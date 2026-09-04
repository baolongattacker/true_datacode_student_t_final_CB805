# -*- coding: utf-8 -*-
"""
stages/stage_dtw.py

DTW / TWT 更新阶段。

职责：
1. 固定当前子波，只更新 TWT(z) 与 r_time(t)
2. 使用 candidate-gated DTW，坏更新自动回退
3. 输出 DTW 后的 r_time、twt、v、s_syn 和 history
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.signal import hilbert



from utils.dtw_timeoffset import (
    calculate_nonstationary_time_shift,
    compute_dtw_window_scalar,
)
from utils.wavelet_inversion_robust import stationary_wavelet_inversion

from core.forward_operator import stationary_convolution
from core.signal_utils import match_rms, normalize_for_dtw, best_fit_scale
from core.metrics import corrcoef_safe, envelope_cc, max_lag_cc
from core.reflectivity_mapping import (
    scatter_reflectivity_to_time,
    enforce_monotonic_twt,
    velocity_from_twt,
)


@dataclass
# DTW各阶段的配置项，描述这一阶段该怎么跑
class DtwPhaseSpec:
    phase: str  # 当前阶段名称：envelope / waveform / fine
    n_iter: int  # 该阶段重复执行的迭代次数
    alpha_candidates: list[float]  # 候选 alpha 列表，用于控制位移幅度
    max_shift_ms: float  # 单次允许的最大时间位移（毫秒）
    smooth_ms: float  # 对位移曲线进行平滑的尺度（毫秒）
    accept_min_cc_gain: float  # 接受更新所需的最小相关系数增益
    accept_env_drop: float  # 允许的最大包络相关下降
    accept_max_lag_ms: float  # 接受更新所允许的最大滞后（毫秒）


@dataclass
# 单次DTW的诊断记录，包含当前模式、候选结果、最终选择的结果，以及是否接受更新等信息
class DtwStepDiag:
    mode: str  # 当前 DTW 模式
    window: int  # DTW 窗口大小（采样点）
    best_alpha: float  # 最终选中的 alpha
    sign_used: int  # 位移方向：+1 / -1 / 0
    cc_before: float  # 更新前相关系数
    cc_after: float  # 更新后相关系数
    env_before: float  # 更新前包络相关系数
    env_after: float  # 更新后包络相关系数
    maxlag_after: float  # 更新后最大互相关值
    lag_after_ms: float  # 更新后滞后时间（毫秒）
    accept: bool  # 该候选是否被接受
    shift_min_ms: float  # 位移曲线最小值（毫秒）
    shift_max_ms: float  # 位移曲线最大值（毫秒）
    twt_change_ms: float  # TWT 最大变化量（毫秒）
    delta_t: np.ndarray  # 时间域位移曲线
    dt_depth: np.ndarray  # 映射到深度/TWT 轴上的位移
    twt_before: np.ndarray  # 更新前的 TWT
    twt_trial: np.ndarray  # 试探更新后的 TWT
    s_syn_before: np.ndarray  # 更新前的合成记录
    s_syn_after: np.ndarray  # 更新后的合成记录
    aligned_syn: np.ndarray  # DTW 对齐后的合成信号
    candidate_table: dict[str, np.ndarray]  # 所有候选结果的汇总表


@dataclass
# 整个DTW阶段的结果，包含最终的 TWT、反射率、速度、合成记录，以及整个 DTW 过程中的历史诊断信息等
class DtwResult:
    r_time: np.ndarray  # 最终时域反射率
    twt: np.ndarray  # 最终双程时曲线
    v: np.ndarray  # 最终速度曲线
    s_syn: np.ndarray  # 最终合成地震记录
    s_before_dtw: np.ndarray  # DTW 之前的合成记录
    cc_before: float  # DTW 前相关系数
    cc_after: float  # DTW 后相关系数
    env_before: float  # DTW 前包络相关系数
    env_after: float  # DTW 后包络相关系数
    history: list[dict[str, Any]]  # 每一步的历史诊断信息
    last_diag_by_mode: dict[str, dict[str, Any]]  # 各模式最后一次诊断结果
    total_accepts: int  # 总接受次数
    total_rejects: int  # 总拒绝次数
    alignment: str  # 卷积对齐方式
    w_for_dtw_final: np.ndarray  # DTW 阶段最终使用的子波
    scale: float  # 最终合成记录到观测地震道的最佳缩放因子

# 返回值类型注解，表示这个函数的返回值应该是一个 列表（list），列表中的每个元素类型都是 DtwPhaseSpec
def default_dtw_phase_specs() -> list[DtwPhaseSpec]:
    return [
        DtwPhaseSpec(
            phase="envelope",
            n_iter=2,
            alpha_candidates=[0.0, 0.05, 0.10, 0.15],
            max_shift_ms=12.0,
            smooth_ms=25.0,
            accept_min_cc_gain=0.001,
            accept_env_drop=0.02,
            accept_max_lag_ms=10.0,
        ),
        DtwPhaseSpec(
            phase="waveform",
            n_iter=2,
            alpha_candidates=[0.0, 0.03, 0.06, 0.10],
            max_shift_ms=8.0,
            smooth_ms=20.0,
            accept_min_cc_gain=0.0015,
            accept_env_drop=0.03,
            accept_max_lag_ms=8.0,
        ),
        DtwPhaseSpec(
            phase="fine",
            n_iter=1,
            alpha_candidates=[0.0, 0.02, 0.04, 0.06],
            max_shift_ms=5.0,
            smooth_ms=12.0,
            accept_min_cc_gain=0.0005,
            accept_env_drop=0.03,
            accept_max_lag_ms=6.0,
        ),
    ]


def dtw_update_twt_and_reflectivity(
    obs_work: np.ndarray,
    r_time_current: np.ndarray,
    w_current: np.ndarray,
    t_work: np.ndarray,
    depth: np.ndarray,
    twt_current: np.ndarray,
    r_depth_fixed: np.ndarray,
    dt: float,
    f_dom: float,
    mode: str = "envelope",
    alpha: float = 0.10,
    alpha_candidates: list[float] | None = None,
    max_shift_ms: float = 12.0,
    smooth_ms: float = 20.0,
    accept_min_cc_gain: float = 0.0,
    accept_env_drop: float = 0.03,
    accept_max_lag_ms: float = 10.0,
    alignment: str = "center",
    verbose: bool = True,
):
    if alpha_candidates is None:
        alpha_candidates = [alpha]

    alpha_candidates = [float(a) for a in alpha_candidates]

    if not any(abs(a) <= 1e-15 for a in alpha_candidates):
        alpha_candidates = [0.0] + alpha_candidates

    s_syn_current = stationary_convolution(
        r_time=r_time_current,
        w=w_current,
        alignment=alignment,
    )
    s_syn_current = match_rms(s_syn_current, obs_work)

    cc_before = corrcoef_safe(obs_work, s_syn_current)
    env_before = envelope_cc(obs_work, s_syn_current)

    if mode == "envelope":
        raw_obs = np.abs(hilbert(obs_work))
        raw_syn = np.abs(hilbert(s_syn_current))
        current_window = compute_dtw_window_scalar(
            twt=t_work,
            dt=dt,
            f_dom=f_dom,
            mode="envelope",
            base_ms=6.0,
            vel_err=0.02,
            T0_ms=20.0,
            min_samples=5,
            max_samples=18,
        )

    elif mode == "waveform":
        raw_obs = obs_work
        raw_syn = s_syn_current
        current_window = compute_dtw_window_scalar(
            twt=t_work,
            dt=dt,
            f_dom=f_dom,
            mode="waveform",
            base_ms=5.0,
            vel_err=0.015,
            T0_ms=20.0,
            min_samples=4,
            max_samples=16,
        )

    elif mode == "fine":
        raw_obs = obs_work
        raw_syn = s_syn_current
        current_window = compute_dtw_window_scalar(
            twt=t_work,
            dt=dt,
            f_dom=f_dom,
            mode="fine",
            base_ms=3.0,
            vel_err=0.008,
            T0_ms=20.0,
            min_samples=2,
            max_samples=10,
        )

    else:
        raise ValueError("mode must be envelope, waveform, or fine")

    sig_obs = normalize_for_dtw(raw_obs)
    sig_syn = normalize_for_dtw(raw_syn)

    f_t, aligned_syn = calculate_nonstationary_time_shift(
        sig_obs,
        sig_syn,
        dt,
        window=current_window,
    )

    f_t = np.asarray(f_t, dtype=float)

    sigma_samples = max(2.0, smooth_ms / 1000.0 / dt)
    max_shift_s = max_shift_ms / 1000.0
    maxlag_samples = int(round(accept_max_lag_ms / (dt * 1000.0)))

    candidates = []

    for alpha_i in alpha_candidates:
        delta_t_i = alpha_i * f_t
        delta_t_i = gaussian_filter1d(
            delta_t_i,
            sigma=sigma_samples,
            mode="nearest",
        )
        delta_t_i = np.clip(delta_t_i, -max_shift_s, max_shift_s)

        interp_shift = interp1d(
            t_work,
            delta_t_i,
            kind="linear",
            bounds_error=False,
            fill_value=0.0,
        )

        dt_depth_i = interp_shift(twt_current)

        signs = [0] if abs(alpha_i) <= 1e-15 else [+1, -1]

        for sign_i in signs:
            if sign_i == 0:
                twt_i = np.asarray(twt_current, dtype=float).copy()
                r_i = np.asarray(r_time_current, dtype=float).copy()
                s_i = s_syn_current.copy()
            else:
                twt_i = enforce_monotonic_twt(
                    twt_current + sign_i * dt_depth_i
                )
                r_i = scatter_reflectivity_to_time(
                    r_depth_fixed,
                    twt_i,
                    t_work,
                )
                s_i = stationary_convolution(
                    r_time=r_i,
                    w=w_current,
                    alignment=alignment,
                )
                s_i = match_rms(s_i, obs_work)

            cc_i = corrcoef_safe(obs_work, s_i)
            env_i = envelope_cc(obs_work, s_i)

            maxlag_i, lag_i = max_lag_cc(
                obs_work,
                s_i,
                max_lag_samples=maxlag_samples,
            )
            lag_i_ms = lag_i * dt * 1000.0

            candidates.append({
                "alpha": float(alpha_i),
                "sign": int(sign_i),
                "delta_t": delta_t_i.copy(),
                "dt_depth": dt_depth_i.copy(),
                "twt": twt_i,
                "r_time": r_i,
                "s_syn": s_i,
                "cc_after": float(cc_i),
                "env_after": float(env_i),
                "maxlag_after": float(maxlag_i),
                "lag_after_ms": float(lag_i_ms),
                "shift_min_ms": float(np.nanmin(delta_t_i) * 1000.0),
                "shift_max_ms": float(np.nanmax(delta_t_i) * 1000.0),
                "twt_change_ms": float(
                    np.nanmax(np.abs(twt_i - twt_current)) * 1000.0
                ),
            })

    def candidate_score(c):
        cc_gain = c["cc_after"] - cc_before
        env_loss = max(0.0, env_before - c["env_after"])
        lag_penalty = 0.002 * abs(c["lag_after_ms"]) / max(
            accept_max_lag_ms,
            1e-12,
        )
        return cc_gain - 0.25 * env_loss - lag_penalty

    best = max(candidates, key=candidate_score)

    accept = (
        abs(best["alpha"]) > 1e-15
        and best["sign"] != 0
        and best["cc_after"] >= cc_before + accept_min_cc_gain
        and best["env_after"] >= env_before - accept_env_drop
        and abs(best["lag_after_ms"]) <= accept_max_lag_ms
    )

    candidate_table = {
        "alpha": np.array([c["alpha"] for c in candidates], dtype=float),
        "sign": np.array([c["sign"] for c in candidates], dtype=int),
        "cc_after": np.array([c["cc_after"] for c in candidates], dtype=float),
        "env_after": np.array([c["env_after"] for c in candidates], dtype=float),
        "maxlag_after": np.array([c["maxlag_after"] for c in candidates], dtype=float),
        "lag_after_ms": np.array([c["lag_after_ms"] for c in candidates], dtype=float),
        "shift_min_ms": np.array([c["shift_min_ms"] for c in candidates], dtype=float),
        "shift_max_ms": np.array([c["shift_max_ms"] for c in candidates], dtype=float),
        "twt_change_ms": np.array([c["twt_change_ms"] for c in candidates], dtype=float),
    }

    diag = {
        "mode": mode,
        "window": current_window,
        "alpha_candidates": np.array(alpha_candidates, dtype=float),
        "best_alpha": float(best["alpha"]),
        "delta_t": best["delta_t"],
        "dt_depth": best["dt_depth"],
        "twt_before": np.asarray(twt_current, dtype=float).copy(),
        "twt_trial": best["twt"].copy(),
        "cc_before": float(cc_before),
        "cc_after": float(best["cc_after"]),
        "env_before": float(env_before),
        "env_after": float(best["env_after"]),
        "maxlag_after": float(best["maxlag_after"]),
        "lag_after_ms": float(best["lag_after_ms"]),
        "accept": bool(accept),
        "s_syn_before": s_syn_current,
        "s_syn_after": best["s_syn"],
        "aligned_syn": aligned_syn,
        "sign_used": int(best["sign"]),
        "candidate_table": candidate_table,
        "shift_min_ms": float(best["shift_min_ms"]),
        "shift_max_ms": float(best["shift_max_ms"]),
        "twt_change_ms": float(best["twt_change_ms"]),
    }

    if verbose:
        print(f"  [DTW-{mode}] window = {current_window} samples")
        print(f"  [DTW-{mode}] alpha candidates = {alpha_candidates}")
        print(f"  [DTW-{mode}] best alpha/sign = {best['alpha']:.3f} / {best['sign']:+d}")
        print(
            f"  [DTW-{mode}] shift range = "
            f"{best['shift_min_ms']:.2f} ~ {best['shift_max_ms']:.2f} ms"
        )
        print(f"  [DTW-{mode}] CC before/after = {cc_before:.4f} / {best['cc_after']:.4f}")
        print(f"  [DTW-{mode}] Env before/after = {env_before:.4f} / {best['env_after']:.4f}")
        print(
            f"  [DTW-{mode}] maxlag after = {best['maxlag_after']:.4f}, "
            f"lag = {best['lag_after_ms']:.1f} ms"
        )
        print(f"  [DTW-{mode}] TWT max change = {best['twt_change_ms']:.2f} ms")
        print(f"  [DTW-{mode}] accept = {accept}")

    if accept:
        v_trial = velocity_from_twt(depth, best["twt"])
        return best["r_time"], best["twt"], v_trial, best["s_syn"], diag

    v_current = velocity_from_twt(depth, twt_current)
    return r_time_current, twt_current, v_current, s_syn_current, diag


def run_dtw_stage(
    obs_work: np.ndarray,
    r_time_init: np.ndarray,
    w_initial: np.ndarray,
    t_work: np.ndarray,
    depth: np.ndarray,
    twt_init: np.ndarray,
    r_depth_fixed: np.ndarray,
    dt: float,
    f_dom: float,
    wavelet_length_pts: int,
    alignment: str = "center",
    center_max_peak_shift_ms: float = 15.0,
    phase_specs: list[DtwPhaseSpec] | None = None,
    refresh_stationary_before_fine: bool = False,
    w_source_for_dtw: str = "unknown",
    verbose: bool = True,
) -> DtwResult:
    if phase_specs is None:
        phase_specs = default_dtw_phase_specs()

    r_current = np.asarray(r_time_init, dtype=float).copy()
    twt_current = np.asarray(twt_init, dtype=float).copy()
    w_current = np.asarray(w_initial, dtype=float).copy()

    v_current = velocity_from_twt(depth, twt_current)

    s_current = stationary_convolution(
        r_time=r_current,
        w=w_current,
        alignment=alignment,
    )
    s_current = match_rms(s_current, obs_work)

    s_before_dtw = s_current.copy()
    cc_before_all = corrcoef_safe(obs_work, s_current)
    env_before_all = envelope_cc(obs_work, s_current)

    history: list[dict[str, Any]] = []
    last_diag_by_mode: dict[str, dict[str, Any]] = {}
    total_accepts = 0
    total_rejects = 0
    reject_streak = 0
    if verbose:
        print("  [DTW-iter] phased candidate DTW begins")

    for spec in phase_specs:
        phase_accepts = 0
        phase_rejects = 0
        mode = spec.phase

        if mode == "fine" and refresh_stationary_before_fine:
            peak_max_samples = int(
                round(center_max_peak_shift_ms / 1000.0 / dt)
            )

            try:
                w_current = stationary_wavelet_inversion(
                    r_time=r_current,
                    s_obs=obs_work,
                    wavelet_length=wavelet_length_pts,
                    mu1=0.2,
                    mu2=2.0,
                    mu_dc=10.0,
                    damping_ratio=3e-3,
                    svd_cutoff_ratio=1e-3,
                    peak_lock=True,
                    max_peak_shift_samples=peak_max_samples,
                    wavelet_alignment=alignment,
                )

                if verbose:
                    print("  [DTW-iter] refreshed stationary prior before fine DTW")

            except ValueError as e:
                if verbose:
                    print(f"  [DTW-iter] keep inherited prior before fine DTW: {e}")
                    
        if mode == "fine" and (not refresh_stationary_before_fine):
            if verbose:
                print("  [DTW-iter] keep fixed initial wavelet before fine DTW")

        for it in range(spec.n_iter):
            r_next, twt_next, v_next, s_next, diag = dtw_update_twt_and_reflectivity(
                obs_work=obs_work,
                r_time_current=r_current,
                w_current=w_current,
                t_work=t_work,
                depth=depth,
                twt_current=twt_current,
                r_depth_fixed=r_depth_fixed,
                dt=dt,
                f_dom=f_dom,
                mode=mode,
                alpha_candidates=spec.alpha_candidates,
                max_shift_ms=spec.max_shift_ms,
                smooth_ms=spec.smooth_ms,
                accept_min_cc_gain=spec.accept_min_cc_gain,
                accept_env_drop=spec.accept_env_drop,
                accept_max_lag_ms=spec.accept_max_lag_ms,
                alignment=alignment,
                verbose=verbose,
            )

            diag["phase"] = mode
            diag["phase_iteration"] = it + 1
            diag["global_iteration"] = len(history) + 1
            diag["w_source"] = w_source_for_dtw
            history.append(diag)
            last_diag_by_mode[mode] = diag

            if diag["accept"]:
                r_current = r_next
                twt_current = twt_next
                v_current = v_next
                s_current = s_next
                phase_accepts += 1
                total_accepts += 1
                reject_streak = 0
            else:
                phase_rejects += 1
                total_rejects += 1
                reject_streak += 1

            if phase_rejects >= 2 or reject_streak >= 2:
                if verbose:
                    print(
                        f"  [DTW-iter] stop {mode}: "
                        f"phase rejects={phase_rejects}, total rejects={reject_streak}"
                    )
                # stop_all = True
                break

        if verbose:
            print(f"  [DTW-iter] phase {mode}: accepted {phase_accepts}/{spec.n_iter}")

    cc_after_all = corrcoef_safe(obs_work, s_current)
    env_after_all = envelope_cc(obs_work, s_current)
    scale_val = best_fit_scale(s_current, obs_work, allow_negative=False)

    return DtwResult(
        r_time=r_current,
        twt=twt_current,
        v=v_current,
        s_syn=s_current,
        s_before_dtw=s_before_dtw,
        cc_before=cc_before_all,
        cc_after=cc_after_all,
        env_before=env_before_all,
        env_after=env_after_all,
        history=history,
        last_diag_by_mode=last_diag_by_mode,
        total_accepts=total_accepts,
        total_rejects=total_rejects,
        alignment=alignment,
        w_for_dtw_final=w_current,
        scale=scale_val,
    )
