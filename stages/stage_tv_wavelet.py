# -*- coding: utf-8 -*-
"""
stages/stage_tv_wavelet.py

时变子波反演阶段。

职责：
1. 固定 r_time_after_DTW
2. 反演 W_est
3. 用 W_est 直接合成
4. 做正负极性选择
5. 计算 W_est 直接候选的 QC 指标
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np



from utils.wavelet_inversion_robust import (
    time_varying_wavelet_inversion,
)

from core.forward_operator import (
    nonstationary_convolution,
    wavelet_matrix_peak_metric_ms,
)
from core.signal_utils import match_rms, best_fit_scale
from core.metrics import (
    corrcoef_safe,
    envelope_cc,
    max_lag_cc,
    tie_similarity_score,
)
from core.acceptance import (
    TvAcceptanceConfig,
    evaluate_tv_wavelet_candidate,
)
from core.wavelet_qc import diag_get_any


@dataclass
class TvWaveletResult:
    W_raw: np.ndarray                 # 原始反演出的时变子波矩阵 (时间 x 子波长度)
    W_best: np.ndarray                # 经过极性选择和处理后的最优子波矩阵
    s_syn: np.ndarray                 # 最终生成的合成记录
    s_pos: np.ndarray                 # 假设正极性（positive polarity）时的合成记录候选
    s_neg: np.ndarray                 # 假设负极性（negative polarity）时的合成记录候选
    diag: dict[str, Any]              # 反演过程中的诊断详细信息
    use_negative: bool                # 是否最终选用了负极性子波
    cc_pos: float                     # 正极性候选的相关系数
    cc_neg: float                     # 负极性候选的相关系数
    cc_direct: float                  # 直接合成记录（无时移）与观测记录的相关系数
    cc_maxlag: float                  # 考虑最大时移匹配后的相关系数
    best_lag_samples: int             # 最佳匹配时移（采样点数）
    best_lag_ms: float                # 最佳匹配时移（毫秒）
    env_cc: float                     # 合成与观测记录的包络相关系数
    direct_gain_vs_dtw: float         # 相比 DTW 阶段的相关系数提升
    gain_vs_stationary: float         # 相比平稳子波阶段的相关系数提升
    env_gain_vs_stationary: float     # 相比平稳子波阶段的包络相关系数提升
    peak_metric_ms: np.ndarray        # 每一时刻子波的峰值偏移指标 (ms)
    peak_metric_p10: float            # 峰值偏移指标的 10% 分位数
    peak_metric_med: float            # 峰值偏移指标的中位数
    peak_metric_p90: float            # 峰值偏移指标的 90% 分位数
    peak_abs_p90: float               # 峰值偏移绝对值的 90% 分位数
    final_peak_violation_ratio: float # 最终平滑后子波主峰越界点比例
    valid_mask: np.ndarray            # 标记哪些时间点是有效计算的掩码
    skip_code: np.ndarray             # 记录每个时间点被跳过计算的具体原因代码
    reflectivity_energy: np.ndarray   # 随时间变化的反射系数能量
    wavelet_energy: np.ndarray        # 随时间变化的子波能量
    centroid_frequency_hz: np.ndarray # 随时间变化的子波质心频率 (Hz)
    attempted_windows: float          # 尝试计算的总窗口数量
    valid_windows: float              # 计算成功的有效窗口数量
    valid_ratio: float                # 有效窗口所占的比例 (0-1)
    numerical_pass: bool              # 数值计算层面是否通过验收
    physical_pass: bool               # 物理规律（如频率、稳定性）层面是否通过验收
    W_pass: bool                      # 最终子波矩阵是否整体通过验收
    acceptance_reasons: list[str]     # 验收通过或失败的具体文字原因列表
    acceptance_details: dict[str, Any]# 验收过程中的详细指标数值
    alignment: str                    # 子波的对齐方式（如 center 或 causal）
    params: dict[str, Any]            # 运行此阶段时使用的参数副本
    scale: float                      # 最终合成记录到观测地震道的最佳缩放因子
    residual_median: np.ndarray        # 每一个滑动窗口的局部残差中位数


def run_tv_wavelet_stage(
    r_time: np.ndarray,
    obs_work: np.ndarray,
    w_prior: np.ndarray,
    dt: float,
    wavelet_length_pts: int,
    data_window_length: int,
    alignment: str = "center",
    peak_allowed_ms: tuple[float, float] = (-15.0, 15.0),
    causal_peak_allowed_ms: tuple[float, float] = (0.0, 40.0),
    cc_stationary: float = np.nan,
    cc_after_dtw: float = np.nan,
    env_stationary: float = np.nan,
    w_prior_source: str = "unknown",
    center_max_peak_shift_ms: float = 15.0,
    mu1: float = 0.2,
    mu2: float = 3.0,
    mu_dc: float = 10.0,
    mu_prior_strict: float = 0.3,
    mu_prior_fallback: float = 0.15,
    mu_time: float = 0.5,
    edge_penalty_enabled: bool = False,
    mu_edge: float = 0.0,
    edge_fraction: float = 0.12,
    edge_taper: str = "cosine",
    energy_percentile: float = 20.0,
    damping_ratio: float = 3e-3,
    svd_cutoff_ratio: float = 1e-3,
    estimate_step_ms: float = 5.0,
    time_smooth_ms: float = 15.0,
    wavelet_smooth_sigma: float = 1.2,
    reject_ill_conditioned: bool = True,
    reject_amplitude_jumps: bool = True,
    loss_type: str = "l2",
    student_nu: float = 10.0,
    irls_max_iter: int = 10,
    irls_tol: float = 1e-4,
    robust_scale_mode: str = "local_mad",
    robust_scale_floor_ratio: float = 0.05,
    robust_weight_floor: float = 1e-3,
    min_effective_sample_ratio: float = 0.30,
    robust_fallback: str = "l2",
    robust_outlier_weight_threshold: float = 0.5,
    store_weight_map: bool = False,
    acceptance_config: TvAcceptanceConfig | None = None,
    verbose: bool = True,
) -> TvWaveletResult:
    r_time = np.asarray(r_time, dtype=float).ravel()
    obs_work = np.asarray(obs_work, dtype=float).ravel()
    w_prior = np.asarray(w_prior, dtype=float)
    if w_prior.ndim not in {1, 2}:
        raise ValueError("w_prior 必须是一维或二维数组。")

    if len(r_time) != len(obs_work):
        raise ValueError("r_time 和 obs_work 长度必须一致。")

    if alignment == "center":
        peak_allowed = peak_allowed_ms
    elif alignment == "causal":
        peak_allowed = causal_peak_allowed_ms
    else:
        raise ValueError("alignment 必须是 'center' 或 'causal'。")

    mu_prior = (
        mu_prior_strict
        if w_prior_source == "strict_stationary_center_after_DTW"
        else mu_prior_fallback
    )

    mu_edge_configured_array = np.asarray(mu_edge, dtype=float)
    if (
        mu_edge_configured_array.ndim != 0
        or not np.isfinite(mu_edge_configured_array)
        or float(mu_edge_configured_array) < 0.0
    ):
        raise ValueError("mu_edge 必须是有限非负数。")
    mu_edge_configured = float(mu_edge_configured_array)
    edge_penalty_enabled = bool(edge_penalty_enabled)
    effective_mu_edge = (
        mu_edge_configured
        if edge_penalty_enabled
        else 0.0
    )

    if verbose:
        print("\n--- Step 2: 固定 r_work_final，反演时变子波 ---")

    W_est, diag = time_varying_wavelet_inversion(
        r_time=r_time,
        s_obs=obs_work,
        wavelet_length=wavelet_length_pts,
        data_window_length=data_window_length,
        dt=dt,
        wavelet_alignment=alignment,
        peak_allowed_ms=peak_allowed,
        mu1=mu1,
        mu2=mu2,
        mu_dc=mu_dc,
        w_prior=w_prior,
        mu_prior=mu_prior,
        mu_time=mu_time,
        mu_edge=effective_mu_edge,
        edge_fraction=edge_fraction,
        edge_taper=edge_taper,
        energy_percentile=energy_percentile,
        damping_ratio=damping_ratio,
        svd_cutoff_ratio=svd_cutoff_ratio,
        estimate_step_ms=estimate_step_ms,
        time_smooth_ms=time_smooth_ms,
        wavelet_smooth_sigma=wavelet_smooth_sigma,
        peak_lock=True,
        max_peak_shift_ms=center_max_peak_shift_ms,
        reject_ill_conditioned=reject_ill_conditioned,
        reject_amplitude_jumps=reject_amplitude_jumps,
        loss_type=loss_type,
        student_nu=student_nu,
        irls_max_iter=irls_max_iter,
        irls_tol=irls_tol,
        robust_scale_mode=robust_scale_mode,
        robust_scale_floor_ratio=robust_scale_floor_ratio,
        robust_weight_floor=robust_weight_floor,
        min_effective_sample_ratio=min_effective_sample_ratio,
        robust_fallback=robust_fallback,
        robust_outlier_weight_threshold=(
            robust_outlier_weight_threshold
        ),
        store_weight_map=store_weight_map,
        return_diagnostics=True,
        verbose=verbose,
    )

    if verbose:
        print("\n--- Step 3: W_est 直接合成与质量验收 ---")

    s_est_pos = nonstationary_convolution(
        r_time=r_time,
        W=W_est,
        alignment=alignment,
    )
    s_est_pos = match_rms(s_est_pos, obs_work)
    s_est_neg = -s_est_pos

    # 原始零时移相关，仅作为诊断量
    cc_direct_pos = corrcoef_safe(obs_work, s_est_pos)
    cc_direct_neg = corrcoef_safe(obs_work, s_est_neg)

    maxlag_ms = 8.0
    maxlag_samples = int(round(maxlag_ms / (dt * 1000.0)))

    maxlag_pos, lag_pos = max_lag_cc(
        obs_work,
        s_est_pos,
        max_lag_samples=maxlag_samples,
    )
    maxlag_neg, lag_neg = max_lag_cc(
        obs_work,
        s_est_neg,
        max_lag_samples=maxlag_samples,
    )

    # 新增：井震标定综合相似性
    similarity_score_pos, similarity_pos = tie_similarity_score(
        s_obs=obs_work,
        s_syn=s_est_pos,
        dt=dt,
        max_lag_ms=maxlag_ms,
    )

    similarity_score_neg, similarity_neg = tie_similarity_score(
        s_obs=obs_work,
        s_syn=s_est_neg,
        dt=dt,
        max_lag_ms=maxlag_ms,
    )

    # 极性选择以零时移直接相关系数最大为准进行
    if cc_direct_neg > cc_direct_pos:
        use_negative = True
        W_est_best = -W_est
        s_syn_tv_direct = s_est_neg
        cc_tv_direct = cc_direct_neg
        cc_tv_maxlag = maxlag_neg
        best_lag = lag_neg
        env_cc = similarity_neg["env_cc"]
        cc_pos = cc_direct_pos
        cc_neg = cc_direct_neg
        composite_score = similarity_score_neg
    else:
        use_negative = False
        W_est_best = W_est
        s_syn_tv_direct = s_est_pos
        cc_tv_direct = cc_direct_pos
        cc_tv_maxlag = maxlag_pos
        best_lag = lag_pos
        env_cc = similarity_pos["env_cc"]
        cc_pos = cc_direct_pos
        cc_neg = cc_direct_neg
        composite_score = similarity_score_pos

    env_tv = envelope_cc(obs_work, s_syn_tv_direct)
    best_lag_ms = best_lag * dt * 1000.0

    peak_metric_ms = wavelet_matrix_peak_metric_ms(
        W=W_est_best,
        dt=dt,
        alignment=alignment,
    )
    peak_metric_med = float(np.nanmedian(peak_metric_ms))
    peak_metric_p10 = float(np.nanpercentile(peak_metric_ms, 10))
    peak_metric_p90 = float(np.nanpercentile(peak_metric_ms, 90))

    if alignment == "center":
        peak_abs_p90 = float(np.nanpercentile(np.abs(peak_metric_ms), 90))
    else:
        peak_abs_p90 = np.nan

    attempted = diag_get_any(
        diag,
        ["attempted_windows", "n_attempted", "attempted"],
        np.nan,
    )
    valid = diag_get_any(
        diag,
        ["valid_windows", "valid_inverted_windows", "n_valid", "valid"],
        np.nan,
    )

    if np.isfinite(attempted) and attempted > 0 and np.isfinite(valid):
        valid_ratio = float(valid) / float(attempted)
    else:
        valid_ratio = np.nan

    valid_mask = np.asarray(
        diag_get_any(diag, ["valid_mask"], np.zeros_like(r_time, dtype=bool)),
        dtype=bool,
    )
    skip_code = np.asarray(
        diag_get_any(diag, ["skip_code"], np.zeros_like(r_time, dtype=int)),
        dtype=int,
    )
    reflectivity_energy = np.asarray(
        diag_get_any(diag, ["r_energy"], np.full_like(r_time, np.nan, dtype=float)),
        dtype=float,
    )
    wavelet_energy = np.asarray(
        diag_get_any(diag, ["qc_energy", "energy"], np.full_like(r_time, np.nan, dtype=float)),
        dtype=float,
    )
    centroid_frequency_hz = np.asarray(
        diag_get_any(
            diag,
            ["qc_centroid_freq", "centroid_frequency_hz", "centroid_freq"],
            np.full_like(r_time, np.nan, dtype=float),
        ),
        dtype=float,
    )

    direct_gain = cc_tv_direct - cc_after_dtw
    stat_gain = cc_tv_direct - cc_stationary
    env_gain = env_tv - env_stationary

    if acceptance_config is None:
        acceptance_config = TvAcceptanceConfig(
            min_gain_vs_dtw=0.015,
            min_gain_vs_stationary=0.020,
            max_env_drop_vs_stationary=0.02,
            max_lag_ms=8.0,
            center_peak_median_ms=5.0,
            center_peak_abs_p90_ms=15.0,
            causal_peak_extra_margin_ms=2.0,
            tolerance_ms=0.0,
            tolerance_cc=0.0,
        )

    if alignment == "center":
        final_peak_limit_ms = float(acceptance_config.center_peak_abs_p90_ms)
        final_peak_violation_mask = np.abs(peak_metric_ms) > final_peak_limit_ms
    else:
        lo, hi = causal_peak_allowed_ms
        margin = float(acceptance_config.causal_peak_extra_margin_ms)
        final_peak_violation_mask = (
            (peak_metric_ms < lo - margin) |
            (peak_metric_ms > hi + margin)
        )

    valid_peaks = np.isfinite(peak_metric_ms)
    if np.any(valid_peaks):
        final_peak_violation_ratio = float(np.mean(final_peak_violation_mask[valid_peaks]))
    else:
        final_peak_violation_ratio = 0.0

    acceptance = evaluate_tv_wavelet_candidate(
        alignment=alignment,
        cc_tv_direct=cc_tv_direct,
        cc_after_dtw=cc_after_dtw,
        cc_stationary=cc_stationary,
        env_tv=env_tv,
        env_stationary=env_stationary,
        best_lag_ms=best_lag_ms,
        peak_metric_p10=peak_metric_p10,
        peak_metric_med=peak_metric_med,
        peak_metric_p90=peak_metric_p90,
        peak_abs_p90=peak_abs_p90,
        causal_peak_allowed_ms=causal_peak_allowed_ms,
        final_peak_violation_ratio=final_peak_violation_ratio,
        config=acceptance_config,
    )

    numerical_pass = acceptance.numerical_pass
    physical_pass = acceptance.physical_pass
    W_pass = acceptance.passed

    if verbose:
        print(f"  use_negative_W      = {use_negative}")
        print(f"  CC_stationary       = {cc_stationary:.4f}")
        print(f"  CC_after_DTW        = {cc_after_dtw:.4f}")
        print(f"  CC_tv_direct        = {cc_tv_direct:.4f}")
        print(f"  CC gain vs DTW      = {direct_gain:+.4f}")
        print(f"  CC gain vs stat     = {stat_gain:+.4f}")
        print(f"  Maxlag CC           = {cc_tv_maxlag:.4f}, lag={best_lag_ms:.1f} ms")
        print(f"  Envelope CC         = {env_tv:.4f}, gain vs stat={env_gain:+.4f}")
        print(
            "  Peak metric p10/med/p90 = "
            f"{peak_metric_p10:.2f} / {peak_metric_med:.2f} / {peak_metric_p90:.2f} ms"
        )
        print(f"  wavelet_alignment   = {alignment}")
        print(f"  valid_ratio         = {valid_ratio}")
        print(f"  numerical_pass      = {numerical_pass}")
        print(f"  physical_pass       = {physical_pass}")
        print(f"  W_est pass gate     = {W_pass}")

    params = {
        "composite_score": float(composite_score),
        "mu1": mu1,
        "mu2": mu2,
        "mu_dc": mu_dc,
        "mu_prior": mu_prior,
        "mu_time": mu_time,
        "edge_penalty_enabled": edge_penalty_enabled,
        "mu_edge_configured": mu_edge_configured,
        "mu_edge_effective": effective_mu_edge,
        "edge_fraction": float(edge_fraction),
        "edge_taper": str(edge_taper),
        "energy_percentile": energy_percentile,
        "damping_ratio": damping_ratio,
        "svd_cutoff_ratio": svd_cutoff_ratio,
        "estimate_step_ms": estimate_step_ms,
        "time_smooth_ms": time_smooth_ms,
        "wavelet_smooth_sigma": wavelet_smooth_sigma,
        "reject_ill_conditioned": reject_ill_conditioned,
        "reject_amplitude_jumps": reject_amplitude_jumps,
        "loss_type": loss_type,
        "student_nu": student_nu,
        "irls_max_iter": irls_max_iter,
        "irls_tol": irls_tol,
        "robust_scale_mode": robust_scale_mode,
        "robust_scale_floor_ratio": robust_scale_floor_ratio,
        "robust_weight_floor": robust_weight_floor,
        "min_effective_sample_ratio": min_effective_sample_ratio,
        "robust_fallback": robust_fallback,
        "robust_outlier_weight_threshold": (
            robust_outlier_weight_threshold
        ),
        "store_weight_map": bool(store_weight_map),
        "w_prior_source": w_prior_source,
        "final_peak_violation_ratio": final_peak_violation_ratio,

        # 新增：相关性诊断
        "cc_direct_pos": cc_direct_pos,
        "cc_direct_neg": cc_direct_neg,
        "tie_similarity_pos": similarity_pos,
        "tie_similarity_neg": similarity_neg,
    }

    scale_val = best_fit_scale(s_syn_tv_direct, obs_work, allow_negative=False)

    return TvWaveletResult(
        W_raw=W_est,                          # 原始反演出的时变子波矩阵
        W_best=W_est_best,                    # 经过极性选择和处理后的最优子波矩阵
        s_syn=s_syn_tv_direct,                # 最终生成的合成记录
        s_pos=s_est_pos,                      # 假设正极性时的合成记录候选
        s_neg=s_est_neg,                      # 假设负极性时的合成记录候选
        diag=diag,                            # 反演过程中的诊断详细信息
        use_negative=use_negative,            # 是否最终选用了负极性子波
        cc_pos=cc_pos,                        # 正极性候选的相关系数
        cc_neg=cc_neg,                        # 负极性候选的相关系数
        cc_direct=cc_tv_direct,               # 直接合成记录（无时移）与观测记录的相关系数
        cc_maxlag=cc_tv_maxlag,               # 考虑最大时移匹配后的相关系数
        best_lag_samples=int(best_lag),       # 最佳匹配时移（采样点数）
        best_lag_ms=float(best_lag_ms),       # 最佳匹配时移（毫秒）
        env_cc=env_tv,                        # 合成与观测记录的包络相关系数
        direct_gain_vs_dtw=float(direct_gain),# 相比 DTW 阶段的相关系数提升
        gain_vs_stationary=float(stat_gain),  # 相比平稳子波阶段的相关系数提升
        env_gain_vs_stationary=float(env_gain),# 相比平稳子波阶段的包络相关系数提升
        peak_metric_ms=peak_metric_ms,        # 每一时刻子波的峰值偏移指标 (ms)
        peak_metric_p10=peak_metric_p10,      # 峰值偏移指标的 10% 分位数
        peak_metric_med=peak_metric_med,      # 峰值偏移指标的中位数
        peak_metric_p90=peak_metric_p90,      # 峰值偏移指标的 90% 分位数
        peak_abs_p90=peak_abs_p90,            # 峰值偏移绝对值的 90% 分位数
        final_peak_violation_ratio=final_peak_violation_ratio, # 最终平滑后子波主峰越界点比例
        valid_mask=valid_mask,                # 标记哪些时间点是有效计算的掩码
        skip_code=skip_code,                  # 记录每个时间点被跳过计算的具体原因代码
        reflectivity_energy=reflectivity_energy,# 随时间变化的反射系数能量
        wavelet_energy=wavelet_energy,        # 随时间变化的子波能量
        centroid_frequency_hz=centroid_frequency_hz, # 随时间变化的子波质心频率 (Hz)
        attempted_windows=float(attempted) if np.isfinite(attempted) else np.nan, # 尝试计算的总窗口数量
        valid_windows=float(valid) if np.isfinite(valid) else np.nan,             # 计算成功的有效窗口数量
        valid_ratio=float(valid_ratio) if np.isfinite(valid_ratio) else np.nan,   # 有效窗口所占的比例 (0-1)
        numerical_pass=bool(numerical_pass),   # 数值计算层面是否通过验收
        physical_pass=bool(physical_pass),     # 物理规律（如频率、稳定性）层面是否通过验收
        W_pass=bool(W_pass),                   # 最终子波矩阵是否整体通过验收
        acceptance_reasons=acceptance.reasons, # 验收通过或失败的具体文字原因列表
        acceptance_details=acceptance.details, # 验收过程中的详细指标数值
        alignment=alignment,                   # 子波的对齐方式（center 或 causal）
        params=params,                         # 运行此阶段时使用的参数副本
        scale=scale_val,
        residual_median=diag.get("residual_median", np.zeros_like(r_time, dtype=float)),
    )
