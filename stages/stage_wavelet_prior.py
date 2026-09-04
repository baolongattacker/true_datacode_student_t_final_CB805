# -*- coding: utf-8 -*-
"""
stages/stage_wavelet_prior.py

Stationary-prior policy for strict inversion with Ricker fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from core.forward_operator import stationary_convolution, wavelet_lag_axis_ms
from core.metrics import corrcoef_safe, envelope_cc
from core.signal_utils import match_rms
from stages.stage_stationary import StationaryResult, make_odd, run_stationary_stage


@dataclass
class WaveletPriorResult:
    w: np.ndarray
    source: str
    s_syn: np.ndarray
    cc: float
    env_cc: float
    wavelet_length_pts: int
    data_window_length: int
    stationary_result: StationaryResult | None
    stationary_candidate_result: StationaryResult | None
    stationary_candidate_accepted: bool
    stationary_candidate_peak_ms: float
    stationary_rejection_code: str
    stationary_rejection_reason: str
    stationary_prior_source: str


def _stationary_peak_metric_ms(
    w: np.ndarray,
    dt: float,
    alignment: str,
) -> float:
    """返回平稳子波绝对主峰对应的 center/causal 峰值指标，单位 ms。"""
    w = np.asarray(w, dtype=float)
    if w.ndim != 1 or w.size == 0:
        raise ValueError("平稳候选子波必须是一维非空数组。")
    if not np.all(np.isfinite(w)):
        raise ValueError("平稳候选子波包含 NaN 或 Inf。")

    lag_axis_ms = wavelet_lag_axis_ms(
        wavelet_length=w.size,
        dt=dt,
        alignment=alignment,
    )
    peak_index = int(np.argmax(np.abs(w)))
    return float(lag_axis_ms[peak_index])


def _stationary_rejection_code(exc: ValueError) -> str:
    """将当前平稳阶段可识别的失败原因转换为稳定机器码。"""
    reason = str(exc)
    if "主峰位置不满足约束" in reason or "peak_metric" in reason:
        return "peak_out_of_range"
    return "stationary_solver_error"


def make_center_ricker_wavelet(f0: float, dt: float, length: int) -> np.ndarray:
    """Create a normalized center-aligned Ricker wavelet."""
    length = int(length)
    if length % 2 == 0:
        length += 1

    t = (np.arange(length) - length // 2) * dt
    a = (np.pi * f0 * t) ** 2
    w = (1.0 - 2.0 * a) * np.exp(-a)
    w = w - np.mean(w)
    w = w / (np.max(np.abs(w)) + 1e-12)
    return w


def build_ricker_prior(
    *,
    r_time: np.ndarray,
    s_obs: np.ndarray,
    dt: float,
    f_dom: float,
    wavelet_length_s: float,
    alignment: str,
    source: str = "center_ricker_for_DTW",
    data_window_factor: float = 3.0,
    verbose: bool = True,
) -> WaveletPriorResult:
    """
    构造 DTW 前使用的中心 Ricker 子波先验。

    输入：
        r_time: 当前时域反射系数
        s_obs: 井旁道 / 观测地震道
        dt: 时间采样间隔，单位 s
        f_dom: 主频，单位 Hz
        wavelet_length_s: 子波长度，单位 s
        alignment: 卷积对齐方式，center 或 causal

    输出：
        WaveletPriorResult

    物理意义：
        DTW 前只提供一个物理中性的零相位子波，
        不通过平稳子波反演吸收井震残余时差。

    数学作用：
        用 Ricker 子波生成初始合成记录，并做 RMS 匹配。
    """
    r_time = np.asarray(r_time, dtype=float).ravel()
    s_obs = np.asarray(s_obs, dtype=float).ravel()

    if len(r_time) != len(s_obs):
        raise ValueError("r_time 和 s_obs 长度必须一致。")

    if dt <= 0:
        raise ValueError("dt 必须大于 0。")

    wavelet_length_pts = make_odd(round(wavelet_length_s / dt))
    data_window_length = make_odd(round(data_window_factor * wavelet_length_pts))

    w = make_center_ricker_wavelet(
        f0=f_dom,
        dt=dt,
        length=wavelet_length_pts,
    )

    s_syn = stationary_convolution(
        r_time=r_time,
        w=w,
        alignment=alignment,
    )
    s_syn = match_rms(s_syn, s_obs)

    cc = corrcoef_safe(s_obs, s_syn)
    env = envelope_cc(s_obs, s_syn)

    if verbose:
        print("  [Ricker prior] use fixed center Ricker before DTW")
        print(f"  [Ricker prior] f_dom = {f_dom:.2f} Hz")
        print(f"  [Ricker prior] wavelet_length_pts = {wavelet_length_pts}")
        print(f"  [Ricker prior] CC = {cc:.6f}")

    return WaveletPriorResult(
        w=w,
        source=source,
        s_syn=s_syn,
        cc=cc,
        env_cc=env,
        wavelet_length_pts=wavelet_length_pts,
        data_window_length=data_window_length,
        stationary_result=None,
        stationary_candidate_result=None,
        stationary_candidate_accepted=False,
        stationary_candidate_peak_ms=np.nan,
        stationary_rejection_code="not_attempted",
        stationary_rejection_reason="DTW 前固定 Ricker 先验未运行平稳候选反演。",
        stationary_prior_source=source,
    )


def build_stationary_prior_with_fallback(
    *,
    r_time: np.ndarray,
    s_obs: np.ndarray,
    dt: float,
    f_dom: float,
    wavelet_length_s: float,
    alignment: str,
    strict_source: str,
    fallback_source: str,
    mu1: float = 0.2,
    mu2: float = 2.0,
    mu_dc: float = 10.0,
    damping_ratio: float = 3e-3,
    svd_cutoff_ratio: float = 1e-3,
    peak_lock: bool = True,
    max_peak_shift_ms: float = 15.0,
    data_window_factor: float = 3.0,
    stationary_runner: Callable | None = None,
    verbose: bool = True,
) -> WaveletPriorResult:
    """
    Run strict stationary inversion and fall back to center Ricker on failure.
    """
    if stationary_runner is None:
        stationary_runner = run_stationary_stage

    wavelet_length_pts = make_odd(round(wavelet_length_s / dt))
    data_window_length = make_odd(round(data_window_factor * wavelet_length_pts))

    try:
        result = stationary_runner(
            r_time=r_time,
            s_obs=s_obs,
            dt=dt,
            wavelet_length_s=wavelet_length_s,
            alignment=alignment,
            mu1=mu1,
            mu2=mu2,
            mu_dc=mu_dc,
            damping_ratio=damping_ratio,
            svd_cutoff_ratio=svd_cutoff_ratio,
            peak_lock=peak_lock,
            max_peak_shift_ms=max_peak_shift_ms,
            data_window_factor=data_window_factor,
        )

        return WaveletPriorResult(
            w=result.w,
            source=strict_source,
            s_syn=result.s_syn,
            cc=result.cc,
            env_cc=result.env_cc,
            wavelet_length_pts=result.wavelet_length_pts,
            data_window_length=result.data_window_length,
            stationary_result=result,
            stationary_candidate_result=result,
            stationary_candidate_accepted=True,
            stationary_candidate_peak_ms=_stationary_peak_metric_ms(
                w=result.w,
                dt=dt,
                alignment=alignment,
            ),
            stationary_rejection_code="accepted",
            stationary_rejection_reason="",
            stationary_prior_source=strict_source,
        )

    except ValueError as exc:
        rejection_code = _stationary_rejection_code(exc)
        rejection_reason = str(exc)

        if verbose:
            print("  [warning] strict stationary prior rejected.")
            print(f"  [reason] {exc}")
            print(f"  [fallback] using {fallback_source}.")

        # 仅为保留被拒绝候选的原始 CC/峰值诊断，使用完全相同参数关闭峰值门重跑。
        # 最终先验仍保持 Ricker fallback，不改变原有分支选择和物理流程。
        stationary_candidate_result = None
        stationary_candidate_peak_ms = np.nan
        try:
            stationary_candidate_result = stationary_runner(
                r_time=r_time,
                s_obs=s_obs,
                dt=dt,
                wavelet_length_s=wavelet_length_s,
                alignment=alignment,
                mu1=mu1,
                mu2=mu2,
                mu_dc=mu_dc,
                damping_ratio=damping_ratio,
                svd_cutoff_ratio=svd_cutoff_ratio,
                peak_lock=False,
                max_peak_shift_ms=max_peak_shift_ms,
                data_window_factor=data_window_factor,
            )
            stationary_candidate_peak_ms = _stationary_peak_metric_ms(
                w=stationary_candidate_result.w,
                dt=dt,
                alignment=alignment,
            )
        except ValueError as diagnostic_exc:
            rejection_reason = (
                f"{rejection_reason} | diagnostic rerun unavailable: "
                f"{diagnostic_exc}"
            )

        w = make_center_ricker_wavelet(f0=f_dom, dt=dt, length=wavelet_length_pts)
        s_syn = stationary_convolution(
            r_time=r_time,
            w=w,
            alignment=alignment,
        )
        s_syn = match_rms(s_syn, s_obs)

        return WaveletPriorResult(
            w=w,
            source=fallback_source,
            s_syn=s_syn,
            cc=corrcoef_safe(s_obs, s_syn),
            env_cc=envelope_cc(s_obs, s_syn),
            wavelet_length_pts=wavelet_length_pts,
            data_window_length=data_window_length,
            stationary_result=None,
            stationary_candidate_result=stationary_candidate_result,
            stationary_candidate_accepted=False,
            stationary_candidate_peak_ms=stationary_candidate_peak_ms,
            stationary_rejection_code=rejection_code,
            stationary_rejection_reason=rejection_reason,
            stationary_prior_source=fallback_source,
        )
