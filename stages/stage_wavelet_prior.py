# -*- coding: utf-8 -*-
"""
stages/stage_wavelet_prior.py

Stationary-prior policy for strict inversion with Ricker fallback.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from core.forward_operator import stationary_convolution, wavelet_lag_axis_ms
from core.metrics import corrcoef_safe, envelope_cc
from core.signal_utils import match_rms
from stages.stage_stationary import StationaryResult, make_odd, run_stationary_stage
from utils.constant_phase_wavelet import (
    build_constant_phase_wavelet,
    default_phase_grid_deg,
    estimate_statistical_amplitude_spectrum,
)


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
    constant_phase_deg: float = np.nan
    constant_phase_spectral_peak_hz: float = np.nan
    constant_phase_band_low_hz: float = np.nan
    constant_phase_band_high_hz: float = np.nan
    constant_phase_candidate_cc: float = np.nan
    constant_phase_ricker_cc: float = np.nan
    constant_phase_candidate_accepted: bool = False
    constant_phase_rejection_reason: str = ""
    constant_phase_zero_phase_wavelet: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=float)
    )
    constant_phase_phase_grid_deg: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=float)
    )
    constant_phase_phase_cc: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=float)
    )
    constant_phase_spectrum_hz: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=float)
    )
    constant_phase_spectrum_amplitude: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=float)
    )


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
    mu_edge: float = 0.0,
    edge_fraction: float = 0.12,
    edge_taper: str = "cosine",
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
            mu_edge=mu_edge,
            edge_fraction=edge_fraction,
            edge_taper=edge_taper,
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


def build_constant_phase_prior_with_fallback(
    *,
    r_time: np.ndarray,
    s_obs: np.ndarray,
    dt: float,
    f_dom: float,
    wavelet_length_s: float,
    alignment: str,
    source: str = "statistical_constant_phase_prior_after_DTW",
    fallback_source: str = "center_ricker_prior_after_DTW",
    max_peak_shift_ms: float = 15.0,
    data_window_factor: float = 3.0,
    verbose: bool = True,
) -> WaveletPriorResult:
    """
    Statistical constant-phase prior after DTW.

    Amplitude spectrum comes from the observed seismic statistics. The only
    searched shape degree of freedom is one constant phase angle. Selection
    uses zero-lag direct CC and the existing peak-position QC, so this stage
    cannot hide residual timing error inside a max-lag score.

    The candidate must perform at least as well as the Ricker baseline in
    direct CC. Otherwise the existing Ricker fallback is returned.
    """
    r_time = np.asarray(r_time, dtype=float).ravel()
    s_obs = np.asarray(s_obs, dtype=float).ravel()

    if r_time.size != s_obs.size:
        raise ValueError("r_time 和 s_obs 长度必须一致。")
    if not np.all(np.isfinite(r_time)) or not np.all(np.isfinite(s_obs)):
        raise ValueError("r_time 和 s_obs 必须全部为有限值。")
    if dt <= 0.0:
        raise ValueError("dt 必须大于 0。")

    wavelet_length_pts = make_odd(round(wavelet_length_s / dt))
    data_window_length = make_odd(
        round(data_window_factor * wavelet_length_pts)
    )

    ricker = make_center_ricker_wavelet(
        f0=f_dom,
        dt=dt,
        length=wavelet_length_pts,
    )
    ricker_syn = stationary_convolution(
        r_time=r_time,
        w=ricker,
        alignment=alignment,
    )
    ricker_syn = match_rms(ricker_syn, s_obs)
    ricker_cc = corrcoef_safe(s_obs, ricker_syn)
    ricker_env = envelope_cc(s_obs, ricker_syn)

    if alignment != "center":
        reason = (
            "constant-phase statistical prior 当前只对 center alignment 开启；"
            "causal 模式保持 Ricker fallback。"
        )
        if verbose:
            print(f"  [Constant-phase prior] {reason}")
        return WaveletPriorResult(
            w=ricker,
            source=fallback_source,
            s_syn=ricker_syn,
            cc=ricker_cc,
            env_cc=ricker_env,
            wavelet_length_pts=wavelet_length_pts,
            data_window_length=data_window_length,
            stationary_result=None,
            stationary_candidate_result=None,
            stationary_candidate_accepted=False,
            stationary_candidate_peak_ms=np.nan,
            stationary_rejection_code="constant_phase_not_supported_for_alignment",
            stationary_rejection_reason=reason,
            stationary_prior_source=fallback_source,
            constant_phase_ricker_cc=float(ricker_cc),
            constant_phase_candidate_accepted=False,
            constant_phase_rejection_reason=reason,
        )

    try:
        spectrum = estimate_statistical_amplitude_spectrum(
            s_obs=s_obs,
            dt=dt,
            f_dom=f_dom,
        )

        zero_phase_wavelet = build_constant_phase_wavelet(
            spectrum=spectrum,
            dt=dt,
            wavelet_length=wavelet_length_pts,
            phase_deg=0.0,
        )

        phase_grid = default_phase_grid_deg()
        phase_cc = np.full(phase_grid.size, np.nan, dtype=float)
        phase_peak_ms = np.full(phase_grid.size, np.nan, dtype=float)

        best_index = -1
        best_cc = -np.inf
        best_env = -np.inf
        best_wavelet = None
        best_synthetic = None

        peak_tolerance_ms = 0.5 * dt * 1000.0 + 1e-9

        for index, phase_deg in enumerate(phase_grid):
            w = build_constant_phase_wavelet(
                spectrum=spectrum,
                dt=dt,
                wavelet_length=wavelet_length_pts,
                phase_deg=float(phase_deg),
            )

            peak_ms = _stationary_peak_metric_ms(
                w=w,
                dt=dt,
                alignment=alignment,
            )
            phase_peak_ms[index] = peak_ms

            if abs(peak_ms) > max_peak_shift_ms + peak_tolerance_ms:
                continue

            s_syn = stationary_convolution(
                r_time=r_time,
                w=w,
                alignment=alignment,
            )
            s_syn = match_rms(s_syn, s_obs)

            cc = corrcoef_safe(s_obs, s_syn)
            env = envelope_cc(s_obs, s_syn)
            phase_cc[index] = cc

            better = cc > best_cc + 1e-12
            tie = abs(cc - best_cc) <= 1e-12 and env > best_env
            if better or tie:
                best_index = index
                best_cc = float(cc)
                best_env = float(env)
                best_wavelet = w
                best_synthetic = s_syn

        if best_index < 0 or best_wavelet is None or best_synthetic is None:
            raise ValueError(
                "constant-phase phase grid 中没有任何候选通过现有主峰位置约束。"
            )

        best_phase_deg = float(phase_grid[best_index])
        best_peak_ms = float(phase_peak_ms[best_index])

        if best_cc + 1e-12 < ricker_cc:
            raise ValueError(
                "constant-phase candidate 的直接 CC 低于 Ricker fallback："
                f"{best_cc:.6f} < {ricker_cc:.6f}。"
            )

        if verbose:
            print(
                "  [Constant-phase prior] "
                "statistical amplitude spectrum + constant phase search"
            )
            print(
                "  [Constant-phase prior] "
                f"spectral peak = {spectrum.spectral_peak_hz:.2f} Hz"
            )
            print(
                "  [Constant-phase prior] "
                f"auto band = {spectrum.band_low_hz:.2f} - "
                f"{spectrum.band_high_hz:.2f} Hz"
            )
            print(
                "  [Constant-phase prior] "
                f"phase = {best_phase_deg:+.1f} deg, "
                f"peak = {best_peak_ms:+.1f} ms"
            )
            print(
                "  [Constant-phase prior] "
                f"CC = {best_cc:.6f}, Ricker CC = {ricker_cc:.6f}"
            )

        return WaveletPriorResult(
            w=np.asarray(best_wavelet, dtype=float),
            source=source,
            s_syn=np.asarray(best_synthetic, dtype=float),
            cc=float(best_cc),
            env_cc=float(best_env),
            wavelet_length_pts=wavelet_length_pts,
            data_window_length=data_window_length,
            stationary_result=None,
            stationary_candidate_result=None,
            stationary_candidate_accepted=False,
            stationary_candidate_peak_ms=np.nan,
            stationary_rejection_code="not_attempted_constant_phase_policy",
            stationary_rejection_reason=(
                "after-DTW prior 使用 statistical constant-phase policy，"
                "未运行 free-form stationary LS candidate。"
            ),
            stationary_prior_source=source,
            constant_phase_deg=best_phase_deg,
            constant_phase_spectral_peak_hz=float(
                spectrum.spectral_peak_hz
            ),
            constant_phase_band_low_hz=float(spectrum.band_low_hz),
            constant_phase_band_high_hz=float(spectrum.band_high_hz),
            constant_phase_candidate_cc=float(best_cc),
            constant_phase_ricker_cc=float(ricker_cc),
            constant_phase_candidate_accepted=True,
            constant_phase_rejection_reason="",
            constant_phase_zero_phase_wavelet=zero_phase_wavelet,
            constant_phase_phase_grid_deg=phase_grid,
            constant_phase_phase_cc=phase_cc,
            constant_phase_spectrum_hz=spectrum.frequencies_hz,
            constant_phase_spectrum_amplitude=spectrum.amplitude,
        )

    except ValueError as exc:
        reason = str(exc)
        if verbose:
            print("  [warning] constant-phase prior rejected.")
            print(f"  [reason] {reason}")
            print(f"  [fallback] using {fallback_source}.")

        return WaveletPriorResult(
            w=ricker,
            source=fallback_source,
            s_syn=ricker_syn,
            cc=float(ricker_cc),
            env_cc=float(ricker_env),
            wavelet_length_pts=wavelet_length_pts,
            data_window_length=data_window_length,
            stationary_result=None,
            stationary_candidate_result=None,
            stationary_candidate_accepted=False,
            stationary_candidate_peak_ms=np.nan,
            stationary_rejection_code="constant_phase_rejected",
            stationary_rejection_reason=reason,
            stationary_prior_source=fallback_source,
            constant_phase_ricker_cc=float(ricker_cc),
            constant_phase_candidate_accepted=False,
            constant_phase_rejection_reason=reason,
        )
