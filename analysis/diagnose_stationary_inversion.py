# -*- coding: utf-8 -*-
"""
analysis/diagnose_stationary_inversion.py

只读诊断模块：定位全局平稳子波反演不稳定、边缘能量异常的主要来源。

设计原则
--------
1. 不修改任何现有反演算法、配置文件或结果文件。
2. 优先读取主实验已经保存的 result_bundle.npz：
       r_work_final = DTW 后反射系数
       obs_work     = 观测地震
       t_work       = 工作时间轴
3. 全局长度扫描直接调用项目现有 stationary_wavelet_inversion，
   peak_lock=False 仅用于取得被 QC 拒绝前的原始候选，不改变求解器。
4. 局部窗口诊断直接复用项目已有 _solve_regularized_window，
   关闭 prior / time coupling，只考察局部数据本身能否稳定确定子波。
5. 所有输出写到独立 stationary_diagnostics 目录。

主要输出
--------
01_global_wavelets_by_length.png
02_global_common_support_abs_correlation.png
03_raw_R_singular_spectra.png
04_augmented_A_singular_spectra.png
05_regularization_hessian_ratios.png
06_local_wavelets.png
07_local_wavelet_abs_correlation.png
08_small_singular_vectors_Lxxx.png

global_length_diagnostics.csv
global_common_support_correlation.csv
local_window_diagnostics.csv
local_wavelet_correlation.csv
diagnostic_summary.json
diagnostic_arrays.npz

推荐运行
--------
python analysis/diagnose_stationary_inversion.py \
    --result-bundle path/to/result_bundle.npz

也可以显式指定配置：
python analysis/diagnose_stationary_inversion.py \
    --result-bundle path/to/result_bundle.npz \
    --config path/to/config_used.yaml \
    --lengths 65 79 101 129 201
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from configs.config_loader import load_config
from core.forward_operator import (
    build_local_convolution_matrix,
    wavelet_lag_axis_ms,
)
from utils.wavelet_inversion_robust import (
    _safe_svd_solve,
    _second_derivative_matrix,
    _solve_regularized_window,
    build_edge_penalty_weights,
    stationary_wavelet_inversion,
)


EPS = 1e-12


@dataclass(frozen=True)
class StationaryParams:
    alignment: str
    mu1: float
    mu2: float
    mu_dc: float
    damping_ratio: float
    svd_cutoff_ratio: float
    mu_edge: float
    edge_fraction: float
    edge_taper: str


@dataclass
class GlobalLengthResult:
    wavelet_length: int
    wavelet: np.ndarray
    lag_ms: np.ndarray
    R_singular_values: np.ndarray
    A_singular_values: np.ndarray
    peak_lag_ms: float
    edge_energy_ratio: float
    side_lobe_ratio: float
    wavelet_l2: float
    raw_effective_rank: int
    raw_effective_rank_ratio: float
    raw_condition_ratio: float
    augmented_effective_rank: int
    augmented_effective_rank_ratio: float
    augmented_condition_ratio: float
    data_hessian_strength: float
    ridge_hessian_ratio: float
    smooth_hessian_ratio: float
    dc_hessian_ratio: float
    edge_hessian_ratio: float
    total_regularization_hessian_ratio: float
    objective_data: float
    objective_ridge: float
    objective_smooth: float
    objective_dc: float
    objective_edge: float
    objective_regularization_over_data: float
    damping_probe_relative_change: float


@dataclass
class LocalWindowResult:
    center_index: int
    center_time_s: float
    window_start_s: float
    window_end_s: float
    r_energy: float
    s_energy: float
    data_scale: float
    wavelet: np.ndarray
    lag_ms: np.ndarray
    peak_lag_ms: float
    edge_energy_ratio: float
    side_lobe_ratio: float
    wavelet_l2: float
    effective_rank: int
    effective_rank_ratio: float
    condition_ratio: float
    residual_rms: float
    polarity_flipped_for_plot: bool = False


def _make_odd(n: int) -> int:
    n = int(n)
    return n if n % 2 == 1 else n + 1


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size != y.size or x.size < 2:
        return float("nan")

    x0 = x - np.mean(x)
    y0 = y - np.mean(y)
    denom = np.linalg.norm(x0) * np.linalg.norm(y0)

    if denom <= EPS:
        return float("nan")

    return float(np.dot(x0, y0) / denom)


def _finite_median(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    return float(np.median(array)) if array.size else float("nan")


def _finite_min(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    return float(np.min(array)) if array.size else float("nan")


def _normalize_for_shape(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=float).copy()
    scale = float(np.max(np.abs(w)))
    if scale <= EPS:
        return w
    return w / scale


def _second_derivative_matrix_public_equivalent(n: int) -> np.ndarray:
    return _second_derivative_matrix(int(n))


def _build_global_convolution_matrix(
    r_time: np.ndarray,
    wavelet_length: int,
    alignment: str,
) -> np.ndarray:
    """
    与 stationary_wavelet_inversion 中 R[row, col] = r[row - lag] 完全相同，
    这里只采用按列填充以减少 Python 双重循环开销。
    """
    r_time = np.asarray(r_time, dtype=float).ravel()
    n_time = r_time.size
    wavelet_length = _make_odd(wavelet_length)

    if alignment == "center":
        lags = np.arange(wavelet_length, dtype=int) - wavelet_length // 2
    elif alignment == "causal":
        lags = np.arange(wavelet_length, dtype=int)
    else:
        raise ValueError("alignment 必须为 'center' 或 'causal'。")

    R = np.zeros((n_time, wavelet_length), dtype=float)

    for col, lag in enumerate(lags):
        lag = int(lag)

        if lag >= 0:
            if lag < n_time:
                R[lag:, col] = r_time[: n_time - lag]
        else:
            advance = -lag
            if advance < n_time:
                R[: n_time - advance, col] = r_time[advance:]

    return R


def _extract_centered_window(
    x: np.ndarray,
    center_index: int,
    window_length: int,
) -> np.ndarray:
    x = np.asarray(x, dtype=float).ravel()
    half = window_length // 2
    start = int(center_index) - half
    end = int(center_index) + half + 1

    if start < 0 or end > x.size:
        raise ValueError("局部诊断窗口越过数据边界。")

    return x[start:end].copy()


def _wavelet_shape_metrics(
    w: np.ndarray,
    dt: float,
    alignment: str,
    edge_fraction: float = 0.15,
    side_lobe_guard_ms: float = 20.0,
) -> tuple[float, float, float, float]:
    w = np.asarray(w, dtype=float).ravel()
    lag_ms = wavelet_lag_axis_ms(w.size, dt, alignment)

    peak_index = int(np.argmax(np.abs(w)))
    peak_lag_ms = float(lag_ms[peak_index])

    n_edge = max(1, int(np.ceil(float(edge_fraction) * w.size)))
    edge_mask = np.zeros(w.size, dtype=bool)
    edge_mask[:n_edge] = True
    edge_mask[-n_edge:] = True

    total_energy = float(np.sum(w ** 2)) + EPS
    edge_energy_ratio = float(np.sum(w[edge_mask] ** 2) / total_energy)

    if alignment == "center":
        main_mask = np.abs(lag_ms) <= float(side_lobe_guard_ms)
    else:
        main_mask = lag_ms <= float(side_lobe_guard_ms)

    outside_mask = ~main_mask
    main_peak = float(np.max(np.abs(w[main_mask]))) if np.any(main_mask) else 0.0
    outside_peak = (
        float(np.max(np.abs(w[outside_mask])))
        if np.any(outside_mask)
        else 0.0
    )
    side_lobe_ratio = outside_peak / (main_peak + EPS)

    return (
        peak_lag_ms,
        edge_energy_ratio,
        side_lobe_ratio,
        float(np.linalg.norm(w)),
    )


def _effective_rank_and_condition(
    singular_values: np.ndarray,
    cutoff_ratio: float,
) -> tuple[int, float, float]:
    s = np.asarray(singular_values, dtype=float).ravel()

    if s.size == 0 or s[0] <= EPS:
        return 0, 0.0, 0.0

    effective_rank = int(np.sum(s > float(cutoff_ratio) * s[0]))
    effective_rank_ratio = float(effective_rank / s.size)
    condition_ratio = float(s[-1] / (s[0] + EPS))

    return effective_rank, effective_rank_ratio, condition_ratio


def _build_regularized_system(
    R: np.ndarray,
    s_zero_mean: np.ndarray,
    params: StationaryParams,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    float,
]:
    wavelet_length = R.shape[1]
    D = _second_derivative_matrix_public_equivalent(wavelet_length)
    I = np.eye(wavelet_length, dtype=float)
    C_dc = (
        np.ones((1, wavelet_length), dtype=float)
        / np.sqrt(wavelet_length)
    )

    data_scale = max(
        float(np.sum(s_zero_mean ** 2) / s_zero_mean.size),
        EPS,
    )

    A_blocks = [
        R,
        np.sqrt(params.mu1 * data_scale) * I,
        np.sqrt(params.mu2 * data_scale) * D,
        np.sqrt(params.mu_dc * data_scale) * C_dc,
    ]

    b_blocks = [
        s_zero_mean,
        np.zeros(wavelet_length, dtype=float),
        np.zeros(D.shape[0], dtype=float),
        np.zeros(1, dtype=float),
    ]

    edge_weights: np.ndarray | None = None

    if params.mu_edge > 0.0:
        edge_weights = build_edge_penalty_weights(
            wavelet_length=wavelet_length,
            edge_fraction=params.edge_fraction,
            taper=params.edge_taper,
        )
        E_edge = np.diag(edge_weights)
        A_blocks.append(
            np.sqrt(params.mu_edge * data_scale) * E_edge
        )
        b_blocks.append(np.zeros(wavelet_length, dtype=float))

    A_aug = np.vstack(A_blocks)
    b_aug = np.concatenate(b_blocks)

    return A_aug, b_aug, D, I, C_dc, edge_weights, data_scale


def _regularization_strength_diagnostics(
    R: np.ndarray,
    w: np.ndarray,
    s_zero_mean: np.ndarray,
    D: np.ndarray,
    I: np.ndarray,
    C_dc: np.ndarray,
    edge_weights: np.ndarray | None,
    data_scale: float,
    params: StationaryParams,
) -> dict[str, float]:
    H_data = R.T @ R
    data_strength = float(np.linalg.norm(H_data, ord=2))

    H_ridge = params.mu1 * data_scale * I
    H_smooth = params.mu2 * data_scale * (D.T @ D)
    H_dc = params.mu_dc * data_scale * (C_dc.T @ C_dc)

    if edge_weights is not None and params.mu_edge > 0.0:
        E = np.diag(edge_weights)
        H_edge = params.mu_edge * data_scale * (E.T @ E)
    else:
        H_edge = np.zeros_like(I)

    H_reg = H_ridge + H_smooth + H_dc + H_edge

    denom_h = data_strength + EPS

    residual = R @ w - s_zero_mean
    objective_data = float(np.sum(residual ** 2))
    objective_ridge = float(
        params.mu1 * data_scale * np.sum(w ** 2)
    )
    objective_smooth = float(
        params.mu2 * data_scale * np.sum((D @ w) ** 2)
    )
    objective_dc = float(
        params.mu_dc * data_scale * np.sum((C_dc @ w) ** 2)
    )

    if edge_weights is not None and params.mu_edge > 0.0:
        objective_edge = float(
            params.mu_edge
            * data_scale
            * np.sum((edge_weights * w) ** 2)
        )
    else:
        objective_edge = 0.0

    objective_reg = (
        objective_ridge
        + objective_smooth
        + objective_dc
        + objective_edge
    )

    return {
        "data_hessian_strength": data_strength,
        "ridge_hessian_ratio": (
            float(np.linalg.norm(H_ridge, ord=2)) / denom_h
        ),
        "smooth_hessian_ratio": (
            float(np.linalg.norm(H_smooth, ord=2)) / denom_h
        ),
        "dc_hessian_ratio": (
            float(np.linalg.norm(H_dc, ord=2)) / denom_h
        ),
        "edge_hessian_ratio": (
            float(np.linalg.norm(H_edge, ord=2)) / denom_h
        ),
        "total_regularization_hessian_ratio": (
            float(np.linalg.norm(H_reg, ord=2)) / denom_h
        ),
        "objective_data": objective_data,
        "objective_ridge": objective_ridge,
        "objective_smooth": objective_smooth,
        "objective_dc": objective_dc,
        "objective_edge": objective_edge,
        "objective_regularization_over_data": (
            objective_reg / (objective_data + EPS)
        ),
    }


def _solver_damping_probe(
    A_aug: np.ndarray,
    b_aug: np.ndarray,
    svd_cutoff_ratio: float,
) -> float:
    """
    不改求解器，只用同一个线性系统探测 damping_ratio 是否影响结果。

    使用 0 与 0.25 两个阻尼比例。若当前 _safe_svd_solve 完全忽略 damping，
    两个结果会在机器精度内相同。
    """
    x_no_damping = _safe_svd_solve(
        A=A_aug,
        b=b_aug,
        damping_ratio=0.0,
        svd_cutoff_ratio=svd_cutoff_ratio,
    )
    x_probe = _safe_svd_solve(
        A=A_aug,
        b=b_aug,
        damping_ratio=0.25,
        svd_cutoff_ratio=svd_cutoff_ratio,
    )

    return float(
        np.linalg.norm(x_probe - x_no_damping)
        / (np.linalg.norm(x_no_damping) + EPS)
    )


def _stationary_call_kwargs(
    params: StationaryParams,
    wavelet_length: int,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "wavelet_length": int(wavelet_length),
        "mu1": params.mu1,
        "mu2": params.mu2,
        "mu_dc": params.mu_dc,
        "damping_ratio": params.damping_ratio,
        "svd_cutoff_ratio": params.svd_cutoff_ratio,
        "peak_lock": False,
        "wavelet_alignment": params.alignment,
    }

    signature = inspect.signature(stationary_wavelet_inversion)

    if "mu_edge" in signature.parameters:
        kwargs["mu_edge"] = params.mu_edge

    if "edge_fraction" in signature.parameters:
        kwargs["edge_fraction"] = params.edge_fraction

    if "edge_taper" in signature.parameters:
        kwargs["edge_taper"] = params.edge_taper

    return kwargs


def _run_global_length_diagnostics(
    r_time: np.ndarray,
    s_obs: np.ndarray,
    dt: float,
    params: StationaryParams,
    lengths: list[int],
) -> dict[int, GlobalLengthResult]:
    s_zero = np.asarray(s_obs, dtype=float).ravel()
    s_zero = s_zero - np.mean(s_zero)

    results: dict[int, GlobalLengthResult] = {}

    for requested_length in lengths:
        wavelet_length = _make_odd(int(requested_length))

        if wavelet_length > r_time.size:
            raise ValueError(
                f"wavelet_length={wavelet_length} 大于数据长度 {r_time.size}。"
            )

        w = stationary_wavelet_inversion(
            r_time=np.asarray(r_time, dtype=float),
            s_obs=np.asarray(s_obs, dtype=float),
            **_stationary_call_kwargs(
                params=params,
                wavelet_length=wavelet_length,
            ),
        )

        R = _build_global_convolution_matrix(
            r_time=r_time,
            wavelet_length=wavelet_length,
            alignment=params.alignment,
        )

        (
            A_aug,
            b_aug,
            D,
            I,
            C_dc,
            edge_weights,
            data_scale,
        ) = _build_regularized_system(
            R=R,
            s_zero_mean=s_zero,
            params=params,
        )

        s_R = np.linalg.svd(R, compute_uv=False)
        s_A = np.linalg.svd(A_aug, compute_uv=False)

        raw_rank, raw_rank_ratio, raw_cond_ratio = (
            _effective_rank_and_condition(
                s_R,
                cutoff_ratio=params.svd_cutoff_ratio,
            )
        )
        aug_rank, aug_rank_ratio, aug_cond_ratio = (
            _effective_rank_and_condition(
                s_A,
                cutoff_ratio=params.svd_cutoff_ratio,
            )
        )

        shape = _wavelet_shape_metrics(
            w=w,
            dt=dt,
            alignment=params.alignment,
        )

        strength = _regularization_strength_diagnostics(
            R=R,
            w=w,
            s_zero_mean=s_zero,
            D=D,
            I=I,
            C_dc=C_dc,
            edge_weights=edge_weights,
            data_scale=data_scale,
            params=params,
        )

        damping_probe = _solver_damping_probe(
            A_aug=A_aug,
            b_aug=b_aug,
            svd_cutoff_ratio=params.svd_cutoff_ratio,
        )

        results[wavelet_length] = GlobalLengthResult(
            wavelet_length=wavelet_length,
            wavelet=np.asarray(w, dtype=float),
            lag_ms=wavelet_lag_axis_ms(
                wavelet_length,
                dt,
                params.alignment,
            ),
            R_singular_values=s_R,
            A_singular_values=s_A,
            peak_lag_ms=shape[0],
            edge_energy_ratio=shape[1],
            side_lobe_ratio=shape[2],
            wavelet_l2=shape[3],
            raw_effective_rank=raw_rank,
            raw_effective_rank_ratio=raw_rank_ratio,
            raw_condition_ratio=raw_cond_ratio,
            augmented_effective_rank=aug_rank,
            augmented_effective_rank_ratio=aug_rank_ratio,
            augmented_condition_ratio=aug_cond_ratio,
            damping_probe_relative_change=damping_probe,
            **strength,
        )

        print(
            "[Global] "
            f"L={wavelet_length:3d} | "
            f"raw_rank={raw_rank:3d}/{wavelet_length:<3d} | "
            f"raw_cond_ratio={raw_cond_ratio:.3e} | "
            f"reg/data={strength['total_regularization_hessian_ratio']:.3e} | "
            f"peak={shape[0]:+.1f} ms | "
            f"edge={shape[1]:.3f} | "
            f"side={shape[2]:.3f}"
        )

    return results


def _common_axis_ms(
    results: dict[int, GlobalLengthResult],
    dt: float,
    alignment: str,
    common_half_ms: float,
) -> np.ndarray:
    dt_ms = float(dt) * 1000.0

    if alignment == "center":
        available = min(
            min(
                abs(float(result.lag_ms[0])),
                abs(float(result.lag_ms[-1])),
            )
            for result in results.values()
        )
        half_ms = min(float(common_half_ms), available)
        half_samples = max(1, int(np.floor(half_ms / dt_ms)))
        return np.arange(-half_samples, half_samples + 1) * dt_ms

    available = min(float(result.lag_ms[-1]) for result in results.values())
    max_ms = min(float(common_half_ms), available)
    n_samples = max(2, int(np.floor(max_ms / dt_ms)) + 1)
    return np.arange(n_samples) * dt_ms


def _interpolate_wavelet_to_axis(
    result: GlobalLengthResult,
    target_axis_ms: np.ndarray,
) -> np.ndarray:
    return np.interp(
        target_axis_ms,
        result.lag_ms,
        result.wavelet,
    )


def _correlation_matrix_from_wavelets(
    wavelets: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    n = len(wavelets)
    signed = np.full((n, n), np.nan, dtype=float)

    for i in range(n):
        for j in range(n):
            signed[i, j] = _safe_corr(wavelets[i], wavelets[j])

    return signed, np.abs(signed)


def _global_common_support_correlations(
    results: dict[int, GlobalLengthResult],
    dt: float,
    alignment: str,
    common_half_ms: float,
) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray]:
    lengths = sorted(results)
    axis = _common_axis_ms(
        results=results,
        dt=dt,
        alignment=alignment,
        common_half_ms=common_half_ms,
    )

    wavelets = [
        _interpolate_wavelet_to_axis(results[length], axis)
        for length in lengths
    ]

    signed, absolute = _correlation_matrix_from_wavelets(wavelets)

    return lengths, axis, signed, absolute


def _select_local_centers(
    r_time: np.ndarray,
    data_window_length: int,
    n_windows: int,
) -> list[int]:
    r_time = np.asarray(r_time, dtype=float).ravel()
    n_time = r_time.size
    half = data_window_length // 2

    valid_start = half
    valid_stop = n_time - half

    if valid_stop <= valid_start:
        raise ValueError(
            "数据长度不足以容纳局部诊断窗口。"
        )

    n_windows = int(max(1, n_windows))
    valid_indices = np.arange(valid_start, valid_stop, dtype=int)

    energy_curve = np.convolve(
        r_time ** 2,
        np.ones(data_window_length, dtype=float),
        mode="same",
    )

    segments = np.array_split(valid_indices, n_windows)
    centers: list[int] = []

    for segment in segments:
        if segment.size == 0:
            continue

        segment_energy = energy_curve[segment]
        local_index = int(np.argmax(segment_energy))
        centers.append(int(segment[local_index]))

    return centers


def _run_local_window_diagnostics(
    r_time: np.ndarray,
    s_obs: np.ndarray,
    t_work: np.ndarray,
    dt: float,
    params: StationaryParams,
    wavelet_length: int,
    data_window_factor: float,
    n_windows: int,
) -> list[LocalWindowResult]:
    wavelet_length = _make_odd(wavelet_length)

    data_window_length = _make_odd(
        int(round(float(data_window_factor) * wavelet_length))
    )

    if data_window_length <= wavelet_length:
        data_window_length = _make_odd(wavelet_length + 2)

    if data_window_length >= len(r_time):
        data_window_length = _make_odd(len(r_time) - 2)

    if data_window_length <= wavelet_length:
        raise ValueError(
            "局部窗口无法同时满足 data_window_length > wavelet_length "
            "和小于数据总长度。"
        )

    centers = _select_local_centers(
        r_time=r_time,
        data_window_length=data_window_length,
        n_windows=n_windows,
    )

    s_zero = np.asarray(s_obs, dtype=float).ravel()
    s_zero = s_zero - np.mean(s_zero)

    global_data_scale = max(
        float(np.mean(s_zero ** 2)),
        EPS,
    )

    D = _second_derivative_matrix_public_equivalent(wavelet_length)
    I = np.eye(wavelet_length, dtype=float)
    C_dc = (
        np.ones((1, wavelet_length), dtype=float)
        / np.sqrt(wavelet_length)
    )

    edge_weights: np.ndarray | None = None

    if params.mu_edge > 0.0:
        edge_weights = build_edge_penalty_weights(
            wavelet_length=wavelet_length,
            edge_fraction=params.edge_fraction,
            taper=params.edge_taper,
        )

    solver_signature = inspect.signature(_solve_regularized_window)
    supports_edge = (
        "mu_edge" in solver_signature.parameters
        and "edge_weights" in solver_signature.parameters
    )

    results: list[LocalWindowResult] = []
    half = data_window_length // 2

    for center in centers:
        s_win = _extract_centered_window(
            s_zero,
            center,
            data_window_length,
        )
        r_win = _extract_centered_window(
            r_time,
            center,
            data_window_length,
        )

        R = build_local_convolution_matrix(
            r_time=np.asarray(r_time, dtype=float),
            center_index=int(center),
            data_window_length=data_window_length,
            wavelet_length=wavelet_length,
            alignment=params.alignment,
        )

        s_energy = float(np.sum(s_win ** 2))
        r_energy = float(np.sum(r_win ** 2))

        local_data_scale = s_energy / data_window_length
        data_scale = max(
            local_data_scale,
            0.25 * global_data_scale,
            EPS,
        )

        solve_kwargs: dict[str, Any] = {
            "R": R,
            "s_win": s_win,
            "data_scale": data_scale,
            "I": I,
            "D": D,
            "C_dc": C_dc,
            "mu1": params.mu1,
            "mu2": params.mu2,
            "mu_dc": params.mu_dc,
            "prior_i": None,
            "mu_prior": 0.0,
            "previous_wavelet": None,
            "mu_time": 0.0,
            "damping_ratio": params.damping_ratio,
            "svd_cutoff_ratio": params.svd_cutoff_ratio,
            "sample_weights": None,
        }

        if supports_edge:
            solve_kwargs["edge_weights"] = edge_weights
            solve_kwargs["mu_edge"] = params.mu_edge
        else:
            solve_kwargs["edge_weights"] = None
            solve_kwargs["mu_edge"] = 0.0

        w = _solve_regularized_window(**solve_kwargs)
        w = np.asarray(w, dtype=float)
        w = w - np.mean(w)

        s_R = np.linalg.svd(R, compute_uv=False)
        effective_rank, rank_ratio, condition_ratio = (
            _effective_rank_and_condition(
                s_R,
                cutoff_ratio=params.svd_cutoff_ratio,
            )
        )

        residual = s_win - R @ w
        residual_rms = float(np.sqrt(np.mean(residual ** 2)))

        shape = _wavelet_shape_metrics(
            w=w,
            dt=dt,
            alignment=params.alignment,
        )

        start_index = center - half
        end_index = center + half

        results.append(
            LocalWindowResult(
                center_index=int(center),
                center_time_s=float(t_work[center]),
                window_start_s=float(t_work[start_index]),
                window_end_s=float(t_work[end_index]),
                r_energy=r_energy,
                s_energy=s_energy,
                data_scale=data_scale,
                wavelet=w,
                lag_ms=wavelet_lag_axis_ms(
                    wavelet_length,
                    dt,
                    params.alignment,
                ),
                peak_lag_ms=shape[0],
                edge_energy_ratio=shape[1],
                side_lobe_ratio=shape[2],
                wavelet_l2=shape[3],
                effective_rank=effective_rank,
                effective_rank_ratio=rank_ratio,
                condition_ratio=condition_ratio,
                residual_rms=residual_rms,
            )
        )

        print(
            "[Local] "
            f"t={t_work[center]:.3f} s | "
            f"rank={effective_rank:3d}/{wavelet_length:<3d} | "
            f"cond_ratio={condition_ratio:.3e} | "
            f"peak={shape[0]:+.1f} ms | "
            f"edge={shape[1]:.3f} | "
            f"side={shape[2]:.3f} | "
            f"res_rms={residual_rms:.3e}"
        )

    return results


def _align_local_polarities_for_plot(
    results: list[LocalWindowResult],
) -> list[np.ndarray]:
    if not results:
        return []

    reference_index = len(results) // 2
    reference = _normalize_for_shape(results[reference_index].wavelet)

    aligned: list[np.ndarray] = []

    for result in results:
        w = _normalize_for_shape(result.wavelet)
        flip = bool(np.dot(w, reference) < 0.0)

        if flip:
            w = -w

        result.polarity_flipped_for_plot = flip
        aligned.append(w)

    return aligned


def _local_correlation_matrices(
    results: list[LocalWindowResult],
) -> tuple[np.ndarray, np.ndarray]:
    wavelets = [
        _normalize_for_shape(result.wavelet)
        for result in results
    ]
    return _correlation_matrix_from_wavelets(wavelets)


def _upper_triangle_values(
    matrix: np.ndarray,
) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("相关矩阵必须为方阵。")

    if matrix.shape[0] < 2:
        return np.array([], dtype=float)

    indices = np.triu_indices(matrix.shape[0], k=1)
    return matrix[indices]


def _write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    if fieldnames is None:
        fieldnames = list(rows[0].keys())

    with open(path, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def _save_global_table(
    results: dict[int, GlobalLengthResult],
    path: Path,
) -> None:
    rows: list[dict[str, Any]] = []

    for length in sorted(results):
        result = results[length]
        row = asdict(result)

        for key in (
            "wavelet",
            "lag_ms",
            "R_singular_values",
            "A_singular_values",
        ):
            row.pop(key, None)

        rows.append(row)

    _write_csv(path, rows)


def _save_global_correlation_table(
    lengths: list[int],
    signed: np.ndarray,
    absolute: np.ndarray,
    path: Path,
) -> None:
    rows: list[dict[str, Any]] = []

    for i, length_i in enumerate(lengths):
        for j, length_j in enumerate(lengths):
            rows.append(
                {
                    "length_i": length_i,
                    "length_j": length_j,
                    "signed_correlation": float(signed[i, j]),
                    "absolute_correlation": float(absolute[i, j]),
                }
            )

    _write_csv(path, rows)


def _save_local_table(
    results: list[LocalWindowResult],
    path: Path,
) -> None:
    rows: list[dict[str, Any]] = []

    for result in results:
        row = asdict(result)
        row.pop("wavelet", None)
        row.pop("lag_ms", None)
        rows.append(row)

    _write_csv(path, rows)


def _save_local_correlation_table(
    results: list[LocalWindowResult],
    signed: np.ndarray,
    absolute: np.ndarray,
    path: Path,
) -> None:
    rows: list[dict[str, Any]] = []

    for i, result_i in enumerate(results):
        for j, result_j in enumerate(results):
            rows.append(
                {
                    "time_i_s": result_i.center_time_s,
                    "time_j_s": result_j.center_time_s,
                    "signed_correlation": float(signed[i, j]),
                    "absolute_correlation": float(absolute[i, j]),
                }
            )

    _write_csv(path, rows)


def _plot_global_wavelets(
    results: dict[int, GlobalLengthResult],
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))

    for length in sorted(results):
        result = results[length]
        ax.plot(
            result.lag_ms,
            _normalize_for_shape(result.wavelet),
            linewidth=1.8,
            label=f"L={length}",
        )

    ax.axhline(0.0, linewidth=0.8, linestyle="--")
    ax.axvline(0.0, linewidth=0.8, linestyle="--")
    ax.set_xlabel("Lag (ms)")
    ax.set_ylabel("Normalized amplitude")
    ax.set_title("Raw stationary candidates by wavelet length")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_heatmap(
    matrix: np.ndarray,
    labels: list[str],
    path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))

    image = ax.imshow(
        matrix,
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
    )

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title(title)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            text = "nan" if not np.isfinite(value) else f"{value:.2f}"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=9,
            )

    fig.colorbar(image, ax=ax, label="|correlation|")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_singular_spectra(
    results: dict[int, GlobalLengthResult],
    attribute_name: str,
    path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 7))

    for length in sorted(results):
        singular_values = np.asarray(
            getattr(results[length], attribute_name),
            dtype=float,
        )

        if singular_values.size == 0 or singular_values[0] <= EPS:
            continue

        normalized = singular_values / singular_values[0]

        ax.semilogy(
            np.arange(1, normalized.size + 1),
            normalized,
            linewidth=1.8,
            label=f"L={length}",
        )

    ax.set_xlabel("Singular-value index")
    ax.set_ylabel(r"$\sigma_i / \sigma_1$")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_regularization_ratios(
    results: dict[int, GlobalLengthResult],
    path: Path,
) -> None:
    lengths = sorted(results)

    curves = {
        "ridge": [
            results[length].ridge_hessian_ratio
            for length in lengths
        ],
        "smooth": [
            results[length].smooth_hessian_ratio
            for length in lengths
        ],
        "dc": [
            results[length].dc_hessian_ratio
            for length in lengths
        ],
        "edge": [
            results[length].edge_hessian_ratio
            for length in lengths
        ],
        "total": [
            results[length].total_regularization_hessian_ratio
            for length in lengths
        ],
    }

    fig, ax = plt.subplots(figsize=(11, 7))

    for name, values in curves.items():
        values_array = np.maximum(
            np.asarray(values, dtype=float),
            1e-16,
        )
        ax.semilogy(
            lengths,
            values_array,
            marker="o",
            linewidth=1.8,
            label=name,
        )

    ax.set_xlabel("Wavelet length (samples)")
    ax.set_ylabel(
        "Regularization Hessian norm / data Hessian norm"
    )
    ax.set_title("Actual regularization strength relative to data term")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_local_wavelets(
    results: list[LocalWindowResult],
    path: Path,
) -> None:
    aligned = _align_local_polarities_for_plot(results)

    fig, ax = plt.subplots(figsize=(12, 8))
    offset_step = 2.4

    for index, (result, w) in enumerate(zip(results, aligned)):
        offset = index * offset_step
        ax.plot(
            result.lag_ms,
            w + offset,
            linewidth=1.8,
        )
        ax.axhline(
            offset,
            linewidth=0.6,
            linestyle="--",
            alpha=0.4,
        )
        ax.text(
            float(result.lag_ms[0]),
            offset + 0.9,
            f"t={result.center_time_s:.3f}s",
            fontsize=9,
        )

    ax.axvline(0.0, linewidth=0.8, linestyle="--")
    ax.set_xlabel("Lag (ms)")
    ax.set_ylabel("Normalized wavelets + vertical offset")
    ax.set_title(
        "Independent local-window wavelets "
        "(polarity aligned for visualization only)"
    )
    ax.grid(alpha=0.20)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_small_singular_vectors(
    r_time: np.ndarray,
    dt: float,
    alignment: str,
    wavelet_length: int,
    path: Path,
    n_vectors: int = 5,
) -> None:
    R = _build_global_convolution_matrix(
        r_time=r_time,
        wavelet_length=wavelet_length,
        alignment=alignment,
    )

    _, singular_values, Vt = np.linalg.svd(
        R,
        full_matrices=False,
    )

    count = min(int(n_vectors), Vt.shape[0])
    lag_ms = wavelet_lag_axis_ms(
        wavelet_length,
        dt,
        alignment,
    )

    fig, ax = plt.subplots(figsize=(12, 7))

    for offset in range(1, count + 1):
        vector = Vt[-offset]
        vector = _normalize_for_shape(vector)

        sigma_ratio = (
            singular_values[-offset]
            / (singular_values[0] + EPS)
        )

        ax.plot(
            lag_ms,
            vector,
            linewidth=1.6,
            label=(
                f"v[-{offset}], "
                f"sigma/sigma1={sigma_ratio:.2e}"
            ),
        )

    ax.axhline(0.0, linewidth=0.8, linestyle="--")
    ax.axvline(0.0, linewidth=0.8, linestyle="--")
    ax.set_xlabel("Lag (ms)")
    ax.set_ylabel("Normalized right singular vector")
    ax.set_title(
        f"Weakly constrained right-singular directions, L={wavelet_length}"
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_npz(
    path: Path,
    global_results: dict[int, GlobalLengthResult],
    common_axis_ms: np.ndarray,
    global_signed_corr: np.ndarray,
    global_abs_corr: np.ndarray,
    local_results: list[LocalWindowResult],
    local_signed_corr: np.ndarray,
    local_abs_corr: np.ndarray,
) -> None:
    arrays: dict[str, np.ndarray] = {
        "common_axis_ms": np.asarray(common_axis_ms),
        "global_signed_correlation": np.asarray(global_signed_corr),
        "global_absolute_correlation": np.asarray(global_abs_corr),
        "local_signed_correlation": np.asarray(local_signed_corr),
        "local_absolute_correlation": np.asarray(local_abs_corr),
    }

    for length, result in global_results.items():
        arrays[f"wavelet_L{length}"] = result.wavelet
        arrays[f"lag_ms_L{length}"] = result.lag_ms
        arrays[f"singular_R_L{length}"] = result.R_singular_values
        arrays[f"singular_A_L{length}"] = result.A_singular_values

    for index, result in enumerate(local_results):
        arrays[f"local_wavelet_{index:02d}"] = result.wavelet
        arrays[f"local_lag_ms_{index:02d}"] = result.lag_ms
        arrays[f"local_center_time_s_{index:02d}"] = np.asarray(
            result.center_time_s
        )

    np.savez_compressed(path, **arrays)


def _build_summary(
    params: StationaryParams,
    global_results: dict[int, GlobalLengthResult],
    global_abs_corr: np.ndarray,
    local_results: list[LocalWindowResult],
    local_abs_corr: np.ndarray,
    local_wavelet_length: int,
    local_data_window_factor: float,
) -> dict[str, Any]:
    global_pairs = _upper_triangle_values(global_abs_corr)
    local_pairs = _upper_triangle_values(local_abs_corr)

    global_median_corr = _finite_median(global_pairs)
    global_min_corr = _finite_min(global_pairs)
    local_median_corr = _finite_median(local_pairs)
    local_min_corr = _finite_min(local_pairs)

    raw_rank_ratios = [
        result.raw_effective_rank_ratio
        for result in global_results.values()
    ]
    raw_condition_ratios = [
        result.raw_condition_ratio
        for result in global_results.values()
    ]
    reg_ratios = [
        result.total_regularization_hessian_ratio
        for result in global_results.values()
    ]
    objective_ratios = [
        result.objective_regularization_over_data
        for result in global_results.values()
    ]
    damping_probes = [
        result.damping_probe_relative_change
        for result in global_results.values()
    ]

    local_condition_ratios = [
        result.condition_ratio
        for result in local_results
    ]
    local_rank_ratios = [
        result.effective_rank_ratio
        for result in local_results
    ]

    median_global_corr = global_median_corr
    median_raw_rank_ratio = _finite_median(raw_rank_ratios)
    median_raw_cond_ratio = _finite_median(raw_condition_ratios)
    median_reg_ratio = _finite_median(reg_ratios)
    median_objective_ratio = _finite_median(objective_ratios)
    max_damping_probe = (
        float(np.nanmax(damping_probes))
        if damping_probes
        else float("nan")
    )

    flags = {
        "length_sensitive_core_solution": bool(
            np.isfinite(median_global_corr)
            and median_global_corr < 0.90
        ),
        "raw_operator_ill_conditioning_likely": bool(
            (
                np.isfinite(median_raw_cond_ratio)
                and median_raw_cond_ratio < 1e-3
            )
            or (
                np.isfinite(median_raw_rank_ratio)
                and median_raw_rank_ratio < 0.80
            )
        ),
        "regularization_hessian_very_weak_relative_to_data": bool(
            np.isfinite(median_reg_ratio)
            and median_reg_ratio < 1e-3
        ),
        "independent_local_wavelets_vary_strongly": bool(
            np.isfinite(local_median_corr)
            and local_median_corr < 0.85
        ),
        "damping_parameter_appears_numerically_inactive": bool(
            np.isfinite(max_damping_probe)
            and max_damping_probe < 1e-12
        ),
    }

    interpretation: list[str] = []

    if flags["length_sensitive_core_solution"]:
        interpretation.append(
            "不同 wavelet length 在共同中心区间内仍明显改变形状；"
            "这更支持逆解对参数化敏感，而不是单纯只有尾部 support 过长。"
        )
    else:
        interpretation.append(
            "不同 wavelet length 的共同中心区间总体较稳定；"
            "若边缘仍异常，应重点检查尾部自由度和边界附近弱约束方向。"
        )

    if flags["raw_operator_ill_conditioning_likely"]:
        interpretation.append(
            "原始卷积矩阵 R 的有效秩比例或条件比例提示明显病态；"
            "小奇异值对应方向可能允许大幅改变子波而只弱改变 synthetic。"
        )
    else:
        interpretation.append(
            "按当前启发式阈值，R 的奇异谱没有表现出极端病态；"
            "应进一步关注模型失配、时变性或正则化尺度。"
        )

    if flags["regularization_hessian_very_weak_relative_to_data"]:
        interpretation.append(
            "当前正则 Hessian 相对数据 Hessian 的谱范数很小；"
            "配置中的 mu 数值可能看似不小，但实际数值约束可能很弱。"
        )

    if flags["independent_local_wavelets_vary_strongly"]:
        interpretation.append(
            "互不耦合的局部窗口子波差异明显。"
            "这支持全局 stationary 假设不足，或不同窗口的数据约束质量差异较大；"
            "请结合 local condition_ratio 和 r_energy 区分两者。"
        )
    else:
        interpretation.append(
            "独立局部窗口的核心子波总体相似；"
            "全局 stationary 假设未被该项诊断明显否定。"
        )

    if flags["damping_parameter_appears_numerically_inactive"]:
        interpretation.append(
            "_safe_svd_solve 的数值探针对 damping_ratio 基本无响应；"
            "请检查当前实现是否仍使用截断伪逆 1/S，而没有使用阻尼滤波因子。"
        )

    return {
        "stationary_parameters": asdict(params),
        "global_wavelet_lengths": sorted(global_results),
        "local_wavelet_length": int(local_wavelet_length),
        "local_data_window_factor": float(local_data_window_factor),
        "metrics": {
            "global_common_support_median_abs_correlation": (
                global_median_corr
            ),
            "global_common_support_min_abs_correlation": global_min_corr,
            "median_raw_effective_rank_ratio": median_raw_rank_ratio,
            "median_raw_condition_ratio": median_raw_cond_ratio,
            "median_total_regularization_hessian_ratio": median_reg_ratio,
            "median_objective_regularization_over_data": (
                median_objective_ratio
            ),
            "local_wavelet_median_abs_correlation": local_median_corr,
            "local_wavelet_min_abs_correlation": local_min_corr,
            "median_local_effective_rank_ratio": (
                _finite_median(local_rank_ratios)
            ),
            "median_local_condition_ratio": (
                _finite_median(local_condition_ratios)
            ),
            "max_damping_probe_relative_change": max_damping_probe,
        },
        "heuristic_flags": flags,
        "interpretation": interpretation,
        "threshold_note": (
            "heuristic_flags 只用于快速定位，不是物理验收门槛；"
            "最终判断应结合图件、奇异谱、局部能量及井震地质背景。"
        ),
    }


def _resolve_config_path(
    result_bundle_path: Path,
    config_path: str | None,
) -> Path:
    if config_path is not None:
        resolved = Path(config_path).expanduser().resolve()
    else:
        resolved = result_bundle_path.parent / "config_used.yaml"

    if not resolved.exists():
        raise FileNotFoundError(
            "找不到配置文件。请通过 --config 显式指定 config_used.yaml "
            f"或原始配置。当前尝试路径：{resolved}"
        )

    return resolved


def _load_stationary_params(cfg: Any) -> StationaryParams:
    stationary_cfg = cfg.stationary
    wavelet_cfg = cfg.wavelet

    return StationaryParams(
        alignment=str(wavelet_cfg.alignment),
        mu1=float(stationary_cfg.mu1),
        mu2=float(stationary_cfg.mu2),
        mu_dc=float(stationary_cfg.mu_dc),
        damping_ratio=float(stationary_cfg.damping_ratio),
        svd_cutoff_ratio=float(stationary_cfg.svd_cutoff_ratio),
        mu_edge=float(getattr(stationary_cfg, "mu_edge", 0.0)),
        edge_fraction=float(
            getattr(stationary_cfg, "edge_fraction", 0.12)
        ),
        edge_taper=str(
            getattr(stationary_cfg, "edge_taper", "cosine")
        ),
    )


def _load_bundle(
    bundle_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(bundle_path, allow_pickle=True) as bundle:
        required = ("r_work_final", "obs_work", "t_work")
        missing = [
            key for key in required
            if key not in bundle.files
        ]

        if missing:
            raise KeyError(
                "result_bundle.npz 缺少诊断所需数组："
                + ", ".join(missing)
            )

        r_time = np.asarray(
            bundle["r_work_final"],
            dtype=float,
        ).ravel()
        s_obs = np.asarray(
            bundle["obs_work"],
            dtype=float,
        ).ravel()
        t_work = np.asarray(
            bundle["t_work"],
            dtype=float,
        ).ravel()

    if not (
        r_time.size == s_obs.size == t_work.size
    ):
        raise ValueError(
            "r_work_final、obs_work、t_work 长度必须一致。"
        )

    if r_time.size < 10:
        raise ValueError("工作数据长度过短，无法进行诊断。")

    if not (
        np.all(np.isfinite(r_time))
        and np.all(np.isfinite(s_obs))
        and np.all(np.isfinite(t_work))
    ):
        raise ValueError("诊断输入包含 NaN 或 Inf。")

    return r_time, s_obs, t_work


def run_diagnostics(
    result_bundle: str | Path,
    config: str | Path | None = None,
    output_dir: str | Path | None = None,
    lengths: list[int] | None = None,
    common_half_ms: float = 25.0,
    local_wavelet_length: int | None = None,
    local_windows: int = 5,
    local_window_factor: float | None = None,
) -> Path:
    result_bundle_path = Path(result_bundle).expanduser().resolve()

    if not result_bundle_path.exists():
        raise FileNotFoundError(
            f"找不到 result bundle：{result_bundle_path}"
        )

    config_path = _resolve_config_path(
        result_bundle_path=result_bundle_path,
        config_path=str(config) if config is not None else None,
    )

    cfg = load_config(config_path)
    params = _load_stationary_params(cfg)

    r_time, s_obs, t_work = _load_bundle(result_bundle_path)

    dt_values = np.diff(t_work)
    dt = float(np.median(dt_values))

    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("无法从 t_work 得到有效 dt。")

    dt_nonuniformity = float(
        np.max(np.abs(dt_values - dt))
        / (abs(dt) + EPS)
    )

    if dt_nonuniformity > 1e-4:
        print(
            "[Warning] t_work 不是严格均匀采样："
            f"relative max deviation={dt_nonuniformity:.3e}"
        )

    if lengths is None:
        lengths = [65, 79, 101, 129, 201]

    lengths = sorted(
        {
            _make_odd(int(length))
            for length in lengths
            if int(length) >= 3
        }
    )

    if not lengths:
        raise ValueError("至少需要一个 >= 3 的 wavelet length。")

    if output_dir is None:
        out_dir = result_bundle_path.parent / "stationary_diagnostics"
    else:
        out_dir = Path(output_dir).expanduser().resolve()

    out_dir.mkdir(parents=True, exist_ok=True)

    configured_length = _make_odd(
        int(round(float(cfg.wavelet.length_s) / dt))
    )

    if local_wavelet_length is None:
        local_wavelet_length = configured_length
    else:
        local_wavelet_length = _make_odd(local_wavelet_length)

    if local_window_factor is None:
        local_window_factor = float(
            getattr(cfg.wavelet, "data_window_factor", 2.0)
        )

    print("=" * 72)
    print("Stationary inversion diagnostics")
    print(f"bundle                 = {result_bundle_path}")
    print(f"config                 = {config_path}")
    print(f"output                 = {out_dir}")
    print(f"N                      = {r_time.size}")
    print(f"dt                     = {dt:.6f} s")
    print(f"alignment              = {params.alignment}")
    print(f"lengths                = {lengths}")
    print(f"configured length      = {configured_length}")
    print(f"local diagnostic L     = {local_wavelet_length}")
    print(f"local windows          = {local_windows}")
    print(f"local window factor    = {local_window_factor}")
    print(f"mu1 / mu2 / mu_dc      = {params.mu1} / {params.mu2} / {params.mu_dc}")
    print(f"mu_edge                = {params.mu_edge}")
    print(f"damping_ratio          = {params.damping_ratio}")
    print(f"svd_cutoff_ratio       = {params.svd_cutoff_ratio}")
    print("=" * 72)

    global_results = _run_global_length_diagnostics(
        r_time=r_time,
        s_obs=s_obs,
        dt=dt,
        params=params,
        lengths=lengths,
    )

    (
        correlation_lengths,
        common_axis_ms,
        global_signed_corr,
        global_abs_corr,
    ) = _global_common_support_correlations(
        results=global_results,
        dt=dt,
        alignment=params.alignment,
        common_half_ms=common_half_ms,
    )

    local_results = _run_local_window_diagnostics(
        r_time=r_time,
        s_obs=s_obs,
        t_work=t_work,
        dt=dt,
        params=params,
        wavelet_length=local_wavelet_length,
        data_window_factor=float(local_window_factor),
        n_windows=int(local_windows),
    )

    local_signed_corr, local_abs_corr = (
        _local_correlation_matrices(local_results)
    )

    _save_global_table(
        global_results,
        out_dir / "global_length_diagnostics.csv",
    )
    _save_global_correlation_table(
        lengths=correlation_lengths,
        signed=global_signed_corr,
        absolute=global_abs_corr,
        path=out_dir / "global_common_support_correlation.csv",
    )
    _save_local_table(
        local_results,
        out_dir / "local_window_diagnostics.csv",
    )
    _save_local_correlation_table(
        results=local_results,
        signed=local_signed_corr,
        absolute=local_abs_corr,
        path=out_dir / "local_wavelet_correlation.csv",
    )

    _plot_global_wavelets(
        global_results,
        out_dir / "01_global_wavelets_by_length.png",
    )
    _plot_heatmap(
        global_abs_corr,
        labels=[f"L={length}" for length in correlation_lengths],
        path=(
            out_dir
            / "02_global_common_support_abs_correlation.png"
        ),
        title=(
            f"Common-support |correlation| "
            f"(requested ±{common_half_ms:g} ms)"
        ),
    )
    _plot_singular_spectra(
        global_results,
        attribute_name="R_singular_values",
        path=out_dir / "03_raw_R_singular_spectra.png",
        title="Singular spectra of raw global convolution matrices R",
    )
    _plot_singular_spectra(
        global_results,
        attribute_name="A_singular_values",
        path=out_dir / "04_augmented_A_singular_spectra.png",
        title="Singular spectra of augmented regularized systems A",
    )
    _plot_regularization_ratios(
        global_results,
        out_dir / "05_regularization_hessian_ratios.png",
    )
    _plot_local_wavelets(
        local_results,
        out_dir / "06_local_wavelets.png",
    )
    _plot_heatmap(
        local_abs_corr,
        labels=[
            f"{result.center_time_s:.3f}s"
            for result in local_results
        ],
        path=out_dir / "07_local_wavelet_abs_correlation.png",
        title="Independent local-window wavelet |correlation|",
    )

    largest_length = max(global_results)

    _plot_small_singular_vectors(
        r_time=r_time,
        dt=dt,
        alignment=params.alignment,
        wavelet_length=largest_length,
        path=(
            out_dir
            / f"08_small_singular_vectors_L{largest_length}.png"
        ),
        n_vectors=5,
    )

    summary = _build_summary(
        params=params,
        global_results=global_results,
        global_abs_corr=global_abs_corr,
        local_results=local_results,
        local_abs_corr=local_abs_corr,
        local_wavelet_length=local_wavelet_length,
        local_data_window_factor=float(local_window_factor),
    )

    summary["input"] = {
        "result_bundle": str(result_bundle_path),
        "config": str(config_path),
        "dt_s": dt,
        "sample_count": int(r_time.size),
        "t_start_s": float(t_work[0]),
        "t_end_s": float(t_work[-1]),
        "dt_relative_nonuniformity": dt_nonuniformity,
        "common_axis_start_ms": float(common_axis_ms[0]),
        "common_axis_end_ms": float(common_axis_ms[-1]),
    }

    with open(
        out_dir / "diagnostic_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    _save_npz(
        path=out_dir / "diagnostic_arrays.npz",
        global_results=global_results,
        common_axis_ms=common_axis_ms,
        global_signed_corr=global_signed_corr,
        global_abs_corr=global_abs_corr,
        local_results=local_results,
        local_signed_corr=local_signed_corr,
        local_abs_corr=local_abs_corr,
    )

    print()
    print("=" * 72)
    print("Diagnostic summary")
    print("=" * 72)

    for line in summary["interpretation"]:
        print(f"- {line}")

    print()
    print("Heuristic flags:")

    for key, value in summary["heuristic_flags"].items():
        print(f"  {key}: {value}")

    print()
    print(f"All diagnostic outputs saved to: {out_dir}")

    return out_dir


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "只读诊断全局平稳子波的长度敏感性、病态性、"
            "正则化实际强度和局部平稳性。"
        )
    )

    parser.add_argument(
        "--result-bundle",
        required=True,
        help="主程序保存的 result_bundle.npz 路径。",
    )
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "配置文件路径。省略时自动读取 result_bundle 同目录的 "
            "config_used.yaml。"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "诊断输出目录。省略时写入 result bundle 同目录下的 "
            "stationary_diagnostics/。"
        ),
    )
    parser.add_argument(
        "--lengths",
        nargs="+",
        type=int,
        default=[65, 79, 101, 129, 201],
        help="全局 stationary 长度扫描，默认 65 79 101 129 201。",
    )
    parser.add_argument(
        "--common-half-ms",
        type=float,
        default=25.0,
        help="center 模式下比较共同核心区间的半宽，默认 25 ms。",
    )
    parser.add_argument(
        "--local-wavelet-length",
        type=int,
        default=None,
        help=(
            "局部窗口诊断使用的子波长度。省略时使用配置中的 "
            "wavelet.length_s 换算后的采样点数。"
        ),
    )
    parser.add_argument(
        "--local-windows",
        type=int,
        default=5,
        help="局部独立窗口数量，默认 5。",
    )
    parser.add_argument(
        "--local-window-factor",
        type=float,
        default=None,
        help=(
            "局部数据窗长度 / 子波长度。省略时读取 "
            "wavelet.data_window_factor。"
        ),
    )

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    run_diagnostics(
        result_bundle=args.result_bundle,
        config=args.config,
        output_dir=args.output_dir,
        lengths=args.lengths,
        common_half_ms=args.common_half_ms,
        local_wavelet_length=args.local_wavelet_length,
        local_windows=args.local_windows,
        local_window_factor=args.local_window_factor,
    )


if __name__ == "__main__":
    main()
