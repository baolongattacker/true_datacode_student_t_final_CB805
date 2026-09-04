# -*- coding: utf-8 -*-
"""
core/forward_operator.py

统一管理 center / causal 两种子波约定下的正演算子。

这里是全流程中最关键的约定层：
1. stationary convolution
2. nonstationary convolution
3. local convolution matrix
4. wavelet lag axis
5. peak metric

后续所有 stage 都必须调用这里，不能各自重新写 center/causal 分支。
"""

from __future__ import annotations

import numpy as np


def _as_finite_1d(x: np.ndarray, name: str) -> np.ndarray:
    """将正演输入检查为非空、有限的一维数组，shape=(N,)。"""
    array = np.asarray(x, dtype=float)

    if array.ndim != 1:
        raise ValueError(f"{name} 必须是一维数组，实际 shape={array.shape}。")
    if array.size == 0:
        raise ValueError(f"{name} 不能为空。")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} 包含 NaN 或 Inf，无法构造正演算子。")

    return array


def _as_finite_2d(x: np.ndarray, name: str) -> np.ndarray:
    """将正演输入检查为两个维度均非空的有限二维数组。"""
    array = np.asarray(x, dtype=float)

    if array.ndim != 2:
        raise ValueError(f"{name} 必须是二维数组，实际 shape={array.shape}。")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} 的两个维度都必须非空，实际 shape={array.shape}。")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} 包含 NaN 或 Inf，无法构造正演算子。")

    return array


def _positive_finite_scalar(value: float, name: str) -> float:
    """检查采样间隔等物理标量为有限正数。"""
    scalar = np.asarray(value, dtype=float)

    if scalar.ndim != 0 or not np.isfinite(scalar) or scalar <= 0:
        raise ValueError(f"{name} 必须是有限正数。")

    return float(scalar)


def _positive_integer(value: int, name: str) -> int:
    """检查数组长度等离散量为正整数。"""
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value <= 0
    ):
        raise ValueError(f"{name} 必须是正整数。")

    return int(value)


def validate_alignment(alignment: str) -> str:
    """
    检查子波对齐方式。

    center:
        零相位/中心子波，lag 轴为 [-L//2, ..., 0, ..., +L//2]

    causal:
        因果/近似最小相位子波，lag 轴为 [0, 1, 2, ..., L-1]
    """
    alignment = str(alignment).lower()

    if alignment not in ("center", "causal"):
        raise ValueError("alignment 必须是 'center' 或 'causal'。")

    return alignment


def get_lag(col: int, wavelet_length: int, alignment: str = "center") -> int:
    """
    返回第 col 个子波采样点对应的 lag。

    center:
        lag = col - L//2

    causal:
        lag = col
    """
    alignment = validate_alignment(alignment)

    if alignment == "center":
        return int(col - wavelet_length // 2)

    return int(col)


def wavelet_lag_axis_samples(
    wavelet_length: int,
    alignment: str = "center",
) -> np.ndarray:
    """
    返回子波横轴，单位为 samples。
    """
    alignment = validate_alignment(alignment)
    wavelet_length = _positive_integer(wavelet_length, "wavelet_length")

    if alignment == "center":
        return np.arange(wavelet_length) - wavelet_length // 2

    return np.arange(wavelet_length)


def wavelet_lag_axis_ms(
    wavelet_length: int,
    dt: float,
    alignment: str = "center",
) -> np.ndarray:
    """
    生成子波横轴，单位为 ms。
    """
    dt = _positive_finite_scalar(dt, "dt")

    return wavelet_lag_axis_samples(wavelet_length, alignment) * dt * 1000.0


def stationary_convolution(
    r_time: np.ndarray,
    w: np.ndarray,
    alignment: str = "center",
) -> np.ndarray:
    """
    平稳子波褶积。

    center:
        使用 np.convolve(..., mode="same")

    causal:
        使用 np.convolve(..., mode="full")[:N]

    注意：
        center 和 causal 的输出长度都固定为 len(r_time)。
    """
    alignment = validate_alignment(alignment)
    r_time = _as_finite_1d(r_time, "r_time")
    w = _as_finite_1d(w, "w")

    n = len(r_time)

    if alignment == "center":
        if len(w) > n:
            raise ValueError(
                "center 卷积要求 wavelet_length <= len(r_time)，以保持输出长度等于 len(r_time)。"
            )
        return np.convolve(r_time, w, mode="same")

    return np.convolve(r_time, w, mode="full")[:n]


def nonstationary_convolution(
    r_time: np.ndarray,
    W: np.ndarray,
    alignment: str = "center",
) -> np.ndarray:
    """
    时变子波非平稳褶积。

    W.shape = (N_time, wavelet_length)

    center:
        s[i] = sum_j r[i - (j - L//2)] * W[i, j]

    causal:
        s[i] = sum_j r[i - j] * W[i, j]

    这个函数必须和 build_local_convolution_matrix 的 convention 完全一致。
    """
    alignment = validate_alignment(alignment)

    r_time = _as_finite_1d(r_time, "r_time")
    W = _as_finite_2d(W, "W")

    n_time, wavelet_length = W.shape

    if len(r_time) != n_time:
        raise ValueError("len(r_time) 必须等于 W.shape[0]。")

    s = np.zeros(n_time, dtype=float)

    for i in range(n_time):
        acc = 0.0

        for col in range(wavelet_length):
            lag = get_lag(col, wavelet_length, alignment)
            r_index = i - lag

            if 0 <= r_index < n_time:
                acc += r_time[r_index] * W[i, col]

        s[i] = acc

    return s


def build_local_convolution_matrix(
    r_time: np.ndarray,
    center_index: int,
    data_window_length: int,
    wavelet_length: int,
    alignment: str = "center",
) -> np.ndarray:
    """
    构造局部褶积矩阵 R，使得：

        s_window ≈ R @ w

    这个矩阵必须和 stationary_convolution / nonstationary_convolution 使用同一个 lag 约定。
    """
    alignment = validate_alignment(alignment)

    r_time = _as_finite_1d(r_time, "r_time")
    data_window_length = _positive_integer(
        data_window_length,
        "data_window_length",
    )
    wavelet_length = _positive_integer(wavelet_length, "wavelet_length")

    if data_window_length % 2 == 0:
        raise ValueError("data_window_length 建议为奇数。")

    if wavelet_length % 2 == 0:
        raise ValueError("wavelet_length 建议为奇数。")

    n_time = len(r_time)

    if (
        isinstance(center_index, (bool, np.bool_))
        or not isinstance(center_index, (int, np.integer))
        or not 0 <= center_index < n_time
    ):
        raise ValueError(
            f"center_index 必须是 [0, {n_time}) 内的整数，实际为 {center_index}。"
        )

    center_index = int(center_index)
    half_data = data_window_length // 2

    R = np.zeros((data_window_length, wavelet_length), dtype=float)

    for row in range(data_window_length):
        global_t = center_index - half_data + row

        for col in range(wavelet_length):
            lag = get_lag(col, wavelet_length, alignment)
            r_index = global_t - lag

            if 0 <= r_index < n_time:
                R[row, col] = r_time[r_index]

    return R


def wavelet_peak_metric_samples(
    w: np.ndarray,
    alignment: str = "center",
) -> int:
    """
    子波主峰位置指标。衡量反演子波的主峰偏离时间零点多少个采样点。

    center:
        返回 peak_index - center_index

    causal:
        返回 peak_index
    """
    alignment = validate_alignment(alignment)

    w = _as_finite_1d(w, "w")
    peak_index = int(np.argmax(np.abs(w)))

    if alignment == "center":
        return peak_index - len(w) // 2

    return peak_index


def wavelet_peak_metric_ms(
    w: np.ndarray,
    dt: float,
    alignment: str = "center",
) -> float:
    """
    子波主峰位置指标，单位 ms，单位转换。
    """
    dt = _positive_finite_scalar(dt, "dt")
    return wavelet_peak_metric_samples(w, alignment) * dt * 1000.0


def wavelet_matrix_peak_metric_ms(
    W: np.ndarray,
    dt: float,
    alignment: str = "center",
) -> np.ndarray:
    """
    计算时变子波矩阵每一行的主峰位置指标，单位 ms。
    对时变子波矩阵（W）的每一行独立计算主峰位置，并以毫秒（ms）为单位返回。这是用于分析时变子波（例如反演得到的子波）在时间上如何变化的工具。
    批量处理版。
    """
    W = _as_finite_2d(W, "W")
    dt = _positive_finite_scalar(dt, "dt")

    return np.array(
        [wavelet_peak_metric_ms(W[i, :], dt, alignment) for i in range(W.shape[0])],
        dtype=float,
    )
