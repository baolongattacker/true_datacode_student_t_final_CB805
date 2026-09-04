# -*- coding: utf-8 -*-
"""
core/signal_utils.py

信号级工具函数：
1. NaN/Inf 处理
2. 归一化
3. RMS 匹配
4. 子波频谱属性
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import tukey


def _as_finite_1d(
    x: np.ndarray,
    name: str,
    *,
    copy: bool = False,
) -> np.ndarray:
    """将输入检查为非空、有限的一维浮点数组，shape=(N,)。"""
    array = np.asarray(x, dtype=float)

    if array.ndim != 1:
        raise ValueError(f"{name} 必须是一维数组，实际 shape={array.shape}。")
    if array.size == 0:
        raise ValueError(f"{name} 不能为空。")
    if not np.all(np.isfinite(array)):
        raise ValueError(
            f"{name} 包含 NaN 或 Inf；请先显式调用 fill_nan_by_interp 或采用明确的掩膜处理。"
        )

    return array.copy() if copy else array


def _positive_finite_scalar(value: float, name: str) -> float:
    """检查物理标量为有限正数。"""
    scalar = np.asarray(value, dtype=float)

    if scalar.ndim != 0 or not np.isfinite(scalar) or scalar <= 0:
        raise ValueError(f"{name} 必须是有限正数。")

    return float(scalar)


def fill_nan_by_interp(x: np.ndarray) -> np.ndarray:
    """
    使用线性插值填充数组中的 NaN 值。

    Args:
        x (np.ndarray): 输入的一维数组，可能包含 NaN 或 Inf。

    Returns:
        np.ndarray: 插值后的数组副本。

    Raises:
        ValueError: 如果有效数据点少于 2 个，无法进行插值。
    """
    x = np.asarray(x, dtype=float)

    if x.ndim != 1:
        raise ValueError(f"x 必须是一维数组，实际 shape={x.shape}。")

    x = x.copy()
    idx = np.arange(len(x))
    valid = np.isfinite(x)

    if np.sum(valid) < 2:
        raise ValueError("有效点太少，无法插值。")

    x[~valid] = np.interp(idx[~valid], idx[valid], x[valid])
    return x


def robust_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    基于百分位数的稳健归一化。
    减去均值后，除以绝对值的第 99 百分位数，以减少异常值的影响。
    抗干扰，抗离群值。

    Args:
        x (np.ndarray): 输入的信号数组。
        eps (float): 防止除零的微小值，默认为 1e-12。

    Returns:
        np.ndarray: 归一化后的信号数组。
    """
    eps = _positive_finite_scalar(eps, "eps")
    x = _as_finite_1d(x, "x", copy=True)
    x = x - np.mean(x)

    scale = np.percentile(np.abs(x), 99)

    if scale <= eps:
        scale = np.max(np.abs(x)) + eps

    return x / scale


def normalize_for_dtw(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    为 DTW（动态时间规整）准备的归一化。
    将 NaN 替换为 0，减去均值，并除以标准差（Z-score 归一化）。

    Args:
        x (np.ndarray): 输入的信号数组。
        eps (float): 防止除零的微小值，默认为 1e-12。

    Returns:
        np.ndarray: 归一化后的信号数组。
    """
    eps = _positive_finite_scalar(eps, "eps")
    x = _as_finite_1d(x, "x")
    x = x - np.mean(x)

    std = np.std(x)
    if std < eps:
        return x * 0.0

    return x / std

# 和上面的函数功能上有点重复

def normalize_trace(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    对地震道进行标准的 Z-score 归一化。

    Args:
        x (np.ndarray): 输入的地震道数据。
        eps (float): 防止除零的微小值，默认为 1e-12。

    Returns:
        np.ndarray: 归一化后的地震道。
    """
    eps = _positive_finite_scalar(eps, "eps")
    x = _as_finite_1d(x, "x")
    x = x - np.mean(x)
    return x / (np.std(x) + eps)


def match_rms(
    s_syn: np.ndarray,
    s_obs: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    将合成记录的 RMS（均方根）振幅匹配到观测记录。

    Args:
        s_syn (np.ndarray): 待调整的合成地震记录。
        s_obs (np.ndarray): 目标观测地震记录。
        eps (float): 防止除零的微小值，默认为 1e-12。

    Returns:
        np.ndarray: 振幅匹配后的合成记录。
    """
    eps = _positive_finite_scalar(eps, "eps")
    s_syn = _as_finite_1d(s_syn, "s_syn", copy=True)
    s_obs = _as_finite_1d(s_obs, "s_obs", copy=True)

    if len(s_syn) != len(s_obs):
        raise ValueError(
            f"s_syn 与 s_obs 长度必须一致，实际为 {len(s_syn)} 和 {len(s_obs)}。"
        )

    rms_syn = np.sqrt(np.mean(s_syn ** 2)) + eps
    rms_obs = np.sqrt(np.mean(s_obs ** 2)) + eps

    return s_syn * (rms_obs / rms_syn)


def estimate_wavelet_centroid_frequency(
    W: np.ndarray,
    dt: float,
    fmin: float = 5.0,
    fmax: float = 80.0,
) -> np.ndarray:
    """
    估计时变子波矩阵中每个子波的质心频率。

    Args:
        W (np.ndarray): 时变子波矩阵，形状为 (N_time, wavelet_length)。
        dt (float): 时间采样间隔（秒）。
        fmin (float): 计算质心的频率下限（Hz），默认为 5.0。
        fmax (float): 计算质心的频率上限（Hz），默认为 80.0。

    Returns:
        np.ndarray: 每个时间点的质心频率数组，形状为 (N_time,)。

    Raises:
        ValueError: 如果输入 W 不是二维数组。
    """
    W = np.asarray(W, dtype=float)

    if W.ndim != 2:
        raise ValueError("W 必须是二维数组，shape=(N_time, wavelet_length)。")

    if W.shape[0] == 0 or W.shape[1] == 0:
        raise ValueError(f"W 的两个维度都必须非空，实际 shape={W.shape}。")
    if not np.all(np.isfinite(W)):
        raise ValueError("W 包含 NaN 或 Inf，无法计算子波质心频率。")

    dt = _positive_finite_scalar(dt, "dt")

    n_time, wavelet_length = W.shape

    freqs = np.fft.rfftfreq(wavelet_length, dt)
    mask = (freqs >= fmin) & (freqs <= fmax)

    f_centroid = np.full(n_time, np.nan)

    for i in range(n_time):
        w = W[i] - np.mean(W[i])
        w = w * tukey(len(w), alpha=0.2)

        power = np.abs(np.fft.rfft(w)) ** 2
        p = power[mask]
        f = freqs[mask]

        if np.sum(p) > 1e-12:
            f_centroid[i] = np.sum(f * p) / np.sum(p)

    good = np.isfinite(f_centroid)

    if np.sum(good) > 2:
        idx = np.arange(n_time)
        f_centroid[~good] = np.interp(
            idx[~good],
            idx[good],
            f_centroid[good],
        )
        f_centroid = gaussian_filter1d(
            f_centroid,
            sigma=5,
            mode="nearest",
        )

    return f_centroid
def best_fit_scale(
    s_syn: np.ndarray,
    s_obs: np.ndarray,
    allow_negative: bool = False,
) -> float:
    """
    计算合成记录到观测地震道的最佳整体振幅缩放因子。

    输入：
        s_syn : 合成地震记录
        s_obs : 井旁道 / 观测地震道
        allow_negative : 是否允许负缩放（允许则可能发生相位翻转，否则强制 >= 0）

    输出：
        scale : 标量，使 scale * s_syn 在最小二乘意义下最接近 s_obs

    物理意义：
        只校正整体振幅尺度差异，不改变反射系数、时深关系、子波相位和同相轴位置。
    """
    s_syn = _as_finite_1d(s_syn, "s_syn")
    s_obs = _as_finite_1d(s_obs, "s_obs")

    if len(s_syn) != len(s_obs):
        raise ValueError(
            f"s_syn 与 s_obs 长度必须一致，实际为 {len(s_syn)} 和 {len(s_obs)}。"
        )

    numerator = np.sum(s_syn * s_obs)
    denominator = np.sum(s_syn * s_syn) + 1e-12

    scale = numerator / denominator

    if not allow_negative:
        scale = max(scale, 0.0)

    return float(scale)
