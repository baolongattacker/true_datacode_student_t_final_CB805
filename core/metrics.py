# -*- coding: utf-8 -*-
"""
core/metrics.py

评价指标：
1. 零滞后相关
2. 最大滞后相关
3. 包络相关
"""

from __future__ import annotations

import numpy as np
from scipy.signal import hilbert

from core.signal_utils import normalize_trace


def _as_finite_1d(x: np.ndarray, name: str) -> np.ndarray:
    """将指标输入检查为非空、有限的一维数组，shape=(N,)。"""
    array = np.asarray(x, dtype=float)

    if array.ndim != 1:
        raise ValueError(f"{name} 必须是一维数组，实际 shape={array.shape}。")
    if array.size == 0:
        raise ValueError(f"{name} 不能为空。")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} 包含 NaN 或 Inf，不能计算相关指标。")

    return array


def _as_equal_finite_1d_pair(
    a: np.ndarray,
    b: np.ndarray,
    a_name: str,
    b_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """检查两条输入道均为有限的一维等长数组。"""
    a = _as_finite_1d(a, a_name)
    b = _as_finite_1d(b, b_name)

    if len(a) != len(b):
        raise ValueError(
            f"{a_name} 与 {b_name} 长度必须一致，实际为 {len(a)} 和 {len(b)}。"
        )

    return a, b


def _positive_finite_scalar(value: float, name: str) -> float:
    """检查物理标量为有限正数。"""
    scalar = np.asarray(value, dtype=float)

    if scalar.ndim != 0 or not np.isfinite(scalar) or scalar <= 0:
        raise ValueError(f"{name} 必须是有限正数。")

    return float(scalar)


def corrcoef_safe(
    a: np.ndarray,
    b: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """
    计算两个信号的安全相关系数（处理了均值和分母可能为零的情况）。

    参数:
        a (np.ndarray): 输入的第一个信号序列。
        b (np.ndarray): 输入的第二个信号序列。
        eps (float, optional): 防止分母为零的微小正数，默认为 1e-12。

    返回:
        float: 两个信号之间的相关系数。
    """
    eps = _positive_finite_scalar(eps, "eps")
    a, b = _as_equal_finite_1d_pair(a, b, "a", "b")
    # 去直流分量（中心化）
    a = a - np.mean(a)
    b = b - np.mean(b)

    den = np.sqrt(np.sum(a ** 2) * np.sum(b ** 2)) + eps

    return float(np.sum(a * b) / den)


def zero_lag_cc(x: np.ndarray, y: np.ndarray) -> float:
    """
    计算两个信号的零滞后互相关（零延迟相关）。

    参数:
        x (np.ndarray): 输入的第一个信号序列。
        y (np.ndarray): 输入的第二个信号序列。

    返回:
        float: 零滞后条件下的互相关系数。
    """
    x, y = _as_equal_finite_1d_pair(x, y, "x", "y")
    x = normalize_trace(x)
    y = normalize_trace(y)

    return float(np.mean(x * y))


def max_lag_cc(
    x: np.ndarray,
    y: np.ndarray,
    max_lag_samples: int = 12,
) -> tuple[float, int]:
    """
    计算在允许的最大滞后样本数范围内的最大互相关系数及其对应的滞后量。

    参数:
        x (np.ndarray): 输入的参考信号序列。
        y (np.ndarray): 输入的待比较信号序列。
        max_lag_samples (int, optional): 允许的最大正负滞后样本数，默认为 12。

    返回:
        tuple[float, int]: 
            - best_cc (float): 搜索范围内的最大互相关系数。
            - best_lag (int): 达到最大互相关时的滞后样本数（正数表示 y 相对 x 延迟，负数表示提前）。
    """
    x, y = _as_equal_finite_1d_pair(x, y, "x", "y")

    if (
        isinstance(max_lag_samples, (bool, np.bool_))
        or not isinstance(max_lag_samples, (int, np.integer))
        or max_lag_samples < 0
    ):
        raise ValueError("max_lag_samples 必须是非负整数。")

    x = normalize_trace(x)
    y = normalize_trace(y)

    n = len(x)

    best_cc = -np.inf
    best_lag = 0

    for lag in range(-max_lag_samples, max_lag_samples + 1):
        if lag < 0:
            xx = x[-lag:]
            yy = y[: n + lag]
        elif lag > 0:
            xx = x[: n - lag]
            yy = y[lag:]
        else:
            xx = x
            yy = y

        if len(xx) < 20:
            continue

        cc = float(np.mean(xx * yy))

        if cc > best_cc:
            best_cc = cc
            best_lag = lag

    return best_cc, best_lag


def envelope_cc(x: np.ndarray, y: np.ndarray) -> float:
    """
    计算两个信号包络之间的零滞后互相关系数。

    包络通过希尔伯特变换 (Hilbert Transform) 提取。此指标用于衡量两个信号能量分布的相似度。

    参数:
        x (np.ndarray): 输入的第一个信号序列。
        y (np.ndarray): 输入的第二个信号序列。

    返回:
        float: 两个信号包络的互相关系数。
    """
    x, y = _as_equal_finite_1d_pair(x, y, "x", "y")
    x_env = np.abs(hilbert(x))
    y_env = np.abs(hilbert(y))

    return zero_lag_cc(x_env, y_env)

def tie_similarity_score(
    s_obs: np.ndarray,
    s_syn: np.ndarray,
    dt: float,
    max_lag_ms: float = 8.0,
    weight_maxlag: float = 0.55,
    weight_envelope: float = 0.35,
    weight_direct: float = 0.10,
    lag_penalty_weight: float = 0.10,
) -> tuple[float, dict]:
    """
    计算适用于井震标定的综合相似性指标。

    输入：
        s_obs: 观测地震道
        s_syn: 合成地震道
        dt: 时间采样间隔，单位 s
        max_lag_ms: 允许搜索的最大整体时移，单位 ms

    输出：
        score: 综合相似性分数
        details: 相关诊断指标

    物理意义：
        不要求两个地震道逐样点完全一致；
        更强调同相轴、小范围时移内的波形对应，以及包络能量带对应。

    数学作用：
        用 max_lag_cc 作为主指标，
        用 envelope_cc 补充波组/能量相似性，
        用 corrcoef_safe 保留零时移波形约束，
        用 lag_penalty 防止靠大时移获得虚高相关。
    """
    s_obs, s_syn = _as_equal_finite_1d_pair(
        s_obs,
        s_syn,
        "s_obs",
        "s_syn",
    )
    dt = _positive_finite_scalar(dt, "dt")

    max_lag_ms_array = np.asarray(max_lag_ms, dtype=float)
    if (
        max_lag_ms_array.ndim != 0
        or not np.isfinite(max_lag_ms_array)
        or max_lag_ms_array < 0
    ):
        raise ValueError("max_lag_ms 必须是有限非负数。")
    max_lag_ms = float(max_lag_ms_array)

    cc_direct = corrcoef_safe(s_obs, s_syn)
    env_cc = envelope_cc(s_obs, s_syn)

    max_lag_samples = int(round(max_lag_ms / (dt * 1000.0)))
    cc_maxlag, best_lag_samples = max_lag_cc(
        s_obs,
        s_syn,
        max_lag_samples=max_lag_samples,
    )
    best_lag_ms = best_lag_samples * dt * 1000.0

    # 时移惩罚：lag 越接近允许上限，惩罚越大
    lag_ratio = abs(best_lag_ms) / (max_lag_ms + 1e-12)
    lag_ratio = min(lag_ratio, 1.0)
    lag_penalty = lag_penalty_weight * lag_ratio

    # 综合相似性：主看 maxlag，其次看包络，少量保留零时移逐点相关
    score = (
        weight_maxlag * cc_maxlag
        + weight_envelope * env_cc
        + weight_direct * cc_direct
        - lag_penalty
    )

    details = {
        "score": float(score),
        "cc_direct": float(cc_direct),
        "cc_maxlag": float(cc_maxlag),
        "env_cc": float(env_cc),
        "best_lag_samples": int(best_lag_samples),
        "best_lag_ms": float(best_lag_ms),
        "lag_penalty": float(lag_penalty),
        "max_lag_ms": float(max_lag_ms),
        "weight_maxlag": float(weight_maxlag),
        "weight_envelope": float(weight_envelope),
        "weight_direct": float(weight_direct),
        "lag_penalty_weight": float(lag_penalty_weight),
    }

    return float(score), details
