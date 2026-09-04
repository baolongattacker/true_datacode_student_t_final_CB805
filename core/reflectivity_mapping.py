# -*- coding: utf-8 -*-
"""
core/reflectivity_mapping.py

井震标定中的时深关系和反射系数映射工具。
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import cumulative_trapezoid


def calculate_twt_from_velocity(
    depth: np.ndarray,
    velocity: np.ndarray,
    twt_top: float = 0.0,
) -> np.ndarray:
    """
    根据层速度计算每个深度点对应的双程时（TWT）。

    原理：TWT = twt_top + 2 * integral(1/v, dz)

    参数:
        depth (np.ndarray): 深度序列（通常单位为 m）。
        velocity (np.ndarray): 对应深度点的层速度（通常单位为 m/s）。
        twt_top (float, optional): 起始深度点的初始双程时（通常单位为 s），默认为 0.0。

    返回:
        np.ndarray: 每个深度点对应的双程时序列（s）。
    """
    depth = np.asarray(depth, dtype=float)
    velocity = np.asarray(velocity, dtype=float)

    if len(depth) != len(velocity):
        raise ValueError("depth 和 velocity 长度必须一致。")

    twt = np.zeros_like(depth, dtype=float)
    twt[:] = twt_top

    if len(depth) > 1:
        twt += 2.0 * cumulative_trapezoid(
            1.0 / velocity,
            depth,
            initial=0.0,
        )

    return twt


def scatter_reflectivity_to_time(
    r_depth: np.ndarray,
    twt_depth: np.ndarray,
    t_ref: np.ndarray,
) -> np.ndarray:
    """
    将深度域的反射系数映射到均匀的时间采样网格上。

    采用线性散射（Scattering/Splitting）方法：如果一个反射系数落在两个时间采样点之间，
    则按比例分配到相邻的两个采样点上，以保持能量守恒。

    参数:
        r_depth (np.ndarray): 深度域的反射系数序列。
        twt_depth (np.ndarray): 深度点对应的双程时（s）。
        t_ref (np.ndarray): 目标时间参考网格（均匀采样，s）。

    返回:
        np.ndarray: 映射到时间网格上的反射系数序列。
    """
    r_depth = np.asarray(r_depth, dtype=float).ravel()
    twt_depth = np.asarray(twt_depth, dtype=float).ravel()
    t_ref = np.asarray(t_ref, dtype=float).ravel()

    if len(r_depth) != len(twt_depth):
        n = min(len(r_depth), len(twt_depth))
        r_depth = r_depth[:n]
        twt_depth = twt_depth[:n]

    if np.any(np.diff(t_ref) <= 0):
        raise ValueError("t_ref 必须严格递增。")

    dt = float(np.median(np.diff(t_ref)))
    n_time = len(t_ref)

    r_time = np.zeros(n_time, dtype=float)

    r_if = r_depth[:-1]
    t_if = twt_depth[1:]

    mask = (
        (t_if >= t_ref[0])
        & (t_if <= t_ref[-1])
        & np.isfinite(t_if)
        & np.isfinite(r_if)
    )

    ti = t_if[mask]
    ai = r_if[mask]

    if len(ti) == 0:
        return r_time

    k = np.floor((ti - t_ref[0]) / dt).astype(int)
    k = np.clip(k, 0, n_time - 2)

    frac = (ti - t_ref[k]) / dt
    frac = np.clip(frac, 0.0, 1.0)

    np.add.at(r_time, k, (1.0 - frac) * ai)
    np.add.at(r_time, k + 1, frac * ai)

    return r_time


def enforce_monotonic_twt(
    twt: np.ndarray,
    min_dt: float = 1e-6,
) -> np.ndarray:
    """
    强制双程时序列严格单调递增。

    在井震标定或时深转换中，由于速度异常或人工调整，可能出现时间倒置，
    本函数通过微调确保后一个点的时间始终大于前一个点。

    参数:
        twt (np.ndarray): 输入的双程时序列。
        min_dt (float, optional): 最小时间步长增量，默认为 1e-6。

    返回:
        np.ndarray: 修正后的严格单调递增双程时序列。
    """
    twt = np.asarray(twt, dtype=float).copy()

    for i in range(1, len(twt)):
        if twt[i] <= twt[i - 1]:
            twt[i] = twt[i - 1] + min_dt

    return twt


def velocity_from_twt(
    depth: np.ndarray,
    twt: np.ndarray,
    v_min: float = 1000.0,
    v_max: float = 6500.0,
) -> np.ndarray:
    """
    根据时深关系反算区间速度（层速度）。
    用以更新层速度

    原理：v = 2 * dz / dtwt

    参数:
        depth (np.ndarray): 深度序列（m）。
        twt (np.ndarray): 对应的双程时序列（s）。
        v_min (float, optional): 允许的最小速度限制，默认为 1000.0。
        v_max (float, optional): 允许的最大速度限制，默认为 6500.0。

    返回:
        np.ndarray: 计算出的区间速度序列（m/s）。
    """
    depth = np.asarray(depth, dtype=float)
    twt = np.asarray(twt, dtype=float)

    if len(depth) != len(twt):
        raise ValueError("depth 和 twt 长度必须一致。")

    dz = np.gradient(depth)
    dtwt = np.gradient(twt)

    v = 2.0 * dz / (dtwt + 1e-12)
    v = np.clip(v, v_min, v_max)

    return v
