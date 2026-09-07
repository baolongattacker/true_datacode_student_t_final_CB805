# -*- coding: utf-8 -*-
"""
stages/stage_stationary.py

平稳子波阶段：
1. 反演全局平稳子波
2. 生成平稳合成记录
3. RMS 匹配
4. 计算 baseline 指标
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np



from utils.wavelet_inversion_robust import stationary_wavelet_inversion

from core.forward_operator import stationary_convolution
from core.signal_utils import match_rms
from core.metrics import (
    corrcoef_safe,
    envelope_cc,
    max_lag_cc,
    tie_similarity_score,
)


@dataclass
class StationaryResult:
    w: np.ndarray
    s_syn_raw: np.ndarray
    s_syn: np.ndarray
    cc: float
    env_cc: float
    maxlag_cc: float
    best_lag_ms: float
    wavelet_length_pts: int
    data_window_length: int
    alignment: str
    params: dict


def make_odd(n: int) -> int:
    n = int(n)
    return n if n % 2 == 1 else n + 1


def run_stationary_stage(
    r_time: np.ndarray,
    s_obs: np.ndarray,
    dt: float,
    wavelet_length_s: float,
    alignment: str = "center",
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
    data_window_factor: float = 2.0,
) -> StationaryResult:
    r_time = np.asarray(r_time, dtype=float).ravel()
    s_obs = np.asarray(s_obs, dtype=float).ravel()

    if len(r_time) != len(s_obs):
        raise ValueError("r_time 和 s_obs 长度必须一致。")

    if dt <= 0:
        raise ValueError("dt 必须大于 0。")

    wavelet_length_pts = make_odd(round(wavelet_length_s / dt))
    data_window_length = make_odd(round(data_window_factor * wavelet_length_pts))

    max_peak_shift_samples = int(round(max_peak_shift_ms / (dt * 1000.0)))

    w = stationary_wavelet_inversion(
        r_time=r_time,
        s_obs=s_obs,
        wavelet_length=wavelet_length_pts,
        mu1=mu1,
        mu2=mu2,
        mu_dc=mu_dc,
        mu_edge=mu_edge,
        edge_fraction=edge_fraction,
        edge_taper=edge_taper,
        damping_ratio=damping_ratio,
        svd_cutoff_ratio=svd_cutoff_ratio,
        peak_lock=peak_lock,
        max_peak_shift_samples=max_peak_shift_samples,
        wavelet_alignment=alignment,
    )

    s_syn_raw = stationary_convolution(
        r_time=r_time,
        w=w,
        alignment=alignment,
    )

    s_syn = match_rms(s_syn_raw, s_obs)

    # 仍然保留原始诊断量
    cc_direct = corrcoef_safe(s_obs, s_syn)
    env = envelope_cc(s_obs, s_syn)

    max_lag_samples = int(round(8.0 / (dt * 1000.0)))
    maxlag_value, best_lag_samples = max_lag_cc(
        s_obs,
        s_syn,
        max_lag_samples=max_lag_samples,
    )
    best_lag_ms = best_lag_samples * dt * 1000.0

    # 用于井震标定评价的综合相似性
    cc, similarity_details = tie_similarity_score(
        s_obs=s_obs,
        s_syn=s_syn,
        dt=dt,
        max_lag_ms=8.0,
    )

    params = {
        "mu1": mu1,
        "mu2": mu2,
        "mu_dc": mu_dc,
        "mu_edge": mu_edge,
        "edge_fraction": edge_fraction,
        "edge_taper": edge_taper,
        "damping_ratio": damping_ratio,
        "svd_cutoff_ratio": svd_cutoff_ratio,
        "peak_lock": peak_lock,
        "max_peak_shift_ms": max_peak_shift_ms,
        "data_window_factor": data_window_factor,

        # 新增：相关性诊断
        "cc_direct": cc_direct,
        "tie_similarity_score": similarity_details["score"],
        "tie_similarity_cc_maxlag": similarity_details["cc_maxlag"],
        "tie_similarity_env_cc": similarity_details["env_cc"],
        "tie_similarity_best_lag_ms": similarity_details["best_lag_ms"],
        "tie_similarity_lag_penalty": similarity_details["lag_penalty"],
    }

    return StationaryResult(
        w=w,
        s_syn_raw=s_syn_raw,
        s_syn=s_syn,
        cc=cc_direct,
        env_cc=env,
        maxlag_cc=maxlag_value,
        best_lag_ms=best_lag_ms,
        wavelet_length_pts=wavelet_length_pts,
        data_window_length=data_window_length,
        alignment=alignment,
        params=params,
    )
