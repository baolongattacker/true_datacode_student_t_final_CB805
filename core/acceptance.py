# -*- coding: utf-8 -*-
"""
core/acceptance.py

Candidate acceptance logic.

This module only evaluates pass/fail gates. It does not run inversion,
forward modeling, plotting, or state mutation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
# pyrefly: ignore [missing-import]
import numpy as np


@dataclass
class TvAcceptanceConfig:
    """
    直接合成的时变波形(W_est)候选者的验收阈值配置。

    Attributes:
        min_gain_vs_dtw (float): 相对于DTW对齐后的基准，最小需要的CC增益。默认为 0.015。
        min_gain_vs_stationary (float): 相对于平稳波形(Stationary)基准，最小需要的CC增益。默认为 0.020。
        max_env_drop_vs_stationary (float): 相对于平稳波形，允许的最大包络相关性下降值。默认为 0.02。
        max_lag_ms (float): 允许的最大时延(毫秒)。默认为 8.0。
        center_peak_median_ms (float): 中心对齐时，允许的波峰偏移中位数最大值(毫秒)。默认为 5.0。
        center_peak_abs_p90_ms (float): 中心对齐时，允许的波峰绝对偏移第90百分位数最大值(毫秒)。默认为 15.0。
        causal_peak_extra_margin_ms (float): 因果对齐时，允许超出范围的额外边缘裕量(毫秒)。默认为 2.0。
        tolerance_ms (float): 时间相关的容差(毫秒)，会增加到所有的毫秒阈值上。默认为 0.0。
        tolerance_cc (float): 相关性相关的容差，会增加到所有的CC阈值上。默认为 0.0。
    """

    min_gain_vs_dtw: float = 0.015
    min_gain_vs_stationary: float = 0.020
    max_env_drop_vs_stationary: float = 0.02
    max_lag_ms: float = 8.0

    center_peak_median_ms: float = 5.0
    center_peak_abs_p90_ms: float = 15.0

    causal_peak_extra_margin_ms: float = 2.0
    tolerance_ms: float = 0.0
    tolerance_cc: float = 0.0


@dataclass
class TvAcceptanceResult:
    """
    时变波形验收结果。

    Attributes:
        passed (bool): 总体是否通过验收（数值和物理门槛均需通过）。
        numerical_pass (bool): 数值门槛（CC、包络、时延）是否通过。
        physical_pass (bool): 物理门槛（波峰位置）是否通过。
        reasons (list[str]): 未通过的具体原因列表。
        details (dict[str, Any]): 包含评估过程中各项中间指标的详细字典。
    """

    passed: bool
    numerical_pass: bool
    physical_pass: bool
    reasons: list[str]
    details: dict[str, Any]


@dataclass
class QAcceptanceConfig:
    """
    Q约束波形候选者的验收阈值配置。

    Attributes:
        max_cc_drop_vs_tv (float): 相对于未受Q约束的TV波形，允许的最大CC下降值。默认为 0.005。
        max_env_drop_vs_tv (float): 相对于未受Q约束的TV波形，允许的最大包络相关性下降值。默认为 0.03。
        max_lag_ms (float): 允许的最大时延(毫秒)。默认为 8.0。
    """

    max_cc_drop_vs_tv: float = 0.005
    max_env_drop_vs_tv: float = 0.03
    max_lag_ms: float = 8.0


@dataclass
class QAcceptanceResult:
    """
    Q约束波形验收结果。

    Attributes:
        passed (bool): 是否通过验收。
        reasons (list[str]): 未通过的具体原因列表。
        details (dict[str, Any]): 包含评估过程中各项中间指标的详细字典。
    """

    passed: bool
    reasons: list[str]
    details: dict[str, Any]


def evaluate_tv_wavelet_candidate(
    *,
    alignment: str,
    cc_tv_direct: float,
    cc_after_dtw: float,
    cc_stationary: float,
    env_tv: float,
    env_stationary: float,
    best_lag_ms: float,
    peak_metric_p10: float,
    peak_metric_med: float,
    peak_metric_p90: float,
    peak_abs_p90: float,
    causal_peak_allowed_ms: tuple[float, float] = (0.0, 40.0),
    final_peak_violation_ratio: float = 0.0,
    config: TvAcceptanceConfig | None = None,
) -> TvAcceptanceResult:
    """
    评估直接合成的时变波形(W_est)候选者是否可以接受。

    数值门槛控制相关性、包络相似度以及时延。
    物理门槛控制子波峰值的位置。

    Args:
        alignment (str): 子波对齐方式，可选 'center' (中心对齐) 或 'causal' (因果/最小相位对齐)。
        cc_tv_direct (float): 当前TV波形合成后的互相关系数(CC)。
        cc_after_dtw (float): 经过动态时间规整(DTW)对齐后的基准CC。
        cc_stationary (float): 使用平稳波形(Stationary)时的基准CC。
        env_tv (float): TV波形的包络相关性。
        env_stationary (float): 平稳波形的包络相关性。
        best_lag_ms (float): 估计的最佳时延(毫秒)。
        peak_metric_p10 (float): 波峰位置偏移量的第10百分位数。
        peak_metric_med (float): 波峰位置偏移量的中位数。
        peak_metric_p90 (float): 波峰位置偏移量的第90百分位数。
        peak_abs_p90 (float): 波峰绝对偏移量的第90百分位数。
        causal_peak_allowed_ms (tuple[float, float]): 因果对齐时允许的波峰范围[下限, 上限]。默认为 (0.0, 40.0)。
        final_peak_violation_ratio (float): 最终平滑后的子波主峰越界点比例。
        config (TvAcceptanceConfig | None): 包含阈值的配置对象。如果为 None，则使用默认配置。

    Returns:
        TvAcceptanceResult: 包含验收结果的对象，包括是否通过(passed)、数值/物理是否通过以及未通过的具体原因。
    """
    if config is None:
        config = TvAcceptanceConfig()

    alignment = str(alignment).lower()
    reasons: list[str] = []
    tol_ms = max(float(config.tolerance_ms), 0.0)
    tol_cc = max(float(config.tolerance_cc), 0.0)

    numerical_pass = (
        cc_tv_direct + tol_cc >= cc_after_dtw + config.min_gain_vs_dtw
        and cc_tv_direct + tol_cc >= cc_stationary + config.min_gain_vs_stationary
        and env_tv + tol_cc >= env_stationary - config.max_env_drop_vs_stationary
        and abs(best_lag_ms) <= config.max_lag_ms + tol_ms
    )

    if not numerical_pass:
        if cc_tv_direct + tol_cc < cc_after_dtw + config.min_gain_vs_dtw:
            reasons.append("cc_gain_vs_dtw_too_small")
        if cc_tv_direct + tol_cc < cc_stationary + config.min_gain_vs_stationary:
            reasons.append("cc_gain_vs_stationary_too_small")
        if env_tv + tol_cc < env_stationary - config.max_env_drop_vs_stationary:
            reasons.append("envelope_cc_drop_too_large")
        if abs(best_lag_ms) > config.max_lag_ms + tol_ms:
            reasons.append("best_lag_too_large")

    if alignment == "center":
        physical_pass = (
            abs(peak_metric_med) <= config.center_peak_median_ms + tol_ms
            and peak_abs_p90 <= config.center_peak_abs_p90_ms + tol_ms
            and final_peak_violation_ratio <= 0.10
        )
        # 如果物理验收没有通过，则进一步检查没有通过的原因
        if not physical_pass:
            if abs(peak_metric_med) > config.center_peak_median_ms + tol_ms:
                reasons.append("center_peak_median_shift_too_large")
            if peak_abs_p90 > config.center_peak_abs_p90_ms + tol_ms:
                reasons.append("center_peak_p90_shift_too_large")
            if final_peak_violation_ratio > 0.10:
                reasons.append("center_peak_violation_ratio_too_large")

    elif alignment == "causal":
        lo, hi = causal_peak_allowed_ms
        margin = config.causal_peak_extra_margin_ms

        physical_pass = (
            peak_metric_p10 >= lo - margin
            and peak_metric_p90 <= hi + margin
            and final_peak_violation_ratio <= 0.10
        )

        if not physical_pass:
            if peak_metric_p10 < lo - margin:
                reasons.append("causal_peak_too_early")
            if peak_metric_p90 > hi + margin:
                reasons.append("causal_peak_too_late")
            if final_peak_violation_ratio > 0.10:
                reasons.append("causal_peak_violation_ratio_too_large")

    else:
        raise ValueError("alignment must be 'center' or 'causal'.")

    passed = bool(numerical_pass and physical_pass)

    details = {
        "alignment": alignment,
        "cc_tv_direct": float(cc_tv_direct),
        "cc_after_dtw": float(cc_after_dtw),
        "cc_stationary": float(cc_stationary),
        "env_tv": float(env_tv),
        "env_stationary": float(env_stationary),
        "best_lag_ms": float(best_lag_ms),
        "peak_metric_p10": float(peak_metric_p10),
        "peak_metric_med": float(peak_metric_med),
        "peak_metric_p90": float(peak_metric_p90),
        "peak_abs_p90": (
            float(peak_abs_p90) if np.isfinite(peak_abs_p90) else np.nan
        ),
        "final_peak_violation_ratio": float(final_peak_violation_ratio),
        "tolerance_ms": tol_ms,
        "tolerance_cc": tol_cc,
        "numerical_pass": bool(numerical_pass),
        "physical_pass": bool(physical_pass),
        "passed": passed,
    }

    return TvAcceptanceResult(
        passed=passed,
        numerical_pass=bool(numerical_pass),
        physical_pass=bool(physical_pass),
        reasons=reasons,
        details=details,
    )


def evaluate_q_candidate(
    *,
    cc_q: float,
    cc_tv: float,
    env_q: float,
    env_tv: float,
    lag_q_ms: float,
    config: QAcceptanceConfig | None = None,
) -> QAcceptanceResult:
    """
    评估受Q约束的波形候选者是否可以接受。

    Q约束可能会平滑或正则化子波，但它不应显著降低已接受的TV子波候选者的质量。

    Args:
        cc_q (float): 应用Q约束后的互相关系数。
        cc_tv (float): 未受Q约束的时变波形的互相关系数。
        env_q (float): 应用Q约束后的包络相关性。
        env_tv (float): 时变波形的包络相关性。
        lag_q_ms (float): Q约束波形产生的时延。
        config (QAcceptanceConfig | None): 包含验收阈值的配置对象。如果为 None，则使用默认配置。

    Returns:
        QAcceptanceResult: 包含验收结果的对象，包括是否通过(passed)以及详细原因。
    """
    if config is None:
        config = QAcceptanceConfig()

    reasons: list[str] = []

    cc_pass = cc_q >= cc_tv - config.max_cc_drop_vs_tv
    env_pass = env_q >= env_tv - config.max_env_drop_vs_tv
    lag_pass = abs(lag_q_ms) <= config.max_lag_ms

    if not cc_pass:
        reasons.append("q_cc_drop_too_large")
    if not env_pass:
        reasons.append("q_envelope_drop_too_large")
    if not lag_pass:
        reasons.append("q_lag_too_large")

    passed = bool(cc_pass and env_pass and lag_pass)

    details = {
        "cc_q": float(cc_q),
        "cc_tv": float(cc_tv),
        "env_q": float(env_q),
        "env_tv": float(env_tv),
        "lag_q_ms": float(lag_q_ms),
        "cc_pass": bool(cc_pass),
        "env_pass": bool(env_pass),
        "lag_pass": bool(lag_pass),
        "passed": passed,
    }

    return QAcceptanceResult(
        passed=passed,
        reasons=reasons,
        details=details,
    )
