# -*- coding: utf-8 -*-
"""
Final thin runner for real-data tying experiments.
运行实际数据的主流程，包含 DTW、时变子波反演、Q 约束和最终模型选择。
模板文件，包含所需的函数等，运行不同的数据就新建对应的名字，然后直接调用就可以·
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter1d
# 把当前文件的路径转换为绝对路径（.resolve），并添加到 sys.path 中
# .parents[1] 表示当前文件所在的目录的上一级目录
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.config_loader import load_config
from core.acceptance import TvAcceptanceConfig, evaluate_tv_wavelet_candidate
from core.forward_operator import nonstationary_convolution, wavelet_lag_axis_ms, stationary_convolution
from core.metrics import tie_similarity_score
from core.signal_utils import match_rms
from core.wavelet_qc import (
    compute_boundary_peak_mask,
    compute_wavelet_qc_attributes,
)
from io_utils.load_initial_npz import load_run_data
from io_utils.save_results import copy_config_file, build_metrics_dict, save_result_bundle
from plotting.plot_qc import plot_all
from stages.stage_dtw import run_dtw_stage, DtwPhaseSpec
from stages.stage_q_constraint import try_q_constraint_stage
from stages.stage_tv_wavelet import run_tv_wavelet_stage
from stages.stage_wavelet_prior import (
    build_constant_phase_prior_with_fallback,
    build_ricker_prior,
    build_stationary_prior_with_fallback,
    make_center_ricker_wavelet,
)


FINAL_SOURCE_TV_CANDIDATE = 0
FINAL_SOURCE_Q_CONSTRAINED = 1
FINAL_SOURCE_STATIONARY_PRIOR = 2
FINAL_SOURCE_LOCAL_FALLBACK_PRIOR = 3
FINAL_SOURCE_LOCAL_BLEND = 4

def choose_final_model(
    *,
    alignment,
    W_pass,
    q_result,
    W_est_best,
    s_syn_tv_direct,
    w_prior,
    s_syn_prior,
    prior_source: str | None = None,
):
    """
    选择最终用于保存和绘图的模型。

    输入：Q 约束候选、时变子波候选、DTW 后平稳子波先验。
    输出：最终子波矩阵、最终合成记录、模型类型和 q_pass 标记。
    单位：子波矩阵单位沿用反演结果，合成记录单位与 obs_work 的 RMS 标定一致。
    物理意义：只决定采用哪个已经计算好的候选结果，不重新定义子波或反射系数。
    数学作用：按 Q 通过优先、W 通过其次、平稳先验兜底的顺序做候选选择。
    """
    if q_result is not None and q_result.accepted:
        return {
            "model_type": "Q_constrained",
            "W_final": q_result.W_q,
            "s_syn_final": q_result.s_syn_q,
            "q_pass": True,
        }

    if W_pass:
        return {
            "model_type": "W_est_best_no_Q",
            "W_final": W_est_best,
            "s_syn_final": s_syn_tv_direct,
            "q_pass": False,
        }

    fallback_model_type = "prior_after_DTW_global_fallback"
    return {
        "model_type": fallback_model_type,
        # 将一维子波 w_prior 扩展为二维矩阵，
        # 行数与 s_syn_prior 的长度相同，列数与 w_prior 的长度相同
        "W_final": np.tile(w_prior[None, :], (len(s_syn_prior), 1)),
        "s_syn_final": s_syn_prior,
        "q_pass": False,
    }


def _q_frequency_bounds(cfg, f_dom: float) -> tuple[float, float]:
    """
    计算 Q 约束使用的频带范围。

    输入：配置中的 Q 频带阈值和主频 f_dom。
    输出：Q 估计使用的 f_min、f_max，单位 Hz。
    单位：输入和输出频率均为 Hz。
    物理意义：Q 估计只在可信频带内比较浅部和深部频谱衰减。
    数学作用：把主频比例限制和绝对频带上下限合并为实际频带。
    """
    f_min = max(cfg.q.f_min_base, cfg.q.f_min_ratio_of_f_dom * f_dom)
    f_max = min(cfg.q.f_max_cap, cfg.q.f_max_ratio_of_f_dom * f_dom)
    return float(f_min), float(f_max)


def _build_tv_acceptance_config(cfg) -> TvAcceptanceConfig:
    """
    组装时变子波候选的验收阈值。

    输入：实验配置 cfg.acceptance。
    输出：TvAcceptanceConfig。
    单位：时间阈值为 ms，相关系数阈值为无量纲。
    物理意义：把时变子波是否可信的 QC 门槛集中展示，避免散落在主流程中。
    数学作用：只搬运阈值，不改变任何验收判据或数值大小。
    """
    acceptance_config = TvAcceptanceConfig(
        min_gain_vs_dtw=cfg.acceptance.min_gain_vs_dtw,
        min_gain_vs_stationary=cfg.acceptance.min_gain_vs_stationary,
        max_env_drop_vs_stationary=cfg.acceptance.max_env_drop_vs_stationary,
        max_lag_ms=cfg.acceptance.max_lag_ms,
        center_peak_median_ms=cfg.acceptance.center_peak_median_ms,
        center_peak_abs_p90_ms=cfg.acceptance.center_peak_abs_p90_ms,
        causal_peak_extra_margin_ms=cfg.acceptance.causal_peak_extra_margin_ms,
        tolerance_ms=cfg.acceptance.tolerance_ms,
        tolerance_cc=cfg.acceptance.tolerance_cc,
    )
    return acceptance_config


def _build_dtw_phase_specs(cfg) -> list[DtwPhaseSpec] | None:
    """
    从 YAML 读取可选的 dtw.phase_specs 参数组。
    输入：cfg.dtw.phase_specs（列表，每项为单个 DTW 阶段配置）。
    输出：DtwPhaseSpec 列表；若未配置则返回 None，沿用默认阶段参数。
    单位：max_shift_ms / smooth_ms / accept_max_lag_ms 单位为 ms。
    物理意义：仅参数化 DTW 候选与门控阈值，不改变 DTW 更新流程。
    数学作用：将配置映射为 run_dtw_stage 的 phase_specs 入参。
    """
    dtw_cfg = getattr(cfg, "dtw", None)
    if dtw_cfg is None or not hasattr(dtw_cfg, "phase_specs"):
        return None

    raw_phase_specs = getattr(dtw_cfg, "phase_specs")
    if raw_phase_specs is None:
        return None

    phase_specs: list[DtwPhaseSpec] = []
    for spec in raw_phase_specs:
        phase_specs.append(
            DtwPhaseSpec(
                phase=str(spec.phase),
                n_iter=int(spec.n_iter),
                alpha_candidates=[float(val) for val in spec.alpha_candidates],
                max_shift_ms=float(spec.max_shift_ms),
                smooth_ms=float(spec.smooth_ms),
                accept_min_cc_gain=float(spec.accept_min_cc_gain),
                accept_env_drop=float(spec.accept_env_drop),
                accept_max_lag_ms=float(spec.accept_max_lag_ms),
            )
        )

    if len(phase_specs) == 0:
        return None

    return phase_specs


def _get_local_wavelet_fallback_config(cfg) -> dict:
    """
    读取局部子波回退配置。旧配置没有 local_wavelet_fallback 时默认关闭。
    """
    local_cfg = getattr(cfg, "local_wavelet_fallback", None)
    if local_cfg is None:
        local_cfg = object()

    return {
        "enable": bool(getattr(local_cfg, "enable", False)),
        "peak_limit_ms": float(getattr(local_cfg, "peak_limit_ms", 15.0)),
        "energy_min": float(getattr(local_cfg, "energy_min", 0.15)),
        "pad_ms": float(getattr(local_cfg, "pad_ms", 0.0)),
        "alpha_smooth_samples": float(getattr(local_cfg, "alpha_smooth_samples", 10.0)),
        "use_shape_qc": bool(getattr(local_cfg, "use_shape_qc", True)),
        "side_lobe_limit_reliable": float(
            getattr(local_cfg, "side_lobe_limit_reliable", 0.75)
        ),
        "side_lobe_limit_fallback": float(
            getattr(local_cfg, "side_lobe_limit_fallback", 0.90)
        ),
        "side_lobe_guard_ms": float(getattr(local_cfg, "side_lobe_guard_ms", 12.0)),
        "edge_energy_limit_reliable": float(
            getattr(local_cfg, "edge_energy_limit_reliable", 0.15)
        ),
        "edge_energy_limit_fallback": float(
            getattr(local_cfg, "edge_energy_limit_fallback", 0.25)
        ),
        "edge_fraction": float(getattr(local_cfg, "edge_fraction", 0.15)),
    }


def compute_side_lobe_ratio(
    W: np.ndarray,
    dt: float,
    alignment: str,
    guard_ms: float,
) -> np.ndarray:
    """
    计算每个时刻子波远离主瓣区域的最大旁瓣比例。

    输入 W 为回退前的时变子波矩阵，dt 单位为 s，guard_ms 单位为 ms。
    输出 side_lobe_ratio 越大，表示远端旁瓣相对主瓣越强。
    """
    W = np.asarray(W, dtype=float)
    if W.ndim != 2:
        raise ValueError("W must be a 2D wavelet matrix.")
    if dt <= 0:
        raise ValueError("dt must be positive for side-lobe QC.")

    n_time, wavelet_length = W.shape
    lag_ms = wavelet_lag_axis_ms(wavelet_length, dt, alignment)
    guard_ms = max(0.0, float(guard_ms))
    main_mask = np.abs(lag_ms) <= guard_ms
    side_mask = ~main_mask

    side_lobe_ratio = np.full(n_time, np.nan, dtype=float)
    for it in range(n_time):
        w = W[it, :]
        main = np.abs(w[main_mask])
        side = np.abs(w[side_mask])
        if main.size == 0 or side.size == 0:
            continue

        main_amp = np.nanmax(main)
        side_amp = np.nanmax(side)
        if np.isfinite(main_amp) and np.isfinite(side_amp) and main_amp > 1e-12:
            side_lobe_ratio[it] = side_amp / (main_amp + 1e-12)

    return side_lobe_ratio


def compute_edge_energy_ratio(W: np.ndarray, edge_fraction: float) -> np.ndarray:
    """
    计算每个时刻子波两端能量占总能量的比例。

    输入 W 为回退前的时变子波矩阵。输出越大，表示子波边界振荡越强。
    """
    W = np.asarray(W, dtype=float)
    if W.ndim != 2:
        raise ValueError("W must be a 2D wavelet matrix.")

    n_time, wavelet_length = W.shape
    edge_fraction = float(edge_fraction)
    edge_fraction = min(max(edge_fraction, 0.0), 0.5)
    edge_n = max(1, int(round(edge_fraction * wavelet_length)))

    edge_energy_ratio = np.full(n_time, np.nan, dtype=float)
    for it in range(n_time):
        w = W[it, :]
        total_energy = np.nansum(w ** 2)
        edge_energy = np.nansum(w[:edge_n] ** 2) + np.nansum(w[-edge_n:] ** 2)
        if np.isfinite(total_energy) and np.isfinite(edge_energy) and total_energy > 1e-12:
            edge_energy_ratio[it] = edge_energy / (total_energy + 1e-12)

    return edge_energy_ratio


def build_shape_qc_masks(
    *,
    side_lobe_ratio: np.ndarray,
    edge_energy_ratio: np.ndarray,
    side_lobe_limit_reliable: float,
    side_lobe_limit_fallback: float,
    edge_energy_limit_reliable: float,
    edge_energy_limit_fallback: float,
) -> tuple[np.ndarray, np.ndarray]:
    """构造候选/最终子波通用的形态合格与严重异常掩码。

    输入数组 shape=(N_time,)，均为无量纲比值。输出两个同 shape 布尔数组：
    shape_ok_mask 表示旁瓣和边缘能量同时满足可靠阈值；
    shape_fallback_request_mask 表示至少一个指标超过严重异常阈值。
    本函数只给出诊断，不直接改变 W 或触发模型分支。
    """
    side_lobe_ratio = np.asarray(side_lobe_ratio, dtype=float).ravel()
    edge_energy_ratio = np.asarray(edge_energy_ratio, dtype=float).ravel()
    if side_lobe_ratio.size == 0 or side_lobe_ratio.size != edge_energy_ratio.size:
        raise ValueError("side_lobe_ratio 与 edge_energy_ratio 必须是一维等长非空数组。")

    shape_ok_mask = (
        np.isfinite(side_lobe_ratio)
        & np.isfinite(edge_energy_ratio)
        & (side_lobe_ratio <= side_lobe_limit_reliable)
        & (edge_energy_ratio <= edge_energy_limit_reliable)
    )
    shape_fallback_request_mask = (
        (
            np.isfinite(side_lobe_ratio)
            & (side_lobe_ratio > side_lobe_limit_fallback)
        )
        | (
            np.isfinite(edge_energy_ratio)
            & (edge_energy_ratio > edge_energy_limit_fallback)
        )
    )
    return shape_ok_mask, shape_fallback_request_mask


def build_final_qc_pass_mask(
    *,
    peak_metric_ms: np.ndarray,
    wavelet_energy_norm: np.ndarray,
    shape_ok_mask: np.ndarray,
    peak_limit_ms: float,
    energy_min: float,
    use_shape_qc: bool,
) -> np.ndarray:
    """计算最终 W 的峰值、能量和形态诊断通过掩码。

    所有输入均为 shape=(N_time,)；peak_metric_ms 单位 ms，其余无量纲。
    输出不包含 ``inversion_valid_mask``，因此只表示最终被使用子波的波形 QC，
    不能解释为“该样点由局部反演直接约束”。
    """
    peak_metric_ms = np.asarray(peak_metric_ms, dtype=float).ravel()
    wavelet_energy_norm = np.asarray(wavelet_energy_norm, dtype=float).ravel()
    shape_ok_mask = np.asarray(shape_ok_mask, dtype=bool).ravel()
    if not (
        peak_metric_ms.size
        == wavelet_energy_norm.size
        == shape_ok_mask.size
    ):
        raise ValueError("最终 QC 数组必须等长。")

    final_qc_pass_mask = (
        np.isfinite(peak_metric_ms)
        & (np.abs(peak_metric_ms) <= peak_limit_ms)
        & np.isfinite(wavelet_energy_norm)
        & (wavelet_energy_norm >= energy_min)
    )
    if use_shape_qc:
        final_qc_pass_mask = final_qc_pass_mask & shape_ok_mask
    return final_qc_pass_mask


def longest_true_gap_duration_ms(
    *,
    gap_mask_at_centers: np.ndarray,
    center_times_s: np.ndarray,
    fallback_center_step_s: float,
) -> float:
    """按稀疏反演中心时间计算最长连续 True 区间，输出单位 ms。

    输入两个数组 shape=(N_centers,)。区间长度使用
    ``t_end - t_start + median_center_step``，避免把 1 ms 地震采样间隔
    错当成通常为 5 ms 的子波反演中心步长。
    """
    gap_mask_at_centers = np.asarray(gap_mask_at_centers, dtype=bool).ravel()
    center_times_s = np.asarray(center_times_s, dtype=float).ravel()
    if gap_mask_at_centers.size != center_times_s.size:
        raise ValueError("gap mask 与 center_times_s 必须等长。")
    if center_times_s.size == 0 or not np.any(gap_mask_at_centers):
        return 0.0
    if not np.all(np.isfinite(center_times_s)):
        raise ValueError("center_times_s 包含 NaN 或 Inf。")

    if center_times_s.size > 1:
        center_time_differences_s = np.diff(center_times_s)
        if np.any(center_time_differences_s <= 0.0):
            raise ValueError("center_times_s 必须严格单调递增。")
        center_step_s = float(np.median(center_time_differences_s))
    else:
        center_step_s = float(fallback_center_step_s)
    if not np.isfinite(center_step_s) or center_step_s <= 0.0:
        raise ValueError("反演中心步长必须是有限正数，单位 s。")

    padded_mask = np.concatenate((
        np.array([False]),
        gap_mask_at_centers,
        np.array([False]),
    ))
    transitions = np.diff(padded_mask.astype(int))
    run_starts = np.where(transitions == 1)[0]
    run_stops = np.where(transitions == -1)[0] - 1

    longest_duration_s = 0.0
    for run_start, run_stop in zip(run_starts, run_stops):
        duration_s = (
            center_times_s[run_stop]
            - center_times_s[run_start]
            + center_step_s
        )
        longest_duration_s = max(longest_duration_s, float(duration_s))
    return longest_duration_s * 1000.0


def build_final_wavelet_source_codes(
    *,
    base_model_type: str,
    local_fallback_alpha: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """构造最终 W 的互斥模型来源编码。

    输入 alpha shape=(N_time,)，取值 1 为纯基础候选、0 为纯平稳先验、
    0~1 之间为局部回退的时间平滑混合。输出编码与五个互斥布尔掩码
    shape 均为 (N_time,)，只描述最终模型来源，不与候选 W 的插值谱系混用。
    """
    alpha = np.asarray(local_fallback_alpha, dtype=float).ravel()
    if alpha.size == 0 or not np.all(np.isfinite(alpha)):
        raise ValueError("local_fallback_alpha 必须是一维非空有限数组。")
    if np.any(alpha < 0.0) or np.any(alpha > 1.0):
        raise ValueError("local_fallback_alpha 必须位于 [0, 1]。")

    pure_prior_mask = np.isclose(alpha, 0.0, rtol=0.0, atol=1e-15)
    pure_base_mask = np.isclose(alpha, 1.0, rtol=0.0, atol=1e-15)
    local_blend_mask = (~pure_prior_mask) & (~pure_base_mask)

    if base_model_type == "W_est_best_no_Q":
        base_code = FINAL_SOURCE_TV_CANDIDATE
    elif base_model_type == "Q_constrained":
        base_code = FINAL_SOURCE_Q_CONSTRAINED
    elif base_model_type.startswith("stationary_"):
        base_code = FINAL_SOURCE_STATIONARY_PRIOR
    else:
        raise ValueError(f"未知最终模型类型: {base_model_type}")

    source_code = np.full(alpha.size, base_code, dtype=np.int8)
    source_code[pure_prior_mask] = FINAL_SOURCE_LOCAL_FALLBACK_PRIOR
    source_code[local_blend_mask] = FINAL_SOURCE_LOCAL_BLEND

    masks = {
        "tv_candidate_mask": source_code == FINAL_SOURCE_TV_CANDIDATE,
        "q_constrained_mask": source_code == FINAL_SOURCE_Q_CONSTRAINED,
        "stationary_prior_mask": source_code == FINAL_SOURCE_STATIONARY_PRIOR,
        "local_fallback_prior_mask": (
            source_code == FINAL_SOURCE_LOCAL_FALLBACK_PRIOR
        ),
        "local_blend_mask": source_code == FINAL_SOURCE_LOCAL_BLEND,
    }
    source_count = np.zeros(alpha.size, dtype=int)
    for source_mask in masks.values():
        source_count += source_mask.astype(int)
    if not np.all(source_count == 1):
        raise AssertionError("最终子波来源编码必须互斥且完整。")
    return source_code, masks


def build_reliable_wavelet_mask(
    *,
    valid_mask: np.ndarray,
    peak_metric_ms: np.ndarray,
    wavelet_energy: np.ndarray,
    side_lobe_ratio: np.ndarray,
    edge_energy_ratio: np.ndarray,
    dt: float,
    peak_limit_ms: float,
    energy_min: float,
    pad_ms: float,
    use_shape_qc: bool,
    side_lobe_limit_reliable: float,
    side_lobe_limit_fallback: float,
    edge_energy_limit_reliable: float,
    edge_energy_limit_fallback: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    构造时变子波局部可靠性掩码。
    输入：TVWI 有效掩码、波峰偏移、子波能量和阈值参数。
    输出：reliable_mask、fallback_mask、wavelet_energy_norm。
    单位：peak_limit_ms 和 pad_ms 为 ms，dt 为 s。
    物理意义：标记哪些时刻的时变子波可以解释，哪些时刻应回退到平稳子波。
    数学作用：将局部 QC 条件转化为最终 W_final 的选择掩码。
    """
    valid_mask = np.asarray(valid_mask, dtype=bool).ravel()
    peak_metric_ms = np.asarray(peak_metric_ms, dtype=float).ravel()
    wavelet_energy = np.asarray(wavelet_energy, dtype=float).ravel()
    side_lobe_ratio = np.asarray(side_lobe_ratio, dtype=float).ravel()
    edge_energy_ratio = np.asarray(edge_energy_ratio, dtype=float).ravel()

    if dt <= 0:
        raise ValueError("dt must be positive for local wavelet fallback.")

    if (
        len(valid_mask) != len(peak_metric_ms)
        or len(valid_mask) != len(wavelet_energy)
        or len(valid_mask) != len(side_lobe_ratio)
        or len(valid_mask) != len(edge_energy_ratio)
    ):
        raise ValueError(
            "valid_mask, peak_metric_ms, wavelet_energy, and shape QC arrays must have the same length."
        )

    energy_abs = np.abs(wavelet_energy)
    finite_energy = energy_abs[np.isfinite(energy_abs)]
    if finite_energy.size == 0:
        energy_max = np.nan
    else:
        energy_max = np.nanmax(finite_energy)

    if (not np.isfinite(energy_max)) or energy_max <= 0:
        wavelet_energy_norm = np.zeros_like(wavelet_energy, dtype=float)
    else:
        wavelet_energy_norm = energy_abs / (energy_max + 1e-12)
        wavelet_energy_norm = np.nan_to_num(
            wavelet_energy_norm,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    raw_shape_reliable_mask, raw_shape_fallback_mask = build_shape_qc_masks(
        side_lobe_ratio=side_lobe_ratio,
        edge_energy_ratio=edge_energy_ratio,
        side_lobe_limit_reliable=side_lobe_limit_reliable,
        side_lobe_limit_fallback=side_lobe_limit_fallback,
        edge_energy_limit_reliable=edge_energy_limit_reliable,
        edge_energy_limit_fallback=edge_energy_limit_fallback,
    )

    if use_shape_qc:
        shape_reliable_mask = raw_shape_reliable_mask
        shape_fallback_mask = raw_shape_fallback_mask
    else:
        shape_reliable_mask = np.ones_like(valid_mask, dtype=bool)
        shape_fallback_mask = np.zeros_like(valid_mask, dtype=bool)

    reliable_mask = (
        valid_mask
        & np.isfinite(peak_metric_ms)
        & (np.abs(peak_metric_ms) <= peak_limit_ms)
        & (wavelet_energy_norm >= energy_min)
        & shape_reliable_mask
    )

    fallback_mask = (
        (~np.isfinite(peak_metric_ms))
        | (np.abs(peak_metric_ms) > peak_limit_ms)
        | shape_fallback_mask
    )

    pad_samples = int(round(max(0.0, pad_ms) / 1000.0 / dt))
    if pad_samples > 0:
        fallback_mask = binary_dilation(
            fallback_mask,
            iterations=pad_samples,
        )

    return reliable_mask, fallback_mask, wavelet_energy_norm, shape_fallback_mask


def apply_local_wavelet_fallback(
    *,
    W_final: np.ndarray,
    w_stationary: np.ndarray,
    fallback_mask: np.ndarray,
    alpha_smooth_samples: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    对不可信时间段的最终时变子波矩阵做局部平稳子波回退。

    输入：W_final 为 shape (N_time, N_wavelet) 的子波矩阵，w_stationary 为
    shape (N_wavelet,) 的平稳子波，fallback_mask 为 shape (N_time,) 的布尔掩码。
    输出：相同 shape 的混合子波矩阵和 shape (N_time,) 的时变子波权重 alpha。
    物理意义：alpha=0 的时间点使用平稳子波，alpha=1 的时间点保留反演子波；
    仅在时间方向平滑过渡权重，不改变子波的时间/相位参考和采样单位。
    """
    W_final = np.asarray(W_final, dtype=float)
    if W_final.ndim != 2:
        raise ValueError("W_final must be a 2D wavelet matrix.")
    if W_final.shape[0] == 0 or W_final.shape[1] == 0:
        raise ValueError("W_final time and wavelet dimensions must be non-empty.")
    if not np.all(np.isfinite(W_final)):
        raise ValueError("W_final must contain only finite values.")

    w_stationary = np.asarray(w_stationary, dtype=float)
    if w_stationary.ndim != 1:
        raise ValueError("w_stationary must be a 1D wavelet.")
    if w_stationary.size == 0:
        raise ValueError("w_stationary must be non-empty.")
    if not np.all(np.isfinite(w_stationary)):
        raise ValueError("w_stationary must contain only finite values.")

    fallback_mask_array = np.asarray(fallback_mask)
    if fallback_mask_array.ndim != 1:
        raise ValueError("fallback_mask must be a 1D mask.")
    if fallback_mask_array.size == 0:
        raise ValueError("fallback_mask must be non-empty.")
    if not (
        np.issubdtype(fallback_mask_array.dtype, np.bool_)
        or np.issubdtype(fallback_mask_array.dtype, np.number)
    ):
        raise ValueError("fallback_mask must contain boolean or numeric values.")
    if not np.all(np.isfinite(fallback_mask_array)):
        raise ValueError("fallback_mask must contain only finite values.")
    fallback_mask = fallback_mask_array.astype(bool, copy=False)

    if W_final.shape[1] != len(w_stationary):
        raise ValueError("w_stationary length must match W_final wavelet length.")

    if W_final.shape[0] != len(fallback_mask):
        raise ValueError("fallback_mask length must match W_final time length.")

    alpha_smooth_samples_array = np.asarray(alpha_smooth_samples, dtype=float)
    if (
        alpha_smooth_samples_array.ndim != 0
        or not np.isfinite(alpha_smooth_samples_array)
        or alpha_smooth_samples_array < 0.0
    ):
        raise ValueError("alpha_smooth_samples must be a finite non-negative value.")
    alpha_smooth_samples = float(alpha_smooth_samples_array)

    alpha = np.ones(W_final.shape[0], dtype=float)
    alpha[fallback_mask] = 0.0

    if alpha_smooth_samples > 0:
        alpha = gaussian_filter1d(
            alpha,
            sigma=float(alpha_smooth_samples),
            mode="nearest",
        )

    alpha[fallback_mask] = 0.0
    alpha = np.clip(alpha, 0.0, 1.0)

    W_final_local = (
        alpha[:, None] * W_final
        + (1.0 - alpha[:, None]) * w_stationary[None, :]
    )
    return W_final_local, alpha


def _diag_bool_mask(
    diag: dict,
    key: str,
    n_time: int,
    *,
    default: bool = False,
) -> np.ndarray:
    value = diag.get(key, None)

    if value is None:
        return np.full(
            n_time,
            default,
            dtype=bool,
        )

    mask = np.asarray(
        value,
        dtype=bool,
    ).ravel()

    if mask.size != n_time:
        raise ValueError(
            f"tv.diag[{key!r}] length mismatch: "
            f"{mask.size} != {n_time}."
        )

    return mask


def bridge_short_gaps_between_reliable_centers(
    reliable_center_mask: np.ndarray,
    *,
    dt: float,
    short_gap_limit_ms: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    根据直接可靠中心构造 TV 可保留区域。

    direct reliable:
        始终保留。

    两个 reliable center 之间：
        若距离 <= short_gap_limit_ms，
        允许保留 TVWI 已经生成的插值结果。

    两端、长 gap：
        本函数不填充，后续会回 prior。
    """
    reliable_center_mask = np.asarray(
        reliable_center_mask,
        dtype=bool,
    ).ravel()

    if reliable_center_mask.size == 0:
        raise ValueError(
            "reliable_center_mask must be non-empty."
        )

    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(
            "dt must be finite and positive."
        )

    if (
        not np.isfinite(short_gap_limit_ms)
        or short_gap_limit_ms < 0.0
    ):
        raise ValueError(
            "short_gap_limit_ms must be finite "
            "and non-negative."
        )

    tv_support_mask = (
        reliable_center_mask.copy()
    )

    reliable_indices = np.flatnonzero(
        reliable_center_mask
    )

    dt_ms = float(dt) * 1000.0

    if reliable_indices.size >= 2:
        for left, right in zip(
            reliable_indices[:-1],
            reliable_indices[1:],
        ):
            left = int(left)
            right = int(right)

            span_ms = (
                right - left
            ) * dt_ms

            if (
                span_ms
                <= short_gap_limit_ms + 1e-12
            ):
                tv_support_mask[
                    left:right + 1
                ] = True

    short_gap_mask = (
        tv_support_mask
        & (~reliable_center_mask)
    )

    return (
        tv_support_mask,
        short_gap_mask,
    )


def _summarize_peak_metric(
    peak_metric_ms: np.ndarray,
    alignment: str,
) -> dict[str, float]:
    """汇总最终子波主峰位置；输入单位为 ms，输出为分位数标量（ms）。"""
    peak_metric_ms = np.asarray(peak_metric_ms, dtype=float)
    if peak_metric_ms.ndim != 1 or peak_metric_ms.size == 0:
        raise ValueError("peak_metric_ms 必须是一维非空数组。")
    if not np.all(np.isfinite(peak_metric_ms)):
        raise ValueError("peak_metric_ms 必须全部为有限值。")

    alignment = str(alignment).lower()
    if alignment not in ("center", "causal"):
        raise ValueError("alignment 必须是 'center' 或 'causal'。")

    peak_abs_p90 = np.nan
    if alignment == "center":
        peak_abs_p90 = float(np.percentile(np.abs(peak_metric_ms), 90))

    return {
        "p10": float(np.percentile(peak_metric_ms, 10)),
        "median": float(np.median(peak_metric_ms)),
        "p90": float(np.percentile(peak_metric_ms, 90)),
        "abs_p90": peak_abs_p90,
    }


def evaluate_wavelet_matrix_candidate(
    *,
    W: np.ndarray,
    r_time: np.ndarray,
    obs_work: np.ndarray,
    dt: float,
    alignment: str,
    cc_after_dtw: float,
    cc_prior: float,
    env_prior: float,
    cfg,
) -> dict:
    """
    对经过局部修复的完整 W 重新进行
    forward + global numerical/physical acceptance。
    """
    W = np.asarray(
        W,
        dtype=float,
    )

    if W.ndim != 2:
        raise ValueError(
            "W must be a 2D wavelet matrix."
        )

    if not np.all(np.isfinite(W)):
        raise ValueError(
            "W must contain only finite values."
        )

    s_syn = nonstationary_convolution(
        r_time=r_time,
        W=W,
        alignment=alignment,
    )

    s_syn = match_rms(
        s_syn,
        obs_work,
    )

    similarity_score, similarity = (
        tie_similarity_score(
            s_obs=obs_work,
            s_syn=s_syn,
            dt=dt,
            max_lag_ms=float(
                cfg.acceptance.max_lag_ms
            ),
        )
    )

    qc = compute_wavelet_qc_attributes(
        W,
        dt=dt,
        wavelet_alignment=alignment,
    )

    peak_summary = _summarize_peak_metric(
        qc["peak_metric_ms"],
        alignment,
    )

    acceptance_cfg = (
        _build_tv_acceptance_config(cfg)
    )

    peak_metric_ms = np.asarray(
        qc["peak_metric_ms"],
        dtype=float,
    )

    finite_peak_mask = np.isfinite(
        peak_metric_ms
    )

    if not np.any(finite_peak_mask):
        peak_violation_ratio = 1.0

    elif alignment == "center":
        peak_limit_ms = (
            float(
                acceptance_cfg
                .center_peak_abs_p90_ms
            )
            + float(
                acceptance_cfg.tolerance_ms
            )
        )

        peak_violation_mask = (
            np.abs(peak_metric_ms)
            > peak_limit_ms
        )

        peak_violation_ratio = float(
            np.mean(
                peak_violation_mask[
                    finite_peak_mask
                ]
            )
        )

    elif alignment == "causal":
        lo, hi = tuple(
            cfg.wavelet
            .causal_peak_allowed_ms
        )

        margin = (
            float(
                acceptance_cfg
                .causal_peak_extra_margin_ms
            )
            + float(
                acceptance_cfg.tolerance_ms
            )
        )

        peak_violation_mask = (
            (
                peak_metric_ms
                < float(lo) - margin
            )
            |
            (
                peak_metric_ms
                > float(hi) + margin
            )
        )

        peak_violation_ratio = float(
            np.mean(
                peak_violation_mask[
                    finite_peak_mask
                ]
            )
        )

    else:
        raise ValueError(
            "alignment must be "
            "'center' or 'causal'."
        )

    acceptance = (
        evaluate_tv_wavelet_candidate(
            alignment=alignment,

            cc_tv_direct=float(
                similarity["cc_direct"]
            ),

            cc_after_dtw=float(
                cc_after_dtw
            ),

            cc_stationary=float(
                cc_prior
            ),

            env_tv=float(
                similarity["env_cc"]
            ),

            env_stationary=float(
                env_prior
            ),

            best_lag_ms=float(
                similarity["best_lag_ms"]
            ),

            peak_metric_p10=float(
                peak_summary["p10"]
            ),

            peak_metric_med=float(
                peak_summary["median"]
            ),

            peak_metric_p90=float(
                peak_summary["p90"]
            ),

            peak_abs_p90=float(
                peak_summary["abs_p90"]
            ),

            causal_peak_allowed_ms=tuple(
                cfg.wavelet
                .causal_peak_allowed_ms
            ),

            final_peak_violation_ratio=float(
                peak_violation_ratio
            ),

            config=acceptance_cfg,
        )
    )

    return {
        "W": W,

        "s_syn": s_syn,

        "similarity_score": float(
            similarity_score
        ),

        "similarity": similarity,

        "qc": qc,

        "peak_summary": peak_summary,

        "peak_violation_ratio": float(
            peak_violation_ratio
        ),

        "acceptance": acceptance,
    }


def build_preacceptance_hybrid_candidate(
    *,
    tv,
    w_prior: np.ndarray,
    dt: float,
    alignment: str,
    local_fallback_cfg: dict,
) -> dict:
    """
    在 global acceptance 之前修复 raw TV candidate。

    不增加新的 YAML 参数。
    """
    W_tv_candidate = np.asarray(
        tv.W_best,
        dtype=float,
    )

    if W_tv_candidate.ndim != 2:
        raise ValueError(
            "tv.W_best must be 2D."
        )

    n_time = W_tv_candidate.shape[0]

    # ----------------------------------
    # 1. raw TV 完整 QC
    # ----------------------------------

    candidate_qc = (
        compute_wavelet_qc_attributes(
            W_tv_candidate,
            dt=dt,
            wavelet_alignment=alignment,
        )
    )

    candidate_side_lobe_ratio = (
        compute_side_lobe_ratio(
            W_tv_candidate,
            dt=dt,
            alignment=alignment,
            guard_ms=(
                local_fallback_cfg[
                    "side_lobe_guard_ms"
                ]
            ),
        )
    )

    candidate_edge_energy_ratio = (
        compute_edge_energy_ratio(
            W_tv_candidate,
            edge_fraction=(
                local_fallback_cfg[
                    "edge_fraction"
                ]
            ),
        )
    )

    (
        candidate_shape_ok_mask,
        candidate_shape_fallback_raw_mask,
    ) = build_shape_qc_masks(
        side_lobe_ratio=(
            candidate_side_lobe_ratio
        ),

        edge_energy_ratio=(
            candidate_edge_energy_ratio
        ),

        side_lobe_limit_reliable=(
            local_fallback_cfg[
                "side_lobe_limit_reliable"
            ]
        ),

        side_lobe_limit_fallback=(
            local_fallback_cfg[
                "side_lobe_limit_fallback"
            ]
        ),

        edge_energy_limit_reliable=(
            local_fallback_cfg[
                "edge_energy_limit_reliable"
            ]
        ),

        edge_energy_limit_fallback=(
            local_fallback_cfg[
                "edge_energy_limit_fallback"
            ]
        ),
    )

    # ----------------------------------
    # 2. A 区：direct + reliable
    # ----------------------------------

    (
        direct_reliable_mask,
        pointwise_fallback_request_mask,
        candidate_wavelet_energy_norm,
        candidate_shape_fallback_request_mask,
    ) = build_reliable_wavelet_mask(
        valid_mask=tv.valid_mask,

        peak_metric_ms=(
            candidate_qc["peak_metric_ms"]
        ),

        wavelet_energy=(
            candidate_qc["energy_l2"]
        ),

        side_lobe_ratio=(
            candidate_side_lobe_ratio
        ),

        edge_energy_ratio=(
            candidate_edge_energy_ratio
        ),

        dt=dt,

        peak_limit_ms=(
            local_fallback_cfg[
                "peak_limit_ms"
            ]
        ),

        energy_min=(
            local_fallback_cfg[
                "energy_min"
            ]
        ),

        pad_ms=(
            local_fallback_cfg[
                "pad_ms"
            ]
        ),

        use_shape_qc=(
            local_fallback_cfg[
                "use_shape_qc"
            ]
        ),

        side_lobe_limit_reliable=(
            local_fallback_cfg[
                "side_lobe_limit_reliable"
            ]
        ),

        side_lobe_limit_fallback=(
            local_fallback_cfg[
                "side_lobe_limit_fallback"
            ]
        ),

        edge_energy_limit_reliable=(
            local_fallback_cfg[
                "edge_energy_limit_reliable"
            ]
        ),

        edge_energy_limit_fallback=(
            local_fallback_cfg[
                "edge_energy_limit_fallback"
            ]
        ),
    )

    # ----------------------------------
    # 3. 自动定义 short gap
    # ----------------------------------

    estimate_step_samples = int(
        tv.diag.get(
            "estimate_step_samples",
            1,
        )
    )

    estimate_step_ms = (
        estimate_step_samples
        * float(dt)
        * 1000.0
    )

    time_smooth_sigma_samples = float(
        tv.diag.get(
            "time_smooth_sigma",
            0.0,
        )
    )

    time_smooth_ms = (
        time_smooth_sigma_samples
        * float(dt)
        * 1000.0
    )

    short_gap_limit_ms = max(
        2.0 * time_smooth_ms,
        2.0 * estimate_step_ms,
    )

    # ----------------------------------
    # 4. B 区：短 gap
    # ----------------------------------

    (
        tv_support_mask,
        short_gap_mask,
    ) = (
        bridge_short_gaps_between_reliable_centers(
            direct_reliable_mask,
            dt=dt,
            short_gap_limit_ms=(
                short_gap_limit_ms
            ),
        )
    )

    # ----------------------------------
    # 5. 来源谱系
    # ----------------------------------

    extrapolated_mask = _diag_bool_mask(
        tv.diag,
        "candidate_extrapolated_mask",
        n_time,
    )

    prior_fill_mask = _diag_bool_mask(
        tv.diag,
        "candidate_prior_fill_mask",
        n_time,
    )

    unavailable_mask = _diag_bool_mask(
        tv.diag,
        "candidate_unavailable_mask",
        n_time,
    )

    provenance_forced_prior_mask = (
        extrapolated_mask
        | prior_fill_mask
        | unavailable_mask
    )

    # endpoint extrapolation 永远不能
    # 被 short-gap bridge 救回来
    tv_support_mask = (
        tv_support_mask
        & (~provenance_forced_prior_mask)
    )

    # ----------------------------------
    # 6. C 区：必须回 prior
    # ----------------------------------

    proposed_fallback_mask = (
        (~tv_support_mask)

        | pointwise_fallback_request_mask

        | provenance_forced_prior_mask
    )

    short_gap_mask = (
        short_gap_mask
        & (~proposed_fallback_mask)
    )

    # ----------------------------------
    # 7. 混合
    # ----------------------------------

    enabled = bool(
        local_fallback_cfg["enable"]
    )

    if enabled:

        prior_for_hybrid = np.asarray(
            w_prior,
            dtype=float,
        ).copy()

        # W_best 已经过全局极性选择。
        # prior 必须与它保持同一极性。
        if bool(tv.use_negative):
            prior_for_hybrid = (
                -prior_for_hybrid
            )

        W_hybrid, alpha = (
            apply_local_wavelet_fallback(
                W_final=W_tv_candidate,

                w_stationary=(
                    prior_for_hybrid
                ),

                fallback_mask=(
                    proposed_fallback_mask
                ),

                alpha_smooth_samples=(
                    local_fallback_cfg[
                        "alpha_smooth_samples"
                    ]
                ),
            )
        )

        applied_fallback_mask = (
            proposed_fallback_mask.copy()
        )

    else:

        W_hybrid = (
            W_tv_candidate.copy()
        )

        alpha = np.ones(
            n_time,
            dtype=float,
        )

        applied_fallback_mask = (
            np.zeros(
                n_time,
                dtype=bool,
            )
        )

    # ----------------------------------
    # 8. 输出来源
    # ----------------------------------

    hard_prior_mask = np.isclose(
        alpha,
        0.0,
        rtol=0.0,
        atol=1e-15,
    )

    pure_tv_mask = np.isclose(
        alpha,
        1.0,
        rtol=0.0,
        atol=1e-15,
    )

    blend_mask = (
        (~hard_prior_mask)
        & (~pure_tv_mask)
    )

    return {
        "W_raw": W_tv_candidate,

        "W_hybrid": W_hybrid,

        "candidate_qc": candidate_qc,

        "candidate_side_lobe_ratio":
            candidate_side_lobe_ratio,

        "candidate_edge_energy_ratio":
            candidate_edge_energy_ratio,

        "candidate_shape_ok_mask":
            candidate_shape_ok_mask,

        "candidate_shape_fallback_raw_mask":
            candidate_shape_fallback_raw_mask,

        "direct_reliable_mask":
            direct_reliable_mask,

        "pointwise_fallback_request_mask":
            pointwise_fallback_request_mask,

        "candidate_shape_fallback_request_mask":
            candidate_shape_fallback_request_mask,

        "candidate_wavelet_energy_norm":
            candidate_wavelet_energy_norm,

        "tv_support_mask":
            tv_support_mask,

        "short_gap_mask":
            short_gap_mask,

        "extrapolated_mask":
            extrapolated_mask,

        "prior_fill_mask":
            prior_fill_mask,

        "unavailable_mask":
            unavailable_mask,

        "provenance_forced_prior_mask":
            provenance_forced_prior_mask,

        "proposed_fallback_mask":
            proposed_fallback_mask,

        "applied_fallback_mask":
            applied_fallback_mask,

        "alpha":
            alpha,

        "pure_tv_mask":
            pure_tv_mask,

        "hard_prior_mask":
            hard_prior_mask,

        "blend_mask":
            blend_mask,

        "short_gap_limit_ms":
            float(short_gap_limit_ms),

        "estimate_step_ms":
            float(estimate_step_ms),

        "time_smooth_ms":
            float(time_smooth_ms),

        "enabled":
            enabled,
    }


def _save_and_plot_results(
    *,
    config_path: str,
    cfg,
    data,
    initial_prior,
    dtw,
    after_prior,
    tv,
    q_result,
    metrics: dict,
    W_final: np.ndarray,
    s_syn_final: np.ndarray,
    final_qc_before_local_fallback: dict[str, np.ndarray],
    final_qc: dict[str, np.ndarray],
    w_ricker_qc: np.ndarray,
    s_syn_ricker_qc_initial: np.ndarray,
    s_syn_ricker_qc_after_dtw: np.ndarray,
    reliable_mask: np.ndarray | None = None,
    local_fallback_mask: np.ndarray | None = None,
    local_fallback_alpha: np.ndarray | None = None,
    wavelet_energy_norm: np.ndarray | None = None,
    side_lobe_ratio: np.ndarray | None = None,
    edge_energy_ratio: np.ndarray | None = None,
    W_final_before_local_fallback: np.ndarray | None = None,
    s_syn_final_before_local_fallback: np.ndarray | None = None,
    extended_wavelet_diagnostics: dict[str, np.ndarray] | None = None,
) -> None:
    """
    保存主流程结果并生成 QC 图件。

    输入：预处理数据、各阶段结果、最终模型和指标字典。
    输出：result_bundle.npz、metrics.json、诊断 JSON、配置副本和 QC 图件。
    单位：所有数组保持各阶段原单位，dt 为 s，频率为 Hz，峰值时间为 ms。
    物理意义：完整保留 obs、synthetic、r、TWT、W、Q 和 QC 中间量，便于复现。
    数学作用：只做持久化和绘图，不改变反演结果、不重新筛选模型。
    """
    result_dir = cfg.paths.result_dir
    alignment = cfg.wavelet.alignment
    center_max_peak_shift_ms = cfg.dtw.center_max_peak_shift_ms
    if extended_wavelet_diagnostics is None:
        extended_wavelet_diagnostics = {}

    stationary_candidate = after_prior.stationary_candidate_result
    stationary_candidate_w = (
        stationary_candidate.w
        if stationary_candidate is not None
        else np.array([], dtype=float)
    )
    stationary_candidate_s_syn_raw = (
        stationary_candidate.s_syn_raw
        if stationary_candidate is not None
        else np.array([], dtype=float)
    )

    save_result_bundle(
        result_dir=result_dir,
        depth=data.depth,
        t_work=data.t_work,
        obs_work=data.obs_work,
        r_work_init=data.r_work_init,
        r_work_final=dtw.r_time,
        r_depth_fixed=data.r_depth_fixed,
        twt_init=data.twt_init,
        twt_final=dtw.twt,
        v_final=dtw.v,
        w_prior=after_prior.w,
        s_syn_stationary=after_prior.s_syn,
        s_syn_after_dtw=dtw.s_syn,
        W_est=tv.W_raw,
        W_est_best=tv.W_best,
        s_syn_tv_direct=tv.s_syn,
        peak_metric_ms=tv.peak_metric_ms,
        W_Q=q_result.W_q,
        s_syn_Q=q_result.s_syn_q,
        W_final=W_final,
        s_syn_final=s_syn_final,
        metrics=metrics,
        tv_diag=tv.diag,
        dtw_history=dtw.history,
        acceptance_reasons=tv.acceptance_reasons,
        q_reason=q_result.reason,
        extra_arrays={
            "v_eff": data.v_eff,
            "rho_eff": data.rho_eff,
            "w_ricker_for_dtw": initial_prior.w,
            "s_syn_ricker_before_dtw": initial_prior.s_syn,
            "w0_for_dtw_final": dtw.w_for_dtw_final,
            "s_syn_prior": after_prior.s_syn,
            "stationary_candidate_w_raw": stationary_candidate_w,
            "stationary_candidate_s_syn_raw": stationary_candidate_s_syn_raw,
            "constant_phase_zero_phase_wavelet": (
                after_prior.constant_phase_zero_phase_wavelet
            ),
            "constant_phase_phase_grid_deg": (
                after_prior.constant_phase_phase_grid_deg
            ),
            "constant_phase_phase_cc": after_prior.constant_phase_phase_cc,
            "constant_phase_spectrum_hz": after_prior.constant_phase_spectrum_hz,
            "constant_phase_spectrum_amplitude": (
                after_prior.constant_phase_spectrum_amplitude
            ),
            "dt": np.asarray(data.dt),
            "f_dom": np.asarray(data.f_dom),
            "tv_valid_mask": tv.valid_mask,
            "tv_skip_code": tv.skip_code,
            "tv_robust_sigma": tv.diag.get("robust_sigma", np.array([])),
            "tv_irls_iterations": tv.diag.get("irls_iterations", np.array([])),
            "tv_irls_converged": tv.diag.get("irls_converged", np.array([])),
            "tv_robust_code": tv.diag.get("robust_code", np.array([])),
            "tv_robust_attempted": tv.diag.get("robust_attempted", np.array([])),
            "tv_robust_solution_used": tv.diag.get(
                "robust_solution_used",
                np.array([]),
            ),
            "tv_weight_mean": tv.diag.get("weight_mean", np.array([])),
            "tv_weight_min": tv.diag.get("weight_min", np.array([])),
            "tv_outlier_fraction": tv.diag.get("outlier_fraction", np.array([])),
            "tv_effective_sample_size": tv.diag.get(
                "effective_sample_size",
                np.array([]),
            ),
            "tv_kish_ratio": tv.diag.get("kish_ratio", np.array([])),
            "tv_effective_sample_ratio": tv.diag.get(
                "effective_sample_ratio",
                np.array([]),
            ),
            "tv_student_objective_initial": tv.diag.get(
                "student_objective_initial",
                np.array([]),
            ),
            "tv_student_objective_final": tv.diag.get(
                "student_objective_final",
                np.array([]),
            ),
            "tv_student_objective_relative_decrease": tv.diag.get(
                "student_objective_relative_decrease",
                np.array([]),
            ),
            "tv_edge_penalty_weights": tv.diag.get(
                "edge_penalty_weights",
                np.array([], dtype=float),
            ),
            "tv_robust_center_indices": tv.diag.get(
                "robust_center_indices",
                np.array([]),
            ),
            "tv_robust_weight_map": tv.diag.get(
                "robust_weight_map",
                np.empty((0, 0), dtype=np.float32),
            ),
            "tv_robust_weight_center_indices": tv.diag.get(
                "robust_weight_center_indices",
                np.array([], dtype=int),
            ),
            "tv_robust_window_offsets_samples": tv.diag.get(
                "robust_window_offsets_samples",
                np.array([], dtype=int),
            ),
            "tv_residual_median": tv.residual_median,
            "tv_robust_weight_solution_used": tv.diag.get(
                "robust_weight_solution_used",
                np.array([], dtype=bool),
            ),
            "tv_robust_weight_post_qc_valid": tv.diag.get(
                "robust_weight_post_qc_valid",
                np.array([], dtype=bool),
            ),
            "tv_robust_weight_skip_code": tv.diag.get(
                "robust_weight_skip_code",
                np.array([], dtype=int),
            ),
            "tv_reflectivity_energy": tv.reflectivity_energy,
            "tv_wavelet_energy": tv.wavelet_energy,
            "tv_centroid_frequency_hz": tv.centroid_frequency_hz,
            "tv_candidate_peak_metric_ms": tv.peak_metric_ms,
            "tv_reliable_mask": reliable_mask,
            "tv_local_fallback_mask": local_fallback_mask,
            "tv_local_fallback_alpha": local_fallback_alpha,
            "tv_wavelet_energy_norm": wavelet_energy_norm,
            "tv_side_lobe_ratio": side_lobe_ratio,
            "tv_edge_energy_ratio": edge_energy_ratio,
            "W_final_before_local_fallback": W_final_before_local_fallback,
            "s_syn_final_before_local_fallback": s_syn_final_before_local_fallback,
            "final_peak_metric_ms_before_local_fallback": (
                final_qc_before_local_fallback["peak_metric_ms"]
            ),
            "final_wavelet_energy_l2_before_local_fallback": (
                final_qc_before_local_fallback["energy_l2"]
            ),
            "final_wavelet_energy_norm_before_local_fallback": (
                final_qc_before_local_fallback["energy_l2_norm"]
            ),
            "final_centroid_frequency_hz_before_local_fallback": (
                final_qc_before_local_fallback["centroid_frequency_hz"]
            ),
            "final_bandwidth_hz_before_local_fallback": (
                final_qc_before_local_fallback["bandwidth_hz"]
            ),
            "final_peak_metric_ms_after_local_fallback": final_qc["peak_metric_ms"],
            "final_wavelet_energy_l2_after_local_fallback": final_qc["energy_l2"],
            "final_wavelet_energy_norm_after_local_fallback": final_qc[
                "energy_l2_norm"
            ],
            "final_centroid_frequency_hz_after_local_fallback": final_qc[
                "centroid_frequency_hz"
            ],
            "final_bandwidth_hz_after_local_fallback": final_qc["bandwidth_hz"],
            "w_ricker_35hz_qc": w_ricker_qc,
            "s_syn_ricker_35hz_initial_reflectivity": s_syn_ricker_qc_initial,
            "s_syn_ricker_35hz_after_dtw_reflectivity": s_syn_ricker_qc_after_dtw,
            **extended_wavelet_diagnostics,
        },
    )
    copy_config_file(config_path, result_dir, out_name="config_used.yaml")

    # 图件展示的是最终 W，因此绘图掩码和形态曲线也必须使用最终口径。
    plot_reliable_mask = extended_wavelet_diagnostics.get(
        "final_reliable_mask",
        reliable_mask,
    )
    plot_side_lobe_ratio = extended_wavelet_diagnostics.get(
        "final_side_lobe_ratio",
        side_lobe_ratio,
    )
    plot_edge_energy_ratio = extended_wavelet_diagnostics.get(
        "final_edge_energy_ratio",
        edge_energy_ratio,
    )
    plot_all(
        result_dir=result_dir,
        t_work=data.t_work,
        obs_work=data.obs_work,
        s_syn_stationary=after_prior.s_syn,
        s_syn_after_dtw=dtw.s_syn,
        s_syn_tv_direct=tv.s_syn,
        s_syn_final=s_syn_final,
        W_final=W_final,
        W_est_best=tv.W_best,
        peak_metric_ms=final_qc["peak_metric_ms"],
        dt=data.dt,
        alignment=alignment,
        dtw_history=dtw.history,
        metrics=metrics,
        wavelet_energy=final_qc["energy_l2"],
        valid_mask=tv.valid_mask,
        reliable_mask=plot_reliable_mask,
        fallback_mask=local_fallback_mask,
        fallback_alpha=local_fallback_alpha,
        wavelet_energy_norm=final_qc["energy_l2_norm"],
        side_lobe_ratio=plot_side_lobe_ratio,
        edge_energy_ratio=plot_edge_energy_ratio,
        centroid_frequency_hz=final_qc["centroid_frequency_hz"],
        candidate_diagnostics=extended_wavelet_diagnostics,
        final_wavelet_source_code=extended_wavelet_diagnostics.get(
            "final_wavelet_source_code"
        ),
        robust_sigma=tv.diag.get("robust_sigma"),
        weight_mean=tv.diag.get("weight_mean"),
        weight_min=tv.diag.get("weight_min"),
        outlier_fraction=tv.diag.get("outlier_fraction"),
        effective_sample_ratio=tv.diag.get("effective_sample_ratio"),
        robust_code=tv.diag.get("robust_code"),
        robust_attempted=tv.diag.get("robust_attempted"),
        robust_solution_used=tv.diag.get("robust_solution_used"),
        irls_iterations=tv.diag.get("irls_iterations"),
        irls_converged=tv.diag.get("irls_converged"),
        objective_relative_decrease=tv.diag.get(
            "student_objective_relative_decrease"
        ),
        min_effective_sample_ratio=float(
            tv.diag.get("min_effective_sample_ratio", 0.30)
        ),
        robust_scale_floor=tv.diag.get("robust_scale_floor"),
        robust_outlier_weight_threshold=float(
            tv.diag.get("robust_outlier_weight_threshold", 0.5)
        ),
        robust_weight_map=tv.diag.get("robust_weight_map"),
        robust_weight_center_indices=tv.diag.get(
            "robust_weight_center_indices"
        ),
        robust_window_offsets_samples=tv.diag.get(
            "robust_window_offsets_samples"
        ),
        robust_weight_solution_used=tv.diag.get(
            "robust_weight_solution_used"
        ),
        robust_weight_post_qc_valid=tv.diag.get(
            "robust_weight_post_qc_valid"
        ),
        robust_weight_skip_code=tv.diag.get(
            "robust_weight_skip_code"
        ),
        robust_weight_row_filter=getattr(
            cfg.tv_wavelet,
            "weight_map_row_filter",
            "all",
        ),
        center_limit_ms=center_max_peak_shift_ms,
        center_peak_allowed_ms=tuple(cfg.wavelet.center_peak_allowed_ms),
        causal_peak_allowed_ms=tuple(cfg.wavelet.causal_peak_allowed_ms),
        r_time=dtw.r_time,
        w_ricker=w_ricker_qc,
        w_prior=after_prior.w,
        w_rejected_stationary=(
            after_prior.stationary_candidate_result.w
            if (
                after_prior.stationary_candidate_result is not None
                and not after_prior.stationary_candidate_accepted
            )
            else None
        ),
    )

    comparison_cfg = getattr(cfg, "comparison", None)
    comparison_enabled = bool(
        getattr(comparison_cfg, "enable", False)
    ) if comparison_cfg is not None else False

    if comparison_enabled:
        l2_result_dir = getattr(
            comparison_cfg,
            "l2_result_dir",
            None,
        )
        output_subdir = getattr(
            comparison_cfg,
            "output_subdir",
            "l2_student_t_comparison",
        )

        if l2_result_dir:
            try:
                from analysis.compare_l2_student_t import (
                    compare_result_directories,
                )

                compare_result_directories(
                    l2_result_dir=l2_result_dir,
                    student_t_result_dir=result_dir,
                    output_dir=(
                        Path(result_dir) / str(output_subdir)
                    ),
                    strict_fairness=bool(
                        getattr(
                            comparison_cfg,
                            "strict_fairness",
                            True,
                        )
                    ),
                )
            except (FileNotFoundError, ValueError, OSError) as exc:
                print(
                    "[Comparison warning] 未生成 L2/Student-t 对比表："
                    f"{exc}"
                )
        else:
            print(
                "[Comparison warning] comparison.enable=true，"
                "但未设置 comparison.l2_result_dir。"
            )


class Tee:
    """
    同时向终端和日志文件写入输出的辅助类
    """
    def __init__(self, log_path: Path):
        self.file = open(log_path, "w", encoding="utf-8")
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        sys.stdout = self
        sys.stderr = self

    def write(self, data):
        self.file.write(data)
        self.file.flush()  # 确保崩溃时日志也完整写入磁盘
        self.stdout.write(data)

    def flush(self):
        self.file.flush()
        self.stdout.flush()

    def close(self):
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        self.file.close()


def main(config_path: str):
    cfg = load_config(config_path)
    result_dir = Path(cfg.paths.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    log_path = result_dir / "run.log"
    
    tee = Tee(log_path)
    try:
        return _main_impl(config_path)
    finally:
        tee.close()


def _main_impl(config_path: str):
    cfg = load_config(config_path)
    data = load_run_data(cfg.paths.input_npz)

    result_dir = cfg.paths.result_dir
    alignment = cfg.wavelet.alignment
    center_max_peak_shift_ms = cfg.dtw.center_max_peak_shift_ms
    well_name = getattr(cfg.experiment, "well", "unknown")

    print("=" * 60)
    print(f"Real-data experiment ({well_name}): {cfg.experiment.name}")
    print(f"alignment = {alignment}")
    print(f"result_dir = {result_dir}")
    print("=" * 60)

    # DTW 前只使用中心 Ricker 子波。
    # 物理意义：避免平稳子波在初始时深关系未校正前吸收残余时差。
    initial_prior = build_ricker_prior(
        r_time=data.r_work_init,
        s_obs=data.obs_work,
        dt=data.dt,
        f_dom=data.f_dom,
        wavelet_length_s=cfg.wavelet.length_s,
        alignment=alignment,
        source="center_ricker_for_DTW",
        data_window_factor=cfg.wavelet.data_window_factor,
        verbose=True,
    )

    print(f"[Initial prior] source = {initial_prior.source}")
    print(f"[Initial prior] CC = {initial_prior.cc:.6f}")

    if not cfg.dtw.enable:
        raise NotImplementedError("This runner expects dtw.enable=true.")

    dtw_phase_specs = _build_dtw_phase_specs(cfg)

    # DTW 阶段固定当前子波，用观测与合成的时间差更新 TWT(z) 和 r_time(t)。
    dtw = run_dtw_stage(
        obs_work=data.obs_work,
        r_time_init=data.r_work_init,
        w_initial=initial_prior.w,
        t_work=data.t_work,
        depth=data.depth,
        twt_init=data.twt_init,
        r_depth_fixed=data.r_depth_fixed,
        dt=data.dt,
        f_dom=data.f_dom,
        wavelet_length_pts=initial_prior.wavelet_length_pts,
        alignment=alignment,
        center_max_peak_shift_ms=center_max_peak_shift_ms,
        phase_specs=dtw_phase_specs,
        refresh_stationary_before_fine=False,
        w_source_for_dtw=initial_prior.source,
        verbose=True,
    )

    # DTW 后重新估计平稳子波先验，为时变子波反演提供物理可解释的参考子波。
    prior_method = str(
        getattr(cfg.stationary, "prior_method", "strict_stationary")
    ).strip().lower()

    if prior_method in {"constant_phase", "statistical_constant_phase"}:
        after_prior = build_constant_phase_prior_with_fallback(
            r_time=dtw.r_time,
            s_obs=data.obs_work,
            dt=data.dt,
            f_dom=data.f_dom,
            wavelet_length_s=cfg.wavelet.length_s,
            alignment=alignment,
            source="statistical_constant_phase_prior_after_DTW",
            fallback_source="center_ricker_prior_after_DTW",
            max_peak_shift_ms=center_max_peak_shift_ms,
            data_window_factor=cfg.wavelet.data_window_factor,
            verbose=True,
        )
    elif prior_method in {"strict_stationary", "stationary", "free_form"}:
        after_prior = build_stationary_prior_with_fallback(
            r_time=dtw.r_time,
            s_obs=data.obs_work,
            dt=data.dt,
            f_dom=data.f_dom,
            wavelet_length_s=cfg.wavelet.length_s,
            alignment=alignment,
            strict_source="strict_stationary_center_after_DTW",
            fallback_source="center_ricker_prior_after_DTW",
            mu1=cfg.stationary.mu1,
            mu2=cfg.stationary.mu2,
            mu_dc=cfg.stationary.mu_dc,
            mu_edge=float(getattr(cfg.stationary, "mu_edge", 0.0)),
            edge_fraction=float(getattr(cfg.stationary, "edge_fraction", 0.12)),
            edge_taper=str(getattr(cfg.stationary, "edge_taper", "cosine")),
            damping_ratio=cfg.stationary.damping_ratio,
            svd_cutoff_ratio=cfg.stationary.svd_cutoff_ratio,
            peak_lock=cfg.stationary.peak_lock,
            max_peak_shift_ms=center_max_peak_shift_ms,
            data_window_factor=cfg.wavelet.data_window_factor,
            verbose=True,
        )
    else:
        raise ValueError(
            "stationary.prior_method 必须是 "
            "'strict_stationary' 或 'constant_phase'，"
            f"实际为 {prior_method!r}。"
        )

    print(f"[DTW] CC = {dtw.cc_after:.6f}")
    print(f"[After-DTW prior] source = {after_prior.source}")
    print(f"[After-DTW prior] CC = {after_prior.cc:.6f}")

    # 时变子波反演固定 DTW 后反射系数，只估计随时间变化的子波矩阵 W(t, tau)。
    tv = run_tv_wavelet_stage(
        r_time=dtw.r_time,
        obs_work=data.obs_work,
        w_prior=after_prior.w,
        dt=data.dt,
        wavelet_length_pts=after_prior.wavelet_length_pts,
        data_window_length=after_prior.data_window_length,
        alignment=alignment,
        peak_allowed_ms=tuple(cfg.wavelet.center_peak_allowed_ms),
        causal_peak_allowed_ms=tuple(cfg.wavelet.causal_peak_allowed_ms),
        cc_stationary=after_prior.cc,
        cc_after_dtw=dtw.cc_after,
        env_stationary=after_prior.env_cc,
        w_prior_source=after_prior.source,
        center_max_peak_shift_ms=center_max_peak_shift_ms,
        mu1=cfg.tv_wavelet.mu1,
        mu2=cfg.tv_wavelet.mu2,
        mu_dc=cfg.tv_wavelet.mu_dc,
        mu_prior_strict=cfg.tv_wavelet.mu_prior_strict,
        mu_prior_fallback=cfg.tv_wavelet.mu_prior_fallback,
        mu_time=cfg.tv_wavelet.mu_time,
        edge_penalty_enabled=getattr(
            cfg.tv_wavelet,
            "edge_penalty_enabled",
            False,
        ),
        mu_edge=getattr(cfg.tv_wavelet, "mu_edge", 0.0),
        edge_fraction=getattr(cfg.tv_wavelet, "edge_fraction", 0.12),
        edge_taper=getattr(cfg.tv_wavelet, "edge_taper", "cosine"),
        energy_percentile=cfg.tv_wavelet.energy_percentile,
        damping_ratio=cfg.tv_wavelet.damping_ratio,
        svd_cutoff_ratio=cfg.tv_wavelet.svd_cutoff_ratio,
        estimate_step_ms=cfg.tv_wavelet.estimate_step_ms,
        time_smooth_ms=cfg.tv_wavelet.time_smooth_ms,
        wavelet_smooth_sigma=cfg.tv_wavelet.wavelet_smooth_sigma,
        reject_ill_conditioned=cfg.tv_wavelet.reject_ill_conditioned,
        reject_amplitude_jumps=cfg.tv_wavelet.reject_amplitude_jumps,
        loss_type=getattr(cfg.tv_wavelet, "loss_type", "l2"),
        student_nu=getattr(cfg.tv_wavelet, "student_nu", 10.0),
        irls_max_iter=getattr(cfg.tv_wavelet, "irls_max_iter", 10),
        irls_tol=getattr(cfg.tv_wavelet, "irls_tol", 1e-4),
        robust_scale_mode=getattr(
            cfg.tv_wavelet,
            "robust_scale_mode",
            "local_mad",
        ),
        robust_scale_floor_ratio=getattr(
            cfg.tv_wavelet,
            "robust_scale_floor_ratio",
            0.05,
        ),
        robust_weight_floor=getattr(
            cfg.tv_wavelet,
            "robust_weight_floor",
            1e-3,
        ),
        min_effective_sample_ratio=getattr(
            cfg.tv_wavelet,
            "min_effective_sample_ratio",
            0.30,
        ),
        robust_fallback=getattr(
            cfg.tv_wavelet,
            "robust_fallback",
            "l2",
        ),
        robust_outlier_weight_threshold=getattr(
            cfg.tv_wavelet,
            "robust_outlier_weight_threshold",
            0.5,
        ),
        store_weight_map=getattr(
            cfg.tv_wavelet,
            "store_weight_map",
            False,
        ),
        acceptance_config=_build_tv_acceptance_config(cfg),
        verbose=True,
    )

    local_fallback_cfg = (
        _get_local_wavelet_fallback_config(
            cfg
        )
    )

    hybrid = (
        build_preacceptance_hybrid_candidate(
            tv=tv,
            w_prior=after_prior.w,
            dt=data.dt,
            alignment=alignment,
            local_fallback_cfg=(
                local_fallback_cfg
            ),
        )
    )

    hybrid_eval = (
        evaluate_wavelet_matrix_candidate(
            W=hybrid["W_hybrid"],
            r_time=dtw.r_time,
            obs_work=data.obs_work,
            dt=data.dt,
            alignment=alignment,
            cc_after_dtw=dtw.cc_after,
            cc_prior=after_prior.cc,
            env_prior=after_prior.env_cc,
            cfg=cfg,
        )
    )

    raw_tv_W_pass = bool(
        tv.W_pass
    )

    hybrid_W_pass = bool(
        hybrid_eval[
            "acceptance"
        ].passed
    )

    print(
        "[Pre-acceptance hybrid] "
        f"enabled = {hybrid['enabled']}"
    )
    print(
        "[Pre-acceptance hybrid] "
        f"short gap limit = "
        f"{hybrid['short_gap_limit_ms']:.1f} ms"
    )
    print(
        "[Pre-acceptance hybrid] "
        f"direct reliable ratio = "
        f"{np.mean(hybrid['direct_reliable_mask']):.3f}"
    )
    print(
        "[Pre-acceptance hybrid] "
        f"short gap ratio = "
        f"{np.mean(hybrid['short_gap_mask']):.3f}"
    )
    print(
        "[Pre-acceptance hybrid] "
        f"hard prior ratio = "
        f"{np.mean(hybrid['hard_prior_mask']):.3f}"
    )
    print(
        "[Pre-acceptance hybrid] "
        f"blend ratio = "
        f"{np.mean(hybrid['blend_mask']):.3f}"
    )
    print(
        "[Pre-acceptance hybrid] "
        f"raw CC = {tv.cc_direct:.6f}"
    )
    print(
        "[Pre-acceptance hybrid] "
        f"hybrid CC = "
        f"{hybrid_eval['similarity']['cc_direct']:.6f}"
    )
    print(
        "[Pre-acceptance hybrid] "
        f"raw pass = {raw_tv_W_pass}"
    )
    print(
        "[Pre-acceptance hybrid] "
        f"hybrid pass = {hybrid_W_pass}"
    )
    print(
        "[Pre-acceptance hybrid] "
        f"reasons = "
        f"{hybrid_eval['acceptance'].reasons}"
    )

    # Q 约束只作为候选模型：通过 QC 才会进入最终结果，否则保留 TV 或平稳兜底。
    f_min, f_max = _q_frequency_bounds(
        cfg,
        data.f_dom,
    )

    q_result = try_q_constraint_stage(
        W_est_best=(
            hybrid["W_hybrid"]
        ),
        r_time=dtw.r_time,
        obs_work=data.obs_work,
        dt=data.dt,
        f_dom=data.f_dom,
        alignment=alignment,
        W_pass=hybrid_W_pass,
        enable_q=cfg.q.enable,
        cc_tv_direct=float(
            hybrid_eval[
                "similarity"
            ]["cc_direct"]
        ),
        env_tv=float(
            hybrid_eval[
                "similarity"
            ]["env_cc"]
        ),
        f_min=f_min,
        f_max=f_max,
        ref_range=tuple(
            cfg.q.ref_range
        ),
        deep_range=tuple(
            cfg.q.deep_range
        ),
        Q_min=cfg.q.Q_min,
        Q_max=cfg.q.Q_max,
        freeze_above_ref=(
            cfg.q.freeze_above_ref
        ),
        max_cc_drop_vs_tv=(
            cfg.q.max_cc_drop_vs_tv
        ),
        max_env_drop_vs_tv=(
            cfg.q.max_env_drop_vs_tv
        ),
        max_lag_ms=(
            cfg.q.max_lag_ms
        ),
        verbose=True,
    )

    # 最终模型选择只在已经完成的候选之间切换，不重新计算任何物理量。
    final = choose_final_model(
        alignment=alignment,
        W_pass=hybrid_W_pass,
        q_result=q_result,
        W_est_best=(
            hybrid["W_hybrid"]
        ),
        s_syn_tv_direct=(
            hybrid_eval["s_syn"]
        ),
        w_prior=after_prior.w,
        s_syn_prior=after_prior.s_syn,
    )

    W_final = np.asarray(
        final["W_final"],
        dtype=float,
    )
    s_syn_final = np.asarray(
        final["s_syn_final"],
        dtype=float,
    )
    final_model_type = (
        final["model_type"]
    )
    q_pass = bool(
        final["q_pass"]
    )

    W_final_before_local_fallback = (
        np.asarray(
            tv.W_best,
            dtype=float,
        ).copy()
    )

    s_syn_final_before_local_fallback = (
        np.asarray(
            tv.s_syn,
            dtype=float,
        ).copy()
    )

    W_tv_candidate = np.asarray(
        tv.W_best,
        dtype=float,
    )

    local_peak_limit_ms = float(
        local_fallback_cfg["peak_limit_ms"]
    )

    local_energy_min = float(
        local_fallback_cfg["energy_min"]
    )

    local_pad_ms = float(
        local_fallback_cfg["pad_ms"]
    )

    local_alpha_smooth_samples = float(
        local_fallback_cfg["alpha_smooth_samples"]
    )

    candidate_reliable_mask = (
        hybrid["direct_reliable_mask"]
    )

    candidate_fallback_request_mask = (
        hybrid[
            "pointwise_fallback_request_mask"
        ]
    )

    candidate_wavelet_energy_norm = (
        hybrid[
            "candidate_wavelet_energy_norm"
        ]
    )

    candidate_shape_fallback_request_mask = (
        hybrid[
            "candidate_shape_fallback_request_mask"
        ]
    )

    candidate_shape_fallback_raw_mask = (
        hybrid[
            "candidate_shape_fallback_raw_mask"
        ]
    )

    candidate_shape_ok_mask = (
        hybrid["candidate_shape_ok_mask"]
    )

    candidate_side_lobe_ratio = (
        hybrid[
            "candidate_side_lobe_ratio"
        ]
    )

    candidate_edge_energy_ratio = (
        hybrid[
            "candidate_edge_energy_ratio"
        ]
    )

    candidate_qc = (
        hybrid["candidate_qc"]
    )

    reliable_mask = (
        candidate_reliable_mask
    )

    wavelet_energy_norm = (
        candidate_wavelet_energy_norm
    )

    side_lobe_ratio = (
        candidate_side_lobe_ratio
    )

    edge_energy_ratio = (
        candidate_edge_energy_ratio
    )

    shape_fallback_mask = (
        candidate_shape_fallback_request_mask
    )

    local_fallback_enabled = bool(
        hybrid["enabled"]
    )

    local_fallback_mask = (
        hybrid["applied_fallback_mask"]
    )

    local_fallback_alpha = (
        hybrid["alpha"]
    )

    local_fallback_hard_ratio = float(
        np.mean(
            local_fallback_mask
        )
    )

    local_fallback_effective_ratio = float(
        np.mean(
            1.0
            - local_fallback_alpha
        )
    )

    local_fallback_ratio = (
        local_fallback_effective_ratio
    )

    final_uses_preacceptance_hybrid = bool(
        local_fallback_enabled
        and final_model_type == "W_est_best_no_Q"
    )

    final_model_display_type = (
        "W_est_best_no_Q_preacceptance_hybrid"
        if final_uses_preacceptance_hybrid
        else final_model_type
    )

    print(
        "[Local fallback] "
        f"enabled = {local_fallback_enabled}"
    )
    print(
        "[Local fallback] "
        f"hard_ratio      = "
        f"{local_fallback_hard_ratio:.3f}"
    )
    print(
        "[Local fallback] "
        f"effective_ratio = "
        f"{local_fallback_effective_ratio:.3f}"
    )
    print(
        "[Candidate QC] "
        f"reliable_ratio  = "
        f"{float(np.mean(candidate_reliable_mask)):.3f}"
    )
    print(
        "[Candidate QC] "
        f"shape_bad_ratio = "
        f"{float(np.mean(candidate_shape_fallback_raw_mask)):.3f}"
    )
    print(
        "[Local fallback] "
        f"shape_ratio     = "
        f"{float(np.mean(local_fallback_mask & candidate_shape_fallback_request_mask)):.3f}"
    )

    # 最终 QC 必须对应最终被保存和绘图的 W，而不是 local fallback 之前的 TV 候选。
    final_qc_before_local_fallback = compute_wavelet_qc_attributes(
        W_final_before_local_fallback,
        dt=data.dt,
        wavelet_alignment=alignment,
    )
    final_qc = compute_wavelet_qc_attributes(
        W_final,
        dt=data.dt,
        wavelet_alignment=alignment,
    )
    final_side_lobe_ratio = compute_side_lobe_ratio(
        W_final,
        dt=data.dt,
        alignment=alignment,
        guard_ms=local_fallback_cfg["side_lobe_guard_ms"],
    )
    final_edge_energy_ratio = compute_edge_energy_ratio(
        W_final,
        edge_fraction=local_fallback_cfg["edge_fraction"],
    )
    final_shape_ok_mask, final_shape_fallback_request_mask = build_shape_qc_masks(
        side_lobe_ratio=final_side_lobe_ratio,
        edge_energy_ratio=final_edge_energy_ratio,
        side_lobe_limit_reliable=local_fallback_cfg["side_lobe_limit_reliable"],
        side_lobe_limit_fallback=local_fallback_cfg["side_lobe_limit_fallback"],
        edge_energy_limit_reliable=local_fallback_cfg["edge_energy_limit_reliable"],
        edge_energy_limit_fallback=local_fallback_cfg["edge_energy_limit_fallback"],
    )
    final_reliable_mask = build_final_qc_pass_mask(
        peak_metric_ms=final_qc["peak_metric_ms"],
        wavelet_energy_norm=final_qc["energy_l2_norm"],
        shape_ok_mask=final_shape_ok_mask,
        peak_limit_ms=local_peak_limit_ms,
        energy_min=local_energy_min,
        use_shape_qc=local_fallback_cfg["use_shape_qc"],
    )

    boundary_peak_fraction_parameter = float(
        tv.params.get("edge_fraction", 0.12)
    )
    candidate_boundary_peak_mask = compute_boundary_peak_mask(
        W_tv_candidate,
        boundary_fraction=boundary_peak_fraction_parameter,
    )
    final_boundary_peak_mask = compute_boundary_peak_mask(
        W_final,
        boundary_fraction=boundary_peak_fraction_parameter,
    )
    candidate_boundary_peak_fraction = float(
        np.mean(candidate_boundary_peak_mask)
    )
    final_boundary_peak_fraction = float(np.mean(final_boundary_peak_mask))

    # gap 只在实际尝试反演的稀疏中心上统计，不能把 skip_code=0 的未尝试样点
    # 按 1 ms 地震采样误计为无效窗口。
    attempted_center_mask = np.asarray(tv.skip_code != 0, dtype=bool)
    attempted_center_indices = np.where(attempted_center_mask)[0]
    attempted_center_times_s = np.asarray(
        data.t_work[attempted_center_indices],
        dtype=float,
    )
    candidate_invalid_at_centers = ~np.asarray(
        tv.valid_mask[attempted_center_indices],
        dtype=bool,
    )
    candidate_unreliable_at_centers = ~np.asarray(
        candidate_reliable_mask[attempted_center_indices],
        dtype=bool,
    )
    fallback_center_step_s = (
        float(tv.diag.get("estimate_step_samples", 1)) * data.dt
    )
    if attempted_center_times_s.size > 1:
        candidate_center_step_ms = float(
            np.median(np.diff(attempted_center_times_s)) * 1000.0
        )
    else:
        candidate_center_step_ms = fallback_center_step_s * 1000.0
    longest_candidate_invalid_gap_ms = longest_true_gap_duration_ms(
        gap_mask_at_centers=candidate_invalid_at_centers,
        center_times_s=attempted_center_times_s,
        fallback_center_step_s=fallback_center_step_s,
    )
    longest_candidate_unreliable_gap_ms = longest_true_gap_duration_ms(
        gap_mask_at_centers=candidate_unreliable_at_centers,
        center_times_s=attempted_center_times_s,
        fallback_center_step_s=fallback_center_step_s,
    )

    base_final_model_type = (
        final["model_type"]
    )

    if (
        base_final_model_type
        == "W_est_best_no_Q"
    ):
        final_source_alpha = (
            hybrid["alpha"]
        )
    else:
        final_source_alpha = np.ones(
            len(data.t_work),
            dtype=float,
        )

    (
        final_wavelet_source_code,
        final_wavelet_source_masks,
    ) = build_final_wavelet_source_codes(
        base_model_type=(
            base_final_model_type
        ),
        local_fallback_alpha=(
            final_source_alpha
        ),
    )
    peak_summary_before_local_fallback = _summarize_peak_metric(
        final_qc_before_local_fallback["peak_metric_ms"],
        alignment,
    )
    final_peak_summary = _summarize_peak_metric(
        final_qc["peak_metric_ms"],
        alignment,
    )

    _, final_similarity_before_local_fallback = (
        tie_similarity_score(
            s_obs=data.obs_work,
            s_syn=s_syn_final_before_local_fallback,
            dt=data.dt,
            max_lag_ms=8.0,
        )
    )
    CC_final_before_local_fallback = float(final_similarity_before_local_fallback["cc_direct"])

    _, final_similarity_details = tie_similarity_score(
        s_obs=data.obs_work,
        s_syn=s_syn_final,
        dt=data.dt,
        max_lag_ms=8.0,
    )
    CC_final = float(final_similarity_details["cc_direct"])
    CC_final_direct = CC_final

    # 仅诊断 fallback 后的最终模型是否仍满足原 TV 门槛，不改变 W_pass 或模型选择。
    post_local_fallback_acceptance = None
    if local_fallback_enabled:
        post_local_fallback_acceptance = evaluate_tv_wavelet_candidate(
            alignment=alignment,
            # 保持现有 TV 验收的数据口径：这里传综合相似度，不改动历史阈值语义。
            cc_tv_direct=CC_final,
            cc_after_dtw=dtw.cc_after,
            cc_stationary=after_prior.cc,
            env_tv=final_similarity_details["env_cc"],
            env_stationary=after_prior.env_cc,
            best_lag_ms=final_similarity_details["best_lag_ms"],
            peak_metric_p10=final_peak_summary["p10"],
            peak_metric_med=final_peak_summary["median"],
            peak_metric_p90=final_peak_summary["p90"],
            peak_abs_p90=final_peak_summary["abs_p90"],
            causal_peak_allowed_ms=tuple(cfg.wavelet.causal_peak_allowed_ms),
            config=_build_tv_acceptance_config(cfg),
        )
        print(
            "[Local fallback] post-QC pass = "
            f"{post_local_fallback_acceptance.passed}; "
            f"reasons = {post_local_fallback_acceptance.reasons}"
        )

    # 35 Hz Ricker 只作为 QC 基准；同一子波分别检查初始和 DTW 后反射系数。
    ricker_qc_frequency_hz = 35.0
    w_ricker_qc = make_center_ricker_wavelet(
        f0=ricker_qc_frequency_hz,
        dt=data.dt,
        length=initial_prior.wavelet_length_pts,
    )
    s_syn_ricker_qc_initial = stationary_convolution(
        data.r_work_init,
        w_ricker_qc,
        alignment=alignment,
    )
    s_syn_ricker_qc_after_dtw = stationary_convolution(
        dtw.r_time,
        w_ricker_qc,
        alignment=alignment,
    )
    CC_ricker_qc_initial, ricker_qc_initial_details = tie_similarity_score(
        s_obs=data.obs_work,
        s_syn=s_syn_ricker_qc_initial,
        dt=data.dt,
        max_lag_ms=8.0,
    )
    CC_ricker_qc_after_dtw, ricker_qc_after_dtw_details = tie_similarity_score(
        s_obs=data.obs_work,
        s_syn=s_syn_ricker_qc_after_dtw,
        dt=data.dt,
        max_lag_ms=8.0,
    )

    stationary_candidate = after_prior.stationary_candidate_result
    if stationary_candidate is not None:
        CC_stationary_raw = float(stationary_candidate.cc)
        stationary_candidate_env_cc_raw = float(stationary_candidate.env_cc)
    else:
        CC_stationary_raw = np.nan
        stationary_candidate_env_cc_raw = np.nan

    candidate_temporal_difference_energy = float(
        np.mean(np.sum(np.diff(W_tv_candidate, axis=0) ** 2, axis=1))
    )
    final_temporal_difference_energy = float(
        np.mean(np.sum(np.diff(W_final, axis=0) ** 2, axis=1))
    )

    robust_attempted_mask = np.asarray(
        tv.diag.get("robust_attempted", np.zeros_like(data.t_work, dtype=bool)),
        dtype=bool,
    )
    robust_iterations = np.asarray(
        tv.diag.get("irls_iterations", np.zeros_like(data.t_work, dtype=int)),
        dtype=int,
    )
    robust_iterations_used = robust_iterations[robust_attempted_mask]
    if robust_iterations_used.size > 0:
        irls_iterations_median = float(np.median(robust_iterations_used))
        irls_iterations_max = int(np.max(robust_iterations_used))
    else:
        irls_iterations_median = np.nan
        irls_iterations_max = 0

    metrics = build_metrics_dict(
        CC_initial_ricker_before_DTW=initial_prior.cc,
        CC_stationary=after_prior.cc,
        CC_after_DTW=dtw.cc_after,
        CC_prior_after_DTW=after_prior.cc,
        CC_tv_direct=float(
            hybrid_eval["similarity"]["cc_direct"]
        ),
        W_pass=hybrid_W_pass,
        q_pass=q_pass,
        CC_Q=q_result.cc_q,
        Q_global=q_result.Q_global,
        CC_final=CC_final,
        final_model_type=final_model_type,
        best_lag_ms=tv.best_lag_ms,
        peak_metric_p10=tv.peak_metric_p10,
        peak_metric_med=tv.peak_metric_med,
        peak_metric_p90=tv.peak_metric_p90,
        peak_abs_p90=tv.peak_abs_p90,
        valid_ratio=tv.valid_ratio,
        use_negative_W=tv.use_negative,
        acceptance_reasons=tv.acceptance_reasons,
        q_reason=q_result.reason,
    )
    valid_res_med = tv.residual_median[tv.valid_mask]
    abs_res_med = np.abs(valid_res_med[np.isfinite(valid_res_med)])
    residual_median_p90 = float(np.percentile(abs_res_med, 90)) if abs_res_med.size > 0 else np.nan
    residual_median_med = float(np.median(valid_res_med[np.isfinite(valid_res_med)])) if valid_res_med.size > 0 else np.nan

    metrics.update(
        {
            "raw_tv_W_pass": bool(tv.W_pass),
            "raw_tv_CC_direct": float(tv.cc_direct),
            "raw_tv_best_lag_ms": float(tv.best_lag_ms),
            "raw_tv_peak_abs_p90": float(tv.peak_abs_p90),
            "raw_tv_acceptance_reasons": list(tv.acceptance_reasons),
            "hybrid_tv_W_pass": bool(hybrid_W_pass),
            "hybrid_tv_CC_direct": float(
                hybrid_eval["similarity"]["cc_direct"]
            ),
            "hybrid_tv_env_cc": float(
                hybrid_eval["similarity"]["env_cc"]
            ),
            "hybrid_tv_best_lag_ms": float(
                hybrid_eval["similarity"]["best_lag_ms"]
            ),
            "hybrid_tv_peak_p10_ms": float(
                hybrid_eval["peak_summary"]["p10"]
            ),
            "hybrid_tv_peak_median_ms": float(
                hybrid_eval["peak_summary"]["median"]
            ),
            "hybrid_tv_peak_p90_ms": float(
                hybrid_eval["peak_summary"]["p90"]
            ),
            "hybrid_tv_peak_abs_p90_ms": float(
                hybrid_eval["peak_summary"]["abs_p90"]
            ),
            "hybrid_tv_peak_violation_ratio": float(
                hybrid_eval["peak_violation_ratio"]
            ),
            "hybrid_short_gap_limit_ms": float(
                hybrid["short_gap_limit_ms"]
            ),
            "hybrid_direct_reliable_ratio": float(
                np.mean(hybrid["direct_reliable_mask"])
            ),
            "hybrid_short_gap_ratio": float(
                np.mean(hybrid["short_gap_mask"])
            ),
            "hybrid_hard_prior_ratio": float(
                np.mean(hybrid["hard_prior_mask"])
            ),
            "hybrid_blend_ratio": float(
                np.mean(hybrid["blend_mask"])
            ),
            "hybrid_tv_support_ratio": float(
                np.mean(hybrid["tv_support_mask"])
            ),
            "hybrid_candidate_scope": "local_repair_before_global_acceptance",
            "residual_median_p90": residual_median_p90,
            "residual_median_med": residual_median_med,
            "tv_loss_type": str(getattr(cfg.tv_wavelet, "loss_type", "l2")),
            "tv_student_nu": float(getattr(cfg.tv_wavelet, "student_nu", 5.0)),
            "tv_robust_fallback": str(getattr(cfg.tv_wavelet, "robust_fallback", "l2")),
            "tv_irls_max_iter": int(getattr(cfg.tv_wavelet, "irls_max_iter", 15)),
            "tv_irls_tol": float(getattr(cfg.tv_wavelet, "irls_tol", 1e-4)),
            "edge_penalty_enabled": bool(
                tv.params.get("edge_penalty_enabled", False)
            ),
            "mu_edge_configured": float(
                tv.params.get("mu_edge_configured", 0.0)
            ),
            "mu_edge_effective": float(
                tv.params.get("mu_edge_effective", 0.0)
            ),
            "edge_penalty_fraction": float(
                tv.params.get("edge_fraction", 0.12)
            ),
            "edge_penalty_taper": str(
                tv.params.get("edge_taper", "cosine")
            ),
            "student_objective_includes_edge": bool(
                tv.diag.get("student_objective_includes_edge", False)
            ),
            "stationary_candidate_accepted": bool(
                after_prior.stationary_candidate_accepted
            ),
            "stationary_candidate_peak_ms": float(
                after_prior.stationary_candidate_peak_ms
            ),
            "stationary_rejection_code": str(
                after_prior.stationary_rejection_code
            ),
            "stationary_rejection_reason": str(
                after_prior.stationary_rejection_reason
            ),
            "stationary_prior_source": str(after_prior.stationary_prior_source),
            "prior_source": str(after_prior.source),
            "CC_selected_prior_after_DTW": float(after_prior.cc),
            "stationary_prior_method_configured": str(
                getattr(cfg.stationary, "prior_method", "strict_stationary")
            ),
            "constant_phase_candidate_accepted": bool(
                after_prior.constant_phase_candidate_accepted
            ),
            "constant_phase_deg": float(after_prior.constant_phase_deg),
            "constant_phase_spectral_peak_hz": float(
                after_prior.constant_phase_spectral_peak_hz
            ),
            "constant_phase_band_low_hz": float(
                after_prior.constant_phase_band_low_hz
            ),
            "constant_phase_band_high_hz": float(
                after_prior.constant_phase_band_high_hz
            ),
            "constant_phase_candidate_cc": float(
                after_prior.constant_phase_candidate_cc
            ),
            "constant_phase_ricker_cc": float(
                after_prior.constant_phase_ricker_cc
            ),
            "constant_phase_rejection_reason": str(
                after_prior.constant_phase_rejection_reason
            ),
            "CC_stationary_raw": float(CC_stationary_raw),
            "CC_stationary_raw_env": float(stationary_candidate_env_cc_raw),
            "CC_stationary_raw_scope": (
                "accepted_stationary_candidate"
                if after_prior.stationary_candidate_accepted
                else "unlocked_candidate_before_ricker_fallback"
            ),
            "CC_prior_after_DTW_scope": "selected_prior_after_DTW",
            # 以下 candidate_* 字段是 TV 直接反演候选的显式别名；保留旧字段兼容历史脚本。
            "candidate_valid_ratio": float(tv.valid_ratio),
            "candidate_best_lag_ms": float(tv.best_lag_ms),
            "candidate_peak_metric_p10": float(tv.peak_metric_p10),
            "candidate_peak_metric_med": float(tv.peak_metric_med),
            "candidate_peak_metric_p90": float(tv.peak_metric_p90),
            "candidate_peak_abs_p90": float(tv.peak_abs_p90),
            "local_fallback_enabled": bool(local_fallback_enabled),
            "local_fallback_ratio": float(local_fallback_ratio),
            "local_fallback_hard_ratio": float(local_fallback_hard_ratio),
            "local_fallback_effective_ratio": float(local_fallback_effective_ratio),
            "fallback_by_shape_ratio": float(
                np.mean(
                    local_fallback_mask
                    & candidate_shape_fallback_request_mask
                )
            ),
            "candidate_shape_fallback_request_ratio": float(
                np.mean(candidate_shape_fallback_request_mask)
            ),
            "candidate_shape_fallback_raw_ratio": float(
                np.mean(candidate_shape_fallback_raw_mask)
            ),
            "candidate_reliable_ratio": float(
                np.mean(candidate_reliable_mask)
            ),
            "candidate_reliable_scope": (
                "direct_inversion_and_peak_energy_shape_qc"
                if bool(local_fallback_cfg["use_shape_qc"])
                else "direct_inversion_and_peak_energy_qc; shape_diagnostic_only"
            ),
            "final_reliable_ratio": float(np.mean(final_reliable_mask)),
            "final_reliable_scope": (
                "shape_peak_energy_only; does_not_mean_direct_inversion"
            ),
            "side_lobe_p90": (
                float(np.nanpercentile(side_lobe_ratio, 90))
                if np.any(np.isfinite(side_lobe_ratio))
                else np.nan
            ),
            "edge_energy_p90": (
                float(np.nanpercentile(edge_energy_ratio, 90))
                if np.any(np.isfinite(edge_energy_ratio))
                else np.nan
            ),
            "candidate_side_lobe_p50": float(
                np.nanpercentile(candidate_side_lobe_ratio, 50)
            ),
            "candidate_side_lobe_p90": float(
                np.nanpercentile(candidate_side_lobe_ratio, 90)
            ),
            "candidate_side_lobe_max": float(
                np.nanmax(candidate_side_lobe_ratio)
            ),
            "candidate_edge_energy_p50": float(
                np.nanpercentile(candidate_edge_energy_ratio, 50)
            ),
            "candidate_edge_energy_p90": float(
                np.nanpercentile(candidate_edge_energy_ratio, 90)
            ),
            "candidate_edge_energy_max": float(
                np.nanmax(candidate_edge_energy_ratio)
            ),
            "final_side_lobe_p50": float(
                np.nanpercentile(final_side_lobe_ratio, 50)
            ),
            "final_side_lobe_p90": float(
                np.nanpercentile(final_side_lobe_ratio, 90)
            ),
            "final_side_lobe_max": float(np.nanmax(final_side_lobe_ratio)),
            "final_edge_energy_p50": float(
                np.nanpercentile(final_edge_energy_ratio, 50)
            ),
            "final_edge_energy_p90": float(
                np.nanpercentile(final_edge_energy_ratio, 90)
            ),
            "final_edge_energy_max": float(np.nanmax(final_edge_energy_ratio)),
            "candidate_boundary_peak_fraction": float(
                candidate_boundary_peak_fraction
            ),
            "final_boundary_peak_fraction": float(final_boundary_peak_fraction),
            "boundary_peak_fraction_parameter": float(
                boundary_peak_fraction_parameter
            ),
            "candidate_peak_violation_ratio": float(
                tv.final_peak_violation_ratio
            ),
            "candidate_center_step_ms": float(candidate_center_step_ms),
            "longest_candidate_invalid_gap_ms": float(
                longest_candidate_invalid_gap_ms
            ),
            "longest_candidate_unreliable_gap_ms": float(
                longest_candidate_unreliable_gap_ms
            ),
            "candidate_temporal_difference_energy": float(
                candidate_temporal_difference_energy
            ),
            "final_temporal_difference_energy": float(
                final_temporal_difference_energy
            ),
            "irls_iterations_median": float(irls_iterations_median),
            "irls_iterations_max": int(irls_iterations_max),
            "candidate_origin_scope": "pre_gaussian_fill_lineage",
            "candidate_origin_codes": {
                "direct_inverted_center": 0,
                "linear_interpolated": 1,
                "endpoint_extrapolated": 2,
                "prior_fill_when_no_valid_center": 3,
                "unavailable": 5,
            },
            "final_wavelet_source_scope": "final_model_and_local_fallback_mix",
            "final_wavelet_source_codes": {
                "tv_candidate": FINAL_SOURCE_TV_CANDIDATE,
                "q_constrained_candidate": FINAL_SOURCE_Q_CONSTRAINED,
                "stationary_prior": FINAL_SOURCE_STATIONARY_PRIOR,
                "local_fallback_prior": FINAL_SOURCE_LOCAL_FALLBACK_PRIOR,
                "local_fallback_blend": FINAL_SOURCE_LOCAL_BLEND,
            },
            "local_fallback_peak_limit_ms": float(local_peak_limit_ms),
            "local_fallback_energy_min": float(local_energy_min),
            "local_fallback_pad_ms": float(local_pad_ms),
            "local_fallback_alpha_smooth_samples": float(local_alpha_smooth_samples),
            "local_fallback_use_shape_qc": bool(local_fallback_cfg["use_shape_qc"]),
            "local_fallback_side_lobe_limit_reliable": float(
                local_fallback_cfg["side_lobe_limit_reliable"]
            ),
            "local_fallback_side_lobe_limit_fallback": float(
                local_fallback_cfg["side_lobe_limit_fallback"]
            ),
            "local_fallback_side_lobe_guard_ms": float(
                local_fallback_cfg["side_lobe_guard_ms"]
            ),
            "local_fallback_edge_energy_limit_reliable": float(
                local_fallback_cfg["edge_energy_limit_reliable"]
            ),
            "local_fallback_edge_energy_limit_fallback": float(
                local_fallback_cfg["edge_energy_limit_fallback"]
            ),
            "local_fallback_edge_fraction": float(local_fallback_cfg["edge_fraction"]),
            "CC_final_direct": float(CC_final_direct),
            "CC_final_similarity": float(final_similarity_details["score"]),
            "CC_final_maxlag": float(final_similarity_details["cc_maxlag"]),
            "CC_final_env": float(final_similarity_details["env_cc"]),
            "CC_final_best_lag_ms": float(final_similarity_details["best_lag_ms"]),
            "CC_final_lag_penalty": float(final_similarity_details["lag_penalty"]),
            "CC_final_before_local_fallback_direct": float(
                final_similarity_before_local_fallback["cc_direct"]
            ),
            "CC_final_before_local_fallback_similarity": float(
                final_similarity_before_local_fallback["score"]
            ),
            "CC_final_before_local_fallback_maxlag": float(
                final_similarity_before_local_fallback["cc_maxlag"]
            ),
            "CC_final_before_local_fallback_env": float(
                final_similarity_before_local_fallback["env_cc"]
            ),
            "CC_final_before_local_fallback_best_lag_ms": float(
                final_similarity_before_local_fallback["best_lag_ms"]
            ),
            "CC_final_similarity_delta_local_fallback": float(
                final_similarity_details["score"]
                - final_similarity_before_local_fallback["score"]
            ),
            "CC_final_direct_delta_local_fallback": float(
                final_similarity_details["cc_direct"]
                - final_similarity_before_local_fallback["cc_direct"]
            ),
            "final_peak_metric_p10_before_local_fallback": (
                peak_summary_before_local_fallback["p10"]
            ),
            "final_peak_metric_med_before_local_fallback": (
                peak_summary_before_local_fallback["median"]
            ),
            "final_peak_metric_p90_before_local_fallback": (
                peak_summary_before_local_fallback["p90"]
            ),
            "final_peak_abs_p90_before_local_fallback": (
                peak_summary_before_local_fallback["abs_p90"]
            ),
            "final_peak_metric_p10_after_local_fallback": final_peak_summary["p10"],
            "final_peak_metric_med_after_local_fallback": final_peak_summary["median"],
            "final_peak_metric_p90_after_local_fallback": final_peak_summary["p90"],
            "final_peak_abs_p90_after_local_fallback": final_peak_summary["abs_p90"],
            "final_wavelet_energy_l2_median_before_local_fallback": float(
                np.median(final_qc_before_local_fallback["energy_l2"])
            ),
            "final_wavelet_energy_l2_median_after_local_fallback": float(
                np.median(final_qc["energy_l2"])
            ),
            "final_centroid_frequency_hz_median_before_local_fallback": float(
                np.median(final_qc_before_local_fallback["centroid_frequency_hz"])
            ),
            "final_centroid_frequency_hz_median_after_local_fallback": float(
                np.median(final_qc["centroid_frequency_hz"])
            ),
            "W_pass_scope": "tv_candidate_before_local_fallback",
            "post_local_fallback_acceptance_evaluated": bool(
                post_local_fallback_acceptance is not None
            ),
            "post_local_fallback_acceptance_passed": (
                bool(post_local_fallback_acceptance.passed)
                if post_local_fallback_acceptance is not None
                else None
            ),
            "post_local_fallback_acceptance_reasons": (
                list(post_local_fallback_acceptance.reasons)
                if post_local_fallback_acceptance is not None
                else []
            ),
            "post_local_fallback_acceptance_scope": (
                "diagnostic_only; keeps legacy composite-score gate semantics"
            ),
            # 旧 Ricker 键继续对应初始反射系数，避免改变历史字段含义。
            "CC_final_ricker_similarity": float(CC_ricker_qc_initial),
            "CC_final_ricker_maxlag": float(ricker_qc_initial_details["cc_maxlag"]),
            "CC_final_ricker_env": float(ricker_qc_initial_details["env_cc"]),
            "CC_final_ricker_scope": "35hz_initial_reflectivity",
            "ricker_qc_frequency_hz": float(ricker_qc_frequency_hz),
            "ricker_qc_wavelet_length_pts": int(len(w_ricker_qc)),
            "CC_ricker_35hz_initial_reflectivity_direct": float(
                ricker_qc_initial_details["cc_direct"]
            ),
            "CC_ricker_35hz_initial_reflectivity_similarity": float(
                CC_ricker_qc_initial
            ),
            "CC_ricker_35hz_initial_reflectivity_maxlag": float(
                ricker_qc_initial_details["cc_maxlag"]
            ),
            "CC_ricker_35hz_initial_reflectivity_env": float(
                ricker_qc_initial_details["env_cc"]
            ),
            "CC_ricker_35hz_initial_reflectivity_best_lag_ms": float(
                ricker_qc_initial_details["best_lag_ms"]
            ),
            "CC_ricker_35hz_after_dtw_reflectivity_direct": float(
                ricker_qc_after_dtw_details["cc_direct"]
            ),
            "CC_ricker_35hz_after_dtw_reflectivity_similarity": float(
                CC_ricker_qc_after_dtw
            ),
            "CC_ricker_35hz_after_dtw_reflectivity_maxlag": float(
                ricker_qc_after_dtw_details["cc_maxlag"]
            ),
            "CC_ricker_35hz_after_dtw_reflectivity_env": float(
                ricker_qc_after_dtw_details["env_cc"]
            ),
            "CC_ricker_35hz_after_dtw_reflectivity_best_lag_ms": float(
                ricker_qc_after_dtw_details["best_lag_ms"]
            ),
        }
    )

    extended_wavelet_diagnostics = {
        "candidate_inversion_valid_mask": np.asarray(tv.valid_mask, dtype=bool),
        "candidate_shape_ok_mask": candidate_shape_ok_mask,
        "candidate_reliable_mask": candidate_reliable_mask,
        "candidate_fallback_request_mask": candidate_fallback_request_mask,
        "candidate_shape_fallback_request_mask": (
            candidate_shape_fallback_request_mask
        ),
        "candidate_shape_fallback_raw_mask": candidate_shape_fallback_raw_mask,
        "candidate_side_lobe_ratio": candidate_side_lobe_ratio,
        "candidate_edge_energy_ratio": candidate_edge_energy_ratio,
        "candidate_boundary_peak_mask": candidate_boundary_peak_mask,
        "candidate_peak_metric_ms": candidate_qc["peak_metric_ms"],
        "candidate_wavelet_energy_l2": candidate_qc["energy_l2"],
        "candidate_centroid_frequency_hz": candidate_qc["centroid_frequency_hz"],
        "candidate_wavelet_energy_norm": candidate_wavelet_energy_norm,
        "candidate_wavelet_origin_code": tv.diag.get(
            "candidate_wavelet_origin_code",
            np.array([], dtype=np.int8),
        ),
        "candidate_inverted_mask": tv.diag.get(
            "candidate_inverted_mask",
            np.array([], dtype=bool),
        ),
        "candidate_interpolated_mask": tv.diag.get(
            "candidate_interpolated_mask",
            np.array([], dtype=bool),
        ),
        "candidate_extrapolated_mask": tv.diag.get(
            "candidate_extrapolated_mask",
            np.array([], dtype=bool),
        ),
        "candidate_prior_fill_mask": tv.diag.get(
            "candidate_prior_fill_mask",
            np.array([], dtype=bool),
        ),
        "candidate_unavailable_mask": tv.diag.get(
            "candidate_unavailable_mask",
            np.array([], dtype=bool),
        ),
        "tv_hybrid_W": hybrid["W_hybrid"],
        "tv_hybrid_alpha": hybrid["alpha"],
        "tv_hybrid_support_mask": hybrid["tv_support_mask"],
        "tv_hybrid_short_gap_mask": hybrid["short_gap_mask"],
        "tv_hybrid_hard_prior_mask": hybrid["hard_prior_mask"],
        "tv_hybrid_blend_mask": hybrid["blend_mask"],
        "tv_hybrid_extrapolated_mask": hybrid["extrapolated_mask"],
        "tv_hybrid_provenance_forced_prior_mask": hybrid[
            "provenance_forced_prior_mask"
        ],
        "tv_hybrid_peak_metric_ms": hybrid_eval[
            "qc"
        ]["peak_metric_ms"],
        "attempted_center_indices": attempted_center_indices,
        "candidate_invalid_at_centers": candidate_invalid_at_centers,
        "candidate_unreliable_at_centers": candidate_unreliable_at_centers,
        "final_shape_ok_mask": final_shape_ok_mask,
        "final_reliable_mask": final_reliable_mask,
        "final_shape_fallback_request_mask": final_shape_fallback_request_mask,
        "final_side_lobe_ratio": final_side_lobe_ratio,
        "final_edge_energy_ratio": final_edge_energy_ratio,
        "final_boundary_peak_mask": final_boundary_peak_mask,
        "final_wavelet_source_code": final_wavelet_source_code,
        "final_source_tv_candidate_mask": final_wavelet_source_masks[
            "tv_candidate_mask"
        ],
        "final_source_q_constrained_mask": final_wavelet_source_masks[
            "q_constrained_mask"
        ],
        "final_source_stationary_prior_mask": final_wavelet_source_masks[
            "stationary_prior_mask"
        ],
        "final_source_local_fallback_prior_mask": final_wavelet_source_masks[
            "local_fallback_prior_mask"
        ],
        "final_source_local_blend_mask": final_wavelet_source_masks[
            "local_blend_mask"
        ],
    }

    _save_and_plot_results(
        config_path=config_path,
        cfg=cfg,
        data=data,
        initial_prior=initial_prior,
        dtw=dtw,
        after_prior=after_prior,
        tv=tv,
        q_result=q_result,
        metrics=metrics,
        W_final=W_final,
        s_syn_final=s_syn_final,
        final_qc_before_local_fallback=final_qc_before_local_fallback,
        final_qc=final_qc,
        w_ricker_qc=w_ricker_qc,
        s_syn_ricker_qc_initial=s_syn_ricker_qc_initial,
        s_syn_ricker_qc_after_dtw=s_syn_ricker_qc_after_dtw,
        reliable_mask=reliable_mask,
        local_fallback_mask=local_fallback_mask,
        local_fallback_alpha=local_fallback_alpha,
        wavelet_energy_norm=wavelet_energy_norm,
        side_lobe_ratio=side_lobe_ratio,
        edge_energy_ratio=edge_energy_ratio,
        W_final_before_local_fallback=W_final_before_local_fallback,
        s_syn_final_before_local_fallback=s_syn_final_before_local_fallback,
        extended_wavelet_diagnostics=extended_wavelet_diagnostics,
    )

    print("\n[Final]")
    # 对metrics的每个键值对便利，返回键和值，打印出来
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/cb323_center_student_t_nu10_edge_mu1p00_v05.yaml",
        help="Path to experiment config yaml.",
    )
    args = parser.parse_args()
    main(args.config)
