# -*- coding: utf-8 -*-
"""
wavelet_inversion_robust.py

Robust time-varying wavelet inversion for nonstationary seismic-well tying.

核心改动：
1. 不再默认每 1 个采样点反演一个子波，而是按 estimate_step_samples / estimate_step_ms 稀疏估计，
   再沿时间插值和平滑，降低自由度。
2. 增加零直流约束，防止子波吸收低频漂移。
3. 支持平稳子波或外部时变子波作为先验，抑制局部子波乱跑。
4. 增加相邻子波连续约束，降低时间方向跳变。
5. 增加峰值位置限制，防止子波把井震时差吸收为子波时延。
6. 增加病态矩阵拒绝机制，反射能量足够但矩阵约束不足时不反演。
7. 增加振幅突变拒绝机制，防止局部异常子波污染 W 矩阵。
8. 增加可选诊断输出，用于 QC：峰值漂移、DC 比例、子波能量、有效秩、跳过原因等。

兼容性：
- time_varying_wavelet_inversion(..., return_diagnostics=False) 默认仍然只返回 W。
- 如果需要诊断信息，设置 return_diagnostics=True，返回 (W, diagnostics)。
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from core.forward_operator import (
    build_local_convolution_matrix,
    nonstationary_convolution,
)


# 子波来源编码。候选 W 在时间插值前按下列互斥类别记录来源；
# 最终 W 若经过局部平稳先验的平滑混合，则由主流程使用 ORIGIN_LOCAL_BLEND。
ORIGIN_INVERTED = 0
ORIGIN_INTERPOLATED = 1
ORIGIN_EXTRAPOLATED = 2
ORIGIN_FALLBACK_PRIOR = 3
ORIGIN_LOCAL_BLEND = 4
ORIGIN_UNAVAILABLE = 5


# =========================================================
# 基础工具函数
# =========================================================
def _make_odd(n: int) -> int:
    """确保窗口长度为奇数。"""
    n = int(n)
    return n if n % 2 == 1 else n + 1

def _validate_wavelet_alignment(wavelet_alignment: str) -> str:
    """检查子波时间轴定义。"""
    wavelet_alignment = str(wavelet_alignment).lower()
    if wavelet_alignment not in ("center", "causal"):
        raise ValueError("wavelet_alignment 必须是 'center' 或 'causal'。")
    return wavelet_alignment


def _validate_nonnegative_scalar(
    value: float,
    name: str,
) -> float:
    """Validate a finite non-negative scalar parameter."""
    array = np.asarray(value, dtype=float)
    if (
        array.ndim != 0
        or not np.isfinite(array)
        or float(array) < 0.0
    ):
        raise ValueError(f"{name} 必须是有限非负数。")
    return float(array)

def _second_derivative_matrix(n: int) -> np.ndarray:
    """
    构造二阶差分矩阵 D。

    D @ w 约等于子波二阶导数，用于抑制不必要的高频振荡。
    """
    if n < 3:
        raise ValueError("wavelet_length 必须 >= 3。")

    D = np.zeros((n - 2, n), dtype=float)
    for i in range(n - 2):
        D[i, i] = 1.0
        D[i, i + 1] = -2.0
        D[i, i + 2] = 1.0
    return D


def build_edge_penalty_weights(
    wavelet_length: int,
    edge_fraction: float = 0.12,
    taper: str = "cosine",
) -> np.ndarray:
    """构造对子波首尾能量进行惩罚的一维权重。

    输入参数：
        wavelet_length: 子波采样点数，无单位，必须 >= 3。
        edge_fraction: 左、右各自参与惩罚的长度比例，必须位于 (0, 0.5)。
        taper: 当前仅支持 ``cosine``。

    输出结果：
        edge_weights: shape=(wavelet_length,)，无单位。中心区域严格为 0，
        最外端为 1，且从中心侧向左右边界单调增加。

    数学作用与物理假设：
        边缘正则写为 ``||E @ w||_2^2``，其中
        ``E = diag(edge_weights)``。这里先构造期望的余弦惩罚曲线 p，
        再返回 ``sqrt(p)``，避免矩阵平方后实际惩罚误变为 p^2。
        该项只抑制子波边界异常能量，不改变子波的时间轴、相位参考或卷积定义。
    """
    if (
        isinstance(wavelet_length, (bool, np.bool_))
        or not isinstance(wavelet_length, (int, np.integer))
        or int(wavelet_length) < 3
    ):
        raise ValueError("wavelet_length 必须是 >= 3 的整数。")

    edge_fraction_value = np.asarray(edge_fraction, dtype=float)
    if (
        edge_fraction_value.ndim != 0
        or not np.isfinite(edge_fraction_value)
        or not 0.0 < float(edge_fraction_value) < 0.5
    ):
        raise ValueError("edge_fraction 必须位于 (0, 0.5)。")

    taper = str(taper).strip().lower()
    if taper != "cosine":
        raise ValueError("edge_taper 当前仅支持 'cosine'。")

    wavelet_length = int(wavelet_length)
    edge_fraction = float(edge_fraction_value)
    max_edge_samples = max(1, (wavelet_length - 1) // 2)
    edge_samples = min(
        max_edge_samples,
        max(1, int(np.ceil(edge_fraction * wavelet_length))),
    )

    # penalty_ramp 是真正希望施加到 w_j^2 上的惩罚强度 p_j。
    # shape=(edge_samples,)，从中心侧的弱惩罚单调增加到边界处的 1。
    normalized_distance = (
        np.arange(1, edge_samples + 1, dtype=float) / float(edge_samples)
    )
    penalty_ramp = 0.5 * (
        1.0 - np.cos(np.pi * normalized_distance)
    )
    edge_weight_ramp = np.sqrt(penalty_ramp)

    edge_weights = np.zeros(wavelet_length, dtype=float)
    edge_weights[:edge_samples] = edge_weight_ramp[::-1]
    edge_weights[-edge_samples:] = edge_weight_ramp
    return edge_weights


def _build_candidate_wavelet_origin_masks(
    valid_mask: np.ndarray,
    prior_available: bool,
) -> dict[str, np.ndarray]:
    """根据直接反演中心构造互斥的候选子波来源掩码。

    输入：
        valid_mask: shape=(N_time,)，True 表示该时间中心直接反演成功。
        prior_available: 无有效中心时是否存在可用于全道填充的先验子波。

    输出：
        字典中的每个布尔掩码和 ``origin_code`` 均为 shape=(N_time,)。
        插值表示位于首末有效中心之间但未直接反演；外推表示位于首末
        有效中心之外。所有来源互斥且每个时间样点恰好属于一种来源。

    数学作用与物理假设：
        该函数只记录 ``_fill_unestimated_wavelets`` 的来源谱系，不改变 W。
        后续二维平滑会传播邻近信息，但不会把来源标签改写成“直接反演”。
    """
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if valid_mask.ndim != 1 or valid_mask.size == 0:
        raise ValueError("valid_mask 必须是一维非空布尔数组。")

    n_time = valid_mask.size
    inverted_mask = valid_mask.copy()
    interpolated_mask = np.zeros(n_time, dtype=bool)
    extrapolated_mask = np.zeros(n_time, dtype=bool)
    fallback_prior_mask = np.zeros(n_time, dtype=bool)
    unavailable_mask = np.zeros(n_time, dtype=bool)

    valid_indices = np.where(valid_mask)[0]
    if valid_indices.size > 0:
        first_valid = int(valid_indices[0])
        last_valid = int(valid_indices[-1])
        between_valid_bounds = np.zeros(n_time, dtype=bool)
        between_valid_bounds[first_valid:last_valid + 1] = True
        interpolated_mask = between_valid_bounds & (~valid_mask)
        extrapolated_mask[:first_valid] = True
        extrapolated_mask[last_valid + 1:] = True
    elif prior_available:
        fallback_prior_mask[:] = True
    else:
        unavailable_mask[:] = True

    origin_count = (
        inverted_mask.astype(int)
        + interpolated_mask.astype(int)
        + extrapolated_mask.astype(int)
        + fallback_prior_mask.astype(int)
        + unavailable_mask.astype(int)
    )
    if not np.all(origin_count == 1):
        raise AssertionError("候选子波来源掩码必须互斥且完整。")

    origin_code = np.full(n_time, ORIGIN_UNAVAILABLE, dtype=np.int8)
    origin_code[inverted_mask] = ORIGIN_INVERTED
    origin_code[interpolated_mask] = ORIGIN_INTERPOLATED
    origin_code[extrapolated_mask] = ORIGIN_EXTRAPOLATED
    origin_code[fallback_prior_mask] = ORIGIN_FALLBACK_PRIOR

    return {
        "origin_code": origin_code,
        "inverted_mask": inverted_mask,
        "interpolated_mask": interpolated_mask,
        "extrapolated_mask": extrapolated_mask,
        "fallback_prior_mask": fallback_prior_mask,
        "unavailable_mask": unavailable_mask,
    }



def _extract_window(
    x: np.ndarray,
    center_index: int,
    window_length: int,
    fill_value: float = 0.0,
) -> np.ndarray:
    """从一维数组中提取以 center_index 为中心的窗口；越界用 fill_value 填充。"""
    N = len(x)
    half = window_length // 2

    out = np.full(window_length, fill_value, dtype=float)

    start_global = center_index - half
    end_global = center_index + half + 1

    valid_start = max(start_global, 0)
    valid_end = min(end_global, N)

    out_start = valid_start - start_global
    out_end = out_start + (valid_end - valid_start)

    if valid_end > valid_start:
        out[out_start:out_end] = x[valid_start:valid_end]

    return out

def _safe_svd_solve(
    A: np.ndarray,
    b: np.ndarray,
    damping_ratio: float = 1e-3,
    svd_cutoff_ratio: float = 1e-4,
) -> np.ndarray:
    """
    使用阻尼 SVD 求解最小二乘问题 min ||A x - b||_2。

    阻尼形式：
        x = V diag(s / (s^2 + eps^2)) U.T b
    """
    U, S, Vt = np.linalg.svd(A, full_matrices=False)

    if len(S) == 0:
        return np.zeros(A.shape[1], dtype=float)

    max_s = float(np.max(S))
    if max_s <= 0:
        return np.zeros(A.shape[1], dtype=float)

    damping = damping_ratio * max_s
    cutoff = svd_cutoff_ratio * max_s

    S_inv = np.zeros_like(S)
    valid = S > cutoff
    # S_inv[valid] = S[valid] / (S[valid] ** 2 + damping ** 2)
    S_inv[valid] = 1.0 / S[valid]

    return Vt.T @ (S_inv * (U.T @ b))


def _solve_regularized_window(
    *,
    R: np.ndarray,
    s_win: np.ndarray,
    data_scale: float,
    I: np.ndarray,
    D: np.ndarray,
    C_dc: np.ndarray,
    mu1: float,
    mu2: float,
    mu_dc: float,
    prior_i: np.ndarray | None,
    mu_prior: float,
    previous_wavelet: np.ndarray | None,
    mu_time: float,
    damping_ratio: float,
    svd_cutoff_ratio: float,
    sample_weights: np.ndarray | None = None,
    edge_weights: np.ndarray | None = None,
    mu_edge: float = 0.0,
) -> np.ndarray:
    """求解一个局部窗口的增广正则化最小二乘问题。

    第一阶段仅使用 ``sample_weights=None``，数值结构与原有 L2 代码保持一致。
    ``sample_weights`` 是为后续 Student-t IRLS 预留的接口；局部残差尺度 sigma
    不在本函数中除到 R 或 s_win 上。
    """
    R = np.asarray(R, dtype=float)
    s_win = np.asarray(s_win, dtype=float).ravel()

    if R.ndim != 2:
        raise ValueError("R 必须是二维矩阵。")
    if R.shape[0] != s_win.size:
        raise ValueError("R 的行数必须等于 s_win 的长度。")

    if sample_weights is None:
        # L2 路径直接复用原矩阵，避免不必要的乘法，保证重构前后数值一致。
        R_data = R
        s_data = s_win
    else:
        weights = np.asarray(sample_weights, dtype=float).ravel()
        if weights.size != s_win.size:
            raise ValueError("sample_weights 的长度必须等于 s_win 的长度。")
        if not np.all(np.isfinite(weights)):
            raise ValueError("sample_weights 包含 NaN 或 Inf。")
        if np.any(weights < 0):
            raise ValueError("sample_weights 不能为负数。")

        sqrt_weight = np.sqrt(weights)
        R_data = sqrt_weight[:, None] * R
        s_data = sqrt_weight * s_win

    A_blocks = [R_data]
    b_blocks = [s_data]

    # L2 / ridge wavelet-energy regularization.
    # mu1 == 0 时严格不构造、不追加零块。
    if mu1 > 0.0:
        A_blocks.append(
            np.sqrt(mu1 * data_scale) * I
        )
        b_blocks.append(
            np.zeros(I.shape[0], dtype=float)
        )

    # Lag-domain second-derivative regularization.
    # Round 7 暂时保持原有路径；当前生产参数 mu2 = 3.0。
    A_blocks.append(
        np.sqrt(mu2 * data_scale) * D
    )
    b_blocks.append(
        np.zeros(D.shape[0], dtype=float)
    )

    # DC regularization.
    # mu_dc == 0 时严格不构造、不追加零块。
    if mu_dc > 0.0:
        A_blocks.append(
            np.sqrt(mu_dc * data_scale) * C_dc
        )
        b_blocks.append(
            np.zeros(C_dc.shape[0], dtype=float)
        )

    if prior_i is not None and mu_prior > 0:
        A_blocks.append(np.sqrt(mu_prior * data_scale) * I)
        b_blocks.append(np.sqrt(mu_prior * data_scale) * prior_i)

    if previous_wavelet is not None and mu_time > 0:
        A_blocks.append(np.sqrt(mu_time * data_scale) * I)
        b_blocks.append(np.sqrt(mu_time * data_scale) * previous_wavelet)

    # mu_edge=0 时严格跳过：不构造矩阵、不追加零块，保持旧 SVD 输入不变。
    if mu_edge > 0:
        if edge_weights is None:
            raise ValueError("mu_edge > 0 时必须提供 edge_weights。")
        edge_weights = np.asarray(edge_weights, dtype=float).ravel()
        if edge_weights.size != I.shape[0]:
            raise ValueError("edge_weights 长度必须等于 wavelet_length。")
        if not np.all(np.isfinite(edge_weights)):
            raise ValueError("edge_weights 包含 NaN 或 Inf。")

        # E=diag(edge_weights)，shape=(wavelet_length, wavelet_length)。
        # 与 mu1/mu2/mu_dc 使用同一 data_scale 层级，不重复缩放。
        E_edge = np.diag(edge_weights)
        A_blocks.append(np.sqrt(mu_edge * data_scale) * E_edge)
        b_blocks.append(np.zeros(I.shape[0]))

    A_aug = np.vstack(A_blocks)
    b_aug = np.concatenate(b_blocks)

    return _safe_svd_solve(
        A=A_aug,
        b=b_aug,
        damping_ratio=damping_ratio,
        svd_cutoff_ratio=svd_cutoff_ratio,
    )


# =========================================================
# Student-t 稳健数据项工具
# =========================================================
def _mad_scale(
    residual: np.ndarray,
    scale_floor: float,
) -> float:
    """用 MAD 估计残差标准差，并施加全局尺度下限。"""
    residual = np.asarray(residual, dtype=float).ravel()
    finite = residual[np.isfinite(residual)]

    floor = max(float(scale_floor), 1e-12)
    if finite.size == 0:
        return floor

    center = float(np.median(finite))
    mad = float(np.median(np.abs(finite - center)))
    sigma = 1.4826 * mad
    return float(max(sigma, floor))


def _student_t_raw_weights(
    residual: np.ndarray,
    sigma: float,
    nu: float,
) -> np.ndarray:
    """返回未截断的 Student-t IRLS 权重。"""
    if nu <= 0:
        raise ValueError("student_nu 必须大于 0。")

    sigma = max(float(sigma), 1e-12)
    z = np.asarray(residual, dtype=float) / sigma
    return float(nu) / (float(nu) + z ** 2)


def _student_t_weights(
    residual: np.ndarray,
    sigma: float,
    nu: float,
    weight_floor: float = 1e-3,
) -> np.ndarray:
    """返回实际用于加权最小二乘的截断权重。"""
    if not 0.0 <= weight_floor <= 1.0:
        raise ValueError("robust_weight_floor 必须位于 [0, 1]。")

    raw = _student_t_raw_weights(
        residual=residual,
        sigma=sigma,
        nu=nu,
    )
    return np.clip(raw, float(weight_floor), 1.0)


def _effective_sample_size(weights: np.ndarray) -> float:
    """计算 Kish 有效样本数。"""
    weights = np.asarray(weights, dtype=float).ravel()
    numerator = float(np.sum(weights)) ** 2
    denominator = float(np.sum(weights ** 2)) + 1e-12
    return numerator / denominator


def _student_t_data_loss(
    residual: np.ndarray,
    sigma: float,
    nu: float,
) -> float:
    """计算与 omega=nu/(nu+z^2) 一致的缩放 Student-t 数据损失。"""
    if nu <= 0:
        raise ValueError("student_nu 必须大于 0。")

    sigma = max(float(sigma), 1e-12)
    residual = np.asarray(residual, dtype=float)
    value = 0.5 * float(nu) * sigma ** 2 * np.log1p(
        residual ** 2 / (float(nu) * sigma ** 2)
    )
    return float(np.sum(value))


def _regularization_value(
    *,
    w: np.ndarray,
    data_scale: float,
    D: np.ndarray,
    C_dc: np.ndarray,
    mu1: float,
    mu2: float,
    mu_dc: float,
    prior_i: np.ndarray | None,
    mu_prior: float,
    previous_wavelet: np.ndarray | None,
    mu_time: float,
    edge_weights: np.ndarray | None = None,
    mu_edge: float = 0.0,
) -> float:
    """计算与增广最小二乘矩阵一致的二次正则目标值。"""
    w = np.asarray(w, dtype=float)

    value = 0.0
    if mu1 > 0.0:
        value += float(mu1) * float(np.sum(w ** 2))
    value += float(mu2) * float(np.sum((D @ w) ** 2))
    if mu_dc > 0.0:
        value += float(mu_dc) * float(np.sum((C_dc @ w) ** 2))

    if prior_i is not None and mu_prior > 0:
        value += float(mu_prior) * float(np.sum((w - prior_i) ** 2))

    if previous_wavelet is not None and mu_time > 0:
        value += float(mu_time) * float(
            np.sum((w - previous_wavelet) ** 2)
        )

    # 与增广矩阵 sqrt(mu_edge * data_scale) * E 完全对应。
    # 零值时不执行任何新增浮点运算，保持旧 Student-t 目标路径不变。
    if mu_edge > 0:
        if edge_weights is None:
            raise ValueError("mu_edge > 0 时必须提供 edge_weights。")
        edge_weights = np.asarray(edge_weights, dtype=float).ravel()
        if edge_weights.size != w.size:
            raise ValueError("edge_weights 长度必须等于 wavelet_length。")
        if not np.all(np.isfinite(edge_weights)):
            raise ValueError("edge_weights 包含 NaN 或 Inf。")
        edge_weighted_wavelet = edge_weights * w
        value += float(mu_edge) * float(
            np.sum(edge_weighted_wavelet ** 2)
        )

    # 数据损失采用 1/2 约定，因此完整目标中的正则项也采用 1/2。
    return 0.5 * float(data_scale) * value


def _full_student_t_objective(
    *,
    w: np.ndarray,
    R: np.ndarray,
    s_win: np.ndarray,
    sigma: float,
    nu: float,
    data_scale: float,
    D: np.ndarray,
    C_dc: np.ndarray,
    mu1: float,
    mu2: float,
    mu_dc: float,
    prior_i: np.ndarray | None,
    mu_prior: float,
    previous_wavelet: np.ndarray | None,
    mu_time: float,
    edge_weights: np.ndarray | None = None,
    mu_edge: float = 0.0,
) -> float:
    """计算单个窗口的完整 Student-t 目标函数。"""
    residual = np.asarray(s_win, dtype=float) - np.asarray(R, dtype=float) @ w
    return _student_t_data_loss(residual, sigma, nu) + _regularization_value(
        w=w,
        data_scale=data_scale,
        D=D,
        C_dc=C_dc,
        mu1=mu1,
        mu2=mu2,
        mu_dc=mu_dc,
        prior_i=prior_i,
        mu_prior=mu_prior,
        previous_wavelet=previous_wavelet,
        mu_time=mu_time,
        edge_weights=edge_weights,
        mu_edge=mu_edge,
    )


def _estimate_reflectivity_energy_threshold(
    r_time: np.ndarray,
    data_window_length: int,
    energy_percentile: float = 35.0,
) -> float:
    """
    根据所有局部反射能量的分位数估计低能量窗口阈值。

    energy_percentile 越高，跳过窗口越多。
    真实资料建议 35~50，合成资料建议 10~25。
    """
    N = len(r_time)
    half = data_window_length // 2
    energies = []

    step = max(1, data_window_length // 10)
    for i in range(half, N - half, step):
        rw = _extract_window(r_time, i, data_window_length)
        energies.append(np.sum(rw ** 2))

    energies = np.asarray(energies, dtype=float)
    positive = energies[np.isfinite(energies) & (energies > 0)]

    if len(positive) == 0:
        return 0.0

    return float(np.percentile(positive, energy_percentile))

def _effective_rank_and_condition(
    R: np.ndarray,
    rank_cutoff_ratio: float = 1e-3,
) -> tuple[int, float, np.ndarray]:
    """
    计算局部卷积矩阵的有效秩和条件比例。

    cond_ratio = smallest_singular / largest_singular。
    越接近 0，矩阵越病态。
    """
    S = np.linalg.svd(R, compute_uv=False)
    if len(S) == 0 or S[0] <= 1e-15:
        return 0, 0.0, S # 如果满足上面的条件，返回有效秩为0，条件

    effective_rank = int(np.sum(S > rank_cutoff_ratio * S[0]))
    cond_ratio = float(S[-1] / (S[0] + 1e-15))

    return effective_rank, cond_ratio, S

def _zero_mean_wavelet(w: np.ndarray) -> np.ndarray:
    """显式去掉子波直流分量。"""
    w = np.asarray(w, dtype=float).copy()
    return w - np.mean(w)


def _dc_ratio(w: np.ndarray) -> float:
    """计算 DC 比例：|sum(w)| / sum(|w|)。越小越好。"""
    denom = np.sum(np.abs(w)) + 1e-12
    return float(np.abs(np.sum(w)) / denom)

def _peak_position_samples(w: np.ndarray) -> int:
    """返回绝对振幅主峰所在的子波采样点位置。"""
    w = np.asarray(w, dtype=float)
    return int(np.argmax(np.abs(w)))


def _peak_shift_samples(w: np.ndarray) -> int:
    """center 模式：返回主峰相对中心点的偏移采样数。"""
    w = np.asarray(w, dtype=float)
    center = len(w) // 2
    peak = _peak_position_samples(w)
    return peak - center


def _peak_metric_samples(
    w: np.ndarray,
    wavelet_alignment: str = "center",
) -> int:
    """
    返回用于 QC 的峰值指标。

    - center: 返回 peak - center
    - causal: 返回 peak_position
    """
    wavelet_alignment = _validate_wavelet_alignment(wavelet_alignment)
    # 将零点放在数组中心，返回的是**“主峰索引 - 中心点索引”**（可以是正数、零或负数）。
    if wavelet_alignment == "center":
        return _peak_shift_samples(w)
    # 返回的是主峰在子波数组中的绝对索引（从 0 开始）。
    else:
        return _peak_position_samples(w)

def _stabilize_wavelet_peak(
    w: np.ndarray,
    max_peak_shift_samples: int | None = None,
    wavelet_alignment: str = "center",
    peak_allowed_samples: tuple[int, int] | None = None,
) -> tuple[np.ndarray | None, int]:
    """
    峰值位置 QC。

    center 模式：
        要求主峰距离中心不超过 max_peak_shift_samples。

    causal 模式：
        要求主峰位置落在 peak_allowed_samples = (lo, hi) 内。
        例如 dt=0.001, peak_allowed_ms=(0, 40) 对应 (0, 40) samples。
    """
    wavelet_alignment = _validate_wavelet_alignment(wavelet_alignment)

    w = _zero_mean_wavelet(w)
    metric = _peak_metric_samples(w, wavelet_alignment)

    if wavelet_alignment == "center":
        if max_peak_shift_samples is None:
            max_peak_shift_samples = max(3, len(w) // 20)

        if abs(metric) > max_peak_shift_samples:
            return None, metric

        return w, metric

    else:
        if peak_allowed_samples is None:
            if max_peak_shift_samples is None:
                peak_allowed_samples = (0, len(w) - 1)
            else:
                peak_allowed_samples = (0, max_peak_shift_samples)

        lo, hi = peak_allowed_samples
        lo = max(0, int(lo))
        hi = min(len(w) - 1, int(hi))

        if metric < lo or metric > hi:
            return None, metric

        return w, metric

def _get_prior_wavelet(
    w_prior: np.ndarray | None,
    index: int,
    wavelet_length: int,
) -> np.ndarray | None:
    """
    获取当前时间点的先验子波。

    支持两种格式：
    - w_prior.shape == (L,)：全局平稳先验；
    - w_prior.shape == (N, L)：时变先验。
    """
    if w_prior is None:
        return None

    wp = np.asarray(w_prior, dtype=float)

    if wp.ndim == 1:
        if len(wp) != wavelet_length:
            raise ValueError("一维 w_prior 的长度必须等于 wavelet_length。")
        return _zero_mean_wavelet(wp)

    if wp.ndim == 2:
        if wp.shape[1] != wavelet_length:
            raise ValueError("二维 w_prior 的第二维必须等于 wavelet_length。")
        idx = int(np.clip(index, 0, wp.shape[0] - 1))
        return _zero_mean_wavelet(wp[idx, :])

    raise ValueError("w_prior 只能是一维或二维数组。")

def _fill_unestimated_wavelets(
    W: np.ndarray,
    valid_mask: np.ndarray,
    w_prior: np.ndarray | None = None,
) -> np.ndarray:
    """
    用时间方向线性插值填充未直接反演的位置。

    相比简单前向填充，插值更适合稀疏估计的时变子波。
    如果没有任何有效反演点，则优先用 w_prior 填充。
    """
    W = np.asarray(W, dtype=float).copy()
    N, L = W.shape
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) == 0:
        if w_prior is not None:
            wp = np.asarray(w_prior, dtype=float)
            if wp.ndim == 1:
                if len(wp) != L:
                    raise ValueError("w_prior 长度必须等于 wavelet_length。")
                return np.tile(_zero_mean_wavelet(wp), (N, 1))
            if wp.ndim == 2:
                if wp.shape != W.shape:
                    raise ValueError("二维 w_prior 的 shape 必须与 W 相同。")
                return np.apply_along_axis(_zero_mean_wavelet, 1, wp)
        return W

    x_all = np.arange(N)
    x_valid = valid_indices

    for j in range(L):
        W[:, j] = np.interp(x_all, x_valid, W[x_valid, j])

    return W

def _wavelet_qc_attributes(
    W: np.ndarray,
    dt: float | None = None,
    wavelet_alignment: str = "center",
) -> dict[str, np.ndarray]:
    """
    计算 W 矩阵的基础 QC 属性。

    返回：
    - energy: 子波能量 ||w||_2
    - peak_shift_samples: 主峰相对中心偏移
    - dc_ratio: DC 比例
    - centroid_freq: 频谱质心频率，dt 非 None 时返回 Hz，否则返回归一化频率坐标
    - bandwidth: 频谱标准差
    """
    W = np.asarray(W, dtype=float)
    N, L = W.shape
    # 计算行向量的几何长度，即行向量的二范数
    energy = np.linalg.norm(W, axis=1)
    wavelet_alignment = _validate_wavelet_alignment(wavelet_alignment)
    # 计算每个行向量的主峰位置
    peak_position = np.array(
        [_peak_position_samples(W[i]) for i in range(N)],
        dtype=float
    )
    # 计算主峰相对于中心点的偏移
    peak_metric = np.array(
        [_peak_metric_samples(W[i], wavelet_alignment) for i in range(N)],
        dtype=float
    )
    # 计算每个行向量的非零均值
    dc = np.array([_dc_ratio(W[i]) for i in range(N)], dtype=float)

    if dt is None:
        freqs = np.fft.rfftfreq(L, d=1.0)
    else:
        freqs = np.fft.rfftfreq(L, d=float(dt))

    spec = np.abs(np.fft.rfft(W, axis=1)) ** 2
    spec_sum = np.sum(spec, axis=1) + 1e-12
    centroid = np.sum(spec * freqs[None, :], axis=1) / spec_sum
    bandwidth = np.sqrt(
        np.sum(((freqs[None, :] - centroid[:, None]) ** 2) * spec, axis=1) / spec_sum
    )

    return {
    "energy": energy,  
    "peak_shift_samples": peak_metric,
    "peak_position_samples": peak_position,
    "dc_ratio": dc,
    "centroid_freq": centroid,
    "bandwidth": bandwidth,
}

# =========================================================
# 平稳子波反演
# =========================================================
def stationary_wavelet_inversion(
    r_time: np.ndarray,
    s_obs: np.ndarray,
    wavelet_length: int,
    mu1: float = 0.2, # 子波能量惩罚项
    mu2: float = 2.0,# 子波二阶导数惩罚项
    mu_dc: float = 10.0,# 非零均值惩罚，抑制低频漂移和非零均值
    mu_edge: float = 0.0,# 子波两端边界衰减惩罚项
    edge_fraction: float = 0.12,# 边界衰减时窗两端所占比例
    edge_taper: str = "cosine",# 边缘衰减窗形式
    damping_ratio: float = 3e-3,# SVD 阻尼比例，越大越稳定但越模糊
    svd_cutoff_ratio: float = 1e-3,
    peak_lock: bool = True,
    max_peak_shift_samples: int | None = None,
    wavelet_alignment: str = "center",
) -> np.ndarray:
    """
    反演一个全局平稳子波。

    主要用途：
    1. 作为时变子波反演的先验 w_prior；
    2. 作为 stationary baseline；
    3. 当局部窗口不稳定时提供合理填充值。
    peak_lock=True 时，限制子波主峰位置，防止把时差吸收为子波时延。
    """
    r_time = np.asarray(r_time, dtype=float).copy()
    s_obs = np.asarray(s_obs, dtype=float).copy()

    if r_time.ndim != 1 or s_obs.ndim != 1:
        raise ValueError("r_time 和 s_obs 必须是一维数组。")
    if len(r_time) != len(s_obs):
        raise ValueError("r_time 和 s_obs 长度必须一致。")

    N = len(r_time)
    wavelet_length = _make_odd(wavelet_length)
    wavelet_alignment = _validate_wavelet_alignment(wavelet_alignment)
    half_wavelet = wavelet_length // 2

    R = np.zeros((N, wavelet_length), dtype=float)
    for row in range(N):
        for col in range(wavelet_length):
            if wavelet_alignment == "center":
                lag = col - half_wavelet
            else:
                lag = col

            r_index = row - lag

            if 0 <= r_index < N:
                R[row, col] = r_time[r_index]

    D = _second_derivative_matrix(wavelet_length)
    I = np.eye(wavelet_length, dtype=float)
    C_dc = np.ones((1, wavelet_length), dtype=float) / np.sqrt(wavelet_length)

    s_obs = s_obs - np.mean(s_obs)
    data_scale = max(np.sum(s_obs ** 2) / N, 1e-12)

    # 注意：平稳子波反演作为先验基线（prior baseline）计算工具，保持独立的参数体系
    # （如 stationary.mu1=0.5, stationary.mu_dc=15.0）。此处无条件保留 mu1 与 mu_dc 块。
    # Round 7 的正则精简仅严格针对 TV objective（时变反演目标函数），不影响平稳先验反演。
    A_blocks = [
        R,
        np.sqrt(mu1 * data_scale) * I,
        np.sqrt(mu2 * data_scale) * D,
        np.sqrt(mu_dc * data_scale) * C_dc,
    ]

    b_blocks = [
        s_obs,
        np.zeros(wavelet_length),
        np.zeros(wavelet_length - 2),
        np.zeros(1),
    ]

    # 当启用边缘惩罚时，追加平滑衰减项 ||E @ w||_2^2
    # 抑制平稳子波两端边界系数由于自由度过多而吸收残余误差产生的虚假振荡
    if mu_edge > 0:
        edge_weights = build_edge_penalty_weights(
            wavelet_length=wavelet_length,
            edge_fraction=edge_fraction,
            taper=edge_taper,
        )
        # E_edge 为对角阵，shape=(wavelet_length, wavelet_length)
        # 与 mu1/mu2/mu_dc 保持同一 data_scale 能量缩放基准
        E_edge = np.diag(edge_weights)
        A_blocks.append(np.sqrt(mu_edge * data_scale) * E_edge)
        b_blocks.append(np.zeros(wavelet_length))

    A_aug = np.vstack(A_blocks)
    b_aug = np.concatenate(b_blocks)

    w = _safe_svd_solve(
        A=A_aug,
        b=b_aug,
        damping_ratio=damping_ratio,
        svd_cutoff_ratio=svd_cutoff_ratio,
    )

    w = _zero_mean_wavelet(w)

    # # 当前 peak_lock=True 时，并不是“强制锁主峰”；
    # # 它只是“如果主峰合格，就接受处理后的 w；如果不合格，就原样返回”。
    # if peak_lock:
    #     if max_peak_shift_samples is None:
    #         max_peak_shift_samples = max(3, wavelet_length // 10)
    #     w_checked, _ = _stabilize_wavelet_peak(w, max_peak_shift_samples)
    #     if w_checked is not None:
    #         w = w_checked

    # 如果平稳子波主峰明显偏离中心，程序会直接提醒你：
    # 当前固定 TWT 条件下反演出来的平稳子波不满足物理约束。
    if peak_lock:
        if max_peak_shift_samples is None:
            max_peak_shift_samples = max(3, wavelet_length // 20)

        w_checked, peak_metric = _stabilize_wavelet_peak(
            w,
            max_peak_shift_samples=max_peak_shift_samples,
            wavelet_alignment=wavelet_alignment,
        )

        if w_checked is None:
            raise ValueError(
                f"平稳子波主峰位置不满足约束: peak_metric={peak_metric} samples, "
                f"alignment={wavelet_alignment}. "
                "说明该平稳子波可能正在吸收残余时差，不建议作为物理先验。"
            )
        # 若通过 QC，则采用经稳定化处理后的子波（确保 peak_lock 生效）
        w = w_checked
    return w



# Student-t 鲁棒反演状态码常量定义
_ROBUST_CODE_SUCCESS = 1                     # IRLS 正常收敛
_ROBUST_CODE_MAX_ITER = 2                    # 达到最大迭代次数后接受最佳解
_ROBUST_CODE_BACKTRACK_IMPROVED = 3          # 回溯中途停止，但最佳解相比初始解有改善
_ROBUST_CODE_LOW_EFFECTIVE_SAMPLE = -1       # 有效样本比例低于阈值被拒绝
_ROBUST_CODE_NONFINITE = -3                  # 权重解中包含无穷大或 NaN 异常被拒绝
_ROBUST_CODE_OBJECTIVE_INCREASE = -4         # 回溯拒绝且最佳解无改善被拒绝



@dataclass
class _RobustDiagnosticsStorage:
    attempted: np.ndarray
    sigma: np.ndarray
    irls_iterations: np.ndarray
    irls_converged: np.ndarray
    robust_code: np.ndarray
    solution_used: np.ndarray
    objective_initial: np.ndarray
    objective_final: np.ndarray
    objective_relative_decrease: np.ndarray
    residual_median: np.ndarray
    weight_mean: np.ndarray
    weight_min: np.ndarray
    outlier_fraction: np.ndarray
    effective_sample_size: np.ndarray
    effective_sample_ratio: np.ndarray
    weight_rows: list[np.ndarray]
    weight_center_indices: list[int]


# =========================================================
# Student-t 鲁棒反演辅助函数
# =========================================================
def _solve_student_t_window(
    *,
    R: np.ndarray,
    s_win: np.ndarray,
    w_l2: np.ndarray,
    sigma: float,
    nu: float,
    irls_max_iter: int,
    irls_tol: float,
    weight_floor: float,
    min_effective_sample_ratio: float,
    data_scale: float,
    I: np.ndarray,
    D: np.ndarray,
    C_dc: np.ndarray,
    mu1: float,
    mu2: float,
    mu_dc: float,
    prior_i: np.ndarray | None,
    mu_prior: float,
    previous_wavelet: np.ndarray | None,
    mu_time: float,
    damping_ratio: float,
    svd_cutoff_ratio: float,
    robust_outlier_weight_threshold: float,
    edge_weights: np.ndarray | None = None,
    mu_edge: float = 0.0,
) -> tuple[np.ndarray, dict[str, object]]:
    """在局部窗口中运行 Student-t 鲁棒 IRLS 迭代。

    输入参数:
        R: 局部卷积矩阵，shape: (data_window_length, wavelet_length)
        s_win: 局部观测地震记录窗口，shape: (data_window_length,)
        w_l2: L2 初始解子波，shape: (wavelet_length,)
        sigma: 局部残差尺度 (MAD)
        nu: Student-t 分布的自由度参数
        irls_max_iter: IRLS 最大迭代次数
        irls_tol: 迭代收敛容差
        weight_floor: IRLS 权重下限
        min_effective_sample_ratio: 最小有效样本比例阈值
        data_scale: 局部正则化尺度因子
        I: 单位阵，shape: (wavelet_length, wavelet_length)
        D: 二阶差分算子矩阵，shape: (wavelet_length - 2, wavelet_length)
        C_dc: 去直流约束算子，shape: (1, wavelet_length)
        mu1: 子波能量惩罚因子
        mu2: 子波二阶导数惩罚因子
        mu_dc: 非零均值惩罚因子
        prior_i: 平稳/时变子波先验值，shape: (wavelet_length,) 或 None
        mu_prior: 先验子波约束惩罚因子
        previous_wavelet: 相邻窗口上一个有效子波值，shape: (wavelet_length,) 或 None
        mu_time: 时间方向连续性惩罚因子
        damping_ratio: SVD 阻尼比例
        svd_cutoff_ratio: SVD 奇异值截断比例
        robust_outlier_weight_threshold: 异常值权重阈值，用于统计异常值比例

    返回:
        w_i: 求解所得子波，shape: (wavelet_length,)
        diagnostics: 包含求解过程诊断信息的字典
    """
    w_current = w_l2.copy()
    w_best = w_l2.copy()

    objective_initial = _full_student_t_objective(
        w=w_current,
        R=R,
        s_win=s_win,
        sigma=sigma,
        nu=nu,
        data_scale=data_scale,
        D=D,
        C_dc=C_dc,
        mu1=mu1,
        mu2=mu2,
        mu_dc=mu_dc,
        prior_i=prior_i,
        mu_prior=mu_prior,
        previous_wavelet=previous_wavelet,
        mu_time=mu_time,
        edge_weights=edge_weights,
        mu_edge=mu_edge,
    )
    objective_current = objective_initial
    objective_best = objective_initial

    robust_code = _ROBUST_CODE_MAX_ITER  # 默认：达到最大迭代次数后接受最佳解。
    irls_iterations = 0
    irls_converged = False

    for iteration in range(1, irls_max_iter + 1):
        irls_iterations = iteration

        residual_current = s_win - R @ w_current
        raw_weights = _student_t_raw_weights(
            residual=residual_current,
            sigma=sigma,
            nu=nu,
        )
        solve_weights = np.clip(raw_weights, weight_floor, 1.0)

        mean_weight = float(np.mean(raw_weights))
        if mean_weight < min_effective_sample_ratio:
            robust_code = _ROBUST_CODE_LOW_EFFECTIVE_SAMPLE
            break

        w_weighted = _solve_regularized_window(
            R=R,
            s_win=s_win,
            data_scale=data_scale,
            I=I,
            D=D,
            C_dc=C_dc,
            mu1=mu1,
            mu2=mu2,
            mu_dc=mu_dc,
            prior_i=prior_i,
            mu_prior=mu_prior,
            previous_wavelet=previous_wavelet,
            mu_time=mu_time,
            damping_ratio=damping_ratio,
            svd_cutoff_ratio=svd_cutoff_ratio,
            sample_weights=solve_weights,
            edge_weights=edge_weights,
            mu_edge=mu_edge,
        )

        if not np.all(np.isfinite(w_weighted)):
            robust_code = _ROBUST_CODE_NONFINITE
            break

        # SVD 阻尼和截断可能使完整步长不能严格下降，
        # 因此沿当前方向进行一个很轻量的回溯。
        direction = w_weighted - w_current
        objective_tol = 1e-10 * max(
            1.0,
            abs(objective_current),
        )
        accepted = False

        for alpha in (1.0, 0.5, 0.25, 0.125, 0.0625):
            w_candidate = w_current + alpha * direction
            objective_candidate = _full_student_t_objective(
                w=w_candidate,
                R=R,
                s_win=s_win,
                sigma=sigma,
                nu=nu,
                data_scale=data_scale,
                D=D,
                C_dc=C_dc,
                mu1=mu1,
                mu2=mu2,
                mu_dc=mu_dc,
                prior_i=prior_i,
                mu_prior=mu_prior,
                previous_wavelet=previous_wavelet,
                mu_time=mu_time,
                edge_weights=edge_weights,
                mu_edge=mu_edge,
            )

            if (
                np.isfinite(objective_candidate)
                and objective_candidate <= objective_current + objective_tol
            ):
                accepted = True
                break

        if not accepted:
            improvement_tol = 1e-10 * max(
                1.0,
                abs(objective_initial),
            )
            if objective_best < objective_initial - improvement_tol:
                robust_code = _ROBUST_CODE_BACKTRACK_IMPROVED
            else:
                robust_code = _ROBUST_CODE_OBJECTIVE_INCREASE
            break

        relative_change = (
            np.linalg.norm(w_candidate - w_current)
            / (np.linalg.norm(w_current) + 1e-12)
        )

        w_current = w_candidate
        objective_current = objective_candidate

        if objective_candidate < objective_best:
            w_best = w_candidate.copy()
            objective_best = objective_candidate

        if relative_change < irls_tol:
            robust_code = _ROBUST_CODE_SUCCESS
            irls_converged = True
            break

    improvement_tol = 1e-10 * max(
        1.0,
        abs(objective_initial),
    )
    robust_improved = (
        objective_best < objective_initial - improvement_tol
    )

    accepted = False
    if robust_code in (
        _ROBUST_CODE_SUCCESS,
        _ROBUST_CODE_MAX_ITER,
        _ROBUST_CODE_BACKTRACK_IMPROVED,
    ) and robust_improved:
        w_i = w_best
        accepted = True
    else:
        # 当前实现采用保守的窗口级 L2 回退。
        w_i = w_l2

    objective_final = (
        objective_best
        if accepted
        else objective_initial
    )

    # 诊断权重始终根据最终进入后续 QC 的原始解重新计算。
    residual_final = s_win - R @ w_i
    residual_median = float(np.median(residual_final))
    final_raw_weights = _student_t_raw_weights(
        residual=residual_final,
        sigma=sigma,
        nu=nu,
    )
    final_n_eff = _effective_sample_size(final_raw_weights)

    diagnostics = {
        "robust_attempted": True,                                                       # 是否尝试了稳健反演
        "robust_sigma": sigma,                                                          # 估算出的数据残差噪声尺度 (MAD)
        "irls_iterations": irls_iterations,                                             # 实际迭代次数
        "irls_converged": irls_converged,                                               # 是否收敛
        "robust_code": robust_code,                                                     # 退出状态码 (指示退出原因)
        "accepted": accepted,                                                           # 是否最终采纳了稳健估计解 (若为 False 则回退到 L2)
        "objective_initial": objective_initial,                                         # 初始目标函数值 (L2 起点)
        "objective_final": objective_final,                                             # 最终目标函数值 (优化终点)
        "objective_relative_decrease": (objective_initial - objective_final) / (abs(objective_initial) + 1e-12), # 目标函数相对下降比例
        "residual_median": residual_median,                                             # 最终拟合残差的中位数
        "weight_mean": float(np.mean(final_raw_weights)),                               # 最终样本权重的平均值
        "weight_min": float(np.min(final_raw_weights)),                                 # 最小样本权重
        "outlier_fraction": float(np.mean(final_raw_weights < robust_outlier_weight_threshold)), # 被判定为异常值的样本比例
        "effective_sample_size": final_n_eff,                                           # Kish 有效样本数
        "effective_sample_ratio": final_n_eff / float(final_raw_weights.size),          # 有效样本比例
        "final_raw_weights": final_raw_weights,                                         # 最终的样本原始权重数组
    }

    return w_i, diagnostics


def _store_robust_diagnostics(
    *,
    window_index: int,
    center_sample_index: int,
    w_i: np.ndarray,
    R: np.ndarray,
    s_win: np.ndarray,
    sigma: float,
    nu: float,
    robust_outlier_weight_threshold: float,
    robust_diag: dict[str, object],
    robust_solution_used: bool,
    store_weight_map: bool,
    storage: _RobustDiagnosticsStorage,
) -> None:
    """计算最终选定解的各权重指标，并将其记录到全局诊断数组中。"""
    # 这里的全局诊断数组长度均为 N，因此应当统一使用 center_sample_index 进行定址。
    storage.attempted[center_sample_index] = robust_diag["robust_attempted"]
    storage.sigma[center_sample_index] = robust_diag["robust_sigma"]
    storage.irls_iterations[center_sample_index] = robust_diag["irls_iterations"]
    storage.irls_converged[center_sample_index] = robust_diag["irls_converged"]
    storage.robust_code[center_sample_index] = robust_diag["robust_code"]
    storage.solution_used[center_sample_index] = robust_solution_used
    storage.objective_initial[center_sample_index] = robust_diag["objective_initial"]

    if robust_solution_used:
        storage.objective_final[center_sample_index] = robust_diag["objective_final"]
        storage.objective_relative_decrease[center_sample_index] = robust_diag["objective_relative_decrease"]
    else:
        storage.objective_final[center_sample_index] = robust_diag["objective_initial"]
        storage.objective_relative_decrease[center_sample_index] = 0.0

    # 诊断权重始终根据最终进入后续 QC 的原始解重新计算。
    residual_final = s_win - R @ w_i
    storage.residual_median[center_sample_index] = float(np.median(residual_final))
    final_raw_weights = _student_t_raw_weights(
        residual=residual_final,
        sigma=sigma,
        nu=nu,
    )
    final_n_eff = _effective_sample_size(final_raw_weights)

    storage.weight_mean[center_sample_index] = float(np.mean(final_raw_weights))
    storage.weight_min[center_sample_index] = float(np.min(final_raw_weights))
    storage.outlier_fraction[center_sample_index] = float(
        np.mean(final_raw_weights < robust_outlier_weight_threshold)
    )
    storage.effective_sample_size[center_sample_index] = final_n_eff
    storage.effective_sample_ratio[center_sample_index] = final_n_eff / float(final_raw_weights.size)

    if store_weight_map:
        storage.weight_rows.append(
            np.asarray(final_raw_weights, dtype=np.float32).copy()
        )
        storage.weight_center_indices.append(center_sample_index)


# =========================================================
# 时变子波反演
# =========================================================
def time_varying_wavelet_inversion(
    r_time: np.ndarray,
    s_obs: np.ndarray,
    wavelet_length: int,
    data_window_length: int | None = None,
    dt: float = 0.001,
    wavelet_alignment: str = "center",
    peak_allowed_ms: tuple[float, float] | None = None,
    # 基础正则
    mu1: float = 0.2,
    mu2: float = 3.0,
    mu_dc: float = 10.0,
    # 先验与时间连续性
    w_prior: np.ndarray | None = None,
    use_stationary_prior: bool = True,
    mu_prior: float = 0.3,
    mu_time: float = 0.5,
    # 子波边缘能量约束；mu_edge=0 时严格保持旧求解路径
    mu_edge: float = 0.0,
    edge_fraction: float = 0.12,
    edge_taper: str = "cosine",
    # 低反射能量跳过
    energy_threshold: float | None = None,
    energy_percentile: float = 35.0,
    # SVD 稳定性
    damping_ratio: float = 3e-3,
    svd_cutoff_ratio: float = 1e-3,
    # 稀疏估计步长
    estimate_step_samples: int | None = None,
    estimate_step_ms: float = 5.0,
    # 病态窗口拒绝
    reject_ill_conditioned: bool = True,
    rank_cutoff_ratio: float = 1e-3,
    min_effective_rank: int | None = None,
    min_condition_ratio: float = 0.0,
    # 峰值与振幅 QC
    polarity_continuity: bool = True,
    peak_lock: bool = True,
    max_peak_shift_ms: float = 15.0,
    reject_amplitude_jumps: bool = True,
    max_norm_ratio: float = 3.0,
    min_norm_ratio: float = 1.0 / 3.0,
    # 反演后平滑
    time_smooth_sigma: float | None = None,
    time_smooth_ms: float = 15.0,
    wavelet_smooth_sigma: float = 1.2,
    # 数据拟合类型
    loss_type: str = "l2",
    # Student-t IRLS 参数
    student_nu: float = 10.0,
    irls_max_iter: int = 10,
    irls_tol: float = 1e-4,
    robust_scale_mode: str = "local_mad",
    robust_scale_floor_ratio: float = 0.05,
    robust_weight_floor: float = 1e-3,
    min_effective_sample_ratio: float = 0.30,
    robust_fallback: str = "l2",
    robust_outlier_weight_threshold: float = 0.5,
    store_weight_map: bool = False,
    # 输出控制
    return_diagnostics: bool = False,
    verbose: bool = True,
    store_stage_wavelets: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray | int | float]]:
    """
    反演时变子波矩阵 W。

    推荐真实资料默认设置：
        wavelet_length        = 129 或 161
        data_window_length    = 3 * wavelet_length 左右，且取奇数
        estimate_step_ms      = 5~10 ms
        energy_percentile     = 35~50
        mu1                  = 0.2~0.5
        mu2                  = 3~10
        mu_dc                = 5~20
        mu_prior             = 0.2~0.5
        mu_time              = 0.3~1.0
        time_smooth_ms        = 15~30 ms

    返回：
        默认返回 W。
        若 return_diagnostics=True，返回 (W, diagnostics)。
        同时启用 store_stage_wavelets 时，诊断增加三个 shape=(N, wavelet_length)
        的独立快照，单位沿用反演子波振幅：W_direct_centers 为通过局部 QC 的
        直接反演中心，其余行标为 NaN；W_pre_gaussian 为原有填充之后的矩阵；
        W_post_gaussian 为原有二维 Gaussian 之后、最终去均值之前的矩阵。
        快照不参与求解、插值、QC 或模型选择；时间和 lag 采样间隔均为 dt (s)。
    """
    r_time = np.asarray(r_time, dtype=float).copy()
    s_obs = np.asarray(s_obs, dtype=float).copy()

    if r_time.ndim != 1 or s_obs.ndim != 1:
        raise ValueError("r_time 和 s_obs 必须是一维数组。")
    if len(r_time) != len(s_obs):
        raise ValueError("r_time 和 s_obs 长度必须一致。")
    if dt <= 0:
        raise ValueError("dt 必须大于 0。")
    wavelet_alignment = _validate_wavelet_alignment(wavelet_alignment)

    mu1 = _validate_nonnegative_scalar(mu1, "mu1")
    mu2 = _validate_nonnegative_scalar(mu2, "mu2")
    mu_dc = _validate_nonnegative_scalar(mu_dc, "mu_dc")
    mu_prior = _validate_nonnegative_scalar(mu_prior, "mu_prior")
    mu_time = _validate_nonnegative_scalar(mu_time, "mu_time")
    mu_edge = _validate_nonnegative_scalar(mu_edge, "mu_edge")

    edge_fraction_value = np.asarray(edge_fraction, dtype=float)
    if (
        edge_fraction_value.ndim != 0
        or not np.isfinite(edge_fraction_value)
        or not 0.0 < float(edge_fraction_value) < 0.5
    ):
        raise ValueError("edge_fraction 必须位于 (0, 0.5)。")
    edge_fraction = float(edge_fraction_value)

    edge_taper = str(edge_taper).strip().lower()
    if edge_taper != "cosine":
        raise ValueError("edge_taper 当前仅支持 'cosine'。")

    loss_type = str(loss_type).strip().lower()
    if loss_type not in ("l2", "student_t"):
        raise ValueError("loss_type 必须是 'l2' 或 'student_t'。")

    if loss_type == "student_t":
        robust_scale_mode = str(robust_scale_mode).strip().lower()
        robust_fallback = str(robust_fallback).strip().lower()

        if student_nu <= 0:
            raise ValueError("student_nu 必须大于 0。")
        if int(irls_max_iter) < 1:
            raise ValueError("irls_max_iter 必须至少为 1。")
        if irls_tol <= 0:
            raise ValueError("irls_tol 必须大于 0。")
        if robust_scale_mode != "local_mad":
            raise ValueError("robust_scale_mode 当前仅支持 'local_mad'。")
        if robust_scale_floor_ratio < 0:
            raise ValueError("robust_scale_floor_ratio 不能为负数。")
        if not 0.0 <= robust_weight_floor <= 1.0:
            raise ValueError("robust_weight_floor 必须位于 [0, 1]。")
        if not 0.0 < min_effective_sample_ratio <= 1.0:
            raise ValueError("min_effective_sample_ratio 必须位于 (0, 1]。")
        if robust_fallback != "l2":
            raise ValueError("当前 robust_fallback 仅支持 'l2'。")
        if not 0.0 < robust_outlier_weight_threshold < 1.0:
            raise ValueError(
                "robust_outlier_weight_threshold 必须位于 (0, 1)。"
            )
        if robust_weight_floor >= robust_outlier_weight_threshold:
            raise ValueError(
                "robust_weight_floor 必须小于 robust_outlier_weight_threshold。"
            )
        irls_max_iter = int(irls_max_iter)

    store_weight_map = bool(store_weight_map)
    store_stage_wavelets = bool(store_stage_wavelets) and bool(return_diagnostics)

    N = len(r_time)
    wavelet_length = _make_odd(wavelet_length)

    # 仅在真正启用边缘约束时构造权重。mu_edge=0 时 edge_weights 保持 None，
    # 从而不改变旧版增广矩阵的 shape、块顺序和 SVD 输入。
    edge_weights = None
    if mu_edge > 0.0:
        edge_weights = build_edge_penalty_weights(
            wavelet_length=wavelet_length,
            edge_fraction=edge_fraction,
            taper=edge_taper,
        )

    if data_window_length is None:
        # 真实资料建议比 2L+1 更长一点，降低自由度；这里给保守默认。
        data_window_length = max(2 * wavelet_length + 1, 3 * wavelet_length)

    data_window_length = _make_odd(data_window_length)

    if data_window_length <= wavelet_length:
        raise ValueError("data_window_length 必须大于 wavelet_length。")
    if N < data_window_length:
        raise ValueError("N 小于 data_window_length，请缩短数据窗或检查输入。")
    # 稀疏估计步长，避免逐样点求解，相邻子波重叠部分多。
    if estimate_step_samples is None:
        estimate_step_samples = max(1, int(round((estimate_step_ms / 1000.0) / dt)))
    else:
        estimate_step_samples = max(1, int(estimate_step_samples))

    if time_smooth_sigma is None:
        time_smooth_sigma = max(0.0, float((time_smooth_ms / 1000.0) / dt))

    max_peak_shift_samples = max(1, int(round((max_peak_shift_ms / 1000.0) / dt)))
    if peak_allowed_ms is not None:
        peak_allowed_samples = (
            int(round((peak_allowed_ms[0] / 1000.0) / dt)),
            int(round((peak_allowed_ms[1] / 1000.0) / dt)),
        )
    else:
        peak_allowed_samples = None

    if wavelet_alignment == "causal" and peak_allowed_samples is None:
        # 默认沿用 max_peak_shift_ms 的数值，但语义变为：
        # causal 子波主峰允许出现在 0 ~ max_peak_shift_ms 之间。
        peak_allowed_samples = (0, max_peak_shift_samples)
    # 去掉观测道直流，防止局部 LS 吸收基线漂移。
    s_obs = s_obs - np.mean(s_obs)

    D = _second_derivative_matrix(wavelet_length)
    I = np.eye(wavelet_length, dtype=float)
    C_dc = np.ones((1, wavelet_length), dtype=float) / np.sqrt(wavelet_length)

    # 自动计算平稳先验。
    if w_prior is None and use_stationary_prior and mu_prior > 0:
        w_prior = stationary_wavelet_inversion(
            r_time=r_time,
            s_obs=s_obs,
            wavelet_length=wavelet_length,
            mu1=max(mu1, 0.2),
            mu2=max(mu2, 2.0),
            mu_dc=mu_dc,
            mu_edge=mu_edge,
            edge_fraction=edge_fraction,
            edge_taper=edge_taper,
            damping_ratio=damping_ratio,
            svd_cutoff_ratio=svd_cutoff_ratio,
            peak_lock=False,
            wavelet_alignment=wavelet_alignment,
        )

    if w_prior is not None:
        # 提前校验一次，避免在循环内部反复报错。
        _ = _get_prior_wavelet(w_prior, 0, wavelet_length)

    W = np.zeros((N, wavelet_length), dtype=float)
    valid_mask = np.zeros(N, dtype=bool)

    if energy_threshold is None:
        energy_threshold = _estimate_reflectivity_energy_threshold(
            r_time=r_time,
            data_window_length=data_window_length,
            energy_percentile=energy_percentile,
        )

    half_data = data_window_length // 2
    global_data_scale = max(np.sum(s_obs ** 2) / N, 1e-12)
    global_data_rms = float(np.sqrt(global_data_scale))
    # 噪声尺度标准差的“绝对下限值”（robust_scale_floor）
    robust_scale_floor = max(
        float(robust_scale_floor_ratio) * global_data_rms,
        1e-8,
    )

    if min_effective_rank is None:
        min_effective_rank = min(
            wavelet_length,
            max(3, wavelet_length // 8),
        )

    previous_valid_wavelet = None

    # 诊断数组
    diag_effective_rank = np.full(N, np.nan)
    diag_condition_ratio = np.full(N, np.nan)
    diag_peak_shift = np.full(N, np.nan)
    diag_r_energy = np.full(N, np.nan)
    diag_s_energy = np.full(N, np.nan)
    diag_skip_code = np.zeros(N, dtype=int)
    # skip_code: 0 未尝试；1 有效；-1低反射能量；-2病态矩阵；-3NaN；-4峰值过远；-5振幅跳变

    # Student-t 诊断与原 skip_code 分离。
    diag_robust_sigma = np.full(N, np.nan)
    diag_irls_iterations = np.zeros(N, dtype=int)
    diag_irls_converged = np.zeros(N, dtype=bool)
    diag_robust_code = np.zeros(N, dtype=int)
    diag_robust_attempted = np.zeros(N, dtype=bool)
    diag_robust_solution_used = np.zeros(N, dtype=bool)
    diag_weight_mean = np.full(N, np.nan)
    diag_weight_min = np.full(N, np.nan)
    diag_outlier_fraction = np.full(N, np.nan)
    diag_effective_sample_size = np.full(N, np.nan)
    diag_effective_sample_ratio = np.full(N, np.nan)
    diag_objective_initial = np.full(N, np.nan)
    diag_objective_final = np.full(N, np.nan)
    diag_objective_relative_decrease = np.full(N, np.nan)
    diag_residual_median = np.full(N, np.nan)

    # 只保存真正进入 Student-t 的稀疏估计中心，避免构造 N x M 大矩阵。
    robust_weight_rows: list[np.ndarray] = []
    robust_weight_center_indices: list[int] = []

    # 将所有诊断存储容器打包
    robust_storage = _RobustDiagnosticsStorage(
        attempted=diag_robust_attempted,
        sigma=diag_robust_sigma,
        irls_iterations=diag_irls_iterations,
        irls_converged=diag_irls_converged,
        robust_code=diag_robust_code,
        solution_used=diag_robust_solution_used,
        objective_initial=diag_objective_initial,
        objective_final=diag_objective_final,
        objective_relative_decrease=diag_objective_relative_decrease,
        residual_median=diag_residual_median,
        weight_mean=diag_weight_mean,
        weight_min=diag_weight_min,
        outlier_fraction=diag_outlier_fraction,
        effective_sample_size=diag_effective_sample_size,
        effective_sample_ratio=diag_effective_sample_ratio,
        weight_rows=robust_weight_rows,
        weight_center_indices=robust_weight_center_indices,
    )
    robust_window_offsets_samples = (
        np.arange(data_window_length, dtype=int) - half_data
    )

    active_regularizers = {
        "mu1": bool(mu1 > 0.0),
        "mu2": bool(mu2 > 0.0),
        "mu_dc": bool(mu_dc > 0.0),
        "mu_prior": bool(mu_prior > 0.0),
        "mu_time": bool(mu_time > 0.0),
        "mu_edge": bool(mu_edge > 0.0),
    }

    if verbose:
        print("========== Robust time-varying wavelet inversion ==========")
        print(f"  N_time                = {N}")
        print(f"  dt                    = {dt:.6f} s")
        print(f"  wavelet_length        = {wavelet_length}")
        print(f"  data_window_length    = {data_window_length}")
        print(f"  estimate_step_samples = {estimate_step_samples}")
        print(f"  estimate_step_ms      = {estimate_step_samples * dt * 1000:.2f} ms")
        print(f"  mu1                   = {mu1}")
        print(f"  mu2                   = {mu2}")
        print(f"  mu_dc                 = {mu_dc}")
        print(f"  mu_prior              = {mu_prior}")
        print(f"  mu_time               = {mu_time}")
        print(f"  mu_edge               = {mu_edge}")
        print(f"  active_regularizers   = {active_regularizers}")
        print(f"  edge_fraction         = {edge_fraction}")
        print(f"  edge_taper            = {edge_taper}")
        print(f"  energy_threshold      = {energy_threshold:.6e}")
        print(f"  damping_ratio         = {damping_ratio}")
        print(f"  svd_cutoff_ratio      = {svd_cutoff_ratio}")
        print(f"  min_effective_rank    = {min_effective_rank}")
        print(f"  max_peak_shift        = {max_peak_shift_samples} samples")
        print(f"  time_smooth_sigma     = {time_smooth_sigma:.2f} samples")
        print(f"  wavelet_smooth_sigma  = {wavelet_smooth_sigma:.2f} samples")
        print(f"  has_wavelet_prior     = {w_prior is not None}")
        print(f"  loss_type             = {loss_type}")
        if loss_type == "student_t":
            print(f"  student_nu            = {student_nu}")
            print(f"  irls_max_iter         = {irls_max_iter}")
            print(f"  irls_tol              = {irls_tol}")
            print(f"  robust_scale_floor    = {robust_scale_floor:.6e}")
            print(f"  robust_weight_floor   = {robust_weight_floor}")
            print(f"  min_effective_ratio   = {min_effective_sample_ratio}")
            print(
                f"  outlier_weight_thresh = "
                f"{robust_outlier_weight_threshold}"
            )
            print(f"  store_weight_map      = {store_weight_map}")
    # enumerate 函数会自动把 range 产生的数据包装成 (序号, 数值) 的元组：window_index（序号）；center_sample_index（数值）
    for window_index, center_sample_index in enumerate(
        range(half_data, N - half_data, estimate_step_samples)
    ):
        center_sample_index = int(center_sample_index)
        # 为保证现有数组切片与局部变量兼容，提供别名 i 指向 center_sample_index
        i = center_sample_index

        r_win = _extract_window(r_time, i, data_window_length)
        s_win = _extract_window(s_obs, i, data_window_length)

        r_energy = float(np.sum(r_win ** 2))
        s_energy = float(np.sum(s_win ** 2))
        diag_r_energy[i] = r_energy
        diag_s_energy[i] = s_energy

        if r_energy < energy_threshold:
            diag_skip_code[i] = -1
            continue

        R = build_local_convolution_matrix(
            r_time=r_time,
            center_index=i,
            data_window_length=data_window_length,
            wavelet_length=wavelet_length,
            alignment=wavelet_alignment,
          )

        effective_rank, condition_ratio, _ = _effective_rank_and_condition(
            R,
            rank_cutoff_ratio=rank_cutoff_ratio,
        )
        diag_effective_rank[i] = effective_rank
        diag_condition_ratio[i] = condition_ratio

        if reject_ill_conditioned:
            if effective_rank < min_effective_rank or condition_ratio < min_condition_ratio:
                diag_skip_code[i] = -2
                continue

        # 正则尺度：同时参考局部能量和全局能量。
        # 避免弱能量窗中正则过弱，也避免强事件窗完全主导反演。
        local_data_scale = s_energy / data_window_length
        data_scale = max(local_data_scale, 0.25 * global_data_scale, 1e-12)

        # 当前窗口的平稳先验；与原代码保持相同的获取时机和缩放。
        prior_i = (
            _get_prior_wavelet(w_prior, i, wavelet_length)
            if w_prior is not None
            else None
        )

        # 无论使用哪种损失，每个窗口都先求一次原始 L2 解。
        w_l2 = _solve_regularized_window(
            R=R,
            s_win=s_win,
            data_scale=data_scale,
            I=I,
            D=D,
            C_dc=C_dc,
            mu1=mu1,
            mu2=mu2,
            mu_dc=mu_dc,
            prior_i=prior_i,
            mu_prior=mu_prior,
            previous_wavelet=previous_valid_wavelet,
            mu_time=mu_time,
            damping_ratio=damping_ratio,
            svd_cutoff_ratio=svd_cutoff_ratio,
            sample_weights=None,
            edge_weights=edge_weights,
            mu_edge=mu_edge,
        )

        if loss_type == "l2":
            # 保持第一阶段的数值路径不变。
            w_i = w_l2
            diag_residual_median[i] = float(np.median(s_win - R @ w_i))

        else:
            sigma = _mad_scale(
                residual=s_win - R @ w_l2,
                scale_floor=robust_scale_floor,
            )

            w_candidate, robust_diag = _solve_student_t_window(
                R=R,
                s_win=s_win,
                w_l2=w_l2,
                sigma=sigma,
                nu=student_nu,
                irls_max_iter=irls_max_iter,
                irls_tol=irls_tol,
                weight_floor=robust_weight_floor,
                min_effective_sample_ratio=min_effective_sample_ratio,
                data_scale=data_scale,
                I=I,
                D=D,
                C_dc=C_dc,
                mu1=mu1,
                mu2=mu2,
                mu_dc=mu_dc,
                prior_i=prior_i,
                mu_prior=mu_prior,
                previous_wavelet=previous_valid_wavelet,
                mu_time=mu_time,
                damping_ratio=damping_ratio,
                svd_cutoff_ratio=svd_cutoff_ratio,
                robust_outlier_weight_threshold=robust_outlier_weight_threshold,
                edge_weights=edge_weights,
                mu_edge=mu_edge,
            )

            # 主循环明确决定是否回退 (方案 B)
            if robust_diag["accepted"]:
                w_i = w_candidate
                candidate_is_student_t = True
            else:
                w_i = w_l2
                candidate_is_student_t = False

            # 注意：
            # robust_solution_used 表示 Student-t 候选解在鲁棒求解阶段
            # 替代了 L2 解，并进入后续公共物理 QC。
            #
            # 它不表示该 Student-t 解最终一定被写入输出；
            # 后续峰值、能量、连续性等物理 QC 仍可能拒绝该候选解。
            robust_solution_used = candidate_is_student_t
            #robust_solution_used 的值存入 robust_diag 字典中
            robust_diag["robust_solution_used"] = robust_solution_used

            _store_robust_diagnostics(
                window_index=window_index,
                center_sample_index=center_sample_index,
                w_i=w_i,
                R=R,
                s_win=s_win,
                sigma=sigma,
                nu=student_nu,
                robust_outlier_weight_threshold=robust_outlier_weight_threshold,
                robust_diag=robust_diag,
                robust_solution_used=robust_solution_used,
                store_weight_map=store_weight_map,
                storage=robust_storage,
            )
# 检查反演出来的子波是否包含NaN或inf
        if not np.all(np.isfinite(w_i)):
            diag_skip_code[i] = -3
            continue

        w_i = _zero_mean_wavelet(w_i)

        if polarity_continuity and previous_valid_wavelet is not None:
            if np.dot(w_i, previous_valid_wavelet) < 0:
                w_i = -w_i
# 峰值判断，反演子波峰值是否超过偏离中心的极限，15样点
        if peak_lock:
            w_checked, peak_metric = _stabilize_wavelet_peak(
                w_i,
                max_peak_shift_samples=max_peak_shift_samples,
                wavelet_alignment=wavelet_alignment,
                # 只有当子波模式为causal的模型起作用，center模式不起作用
                peak_allowed_samples=peak_allowed_samples,
            )
            # 每个计算窗口的子波峰值指标（偏移点数或绝对位置）保存到全局诊断数组中。
            diag_peak_shift[i] = peak_metric
            if w_checked is None:
                diag_skip_code[i] = -4
                continue
            w_i = w_checked
        # 关闭主峰判断只计算偏移值作监控    
        else:
            diag_peak_shift[i] = _peak_metric_samples(w_i, wavelet_alignment)

        if reject_amplitude_jumps and previous_valid_wavelet is not None:
            norm_prev = np.linalg.norm(previous_valid_wavelet) + 1e-12
            norm_now = np.linalg.norm(w_i) + 1e-12
            ratio = norm_now / norm_prev
            if ratio > max_norm_ratio or ratio < min_norm_ratio:
                diag_skip_code[i] = -5
                continue

        W[i, :] = w_i
        valid_mask[i] = True
        diag_skip_code[i] = 1
        previous_valid_wavelet = w_i.copy()

    # 在填充前记录候选子波来源谱系。来源掩码只描述“直接反演/插值/外推/先验填充”，
    # 不把后续二维高斯平滑误标成新的直接反演结果。
    candidate_origin = _build_candidate_wavelet_origin_masks(
        valid_mask=valid_mask,
        prior_available=w_prior is not None,
    )

    if store_stage_wavelets:
        # shape=(N, wavelet_length)。只复制已通过局部 QC 的中心子波。
        # 未反演/被拒绝的行没有直接估计值，用 NaN 标识缺测，不能当作零子波。
        W_direct_centers = W.copy()
        W_direct_centers[~valid_mask, :] = np.nan

    # 用线性插值填充稀疏估计点；若一个有效点都没有，则退回 w_prior。
    W = _fill_unestimated_wavelets(W, valid_mask, w_prior=w_prior)

    if store_stage_wavelets:
        # shape=(N, wavelet_length)。保存原有 fill 的结果，不增加任何插值操作。
        W_pre_gaussian = W.copy()

    # 反演后平滑。
    if time_smooth_sigma is not None and wavelet_smooth_sigma is not None:
        if time_smooth_sigma > 0 or wavelet_smooth_sigma > 0:
            W = gaussian_filter(
                W,
                sigma=(float(time_smooth_sigma), float(wavelet_smooth_sigma)),
                mode="nearest",
            )

    if store_stage_wavelets:
        # shape=(N, wavelet_length)。记录 G(W_fill)，尚未执行下一行去均值。
        W_post_gaussian = W.copy()

    # 平滑后再次去直流，防止二维滤波引入极小 DC。
    W = W - np.mean(W, axis=1, keepdims=True)

    qc = _wavelet_qc_attributes(
        W,
        dt=dt,
        wavelet_alignment=wavelet_alignment,
    )

    if robust_weight_rows:
        robust_weight_map = np.vstack(robust_weight_rows).astype(
            np.float32,
            copy=False,
        )
        robust_weight_center_indices_array = np.asarray(
            robust_weight_center_indices,
            dtype=int,
        )
    else:
        robust_weight_map = np.empty(
            (0, data_window_length),
            dtype=np.float32,
        )
        robust_weight_center_indices_array = np.array([], dtype=int)

    diagnostics: dict[str, object] = {
        "valid_mask": valid_mask,                                      # 布尔掩码 (N,)：指示哪些时间点直接成功反演了子波
        "active_regularizers": active_regularizers,                   # 真实激活状态布尔字典（便于科研分析与消融溯源）
        "residual_median": diag_residual_median,                      # 最终拟合残差的中位数 (N,)
        "skip_code": diag_skip_code,                    # 决策状态码 (N,)：0-未尝试, 1-成功, 负数-各级拒绝原因
        "effective_rank": diag_effective_rank,          # 局部有效秩 (N,)：局部卷积矩阵在SVD下的有效自由度
        "condition_ratio": diag_condition_ratio,        # 局部条件比例 (N,)：最小与最大奇异值之比，越接近0越病态
        "peak_shift_raw_samples": diag_peak_shift,      # 原始主峰偏移量 (N,)：平滑前反演子波主峰偏离中心的采样点数
        "r_energy": diag_r_energy,                      # 局部反射系数能量 (N,)：滑动窗口内反射系数的平方和
        "s_energy": diag_s_energy,                      # 局部观测地震能量 (N,)：滑动窗口内地震振幅的平方和
        "energy_threshold": float(energy_threshold),    # 估算/使用的绝对反射能量阈值
        "estimate_step_samples": int(estimate_step_samples), # 子波稀疏估计的时间步长（采样点数）
        "time_smooth_sigma": float(time_smooth_sigma),  # 时间轴方向高斯平滑的标准差（采样点数）
        "wavelet_alignment": wavelet_alignment,          # 子波时间轴对齐方式：'center' (中心) 或 'causal' (因果)
        "loss_type": loss_type,                          # 当前数据拟合项：l2 或 student_t
        "mu_edge": float(mu_edge),
        "edge_penalty_active": bool(mu_edge > 0.0),
        "edge_fraction": float(edge_fraction),
        "edge_taper": edge_taper,
        "edge_penalty_weights": (
            edge_weights.copy()
            if edge_weights is not None
            else np.array([], dtype=float)
        ),
        "student_objective_includes_edge": bool(
            loss_type == "student_t" and mu_edge > 0.0
        ),
        "candidate_wavelet_origin_code": candidate_origin["origin_code"],
        "candidate_inverted_mask": candidate_origin["inverted_mask"],
        "candidate_interpolated_mask": candidate_origin["interpolated_mask"],
        "candidate_extrapolated_mask": candidate_origin["extrapolated_mask"],
        "candidate_prior_fill_mask": candidate_origin["fallback_prior_mask"],
        "candidate_unavailable_mask": candidate_origin["unavailable_mask"],
        "candidate_origin_scope": "pre_gaussian_fill_lineage",
        "student_nu": float(student_nu),
        "irls_max_iter": int(irls_max_iter),
        "irls_tol": float(irls_tol),
        "robust_scale_mode": robust_scale_mode,
        "robust_scale_floor_ratio": float(robust_scale_floor_ratio),
        "robust_scale_floor": float(robust_scale_floor),
        "robust_weight_floor": float(robust_weight_floor),
        "min_effective_sample_ratio": float(min_effective_sample_ratio),
        "robust_fallback": robust_fallback,
        "robust_outlier_weight_threshold": float(
            robust_outlier_weight_threshold
        ),
        "store_weight_map": bool(store_weight_map),
        "robust_sigma": diag_robust_sigma,
        "irls_iterations": diag_irls_iterations,
        "irls_converged": diag_irls_converged,
        "irls_step_converged": diag_irls_converged,
        "post_qc_valid": diag_robust_solution_used & valid_mask,
        "robust_code": diag_robust_code,
        "robust_attempted": diag_robust_attempted,
        "robust_solution_used": diag_robust_solution_used,
        "weight_mean": diag_weight_mean,
        "weight_min": diag_weight_min,
        "outlier_fraction": diag_outlier_fraction,
        "kish_size": diag_effective_sample_size,
        "kish_ratio": diag_effective_sample_ratio,
        "effective_sample_size": diag_effective_sample_size,
        "effective_sample_ratio": diag_effective_sample_ratio,
        "student_objective_initial": diag_objective_initial,
        "student_objective_final": diag_objective_final,
        "student_objective_relative_decrease": diag_objective_relative_decrease,
        "robust_center_indices": np.where(diag_robust_attempted)[0],
        "robust_weight_map": robust_weight_map,
        "robust_weight_center_indices": robust_weight_center_indices_array,
        "robust_window_offsets_samples": robust_window_offsets_samples,
        # 权重矩阵是在原始 IRLS 解上计算；以下逐行状态说明该窗口
        # 是否真正采用了 Student-t 解，以及是否通过后续物理 QC。
        "robust_weight_solution_used": (
            diag_robust_solution_used[robust_weight_center_indices_array]
            if robust_weight_center_indices_array.size
            else np.array([], dtype=bool)
        ),
        "robust_weight_post_qc_valid": (
            valid_mask[robust_weight_center_indices_array]
            if robust_weight_center_indices_array.size
            else np.array([], dtype=bool)
        ),
        "robust_weight_skip_code": (
            diag_skip_code[robust_weight_center_indices_array]
            if robust_weight_center_indices_array.size
            else np.array([], dtype=int)
        ),
        "n_robust_attempted": int(np.sum(diag_robust_attempted)),
        "n_robust_solution_used": int(np.sum(diag_robust_solution_used)),
        "n_robust_converged": int(np.sum(diag_irls_converged)),
        "peak_allowed_samples": peak_allowed_samples,    # 因果模式下允许的主峰采样点范围 (min, max) 或 None
        "n_valid": int(np.sum(valid_mask)),              # 整个地震道中成功进行直接反演的窗口总数
        "n_attempted": int(np.sum(diag_skip_code != 0)), # 整个地震道中尝试进行反演的窗口总数
        "qc_energy": qc["energy"],                      # 最终平滑后子波的能量 (N,)：每行子波向量的L2范数
        "qc_peak_shift_samples": qc["peak_shift_samples"], # 最终平滑后子波的主峰偏移采样点数 (N,)
        "qc_peak_position_samples": qc["peak_position_samples"], # 最终平滑后子波的主峰绝对位置索引 (N,)
        "qc_dc_ratio": qc["dc_ratio"],                  # 最终平滑后子波的DC(直流)比例 (N,)：越接近0越符合物理特性
        "qc_centroid_freq": qc["centroid_freq"],        # 最终平滑后子波的频谱质心频率 (N,)，单位：Hz
        "qc_bandwidth": qc["bandwidth"],                # 最终平滑后子波的频谱频带宽度 (N,)，单位：Hz
    }
    if store_stage_wavelets:
        diagnostics["store_stage_wavelets"] = True
        diagnostics["W_direct_centers"] = W_direct_centers
        diagnostics["W_pre_gaussian"] = W_pre_gaussian
        diagnostics["W_post_gaussian"] = W_post_gaussian

#　输出运行参数和日志
    if verbose:
        n_valid = int(np.sum(valid_mask))
        n_attempted = int(np.sum(diag_skip_code != 0))
        n_low_energy = int(np.sum(diag_skip_code == -1))
        n_ill = int(np.sum(diag_skip_code == -2))
        n_nan = int(np.sum(diag_skip_code == -3))
        n_peak = int(np.sum(diag_skip_code == -4))
        n_amp = int(np.sum(diag_skip_code == -5))
        print(f"  attempted windows      = {n_attempted}")
        print(f"  valid inverted centers = {n_valid} / {n_attempted}")
        print(f"  skipped low energy     = {n_low_energy}")
        print(f"  skipped ill-cond       = {n_ill}")
        print(f"  skipped NaN/Inf        = {n_nan}")
        print(f"  skipped peak shift     = {n_peak}")
        print(f"  skipped amp jump       = {n_amp}")
        if loss_type == "student_t":
            n_robust_attempted = int(np.sum(diag_robust_attempted))
            n_irls_step_converged = int(np.sum(diag_irls_converged)) # 布尔变量true=1,false=0，求和就是true的个数，统计收敛的个数
            n_robust_solution_used = int(np.sum(diag_robust_solution_used))
            n_post_qc_valid = int(np.sum(diag_robust_solution_used & valid_mask))
            n_robust_fallback = n_robust_attempted - n_robust_solution_used
            print(f"  robust attempted       = {n_robust_attempted}")
            print(f"  irls_step_converged    = {n_irls_step_converged}")
            print(f"  robust solution used   = {n_robust_solution_used}")
            print(f"  post_qc_valid (robust) = {n_post_qc_valid}")
            print(f"  robust fallback to L2  = {n_robust_fallback}")
            if n_robust_attempted > 0:
                attempted = diag_robust_attempted
                print(
                    "  median robust sigma    = "
                    f"{np.nanmedian(diag_robust_sigma[attempted]):.6e}"
                )
                print(
                    "  median weight mean     = "
                    f"{np.nanmedian(diag_weight_mean[attempted]):.4f}"
                )
                print(
                    "  median effective ratio = "
                    f"{np.nanmedian(diag_effective_sample_ratio[attempted]):.4f}"
                )
        print(f"  median qc_dc_ratio     = {np.nanmedian(qc['dc_ratio']):.3e}")
        print(f"  median centroid_freq   = {np.nanmedian(qc['centroid_freq']):.2f} Hz")
        print("============================================================")
        print(f"  wavelet_alignment     = {wavelet_alignment}")
        if wavelet_alignment == "causal":
            print(f"  peak_allowed_samples  = {peak_allowed_samples}")
    if return_diagnostics:
        return W, diagnostics
    return W




# =========================================================
# 自测
# =========================================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    np.random.seed(123)

    dt = 0.001
    N = 1200
    wavelet_length = 101
    data_window_length = 301

    r = np.zeros(N)
    spike_idx = np.random.choice(np.arange(100, N - 100), 90, replace=False)
    r[spike_idx] = 0.2 * np.random.randn(len(spike_idx))

    t = (np.arange(wavelet_length) - wavelet_length // 2) * dt
    f0 = 30.0
    w0 = (1.0 - 2.0 * (np.pi * f0 * t) ** 2) * np.exp(-(np.pi * f0 * t) ** 2)
    w0 = w0 - np.mean(w0)

    s = np.convolve(r, w0, mode="same")
    s += 0.02 * np.random.randn(N)

    W, diag = time_varying_wavelet_inversion(
        r_time=r,
        s_obs=s,
        wavelet_length=wavelet_length,
        data_window_length=data_window_length,
        dt=dt,
        mu1=0.2,
        mu2=3.0,
        mu_dc=10.0,
        mu_prior=0.3,
        mu_time=0.5,
        energy_percentile=25.0,
        estimate_step_ms=5.0,
        time_smooth_ms=15.0,
        wavelet_smooth_sigma=1.2,
        return_diagnostics=True,
        verbose=True,
    )

    s_syn = nonstationary_convolution(r, W)
    cc = np.corrcoef(s, s_syn)[0, 1]
    print(f"Self-test CC = {cc:.4f}")

    plt.figure(figsize=(10, 5))
    plt.imshow(W.T, aspect="auto", origin="lower", cmap="seismic")
    plt.title("Robust estimated time-varying wavelet matrix")
    plt.xlabel("Time sample")
    plt.ylabel("Wavelet sample")
    plt.colorbar()
    plt.tight_layout()
    plt.show()
