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

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d


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

def _local_convolution_matrix(
    r_time: np.ndarray,
    center_index: int,
    data_window_length: int,
    wavelet_length: int,
    wavelet_alignment: str = "center",
) -> np.ndarray:
    """
    构造局部卷积矩阵 R，使得：

        s_window ≈ R @ w

    wavelet_alignment:
    - "center": 零相位/中心子波
          lag = col - half_wavelet
          s[t] = sum r[t - lag] w[col]

    - "causal": 因果/近似最小相位子波
          lag = col
          s[t] = r[t] w[0] + r[t-1] w[1] + ...
    """
    wavelet_alignment = _validate_wavelet_alignment(wavelet_alignment)

    N = len(r_time)
    half_data = data_window_length // 2
    half_wavelet = wavelet_length // 2

    R = np.zeros((data_window_length, wavelet_length), dtype=float)

    for row in range(data_window_length):
        global_t = center_index - half_data + row

        for col in range(wavelet_length):
            if wavelet_alignment == "center":
                lag = col - half_wavelet
            else:
                lag = col

            r_index = global_t - lag

            if 0 <= r_index < N:
                R[row, col] = r_time[r_index]

    return R

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
    S_inv[valid] = S[valid] / (S[valid] ** 2 + damping ** 2)

    return Vt.T @ (S_inv * (U.T @ b))

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

    if wavelet_alignment == "center":
        return _peak_shift_samples(w)
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

    A_aug = np.vstack([
        R,
        np.sqrt(mu1 * data_scale) * I,
        np.sqrt(mu2 * data_scale) * D,
        np.sqrt(mu_dc * data_scale) * C_dc,
    ])

    b_aug = np.concatenate([
        s_obs,
        np.zeros(wavelet_length),
        np.zeros(wavelet_length - 2),
        np.zeros(1),
    ])

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
    # 输出控制
    return_diagnostics: bool = False,
    verbose: bool = True,
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

    N = len(r_time)
    wavelet_length = _make_odd(wavelet_length)

    if data_window_length is None:
        # 真实资料建议比 2L+1 更长一点，降低自由度；这里给保守默认。
        data_window_length = max(2 * wavelet_length + 1, 3 * wavelet_length)

    data_window_length = _make_odd(data_window_length)

    if data_window_length <= wavelet_length:
        raise ValueError("data_window_length 必须大于 wavelet_length。")
    if N < data_window_length:
        raise ValueError("N 小于 data_window_length，请缩短数据窗或检查输入。")

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

    if min_effective_rank is None:
        min_effective_rank = max(10, wavelet_length // 8)

    previous_valid_wavelet = None

    # 诊断数组
    diag_effective_rank = np.full(N, np.nan)
    diag_condition_ratio = np.full(N, np.nan)
    diag_peak_shift = np.full(N, np.nan)
    diag_r_energy = np.full(N, np.nan)
    diag_s_energy = np.full(N, np.nan)
    diag_skip_code = np.zeros(N, dtype=int)
    # skip_code: 0 未尝试；1 有效；-1低反射能量；-2病态矩阵；-3NaN；-4峰值过远；-5振幅跳变

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
        print(f"  energy_threshold      = {energy_threshold:.6e}")
        print(f"  damping_ratio         = {damping_ratio}")
        print(f"  svd_cutoff_ratio      = {svd_cutoff_ratio}")
        print(f"  min_effective_rank    = {min_effective_rank}")
        print(f"  max_peak_shift        = {max_peak_shift_samples} samples")
        print(f"  time_smooth_sigma     = {time_smooth_sigma:.2f} samples")
        print(f"  wavelet_smooth_sigma  = {wavelet_smooth_sigma:.2f} samples")
        print(f"  stationary_prior      = {w_prior is not None}")

    for i in range(half_data, N - half_data, estimate_step_samples):
        r_win = _extract_window(r_time, i, data_window_length)
        s_win = _extract_window(s_obs, i, data_window_length)

        r_energy = float(np.sum(r_win ** 2))
        s_energy = float(np.sum(s_win ** 2))
        diag_r_energy[i] = r_energy
        diag_s_energy[i] = s_energy

        if r_energy < energy_threshold:
            diag_skip_code[i] = -1
            continue

        R = _local_convolution_matrix(
            r_time=r_time,
            center_index=i,
            data_window_length=data_window_length,
            wavelet_length=wavelet_length,
            wavelet_alignment=wavelet_alignment,
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

        A_blocks = [
            R,
            np.sqrt(mu1 * data_scale) * I,
            np.sqrt(mu2 * data_scale) * D,
            np.sqrt(mu_dc * data_scale) * C_dc,
        ]

        b_blocks = [
            s_win,
            np.zeros(wavelet_length),
            np.zeros(wavelet_length - 2),
            np.zeros(1),
        ]
    # 当先验子波存在时，添加先验子波的约束
        prior_i = _get_prior_wavelet(w_prior, i, wavelet_length) if w_prior is not None else None
        if prior_i is not None and mu_prior > 0:
            A_blocks.append(np.sqrt(mu_prior * data_scale) * I)
            b_blocks.append(np.sqrt(mu_prior * data_scale) * prior_i)

        if previous_valid_wavelet is not None and mu_time > 0:
            A_blocks.append(np.sqrt(mu_time * data_scale) * I)
            b_blocks.append(np.sqrt(mu_time * data_scale) * previous_valid_wavelet)

        A_aug = np.vstack(A_blocks)
        b_aug = np.concatenate(b_blocks) # 弄成一维矩阵

        w_i = _safe_svd_solve(
            A=A_aug,
            b=b_aug,
            damping_ratio=damping_ratio,
            svd_cutoff_ratio=svd_cutoff_ratio,
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
                peak_allowed_samples=peak_allowed_samples,
            )
            diag_peak_shift[i] = peak_metric
            if w_checked is None:
                diag_skip_code[i] = -4
                continue
            w_i = w_checked
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

    # 用线性插值填充稀疏估计点；若一个有效点都没有，则退回 w_prior。
    W = _fill_unestimated_wavelets(W, valid_mask, w_prior=w_prior)

    # 反演后平滑。
    if time_smooth_sigma is not None and wavelet_smooth_sigma is not None:
        if time_smooth_sigma > 0 or wavelet_smooth_sigma > 0:
            W = gaussian_filter(
                W,
                sigma=(float(time_smooth_sigma), float(wavelet_smooth_sigma)),
                mode="nearest",
            )

    # 平滑后再次去直流，防止二维滤波引入极小 DC。
    W = W - np.mean(W, axis=1, keepdims=True)

    qc = _wavelet_qc_attributes(
        W,
        dt=dt,
        wavelet_alignment=wavelet_alignment,
    )

    diagnostics: dict[str, np.ndarray | int | float | str | tuple[int, int] | None] = {
        "valid_mask": valid_mask,                      # 布尔掩码 (N,)：指示哪些时间点直接成功反演了子波
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
        print(f"  valid inverted windows = {n_valid} / {N}")
        print(f"  skipped low energy     = {n_low_energy}")
        print(f"  skipped ill-cond       = {n_ill}")
        print(f"  skipped NaN/Inf        = {n_nan}")
        print(f"  skipped peak shift     = {n_peak}")
        print(f"  skipped amp jump       = {n_amp}")
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
# 非平稳褶积工具：用于用 W 生成合成记录
# =========================================================
def nonstationary_convolution(
    r_time: np.ndarray,
    W: np.ndarray,
    wavelet_alignment: str = "center",
) -> np.ndarray:
    """
    用时变子波矩阵 W 对时间域反射系数做非平稳褶积。

    wavelet_alignment:
    - "center":
          lag = col - half
          s[i] = sum r[i - lag] * W[i, col]

    - "causal":
          lag = col
          s[i] = r[i] W[i,0] + r[i-1] W[i,1] + ...
    """
    wavelet_alignment = _validate_wavelet_alignment(wavelet_alignment)

    r_time = np.asarray(r_time, dtype=float)
    W = np.asarray(W, dtype=float)

    if W.ndim != 2:
        raise ValueError("W 必须是二维矩阵，shape=(N, L)。")
    if len(r_time) != W.shape[0]:
        raise ValueError("r_time 长度必须等于 W.shape[0]。")

    N, L = W.shape
    half = L // 2
    s = np.zeros(N, dtype=float)

    for i in range(N):
        acc = 0.0

        for col in range(L):
            if wavelet_alignment == "center":
                lag = col - half
            else:
                lag = col

            idx = i - lag

            if 0 <= idx < N:
                acc += r_time[idx] * W[i, col]

        s[i] = acc

    return s


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
