# -*- coding: utf-8 -*-
"""
core/wavelet_qc.py

子波 QC 指标。
只计算指标，不决定是否 fallback。
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import tukey


def compute_wavelet_qc_attributes(
    W: np.ndarray,
    dt: float,
    wavelet_alignment: str = "center",
) -> dict[str, np.ndarray]:
    """
    按反演模块的原始定义计算时变子波矩阵的基础 QC 属性。

    输入：
    - W: shape (N_time, N_wavelet) 的二维时变子波矩阵；每一行对应一个时间采样点。
    - dt: 时间采样间隔，单位 s，必须为有限正数。
    - wavelet_alignment: ``center`` 或 ``causal``，决定峰值时间的参考原点。

    输出：
    - peak_metric_ms: shape (N_time,)，主峰相对参考原点的位置，单位 ms。
    - energy_l2: shape (N_time,)，每行子波的 L2 范数。
    - energy_l2_norm: shape (N_time,)，相对全时窗最大 L2 范数的归一化值。
    - centroid_frequency_hz: shape (N_time,)，全 RFFT 功率谱质心，单位 Hz。
    - bandwidth_hz: shape (N_time,)，全 RFFT 功率谱标准差，单位 Hz。

    物理与数学约定：不去均值、不加窗、不截取频带、也不沿时间方向平滑，
    从而与时变子波反演诊断中的能量和频谱属性定义保持一致。本函数只计算
    QC，不改变子波、反射系数或正演数据流。
    """
    W = np.asarray(W, dtype=float)
    if W.ndim != 2:
        raise ValueError("W 必须是 shape (N_time, N_wavelet) 的二维子波矩阵。")
    if W.shape[0] == 0 or W.shape[1] == 0:
        raise ValueError("W 的时间维和子波长度均必须大于 0。")
    if not np.all(np.isfinite(W)):
        raise ValueError("W 必须全部为有限值，不能包含 NaN 或 Inf。")

    dt_array = np.asarray(dt, dtype=float)
    if dt_array.ndim != 0 or not np.isfinite(dt_array) or dt_array <= 0.0:
        raise ValueError("dt 必须是有限正数，单位为 s。")
    dt = float(dt_array)

    wavelet_alignment = str(wavelet_alignment).lower()
    if wavelet_alignment not in ("center", "causal"):
        raise ValueError("wavelet_alignment 必须是 'center' 或 'causal'。")

    n_wavelet = W.shape[1]
    peak_index = np.argmax(np.abs(W), axis=1)
    if wavelet_alignment == "center":
        peak_metric_samples = peak_index - n_wavelet // 2
    else:
        peak_metric_samples = peak_index
    peak_metric_ms = peak_metric_samples.astype(float) * dt * 1000.0

    energy_l2 = np.linalg.norm(W, axis=1)
    energy_l2_norm = energy_l2 / (np.max(energy_l2) + 1e-12)
    frequencies_hz = np.fft.rfftfreq(n_wavelet, d=dt)
    power_spectrum = np.abs(np.fft.rfft(W, axis=1)) ** 2
    power_sum = np.sum(power_spectrum, axis=1) + 1e-12
    centroid_frequency_hz = np.sum(
        power_spectrum * frequencies_hz[None, :],
        axis=1,
    ) / power_sum
    bandwidth_hz = np.sqrt(
        np.sum(
            (frequencies_hz[None, :] - centroid_frequency_hz[:, None]) ** 2
            * power_spectrum,
            axis=1,
        )
        / power_sum
    )

    return {
        "peak_metric_ms": peak_metric_ms,
        "energy_l2": energy_l2,
        "energy_l2_norm": energy_l2_norm,
        "centroid_frequency_hz": centroid_frequency_hz,
        "bandwidth_hz": bandwidth_hz,
    }


def compute_wavelet_peak_metric_ms(W, dt, wavelet_alignment="center"):
    """计算时变子波主峰位置指标。

    center: 返回主峰相对中心的偏移 (ms)。
    causal: 返回主峰相对子波起点的位置 (ms)。

    输入:
    - W: 2D array (N, L), 时变子波矩阵。
    - dt: float, 采样间隔 (s)。
    - wavelet_alignment: str, "center" 或 "causal"。

    返回:
    - numpy.ndarray (len = N): 主峰位置指标 (ms)。
    """
    W = np.asarray(W, dtype=float)
    L = W.shape[1]
    peak_idx = np.argmax(np.abs(W), axis=1)

    if wavelet_alignment == "center":
        return (peak_idx - L // 2) * dt * 1000.0
    elif wavelet_alignment == "causal":
        return peak_idx * dt * 1000.0
    else:
        raise ValueError("wavelet_alignment 必须是 center 或 causal")


def compute_boundary_peak_mask(
    W: np.ndarray,
    boundary_fraction: float = 0.12,
) -> np.ndarray:
    """判断每条子波的绝对主峰是否落在首尾边界区。

    输入参数：
        W: 子波矩阵，shape=(N_time, N_wavelet)。
        boundary_fraction: 左、右边界各自占子波长度的比例，必须位于 (0, 0.5)。

    输出结果：
        boundary_peak_mask: shape=(N_time,) 的布尔数组。True 表示绝对主峰
        落在左侧或右侧边界区。

    单位与物理意义：
        本指标只使用离散采样点，无时间单位；它用于识别主峰跑到子波边界的
        灾难性退化，不替代以 ms 表示的 center/causal 峰值位置 QC。
    """
    W = np.asarray(W, dtype=float)
    if W.ndim != 2 or W.shape[0] == 0 or W.shape[1] == 0:
        raise ValueError("W 必须是两个维度均非空的二维数组。")
    if not np.all(np.isfinite(W)):
        raise ValueError("W 包含 NaN 或 Inf。")

    boundary_fraction_value = np.asarray(boundary_fraction, dtype=float)
    if (
        boundary_fraction_value.ndim != 0
        or not np.isfinite(boundary_fraction_value)
        or not 0.0 < float(boundary_fraction_value) < 0.5
    ):
        raise ValueError("boundary_fraction 必须位于 (0, 0.5)。")

    wavelet_length = W.shape[1]
    max_edge_samples = max(1, (wavelet_length - 1) // 2)
    edge_samples = min(
        max_edge_samples,
        max(
            1,
            int(np.ceil(float(boundary_fraction_value) * wavelet_length)),
        ),
    )
    peak_index = np.argmax(np.abs(W), axis=1)
    boundary_peak_mask = (
        (peak_index < edge_samples)
        | (peak_index >= wavelet_length - edge_samples)
    )
    return boundary_peak_mask


def estimate_wavelet_centroid_frequency(W, dt, fmin=5.0, fmax=80.0):
    """估计时变子波矩阵每一行的质心频率（在 fmin-fmax 范围内）。

    输入:
    - W: 2D array (N, L), 时变子波矩阵，每行对应一个时间点的局部子波。
    - dt: float, 采样间隔（秒）。
    - fmin, fmax: float, 频率范围（Hz）。

    返回:
    - numpy.ndarray (len = N): 每个时间点的质心频率 (Hz)。
    """
    W = np.asarray(W, dtype=float)
    N, L = W.shape
    # np.fft.rfftfreq 用于生成实值信号的离散傅里叶变换（RFFT）对应的频率向量
    # 即生成频率域的坐标轴，长度为L，间隔为dt。
    freqs = np.fft.rfftfreq(L, dt)
    mask = (freqs >= fmin) & (freqs <= fmax)
    f_centroid = np.full(N, np.nan)
    for i in range(N):
        w = W[i] - np.nanmean(W[i])
        w = np.nan_to_num(w) * tukey(len(w), alpha=0.2)
        # 功率谱
        power = np.abs(np.fft.rfft(w)) ** 2
        p = power[mask]
        f = freqs[mask]
        if np.sum(p) > 1e-12:
            f_centroid[i] = np.sum(f * p) / np.sum(p)
    good = np.isfinite(f_centroid)
    if np.sum(good) > 2:
        idx = np.arange(N)
        f_centroid[~good] = np.interp(idx[~good], idx[good], f_centroid[good])
        f_centroid = gaussian_filter1d(f_centroid, sigma=5, mode="nearest")
    return f_centroid


def compute_wavelet_energy(W):
    """计算时变子波矩阵每一行的能量及归一化能量。

    输入:
    - W: 2D array (N, L), 时变子波矩阵。

    返回:
    - energy: numpy.ndarray (N,), 每行的能量。
    - energy_norm: numpy.ndarray (N,), 归一化后的能量。
    """
    W = np.asarray(W, dtype=float)
    energy = np.sum(W ** 2, axis=1)
    energy_norm = energy / (np.nanmax(energy) + 1e-12)
    return energy, energy_norm


def diag_get_any(diag, names, default=np.nan):
    """从诊断字典 diag 中按候选键名列表 names 依次取值。

    兼容不同版本输出格式。

    输入:
    - diag: dict 或其他。
    - names: iterable of str, 候选键名。
    - default: 默认值。

    返回:
    - 第一个匹配键的值，或 default。
    """
    if not isinstance(diag, dict):
        return default
    for name in names:
        if name in diag:
            return diag[name]
    return default
