# -*- coding: utf-8 -*-
"""
wavelet_inversion.py

Time-varying wavelet inversion for nonstationary seismic-well tying.

设计目标：
1. 使用比子波更长的数据窗，形成超定最小二乘问题；
2. 对低反射能量窗口自动跳过，避免反演出纯噪声子波；
3. 使用 Tikhonov + 二阶差分平滑正则化；
4. 使用 SVD / damped least squares 稳定求解；
5. 保持相邻时变子波极性连续；
6. 不对每个子波做最大值归一化，从而保留真实振幅衰减趋势；
7. 输出的 W_estimate 可继续送入 Q 估计与 Q 约束重构模块。

推荐调用方式：
    W_estimate = time_varying_wavelet_inversion(
        r_time=r_time_k,
        s_obs=s_obs_noisy,
        wavelet_length=wavelet_length_pts,
        data_window_length=2 * wavelet_length_pts + 1,
        mu1=0.05,
        mu2=0.5,
        energy_percentile=20,
        time_smooth_sigma=3.0,
        wavelet_smooth_sigma=0.8
    )
"""

import numpy as np
from scipy.ndimage import gaussian_filter


def _make_odd(n: int) -> int:
    """确保窗口长度为奇数。"""
    n = int(n)
    return n if n % 2 == 1 else n + 1


def _second_derivative_matrix(n: int) -> np.ndarray:
    """
    构造二阶差分矩阵 D。

    D @ w 约等于子波的二阶导数，用于抑制不必要的高频振荡。
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
    wavelet_length: int
) -> np.ndarray:
    """
    构造局部卷积矩阵 R，使得：

        s_window ≈ R @ w

    这里的含义是：
    对于局部地震窗中的每一个输出采样点 t_out，
    它由附近 reflectivity 与子波 w 的加权和构成。

    参数：
        r_time:
            时间域反射系数，长度 N。
        center_index:
            当前要反演子波的中心时间样点。
        data_window_length:
            局部地震数据窗长度，应该大于 wavelet_length。
        wavelet_length:
            子波长度。

    返回：
        R:
            shape = (data_window_length, wavelet_length)
    """
    N = len(r_time)

    half_data = data_window_length // 2
    half_wavelet = wavelet_length // 2

    R = np.zeros((data_window_length, wavelet_length), dtype=float)

    for row in range(data_window_length):
        # 当前局部地震窗中的输出点，对应全局地震样点
        global_t = center_index - half_data + row

        for col in range(wavelet_length):
            # 子波 col 对应相对中心的时间滞后
            lag = col - half_wavelet

            # 卷积关系：s[t] = sum_lag r[t - lag] * w[lag]
            r_index = global_t - lag

            if 0 <= r_index < N:
                R[row, col] = r_time[r_index]

    return R


def _extract_window(
    x: np.ndarray,
    center_index: int,
    window_length: int,
    fill_value: float = 0.0
) -> np.ndarray:
    """
    从一维数组 x 中提取以 center_index 为中心的窗口。
    超出边界的位置用 fill_value 填充。
    """
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
    svd_cutoff_ratio: float = 1e-4
) -> np.ndarray:
    """
    使用阻尼 SVD 求解最小二乘问题。

    解：
        min ||A x - b||_2

    阻尼形式：
        x = V diag(s / (s^2 + eps^2)) U.T b

    参数：
        damping_ratio:
            阻尼强度，相对于最大奇异值。
        svd_cutoff_ratio:
            小于 max_s * svd_cutoff_ratio 的奇异值直接丢弃。
    """
    U, S, Vt = np.linalg.svd(A, full_matrices=False)

    if len(S) == 0:
        return np.zeros(A.shape[1], dtype=float)

    max_s = np.max(S)

    if max_s <= 0:
        return np.zeros(A.shape[1], dtype=float)

    damping = damping_ratio * max_s
    cutoff = svd_cutoff_ratio * max_s

    S_inv = np.zeros_like(S)

    valid = S > cutoff
    S_inv[valid] = S[valid] / (S[valid] ** 2 + damping ** 2)

    x = Vt.T @ (S_inv * (U.T @ b))

    return x


def _estimate_reflectivity_energy_threshold(
    r_time: np.ndarray,
    data_window_length: int,
    energy_percentile: float = 20.0
) -> float:
    """
    根据所有局部反射能量的分位数，估计低能量窗口阈值。

    energy_percentile 越高，跳过的窗口越多。
    建议：
        合成数据：10 ~ 20
        含噪真实数据：20 ~ 40
    """
    N = len(r_time)
    half = data_window_length // 2

    energies = []

    step = max(1, data_window_length // 10)

    for i in range(half, N - half, step):
        rw = _extract_window(r_time, i, data_window_length)
        energies.append(np.sum(rw ** 2))

    energies = np.asarray(energies, dtype=float)

    if len(energies) == 0:
        return 0.0

    positive = energies[energies > 0]

    if len(positive) == 0:
        return 0.0

    threshold = np.percentile(positive, energy_percentile)

    return float(threshold)


def _fill_unestimated_wavelets(
    W: np.ndarray,
    valid_mask: np.ndarray
) -> np.ndarray:
    """
    对没有成功反演的时间位置，用最近的有效子波填充。

    处理顺序：
    1. 向前填充；
    2. 向后填充；
    3. 如果全都无效，则保持零矩阵。
    """
    W = W.copy()
    N = W.shape[0]

    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) == 0:
        return W

    first_valid = valid_indices[0]
    last_valid = valid_indices[-1]

    # 顶部填充
    for i in range(0, first_valid):
        W[i, :] = W[first_valid, :]

    # 中间向前填充
    last = first_valid
    for i in range(first_valid + 1, N):
        if valid_mask[i]:
            last = i
        else:
            W[i, :] = W[last, :]

    # 底部已经被向前填充，这里保险处理
    for i in range(last_valid + 1, N):
        W[i, :] = W[last_valid, :]

    return W


def time_varying_wavelet_inversion(
    r_time: np.ndarray,
    s_obs: np.ndarray,
    wavelet_length: int,
    data_window_length: int | None = None,
    mu1: float = 0.05,
    mu2: float = 0.5,
    energy_threshold: float | None = None,
    energy_percentile: float = 20.0,
    damping_ratio: float = 1e-3,
    svd_cutoff_ratio: float = 1e-4,
    time_smooth_sigma: float = 3.0,
    wavelet_smooth_sigma: float = 0.8,
    polarity_continuity: bool = True,
    verbose: bool = True
) -> np.ndarray:
    """
    反演时变子波矩阵 W。

    参数：
        r_time:
            当前时深关系下的时间域反射系数，shape = (N,)
        s_obs:
            观测地震道，shape = (N,)
        wavelet_length:
            子波长度，建议为奇数，例如 101、151、201。
        data_window_length:
            局部数据窗长度，必须大于 wavelet_length。
            如果为 None，默认使用 2 * wavelet_length + 1。
        mu1:
            子波能量正则化系数，对应 ||w||^2。
        mu2:
            子波二阶导平滑正则化系数，对应 ||D w||^2。
        energy_threshold:
            低反射能量阈值。
            如果为 None，则根据 energy_percentile 自动估计。
        energy_percentile:
            自动估计低能量阈值所用分位数。
        damping_ratio:
            SVD 阻尼强度。
        svd_cutoff_ratio:
            SVD 小奇异值截断比例。
        time_smooth_sigma:
            反演后沿时间方向平滑 W 的 sigma。
            设为 0 或 None 则不平滑。
        wavelet_smooth_sigma:
            反演后沿子波内部方向平滑 W 的 sigma。
            设为 0 或 None 则不平滑。
        polarity_continuity:
            是否强制相邻子波极性连续。
        verbose:
            是否打印诊断信息。

    返回：
        W:
            时变子波矩阵，shape = (N, wavelet_length)。
            W[i, :] 表示中心位于第 i 个时间采样点的局部子波。
    """
    r_time = np.asarray(r_time, dtype=float).copy()
    s_obs = np.asarray(s_obs, dtype=float).copy()

    if r_time.ndim != 1 or s_obs.ndim != 1:
        raise ValueError("r_time 和 s_obs 必须是一维数组。")

    if len(r_time) != len(s_obs):
        raise ValueError("r_time 和 s_obs 长度必须一致。")

    N = len(r_time)

    wavelet_length = _make_odd(wavelet_length)

    if data_window_length is None:
        data_window_length = 2 * wavelet_length + 1

    data_window_length = _make_odd(data_window_length)

    if data_window_length <= wavelet_length:
        raise ValueError(
            "data_window_length 必须大于 wavelet_length，"
            "否则局部反演不是稳定的超定问题。"
        )

    if N < data_window_length:
        raise ValueError(
            "地震道长度 N 小于 data_window_length，"
            "请缩短 data_window_length 或检查输入数据。"
        )

    # 去掉极小直流，避免局部反演吸收 DC 漂移
    s_obs = s_obs - np.mean(s_obs)

    D = _second_derivative_matrix(wavelet_length)
    I = np.eye(wavelet_length, dtype=float)

    W = np.zeros((N, wavelet_length), dtype=float)
    valid_mask = np.zeros(N, dtype=bool)

    if energy_threshold is None:
        energy_threshold = _estimate_reflectivity_energy_threshold(
            r_time=r_time,
            data_window_length=data_window_length,
            energy_percentile=energy_percentile
        )

    half_data = data_window_length // 2

    previous_valid_wavelet = None

    if verbose:
        print("========== Time-varying wavelet inversion ==========")
        print(f"  N_time              = {N}")
        print(f"  wavelet_length      = {wavelet_length}")
        print(f"  data_window_length  = {data_window_length}")
        print(f"  mu1                 = {mu1}")
        print(f"  mu2                 = {mu2}")
        print(f"  energy_threshold    = {energy_threshold:.6e}")
        print(f"  damping_ratio       = {damping_ratio}")
        print(f"  svd_cutoff_ratio    = {svd_cutoff_ratio}")
    estimate_step_samples = max(1, int(round(0.005 / dt))) 
    for i in range(half_data, N - half_data, estimate_step_samples):
        r_win = _extract_window(r_time, i, data_window_length)
        s_win = _extract_window(s_obs, i, data_window_length)

        r_energy = np.sum(r_win ** 2)
        s_energy = np.sum(s_win ** 2)

        # 低反射能量窗口：不反演，直接沿用上一个可靠子波
        if r_energy < energy_threshold:
            if previous_valid_wavelet is not None:
                W[i, :] = previous_valid_wavelet
            continue

        R = _local_convolution_matrix(
            r_time=r_time,
            center_index=i,
            data_window_length=data_window_length,
            wavelet_length=wavelet_length
        )

        # 正则化尺度不要只跟 s_energy 走，否则弱能量窗会正则过弱
        data_scale = max(s_energy / data_window_length, 1e-12)

        A_aug = np.vstack([
            R,
            np.sqrt(mu1 * data_scale) * I,
            np.sqrt(mu2 * data_scale) * D
        ])

        b_aug = np.concatenate([
            s_win,
            np.zeros(wavelet_length),
            np.zeros(wavelet_length - 2)
        ])

        w_i = _safe_svd_solve(
            A=A_aug,
            b=b_aug,
            damping_ratio=damping_ratio,
            svd_cutoff_ratio=svd_cutoff_ratio
        )

        # 防止 NaN / inf 污染整个 W
        if not np.all(np.isfinite(w_i)):
            if previous_valid_wavelet is not None:
                W[i, :] = previous_valid_wavelet
            continue

        # 极性连续性：避免相邻窗口解空间跳变
        if polarity_continuity and previous_valid_wavelet is not None:
            if np.dot(w_i, previous_valid_wavelet) < 0:
                w_i = -w_i

        W[i, :] = w_i
        valid_mask[i] = True
        previous_valid_wavelet = w_i.copy()

    # 对没有估计出的窗口做最近有效子波填充
    W = _fill_unestimated_wavelets(W, valid_mask)

    # 轻微二维平滑：时间方向平滑大于子波内部方向平滑
    # 注意：这里不是最大值归一化，所以仍然保留振幅衰减趋势
    if time_smooth_sigma is not None and wavelet_smooth_sigma is not None:
        if time_smooth_sigma > 0 or wavelet_smooth_sigma > 0:
            W = gaussian_filter(
                W,
                sigma=(float(time_smooth_sigma), float(wavelet_smooth_sigma)),
                mode="nearest"
            )

    if verbose:
        n_valid = int(np.sum(valid_mask))
        n_attempted = len(range(half_data, N - half_data, estimate_step_samples))
        print(f"  attempted windows      = {n_attempted}")
        print(f"  valid inverted centers = {n_valid} / {n_attempted}")
        print("====================================================")

    return W


def stationary_wavelet_inversion(
    r_time: np.ndarray,
    s_obs: np.ndarray,
    wavelet_length: int,
    mu1: float = 0.05,
    mu2: float = 0.5,
    damping_ratio: float = 1e-3,
    svd_cutoff_ratio: float = 1e-4
) -> np.ndarray:
    """
    可选：反演一个全局平稳子波。

    用途：
    1. 作为初始子波诊断；
    2. 当时变子波反演不稳定时，用全局子波填充；
    3. 对比 stationary tying 与 nonstationary tying。

    返回：
        w:
            shape = (wavelet_length,)
    """
    r_time = np.asarray(r_time, dtype=float)
    s_obs = np.asarray(s_obs, dtype=float)

    if len(r_time) != len(s_obs):
        raise ValueError("r_time 和 s_obs 长度必须一致。")

    N = len(r_time)
    wavelet_length = _make_odd(wavelet_length)

    R = np.zeros((N, wavelet_length), dtype=float)
    half_wavelet = wavelet_length // 2

    for row in range(N):
        for col in range(wavelet_length):
            lag = col - half_wavelet
            r_index = row - lag
            if 0 <= r_index < N:
                R[row, col] = r_time[r_index]

    D = _second_derivative_matrix(wavelet_length)
    I = np.eye(wavelet_length)

    s_obs = s_obs - np.mean(s_obs)
    data_scale = max(np.sum(s_obs ** 2) / N, 1e-12)

    A_aug = np.vstack([
        R,
        np.sqrt(mu1 * data_scale) * I,
        np.sqrt(mu2 * data_scale) * D
    ])

    b_aug = np.concatenate([
        s_obs,
        np.zeros(wavelet_length),
        np.zeros(wavelet_length - 2)
    ])

    w = _safe_svd_solve(
        A=A_aug,
        b=b_aug,
        damping_ratio=damping_ratio,
        svd_cutoff_ratio=svd_cutoff_ratio
    )

    return w


if __name__ == "__main__":
    # 简单自测，不依赖你的主程序
    import matplotlib.pyplot as plt

    np.random.seed(123)

    dt = 0.001
    N = 1200
    wavelet_length = 101
    data_window_length = 301

    # 构造稀疏反射系数
    r = np.zeros(N)
    spike_idx = np.random.choice(np.arange(100, N - 100), 80, replace=False)
    r[spike_idx] = 0.2 * np.random.randn(len(spike_idx))

    # 构造一个简单 Ricker 子波
    t = (np.arange(wavelet_length) - wavelet_length // 2) * dt
    f0 = 30.0
    w0 = (1.0 - 2.0 * (np.pi * f0 * t) ** 2) * np.exp(-(np.pi * f0 * t) ** 2)

    # 简单平稳合成记录
    s = np.convolve(r, w0, mode="same")
    s += 0.02 * np.random.randn(N)

    W = time_varying_wavelet_inversion(
        r_time=r,
        s_obs=s,
        wavelet_length=wavelet_length,
        data_window_length=data_window_length,
        mu1=0.05,
        mu2=0.5,
        energy_percentile=20,
        time_smooth_sigma=3.0,
        wavelet_smooth_sigma=0.8,
        verbose=True
    )

    plt.figure(figsize=(10, 5))
    plt.imshow(W.T, aspect="auto", origin="lower", cmap="seismic")
    plt.title("Estimated time-varying wavelet matrix")
    plt.xlabel("Time sample")
    plt.ylabel("Wavelet sample")
    plt.colorbar()
    plt.tight_layout()
    plt.show()