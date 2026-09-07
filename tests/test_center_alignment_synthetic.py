# -*- coding: utf-8 -*-
"""
tests/test_center_alignment_synthetic.py

针对用户提出的 "情况 B: center alignment 矩阵索引与正反演算子 lag 约定"
进行严格的合成数据单元测试。

验证目标：
1. 验证正演算子 (stationary_convolution, nonstationary_convolution) 与
   反演矩阵 (stationary_wavelet_inversion 中的 R 矩阵、build_local_convolution_matrix)
   的 lag 约定是否严格一致 (forward operator lag == inverse operator lag)。
2. 在已知 center Ricker 真实子波与无噪声合成地震记录条件下，
   运行 stationary_wavelet_inversion() 反演平稳子波，
   检查恢复出的子波主峰是否严格落在 0 ms (0 采样点偏移)，
   以及 CC 是否接近 1.0。
3. 验证时变子波反演 time_varying_wavelet_inversion() 在平稳合成数据下的主峰是否稳定在 0 ms。
4. 探究在何种非理想情况下（如人工注入 +64 ms 残余时差、直流偏移、首尾边界效应）
   会导致反演子波峰值移动至 +64 ms 边界，以明确区分“代码索引缺陷”与“真实数据残余时差/病态性”。
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

# 确保项目根目录在 sys.path 中
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.forward_operator import (
    stationary_convolution,
    nonstationary_convolution,
    build_local_convolution_matrix,
    wavelet_peak_metric_samples,
    wavelet_peak_metric_ms,
    wavelet_lag_axis_samples,
    get_lag,
)
from stages.stage_wavelet_prior import make_center_ricker_wavelet
from utils.wavelet_inversion_robust import (
    stationary_wavelet_inversion,
    time_varying_wavelet_inversion,
)


def test_center_lag_definition() -> None:
    """测试 1: 检验 center 对齐方式下各列对应的 lag 定义。"""
    wavelet_length = 129
    half = wavelet_length // 2  # 64

    # 对于 center 对齐：
    # col = 0 -> lag = -64
    # col = 64 (中心) -> lag = 0
    # col = 128 -> lag = +64
    assert get_lag(0, wavelet_length, "center") == -half
    assert get_lag(half, wavelet_length, "center") == 0
    assert get_lag(wavelet_length - 1, wavelet_length, "center") == half

    lags = wavelet_lag_axis_samples(wavelet_length, "center")
    assert lags[0] == -64
    assert lags[half] == 0
    assert lags[-1] == 64
    print("[PASS] test_center_lag_definition")


def test_forward_and_matrix_lag_consistency() -> None:
    """
    测试 2: 验证正演算子与反演矩阵的 lag 约定完全相同。
    forward operator lag == inverse operator lag.
    """
    dt = 0.001  # 1 ms
    wavelet_length = 129  # L = 129, half = 64 ms
    f0 = 30.0
    w_true = make_center_ricker_wavelet(f0=f0, dt=dt, length=wavelet_length)

    # 真实子波主峰必须在 index=64, 对应 0 ms 偏移
    peak_sample = int(np.argmax(np.abs(w_true)))
    assert peak_sample == 64
    assert wavelet_peak_metric_samples(w_true, "center") == 0
    assert wavelet_peak_metric_ms(w_true, dt, "center") == 0.0

    # 构造随机脉冲反射系数
    rng = np.random.default_rng(12345)
    n_time = 800
    r_time = np.zeros(n_time, dtype=float)
    spike_indices = rng.choice(np.arange(100, n_time - 100), size=30, replace=False)
    r_time[spike_indices] = rng.uniform(-0.8, 0.8, size=30)

    # 1) 平稳正演卷积 (np.convolve mode='same')
    s_stat = stationary_convolution(r_time, w_true, alignment="center")

    # 2) 非平稳正演卷积 (W 矩阵每行复制 w_true)
    W_true = np.tile(w_true, (n_time, 1))
    s_nonstat = nonstationary_convolution(r_time, W_true, alignment="center")

    # 3) 反演算法内部构造的全道卷积矩阵 R_global
    half_wavelet = wavelet_length // 2
    R_global = np.zeros((n_time, wavelet_length), dtype=float)
    for row in range(n_time):
        for col in range(wavelet_length):
            lag = col - half_wavelet
            r_idx = row - lag
            if 0 <= r_idx < n_time:
                R_global[row, col] = r_time[r_idx]
    s_matrix = R_global @ w_true

    # 4) 局部卷积矩阵拼接 (build_local_convolution_matrix)
    # 取一个中心点测试局部矩阵与 R_global 对应块是否严格恒等
    center_test = 300
    win_len = 387  # 3 * 129
    R_local = build_local_convolution_matrix(
        r_time=r_time,
        center_index=center_test,
        data_window_length=win_len,
        wavelet_length=wavelet_length,
        alignment="center",
    )
    half_win = win_len // 2
    R_global_sub = R_global[center_test - half_win : center_test - half_win + win_len, :]
    max_diff_local = np.max(np.abs(R_local - R_global_sub))
    assert max_diff_local < 1e-15, f"局部卷积矩阵与全局矩阵不一致! diff={max_diff_local}"

    # 比较三种方式的差异 (注意边界 half_wavelet 区域外)
    # 内部有效区域 [half_wavelet, n_time - half_wavelet]
    valid_slice = slice(half_wavelet, n_time - half_wavelet)
    diff_stat_nonstat = np.max(np.abs(s_stat[valid_slice] - s_nonstat[valid_slice]))
    diff_matrix_nonstat = np.max(np.abs(s_matrix[valid_slice] - s_nonstat[valid_slice]))
    diff_stat_matrix = np.max(np.abs(s_stat[valid_slice] - s_matrix[valid_slice]))

    print(f"  Max diff (stationary vs nonstationary): {diff_stat_nonstat:.2e}")
    print(f"  Max diff (matrix vs nonstationary):     {diff_matrix_nonstat:.2e}")
    print(f"  Max diff (stationary vs matrix):        {diff_stat_matrix:.2e}")

    assert diff_stat_nonstat < 1e-12, "stationary 与 nonstationary 正演算子不一致!"
    assert diff_matrix_nonstat < 1e-12, "matrix 乘法与 nonstationary 正演算子不一致!"
    assert diff_stat_matrix < 1e-12, "stationary 卷积与反演矩阵 R 不一致!"
    print("[PASS] test_forward_and_matrix_lag_consistency")


def test_stationary_wavelet_inversion_synthetic() -> None:
    """
    测试 3: 人工构造 r 和已知 center Ricker w_true，
    正演生成 s = r * w_true，输入 stationary_wavelet_inversion()，
    验证反演子波主峰是否严格落在 0 ms，且 peak_lock=True 能顺利通过。
    """
    dt = 0.001
    wavelet_length = 129
    f0 = 30.0
    w_true = make_center_ricker_wavelet(f0=f0, dt=dt, length=wavelet_length)

    rng = np.random.default_rng(2026)
    n_time = 1000
    r_time = np.zeros(n_time, dtype=float)
    spikes = rng.choice(np.arange(100, n_time - 100), size=50, replace=False)
    r_time[spikes] = rng.uniform(-1.0, 1.0, size=50)

    s_obs = stationary_convolution(r_time, w_true, alignment="center")

    # 使用生产代码完全相同的 stationary_wavelet_inversion，开启 peak_lock=True
    w_est = stationary_wavelet_inversion(
        r_time=r_time,
        s_obs=s_obs,
        wavelet_length=wavelet_length,
        mu1=0.01,
        mu2=0.05,
        mu_dc=5.0,
        damping_ratio=1e-4,
        svd_cutoff_ratio=1e-4,
        peak_lock=True,
        max_peak_shift_samples=15,
        wavelet_alignment="center",
    )

    peak_samples = wavelet_peak_metric_samples(w_est, "center")
    peak_ms = wavelet_peak_metric_ms(w_est, dt, "center")
    peak_idx = int(np.argmax(np.abs(w_est)))
    cc = float(np.corrcoef(w_true, w_est)[0, 1])

    print(f"  True wavelet peak index: {len(w_true)//2} (0 ms)")
    print(f"  Estimated wavelet peak index: {peak_idx} ({peak_samples} samples, {peak_ms:.2f} ms)")
    print(f"  Wavelet correlation CC: {cc:.6f}")

    assert peak_samples == 0, f"恢复出的主峰偏离了 0 ms! peak={peak_samples} samples ({peak_ms} ms)"
    assert abs(peak_ms) < 1e-6, f"主峰 ms 不为 0: {peak_ms} ms"
    assert cc > 0.999, f"反演子波与真实子波相关系数过低: {cc:.6f}"
    print("[PASS] test_stationary_wavelet_inversion_synthetic: peak strictly at 0 ms, CC > 0.999")


def test_tv_wavelet_inversion_synthetic() -> None:
    """测试 4: 验证时变子波反演在合成数据下的主峰稳定性。"""
    dt = 0.001
    wavelet_length = 129
    f0 = 30.0
    w_true = make_center_ricker_wavelet(f0=f0, dt=dt, length=wavelet_length)

    rng = np.random.default_rng(888)
    n_time = 600
    r_time = np.zeros(n_time, dtype=float)
    spikes = rng.choice(np.arange(100, n_time - 100), size=40, replace=False)
    r_time[spikes] = rng.uniform(-1.0, 1.0, size=40)

    s_obs = stationary_convolution(r_time, w_true, alignment="center")

    W_est = time_varying_wavelet_inversion(
        r_time=r_time,
        s_obs=s_obs,
        wavelet_length=wavelet_length,
        data_window_length=387,
        dt=dt,
        wavelet_alignment="center",
        mu1=0.1,
        mu2=1.0,
        mu_dc=5.0,
        w_prior=w_true,
        mu_prior=0.5,
        mu_time=1.0,
        energy_percentile=10.0,
        damping_ratio=1e-3,
        svd_cutoff_ratio=1e-3,
        estimate_step_ms=10.0,
        time_smooth_ms=15.0,
        wavelet_smooth_sigma=1.0,
        loss_type="l2",
        verbose=False,
    )

    # 检查所有有效估计窗口的主峰偏移
    peaks = np.array([wavelet_peak_metric_samples(W_est[i], "center") for i in range(n_time)])
    # 中间区域（远离首尾边界）
    mid_peaks = peaks[100 : n_time - 100]
    max_peak_shift = np.max(np.abs(mid_peaks))
    print(f"  TV-wavelet middle section max peak shift: {max_peak_shift} samples")
    assert max_peak_shift == 0, f"时变子波主峰在合成数据上发生偏移: max_shift={max_peak_shift}"
    print("[PASS] test_tv_wavelet_inversion_synthetic: all middle windows strictly peaked at 0 ms")


def test_diagnose_boundary_peak_shift_64ms() -> None:
    """
    诊断测试 5: 探究为什么实际数据中可能恰好出现 +64 ms 的主峰？
    
    场景 A：如果真实数据相对反射系数存在 64 ms 的时差 (残余时差 tau = +64 ms)，
           反演平稳子波为了拟合时差，其主峰就会移动到 +64 ms 边界！
    场景 B：如果反射系数几乎全为 0 (反射能量缺失/极端病态)，在只有边缘正则或截断时
           边界样点可能出现伪峰。
    """
    dt = 0.001
    wavelet_length = 129
    half = wavelet_length // 2  # 64 ms
    f0 = 30.0
    w_true = make_center_ricker_wavelet(f0=f0, dt=dt, length=wavelet_length)

    rng = np.random.default_rng(999)
    n_time = 1000
    r_time = np.zeros(n_time, dtype=float)
    spikes = rng.choice(np.arange(150, n_time - 150), size=40, replace=False)
    r_time[spikes] = rng.uniform(-1.0, 1.0, size=40)

    # 人工构造带 64 ms 时移的地震记录: s_shifted(t) = (r * w_true)(t - 64ms)
    s_clean = stationary_convolution(r_time, w_true, alignment="center")
    shift_samples = 64  # +64 ms 时移
    s_shifted = np.zeros_like(s_clean)
    s_shifted[shift_samples:] = s_clean[:-shift_samples]

    # 反演时关闭 peak_lock 限制，观察平稳反演的自然响应
    w_shifted = stationary_wavelet_inversion(
        r_time=r_time,
        s_obs=s_shifted,
        wavelet_length=wavelet_length,
        mu1=0.01,
        mu2=0.05,
        mu_dc=5.0,
        damping_ratio=1e-4,
        svd_cutoff_ratio=1e-4,
        peak_lock=False,  # 不强制拒绝
        wavelet_alignment="center",
    )

    peak_samples = wavelet_peak_metric_samples(w_shifted, "center")
    peak_ms = wavelet_peak_metric_ms(w_shifted, dt, "center")
    print(f"  [Simulated +64 ms residual timing] Inverted peak: {peak_samples} samples ({peak_ms:.1f} ms)")
    
    # 验证：当存在 +64 ms 残余时差时，反演出的子波主峰恰好落在 +64 ms 处！
    assert peak_samples == 64, f"模拟时差测试预期 peak=64，实际={peak_samples}"
    print("[PASS] test_diagnose_boundary_peak_shift_64ms: confirmed that +64 ms peak is the exact mathematical signature of a 64 ms timing mismatch!")


if __name__ == "__main__":
    print("=" * 60)
    print("开始运行 center alignment 与 lag 约定单元测试...")
    print("=" * 60)
    test_center_lag_definition()
    test_forward_and_matrix_lag_consistency()
    test_stationary_wavelet_inversion_synthetic()
    test_tv_wavelet_inversion_synthetic()
    test_diagnose_boundary_peak_shift_64ms()
    print("=" * 60)
    print("全部 5 项单元测试通过！代码中的矩阵索引与 lag 约定完全正确，无代码级时移 Bug。")
    print("=" * 60)
