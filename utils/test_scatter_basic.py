# -*- coding: utf-8 -*-
"""
Geophysical diagnostic script to analyze if reflectivity energy dampening
is caused by depth-to-time domain projection (scattering).
判断是否是因为采样粗导致的误差
"""

import sys
from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import windows

# Fix import paths
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.data_preprocess import scatter_reflectivity_to_time

# =====================================================================
# 1. Test 1: Basic Math & Index Verification
# =====================================================================
def test_scatter_basic():
    print("=" * 60)
    print("TEST 1: BASIC SCATTER MATH VERIFICATION")
    print("=" * 60)
    dt = 0.001
    t_ref = np.arange(0.0, 0.011, dt)

    # 情况 1：一个反射点刚好落在 3 ms
    r_depth = np.zeros(4)
    r_depth[0] = 1.0
    twt_depth = np.array([0.000, 0.003, 0.006, 0.009])

    r_time = scatter_reflectivity_to_time(r_depth, twt_depth, t_ref)

    print("[Test 1.1] Exact sample boundary alignment")
    print("  r_time =", r_time)
    print("  expected: r_time[3] should be close to 1.0")
    assert np.allclose(r_time[3], 1.0), f"Exact alignment failed, got {r_time[3]}"

    # 情况 2：一个反射点落在 3.5 ms，应分到 3 ms 和 4 ms
    r_depth = np.zeros(4)
    r_depth[0] = 1.0
    twt_depth = np.array([0.000, 0.0035, 0.006, 0.009])

    r_time = scatter_reflectivity_to_time(r_depth, twt_depth, t_ref)

    print("[Test 1.2] Half sample splitting")
    print("  r_time =", r_time)
    print("  expected: r_time[3] and r_time[4] should be about 0.5")
    assert np.allclose(r_time[3], 0.5) and np.allclose(r_time[4], 0.5), f"Linear splitting failed, got {r_time[3]}, {r_time[4]}"

    # 情况 3：两个相反反射点落入相近时间，会发生相消
    r_depth = np.zeros(5)
    r_depth[0] = 1.0
    r_depth[1] = -1.0
    twt_depth = np.array([0.000, 0.0032, 0.0033, 0.007, 0.009])

    r_time = scatter_reflectivity_to_time(r_depth, twt_depth, t_ref)

    print("[Test 1.3] Destructive interference (cancellation)")
    print("  r_time =", r_time)
    print("  expected: positive and negative events partly cancel")
    print("  Success: Basic scatter function mathematically consistent.\n")


# =====================================================================
# 2. Test 2: Energy Comparison Function (Depth vs Time Domain)
# =====================================================================
def rms_energy(x):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return np.nan
    return np.sqrt(np.nanmean(x * x))


def compare_depth_time_energy(
    r_depth_fixed,
    twt_init,
    r_time_init,
    time_axis,
    t1=1.55,
    t2=2.05
):
    print("=" * 60)
    print(f"TEST 2: DEPTH VS TIME ENERGY RATIO COMPARISON ({t1} - {t2} s)")
    print("=" * 60)
    
    # 深度域界面反射系数对应时间：scatter 函数使用 twt_init[1:]
    r_if = r_depth_fixed[:-1]
    t_if = twt_init[1:]

    mask_depth_global = np.isfinite(r_if) & np.isfinite(t_if)
    mask_depth_local = mask_depth_global & (t_if >= t1) & (t_if <= t2)

    mask_time_global = np.isfinite(r_time_init)
    mask_time_local = mask_time_global & (time_axis >= t1) & (time_axis <= t2)

    depth_global_rms = rms_energy(r_if[mask_depth_global])
    depth_local_rms = rms_energy(r_if[mask_depth_local])

    time_global_rms = rms_energy(r_time_init[mask_time_global])
    time_local_rms = rms_energy(r_time_init[mask_time_local])

    depth_ratio = depth_local_rms / (depth_global_rms + 1e-12)
    time_ratio = time_local_rms / (time_global_rms + 1e-12)

    print("[Depth-domain reflectivity]")
    print(f"  local RMS / global RMS = {depth_ratio:.4f}")
    print(f"  local RMS              = {depth_local_rms:.6f}")
    print(f"  global RMS             = {depth_global_rms:.6f}")
    print(f"  local sample count     = {np.sum(mask_depth_local)}")

    print("[Time-domain reflectivity]")
    print(f"  local RMS / global RMS = {time_ratio:.4f}")
    print(f"  local RMS              = {time_local_rms:.6f}")
    print(f"  global RMS             = {time_global_rms:.6f}")
    print(f"  local sample count     = {np.sum(mask_time_local)}")

    print("\n[Interpretation]")
    if depth_ratio > 0.7 and time_ratio < 0.4:
        print("  >>> 结论：深度域不弱，但时间域明显变弱！重点怀疑 scatter / 时间采样 / 相消。")
    elif depth_ratio < 0.5 and time_ratio < 0.5:
        print("  >>> 结论：深度域和时间域都弱！更可能是 Backus、密度估计或真实弱反射。")
    elif depth_ratio > 0.7 and time_ratio > 0.7:
        print("  >>> 结论：深度域和时间域都不弱！问题不主要在反射系数能量上。")
    else:
        print("  >>> 结论：结果介于中间，请结合 nearest/linear 和 fine-grid 测试综合判断。")
    print()


# =====================================================================
# 3. Test 3: Nearest Neighbor Projection Comparison
# =====================================================================
def scatter_reflectivity_to_time_nearest(r_depth, twt_depth, t_ref):
    r_depth = np.asarray(r_depth, dtype=float)
    twt_depth = np.asarray(twt_depth, dtype=float)
    t_ref = np.asarray(t_ref, dtype=float)

    if len(r_depth) != len(twt_depth):
        raise ValueError("r_depth 和 twt_depth 长度必须一致。")

    dt = np.median(np.diff(t_ref))
    n_time = len(t_ref)

    r_time = np.zeros(n_time)

    r_if = r_depth[:-1]
    t_if = twt_depth[1:]

    mask = (t_if >= t_ref[0]) & (t_if <= t_ref[-1])

    ti = t_if[mask]
    ai = r_if[mask]

    if len(ti) == 0:
        return r_time

    k = np.round((ti - t_ref[0]) / dt).astype(int)
    k = np.clip(k, 0, n_time - 1)

    np.add.at(r_time, k, ai)

    return r_time


# =====================================================================
# 4. Test 4: Time Bin Cancellation Diagnostic
# =====================================================================
def diagnose_time_bin_cancellation(r_depth, twt_depth, t_ref, t1=1.55, t2=2.05):
    print("=" * 60)
    print(f"TEST 4: TIME BIN CANCELLATION DIAGNOSTIC ({t1} - {t2} s)")
    print("=" * 60)
    r_depth = np.asarray(r_depth, dtype=float)
    twt_depth = np.asarray(twt_depth, dtype=float)
    t_ref = np.asarray(t_ref, dtype=float)

    dt = np.median(np.diff(t_ref))
    n_time = len(t_ref)

    r_if = r_depth[:-1]
    t_if = twt_depth[1:]

    mask = (t_if >= t1) & (t_if <= t2)

    ti = t_if[mask]
    ai = r_if[mask]

    if len(ti) == 0:
        print("No reflectivity samples in this TWT interval.")
        return

    k = np.floor((ti - t_ref[0]) / dt).astype(int)
    k = np.clip(k, 0, n_time - 1)

    signed_sum = np.zeros(n_time)
    abs_sum = np.zeros(n_time)

    np.add.at(signed_sum, k, ai)
    np.add.at(abs_sum, k, np.abs(ai))

    mask_bin = abs_sum > 0

    total_signed_abs = np.sum(np.abs(signed_sum[mask_bin]))
    total_abs = np.sum(abs_sum[mask_bin])

    cancellation_ratio = 1.0 - total_signed_abs / (total_abs + 1e-12)

    print(f"  Number of interface samples = {len(ai)}")
    print(f"  Occupied time bins          = {np.sum(mask_bin)}")
    print(f"  sum(abs(signed bin sum))    = {total_signed_abs:.6f}")
    print(f"  sum(abs(original samples))  = {total_abs:.6f}")
    print(f"  Cancellation ratio          = {cancellation_ratio:.4f}")

    print("\n[Interpretation]")
    if cancellation_ratio > 0.6:
        print("  >>> 警告：相消极强！同一采样格点内大量的正负反射发生干涉抵消。")
    elif cancellation_ratio > 0.3:
        print("  >>> 提示：存在中等强度的相消，可能对局部能量有一定衰减作用。")
    else:
        print("  >>> 结论：相消很弱。重采样相消不是造成局部弱反射的主因。")
    print()


# =====================================================================
# 5. Test 5: Coarse vs Fine Grid Scatter-Convolution Comparison
# =====================================================================
def make_ricker_wavelet(f_dom, dt, wavelet_length_s):
    n = int(round(wavelet_length_s / dt)) + 1
    if n % 2 == 0:
        n += 1
    half = n // 2
    tw = (np.arange(n) - half) * dt
    pf = np.pi * f_dom * tw
    w = (1.0 - 2.0 * pf * pf) * np.exp(-pf * pf)
    # Energy normalization
    w = w / (np.sqrt(np.sum(w * w)) + 1e-12)
    return w


def local_cc(a, b):
    a = np.asarray(a, dtype=float).copy()
    b = np.asarray(b, dtype=float).copy()
    a = a - np.mean(a)
    b = b - np.mean(b)
    denom = np.sqrt(np.sum(a * a) * np.sum(b * b)) + 1e-12
    return np.sum(a * b) / denom


def compare_coarse_vs_fine_scatter_convolution(
    r_depth_fixed,
    twt_init,
    time_axis,
    obs_trace,
    f_dom,
    wavelet_length_s=0.129,
    oversample=10,
    t1=1.55,
    t2=2.05
):
    print("=" * 60)
    print(f"TEST 5: COARSE VS FINE SCATTER-CONVOLUTION COMPARISON")
    print("=" * 60)

    dt = np.median(np.diff(time_axis))
    dt_fine = dt / oversample

    # A. 粗时间轴：常规流程
    r_coarse = scatter_reflectivity_to_time(
        r_depth=r_depth_fixed,
        twt_depth=twt_init,
        t_ref=time_axis
    )
    w_coarse = make_ricker_wavelet(f_dom=f_dom, dt=dt, wavelet_length_s=wavelet_length_s)
    s_coarse = np.convolve(r_coarse, w_coarse, mode="same")

    # B. 细时间轴：高分辨率卷积，再下采样
    t_fine = np.arange(time_axis[0], time_axis[-1] + 0.5 * dt_fine, dt_fine)
    r_fine = scatter_reflectivity_to_time(
        r_depth=r_depth_fixed,
        twt_depth=twt_init,
        t_ref=t_fine
    )
    w_fine = make_ricker_wavelet(f_dom=f_dom, dt=dt_fine, wavelet_length_s=wavelet_length_s)
    s_fine = np.convolve(r_fine, w_fine, mode="same")
    s_fine_to_coarse = np.interp(time_axis, t_fine, s_fine)

    # 局部指标统计
    mask = (time_axis >= t1) & (time_axis <= t2)
    obs_local = obs_trace[mask]
    s_coarse_local = s_coarse[mask]
    s_fine_local = s_fine_to_coarse[mask]

    cc_coarse_obs = local_cc(obs_local, s_coarse_local)
    cc_fine_obs = local_cc(obs_local, s_fine_local)
    cc_coarse_fine = local_cc(s_coarse_local, s_fine_local)

    rms_coarse = rms_energy(s_coarse_local)
    rms_fine = rms_energy(s_fine_local)
    ratio_rms = rms_fine / (rms_coarse + 1e-12)

    print(f"  Coarse dt              = {dt:.6f} s")
    print(f"  Fine dt (oversample)   = {dt_fine:.6f} s (x{oversample})")
    print(f"  CC(obs, coarse syn)    = {cc_coarse_obs:.4f}")
    print(f"  CC(obs, fine syn)      = {cc_fine_obs:.4f}")
    print(f"  CC(coarse, fine syn)   = {cc_coarse_fine:.4f}")
    print(f"  RMS coarse local       = {rms_coarse:.6f}")
    print(f"  RMS fine local         = {rms_fine:.6f}")
    print(f"  RMS ratio (fine/coarse)= {ratio_rms:.4f}")

    print("\n[Interpretation]")
    if cc_coarse_fine > 0.95 and 0.8 < ratio_rms < 1.25:
        print("  >>> 结论：粗采样与细采样卷积高度相似！当前 1ms 重采样不是局部能量弱的主因。")
    elif cc_fine_obs > cc_coarse_obs + 0.05:
        print("  >>> 警告：细采样路线显著优于粗采样！说明先 scatter 到粗网格会导致严重的信息丢失。")
    else:
        print("  >>> 结论：细采样无法提供实质性改善，原因更可能是 Backus、密度、TWT 或子波参数问题。")
    print()

    return r_coarse, s_coarse, r_fine, s_fine_to_coarse


# =====================================================================
# Main Experiment Diagnostic Runner
# =====================================================================
if __name__ == "__main__":
    # Load actual experiment preprocessing output NPZ
    npz_path = r"D:\python_code\python_project\seismic_well\true_data_tying3\true_datacode\_experiment_data\initial_model_data\initial_real_model_data_CB323.npz"
    
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Missing NPZ input data: {npz_path}. Please run data_preprocess.py first.")
        
    data = np.load(npz_path)
    
    r_depth_fixed = data['r_depth_fixed']
    twt_init = data['twt_init']
    r_time_init = data['r_time_init']
    time_axis = data['time_axis']
    seismic_trace_norm = data['seismic_trace_norm']
    f_dom = float(data['f_dom'])
    
    # -----------------------------
    # 运行基本单元测试
    # -----------------------------
    test_scatter_basic()

    # 设定我们重点怀疑的弱反射异常 TWT 时间段：1.10 - 1.70 秒
    t1_target, t2_target = 1.10, 1.70

    # -----------------------------
    # 运行真实数据能量比例测试 (Test 2)
    # -----------------------------
    compare_depth_time_energy(
        r_depth_fixed=r_depth_fixed,
        twt_init=twt_init,
        r_time_init=r_time_init,
        time_axis=time_axis,
        t1=t1_target,
        t2=t2_target
    )

    # -----------------------------
    # 运行 Nearest Neighbor 对比 (Test 3)
    # -----------------------------
    r_time_nearest = scatter_reflectivity_to_time_nearest(
        r_depth=r_depth_fixed,
        twt_depth=twt_init,
        t_ref=time_axis
    )
    print("=" * 60)
    print("TEST 3: NEAREST NEIGHBOR SCATTER VS LINEAR SCATTER ENERGY")
    print("=" * 60)
    print("[LINEAR SCATTER]:")
    compare_depth_time_energy(
        r_depth_fixed=r_depth_fixed,
        twt_init=twt_init,
        r_time_init=r_time_init,
        time_axis=time_axis,
        t1=t1_target,
        t2=t2_target
    )
    print("[NEAREST SCATTER]:")
    compare_depth_time_energy(
        r_depth_fixed=r_depth_fixed,
        twt_init=twt_init,
        r_time_init=r_time_nearest,
        time_axis=time_axis,
        t1=t1_target,
        t2=t2_target
    )

    # -----------------------------
    # 运行时间网格相消程度诊断 (Test 4)
    # -----------------------------
    diagnose_time_bin_cancellation(
        r_depth=r_depth_fixed,
        twt_depth=twt_init,
        t_ref=time_axis,
        t1=t1_target,
        t2=t2_target
    )

    # -----------------------------
    # 运行高采样率 vs 粗采样率卷积测试 (Test 5)
    # -----------------------------
    r_coarse, s_coarse, r_fine, s_fine_to_coarse = compare_coarse_vs_fine_scatter_convolution(
        r_depth_fixed=r_depth_fixed,
        twt_init=twt_init,
        time_axis=time_axis,
        obs_trace=seismic_trace_norm,
        f_dom=f_dom,
        oversample=10,
        t1=t1_target,
        t2=t2_target
    )

    # -----------------------------
    # 6. 生成并保存诊断 QC 图像
    # -----------------------------
    t1_plot, t2_plot = 0.80, 2.20
    mask_plot = (time_axis >= t1_plot) & (time_axis <= t2_plot)

    obs_plot = seismic_trace_norm[mask_plot]
    s_coarse_plot = s_coarse[mask_plot]
    s_fine_plot = s_fine_to_coarse[mask_plot]
    t_plot = time_axis[mask_plot]

    obs_plot = obs_plot / (np.max(np.abs(obs_plot)) + 1e-12)
    s_coarse_plot = s_coarse_plot / (np.max(np.abs(s_coarse_plot)) + 1e-12)
    s_fine_plot = s_fine_plot / (np.max(np.abs(s_fine_plot)) + 1e-12)

    plt.figure(figsize=(8, 10))
    plt.plot(obs_plot, t_plot, label="Observed Seismic (Norm)", color="black", linewidth=1.5)
    plt.plot(s_coarse_plot, t_plot, label="Coarse Grid Scatter + Conv (1ms)", color="orange", linestyle="--", linewidth=1.2)
    plt.plot(s_fine_plot, t_plot, label="Fine Grid Scatter + Conv -> Coarse (0.1ms -> 1ms)", color="red", alpha=0.8, linewidth=1.2)

    # 标记出异常时间段
    plt.axhspan(t1_target, t2_target, color="red", alpha=0.1, label=f"Anomaly Zone ({t1_target}-{t2_target} s)")
    
    plt.gca().invert_yaxis()
    plt.xlabel("Normalized Synthetic/Obs Amplitude")
    plt.ylabel("TWT (s)")
    plt.title(f"Scatter Route & Discretization QC\nZone: {t1_plot:.2f} - {t2_plot:.2f} s (Anomaly highlighted)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    output_fig_path = "fig_scatter_route_qc.png"
    plt.savefig(output_fig_path, dpi=300)
    print(f"QC figure saved successfully to: {output_fig_path}")
    print("\nAll diagnostics complete. Check logs above for detailed geophysical conclusions.")