# -*- coding: utf-8 -*-
"""检查阶段快照只是观测输出，不改变 L2/Student-t 反演及原诊断。

直接运行本脚本；输入为固定种子合成反射系数和已知中心子波，无新增依赖。
"""

import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.forward_operator import stationary_convolution
from stages.stage_wavelet_prior import make_center_ricker_wavelet
from utils.wavelet_inversion_robust import time_varying_wavelet_inversion


def check_snapshot_observational_equivalence():
    """固定输入检查快照开关、缺测标记、滤波位置及内存独立性。

    输入：函数内部生成 shape=(241,) 的反射系数/合成道和 shape=(17,) 子波；
    dt=0.001 s，频率 50 Hz，振幅为合成相对单位，不读取或修改实际井数据。
    输出：通过计数并 print；失败抛 AssertionError，不产生科研结果文件。
    数学作用：验证快照开/关输出逐元素相等，并检验保存位置遵循既有 Gaussian
    与去均值顺序；比较 L2、Student-t、两个滤波轴和零有效中心的情况。
    """
    random_generator = np.random.default_rng(20260908)
    r_time = random_generator.normal(0.0, 0.1, 241)  # shape=(N_time,)，反射系数。
    w_prior = make_center_ricker_wavelet(f0=50.0, dt=0.001, length=17)
    s_clean = stationary_convolution(r_time, w_prior, alignment="center")
    noise = random_generator.normal(0.0, 0.03 * np.std(s_clean), r_time.size)
    s_obs = s_clean + noise  # shape=(N_time,)，相对地震振幅。
    assert r_time.shape == s_obs.shape and w_prior.shape == (17,)

    checks_passed = 0
    cases = [(0.0, 0.0, 0.0), (3.0, 0.0, 0.0), (0.0, 2.0, 0.0),
             (3.0, 2.0, 0.0), (3.0, 2.0, 1e12)]
    for loss_type in ("l2", "student_t"):
        for time_sigma, lag_sigma, energy_threshold in cases:
            parameters = dict(
                r_time=r_time, s_obs=s_obs, w_prior=w_prior,
                wavelet_length=17, data_window_length=51, dt=0.001,
                mu1=0.2, mu2=6.0, mu_dc=0.0, mu_prior=1.0, mu_time=8.0,
                mu_edge=1.0, damping_ratio=0.0, svd_cutoff_ratio=0.001,
                energy_threshold=energy_threshold, estimate_step_samples=10,
                time_smooth_sigma=time_sigma, wavelet_smooth_sigma=lag_sigma,
                loss_type=loss_type, student_nu=10.0, irls_max_iter=12,
                irls_tol=3e-4, return_diagnostics=True, verbose=False,
            )
            W_without, diag_without = time_varying_wavelet_inversion(**parameters)
            W_with, diag_with = time_varying_wavelet_inversion(
                **parameters, store_stage_wavelets=True,
            )
            np.testing.assert_array_equal(W_without, W_with)
            for diagnostic_name, previous_value in diag_without.items():
                np.testing.assert_equal(previous_value, diag_with[diagnostic_name])
            assert "W_pre_gaussian" not in diag_without

            direct_mask = diag_with["valid_mask"]  # shape=(N_time,)，有效中心掩码。
            W_direct = diag_with["W_direct_centers"]
            W_pre = diag_with["W_pre_gaussian"]
            W_post = diag_with["W_post_gaussian"]
            for snapshot in (W_direct, W_pre, W_post):
                assert snapshot.shape == W_with.shape
                assert not np.shares_memory(snapshot, W_with)
            assert not np.shares_memory(W_direct, W_pre)
            assert not np.shares_memory(W_pre, W_post)
            assert np.all(np.isnan(W_direct[~direct_mask]))
            assert np.all(np.isfinite(W_direct[direct_mask]))
            np.testing.assert_array_equal(W_direct[direct_mask], W_pre[direct_mask])

            expected_post = gaussian_filter(
                W_pre, sigma=(time_sigma, lag_sigma), mode="nearest",
            )
            np.testing.assert_array_equal(W_post, expected_post)
            post_row_mean = np.mean(W_post, axis=1, keepdims=True)  # shape=(N_time,1)。
            expected_output = W_post - post_row_mean
            np.testing.assert_array_equal(W_with, expected_output)
            if energy_threshold > 0.0:
                assert not np.any(direct_mask)
                # 原有零有效中心分支先对 prior 去均值，再沿时间平铺。
                prior_zero_mean = w_prior - np.mean(w_prior)  # shape=(17,)。
                prior_rows = np.tile(prior_zero_mean[None, :], (r_time.size, 1))  # shape=(N_time,17)。
                np.testing.assert_array_equal(W_pre, prior_rows)
            else:
                assert np.any(direct_mask)
            checks_passed += 1

    print(f"PASS: {checks_passed} L2/Student-t snapshot cases; numerical outputs and old diagnostics unchanged.")
    return checks_passed


if __name__ == "__main__":
    check_snapshot_observational_equivalence()
