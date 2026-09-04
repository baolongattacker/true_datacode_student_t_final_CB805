# -*- coding: utf-8 -*-
"""第一阶段局部整理的最小数学与功能回归测试。"""

from __future__ import annotations

import unittest

import numpy as np

from core.forward_operator import (
    build_local_convolution_matrix,
    nonstationary_convolution,
    stationary_convolution,
)
from core.wavelet_qc import (
    compute_wavelet_energy,
    compute_wavelet_peak_metric_ms,
    compute_wavelet_qc_attributes,
)


RANDOM_SEED = 20260722


class ForwardOperatorRegressionTest(unittest.TestCase):
    """检查统一正演算子的 center/causal 数学约定。"""

    def setUp(self):
        self.rng = np.random.default_rng(RANDOM_SEED)

    def test_tiled_wavelet_matches_stationary_convolution(self):
        """同一子波逐时刻平铺后，非平稳正演应退化为平稳卷积。"""
        n_time = 67
        wavelet_length = 11
        r_time = self.rng.normal(size=n_time)
        w_stationary = self.rng.normal(size=wavelet_length)
        w_stationary = w_stationary - np.mean(w_stationary)
        W_tiled = np.tile(w_stationary[None, :], (n_time, 1))

        for alignment in ("center", "causal"):
            with self.subTest(alignment=alignment):
                s_stationary = stationary_convolution(
                    r_time,
                    w_stationary,
                    alignment=alignment,
                )
                s_nonstationary = nonstationary_convolution(
                    r_time,
                    W_tiled,
                    alignment=alignment,
                )
                np.testing.assert_allclose(
                    s_nonstationary,
                    s_stationary,
                    rtol=1e-13,
                    atol=1e-13,
                )

    def test_local_matrix_matches_forward_windows(self):
        """局部 R@w 应与完整正演在同一时间窗内完全一致。"""
        n_time = 67
        data_window_length = 15
        wavelet_length = 11
        half_data_window = data_window_length // 2
        center_indices = (
            half_data_window,
            n_time // 2,
            n_time - 1 - half_data_window,
        )

        r_time = self.rng.normal(size=n_time)
        w_stationary = self.rng.normal(size=wavelet_length)
        W_tiled = np.tile(w_stationary[None, :], (n_time, 1))

        for alignment in ("center", "causal"):
            s_stationary = stationary_convolution(
                r_time,
                w_stationary,
                alignment=alignment,
            )
            s_nonstationary = nonstationary_convolution(
                r_time,
                W_tiled,
                alignment=alignment,
            )

            for center_index in center_indices:
                with self.subTest(
                    alignment=alignment,
                    center_index=center_index,
                ):
                    R_local = build_local_convolution_matrix(
                        r_time=r_time,
                        center_index=center_index,
                        data_window_length=data_window_length,
                        wavelet_length=wavelet_length,
                        alignment=alignment,
                    )
                    s_local = R_local @ w_stationary
                    start_index = center_index - half_data_window
                    stop_index = center_index + half_data_window + 1

                    np.testing.assert_allclose(
                        s_local,
                        s_stationary[start_index:stop_index],
                        rtol=1e-13,
                        atol=1e-13,
                    )
                    np.testing.assert_allclose(
                        s_local,
                        s_nonstationary[start_index:stop_index],
                        rtol=1e-13,
                        atol=1e-13,
                    )


class ValidationRegressionTest(unittest.TestCase):
    """非法 Shape、长度和非有限值必须明确失败，不能静默展平或传播。"""

    def setUp(self):
        self.rng = np.random.default_rng(RANDOM_SEED)
        self.r_time = self.rng.normal(size=24)
        self.w_stationary = self.rng.normal(size=9)
        self.W_tiled = np.tile(self.w_stationary[None, :], (24, 1))

    def test_stationary_convolution_rejects_invalid_shape_and_nan(self):
        with self.assertRaises(ValueError):
            stationary_convolution(
                self.r_time.reshape(4, 6),
                self.w_stationary,
                alignment="center",
            )

        w_with_nan = self.w_stationary.copy()
        w_with_nan[3] = np.nan
        with self.assertRaises(ValueError):
            stationary_convolution(
                self.r_time,
                w_with_nan,
                alignment="center",
            )

    def test_nonstationary_convolution_rejects_length_and_nan(self):
        with self.assertRaises(ValueError):
            nonstationary_convolution(
                self.r_time,
                self.W_tiled[:-1, :],
                alignment="causal",
            )

        W_with_nan = self.W_tiled.copy()
        W_with_nan[5, 4] = np.nan
        with self.assertRaises(ValueError):
            nonstationary_convolution(
                self.r_time,
                W_with_nan,
                alignment="causal",
            )

    def test_local_matrix_and_qc_reject_nan_or_invalid_shape(self):
        r_time_with_nan = self.r_time.copy()
        r_time_with_nan[7] = np.nan
        with self.assertRaises(ValueError):
            build_local_convolution_matrix(
                r_time=r_time_with_nan,
                center_index=12,
                data_window_length=11,
                wavelet_length=9,
                alignment="center",
            )

        with self.assertRaises(ValueError):
            compute_wavelet_qc_attributes(
                self.w_stationary,
                dt=0.001,
                wavelet_alignment="center",
            )

        W_with_nan = self.W_tiled.copy()
        W_with_nan[2, 2] = np.nan
        with self.assertRaises(ValueError):
            compute_wavelet_qc_attributes(
                W_with_nan,
                dt=0.001,
                wavelet_alignment="center",
            )


class FinalWaveletQcRegressionTest(unittest.TestCase):
    """检查 local fallback 后的 QC 确实来自最终子波矩阵。"""

    def setUp(self):
        self.rng = np.random.default_rng(RANDOM_SEED)

    def test_final_qc_matches_direct_calculation_after_local_fallback(self):
        n_time = 9
        wavelet_length = 11
        dt_s = 0.001

        lag_samples = np.arange(wavelet_length) - wavelet_length // 2
        w_stationary = (1.0 - 0.18 * lag_samples ** 2) * np.exp(
            -0.09 * lag_samples ** 2
        )
        w_stationary = w_stationary - np.mean(w_stationary)

        W_before_fallback = self.rng.normal(
            scale=0.15,
            size=(n_time, wavelet_length),
        )
        W_before_fallback[:, wavelet_length // 2 + 2] += 1.5
        fallback_mask = np.zeros(n_time, dtype=bool)
        fallback_mask[[2, 5, 6]] = True

        # 与主流程 alpha_smooth_samples=0 时的局部 fallback 公式一致：
        # W_final[i] = alpha[i] * W_raw[i] + (1-alpha[i]) * w_stationary。
        fallback_alpha = np.ones(n_time, dtype=float)
        fallback_alpha[fallback_mask] = 0.0
        W_final = (
            fallback_alpha[:, None] * W_before_fallback
            + (1.0 - fallback_alpha[:, None]) * w_stationary[None, :]
        )

        np.testing.assert_array_equal(fallback_alpha[fallback_mask], 0.0)
        np.testing.assert_allclose(
            W_final[fallback_mask, :],
            np.tile(w_stationary[None, :], (np.sum(fallback_mask), 1)),
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            W_final[~fallback_mask, :],
            W_before_fallback[~fallback_mask, :],
            rtol=0.0,
            atol=0.0,
        )

        qc_before_fallback = compute_wavelet_qc_attributes(
            W_before_fallback,
            dt=dt_s,
            wavelet_alignment="center",
        )
        qc_final = compute_wavelet_qc_attributes(
            W_final,
            dt=dt_s,
            wavelet_alignment="center",
        )
        self.assertFalse(
            np.array_equal(
                qc_before_fallback["peak_metric_ms"][fallback_mask],
                qc_final["peak_metric_ms"][fallback_mask],
            )
        )

        peak_metric_direct = compute_wavelet_peak_metric_ms(
            W_final,
            dt=dt_s,
            wavelet_alignment="center",
        )
        energy_squared_direct, _ = compute_wavelet_energy(W_final)
        energy_l2_direct = np.sqrt(energy_squared_direct)
        energy_l2_norm_direct = energy_l2_direct / (
            np.max(energy_l2_direct) + 1e-12
        )

        frequencies_hz = np.fft.rfftfreq(wavelet_length, d=dt_s)
        power_spectrum = np.abs(np.fft.rfft(W_final, axis=1)) ** 2
        centroid_frequency_direct = np.sum(
            power_spectrum * frequencies_hz[None, :],
            axis=1,
        ) / np.sum(power_spectrum, axis=1)

        np.testing.assert_allclose(
            qc_final["peak_metric_ms"],
            peak_metric_direct,
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            qc_final["energy_l2"],
            energy_l2_direct,
            rtol=1e-13,
            atol=1e-13,
        )
        np.testing.assert_allclose(
            qc_final["energy_l2_norm"],
            energy_l2_norm_direct,
            rtol=1e-13,
            atol=1e-13,
        )
        np.testing.assert_allclose(
            qc_final["centroid_frequency_hz"],
            centroid_frequency_direct,
            rtol=1e-12,
            atol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
