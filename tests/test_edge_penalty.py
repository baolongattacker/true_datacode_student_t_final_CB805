# -*- coding: utf-8 -*-
"""边缘能量约束与候选来源编码的最小数学回归测试。"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

import utils.wavelet_inversion_robust as robust_module
from core.wavelet_qc import compute_boundary_peak_mask
from experiments.run_real_experiment import (
    FINAL_SOURCE_LOCAL_BLEND,
    FINAL_SOURCE_LOCAL_FALLBACK_PRIOR,
    FINAL_SOURCE_TV_CANDIDATE,
    build_final_wavelet_source_codes,
    longest_true_gap_duration_ms,
)
from stages.stage_stationary import StationaryResult
from stages.stage_wavelet_prior import build_stationary_prior_with_fallback
from utils.wavelet_inversion_robust import (
    ORIGIN_EXTRAPOLATED,
    ORIGIN_INTERPOLATED,
    ORIGIN_INVERTED,
    _build_candidate_wavelet_origin_masks,
    _full_student_t_objective,
    _regularization_value,
    _solve_regularized_window,
    build_edge_penalty_weights,
    time_varying_wavelet_inversion,
)


class EdgePenaltyWeightTest(unittest.TestCase):
    """检查权重的物理对称性、中心零值和参数边界。"""

    def test_odd_and_even_lengths_are_symmetric(self):
        for wavelet_length in (9, 10):
            with self.subTest(wavelet_length=wavelet_length):
                edge_fraction = 0.22
                edge_weights = build_edge_penalty_weights(
                    wavelet_length=wavelet_length,
                    edge_fraction=edge_fraction,
                    taper="cosine",
                )
                edge_samples = min(
                    (wavelet_length - 1) // 2,
                    int(np.ceil(edge_fraction * wavelet_length)),
                )

                np.testing.assert_allclose(
                    edge_weights,
                    edge_weights[::-1],
                    rtol=0.0,
                    atol=0.0,
                )
                self.assertEqual(edge_weights[0], 1.0)
                self.assertEqual(edge_weights[-1], 1.0)
                np.testing.assert_array_equal(
                    edge_weights[edge_samples:wavelet_length - edge_samples],
                    0.0,
                )
                self.assertTrue(np.all(np.diff(edge_weights[:edge_samples]) <= 0.0))
                self.assertTrue(np.all(np.diff(edge_weights[-edge_samples:]) >= 0.0))

    def test_invalid_edge_parameters_fail_explicitly(self):
        for edge_fraction in (0.0, -0.1, 0.5, 0.8, np.nan):
            with self.subTest(edge_fraction=edge_fraction):
                with self.assertRaises(ValueError):
                    build_edge_penalty_weights(9, edge_fraction=edge_fraction)

        with self.assertRaises(ValueError):
            build_edge_penalty_weights(9, taper="linear")
        with self.assertRaises(ValueError):
            build_edge_penalty_weights(2)


class EdgePenaltyObjectiveTest(unittest.TestCase):
    """检查增广矩阵、L2 和 Student-t 真实目标使用同一个边缘项。"""

    def setUp(self):
        self.wavelet_length = 9
        self.edge_weights = build_edge_penalty_weights(
            self.wavelet_length,
            edge_fraction=0.22,
        )
        self.I = np.eye(self.wavelet_length)
        self.D = robust_module._second_derivative_matrix(self.wavelet_length)
        self.C_dc = np.ones((1, self.wavelet_length)) / np.sqrt(
            self.wavelet_length
        )

    def test_augmented_edge_block_matches_regularization_value(self):
        data_scale = 2.5
        mu_edge = 1.7
        captured = {}

        def capture_solve(*, A, b, damping_ratio, svd_cutoff_ratio):
            captured["A"] = np.asarray(A, dtype=float)
            captured["b"] = np.asarray(b, dtype=float)
            return np.zeros(A.shape[1], dtype=float)

        with patch.object(robust_module, "_safe_svd_solve", side_effect=capture_solve):
            _solve_regularized_window(
                R=np.eye(self.wavelet_length),
                s_win=np.ones(self.wavelet_length),
                data_scale=data_scale,
                I=self.I,
                D=self.D,
                C_dc=self.C_dc,
                mu1=0.0,
                mu2=0.0,
                mu_dc=0.0,
                prior_i=None,
                mu_prior=0.0,
                previous_wavelet=None,
                mu_time=0.0,
                damping_ratio=0.0,
                svd_cutoff_ratio=0.0,
                edge_weights=self.edge_weights,
                mu_edge=mu_edge,
            )

        edge_block = captured["A"][-self.wavelet_length:, :]
        test_wavelet = np.linspace(-1.0, 1.0, self.wavelet_length)
        matrix_value = 0.5 * float(np.sum((edge_block @ test_wavelet) ** 2))
        objective_value = _regularization_value(
            w=test_wavelet,
            data_scale=data_scale,
            D=self.D,
            C_dc=self.C_dc,
            mu1=0.0,
            mu2=0.0,
            mu_dc=0.0,
            prior_i=None,
            mu_prior=0.0,
            previous_wavelet=None,
            mu_time=0.0,
            edge_weights=self.edge_weights,
            mu_edge=mu_edge,
        )
        self.assertAlmostEqual(matrix_value, objective_value, places=14)

    def test_student_t_objective_contains_same_edge_value(self):
        wavelet = np.linspace(-0.8, 0.9, self.wavelet_length)
        R = np.eye(self.wavelet_length)
        s_win = np.zeros(self.wavelet_length)
        data_scale = 1.8
        mu_edge = 0.75

        objective_without_edge = _full_student_t_objective(
            w=wavelet,
            R=R,
            s_win=s_win,
            sigma=1.0,
            nu=10.0,
            data_scale=data_scale,
            D=self.D,
            C_dc=self.C_dc,
            mu1=0.0,
            mu2=0.0,
            mu_dc=0.0,
            prior_i=None,
            mu_prior=0.0,
            previous_wavelet=None,
            mu_time=0.0,
        )
        objective_with_edge = _full_student_t_objective(
            w=wavelet,
            R=R,
            s_win=s_win,
            sigma=1.0,
            nu=10.0,
            data_scale=data_scale,
            D=self.D,
            C_dc=self.C_dc,
            mu1=0.0,
            mu2=0.0,
            mu_dc=0.0,
            prior_i=None,
            mu_prior=0.0,
            previous_wavelet=None,
            mu_time=0.0,
            edge_weights=self.edge_weights,
            mu_edge=mu_edge,
        )
        expected_edge_value = 0.5 * data_scale * mu_edge * float(
            np.sum((self.edge_weights * wavelet) ** 2)
        )
        self.assertAlmostEqual(
            objective_with_edge - objective_without_edge,
            expected_edge_value,
            places=14,
        )

    def test_l2_edge_norm_decreases_for_fixed_small_problem(self):
        s_win = np.zeros(self.wavelet_length)
        s_win[0] = 1.0
        s_win[-1] = -0.8
        s_win[self.wavelet_length // 2] = 0.6

        common_arguments = {
            "R": np.eye(self.wavelet_length),
            "s_win": s_win,
            "data_scale": 1.0,
            "I": self.I,
            "D": self.D,
            "C_dc": self.C_dc,
            "mu1": 0.0,
            "mu2": 0.0,
            "mu_dc": 0.0,
            "prior_i": None,
            "mu_prior": 0.0,
            "previous_wavelet": None,
            "mu_time": 0.0,
            "damping_ratio": 0.0,
            "svd_cutoff_ratio": 0.0,
        }
        wavelet_without_edge = _solve_regularized_window(**common_arguments)
        wavelet_with_edge = _solve_regularized_window(
            **common_arguments,
            edge_weights=self.edge_weights,
            mu_edge=2.0,
        )

        norm_without_edge = np.linalg.norm(
            self.edge_weights * wavelet_without_edge
        )
        norm_with_edge = np.linalg.norm(self.edge_weights * wavelet_with_edge)
        self.assertLess(norm_with_edge, norm_without_edge)

    def test_mu_edge_zero_never_builds_edge_weights(self):
        n_time = 31
        wavelet_length = 5
        r_time = np.zeros(n_time)
        r_time[8] = 1.0
        r_time[15] = -0.7
        r_time[22] = 0.5
        s_obs = r_time.copy()
        w_prior = np.zeros(wavelet_length)
        w_prior[wavelet_length // 2] = 1.0

        with patch.object(
            robust_module,
            "build_edge_penalty_weights",
            side_effect=AssertionError("mu_edge=0 不应构造边缘权重"),
        ):
            W, diagnostics = time_varying_wavelet_inversion(
                r_time=r_time,
                s_obs=s_obs,
                wavelet_length=wavelet_length,
                data_window_length=15,
                w_prior=w_prior,
                use_stationary_prior=False,
                mu_edge=0.0,
                estimate_step_samples=2,
                energy_threshold=0.0,
                reject_ill_conditioned=False,
                reject_amplitude_jumps=False,
                peak_lock=False,
                time_smooth_sigma=0.0,
                wavelet_smooth_sigma=0.0,
                return_diagnostics=True,
                verbose=False,
            )
        self.assertEqual(W.shape, (n_time, wavelet_length))
        self.assertFalse(diagnostics["edge_penalty_active"])
        self.assertEqual(diagnostics["edge_penalty_weights"].size, 0)

        with self.assertRaises(ValueError):
            time_varying_wavelet_inversion(
                r_time=r_time,
                s_obs=s_obs,
                wavelet_length=wavelet_length,
                data_window_length=15,
                w_prior=w_prior,
                mu_edge=-0.1,
                return_diagnostics=True,
                verbose=False,
            )

    def test_l2_and_student_t_paths_both_use_active_edge_penalty(self):
        n_time = 41
        wavelet_length = 7
        r_time = np.zeros(n_time)
        r_time[[10, 20, 30]] = [1.0, -0.8, 0.6]
        s_obs = r_time.copy()
        w_prior = np.zeros(wavelet_length)
        w_prior[wavelet_length // 2] = 1.0

        for loss_type in ("l2", "student_t"):
            with self.subTest(loss_type=loss_type):
                W, diagnostics = time_varying_wavelet_inversion(
                    r_time=r_time,
                    s_obs=s_obs,
                    wavelet_length=wavelet_length,
                    data_window_length=15,
                    w_prior=w_prior,
                    use_stationary_prior=False,
                    mu_edge=0.5,
                    edge_fraction=0.20,
                    estimate_step_samples=2,
                    energy_threshold=0.0,
                    reject_ill_conditioned=False,
                    reject_amplitude_jumps=False,
                    peak_lock=False,
                    time_smooth_sigma=0.0,
                    wavelet_smooth_sigma=0.0,
                    loss_type=loss_type,
                    irls_max_iter=3,
                    return_diagnostics=True,
                    verbose=False,
                )
                self.assertTrue(np.all(np.isfinite(W)))
                self.assertTrue(diagnostics["edge_penalty_active"])
                self.assertEqual(
                    diagnostics["edge_penalty_weights"].shape,
                    (wavelet_length,),
                )
                if loss_type == "student_t":
                    robust_attempted = diagnostics["robust_attempted"]
                    self.assertTrue(np.any(robust_attempted))
                    self.assertTrue(
                        np.all(
                            np.isfinite(
                                diagnostics["student_objective_initial"][
                                    robust_attempted
                                ]
                            )
                        )
                    )


class CandidateOriginTest(unittest.TestCase):
    """检查直接反演、插值和外推来源严格互斥。"""

    def test_candidate_origin_masks_are_mutually_exclusive(self):
        valid_mask = np.array(
            [False, False, True, False, True, False, False],
            dtype=bool,
        )
        origin = _build_candidate_wavelet_origin_masks(
            valid_mask=valid_mask,
            prior_available=True,
        )
        np.testing.assert_array_equal(
            origin["origin_code"],
            np.array(
                [
                    ORIGIN_EXTRAPOLATED,
                    ORIGIN_EXTRAPOLATED,
                    ORIGIN_INVERTED,
                    ORIGIN_INTERPOLATED,
                    ORIGIN_INVERTED,
                    ORIGIN_EXTRAPOLATED,
                    ORIGIN_EXTRAPOLATED,
                ],
                dtype=np.int8,
            ),
        )
        origin_count = (
            origin["inverted_mask"].astype(int)
            + origin["interpolated_mask"].astype(int)
            + origin["extrapolated_mask"].astype(int)
            + origin["fallback_prior_mask"].astype(int)
            + origin["unavailable_mask"].astype(int)
        )
        np.testing.assert_array_equal(origin_count, 1)


class DiagnosticSemanticsTest(unittest.TestCase):
    """检查边界峰、中心间断时长和最终混合来源的诊断口径。"""

    def test_boundary_peak_mask(self):
        W = np.zeros((3, 9), dtype=float)
        W[0, 0] = 1.0
        W[1, 4] = 1.0
        W[2, 8] = -1.0
        boundary_mask = compute_boundary_peak_mask(W, boundary_fraction=0.20)
        np.testing.assert_array_equal(
            boundary_mask,
            np.array([True, False, True]),
        )

    def test_longest_gap_uses_center_step_not_seismic_dt(self):
        center_times_s = np.array([0.000, 0.005, 0.010, 0.015, 0.020])
        gap_mask = np.array([False, True, True, False, True])
        duration_ms = longest_true_gap_duration_ms(
            gap_mask_at_centers=gap_mask,
            center_times_s=center_times_s,
            fallback_center_step_s=0.005,
        )
        self.assertAlmostEqual(duration_ms, 10.0, places=12)

    def test_final_source_codes_keep_blend_separate(self):
        source_code, source_masks = build_final_wavelet_source_codes(
            base_model_type="W_est_best_no_Q",
            local_fallback_alpha=np.array([1.0, 0.5, 0.0]),
        )
        np.testing.assert_array_equal(
            source_code,
            np.array(
                [
                    FINAL_SOURCE_TV_CANDIDATE,
                    FINAL_SOURCE_LOCAL_BLEND,
                    FINAL_SOURCE_LOCAL_FALLBACK_PRIOR,
                ],
                dtype=np.int8,
            ),
        )
        source_count = np.zeros(3, dtype=int)
        for source_mask in source_masks.values():
            source_count += source_mask.astype(int)
        np.testing.assert_array_equal(source_count, 1)

    def test_stationary_rejection_keeps_unlocked_candidate_diagnostics(self):
        n_time = 21
        wavelet_length = 9
        r_time = np.zeros(n_time)
        r_time[n_time // 2] = 1.0
        s_obs = r_time.copy()
        candidate_wavelet = np.zeros(wavelet_length)
        candidate_wavelet[-1] = 1.0
        candidate_synthetic = np.convolve(
            r_time,
            candidate_wavelet,
            mode="same",
        )

        candidate_result = StationaryResult(
            w=candidate_wavelet,
            s_syn_raw=candidate_synthetic,
            s_syn=candidate_synthetic,
            cc=0.25,
            env_cc=0.30,
            maxlag_cc=0.40,
            best_lag_ms=0.0,
            wavelet_length_pts=wavelet_length,
            data_window_length=27,
            alignment="center",
            params={},
        )

        def stationary_runner(**kwargs):
            if kwargs["peak_lock"]:
                raise ValueError(
                    "平稳子波主峰位置不满足约束: peak_metric=4 samples"
                )
            return candidate_result

        prior = build_stationary_prior_with_fallback(
            r_time=r_time,
            s_obs=s_obs,
            dt=0.001,
            f_dom=30.0,
            wavelet_length_s=0.009,
            alignment="center",
            strict_source="strict_stationary_center_after_DTW",
            fallback_source="center_ricker_prior_after_DTW",
            stationary_runner=stationary_runner,
            verbose=False,
        )
        self.assertFalse(prior.stationary_candidate_accepted)
        self.assertEqual(prior.stationary_rejection_code, "peak_out_of_range")
        self.assertIs(prior.stationary_candidate_result, candidate_result)
        self.assertAlmostEqual(prior.stationary_candidate_peak_ms, 4.0)
        self.assertEqual(prior.stationary_prior_source, "center_ricker_prior_after_DTW")


if __name__ == "__main__":
    unittest.main()
