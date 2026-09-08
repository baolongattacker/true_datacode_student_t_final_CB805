# -*- coding: utf-8 -*-
"""Round 7 单元测试：直接验证增广系统 zero-block removal 的数值等价性。

测试内容：
1. Test A: 普通 L2 路径（sample_weights=None）
   对比包含全零行块（legacy: 0*I 与 0*C_dc）与不包含全零行块（new: absent block）的 SVD 解，
   要求逐元素差异满足 rtol=1e-12, atol=1e-12。
2. Test B: Student-t 加权路径（sample_weights 显式提供）
   验证 IRLS 加权卷积数据块下，移除零块依然保持 rtol=1e-12, atol=1e-12。
3. Test C: 正则目标函数数值等价性
   验证 _regularization_value 在 mu1=0, mu_dc=0 时与旧版展开计算完全一致。
4. Test D: 非负及有限性参数校验
   验证 negative, NaN, Inf, non-scalar 正则参数被明确拒绝并抛出 ValueError。
"""

from __future__ import annotations

import unittest
import numpy as np

from utils.wavelet_inversion_robust import (
    _safe_svd_solve,
    _second_derivative_matrix,
    _solve_regularized_window,
    _student_t_weights,
    _regularization_value,
    _validate_nonnegative_scalar,
    build_edge_penalty_weights,
)


class TestTvZeroBlockRemoval(unittest.TestCase):
    """验证零矩阵块物理移除后的严格数值回归。"""

    def setUp(self) -> None:
        np.random.seed(20260908)
        self.wavelet_length = 129
        self.data_window_length = 387
        self.damping_ratio = 1e-3
        self.svd_cutoff_ratio = 1e-4

        # 构造局部卷积矩阵 R 与观测地震窗口 s_win
        self.R = np.random.randn(self.data_window_length, self.wavelet_length)
        self.s_win = np.random.randn(self.data_window_length)
        self.data_scale = float(np.sum(self.s_win ** 2) / self.data_window_length)

        # 基础算子
        self.I = np.eye(self.wavelet_length, dtype=float)
        self.D = _second_derivative_matrix(self.wavelet_length)
        self.C_dc = np.ones((1, self.wavelet_length), dtype=float) / np.sqrt(self.wavelet_length)

        # 先验、前一时刻子波、边缘权重
        self.prior_i = np.random.randn(self.wavelet_length)
        self.previous_wavelet = np.random.randn(self.wavelet_length)
        self.edge_weights = build_edge_penalty_weights(
            wavelet_length=self.wavelet_length,
            edge_fraction=0.12,
            taper="cosine",
        )
        self.E_edge = np.diag(self.edge_weights)

        # 生产参数
        self.mu1 = 0.0
        self.mu2 = 3.0
        self.mu_dc = 0.0
        self.mu_prior = 1.0
        self.mu_time = 8.0
        self.mu_edge = 1.0

    def test_a_l2_path_zero_block_removal(self) -> None:
        """Test A — 普通 L2 路径 (sample_weights=None)。"""
        # 1. 使用当前函数求解 (new: 物理移除 mu1 和 mu_dc 块)
        w_new = _solve_regularized_window(
            R=self.R,
            s_win=self.s_win,
            data_scale=self.data_scale,
            I=self.I,
            D=self.D,
            C_dc=self.C_dc,
            mu1=self.mu1,
            mu2=self.mu2,
            mu_dc=self.mu_dc,
            prior_i=self.prior_i,
            mu_prior=self.mu_prior,
            previous_wavelet=self.previous_wavelet,
            mu_time=self.mu_time,
            damping_ratio=self.damping_ratio,
            svd_cutoff_ratio=self.svd_cutoff_ratio,
            sample_weights=None,
            edge_weights=self.edge_weights,
            mu_edge=self.mu_edge,
        )

        # 2. 显式构造旧版带零块的增广矩阵 (legacy: 1031 rows)
        A_old = np.vstack([
            self.R,
            np.sqrt(self.mu1 * self.data_scale) * self.I,      # 全零块 (129, 129)
            np.sqrt(self.mu2 * self.data_scale) * self.D,      # 二阶差分 (127, 129)
            np.sqrt(self.mu_dc * self.data_scale) * self.C_dc,  # 全零块 (1, 129)
            np.sqrt(self.mu_prior * self.data_scale) * self.I, # prior (129, 129)
            np.sqrt(self.mu_time * self.data_scale) * self.I,  # time (129, 129)
            np.sqrt(self.mu_edge * self.data_scale) * self.E_edge, # edge (129, 129)
        ])
        b_old = np.concatenate([
            self.s_win,
            np.zeros(self.I.shape[0]),
            np.zeros(self.D.shape[0]),
            np.zeros(self.C_dc.shape[0]),
            np.sqrt(self.mu_prior * self.data_scale) * self.prior_i,
            np.sqrt(self.mu_time * self.data_scale) * self.previous_wavelet,
            np.zeros(self.I.shape[0]),
        ])

        self.assertEqual(A_old.shape, (1031, 129))
        self.assertEqual(b_old.shape, (1031,))

        w_old = _safe_svd_solve(
            A=A_old,
            b=b_old,
            damping_ratio=self.damping_ratio,
            svd_cutoff_ratio=self.svd_cutoff_ratio,
        )

        # 3. 严格逐元素对比 (rtol=1e-12, atol=1e-12)
        max_abs_diff = float(np.max(np.abs(w_new - w_old)))
        max_rel_diff = float(np.max(np.abs(w_new - w_old) / (np.abs(w_old) + 1e-15)))

        np.testing.assert_allclose(
            w_new,
            w_old,
            rtol=1e-12,
            atol=1e-12,
            err_msg=f"Test A 失败: 最大绝对误差={max_abs_diff:.3e}, 最大相对误差={max_rel_diff:.3e}",
        )

    def test_b_student_t_weighted_path_zero_block_removal(self) -> None:
        """Test B — Student-t IRLS 加权路径 (sample_weights != None)。"""
        # 生成一组典型的 IRLS 权重（处于 (0.01, 1.0] 区间）
        weights = np.random.uniform(0.01, 1.0, size=self.data_window_length)

        # 1. 使用当前函数求解 (new)
        w_new = _solve_regularized_window(
            R=self.R,
            s_win=self.s_win,
            data_scale=self.data_scale,
            I=self.I,
            D=self.D,
            C_dc=self.C_dc,
            mu1=self.mu1,
            mu2=self.mu2,
            mu_dc=self.mu_dc,
            prior_i=self.prior_i,
            mu_prior=self.mu_prior,
            previous_wavelet=self.previous_wavelet,
            mu_time=self.mu_time,
            damping_ratio=self.damping_ratio,
            svd_cutoff_ratio=self.svd_cutoff_ratio,
            sample_weights=weights,
            edge_weights=self.edge_weights,
            mu_edge=self.mu_edge,
        )

        # 2. 显式加权数据块并拼入 legacy 零块
        sqrt_w = np.sqrt(weights)
        R_weighted = sqrt_w[:, None] * self.R
        s_weighted = sqrt_w * self.s_win

        A_old = np.vstack([
            R_weighted,
            np.sqrt(self.mu1 * self.data_scale) * self.I,
            np.sqrt(self.mu2 * self.data_scale) * self.D,
            np.sqrt(self.mu_dc * self.data_scale) * self.C_dc,
            np.sqrt(self.mu_prior * self.data_scale) * self.I,
            np.sqrt(self.mu_time * self.data_scale) * self.I,
            np.sqrt(self.mu_edge * self.data_scale) * self.E_edge,
        ])
        b_old = np.concatenate([
            s_weighted,
            np.zeros(self.I.shape[0]),
            np.zeros(self.D.shape[0]),
            np.zeros(self.C_dc.shape[0]),
            np.sqrt(self.mu_prior * self.data_scale) * self.prior_i,
            np.sqrt(self.mu_time * self.data_scale) * self.previous_wavelet,
            np.zeros(self.I.shape[0]),
        ])

        w_old = _safe_svd_solve(
            A=A_old,
            b=b_old,
            damping_ratio=self.damping_ratio,
            svd_cutoff_ratio=self.svd_cutoff_ratio,
        )

        max_abs_diff = float(np.max(np.abs(w_new - w_old)))
        max_rel_diff = float(np.max(np.abs(w_new - w_old) / (np.abs(w_old) + 1e-15)))

        np.testing.assert_allclose(
            w_new,
            w_old,
            rtol=1e-12,
            atol=1e-12,
            err_msg=f"Test B 失败: 最大绝对误差={max_abs_diff:.3e}, 最大相对误差={max_rel_diff:.3e}",
        )

    def test_b2_production_student_t_weights_path(self) -> None:
        """Test B2 — 生产级真实 Student-t 权重的加权最小二乘路径。"""
        # 使用初始残差与局部尺度估计生成实际生产形态的 Student-t 权重
        w_init = np.random.randn(self.wavelet_length)
        residual = self.s_win - self.R @ w_init
        mad = float(np.median(np.abs(residual - np.median(residual))))
        sigma = max(1.4826 * mad, 1e-6)
        student_weights = _student_t_weights(
            residual=residual,
            sigma=sigma,
            nu=10.0,
            weight_floor=1e-3,
        )

        # 1. 使用当前函数求解 (new: 901 rows)
        w_new = _solve_regularized_window(
            R=self.R,
            s_win=self.s_win,
            data_scale=self.data_scale,
            I=self.I,
            D=self.D,
            C_dc=self.C_dc,
            mu1=self.mu1,
            mu2=self.mu2,
            mu_dc=self.mu_dc,
            prior_i=self.prior_i,
            mu_prior=self.mu_prior,
            previous_wavelet=self.previous_wavelet,
            mu_time=self.mu_time,
            damping_ratio=self.damping_ratio,
            svd_cutoff_ratio=self.svd_cutoff_ratio,
            sample_weights=student_weights,
            edge_weights=self.edge_weights,
            mu_edge=self.mu_edge,
        )

        # 2. 显式构造带 130 行零块的旧版增广系统 (legacy: 1031 rows)
        sqrt_w = np.sqrt(student_weights)
        R_weighted = sqrt_w[:, None] * self.R
        s_weighted = sqrt_w * self.s_win

        A_old = np.vstack([
            R_weighted,
            np.sqrt(self.mu1 * self.data_scale) * self.I,
            np.sqrt(self.mu2 * self.data_scale) * self.D,
            np.sqrt(self.mu_dc * self.data_scale) * self.C_dc,
            np.sqrt(self.mu_prior * self.data_scale) * self.I,
            np.sqrt(self.mu_time * self.data_scale) * self.I,
            np.sqrt(self.mu_edge * self.data_scale) * self.E_edge,
        ])
        b_old = np.concatenate([
            s_weighted,
            np.zeros(self.I.shape[0]),
            np.zeros(self.D.shape[0]),
            np.zeros(self.C_dc.shape[0]),
            np.sqrt(self.mu_prior * self.data_scale) * self.prior_i,
            np.sqrt(self.mu_time * self.data_scale) * self.previous_wavelet,
            np.zeros(self.I.shape[0]),
        ])

        w_old = _safe_svd_solve(
            A=A_old,
            b=b_old,
            damping_ratio=self.damping_ratio,
            svd_cutoff_ratio=self.svd_cutoff_ratio,
        )

        max_abs_diff = float(np.max(np.abs(w_new - w_old)))
        max_rel_diff = float(np.max(np.abs(w_new - w_old) / (np.abs(w_old) + 1e-15)))

        np.testing.assert_allclose(
            w_new,
            w_old,
            rtol=1e-12,
            atol=1e-12,
            err_msg=f"Test B2 失败: 最大绝对误差={max_abs_diff:.3e}, 最大相对误差={max_rel_diff:.3e}",
        )

    def test_c_regularization_objective_equivalence(self) -> None:
        """Test C — 目标函数值计算完全等价性。"""
        w_test = np.random.randn(self.wavelet_length)

        # new 逻辑
        val_new = _regularization_value(
            w=w_test,
            data_scale=self.data_scale,
            D=self.D,
            C_dc=self.C_dc,
            mu1=self.mu1,
            mu2=self.mu2,
            mu_dc=self.mu_dc,
            prior_i=self.prior_i,
            mu_prior=self.mu_prior,
            previous_wavelet=self.previous_wavelet,
            mu_time=self.mu_time,
            edge_weights=self.edge_weights,
            mu_edge=self.mu_edge,
        )

        # legacy 逻辑（展开所有项，包括 0 * ||w||^2 和 0 * ||C_dc w||^2）
        val_old_inner = (
            float(self.mu1) * float(np.sum(w_test ** 2))
            + float(self.mu2) * float(np.sum((self.D @ w_test) ** 2))
            + float(self.mu_dc) * float(np.sum((self.C_dc @ w_test) ** 2))
            + float(self.mu_prior) * float(np.sum((w_test - self.prior_i) ** 2))
            + float(self.mu_time) * float(np.sum((w_test - self.previous_wavelet) ** 2))
            + float(self.mu_edge) * float(np.sum((self.edge_weights * w_test) ** 2))
        )
        val_old = 0.5 * float(self.data_scale) * val_old_inner

        self.assertAlmostEqual(val_new, val_old, delta=1e-15)

    def test_d_nonnegative_parameter_validation(self) -> None:
        """Test D — 非负与有限标量参数校验。"""
        # 合法输入应原样返回 float
        self.assertEqual(_validate_nonnegative_scalar(0.0, "mu1"), 0.0)
        self.assertEqual(_validate_nonnegative_scalar(3, "mu2"), 3.0)
        self.assertEqual(_validate_nonnegative_scalar(np.float64(8.0), "mu_time"), 8.0)

        # 负数必须抛出 ValueError
        with self.assertRaises(ValueError):
            _validate_nonnegative_scalar(-0.1, "mu1")

        # NaN 必须抛出 ValueError
        with self.assertRaises(ValueError):
            _validate_nonnegative_scalar(float("nan"), "mu1")

        # Inf 必须抛出 ValueError
        with self.assertRaises(ValueError):
            _validate_nonnegative_scalar(float("inf"), "mu2")

        # 数组 / 非标量必须抛出 ValueError
        with self.assertRaises(ValueError):
            _validate_nonnegative_scalar([1.0, 2.0], "mu_dc")  # type: ignore[arg-type]

    def test_e_nonzero_regularization_identity(self) -> None:
        """Test E — 第三层测试：非零正则下绝对不能发生行为变化 (mu1=0.2, mu_dc=15.0)。"""
        mu1_nonzero = 0.2
        mu_dc_nonzero = 15.0

        # 1. 使用当前修改后的 _solve_regularized_window (触发 mu1 > 0 和 mu_dc > 0 追加分支)
        w_new = _solve_regularized_window(
            R=self.R,
            s_win=self.s_win,
            data_scale=self.data_scale,
            I=self.I,
            D=self.D,
            C_dc=self.C_dc,
            mu1=mu1_nonzero,
            mu2=self.mu2,
            mu_dc=mu_dc_nonzero,
            prior_i=self.prior_i,
            mu_prior=self.mu_prior,
            previous_wavelet=self.previous_wavelet,
            mu_time=self.mu_time,
            damping_ratio=self.damping_ratio,
            svd_cutoff_ratio=self.svd_cutoff_ratio,
            sample_weights=None,
            edge_weights=self.edge_weights,
            mu_edge=self.mu_edge,
        )

        # 2. 显式构造旧版非零增广矩阵，严格保持原 block 顺序：
        #    data -> mu1 -> mu2 -> mu_dc -> prior -> time -> edge
        A_old = np.vstack([
            self.R,
            np.sqrt(mu1_nonzero * self.data_scale) * self.I,
            np.sqrt(self.mu2 * self.data_scale) * self.D,
            np.sqrt(mu_dc_nonzero * self.data_scale) * self.C_dc,
            np.sqrt(self.mu_prior * self.data_scale) * self.I,
            np.sqrt(self.mu_time * self.data_scale) * self.I,
            np.sqrt(self.mu_edge * self.data_scale) * self.E_edge,
        ])
        b_old = np.concatenate([
            self.s_win,
            np.zeros(self.I.shape[0]),
            np.zeros(self.D.shape[0]),
            np.zeros(self.C_dc.shape[0]),
            np.sqrt(self.mu_prior * self.data_scale) * self.prior_i,
            np.sqrt(self.mu_time * self.data_scale) * self.previous_wavelet,
            np.zeros(self.I.shape[0]),
        ])

        w_old = _safe_svd_solve(
            A=A_old,
            b=b_old,
            damping_ratio=self.damping_ratio,
            svd_cutoff_ratio=self.svd_cutoff_ratio,
        )

        max_abs_diff = float(np.max(np.abs(w_new - w_old)))
        max_rel_diff = float(np.max(np.abs(w_new - w_old) / (np.abs(w_old) + 1e-15)))

        np.testing.assert_allclose(
            w_new,
            w_old,
            rtol=1e-12,
            atol=1e-12,
            err_msg=f"Test E 失败: 最大绝对误差={max_abs_diff:.3e}, 最大相对误差={max_rel_diff:.3e}",
        )


if __name__ == "__main__":
    unittest.main()
