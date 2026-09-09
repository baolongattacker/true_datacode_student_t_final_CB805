# -*- coding: utf-8 -*-
"""
单测套件：验证 Soft Fallback v2 / v2.1 / v2.2 核心算法机制。
覆盖：
1. 平滑步阶、连续物理可信度与加权几何平均
2. 支撑骨架锚点解耦与短间隙桥接
3. Soft Fallback Guard 判定指标
4. v2.1 Provenance Cap (来源上限高斯衰减、外推压制)
5. v2.2 Boundary Amplitude Ratio 与三级 Shape Veto Cap
6. 关键安全回归：beta=0 绝不能绕过 origin_cap 与 shape_cap 物理上限
"""
import sys
from pathlib import Path
import unittest
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_real_experiment import (
    _smoothstep01,
    _reliability_low_is_good,
    _reliability_high_is_good,
    _weighted_geometric_reliability,
    _temporal_wavelet_difference_energy,
    _build_hard_transition_cap,
    _compose_wavelet_matrix,
    build_alpha_for_strength,
    evaluate_soft_fallback_guard,
    build_soft_support_anchor_mask,
    build_soft_fallback_reliability,
    apply_final_model_guard,
    build_origin_alpha_cap,
    compute_boundary_amplitude_ratio,
    build_shape_alpha_cap,
)


class TestSoftFallbackV2(unittest.TestCase):
    def test_smoothstep01(self):
        """验证 smoothstep01 边界值、单调性及导数连续性。"""
        self.assertAlmostEqual(_smoothstep01(np.array([0.0]))[0], 0.0)
        self.assertAlmostEqual(_smoothstep01(np.array([1.0]))[0], 1.0)
        self.assertAlmostEqual(_smoothstep01(np.array([0.5]))[0], 0.5)
        self.assertAlmostEqual(_smoothstep01(np.array([-0.5]))[0], 0.0)
        self.assertAlmostEqual(_smoothstep01(np.array([1.5]))[0], 1.0)

        # 检查单调性
        x = np.linspace(0, 1, 100)
        y = _smoothstep01(x)
        self.assertTrue(np.all(np.diff(y) >= 0))

    def test_reliability_low_is_good(self):
        """验证低值优秀（如旁瓣、边缘能量）的平滑过渡。"""
        vals = np.array([0.70, 0.75, 0.85, 0.95, 1.10, np.nan])
        scores = _reliability_low_is_good(vals, 0.75, 0.95, 0.0)
        self.assertAlmostEqual(scores[0], 1.0)
        self.assertAlmostEqual(scores[1], 1.0)
        self.assertAlmostEqual(scores[2], 0.5)
        self.assertAlmostEqual(scores[3], 0.0)
        self.assertAlmostEqual(scores[4], 0.0)
        self.assertAlmostEqual(scores[5], 0.0)  # NaN 映射为 floor

    def test_reliability_high_is_good(self):
        """验证高值优秀（如子波能量）的平滑过渡。"""
        vals = np.array([0.10, 0.15, 0.50, 0.80])
        scores = _reliability_high_is_good(vals, 0.15, 0.50, 0.1)
        self.assertAlmostEqual(scores[0], 0.1)
        self.assertAlmostEqual(scores[1], 0.1)
        self.assertAlmostEqual(scores[2], 1.0)
        self.assertAlmostEqual(scores[3], 1.0)

    def test_weighted_geometric_reliability(self):
        """验证加权几何平均的综合评分性质（避免一票否决）。"""
        scores = {
            "sl": np.array([1.0, 0.8, 0.5]),
            "ee": np.array([1.0, 0.9, 0.5]),
        }
        weights = {"sl": 1.0, "ee": 1.0}
        geo = _weighted_geometric_reliability(scores, weights)
        self.assertAlmostEqual(geo[0], 1.0)
        self.assertAlmostEqual(geo[1], np.sqrt(0.8 * 0.9))
        self.assertAlmostEqual(geo[2], 0.5)

    def test_hard_transition_cap(self):
        """验证欧氏距离变换与平滑 Hard cap 边界。"""
        hard_mask = np.zeros(10, dtype=bool)
        hard_mask[0:2] = True
        cap = _build_hard_transition_cap(hard_mask, transition_samples=4.0)

        # hard 区域内必须为 0.0
        self.assertAlmostEqual(cap[0], 0.0)
        self.assertAlmostEqual(cap[1], 0.0)
        # 随距离递增并平滑达到 1.0
        self.assertTrue(np.all(np.diff(cap[1:]) >= 0.0))
        self.assertAlmostEqual(cap[6], 1.0)

    def test_soft_support_anchor_decoupling(self):
        """核心验证：形态（旁瓣/边缘能量）不再决定 Anchor 资格，仅由有效性、有限峰值偏移和能量决定。"""
        n_time = 5
        valid_mask = np.array([True, True, True, True, True])
        peak_metric_ms = np.array([2.0, 2.0, 2.0, 25.0, 2.0])  # i=3 峰值超限
        wavelet_energy_norm = np.array([0.5, 0.5, 0.02, 0.5, 0.5])  # i=2 能量过低
        provenance_forced_prior_mask = np.array([False, False, False, False, True])  # i=4 外推/填充强制为先验

        anchor = build_soft_support_anchor_mask(
            valid_mask=valid_mask,
            peak_metric_ms=peak_metric_ms,
            wavelet_energy_norm=wavelet_energy_norm,
            provenance_forced_prior_mask=provenance_forced_prior_mask,
            peak_soft_limit_ms=18.0,
            energy_soft_min=0.05,
        )
        self.assertTrue(anchor[0])
        self.assertTrue(anchor[1])
        self.assertFalse(anchor[2])  # 能量不足
        self.assertFalse(anchor[3])  # 峰值偏移过大
        self.assertFalse(anchor[4])  # 来源先验剔除

    def test_soft_fallback_guard_metrics(self):
        """验证 guard 对 CC drop 和粗糙度比率的客观判断。"""
        class DummyAcceptance:
            def __init__(self, passed):
                self.passed = passed
                self.reasons = []

        cand_eval = {
            "similarity": {"cc_direct": 0.440},
            "similarity_score": 0.440,
            "acceptance": DummyAcceptance(True),
        }
        base_eval = {
            "similarity": {"cc_direct": 0.445},
            "similarity_score": 0.445,
            "acceptance": DummyAcceptance(True),
        }
        W_base = np.zeros((10, 21))
        W_base[:, 10] = 1.0
        W_cand = W_base.copy()

        guard = evaluate_soft_fallback_guard(
            candidate_eval=cand_eval,
            baseline_eval=base_eval,
            candidate_W=W_cand,
            baseline_W=W_base,
            max_cc_drop=0.010,
            max_similarity_drop=0.015,
            max_temporal_roughness_ratio=3.0,
        )
        self.assertTrue(guard["passed"])
        self.assertAlmostEqual(guard["cc_drop"], 0.005)

    # -------------------------------------------------------------
    # Section 16: v2.1 Provenance Cap 单元测试
    # -------------------------------------------------------------
    def test_origin_cap_non_extrapolated_is_one(self):
        """测试用例：非外推区域的 origin_cap 恒为 1.0，距离为 0。"""
        extrapolated = np.zeros(20, dtype=bool)
        direct = np.zeros(20, dtype=bool)
        direct[5:15] = True

        cap, distance = build_origin_alpha_cap(
            extrapolated_mask=extrapolated,
            valid_direct_mask=direct,
            dt=0.001,
            alpha_near=0.40,
            alpha_far=0.10,
            decay_ms=80.0,
        )

        np.testing.assert_allclose(cap, 1.0)
        np.testing.assert_allclose(distance, 0.0)

    def test_origin_cap_shallow_decay_is_monotonic(self):
        """测试用例：浅部外推区域随远离首个有效中心单调高斯衰减。"""
        n = 300
        extrapolated = np.zeros(n, dtype=bool)
        direct = np.zeros(n, dtype=bool)

        direct[200:250] = True
        extrapolated[:200] = True

        cap, _ = build_origin_alpha_cap(
            extrapolated_mask=extrapolated,
            valid_direct_mask=direct,
            dt=0.001,
            alpha_near=0.40,
            alpha_far=0.10,
            decay_ms=80.0,
        )

        self.assertTrue(cap[199] > cap[150])
        self.assertTrue(cap[150] > cap[50])
        self.assertAlmostEqual(cap[199], 0.40, delta=1e-3)
        self.assertAlmostEqual(cap[0], 0.10, delta=0.01)

    def test_origin_cap_deep_decay_is_monotonic(self):
        """测试用例：深部外推区域随远离最后一个有效中心单调衰减。"""
        n = 300
        extrapolated = np.zeros(n, dtype=bool)
        direct = np.zeros(n, dtype=bool)

        direct[50:100] = True
        extrapolated[100:] = True

        cap, _ = build_origin_alpha_cap(
            extrapolated_mask=extrapolated,
            valid_direct_mask=direct,
            dt=0.001,
            alpha_near=0.40,
            alpha_far=0.10,
            decay_ms=80.0,
        )

        self.assertTrue(cap[100] > cap[150])
        self.assertTrue(cap[150] > cap[250])

    def test_origin_cap_survives_beta_zero(self):
        """测试用例：beta=0 时仍受 origin_cap 严格压制，绝不恢复为 raw TV。"""
        alpha_soft = np.full(10, 0.3)
        hard_cap = np.ones(10)
        origin_cap = np.ones(10)
        origin_cap[:4] = 0.2

        alpha = build_alpha_for_strength(
            alpha_soft=alpha_soft,
            hard_cap=hard_cap,
            beta=0.0,
            origin_cap=origin_cap,
        )

        np.testing.assert_allclose(alpha[:4], 0.2)
        np.testing.assert_allclose(alpha[4:], 1.0)

    def test_origin_cap_no_valid_direct_is_not_one(self):
        """测试用例：没有任何直接反演中心时，外推区安全返回 alpha_far，距离为 inf。"""
        extrapolated = np.ones(20, dtype=bool)
        direct = np.zeros(20, dtype=bool)

        cap, distance = build_origin_alpha_cap(
            extrapolated_mask=extrapolated,
            valid_direct_mask=direct,
            dt=0.001,
            alpha_near=0.40,
            alpha_far=0.10,
            decay_ms=80.0,
        )

        np.testing.assert_allclose(cap, 0.10)
        self.assertTrue(np.all(np.isinf(distance)))

    # -------------------------------------------------------------
    # Section 17: v2.2 Boundary Amplitude & Shape Veto 单元测试
    # -------------------------------------------------------------
    def test_boundary_amplitude_ratio_detects_edge_ringing(self):
        """测试用例：边界振幅比能灵敏检测到子波两端的截断振荡。"""
        W = np.zeros((2, 21), dtype=float)
        W[:, 10] = 1.0

        W[0, 0] = 0.10
        W[0, -1] = 0.20

        W[1, 0] = 0.70
        W[1, -1] = 0.60

        ratio = compute_boundary_amplitude_ratio(W, boundary_fraction=0.05)

        self.assertAlmostEqual(ratio[0], 0.20, delta=1e-6)
        self.assertAlmostEqual(ratio[1], 0.70, delta=1e-6)

    def test_shape_cap_boundary_three_levels(self):
        """测试用例：三级边界振幅比阈值 (<=0.25 -> 1.0; <=0.45 -> 0.65; <0.60 -> 0.45; >=0.60 -> 0.25)。"""
        side = np.array([0.5, 0.5, 0.5, 0.5])
        edge = np.array([0.05, 0.05, 0.05, 0.05])
        boundary = np.array([0.20, 0.30, 0.50, 0.70])

        info = build_shape_alpha_cap(
            side_lobe_ratio=side,
            edge_energy_ratio=edge,
            boundary_amp_ratio=boundary,
            side_lobe_reliable=0.70,
            side_lobe_fallback=0.90,
            side_lobe_severe=1.00,
            edge_energy_reliable=0.12,
            edge_energy_fallback=0.20,
            boundary_amp_reliable=0.25,
            boundary_amp_fallback=0.45,
            boundary_amp_severe=0.60,
            unreliable_cap=0.65,
            fallback_cap=0.45,
            severe_cap=0.25,
        )

        np.testing.assert_allclose(
            info["shape_cap"],
            np.array([1.00, 0.65, 0.45, 0.25]),
        )

    def test_shape_cap_side_lobe_unreliable(self):
        """测试用例：旁瓣比处于 unreliable 区间时激活 0.65 cap。"""
        info = build_shape_alpha_cap(
            side_lobe_ratio=np.array([0.808]),
            edge_energy_ratio=np.array([0.08]),
            boundary_amp_ratio=np.array([0.23]),
            side_lobe_reliable=0.70,
            side_lobe_fallback=0.90,
            side_lobe_severe=1.00,
            edge_energy_reliable=0.12,
            edge_energy_fallback=0.20,
            boundary_amp_reliable=0.25,
            boundary_amp_fallback=0.45,
            boundary_amp_severe=0.60,
            unreliable_cap=0.65,
            fallback_cap=0.45,
            severe_cap=0.25,
        )

        self.assertAlmostEqual(info["shape_cap"][0], 0.65)

    def test_shape_cap_survives_beta_zero(self):
        """测试用例：beta=0 时仍然受 shape_cap 压制。"""
        alpha = build_alpha_for_strength(
            alpha_soft=np.array([0.99]),
            hard_cap=np.array([1.0]),
            beta=0.0,
            origin_cap=np.array([1.0]),
            shape_cap=np.array([0.25]),
        )

        self.assertAlmostEqual(alpha[0], 0.25)

    def test_invalid_shape_metric_is_not_trusted(self):
        """测试用例：包含 NaN 的未定义形态指标绝不能直接信任，必须压制为 severe_cap。"""
        info = build_shape_alpha_cap(
            side_lobe_ratio=np.array([np.nan]),
            edge_energy_ratio=np.array([0.05]),
            boundary_amp_ratio=np.array([0.10]),
            side_lobe_reliable=0.70,
            side_lobe_fallback=0.90,
            side_lobe_severe=1.00,
            edge_energy_reliable=0.12,
            edge_energy_fallback=0.20,
            boundary_amp_reliable=0.25,
            boundary_amp_fallback=0.45,
            boundary_amp_severe=0.60,
            unreliable_cap=0.65,
            fallback_cap=0.45,
            severe_cap=0.25,
        )

        self.assertAlmostEqual(info["shape_cap"][0], 0.25)

    # -------------------------------------------------------------
    # Section 18: 最关键回归测试
    # -------------------------------------------------------------
    def test_beta_zero_cannot_bypass_origin_or_shape_caps(self):
        """关键回归验证：无论 beta 为何值（即使为 0.0），程序绝不能突破 origin_cap 或 shape_cap 物理上限。"""
        alpha_soft = np.array([0.2, 0.8, 1.0])
        hard_cap = np.ones(3)
        origin_cap = np.array([0.10, 1.00, 1.00])
        shape_cap = np.array([1.00, 0.45, 0.25])

        alpha = build_alpha_for_strength(
            alpha_soft=alpha_soft,
            hard_cap=hard_cap,
            beta=0.0,
            origin_cap=origin_cap,
            shape_cap=shape_cap,
        )

        np.testing.assert_allclose(
            alpha,
            np.array([0.10, 0.45, 0.25]),
        )


if __name__ == "__main__":
    unittest.main()
