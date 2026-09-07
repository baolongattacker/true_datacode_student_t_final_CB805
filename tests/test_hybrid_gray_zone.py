# -*- coding: utf-8 -*-
"""固定输入检查：gray-zone 分类、硬回退和主流程输出口径；不代表完整 DTW 验证。"""
import ast
import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from configs.config_loader import load_config
from experiments import run_real_experiment as runner
from plotting.plot_wavelets import _print_selected_time_qc


def test_direct_partition_and_nonfinite_rejection():
    """输入为 8 个时间采样，峰值 ms、能量无量纲；输出检查三类互斥及异常拒绝。"""
    admissible, gray, rejected = runner.build_admissible_direct_masks(
        valid_mask=np.array([1, 1, 1, 1, 1, 1, 1, 0], dtype=bool),
        strict_reliable_mask=np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=bool),
        peak_metric_ms=np.array([0, 15, 16, np.nan, 0, 0, 0, 0]),
        wavelet_energy_norm=np.array([1, .15, 1, 1, np.nan, 1, 1, 1]),
        pointwise_fallback_request_mask=np.array([0, 0, 0, 0, 0, 1, 0, 0], dtype=bool),
        provenance_forced_prior_mask=np.array([0, 0, 0, 0, 0, 0, 1, 0], dtype=bool),
        peak_limit_ms=15, energy_min=.15,
    )
    np.testing.assert_array_equal(np.flatnonzero(admissible), [0, 1])
    np.testing.assert_array_equal(np.flatnonzero(gray), [1])
    np.testing.assert_array_equal(np.flatnonzero(rejected), [2, 3, 4, 5, 6])


def test_direct_shape_and_strict_invariants():
    """一维单采样输入，无量纲；错误维度和 strict/hard 冲突必须显式报错。"""
    inputs = dict(
        valid_mask=[True], strict_reliable_mask=[True], peak_metric_ms=[0.0],
        wavelet_energy_norm=[1.0], pointwise_fallback_request_mask=[False],
        provenance_forced_prior_mask=[False], peak_limit_ms=15.0, energy_min=.15,
    )
    with pytest.raises(ValueError, match="shape"):
        runner.build_admissible_direct_masks(**dict(inputs, peak_metric_ms=[[0.0]]))
    with pytest.raises(AssertionError, match="strict"):
        runner.build_admissible_direct_masks(**dict(inputs, pointwise_fallback_request_mask=[True]))


@pytest.mark.parametrize("enabled", [True, False])
def test_gray_bridge_keeps_rejected_and_provenance_in_prior(enabled):
    """合成 W shape (13,129)，dt=0.01 s；检查短间隙、低能量及来源硬约束。"""
    cfg = load_config(str(Path(__file__).resolve().parents[1] / "configs/cb805_center_student_t_L129_v01.yaml"))
    local = runner._get_local_wavelet_fallback_config(cfg)
    local["enable"] = enabled
    local["alpha_smooth_samples"] = 1.0
    # 主峰居中；单侧次峰 0.8 位于 20 ms，属于 strict 与 hard 之间的 gray。
    w_prior = np.zeros(129)  # shape: (N_wavelet,)，相对振幅
    w_prior[64] = 1.0
    W = np.tile(w_prior, (13, 1))  # shape: (N_time, N_wavelet)
    W[4, 66] = .8
    W[2] *= .01  # 两锚点之间低能量 direct，不能被插值重新救回。
    W[9, 66] = .96  # hard shape reject
    valid = np.zeros(13, dtype=bool)  # shape: (N_time,)
    valid[[0, 2, 4, 9, 12]] = True
    forced = np.zeros(13, dtype=bool)  # shape: (N_time,)
    forced[3] = True
    tv = SimpleNamespace(W_best=W, valid_mask=valid, use_negative=False, diag={
        "estimate_step_samples": 1, "time_smooth_sigma": 3.0,
        "candidate_prior_fill_mask": forced,
    })
    hybrid = runner.build_preacceptance_hybrid_candidate(
        tv=tv, w_prior=w_prior, dt=.01, alignment="center", local_fallback_cfg=local,
    )
    np.testing.assert_array_equal(np.flatnonzero(hybrid["admissible_direct_mask"]), [0, 4, 12])
    np.testing.assert_array_equal(np.flatnonzero(hybrid["gray_zone_direct_mask"]), [4])
    np.testing.assert_array_equal(np.flatnonzero(hybrid["rejected_direct_mask"]), [2, 9])
    np.testing.assert_array_equal(np.flatnonzero(hybrid["short_gap_mask"]), [1])
    assert hybrid["proposed_fallback_mask"][2:4].all()
    assert hybrid["proposed_fallback_mask"][5:12].all()  # 80 ms 超出 60 ms
    assert hybrid["strict_reliable_center_ratio"] == 2 / 5
    assert hybrid["strict_reliable_sample_ratio"] == 2 / 13
    if enabled:
        mask = hybrid["proposed_fallback_mask"]
        np.testing.assert_array_equal(hybrid["alpha"][mask], 0.0)
        np.testing.assert_array_equal(hybrid["W_hybrid"][mask], np.tile(w_prior, (mask.sum(), 1)))
    else:
        np.testing.assert_array_equal(hybrid["W_hybrid"], W)
        np.testing.assert_array_equal(hybrid["alpha"], 1.0)
        assert not hybrid["applied_fallback_mask"].any()


def test_no_valid_centers_has_zero_ratios():
    """无有效中心时输出零比例，不产生 NaN；W shape (3,129)，dt 单位 s。"""
    cfg = load_config(str(Path(__file__).resolve().parents[1] / "configs/cb805_center_student_t_L129_v01.yaml"))
    w_prior = np.zeros(129)  # shape: (N_wavelet,)
    w_prior[64] = 1.0
    tv = SimpleNamespace(W_best=np.tile(w_prior, (3, 1)), valid_mask=np.zeros(3, dtype=bool), diag={}, use_negative=False)
    hybrid = runner.build_preacceptance_hybrid_candidate(
        tv=tv, w_prior=w_prior, dt=.001, alignment="center",
        local_fallback_cfg=runner._get_local_wavelet_fallback_config(cfg),
    )
    assert hybrid["valid_direct_count"] == 0
    assert hybrid["admissible_direct_center_ratio"] == 0.0
    assert hybrid["hard_prior_mask"].all()


@pytest.mark.parametrize("passed", [True, False])
def test_actual_metrics_and_bundle_reason_wiring(passed):
    """执行主流程真实 metrics 表达式；区分 raw/hybrid/final，峰值及 lag 单位 ms。

    仅抽取输出装配语句，避免为字段回归重复运行 DTW；没有复制生产字段赋值。
    """
    tree = ast.parse(inspect.getsource(runner._main_impl))
    acceptance = SimpleNamespace(passed=passed, reasons=[] if passed else ["hybrid_failed"])
    tv = SimpleNamespace(cc_direct=.44, W_pass=False, best_lag_ms=-8., peak_metric_p10=-64.,
                         peak_metric_med=50., peak_metric_p90=64., peak_abs_p90=64.,
                         valid_ratio=.03, use_negative=False, acceptance_reasons=["raw_failed"])
    namespace = dict(
        np=np, build_metrics_dict=runner.build_metrics_dict, tv=tv,
        initial_prior=SimpleNamespace(cc=.1), after_prior=SimpleNamespace(cc=.16),
        dtw=SimpleNamespace(cc_after=.15), q_result=SimpleNamespace(cc_q=np.nan, Q_global=np.nan, reason="disabled"),
        q_pass=False, CC_final=.12, final_model_type="stationary_prior",
        hybrid_W_pass=passed, hybrid_acceptance=acceptance,
        hybrid_similarity={"cc_direct": .288, "best_lag_ms": 0.},
        hybrid_peak_summary={"p10": -1., "median": 0., "p90": 0., "abs_p90": 1.},
    )
    for statement in tree.body[0].body:
        if isinstance(statement, ast.Assign) and isinstance(statement.value, ast.Call):
            if isinstance(statement.value.func, ast.Name) and statement.value.func.id == "build_metrics_dict":
                exec(compile(ast.Module(body=[statement], type_ignores=[]), "metrics_wiring", "exec"), namespace)
    metrics = namespace["metrics"]
    # 同时执行实际 scope/alias/raw 字段；若被后续同名字段覆盖，这里也会捕获。
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values):
            if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                continue
            name = key.value
            if name.startswith(("raw_tv_", "hybrid_acceptance_", "post_local_fallback_acceptance_")) or name in (
                "W_pass_scope", "valid_ratio_scope", "acceptance_reasons_scope", "peak_metric_scope", "best_lag_scope",
            ):
                metrics[name] = eval(compile(ast.Expression(value), "metric_field", "eval"), namespace)
    assert metrics["W_pass"] == passed
    assert metrics["CC_tv_direct"] == .288
    assert metrics["CC_final"] == .12
    assert metrics["best_lag_ms"] == 0.
    assert [metrics[k] for k in ("peak_metric_p10", "peak_metric_med", "peak_metric_p90", "peak_abs_p90")] == [-1., 0., 0., 1.]
    assert metrics["acceptance_reasons"] == acceptance.reasons
    assert metrics["raw_tv_peak_abs_p90"] == 64.
    assert metrics["raw_tv_acceptance_reasons"] == ["raw_failed"]
    assert metrics["post_local_fallback_acceptance_passed"] == passed
    assert metrics["post_local_fallback_acceptance_reasons"] == acceptance.reasons
    assert metrics["W_pass_scope"] == "preacceptance_hybrid_candidate"
    assert metrics["post_local_fallback_acceptance_scope"] == "legacy_alias_of_decisive_preacceptance_hybrid_gate"
    save_tree = ast.parse(inspect.getsource(runner._save_and_plot_results))
    for node in ast.walk(save_tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "save_result_bundle":
            for keyword in node.keywords:
                if keyword.arg == "acceptance_reasons":
                    saved_reasons = eval(compile(ast.Expression(keyword.value), "bundle_reasons", "eval"), namespace)
    assert saved_reasons == metrics["acceptance_reasons"]


def test_selected_time_log_separates_candidate_and_final(capsys):
    """单采样日志检查，无量纲状态；final 可靠并不意味着该点曾直接反演。"""
    _print_selected_time_qc(
        figure_name="test", target_times_s=[1.], selected_indices=[0], raw_indices=[0], t_work=[1.],
        valid_mask=[False], reliable_mask=[True], fallback_mask=[True], fallback_alpha=[0.],
        candidate_diagnostics={"tv_strict_reliable_direct_mask": [False], "tv_admissible_direct_mask": [False],
                               "tv_gray_zone_direct_mask": [False], "tv_rejected_direct_mask": [False],
                               "tv_hybrid_support_mask": [False], "tv_hybrid_applied_fallback_mask": [True]},
    )
    output = capsys.readouterr().out
    for field in ("direct_inverted=0", "strict_reliable=0", "gray_zone=0", "hybrid_support=0", "final_shape_reliable=1", "fallback=1"):
        assert field in output
