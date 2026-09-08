# -*- coding: utf-8 -*-
"""第三轮 T00–T03 离线验收；直接运行，不改变反演和前两轮结果。

使用 round3_plan.json 中运行前冻结的门槛；输出 time_acceptance.json。
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from experiments.check_lag_ablation import summarize_stage, amplitude_diagnostics
from core.forward_operator import nonstationary_convolution
from core.signal_utils import match_rms
from core.metrics import corrcoef_safe

RESULT_ROOT = ROOT / "_experiment_results/ablation/round3_time"
STAGES = ("W_direct_centers", "W_pre_gaussian", "W_post_gaussian", "W_est_best", "W_final_hybrid", "W_final")


def summarize_time_stage(W_matrix, time_s, row_mask, dt_s, epsilon):
    """输入 W:(N,L)、时间 s:(N,)、统计掩码:(N,)、dt_s(s)、正数 epsilon。

    输出阶段标量字典；峰位 ms、频率 Hz、能量为原子波振幅平方单位。
    数学作用：E_t / (mean(||w||²) + epsilon) 剔除整体能量尺度影响；
    它不是逐行单位范数归一化，仍包含相对幅度变化。不对缺测中心补零。
    """
    assert W_matrix.ndim == 2 and row_mask.shape == time_s.shape == (W_matrix.shape[0],)
    assert epsilon > 0
    summary = summarize_stage(W_matrix, time_s, row_mask, dt_s)
    summary["normalized_temporal_energy"] = summary["temporal_difference_energy"] / (summary["mean_wavelet_energy"] + epsilon)
    return summary


def check_time_ablation():
    """输入为冻结计划和四组已保存结果；输出可复算的诊断 JSON/NPZ。

    检查 W:(N_time,N_lag)、t:(N_time,)；时间单位 s，峰位单位 ms。
    全时间轴与共同直接中心分别统计；直接中心间隔可能不等，E_t 未除 dt²。
    重算正演、滤波和 CC 只用于核验，不反馈到反演、QC 或 hybrid。
    """
    assert RESULT_ROOT.is_dir()
    plan = json.loads((RESULT_ROOT / "round3_plan.json").read_text(encoding="utf-8"))
    criteria = plan["criteria"]
    assert hashlib.sha256(Path(plan["input_file"]).read_bytes()).hexdigest() == plan["input_sha256"]
    for relative, digest in plan["source_sha256"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest, relative
    bundles = []
    records = []
    profiles = {}
    for config_record in plan["configs"]:
        config_path = ROOT / config_record["config_path"]
        result_dir = ROOT / config_record["result_dir"]
        assert hashlib.sha256(config_path.read_bytes()).hexdigest() == config_record["sha256"]
        assert config_path.read_bytes() == (result_dir / "config_used.yaml").read_bytes()
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        metrics = json.loads((result_dir / "metrics.json").read_text(encoding="utf-8"))
        bundle = {}
        with np.load(result_dir / "result_bundle.npz", allow_pickle=False) as saved:
            for key in saved.files:
                try:
                    bundle[key] = saved[key]
                except ValueError:
                    # object 标量沿用保存格式；验收原因读取 metrics，不反序列化。
                    continue
        bundles.append(bundle)
        time_s = bundle["t_work"]
        dt_s = float(bundle["dt"])
        direct_mask = bundle["tv_valid_mask"]
        assert bundle["W_pre_gaussian"].shape == (2546, 129)
        assert direct_mask.shape == time_s.shape and np.all(np.diff(time_s) > 0)
        assert np.all(np.isnan(bundle["W_direct_centers"][~direct_mask]))
        np.testing.assert_array_equal(bundle["W_direct_centers"][direct_mask], bundle["W_pre_gaussian"][direct_mask])
        time_sigma_samples = cfg["tv_wavelet"]["time_smooth_ms"] / 1000.0 / dt_s
        expected_gaussian = gaussian_filter(bundle["W_pre_gaussian"], sigma=(time_sigma_samples, 0.0), mode="nearest")
        np.testing.assert_array_equal(bundle["W_post_gaussian"], expected_gaussian)
        row_mean = np.mean(expected_gaussian, axis=1, keepdims=True)  # shape=(N_time,1)。
        np.testing.assert_array_equal(bundle["W_est"], expected_gaussian - row_mean)
        np.testing.assert_array_equal(bundle["W_final_hybrid"], bundle["tv_hybrid_W"])
        final_syn = nonstationary_convolution(bundle["r_work_final"], bundle["W_final"], alignment="center")
        np.testing.assert_allclose(match_rms(final_syn, bundle["obs_work"]), bundle["s_syn_final"], rtol=1e-12, atol=1e-12)
        assert abs(corrcoef_safe(bundle["obs_work"], bundle["s_syn_final"]) - metrics["CC_final"]) < 1e-12
        raw_syn = nonstationary_convolution(bundle["r_work_final"], bundle["W_est_best"], alignment="center")
        assert abs(corrcoef_safe(bundle["obs_work"], raw_syn) - metrics["raw_tv_CC_direct"]) < 1e-10
        attempted = bundle["tv_robust_attempted"]
        irls_ok = bool(np.all(bundle["tv_irls_converged"][attempted]) and np.all(bundle["tv_robust_solution_used"][attempted]))
        objective_ok = bool(np.all(bundle["tv_student_objective_final"][attempted] <= bundle["tv_student_objective_initial"][attempted] + 1e-12))
        records.append(dict(id=result_dir.name[:3], metrics=metrics, stages={}, amplitude={}, valid_centers=int(np.count_nonzero(direct_mask)), IRLS_attempted=int(np.count_nonzero(attempted)), IRLS_all_converged_without_fallback=irls_ok, IRLS_objective_nonincreasing=objective_ok))

    # shape=(N_time,)，交集排除有效中心数量不同导致的采样支撑混淆。
    common_centers = np.logical_and.reduce([bundle["tv_valid_mask"] for bundle in bundles])
    assert np.count_nonzero(common_centers) >= 2
    for record, bundle in zip(records, bundles):
        time_s = bundle["t_work"]
        dt_s = float(bundle["dt"])
        direct_mask = bundle["tv_valid_mask"]
        dense_mask = np.ones(time_s.size, dtype=bool)  # shape=(N_time,)。
        for stage in STAGES:
            stage_mask = direct_mask if stage == "W_direct_centers" else dense_mask
            record["stages"][stage] = {}
            for label, mask in (("available_rows", stage_mask), ("same_case_direct_centers", direct_mask), ("common_direct_centers", common_centers)):
                record["stages"][stage][label] = summarize_time_stage(bundle[stage], time_s, mask, dt_s, criteria["normalization_epsilon"])
            if stage != "W_direct_centers":
                window_samples = bundle["tv_robust_window_offsets_samples"].size
                summary, before, after = amplitude_diagnostics(bundle[stage], bundle["r_work_final"], bundle["obs_work"], window_samples)
                record["amplitude"][stage] = summary
                profiles[record["id"] + "_" + stage + "_before_scale"] = before
                profiles[record["id"] + "_" + stage + "_after_global_scale"] = after
                half_window = window_samples // 2
                profiles["window_center_time_s"] = time_s[half_window:-half_window]
        for stage, metric_key in (("W_est_best", "candidate_temporal_difference_energy"), ("W_final", "final_temporal_difference_energy")):
            assert np.isclose(record["stages"][stage]["available_rows"]["temporal_difference_energy"], record["metrics"][metric_key], rtol=1e-12, atol=1e-14)

    previous_dir = ROOT / "_experiment_results/ablation/round2_lag/L02_no_lag_gaussian"
    previous_metrics = json.loads((previous_dir / "metrics.json").read_text(encoding="utf-8"))
    assert records[0]["metrics"] == previous_metrics
    with np.load(previous_dir / "result_bundle.npz", allow_pickle=False) as previous:
        for key in bundles[0]:
            np.testing.assert_array_equal(bundles[0][key], previous[key])
    for bundle in bundles[1:]:
        for key in ("t_work", "obs_work", "r_work_final", "twt_final", "w_prior", "s_syn_after_dtw", "constant_phase_phase_cc"):
            np.testing.assert_array_equal(bundle[key], bundles[0][key])
    for on_index, off_index in ((0, 2), (1, 3)):
        for key in ("W_direct_centers", "W_pre_gaussian", "tv_valid_mask", "tv_student_objective_final"):
            np.testing.assert_array_equal(bundles[on_index][key], bundles[off_index][key])

    baseline = records[0]
    for record in records:
        metrics = record["metrics"]
        record["delta_CC_final"] = metrics["CC_final"] - baseline["metrics"]["CC_final"]
        changes = {}
        for stage in STAGES:
            changes[stage] = {}
            for scope in ("available_rows", "common_direct_centers"):
                changes[stage][scope] = {}
                for key in ("temporal_difference_energy", "normalized_temporal_energy", "mean_wavelet_energy"):
                    reference = baseline["stages"][stage][scope][key]
                    changes[stage][scope][key] = 100 * (record["stages"][stage][scope][key] / reference - 1)
        record["stage_change_pct_vs_T00"] = changes
        gates = dict(
            CC_final_noninferior=record["delta_CC_final"] >= criteria["delta_CC_final_floor"],
            raw_W_pass_no_regression=bool(metrics["raw_tv_W_pass"] or not baseline["metrics"]["raw_tv_W_pass"]),
            hybrid_W_pass_no_regression=bool(metrics["W_pass"] or not baseline["metrics"]["W_pass"]),
            raw_peak_p90_le_17ms=metrics["candidate_peak_abs_p90"] <= criteria["peak_abs_p90_ms"] + 1e-6,
            final_peak_p90_le_17ms=metrics["final_peak_abs_p90_after_local_fallback"] <= criteria["peak_abs_p90_ms"] + 1e-6,
            candidate_normalized_Et_le_15pct=changes["W_est_best"]["available_rows"]["normalized_temporal_energy"] <= criteria["normalized_Et_increase_limit_pct"],
            final_normalized_Et_le_15pct=changes["W_final"]["available_rows"]["normalized_temporal_energy"] <= criteria["normalized_Et_increase_limit_pct"],
            IRLS_stable=record["IRLS_all_converged_without_fallback"] and record["IRLS_objective_nonincreasing"],
        )
        record["main_gates"] = gates
        record["main_gates_pass"] = all(gates.values())
        record["failed_main_gates"] = [key for key, passed in gates.items() if not passed]
        record["valid_centers_conservative_flag_pass"] = record["valid_centers"] >= baseline["valid_centers"] - criteria["valid_centers_allowed_drop"]
        record["final_edge_preferred"] = metrics["final_edge_energy_p90"] <= criteria["edge_p90_preferred"]
        print(record["id"], "CC_final=", metrics["CC_final"], "valid=", record["valid_centers"], "failed=", record["failed_main_gates"])
    effects = {}
    for label, before, after in (("Gaussian_off_mu_time_on", 0, 2), ("Gaussian_off_mu_time_off", 1, 3), ("mu_time_off_Gaussian_on", 0, 1), ("mu_time_off_Gaussian_off", 2, 3)):
        effects[label] = records[after]["metrics"]["CC_final"] - records[before]["metrics"]["CC_final"]
    acceptance = dict(status="completed", integrity_checks_passed=True, T00_equals_L02_all_metrics_and_numeric_arrays=True, numeric_arrays_compared=len(bundles[0]), time_Gaussian_pairs_have_identical_pre_Gaussian_inputs=True, common_direct_centers_count=int(np.count_nonzero(common_centers)), criteria=criteria, factorial_CC_effects=effects, records=records, production_config_modified=False)
    (RESULT_ROOT / "time_acceptance.json").write_text(json.dumps(acceptance, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    profiles["common_direct_center_time_s"] = bundles[0]["t_work"][common_centers]
    np.savez_compressed(RESULT_ROOT / "rms_profiles.npz", **profiles)
    return records


if __name__ == "__main__":
    check_time_ablation()
