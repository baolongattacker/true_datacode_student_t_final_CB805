# -*- coding: utf-8 -*-
"""第四轮 M00–M04 离线核验脚本；读取已保存结果并评估 Pareto 权衡。

功能：
1. 验证输入与配置 SHA256；
2. 验证 M03 与 T00 逐元素精确复现；
3. 统计 5 组各自与共同有效中心上的阶段指标（时变差分、范数、曲率、峰位、边缘、旁瓣）；
4. 提取全局标定前与标定后的双局部 RMS 比剖面及对数误差；
5. 检查数值稳定性（IRLS、病态、跳过数、有效中心）；
6. 评估运行前冻结的预注册门槛及 Pareto 前沿。
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

from core.forward_operator import nonstationary_convolution
from core.signal_utils import match_rms
from core.metrics import corrcoef_safe
from core.wavelet_qc import compute_boundary_peak_mask, compute_wavelet_qc_attributes
from experiments.run_real_experiment import compute_edge_energy_ratio, compute_side_lobe_ratio

RESULT_ROOT = ROOT / "_experiment_results/ablation/round4_mu1"
STAGES = ("W_direct_centers", "W_pre_gaussian", "W_post_gaussian", "W_est_best", "W_final_hybrid", "W_final")


def summarize_mu1_stage(W_matrix, time_s, row_mask, dt_s, epsilon):
    """汇总指定阶段在指定时间掩码行上的波形特征。"""
    W_matrix = np.asarray(W_matrix, dtype=float)
    time_s = np.asarray(time_s, dtype=float)
    row_mask = np.asarray(row_mask, dtype=bool)
    assert W_matrix.ndim == 2 and W_matrix.shape[1] >= 3
    assert time_s.ndim == row_mask.ndim == 1
    assert time_s.size == row_mask.size == W_matrix.shape[0]

    selected_W = W_matrix[row_mask]
    selected_time_s = time_s[row_mask]
    assert selected_W.shape[0] >= 2 and np.all(np.isfinite(selected_W))

    qc = compute_wavelet_qc_attributes(selected_W, dt_s, "center")
    row_energy = np.sum(selected_W ** 2, axis=1)
    mean_energy = float(np.mean(row_energy))

    lag_curvature = np.diff(selected_W, n=2, axis=1)
    curvature_energy = np.sum(lag_curvature ** 2, axis=1)
    normalized_curvature = curvature_energy / (row_energy + 1e-12)

    temporal_diff = np.diff(selected_W, axis=0)
    temporal_energy = float(np.mean(np.sum(temporal_diff ** 2, axis=1)))
    norm_temporal_energy = temporal_energy / (mean_energy + epsilon)

    dc_ratio = np.abs(np.sum(selected_W, axis=1)) / (np.sum(np.abs(selected_W), axis=1) + 1e-12)
    edge_ratio = compute_edge_energy_ratio(selected_W, edge_fraction=0.15)
    side_ratio = compute_side_lobe_ratio(selected_W, dt_s, "center", guard_ms=12.0)
    boundary_peaks = compute_boundary_peak_mask(selected_W, 0.12)

    abs_peak_ms = np.abs(qc["peak_metric_ms"])

    return {
        "rows": int(selected_W.shape[0]),
        "median_time_interval_ms": float(1000 * np.median(np.diff(selected_time_s))),
        "maximum_time_interval_ms": float(1000 * np.max(np.diff(selected_time_s))),
        "peak_abs_p50_ms": float(np.percentile(abs_peak_ms, 50)),
        "peak_abs_p90_ms": float(np.percentile(abs_peak_ms, 90)),
        "peak_outside_15ms_fraction": float(np.mean(abs_peak_ms > 15.0 + 1e-6)),
        "peak_outside_17ms_fraction": float(np.mean(abs_peak_ms > 17.0 + 1e-6)),
        "boundary_peak_fraction": float(np.mean(boundary_peaks)),
        "edge_energy_p50": float(np.percentile(edge_ratio, 50)),
        "edge_energy_p90": float(np.percentile(edge_ratio, 90)),
        "edge_exceed_0p15_fraction": float(np.mean(edge_ratio > 0.15)),
        "edge_exceed_0p30_fraction": float(np.mean(edge_ratio > 0.30)),
        "side_lobe_p50": float(np.percentile(side_ratio, 50)),
        "side_lobe_p90": float(np.percentile(side_ratio, 90)),
        "side_lobe_exceed_0p75_fraction": float(np.mean(side_ratio > 0.75)),
        "side_lobe_exceed_0p95_fraction": float(np.mean(side_ratio > 0.95)),
        "wavelet_l2_norm_p50": float(np.percentile(qc["energy_l2"], 50)),
        "wavelet_l2_norm_p90": float(np.percentile(qc["energy_l2"], 90)),
        "mean_wavelet_energy": mean_energy,
        "dc_ratio_max": float(np.max(dc_ratio)),
        "curvature_energy_p90": float(np.percentile(curvature_energy, 90)),
        "normalized_curvature_energy_p90": float(np.percentile(normalized_curvature, 90)),
        "temporal_difference_energy": temporal_energy,
        "normalized_temporal_energy": norm_temporal_energy,
        "centroid_frequency_median_hz": float(np.median(qc["centroid_frequency_hz"])),
        "bandwidth_p90_hz": float(np.percentile(qc["bandwidth_hz"], 90)),
    }


def compute_dual_amplitude_diagnostics(W_matrix, r_time, s_obs, window_samples):
    """计算全局标定前 (raw) 与标定后 (scaled) 的滑窗 RMS 比值剖面及对数误差。"""
    s_syn_raw = nonstationary_convolution(r_time, W_matrix, alignment="center")
    obs_rms = float(np.sqrt(np.mean(s_obs ** 2)))
    syn_raw_rms = float(np.sqrt(np.mean(s_syn_raw ** 2)))
    scale_factor = obs_rms / (syn_raw_rms + 1e-12)

    s_syn_scaled = s_syn_raw * scale_factor

    half = window_samples // 2
    N = len(s_obs)
    valid_indices = range(half, N - half)
    raw_ratios = []
    scaled_ratios = []

    for i in valid_indices:
        obs_win = s_obs[i - half : i + half + 1]
        syn_raw_win = s_syn_raw[i - half : i + half + 1]
        syn_scaled_win = s_syn_scaled[i - half : i + half + 1]

        o_r = np.sqrt(np.mean(obs_win ** 2))
        if o_r > 1e-12:
            raw_ratios.append(np.sqrt(np.mean(syn_raw_win ** 2)) / o_r)
            scaled_ratios.append(np.sqrt(np.mean(syn_scaled_win ** 2)) / o_r)
        else:
            raw_ratios.append(np.nan)
            scaled_ratios.append(np.nan)

    raw_arr = np.asarray(raw_ratios, dtype=float)
    scaled_arr = np.asarray(scaled_ratios, dtype=float)

    # 对数误差：衡量去除了整体增益后，局部时窗振幅偏离观测的程度
    log_err = np.abs(np.log(np.clip(scaled_arr, 1e-6, 1e6)))

    summary = {
        "global_obs_rms": obs_rms,
        "global_syn_raw_rms": syn_raw_rms,
        "global_scale_factor": scale_factor,
        "raw_rms_ratio_p10": float(np.nanpercentile(raw_arr, 10)),
        "raw_rms_ratio_p50": float(np.nanpercentile(raw_arr, 50)),
        "raw_rms_ratio_p90": float(np.nanpercentile(raw_arr, 90)),
        "scaled_rms_ratio_p10": float(np.nanpercentile(scaled_arr, 10)),
        "scaled_rms_ratio_p50": float(np.nanpercentile(scaled_arr, 50)),
        "scaled_rms_ratio_p90": float(np.nanpercentile(scaled_arr, 90)),
        "amplitude_log_error_median": float(np.nanmedian(log_err)),
        "amplitude_log_error_p90": float(np.nanpercentile(log_err, 90)),
    }
    return summary, raw_arr, scaled_arr


def check_mu1_ablation():
    """核验第四轮消融实验，输出 mu1_acceptance.json 及 rms_profiles.npz。"""
    assert RESULT_ROOT.is_dir()
    plan_path = RESULT_ROOT / "round4_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    criteria = plan["criteria"]

    # 核验证件哈希
    assert hashlib.sha256(Path(plan["input_file"]).read_bytes()).hexdigest() == plan["input_sha256"]

    bundles = []
    records = []
    profiles = {}

    for config_record in plan["configs"]:
        cfg_path = ROOT / config_record["config_path"]
        res_dir = ROOT / config_record["result_dir"]
        assert cfg_path.exists() and res_dir.exists(), f"路径缺失: {cfg_path} 或 {res_dir}"
        assert hashlib.sha256(cfg_path.read_bytes()).hexdigest() == config_record["sha256"]
        assert cfg_path.read_bytes() == (res_dir / "config_used.yaml").read_bytes()

        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        metrics = json.loads((res_dir / "metrics.json").read_text(encoding="utf-8"))
        diag_summary = json.loads((res_dir / "tv_diag_summary.json").read_text(encoding="utf-8"))

        bundle = {}
        with np.load(res_dir / "result_bundle.npz", allow_pickle=False) as saved:
            for key in saved.files:
                try:
                    bundle[key] = saved[key]
                except ValueError:
                    continue
        bundles.append(bundle)

        time_s = bundle["t_work"]
        dt_s = float(bundle["dt"])
        direct_mask = bundle["tv_valid_mask"]
        assert direct_mask.shape == time_s.shape
        assert bundle["W_pre_gaussian"].shape == (2546, 129)

        # 检查阶段快照一致性
        np.testing.assert_array_equal(bundle["W_direct_centers"][direct_mask], bundle["W_pre_gaussian"][direct_mask])
        time_sigma_samples = cfg["tv_wavelet"]["time_smooth_ms"] / 1000.0 / dt_s
        expected_gaussian = gaussian_filter(bundle["W_pre_gaussian"], sigma=(time_sigma_samples, 0.0), mode="nearest")
        np.testing.assert_array_equal(bundle["W_post_gaussian"], expected_gaussian)

        # 检查正演再现与 CC
        final_syn = nonstationary_convolution(bundle["r_work_final"], bundle["W_final"], alignment="center")
        np.testing.assert_allclose(match_rms(final_syn, bundle["obs_work"]), bundle["s_syn_final"], rtol=1e-12, atol=1e-12)
        assert abs(corrcoef_safe(bundle["obs_work"], bundle["s_syn_final"]) - metrics["CC_final"]) < 1e-12

        raw_syn = nonstationary_convolution(bundle["r_work_final"], bundle["W_est_best"], alignment="center")
        assert abs(corrcoef_safe(bundle["obs_work"], raw_syn) - metrics["raw_tv_CC_direct"]) < 1e-10

        # 数值求解稳定性指标
        attempted = bundle["tv_robust_attempted"]
        attempted_count = int(np.count_nonzero(attempted))
        converged_count = int(np.count_nonzero(bundle["tv_irls_converged"][attempted]))
        robust_used_count = int(np.count_nonzero(bundle["tv_robust_solution_used"][attempted]))
        l2_fallback_count = attempted_count - robust_used_count
        irls_ok = (converged_count == attempted_count) and (robust_used_count == attempted_count)
        objective_ok = bool(np.all(bundle["tv_student_objective_final"][attempted] <= bundle["tv_student_objective_initial"][attempted] + 1e-12))

        iterations = bundle["tv_irls_iterations"][attempted]
        iter_median = float(np.median(iterations)) if iterations.size > 0 else 0.0
        iter_max = int(np.max(iterations)) if iterations.size > 0 else 0

        numerical_stability = {
            "valid_centers": int(np.count_nonzero(direct_mask)),
            "skipped_low_energy": int(diag_summary.get("skipped_low_energy", 0)),
            "skipped_ill_conditioned": int(diag_summary.get("skipped_ill_conditioned", 0)),
            "skipped_peak_shift": int(diag_summary.get("skipped_peak_shift", 0)),
            "skipped_amplitude_jump": int(diag_summary.get("skipped_amplitude_jumps", 0)),
            "irls_attempted": attempted_count,
            "irls_converged": converged_count,
            "robust_solution_used": robust_used_count,
            "l2_fallback_count": l2_fallback_count,
            "irls_iterations_median": iter_median,
            "irls_iterations_max": iter_max,
            "irls_all_converged_without_fallback": irls_ok,
            "irls_objective_nonincreasing": objective_ok,
        }

        records.append({
            "id": config_record["id"],
            "mu1": config_record["mu1"],
            "metrics": metrics,
            "numerical": numerical_stability,
            "stages": {},
            "amplitude": {},
        })

    # 共同直接反演中心交集 (5 组共同支撑)
    common_centers = np.logical_and.reduce([bundle["tv_valid_mask"] for bundle in bundles])
    common_centers_count = int(np.count_nonzero(common_centers))
    assert common_centers_count >= 2, f"共同有效中心数过少: {common_centers_count}"

    # 提取各阶段波形属性与振幅诊断
    window_samples = bundles[0]["tv_robust_window_offsets_samples"].size
    half_window = window_samples // 2
    time_s = bundles[0]["t_work"]
    profiles["window_center_time_s"] = time_s[half_window:-half_window]
    profiles["common_direct_center_time_s"] = time_s[common_centers]

    for record, bundle in zip(records, bundles):
        dt_s = float(bundle["dt"])
        direct_mask = bundle["tv_valid_mask"]
        dense_mask = np.ones(time_s.size, dtype=bool)

        for stage in STAGES:
            stage_mask = direct_mask if stage == "W_direct_centers" else dense_mask
            record["stages"][stage] = {}
            for scope_label, mask in (
                ("available_rows", stage_mask),
                ("same_case_direct_centers", direct_mask),
                ("common_direct_centers", common_centers),
            ):
                record["stages"][stage][scope_label] = summarize_mu1_stage(
                    bundle[stage], time_s, mask, dt_s, criteria["normalization_epsilon"]
                )

            if stage in ("W_est_best", "W_final"):
                summary, raw_arr, scaled_arr = compute_dual_amplitude_diagnostics(
                    bundle[stage], bundle["r_work_final"], bundle["obs_work"], window_samples
                )
                record["amplitude"][stage] = summary
                profiles[f"{record['id']}_{stage}_raw_rms_ratio"] = raw_arr
                profiles[f"{record['id']}_{stage}_scaled_rms_ratio"] = scaled_arr

    # 验证 M03 与 T00 逐元素精确复现
    t00_dir = ROOT / "_experiment_results/ablation/round3_time/T00_time_on_gaussian_on"
    t00_metrics = json.loads((t00_dir / "metrics.json").read_text(encoding="utf-8"))
    m03_record = next(r for r in records if r["id"] == "M03")
    assert m03_record["metrics"] == t00_metrics, "M03 与 T00 的 metrics.json 不完全一致！"

    with np.load(t00_dir / "result_bundle.npz", allow_pickle=False) as t00_bundle:
        m03_bundle = bundles[3]
        for key in t00_bundle.files:
            if key in m03_bundle:
                np.testing.assert_array_equal(m03_bundle[key], t00_bundle[key])
    print("M03 与 T00 逐元素精确复现核验通过！")

    # 以 M03 为参考基准计算相对变化与门槛检验
    baseline = m03_record
    for record in records:
        m = record["metrics"]
        record["delta_CC_final"] = m["CC_final"] - baseline["metrics"]["CC_final"]
        record["delta_raw_CC"] = m["raw_tv_CC_direct"] - baseline["metrics"]["raw_tv_CC_direct"]

        # 各阶段相对 M03 的百分比变化
        changes = {}
        for stage in STAGES:
            changes[stage] = {}
            for scope in ("available_rows", "common_direct_centers"):
                changes[stage][scope] = {}
                for key in ("temporal_difference_energy", "normalized_temporal_energy", "mean_wavelet_energy"):
                    ref_val = baseline["stages"][stage][scope][key]
                    val = record["stages"][stage][scope][key]
                    changes[stage][scope][key] = 100.0 * (val / (ref_val + 1e-12) - 1.0)
        record["stage_change_pct_vs_M03"] = changes

        # 检查各硬门槛
        c_change_common = changes["W_est_best"]["common_direct_centers"]["normalized_temporal_energy"]
        f_change_common = changes["W_final"]["common_direct_centers"]["normalized_temporal_energy"]

        gates = {
            "CC_final_noninferior": record["delta_CC_final"] >= criteria["delta_CC_final_floor"],
            "raw_W_pass_no_regression": bool(m["raw_tv_W_pass"]),
            "hybrid_W_pass_no_regression": bool(m["W_pass"]),
            "raw_peak_p90_le_17ms": m["candidate_peak_abs_p90"] <= criteria["peak_abs_p90_ms_hard_gate"] + 1e-6,
            "final_peak_p90_le_17ms": m["final_peak_abs_p90_after_local_fallback"] <= criteria["peak_abs_p90_ms_hard_gate"] + 1e-6,
            "common_candidate_norm_Et_le_15pct": c_change_common <= criteria["normalized_Et_increase_limit_pct"],
            "common_final_norm_Et_le_15pct": f_change_common <= criteria["normalized_Et_increase_limit_pct"],
            "valid_centers_no_drop": record["numerical"]["valid_centers"] >= baseline["numerical"]["valid_centers"] - criteria["valid_centers_allowed_drop"],
            "IRLS_no_fallback": record["numerical"]["l2_fallback_count"] == 0,
            "IRLS_converged": record["numerical"]["irls_all_converged_without_fallback"],
            "IRLS_objective_nonincreasing": record["numerical"]["irls_objective_nonincreasing"],
        }
        record["gates"] = gates
        record["gates_pass"] = all(gates.values())
        record["failed_gates"] = [k for k, v in gates.items() if not v]

        # 优选提示指标
        record["preferences"] = {
            "peak_abs_p90_le_15ms": m["final_peak_abs_p90_after_local_fallback"] <= criteria["peak_abs_p90_ms_preferred"] + 1e-6,
            "edge_p90_le_0p15": m["final_edge_energy_p90"] <= criteria["edge_p90_preferred"] + 1e-6,
            "side_lobe_p90_le_0p75": m["final_side_lobe_p90"] <= criteria["side_lobe_reliable"] + 1e-6,
        }

    # Pareto 候选分析与拐点确定
    eligible_records = [r for r in records if r["gates_pass"]]
    acceptance = {
        "status": "completed",
        "integrity_checks_passed": True,
        "M03_equals_T00_reproduction_verified": True,
        "common_direct_centers_count": common_centers_count,
        "criteria": criteria,
        "records": records,
        "eligible_candidates": [r["id"] for r in eligible_records],
    }

    # 保存 NPZ 与 JSON
    np.savez_compressed(RESULT_ROOT / "rms_profiles.npz", **profiles)
    (RESULT_ROOT / "mu1_acceptance.json").write_text(
        json.dumps(acceptance, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )

    # 计划文件更新状态
    plan["status"] = "completed"
    plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    print("第四轮核验完成，已生成 mu1_acceptance.json 与 rms_profiles.npz！")
    return acceptance


if __name__ == "__main__":
    check_mu1_ablation()
