# -*- coding: utf-8 -*-
"""第六轮 S00–S03 离线核验脚本；核验 S03 与 C02 复现，检验因果隔离不变量，评估时间平滑收益与修改代价。"""

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

RESULT_ROOT = ROOT / "_experiment_results/ablation/round6_time_smooth"
ROUND5_DIR = ROOT / "_experiment_results/ablation/round5_mu2"
STAGES = ("W_direct_centers", "W_pre_gaussian", "W_post_gaussian", "W_est_best", "W_final_hybrid", "W_final")


def summarize_time_smooth_stage(W_matrix, time_s, row_mask, dt_s, epsilon):
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

    # 二阶差分算子曲率：D2 @ w
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
        "curvature_energy_p50": float(np.percentile(curvature_energy, 50)),
        "curvature_energy_p90": float(np.percentile(curvature_energy, 90)),
        "normalized_curvature_energy_p50": float(np.percentile(normalized_curvature, 50)),
        "normalized_curvature_energy_p90": float(np.percentile(normalized_curvature, 90)),
        "temporal_difference_energy": temporal_energy,
        "normalized_temporal_energy": norm_temporal_energy,
        "centroid_frequency_median_hz": float(np.median(qc["centroid_frequency_hz"])),
        "bandwidth_p90_hz": float(np.percentile(qc["bandwidth_hz"], 90)),
    }


def compute_dual_amplitude_diagnostics(W_matrix, r_time, s_obs, window_samples):
    """计算全局标定前与标定后的局部 RMS 比值剖面及对数误差。"""
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


def compute_gaussian_modification_cost(W_post, W_pre, epsilon=1e-12):
    """计算高斯平滑对波形带来的相对修改代价 D_W(t) = ||w_post - w_pre||_2 / (||w_pre||_2 + eps)。"""
    diff = W_post - W_pre
    diff_norm = np.linalg.norm(diff, axis=1)
    pre_norm = np.linalg.norm(W_pre, axis=1)
    rel_cost = diff_norm / (pre_norm + epsilon)
    return {
        "D_W_median": float(np.median(rel_cost)),
        "D_W_p90": float(np.percentile(rel_cost, 90)),
        "D_W_mean": float(np.mean(rel_cost)),
        "D_W_max": float(np.max(rel_cost)),
        "relative_cost_profile": rel_cost
    }


def check_time_smooth_ablation():
    """核验第六轮时间平滑消融实验，输出 time_smooth_acceptance.json 及 rms_profiles.npz。"""
    assert RESULT_ROOT.is_dir(), f"目录不存在: {RESULT_ROOT}"
    plan_path = RESULT_ROOT / "round6_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    criteria = plan["criteria"]

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

        # 阶段快照检查
        np.testing.assert_array_equal(bundle["W_direct_centers"][direct_mask], bundle["W_pre_gaussian"][direct_mask])
        time_smooth_ms = cfg["tv_wavelet"]["time_smooth_ms"]
        time_sigma_samples = time_smooth_ms / 1000.0 / dt_s
        if time_sigma_samples > 0.0:
            expected_gaussian = gaussian_filter(bundle["W_pre_gaussian"], sigma=(time_sigma_samples, 0.0), mode="nearest")
            np.testing.assert_array_equal(bundle["W_post_gaussian"], expected_gaussian)
        else:
            np.testing.assert_array_equal(bundle["W_post_gaussian"], bundle["W_pre_gaussian"])

        # 正演核验
        final_syn = nonstationary_convolution(bundle["r_work_final"], bundle["W_final"], alignment="center")
        np.testing.assert_allclose(match_rms(final_syn, bundle["obs_work"]), bundle["s_syn_final"], rtol=1e-12, atol=1e-12)
        assert abs(corrcoef_safe(bundle["obs_work"], bundle["s_syn_final"]) - metrics["CC_final"]) < 1e-12

        raw_syn = nonstationary_convolution(bundle["r_work_final"], bundle["W_est_best"], alignment="center")
        assert abs(corrcoef_safe(bundle["obs_work"], raw_syn) - metrics["raw_tv_CC_direct"]) < 1e-10

        # 数值求解稳定性
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

        # 高斯修改代价
        cost_info = compute_gaussian_modification_cost(bundle["W_post_gaussian"], bundle["W_pre_gaussian"])

        # Hybrid 依赖指标
        hybrid_dependency = {
            "hard_prior_ratio": float(metrics.get("hybrid_hard_prior_ratio", 0.0)),
            "tv_support_ratio": float(metrics.get("hybrid_tv_support_ratio", 0.0)),
            "blend_ratio": float(metrics.get("hybrid_blend_ratio", 0.0)),
            "short_gap_ratio": float(metrics.get("hybrid_short_gap_ratio", 0.0)),
            "fallback_by_shape_ratio": float(metrics.get("fallback_by_shape_ratio", 0.0)),
            "candidate_shape_fallback_request_ratio": float(metrics.get("candidate_shape_fallback_request_ratio", 0.0)),
            "strict_reliable_direct_count": int(metrics.get("hybrid_strict_reliable_direct_count", 0)),
            "admissible_direct_count": int(metrics.get("hybrid_admissible_direct_count", 0)),
            "gray_zone_direct_count": int(metrics.get("hybrid_gray_zone_direct_count", 0)),
            "rejected_direct_count": int(metrics.get("hybrid_rejected_direct_count", 0)),
            "strict_reliable_center_ratio": float(metrics.get("hybrid_strict_reliable_center_ratio", 0.0)),
            "admissible_direct_center_ratio": float(metrics.get("hybrid_admissible_direct_center_ratio", 0.0)),
            "gray_zone_center_ratio": float(metrics.get("hybrid_gray_zone_center_ratio", 0.0)),
            "rejected_direct_center_ratio": float(metrics.get("hybrid_rejected_direct_center_ratio", 0.0)),
        }

        records.append({
            "id": config_record["id"],
            "time_smooth_ms": config_record["time_smooth_ms"],
            "metrics": metrics,
            "numerical": numerical_stability,
            "gaussian_cost": {k: v for k, v in cost_info.items() if k != "relative_cost_profile"},
            "hybrid_dependency": hybrid_dependency,
            "stages": {},
            "amplitude": {},
        })

    # ==================== 1. 因果隔离不变量核验 (Regression Invariants) ====================
    # S00–S03 理论上 W_direct_centers 与 W_pre_gaussian 必须完全一致！
    for i in range(len(bundles)):
        for j in range(i + 1, len(bundles)):
            id_i = records[i]["id"]
            id_j = records[j]["id"]
            np.testing.assert_allclose(
                bundles[i]["W_direct_centers"], bundles[j]["W_direct_centers"],
                rtol=1e-12, atol=1e-12,
                err_msg=f"因果隔离失败: W_direct_centers 在 {id_i} 与 {id_j} 之间存在差异！"
            )
            np.testing.assert_allclose(
                bundles[i]["W_pre_gaussian"], bundles[j]["W_pre_gaussian"],
                rtol=1e-12, atol=1e-12,
                err_msg=f"因果隔离失败: W_pre_gaussian 在 {id_i} 与 {id_j} 之间存在差异！"
            )
            np.testing.assert_array_equal(
                bundles[i]["tv_valid_mask"], bundles[j]["tv_valid_mask"],
                err_msg=f"因果隔离失败: tv_valid_mask 在 {id_i} 与 {id_j} 之间不相等！"
            )
    print(">>> 因果隔离核验通过：W_direct 与 W_pre_gaussian 在 S00–S03 间逐元素完全一致 (误差 < 1e-12)！")

    # ==================== 2. 确定性锚点核验 (S03 vs C02) ====================
    c02_dir = ROUND5_DIR / "C02_mu2_3p0"
    assert c02_dir.is_dir(), f"未找到基线 C02 目录: {c02_dir}"
    c02_metrics = json.loads((c02_dir / "metrics.json").read_text(encoding="utf-8"))
    s03_record = next(r for r in records if r["id"] == "S03")
    s03_bundle = bundles[3]

    assert abs(s03_record["metrics"]["CC_final"] - c02_metrics["CC_final"]) < 1e-12
    assert abs(s03_record["metrics"]["raw_tv_CC_direct"] - c02_metrics["raw_tv_CC_direct"]) < 1e-12
    assert s03_record["metrics"]["W_pass"] == c02_metrics["W_pass"]

    with np.load(c02_dir / "result_bundle.npz", allow_pickle=False) as c02_saved:
        for array_key in ("W_direct_centers", "W_pre_gaussian", "W_post_gaussian", "W_est_best", "W_final_hybrid", "W_final"):
            np.testing.assert_allclose(
                s03_bundle[array_key], c02_saved[array_key],
                rtol=1e-12, atol=1e-12,
                err_msg=f"锚点核验失败: S03 的 {array_key} 与 C02 不一致！"
            )
    print(">>> 确定性锚点核验通过：S03 完全逐元素复现 C02 (误差 < 1e-12)！")

    # ==================== 3. 共同直接中心与阶段波形特征 ====================
    common_centers = np.logical_and.reduce([bundle["tv_valid_mask"] for bundle in bundles])
    common_centers_count = int(np.count_nonzero(common_centers))
    assert common_centers_count >= 2

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
            record["stages"][stage] = {
                "available_rows": summarize_time_smooth_stage(bundle[stage], time_s, stage_mask, dt_s, 1e-12),
                "same_case_direct_centers": summarize_time_smooth_stage(bundle[stage], time_s, direct_mask, dt_s, 1e-12),
                "common_direct_centers": summarize_time_smooth_stage(bundle[stage], time_s, common_centers, dt_s, 1e-12),
            }

        diag_summary, raw_arr, scaled_arr = compute_dual_amplitude_diagnostics(
            bundle["W_final"], bundle["r_work_final"], bundle["obs_work"], window_samples
        )
        record["amplitude"] = diag_summary
        profiles[f"{record['id']}_raw_rms_ratio"] = raw_arr
        profiles[f"{record['id']}_scaled_rms_ratio"] = scaled_arr

    # ==================== 4. 时间平滑收益比例 B(sigma) 计算 ====================
    # 定义 E_0 = E_{t,norm}(0 ms), E_30 = E_{t,norm}(30 ms)
    s00_record = next(r for r in records if r["id"] == "S00")
    # 分别基于 post_gaussian (候选平滑后), est_best (候选裁切后), final_hybrid (最终结果)
    e0_postG = s00_record["stages"]["W_post_gaussian"]["available_rows"]["normalized_temporal_energy"]
    e30_postG = s03_record["stages"]["W_post_gaussian"]["available_rows"]["normalized_temporal_energy"]
    denom_postG = (e0_postG - e30_postG) if abs(e0_postG - e30_postG) > 1e-12 else 1e-12

    e0_best = s00_record["stages"]["W_est_best"]["available_rows"]["normalized_temporal_energy"]
    e30_best = s03_record["stages"]["W_est_best"]["available_rows"]["normalized_temporal_energy"]
    denom_best = (e0_best - e30_best) if abs(e0_best - e30_best) > 1e-12 else 1e-12

    e0_final = s00_record["stages"]["W_final"]["available_rows"]["normalized_temporal_energy"]
    e30_final = s03_record["stages"]["W_final"]["available_rows"]["normalized_temporal_energy"]
    denom_final = (e0_final - e30_final) if abs(e0_final - e30_final) > 1e-12 else 1e-12

    for record in records:
        e_postG = record["stages"]["W_post_gaussian"]["available_rows"]["normalized_temporal_energy"]
        b_postG = float((e0_postG - e_postG) / denom_postG)

        e_best = record["stages"]["W_est_best"]["available_rows"]["normalized_temporal_energy"]
        b_best = float((e0_best - e_best) / denom_best)

        e_final = record["stages"]["W_final"]["available_rows"]["normalized_temporal_energy"]
        b_final = float((e0_final - e_final) / denom_final)

        record["continuity_benefit"] = {
            "E_t_norm_postG": e_postG,
            "B_postG": b_postG,
            "E_t_norm_candidate": e_best,
            "B_candidate": b_best,
            "E_t_norm_final": e_final,
            "B_final": b_final,
        }

    # ==================== 5. 验收判据评估 ====================
    baseline_cc = s03_record["metrics"]["CC_final"]
    acceptance_summary = {}

    for record in records:
        metrics = record["metrics"]
        num = record["numerical"]
        delta_cc = metrics["CC_final"] - baseline_cc
        peak_abs_p90 = metrics["hybrid_tv_peak_abs_p90_ms"]
        hard_prior_ratio = record["hybrid_dependency"]["hard_prior_ratio"]

        pass_cc = delta_cc >= criteria["delta_CC_final_floor"]
        pass_raw_w = bool(metrics["raw_tv_W_pass"])
        pass_hybrid_w = bool(metrics["hybrid_tv_W_pass"])
        pass_peak = peak_abs_p90 <= criteria["peak_abs_p90_ms_hard_gate"]
        pass_centers = (num["valid_centers"] == 70)
        pass_irls = (num["l2_fallback_count"] == 0) and num["irls_all_converged_without_fallback"]
        pass_all = all([pass_cc, pass_raw_w, pass_hybrid_w, pass_peak, pass_centers, pass_irls])

        acceptance_summary[record["id"]] = {
            "time_smooth_ms": record["time_smooth_ms"],
            "CC_final": metrics["CC_final"],
            "delta_CC_final": delta_cc,
            "raw_tv_CC_direct": metrics["raw_tv_CC_direct"],
            "candidate_E_t_norm": record["continuity_benefit"]["E_t_norm_candidate"],
            "B_candidate": record["continuity_benefit"]["B_candidate"],
            "D_W_median": record["gaussian_cost"]["D_W_median"],
            "hard_prior_ratio": hard_prior_ratio,
            "peak_abs_p90_ms": peak_abs_p90,
            "pass_all": pass_all,
            "status": "PASS" if pass_all else "FAIL",
        }

    out_json = RESULT_ROOT / "time_smooth_acceptance.json"
    result_data = {
        "status": "completed",
        "causal_isolation_passed": True,
        "anchor_S03_verified": True,
        "common_direct_centers_count": common_centers_count,
        "criteria": criteria,
        "acceptance_summary": acceptance_summary,
        "records": records,
    }
    out_json.write_text(json.dumps(result_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"已生成验收指标 JSON: {out_json}")

    profiles_path = RESULT_ROOT / "rms_profiles.npz"
    np.savez_compressed(profiles_path, **profiles)
    print(f"已保存双振幅剖面: {profiles_path}")

    return result_data


if __name__ == "__main__":
    check_time_smooth_ablation()
