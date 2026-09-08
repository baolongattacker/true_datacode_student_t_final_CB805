# -*- coding: utf-8 -*-
"""读取 L00-L03 已保存结果，核验冻结条件并汇总各阶段子波诊断。

直接运行本脚本；只生成 round2_lag/lag_acceptance.json 与 rms_profiles.npz。
不重跑反演，不改变原验收阈值，不安装依赖。
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
from core.metrics import corrcoef_safe
from core.signal_utils import match_rms
from core.wavelet_qc import compute_boundary_peak_mask, compute_wavelet_qc_attributes
from experiments.run_real_experiment import compute_edge_energy_ratio, compute_side_lobe_ratio

RESULT_ROOT = ROOT / "_experiment_results/ablation/round2_lag"
CASE_NAMES = ("L00_reduced_baseline", "L01_no_mu2", "L02_no_lag_gaussian", "L03_no_mu2_no_lag_gaussian")


def summarize_stage(W_matrix, time_s, row_mask, dt_s):
    """汇总指定时间行的形态，不改写 W。

    输入：W_matrix shape=(N,L)，振幅单位继承反演；time_s shape=(N,) 和 dt_s
    单位 s；row_mask shape=(N,) 指定统计支撑，直接中心以外的 NaN 不参与。
    输出：标量字典；峰值 ms、频谱 Hz、范数为相对振幅，其余比例无量纲。
    E_t 为相邻选中行的平方差均值，未除以 dt²；直接中心可能不等间距，
    因此同时记录时间间隔，只在相同 row_mask 下比较阶段时间变化。
    """
    W_matrix = np.asarray(W_matrix, dtype=float)
    time_s = np.asarray(time_s, dtype=float)
    row_mask = np.asarray(row_mask, dtype=bool)
    assert W_matrix.ndim == 2 and W_matrix.shape[1] >= 3
    assert time_s.ndim == row_mask.ndim == 1
    assert time_s.size == row_mask.size == W_matrix.shape[0]
    assert np.isfinite(dt_s) and dt_s > 0 and np.all(np.diff(time_s) > 0)
    selected_W = W_matrix[row_mask]  # shape=(N_selected,L)，未对缺测行补零。
    selected_time_s = time_s[row_mask]  # shape=(N_selected,)。
    assert selected_W.shape[0] >= 2 and np.all(np.isfinite(selected_W))
    qc = compute_wavelet_qc_attributes(selected_W, dt_s, "center")
    row_energy = np.sum(selected_W ** 2, axis=1)  # shape=(N_selected,)。
    lag_curvature = np.diff(selected_W, n=2, axis=1)  # shape=(N_selected,L-2)，未除 dt²。
    curvature_energy = np.sum(lag_curvature ** 2, axis=1)
    normalized_curvature = curvature_energy / (row_energy + 1e-12)
    temporal_difference = np.diff(selected_W, axis=0)  # shape=(N_selected-1,L)。
    temporal_energy = float(np.mean(np.sum(temporal_difference ** 2, axis=1)))
    mean_energy = float(np.mean(row_energy))
    dc_ratio = np.abs(np.sum(selected_W, axis=1)) / (np.sum(np.abs(selected_W), axis=1) + 1e-12)
    edge_ratio = compute_edge_energy_ratio(selected_W, edge_fraction=0.15)
    side_ratio = compute_side_lobe_ratio(selected_W, dt_s, "center", guard_ms=12.0)
    return {
        "rows": int(selected_W.shape[0]),
        "median_time_interval_ms": float(1000 * np.median(np.diff(selected_time_s))),
        "maximum_time_interval_ms": float(1000 * np.max(np.diff(selected_time_s))),
        "peak_abs_p90_ms": float(np.percentile(np.abs(qc["peak_metric_ms"]), 90)),
        "peak_outside_15ms_fraction": float(np.mean(np.abs(qc["peak_metric_ms"]) > 15.0 + 1e-6)),
        "boundary_peak_fraction": float(np.mean(compute_boundary_peak_mask(selected_W, 0.12))),
        "edge_energy_p90": float(np.percentile(edge_ratio, 90)),
        "edge_reliable_exceed_fraction": float(np.mean(edge_ratio > 0.15)),
        "side_lobe_p90": float(np.percentile(side_ratio, 90)),
        "side_severe_exceed_fraction": float(np.mean(side_ratio > 0.95)),
        "wavelet_l2_norm_p90": float(np.percentile(qc["energy_l2"], 90)),
        "mean_wavelet_energy": mean_energy,
        "dc_ratio_max": float(np.max(dc_ratio)),
        "curvature_energy_p90": float(np.percentile(curvature_energy, 90)),
        "normalized_curvature_energy_p90": float(np.percentile(normalized_curvature, 90)),
        "temporal_difference_energy": temporal_energy,
        "normalized_temporal_energy": temporal_energy / mean_energy,
        "centroid_frequency_median_hz": float(np.median(qc["centroid_frequency_hz"])),
        "bandwidth_p90_hz": float(np.percentile(qc["bandwidth_hz"], 90)),
    }


def amplitude_diagnostics(W_matrix, r_time, s_obs, window_samples):
    """计算全局标定前和标定后的 RMS 比例，不调整 W 或最终合成记录。

    输入：W_matrix shape=(N,L)，r_time/s_obs shape=(N,)，振幅沿用已保存数据；
    window_samples 为正奇数，本轮沿用反演窗口 387 samples（约 387 ms）。
    输出：无量纲标量统计、shape=(N-window_samples+1,) 的局部 RMS 比例。
    物理意义：RMS(s_syn)/RMS(s_obs) 区分整体幅度和时间幅度分布；窗口只用
    完整支撑，不补零。观测 RMS 为零时局部比例保留 NaN，不构造虚假比值。
    """
    assert W_matrix.ndim == 2 and r_time.ndim == s_obs.ndim == 1
    assert W_matrix.shape[0] == r_time.size == s_obs.size
    assert 0 < window_samples <= s_obs.size and window_samples % 2 == 1
    assert np.all(np.isfinite(W_matrix)) and np.all(np.isfinite(s_obs))
    s_syn_before_scale = nonstationary_convolution(r_time, W_matrix, alignment="center")
    obs_rms = float(np.sqrt(np.mean(s_obs ** 2)))
    syn_rms = float(np.sqrt(np.mean(s_syn_before_scale ** 2)))
    assert obs_rms > 0 and syn_rms > 0
    global_ratio = syn_rms / obs_rms
    s_syn_after_scale = match_rms(s_syn_before_scale, s_obs)
    global_ratio_after_scale = float(np.sqrt(np.mean(s_syn_after_scale ** 2)) / obs_rms)
    window_weights = np.ones(window_samples) / window_samples  # shape=(window_samples,)。
    local_obs_energy = np.convolve(s_obs ** 2, window_weights, mode="valid")
    local_syn_energy = np.convolve(s_syn_before_scale ** 2, window_weights, mode="valid")
    valid_rms = local_obs_energy > 0  # shape=(N-window_samples+1,)。
    local_ratio = np.full(local_obs_energy.shape, np.nan)
    local_ratio[valid_rms] = np.sqrt(local_syn_energy[valid_rms] / local_obs_energy[valid_rms])
    local_after_global_scale = local_ratio / global_ratio
    return {
        "global_R_amp_before_scale": global_ratio,
        "global_R_amp_after_match_rms": global_ratio_after_scale,
        "local_window_samples": int(window_samples),
        "local_zero_observation_rms_windows": int(np.count_nonzero(~valid_rms)),
        "local_R_amp_before_scale_p10_p50_p90": np.nanpercentile(local_ratio, [10, 50, 90]).tolist(),
        "local_R_amp_after_global_scale_p10_p50_p90": np.nanpercentile(local_after_global_scale, [10, 50, 90]).tolist(),
    }, local_ratio, local_after_global_scale


def check_lag_ablation():
    """核验四个固定路径实验并保存诊断 JSON/NPZ，失败即抛异常。

    输入：CASE_NAMES 对应 NPZ、metrics、配置及 round2_plan.json；物理维度/单位
    由保存的 dt、时间轴和配置检查。输出 records（四个字典），并写验收文件。
    数学作用：复算既有正演/QC，检查诊断开关的数值等价和 2×2 上游冻结条件；
    不修改 baseline，不根据结果调整预先保存的非劣门槛。
    """
    assert RESULT_ROOT.is_dir()
    plan = json.loads((RESULT_ROOT / "round2_plan.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(Path(plan["input_file"]).read_bytes()).hexdigest() == plan["input_sha256"]
    for relative, digest in plan["source_sha256"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest, relative
    records = []
    bundles = []
    profiles = {}
    base_config = yaml.safe_load((ROOT / plan["baseline_config"]).read_text(encoding="utf-8"))
    for index, case_name in enumerate(CASE_NAMES):
        result_dir = RESULT_ROOT / case_name
        config_record = plan["configs"][index]
        config_path = ROOT / config_record["config_path"]
        assert config_path.read_bytes() == (result_dir / "config_used.yaml").read_bytes()
        assert hashlib.sha256(config_path.read_bytes()).hexdigest() == config_record["sha256"]
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        expected_cfg = json.loads(json.dumps(base_config))
        expected_cfg["experiment"]["name"] = cfg["experiment"]["name"]
        expected_cfg["paths"]["result_dir"] = cfg["paths"]["result_dir"]
        expected_cfg["tv_wavelet"].update(mu2=config_record["mu2"], wavelet_smooth_sigma=config_record["lag_gaussian_sigma_samples"], store_stage_wavelets=True)
        assert cfg == expected_cfg
        metrics = json.loads((result_dir / "metrics.json").read_text(encoding="utf-8"))
        with np.load(result_dir / "result_bundle.npz", allow_pickle=False) as bundle_file:
            bundle = {}
            for key in bundle_file.files:
                try:
                    bundle[key] = bundle_file[key]
                except ValueError:
                    # 原结果包的 object 标量由 metrics.json 核验，不反序列化 pickle。
                    continue
        bundles.append(bundle)
        time_s = bundle["t_work"]
        dt_s = float(bundle["dt"])
        direct_mask = bundle["tv_valid_mask"]
        dense_mask = np.ones(time_s.size, dtype=bool)  # shape=(N_time,)。
        assert bundle["W_pre_gaussian"].shape == (2546, 129)
        assert np.all(np.isnan(bundle["W_direct_centers"][~direct_mask]))
        np.testing.assert_array_equal(bundle["W_direct_centers"][direct_mask], bundle["W_pre_gaussian"][direct_mask])
        time_sigma_samples = (cfg["tv_wavelet"]["time_smooth_ms"] / 1000.0) / dt_s
        expected_gaussian = gaussian_filter(bundle["W_pre_gaussian"], sigma=(time_sigma_samples, cfg["tv_wavelet"]["wavelet_smooth_sigma"]), mode="nearest")
        np.testing.assert_array_equal(bundle["W_post_gaussian"], expected_gaussian)
        post_mean = np.mean(bundle["W_post_gaussian"], axis=1, keepdims=True)  # shape=(N_time,1)。
        np.testing.assert_array_equal(bundle["W_est"], bundle["W_post_gaussian"] - post_mean)
        np.testing.assert_array_equal(bundle["W_final_hybrid"], bundle["tv_hybrid_W"])
        final_synthetic = nonstationary_convolution(bundle["r_work_final"], bundle["W_final"], alignment="center")
        np.testing.assert_allclose(match_rms(final_synthetic, bundle["obs_work"]), bundle["s_syn_final"], rtol=1e-12, atol=1e-12)
        assert abs(corrcoef_safe(bundle["obs_work"], bundle["s_syn_final"]) - metrics["CC_final"]) < 1e-12
        attempted = bundle["tv_robust_attempted"]
        assert np.all(bundle["tv_irls_converged"][attempted])
        assert np.all(bundle["tv_robust_solution_used"][attempted])
        assert np.all(bundle["tv_student_objective_final"][attempted] <= bundle["tv_student_objective_initial"][attempted] + 1e-12)
        record = {"id": case_name[:3], "metrics": metrics, "stages": {}, "amplitude": {}, "IRLS_attempted": int(np.count_nonzero(attempted))}
        stage_names = ("W_direct_centers", "W_pre_gaussian", "W_post_gaussian", "W_est_best", "W_final_hybrid", "W_final")
        for stage_name in stage_names:
            stage_mask = direct_mask if stage_name == "W_direct_centers" else dense_mask
            record["stages"][stage_name] = {
                "available_rows": summarize_stage(bundle[stage_name], time_s, stage_mask, dt_s),
                "same_direct_centers": summarize_stage(bundle[stage_name], time_s, direct_mask, dt_s),
            }
            if stage_name != "W_direct_centers":
                window_samples = bundle["tv_robust_window_offsets_samples"].size
                summary, local_before, local_after = amplitude_diagnostics(bundle[stage_name], bundle["r_work_final"], bundle["obs_work"], window_samples)
                record["amplitude"][stage_name] = summary
                profiles[case_name[:3] + "_" + stage_name + "_before_scale"] = local_before
                profiles[case_name[:3] + "_" + stage_name + "_after_global_scale"] = local_after
                half_window = window_samples // 2
                profiles["window_center_time_s"] = time_s[half_window:-half_window]
        assert np.isclose(record["stages"]["W_est_best"]["available_rows"]["temporal_difference_energy"], metrics["candidate_temporal_difference_energy"])
        assert np.isclose(record["stages"]["W_final"]["available_rows"]["temporal_difference_energy"], metrics["final_temporal_difference_energy"])
        record["boundary_localization"] = {}
        for stage_name in ("W_pre_gaussian", "W_post_gaussian", "W_est_best"):
            boundary_mask = compute_boundary_peak_mask(bundle[stage_name], 0.12)  # shape=(N_time,)。
            origin_counts = {}
            for origin in ("inverted", "interpolated", "extrapolated", "prior_fill", "unavailable"):
                # 来源只描述填充前的谱系；不把滤波后的行视为独立直接反演结果。
                origin_mask = bundle["candidate_" + origin + "_mask"]
                origin_counts[origin] = int(np.count_nonzero(boundary_mask & origin_mask))
            indices = np.flatnonzero(boundary_mask)  # shape=(N_boundary,)。
            segments_s = []
            if indices.size:
                segment_start = int(indices[0])
                previous_index = segment_start
                for current_index in indices[1:]:
                    if current_index != previous_index + 1:
                        segments_s.append([float(time_s[segment_start]), float(time_s[previous_index])])
                        segment_start = int(current_index)
                    previous_index = int(current_index)
                segments_s.append([float(time_s[segment_start]), float(time_s[previous_index])])
            record["boundary_localization"][stage_name] = {"count": int(indices.size), "by_pre_gaussian_origin": origin_counts, "time_segments_s": segments_s}
        records.append(record)

    with np.load(ROOT / "_experiment_results/ablation/A02_no_dc/result_bundle.npz", allow_pickle=False) as previous_bundle:
        compared_array_count = 0
        for key in previous_bundle.files:
            if key in bundles[0]:
                np.testing.assert_array_equal(previous_bundle[key], bundles[0][key])
                compared_array_count += 1
    previous_metrics = json.loads((ROOT / "_experiment_results/ablation/A02_no_dc/metrics.json").read_text(encoding="utf-8"))
    assert records[0]["metrics"] == previous_metrics
    for bundle in bundles[1:]:
        for key in ("t_work", "obs_work", "r_work_final", "twt_final", "w_prior", "s_syn_after_dtw", "constant_phase_phase_cc"):
            np.testing.assert_array_equal(bundle[key], bundles[0][key])
    for on_index, off_index in ((0, 2), (1, 3)):
        for key in ("W_direct_centers", "W_pre_gaussian", "tv_valid_mask", "tv_student_objective_final"):
            np.testing.assert_array_equal(bundles[on_index][key], bundles[off_index][key])

    baseline_metrics = records[0]["metrics"]
    for record in records:
        metrics = record["metrics"]
        delta_cc = metrics["CC_final"] - baseline_metrics["CC_final"]
        record["delta_CC_final"] = delta_cc
        record["temporal_change_pct"] = {}
        for name, stage in (("candidate", "W_est_best"), ("final", "W_final")):
            current = record["stages"][stage]["available_rows"]
            reference = records[0]["stages"][stage]["available_rows"]
            record["temporal_change_pct"][name] = 100 * (current["temporal_difference_energy"] / reference["temporal_difference_energy"] - 1)
            record["temporal_change_pct"][name + "_normalized"] = 100 * (current["normalized_temporal_energy"] / reference["normalized_temporal_energy"] - 1)
        gates = {
            "CC_final_noninferior": delta_cc >= plan["criteria"]["delta_CC_floor"],
            "CC_tv_direct_noninferior": metrics["CC_tv_direct"] - baseline_metrics["CC_tv_direct"] >= plan["criteria"]["delta_CC_floor"],
            "W_pass_no_regression": bool(metrics["W_pass"]),
            "valid_ratio_no_regression": metrics["valid_ratio"] >= baseline_metrics["valid_ratio"],
            "final_edge_p90_within_reliable_threshold": metrics["final_edge_energy_p90"] <= plan["criteria"]["edge_reliable"],
            "final_side_p90_no_new_severe_crossing": metrics["final_side_lobe_p90"] <= plan["criteria"]["side_lobe_severe"],
            "final_peak_p90_no_increase": metrics["final_peak_abs_p90_after_local_fallback"] <= baseline_metrics["final_peak_abs_p90_after_local_fallback"] + 1e-6,
            "final_abs_lag_no_increase": abs(metrics["CC_final_best_lag_ms"]) <= abs(baseline_metrics["CC_final_best_lag_ms"]) + 1e-6,
            "candidate_Et_increase_le_15pct": record["temporal_change_pct"]["candidate"] <= plan["criteria"]["non_temporal_Et_increase_limit_pct"],
            "final_Et_increase_le_15pct": record["temporal_change_pct"]["final"] <= plan["criteria"]["non_temporal_Et_increase_limit_pct"],
        }
        record["gates"] = gates
        record["failed_gates"] = [key for key, passed in gates.items() if not passed]
        record["strict_screening_pass"] = all(gates.values())
        print(record["id"], f"CC_final={metrics['CC_final']:.9f}", "raw_W_pass=", metrics["raw_tv_W_pass"], "failed gates:", record["failed_gates"])
    factorial = {}
    for label, from_index, to_index in (("mu2_off_lag_on", 0, 1), ("mu2_off_lag_off", 2, 3), ("lag_off_mu2_on", 0, 2), ("lag_off_mu2_off", 1, 3)):
        factorial[label] = records[to_index]["metrics"]["CC_final"] - records[from_index]["metrics"]["CC_final"]
    acceptance = {"status": "completed", "execution_acceptance": "4_of_4_passed", "L00_equals_A02": True, "old_numeric_arrays_compared": compared_array_count, "lag_pairs_have_identical_pre_gaussian_inputs": True, "criteria": plan["criteria"], "factorial_CC_differences": factorial, "records": records, "baseline_replaced": False}
    (RESULT_ROOT / "lag_acceptance.json").write_text(json.dumps(acceptance, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    np.savez_compressed(RESULT_ROOT / "rms_profiles.npz", **profiles)
    return records


if __name__ == "__main__":
    check_lag_ablation()
