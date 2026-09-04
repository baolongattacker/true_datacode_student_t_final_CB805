# -*- coding: utf-8 -*-
"""
Run a list of configs through the final experiment runner and summarize metrics.
顺序执行：像流水线一样逐个跑完你给出的所有 YAML 配置。
指标提取：从每个实验产生的 metrics.json 中提取核心数据（如 CC_final, W_pass, Q_global）。
报表生成：自动创建一个 CSV，每一行对应一个配置文件，方便你一眼看出哪组参数的校正效果最好。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.config_loader import load_config
from experiments.run_real_experiment import main as run_experiment

# 定义汇总字段，决定了最终CSV的列名
SUMMARY_FIELDS = [
    "config_name",
    "experiment_name",
    "result_dir",
    "alignment",
    "wavelet_length_s",
    "q_enable",
    "edge_penalty_enabled",
    "mu_edge_configured",
    "mu_edge_effective",
    "CC_stationary",
    "CC_after_DTW",
    "CC_prior_after_DTW",
    "CC_tv_direct",
    "CC_final",
    "W_pass",
    "q_pass",
    "CC_Q",
    "Q_global",
    "valid_ratio",
    "candidate_valid_ratio",
    "candidate_edge_energy_p50",
    "candidate_edge_energy_p90",
    "candidate_edge_energy_max",
    "candidate_side_lobe_p50",
    "candidate_side_lobe_p90",
    "final_edge_energy_p50",
    "final_edge_energy_p90",
    "final_edge_energy_max",
    "final_side_lobe_p50",
    "final_side_lobe_p90",
    "candidate_boundary_peak_fraction",
    "final_boundary_peak_fraction",
    "candidate_reliable_ratio",
    "final_reliable_ratio",
    "longest_candidate_invalid_gap_ms",
    "longest_candidate_unreliable_gap_ms",
    "stationary_candidate_accepted",
    "stationary_candidate_peak_ms",
    "stationary_rejection_code",
    "stationary_prior_source",
    "CC_stationary_raw",
    "candidate_peak_violation_ratio",
    "student_objective_includes_edge",
    "peak_metric_p10",
    "peak_metric_med",
    "peak_metric_p90",
    "peak_abs_p90",
    "candidate_peak_metric_p10",
    "candidate_peak_metric_med",
    "candidate_peak_metric_p90",
    "candidate_peak_abs_p90",
    "final_peak_metric_p10",
    "final_peak_metric_med",
    "final_peak_metric_p90",
    "final_peak_abs_p90",
    "candidate_temporal_difference_energy",
    "final_temporal_difference_energy",
    "irls_iterations_median",
    "irls_iterations_max",
    "best_lag_ms",
    "candidate_best_lag_ms",
    "final_best_lag_ms",
    "final_model_type",
    "acceptance_reasons",
    "q_reason",
]


def _load_metrics(metrics_path: str | Path) -> dict[str, Any]:
    with open(metrics_path, "r", encoding="utf-8") as file:
        return json.load(file)


def build_summary_row(config_path: str | Path, cfg, metrics: dict[str, Any]) -> dict[str, Any]:
    config_path = Path(config_path)

    acceptance_reasons = metrics.get("acceptance_reasons", [])
    acceptance_reason_text = "|".join(acceptance_reasons)

    row: dict[str, Any] = {}
    row["config_name"] = config_path.stem
    row["experiment_name"] = cfg.experiment.name
    row["result_dir"] = str(cfg.paths.result_dir)
    row["alignment"] = cfg.wavelet.alignment
    row["wavelet_length_s"] = float(cfg.wavelet.length_s)
    row["q_enable"] = bool(cfg.q.enable)
    row["edge_penalty_enabled"] = metrics.get("edge_penalty_enabled")
    row["mu_edge_configured"] = metrics.get("mu_edge_configured")
    row["mu_edge_effective"] = metrics.get("mu_edge_effective")

    row["CC_stationary"] = metrics.get("CC_stationary")
    row["CC_after_DTW"] = metrics.get("CC_after_DTW")
    row["CC_prior_after_DTW"] = metrics.get("CC_prior_after_DTW")
    row["CC_tv_direct"] = metrics.get("CC_tv_direct")
    row["CC_final"] = metrics.get("CC_final")

    row["W_pass"] = metrics.get("W_pass")
    row["q_pass"] = metrics.get("q_pass")
    row["CC_Q"] = metrics.get("CC_Q")
    row["Q_global"] = metrics.get("Q_global")

    row["valid_ratio"] = metrics.get("valid_ratio")
    row["candidate_valid_ratio"] = metrics.get(
        "candidate_valid_ratio",
        metrics.get("valid_ratio"),
    )
    for field_name in (
        "candidate_edge_energy_p50",
        "candidate_edge_energy_p90",
        "candidate_edge_energy_max",
        "candidate_side_lobe_p50",
        "candidate_side_lobe_p90",
        "final_edge_energy_p50",
        "final_edge_energy_p90",
        "final_edge_energy_max",
        "final_side_lobe_p50",
        "final_side_lobe_p90",
        "candidate_boundary_peak_fraction",
        "final_boundary_peak_fraction",
        "candidate_reliable_ratio",
        "final_reliable_ratio",
        "longest_candidate_invalid_gap_ms",
        "longest_candidate_unreliable_gap_ms",
        "stationary_candidate_accepted",
        "stationary_candidate_peak_ms",
        "stationary_rejection_code",
        "stationary_prior_source",
        "CC_stationary_raw",
        "candidate_peak_violation_ratio",
        "student_objective_includes_edge",
        "peak_metric_p10",
        "peak_metric_med",
        "peak_metric_p90",
        "candidate_temporal_difference_energy",
        "final_temporal_difference_energy",
        "irls_iterations_median",
        "irls_iterations_max",
    ):
        row[field_name] = metrics.get(field_name)
    row["peak_abs_p90"] = metrics.get("peak_abs_p90")
    row["candidate_peak_metric_p10"] = metrics.get(
        "candidate_peak_metric_p10",
        metrics.get("peak_metric_p10"),
    )
    row["candidate_peak_metric_med"] = metrics.get(
        "candidate_peak_metric_med",
        metrics.get("peak_metric_med"),
    )
    row["candidate_peak_metric_p90"] = metrics.get(
        "candidate_peak_metric_p90",
        metrics.get("peak_metric_p90"),
    )
    row["candidate_peak_abs_p90"] = metrics.get(
        "candidate_peak_abs_p90",
        metrics.get("peak_abs_p90"),
    )
    row["final_peak_metric_p10"] = metrics.get(
        "final_peak_metric_p10_after_local_fallback"
    )
    row["final_peak_metric_med"] = metrics.get(
        "final_peak_metric_med_after_local_fallback"
    )
    row["final_peak_metric_p90"] = metrics.get(
        "final_peak_metric_p90_after_local_fallback"
    )
    row["final_peak_abs_p90"] = metrics.get(
        "final_peak_abs_p90_after_local_fallback"
    )
    row["best_lag_ms"] = metrics.get("best_lag_ms")
    row["candidate_best_lag_ms"] = metrics.get(
        "candidate_best_lag_ms",
        metrics.get("best_lag_ms"),
    )
    row["final_best_lag_ms"] = metrics.get("CC_final_best_lag_ms")
    row["final_model_type"] = metrics.get("final_model_type")
    row["acceptance_reasons"] = acceptance_reason_text
    row["q_reason"] = metrics.get("q_reason")

    return row


def write_summary_csv(rows: list[dict[str, Any]], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def run_matrix(config_paths: list[str], summary_csv: str | Path) -> tuple[list[dict[str, Any]], Path]:
    rows: list[dict[str, Any]] = []

    for config_path in config_paths:
        cfg = load_config(config_path)
        print(f"[Matrix] running {config_path}")
        run_experiment(config_path)

        metrics_path = Path(cfg.paths.result_dir) / "metrics.json"
        metrics = _load_metrics(metrics_path)
        rows.append(build_summary_row(config_path, cfg, metrics))

    summary_path = write_summary_csv(rows, summary_csv)
    print(f"[Matrix] summary: {summary_path}")
    return rows, summary_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="+", help="Config files to run in sequence.")
    parser.add_argument(
        "--summary-csv",
        default="_experiment_results/matrix_summary.csv",
        help="Path to write aggregated metrics CSV.",
    )
    args = parser.parse_args()

    run_matrix(args.configs, args.summary_csv)
