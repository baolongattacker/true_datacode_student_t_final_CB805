# -*- coding: utf-8 -*-
"""Focused tests for phase-5 paired statistics and global weight metrics."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.analyze_synthetic_student_t_benchmark import analyze_benchmark
from experiments.run_synthetic_student_t_benchmark import _weight_detection_metrics


def _write_rows(path: Path) -> None:
    rows = []
    for seed in range(5):
        # Clean case: all Student-t methods remain within the 0.005 margin.
        rows.append(
            {
                "method": "L2",
                "method_key": "l2",
                "loss_type": "l2",
                "student_nu": "",
                "seed": seed,
                "snr_db": 25.0,
                "outlier_ratio": 0.0,
                "outlier_amplitude_sigma": 0.0,
                "wavelet_nrmse_scaled": 0.060 + 0.001 * seed,
                "wavelet_nrmse_absolute": 0.080,
                "wavelet_median_row_correlation": 0.99,
                "centroid_frequency_mae_hz": 1.0,
                "clean_trace_nrmse": 0.07,
                "clean_trace_cc": 0.99,
                "observed_trace_cc": 0.98,
                "valid_ratio": 0.90,
                "elapsed_seconds": 0.1,
            }
        )
        for nu, clean_delta in ((10.0, 0.001), (5.0, 0.002), (3.0, 0.004)):
            rows.append(
                {
                    "method": f"Student-t nu={nu:g}",
                    "method_key": f"student_t_nu{nu:g}".replace(".", "p"),
                    "loss_type": "student_t",
                    "student_nu": nu,
                    "seed": seed,
                    "snr_db": 25.0,
                    "outlier_ratio": 0.0,
                    "outlier_amplitude_sigma": 0.0,
                    "wavelet_nrmse_scaled": 0.060 + 0.001 * seed + clean_delta,
                    "wavelet_nrmse_absolute": 0.081,
                    "wavelet_median_row_correlation": 0.99,
                    "centroid_frequency_mae_hz": 1.0,
                    "clean_trace_nrmse": 0.07,
                    "clean_trace_cc": 0.99,
                    "observed_trace_cc": 0.98,
                    "valid_ratio": 0.90,
                    "weight_global_f1": 0.0,
                    "weight_global_false_positive_rate": 0.01 * (11.0 - nu),
                    "weight_global_coverage_ratio": 0.95,
                    "elapsed_seconds": 0.4,
                }
            )

        # Contaminated case: nu=10 has the strongest reliable improvement.
        l2_nrmse = 0.100 + 0.002 * seed
        rows.append(
            {
                "method": "L2",
                "method_key": "l2",
                "loss_type": "l2",
                "student_nu": "",
                "seed": seed,
                "snr_db": 25.0,
                "outlier_ratio": 0.05,
                "outlier_amplitude_sigma": 10.0,
                "wavelet_nrmse_scaled": l2_nrmse,
                "wavelet_nrmse_absolute": 0.13,
                "wavelet_median_row_correlation": 0.95,
                "centroid_frequency_mae_hz": 2.0,
                "clean_trace_nrmse": 0.12,
                "clean_trace_cc": 0.95,
                "observed_trace_cc": 0.90,
                "valid_ratio": 0.85,
                "elapsed_seconds": 0.1,
            }
        )
        for nu, gain, f1, fpr in (
            (10.0, 0.020, 0.92, 0.01),
            (5.0, 0.016, 0.82, 0.03),
            (3.0, 0.012, 0.65, 0.08),
        ):
            rows.append(
                {
                    "method": f"Student-t nu={nu:g}",
                    "method_key": f"student_t_nu{nu:g}".replace(".", "p"),
                    "loss_type": "student_t",
                    "student_nu": nu,
                    "seed": seed,
                    "snr_db": 25.0,
                    "outlier_ratio": 0.05,
                    "outlier_amplitude_sigma": 10.0,
                    "wavelet_nrmse_scaled": l2_nrmse - gain,
                    "wavelet_nrmse_absolute": 0.11,
                    "wavelet_median_row_correlation": 0.97,
                    "centroid_frequency_mae_hz": 1.5,
                    "clean_trace_nrmse": 0.10,
                    "clean_trace_cc": 0.97,
                    "observed_trace_cc": 0.91,
                    "valid_ratio": 0.90,
                    "weight_global_f1": f1,
                    "weight_global_false_positive_rate": fpr,
                    "weight_global_coverage_ratio": 0.95,
                    "elapsed_seconds": 0.4,
                }
            )

    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_global_weight_aggregation() -> None:
    diagnostics = {
        "robust_weight_map": np.asarray(
            [
                [0.9, 0.1, 0.1],
                [0.1, 0.1, 0.9],
            ]
        ),
        "robust_weight_center_indices": np.asarray([1, 2]),
        "robust_window_offsets_samples": np.asarray([-1, 0, 1]),
    }
    outlier_mask = np.asarray([False, True, True, False, False])
    metrics = _weight_detection_metrics(
        diagnostics=diagnostics,
        outlier_mask=outlier_mask,
        threshold=0.5,
    )
    assert metrics["weight_global_recall"] == 1.0
    assert metrics["weight_global_precision"] == 1.0
    assert metrics["weight_global_f1"] == 1.0
    assert metrics["weight_global_coverage_ratio"] == 0.8


def test_paired_recommendation() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        raw_csv = root / "raw.csv"
        _write_rows(raw_csv)
        outputs = analyze_benchmark(
            raw_csv=raw_csv,
            output_dir=root / "analysis",
            bootstrap_reps=300,
            clean_noninferiority_margin=0.005,
            min_contaminated_win_rate=0.60,
            max_global_false_positive_rate=0.10,
            random_seed=123,
        )
        recommendation = json.loads(
            Path(outputs["recommendation_json"]).read_text(encoding="utf-8")
        )
        assert recommendation["recommendation_available"] is True
        assert recommendation["recommended_student_nu"] == 10.0
        assert Path(outputs["paired_csv"]).exists()
        assert Path(outputs["summary_csv"]).exists()


def main() -> None:
    test_global_weight_aggregation()
    test_paired_recommendation()
    print("Student-t phase-5 statistical tests PASSED")


if __name__ == "__main__":
    main()
