# -*- coding: utf-8 -*-
"""Validate phase-3 weight-map storage, plots and comparison tables."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.compare_l2_student_t import compare_result_directories
from plotting.plot_qc import (
    plot_student_t_time_diagnostics,
    plot_student_t_weight_map,
)
from utils.wavelet_inversion_robust import time_varying_wavelet_inversion


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--output-dir", default="_phase3_test_output")
    args = parser.parse_args()

    bundle_path = Path(args.bundle)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(bundle_path, allow_pickle=True) as bundle_file:
        bundle = {key: np.asarray(bundle_file[key]) for key in bundle_file.files}

    wavelet_length = int(np.asarray(bundle["w_prior"]).size)
    W, diag = time_varying_wavelet_inversion(
        r_time=bundle["r_work_final"],
        s_obs=bundle["obs_work"],
        wavelet_length=wavelet_length,
        data_window_length=2 * wavelet_length + 1,
        dt=float(bundle["dt"]),
        wavelet_alignment="center",
        peak_allowed_ms=(-15.0, 15.0),
        mu1=0.15,
        mu2=4.0,
        mu_dc=15.0,
        w_prior=bundle["w_prior"],
        mu_prior=1.0,
        mu_time=1.5,
        energy_percentile=25.0,
        damping_ratio=0.003,
        svd_cutoff_ratio=0.001,
        estimate_step_ms=5.0,
        time_smooth_ms=20.0,
        wavelet_smooth_sigma=3.0,
        peak_lock=True,
        max_peak_shift_ms=15.0,
        reject_ill_conditioned=True,
        reject_amplitude_jumps=True,
        loss_type="student_t",
        student_nu=10.0,
        irls_max_iter=10,
        irls_tol=1.0e-4,
        robust_scale_mode="local_mad",
        robust_scale_floor_ratio=0.05,
        robust_weight_floor=1.0e-3,
        min_effective_sample_ratio=0.30,
        robust_fallback="l2",
        robust_outlier_weight_threshold=0.5,
        store_weight_map=True,
        return_diagnostics=True,
        verbose=False,
    )

    weight_map = np.asarray(diag["robust_weight_map"])
    weight_centers = np.asarray(diag["robust_weight_center_indices"])
    offsets = np.asarray(diag["robust_window_offsets_samples"])
    attempted = np.asarray(diag["robust_attempted"], dtype=bool)

    assert weight_map.ndim == 2
    assert weight_map.shape[0] == int(np.sum(attempted))
    assert weight_map.shape[1] == offsets.size
    assert weight_centers.size == weight_map.shape[0]
    assert np.all(np.isfinite(weight_map))
    assert np.all((weight_map >= 0.0) & (weight_map <= 1.0))

    time_plot = plot_student_t_time_diagnostics(
        t_work=bundle["t_work"],
        robust_sigma=diag["robust_sigma"],
        weight_mean=diag["weight_mean"],
        weight_min=diag["weight_min"],
        outlier_fraction=diag["outlier_fraction"],
        effective_sample_ratio=diag["effective_sample_ratio"],
        robust_code=diag["robust_code"],
        robust_attempted=diag["robust_attempted"],
        robust_solution_used=diag["robust_solution_used"],
        irls_iterations=diag["irls_iterations"],
        irls_converged=diag["irls_converged"],
        objective_relative_decrease=diag["student_objective_relative_decrease"],
        min_effective_sample_ratio=float(diag["min_effective_sample_ratio"]),
        robust_scale_floor=float(diag["robust_scale_floor"]),
        outlier_weight_threshold=float(diag["robust_outlier_weight_threshold"]),
        result_dir=output_dir,
    )
    heatmap_plot = plot_student_t_weight_map(
        t_work=bundle["t_work"],
        dt=float(bundle["dt"]),
        weight_map=weight_map,
        center_indices=weight_centers,
        window_offsets_samples=offsets,
        outlier_weight_threshold=0.5,
        result_dir=output_dir,
    )
    assert time_plot is not None and Path(time_plot).exists()
    assert heatmap_plot is not None and Path(heatmap_plot).exists()

    l2_dir = bundle_path.parent
    mock_student_dir = output_dir / "mock_student_result"
    mock_student_dir.mkdir(parents=True, exist_ok=True)

    student_arrays = dict(bundle)
    student_arrays["W_est"] = W
    student_arrays["W_est_best"] = W
    student_arrays["tv_valid_mask"] = np.asarray(diag["valid_mask"])
    student_arrays["tv_skip_code"] = np.asarray(diag["skip_code"])
    mapping = {
        "tv_robust_sigma": "robust_sigma",
        "tv_irls_iterations": "irls_iterations",
        "tv_irls_converged": "irls_converged",
        "tv_robust_code": "robust_code",
        "tv_robust_attempted": "robust_attempted",
        "tv_robust_solution_used": "robust_solution_used",
        "tv_weight_mean": "weight_mean",
        "tv_weight_min": "weight_min",
        "tv_outlier_fraction": "outlier_fraction",
        "tv_effective_sample_size": "effective_sample_size",
        "tv_effective_sample_ratio": "effective_sample_ratio",
        "tv_student_objective_initial": "student_objective_initial",
        "tv_student_objective_final": "student_objective_final",
        "tv_student_objective_relative_decrease": "student_objective_relative_decrease",
        "tv_robust_weight_map": "robust_weight_map",
        "tv_robust_weight_center_indices": "robust_weight_center_indices",
        "tv_robust_window_offsets_samples": "robust_window_offsets_samples",
    }
    for output_key, diag_key in mapping.items():
        student_arrays[output_key] = np.asarray(diag[diag_key])

    np.savez_compressed(mock_student_dir / "result_bundle.npz", **student_arrays)

    source_metrics = l2_dir / "metrics.json"
    if source_metrics.exists():
        shutil.copy2(source_metrics, mock_student_dir / "metrics.json")
    else:
        with open(mock_student_dir / "metrics.json", "w", encoding="utf-8") as file:
            json.dump({}, file)

    comparison_paths = compare_result_directories(
        l2_result_dir=l2_dir,
        student_t_result_dir=mock_student_dir,
        output_dir=output_dir / "comparison",
    )
    assert all(path.exists() for path in comparison_paths.values())

    print(f"weight_map_shape = {weight_map.shape}")
    print(f"time_plot        = {time_plot}")
    print(f"heatmap_plot     = {heatmap_plot}")
    print(f"comparison_csv   = {comparison_paths['csv']}")
    print("Student-t phase-3 output test PASSED")


if __name__ == "__main__":
    main()
