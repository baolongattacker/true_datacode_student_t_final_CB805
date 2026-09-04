# -*- coding: utf-8 -*-
"""Phase-4 regression checks for status-aware weights and prior-source inference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.run_tv_loss_ablation import _infer_w_prior_source
from plotting.plot_qc import plot_student_t_weight_map
from utils.wavelet_inversion_robust import time_varying_wavelet_inversion


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    bundle_path = Path(args.bundle)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(bundle_path, allow_pickle=True) as bundle:
        baseline = {key: np.asarray(bundle[key]) for key in bundle.files}

    source = _infer_w_prior_source(
        requested_source="auto",
        baseline=baseline,
        alignment="center",
    )
    assert source == "center_ricker_prior_after_DTW", source

    r = np.asarray(baseline["r_work_final"], dtype=float)
    obs = np.asarray(baseline["obs_work"], dtype=float)
    w_prior = np.asarray(baseline["w_prior"], dtype=float)
    dt = float(np.asarray(baseline["dt"]).reshape(()))

    _, diag = time_varying_wavelet_inversion(
        r_time=r,
        s_obs=obs,
        wavelet_length=w_prior.size,
        data_window_length=259,
        dt=dt,
        wavelet_alignment="center",
        peak_allowed_ms=(-15.0, 15.0),
        mu1=0.15,
        mu2=4.0,
        mu_dc=15.0,
        w_prior=w_prior,
        use_stationary_prior=False,
        mu_prior=0.8,
        mu_time=1.5,
        energy_percentile=25.0,
        damping_ratio=0.003,
        svd_cutoff_ratio=0.001,
        estimate_step_ms=5.0,
        time_smooth_ms=20.0,
        wavelet_smooth_sigma=3.0,
        reject_ill_conditioned=True,
        reject_amplitude_jumps=True,
        loss_type="student_t",
        student_nu=10.0,
        irls_max_iter=10,
        irls_tol=1e-4,
        robust_scale_mode="local_mad",
        robust_scale_floor_ratio=0.05,
        robust_weight_floor=1e-3,
        min_effective_sample_ratio=0.30,
        robust_fallback="l2",
        robust_outlier_weight_threshold=0.5,
        store_weight_map=True,
        return_diagnostics=True,
        verbose=False,
    )

    centers = np.asarray(diag["robust_weight_center_indices"], dtype=int)
    weight_map = np.asarray(diag["robust_weight_map"])
    solution_used = np.asarray(diag["robust_weight_solution_used"], dtype=bool)
    post_qc_valid = np.asarray(diag["robust_weight_post_qc_valid"], dtype=bool)
    row_skip = np.asarray(diag["robust_weight_skip_code"], dtype=int)
    valid_mask = np.asarray(diag["valid_mask"], dtype=bool)
    skip_code = np.asarray(diag["skip_code"], dtype=int)

    assert weight_map.shape[0] == centers.size
    assert solution_used.size == centers.size
    assert post_qc_valid.size == centers.size
    assert row_skip.size == centers.size
    assert np.array_equal(post_qc_valid, valid_mask[centers])
    assert np.array_equal(row_skip, skip_code[centers])

    figure = plot_student_t_weight_map(
        t_work=np.asarray(baseline["t_work"], dtype=float),
        dt=dt,
        weight_map=weight_map,
        center_indices=centers,
        window_offsets_samples=diag["robust_window_offsets_samples"],
        outlier_weight_threshold=0.5,
        solution_used=solution_used,
        post_qc_valid=post_qc_valid,
        skip_code=row_skip,
        row_filter="all",
        result_dir=output_dir,
    )
    assert figure is not None and Path(figure).exists()

    print(f"prior_source       = {source}")
    print(f"weight_map_shape   = {weight_map.shape}")
    print(f"post_qc_valid_rows = {int(np.sum(post_qc_valid))}")
    print(f"figure             = {figure}")
    print("Student-t phase-4 checks PASSED")


if __name__ == "__main__":
    main()
