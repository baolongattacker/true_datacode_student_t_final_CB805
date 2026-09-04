# -*- coding: utf-8 -*-
"""使用现有 CB803 result_bundle 做 Student-t IRLS 冒烟测试。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.wavelet_inversion_robust import time_varying_wavelet_inversion
from experiments.run_tv_loss_ablation import _infer_w_prior_source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--nu", type=float, default=10.0)
    parser.add_argument("--max-iter", type=int, default=10)
    args = parser.parse_args()

    bundle = np.load(args.bundle, allow_pickle=True)
    baseline = {key: np.asarray(bundle[key]) for key in bundle.files}
    wavelet_length = int(np.asarray(bundle["w_prior"]).size)
    prior_source = _infer_w_prior_source(
        requested_source="auto",
        baseline=baseline,
        alignment="center",
    )
    mu_prior = (
        1.0
        if prior_source == "strict_stationary_center_after_DTW"
        else 0.8
    )

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
        mu_prior=mu_prior,
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
        student_nu=args.nu,
        irls_max_iter=args.max_iter,
        irls_tol=1.0e-4,
        robust_scale_mode="local_mad",
        robust_scale_floor_ratio=0.05,
        robust_weight_floor=1.0e-3,
        min_effective_sample_ratio=0.30,
        robust_fallback="l2",
        return_diagnostics=True,
        verbose=False,
    )

    attempted = np.asarray(diag["robust_attempted"], dtype=bool)
    used = np.asarray(diag["robust_solution_used"], dtype=bool)
    initial = np.asarray(diag["student_objective_initial"], dtype=float)
    final = np.asarray(diag["student_objective_final"], dtype=float)
    n_eff_ratio = np.asarray(diag["effective_sample_ratio"], dtype=float)

    assert np.all(np.isfinite(W))
    assert int(np.sum(attempted)) > 0
    assert int(np.sum(used)) > 0
    assert np.all(final[used] <= initial[used] + 1.0e-10 * np.maximum(1.0, np.abs(initial[used])))
    assert np.all(n_eff_ratio[attempted] > 0.0)
    assert not np.any(np.asarray(diag["skip_code"]) == -3)

    print(f"prior_source          = {prior_source}")
    print(f"mu_prior              = {mu_prior}")
    print(f"robust_attempted      = {int(np.sum(attempted))}")
    print(f"robust_solution_used  = {int(np.sum(used))}")
    print(f"robust_converged      = {int(np.sum(diag['irls_converged']))}")
    print(f"valid_windows         = {int(diag['n_valid'])}")
    print(
        "median_weight_mean    = "
        f"{float(np.nanmedian(np.asarray(diag['weight_mean'])[attempted])):.6f}"
    )
    print(
        "median_effective_ratio = "
        f"{float(np.nanmedian(n_eff_ratio[attempted])):.6f}"
    )
    print("Student-t CB803 smoke test PASSED")


if __name__ == "__main__":
    main()
