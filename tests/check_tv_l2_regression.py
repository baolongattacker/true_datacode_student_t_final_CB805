# -*- coding: utf-8 -*-
"""比较重构前保存的 L2 基线与重构后的完整时变子波结果。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.wavelet_inversion_robust import time_varying_wavelet_inversion


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True, help="现有 result_bundle.npz")
    parser.add_argument("--baseline", required=True, help="修改前导出的 L2 基线 NPZ")
    parser.add_argument("--rtol", type=float, default=1e-8)
    args = parser.parse_args()

    bundle = np.load(args.bundle, allow_pickle=True)
    baseline = np.load(args.baseline, allow_pickle=True)

    wavelet_length = int(np.asarray(bundle["w_prior"]).size)
    data_window_length = 2 * wavelet_length + 1

    W_new, diag_new = time_varying_wavelet_inversion(
        r_time=bundle["r_work_final"],
        s_obs=bundle["obs_work"],
        wavelet_length=wavelet_length,
        data_window_length=data_window_length,
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
        loss_type="l2",
        return_diagnostics=True,
        verbose=False,
    )

    W_old = baseline["W"]
    valid_old = baseline["valid_mask"]
    skip_old = baseline["skip_code"]

    rel_error = np.linalg.norm(W_new - W_old) / (np.linalg.norm(W_old) + 1e-12)
    max_abs_error = float(np.max(np.abs(W_new - W_old)))
    valid_equal = np.array_equal(diag_new["valid_mask"], valid_old)
    skip_equal = np.array_equal(diag_new["skip_code"], skip_old)

    print(f"relative_W_error = {rel_error:.3e}")
    print(f"max_abs_W_error  = {max_abs_error:.3e}")
    print(f"valid_mask_equal = {valid_equal}")
    print(f"skip_code_equal  = {skip_equal}")

    if rel_error >= args.rtol or not valid_equal or not skip_equal:
        raise SystemExit("L2 regression FAILED")

    print("L2 regression PASSED")


if __name__ == "__main__":
    main()
