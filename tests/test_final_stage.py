# -*- coding: utf-8 -*-
"""Final-stage regression tests for import hygiene and heterogeneous stress analysis."""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

import numpy as np

from analysis.analyze_final_student_t_stress_test import analyze_final_stress_test
from core.forward_operator import nonstationary_convolution
from experiments.run_final_student_t_stress_test import (
    _build_scenario_data,
)
from experiments.run_synthetic_student_t_benchmark import (
    DEFAULT_MODEL_SETTINGS,
    _build_reflectivity,
    _build_true_wavelet_matrix,
)


def test_scenario_generation_is_reproducible() -> None:
    n_time = 320
    dt = 0.001
    wavelet_length = 41
    reflectivity = _build_reflectivity(
        n_time,
        np.random.default_rng(7),
    )
    W_true = _build_true_wavelet_matrix(n_time, dt, wavelet_length)
    clean = nonstationary_convolution(reflectivity, W_true, alignment="center")
    scenario = {
        "group": "combined",
        "impulse_ratio": 0.03,
        "impulse_amplitude_sigma": 8.0,
        "ar1_rho": 0.7,
        "missing_reflector_fraction": 0.1,
        "prior_frequency_scale": 1.2,
    }
    first = _build_scenario_data(
        scenario_key="combined_error",
        scenario=scenario,
        seed=4,
        clean_trace=clean,
        reflectivity_true=reflectivity,
        W_true=W_true,
        dt=dt,
        snr_db=25.0,
        model_settings=dict(DEFAULT_MODEL_SETTINGS),
    )
    second = _build_scenario_data(
        scenario_key="combined_error",
        scenario=scenario,
        seed=4,
        clean_trace=clean,
        reflectivity_true=reflectivity,
        W_true=W_true,
        dt=dt,
        snr_db=25.0,
        model_settings=dict(DEFAULT_MODEL_SETTINGS),
    )
    assert np.array_equal(first["observed"], second["observed"])
    assert np.array_equal(
        first["reflectivity_inversion"], second["reflectivity_inversion"]
    )
    assert np.array_equal(first["target_mask"], second["target_mask"])
    assert np.any(first["target_mask"])


def _write_fake_final_rows(path: Path) -> None:
    rows: list[dict[str, object]] = []
    scenarios = ["clean_gaussian", "impulse_noise", "reflectivity_missing_events"]
    for seed in range(5):
        for scenario in scenarios:
            l2_nrmse = 0.060 + 0.002 * seed if scenario == "clean_gaussian" else 0.115 + 0.002 * seed
            st_nrmse = l2_nrmse + 0.001 if scenario == "clean_gaussian" else l2_nrmse - 0.020
            for method_key, method, nu, nrmse, valid in (
                ("l2", "L2", "", l2_nrmse, 0.86),
                ("student_t_nu10", "Student-t nu=10", 10.0, st_nrmse, 0.89),
            ):
                rows.append(
                    {
                        "case_id": f"seed={seed}|scenario={scenario}|method={method_key}",
                        "seed": seed,
                        "scenario_key": scenario,
                        "scenario_group": "clean" if scenario == "clean_gaussian" else "stress",
                        "method_key": method_key,
                        "method": method,
                        "student_nu": nu,
                        "wavelet_nrmse_absolute": nrmse + 0.02,
                        "wavelet_nrmse_scaled": nrmse,
                        "wavelet_median_row_correlation": 0.98,
                        "centroid_frequency_mae_hz": 1.0,
                        "clean_trace_nrmse_true_r": nrmse,
                        "clean_trace_cc_true_r": 0.98,
                        "observed_fit_cc_inversion_r": 0.95,
                        "valid_ratio": valid,
                        "weight_global_false_positive_rate": 0.02,
                        "weight_global_coverage_ratio": 0.95,
                        "elapsed_seconds": 0.1,
                    }
                )
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_final_assessment_accepts_noninferior_robust_method() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        raw = root / "raw.csv"
        _write_fake_final_rows(raw)
        outputs = analyze_final_stress_test(
            raw_csv=raw,
            output_dir=root / "analysis",
            bootstrap_reps=300,
            clean_noninferiority_margin=0.005,
            minimum_stress_win_rate=0.60,
            maximum_worst_scenario_degradation=0.01,
            maximum_valid_ratio_drop=0.05,
            random_seed=123,
        )
        result = json.loads(
            Path(outputs["assessment_json"]).read_text(encoding="utf-8")
        )
        assert result["final_validation_passed"] is True
        assert result["recommended_student_nu"] == 10.0


def test_stage_imports_resolve_inside_project() -> None:
    import stages.stage_dtw as stage_dtw
    import stages.stage_q_constraint as stage_q
    import stages.stage_stationary as stage_stationary
    import stages.stage_tv_wavelet as stage_tv

    project_root = Path(__file__).resolve().parents[1]
    for module in (stage_dtw, stage_q, stage_stationary, stage_tv):
        path = Path(module.__file__).resolve()
        path.relative_to(project_root)
