# -*- coding: utf-8 -*-
"""
Run a strict, fixed-input loss-function ablation for time-varying wavelets.

The script loads the already completed DTW/prior result bundle and keeps
r_work_final, obs_work, w_prior and dt identical for every method.  It then
runs only the TV-wavelet stage for L2 and selected Student-t nu values.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.compare_student_t_sweep import compare_sweep_directories
from configs.config_loader import load_config
from core.acceptance import TvAcceptanceConfig
from core.metrics import corrcoef_safe, envelope_cc
from stages.stage_tv_wavelet import TvWaveletResult, run_tv_wavelet_stage
from stages.stage_wavelet_prior import make_center_ricker_wavelet


def _make_odd(value: int) -> int:
    value = int(value)
    return value if value % 2 == 1 else value + 1


def _load_yaml_dict(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "批量消融需要 PyYAML 来生成可追溯配置。请安装 pyyaml。"
        ) from exc
    with open(path, "r", encoding="utf-8-sig") as file:
        value = yaml.safe_load(file)
    if not isinstance(value, dict):
        raise ValueError(f"配置文件不是字典结构：{path}")
    return value


def _save_yaml_dict(path: Path, value: dict[str, Any]) -> None:
    import yaml  # type: ignore

    with open(path, "w", encoding="utf-8") as file:
        yaml.safe_dump(
            value,
            file,
            allow_unicode=True,
            sort_keys=False,
        )


def _load_baseline_bundle(result_dir: Path) -> dict[str, np.ndarray]:
    bundle_path = result_dir / "result_bundle.npz"
    if not bundle_path.exists():
        raise FileNotFoundError(f"未找到基线结果包：{bundle_path}")

    required = (
        "t_work",
        "obs_work",
        "r_work_final",
        "w_prior",
        "dt",
        "s_syn_stationary",
        "s_syn_after_dtw",
    )
    with np.load(bundle_path, allow_pickle=True) as bundle:
        missing = [key for key in required if key not in bundle.files]
        if missing:
            raise KeyError(f"基线结果包缺少数组：{missing}")
        return {key: np.asarray(bundle[key]) for key in bundle.files}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as file:
        value = json.load(file)
    return value if isinstance(value, dict) else {}


def _acceptance_config(cfg: Any) -> TvAcceptanceConfig:
    return TvAcceptanceConfig(
        min_gain_vs_dtw=float(cfg.acceptance.min_gain_vs_dtw),
        min_gain_vs_stationary=float(cfg.acceptance.min_gain_vs_stationary),
        max_env_drop_vs_stationary=float(
            cfg.acceptance.max_env_drop_vs_stationary
        ),
        max_lag_ms=float(cfg.acceptance.max_lag_ms),
        center_peak_median_ms=float(cfg.acceptance.center_peak_median_ms),
        center_peak_abs_p90_ms=float(
            cfg.acceptance.center_peak_abs_p90_ms
        ),
        causal_peak_extra_margin_ms=float(
            cfg.acceptance.causal_peak_extra_margin_ms
        ),
        tolerance_ms=float(cfg.acceptance.tolerance_ms),
        tolerance_cc=float(cfg.acceptance.tolerance_cc),
    )


def _method_name(loss_type: str, nu: float | None) -> str:
    if loss_type == "l2":
        return "l2"
    assert nu is not None
    if float(nu).is_integer():
        label = str(int(nu))
    else:
        label = str(nu).replace(".", "p")
    return f"student_t_nu{label}"


def _safe_scalar(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _diag_array(diag: dict[str, Any], key: str, dtype=None) -> np.ndarray:
    value = np.asarray(diag.get(key, np.array([])))
    return value.astype(dtype, copy=False) if dtype is not None else value


def _build_metrics(
    result: TvWaveletResult,
    obs_work: np.ndarray,
    elapsed_seconds: float,
) -> dict[str, Any]:
    robust_attempted = _diag_array(result.diag, "robust_attempted", bool)
    robust_used = _diag_array(result.diag, "robust_solution_used", bool)
    robust_converged = _diag_array(result.diag, "irls_converged", bool)

    return {
        "loss_type": str(result.params.get("loss_type", "unknown")),
        "student_nu": _safe_scalar(result.params.get("student_nu")),
        "CC_tv_direct": float(result.cc_direct),
        "CC_tv_raw_corr": float(corrcoef_safe(obs_work, result.s_syn)),
        "CC_tv_maxlag": float(result.cc_maxlag),
        "env_cc": float(result.env_cc),
        "best_lag_ms": float(result.best_lag_ms),
        "peak_metric_p10": float(result.peak_metric_p10),
        "peak_metric_med": float(result.peak_metric_med),
        "peak_metric_p90": float(result.peak_metric_p90),
        "peak_abs_p90": float(result.peak_abs_p90),
        "valid_ratio": float(result.valid_ratio),
        "valid_windows": int(np.sum(result.valid_mask)),
        "attempted_windows": int(np.sum(result.skip_code != 0)),
        "W_pass": bool(result.W_pass),
        "use_negative_W": bool(result.use_negative),
        "robust_attempted_windows": int(np.sum(robust_attempted)),
        "robust_solution_used_windows": int(np.sum(robust_used)),
        "robust_converged_windows": int(np.sum(robust_converged)),
        "robust_fallback_to_l2_windows": int(
            np.sum(robust_attempted) - np.sum(robust_used)
        ),
        "elapsed_seconds": float(elapsed_seconds),
    }


def _save_compact_result(
    *,
    output_dir: Path,
    generated_config: dict[str, Any],
    baseline: dict[str, np.ndarray],
    result: TvWaveletResult,
    metrics: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, np.ndarray] = {
        "t_work": np.asarray(baseline["t_work"]),
        "obs_work": np.asarray(baseline["obs_work"]),
        "r_work_final": np.asarray(baseline["r_work_final"]),
        "w_prior": np.asarray(baseline["w_prior"]),
        "s_syn_stationary": np.asarray(baseline["s_syn_stationary"]),
        "s_syn_after_dtw": np.asarray(baseline["s_syn_after_dtw"]),
        "dt": np.asarray(baseline["dt"]),
        "W_est": np.asarray(result.W_raw),
        "W_est_best": np.asarray(result.W_best),
        "s_syn_tv_direct": np.asarray(result.s_syn),
        "peak_metric_ms": np.asarray(result.peak_metric_ms),
        "tv_valid_mask": np.asarray(result.valid_mask),
        "tv_skip_code": np.asarray(result.skip_code),
    }

    diag_keys = (
        "robust_sigma",
        "irls_iterations",
        "irls_converged",
        "robust_code",
        "robust_attempted",
        "robust_solution_used",
        "weight_mean",
        "weight_min",
        "outlier_fraction",
        "effective_sample_size",
        "effective_sample_ratio",
        "student_objective_initial",
        "student_objective_final",
        "student_objective_relative_decrease",
        "robust_weight_map",
        "robust_weight_center_indices",
        "robust_window_offsets_samples",
        "robust_weight_solution_used",
        "robust_weight_post_qc_valid",
        "robust_weight_skip_code",
        "qc_energy",
        "qc_centroid_freq",
        "qc_bandwidth",
    )
    for key in diag_keys:
        arrays[f"tv_{key}"] = np.asarray(result.diag.get(key, np.array([])))

    np.savez_compressed(output_dir / "result_bundle.npz", **arrays)

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)

    with open(output_dir / "tv_params.json", "w", encoding="utf-8") as file:
        json.dump(result.params, file, ensure_ascii=False, indent=2, default=str)

    _save_yaml_dict(output_dir / "config_used.yaml", generated_config)


def _generated_method_config(
    *,
    base_config: dict[str, Any],
    method_name: str,
    output_dir: Path,
    loss_type: str,
    nu: float | None,
) -> dict[str, Any]:
    cfg = copy.deepcopy(base_config)
    cfg.setdefault("experiment", {})["name"] = method_name
    cfg.setdefault("paths", {})["result_dir"] = str(output_dir)
    tv = cfg.setdefault("tv_wavelet", {})
    tv["loss_type"] = loss_type
    tv["store_weight_map"] = loss_type == "student_t"
    if nu is not None:
        tv["student_nu"] = float(nu)
    cfg.setdefault("comparison", {})["enable"] = False
    cfg["ablation_fixed_inputs"] = True
    return cfg



def _infer_w_prior_source(
    *,
    requested_source: str,
    baseline: dict[str, np.ndarray],
    alignment: str,
) -> str:
    requested_source = str(requested_source).strip()
    if requested_source.lower() != "auto":
        return requested_source

    w_prior = np.asarray(baseline["w_prior"], dtype=float).ravel()
    dt = float(np.asarray(baseline["dt"]).reshape(()))
    f_dom_array = np.asarray(baseline.get("f_dom", np.array([])))

    if alignment == "center" and f_dom_array.size == 1:
        f_dom = float(f_dom_array.reshape(()))
        ricker = make_center_ricker_wavelet(
            f0=f_dom,
            dt=dt,
            length=w_prior.size,
        )
        relative_difference = float(
            np.linalg.norm(w_prior - ricker)
            / (np.linalg.norm(w_prior) + 1e-12)
        )
        if relative_difference < 1e-8:
            print(
                "[prior source auto] w_prior 与中心 Ricker 一致，"
                "使用 center_ricker_prior_after_DTW。"
            )
            return "center_ricker_prior_after_DTW"

    print(
        "[prior source auto] w_prior 不是中心 Ricker，"
        "使用 strict_stationary_center_after_DTW。"
    )
    return "strict_stationary_center_after_DTW"


def _validate_l2_against_baseline(
    *,
    baseline: dict[str, np.ndarray],
    W_l2: np.ndarray,
    skip_code_l2: np.ndarray,
    tolerance: float = 1e-8,
) -> dict[str, Any]:
    if "W_est" not in baseline or "tv_skip_code" not in baseline:
        return {
            "available": False,
            "relative_W_error": None,
            "skip_code_equal": None,
        }

    W_reference = np.asarray(baseline["W_est"], dtype=float)
    skip_reference = np.asarray(baseline["tv_skip_code"], dtype=int)
    W_l2 = np.asarray(W_l2, dtype=float)
    skip_code_l2 = np.asarray(skip_code_l2, dtype=int)

    relative_error = float(
        np.linalg.norm(W_l2 - W_reference)
        / (np.linalg.norm(W_reference) + 1e-12)
    )
    skip_equal = bool(np.array_equal(skip_code_l2, skip_reference))
    passed = relative_error < tolerance and skip_equal

    print("[fixed-input L2 regression]")
    print(f"  relative_W_error = {relative_error:.3e}")
    print(f"  skip_code_equal  = {skip_equal}")
    print(f"  passed           = {passed}")

    if not passed:
        raise RuntimeError(
            "固定输入消融的 L2 路径未复现基线。请检查 w_prior_source、"
            "mu_prior 分支、窗口长度或配置参数；禁止继续比较 Student-t。"
        )

    return {
        "available": True,
        "relative_W_error": relative_error,
        "skip_code_equal": skip_equal,
        "passed": passed,
    }


def run_ablation(
    *,
    baseline_dir: str | Path,
    base_config_path: str | Path,
    output_root: str | Path,
    nus: list[float],
    include_l2: bool = True,
    w_prior_source: str = "auto",
    skip_existing: bool = True,
    verbose: bool = True,
) -> dict[str, Path]:
    baseline_dir = Path(baseline_dir)
    base_config_path = Path(base_config_path)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    baseline = _load_baseline_bundle(baseline_dir)
    baseline_metrics = _load_json(baseline_dir / "metrics.json")
    cfg = load_config(base_config_path)
    base_config_dict = _load_yaml_dict(base_config_path)
    w_prior_source = _infer_w_prior_source(
        requested_source=w_prior_source,
        baseline=baseline,
        alignment=str(cfg.wavelet.alignment),
    )

    t_work = np.asarray(baseline["t_work"], dtype=float).ravel()
    obs_work = np.asarray(baseline["obs_work"], dtype=float).ravel()
    r_work_final = np.asarray(baseline["r_work_final"], dtype=float).ravel()
    w_prior = np.asarray(baseline["w_prior"], dtype=float).ravel()
    dt = float(np.asarray(baseline["dt"]).reshape(()))
    s_stationary = np.asarray(baseline["s_syn_stationary"], dtype=float).ravel()
    s_after_dtw = np.asarray(baseline["s_syn_after_dtw"], dtype=float).ravel()

    if not (
        t_work.size
        == obs_work.size
        == r_work_final.size
        == s_stationary.size
        == s_after_dtw.size
    ):
        raise ValueError("基线结果包的时间序列长度不一致。")

    wavelet_length_pts = int(w_prior.size)
    data_window_length = _make_odd(
        round(float(cfg.wavelet.data_window_factor) * wavelet_length_pts)
    )

    cc_stationary = float(
        baseline_metrics.get("CC_stationary", corrcoef_safe(obs_work, s_stationary))
    )
    cc_after_dtw = float(
        baseline_metrics.get("CC_after_DTW", corrcoef_safe(obs_work, s_after_dtw))
    )
    env_stationary = float(envelope_cc(obs_work, s_stationary))

    methods: list[tuple[str, float | None]] = []
    if include_l2:
        methods.append(("l2", None))
    methods.extend(("student_t", float(nu)) for nu in nus)

    result_dirs: dict[str, Path] = {}
    for loss_type, nu in methods:
        method_name = _method_name(loss_type, nu)
        output_dir = output_root / method_name
        result_dirs[method_name] = output_dir

        if skip_existing and (output_dir / "result_bundle.npz").exists():
            print(f"[skip existing] {method_name}: {output_dir}")
            continue

        print(f"\n========== Fixed-input ablation: {method_name} ==========")
        started = time.perf_counter()
        result = run_tv_wavelet_stage(
            r_time=r_work_final,
            obs_work=obs_work,
            w_prior=w_prior,
            dt=dt,
            wavelet_length_pts=wavelet_length_pts,
            data_window_length=data_window_length,
            alignment=str(cfg.wavelet.alignment),
            peak_allowed_ms=tuple(cfg.wavelet.center_peak_allowed_ms),
            causal_peak_allowed_ms=tuple(cfg.wavelet.causal_peak_allowed_ms),
            cc_stationary=cc_stationary,
            cc_after_dtw=cc_after_dtw,
            env_stationary=env_stationary,
            w_prior_source=w_prior_source,
            center_max_peak_shift_ms=float(cfg.dtw.center_max_peak_shift_ms),
            mu1=float(cfg.tv_wavelet.mu1),
            mu2=float(cfg.tv_wavelet.mu2),
            mu_dc=float(cfg.tv_wavelet.mu_dc),
            mu_prior_strict=float(cfg.tv_wavelet.mu_prior_strict),
            mu_prior_fallback=float(cfg.tv_wavelet.mu_prior_fallback),
            mu_time=float(cfg.tv_wavelet.mu_time),
            edge_penalty_enabled=bool(
                getattr(cfg.tv_wavelet, "edge_penalty_enabled", False)
            ),
            mu_edge=float(getattr(cfg.tv_wavelet, "mu_edge", 0.0)),
            edge_fraction=float(getattr(cfg.tv_wavelet, "edge_fraction", 0.12)),
            edge_taper=str(getattr(cfg.tv_wavelet, "edge_taper", "cosine")),
            energy_percentile=float(cfg.tv_wavelet.energy_percentile),
            damping_ratio=float(cfg.tv_wavelet.damping_ratio),
            svd_cutoff_ratio=float(cfg.tv_wavelet.svd_cutoff_ratio),
            estimate_step_ms=float(cfg.tv_wavelet.estimate_step_ms),
            time_smooth_ms=float(cfg.tv_wavelet.time_smooth_ms),
            wavelet_smooth_sigma=float(cfg.tv_wavelet.wavelet_smooth_sigma),
            reject_ill_conditioned=bool(cfg.tv_wavelet.reject_ill_conditioned),
            reject_amplitude_jumps=bool(cfg.tv_wavelet.reject_amplitude_jumps),
            loss_type=loss_type,
            student_nu=float(nu if nu is not None else 10.0),
            irls_max_iter=int(getattr(cfg.tv_wavelet, "irls_max_iter", 10)),
            irls_tol=float(getattr(cfg.tv_wavelet, "irls_tol", 1e-4)),
            robust_scale_mode=str(
                getattr(cfg.tv_wavelet, "robust_scale_mode", "local_mad")
            ),
            robust_scale_floor_ratio=float(
                getattr(cfg.tv_wavelet, "robust_scale_floor_ratio", 0.05)
            ),
            robust_weight_floor=float(
                getattr(cfg.tv_wavelet, "robust_weight_floor", 1e-3)
            ),
            min_effective_sample_ratio=float(
                getattr(cfg.tv_wavelet, "min_effective_sample_ratio", 0.30)
            ),
            robust_fallback=str(
                getattr(cfg.tv_wavelet, "robust_fallback", "l2")
            ),
            robust_outlier_weight_threshold=float(
                getattr(cfg.tv_wavelet, "robust_outlier_weight_threshold", 0.5)
            ),
            store_weight_map=loss_type == "student_t",
            acceptance_config=_acceptance_config(cfg),
            verbose=verbose,
        )
        elapsed = time.perf_counter() - started
        if loss_type == "l2":
            _validate_l2_against_baseline(
                baseline=baseline,
                W_l2=result.W_raw,
                skip_code_l2=result.skip_code,
            )
        metrics = _build_metrics(result, obs_work, elapsed)
        generated_config = _generated_method_config(
            base_config=base_config_dict,
            method_name=method_name,
            output_dir=output_dir,
            loss_type=loss_type,
            nu=nu,
        )
        _save_compact_result(
            output_dir=output_dir,
            generated_config=generated_config,
            baseline=baseline,
            result=result,
            metrics=metrics,
        )
        print(f"[saved] {output_dir}")
        print(f"[elapsed] {elapsed:.2f} s")

    compare_sweep_directories(
        result_dirs=list(result_dirs.values()),
        output_dir=output_root / "summary",
        strict_fairness=True,
    )
    return result_dirs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--nus", nargs="+", type=float, default=[10.0, 5.0, 3.0])
    parser.add_argument("--no-l2", action="store_true")
    parser.add_argument(
        "--w-prior-source",
        default="auto",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    run_ablation(
        baseline_dir=args.baseline_dir,
        base_config_path=args.base_config,
        output_root=args.output_root,
        nus=list(args.nus),
        include_l2=not args.no_l2,
        w_prior_source=args.w_prior_source,
        skip_existing=not args.overwrite,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
