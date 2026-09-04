# -*- coding: utf-8 -*-
"""
analysis/compare_l2_student_t.py

Compare an L2 time-varying-wavelet experiment with a Student-t experiment.
The module can be called from run_real_experiment.py or executed directly.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as file:
        value = json.load(file)
    return value if isinstance(value, dict) else {}



def _load_yaml_dict(path: Path) -> dict[str, Any]:
    """Load a YAML config when available; return an empty dict for old results."""
    if not path.exists():
        return {}
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError:
        return {}
    with open(path, "r", encoding="utf-8-sig") as file:
        value = yaml.safe_load(file)
    return value if isinstance(value, dict) else {}


def _nested_get(mapping: dict[str, Any], dotted_key: str) -> Any:
    value: Any = mapping
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


# These fields must be identical for a fair loss-function ablation.  The
# Student-t-only fields (loss_type, nu, IRLS controls and diagnostic storage)
# are intentionally excluded.
_FAIR_CONFIG_KEYS = (
    "wavelet.alignment",
    "wavelet.length_s",
    "wavelet.data_window_factor",
    "wavelet.center_peak_allowed_ms",
    "wavelet.causal_peak_allowed_ms",
    "stationary.mu1",
    "stationary.mu2",
    "stationary.mu_dc",
    "stationary.damping_ratio",
    "stationary.svd_cutoff_ratio",
    "stationary.peak_lock",
    "tv_wavelet.mu1",
    "tv_wavelet.mu2",
    "tv_wavelet.mu_dc",
    "tv_wavelet.mu_prior_strict",
    "tv_wavelet.mu_prior_fallback",
    "tv_wavelet.mu_time",
    "tv_wavelet.energy_percentile",
    "tv_wavelet.damping_ratio",
    "tv_wavelet.svd_cutoff_ratio",
    "tv_wavelet.estimate_step_ms",
    "tv_wavelet.time_smooth_ms",
    "tv_wavelet.wavelet_smooth_sigma",
    "tv_wavelet.reject_ill_conditioned",
    "tv_wavelet.reject_amplitude_jumps",
    "acceptance",
    "q",
    "local_wavelet_fallback",
)


def _compare_config_subset(
    l2_config: dict[str, Any],
    student_config: dict[str, Any],
) -> tuple[bool | None, dict[str, dict[str, Any]]]:
    if not l2_config or not student_config:
        return None, {}

    mismatches: dict[str, dict[str, Any]] = {}
    for key in _FAIR_CONFIG_KEYS:
        l2_value = _nested_get(l2_config, key)
        student_value = _nested_get(student_config, key)
        if l2_value != student_value:
            mismatches[key] = {
                "l2": l2_value,
                "student_t": student_value,
            }
    return len(mismatches) == 0, mismatches


def _load_bundle(result_dir: str | Path) -> tuple[Path, dict[str, np.ndarray]]:
    result_dir = Path(result_dir)
    bundle_path = result_dir / "result_bundle.npz"
    if not bundle_path.exists():
        raise FileNotFoundError(f"未找到结果文件：{bundle_path}")

    with np.load(bundle_path, allow_pickle=True) as bundle:
        arrays = {key: np.asarray(bundle[key]) for key in bundle.files}
    return result_dir, arrays


def _scalar(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _array(bundle: dict[str, np.ndarray], key: str) -> np.ndarray:
    value = bundle.get(key, np.array([]))
    return np.asarray(value)


def _safe_count(array: np.ndarray, predicate=None) -> int | None:
    array = np.asarray(array)
    if array.size == 0:
        return None
    if predicate is None:
        return int(array.size)
    return int(np.sum(predicate(array)))


def _safe_nanmedian(array: np.ndarray, mask: np.ndarray | None = None) -> float | None:
    array = np.asarray(array, dtype=float).ravel()
    if array.size == 0:
        return None
    if mask is not None:
        mask = np.asarray(mask, dtype=bool).ravel()
        if mask.size == array.size:
            array = array[mask]
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return None
    return float(np.median(finite))


def _safe_nanmean(array: np.ndarray, mask: np.ndarray | None = None) -> float | None:
    array = np.asarray(array, dtype=float).ravel()
    if array.size == 0:
        return None
    if mask is not None:
        mask = np.asarray(mask, dtype=bool).ravel()
        if mask.size == array.size:
            array = array[mask]
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return None
    return float(np.mean(finite))


def _relative_difference(a: np.ndarray, b: np.ndarray) -> float | None:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0 or a.shape != b.shape:
        return None
    return float(
        np.linalg.norm(b - a)
        / (np.linalg.norm(a) + 1e-12)
    )


def _same_array(a: np.ndarray, b: np.ndarray, *, atol: float = 1e-10) -> bool | None:
    a = np.asarray(a)
    b = np.asarray(b)
    if a.size == 0 or b.size == 0 or a.shape != b.shape:
        return None
    return bool(np.allclose(a, b, rtol=0.0, atol=atol, equal_nan=True))


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "True" if bool(value) else "False"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not np.isfinite(number):
            return ""
        if abs(number) >= 1e4 or (0 < abs(number) < 1e-4):
            return f"{number:.6e}"
        return f"{number:.6f}"
    return str(value)


def _delta(l2_value: Any, student_value: Any) -> float | None:
    l2_number = _scalar(l2_value)
    student_number = _scalar(student_value)
    if l2_number is None or student_number is None:
        return None
    return student_number - l2_number


def _build_rows(
    *,
    l2_bundle: dict[str, np.ndarray],
    student_bundle: dict[str, np.ndarray],
    l2_metrics: dict[str, Any],
    student_metrics: dict[str, Any],
) -> list[dict[str, Any]]:
    l2_valid = _array(l2_bundle, "tv_valid_mask").astype(bool, copy=False)
    st_valid = _array(student_bundle, "tv_valid_mask").astype(bool, copy=False)
    l2_skip = _array(l2_bundle, "tv_skip_code")
    st_skip = _array(student_bundle, "tv_skip_code")

    robust_attempted = _array(student_bundle, "tv_robust_attempted").astype(bool, copy=False)
    robust_used = _array(student_bundle, "tv_robust_solution_used").astype(bool, copy=False)
    robust_converged = _array(student_bundle, "tv_irls_converged").astype(bool, copy=False)

    rows: list[dict[str, Any]] = []

    def add(metric: str, l2_value: Any, student_value: Any, note: str = "") -> None:
        rows.append(
            {
                "metric": metric,
                "l2": l2_value,
                "student_t": student_value,
                "delta_student_minus_l2": _delta(l2_value, student_value),
                "note": note,
            }
        )

    add("CC_tv_direct", l2_metrics.get("CC_tv_direct"), student_metrics.get("CC_tv_direct"), "越大通常越好")
    add("CC_final", l2_metrics.get("CC_final"), student_metrics.get("CC_final"), "越大通常越好")
    add("CC_final_direct", l2_metrics.get("CC_final_direct"), student_metrics.get("CC_final_direct"), "越大通常越好")
    add("best_lag_ms", l2_metrics.get("best_lag_ms"), student_metrics.get("best_lag_ms"), "绝对值越小通常越好")
    add("peak_abs_p90_ms", l2_metrics.get("peak_abs_p90"), student_metrics.get("peak_abs_p90"), "越小通常越稳定")
    add("valid_ratio", l2_metrics.get("valid_ratio"), student_metrics.get("valid_ratio"), "直接有效窗口比例")
    add("valid_windows", _safe_count(l2_valid, lambda x: x), _safe_count(st_valid, lambda x: x))
    add("attempted_windows", _safe_count(l2_skip, lambda x: x != 0), _safe_count(st_skip, lambda x: x != 0))
    add("skip_low_energy", _safe_count(l2_skip, lambda x: x == -1), _safe_count(st_skip, lambda x: x == -1))
    add("skip_ill_conditioned", _safe_count(l2_skip, lambda x: x == -2), _safe_count(st_skip, lambda x: x == -2))
    add("skip_nan_inf", _safe_count(l2_skip, lambda x: x == -3), _safe_count(st_skip, lambda x: x == -3))
    add("skip_peak_shift", _safe_count(l2_skip, lambda x: x == -4), _safe_count(st_skip, lambda x: x == -4))
    add("skip_amplitude_jump", _safe_count(l2_skip, lambda x: x == -5), _safe_count(st_skip, lambda x: x == -5))

    add(
        "W_est_best_relative_difference",
        None,
        _relative_difference(
            _array(l2_bundle, "W_est_best"),
            _array(student_bundle, "W_est_best"),
        ),
        "相对于 L2 的 Frobenius 相对差异",
    )
    add(
        "s_syn_tv_relative_difference",
        None,
        _relative_difference(
            _array(l2_bundle, "s_syn_tv_direct"),
            _array(student_bundle, "s_syn_tv_direct"),
        ),
        "相对于 L2 的二范数相对差异",
    )

    add("robust_attempted_windows", None, _safe_count(robust_attempted, lambda x: x))
    add("robust_solution_used_windows", None, _safe_count(robust_used, lambda x: x))
    add("robust_converged_windows", None, _safe_count(robust_converged, lambda x: x))
    if robust_attempted.size:
        add(
            "robust_fallback_to_l2_windows",
            None,
            int(np.sum(robust_attempted) - np.sum(robust_used)),
        )

    add(
        "median_robust_sigma",
        None,
        _safe_nanmedian(_array(student_bundle, "tv_robust_sigma"), robust_attempted),
    )
    add(
        "median_weight_mean",
        None,
        _safe_nanmedian(_array(student_bundle, "tv_weight_mean"), robust_attempted),
        "越低表示整体降权更强",
    )
    add(
        "median_weight_min",
        None,
        _safe_nanmedian(_array(student_bundle, "tv_weight_min"), robust_attempted),
    )
    add(
        "mean_outlier_fraction",
        None,
        _safe_nanmean(_array(student_bundle, "tv_outlier_fraction"), robust_attempted),
        "默认权重小于 0.5 的样点比例",
    )
    add(
        "median_effective_sample_ratio",
        None,
        _safe_nanmedian(_array(student_bundle, "tv_effective_sample_ratio"), robust_attempted),
    )
    add(
        "median_objective_relative_decrease",
        None,
        _safe_nanmedian(
            _array(student_bundle, "tv_student_objective_relative_decrease"),
            robust_attempted,
        ),
    )

    return rows


def compare_result_directories(
    *,
    l2_result_dir: str | Path,
    student_t_result_dir: str | Path,
    output_dir: str | Path | None = None,
    strict_fairness: bool = True,
) -> dict[str, Path]:
    """Create CSV, Markdown and JSON comparison files."""
    l2_dir, l2_bundle = _load_bundle(l2_result_dir)
    st_dir, st_bundle = _load_bundle(student_t_result_dir)

    l2_metrics = _load_json(l2_dir / "metrics.json")
    st_metrics = _load_json(st_dir / "metrics.json")
    l2_config = _load_yaml_dict(l2_dir / "config_used.yaml")
    st_config = _load_yaml_dict(st_dir / "config_used.yaml")
    same_config, config_mismatches = _compare_config_subset(
        l2_config,
        st_config,
    )

    if output_dir is None:
        output_dir = st_dir / "l2_student_t_comparison"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fairness_checks = {
        "same_t_work": _same_array(_array(l2_bundle, "t_work"), _array(st_bundle, "t_work")),
        "same_obs_work": _same_array(_array(l2_bundle, "obs_work"), _array(st_bundle, "obs_work")),
        "same_r_work_final": _same_array(_array(l2_bundle, "r_work_final"), _array(st_bundle, "r_work_final")),
        "same_w_prior": _same_array(_array(l2_bundle, "w_prior"), _array(st_bundle, "w_prior")),
        "same_W_shape": _array(l2_bundle, "W_est_best").shape == _array(st_bundle, "W_est_best").shape,
        "same_config_except_robust_fields": same_config,
    }

    failed_checks = [
        key for key, value in fairness_checks.items() if value is False
    ]
    if strict_fairness and failed_checks:
        details = "; ".join(failed_checks)
        if config_mismatches:
            details += f"; config mismatches={config_mismatches}"
        raise ValueError(
            "L2 与 Student-t 不是严格公平的损失函数消融：" + details
        )

    rows = _build_rows(
        l2_bundle=l2_bundle,
        student_bundle=st_bundle,
        l2_metrics=l2_metrics,
        student_metrics=st_metrics,
    )

    csv_path = output_dir / "l2_vs_student_t_comparison.csv"
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "metric",
                "l2",
                "student_t",
                "delta_student_minus_l2",
                "note",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _format_value(value) for key, value in row.items()})

    md_path = output_dir / "l2_vs_student_t_comparison.md"
    with open(md_path, "w", encoding="utf-8") as file:
        file.write("# L2 与 Student-t 时变子波反演对比\n\n")
        file.write("## 公平性检查\n\n")
        for key, value in fairness_checks.items():
            file.write(f"- `{key}`: `{value}`\n")
        if config_mismatches:
            file.write("\n### 配置差异（不含 Student-t 专属字段）\n\n")
            for key, values in config_mismatches.items():
                file.write(
                    f"- `{key}`: L2=`{values['l2']}`, "
                    f"Student-t=`{values['student_t']}`\n"
                )
        file.write("\n## 指标表\n\n")
        file.write("| 指标 | L2 | Student-t | Student-t - L2 | 说明 |\n")
        file.write("|---|---:|---:|---:|---|\n")
        for row in rows:
            file.write(
                "| {metric} | {l2} | {student_t} | {delta} | {note} |\n".format(
                    metric=row["metric"],
                    l2=_format_value(row["l2"]),
                    student_t=_format_value(row["student_t"]),
                    delta=_format_value(row["delta_student_minus_l2"]),
                    note=str(row["note"]),
                )
            )

    json_path = output_dir / "l2_vs_student_t_comparison.json"
    payload = {
        "l2_result_dir": str(l2_dir.resolve()),
        "student_t_result_dir": str(st_dir.resolve()),
        "fairness_checks": fairness_checks,
        "config_mismatches": config_mismatches,
        "strict_fairness": bool(strict_fairness),
        "rows": rows,
    }
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, default=_format_value)

    print("[L2 vs Student-t comparison]")
    print(f"  CSV      : {csv_path}")
    print(f"  Markdown : {md_path}")
    print(f"  JSON     : {json_path}")
    print(f"  fairness : {fairness_checks}")

    return {
        "csv": csv_path,
        "markdown": md_path,
        "json": json_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--l2-dir", required=True, help="L2 结果目录")
    parser.add_argument("--student-dir", required=True, help="Student-t 结果目录")
    parser.add_argument("--output-dir", default=None, help="对比表输出目录")
    parser.add_argument(
        "--allow-unfair",
        action="store_true",
        help="允许关键输入或非稳健配置不一致；默认遇到不公平对比立即报错。",
    )
    args = parser.parse_args()

    compare_result_directories(
        l2_result_dir=args.l2_dir,
        student_t_result_dir=args.student_dir,
        output_dir=args.output_dir,
        strict_fairness=not args.allow_unfair,
    )


if __name__ == "__main__":
    main()
