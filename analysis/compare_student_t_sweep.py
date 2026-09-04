# -*- coding: utf-8 -*-
"""Aggregate a fixed-input L2 / Student-t nu sweep."""

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


def _load_bundle(result_dir: Path) -> dict[str, np.ndarray]:
    path = result_dir / "result_bundle.npz"
    if not path.exists():
        raise FileNotFoundError(f"未找到结果包：{path}")
    with np.load(path, allow_pickle=True) as bundle:
        return {key: np.asarray(bundle[key]) for key in bundle.files}


def _array(bundle: dict[str, np.ndarray], key: str) -> np.ndarray:
    return np.asarray(bundle.get(key, np.array([])))


def _finite_stat(
    array: np.ndarray,
    *,
    statistic: str,
    mask: np.ndarray | None = None,
) -> float | None:
    values = np.asarray(array, dtype=float).ravel()
    if mask is not None:
        mask = np.asarray(mask, dtype=bool).ravel()
        if mask.size == values.size:
            values = values[mask]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    if statistic == "mean":
        return float(np.mean(values))
    if statistic == "median":
        return float(np.median(values))
    raise ValueError(statistic)


def _relative_difference(reference: np.ndarray, candidate: np.ndarray) -> float | None:
    reference = np.asarray(reference, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    if reference.size == 0 or reference.shape != candidate.shape:
        return None
    return float(
        np.linalg.norm(candidate - reference)
        / (np.linalg.norm(reference) + 1e-12)
    )


def _normalized_roughness(W: np.ndarray, axis: int, order: int = 1) -> float | None:
    W = np.asarray(W, dtype=float)
    if W.ndim != 2 or W.size == 0 or W.shape[axis] <= order:
        return None
    diff = np.diff(W, n=order, axis=axis)
    return float(np.linalg.norm(diff) / (np.linalg.norm(W) + 1e-12))


def _same_array(a: np.ndarray, b: np.ndarray, atol: float = 1e-10) -> bool:
    a = np.asarray(a)
    b = np.asarray(b)
    return bool(
        a.shape == b.shape
        and a.size > 0
        and np.allclose(a, b, rtol=0.0, atol=atol, equal_nan=True)
    )


def _method_label(metrics: dict[str, Any], result_dir: Path) -> tuple[str, float | None]:
    loss_type = str(metrics.get("loss_type", "")).lower()
    if loss_type == "l2" or result_dir.name.lower() == "l2":
        return "L2", None
    nu_value = metrics.get("student_nu")
    try:
        nu = float(nu_value)
    except (TypeError, ValueError):
        nu = None
    return (f"Student-t nu={nu:g}" if nu is not None else result_dir.name), nu


def _build_row(
    result_dir: Path,
    bundle: dict[str, np.ndarray],
    metrics: dict[str, Any],
    l2_bundle: dict[str, np.ndarray],
) -> dict[str, Any]:
    label, nu = _method_label(metrics, result_dir)
    valid = _array(bundle, "tv_valid_mask").astype(bool, copy=False)
    skip = _array(bundle, "tv_skip_code")
    robust_attempted = _array(bundle, "tv_robust_attempted").astype(bool, copy=False)
    robust_used = _array(bundle, "tv_robust_solution_used").astype(bool, copy=False)
    robust_converged = _array(bundle, "tv_irls_converged").astype(bool, copy=False)
    W = _array(bundle, "W_est_best")

    row: dict[str, Any] = {
        "method": label,
        "loss_type": metrics.get("loss_type"),
        "student_nu": nu,
        "CC_tv_direct": metrics.get("CC_tv_direct"),
        "CC_tv_raw_corr": metrics.get("CC_tv_raw_corr"),
        "CC_tv_maxlag": metrics.get("CC_tv_maxlag"),
        "env_cc": metrics.get("env_cc"),
        "best_lag_ms": metrics.get("best_lag_ms"),
        "peak_abs_p90_ms": metrics.get("peak_abs_p90"),
        "valid_ratio": metrics.get("valid_ratio"),
        "valid_windows": int(np.sum(valid)) if valid.size else None,
        "attempted_windows": int(np.sum(skip != 0)) if skip.size else None,
        "skip_low_energy": int(np.sum(skip == -1)) if skip.size else None,
        "skip_ill_conditioned": int(np.sum(skip == -2)) if skip.size else None,
        "skip_nan_inf": int(np.sum(skip == -3)) if skip.size else None,
        "skip_peak_shift": int(np.sum(skip == -4)) if skip.size else None,
        "skip_amplitude_jump": int(np.sum(skip == -5)) if skip.size else None,
        "robust_attempted_windows": int(np.sum(robust_attempted)) if robust_attempted.size else 0,
        "robust_solution_used_windows": int(np.sum(robust_used)) if robust_used.size else 0,
        "robust_converged_windows": int(np.sum(robust_converged)) if robust_converged.size else 0,
        "robust_fallback_to_l2_windows": (
            int(np.sum(robust_attempted) - np.sum(robust_used))
            if robust_attempted.size
            else 0
        ),
        "median_robust_sigma": _finite_stat(
            _array(bundle, "tv_robust_sigma"),
            statistic="median",
            mask=robust_attempted if robust_attempted.size else None,
        ),
        "median_weight_mean": _finite_stat(
            _array(bundle, "tv_weight_mean"),
            statistic="median",
            mask=robust_attempted if robust_attempted.size else None,
        ),
        "median_weight_min": _finite_stat(
            _array(bundle, "tv_weight_min"),
            statistic="median",
            mask=robust_attempted if robust_attempted.size else None,
        ),
        "mean_outlier_fraction": _finite_stat(
            _array(bundle, "tv_outlier_fraction"),
            statistic="mean",
            mask=robust_attempted if robust_attempted.size else None,
        ),
        "median_effective_sample_ratio": _finite_stat(
            _array(bundle, "tv_effective_sample_ratio"),
            statistic="median",
            mask=robust_attempted if robust_attempted.size else None,
        ),
        "median_objective_relative_decrease": _finite_stat(
            _array(bundle, "tv_student_objective_relative_decrease"),
            statistic="median",
            mask=robust_attempted if robust_attempted.size else None,
        ),
        "W_relative_difference_vs_L2": _relative_difference(
            _array(l2_bundle, "W_est_best"), W
        ),
        "synthetic_relative_difference_vs_L2": _relative_difference(
            _array(l2_bundle, "s_syn_tv_direct"),
            _array(bundle, "s_syn_tv_direct"),
        ),
        "time_roughness": _normalized_roughness(W, axis=0, order=1),
        "wavelet_second_difference_roughness": _normalized_roughness(
            W, axis=1, order=2
        ),
        "elapsed_seconds": metrics.get("elapsed_seconds"),
        "W_pass": metrics.get("W_pass"),
        "result_dir": str(result_dir.resolve()),
    }
    return row


def _format(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "True" if bool(value) else "False"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if not np.isfinite(value):
            return ""
        if abs(value) >= 1e4 or (0 < abs(value) < 1e-4):
            return f"{value:.6e}"
        return f"{value:.6f}"
    return str(value)


def _select_recommended(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    l2 = next((row for row in rows if row["loss_type"] == "l2"), None)
    candidates = [row for row in rows if row["loss_type"] == "student_t"]
    if not candidates:
        return None

    def finite(value: Any, default: float) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return default
        return number if np.isfinite(number) else default

    if l2 is not None:
        l2_valid = finite(l2.get("valid_ratio"), 0.0)
        l2_peak = finite(l2.get("peak_abs_p90_ms"), np.inf)
        eligible = [
            row
            for row in candidates
            if finite(row.get("valid_ratio"), -np.inf) >= l2_valid - 0.05
            and finite(row.get("peak_abs_p90_ms"), np.inf) <= l2_peak + 2.0
        ]
        if eligible:
            candidates = eligible

    return max(
        candidates,
        key=lambda row: (
            finite(row.get("CC_tv_direct"), -np.inf),
            finite(row.get("median_effective_sample_ratio"), -np.inf),
            finite(row.get("student_nu"), -np.inf),
        ),
    )


def _plot_summary(rows: list[dict[str, Any]], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    labels = [str(row["method"]) for row in rows]
    x = np.arange(len(rows))

    def values(key: str) -> np.ndarray:
        result = []
        for row in rows:
            try:
                number = float(row.get(key))
            except (TypeError, ValueError):
                number = np.nan
            result.append(number)
        return np.asarray(result, dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(x, values("CC_tv_direct"), marker="o")
    axes[0, 0].set_title("TV direct similarity")
    axes[0, 0].set_ylabel("Score")

    axes[0, 1].plot(x, values("valid_ratio"), marker="o", label="valid ratio")
    axes[0, 1].plot(
        x,
        values("median_effective_sample_ratio"),
        marker="s",
        label="effective sample ratio",
    )
    axes[0, 1].set_title("Window reliability")
    axes[0, 1].set_ylim(0.0, 1.05)
    axes[0, 1].legend()

    axes[1, 0].plot(x, values("median_weight_mean"), marker="o", label="mean weight")
    axes[1, 0].plot(
        x,
        values("mean_outlier_fraction"),
        marker="s",
        label="outlier fraction",
    )
    axes[1, 0].set_title("Robust down-weighting")
    axes[1, 0].set_ylim(0.0, 1.05)
    axes[1, 0].legend()

    axes[1, 1].plot(
        x,
        values("W_relative_difference_vs_L2"),
        marker="o",
        label="W difference",
    )
    axes[1, 1].plot(
        x,
        values("time_roughness"),
        marker="s",
        label="time roughness",
    )
    axes[1, 1].set_title("Model change and smoothness")
    axes[1, 1].legend()

    for ax in axes.ravel():
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def compare_sweep_directories(
    *,
    result_dirs: list[str | Path],
    output_dir: str | Path,
    strict_fairness: bool = True,
) -> dict[str, Path]:
    directories = [Path(path) for path in result_dirs]
    loaded = [
        (directory, _load_bundle(directory), _load_json(directory / "metrics.json"))
        for directory in directories
    ]

    l2_item = next(
        (
            item
            for item in loaded
            if str(item[2].get("loss_type", "")).lower() == "l2"
        ),
        None,
    )
    if l2_item is None:
        raise ValueError("多实验汇总必须包含一个 L2 结果。")
    _, l2_bundle, _ = l2_item

    fairness_keys = ("t_work", "obs_work", "r_work_final", "w_prior")
    fairness: dict[str, dict[str, bool]] = {}
    for directory, bundle, _ in loaded:
        checks = {
            key: _same_array(_array(l2_bundle, key), _array(bundle, key))
            for key in fairness_keys
        }
        fairness[directory.name] = checks

    failures = {
        name: [key for key, value in checks.items() if not value]
        for name, checks in fairness.items()
        if not all(checks.values())
    }
    if strict_fairness and failures:
        raise ValueError(f"固定输入公平性检查失败：{failures}")

    rows = [
        _build_row(directory, bundle, metrics, l2_bundle)
        for directory, bundle, metrics in loaded
    ]
    rows.sort(
        key=lambda row: (
            0 if row["loss_type"] == "l2" else 1,
            -(float(row["student_nu"]) if row["student_nu"] is not None else 0.0),
        )
    )

    recommendation = _select_recommended(rows)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())
    csv_path = output_dir / "student_t_nu_sweep_summary.csv"
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _format(value) for key, value in row.items()})

    md_path = output_dir / "student_t_nu_sweep_summary.md"
    selected_columns = (
        "method",
        "CC_tv_direct",
        "CC_tv_raw_corr",
        "valid_ratio",
        "peak_abs_p90_ms",
        "median_weight_mean",
        "mean_outlier_fraction",
        "median_effective_sample_ratio",
        "W_relative_difference_vs_L2",
        "elapsed_seconds",
    )
    with open(md_path, "w", encoding="utf-8") as file:
        file.write("# 固定输入的 L2 / Student-t 自由度消融\n\n")
        file.write("## 公平性检查\n\n")
        for name, checks in fairness.items():
            file.write(f"- `{name}`: `{checks}`\n")
        file.write("\n## 核心指标\n\n")
        file.write("| " + " | ".join(selected_columns) + " |\n")
        file.write("|" + "|".join(["---"] * len(selected_columns)) + "|\n")
        for row in rows:
            file.write(
                "| "
                + " | ".join(_format(row.get(key)) for key in selected_columns)
                + " |\n"
            )
        if recommendation is not None:
            file.write("\n## 规则推荐\n\n")
            file.write(
                "在有效窗口比例不低于 L2 超过 5 个百分点、峰值 P90 "
                "不比 L2 恶化超过 2 ms 的候选中，按 TV 直接相似性选择："
                f"**{recommendation['method']}**。\n"
            )
            file.write(
                "该推荐只是工程筛选结果，最终论文结论仍应由合成异常实验决定。\n"
            )

    json_path = output_dir / "student_t_nu_sweep_summary.json"
    payload = {
        "fairness": fairness,
        "strict_fairness": bool(strict_fairness),
        "rows": rows,
        "recommended_method": recommendation,
    }
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, default=_format)

    figure_path = output_dir / "student_t_nu_sweep_summary.png"
    _plot_summary(rows, figure_path)

    print("[Student-t nu sweep summary]")
    print(f"  CSV      : {csv_path}")
    print(f"  Markdown : {md_path}")
    print(f"  JSON     : {json_path}")
    print(f"  Figure   : {figure_path}")

    return {
        "csv": csv_path,
        "markdown": md_path,
        "json": json_path,
        "figure": figure_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dirs", nargs="+")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--allow-unfair", action="store_true")
    args = parser.parse_args()

    compare_sweep_directories(
        result_dirs=args.result_dirs,
        output_dir=args.output_dir,
        strict_fairness=not args.allow_unfair,
    )


if __name__ == "__main__":
    main()
