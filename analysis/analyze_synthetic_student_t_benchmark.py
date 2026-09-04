# -*- coding: utf-8 -*-
"""Statistical analysis for the synthetic Student-t robustness benchmark.

This module deliberately separates expensive inversion runs from statistical
analysis. It can be rerun from the raw CSV without repeating any inversion.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


LOWER_IS_BETTER = {
    "wavelet_nrmse_absolute",
    "wavelet_nrmse_scaled",
    "centroid_frequency_mae_hz",
    "clean_trace_nrmse",
    "weight_pair_false_positive_rate",
    "weight_global_false_positive_rate",
    "elapsed_seconds",
}

HIGHER_IS_BETTER = {
    "wavelet_median_row_correlation",
    "clean_trace_cc",
    "observed_trace_cc",
    "valid_ratio",
    "weight_pair_precision",
    "weight_pair_recall",
    "weight_pair_f1",
    "weight_global_precision",
    "weight_global_recall",
    "weight_global_f1",
}

SUMMARY_METRICS = tuple(sorted(LOWER_IS_BETTER | HIGHER_IS_BETTER))
PRIMARY_METRIC = "wavelet_nrmse_scaled"


def _as_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _method_key(method: str, student_nu: Any = None) -> str:
    method_text = str(method).strip().lower()
    if method_text == "l2" or method_text.startswith("l2"):
        return "l2"
    nu = _as_float(student_nu)
    if nu is None:
        # Backward-compatible parsing of labels such as "Student-t nu=10".
        if "nu=" in method_text:
            nu = _as_float(method_text.split("nu=", 1)[1])
    if nu is None:
        return method_text.replace(" ", "_")
    label = str(int(nu)) if float(nu).is_integer() else str(nu).replace(".", "p")
    return f"student_t_nu{label}"


def _read_raw_csv(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"未找到合成实验原始表：{path}")

    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as file:
        for source in csv.DictReader(file):
            row: dict[str, Any] = dict(source)
            for key in (
                "student_nu",
                "seed",
                "n_time",
                "snr_db",
                "noise_sigma",
                "outlier_ratio",
                "outlier_amplitude_sigma",
                "wavelet_nrmse_absolute",
                "wavelet_nrmse_scaled",
                "wavelet_median_row_correlation",
                "centroid_frequency_mae_hz",
                "clean_trace_nrmse",
                "clean_trace_cc",
                "observed_trace_cc",
                "valid_ratio",
                "weight_pair_precision",
                "weight_pair_recall",
                "weight_pair_f1",
                "weight_pair_false_positive_rate",
                "weight_global_precision",
                "weight_global_recall",
                "weight_global_f1",
                "weight_global_false_positive_rate",
                "weight_global_coverage_ratio",
                "elapsed_seconds",
            ):
                if key in row:
                    row[key] = _as_float(row[key])

            # Backward compatibility with phase-4 raw tables.
            aliases = {
                "weight_pair_precision": "weight_precision",
                "weight_pair_recall": "weight_recall",
                "weight_pair_f1": "weight_f1",
                "weight_pair_false_positive_rate": "weight_false_positive_rate",
            }
            for new_key, old_key in aliases.items():
                if row.get(new_key) is None and old_key in row:
                    row[new_key] = _as_float(row.get(old_key))

            row["method_key"] = str(
                row.get("method_key")
                or _method_key(row.get("method", ""), row.get("student_nu"))
            )
            rows.append(row)
    if not rows:
        raise ValueError(f"原始表为空：{path}")
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _finite_values(values: Iterable[Any]) -> np.ndarray:
    numeric = [_as_float(value) for value in values]
    return np.asarray([value for value in numeric if value is not None], dtype=float)


def _bootstrap_mean_ci(
    values: Iterable[Any],
    *,
    bootstrap_reps: int,
    ci_level: float,
    rng: np.random.Generator,
) -> dict[str, float | int | None]:
    array = _finite_values(values)
    if array.size == 0:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "ci_low": None,
            "ci_high": None,
        }

    mean = float(np.mean(array))
    std = float(np.std(array, ddof=1)) if array.size > 1 else 0.0
    if array.size == 1 or bootstrap_reps <= 0:
        ci_low = ci_high = mean
    else:
        indices = rng.integers(0, array.size, size=(int(bootstrap_reps), array.size))
        bootstrap_means = np.mean(array[indices], axis=1)
        alpha = (1.0 - float(ci_level)) / 2.0
        ci_low, ci_high = np.quantile(bootstrap_means, [alpha, 1.0 - alpha])
        ci_low = float(ci_low)
        ci_high = float(ci_high)

    return {
        "n": int(array.size),
        "mean": mean,
        "std": std,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def _scenario_key(row: dict[str, Any]) -> tuple[float, float]:
    return (
        float(row.get("outlier_ratio") or 0.0),
        float(row.get("outlier_amplitude_sigma") or 0.0),
    )


def _pair_key(row: dict[str, Any]) -> tuple[int, float, float, float]:
    return (
        int(float(row.get("seed") or 0)),
        float(row.get("snr_db") or 0.0),
        float(row.get("outlier_ratio") or 0.0),
        float(row.get("outlier_amplitude_sigma") or 0.0),
    )


def _validate_raw_rows(rows: list[dict[str, Any]]) -> None:
    """Reject duplicate or ambiguous paired results before statistics."""
    seen_case_ids: set[str] = set()
    seen_pairs: set[tuple[tuple[int, float, float, float], str]] = set()
    for row_number, row in enumerate(rows, start=2):
        case_id = str(row.get("case_id") or "").strip()
        if case_id:
            if case_id in seen_case_ids:
                raise ValueError(
                    f"原始表包含重复 case_id：{case_id}（约第 {row_number} 行）。"
                )
            seen_case_ids.add(case_id)

        method_key = str(row.get("method_key") or "").strip()
        pair_method = (_pair_key(row), method_key)
        if pair_method in seen_pairs:
            raise ValueError(
                "同一随机种子、噪声水平、污染场景和方法出现重复结果："
                f"pair={pair_method[0]}, method={method_key}。"
            )
        seen_pairs.add(pair_method)


def _build_group_summary(
    rows: list[dict[str, Any]],
    *,
    bootstrap_reps: int,
    ci_level: float,
    random_seed: int,
) -> list[dict[str, Any]]:
    groups: dict[tuple[float, float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        ratio, amplitude = _scenario_key(row)
        groups[(ratio, amplitude, str(row["method_key"]))].append(row)

    summary_rows: list[dict[str, Any]] = []
    for group_number, ((ratio, amplitude, method_key), group_rows) in enumerate(
        sorted(groups.items())
    ):
        first = group_rows[0]
        summary: dict[str, Any] = {
            "outlier_ratio": ratio,
            "outlier_amplitude_sigma": amplitude,
            "method": first.get("method"),
            "method_key": method_key,
            "student_nu": first.get("student_nu"),
            "n_repeats": len(group_rows),
        }
        rng = np.random.default_rng(random_seed + 1009 * group_number)
        for metric in SUMMARY_METRICS:
            statistics = _bootstrap_mean_ci(
                [row.get(metric) for row in group_rows],
                bootstrap_reps=bootstrap_reps,
                ci_level=ci_level,
                rng=rng,
            )
            for statistic_name, statistic_value in statistics.items():
                summary[f"{metric}_{statistic_name}"] = statistic_value
        summary_rows.append(summary)
    return summary_rows


def _build_paired_rows(
    rows: list[dict[str, Any]],
    *,
    bootstrap_reps: int,
    ci_level: float,
    random_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    by_pair: dict[tuple[int, float, float, float], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_pair[_pair_key(row)][str(row["method_key"])] = row

    raw_improvements_by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair_key, methods in by_pair.items():
        l2 = methods.get("l2")
        if l2 is None:
            continue
        seed, snr_db, ratio, amplitude = pair_key
        for method_key, candidate in methods.items():
            if method_key == "l2":
                continue
            for metric in SUMMARY_METRICS:
                l2_value = _as_float(l2.get(metric))
                candidate_value = _as_float(candidate.get(metric))
                if l2_value is None or candidate_value is None:
                    continue
                if metric in LOWER_IS_BETTER:
                    improvement = l2_value - candidate_value
                else:
                    improvement = candidate_value - l2_value
                relative_improvement = (
                    improvement / (abs(l2_value) + 1e-12)
                    if metric == PRIMARY_METRIC
                    else None
                )
                raw_improvements_by_method[method_key].append(
                    {
                        "seed": seed,
                        "snr_db": snr_db,
                        "outlier_ratio": ratio,
                        "outlier_amplitude_sigma": amplitude,
                        "method_key": method_key,
                        "student_nu": candidate.get("student_nu"),
                        "metric": metric,
                        "l2_value": l2_value,
                        "student_t_value": candidate_value,
                        "improvement": improvement,
                        "relative_improvement": relative_improvement,
                    }
                )

    grouped: dict[tuple[float, float, str, str], list[dict[str, Any]]] = defaultdict(list)
    for method_rows in raw_improvements_by_method.values():
        for row in method_rows:
            grouped[
                (
                    float(row["outlier_ratio"]),
                    float(row["outlier_amplitude_sigma"]),
                    str(row["method_key"]),
                    str(row["metric"]),
                )
            ].append(row)

    paired_summary: list[dict[str, Any]] = []
    for group_number, ((ratio, amplitude, method_key, metric), group_rows) in enumerate(
        sorted(grouped.items())
    ):
        rng = np.random.default_rng(random_seed + 100003 + 1013 * group_number)
        statistics = _bootstrap_mean_ci(
            [row["improvement"] for row in group_rows],
            bootstrap_reps=bootstrap_reps,
            ci_level=ci_level,
            rng=rng,
        )
        values = _finite_values(row["improvement"] for row in group_rows)
        paired_summary.append(
            {
                "outlier_ratio": ratio,
                "outlier_amplitude_sigma": amplitude,
                "method_key": method_key,
                "student_nu": group_rows[0].get("student_nu"),
                "metric": metric,
                "positive_means_better_than_l2": True,
                "n_pairs": statistics["n"],
                "mean_improvement": statistics["mean"],
                "std_improvement": statistics["std"],
                "ci_low": statistics["ci_low"],
                "ci_high": statistics["ci_high"],
                "win_rate": float(np.mean(values > 0.0)) if values.size else None,
                "tie_rate": float(np.mean(values == 0.0)) if values.size else None,
            }
        )
    return paired_summary, raw_improvements_by_method


def _candidate_rows(
    raw_rows: list[dict[str, Any]], method_key: str, contaminated: bool
) -> list[dict[str, Any]]:
    return [
        row
        for row in raw_rows
        if str(row.get("method_key")) == method_key
        and ((float(row.get("outlier_ratio") or 0.0) > 0.0) == contaminated)
    ]


def _build_recommendation(
    rows: list[dict[str, Any]],
    raw_improvements_by_method: dict[str, list[dict[str, Any]]],
    *,
    bootstrap_reps: int,
    ci_level: float,
    clean_noninferiority_margin: float,
    min_contaminated_win_rate: float,
    max_global_false_positive_rate: float,
    min_global_weight_coverage_ratio: float,
    random_seed: int,
) -> dict[str, Any]:
    candidate_keys = sorted(
        key for key in raw_improvements_by_method if key != "l2"
    )
    candidate_summaries: list[dict[str, Any]] = []

    for number, method_key in enumerate(candidate_keys):
        primary_rows = [
            row
            for row in raw_improvements_by_method[method_key]
            if row["metric"] == PRIMARY_METRIC
        ]
        clean_deltas = [
            -float(row["improvement"])
            for row in primary_rows
            if float(row["outlier_ratio"]) == 0.0
        ]
        contaminated_relative_gains = [
            float(row["relative_improvement"])
            for row in primary_rows
            if float(row["outlier_ratio"]) > 0.0
            and row["relative_improvement"] is not None
        ]

        clean_stats = _bootstrap_mean_ci(
            clean_deltas,
            bootstrap_reps=bootstrap_reps,
            ci_level=ci_level,
            rng=np.random.default_rng(random_seed + 200003 + number),
        )
        gain_stats = _bootstrap_mean_ci(
            contaminated_relative_gains,
            bootstrap_reps=bootstrap_reps,
            ci_level=ci_level,
            rng=np.random.default_rng(random_seed + 300007 + number),
        )
        gain_array = _finite_values(contaminated_relative_gains)

        contaminated_rows = _candidate_rows(rows, method_key, contaminated=True)
        global_fpr_values = _finite_values(
            row.get("weight_global_false_positive_rate")
            for row in contaminated_rows
        )
        global_f1_values = _finite_values(
            row.get("weight_global_f1") for row in contaminated_rows
        )
        global_coverage_values = _finite_values(
            row.get("weight_global_coverage_ratio") for row in contaminated_rows
        )
        nu = _as_float(contaminated_rows[0].get("student_nu")) if contaminated_rows else None

        clean_upper = _as_float(clean_stats["ci_high"])
        gain_mean = _as_float(gain_stats["mean"])
        win_rate = float(np.mean(gain_array > 0.0)) if gain_array.size else None
        global_fpr = float(np.mean(global_fpr_values)) if global_fpr_values.size else None
        global_f1 = float(np.mean(global_f1_values)) if global_f1_values.size else None
        global_coverage = (
            float(np.mean(global_coverage_values))
            if global_coverage_values.size
            else None
        )

        clean_pass = clean_upper is not None and clean_upper <= clean_noninferiority_margin
        gain_pass = gain_mean is not None and gain_mean > 0.0
        win_pass = win_rate is not None and win_rate >= min_contaminated_win_rate
        coverage_pass = (
            global_coverage is not None
            and global_coverage >= min_global_weight_coverage_ratio
        )
        # Missing FPR is not evidence of good detection; it now fails closed.
        fpr_pass = (
            global_fpr is not None
            and global_fpr <= max_global_false_positive_rate
        )
        eligible = bool(
            clean_pass and gain_pass and win_pass and coverage_pass and fpr_pass
        )

        candidate_summaries.append(
            {
                "method_key": method_key,
                "student_nu": nu,
                "clean_nrmse_degradation_mean": clean_stats["mean"],
                "clean_nrmse_degradation_ci_low": clean_stats["ci_low"],
                "clean_nrmse_degradation_ci_high": clean_stats["ci_high"],
                "contaminated_relative_nrmse_gain_mean": gain_stats["mean"],
                "contaminated_relative_nrmse_gain_ci_low": gain_stats["ci_low"],
                "contaminated_relative_nrmse_gain_ci_high": gain_stats["ci_high"],
                "contaminated_win_rate": win_rate,
                "mean_global_weight_f1": global_f1,
                "mean_global_false_positive_rate": global_fpr,
                "mean_global_weight_coverage_ratio": global_coverage,
                "clean_noninferiority_pass": clean_pass,
                "contaminated_gain_pass": gain_pass,
                "contaminated_win_rate_pass": win_pass,
                "global_weight_coverage_pass": coverage_pass,
                "global_false_positive_rate_pass": fpr_pass,
                "eligible": eligible,
            }
        )

    eligible = [row for row in candidate_summaries if row["eligible"]]
    selected: dict[str, Any] | None = None
    if eligible:
        eligible.sort(
            key=lambda row: (
                -float(row["contaminated_relative_nrmse_gain_mean"]),
                -float(row["student_nu"] if row["student_nu"] is not None else -np.inf),
            )
        )
        selected = eligible[0]

    return {
        "selection_principle": (
            "先满足干净数据非劣、污染数据胜率、全局权重覆盖率和误报率约束，"
            "再选择污染场景下相对波子 NRMSE 改善最大的固定 nu。"
        ),
        "thresholds": {
            "clean_noninferiority_margin_absolute_nrmse": clean_noninferiority_margin,
            "minimum_contaminated_win_rate": min_contaminated_win_rate,
            "maximum_global_false_positive_rate": max_global_false_positive_rate,
            "minimum_global_weight_coverage_ratio": min_global_weight_coverage_ratio,
            "ci_level": ci_level,
        },
        "candidates": candidate_summaries,
        "recommended_method_key": selected["method_key"] if selected else None,
        "recommended_student_nu": selected["student_nu"] if selected else None,
        "recommendation_available": selected is not None,
    }


def _plot_relative_gain_heatmaps(
    paired_rows: list[dict[str, Any]], output_dir: Path
) -> list[Path]:
    import matplotlib.pyplot as plt

    paths: list[Path] = []
    methods = sorted({str(row["method_key"]) for row in paired_rows})
    for method_key in methods:
        selected = [
            row
            for row in paired_rows
            if row["method_key"] == method_key
            and row["metric"] == PRIMARY_METRIC
            and float(row["outlier_ratio"]) > 0.0
        ]
        if not selected:
            continue
        ratios = sorted({float(row["outlier_ratio"]) for row in selected})
        amplitudes = sorted(
            {float(row["outlier_amplitude_sigma"]) for row in selected}
        )
        matrix = np.full((len(ratios), len(amplitudes)), np.nan)
        for row in selected:
            i = ratios.index(float(row["outlier_ratio"]))
            j = amplitudes.index(float(row["outlier_amplitude_sigma"]))
            # Improvement is absolute. Divide by an approximate L2 scale is not
            # available in this summary, so display absolute NRMSE improvement.
            matrix[i, j] = float(row["mean_improvement"])

        fig, ax = plt.subplots(figsize=(7.5, 5.2))
        image = ax.imshow(matrix, aspect="auto", origin="lower")
        ax.set_xticks(np.arange(len(amplitudes)), [f"{value:g}" for value in amplitudes])
        ax.set_yticks(
            np.arange(len(ratios)), [f"{100.0 * value:g}%" for value in ratios]
        )
        ax.set_xlabel("Outlier amplitude / noise sigma")
        ax.set_ylabel("Outlier ratio")
        ax.set_title(f"Paired scaled-wavelet NRMSE improvement: {method_key}")
        colorbar = fig.colorbar(image, ax=ax)
        colorbar.set_label("L2 NRMSE - Student-t NRMSE (positive is better)")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if np.isfinite(matrix[i, j]):
                    ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center")
        fig.tight_layout()
        path = output_dir / f"paired_nrmse_gain_{method_key}.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        paths.append(path)
    return paths


def _plot_recommendation(recommendation: dict[str, Any], output_dir: Path) -> Path | None:
    import matplotlib.pyplot as plt

    candidates = recommendation.get("candidates", [])
    if not candidates:
        return None
    candidates = sorted(
        candidates,
        key=lambda row: float(row.get("student_nu") or 0.0),
        reverse=True,
    )
    labels = [str(row["method_key"]) for row in candidates]
    means = np.asarray(
        [float(row["contaminated_relative_nrmse_gain_mean"]) for row in candidates]
    )
    low = np.asarray(
        [float(row["contaminated_relative_nrmse_gain_ci_low"]) for row in candidates]
    )
    high = np.asarray(
        [float(row["contaminated_relative_nrmse_gain_ci_high"]) for row in candidates]
    )
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    ax.errorbar(x, means * 100.0, yerr=np.vstack(((means - low), (high - means))) * 100.0, fmt="o", capsize=5)
    ax.axhline(0.0, linewidth=1.0)
    ax.set_xticks(x, labels, rotation=15)
    ax.set_ylabel("Mean relative scaled-NRMSE improvement vs L2 (%)")
    ax.set_title("Student-t fixed-nu statistical recommendation")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    path = output_dir / "student_t_fixed_nu_recommendation.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def _write_recommendation_markdown(path: Path, recommendation: dict[str, Any]) -> None:
    lines = [
        "# Student-t 固定自由度推荐",
        "",
        recommendation["selection_principle"],
        "",
        "| 方法 | 干净数据 NRMSE 退化上界 | 污染场景相对改善 | 污染胜率 | 全局权重 F1 | 全局覆盖率 | 全局误报率 | 合格 |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in recommendation.get("candidates", []):
        def fmt(value: Any, percent: bool = False) -> str:
            number = _as_float(value)
            if number is None:
                return "—"
            return f"{100.0 * number:.2f}%" if percent else f"{number:.4f}"

        lines.append(
            "| {method} | {clean} | {gain} | {win} | {f1} | {coverage} | {fpr} | {eligible} |".format(
                method=row["method_key"],
                clean=fmt(row["clean_nrmse_degradation_ci_high"]),
                gain=fmt(row["contaminated_relative_nrmse_gain_mean"], percent=True),
                win=fmt(row["contaminated_win_rate"], percent=True),
                f1=fmt(row["mean_global_weight_f1"]),
                coverage=fmt(
                    row["mean_global_weight_coverage_ratio"], percent=True
                ),
                fpr=fmt(row["mean_global_false_positive_rate"], percent=True),
                eligible="是" if row["eligible"] else "否",
            )
        )
    lines.extend(
        [
            "",
            f"**推荐方法：** {recommendation.get('recommended_method_key') or '暂无满足约束的方法'}",
            "",
            "该推荐只用于选择一个固定自由度。它不是逐窗口自适应自由度估计。",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def analyze_benchmark(
    *,
    raw_csv: str | Path,
    output_dir: str | Path,
    bootstrap_reps: int = 2000,
    ci_level: float = 0.95,
    clean_noninferiority_margin: float = 0.005,
    min_contaminated_win_rate: float = 0.60,
    max_global_false_positive_rate: float = 0.10,
    min_global_weight_coverage_ratio: float = 0.80,
    random_seed: int = 20260723,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_raw_csv(raw_csv)
    _validate_raw_rows(rows)

    summary_rows = _build_group_summary(
        rows,
        bootstrap_reps=bootstrap_reps,
        ci_level=ci_level,
        random_seed=random_seed,
    )
    paired_rows, raw_improvements = _build_paired_rows(
        rows,
        bootstrap_reps=bootstrap_reps,
        ci_level=ci_level,
        random_seed=random_seed,
    )
    recommendation = _build_recommendation(
        rows,
        raw_improvements,
        bootstrap_reps=bootstrap_reps,
        ci_level=ci_level,
        clean_noninferiority_margin=clean_noninferiority_margin,
        min_contaminated_win_rate=min_contaminated_win_rate,
        max_global_false_positive_rate=max_global_false_positive_rate,
        min_global_weight_coverage_ratio=min_global_weight_coverage_ratio,
        random_seed=random_seed,
    )

    summary_csv = output_dir / "synthetic_benchmark_summary_ci.csv"
    paired_csv = output_dir / "synthetic_paired_comparison.csv"
    recommendation_json = output_dir / "student_t_fixed_nu_recommendation.json"
    recommendation_md = output_dir / "student_t_fixed_nu_recommendation.md"
    _write_csv(summary_csv, summary_rows)
    _write_csv(paired_csv, paired_rows)
    recommendation_json.write_text(
        json.dumps(recommendation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_recommendation_markdown(recommendation_md, recommendation)

    figure_paths = _plot_relative_gain_heatmaps(paired_rows, output_dir)
    recommendation_figure = _plot_recommendation(recommendation, output_dir)

    analysis_json = output_dir / "synthetic_statistical_analysis.json"
    analysis_json.write_text(
        json.dumps(
            {
                "settings": {
                    "bootstrap_reps": bootstrap_reps,
                    "ci_level": ci_level,
                    "clean_noninferiority_margin": clean_noninferiority_margin,
                    "min_contaminated_win_rate": min_contaminated_win_rate,
                    "max_global_false_positive_rate": max_global_false_positive_rate,
                    "min_global_weight_coverage_ratio": min_global_weight_coverage_ratio,
                    "random_seed": random_seed,
                },
                "recommendation": recommendation,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    outputs = {
        "summary_csv": summary_csv,
        "paired_csv": paired_csv,
        "recommendation_json": recommendation_json,
        "recommendation_md": recommendation_md,
        "analysis_json": analysis_json,
    }
    if recommendation_figure is not None:
        outputs["recommendation_figure"] = recommendation_figure
    outputs.update({f"gain_heatmap_{index}": path for index, path in enumerate(figure_paths)})
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--ci-level", type=float, default=0.95)
    parser.add_argument("--clean-margin", type=float, default=0.005)
    parser.add_argument("--min-win-rate", type=float, default=0.60)
    parser.add_argument("--max-global-fpr", type=float, default=0.10)
    parser.add_argument("--min-global-coverage", type=float, default=0.80)
    parser.add_argument("--random-seed", type=int, default=20260723)
    args = parser.parse_args()

    outputs = analyze_benchmark(
        raw_csv=args.raw_csv,
        output_dir=args.output_dir,
        bootstrap_reps=args.bootstrap_reps,
        ci_level=args.ci_level,
        clean_noninferiority_margin=args.clean_margin,
        min_contaminated_win_rate=args.min_win_rate,
        max_global_false_positive_rate=args.max_global_fpr,
        min_global_weight_coverage_ratio=args.min_global_coverage,
        random_seed=args.random_seed,
    )
    for name, path in outputs.items():
        print(f"[{name}] {path}")


if __name__ == "__main__":
    main()
