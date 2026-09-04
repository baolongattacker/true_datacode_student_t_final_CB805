# -*- coding: utf-8 -*-
"""Final heterogeneous-error stress-test analysis for Student-t TV inversion."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

PRIMARY_METRIC = "wavelet_nrmse_scaled"
LOWER_IS_BETTER = {
    "wavelet_nrmse_absolute",
    "wavelet_nrmse_scaled",
    "clean_trace_nrmse_true_r",
    "centroid_frequency_mae_hz",
    "weight_global_false_positive_rate",
    "elapsed_seconds",
}
HIGHER_IS_BETTER = {
    "wavelet_median_row_correlation",
    "clean_trace_cc_true_r",
    "observed_fit_cc_inversion_r",
    "valid_ratio",
    "weight_global_precision",
    "weight_global_recall",
    "weight_global_f1",
    "weight_global_coverage_ratio",
}
SUMMARY_METRICS = tuple(sorted(LOWER_IS_BETTER | HIGHER_IS_BETTER))


def _as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _finite(values: Iterable[Any]) -> np.ndarray:
    output = [_as_float(value) for value in values]
    return np.asarray([value for value in output if value is not None], dtype=float)


def _read_csv(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"未找到最终压力测试原始表：{path}")
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as file:
        for source in csv.DictReader(file):
            row: dict[str, Any] = dict(source)
            for key in (
                "seed",
                "student_nu",
                *SUMMARY_METRICS,
            ):
                if key in row:
                    row[key] = _as_float(row[key])
            rows.append(row)
    if not rows:
        raise ValueError(f"原始表为空：{path}")
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _bootstrap_mean_ci(
    values: Iterable[Any],
    *,
    reps: int,
    ci_level: float,
    rng: np.random.Generator,
) -> dict[str, Any]:
    array = _finite(values)
    if array.size == 0:
        return {"n": 0, "mean": None, "std": None, "ci_low": None, "ci_high": None}
    mean = float(np.mean(array))
    std = float(np.std(array, ddof=1)) if array.size > 1 else 0.0
    if array.size == 1 or reps <= 0:
        low = high = mean
    else:
        indices = rng.integers(0, array.size, size=(int(reps), array.size))
        means = np.mean(array[indices], axis=1)
        alpha = (1.0 - float(ci_level)) / 2.0
        low, high = np.quantile(means, [alpha, 1.0 - alpha])
        low, high = float(low), float(high)
    return {"n": int(array.size), "mean": mean, "std": std, "ci_low": low, "ci_high": high}


def _validate_rows(rows: list[dict[str, Any]]) -> None:
    seen_case: set[str] = set()
    seen_pair_method: set[tuple[int, str, str]] = set()
    for row in rows:
        case_id = str(row.get("case_id") or "").strip()
        if not case_id:
            raise ValueError("最终压力测试结果存在空 case_id。")
        if case_id in seen_case:
            raise ValueError(f"最终压力测试结果存在重复 case_id：{case_id}")
        seen_case.add(case_id)
        key = (
            int(float(row.get("seed") or 0)),
            str(row.get("scenario_key") or ""),
            str(row.get("method_key") or ""),
        )
        if key in seen_pair_method:
            raise ValueError(f"同一 seed/scenario/method 出现重复结果：{key}")
        seen_pair_method.add(key)


def _group_summary(
    rows: list[dict[str, Any]], *, reps: int, ci_level: float, random_seed: int
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["scenario_key"]), str(row["method_key"]))].append(row)
    output: list[dict[str, Any]] = []
    for number, ((scenario, method), group_rows) in enumerate(sorted(groups.items())):
        first = group_rows[0]
        result: dict[str, Any] = {
            "scenario_key": scenario,
            "scenario_group": first.get("scenario_group"),
            "method_key": method,
            "method": first.get("method"),
            "student_nu": first.get("student_nu"),
            "n_repeats": len(group_rows),
        }
        rng = np.random.default_rng(random_seed + 1013 * number)
        for metric in SUMMARY_METRICS:
            stats = _bootstrap_mean_ci(
                [row.get(metric) for row in group_rows],
                reps=reps,
                ci_level=ci_level,
                rng=rng,
            )
            for name, value in stats.items():
                result[f"{metric}_{name}"] = value
        output.append(result)
    return output


def _paired_summary(
    rows: list[dict[str, Any]], *, reps: int, ci_level: float, random_seed: int
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    by_pair: dict[tuple[int, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_pair[(int(float(row["seed"])), str(row["scenario_key"]))][
            str(row["method_key"])
        ] = row

    raw_by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (seed, scenario), methods in by_pair.items():
        l2 = methods.get("l2")
        if l2 is None:
            continue
        for method_key, candidate in methods.items():
            if method_key == "l2":
                continue
            for metric in SUMMARY_METRICS:
                l2_value = _as_float(l2.get(metric))
                st_value = _as_float(candidate.get(metric))
                if l2_value is None or st_value is None:
                    continue
                improvement = (
                    l2_value - st_value
                    if metric in LOWER_IS_BETTER
                    else st_value - l2_value
                )
                raw_by_method[method_key].append(
                    {
                        "seed": seed,
                        "scenario_key": scenario,
                        "scenario_group": candidate.get("scenario_group"),
                        "method_key": method_key,
                        "student_nu": candidate.get("student_nu"),
                        "metric": metric,
                        "l2_value": l2_value,
                        "student_t_value": st_value,
                        "improvement": improvement,
                    }
                )

    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for method_rows in raw_by_method.values():
        for row in method_rows:
            groups[(row["scenario_key"], row["method_key"], row["metric"])].append(row)

    output: list[dict[str, Any]] = []
    for number, ((scenario, method, metric), group_rows) in enumerate(sorted(groups.items())):
        values = _finite(row["improvement"] for row in group_rows)
        stats = _bootstrap_mean_ci(
            values,
            reps=reps,
            ci_level=ci_level,
            rng=np.random.default_rng(random_seed + 200003 + number * 1019),
        )
        output.append(
            {
                "scenario_key": scenario,
                "scenario_group": group_rows[0].get("scenario_group"),
                "method_key": method,
                "student_nu": group_rows[0].get("student_nu"),
                "metric": metric,
                "positive_means_better_than_l2": True,
                "n_pairs": stats["n"],
                "mean_improvement": stats["mean"],
                "std_improvement": stats["std"],
                "ci_low": stats["ci_low"],
                "ci_high": stats["ci_high"],
                "win_rate": float(np.mean(values > 0.0)) if values.size else None,
            }
        )
    return output, raw_by_method


def _final_assessment(
    raw_by_method: dict[str, list[dict[str, Any]]],
    *,
    reps: int,
    ci_level: float,
    clean_noninferiority_margin: float,
    minimum_stress_win_rate: float,
    minimum_mean_stress_gain: float,
    maximum_worst_scenario_degradation: float,
    maximum_valid_ratio_drop: float,
    random_seed: int,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for number, (method, rows) in enumerate(sorted(raw_by_method.items())):
        primary = [row for row in rows if row["metric"] == PRIMARY_METRIC]
        clean = [row["improvement"] for row in primary if row["scenario_key"] == "clean_gaussian"]
        stress = [row["improvement"] for row in primary if row["scenario_key"] != "clean_gaussian"]
        clean_degradation = [-float(value) for value in clean]
        clean_stats = _bootstrap_mean_ci(
            clean_degradation,
            reps=reps,
            ci_level=ci_level,
            rng=np.random.default_rng(random_seed + 300007 + number),
        )
        stress_stats = _bootstrap_mean_ci(
            stress,
            reps=reps,
            ci_level=ci_level,
            rng=np.random.default_rng(random_seed + 400009 + number),
        )
        stress_array = _finite(stress)

        scenario_means: dict[str, float] = {}
        for scenario in sorted({row["scenario_key"] for row in primary if row["scenario_key"] != "clean_gaussian"}):
            values = _finite(
                row["improvement"] for row in primary if row["scenario_key"] == scenario
            )
            if values.size:
                scenario_means[scenario] = float(np.mean(values))
        worst_scenario_gain = min(scenario_means.values()) if scenario_means else None

        valid_rows = [row for row in rows if row["metric"] == "valid_ratio" and row["scenario_key"] != "clean_gaussian"]
        valid_drops = [-float(row["improvement"]) for row in valid_rows]
        max_valid_drop = max(valid_drops) if valid_drops else None

        clean_upper = _as_float(clean_stats["ci_high"])
        stress_mean = _as_float(stress_stats["mean"])
        stress_win_rate = float(np.mean(stress_array > 0.0)) if stress_array.size else None

        checks = {
            "clean_noninferiority": clean_upper is not None and clean_upper <= clean_noninferiority_margin,
            "positive_mean_stress_gain": stress_mean is not None and stress_mean >= minimum_mean_stress_gain,
            "stress_win_rate": stress_win_rate is not None and stress_win_rate >= minimum_stress_win_rate,
            "worst_scenario_guard": worst_scenario_gain is not None and worst_scenario_gain >= -maximum_worst_scenario_degradation,
            "valid_ratio_guard": max_valid_drop is not None and max_valid_drop <= maximum_valid_ratio_drop,
        }
        candidates.append(
            {
                "method_key": method,
                "student_nu": rows[0].get("student_nu") if rows else None,
                "clean_nrmse_degradation_mean": clean_stats["mean"],
                "clean_nrmse_degradation_ci_low": clean_stats["ci_low"],
                "clean_nrmse_degradation_ci_high": clean_stats["ci_high"],
                "mean_stress_nrmse_gain": stress_stats["mean"],
                "mean_stress_nrmse_gain_ci_low": stress_stats["ci_low"],
                "mean_stress_nrmse_gain_ci_high": stress_stats["ci_high"],
                "stress_win_rate": stress_win_rate,
                "worst_scenario_nrmse_gain": worst_scenario_gain,
                "maximum_valid_ratio_drop": max_valid_drop,
                "scenario_mean_gains": scenario_means,
                "checks": checks,
                "validated": all(checks.values()),
            }
        )

    validated = [row for row in candidates if row["validated"]]
    validated.sort(key=lambda row: -float(row["mean_stress_nrmse_gain"]))
    selected = validated[0] if validated else None
    return {
        "decision_rule": (
            "干净数据非劣、压力场景平均改善与胜率达标、最差场景退化受控，"
            "且有效窗口比例不得明显下降。"
        ),
        "thresholds": {
            "clean_noninferiority_margin": clean_noninferiority_margin,
            "minimum_stress_win_rate": minimum_stress_win_rate,
            "minimum_mean_stress_gain": minimum_mean_stress_gain,
            "maximum_worst_scenario_degradation": maximum_worst_scenario_degradation,
            "maximum_valid_ratio_drop": maximum_valid_ratio_drop,
            "ci_level": ci_level,
        },
        "candidates": candidates,
        "final_validation_passed": selected is not None,
        "recommended_method_key": selected["method_key"] if selected else None,
        "recommended_student_nu": selected["student_nu"] if selected else None,
    }


def _plot_scenario_gains(paired_rows: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    paths: list[Path] = []
    methods = sorted({str(row["method_key"]) for row in paired_rows})
    for method in methods:
        rows = [
            row
            for row in paired_rows
            if row["method_key"] == method and row["metric"] == PRIMARY_METRIC
        ]
        if not rows:
            continue
        rows.sort(key=lambda row: (row["scenario_key"] != "clean_gaussian", row["scenario_key"]))
        labels = [str(row["scenario_key"]) for row in rows]
        means = np.asarray([float(row["mean_improvement"]) for row in rows])
        lows = np.asarray([float(row["ci_low"]) for row in rows])
        highs = np.asarray([float(row["ci_high"]) for row in rows])
        y = np.arange(len(rows))
        fig, ax = plt.subplots(figsize=(9.2, max(5.0, 0.48 * len(rows) + 1.8)))
        ax.errorbar(
            means,
            y,
            xerr=np.vstack((means - lows, highs - means)),
            fmt="o",
            capsize=4,
        )
        ax.axvline(0.0, linewidth=1.0)
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_xlabel("L2 scaled-wavelet NRMSE - Student-t NRMSE")
        ax.set_title(f"Final heterogeneous-error stress test: {method}")
        ax.grid(True, axis="x", alpha=0.25)
        fig.tight_layout()
        path = output_dir / f"final_stress_scenario_gain_{method}.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        paths.append(path)
    return paths


def _write_markdown(path: Path, assessment: dict[str, Any], paired_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Student-t 最终异构误差压力测试",
        "",
        assessment["decision_rule"],
        "",
        f"**最终通过：** {'是' if assessment['final_validation_passed'] else '否'}",
        f"**推荐方法：** {assessment.get('recommended_method_key') or '暂无'}",
        "",
        "## 候选方法",
        "",
        "| 方法 | 干净退化上界 | 压力场景平均改善 | 压力胜率 | 最差场景改善 | 最大有效率下降 | 通过 |",
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in assessment.get("candidates", []):
        def fmt(value: Any, percent: bool = False) -> str:
            number = _as_float(value)
            if number is None:
                return "—"
            return f"{100.0 * number:.2f}%" if percent else f"{number:.5f}"
        lines.append(
            "| {method} | {clean} | {gain} | {win} | {worst} | {valid} | {passed} |".format(
                method=row["method_key"],
                clean=fmt(row["clean_nrmse_degradation_ci_high"]),
                gain=fmt(row["mean_stress_nrmse_gain"]),
                win=fmt(row["stress_win_rate"], percent=True),
                worst=fmt(row["worst_scenario_nrmse_gain"]),
                valid=fmt(row["maximum_valid_ratio_drop"], percent=True),
                passed="是" if row["validated"] else "否",
            )
        )

    lines.extend(["", "## 逐场景缩放后子波 NRMSE 配对改善", "", "| 场景 | 方法 | 平均改善 | 95%区间 | 胜率 |", "|---|---|---:|---:|---:|"])
    for row in paired_rows:
        if row["metric"] != PRIMARY_METRIC:
            continue
        lines.append(
            "| {scenario} | {method} | {mean:.5f} | [{low:.5f}, {high:.5f}] | {win:.1%} |".format(
                scenario=row["scenario_key"],
                method=row["method_key"],
                mean=float(row["mean_improvement"]),
                low=float(row["ci_low"]),
                high=float(row["ci_high"]),
                win=float(row["win_rate"]),
            )
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def analyze_final_stress_test(
    *,
    raw_csv: str | Path,
    output_dir: str | Path,
    bootstrap_reps: int = 2000,
    ci_level: float = 0.95,
    clean_noninferiority_margin: float = 0.005,
    minimum_stress_win_rate: float = 0.60,
    minimum_mean_stress_gain: float = 0.0,
    maximum_worst_scenario_degradation: float = 0.01,
    maximum_valid_ratio_drop: float = 0.05,
    random_seed: int = 20260723,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_csv(raw_csv)
    _validate_rows(rows)
    group_rows = _group_summary(
        rows, reps=bootstrap_reps, ci_level=ci_level, random_seed=random_seed
    )
    paired_rows, raw_by_method = _paired_summary(
        rows, reps=bootstrap_reps, ci_level=ci_level, random_seed=random_seed
    )
    assessment = _final_assessment(
        raw_by_method,
        reps=bootstrap_reps,
        ci_level=ci_level,
        clean_noninferiority_margin=clean_noninferiority_margin,
        minimum_stress_win_rate=minimum_stress_win_rate,
        minimum_mean_stress_gain=minimum_mean_stress_gain,
        maximum_worst_scenario_degradation=maximum_worst_scenario_degradation,
        maximum_valid_ratio_drop=maximum_valid_ratio_drop,
        random_seed=random_seed,
    )

    summary_csv = output_dir / "final_stress_summary_ci.csv"
    paired_csv = output_dir / "final_stress_paired_comparison.csv"
    assessment_json = output_dir / "final_student_t_validation.json"
    report_md = output_dir / "final_student_t_validation.md"
    _write_csv(summary_csv, group_rows)
    _write_csv(paired_csv, paired_rows)
    assessment_json.write_text(
        json.dumps(assessment, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_markdown(report_md, assessment, paired_rows)
    figures = _plot_scenario_gains(paired_rows, output_dir)

    outputs = {
        "summary_csv": summary_csv,
        "paired_csv": paired_csv,
        "assessment_json": assessment_json,
        "report_md": report_md,
    }
    outputs.update({f"scenario_figure_{i}": path for i, path in enumerate(figures)})
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--ci-level", type=float, default=0.95)
    parser.add_argument("--clean-margin", type=float, default=0.005)
    parser.add_argument("--min-stress-win-rate", type=float, default=0.60)
    parser.add_argument("--min-mean-stress-gain", type=float, default=0.0)
    parser.add_argument("--max-worst-degradation", type=float, default=0.01)
    parser.add_argument("--max-valid-ratio-drop", type=float, default=0.05)
    args = parser.parse_args()
    outputs = analyze_final_stress_test(
        raw_csv=args.raw_csv,
        output_dir=args.output_dir,
        bootstrap_reps=args.bootstrap_reps,
        ci_level=args.ci_level,
        clean_noninferiority_margin=args.clean_margin,
        minimum_stress_win_rate=args.min_stress_win_rate,
        minimum_mean_stress_gain=args.min_mean_stress_gain,
        maximum_worst_scenario_degradation=args.max_worst_degradation,
        maximum_valid_ratio_drop=args.max_valid_ratio_drop,
    )
    for name, path in outputs.items():
        print(f"[{name}] {path}")


if __name__ == "__main__":
    main()
