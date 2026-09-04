# -*- coding: utf-8 -*-
"""Validate a completed phase-5 synthetic benchmark directory."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def check_output_directory(output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    required_files = (
        "experiment_manifest.json",
        "experiment_completion.json",
        "synthetic_benchmark_raw.csv",
        "synthetic_benchmark_summary_ci.csv",
        "synthetic_paired_comparison.csv",
        "student_t_fixed_nu_recommendation.json",
        "student_t_fixed_nu_recommendation.md",
    )
    missing = [name for name in required_files if not (output_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"第五阶段输出缺少文件：{missing}")

    manifest = json.loads(
        (output_dir / "experiment_manifest.json").read_text(encoding="utf-8")
    )
    completion = json.loads(
        (output_dir / "experiment_completion.json").read_text(encoding="utf-8")
    )
    settings = manifest["settings"]
    expected_cases = (
        len(settings["seeds"])
        * (1 + len(settings["ratios"]) * len(settings["amplitudes"]))
        * (1 + len(settings["nus"]))
    )

    with open(
        output_dir / "synthetic_benchmark_raw.csv",
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        rows = list(csv.DictReader(file))
    case_ids = [str(row.get("case_id", "")) for row in rows]
    if len(rows) != expected_cases:
        raise AssertionError(
            f"原始结果行数错误：actual={len(rows)}, expected={expected_cases}"
        )
    if len(set(case_ids)) != len(case_ids):
        raise AssertionError("synthetic_benchmark_raw.csv 中存在重复 case_id。")
    if not completion.get("complete", False):
        raise AssertionError("experiment_completion.json 未标记为 complete。")
    if int(completion.get("completed_cases", -1)) != expected_cases:
        raise AssertionError("完成案例数与参数网格不一致。")

    required_columns = {
        "weight_pair_f1",
        "weight_global_f1",
        "weight_global_false_positive_rate",
        "weight_global_coverage_ratio",
    }
    missing_columns = required_columns - set(rows[0])
    if missing_columns:
        raise AssertionError(f"原始结果缺少第五阶段列：{sorted(missing_columns)}")

    recommendation = json.loads(
        (output_dir / "student_t_fixed_nu_recommendation.json").read_text(
            encoding="utf-8"
        )
    )
    if not recommendation.get("candidates"):
        raise AssertionError("固定 nu 推荐文件中没有候选方法。")

    print(f"expected_cases       = {expected_cases}")
    print(f"actual_cases         = {len(rows)}")
    print(f"unique_case_ids      = {len(set(case_ids))}")
    print(
        "recommended_nu      = "
        f"{recommendation.get('recommended_student_nu')}"
    )
    print("Student-t phase-5 output checks PASSED")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    check_output_directory(args.output_dir)


if __name__ == "__main__":
    main()
