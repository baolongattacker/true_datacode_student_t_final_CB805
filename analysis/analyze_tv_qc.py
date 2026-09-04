# -*- coding: utf-8 -*-
"""
Analyze time-varying wavelet QC diagnostics from a saved result bundle.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from io_utils.serialization import to_jsonable


def _as_float_array(value) -> np.ndarray:
    """
    将输入转换为一维浮型 NumPy 数组。、
    统一输入格式

    Args:
        value (Any): 输入数据（列表、数组或标量）。

    Returns:
        np.ndarray: 扁平化的一维浮点数组。
    """
    return np.asarray(value, dtype=float).ravel()


def _safe_percentile(values: np.ndarray, q: float) -> float:
    """
    安全地计算百分位数，忽略 NaN 值并处理空数组。
    P50 (中位数)：反映了误差的平均水平。
    P90 (第 90 分位数)：反映了误差的上限水平。
    Args:
        values (np.ndarray): 输入的数值数组。
        q (float): 百分位数（0-100）。

    Returns:
        float: 计算得到的百分位数，如果无有效数据则返回 NaN。
    """
    # 计算百分数，返回一个布尔数组，表示哪些元素是有限的（即不是 NaN 或 Inf）。
    finite = np.isfinite(values)
    # 如果没有一个是好数据，那么就直接返回 NaN，避免后续计算出错。
    if not np.any(finite):
        return float("nan")
    # values[finite] 布尔索引，是把原数组中有效的元素提取出来，q 是要计算的百分位数，np.nanpercentile 会计算这个百分位数，并且会自动忽略 NaN 值。最后把结果转换为 float 类型返回。
    return float(np.nanpercentile(values[finite], q))


def _safe_corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    """
    安全地计算两个序列的相关系数，处理 NaN 和常量序列。

    Args:
        x (np.ndarray): 序列 X。
        y (np.ndarray): 序列 Y。

    Returns:
        float: 相关系数，如果无法计算则返回 NaN。
    """
    x = _as_float_array(x)
    y = _as_float_array(y)

    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 2:
        return float("nan")

    xx = x[mask]
    yy = y[mask]
    # 忽略nan后计算标准差std
    if np.nanstd(xx) <= 1e-12 or np.nanstd(yy) <= 1e-12:
        return float("nan")

    return float(np.corrcoef(xx, yy)[0, 1])


def _stats_dict(values: np.ndarray) -> dict[str, float]:
    """
    计算数组的统计特征（P10, P50, P90, P99）。
    对一组数据的分位数进行统计，了解数据的分布特征。
    Args:
        values (np.ndarray): 输入数组。

    Returns:
        dict[str, float]: 包含统计分位数的字典。
    """
    values = _as_float_array(values)
    return {
        "p10": _safe_percentile(values, 10.0), #低值区，反应数据下限。
        "p50": _safe_percentile(values, 50.0), #中值，反应数据平均水平。
        "p90": _safe_percentile(values, 90.0), #高值区，反应数据上限。
        "p99": _safe_percentile(values, 99.0), #极高值，反应数据极值。
    }


def _masked_p90(values: np.ndarray, mask: np.ndarray) -> float:
    """
    根据掩码计算有效部分的 P90 值。
    现根据掩码筛选一部分数据，然后计算筛选出来数据的百分位。
    Args:
        values (np.ndarray): 数值数组。
        mask (np.ndarray): 布尔掩码数组。

    Returns:
        float: 掩码部分的 P90 值，如果无有效数据则返回 NaN。
    """
    values = _as_float_array(values)
    mask = np.asarray(mask, dtype=bool).ravel()

    if values.shape != mask.shape:
        raise ValueError("values and mask must have the same shape.")

    if not np.any(mask):
        return float("nan")

    return _safe_percentile(values[mask], 90.0)


def _skip_code_counts(skip_code: np.ndarray) -> dict[str, int]:
    """
    统计各跳过代码（skip code）出现的次数。

    Args:
        skip_code (np.ndarray): 跳过代码数组。

    Returns:
        dict[str, int]: 键为代码、值为频数的字典。
        具体的判定逻辑如下：
        当算法扫描每个时间窗口时，会按顺序进行以下“闯关”检查，如果没通过，就会给该点打上对应的 skip_code：

        能量关 (-1)：

        判定：计算局部反射系数能量，如果能量低于 energy_threshold（通常由 energy_percentile 参数控制），说明这里没信号，强行反演只会放大噪声。
        代码：给该位置标记 -1。
        数学稳定性关 (-2)：

        判定：对局部卷积矩阵进行 SVD 分解，检查矩阵的有效秩 (Effective Rank)。如果矩阵过于病态（不适定），反演结果会极其不稳定。
        代码：给该位置标记 -2。
        计算关 (-3)：

        判定：反演计算过程中如果出现了 NaN 或 Inf（无穷大）。
        代码：给该位置标记 -3。
        物理约束关 (-4)：

        判定：反演出的子波主峰位置是否跑得太远（例如超过了 15ms）。如果子波主峰偏离中心太远，说明它可能在错误地吸收井震之间的时差，而不是在反演子波。
        代码：给该位置标记 -4。
        连续性关 (-5)：

        判定：检查当前子波的振幅与上一个有效子波相比是否发生了“突变”（amplitude_jumps）。如果振幅突然暴涨或暴跌好几倍，通常认为是不物理的。
        代码：给该位置标记 -5。
        总结
        0：代表该点未参与尝试。
        1：代表该点闯关成功，被成功反演。
        负数 (-1 到 -5)：代表在该关卡被“拦下”了。
    """
    skip_code = np.asarray(skip_code).ravel()
    if skip_code.size == 0:
        return {} # 返回空字典
    # 去重加排序统计 skip_code 里每个编号出现了多少次，并返回字典。
    # return_counts=True 的意思是返回每个唯一值的计数，而不是返回一个布尔数组。
    unique, counts = np.unique(skip_code.astype(int), return_counts=True)
    return {str(int(code)): int(count) for code, count in zip(unique, counts)}


def _contiguous_segments(
    t_work: np.ndarray,
    peak_abs_ms: np.ndarray,
    valid_mask: np.ndarray,
    threshold_ms: float,
) -> list[dict[str, Any]]:
    """
    识别超过阈值的连续异常段，并计算段内的统计特征。
    自动识别并提出问题连片的时间段，如果连续几十毫秒都出现了巨大的误差（时移过大），
    那通常意味着该层位存在严重的井震不匹配、地质构造复杂或者处理出现了系统性偏差。

    Args:
        t_work (np.ndarray): 时间轴数组。
        peak_abs_ms (np.ndarray): 时移绝对值数组（毫秒）。
        valid_mask (np.ndarray): 有效点掩码。
        threshold_ms (float): 判定异常的阈值（毫秒）。

    Returns:
        list[dict[str, Any]]: 包含各异常段信息的列表。
    """
    # 子波主峰偏离量的绝对值
    over = peak_abs_ms > float(threshold_ms)
    if not np.any(over):
        return []
    # 返回数组中非零元素 在扁平化之后的索引
    indices = np.flatnonzero(over)
    # 找到相邻索引差值大于1的位置，这就是断点
    splits = np.where(np.diff(indices) > 1)[0] + 1
    # 根据断点索引将indieces切分成若干段
    groups = np.split(indices, splits)

    segments: list[dict[str, Any]] = []
    for group in groups:
        segment_peak = peak_abs_ms[group]
        segment_valid = valid_mask[group]
        # 向列表中添加字典，记录该段的详细特征。
        segments.append(
            {
                "start_index": int(group[0]),
                "end_index": int(group[-1]),
                "start_time_s": float(t_work[group[0]]),
                "end_time_s": float(t_work[group[-1]]),
                "n_samples": int(len(group)),
                "peak_abs_max_ms": float(np.nanmax(segment_peak)),
                "peak_abs_mean_ms": float(np.nanmean(segment_peak)),
                "valid_fraction": float(np.nanmean(segment_valid.astype(float))),
            }
        )

    segments.sort(key=lambda item: item["peak_abs_max_ms"], reverse=True)
    return segments


def _top_abnormal_windows(
    t_work: np.ndarray,
    peak_metric_ms: np.ndarray,
    valid_mask: np.ndarray,
    skip_code: np.ndarray,
    reflectivity_energy: np.ndarray,
    wavelet_energy: np.ndarray,
    centroid_frequency_hz: np.ndarray,
    top_k: int,
) -> list[dict[str, Any]]:
    """
    提取时移最大的前 K 个异常窗口的详细信息。

    Args:
        t_work (np.ndarray): 时间轴数组。
        peak_metric_ms (np.ndarray): 时移值（毫秒）。
        valid_mask (np.ndarray): 有效标记掩码。
        skip_code (np.ndarray): 跳过原因代码。
        reflectivity_energy (np.ndarray): 反射系数能量。
        wavelet_energy (np.ndarray): 子波能量。
        centroid_frequency_hz (np.ndarray): 质心频率。
        top_k (int): 提取的数量。

    Returns:
        list[dict[str, Any]]: 前 K 个异常点的信息字典列表。
    """
    peak_abs = np.abs(peak_metric_ms)
    finite = np.isfinite(peak_abs)
    if not np.any(finite):
        return []

    idx_sorted = np.argsort(peak_abs[finite])[::-1]
    finite_idx = np.flatnonzero(finite)[idx_sorted[:top_k]]

    rows = []
    for idx in finite_idx:
        rows.append(
            {
                "index": int(idx),
                "time_s": float(t_work[idx]),
                "peak_metric_ms": float(peak_metric_ms[idx]),
                "peak_abs_ms": float(peak_abs[idx]),
                "valid_mask": bool(valid_mask[idx]),
                "skip_code": int(skip_code[idx]),
                "reflectivity_energy": float(reflectivity_energy[idx]),
                "wavelet_energy": float(wavelet_energy[idx]),
                "centroid_frequency_hz": float(centroid_frequency_hz[idx]),
            }
        )

    return rows


def resolve_bundle_path(path: str | Path) -> Path:
    """
    解析并验证结果包路径，支持传入目录或直接传入 .npz 文件。

    Args:
        path (str | Path): 结果包所在目录或文件路径。

    Returns:
        Path: 验证通过的结果包文件路径。

    Raises:
        FileNotFoundError: 如果在指定位置找不到结果包文件。
    """
    path = Path(path)
    if path.is_dir():
        bundle = path / "result_bundle.npz"
    else:
        bundle = path

    if not bundle.exists():
        raise FileNotFoundError(f"Result bundle not found: {bundle}")

    return bundle


def load_result_bundle(path: str | Path) -> dict[str, np.ndarray]:
    """
    加载保存的反演结果包（.npz 文件）。

    Args:
        path (str | Path): 结果包路径。

    Returns:
        dict[str, np.ndarray]: 包含反演结果各项数据的字典。
    """
    bundle_path = resolve_bundle_path(path)
    data = np.load(bundle_path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def compute_tv_qc_summary(
    bundle: dict[str, np.ndarray],
    *,
    peak_threshold_ms: float = 15.0,
    top_k: int = 10,
) -> dict[str, Any]:
    """
    计算时变子波反演的全面 QC 摘要信息。

    Args:
        bundle (dict[str, np.ndarray]): 加载的结果包数据字典。
        peak_threshold_ms (float): 时移异常阈值（毫秒）。
        top_k (int): 记录前 K 个最大异常。

    Returns:
        dict[str, Any]: 汇总的 QC 统计及异常诊断字典。
    """
    t_work = _as_float_array(bundle["t_work"])
    peak_metric_ms = _as_float_array(bundle["peak_metric_ms"])
    peak_abs_ms = np.abs(peak_metric_ms)

    valid_mask = np.asarray(bundle.get("tv_valid_mask", np.zeros_like(t_work)), dtype=bool).ravel()
    skip_code = np.asarray(bundle.get("tv_skip_code", np.zeros_like(t_work)), dtype=int).ravel()
    reflectivity_energy = _as_float_array(bundle.get("tv_reflectivity_energy", np.full_like(t_work, np.nan)))
    wavelet_energy = _as_float_array(bundle.get("tv_wavelet_energy", np.full_like(t_work, np.nan)))
    centroid_frequency_hz = _as_float_array(bundle.get("tv_centroid_frequency_hz", np.full_like(t_work, np.nan)))

    if not (
        t_work.shape
        == peak_metric_ms.shape
        == valid_mask.shape
        == skip_code.shape
        == reflectivity_energy.shape
        == wavelet_energy.shape
        == centroid_frequency_hz.shape
    ):
        raise ValueError("QC arrays in result bundle must share the same length.")

    invalid_mask = ~valid_mask
    segments = _contiguous_segments(
        t_work=t_work,
        peak_abs_ms=peak_abs_ms,
        valid_mask=valid_mask,
        threshold_ms=peak_threshold_ms,
    )

    summary = {
        "n_time": int(len(t_work)),
        "valid_count": int(np.sum(valid_mask)),
        "invalid_count": int(np.sum(invalid_mask)),
        "valid_mask_ratio": float(np.mean(valid_mask.astype(float))),
        "skip_code_counts": _skip_code_counts(skip_code),
        "peak_threshold_ms": float(peak_threshold_ms),
        "peak_abs_p90_all": _safe_percentile(peak_abs_ms, 90.0),
        "peak_abs_p90_valid": _masked_p90(peak_abs_ms, valid_mask),
        "peak_abs_p90_invalid": _masked_p90(peak_abs_ms, invalid_mask),
        "peak_over_threshold_count": int(np.sum(peak_abs_ms > peak_threshold_ms)),
        "peak_over_threshold_segments": segments,
        "wavelet_energy_stats": _stats_dict(wavelet_energy),
        "centroid_frequency_stats": {
            "p10": _safe_percentile(centroid_frequency_hz, 10.0),
            "p50": _safe_percentile(centroid_frequency_hz, 50.0),
            "p90": _safe_percentile(centroid_frequency_hz, 90.0),
        },
        "reflectivity_energy_stats": {
            "p10": _safe_percentile(reflectivity_energy, 10.0),
            "p50": _safe_percentile(reflectivity_energy, 50.0),
            "p90": _safe_percentile(reflectivity_energy, 90.0),
        },
        "corr_reflectivity_energy_vs_abs_peak": _safe_corrcoef(
            reflectivity_energy,
            peak_abs_ms,
        ),
        "top_abnormal_windows": _top_abnormal_windows(
            t_work=t_work,
            peak_metric_ms=peak_metric_ms,
            valid_mask=valid_mask,
            skip_code=skip_code,
            reflectivity_energy=reflectivity_energy,
            wavelet_energy=wavelet_energy,
            centroid_frequency_hz=centroid_frequency_hz,
            top_k=top_k,
        ),
    }
    return summary


def _flatten_summary_for_csv(summary: dict[str, Any]) -> dict[str, Any]:
    """
    将复杂的摘要字典转换为扁平化的单行字典，以便保存为 CSV 格式。

    Args:
        summary (dict[str, Any]): 原始摘要字典。

    Returns:
        dict[str, Any]: 用于 CSV 导出的一维字典。
    """
    row = {
        "n_time": summary["n_time"],
        "valid_count": summary["valid_count"],
        "invalid_count": summary["invalid_count"],
        "valid_mask_ratio": summary["valid_mask_ratio"],
        "metrics_valid_ratio": summary.get("metrics_valid_ratio"),
        "peak_threshold_ms": summary["peak_threshold_ms"],
        "peak_abs_p90_all": summary["peak_abs_p90_all"],
        "peak_abs_p90_valid": summary["peak_abs_p90_valid"],
        "peak_abs_p90_invalid": summary["peak_abs_p90_invalid"],
        "peak_over_threshold_count": summary["peak_over_threshold_count"],
        "peak_over_threshold_segment_count": len(summary["peak_over_threshold_segments"]),
        "corr_reflectivity_energy_vs_abs_peak": summary["corr_reflectivity_energy_vs_abs_peak"],
        "skip_code_counts_json": json.dumps(summary["skip_code_counts"], ensure_ascii=False),
    }

    for prefix, stats in (
        ("wavelet_energy", summary["wavelet_energy_stats"]),
        ("centroid_frequency", summary["centroid_frequency_stats"]),
        ("reflectivity_energy", summary["reflectivity_energy_stats"]),
    ):
        for key, value in stats.items():
            row[f"{prefix}_{key}"] = value

    return row


def write_qc_summary(
    summary: dict[str, Any],
    out_dir: str | Path,
    *,
    stem: str = "qc_summary",
) -> tuple[Path, Path]:
    """
    将 QC 摘要数据保存为 JSON 和 CSV 文件。

    Args:
        summary (dict[str, Any]): 摘要数据字典。
        out_dir (str | Path): 保存文件的输出目录。
        stem (str): 文件名的基本前缀，默认为 "qc_summary"。

    Returns:
        tuple[Path, Path]: 保存好的 JSON 和 CSV 文件的完整路径。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"{stem}.json"
    csv_path = out_dir / f"{stem}.csv"

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(to_jsonable(summary), file, ensure_ascii=False, indent=2)

    row = _flatten_summary_for_csv(summary)
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    return json_path, csv_path


def main(
    result_bundle: str,
    *,
    out_dir: str | None = None,
    peak_threshold_ms: float = 15.0,
    top_k: int = 10,
) -> dict[str, Any]:
    """
    QC 分析主入口：加载数据、计算指标、关联现有度量并持久化结果。

    Args:
        result_bundle (str): 结果包路径或所在目录。
        out_dir (str, optional): 指定输出目录，默认保存在包所在目录。
        peak_threshold_ms (float): 异常检测的时移阈值。
        top_k (int): 记录前 K 个最显著的异常。

    Returns:
        dict[str, Any]: 计算得到的完整 QC 摘要。
    """
    bundle_path = resolve_bundle_path(result_bundle)
    bundle = load_result_bundle(bundle_path)
    summary = compute_tv_qc_summary(
        bundle,
        peak_threshold_ms=peak_threshold_ms,
        top_k=top_k,
    )

    metrics_path = bundle_path.parent / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as file:
            metrics = json.load(file)
        summary["metrics_valid_ratio"] = metrics.get("valid_ratio")

    target_dir = Path(out_dir) if out_dir is not None else bundle_path.parent
    json_path, csv_path = write_qc_summary(summary, target_dir)

    print(f"[QC] json: {json_path}")
    print(f"[QC] csv:  {csv_path}")
    print(f"[QC] valid_mask_ratio = {summary['valid_mask_ratio']:.6f}")
    if "metrics_valid_ratio" in summary:
        print(f"[QC] metrics_valid_ratio = {summary['metrics_valid_ratio']:.6f}")
    print(f"[QC] peak_abs_p90_all = {summary['peak_abs_p90_all']:.6f} ms")
    print(f"[QC] peak_abs_p90_valid = {summary['peak_abs_p90_valid']:.6f} ms")
    print(f"[QC] peak_abs_p90_invalid = {summary['peak_abs_p90_invalid']:.6f} ms")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("result_bundle", help="Path to result_bundle.npz or its parent directory.")
    parser.add_argument("--out-dir", default=None, help="Directory to save qc_summary.json/csv.")
    parser.add_argument("--peak-threshold-ms", type=float, default=15.0)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    main(
        args.result_bundle,
        out_dir=args.out_dir,
        peak_threshold_ms=args.peak_threshold_ms,
        top_k=args.top_k,
    )
