# -*- coding: utf-8 -*-
"""
io_utils/save_results.py

Centralized result persistence for real-data experiments.

"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from io_utils.serialization import to_jsonable


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def save_metrics_json(
    metrics: dict[str, Any],
    result_dir: str | Path,
    filename: str = "metrics.json",
) -> Path:
    """Serialize metrics to JSON after converting non-JSON types."""
    result_dir = ensure_dir(result_dir)
    out_path = result_dir / filename

    with open(out_path, "w", encoding="utf-8") as file:
        json.dump(to_jsonable(metrics), file, ensure_ascii=False, indent=2)

    return out_path


def copy_config_file(
    config_path: str | Path | None,
    result_dir: str | Path,
    out_name: str = "config_used.yaml",
) -> Path | None:
    """Copy the resolved config file into the result directory."""
    if config_path is None:
        return None

    config_path = Path(config_path)
    if not config_path.exists():
        return None

    result_dir = ensure_dir(result_dir)
    out_path = result_dir / out_name
    shutil.copy2(config_path, out_path)
    return out_path


def build_metrics_dict(
    *,
    CC_initial_ricker_before_DTW: float,
    CC_stationary: float,
    CC_after_DTW: float,
    CC_prior_after_DTW: float,
    CC_tv_direct: float,
    W_pass: bool,
    q_pass: bool,
    CC_Q: float,
    Q_global: float,
    CC_final: float,
    final_model_type: str,
    best_lag_ms: float,
    peak_metric_p10: float,
    peak_metric_med: float,
    peak_metric_p90: float,
    peak_abs_p90: float,
    valid_ratio: float,
    use_negative_W: bool,
    acceptance_reasons: list[str],
    q_reason: str,
) -> dict[str, Any]:
    """Build a JSON-friendly metrics payload with normalized scalar types."""
    return {
        "CC_initial_ricker_before_DTW": float(CC_initial_ricker_before_DTW),
        "CC_stationary": float(CC_stationary),
        "CC_after_DTW": float(CC_after_DTW),
        "CC_prior_after_DTW": float(CC_prior_after_DTW),
        "CC_tv_direct": float(CC_tv_direct),
        "W_pass": bool(W_pass),
        "q_pass": bool(q_pass),
        "CC_Q": float(CC_Q) if np.isfinite(CC_Q) else None,
        "Q_global": float(Q_global) if np.isfinite(Q_global) else None,
        "CC_final": float(CC_final),
        "final_model_type": str(final_model_type),
        "best_lag_ms": float(best_lag_ms),
        "peak_metric_p10": float(peak_metric_p10),
        "peak_metric_med": float(peak_metric_med),
        "peak_metric_p90": float(peak_metric_p90),
        "peak_abs_p90": (
            float(peak_abs_p90) if np.isfinite(peak_abs_p90) else None
        ),
        "valid_ratio": float(valid_ratio) if np.isfinite(valid_ratio) else None,
        "use_negative_W": bool(use_negative_W),
        "acceptance_reasons": list(acceptance_reasons),
        "q_reason": str(q_reason),
    }


def _array_or_empty(value):
    """Convert an optional array-like value to a NumPy array."""
    if value is None:
        return np.array([])
    return np.asarray(value)


def _diag_summary(diag: dict[str, Any]) -> dict[str, Any]:
    """Reduce diagnostic arrays to lightweight shape and dtype metadata."""
    summary: dict[str, Any] = {}

    for key, value in diag.items():
        if isinstance(value, np.ndarray):
            summary[key] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
        else:
            summary[key] = value

    return summary


def save_result_bundle(
    *,
    result_dir: str | Path,
    filename: str = "result_bundle.npz",
    depth,
    t_work,
    obs_work,
    r_work_init,
    r_work_final,
    r_depth_fixed,
    twt_init,
    twt_final,
    v_final,
    w_prior,
    s_syn_stationary,
    s_syn_after_dtw,
    W_est,
    W_est_best,
    s_syn_tv_direct,
    peak_metric_ms,
    W_Q=None,
    s_syn_Q=None,
    W_final=None,
    s_syn_final=None,
    metrics: dict[str, Any] | None = None,
    tv_diag: dict[str, Any] | None = None,
    dtw_history=None,
    acceptance_reasons=None,
    q_reason: str = "",
    extra_arrays: dict[str, Any] | None = None,
) -> Path:
    """Save core arrays to NPZ and write scalar or diagnostic summaries as JSON."""
    result_dir = ensure_dir(result_dir)
    out_path = result_dir / filename

    if acceptance_reasons is None:
        acceptance_reasons = []

    arrays = {
        "depth": np.asarray(depth),
        "t_work": np.asarray(t_work),
        "obs_work": np.asarray(obs_work),
        "r_work_init": np.asarray(r_work_init),
        "r_work_final": np.asarray(r_work_final),
        "r_depth_fixed": np.asarray(r_depth_fixed),
        "twt_init": np.asarray(twt_init),
        "twt_final": np.asarray(twt_final),
        "v_final": np.asarray(v_final),
        "w_prior": np.asarray(w_prior),
        "s_syn_stationary": np.asarray(s_syn_stationary),
        "s_syn_after_dtw": np.asarray(s_syn_after_dtw),
        "W_est": np.asarray(W_est),
        "W_est_best": np.asarray(W_est_best),
        "s_syn_tv_direct": np.asarray(s_syn_tv_direct),
        "peak_metric_ms": np.asarray(peak_metric_ms),
        "W_Q": _array_or_empty(W_Q),
        "s_syn_Q": _array_or_empty(s_syn_Q),
        "W_final": _array_or_empty(W_final),
        "s_syn_final": _array_or_empty(s_syn_final),
        "acceptance_reasons": np.asarray(acceptance_reasons, dtype=object),
        "q_reason": np.asarray(q_reason, dtype=object),
    }

    if metrics is not None:
        for key, value in metrics.items():
            if isinstance(value, (str, bool, int, float, np.generic)) or value is None:
                arrays[key] = np.asarray(value, dtype=object)

    if extra_arrays:
        for key, value in extra_arrays.items():
            arrays[str(key)] = _array_or_empty(value)

    np.savez_compressed(out_path, **arrays)

    if metrics is not None:
        save_metrics_json(metrics, result_dir, filename="metrics.json")

    if tv_diag is not None:
        save_metrics_json(
            _diag_summary(tv_diag),
            result_dir,
            filename="tv_diag_summary.json",
        )

    if dtw_history is not None:
        save_metrics_json(
            {"dtw_history": dtw_history},
            result_dir,
            filename="dtw_history.json",
        )

    return out_path
