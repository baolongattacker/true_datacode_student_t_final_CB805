# -*- coding: utf-8 -*-
"""
io_utils/load_initial_npz.py

Load preprocessed initial model NPZ data for real-data experiments.
"""
# 延迟注解
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class RunData:
    depth: np.ndarray
    v_eff: np.ndarray
    rho_eff: np.ndarray
    r_depth_fixed: np.ndarray
    twt_init: np.ndarray

    t_work: np.ndarray
    obs_work: np.ndarray
    r_work_init: np.ndarray
    mask_work: np.ndarray | None

    dt: float
    f_dom: float

    raw: dict


def _as_finite_1d(x: np.ndarray, name: str) -> np.ndarray:
    """检查 NPZ 物理曲线为非空、有限的一维数组，shape=(N,)。"""
    array = np.asarray(x, dtype=float)

    if array.ndim != 1:
        raise ValueError(f"NPZ 字段 {name} 必须是一维数组，实际 shape={array.shape}。")
    if array.size == 0:
        raise ValueError(f"NPZ 字段 {name} 不能为空。")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"NPZ 字段 {name} 包含 NaN 或 Inf。")

    return array


def _positive_finite_scalar(value: float, name: str) -> float:
    """检查 NPZ 中的采样间隔或主频为有限正标量。"""
    scalar = np.asarray(value, dtype=float)

    if scalar.ndim != 0 or not np.isfinite(scalar) or scalar <= 0:
        raise ValueError(f"NPZ 字段 {name} 必须是有限正标量。")

    return float(scalar)


# 更健壮性的写法，为的是克服数据命名不标准
# 支持别名：_get_first_existing(npz, ["v_eff","velocity_eff","v_sonic"]) 会按优先级返回第一个存在的字段；直接 npz['v_eff'] 在文件用 velocity_eff 时会抛 KeyError。
def _get_first_existing(npz, names, required=True, default=None):
    for name in names:
        if name in npz:
            return npz[name]

    if required:
        raise KeyError(f"Missing NPZ field. Tried: {names}")

    return default


def load_run_data(input_npz: str | Path) -> RunData:
    input_npz = Path(input_npz)
    if not input_npz.exists():
        raise FileNotFoundError(f"input_npz not found: {input_npz}")

    npz = np.load(input_npz, allow_pickle=True)
    raw = {key: npz[key] for key in npz.files}

    depth = _get_first_existing(npz, ["depth"])
    v_eff = _get_first_existing(npz, ["v_eff", "velocity_eff", "v_sonic"])
    rho_eff = _get_first_existing(npz, ["rho_eff", "rho"])
    r_depth_fixed = _get_first_existing(npz, ["r_depth_fixed", "r_depth"])
    twt_init = _get_first_existing(npz, ["twt_init", "TWT_init", "twt"])

    t_work = _get_first_existing(npz, ["t_work"])
    obs_work = _get_first_existing(npz, ["obs_work", "seismic_work"])
    r_work_init = _get_first_existing(npz, ["r_work", "r_work_init", "r_time_init"])
    mask_work = _get_first_existing(npz, ["mask_work"], required=False, default=None)

    depth = _as_finite_1d(depth, "depth")
    v_eff = _as_finite_1d(v_eff, "v_eff")
    rho_eff = _as_finite_1d(rho_eff, "rho_eff")
    r_depth_fixed = _as_finite_1d(r_depth_fixed, "r_depth_fixed")
    twt_init = _as_finite_1d(twt_init, "twt_init")

    depth_length = len(depth)
    depth_field_lengths = {
        "v_eff": len(v_eff),
        "rho_eff": len(rho_eff),
        "r_depth_fixed": len(r_depth_fixed),
        "twt_init": len(twt_init),
    }
    if any(length != depth_length for length in depth_field_lengths.values()):
        raise ValueError(
            "depth、v_eff、rho_eff、r_depth_fixed 与 twt_init 必须等长；"
            f"实际 depth={depth_length}, {depth_field_lengths}。"
        )

    t_work = _as_finite_1d(t_work, "t_work")
    obs_work = _as_finite_1d(obs_work, "obs_work")
    r_work_init = _as_finite_1d(r_work_init, "r_work_init")

    if not (len(t_work) == len(obs_work) == len(r_work_init)):
        raise ValueError(
            "t_work、obs_work 与 r_work_init 必须等长；"
            f"实际为 {len(t_work)}、{len(obs_work)}、{len(r_work_init)}。"
        )

    if mask_work is not None:
        mask_work = np.asarray(mask_work)
        if mask_work.ndim != 1 or mask_work.size == 0:
            raise ValueError(
                f"NPZ 字段 mask_work 必须是非空一维数组，实际 shape={mask_work.shape}。"
            )
        if not (
            np.issubdtype(mask_work.dtype, np.bool_)
            or np.issubdtype(mask_work.dtype, np.number)
        ):
            raise ValueError("NPZ 字段 mask_work 必须是布尔或数值数组。")
        if not np.all(np.isfinite(mask_work)):
            raise ValueError("NPZ 字段 mask_work 包含 NaN 或 Inf。")

    if "dt_seismic" in npz:
        dt = _positive_finite_scalar(npz["dt_seismic"], "dt_seismic")
    elif "dt" in npz:
        dt = _positive_finite_scalar(npz["dt"], "dt")
    else:
        if len(t_work) < 2:
            raise ValueError("缺少 dt 字段时，t_work 至少需要两个采样点。")
        dt = _positive_finite_scalar(np.median(np.diff(t_work)), "dt")

    f_dom = _positive_finite_scalar(
        _get_first_existing(npz, ["f_dom", "dominant_frequency"]),
        "f_dom",
    )

    return RunData(
        depth=depth,
        v_eff=v_eff,
        rho_eff=rho_eff,
        r_depth_fixed=r_depth_fixed,
        twt_init=twt_init,
        t_work=t_work,
        obs_work=obs_work,
        r_work_init=r_work_init,
        mask_work=mask_work,
        dt=dt,
        f_dom=f_dom,
        raw=raw,
    )
