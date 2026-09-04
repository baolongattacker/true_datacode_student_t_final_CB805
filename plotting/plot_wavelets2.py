# -*- coding: utf-8 -*-
"""
plotting/plot_wavelets.py

Wavelet-only QC figures.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from core.forward_operator import wavelet_lag_axis_ms


def ensure_dir(path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _lag_label(alignment: str) -> str:
    if str(alignment).lower() == "center":
        return "Lag (ms)"
    return "Causal time after reflection (ms)"


def plot_wavelet_matrix(
    *,
    W,
    t_work,
    dt,
    alignment: str,
    result_dir,
    filename: str = "fig02_W_final_matrix.png",
    title: str = "Time-varying wavelet matrix",
):
    result_dir = ensure_dir(result_dir)

    W = np.asarray(W, dtype=float)
    t_work = np.asarray(t_work, dtype=float)

    if W.ndim != 2:
        raise ValueError("W must be a 2D array.")

    lag_ms = wavelet_lag_axis_ms(W.shape[1], dt, alignment)
    vmax = np.nanpercentile(np.abs(W), 99)
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    fig, ax = plt.subplots(figsize=(10, 6))
    image = ax.imshow(
        W,
        aspect="auto",
        origin="lower",
        extent=[lag_ms[0], lag_ms[-1], t_work[0], t_work[-1]],
        cmap="seismic",
        vmin=-vmax,
        vmax=vmax,
    )
    ax.invert_yaxis()
    ax.set_xlabel(_lag_label(alignment))
    ax.set_ylabel("TWT (s)")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label="Amplitude")
    fig.tight_layout()

    out_path = result_dir / filename
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_wavelet_slices(
    *,
    W,
    t_work,
    dt,
    alignment: str,
    result_dir,
    filename: str = "fig03_wavelet_slices.png",
    title: str = "Representative wavelets",
    time_fractions=(0.1, 0.3, 0.5, 0.7, 0.9),
):
    result_dir = ensure_dir(result_dir)

    W = np.asarray(W, dtype=float)
    t_work = np.asarray(t_work, dtype=float)

    if W.ndim != 2:
        raise ValueError("W must be a 2D array.")

    n_time, wavelet_length = W.shape
    lag_ms = wavelet_lag_axis_ms(wavelet_length, dt, alignment)

    fig, ax = plt.subplots(figsize=(10, 5))
    for frac in time_fractions:
        idx = int(round(frac * (n_time - 1)))
        idx = int(np.clip(idx, 0, n_time - 1))
        w = W[idx].copy()
        scale = np.nanmax(np.abs(w))
        if scale > 1e-12:
            w = w / scale
        ax.plot(lag_ms, w, label=f"t={t_work[idx]:.3f}s")

    ax.axhline(0.0, linewidth=0.8)
    if str(alignment).lower() == "center":
        ax.axvline(0.0, linestyle="--", linewidth=0.8)
    ax.set_xlabel(_lag_label(alignment))
    ax.set_ylabel("Normalized amplitude")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    out_path = result_dir / filename
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_peak_metric(
    *,
    t_work,
    peak_metric_ms,
    alignment: str,
    result_dir,
    filename: str = "fig04_peak_metric.png",
    center_limit_ms: float = 15.0,
    causal_peak_allowed_ms=(0.0, 40.0),
):
    result_dir = ensure_dir(result_dir)

    t_work = np.asarray(t_work, dtype=float)
    peak_metric_ms = np.asarray(peak_metric_ms, dtype=float)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_work, peak_metric_ms, linewidth=1.0)

    if str(alignment).lower() == "center":
        ax.axhline(0.0, color="k", linewidth=0.8)
        ax.axhline(center_limit_ms, linestyle="--", linewidth=0.8)
        ax.axhline(-center_limit_ms, linestyle="--", linewidth=0.8)
        ax.set_ylabel("Peak shift from center (ms)")
    else:
        lo, hi = causal_peak_allowed_ms
        ax.axhline(lo, linestyle="--", linewidth=0.8)
        ax.axhline(hi, linestyle="--", linewidth=0.8)
        ax.set_ylabel("Peak time after reflection (ms)")

    ax.set_xlabel("TWT (s)")
    ax.set_title("Wavelet peak metric")
    fig.tight_layout()

    out_path = result_dir / filename
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path



def _normalize_wavelet_for_plot(w, eps=1e-12):
    """
    将单条子波按最大绝对振幅归一化，仅用于绘图。

    物理意义：
        保留子波相位、主瓣、旁瓣和边界形态，便于不同时间子波之间做形态对比。
        这里不改变反演结果，只改变图中的显示尺度。
    """
    w = np.asarray(w, dtype=float).copy()
    scale = np.nanmax(np.abs(w))
    if np.isfinite(scale) and scale > eps:
        w = w / scale
    return w


def _nearest_time_indices(t_work, target_times_s):
    """
    根据目标时间找到最近的 t_work 样点索引。

    输入：
        t_work: 井震标定工作窗时间轴，单位 s。
        target_times_s: 希望展示的目标时间，单位 s。

    输出：
        indices: 每个目标时间对应的最近样点索引。
    """
    t_work = np.asarray(t_work, dtype=float)
    target_times_s = np.asarray(target_times_s, dtype=float)

    if t_work.ndim != 1:
        raise ValueError("t_work must be a 1D array.")

    indices = []
    for target_t in target_times_s:
        idx = int(np.argmin(np.abs(t_work - target_t)))
        indices.append(idx)

    return indices


def _default_target_times(t_work, n_times=4):
    """
    自动生成若干代表性目标时间。

    说明：
        默认不取最边缘时间点，避免边界子波受窗口截断影响太强。
    """
    t_work = np.asarray(t_work, dtype=float)
    if len(t_work) == 0:
        return []

    if n_times <= 1:
        return [float(t_work[len(t_work) // 2])]

    t_min = float(t_work[0])
    t_max = float(t_work[-1])
    margin = 0.10 * (t_max - t_min)
    if margin <= 0:
        return [t_min]

    return np.linspace(t_min + margin, t_max - margin, n_times)


def _optional_bool_vector(x, *, name: str, n_time: int):
    if x is None:
        return None
    arr = np.asarray(x, dtype=bool).ravel()
    if len(arr) != n_time:
        raise ValueError(f"{name} length must match n_time.")
    return arr


def _optional_float_vector(x, *, name: str, n_time: int):
    if x is None:
        return None
    arr = np.asarray(x, dtype=float).ravel()
    if len(arr) != n_time:
        raise ValueError(f"{name} length must match n_time.")
    return arr


def _nearest_indices_with_reliable_priority(
    t_work,
    target_times_s,
    reliable_mask=None,
    fallback_mask=None,
    fallback_alpha=None,
    fallback_alpha_keep_threshold=0.999,
):
    t_work = np.asarray(t_work, dtype=float)
    target_times_s = np.asarray(target_times_s, dtype=float)
    raw_indices = _nearest_time_indices(t_work, target_times_s)
    
    n_time = len(t_work)
    fallback_mask = _optional_bool_vector(fallback_mask, name="fallback_mask", n_time=n_time)
    fallback_alpha = _optional_float_vector(fallback_alpha, name="fallback_alpha", n_time=n_time)
    reliable_mask = _optional_bool_vector(reliable_mask, name="reliable_mask", n_time=n_time)

    if reliable_mask is None:
        return raw_indices, raw_indices

    reliable_indices = np.where(reliable_mask)[0]
    if len(reliable_indices) == 0:
        return raw_indices, raw_indices

    selected_indices = []
    for idx_raw, target_t in zip(raw_indices, target_times_s):
        hard_fallback = False
        if fallback_mask is not None:
            hard_fallback = bool(fallback_mask[idx_raw])

        effective_fallback = False
        if fallback_alpha is not None:
            alpha_raw = float(fallback_alpha[idx_raw])
            effective_fallback = (
                np.isfinite(alpha_raw)
                and alpha_raw < fallback_alpha_keep_threshold
            )

        # Keep fallback times at their original location in QC plots.
        if hard_fallback or effective_fallback:
            selected_indices.append(idx_raw)
            continue

        if reliable_mask[idx_raw]:
            selected_indices.append(idx_raw)
            continue
        # 诊断图优先挑选最近的 reliable 时间，避免把 invalid 时间误当代表性子波。
        local = int(np.argmin(np.abs(t_work[reliable_indices] - target_t)))
        selected_indices.append(int(reliable_indices[local]))
    return selected_indices, raw_indices

# 判断指定索引的子波状态是 valid、reliable、fallback 
def _selected_status_string(
    valid_mask,
    reliable_mask,
    fallback_mask,
    idx: int,
    fallback_alpha=None,
    fallback_alpha_keep_threshold=0.999,
) -> str:
    tags = []
    if valid_mask is not None and not bool(valid_mask[idx]):
        tags.append("invalid")
    if reliable_mask is not None and bool(reliable_mask[idx]):
        tags.append("reliable")
    if fallback_mask is not None and bool(fallback_mask[idx]):
        tags.append("fallback")
    elif fallback_alpha is not None:
        alpha_val = float(fallback_alpha[idx])
        if np.isfinite(alpha_val) and alpha_val < fallback_alpha_keep_threshold:
            tags.append("fallback-mixed")
    if len(tags) == 0:
        return "normal"
    return "/".join(tags)

# 在终端输出画图时间点的物理质控信息
def _print_selected_time_qc(
    *,
    figure_name: str,
    target_times_s,
    selected_indices,
    raw_indices,
    t_work,
    valid_mask=None,
    reliable_mask=None,
    fallback_mask=None,
    fallback_alpha=None,
    peak_metric_ms=None,
    wavelet_energy_norm=None,
    side_lobe_ratio=None,
    edge_energy_ratio=None,
):
    print(f"[Selected-time QC] {figure_name}")
    for target_t, idx, idx_raw in zip(target_times_s, selected_indices, raw_indices):
        valid_val = int(valid_mask[idx]) if valid_mask is not None else -1
        reliable_val = int(reliable_mask[idx]) if reliable_mask is not None else -1
        fallback_val = int(fallback_mask[idx]) if fallback_mask is not None else -1
        alpha_val = float(fallback_alpha[idx]) if fallback_alpha is not None else np.nan
        peak_val = float(peak_metric_ms[idx]) if peak_metric_ms is not None else np.nan
        energy_val = (
            float(wavelet_energy_norm[idx]) if wavelet_energy_norm is not None else np.nan
        )
        side_val = float(side_lobe_ratio[idx]) if side_lobe_ratio is not None else np.nan
        edge_val = float(edge_energy_ratio[idx]) if edge_energy_ratio is not None else np.nan
        print(
            "  "
            f"target={float(target_t):.3f}s "
            f"selected={float(t_work[idx]):.3f}s "
            f"idx={idx} raw_idx={idx_raw} "
            f"valid={valid_val} "
            f"reliable={reliable_val} "
            f"fallback={fallback_val} "
            f"alpha={alpha_val:.3f} "
            f"peak_metric_ms={peak_val:.3f} "
            f"wavelet_energy_norm={energy_val:.3f} "
            f"side_lobe_ratio={side_val:.3f} "
            f"edge_energy_ratio={edge_val:.3f}"
        )

# 绘制所选时间的时变子波的图像
def plot_wavelet_selected_time_comparison(
    *,
    W_est,
    t_work,
    dt,
    alignment: str,
    result_dir,
    W_reference=None,
    target_times_s=None,
    filename: str = "fig10_selected_time_wavelet_shape_comparison.png",
    title: str = "Selected-Time Wavelet Shape Comparison",
    est_label: str = "Final",
    reference_label: str = "TVWI",
    ncols: int = 2,
    valid_mask=None,
    reliable_mask=None,
    fallback_mask=None,
    fallback_alpha=None,
    peak_metric_ms=None,
    wavelet_energy_norm=None,
    side_lobe_ratio=None,
    edge_energy_ratio=None,
    print_selected_qc: bool = True,
):
    """
    绘制“指定时间点子波形态对比图”。

    图像样式：
        类似 2x2 子图。每个子图对应一个目标时间，横轴为子波 lag，纵轴为归一化振幅。
        若提供 W_reference，则叠加参考子波与估计/最终子波；若不提供，则只画 W_est。
    输入：
        W_est: np.ndarray, 形状 [n_time, n_wavelet]。
            要展示的时变子波矩阵，例如 W_final 或 W_est_best。
        t_work: np.ndarray, 形状 [n_time]，单位 s。
            每一条时变子波对应的双程时。
        dt: float, 单位 s。
            时间采样间隔，用于构造 lag 轴。
        alignment: str。
            子波对齐方式，center 或 causal。
        result_dir: str/Path。
            图像保存目录。
        W_reference: np.ndarray 或 None。
            参考时变子波矩阵。合成数据中可以传 W_true；真实数据中可以传 W_est_best
            作为 raw TVWI，与 W_final 做对比。
        target_times_s: list/tuple/np.ndarray 或 None。
            指定展示的目标时间，单位 s。若为 None，自动取 4 个代表性时间。
        filename: str。
            输出文件名。
        title: str。
            图像标题。
        est_label: str。
            W_est 曲线图例名称。
        reference_label: str。
            W_reference 曲线图例名称。
        ncols: int。
            子图列数。

    输出：
        out_path: Path，保存后的图像路径。

    物理意义：
        用于检查不同时刻的时变子波是否仍保持合理的主瓣、旁瓣、相位和紧凑性；
        也可用于比较 raw TVWI 与 local fallback 后 W_final 的差异。
    """
    result_dir = ensure_dir(result_dir)

    W_est = np.asarray(W_est, dtype=float)
    t_work = np.asarray(t_work, dtype=float)

    if W_est.ndim != 2:
        raise ValueError("W_est must be a 2D array.")
    if len(t_work) != W_est.shape[0]:
        raise ValueError("len(t_work) must match W_est.shape[0].")

    if W_reference is not None:
        W_reference = np.asarray(W_reference, dtype=float)
        if W_reference.shape != W_est.shape:
            raise ValueError("W_reference must have the same shape as W_est.")

    n_time = W_est.shape[0]
    valid_mask = _optional_bool_vector(valid_mask, name="valid_mask", n_time=n_time)
    reliable_mask = _optional_bool_vector(reliable_mask, name="reliable_mask", n_time=n_time)
    fallback_mask = _optional_bool_vector(fallback_mask, name="fallback_mask", n_time=n_time)
    fallback_alpha = _optional_float_vector(fallback_alpha, name="fallback_alpha", n_time=n_time)
    peak_metric_ms = _optional_float_vector(peak_metric_ms, name="peak_metric_ms", n_time=n_time)
    wavelet_energy_norm = _optional_float_vector(
        wavelet_energy_norm,
        name="wavelet_energy_norm",
        n_time=n_time,
    )
    side_lobe_ratio = _optional_float_vector(
        side_lobe_ratio,
        name="side_lobe_ratio",
        n_time=n_time,
    )
    edge_energy_ratio = _optional_float_vector(
        edge_energy_ratio,
        name="edge_energy_ratio",
        n_time=n_time,
    )

    if target_times_s is None:
        target_times_s = _default_target_times(t_work, n_times=4)

    target_indices, raw_indices = _nearest_indices_with_reliable_priority(
        t_work=t_work,
        target_times_s=target_times_s,
        reliable_mask=reliable_mask,
        fallback_mask=fallback_mask,
        fallback_alpha=fallback_alpha,
    )
    target_times_s = np.asarray(target_times_s, dtype=float)
    n_plot = len(target_indices)
    if n_plot == 0:
        return None

    ncols = max(1, int(ncols))
    nrows = int(np.ceil(n_plot / ncols))

    wavelet_length = W_est.shape[1]
    lag_ms = wavelet_lag_axis_ms(wavelet_length, dt, alignment)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.2 * ncols, 3.4 * nrows),
        squeeze=False,
        sharex=True,
    )

    for ip, idx in enumerate(target_indices):
        irow = ip // ncols
        icol = ip % ncols
        ax = axes[irow][icol]
        idx_raw = raw_indices[ip]
        target_t = float(target_times_s[ip])

        if W_reference is not None:
            w_ref = _normalize_wavelet_for_plot(W_reference[idx])
            ax.plot(lag_ms, w_ref, color="k", linewidth=1.2, label=reference_label)

        w_est = _normalize_wavelet_for_plot(W_est[idx])
        ax.plot(lag_ms, w_est, color="tab:red", linestyle="--", linewidth=1.2, label=est_label)

        ax.axhline(0.0, color="0.75", linewidth=0.8)
        if str(alignment).lower() == "center":
            ax.axvline(0.0, color="0.35", linestyle=":", linewidth=0.8)

        status_text = _selected_status_string(
            valid_mask,
            reliable_mask,
            fallback_mask,
            idx,
            fallback_alpha=fallback_alpha,
        )
        if valid_mask is not None and not bool(valid_mask[idx]):
            ax.set_facecolor("#f0f0f0")

        title_text = f" {t_work[idx]:.3f}s"
        if idx != idx_raw:
            title_text += " (nearest reliable)"
        title_text += f" [{status_text}]"
        ax.set_title(title_text)
        ax.set_ylabel("Normalized amplitude")
        ax.grid(True, alpha=0.25)

        if ip == 0:
            legend_handles = [
                Line2D([0], [0], color="k", linewidth=1.2, label=reference_label),
                Line2D([0], [0], color="tab:red", linestyle="--", linewidth=1.2, label=est_label),
                # Line2D([0], [0], color="0.55", linewidth=4.0, label="invalid"),
                # Line2D(
                #     [0],
                #     [0],
                #     marker="s",
                #     markersize=6,
                #     markerfacecolor="none",
                #     markeredgecolor="tab:orange",
                #     linestyle="None",
                #     label="fallback",
                # ),
            ]
            ax.legend(handles=legend_handles, loc="best", fontsize=8)

    # 删除多余子图，避免空白面板误导。
    for ip in range(n_plot, nrows * ncols):
        irow = ip // ncols
        icol = ip % ncols
        axes[irow][icol].axis("off")

    for ax in axes[-1, :]:
        if ax.has_data():
            ax.set_xlabel(_lag_label(alignment))

    fig.suptitle(title)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])

    out_path = result_dir / filename
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    if print_selected_qc:
        _print_selected_time_qc(
            figure_name="fig10",
            target_times_s=target_times_s,
            selected_indices=target_indices,
            raw_indices=raw_indices,
            t_work=t_work,
            valid_mask=valid_mask,
            reliable_mask=reliable_mask,
            fallback_mask=fallback_mask,
            fallback_alpha=fallback_alpha,
            peak_metric_ms=peak_metric_ms,
            wavelet_energy_norm=wavelet_energy_norm,
            side_lobe_ratio=side_lobe_ratio,
            edge_energy_ratio=edge_energy_ratio,
        )
    return out_path


def plot_time_varying_wavelet_wiggle_panel(
    *,
    W_est,
    t_work,
    dt,
    alignment: str,
    result_dir,
    W_reference=None,
    target_times_s=None,
    filename: str = "fig11_time_varying_wavelet_wiggle_panel.png",
    title: str = "Time-Varying Wavelet",
    est_label: str = "Final",
    reference_label: str = "TVWI",
    amplitude_scale: float = 0.5,
    valid_mask=None,
    reliable_mask=None,
    fallback_mask=None,
    fallback_alpha=None,
    peak_metric_ms=None,
    wavelet_energy_norm=None,
    side_lobe_ratio=None,
    edge_energy_ratio=None,
    print_selected_qc: bool = True,
):
    """
    绘制“时变子波沿时间轴展开的 wiggle-style 对比图”。

    图像样式：
        横轴是真实时间 Time (s)，每条子波以其中心时刻为横向位置展开：
            x = t_center + lag
        纵轴使用不同的垂向 offset 表示不同目标时间。
        若提供 W_reference，则叠加参考子波和估计/最终子波。

    输入：
        W_est: np.ndarray, 形状 [n_time, n_wavelet]。
            要展示的时变子波矩阵，例如 W_final 或 W_est_best。
        t_work: np.ndarray, 形状 [n_time]，单位 s。
            每一条子波对应的双程时。
        dt: float, 单位 s。
            时间采样间隔。
        alignment: str。
            子波对齐方式，center 或 causal。
        result_dir: str/Path。
            图像保存目录。
        W_reference: np.ndarray 或 None。
            可选参考子波矩阵。合成数据中可以为 W_true；真实数据中可以为 raw TVWI。
        target_times_s: list/tuple/np.ndarray 或 None。
            要展示的子波中心时刻，单位 s。若为 None，自动取 9 个代表性时间。
        filename: str。
            输出文件名。
        title: str。
            图像标题。
        est_label: str。
            W_est 曲线图例名称。
        reference_label: str。
            W_reference 曲线图例名称。
        amplitude_scale: float。
            子波振幅在纵向 offset 上的显示比例，仅影响绘图。
            默认 0.22，适合行间距为 1.0 的多条子波展示，避免振幅过小看不清。

    输出：
        out_path: Path，保存后的图像路径。

    物理意义：
        以“时间轴上移动的子波包”形式检查子波随时间演化是否平滑。
        该图比普通多曲线叠加图更容易观察局部异常、回退段和相位漂移。
    """
    result_dir = ensure_dir(result_dir)

    W_est = np.asarray(W_est, dtype=float)
    t_work = np.asarray(t_work, dtype=float)

    if W_est.ndim != 2:
        raise ValueError("W_est must be a 2D array.")
    if len(t_work) != W_est.shape[0]:
        raise ValueError("len(t_work) must match W_est.shape[0].")

    if W_reference is not None:
        W_reference = np.asarray(W_reference, dtype=float)
        if W_reference.shape != W_est.shape:
            raise ValueError("W_reference must have the same shape as W_est.")

    n_time = W_est.shape[0]
    valid_mask = _optional_bool_vector(valid_mask, name="valid_mask", n_time=n_time)
    reliable_mask = _optional_bool_vector(reliable_mask, name="reliable_mask", n_time=n_time)
    fallback_mask = _optional_bool_vector(fallback_mask, name="fallback_mask", n_time=n_time)
    fallback_alpha = _optional_float_vector(fallback_alpha, name="fallback_alpha", n_time=n_time)
    peak_metric_ms = _optional_float_vector(peak_metric_ms, name="peak_metric_ms", n_time=n_time)
    wavelet_energy_norm = _optional_float_vector(
        wavelet_energy_norm,
        name="wavelet_energy_norm",
        n_time=n_time,
    )
    side_lobe_ratio = _optional_float_vector(
        side_lobe_ratio,
        name="side_lobe_ratio",
        n_time=n_time,
    )
    edge_energy_ratio = _optional_float_vector(
        edge_energy_ratio,
        name="edge_energy_ratio",
        n_time=n_time,
    )

    if target_times_s is None:
        target_times_s = _default_target_times(t_work, n_times=9)

    target_indices, raw_indices = _nearest_indices_with_reliable_priority(
        t_work=t_work,
        target_times_s=target_times_s,
        reliable_mask=reliable_mask,
        fallback_mask=fallback_mask,
        fallback_alpha=fallback_alpha,
    )
    target_times_s = np.asarray(target_times_s, dtype=float)
    if len(target_indices) == 0:
        return None

    wavelet_length = W_est.shape[1]
    lag_ms = wavelet_lag_axis_ms(wavelet_length, dt, alignment)
    lag_s = lag_ms / 1000.0

    # 每一行的垂向位置只用于显示，不改变子波的真实时间位置。
    row_offsets = np.arange(len(target_indices), dtype=float)

    fig, ax = plt.subplots(figsize=(10, 5.5))

    for irow, idx in enumerate(target_indices):
        t_center = float(t_work[idx])
        x_axis = t_center + lag_s
        y0 = row_offsets[irow]
        idx_raw = raw_indices[irow]

        if valid_mask is not None and not bool(valid_mask[idx]):
            ax.axhspan(y0 - amplitude_scale, y0 + amplitude_scale, color="#f0f0f0", zorder=0)

        if W_reference is not None:
            w_ref = _normalize_wavelet_for_plot(W_reference[idx])
            ax.plot(
                x_axis,
                y0 + amplitude_scale * w_ref,
                color="k",
                linewidth=1.1,
                label=reference_label if irow == 0 else None,
            )

        w_est = _normalize_wavelet_for_plot(W_est[idx])
        ax.plot(
            x_axis,
            y0 + amplitude_scale * w_est,
            color="tab:blue",
            linewidth=1.1,
            label=est_label if irow == 0 else None,
        )

        # if fallback_mask is not None and bool(fallback_mask[idx]):
        #     ax.plot(
        #         [t_center],
        #         [y0],
        #         marker="s",
        #         markersize=4.8,
        #         markerfacecolor="none",
        #         markeredgewidth=1.0,
        #         markeredgecolor="tab:orange",
        #         linestyle="None",
        #     )

        # 中心时间参考线，便于检查主峰是否围绕该时刻。
        ax.plot(
            [t_center, t_center],
            [y0 - amplitude_scale, y0 + amplitude_scale],
            color="0.45",
            linestyle=":",
            linewidth=0.8,
        )

    y_labels = []
    for idx, idx_raw in zip(target_indices, raw_indices):
        label = f"{t_work[idx]:.3f}s"
        if idx != idx_raw:
            label += "*"
        status_text = _selected_status_string(
            valid_mask,
            reliable_mask,
            fallback_mask,
            idx,
            fallback_alpha=fallback_alpha,
        )
        y_labels.append(f"{label}")

    # 根据显示振幅自动留出上下边界，避免放大后子波被裁剪。
    y_margin = max(0.5, 1.5 * float(amplitude_scale))
    ax.set_ylim(row_offsets[0] - y_margin, row_offsets[-1] + y_margin)

    ax.set_yticks(row_offsets)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Wavelet center time")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    legend_handles = [
        Line2D([0], [0], color="k", linewidth=1.1, label=reference_label),
        Line2D([0], [0], color="tab:blue", linewidth=1.1, label=est_label),
        # Line2D([0], [0], color="0.55", linewidth=4.0, label="invalid"),
        # Line2D(
        #     [0],
        #     [0],
        #     marker="s",
        #     markersize=6,
        #     markerfacecolor="none",
        #     markeredgecolor="tab:orange",
        #     linestyle="None",
        #     label="fallback",
        # ),
    ]
    ax.legend(handles=legend_handles, loc="best")
    fig.tight_layout()

    out_path = result_dir / filename
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    if print_selected_qc:
        _print_selected_time_qc(
            figure_name="fig11",
            target_times_s=target_times_s,
            selected_indices=target_indices,
            raw_indices=raw_indices,
            t_work=t_work,
            valid_mask=valid_mask,
            reliable_mask=reliable_mask,
            fallback_mask=fallback_mask,
            fallback_alpha=fallback_alpha,
            peak_metric_ms=peak_metric_ms,
            wavelet_energy_norm=wavelet_energy_norm,
            side_lobe_ratio=side_lobe_ratio,
            edge_energy_ratio=edge_energy_ratio,
        )
    return out_path
def plot_single_wavelet(
    *,
    w,
    dt,
    alignment: str,
    result_dir,
    filename: str = "fig00_W_stationary_before_dtw.png",
    title: str = "Pre-DTW Stationary Wavelet",
):
    """绘制一维平稳子波形态"""
    import matplotlib.pyplot as plt
    result_dir = ensure_dir(result_dir)
    w = np.asarray(w, dtype=float)
    
    # 获取时间延迟轴 (ms)
    lag_ms = wavelet_lag_axis_ms(len(w), dt, alignment)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(lag_ms, w, color="tab:blue", linewidth=1.5, label="Wavelet")
    ax.axhline(0.0, color="0.3", linewidth=0.8, linestyle="--")
    if str(alignment).lower() == "center":
        ax.axvline(0.0, color="0.3", linewidth=0.8, linestyle="--")
        
    ax.set_xlabel(_lag_label(alignment))
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    
    out_path = result_dir / filename
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path

