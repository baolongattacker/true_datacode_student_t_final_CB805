# -*- coding: utf-8 -*-
"""
plotting/plot_qc.py

General QC figures for real-data experiments.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plotting.plot_wavelets import (
    plot_peak_metric,
    plot_wavelet_matrix,
    plot_wavelet_slices,
    plot_wavelet_selected_time_comparison,
    plot_time_varying_wavelet_wiggle_panel,
    plot_single_wavelet,
)


def ensure_dir(path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _normalize_plot_trace(x, eps=1e-12):
    x = np.asarray(x, dtype=float)
    x = x - np.nanmean(x)
    scale = np.nanpercentile(np.abs(x), 99)
    if scale <= eps:
        scale = np.nanmax(np.abs(x)) + eps
    return x / scale


def plot_obs_vs_final_wiggle(
    *,
    t_work,
    obs_work,
    s_syn_final,
    result_dir,
    filename: str = "obs_vs_final_wiggle.png",
    title: str = "Observed vs Final Wiggle Tying",
):
    """
    使用 xwigb 变面积地震道绘制函数，仅绘制观测地震道 (obs) 与最终决策合成记录 (final)
    的对比图 (fig03_obs_vs_final_wiggle.png)。

    【图像的意义】：
        采用传统的变面积地震道 (Wiggle Trace with fill) 显示格式，极度清晰地突显观测地震
        与标定最终合成记录之间的波瓣匹配细节、振幅强弱以及零交叉点。

    【参数含义】：
        t_work: np.ndarray, 实验提取窗双程时轴 (s)。
        obs_work: np.ndarray, 质控观测地震道。
        s_syn_final: np.ndarray, 最终输出的合成记录。
        result_dir: str/Path, 图像保存目录。
        filename: str, 保存文件名。
        title: str, 图像标题。
    """
    from utils.xwigb import xwigb
    result_dir = ensure_dir(result_dir)
    t_work = np.asarray(t_work, dtype=float)
    obs_work = np.asarray(obs_work, dtype=float)
    s_syn_final = np.asarray(s_syn_final, dtype=float)

    # 将观测地震道与最终合成记录各重复 4 道，形成 (nt, 8) 的多道变面积对比微剖面
    num_repeats = 4
    obs_repeats = [obs_work] * num_repeats
    final_repeats = [s_syn_final] * num_repeats
    seis_data = np.stack(obs_repeats + final_repeats, axis=1)  # shape: (nt, 8)

    # 构造 X 轴道位置坐标：
    # 前 4 道为实测道 (1.0 ~ 4.0)，中间留空一道间隔，后 4 道为最终合成道 (6.0 ~ 9.0)
    x_obs = np.arange(1.0, num_repeats + 1.0, 1.0)
    x_gap = x_obs[-1] + 2.0
    x_final = np.arange(x_gap, x_gap + num_repeats, 1.0)
    x_coords = np.concatenate([x_obs, x_final])

    fig, ax = plt.subplots(figsize=(7, 8))
    
    # 变面积填充绘制
    xwigb(
        seis_data,
        t=t_work,
        x=x_coords,
        scale=0.6,
        linewidth=1.0,
        mode="vertical",
        wiggle_fill="peak_fill",
        wigb_color="k",
        ax=ax,
    )

    # 设置更友好的 X 轴刻度标签，将标签居中放置在各自 4 道的中心
    mid_obs = float(np.mean(x_obs))
    mid_final = float(np.mean(x_final))
    ax.set_xticks([mid_obs, mid_final])
    ax.set_xticklabels(["Observed (4 traces)", "Final Synthetic (4 traces)"])
    ax.set_xlabel("")
    ax.set_ylabel("TWT (s)")
    ax.set_title(title)
    
    fig.tight_layout()

    out_path = result_dir / filename
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_pure_ricker_vs_obs_vs_tv_wiggle(
    *,
    t_work,              # 标定工作窗双程时轴 (s)
    obs_work,            # 井旁实测观测地震道 (1D array)
    r_time,              # DTW 规整后的最终时域反射系数 (1D array)
    w_ricker,            # 标准的雷克（Ricker）子波 (1D array)
    s_syn_tv_direct,     # 时变反演子波直接合成记录 (1D array)
    alignment,           # 子波对齐方式 ('center' 或 'causal')
    result_dir,          # 图像保存的目录路径
    filename: str = "pure_ricker_vs_obs_vs_tv_wiggle.png",
    title: str = "Pure Ricker vs Observed vs TV Wavelet (xwigb)",
):
    from utils.xwigb import xwigb
    from core.forward_operator import stationary_convolution
    from core.signal_utils import match_rms
    print(">>> NEW plot_pure_ricker_vs_obs_vs_tv_wiggle <<<")
    result_dir = ensure_dir(result_dir)
    t_work = np.asarray(t_work, dtype=float)
    obs_work = np.asarray(obs_work, dtype=float)
    r_time = np.asarray(r_time, dtype=float)
    w_ricker = np.asarray(w_ricker, dtype=float)
    s_syn_tv_direct = np.asarray(s_syn_tv_direct, dtype=float)

    s_syn_ricker = stationary_convolution(
        r_time=r_time,
        w=w_ricker,
        alignment=alignment
    )

    s_syn_ricker_norm = _normalize_plot_trace(match_rms(s_syn_ricker, obs_work))
    obs_norm = _normalize_plot_trace(obs_work)
    s_syn_tv_norm = _normalize_plot_trace(match_rms(s_syn_tv_direct, obs_work))

    # 组合为 2D 矩阵 (nt, ntr=9)
    # 左: 3xRicker, 中: 3xObs, 右: 3xTV
    seis_data = np.stack(
        [s_syn_ricker_norm, s_syn_ricker_norm, s_syn_ricker_norm, 
         obs_norm, obs_norm, obs_norm, 
         s_syn_tv_norm, s_syn_tv_norm, s_syn_tv_norm],
        axis=1
    )

    # ---------------------------------
    # split into three panels
    # ---------------------------------
    total_len = len(t_work)
    num_cols = 3
    # 使用 np.linspace 自动根据实际数据长度计算 3 列的均匀分割区间
    col_boundaries = np.linspace(0, total_len, num_cols + 1, dtype=int)
    
    # 控制三列分别的放大倍数
    amp_factors = [2.0, 1.5, 1.5] 
    
    global_max = np.max(np.abs(seis_data))
    scale_base = 0.6
    
    # 将图像宽度扩展至 20 以容纳三列
    fig, axes = plt.subplots(1, num_cols, figsize=(20, 10), sharex=True)

    for i in range(num_cols):
        start_idx = int(col_boundaries[i])
        end_idx = int(col_boundaries[i + 1])
        
        # 如果当前子区间为空，则关闭多余的坐标系
        if start_idx >= end_idx or start_idx >= total_len:
            axes[i].axis('off')
            continue
            
        seis_sub = seis_data[start_idx:end_idx, :]
        t_sub = t_work[start_idx:end_idx]
        
        max_sub = np.max(np.abs(seis_sub))
        amp_factor = amp_factors[i]
        scale_sub = scale_base * amp_factor * (max_sub / global_max) if global_max > 0 and max_sub > 0 else scale_base * amp_factor
        
        xwigb(
            seis_sub,
            t=t_sub,
            x=np.array([1.0, 2.0, 3.0, 5.0, 6.0, 7.0, 9.0, 10.0, 11.0]),
            scale=scale_sub,
            linewidth=1.0,
            mode="vertical",
            wiggle_fill="peak_fill",
            wigb_color="k",
            ax=axes[i],
        )
        
        # 设置图名与 Y 轴
        ax_title = f"{title}\nSamples {start_idx} - {end_idx - 1}" if i == 0 else f"Samples {start_idx} - {end_idx - 1}"
        axes[i].set_title(ax_title)
        axes[i].set_ylabel("TWT (s)")

    # -------------------------
    # x axis formatting
    # -------------------------
    for ax in axes:
        if not ax.axison: 
            continue
        ax.set_xticks([2.0, 6.0, 10.0])
        ax.set_xticklabels(["Pure Ricker", "Observed", "TV Wavelet"])

    fig.tight_layout()

    out_path = result_dir / filename
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    return out_path
  

def plot_dtw_history(
    *,
    dtw_history,
    result_dir,
    filename: str = "dtw_history.png",
):
    """
    绘制动态时间规整 (DTW) 多阶段候选迭代收敛历史图 (fig07_dtw_history.png)。

    【图像的意义】：
        监控和诊断 DTW 校正时深关系时的收敛过程。展示在不同的校正阶段（如包络对齐、波形对齐、细化对齐），
        每次参数尝试迭代后的相关系数变化。这能帮助地质研究人员评估对齐刚度与参数的收敛性。

    【所含内容】：
        - 折线：展现各迭代步骤对齐后的相关系数 (CC) 变化。
        - 散点标记 'o'：表示该步迭代所作的时深调整通过了局部 CC / 包络质控阈值，被算法“接受 (Accepted)”。
        - 散点标记 'x'：表示该步迭代因指标变差或漂移超标而被“拒绝 (Rejected)”，时深关系回退到上一次有效状态。

    【参数含义】：
        dtw_history: list[dict], 包含各步 DTW 迭代诊断信息的字典列表 (如 mode, cc_after, accept 等)。
        result_dir: str/Path, 图像保存的目录路径。
        filename: str, 保存的文件名，默认为 "fig07_dtw_history.png"。
    """
    result_dir = ensure_dir(result_dir)

    if dtw_history is None or len(dtw_history) == 0:
        return None

    phases = [item.get("phase", item.get("mode", "")) for item in dtw_history]
    cc_after = np.array([item.get("cc_after", np.nan) for item in dtw_history], dtype=float)
    accepted = [bool(item.get("accept", False)) for item in dtw_history]
    x = np.arange(len(dtw_history))

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x, cc_after, marker="o", linewidth=1.0)
    for idx, ok in enumerate(accepted):
        marker = "o" if ok else "x"
        ax.scatter(idx, cc_after[idx], marker=marker)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{idx + 1}:{phase}" for idx, phase in enumerate(phases)],
        rotation=45,
        ha="right",
    )
    ax.set_ylabel("CC after candidate")
    ax.set_title("DTW candidate history")
    fig.tight_layout()

    out_path = result_dir / filename
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_acceptance_summary(
    *,
    metrics,
    result_dir,
    filename: str = "acceptance_summary.png",
):
    """
    绘制标定全流程相关系数 (CC) 柱状对比与最终模型验收决策总结图 (fig08_acceptance_summary.png)。

    【图像的意义】：
        标定方案的“终审判决书”。用直观的柱状图对比每个里程碑阶段的相关系数，并在标题处显著显示
        最终推荐的模型类型，以及时变子波验收闸门 (W_pass) 和 Q 约束闸门 (q_pass) 的通过状态。

    【所含内容】：
        - CC_stationary (柱1)：平稳先验子波合成记录的相关系数。
        - CC_after_DTW (柱2)：仅时深对齐（固定子波）后的相关系数。
        - CC_tv_direct (柱3)：时变子波反演直接合成记录的相关系数。
        - CC_final (柱4)：推荐输出的最终模型合成记录的相关系数（如回退或Q约束）。

    【参数含义】：
        metrics: dict, 全流程最终计算的各项质量指标字典，必须包含各大阶段的 CC 值和 W_pass, q_pass 状态。
        result_dir: str/Path, 图像保存的目录路径。
        filename: str, 保存的文件名，默认为 "fig08_acceptance_summary.png"。
    """
    result_dir = ensure_dir(result_dir)

    keys = ["CC_stationary", "CC_after_DTW", "CC_tv_direct", "CC_final"]
    values = [metrics.get(key, np.nan) for key in keys]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(keys, values)
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set_ylabel("CC")
    ax.margins(y=0.2)
    ax.set_title(
        f"Final after fallback: {metrics.get('final_model_type', 'unknown')}\n"
        f"W_pass={metrics.get('W_pass')}, q_pass={metrics.get('q_pass')}"
    )
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()

    out_path = result_dir / filename
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_peak_metric_with_acceptance_band(
    *,
    t_work,
    peak_metric_ms,
    alignment,
    result_dir,
    filename: str = "peak_metric_with_acceptance_band.png",
    center_peak_allowed_ms=None,
    causal_peak_allowed_ms=(0.0, 40.0),
):
    """
    绘制时变子波波峰随时间偏移特征及质量控制边界带图 (fig09_peak_metric_with_acceptance_band.png)。

    【图像的意义】：
        防止时变子波反演因过度拟合而产生非物理的“相位漂移”。这是地球物理质控（QC Gate）中最核心的图形，
        如果偏移量频繁超出红色阴影质控区，则会被系统自动判定为“非物理子波漂移”并触发 Fallback 安全回退。

    【所含内容】：
        - 黑色实线：时变子波波峰延迟时间（单位毫秒，ms）随双程时（TWT）变化的轨迹。
        - 红色虚线与淡红阴影区：允许的最大子波漂移允许范围（质控区间带）。
        - 0 ms 灰色水平线（中心对齐时）：零偏参考基准线。

    【参数含义】：
        t_work: np.ndarray, 双程时 (TWT) 坐标轴，单位秒 (s)。
        peak_metric_ms: np.ndarray, 每一时刻提取出的子波峰值延迟时间 (ms)。
        alignment: str, 子波卷积对齐方式，'center' (中心对齐/零相位) 或 'causal' (因果/最小相位)。
        result_dir: str/Path, 图像保存的目录路径。
        filename: str, 保存的文件名，默认为 "fig09_peak_metric_with_acceptance_band.png"。
        center_peak_allowed_ms: tuple, 中心对齐时允许的最大漂移上下限 (如 -15.0 到 15.0 ms)。
        causal_peak_allowed_ms: tuple, 因果对齐时允许的波峰时变延迟区间 (如 0.0 到 40.0 ms)。
    """
    result_dir = ensure_dir(result_dir)
    t_work = np.asarray(t_work, dtype=float)
    peak_metric_ms = np.asarray(peak_metric_ms, dtype=float)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_work, peak_metric_ms, linewidth=1.0, color="k")

    if str(alignment).lower() == "center":
        if center_peak_allowed_ms is None:
            center_peak_allowed_ms = (-15.0, 15.0)
        lo, hi = center_peak_allowed_ms
        ax.axhline(0.0, color="0.35", linewidth=0.8)
        ax.axhline(lo, color="tab:red", linestyle="--", linewidth=0.9)
        ax.axhline(hi, color="tab:red", linestyle="--", linewidth=0.9)
        ax.fill_between(t_work, lo, hi, color="tab:red", alpha=0.08)
        ax.set_ylabel("Peak shift from center (ms)")
    else:
        lo, hi = causal_peak_allowed_ms
        ax.axhline(lo, color="tab:red", linestyle="--", linewidth=0.9)
        ax.axhline(hi, color="tab:red", linestyle="--", linewidth=0.9)
        ax.fill_between(t_work, lo, hi, color="tab:red", alpha=0.08)
        ax.set_ylabel("Peak time after reflection (ms)")

    ax.set_xlabel("TWT (s)")
    ax.set_title("Final after fallback\nFinal wavelet peak metric with acceptance band")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    out_path = result_dir / filename
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_wavelet_energy_and_valid_mask(
    *,
    t_work,
    wavelet_energy,
    valid_mask,
    peak_metric_ms,
    result_dir,
    filename: str = "wavelet_energy_and_valid_mask.png",
):
    r"""
    绘制时变子波能量分布、有效计算区域掩码与子波波峰偏移的多轴综合诊断图 (fig10_wavelet_energy_and_valid_mask.png)。

    【图像的意义】：
        诊断低能量反射系数区域（即反射系数极弱或空白的地层）对时变子波反演稳定性的潜在威胁。
        - 绿线 (valid_mask) 为 0 表示该时刻不是通过反演 QC 的直接反演中心，
          不代表记录被静音，也不代表最终子波无效。
        - 蓝线 (wavelet_energy) 代表提取的子波强度变化，归一化子波能量。
        - 红线 (peak_metric_ms) 监测在低反射能段的相位或波峰是否发生不稳定的震荡，子波主峰偏移。

    【所含内容】：
        - 左Y轴 (蓝色曲线)：时变子波随深度归一化后的能量曲线 (wavelet_energy_norm)。
        - 左Y轴 (绿色曲线)：时变反演的有效掩码 (valid_mask)，1为计算成功有效，0为跳过并插值平滑区。
        - 右Y轴 (红色曲线)：时变子波提取的波峰延迟时间曲线 (peak_metric_ms)。

    【参数含义】：
        t_work: np.ndarray, 双程时 (TWT) 坐标轴，单位秒 (s)。
        wavelet_energy: np.ndarray, shape (N_time,)，候选子波的 L2 范数；仅在图中归一化。
        valid_mask: np.ndarray, 1D 布尔或 0-1 数组，指示时变反演对该时刻是否有效。
        peak_metric_ms: np.ndarray, 时变子波的波峰时间延迟 (ms)。
        result_dir: str/Path, 图像保存的目录路径。
        filename: str, 保存的文件名，默认为 "fig10_wavelet_energy_and_valid_mask.png"。
    """
    result_dir = ensure_dir(result_dir)
    t_work = np.asarray(t_work, dtype=float)
    wavelet_energy = np.asarray(wavelet_energy, dtype=float)
    valid_mask = np.asarray(valid_mask, dtype=float)
    peak_metric_ms = np.asarray(peak_metric_ms, dtype=float)

    fig, ax_left = plt.subplots(figsize=(10, 4.5))
    ax_right = ax_left.twinx()  # 共享X轴，创建双Y轴

    # 对子波能量进行归一化，以便在0-1尺度下与有效掩码对比
    energy_scale = np.nanmax(np.abs(wavelet_energy))
    if not np.isfinite(energy_scale) or energy_scale <= 0:
        energy_norm = np.zeros_like(wavelet_energy)
    else:
        energy_norm = wavelet_energy / energy_scale

    # 绘制左轴：能量与掩码
    ax_left.plot(t_work, energy_norm, color="tab:blue", linewidth=1.0, label="wavelet_energy_norm")
    ax_left.plot(t_work, valid_mask, color="tab:green", linewidth=0.9, alpha=0.8, label="valid_mask")
    ax_left.set_ylabel("Normalized energy / valid mask")
    ax_left.set_ylim(-0.05, 1.1)

    # 绘制右轴：波峰偏移
    ax_right.plot(t_work, peak_metric_ms, color="tab:red", linewidth=1.0, label="peak_metric_ms")
    ax_right.set_ylabel("Peak metric (ms)")

    # 整合左右轴的图例 (Legends)
    handles_left, labels_left = ax_left.get_legend_handles_labels()
    handles_right, labels_right = ax_right.get_legend_handles_labels()
    ax_left.legend(handles_left + handles_right, labels_left + labels_right, loc="best")
    ax_left.set_xlabel("TWT (s)")
    ax_left.set_title("Candidate before global acceptance\nTV candidate energy, valid mask, and peak metric")
    ax_left.grid(True, alpha=0.25)
    fig.tight_layout()

    out_path = result_dir / filename
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_centroid_frequency_vs_time(
    *,
    t_work,
    centroid_frequency_hz,
    result_dir,
    filename: str = "fig11_centroid_frequency_vs_time.png",
):
    """
    绘制提取出的时变子波质心频率随深度（时间）变化曲线图 (fig11_centroid_frequency_vs_time.png)。

    【图像的意义】：
        定量的反映地层吸收与高频衰减特征。根据高频衰减物理规律，随着双程时变深，
        子波的质心频率曲线通常呈现平缓的阶梯状或单调下降趋势。如曲线异常剧烈波动或不降反升，
        则代表局部地层存在干扰、或者是反演数值发生了不稳定波动。

    【所含内容】：
        - 蓝色折线：时变子波质心频率（单位赫兹，Hz）随双程时（TWT）变化的轨迹。

    【参数含义】：
        t_work: np.ndarray, 双程时 (TWT) 坐标轴，单位秒 (s)。
        centroid_frequency_hz: np.ndarray, 每一时刻子波功率谱求得的质心频率 (Hz)。
        result_dir: str/Path, 图像保存的目录路径。
        filename: str, 保存的文件名，默认为 "fig11_centroid_frequency_vs_time.png"。
    """
    result_dir = ensure_dir(result_dir)
    t_work = np.asarray(t_work, dtype=float)
    centroid_frequency_hz = np.asarray(centroid_frequency_hz, dtype=float)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_work, centroid_frequency_hz, color="tab:blue", linewidth=1.0)
    ax.set_xlabel("TWT (s)")
    ax.set_ylabel("Centroid frequency (Hz)")
    ax.set_title("Final after fallback\nFinal centroid frequency")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    out_path = result_dir / filename
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path



def _optional_1d(values, n_expected: int, *, dtype=float):
    """Return a length-matched 1D array, or None when diagnostics are absent."""
    if values is None:
        return None
    array = np.asarray(values, dtype=dtype).ravel()
    if array.size != n_expected:
        return None
    return array


def plot_student_t_time_diagnostics(
    *,
    t_work,
    robust_sigma,
    weight_mean,
    weight_min,
    outlier_fraction,
    effective_sample_ratio,
    robust_code,
    robust_attempted,
    robust_solution_used,
    irls_iterations,
    irls_converged,
    objective_relative_decrease,
    min_effective_sample_ratio: float,
    robust_scale_floor: float | None,
    outlier_weight_threshold: float,
    result_dir,
    filename: str = "student_t_time_diagnostics.png",
):
    """Plot Student-t scale, weights, effective information and IRLS status."""
    result_dir = ensure_dir(result_dir)
    t_work = np.asarray(t_work, dtype=float).ravel()
    n_time = t_work.size

    attempted = _optional_1d(robust_attempted, n_time, dtype=bool)
    sigma = _optional_1d(robust_sigma, n_time)
    if attempted is None:
        attempted = np.isfinite(sigma) if sigma is not None else np.zeros(n_time, dtype=bool)

    if not np.any(attempted):
        return None

    weight_mean = _optional_1d(weight_mean, n_time)
    weight_min = _optional_1d(weight_min, n_time)
    outlier_fraction = _optional_1d(outlier_fraction, n_time)
    effective_ratio = _optional_1d(effective_sample_ratio, n_time)
    robust_code = _optional_1d(robust_code, n_time)
    solution_used = _optional_1d(robust_solution_used, n_time, dtype=bool)
    irls_iterations = _optional_1d(irls_iterations, n_time)
    irls_converged = _optional_1d(irls_converged, n_time, dtype=bool)
    objective_drop = _optional_1d(objective_relative_decrease, n_time)

    t = t_work[attempted]
    fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)

    if sigma is not None:
        axes[0].plot(t, sigma[attempted], linewidth=1.0, label="local MAD sigma")
    if robust_scale_floor is not None and np.isfinite(robust_scale_floor):
        axes[0].axhline(
            float(robust_scale_floor),
            linestyle="--",
            linewidth=0.9,
            label="sigma floor",
        )
    axes[0].set_ylabel("Residual scale")
    axes[0].set_title("Candidate before global acceptance\nStudent-t local residual scale")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best")

    if weight_mean is not None:
        axes[1].plot(t, weight_mean[attempted], linewidth=1.0, label="mean weight")
    if weight_min is not None:
        axes[1].plot(t, weight_min[attempted], linewidth=0.9, label="minimum weight")
    if outlier_fraction is not None:
        axes[1].plot(
            t,
            outlier_fraction[attempted],
            linewidth=0.9,
            label=f"fraction(weight < {outlier_weight_threshold:g})",
        )
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].set_ylabel("Weight statistic")
    axes[1].set_title("Candidate before global acceptance\nWeight strength and down-weighted sample fraction")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="best")

    if effective_ratio is not None:
        axes[2].plot(
            t,
            effective_ratio[attempted],
            linewidth=1.0,
            label="effective sample ratio",
        )
    axes[2].axhline(
        float(min_effective_sample_ratio),
        linestyle="--",
        linewidth=0.9,
        label="minimum accepted ratio",
    )
    if solution_used is not None:
        axes[2].step(
            t,
            solution_used[attempted].astype(float),
            where="mid",
            linewidth=0.8,
            alpha=0.75,
            label="Student-t solution used",
        )
    axes[2].set_ylim(-0.03, 1.03)
    axes[2].set_ylabel("Ratio / flag")
    axes[2].set_title("Candidate before global acceptance\nEffective information and local solver fallback state")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(loc="best")

    if robust_code is not None:
        axes[3].scatter(t, robust_code[attempted], s=14, label="robust code")
    if irls_converged is not None:
        converged_mask = attempted & irls_converged
        if np.any(converged_mask):
            code_for_marker = (
                robust_code[converged_mask]
                if robust_code is not None
                else np.ones(np.sum(converged_mask))
            )
            axes[3].scatter(
                t_work[converged_mask],
                code_for_marker,
                s=34,
                facecolors="none",
                edgecolors="black",
                label="converged",
            )
    axes[3].axhline(0.0, linewidth=0.8, alpha=0.6)
    axes[3].set_ylabel("Robust code")
    axes[3].set_xlabel("TWT (s)")
    axes[3].set_title("Candidate before global acceptance\nIRLS state and objective decrease")
    axes[3].grid(True, alpha=0.25)

    ax_objective = axes[3].twinx()
    if objective_drop is not None:
        ax_objective.plot(
            t,
            objective_drop[attempted],
            linewidth=0.9,
            alpha=0.75,
            label="relative objective decrease",
        )
    if irls_iterations is not None:
        # Iterations are included in the legend as a summary rather than another axis.
        finite_iter = irls_iterations[attempted]
        finite_iter = finite_iter[np.isfinite(finite_iter)]
        if finite_iter.size:
            axes[3].text(
                0.01,
                0.04,
                f"median iterations = {np.median(finite_iter):.1f}",
                transform=axes[3].transAxes,
                fontsize=9,
                va="bottom",
            )
    ax_objective.set_ylabel("Relative objective decrease")

    handles1, labels1 = axes[3].get_legend_handles_labels()
    handles2, labels2 = ax_objective.get_legend_handles_labels()
    axes[3].legend(handles1 + handles2, labels1 + labels2, loc="best")

    fig.tight_layout()
    out_path = result_dir / filename
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def plot_student_t_weight_map(
    *,
    t_work,
    dt: float,
    weight_map,
    center_indices,
    window_offsets_samples,
    outlier_weight_threshold: float,
    result_dir,
    solution_used=None,
    post_qc_valid=None,
    skip_code=None,
    row_filter: str = "all",
    filename: str = "student_t_weight_map.png",
):
    """
    Plot Student-t IRLS weights for sparse inversion centers.

    Notes
    -----
    The weight matrix is calculated from the IRLS solution before the later
    zero-mean, polarity, peak-lock and amplitude-jump QC operations.  The
    status strip on the right therefore explicitly shows whether each row:

    1. actually used the Student-t solution; and
    2. passed the later physical QC and became a direct valid wavelet window.
    """
    result_dir = ensure_dir(result_dir)
    t_work = np.asarray(t_work, dtype=float).ravel()

    if (
        weight_map is None
        or center_indices is None
        or window_offsets_samples is None
    ):
        return None

    weights = np.asarray(weight_map, dtype=float)
    center_indices = np.asarray(center_indices, dtype=int).ravel()
    offsets = np.asarray(window_offsets_samples, dtype=float).ravel()

    if (
        weights.ndim != 2
        or weights.size == 0
        or center_indices.size != weights.shape[0]
        or offsets.size != weights.shape[1]
    ):
        return None

    n_rows = weights.shape[0]

    def _optional_row_status(values, *, dtype):
        if values is None:
            return None
        array = np.asarray(values, dtype=dtype).ravel()
        return array if array.size == n_rows else None

    solution_used = _optional_row_status(solution_used, dtype=bool)
    post_qc_valid = _optional_row_status(post_qc_valid, dtype=bool)
    skip_code = _optional_row_status(skip_code, dtype=int)

    row_filter = str(row_filter).strip().lower()
    allowed_filters = {"all", "robust_used", "post_qc_valid"}
    if row_filter not in allowed_filters:
        raise ValueError(
            "row_filter 必须是 'all'、'robust_used' 或 'post_qc_valid'。"
        )

    valid_rows = (
        (center_indices >= 0)
        & (center_indices < t_work.size)
        & np.any(np.isfinite(weights), axis=1)
    )

    if row_filter == "robust_used":
        if solution_used is None:
            return None
        valid_rows &= solution_used
    elif row_filter == "post_qc_valid":
        if post_qc_valid is None:
            return None
        valid_rows &= post_qc_valid

    if not np.any(valid_rows):
        return None

    center_indices = center_indices[valid_rows]
    weights = weights[valid_rows]
    if solution_used is not None:
        solution_used = solution_used[valid_rows]
    if post_qc_valid is not None:
        post_qc_valid = post_qc_valid[valid_rows]
    if skip_code is not None:
        skip_code = skip_code[valid_rows]

    order = np.argsort(center_indices)
    center_indices = center_indices[order]
    weights = weights[order]
    if solution_used is not None:
        solution_used = solution_used[order]
    if post_qc_valid is not None:
        post_qc_valid = post_qc_valid[order]
    if skip_code is not None:
        skip_code = skip_code[order]

    center_times = t_work[center_indices]
    offset_ms = offsets * float(dt) * 1000.0
    masked_weights = np.ma.masked_invalid(weights)

    has_status = solution_used is not None or post_qc_valid is not None
    if has_status:
        fig = plt.figure(figsize=(12, 7), constrained_layout=True)
        grid = fig.add_gridspec(1, 2, width_ratios=(18, 2), wspace=0.08)
        ax = fig.add_subplot(grid[0, 0])
        ax_status = fig.add_subplot(grid[0, 1], sharey=ax)
    else:
        fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
        ax_status = None

    image = ax.pcolormesh(
        offset_ms,
        center_times,
        masked_weights,
        shading="auto",
        vmin=0.0,
        vmax=1.0,
        cmap="viridis",
    )

    finite = weights[np.isfinite(weights)]
    if (
        finite.size
        and np.min(finite) <= outlier_weight_threshold <= np.max(finite)
        and center_times.size >= 2
    ):
        ax.contour(
            offset_ms,
            center_times,
            weights,
            levels=[float(outlier_weight_threshold)],
            linewidths=0.6,
        )

    ax.set_xlabel("Local window offset (ms)")
    ax.set_ylabel("Window center TWT (s)")
    ax.set_title(
        "Candidate before global acceptance\nStudent-t IRLS weight map "
        f"(rows: {row_filter.replace('_', ' ')})"
    )
    ax.invert_yaxis()
    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label("IRLS weight")

    if ax_status is not None:
        status_names: list[str] = []
        status_values: list[np.ndarray] = []
        if solution_used is not None:
            status_names.append("ST used")
            status_values.append(solution_used.astype(float))
        if post_qc_valid is not None:
            status_names.append("QC valid")
            status_values.append(post_qc_valid.astype(float))

        for column, values in enumerate(status_values):
            ax_status.scatter(
                np.full(center_times.size, column, dtype=float),
                center_times,
                c=values,
                cmap="gray_r",
                vmin=0.0,
                vmax=1.0,
                marker="s",
                s=18,
                linewidths=0.0,
            )

        ax_status.set_xlim(-0.6, max(0.6, len(status_names) - 0.4))
        ax_status.set_xticks(np.arange(len(status_names)))
        ax_status.set_xticklabels(status_names, rotation=90)
        ax_status.tick_params(axis="y", labelleft=False)
        ax_status.grid(False)
        ax_status.set_title("Candidate before\nglobal acceptance\nStatus", fontsize=8)

        if skip_code is not None:
            rejected = skip_code[skip_code < 0]
            if rejected.size:
                values, counts = np.unique(rejected, return_counts=True)
                summary = ", ".join(
                    f"{int(value)}:{int(count)}"
                    for value, count in zip(values, counts)
                )
                ax_status.text(
                    0.5,
                    0.01,
                    f"rejected codes\n{summary}",
                    transform=ax_status.transAxes,
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    out_path = result_dir / filename
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path

def plot_candidate_and_final_wavelet_qc(
    *,
    t_work,
    candidate_diagnostics,
    final_diagnostics,
    final_source_codes,
    result_dir,
):
    """分开绘制全局验收前的 TV 候选与全部回退后的最终子波 QC。

    输入：t_work 为双程时，shape (N_time,)，单位 s；两个 diagnostics 字典
    中每个指标均为 shape (N_time,)。峰值位置单位 ms，能量沿用既有 L2 范数
    （子波振幅单位），质心频率单位 Hz，掩码、旁瓣比、边缘能量比无量纲。
    final_source_codes 为已有的 {来源名称: 整数编码} 映射，result_dir 为输出目录。

    输出：包含 candidate_qc、final_qc 两个 PNG 路径的字典，无数组输出。
    数学作用与物理假设：只显示已有诊断值，不重新定义指标、不平滑或填补 NaN，
    不把候选 valid/reliable 掩码解释为最终模型来源；缺失指标明确标为 unavailable。
    """
    t_work = np.asarray(t_work, dtype=float)  # shape: (N_time,)，双程时 s
    if t_work.ndim != 1 or t_work.size == 0 or not np.all(np.isfinite(t_work)):
        raise ValueError("t_work 必须是非空有限一维时间轴，单位 s。")
    for diagnostics in (candidate_diagnostics, final_diagnostics):
        if not isinstance(diagnostics, dict):
            raise ValueError("diagnostics 必须是指标字典。")
        for name, values in diagnostics.items():
            if values is not None and np.asarray(values).shape != t_work.shape:
                raise ValueError(f"{name} 必须与 t_work 同为 shape (N_time,)。")
    if not isinstance(final_source_codes, dict):
        raise ValueError("final_source_codes 必须是来源名称到整数编码的字典。")

    # 仅指定显示顺序和坐标单位；数值直接来自对应阶段的诊断量。
    candidate_panels = [
        ("peak_metric_ms", "TV candidate peak metric", "Peak metric (ms)"),
        ("energy_l2", "TV candidate energy", "Energy (L2 norm)"),
        ("valid_mask", "TV candidate valid mask", "Valid (0/1)"),
        ("reliable_mask", "TV candidate reliable mask", "Reliable (0/1)"),
        ("centroid_frequency_hz", "TV candidate centroid frequency", "Frequency (Hz)"),
        ("side_lobe_ratio", "TV candidate side-lobe ratio", "Ratio"),
        ("edge_energy_ratio", "TV candidate edge-energy ratio", "Ratio"),
    ]
    final_panels = [
        ("peak_metric_ms", "Final peak metric", "Peak metric (ms)"),
        ("energy_l2", "Final energy", "Energy (L2 norm)"),
        ("centroid_frequency_hz", "Final centroid frequency", "Frequency (Hz)"),
        ("source_code", "Final wavelet source", "Source"),
    ]
    groups = [
        ("candidate_qc", "Candidate before global acceptance", candidate_diagnostics,
         candidate_panels, "fig16_candidate_qc.png"),
        ("final_qc", "Final after fallback", final_diagnostics,
         final_panels, "fig17_final_qc.png"),
    ]
    result_dir = ensure_dir(result_dir)
    paths = {}
    for group_name, stage_title, diagnostics, panels, filename in groups:
        # axes shape: (指标数,)，所有子图共用同一条双程时轴。
        fig, axes = plt.subplots(len(panels), 1, figsize=(12, 2.2 * len(panels)), sharex=True)
        for ax, (name, metric_title, unit_label) in zip(axes, panels):
            ax.set_title(f"{stage_title}\n{metric_title}", fontsize=10)
            ax.set_ylabel(unit_label)
            ax.grid(True, alpha=0.25)
            values = diagnostics.get(name)
            if values is None:
                ax.text(0.5, 0.5, "Unavailable", ha="center", transform=ax.transAxes)
                continue
            values = np.asarray(values, dtype=float)  # shape: (N_time,)，不改变诊断值
            if not np.any(np.isfinite(values)):
                ax.text(0.5, 0.5, "Unavailable (no finite values)", ha="center", transform=ax.transAxes)
            if name in ("valid_mask", "reliable_mask", "source_code"):
                # 掩码及模型来源是离散状态，只用阶梯线显示，不对类别做插值。
                ax.step(t_work, values, where="mid", linewidth=0.9)
            else:
                ax.plot(t_work, values, linewidth=1.0)
            if name in ("valid_mask", "reliable_mask"):
                ax.set_yticks([0, 1])
                ax.set_ylim(-0.1, 1.1)
            if name == "source_code":
                if final_source_codes:
                    source_labels = []
                    source_ticks = []
                    for source_name, source_code in final_source_codes.items():
                        source_labels.append(source_name.replace("_", " "))
                        source_ticks.append(source_code)
                    ax.set_yticks(source_ticks)
                    ax.set_yticklabels(source_labels, fontsize=8)
                else:
                    ax.set_ylabel("Source code (labels unavailable)")
        axes[-1].set_xlabel("TWT (s)")
        fig.tight_layout()
        out_path = result_dir / filename
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        paths[group_name] = out_path
    return paths


def plot_all(
    *,
    result_dir,
    t_work,
    obs_work,
    s_syn_stationary,
    s_syn_after_dtw,
    s_syn_tv_direct,
    s_syn_final,
    W_final,
    W_est_best,
    peak_metric_ms,
    dt,
    alignment,
    dtw_history,
    metrics,
    wavelet_energy=None,
    valid_mask=None,
    reliable_mask=None,
    fallback_mask=None,
    fallback_alpha=None,
    wavelet_energy_norm=None,
    side_lobe_ratio=None,
    edge_energy_ratio=None,
    centroid_frequency_hz=None,
    robust_sigma=None,
    weight_mean=None,
    weight_min=None,
    outlier_fraction=None,
    effective_sample_ratio=None,
    robust_code=None,
    robust_attempted=None,
    robust_solution_used=None,
    irls_iterations=None,
    irls_converged=None,
    objective_relative_decrease=None,
    min_effective_sample_ratio: float = 0.30,
    robust_scale_floor: float | None = None,
    robust_outlier_weight_threshold: float = 0.5,
    robust_weight_map=None,
    robust_weight_center_indices=None,
    robust_window_offsets_samples=None,
    robust_weight_solution_used=None,
    robust_weight_post_qc_valid=None,
    robust_weight_skip_code=None,
    robust_weight_row_filter: str = "all",
    selected_wavelet_times_s=None,
    W_reference_for_wavelet_plots=None,
    wavelet_reference_label: str = "TV candidate (before global acceptance)",
    wavelet_estimated_label: str = "Final (after fallback)",
    wavelet_wiggle_amplitude_scale: float = 0.3,
    center_limit_ms=15.0,
    center_peak_allowed_ms=None,
    causal_peak_allowed_ms=(0.0, 40.0),
    r_time=None,
    w_ricker=None,
    candidate_diagnostics=None,
    final_wavelet_source_code=None,
    w_prior=None,
    w_rejected_stationary=None,
):
    """
    井震标定全流程高等级质量控制 (QC) 系列图件生成器。

    【图像的意义】：
        作为一个一键集成接口，自动协调并调用各个诊断画图函数，输出完整的标定诊断报告图件包，
        包括多道波形对比、子波提取矩阵图、二维子波多切片对比、波峰延迟及能量衰减指标等。

    【所含内容】：
        - 生成图 01：时变反演先验子波 (plot_single_wavelet)
        - 生成图 02（可选）：被拒绝的平稳候选畸变子波 (plot_single_wavelet)
        - 生成图 04：二维时变子波矩阵伪彩色图 (plot_wavelet_matrix)
        - 生成图 05：一维代表性子波切片对比图 (plot_wavelet_slices)
        - 生成图 06：子波延迟包络带检验图 (plot_peak_metric)
        - 生成图 07：DTW 校正历史 (plot_dtw_history)
        - 生成图 08：阶段 CC 柱状验收图 (plot_acceptance_summary)
        - 生成图 09：带红色警戒带的波峰漂移图 (plot_peak_metric_with_acceptance_band)
        - 生成图 10：能量、掩码与漂移综合诊断图 (plot_wavelet_energy_and_valid_mask)
        - 生成图 11：子波质心频率吸收曲线图 (plot_centroid_frequency_vs_time)
        - 生成图 12：指定时间点子波形态精细对比图 (plot_wavelet_selected_time_comparison)
        - 生成图 13：时变子波展开面板对比图 (plot_time_varying_wavelet_wiggle_panel)
        - 生成图 14：Student-t 稳健时变诊断图 (plot_student_t_time_diagnostics)
        - 生成图 15：Student-t 稳健权重时空热图 (plot_student_t_weight_map)
        - 生成图 16：验收前候选子波全面质控图 (plot_candidate_and_final_wavelet_qc)
        - 生成图 17：最终模型质量验收图 (plot_candidate_and_final_wavelet_qc)

    【参数含义】：
        result_dir: str/Path, 诊断报告保存的目标文件夹。
        t_work: np.ndarray, 实验提取窗双程时轴 (s)。
        obs_work: np.ndarray, 质控观测地震道。
        s_syn_stationary: np.ndarray, 对应平稳子波阶段的合成记录。
        s_syn_after_dtw: np.ndarray, 时深对齐阶段的合成记录。
        s_syn_tv_direct: np.ndarray, 时变直接反演的合成记录。
        s_syn_final: np.ndarray, 最终输出的合成记录。
        W_final: np.ndarray, (N_t, N_tau) 最终时变子波矩阵。
        W_est_best: np.ndarray, 时变直接反演产生的子波矩阵。
        peak_metric_ms: np.ndarray, 最终回退后子波的波峰位移指标，shape (N_time,)，ms。
        dt: float, 采样间隔 (秒)。
        alignment: str, 'center' 或 'causal' 对齐模式。
        dtw_history: list, DTW 运行迭代轨迹诊断。
        metrics: dict, 最终量化的质量指标包。
        wavelet_energy: np.ndarray (可选), 子波能量向量。
        valid_mask: np.ndarray (可选), 候选直接反演有效掩码，不能解释为最终子波有效性。
        candidate_diagnostics: dict (可选), 明确以 candidate_ 命名的验收前指标，shape (N_time,)。
        final_wavelet_source_code: np.ndarray (可选), 回退后最终模型来源编码，shape (N_time,)，无量纲。
        centroid_frequency_hz: np.ndarray (可选), 时变子波质心频率向量。
        selected_wavelet_times_s: list/tuple/np.ndarray (可选), 指定绘制子波切片的目标时间，单位 s。
        W_reference_for_wavelet_plots: np.ndarray (可选), 参考子波矩阵，例如 raw TVWI 或合成数据 W_true。
        wavelet_reference_label: str, 参考子波图例名称。
        wavelet_estimated_label: str, 最终/估计子波图例名称。
        center_limit_ms: float, 限制的偏置值。
        center_peak_allowed_ms: tuple, 中心质控偏置上下限范围。
        causal_peak_allowed_ms: tuple, 因果质控延迟上下限范围。
        w_prior: np.ndarray (可选), shape (wavelet_length,), 时变反演实际使用的平稳先验子波。
        w_rejected_stationary: np.ndarray (可选), shape (wavelet_length,), 发生畸变被拒绝的平稳候选子波。
    """
    result_dir = ensure_dir(result_dir)

    # 优先展示经过最终评估筛选后的 W_final 矩阵，若未成功通过则回退展示最优反演矩阵 W_est_best
    if W_final is not None and np.asarray(W_final).size > 0:
        W_plot = W_final
        wavelet_stage_title = "Final after fallback"
        matrix_title = f"{wavelet_stage_title}\nFinal wavelet matrix"
    else:
        W_plot = W_est_best
        wavelet_stage_title = "Candidate before global acceptance"
        matrix_title = f"{wavelet_stage_title}\nTV candidate wavelet matrix"

    paths = {}
    if w_prior is not None and np.asarray(w_prior).size > 0:
        paths["prior_wavelet_for_tv"] = plot_single_wavelet(
            w=w_prior,
            dt=dt,
            alignment=alignment,
            result_dir=result_dir,
            filename="fig01_prior_wavelet_for_tv.png",
            title="Prior Wavelet for TV Inversion",
        )
    else:
        paths["prior_wavelet_for_tv"] = None

    if w_rejected_stationary is not None and np.asarray(w_rejected_stationary).size > 0:
        paths["rejected_stationary_candidate"] = plot_single_wavelet(
            w=w_rejected_stationary,
            dt=dt,
            alignment=alignment,
            result_dir=result_dir,
            filename="fig02_rejected_stationary_candidate.png",
            title="Rejected Stationary Candidate (Fallback Triggered)",
        )
    else:
        paths["rejected_stationary_candidate"] = None
    if s_syn_final is not None and np.asarray(s_syn_final).size > 0:
        paths["obs_vs_final_wiggle"] = plot_obs_vs_final_wiggle(
            t_work=t_work,
            obs_work=obs_work,
            s_syn_final=s_syn_final,
            result_dir=result_dir,
            filename="fig03_obs_vs_final_wiggle.png",
        )
    else:
        paths["obs_vs_final_wiggle"] = None

    if r_time is not None and w_ricker is not None:
        paths["pure_ricker_vs_obs_vs_tv_wiggle"] = plot_pure_ricker_vs_obs_vs_tv_wiggle(
            t_work=t_work,
            obs_work=obs_work,
            r_time=r_time,
            w_ricker=w_ricker,
            s_syn_tv_direct=s_syn_tv_direct,
            alignment=alignment,
            result_dir=result_dir,
            filename="fig03_pure_ricker_vs_obs_vs_tv_wiggle.png",
        )
    else:
        paths["pure_ricker_vs_obs_vs_tv_wiggle"] = None

    paths["wavelet_matrix"] = plot_wavelet_matrix(
        W=W_plot,
        t_work=t_work,
        dt=dt,
        alignment=alignment,
        result_dir=result_dir,
        filename="fig04_W_final_matrix.png",
        title=matrix_title,
    )
    paths["wavelet_slices"] = plot_wavelet_slices(
        W=W_plot,
        t_work=t_work,
        dt=dt,
        alignment=alignment,
        result_dir=result_dir,
        filename="wavelet_slices.png",
        title=f"{wavelet_stage_title}\nRepresentative wavelets",
    )

    # 新增图 10：指定时间点子波形态对比。
    # 合成数据可用 W_true 做参考；真实数据可用 W_est_best 与 W_final 对比。
    W_reference_plot = W_reference_for_wavelet_plots
    if W_reference_plot is None:
        W_reference_plot = W_est_best

    paths["wavelet_selected_time_comparison"] = plot_wavelet_selected_time_comparison(
        W_est=W_plot,
        W_reference=W_reference_plot,
        t_work=t_work,
        dt=dt,
        alignment=alignment,
        result_dir=result_dir,
        target_times_s=selected_wavelet_times_s,
        filename="fig12_selected_time_wavelet_shape_comparison.png",
        est_label=wavelet_estimated_label,
        reference_label=wavelet_reference_label,
        title=f"{wavelet_stage_title}\nCandidate before global acceptance: selected-time comparison",
        valid_mask=valid_mask,
        reliable_mask=reliable_mask,
        fallback_mask=fallback_mask,
        fallback_alpha=fallback_alpha,
        peak_metric_ms=peak_metric_ms,
        wavelet_energy_norm=wavelet_energy_norm,
        side_lobe_ratio=side_lobe_ratio,
        edge_energy_ratio=edge_energy_ratio,
        print_selected_qc=True,
        candidate_diagnostics=candidate_diagnostics,
    )

    # 新增图 11：将代表性子波按照中心时间展开，观察时变演化连续性。
    paths["time_varying_wavelet_wiggle_panel"] = plot_time_varying_wavelet_wiggle_panel(
        W_est=W_plot,
        W_reference=W_reference_plot,
        t_work=t_work,
        dt=dt,
        alignment=alignment,
        result_dir=result_dir,
        target_times_s=selected_wavelet_times_s,
        filename="fig13_time_varying_wavelet_wiggle_panel.png",
        est_label=wavelet_estimated_label,
        reference_label=wavelet_reference_label,
        title=f"{wavelet_stage_title}\nCandidate before global acceptance: wavelet wiggle comparison",
        amplitude_scale=wavelet_wiggle_amplitude_scale,
        valid_mask=valid_mask,
        reliable_mask=reliable_mask,
        fallback_mask=fallback_mask,
        fallback_alpha=fallback_alpha,
        peak_metric_ms=peak_metric_ms,
        wavelet_energy_norm=wavelet_energy_norm,
        side_lobe_ratio=side_lobe_ratio,
        edge_energy_ratio=edge_energy_ratio,
        print_selected_qc=True,
        candidate_diagnostics=candidate_diagnostics,
    )
    paths["peak_metric"] = plot_peak_metric(
        t_work=t_work,
        peak_metric_ms=peak_metric_ms,
        alignment=alignment,
        result_dir=result_dir,
        filename="fig06_peak_metric.png",
        center_limit_ms=center_limit_ms,
        causal_peak_allowed_ms=causal_peak_allowed_ms,
    )
    paths["dtw_history"] = plot_dtw_history(
        dtw_history=dtw_history,
        result_dir=result_dir,
        filename="fig07_dtw_history.png",
    )
    paths["acceptance_summary"] = plot_acceptance_summary(
        metrics=metrics,
        result_dir=result_dir,
        filename="fig08_acceptance_summary.png",
    )
    paths["peak_metric_acceptance_band"] = plot_peak_metric_with_acceptance_band(
        t_work=t_work,
        peak_metric_ms=peak_metric_ms,
        alignment=alignment,
        result_dir=result_dir,
        filename="fig09_peak_metric_with_acceptance_band.png",
        center_peak_allowed_ms=center_peak_allowed_ms,
        causal_peak_allowed_ms=causal_peak_allowed_ms,
    )
    # 旧 fig08 也必须使用同一候选阶段的能量、掩码和峰值，避免与 final 混画。
    if candidate_diagnostics is None:
        candidate_diagnostics = {}
    candidate_energy = candidate_diagnostics.get("candidate_wavelet_energy_l2")
    candidate_valid = candidate_diagnostics.get("candidate_inversion_valid_mask")
    candidate_peak = candidate_diagnostics.get("candidate_peak_metric_ms")
    if candidate_energy is not None and candidate_valid is not None and candidate_peak is not None:
        paths["wavelet_energy_valid_mask"] = plot_wavelet_energy_and_valid_mask(
            t_work=t_work,
            wavelet_energy=candidate_energy,
            valid_mask=candidate_valid,
            peak_metric_ms=candidate_peak,
            result_dir=result_dir,
            filename="fig10_wavelet_energy_and_valid_mask.png",
        )
    else:
        paths["wavelet_energy_valid_mask"] = None
    if centroid_frequency_hz is not None:
        paths["centroid_frequency"] = plot_centroid_frequency_vs_time(
            t_work=t_work,
            centroid_frequency_hz=centroid_frequency_hz,
            result_dir=result_dir,
            filename="fig11_centroid_frequency_vs_time.png",
        )
    else:
        paths["centroid_frequency"] = None

    # 两组指标均为 shape (N_time,)，只传递已有诊断，不参与验收或正演。
    candidate_qc_for_plot = {
        "peak_metric_ms": candidate_peak,
        "energy_l2": candidate_energy,
        "valid_mask": candidate_valid,
        "reliable_mask": candidate_diagnostics.get("candidate_reliable_mask"),
        "centroid_frequency_hz": candidate_diagnostics.get("candidate_centroid_frequency_hz"),
        "side_lobe_ratio": candidate_diagnostics.get("candidate_side_lobe_ratio"),
        "edge_energy_ratio": candidate_diagnostics.get("candidate_edge_energy_ratio"),
    }
    final_qc_for_plot = {
        "peak_metric_ms": peak_metric_ms,
        "energy_l2": wavelet_energy,
        "centroid_frequency_hz": centroid_frequency_hz,
        "source_code": final_wavelet_source_code,
    }
    paths.update(plot_candidate_and_final_wavelet_qc(
        t_work=t_work,
        candidate_diagnostics=candidate_qc_for_plot,
        final_diagnostics=final_qc_for_plot,
        final_source_codes=metrics.get("final_wavelet_source_codes", {}),
        result_dir=result_dir,
    ))

    paths["student_t_time_diagnostics"] = plot_student_t_time_diagnostics(
        t_work=t_work,
        robust_sigma=robust_sigma,
        weight_mean=weight_mean,
        weight_min=weight_min,
        outlier_fraction=outlier_fraction,
        effective_sample_ratio=effective_sample_ratio,
        robust_code=robust_code,
        robust_attempted=robust_attempted,
        robust_solution_used=robust_solution_used,
        irls_iterations=irls_iterations,
        irls_converged=irls_converged,
        objective_relative_decrease=objective_relative_decrease,
        min_effective_sample_ratio=min_effective_sample_ratio,
        robust_scale_floor=robust_scale_floor,
        outlier_weight_threshold=robust_outlier_weight_threshold,
        result_dir=result_dir,
        filename="fig14_student_t_time_diagnostics.png",
    )

    paths["student_t_weight_map"] = plot_student_t_weight_map(
        t_work=t_work,
        dt=dt,
        weight_map=robust_weight_map,
        center_indices=robust_weight_center_indices,
        window_offsets_samples=robust_window_offsets_samples,
        outlier_weight_threshold=robust_outlier_weight_threshold,
        solution_used=robust_weight_solution_used,
        post_qc_valid=robust_weight_post_qc_valid,
        skip_code=robust_weight_skip_code,
        row_filter=robust_weight_row_filter,
        result_dir=result_dir,
        filename="fig15_student_t_weight_map.png",
    )
    return paths


def plot_origin_alpha_cap(
    *,
    result_dir,
    t_work: np.ndarray,
    extrapolated_mask: np.ndarray,
    origin_cap: np.ndarray,
    alpha_before_caps: np.ndarray,
    alpha_final: np.ndarray,
    filename: str = "fig_origin_alpha_cap.png",
) -> Path:
    """
    绘制 Soft Fallback v2.1 来源上限 (Provenance Cap) 诊断图。
    展示：
    1. Extrapolated 支撑掩码
    2. Origin Alpha Cap 距离衰减曲线
    3. Cap 前后的最终合成权值 alpha 对比与激活区域
    """
    result_dir = ensure_dir(result_dir)
    t_work = np.asarray(t_work, dtype=float).ravel()
    extrapolated_mask = np.asarray(extrapolated_mask, dtype=bool).ravel()
    origin_cap = np.asarray(origin_cap, dtype=float).ravel()
    alpha_before_caps = np.asarray(alpha_before_caps, dtype=float).ravel()
    alpha_final = np.asarray(alpha_final, dtype=float).ravel()

    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)

    # 1. 外推掩码
    axes[0].set_title("Provenance & Origin Cap Diagnostics (v2.1)\nSupport Lineage", fontsize=10)
    axes[0].step(t_work, np.asarray(extrapolated_mask, dtype=int), where="mid", color="#d95f02", lw=1.2)
    axes[0].set_ylabel("Extrapolated")
    axes[0].set_yticks([0, 1])
    axes[0].set_yticklabels(["Direct / Support", "Extrapolated"])
    axes[0].set_ylim(-0.1, 1.1)
    axes[0].grid(True, alpha=0.25)

    # 2. Origin Cap
    axes[1].set_title("Origin Alpha Cap C_origin(t)", fontsize=10)
    axes[1].plot(t_work, origin_cap, color="#7570b3", lw=1.5, label="C_origin(t)")
    axes[1].axhline(0.40, color="gray", linestyle="--", alpha=0.7, label="Near cap (0.40)")
    axes[1].axhline(0.10, color="gray", linestyle=":", alpha=0.7, label="Far cap (0.10)")
    axes[1].set_ylabel("Cap [0, 1]")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(True, alpha=0.25)

    # 3. Alpha 对比
    axes[2].set_title("Wavelet Blend Alpha: Before Caps vs Final", fontsize=10)
    axes[2].plot(t_work, alpha_before_caps, color="#1b9e77", lw=1.2, linestyle="--", label="alpha (before caps)")
    axes[2].plot(t_work, alpha_final, color="#e7298a", lw=1.5, label="alpha (final)")
    active_mask = (origin_cap < (1.0 - 1e-12)) & extrapolated_mask
    if np.any(active_mask):
        axes[2].fill_between(t_work, 0, 1, where=active_mask, color="#7570b3", alpha=0.15, label="Origin Cap Active")
    axes[2].set_ylabel("Alpha [0, 1]")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].set_xlabel("TWT (s)")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].grid(True, alpha=0.25)

    fig.tight_layout()
    out_path = result_dir / filename
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_shape_qc_v22(
    *,
    result_dir,
    t_work: np.ndarray,
    side_lobe_ratio: np.ndarray,
    edge_energy_ratio: np.ndarray,
    boundary_amp_ratio: np.ndarray,
    shape_alpha_cap: np.ndarray,
    filename: str = "fig_shape_qc_v22.png",
) -> Path:
    """
    绘制 Soft Fallback v2.2 边界振幅与形态否决上限诊断图。
    展示：
    1. Side-Lobe Ratio 与阈值线 (0.70 / 0.90 / 1.00)
    2. Edge Energy Ratio 与阈值线 (0.12 / 0.20)
    3. Boundary Amplitude Ratio 与阈值线 (0.25 / 0.45 / 0.60)
    4. Shape Alpha Cap C_shape(t)
    """
    result_dir = ensure_dir(result_dir)
    t_work = np.asarray(t_work, dtype=float).ravel()
    side_lobe_ratio = np.asarray(side_lobe_ratio, dtype=float).ravel()
    edge_energy_ratio = np.asarray(edge_energy_ratio, dtype=float).ravel()
    boundary_amp_ratio = np.asarray(boundary_amp_ratio, dtype=float).ravel()
    shape_alpha_cap = np.asarray(shape_alpha_cap, dtype=float).ravel()

    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)

    # 1. 旁瓣比
    axes[0].set_title("Shape QC & Veto Diagnostics (v2.2)\nSide-Lobe Ratio", fontsize=10)
    axes[0].plot(t_work, side_lobe_ratio, color="#1f77b4", lw=1.2)
    axes[0].axhline(0.70, color="green", linestyle="--", alpha=0.7, label="Reliable (0.70)")
    axes[0].axhline(0.90, color="orange", linestyle="--", alpha=0.7, label="Fallback (0.90)")
    axes[0].axhline(1.00, color="red", linestyle="--", alpha=0.7, label="Severe (1.00)")
    axes[0].set_ylabel("Side-lobe ratio")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(True, alpha=0.25)

    # 2. 边缘能量比
    axes[1].set_title("Edge Energy Ratio", fontsize=10)
    axes[1].plot(t_work, edge_energy_ratio, color="#ff7f0e", lw=1.2)
    axes[1].axhline(0.12, color="green", linestyle="--", alpha=0.7, label="Reliable (0.12)")
    axes[1].axhline(0.20, color="orange", linestyle="--", alpha=0.7, label="Fallback (0.20)")
    axes[1].set_ylabel("Edge energy ratio")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(True, alpha=0.25)

    # 3. 边界振幅比
    axes[2].set_title("Boundary Amplitude Ratio", fontsize=10)
    axes[2].plot(t_work, boundary_amp_ratio, color="#d62728", lw=1.2)
    axes[2].axhline(0.25, color="green", linestyle="--", alpha=0.7, label="Reliable (0.25)")
    axes[2].axhline(0.45, color="orange", linestyle="--", alpha=0.7, label="Fallback (0.45)")
    axes[2].axhline(0.60, color="red", linestyle="--", alpha=0.7, label="Severe (0.60)")
    axes[2].set_ylabel("Boundary amp ratio")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].grid(True, alpha=0.25)

    # 4. 形态上限
    axes[3].set_title("Shape Alpha Cap C_shape(t)", fontsize=10)
    axes[3].plot(t_work, shape_alpha_cap, color="#9467bd", lw=1.5, label="C_shape(t)")
    axes[3].set_ylabel("Cap [0, 1]")
    axes[3].set_ylim(-0.05, 1.05)
    axes[3].set_xlabel("TWT (s)")
    axes[3].legend(loc="upper right", fontsize=8)
    axes[3].grid(True, alpha=0.25)

    fig.tight_layout()
    out_path = result_dir / filename
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path
