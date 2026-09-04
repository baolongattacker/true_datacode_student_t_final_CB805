from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import segyio
from scipy.signal import butter, sosfiltfilt

import xwigb


def normalize_trace(trace: np.ndarray) -> np.ndarray:
    """按最大绝对振幅对单道数据归一化。"""
    trace = np.asarray(trace, dtype=np.float64)

    max_amplitude = np.max(np.abs(trace))

    if max_amplitude <= 0.0:
        return trace.copy()

    return trace / max_amplitude


def lowpass_filter_zero_phase(
    trace: np.ndarray,
    dt: float,
    cutoff_hz: float = 70.0,
    order: int = 4,
) -> np.ndarray:
    """
    对单道地震数据实施零相位 Butterworth 低通滤波。

    Parameters
    ----------
    trace
        一维地震道。
    dt
        采样间隔，单位为秒。
    cutoff_hz
        低通截止频率，单位为 Hz。
    order
        Butterworth 滤波器阶数。

    Returns
    -------
    filtered_trace
        零相位低通滤波结果。
    """
    trace = np.asarray(trace, dtype=np.float64)

    if trace.ndim != 1:
        raise ValueError("trace 必须是一维数组")

    if trace.size < 20:
        raise ValueError("地震道过短，不能稳定地进行零相位滤波")

    if not np.all(np.isfinite(trace)):
        raise ValueError("trace 中存在 NaN 或 Inf")

    if dt <= 0:
        raise ValueError("dt 必须大于 0")

    if order < 1:
        raise ValueError("order 必须大于或等于 1")

    sampling_frequency_hz = 1.0 / dt
    nyquist_frequency_hz = sampling_frequency_hz / 2.0

    if not 0.0 < cutoff_hz < nyquist_frequency_hz:
        raise ValueError(
            f"cutoff_hz={cutoff_hz} Hz 不合法，必须位于 "
            f"0～{nyquist_frequency_hz:.2f} Hz 之间"
        )

    # 二阶节形式比传统 b、a 系数形式更加稳定
    sos = butter(
        N=order,
        Wn=cutoff_hz,
        btype="lowpass",
        fs=sampling_frequency_hz,
        output="sos",
    )

    # 前向和反向滤波，消除相位延迟
    filtered_trace = sosfiltfilt(sos, trace)

    return filtered_trace


def read_sample_interval_seconds(
    src: segyio.SegyFile,
    dt: float | None,
) -> float:
    """
    获取采样间隔。

    当 dt=None 时，从 SEG-Y 二进制卷头读取。
    SEG-Y 中的采样间隔单位通常为微秒。
    """
    dt_header_us = int(src.bin[segyio.BinField.Interval])

    if dt_header_us > 0:
        dt_header_s = dt_header_us * 1e-6
    else:
        dt_header_s = None

    if dt is None:
        if dt_header_s is None:
            raise ValueError(
                "SEG-Y 卷头中没有有效采样间隔，请手动传入 dt"
            )

        print(
            f"从 SEG-Y 卷头读取采样间隔: "
            f"{dt_header_s * 1000.0:.3f} ms"
        )
        return dt_header_s

    dt = float(dt)

    if dt <= 0:
        raise ValueError("手动传入的 dt 必须大于 0")

    if dt_header_s is not None:
        difference_us = abs(dt - dt_header_s) * 1e6

        if difference_us > 0.5:
            print(
                "警告：手动传入的采样间隔与 SEG-Y 卷头不一致：\n"
                f"  手动传入: {dt * 1000.0:.3f} ms\n"
                f"  SEG-Y卷头: {dt_header_s * 1000.0:.3f} ms\n"
                "程序将采用手动传入的 dt。"
            )

    return dt


def read_trace_start_time_seconds(
    src: segyio.SegyFile,
    trace_index: int,
) -> float:
    """
    从道头读取记录延迟时间。

    DelayRecordingTime 的常见单位为毫秒。
    如果读取失败，则采用 0 秒。
    """
    try:
        raw_val = src.header[trace_index][segyio.TraceField.DelayRecordingTime]
        print(f"【道头诊断】DelayRecordingTime 原始读取值: {raw_val}")
        delay_ms = float(raw_val)
    except Exception as e:
        print(f"【道头诊断】读取或解析 DelayRecordingTime 失败，原因为: {e}")
        delay_ms = 0.0

    return delay_ms * 1e-3


def calculate_filter_qc(
    raw_trace: np.ndarray,
    filtered_trace: np.ndarray,
) -> dict:
    """计算低通滤波的基本质控指标。"""
    raw_trace = np.asarray(raw_trace, dtype=np.float64)
    filtered_trace = np.asarray(filtered_trace, dtype=np.float64)

    removed_component = raw_trace - filtered_trace

    raw_energy = np.sum(raw_trace**2)
    filtered_energy = np.sum(filtered_trace**2)
    removed_energy = np.sum(removed_component**2)

    removed_energy_ratio = removed_energy / (raw_energy + 1e-12)
    retained_energy_ratio = filtered_energy / (raw_energy + 1e-12)

    raw_std = np.std(raw_trace)
    filtered_std = np.std(filtered_trace)

    if raw_std > 0.0 and filtered_std > 0.0:
        correlation = float(
            np.corrcoef(raw_trace, filtered_trace)[0, 1]
        )
    else:
        correlation = np.nan

    return {
        "removed_component": removed_component,
        "removed_energy_ratio": float(removed_energy_ratio),
        "retained_energy_ratio": float(retained_energy_ratio),
        "raw_filtered_correlation": correlation,
    }


def extract_and_save_seismic_trace(
    segy_filepath,
    target_inline,
    target_crossline,
    out_segy_filepath=None,
    out_npz_filepath=None,
    dt=None,
    iline_byte=9,
    xline_byte=21,
    apply_lowpass=True,
    lowpass_cutoff_hz=70.0,
    lowpass_order=4,
    write_normalized_to_segy=False,
):
    """
    提取指定道，进行零相位低通滤波，并保存为 SEG-Y 和 NPZ。

    Parameters
    ----------
    segy_filepath
        输入 SEG-Y 文件。
    target_inline, target_crossline
        待提取道对应的两个道头字段值。
    out_segy_filepath
        输出单道 SEG-Y 文件。
    out_npz_filepath
        输出 NPZ 文件。为 None 时自动生成。
    dt
        采样间隔，单位秒。建议传入 None，由 SEG-Y 卷头读取。
    iline_byte, xline_byte
        用于查找目标道的两个道头字段。
    apply_lowpass
        是否进行低通滤波。
    lowpass_cutoff_hz
        低通截止频率。
    lowpass_order
        低通滤波阶数。
    write_normalized_to_segy
        True：向 SEG-Y 写入归一化滤波道；
        False：向 SEG-Y 写入保留原振幅的滤波道。
    """
    input_path = Path(segy_filepath)

    if not input_path.exists():
        raise FileNotFoundError(f"输入 SEG-Y 文件不存在：{input_path}")

    if out_segy_filepath is not None:
        output_segy_path = Path(out_segy_filepath)

        if input_path.resolve() == output_segy_path.resolve():
            raise ValueError("输出 SEG-Y 不能覆盖输入 SEG-Y")

        output_segy_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
    else:
        output_segy_path = None

    print(f"正在读取 SEG-Y 文件：{input_path}")

    with segyio.open(
        str(input_path),
        mode="r",
        ignore_geometry=True,
    ) as src:
        # ---------------------------------------------------------
        # 1. 获取用于定位目标道的道头字段
        # ---------------------------------------------------------
        inline_all = np.asarray(
            src.attributes(iline_byte)[:]
        )

        crossline_all = np.asarray(
            src.attributes(xline_byte)[:]
        )

        selected_indices = np.where(
            (inline_all == target_inline)
            & (crossline_all == target_crossline)
        )[0]

        if selected_indices.size == 0:
            raise ValueError(
                "没有找到目标地震道："
                f"field_1={target_inline}, "
                f"field_2={target_crossline}"
            )

        if selected_indices.size > 1:
            print(
                f"警告：共找到 {selected_indices.size} 道匹配记录，"
                "程序默认使用第一道。"
            )

        selected_index = int(selected_indices[0])

        # 必须 copy，防止离开 with 后引用失效
        trace_raw = np.asarray(
            src.trace[selected_index],
            dtype=np.float64,
        ).copy()

        number_of_samples = trace_raw.size

        print(
            f"成功找到目标道：道索引={selected_index}，"
            f"采样点数={number_of_samples}"
        )

        # ---------------------------------------------------------
        # 2. 获取真实采样间隔和起始时间
        # ---------------------------------------------------------
        dt_s = read_sample_interval_seconds(
            src=src,
            dt=dt,
        )

        start_time_s = read_trace_start_time_seconds(
            src=src,
            trace_index=selected_index,
        )

        # 第一个样点应位于 start_time_s，而不是 start_time_s + dt
        time_axis = (
            start_time_s
            + np.arange(number_of_samples, dtype=np.float64) * dt_s
        )

        print(
            f"时间范围：{time_axis[0]:.3f}～"
            f"{time_axis[-1]:.3f} s"
        )

        # ---------------------------------------------------------
        # 3. 对整道实施零相位低通滤波
        # ---------------------------------------------------------
        if apply_lowpass:
            trace_lowpass = lowpass_filter_zero_phase(
                trace=trace_raw,
                dt=dt_s,
                cutoff_hz=lowpass_cutoff_hz,
                order=lowpass_order,
            )

            print(
                "低通滤波完成："
                f"截止频率={lowpass_cutoff_hz:.1f} Hz，"
                f"阶数={lowpass_order}，"
                "类型=零相位 Butterworth"
            )
        else:
            trace_lowpass = trace_raw.copy()
            print("未实施低通滤波")

        # 原始道和滤波道分别归一化
        trace_raw_normalized = normalize_trace(trace_raw)
        trace_lowpass_normalized = normalize_trace(trace_lowpass)

        # ---------------------------------------------------------
        # 4. 滤波质控
        # ---------------------------------------------------------
        qc = calculate_filter_qc(
            raw_trace=trace_raw,
            filtered_trace=trace_lowpass,
        )

        removed_component = qc["removed_component"]

        print(
            f"滤除能量比例："
            f"{qc['removed_energy_ratio']:.2%}"
        )
        print(
            f"保留能量比例："
            f"{qc['retained_energy_ratio']:.2%}"
        )
        print(
            f"滤波前后相关系数："
            f"{qc['raw_filtered_correlation']:.6f}"
        )

        # ---------------------------------------------------------
        # 5. 导出单道 SEG-Y
        # ---------------------------------------------------------
        if output_segy_path is not None:
            if write_normalized_to_segy:
                trace_to_write = trace_lowpass_normalized
                output_amplitude_type = "normalized"
            else:
                # 推荐：保留滤波后的真实振幅
                trace_to_write = trace_lowpass
                output_amplitude_type = "original_amplitude"

            spec = segyio.spec()
            spec.format = 5
            spec.tracecount = 1
            spec.samples = src.samples

            with segyio.create(
                str(output_segy_path),
                spec,
            ) as dst:
                # 复制文本头
                dst.text[0] = src.text[0]

                # 复制二进制卷头
                dst.bin = src.bin

                # 由于输出格式指定为 IEEE float，复制卷头后应重新设置格式
                dst.bin[segyio.BinField.Format] = 5
                dst.bin[segyio.BinField.Interval] = int(
                    round(dt_s * 1e6)
                )
                dst.bin[segyio.BinField.Samples] = number_of_samples

                # 复制目标道的完整道头
                dst.header[0] = src.header[selected_index]

                # 写入滤波道
                dst.trace[0] = np.asarray(
                    trace_to_write,
                    dtype=np.float32,
                )

            print(
                f"单道 SEG-Y 已保存：{output_segy_path}"
            )
            print(
                f"SEG-Y 数据类型：{output_amplitude_type}"
            )

        # ---------------------------------------------------------
        # 6. 保存 NPZ
        # ---------------------------------------------------------
        if out_npz_filepath is None:
            if output_segy_path is not None:
                npz_path = (
                    output_segy_path.parent
                    / "seismic_data_CB805.npz"
                )
            else:
                npz_path = (
                    input_path.parent
                    / "seismic_data_CB805.npz"
                )
        else:
            npz_path = Path(out_npz_filepath)

        npz_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        np.savez_compressed(
            npz_path,

            # 坐标
            t_axis=time_axis,
            dt_s=np.float64(dt_s),
            start_time_s=np.float64(start_time_s),

            # 原始数据
            trace_raw=trace_raw.astype(np.float32),
            trace_raw_normalized=(
                trace_raw_normalized.astype(np.float32)
            ),

            # 滤波数据
            trace_lowpass=trace_lowpass.astype(np.float32),
            trace_lowpass_normalized=(
                trace_lowpass_normalized.astype(np.float32)
            ),

            # 被滤除部分
            removed_component=(
                removed_component.astype(np.float32)
            ),

            # 目标道信息
            target_inline=np.int64(target_inline),
            target_crossline=np.int64(target_crossline),
            selected_trace_index=np.int64(selected_index),
            iline_byte=np.int64(iline_byte),
            xline_byte=np.int64(xline_byte),

            # 滤波参数
            lowpass_applied=np.bool_(apply_lowpass),
            lowpass_cutoff_hz=np.float64(
                lowpass_cutoff_hz
            ),
            lowpass_order=np.int64(lowpass_order),
            lowpass_zero_phase=np.bool_(True),

            # 滤波质控
            removed_energy_ratio=np.float64(
                qc["removed_energy_ratio"]
            ),
            retained_energy_ratio=np.float64(
                qc["retained_energy_ratio"]
            ),
            raw_filtered_correlation=np.float64(
                qc["raw_filtered_correlation"]
            ),
        )

        print(f"NPZ 数据已保存：{npz_path}")

    return {
        "t_axis": time_axis,
        "dt_s": dt_s,
        "trace_raw": trace_raw,
        "trace_raw_normalized": trace_raw_normalized,
        "trace_lowpass": trace_lowpass,
        "trace_lowpass_normalized": trace_lowpass_normalized,
        "removed_component": removed_component,
        "selected_trace_index": selected_index,
        "npz_path": npz_path,
        "segy_path": output_segy_path,
        "qc": qc,
    }


def plot_filter_qc(
    result: dict,
    time_range=(2.0, 4.0),
    save_directory=None,
):
    """
    绘制原始道、滤波道、被滤除成分和频谱。
    """
    t = result["t_axis"]
    dt = result["dt_s"]
    raw = result["trace_raw"]
    filtered = result["trace_lowpass"]
    removed = result["removed_component"]

    t_start, t_end = time_range

    mask = (t >= t_start) & (t <= t_end)

    if not np.any(mask):
        raise ValueError(
            f"时间范围 {time_range} 不在地震道范围内"
        )

    if save_directory is not None:
        save_directory = Path(save_directory)
        save_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    # 使用原始道最大振幅作为统一尺度，避免分别归一化掩盖振幅变化
    amplitude_scale = np.max(np.abs(raw)) + 1e-12

    # ---------------------------------------------------------
    # 图3：滤波前后频谱
    # ---------------------------------------------------------
    frequencies = np.fft.rfftfreq(
        raw.size,
        d=dt,
    )

    raw_spectrum = np.abs(
        np.fft.rfft(raw)
    )

    filtered_spectrum = np.abs(
        np.fft.rfft(filtered)
    )

    spectrum_scale = np.max(raw_spectrum) + 1e-12

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        frequencies,
        raw_spectrum / spectrum_scale,
        linewidth=0.8,
        label="Raw",
    )

    ax.plot(
        frequencies,
        filtered_spectrum / spectrum_scale,
        linewidth=1.0,
        label="Low-pass",
    )

    ax.set_xlim(
        0.0,
        min(150.0, 0.5 / dt),
    )
    ax.set_xlabel("Frequency / Hz")
    ax.set_ylabel("Normalized amplitude spectrum")
    ax.set_title("Amplitude spectra before and after filtering")
    ax.legend()
    ax.grid(True, alpha=0.25)

    fig.tight_layout()

    if save_directory is not None:
        save_path = (
            save_directory
            / "CB805_filter_spectrum.png"
        )
        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )
        print(f"频谱对比图已保存：{save_path}")


def plot_segmented_wiggles(
    result: dict,
    target_inline: int,
    target_crossline: int,
    save_directory,
):
    """
    将滤波后的地震道分为三个时间段绘制变面积图。
    """
    t = result["t_axis"]

    # 使用滤波后的未归一化地震道
    trace = result["trace_lowpass"]

    # xwigb 通常需要二维数据。
    # 这里复制为三道，仅用于展示单道波形。
    plot_data = np.column_stack(
        [trace, trace, trace]
    )

    x_positions = np.array([1.0, 2.0, 3.0])

    time_segments = [
        (0.0, 2.0),
        (2.0, 4.0),
        (4.0, 6.0),
    ]

    save_directory = Path(save_directory)
    save_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    for segment_index, (t_start, t_end) in enumerate(
        time_segments,
        start=1,
    ):
        mask = (t >= t_start) & (t <= t_end)

        if not np.any(mask):
            print(
                f"跳过 {t_start:.1f}～{t_end:.1f} s："
                "该时间段不在数据范围内。"
            )
            continue

        time_sub = t[mask]
        data_sub = plot_data[mask, :]

        fig, ax = plt.subplots(figsize=(4, 6))

        xwigb.xwigb(
            seis=data_sub,
            t=time_sub,
            x=x_positions,
            scale=1.5,
            linewidth=0.8,
            mode="vertical",
            wiggle_fill="peak_fill",
            wigb_color="k",
            ax=ax,
        )

        ax.set_title(
            f"Inline {target_inline}, "
            f"Crossline {target_crossline}\n"
            f"Zero-phase low-pass: "
            f"{int(t_start * 1000)}–"
            f"{int(t_end * 1000)} ms"
        )

        fig.tight_layout()

        save_path = (
            save_directory
            / f"seismic_CB805_lowpass_part{segment_index}.png"
        )

        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

        print(
            f"分段滤波道图 {segment_index} 已保存："
            f"{save_path}"
        )


if __name__ == "__main__":
    # ---------------------------------------------------------
    # 1. 输入与输出路径
    # ---------------------------------------------------------
    segy_file = Path(
        r"F:\02JDX_Data\测井实际数据"
        r"\项目第三次数据工区资料（斜井）"
        r"\seismicdata.segy"
    )

    output_directory = Path(
        r"F:\01JDX_code\python_code"
        r"\seismic_well_tying"
        r"\true_datacode_student_t_final_CB805"
        r"\_experiment_data\seismic_CB805"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_segy = (
        output_directory
        / "seismic_CB805_lowpass_70Hz.sgy"
    )

    output_npz = (
        output_directory
        / "seismic_data_CB805_lowpass_70Hz.npz"
    )

    target_inline = 1668
    target_crossline = 1557

    # ---------------------------------------------------------
    # 2. 提取、滤波和保存
    # ---------------------------------------------------------
    result = extract_and_save_seismic_trace(
        segy_filepath=segy_file,
        target_inline=target_inline,
        target_crossline=target_crossline,
        out_segy_filepath=output_segy,
        out_npz_filepath=output_npz,

        # 推荐设为 None，让程序读取 SEG-Y 的真实采样间隔
        dt=None,

        # 你当前数据中使用 FieldRecord 和 CDP 定位
        iline_byte=9,
        xline_byte=21,

        apply_lowpass=True,
        lowpass_cutoff_hz=70.0,
        lowpass_order=4,

        # 推荐 False，避免导出的 SEG-Y 丢失真实振幅比例
        write_normalized_to_segy=False,
    )

    # ---------------------------------------------------------
    # 3. 滤波质控图
    # ---------------------------------------------------------
    plot_filter_qc(
        result=result,
        time_range=(2.0, 4.0),
        save_directory=output_directory,
    )

    # ---------------------------------------------------------
    # 4. 分段变面积图
    # ---------------------------------------------------------
    plot_segmented_wiggles(
        result=result,
        target_inline=target_inline,
        target_crossline=target_crossline,
        save_directory=output_directory,
    )

    plt.show()