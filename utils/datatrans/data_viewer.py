# -*- coding: utf-8 -*-
"""
data_viewer.py

通用数据查看工具：
支持 txt / csv / npy / npz / las / segy / sgy 等常见地球物理数据文件。

主要作用：
1. 查看文件基本信息
2. 打印数组 shape / dtype / 数值范围
3. 预览文本前几行
4. 对一维、二维数组做简单绘图
5. 对 SEGY 文件查看 trace 数量、采样点、采样间隔，并绘制部分道集
6. 对 LAS 文件查看曲线信息，并绘制测井曲线

"""

import os
import numpy as np
import matplotlib.pyplot as plt


def ensure_dir(path):
    """确保输出目录存在。"""
    if path is None:
        return None

    os.makedirs(path, exist_ok=True)
    return path


def print_file_basic_info(file_path):
    """打印文件基础信息。"""
    print("=" * 80)
    print("[文件信息]")
    print(f"  file_path = {file_path}")

    if not os.path.exists(file_path):
        print("  文件不存在。")
        print("=" * 80)
        return False

    file_size = os.path.getsize(file_path)
    file_size_mb = file_size / 1024.0 / 1024.0

    print(f"  file_name = {os.path.basename(file_path)}")
    print(f"  file_size = {file_size_mb:.3f} MB")
    print("=" * 80)

    return True


def print_array_info(name, array, max_print_values=8):
    """
    打印数组基本信息。

    参数：
    name : str
        数组名称。
    array : np.ndarray
        待查看数组。
    max_print_values : int
        一维数组最多打印多少个前几个样点。

    说明：
    这里不直接 print(array)，避免大数组刷屏。
    """

    array = np.asarray(array)

    print("-" * 80)
    print(f"[数组] {name}")
    print(f"  shape = {array.shape}")
    print(f"  ndim  = {array.ndim}")
    print(f"  dtype = {array.dtype}")

    # 标量数组
    if array.ndim == 0:
        print(f"  value = {array}")
        return

    # 结构化数组不强行做数值统计
    if array.dtype.names is not None:
        print(f"  structured fields = {array.dtype.names}")
        return

    # object / string 类型不做数值统计
    if not np.issubdtype(array.dtype, np.number):
        print("  非数值数组，只打印基本信息。")

        flat_array = array.ravel()
        n_print = min(len(flat_array), max_print_values)
        print(f"  first {n_print} values = {flat_array[:n_print]}")
        return

    # 数值数组统计
    finite_mask = np.isfinite(array)
    finite_count = np.sum(finite_mask)
    total_count = array.size

    print(f"  finite_count = {finite_count} / {total_count}")

    if finite_count > 0:
        finite_values = array[finite_mask]

        print(f"  min  = {np.min(finite_values):.6g}")
        print(f"  max  = {np.max(finite_values):.6g}")
        print(f"  mean = {np.mean(finite_values):.6g}")
        print(f"  std  = {np.std(finite_values):.6g}")
    else:
        print("  没有有效数值。")

    # 打印前几个值，方便快速判断数据内容
    flat_array = array.ravel()
    n_print = min(len(flat_array), max_print_values)
    print(f"  first {n_print} values = {flat_array[:n_print]}")


def plot_array_simple(array, title, save_dir=None, filename=None, show=True, close_fig=True):
    """
    对一维或二维数组做简单查看图。

    参数：
    array : np.ndarray
        待绘图数组。
    title : str
        图标题。
    save_dir : str or None
        图片保存目录。
    filename : str or None
        图片文件名。
    show : bool
        是否 plt.show()。
    close_fig : bool
        如果 show 为 False，是否关闭图窗以释放内存。如果要在外部统一调用 plt.show()，请设为 False。

    说明：
    一维数组：画曲线。
    二维数组：画 imshow 图像。
    更复杂的数据结构不在这里处理。
    """

    array = np.asarray(array)

    if not np.issubdtype(array.dtype, np.number):
        print(f"[绘图跳过] {title}: 非数值数组。")
        return

    if array.ndim == 0:
        print(f"[绘图跳过] {title}: 标量数组。")
        return

    if array.ndim == 1:
        plt.figure(figsize=(8, 4))
        plt.plot(array, linewidth=1.0)
        plt.title(title)
        plt.xlabel("Sample Index")
        plt.ylabel("Amplitude")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()

    elif array.ndim == 2:
        # 如果二维数组某一维为 1，则退化为一维曲线
        if array.shape[0] == 1 or array.shape[1] == 1:
            curve = array.ravel()

            plt.figure(figsize=(8, 4))
            plt.plot(curve, linewidth=1.0)
            plt.title(title)
            plt.xlabel("Sample Index")
            plt.ylabel("Amplitude")
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout()

        else:
            plt.figure(figsize=(8, 5))
            plt.imshow(array, aspect="auto", cmap="seismic")
            plt.colorbar(label="Amplitude")
            plt.title(title)
            plt.xlabel("Column Index")
            plt.ylabel("Row Index")
            plt.tight_layout()

    else:
        print(f"[绘图跳过] {title}: ndim = {array.ndim}，暂不直接绘图。")
        return

    if save_dir is not None and filename is not None:
        ensure_dir(save_dir)
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=200)
        print(f"[保存图片] {save_path}")

    if show:
        plt.show()
    elif close_fig:
        plt.close()


def preview_text_lines(file_path, n_lines=10, encoding="utf-8"):
    """
    预览文本文件前几行。

    说明：
    有些 txt/csv 文件前面存在文件头、曲线说明、单位说明。
    所以先打印前几行，有助于判断 skip_header 应该设多少。
    """

    print("-" * 80)
    print(f"[文本预览] 前 {n_lines} 行")

    try:
        with open(file_path, "r", encoding=encoding, errors="ignore") as f:
            for i in range(n_lines):
                line = f.readline()

                if line == "":
                    break

                print(f"{i + 1:03d}: {line.rstrip()}")

    except Exception as e:
        print(f"[文本预览失败] {e}")


def load_text_numeric_data(file_path, skip_header=0, delimiter=None):
    """
    尝试把 txt/csv 读取为数值数组。

    参数：
    skip_header : int
        跳过文件头行数。
    delimiter : str or None
        分隔符。
        None 表示自动按空格/制表符分隔。
        ',' 表示 csv 逗号分隔。

    返回：
    data : np.ndarray or None
        数值数组。
    """

    # 第一种方式：np.loadtxt，适合规则数值文件
    try:
        data = np.loadtxt(
            file_path,
            skiprows=skip_header,
            delimiter=delimiter
        )

        print("[文本读取] 使用 np.loadtxt 成功。")
        return data

    except Exception as e:
        err_loadtxt = str(e)

    # 第二种方式：np.genfromtxt，适合局部缺失值或不规则文本
    try:
        data = np.genfromtxt(
            file_path,
            skip_header=skip_header,
            delimiter=delimiter,
            comments="#",
            invalid_raise=False
        )

        print("[文本读取] 使用 np.genfromtxt 成功。")
        return data

    except Exception as e:
        err_genfromtxt = str(e)

    # 只有当两种方法都失败时，才输出完整的报错信息，避免虚假报警
    print("[文本读取] 错误：无法将该文本读取为数值数组。")
    print(f"  -> np.loadtxt 失败原因: {err_loadtxt}")
    print(f"  -> np.genfromtxt 失败原因: {err_genfromtxt}")
    return None


def view_text_file(
    file_path,
    skip_header=0,
    delimiter=None,
    preview_lines=10,
    plot=True,
    first_col_as_x=True,
    invert_yaxis_for_depth=False,
    save_dir=None,
    show=True
):
    """
    查看 txt / csv 数值文件。

    常见情况：
    1. 测井 txt: 第一列 depth，后面是 AC / RHOB 等曲线
    2. 普通数组 txt: 每列是一个变量
    3. csv: delimiter=','

    参数：
    first_col_as_x : bool
        True 表示第一列作为横轴或纵轴参考变量。
    invert_yaxis_for_depth : bool
        True 表示深度轴向下增加，适合测井曲线。
    """

    print_file_basic_info(file_path)
    preview_text_lines(file_path, n_lines=preview_lines)

    data = load_text_numeric_data(
        file_path=file_path,
        skip_header=skip_header,
        delimiter=delimiter
    )

    if data is None:
        print("[结果] 该文本文件无法读取为纯数值数组。")
        return None

    data = np.asarray(data)
    print_array_info("text_data", data)

    if not plot:
        return data

    if data.ndim == 1:
        plot_array_simple(
            data,
            title=os.path.basename(file_path),
            save_dir=save_dir,
            filename=os.path.basename(file_path) + "_curve.png",
            show=show
        )

    elif data.ndim == 2:
        n_rows, n_cols = data.shape

        if n_cols >= 2 and first_col_as_x:
            x = data[:, 0]

            plt.figure(figsize=(8, 6))

            for col_idx in range(1, n_cols):
                y = data[:, col_idx]
                plt.plot(y, x, linewidth=1.0, label=f"col_{col_idx}")

            plt.xlabel("Value")
            plt.ylabel("Column 0")
            plt.title(os.path.basename(file_path))
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.legend()

            if invert_yaxis_for_depth:
                plt.gca().invert_yaxis()

            plt.tight_layout()

            if save_dir is not None:
                ensure_dir(save_dir)
                save_path = os.path.join(
                    save_dir,
                    os.path.basename(file_path) + "_curves.png"
                )
                plt.savefig(save_path, dpi=200)
                print(f"[保存图片] {save_path}")

            if show:
                plt.show()
            else:
                plt.close()

        else:
            plot_array_simple(
                data,
                title=os.path.basename(file_path),
                save_dir=save_dir,
                filename=os.path.basename(file_path) + "_matrix.png",
                show=show
            )

    return data


def view_npy_file(file_path, plot=True, save_dir=None, show=True):
    """
    查看 npy 文件。

    npy 通常保存单个数组。
    """

    print_file_basic_info(file_path)

    try:
        data = np.load(file_path, allow_pickle=True)
    except Exception as e:
        print(f"[npy读取失败] {e}")
        return None

    print_array_info("npy_data", data)

    if plot:
        plot_array_simple(
            data,
            title=os.path.basename(file_path),
            save_dir=save_dir,
            filename=os.path.basename(file_path) + "_view.png",
            show=show
        )

    return data


def view_npz_file(file_path, plot=True, save_dir=None, show=True, max_plot_items=20):
    """查看 npz 文件（优化版：避免阻塞式弹窗）。"""
    print_file_basic_info(file_path)
    try:
        data = np.load(file_path, allow_pickle=True)
    except Exception as e:
        print(f"[npz读取失败] {e}")
        return None
    keys = list(data.keys())
    print("-" * 80)
    print("[npz keys]")
    for key in keys:
        print(f"  - {key}")
    print("-" * 80)
    print("[npz arrays]")
    for key in keys:
        print_array_info(key, data[key])
    if plot:
        n_plot = 0
        for key in keys:
            if n_plot >= max_plot_items:
                break
            array = np.asarray(data[key])
            if not np.issubdtype(array.dtype, np.number):
                continue
            if array.ndim in [1, 2]:
                plot_array_simple(
                    array,
                    title=f"{os.path.basename(file_path)} | {key}",
                    save_dir=save_dir,
                    filename=os.path.basename(file_path) + f"_{key}.png",
                    show=False,       # 先不 show，防止阻塞循环
                    close_fig=False   # 不要立即释放图窗，以便最后统一弹窗
                )
                n_plot += 1
        
        if show and n_plot > 0:
            plt.show()  # 一次性展示所有图窗，方便对比
    return data


def view_las_file(
    file_path,
    plot=True,
    save_dir=None,
    show=True,
    max_plot_curves=6
):
    """
    查看 LAS 测井文件。

    需要环境中安装 lasio：
    pip install lasio

    输出：
    1. well 信息
    2. curve 信息
    3. 每条曲线 shape / 数值范围
    4. 简单测井曲线图
    """

    print_file_basic_info(file_path)

    try:
        import lasio
    except Exception as e:
        print(f"[LAS读取失败] 当前环境无法 import lasio: {e}")
        return None

    try:
        las = lasio.read(file_path)
    except Exception as e:
        print(f"[LAS读取失败] {e}")
        return None

    print("-" * 80)
    print("[LAS well 信息]")
    for item in las.well:
        print(f"  {item.mnemonic}: {item.value} {item.unit}  {item.descr}")

    print("-" * 80)
    print("[LAS curve 信息]")
    for curve in las.curves:
        print(f"  {curve.mnemonic:12s} unit={curve.unit:10s} descr={curve.descr}")

    # las.index 通常是深度或时间
    depth = np.asarray(las.index, dtype=float)

    print_array_info("LAS_index_depth_or_time", depth)

    curve_names = []
    curve_arrays = []

    for curve in las.curves:
        curve_name = curve.mnemonic

        # 第一列通常是 DEPT，本身已经在 las.index 里
        if curve_name.upper() in ["DEPT", "DEPTH", "TIME"]:
            continue

        try:
            curve_data = np.asarray(las[curve_name], dtype=float)
        except Exception:
            continue

        curve_names.append(curve_name)
        curve_arrays.append(curve_data)

        print_array_info(curve_name, curve_data)

    if plot and len(curve_arrays) > 0:
        n_plot = min(len(curve_arrays), max_plot_curves)

        fig, axes = plt.subplots(
            1,
            n_plot,
            figsize=(3.0 * n_plot, 8),
            sharey=True
        )

        if n_plot == 1:
            axes = [axes]

        for i in range(n_plot):
            curve_name = curve_names[i]
            curve_data = curve_arrays[i]

            axes[i].plot(curve_data, depth, linewidth=1.0)
            axes[i].set_xlabel(curve_name)
            axes[i].grid(True, linestyle="--", alpha=0.5)
            axes[i].invert_yaxis()

            if i == 0:
                axes[i].set_ylabel("Depth / Time")

        plt.suptitle(os.path.basename(file_path))
        plt.tight_layout()

        if save_dir is not None:
            ensure_dir(save_dir)
            save_path = os.path.join(
                save_dir,
                os.path.basename(file_path) + "_las_curves.png"
            )
            plt.savefig(save_path, dpi=200)
            print(f"[保存图片] {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    return las


def view_segy_file(
    file_path,
    plot=True,
    save_dir=None,
    show=True,
    max_plot_traces=200,
    trace_index=0
):
    """
    查看 SEGY / SGY 文件。

    需要环境中安装 segyio：
    pip install segyio

    说明：
    SEGY 文件可能非常大，因此这里只读取部分道用于查看。
    不建议一开始就把全部 trace 读入内存。

    参数：
    max_plot_traces : int
        最多读取多少道用于绘图。
    trace_index : int
        打印和绘制单道时使用的道号。
    """

    print_file_basic_info(file_path)

    try:
        import segyio
    except Exception as e:
        print(f"[SEGY读取失败] 当前环境无法 import segyio: {e}")
        return None

    try:
        segy_file = segyio.open(file_path, "r", ignore_geometry=True)
    except Exception as e:
        print(f"[SEGY读取失败] {e}")
        return None

    with segy_file as f:
        try:
            f.mmap()
        except Exception:
            pass

        trace_count = f.tracecount
        samples = np.asarray(f.samples)

        try:
            dt_us = segyio.tools.dt(f)
        except Exception:
            dt_us = None

        print("-" * 80)
        print("[SEGY 信息]")
        print(f"  trace_count = {trace_count}")
        print(f"  n_samples   = {len(samples)}")

        if len(samples) > 0:
            print(f"  sample_min  = {samples[0]}")
            print(f"  sample_max  = {samples[-1]}")

        if dt_us is not None:
            print(f"  dt_us       = {dt_us}")
            print(f"  dt_s        = {dt_us * 1e-6:.6f}")

        # 读取指定单道
        if trace_index < 0:
            trace_index = 0

        if trace_index >= trace_count:
            trace_index = trace_count - 1

        trace = np.asarray(f.trace[trace_index], dtype=float)
        print_array_info(f"trace_{trace_index}", trace)

        if not plot:
            return {
                "trace_count": trace_count,
                "samples": samples,
                "dt_us": dt_us,
                "trace": trace
            }

        # 绘制单道
        plt.figure(figsize=(5, 8))
        plt.plot(trace, samples, linewidth=1.0)
        plt.gca().invert_yaxis()
        plt.title(f"{os.path.basename(file_path)} | trace {trace_index}")
        plt.xlabel("Amplitude")
        plt.ylabel("Time / Sample")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()

        if save_dir is not None:
            ensure_dir(save_dir)
            save_path = os.path.join(
                save_dir,
                os.path.basename(file_path) + f"_trace_{trace_index}.png"
            )
            plt.savefig(save_path, dpi=200)
            print(f"[保存图片] {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        # 读取部分道集用于二维显示
        n_plot_traces = min(trace_count, max_plot_traces)

        if n_plot_traces <= 0:
            return {
                "trace_count": trace_count,
                "samples": samples,
                "dt_us": dt_us,
                "trace": trace
            }

        step = max(1, trace_count // n_plot_traces)
        selected_trace_indices = np.arange(0, trace_count, step)
        selected_trace_indices = selected_trace_indices[:n_plot_traces]

        gather = np.zeros((len(selected_trace_indices), len(samples)), dtype=float)

        for i, idx in enumerate(selected_trace_indices):
            gather[i, :] = np.asarray(f.trace[idx], dtype=float)

        print_array_info("selected_segy_gather", gather)

        # 图像显示时转置为：纵轴时间，横轴道号
        plt.figure(figsize=(10, 6))
        plt.imshow(
            gather.T,
            aspect="auto",
            cmap="seismic",
            extent=[0, len(selected_trace_indices) - 1, samples[-1], samples[0]]
        )
        plt.colorbar(label="Amplitude")
        plt.title(f"{os.path.basename(file_path)} | selected traces")
        plt.xlabel("Selected Trace Index")
        plt.ylabel("Time / Sample")
        plt.tight_layout()

        if save_dir is not None:
            ensure_dir(save_dir)
            save_path = os.path.join(
                save_dir,
                os.path.basename(file_path) + "_segy_gather.png"
            )
            plt.savefig(save_path, dpi=200)
            print(f"[保存图片] {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    return {
        "trace_count": trace_count,
        "samples": samples,
        "dt_us": dt_us,
        "trace": trace,
        "selected_trace_indices": selected_trace_indices,
        "gather": gather
    }


def view_data_file(
    file_path,
    skip_header=0,
    delimiter=None,
    preview_lines=10,
    plot=True,
    save_dir=None,
    show=True,
    first_col_as_x=True,
    invert_yaxis_for_depth=False,
    max_plot_traces=200,
    trace_index=0
):
    """
    通用数据查看入口函数。

    支持格式：
    - .txt
    - .csv
    - .dat
    - .npy
    - .npz
    - .las
    - .sgy
    - .segy

    返回：
    根据文件类型返回读取结果。
    """

    if not os.path.exists(file_path):
        print(f"[错误] 文件不存在: {file_path}")
        return None

    file_ext = os.path.splitext(file_path)[1].lower()

    print("=" * 80)
    print("[数据查看入口]")
    print(f"  file_ext = {file_ext}")
    print("=" * 80)

    if file_ext in [".txt", ".dat"]:
        data = view_text_file(
            file_path=file_path,
            skip_header=skip_header,
            delimiter=delimiter,
            preview_lines=preview_lines,
            plot=plot,
            first_col_as_x=first_col_as_x,
            invert_yaxis_for_depth=invert_yaxis_for_depth,
            save_dir=save_dir,
            show=show
        )
        return data

    elif file_ext in [".csv"]:
        # csv 默认逗号分隔
        if delimiter is None:
            delimiter = ","

        data = view_text_file(
            file_path=file_path,
            skip_header=skip_header,
            delimiter=delimiter,
            preview_lines=preview_lines,
            plot=plot,
            first_col_as_x=first_col_as_x,
            invert_yaxis_for_depth=invert_yaxis_for_depth,
            save_dir=save_dir,
            show=show
        )
        return data

    elif file_ext == ".npy":
        data = view_npy_file(
            file_path=file_path,
            plot=plot,
            save_dir=save_dir,
            show=show
        )
        return data

    elif file_ext == ".npz":
        data = view_npz_file(
            file_path=file_path,
            plot=plot,
            save_dir=save_dir,
            show=show
        )
        return data

    elif file_ext == ".las":
        las = view_las_file(
            file_path=file_path,
            plot=plot,
            save_dir=save_dir,
            show=show
        )
        return las

    elif file_ext in [".sgy", ".segy"]:
        segy_info = view_segy_file(
            file_path=file_path,
            plot=plot,
            save_dir=save_dir,
            show=show,
            max_plot_traces=max_plot_traces,
            trace_index=trace_index
        )
        return segy_info

    else:
        print(f"[暂不支持] 文件格式: {file_ext}")
        return None


if __name__ == "__main__":

    # 查看 npz 数据
    # file_path = r"D:\python_code\python_project\seismic_well\true_data_tying\data\initial_model_data\initial_real_model_data_CB323.npz"
    file_path = r"F:\01JDX_code\python_code\seismic_well_tying\true_datacode_student_t_final_CB323\_experiment_data\initial_model_data\initial_real_model_data_CB323.npz"
    data = view_data_file(
        file_path=file_path,
        plot=True,
        save_dir=None,
        show=True
    )

    # 示例 2：查看 txt 测井数据
    # file_path = r"F:\Project\基于随钻测井数据的高精度井震标定及速度建模\code\third_data\result\CB323_rawDATA.txt"
    #
    # data = view_data_file(
    #     file_path=file_path,
    #     skip_header=0,
    #     delimiter=None,
    #     plot=True,
    #     first_col_as_x=True,
    #     invert_yaxis_for_depth=True,
    #     save_dir=None,
    #     show=True
    # )

    # 示例 3：查看 SEGY 数据
    # file_path = r"D:\your_data\near_well_trace.sgy"
    #
    # segy_info = view_data_file(
    #     file_path=file_path,
    #     plot=True,
    #     max_plot_traces=200,
    #     trace_index=0,
    #     save_dir=None,
    #     show=True
    # )

    # 示例 4：查看 LAS 数据
    # file_path = r"D:\your_data\CB323.las"
    #
    # las = view_data_file(
    #     file_path=file_path,
    #     plot=True,
    #     save_dir=None,
    #     show=True
    # )