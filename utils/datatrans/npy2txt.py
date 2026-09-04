# -*- coding: utf-8 -*-
"""
数据格式转换工具：将 NumPy 的 .npy 或 .npz 文件转换为文本 .txt 文件。

转换规则
--------
1. .npy 文件：
   直接读取其中唯一的 NumPy 数组，不需要指定 key。

2. .npz 文件：
   可以通过 key 指定要提取的数组。
   如果未指定 key：
   - 只有一个键时，自动选取；
   - 有多个键时，尝试匹配默认候选键；
   - 无法匹配时，抛出错误并显示所有可用键。

3. 数据维度：
   - 一维数组：保存为单列；
   - 二维数组：保持原始行列结构；
   - 三维及更高维数组：默认不自动转换，避免破坏维度物理意义。
"""

from pathlib import Path
from typing import Sequence

import numpy as np


DEFAULT_KEY_CANDIDATES = (
    "trace_lowpass",
    "trace_raw",
    "sel_data",
    "sel_data_norm",
    "data",
    "seismic_trace",
    "seismic",
    "well_data",
)


def _select_npz_key(
    keys: Sequence[str],
    key: str | None,
    default_candidates: Sequence[str],
) -> str:
    """
    从 .npz 文件的键列表中确定需要提取的数组键。
    """
    if not keys:
        raise ValueError(".npz 文件中没有包含任何数据键。")

    # 用户显式指定键
    if key is not None:
        if key not in keys:
            raise KeyError(
                f"指定的键 '{key}' 不存在。\n"
                f"可用键为: {list(keys)}"
            )
        return key

    # 文件中只有一个键，直接使用
    if len(keys) == 1:
        selected_key = keys[0]
        print(f"  [提示] .npz 中只有一个键，自动选择: '{selected_key}'")
        return selected_key

    # 按候选键顺序自动匹配
    for candidate in default_candidates:
        if candidate in keys:
            print(f"  [提示] 未指定 key，自动匹配推荐键: '{candidate}'")
            return candidate

    raise KeyError(
        f".npz 文件中包含多个数据键: {list(keys)}\n"
        "未发现默认推荐键，请显式指定 key 参数。"
    )


def _load_numpy_array(
    input_path: Path,
    key: str | list[str] | tuple[str, ...] | None = None,
    default_candidates: Sequence[str] = DEFAULT_KEY_CANDIDATES,
) -> tuple[np.ndarray, str | None]:
    """
    加载 .npy 或 .npz 文件，并返回选中的 NumPy 数组。

    返回
    ----
    data : np.ndarray
        提取出的数组（如果是多键则为拼合的二维矩阵）。

    selected_key : str or None
        .npz 中选中的键名或拼接组合键；对于普通 .npy 文件返回 None。
    """
    suffix = input_path.suffix.lower()

    if suffix == ".npy":
        if key is not None:
            print(
                f"  [提示] 输入文件是 .npy，标准 .npy 文件没有键值；"
                f"参数 key='{key}' 将被忽略。"
            )

        data = np.load(input_path, allow_pickle=False)
        return np.asarray(data), None

    if suffix == ".npz":
        # 使用 with，确保 NpzFile 被正常关闭
        with np.load(input_path, allow_pickle=False) as npz_file:
            keys = list(npz_file.files)

            print(f"  .npz 可用键: {keys}")

            if isinstance(key, (list, tuple)):
                # 检查所有指定的键是否均存在
                for k in key:
                    if k not in keys:
                        raise KeyError(
                            f"指定的键 '{k}' 不在 .npz 文件的可用键中。\n"
                            f"可用键为: {keys}"
                        )
                
                # 依次读取并拉平为 1D
                arrays = []
                for k in key:
                    arr = np.asarray(npz_file[k]).copy()
                    arrays.append(arr.ravel())
                
                # 校验长度是否对齐
                lengths = [len(a) for a in arrays]
                if len(set(lengths)) > 1:
                    raise ValueError(
                        f"要拼接的多个数据键长度不一致，无法并排对齐:\n"
                        f"  {dict(zip(key, lengths))}"
                    )
                
                # 将多道 1D 数组拼接成 (ns, n_keys) 的多列矩阵
                data = np.column_stack(arrays)
                selected_key = "+".join(key)
            else:
                selected_key = _select_npz_key(
                    keys=keys,
                    key=key,
                    default_candidates=default_candidates,
                )
                data = np.asarray(npz_file[selected_key]).copy()

        return data, selected_key

    raise ValueError(
        f"不支持的输入文件格式: '{suffix}'。\n"
        "仅支持 .npy 和 .npz 文件。"
    )


def npy_to_txt(
    npy_path: str | Path,
    txt_path: str | Path | None = None,
    key: str | list[str] | tuple[str, ...] | None = None,
    dtype: np.dtype | type = np.float32,
    fmt: str = "%.8e",
    delimiter: str = "\t",
    header: str | None = None,
    flatten_high_dim: bool = False,
    check_finite: bool = True,
) -> str:
    """
    将 NumPy 的 .npy 或 .npz 数据转换为文本 .txt 文件。

    参数
    ----
    npy_path : str or pathlib.Path
        输入的 .npy 或 .npz 文件路径。

    txt_path : str or pathlib.Path or None
        输出的 .txt 文件路径。
        如果为 None，则使用输入文件名，仅将后缀替换为 .txt。

    key : str or None
        当输入为 .npz 时，指定要提取的数组键。

        对于普通 .npy 文件，该参数没有作用，因为标准 .npy 文件
        只保存一个数组，不包含键值。

    dtype : numpy dtype or type
        写入文本前的数据类型。
        地球物理数据通常可使用 np.float32；
        若希望降低文本转换中的精度损失，可使用 np.float64。

    fmt : str
        np.savetxt 使用的数值格式。

        常用格式：
        "%.8e"   科学计数法，保留8位小数；
        "%.6f"   普通小数，保留6位；
        "%.10g"  自动选择小数或科学计数法。

    delimiter : str
        列分隔符。

        常用选择：
        "\\t"    制表符，推荐用于 .txt；
        " "      空格；
        ","      逗号，可生成类似 CSV 的内容。

    header : str or None
        写入文件首行的说明文字。
        如果为 None，则自动生成包含源文件、键、shape 和 dtype 的文件头。

    flatten_high_dim : bool
        是否允许处理三维及更高维数组。

        False：
            遇到 ndim > 2 时抛出错误，避免不明确地破坏数据维度。

        True：
            将数组重排为二维矩阵：
            (前面所有维度的乘积, 最后一维长度)。

        例如：
            原始 shape = (10, 20, 30)
            转换后 shape = (200, 30)

    check_finite : bool
        是否检查数据中的 NaN 和正负无穷值。
        若发现非有限值，只给出警告，不会自动修改。

    返回
    ----
    str
        输出 .txt 文件的完整路径。
    """
    input_path = Path(npy_path)

    if not input_path.is_file():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    if input_path.suffix.lower() not in {".npy", ".npz"}:
        raise ValueError(
            f"输入文件必须是 .npy 或 .npz，当前文件为: {input_path.name}"
        )

    print("=" * 70)
    print("开始执行 NumPy → TXT 数据转换")
    print(f"输入文件: {input_path}")

    # 1. 加载数组
    data, selected_key = _load_numpy_array(
        input_path=input_path,
        key=key,
    )

    print("成功加载数据:")
    if selected_key is not None:
        print(f"  提取键值  : {selected_key}")
    print(f"  原始 shape: {data.shape}")
    print(f"  原始 ndim : {data.ndim}")
    print(f"  原始 dtype: {data.dtype}")
    print(f"  元素数量  : {data.size}")

    # 2. 检查数组类型
    if data.dtype == object:
        raise TypeError(
            "当前数组是 object 类型，无法安全地直接保存为数值文本。\n"
            "请先检查源文件是否保存了字典、列表或其他 Python 对象。"
        )

    if not np.issubdtype(data.dtype, np.number):
        raise TypeError(
            f"只支持数值数组，当前数组 dtype 为: {data.dtype}"
        )

    # 3. 转换目标精度
    try:
        output_data = np.asarray(data, dtype=dtype)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"无法将数据转换为目标 dtype={dtype}。"
        ) from exc

    # 4. 处理数据维度
    original_shape = output_data.shape

    if output_data.ndim == 0:
        # 标量转换成一个元素的一维数组
        output_data = output_data.reshape(1)

    elif output_data.ndim == 1:
        # np.savetxt 会将一维数组写成单列
        pass

    elif output_data.ndim == 2:
        # 保持二维矩阵原有结构
        pass

    elif output_data.ndim > 2:
        if not flatten_high_dim:
            raise ValueError(
                f"当前数组维度为 {output_data.ndim}，shape={output_data.shape}。\n"
                "文本文件本身只能直接表达一维或二维数据。\n"
                "请先明确物理维度的排列方式，或者设置 "
                "flatten_high_dim=True 将数据重排为二维矩阵。"
            )

        last_dimension = output_data.shape[-1]
        output_data = output_data.reshape(-1, last_dimension)

        print(
            "  [警告] 高维数组已重排为二维矩阵:"
            f" {original_shape} → {output_data.shape}"
        )

    # 确保数据在内存中连续
    output_data = np.ascontiguousarray(output_data)

    # 5. 有限值诊断
    if check_finite:
        finite_mask = np.isfinite(output_data)
        non_finite_count = int(output_data.size - np.count_nonzero(finite_mask))

        if non_finite_count > 0:
            nan_count = int(np.count_nonzero(np.isnan(output_data)))
            inf_count = int(np.count_nonzero(np.isinf(output_data)))

            print("  [警告] 数据中存在非有限值:")
            print(f"    NaN 数量 : {nan_count}")
            print(f"    Inf 数量 : {inf_count}")
            print(f"    总计     : {non_finite_count}")
            print("  这些值将按照 NumPy 的文本形式直接写入文件。")
        else:
            print("  有限值检查: 通过，未发现 NaN 或 Inf。")

    # 6. 确定输出路径
    if txt_path is None:
        output_path = input_path.with_suffix(".txt")
    else:
        output_path = Path(txt_path)

        # 用户未写后缀时自动添加 .txt
        if output_path.suffix == "":
            output_path = output_path.with_suffix(".txt")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 7. 构造文件头
    if header is None:
        header_items = [
            f"source_file={input_path.name}",
            f"source_shape={original_shape}",
            f"saved_shape={output_data.shape}",
            f"saved_dtype={output_data.dtype}",
        ]

        if selected_key is not None:
            header_items.insert(1, f"selected_key={selected_key}")

        header = "; ".join(header_items)

    # 8. 保存文本
    np.savetxt(
        fname=output_path,
        X=output_data,
        fmt=fmt,
        delimiter=delimiter,
        header=header,
        comments="# ",
    )

    file_size_mb = output_path.stat().st_size / (1024 ** 2)

    print("成功导出 TXT 文件:")
    print(f"  输出路径  : {output_path}")
    print(f"  保存 shape: {output_data.shape}")
    print(f"  保存 dtype: {output_data.dtype}")
    print(f"  数值格式  : {fmt}")
    print(f"  分隔符    : {repr(delimiter)}")
    print(f"  文件大小  : {file_size_mb:.3f} MB")
    print("=" * 70)

    return str(output_path)


if __name__ == "__main__":
    # ==============================================================
    # 示例1：转换普通 .npy 文件
    # .npy 文件只有一个数组，不需要指定 key
    # ==============================================================

    test_numpy_path = Path(
        r"F:\01JDX_code\python_code\seismic_well_tying\true_datacode_student_t_final_CB323\_experiment_data\well_data\well_data_CB323.npy"
    )

    # ==============================================================
    # 示例2：转换包含多个键的 .npz 文件
    # ==============================================================

    # test_numpy_path = Path(
    #     r"F:\01JDX_code\python_code\seismic_well_tying"
    #     r"\true_datacode_student_t_final_CB323"
    #     r"\_experiment_data\seismic_CB323"
    #     r"\seismic_data_CB323_lowpass_70Hz.npz"
    # )

    if not test_numpy_path.is_file():
        print(f"未找到测试文件: {test_numpy_path}")

    else:
        try:
            print("发现测试文件，开始执行转换演示。")

            if test_numpy_path.suffix.lower() == ".npz":
                # 自动提取深度、声波速度和密度三列并合并导出为 3 列文本
                result_path = npy_to_txt(
                    npy_path=test_numpy_path,
                    key=["depth", "v_sonic", "rho"],
                    dtype=np.float32,
                    fmt="%.8e",
                    delimiter="\t",
                )

            else:
                # 对于 .npy，不需要指定 key
                result_path = npy_to_txt(
                    npy_path=test_numpy_path,
                    dtype=np.float32,
                    fmt="%.8e",
                    delimiter="\t",
                )

            print(f"转换完成: {result_path}")

        except Exception as exc:
            print(f"转换失败: {exc}")