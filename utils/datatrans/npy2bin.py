# -*- coding: utf-8 -*-
"""
数据格式转换工具：将 numpy 的 .npy 文件转换为纯二进制 .bin 字节流。

本工具遵循地球物理数据处理规范，将带维度的 numpy 矩阵转换为可供 C/C++/Fortran/MATLAB 读取的 C 排列二进制文件。
"""

from pathlib import Path
import numpy as np


def npy_to_binary(
    npy_path: str | Path,
    bin_path: str | Path | None = None,
    dtype: np.dtype = np.float32,
    key: str | None = None,
) -> str:
    """
    将 numpy 的 .npy 或 .npz 数据文件转换为纯二进制 .bin 数据文件。

    输入参数:
    ----------
    npy_path : str or pathlib.Path
        输入的 .npy 或 .npz 文件路径。
    bin_path : str or pathlib.Path or None
        输出的二进制 .bin 文件路径。如果为 None，则默认使用与输入文件相同的路径及文件名，仅将后缀替换为 .bin。
    dtype : numpy.dtype
        转存二进制时的目标数据精度类型。在地球物理计算中，通常统一为 np.float32。
    key : str or None
        如果输入是 .npz 格式，此参数指定要读取并转换的数组键名。
        如果为 None 且 .npz 中包含多个键，将尝试匹配默认候选键 ("sel_data", "sel_data_norm" 等)，匹配失败则抛出错误。

    输出结果:
    -------
    str
        保存成功的二进制文件路径。

    物理作用与数学假设:
    ------------------
    本函数将带有表头、维度或打包键值信息的 numpy 专属数据文件，剥离元数据后转化为按行排布的纯二进制数据流。
    这对于跨语言（如 Fortran, C++, C, MATLAB）加载原始地震道或测井数据非常关键。
    """
    npy_path = Path(npy_path)
    if not npy_path.is_file():
        raise FileNotFoundError(f"输入的 numpy 文件不存在: {npy_path}")

    # 1. 加载 numpy 数据
    loaded = np.load(npy_path)
    print(f"成功加载文件: {npy_path.name}")

    # 2. 兼容处理 .npz 格式
    if npy_path.suffix.lower() == ".npz" or isinstance(loaded, np.lib.npyio.NpzFile):
        keys = list(loaded.keys())
        if not keys:
            raise ValueError(f".npz 文件中没有包含任何数据键: {npy_path}")
        
        if key is not None:
            if key not in keys:
                raise KeyError(f"指定的键 '{key}' 不在 .npz 文件的可用键中。可用键: {keys}")
            selected_key = key
        else:
            if len(keys) == 1:
                selected_key = keys[0]
            else:
                default_candidates = ["sel_data", "sel_data_norm", "data", "seismic_trace", "seismic"]
                matched_key = None
                for cand in default_candidates:
                    if cand in keys:
                        matched_key = cand
                        break
                if matched_key is not None:
                    selected_key = matched_key
                    print(f"  [提示] 未指定 key，自动匹配推荐键: '{selected_key}'")
                else:
                    raise KeyError(
                        f".npz 文件中包含多个数据键 {keys}，且未发现默认推荐键。请显式指定 key 参数。"
                    )
        data = loaded[selected_key]
        print(f"  已从 .npz 提取键 '{selected_key}' 的数据。")
    else:
        # 直接是 .npy 文件
        data = loaded

    # 打印 Shape 和 Dtype 诊断信息
    print(f"  原始 Shape: {data.shape}")
    print(f"  原始 dtype: {data.dtype}")

    # 3. 确定保存路径
    if bin_path is None:
        bin_path = npy_path.with_suffix(".bin")
    else:
        bin_path = Path(bin_path)

    bin_path.parent.mkdir(parents=True, exist_ok=True)

    # 4. 转换精度并以二进制格式写入文件
    flat_data = np.ascontiguousarray(data, dtype=dtype)
    # Shape: 与 data 相同，仅在内存中转为 C-contiguous 并完成数据类型强制转换
    
    flat_data.tofile(bin_file_str := str(bin_path))

    print(f"成功导出二进制文件:")
    print(f"  目标路径  : {bin_file_str}")
    print(f"  目标 dtype: {dtype}")
    print(f"  文件大小  : {bin_path.stat().st_size / 1024:.2f} KB")

    return bin_file_str


# 独立运行脚本测试
if __name__ == "__main__":
    # 示例测试路径 (这里以 CB323 测井清洗后的 .npy 文件为例)
    test_npy_path = r"F:\01JDX_code\python_code\seismic_well_tying\true_datacode_student_t_final_CB323\_experiment_data\seismic_CB323\seismic_data_CB323_lowpass_70Hz.npz"
    # test_npy_path = r"F:\01JDX_code\python_code\seismic_well_tying\true_datacode_student_t_final_CB323\_experiment_data\well_data\well_data_CB323.npy" 
    test_path = Path(test_npy_path)
    if test_path.is_file():
        print("发现测试 .npy 文件，开始执行转换演示...")
        try:
            # 通过传入 key="trace_raw" 转换原始地震道，或者传入 key="trace_lowpass" 转换滤波地震道
            npy_to_binary(npy_path=test_path, dtype=np.float32, key="trace_lowpass")
        except Exception as e:
            print(f"转换演示失败: {e}")
    else:
        print(f"未在预期路径找到测试文件: {test_path}，已跳过转换测试演示。")
