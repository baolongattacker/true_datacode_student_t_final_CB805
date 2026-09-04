import os
import numpy as np
import segyio
from scipy.io import savemat

def sgy_to_mat(sgy_filepath: str, mat_filepath: str) -> None:
    """
    将 SEG-Y 格式的地震数据读取并转换为 MATLAB 的 .mat 格式。

    输入参数:
    - sgy_filepath (str): 输入 SEG-Y 文件的绝对或相对路径
    - mat_filepath (str): 输出 .mat 文件的绝对或相对路径

    输出结果:
    - 无直接返回值。结果保存至指定的 .mat 文件中。
      .mat 文件包含：
      1. seis_matrix: 地震数据二维矩阵 (Shape: 采样点数 x 水平道数)
      2. t_ms: 采样时间向量，单位：毫秒 (ms) (Shape: 采样点数,)
      3. dt_ms: 采样间隔，单位：毫秒 (ms)

    数学作用与物理假设:
    - 物理意义：将记录地下反射界面的地震波形数据从工业标准 SEG-Y 格式提取为通用矩阵，方便后续进行地震井震标定或反演。
    - 忽略 SEG-Y 中的 3D 几何结构头信息（ignore_geometry=True），将其视为连续的 2D 剖面或道集提取所有地震道。
    """
    
    # 强制输入检查
    assert os.path.exists(sgy_filepath), f"错误：输入文件不存在，请检查路径: {sgy_filepath}"
    
    print(f"开始读取 SEG-Y 数据: {sgy_filepath}")
    
    with segyio.open(sgy_filepath, "r", ignore_geometry=True) as f:
        # 获取基础信息
        num_traces = f.tracecount
        num_samples = len(f.samples)
        dt_us = f.bin[segyio.BinField.Interval]  # 采样间隔，单位：微秒 (microseconds)
        dt_ms = dt_us / 1000.0  # 采样间隔，单位：毫秒 (ms)
        
        print(f"文件信息: 共有 {num_traces} 道, 每道 {num_samples} 采样点, 采样间隔 {dt_ms} ms")
        
        # 提取时间轴（twt_ms，双程走时），单位：ms
        # shape: (采样点数,)
        t_ms = f.samples
        
        # 提取地震数据，原始读取出来的 shape: (水平道数, 深度采样点数)
        raw_traces_matrix = f.trace.raw[:]
        
        # 检查维度
        assert raw_traces_matrix.shape == (num_traces, num_samples), \
            f"数据维度不匹配: 期望 ({num_traces}, {num_samples}), 实际 {raw_traces_matrix.shape}"
            
        # 转置，使得符合物理习惯：行代表深度/时间采样，列代表不同位置的地震道
        # shape 变为: (深度采样点数, 水平道数)
        seis_matrix = raw_traces_matrix.T 
        
    print(f"提取地震数据矩阵完成, 当前维度 Shape: {seis_matrix.shape}")
    
    # 构造待写入 .mat 文件的字典
    mat_dict = {
        'seis_matrix': seis_matrix,  # 地震波形数据矩阵
        't_ms': t_ms,                # 对应的时间采样点
        'dt_ms': dt_ms               # 采样间隔
    }
    
    # 保存 .mat 文件
    print(f"正在保存为 .mat 格式: {mat_filepath}")
    savemat(mat_filepath, mat_dict)
    print("转换完成！")


if __name__ == "__main__":
    # 请根据实际情况修改下方路径
    input_sgy_file = r"example_input.sgy"
    output_mat_file = r"example_output.mat"
    
    if os.path.exists(input_sgy_file):
        sgy_to_mat(input_sgy_file, output_mat_file)
    else:
        print(f"提示: 请将 input_sgy_file 修改为实际存在的 SEG-Y 文件路径。")
