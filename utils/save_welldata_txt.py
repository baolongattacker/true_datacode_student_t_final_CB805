import numpy as np
import os

# 1. 定义路径
npz_path = r"F:\01JDX_code\python_code\seismic_well_tying\true_datacode_student_t_final_CB323\_experiment_data\initial_model_data\initial_real_model_data_CB323.npz"
output_txt_path = r"F:\01JDX_code\python_code\seismic_well_tying\true_datacode_student_t_final_CB323\_experiment_data\well_data\welldata323.txt"

# 确认输入文件存在
if not os.path.exists(npz_path):
    raise FileNotFoundError(f"找不到预处理输出的 NPZ 文件: {npz_path}")

# 2. 读取校正后的数据
data = np.load(npz_path)

depth = data["depth"]                       # 深度曲线 (m)
ac_corrected = data["ac_tdr_corrected"]     # TDR 校正后的时差 (us/ft)
rho_corrected = data["rho_model_used"]       # TDR 校正后的密度 (kg/m3)

# 检查数据合法性
assert len(depth) == len(ac_corrected) == len(rho_corrected), "数据曲线长度不一致"

# 3. 按列堆叠
# 第一列：深度 (m)，第二列：声波时差 (us/ft)，第三列：密度 (kg/m3)
well_data = np.column_stack((depth, ac_corrected, rho_corrected))

# 4. 写入带列名表头的 txt 文件 (使用科学记数法或浮点数格式保存)
header_info = "DEPTH(m)    AC_TDR_CORRECTED(us/ft)    RHO_TDR_CORRECTED(kg/m3)"
np.savetxt(
    output_txt_path,
    well_data,
    fmt=["%.4f", "%.6f", "%.6f"],
    header=header_info,
    comments="# "
)

print(f"提取并保存成功！文本文件路径: {output_txt_path}")
print(f"数据总行数: {len(well_data)}")
