# Student-t 时变子波程序最终逻辑审查

## 审查结论

当前 Student-t 核心反演数学逻辑是成立的：

- 每个有效窗口先求原始 L2 解；
- 用 L2 残差的 MAD 估计固定局部尺度；
- 权重采用 `nu / (nu + (e / sigma)^2)`；
- `sigma` 不直接除入 `R` 和 `s_win`；
- 加权数据块使用 `sqrt(weight) * R` 与 `sqrt(weight) * s_win`；
- 正则块继续保留原 `data_scale` 缩放；
- 完整目标函数的数据项与正则项统一采用 `1/2` 约定；
- IRLS 不下降时保留历史最佳解，彻底失败时回退当前窗口的 L2；
- 原 `skip_code` 与 Student-t `robust_code` 分离；
- 原峰值、极性、振幅和后处理 QC 保持在 IRLS 之后。

## 最终审查发现并修复的问题

### 1. 旧版 `code/utils` 仍可能抢占导入

原来的 `stage_stationary.py`、`stage_dtw.py` 和 `stage_q_constraint.py` 会把父目录的 `code/` 插入 `sys.path` 首位。这样 DTW 和平稳子波可能使用旧求解器，而时变子波使用新求解器，实验不可追溯。

最终修复：所有阶段统一只从当前 `true_datacode/utils` 导入，不再插入旧 `code/`。

### 2. `stage_tv_wavelet.py` 依赖目录必须叫 `true_datacode`

原代码使用 `from true_datacode.utils...`，如果文件夹重命名或解压到其他目录会失败。

最终修复：先将当前项目根目录放入 `sys.path`，再使用 `from utils.wavelet_inversion_robust import ...`。导入不再依赖文件夹名称。

### 3. `stages/core.py` 是早期草稿，可能遮蔽正式 `core/` 包

直接运行 `stages` 下脚本时，Python 可能把 `stages/core.py` 当作 `core`，导致 `core.forward_operator` 导入失败。

最终修复：移动到 `legacy/stage_core_draft.py`，不再参与运行。

### 4. 旧版 `wavelet_inversion_robust_raw.py` 容易被误改或误导入

最终修复：移动到 `legacy/`，仅用于历史比较。

### 5. 没装 `dtaidistance` 时整个主程序无法导入

原代码在模块顶层直接导入 `dtaidistance`，即使只运行 Student-t 合成实验也会被 DTW 依赖阻断。

最终修复：改为延迟检查。非 DTW 模块可正常导入；真正执行 DTW 时若依赖缺失，会明确提示安装命令。

### 6. PyYAML 备用解析器不能解析现有 DTW 列表配置

现有 YAML 包含 `phase_specs` 块列表，旧备用解析器并不支持，却可能让用户误以为无需 PyYAML。

最终修复：明确要求 PyYAML，并提供清晰错误信息。`requirements.txt` 已列出该依赖。

### 7. 第五阶段断点续跑签名不完整

原签名只包含数据长度、异常比例和自由度，没有包含：

- `mu1/mu2/mu_prior/mu_time`；
- IRLS 参数；
- 真子波频率变化；
- 反射系数生成参数；
- 数值实现代码版本。

修改这些内容后可能错误续跑旧结果。

最终修复：签名加入完整模型参数、反演参数和实现文件 SHA-256 指纹。

### 8. 固定自由度推荐在 FPR 缺失时默认通过

原逻辑为 `global_fpr is None or global_fpr <= threshold`，缺少权重诊断反而会通过。

最终修复：缺失即失败，并新增全局权重覆盖率门槛。

### 9. 原始统计表的重复案例可能被静默覆盖

配对统计使用字典索引，重复的 seed/scenario/method 可能覆盖旧行。

最终修复：分析前强制检测重复 `case_id` 和重复配对方法，发现即终止。

### 10. `--overwrite` 后旧报告可能残留

原代码只删除原始 CSV 和 manifest；若新实验中途停止，旧推荐图和旧 completion 文件仍可能存在。

最终修复：覆盖运行前只清理该实验明确生成的全部输出。

### 11. 原实验只验证脉冲异常，尚不足以作为最终结论

最终新增异构误差压力测试，覆盖：

- 干净高斯噪声；
- 脉冲异常；
- AR(1) 相关噪声；
- 局部相干正弦干扰；
- 反射系数振幅误差；
- 反射事件缺失；
- 反射事件时间抖动；
- 平稳先验频率失配；
- 多误差组合。

## 未发现的核心错误

审查没有发现以下方面存在原则性错误：

- Student-t 权重公式与所定义损失不一致；
- `sigma` 被重复除入数据矩阵；
- 正则项 `data_scale` 被意外删除；
- Student-t 状态码破坏原 `skip_code`；
- IRLS 内错误进行极性翻转或峰值平移；
- L2 模式因新代码发生数值改变。

最终 L2 回归结果：

```text
relative_W_error = 0.000e+00
max_abs_W_error  = 0.000e+00
valid_mask_equal = True
skip_code_equal  = True
```
