# Student-t 时变子波程序最终验证报告

验证日期：2026-07-23

## 1. 已完成的代码级验证

### 1.1 全项目编译

```text
python -m compileall -q .
```

结果：通过，无语法错误。

### 1.2 单元测试

```text
11 passed, 8 subtests passed
```

覆盖内容包括：

- Student-t 权重、MAD 尺度和 L2 极限；
- L2 重构等价性；
- 权重全局聚合与统计口径；
- 配对比较和固定自由度推荐；
- 最终压力测试场景生成与结果分析；
- 重复案例、断点续跑和参数签名保护。

### 1.3 导入路径自检

以下模块均确认来自当前发布包内部，不包含父目录旧版 `code/`：

```text
utils.wavelet_inversion_robust
stages.stage_stationary
stages.stage_dtw
stages.stage_tv_wavelet
stages.stage_q_constraint
experiments.run_real_experiment
experiments.run_synthetic_student_t_benchmark
experiments.run_final_student_t_stress_test
```

### 1.4 L2 数值回归

核心 L2 求解路径验证结果：

```text
relative_W_error = 0.000e+00
max_abs_W_error  = 0.000e+00
valid_mask_equal = True
skip_code_equal  = True
```

该测试验证的是：加入 Student-t 接口、诊断和实验代码后，`loss_type="l2"` 仍保持原增广最小二乘数值路径。

### 1.5 CB803 Student-t 冒烟测试

使用现有 CB803 固定输入结果包验证：

```text
prior_source          = center_ricker_prior_after_DTW
mu_prior              = 0.8
robust_attempted      = 276
robust_solution_used  = 276
robust_converged      = 151
valid_windows         = 234
median_weight_mean    = 0.916438
median_effective_ratio = 0.990563
```

结果无 NaN/Inf，目标函数不增，原物理 QC 正常执行。

## 2. 已完成的实验级快速验证

### 2.1 第五阶段固定自由度筛选快速验证

快速网格共运行 16 个严格配对案例；第二次运行全部正确跳过，无重复行。快速结果推荐：

```text
student_nu = 10
```

该结果只证明统计和断点流程闭环，不能替代完整 200 次实验。

### 2.2 最终异构压力测试快速验证

已完成：

- 快速子集测试：20 个案例；
- 九类场景单随机种子测试：18 个案例；
- 第二次运行全部断点跳过。

九类场景单种子测试中，`nu=10` 通过当前工程门槛，但单随机种子不能作为正式统计结论。

## 3. 当前环境未完整执行的内容

### 3.1 完整第五阶段网格

正式配置需要运行 200 次反演。当前交付前只完成快速闭环验证，没有在本环境等待完整 200 次网格结束。

### 3.2 完整最终压力测试

正式配置为 5 个随机种子 × 9 个场景 × 2 个方法，共 90 次反演。当前完成了全部场景的单随机种子运行和快速子集，而非完整 90 次统计。

### 3.3 真实数据完整 DTW 主流程

当前验证环境未安装 `dtaidistance`，因此没有重新执行完整真实数据 DTW 主流程。已经完成以下保护：

- 非 DTW 模块可正常导入；
- 真正执行 DTW 时会给出明确安装提示；
- `requirements.txt` 已列出依赖；
- 安装后可用 `--require-dtw` 进行严格环境自检。

这不影响 Student-t 核心反演、合成实验和统计脚本的验证，但用户本机运行真实资料前必须安装该依赖。

## 4. 最终结论边界

目前能够可靠确认：

1. Student-t 数学实现与所定义损失一致；
2. L2 模式没有因改造发生数值变化；
3. 导入链已隔离旧模块；
4. 断点续跑、配置签名和统计推荐逻辑已加固；
5. 异构误差压力测试程序能够完整运行并生成配对统计。

仍需由正式完整实验决定：

- 最终固定自由度是否稳定为 `nu=10`；
- 不同随机种子下各类模型误差的置信区间；
- CB803 真实资料上的最终解释和论文结论。
