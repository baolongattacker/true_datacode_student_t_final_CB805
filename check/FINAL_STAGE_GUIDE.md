# Student-t 时变子波反演最终阶段修改与运行指南

## 一、最终阶段目标

最终阶段不再修改 Student-t 损失函数本身，而完成四项收尾工作：

1. 统一所有模块导入路径，彻底隔离旧版 `code/`；
2. 加固第五阶段的断点续跑和统计推荐逻辑；
3. 建立不同噪声和模型误差下的最终压力测试；
4. 增加一键自检、依赖说明和完整回归测试。

最终研究链条为：

```text
L2 等价回归
→ Student-t 数学与冒烟测试
→ ν=10/5/3 脉冲异常统计筛选
→ 选择固定 ν
→ 异构噪声与模型误差压力测试
→ CB803 真实资料解释
```

## 二、文件修改清单

### 修改文件

```text
stages/stage_stationary.py
stages/stage_dtw.py
stages/stage_tv_wavelet.py
stages/stage_q_constraint.py
utils/dtw_timeoffset.py
configs/config_loader.py
experiments/run_synthetic_student_t_benchmark.py
experiments/run_synthetic_student_t_benchmark_from_config.py
analysis/analyze_synthetic_student_t_benchmark.py
configs/synthetic_student_t_phase5.yaml
plotting/plot_qc.py
utils/extract_laswelldata_CB803.py
```

### 新增文件

```text
analysis/analyze_final_student_t_stress_test.py
experiments/run_final_student_t_stress_test.py
experiments/run_final_self_check.py
configs/final_student_t_stress_test.yaml
tests/test_final_stage.py
requirements.txt
pytest.ini
CODE_LOGIC_REVIEW.md
FINAL_STAGE_GUIDE.md
```

### 移入历史目录

```text
legacy/stage_core_draft.py
legacy/wavelet_inversion_robust_raw.py
```

## 三、第一步：安装并检查环境

在 `true_datacode` 根目录打开 PowerShell：

```powershell
python -m pip install -r requirements.txt
```

真实 DTW 主流程必须安装：

```powershell
python -m pip install dtaidistance
```

运行最终自检：

```powershell
python experiments/run_final_self_check.py --run-tests
```

准备运行真实 DTW 时使用严格模式：

```powershell
python experiments/run_final_self_check.py --run-tests --require-dtw
```

输出 `passed=true` 才继续。

## 四、第二步：确认 L2 路径没有改变

> 发布包中的实验结果目录保持为空，以避免旧参数结果污染新实验。运行下面的 CB803 回归和冒烟测试前，请把你原项目中的 `_experiment_results/cb803_center_final_smooth_v02/result_bundle.npz` 复制到相同相对路径，或在命令中传入其绝对路径。

```powershell
python tests/check_tv_l2_regression.py `
  --bundle "_experiment_results/cb803_center_final_smooth_v02/result_bundle.npz" `
  --baseline "baseline/tv_l2_baseline_uploaded.npz"
```

必须满足：

```text
relative_W_error < 1e-8
valid_mask_equal = True
skip_code_equal  = True
```

最终包实测为零误差。

## 五、第三步：确认 CB803 Student-t 最小版本正常

```powershell
python tests/check_student_t_cb803_smoke.py `
  --bundle "_experiment_results/cb803_center_final_smooth_v02/result_bundle.npz"
```

程序应自动识别：

```text
prior_source = center_ricker_prior_after_DTW
mu_prior = 0.8
```

并且：

- `robust_attempted > 0`；
- `robust_solution_used > 0`；
- 不出现 NaN/Inf；
- 有效样本比例合理。

## 六、第四步：完整运行第五阶段固定自由度筛选

配置：

```text
configs/synthetic_student_t_phase5.yaml
```

运行：

```powershell
python experiments/run_synthetic_student_t_benchmark_from_config.py `
  --config configs/synthetic_student_t_phase5.yaml
```

完整网格包含：

```text
5 个随机种子
1 个干净场景
3 个异常比例
3 个异常幅值
L2 + Student-t ν=10/5/3
```

总反演次数：

```text
200
```

中断后执行相同命令即可续跑。以下任一内容变化后，旧结果会被签名拒绝：

- 真子波模型；
- 反射系数模型；
- 正则参数；
- IRLS 参数；
- 异常网格；
- 数值实现源码。

输出重点文件：

```text
synthetic_benchmark_raw.csv
synthetic_benchmark_summary_ci.csv
synthetic_paired_comparison.csv
student_t_fixed_nu_recommendation.json
student_t_fixed_nu_recommendation.md
student_t_fixed_nu_recommendation.png
```

固定自由度必须同时满足：

1. 干净数据 NRMSE 非劣；
2. 污染场景平均改善为正；
3. 污染场景胜率不低于 60%；
4. 全局权重覆盖率不低于 80%；
5. 全局低权重误报率不高于 10%。

## 七、第五步：运行最终异构误差压力测试

配置：

```text
configs/final_student_t_stress_test.yaml
```

先运行快速模式：

```powershell
python experiments/run_final_student_t_stress_test.py `
  --config configs/final_student_t_stress_test.yaml `
  --quick `
  --overwrite
```

快速模式只用于确认流程闭环。

正式运行：

```powershell
python experiments/run_final_student_t_stress_test.py `
  --config configs/final_student_t_stress_test.yaml
```

如果第五阶段推荐文件存在，程序自动读取：

```text
recommended_student_nu
```

否则使用配置中的：

```yaml
student_t:
  nus: [10.0]
```

### 最终测试场景

#### 1. `clean_gaussian`

只含高斯噪声，检验 Student-t 是否在干净数据上明显退化。

#### 2. `impulse_noise`

5% 脉冲异常、异常幅值约为 10 倍噪声标准差。

#### 3. `ar1_correlated_noise`

使用 AR(1) 相关噪声，默认相关系数 0.85。

#### 4. `coherent_burst`

在局部时间段加入带窗正弦相干干扰。

#### 5. `reflectivity_amplitude_error`

对部分反射系数施加随机振幅误差。

#### 6. `reflectivity_missing_events`

从反演使用的反射系数中删除部分真实反射事件。

#### 7. `reflectivity_time_jitter`

将部分反射事件随机移动 1～2 个采样点。

#### 8. `prior_frequency_high`

将平稳先验的频率整体提高 25%，检验先验失配。

#### 9. `combined_error`

组合相关噪声、脉冲异常、反射事件缺失和先验频率失配。

## 八、最终压力测试的评价指标

### 1. 子波绝对 NRMSE

```text
wavelet_nrmse_absolute
```

直接比较估计子波矩阵和真子波矩阵。

### 2. 消除单一全局振幅差后的 NRMSE

```text
wavelet_nrmse_scaled
```

这是最终压力测试的主指标。

### 3. 真反射系数下的干净记录误差

```text
clean_trace_nrmse_true_r
clean_trace_cc_true_r
```

该指标用于避免错误反射系数被子波吸收后，表面上拟合很好但真子波变差。

### 4. 使用反演反射系数的观测拟合

```text
observed_fit_cc_inversion_r
```

它只表示数据拟合，不单独作为真子波恢复优劣依据。

### 5. 权重异常定位

对局部异常和局部模型误差统计：

```text
weight_global_precision
weight_global_recall
weight_global_f1
weight_global_false_positive_rate
weight_global_coverage_ratio
```

对全局先验失配不强行定义异常位置，因此权重定位指标留空。

## 九、最终通过条件

最终报告：

```text
final_student_t_validation.json
final_student_t_validation.md
```

固定 Student-t 方法需要同时满足：

```text
干净数据 NRMSE 退化上界 <= 0.005
压力场景平均 NRMSE 改善 >= 0
压力场景配对胜率 >= 60%
最差单场景平均退化 <= 0.01
有效窗口比例最大下降 <= 5 个百分点
```

注意：快速模式样本太少，只能验证程序，不能作为论文统计结论。

## 十、真实资料最终运行

第五阶段与最终压力测试确认固定自由度后，修改：

```text
configs/cb803_center_smooth_student_t.yaml
```

例如：

```yaml
tv_wavelet:
  loss_type: "student_t"
  student_nu: 10.0
  irls_max_iter: 10
  irls_tol: 1.0e-4
  robust_scale_mode: "local_mad"
  robust_scale_floor_ratio: 0.05
  robust_weight_floor: 0.001
  min_effective_sample_ratio: 0.30
  robust_fallback: "l2"
  robust_outlier_weight_threshold: 0.5
  store_weight_map: true
  weight_map_row_filter: "all"
```

运行：

```powershell
python experiments/run_cb803_experiment_smooth.py `
  --config configs/cb803_center_smooth_student_t.yaml
```

## 十一、真实资料结果不能只看相关系数

必须联合检查：

```text
CC_tv_direct
CC_final
valid_ratio
peak_abs_p90_ms
峰值拒绝窗口数
振幅拒绝窗口数
robust_solution_used
robust_fallback_to_l2
median effective sample ratio
权重热图
局部残差位置
```

Student-t 可能降低对含异常观测道的直接 L2 拟合，但提高子波物理稳定性。论文中应使用合成真值实验作为“恢复更准确”的主要证据，真实资料用于解释其稳健行为和 QC 改善。

## 十二、最终阶段之后不建议继续增加的功能

在没有更多真实井和更多合成随机种子之前，不建议立即加入：

- 每窗口自动估计 `nu_k`；
- 每轮联合更新 `sigma_k`；
- 允许 `robust_fallback=prior`；
- 根据权重自动修改正则强度；
- 把加权秩直接设为硬拒绝条件。

这些功能会增加自由度，使论文难以证明改进来自 Student-t 数据项本身。
