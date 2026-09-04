# Student-t 最终异构误差压力测试

干净数据非劣、压力场景平均改善与胜率达标、最差场景退化受控，且有效窗口比例不得明显下降。

**最终通过：** 是
**推荐方法：** student_t_nu10

## 候选方法

| 方法 | 干净退化上界 | 压力场景平均改善 | 压力胜率 | 最差场景改善 | 最大有效率下降 | 通过 |
|---|---:|---:|---:|---:|---:|:---:|
| student_t_nu10 | -0.00184 | 0.00903 | 87.50% | -0.00033 | -0.00% | 是 |

## 逐场景缩放后子波 NRMSE 配对改善

| 场景 | 方法 | 平均改善 | 95%区间 | 胜率 |
|---|---|---:|---:|---:|
| ar1_correlated_noise | student_t_nu10 | 0.00110 | [0.00110, 0.00110] | 100.0% |
| clean_gaussian | student_t_nu10 | 0.00184 | [0.00184, 0.00184] | 100.0% |
| coherent_burst | student_t_nu10 | 0.00348 | [0.00348, 0.00348] | 100.0% |
| combined_error | student_t_nu10 | 0.03888 | [0.03888, 0.03888] | 100.0% |
| impulse_noise | student_t_nu10 | -0.00033 | [-0.00033, -0.00033] | 0.0% |
| prior_frequency_high | student_t_nu10 | 0.00152 | [0.00152, 0.00152] | 100.0% |
| reflectivity_amplitude_error | student_t_nu10 | 0.00429 | [0.00429, 0.00429] | 100.0% |
| reflectivity_missing_events | student_t_nu10 | 0.01807 | [0.01807, 0.01807] | 100.0% |
| reflectivity_time_jitter | student_t_nu10 | 0.00525 | [0.00525, 0.00525] | 100.0% |