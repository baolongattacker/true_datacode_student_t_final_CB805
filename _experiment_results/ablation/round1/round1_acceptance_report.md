# CB805 正则项消融第一轮验收报告

**验收结论：A00–A06 共 7 组完整真实数据流程运行通过。DC、mu2 通过本轮单项非劣初筛；edge、mu_time、prior 当前应保留；mu1 暂缓删除。尚无任何项完成联合删除和 2×2 去耦合验收。**

## 1. 实际运行与冻结核验

- 运行日期：2026-09-07。开始 21:31:47 +0800，矩阵结束 21:33:23 +0800，约 1.59 分钟；之后独立核验保存结果。
- 仓库提交：`9bb8edec0e766b8a13c47db4d8ee3a44765927f4`。直接使用用户提供的 `configs/ablation/A00–A06`，未改任何源码、配置或输入，未安装依赖。
- 解释器：`D:\miniconda\envs\myenv\python.exe`，Python 3.12.3；NumPy 2.3.1、SciPy 1.16.2、Matplotlib 3.10.1、PyYAML 6.0.3、dtaidistance 2.4.0。核心导入路径全部来自本仓库。
- 7 组均使用单线程 BLAS/OMP，`MPLBACKEND=Agg` 保存原有绘图；结果目录均为本轮新建，无覆盖旧实验。
- 输入 SHA256：`ee82c0303c5fab67c7c60fff290b13622e501355d967862fb645ab17668948e4`。工作时窗 0.580–3.125 s，采样间隔 1 ms，观测/反射系数 shape=(2546,)；子波配置 length_s=0.129，实际矩阵 shape=(2546,129)，center lag 为 −64…64 ms。
- A00 与 `configs/cb805_center_student_t_L129_v01.yaml` 除实验名称、输出目录外完全一致。A01–A06 与 A00 的差异严格为各自关闭项及名称/目录；A06 的 strict/fallback prior 权重均为 0。
- 冻结 TV：Student-t ν=10、IRLS max_iter=12/tol=3e-4、mu1=.20、mu2=6、mu_dc=15、prior=1/.8、mu_time=8、mu_edge=1，estimate_step=20 ms，time Gaussian=30 ms，lag Gaussian sigma=4 samples。当前 constant-phase 来源实际选择 strict prior 权重 1.0。
- DTW history 及 16 项上游数组在 7 组之间逐元素一致，包括 DTW 后反射系数、时深关系、平稳先验和相位搜索结果。`CC_after_DTW=0.155402646229`，`CC_prior_after_DTW=0.157126228507`。
- 使用当前提交的 gray-zone support、peak/amplitude QC、local fallback 与 alpha smoothing；本轮没有额外调整这些机制。

## 2. 主表：所有实验独立相对 A00

| 实验 | 关闭项 | raw TV CC | CC_tv_direct = CC_final | ΔCC_final | 有效窗口 | 本轮筛选结论 |
|---|---|---:|---:|---:|---:|---|
| A00 | 无 | 0.435505 | 0.287982 | +0.000000 | 70/108 | 基准 |
| A01 | edge | 0.357692 | 0.287626 | -0.000356 | 65/108 | 不支持删除 |
| A02 | DC | 0.435644 | 0.287986 | +0.000004 | 70/108 | 通过单项初筛 |
| A03 | mu1 | 0.438866 | 0.301852 | +0.013870 | 70/108 | 暂缓，区分幅度效应 |
| A04 | mu2 | 0.435592 | 0.288467 | +0.000485 | 70/108 | 通过单项初筛，待去耦合 |
| A05 | mu_time | 0.449957 | 0.279837 | -0.008145 | 67/108 | 保留 |
| A06 | prior（两权重） | 0.439597 | 0.221599 | -0.066383 | 65/108 | 保留 |

全部 `W_pass=True`，最终模型均为 `W_est_best_no_Q`，`acceptance_reasons=[]`。但全部 `raw_tv_W_pass=False`，raw TV 的拒绝原因都是峰值 P90 偏移与峰值越界比例过大。**主流程通过与无回退 TV 通过物理 QC 是两种结论。**

本轮采用用户建议中的 ΔCC≥−0.003 作宽松界，−0.002 作更严参考；两者下结论相同。有效窗口比例不降低、W_pass 不退化、峰值/时移不增加，非时间消融的原始时间差分能量以 +15% 为上限（+10% 为关注线）。这是同输入确定性点值筛选，未进行统计非劣检验或噪声重复。

## 3. 物理形态与峰值

| 实验 | edge P90 候选 / 最终 | side-lobe P90 候选 / 最终 | 候选边界峰比例 | 候选 / 最终 peak abs P90 (ms) | 最终形态可靠比例 |
|---|---:|---:|---:|---:|---:|
| A00 | 0.232628 / 0.127633 | 1.286572 / 0.801889 | 19.95% | 64 / 1 | 86.10% |
| A01 | 0.246120 / 0.170074 | 1.081696 / 0.798234 | 0.00% | 23 / 1 | 81.11% |
| A02 | 0.232276 / 0.127572 | 1.282098 / 0.802225 | 19.99% | 64 / 1 | 86.10% |
| A03 | 0.236617 / 0.123970 | 1.259085 / 0.791063 | 19.87% | 64 / 1 | 86.88% |
| A04 | 0.230370 / 0.127256 | 1.253431 / 0.799959 | 19.68% | 64 / 1 | 86.17% |
| A05 | 0.232628 / 0.137924 | 1.286572 / 0.797154 | 19.05% | 64 / 1 | 86.49% |
| A06 | 0.259131 / 0.101117 | 1.201430 / 0.811051 | 0.00% | 24 / 1 | 86.21% |

所有组最终 boundary peak fraction=0，最终 best lag=0 ms；candidate best lag=−1 ms。可靠 / 严重回退阈值分别是 edge 0.15 / 0.30、side-lobe 0.75 / 0.95。A00 本身的最终 side-lobe P90=0.801889 已处于灰区，因此不能把所有最终样点称为形态可靠。A00 最终 13.90% 样点超过 side-lobe 可靠阈值，但未超过严重阈值。

`valid_ratio` 分母为 108 个反演候选中心。`candidate_reliable_ratio` 是稀疏可靠中心占全部 2546 个时间样点的比例，不能直接与连续最终子波的 `final_reliable_ratio` 比大小。A00 有 70 个有效中心，其中 strict=42、gray=20、admissible=62；严格保留当前分母定义。

## 4. 时间差分能量与 fallback

| 实验 | E_t 候选 | E_t 最终 | 候选 E_t 变化 | 最终 E_t 变化 | 归一化 E_t 变化 候选 / 最终 | hard-prior 掩码比例 |
|---|---:|---:|---:|---:|---:|---:|
| A00 | 7.20721585e-05 | 3.61947467e-03 | +0.000% | +0.000% | +0.000% / +0.000% | 52.00% |
| A01 | 5.92974265e-05 | 4.35229495e-03 | -17.725% | +20.247% | -32.196% / +16.132% | 52.79% |
| A02 | 7.20719002e-05 | 3.62443058e-03 | -0.000% | +0.137% | -0.021% / +0.132% | 52.00% |
| A03 | 8.29010330e-05 | 4.24838915e-03 | +15.025% | +17.376% | -7.358% / +1.417% | 52.00% |
| A04 | 7.24980513e-05 | 3.64024125e-03 | +0.591% | +0.574% | -0.158% / +0.132% | 52.00% |
| A05 | 1.63292726e-04 | 3.78515669e-03 | +126.568% | +4.578% | +103.831% / +0.054% | 52.79% |
| A06 | 1.48835018e-04 | 1.04306975e-02 | +106.508% | +188.183% | -13.523% / +117.257% | 75.57% |

仓库定义 `E_t = mean_i(sum_lag((W[i+1]-W[i])**2))`，未除以 dt²，单位沿用子波相对振幅平方。补充归一化诊断为 `E_t / mean_i(sum_lag(W[i]**2))`，用于区分幅度变化与相对时间变化，**没有用它替换原先的 +15% 筛选门槛**。hard-prior 掩码表示 hybrid 混合前的分区，边界经 alpha smoothing 后可能包含混合；它不等于最终纯先验来源编码的比例。

指标口径必须区分：

1. `candidate_*` 和 `raw_tv_*` 均来自已经完成稀疏填充、2-D Gaussian、去均值的 TV；这里 raw 的含义是 hybrid/fallback 之前。
2. `CC_tv_direct` 已是 preacceptance hybrid 的零时移相关系数。本轮它与 `CC_final` 相同，不代表未修复 TV 的 CC。
3. `candidate_temporal_difference_energy` 与 `final_temporal_difference_energy` 的差异反映 hybrid/fallback 后处理，不能单独据此推断 Gaussian 的遮蔽作用。本轮未保存 Gaussian 前完整 W，需 T/L 2×2 才能分离 Gaussian 与正则作用。

## 5. DC、振幅与 lag 形态补充检查

| 实验 | 候选 DC ratio 最大值 | 候选 L2 norm P90 | 候选 bandwidth P90 (Hz) | 候选频谱质心中位数 (Hz) | 候选归一化曲率 P90 |
|---|---:|---:|---:|---:|---:|
| A00 | 9.771e-17 | 4.751293 | 25.0244 | 27.0347 | 0.0019812 |
| A01 | 7.469e-17 | 5.113257 | 19.6061 | 28.3348 | 0.0020035 |
| A02 | 9.781e-17 | 4.754283 | 25.0007 | 27.0354 | 0.0019889 |
| A03 | 8.992e-17 | 5.241045 | 24.5748 | 26.9858 | 0.0020574 |
| A04 | 9.729e-17 | 4.760716 | 25.0605 | 27.1277 | 0.0020321 |
| A05 | 9.770e-17 | 4.968241 | 25.0244 | 26.8362 | 0.0019811 |
| A06 | 8.950e-17 | 7.115815 | 20.4327 | 30.9680 | 0.0027982 |

DC ratio 使用仓库定义 `abs(sum(w))/(sum(abs(w))+1e-12)`。曲率使用 `sum(diff(w,n=2)**2)/(sum(w**2)+1e-12)`，未除以 dt⁴。频谱质心和 bandwidth 直接调用仓库 `compute_wavelet_qc_attributes()`；频谱采用全 RFFT 功率权重，未额外加窗、裁频带或平滑。振幅继承数据/反演标定，不赋予 NPZ 未声明的绝对物理单位。

## 6. 逐项裁决

- **A01 / edge：不支持删除。** ΔCC_final=−0.000356 虽在容许范围内，但有效窗口 70→65；最终 edge P90 0.127633→0.170074，跨越可靠阈值；最终 edge 超可靠阈值样点比例 5.58%→15.48%；最终时间差分能量 +20.25%。候选边界峰比例由 19.95% 降至 0、旁瓣 P90 有改善，说明存在权衡，不能笼统称所有形态都恶化；按统一删除规则仍未通过。
- **A02 / DC：通过单项初筛。** ΔCC_final=+0.00000425，有效窗口 70/108、hard-prior 比例不变；最终 E_t 仅 +0.137%；候选和最终 DC ratio 最大值均约 1e−16。它是优先进入下一轮组合验证的项，尚不能由本轮直接宣告永久删除。
- **A03 / mu1：暂缓删除。** ΔCC_final=+0.013870，为本轮最高 CC；原始 E_t 候选 +15.025%、最终 +17.376%，越过预先保留的建议上限。但候选平均能量 +24.16%、最终 +15.74%，归一化 E_t 分别 −7.36%、+1.42%；IRLS 和 amplitude-jump 拒绝均未恶化。这更像幅度与时间能量的耦合，尚无证据称数值求解不稳定。若后续更改判据，应在下一轮运行前确定。
- **A04 / mu2：通过当前全稳定化配置下的单项初筛。** ΔCC_final=+0.000485，有效窗口不变，候选/最终 E_t 均仅约 +0.6%；候选归一化曲率 P90 约 +2.57%，频带宽度变化很小。lag Gaussian sigma=4 仍在，因此不能据此证明曲率正则没有独立作用；优先进入 mu2 × lag Gaussian 的 2×2 验证。
- **A05 / mu_time：保留。** ΔCC_final=−0.008145，超过 −0.003 界；有效窗口 70→67；候选 E_t +126.57%，幅度归一化后仍 +103.83%。最终 E_t 仅 +4.58%，说明后续 hybrid/fallback 减弱了这一差异；本轮无法把该作用归给 Gaussian。
- **A06 / prior：保留。** ΔCC_final=−0.066383，有效窗口 70→65；hybrid TV 支撑比例 48.00%→24.43%，hard-prior 掩码 52.00%→75.57%；最终 E_t +188.18%，归一化后仍 +117.26%。候选严重旁瓣样点比例从 35.23% 增至 57.82%，即使 P90 略降也不能判为改善。

因此，实测结果不支持照预判顺序先删 edge。下一轮优先以 A02/no-DC 作 reduced-baseline 候选，后续组合仍需独立验证；mu2 转入 lag Gaussian 去耦合；mu1 先确定幅度敏感指标的解释；edge、time、prior 保留现状。B/T/L/P 与 strength tuning 均未在本轮执行，当前 baseline 保持原样。

## 7. 数值及产物验收证据

- 7/7 组主进程完成，矩阵 exit code=0；每组有 metrics.json、result_bundle.npz、config_used.yaml、tv_diag_summary.json、dtw_history.json、run.log 与 17 张 PNG，合计 119 张 PNG 可完整解码。
- 所有组 IRLS 均 87 次尝试、87 次收敛、87 次采用 Student-t 解，robust_code=1；L2 fallback=0；迭代中位数=3、最大值=5。108 个候选中心中 21 个因低能量被跳过；峰值拒绝分别 17/22/17/17/17/20/22，amplitude-jump 拒绝全部为 0。
- 所有已尝试 IRLS 窗口 objective_final≤objective_initial，且有限。不同消融目标函数不同，不跨实验比较其绝对数值。
- 关键输出数组 Shape/有限性、来源掩码互斥完整性、strict/gray/admissible/rejected 分区均通过核验。已重算 raw/final CC 和候选/最终 E_t，与保存指标吻合。
- 用保存的 W_final 和 DTW 后反射系数重新执行现有正演及 RMS 标定，与保存的最终合成记录相对误差 <1e−12。
- 代码、输入与配置 SHA256 在前后核对一致。所有结果是本轮实际运行所得，没有把旧目录结果当作本轮结果。

## 8. 文件与复现

- [原有矩阵汇总 CSV](F:/01JDX_code/python_code/seismic_well_tying/true_datacode_student_t_final_CB805/_experiment_results/ablation/round1_summary.csv)
- [机器可读验收与全部补充指标](F:/01JDX_code/python_code/seismic_well_tying/true_datacode_student_t_final_CB805/_experiment_results/ablation/round1_acceptance.json)
- [运行环境、配置/源码/输入 SHA256](F:/01JDX_code/python_code/seismic_well_tying/true_datacode_student_t_final_CB805/_experiment_results/ablation/round1_run_metadata.json)
- [完整矩阵日志](F:/01JDX_code/python_code/seismic_well_tying/true_datacode_student_t_final_CB805/_experiment_results/ablation/round1_console.log)
- [对比图](F:/01JDX_code/python_code/seismic_well_tying/true_datacode_student_t_final_CB805/_experiment_results/ablation/round1_overview.png)

下列命令复现本轮运行；若保留现有本轮产物，复跑前应将输出路径设为另一新目录。此命令本身会使用 YAML 中原有输出路径。

```powershell
$env:MPLBACKEND = 'Agg'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
& D:\miniconda\envs\myenv\python.exe -u -B experiments/run_config_matrix.py configs/ablation/A00_full.yaml configs/ablation/A01_no_edge.yaml configs/ablation/A02_no_dc.yaml configs/ablation/A03_no_mu1.yaml configs/ablation/A04_no_mu2.yaml configs/ablation/A05_no_time.yaml configs/ablation/A06_no_prior.yaml --summary-csv _experiment_results/ablation/round1_summary.csv
```

源码口径依据：[Gaussian 与去均值](F:/01JDX_code/python_code/seismic_well_tying/true_datacode_student_t_final_CB805/utils/wavelet_inversion_robust.py:1813)、[候选/最终时间差分能量](F:/01JDX_code/python_code/seismic_well_tying/true_datacode_student_t_final_CB805/experiments/run_real_experiment.py:2746)、[CC_tv_direct 与 hybrid 映射](F:/01JDX_code/python_code/seismic_well_tying/true_datacode_student_t_final_CB805/experiments/run_real_experiment.py:2768)、[频谱 QC 定义](F:/01JDX_code/python_code/seismic_well_tying/true_datacode_student_t_final_CB805/core/wavelet_qc.py:18)。在线 [baseline 配置](https://github.com/baolongattacker/true_datacode_student_t_final_CB805/blob/main/configs/cb805_center_student_t_L129_v01.yaml) 已核对；本报告数值与结论均以本地本轮输出为准。
