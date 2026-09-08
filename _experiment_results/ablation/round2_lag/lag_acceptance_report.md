# CB805 Gaussian 阶段诊断与 L00–L03 去耦合验收

**本轮已完成建议中优先的下一步：增加可选阶段快照，以 no-DC 为临时基线完整运行 L00–L03。四组运行通过。lag Gaussian=4 与本次候选主峰跑到 ±64 ms 的现象直接相关；L02（保留 mu2，关闭 lag Gaussian）成为后续重点候选，但尚未通过预先保留的全部非劣门槛，未替换正式 baseline。**

## 1. 改动与数学边界

现有源码增加 47 行，涉及 [反演快照](F:/01JDX_code/python_code/seismic_well_tying/true_datacode_student_t_final_CB805/utils/wavelet_inversion_robust.py:1819)、[stage 开关传递](F:/01JDX_code/python_code/seismic_well_tying/true_datacode_student_t_final_CB805/stages/stage_tv_wavelet.py:138)、[结果保存](F:/01JDX_code/python_code/seismic_well_tying/true_datacode_student_t_final_CB805/experiments/run_real_experiment.py:1636)。目标函数、SVD/IRLS、局部 QC、插值、Gaussian、去均值、hybrid/fallback 的原有顺序和公式保持不变。

`store_stage_wavelets` 默认 False，只有本轮新 YAML 设为 True。开启时保存：

- `W_direct_centers`：通过局部 QC 的直接反演中心；shape=(2546,129)，非有效中心行为 NaN，以已有 `tv_valid_mask` 区分。它不包括已被局部峰值/QC 拒绝的解。
- `W_pre_gaussian`：现有 fill 后，未做 Gaussian。
- `W_post_gaussian`：现有二维 Gaussian 后，未执行最后去均值。
- `W_final_hybrid`：主流程构造的全局验收前 hybrid 候选。若后续全局回退，仍须以原有 `W_final` 识别真正最终模型。

后 3 个快照同为 shape=(2546,129)。时间采样 dt=1 ms、center lag −64…64 ms；振幅继承现有标定，不赋予数据未声明的绝对物理单位。原有 `W_est/W_est_best/W_final`、print 与全部原图继续保存。

本轮同时新增 [10 项快照检查脚本](F:/01JDX_code/python_code/seismic_well_tying/true_datacode_student_t_final_CB805/tests/check_wavelet_stage_snapshots.py)、[脚本化验收](F:/01JDX_code/python_code/seismic_well_tying/true_datacode_student_t_final_CB805/experiments/check_lag_ablation.py) 和 4 份完整 YAML。继续使用原来的 `run_config_matrix.py`。

## 2. 公平性与回归证据

- 四组固定 mu_dc=0、mu1=.20、prior=1/.8、mu_time=8、mu_edge=1、time Gaussian=30 ms、ν=10、wavelet length=129 samples；其它 DTW、QC、gray-zone、fallback、alpha smoothing 参数与 A02 相同。
- 新 L00 的全部旧 metrics 及 **135 项旧数值数组与上一轮 A02 逐元素完全一致**，不是仅 CC 接近。
- L00/L02 与 L01/L03 各自的直接中心矩阵、Gaussian 前矩阵、有效中心掩码、Student-t 目标值逐元素一致。关闭 lag Gaussian 没有反向影响求解。
- 四组的输入、DTW 后反射系数、时深关系、constant-phase prior/相位搜索结果一致。配置与输入 SHA256 已核验；输入仍为 `ee82c0303c5fab67c7c60fff290b13622e501355d967862fb645ab17668948e4`。
- `myenv`、单线程 BLAS/OMP、Agg 绘图后台与上一轮一致。四组主流程共约 1.00 分钟，矩阵退出码 0。
- 10 项快照开/关检查通过，覆盖 L2、Student-t、两个 Gaussian 轴开/关及零有效中心；开启快照不改变原数值和原诊断。
- 现有 gray-zone、edge penalty、global-prior fallback 共 12 项回归测试通过。四组均 70/108 有效窗口，87/87 IRLS 收敛并采用 Student-t 解，无 L2 回退；迭代中位数=3，最大值=5。

## 3. L00–L03 主结果

| 实验 | mu2 | lag sigma（samples） | raw TV CC / W_pass | CC_final | ΔCC_final vs L00 | candidate peak P90 (ms) |
|---|---:|---:|---|---:|---:|---:|
| L00 | 6 | 4 | 0.435644 / False | 0.287986 | +0.000000 | 64 |
| L01 | 0 | 4 | 0.435733 / False | 0.288468 | +0.000482 | 64 |
| L02 | 6 | 0 | 0.444446 / True | 0.352148 | +0.064162 | 14 |
| L03 | 0 | 0 | 0.444354 / True | 0.352391 | +0.064405 | 13 |

四组最终 W_pass 均为 True、最终 best lag=0 ms、candidate best lag=−1 ms，最终模型都是 `W_est_best_no_Q`。本轮 `CC_tv_direct=CC_final`，因为该字段来自 hybrid 候选。raw TV 指 Gaussian/去均值之后、hybrid 之前，raw W_pass=True 表示程序现有全局 QC 通过，并不代表每个局部样点都通过 shape QC。

2×2 的配对差值：

- 保留 mu2 时，关闭 lag Gaussian：ΔCC_final=+0.064162。
- 关闭 mu2 时，关闭 lag Gaussian：ΔCC_final=+0.063923。
- lag Gaussian 开启时，关闭 mu2：ΔCC_final=+0.000482。
- lag Gaussian 关闭时，关闭 mu2：ΔCC_final=+0.000243。

因此最大的 CC 变化来自 lag Gaussian 开/关；不能仅依据两组 mu2 差值小就判定曲率正则无作用。

## 4. 阶段诊断：主峰具体在哪一步变坏

| 实验 | 阶段 | 统计行数 | peak abs P90 (ms) | boundary peak 比例 | edge P90 | side-lobe P90 | 归一化曲率 P90 | L2 norm P90 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| L00 | 局部 QC 后直接中心 | 70 | 2 | 0.00% | 0.202040 | 0.913712 | 0.0029671 | 6.501721 |
| L00 | fill 后 / Gaussian 前 | 2546 | 14 | 0.00% | 0.213789 | 1.096103 | 0.0034373 | 6.157725 |
| L00 | Gaussian 后 / 去均值前 | 2546 | 64 | 20.15% | 0.232159 | 1.280712 | 0.0019887 | 4.754283 |
| L00 | Gaussian + 去均值后候选 | 2546 | 64 | 19.99% | 0.232276 | 1.282098 | 0.0019889 | 4.754283 |
| L00 | 全局验收前 hybrid | 2546 | 1 | 0.00% | 0.127572 | 0.802225 | 0.0017810 | 4.676585 |
| L01 | 局部 QC 后直接中心 | 70 | 2 | 0.00% | 0.201612 | 0.902549 | 0.0032428 | 6.559314 |
| L01 | fill 后 / Gaussian 前 | 2546 | 13 | 0.00% | 0.212445 | 1.054120 | 0.0047478 | 6.206856 |
| L01 | Gaussian 后 / 去均值前 | 2546 | 64 | 19.87% | 0.229884 | 1.247827 | 0.0020401 | 4.763133 |
| L01 | Gaussian + 去均值后候选 | 2546 | 64 | 19.72% | 0.230009 | 1.248886 | 0.0020402 | 4.763131 |
| L01 | 全局验收前 hybrid | 2546 | 1 | 0.00% | 0.127194 | 0.800263 | 0.0017810 | 4.688401 |
| L02 | 局部 QC 后直接中心 | 70 | 2 | 0.00% | 0.202040 | 0.913712 | 0.0029671 | 6.501721 |
| L02 | fill 后 / Gaussian 前 | 2546 | 14 | 0.00% | 0.213789 | 1.096103 | 0.0034373 | 6.157725 |
| L02 | Gaussian 后 / 去均值前 | 2546 | 14 | 0.00% | 0.213789 | 1.096103 | 0.0034373 | 6.092253 |
| L02 | Gaussian + 去均值后候选 | 2546 | 14 | 0.00% | 0.213789 | 1.096103 | 0.0034373 | 6.092253 |
| L02 | 全局验收前 hybrid | 2546 | 2 | 0.00% | 0.146457 | 0.835680 | 0.0021053 | 6.078009 |
| L03 | 局部 QC 后直接中心 | 70 | 2 | 0.00% | 0.201612 | 0.902549 | 0.0032428 | 6.559314 |
| L03 | fill 后 / Gaussian 前 | 2546 | 13 | 0.00% | 0.212445 | 1.054120 | 0.0047478 | 6.206856 |
| L03 | Gaussian 后 / 去均值前 | 2546 | 13 | 0.00% | 0.212445 | 1.054120 | 0.0047478 | 6.142079 |
| L03 | Gaussian + 去均值后候选 | 2546 | 13 | 0.00% | 0.212445 | 1.054120 | 0.0047478 | 6.142079 |
| L03 | 全局验收前 hybrid | 2546 | 2 | 0.00% | 0.147124 | 0.829513 | 0.0022375 | 6.124952 |

最清楚的证据是 L00 在相同 2546 行上的变化：**fill 后 peak P90=14 ms，Gaussian 后为 64 ms，最终去均值后仍为 64 ms。** L02 从完全相同的 fill 结果出发，保持 time Gaussian=30 ms、只关闭 lag Gaussian，peak P90 仍为 14 ms、boundary peak fraction=0，并通过 raw TV 全局 QC。

L00 的 Gaussian 后有 513 个 boundary-peak 样点，连续分布在 **0.580–1.092 s**；按填充前来源归类，453 个来自端点外推、57 个来自插值、3 个为原直接中心。去均值后仍有 509 个，区间为 0.580–1.088 s。这将当前问题定位到 lag 平滑与有限 lag 支撑下的形态变化；本轮没有改动 Gaussian 边界模式或替换算法。

直接中心共有 70 行，而完整填充结果有 2546 行；直接中心 P90=2 ms 与填充后 P90=14 ms 分母不同，**不能把这两个分位数的差异直接解释为插值引起峰值偏移**。验收 JSON 同时保存各阶段在相同 70 个直接中心上的统计；时间差分也记录了实际样点间隔，避免把稀疏中心间隔与 1 ms 连续采样混用。

DC 也能分层看到：L00 Gaussian 后 DC ratio 最大值约 0.01329，最终去均值后降至约 1e−16。因此本轮关闭的是求解中的 DC penalty，原有 hard zero-mean 仍然生效。

## 5. mu2 的作用确实有一部分被 lag Gaussian 掩盖

- 关闭 mu2：直接中心的归一化曲率 P90 增加约 **9.29%**；fill 后、Gaussian 前增加约 **38.13%**。
- lag Gaussian=4 时，候选的该差异只剩约 **2.58%**。
- lag Gaussian=0 时，候选的该差异仍为约 **38.13%**；到最终 hybrid 后缩小为约 **6.28%**。

这支持“lag Gaussian/hybrid 会掩盖曲率差异”的机制解释。L03 相比 L02 的最终 CC 仅多 0.000243；在未给曲率退化设定经过验证的容许值前，后续研究优先保留 mu2=6，使用 L02 作为候选比直接同时删掉 mu2 更有依据。这里未新增或事后调整曲率硬阈值。

## 6. 严格验收：改善与代价必须一起看

| 实验 | final edge P90 | final side-lobe P90 | 最终形态可靠比例 | final peak P90 (ms) | candidate / final E_t 变化 | 归一化 candidate / final E_t 变化 |
|---|---:|---:|---:|---:|---:|---:|
| L00 | 0.127572 | 0.802225 | 86.10% | 1 | +0.00% / +0.00% | +0.00% / +0.00% |
| L01 | 0.127194 | 0.800263 | 86.17% | 1 | +0.57% / +0.54% | -0.17% / +0.10% |
| L02 | 0.146457 | 0.835680 | 82.56% | 2 | +63.42% / +58.52% | +0.29% / +12.75% |
| L03 | 0.147124 | 0.829513 | 82.72% | 2 | +65.15% / +61.15% | +0.03% / +13.68% |

按照开跑前保存的规则（ΔCC≥−0.003、有效比例/W_pass 不降低、峰值/时移不增加、非时间消融的原始 E_t 增加不超过 15%）：

- **L01：通过当前单项/组合点值初筛。** 它同时关闭 DC 与 mu2，在 lag Gaussian=4 下仍与 L00/A00 接近。但 raw TV 的 64 ms 峰值问题仍在，且 Gaussian 前曲率增加，不能据此宣告 mu2 永久删除。
- **L02：未通过全部旧门槛，列为重点候选。** 最终 CC +0.064162、raw W_pass False→True、候选 boundary peak 消失；代价是 candidate/final 原始 E_t +63.42%/+58.52%，最终 peak P90 从 1→2 ms，最终形态可靠比例 86.10%→82.56%。edge P90=0.146457 尚在 0.15 内，但余量较小。
- **L03：同样未通过全部旧门槛。** candidate/final 原始 E_t +65.15%/+61.15%，最终 peak P90=2 ms；最终 edge P90=0.147124，接近可靠界。相较 L02 的 CC 增益很小，曲率却进一步增大。

L02 的幅度归一化 candidate/final E_t 分别仅 +0.29%/+12.75%；其候选/最终平均子波能量分别增加约 62.94%/40.60%。所以原始 E_t 增长包含明显幅度效应，但最终归一化时间变化也并非零。**不能把 +58.5% 全部称为时间抖动，也不能事后换成归一化门槛宣布它已通过。**

最终 peak P90 从 1→2 ms 仍远小于程序 17 ms 的全局阈值；本轮拒绝的是原先严格“不增加”的比较规则，与程序本身 W_pass=True 并不矛盾。这些门槛适合下一轮运行前重新明确，而非本轮事后放宽。

## 7. 幅度诊断

| 实验 | 全局 R_amp（最终 W 正演、RMS 标定前） | 标定后局部 R_amp P10 / P50 / P90 |
|---|---:|---|
| L00 | 0.204513 | 0.3146 / 0.8711 / 1.7094 |
| L01 | 0.204824 | 0.3146 / 0.8731 / 1.7074 |
| L02 | 0.218566 | 0.3487 / 0.9644 / 1.5543 |
| L03 | 0.219300 | 0.3501 / 0.9632 / 1.5519 |

R_amp 使用保存的 W 与同一 DTW 后反射系数直接正演，再与观测 RMS 比较。原流程会对最终 synthetic 作全局 `match_rms`，因此全局标定后的 R_amp 约为 1，本身不能识别振幅再分配。滑窗使用与反演相同的 387 个样点（约 387 ms），仅完整窗口、没有零填充，共 2160 个窗口，本轮没有观测 RMS=0 的窗口。局部曲线保存在 `rms_profiles.npz`，诊断不参与拟合或最终模型选择。

## 8. 本轮结论及边界

1. 诊断代码验收通过，四组实验完成，旧 A02 完全复现。
2. 当前 mu_dc=0 的试验中，lag Gaussian=4 会把一部分本来通过局部 peak QC 的子波及其外推/插值结果变成边界峰；新增快照给出了直接的阶段证据。
3. 保留 mu2、关闭 lag Gaussian 的 L02 值得继续验证，但它尚未满足全部旧非劣门槛；原 baseline 没有被替换。
4. mu2 在 Gaussian 前仍抑制曲率，建议保留后进入后续验证；不把 CC 很小的差异解释为数学冗余已证实。
5. 本轮范围为建议中明确优先的“阶段诊断 + L00–L03”。T00–T03、mu1 strength sweep 与最终联合删除尚未运行；下一步应先明确幅度归一化指标与峰值变化容许值，再按建议推进时间去耦合。

这是单一 CB805 固定输入的完整主流程机制验证；未进行多井、噪声重复或统计置信区间验证。

## 9. 文件与复现入口

- [原矩阵 CSV](F:/01JDX_code/python_code/seismic_well_tying/true_datacode_student_t_final_CB805/_experiment_results/ablation/round2_lag/lag_summary.csv)、[机器可读验收及各阶段全指标](F:/01JDX_code/python_code/seismic_well_tying/true_datacode_student_t_final_CB805/_experiment_results/ablation/round2_lag/lag_acceptance.json)
- [结果概览](F:/01JDX_code/python_code/seismic_well_tying/true_datacode_student_t_final_CB805/_experiment_results/ablation/round2_lag/lag_overview.png)、[阶段定位与幅度图](F:/01JDX_code/python_code/seismic_well_tying/true_datacode_student_t_final_CB805/_experiment_results/ablation/round2_lag/lag_stage_mechanism.png)
- [执行计划、冻结条件和哈希](F:/01JDX_code/python_code/seismic_well_tying/true_datacode_student_t_final_CB805/_experiment_results/ablation/round2_lag/round2_plan.json)、[完整实验日志](F:/01JDX_code/python_code/seismic_well_tying/true_datacode_student_t_final_CB805/_experiment_results/ablation/round2_lag/lag_console.log)
- [快照检查日志](F:/01JDX_code/python_code/seismic_well_tying/true_datacode_student_t_final_CB805/_experiment_results/ablation/round2_lag/snapshot_tests.log)、[原 QC 回归日志](F:/01JDX_code/python_code/seismic_well_tying/true_datacode_student_t_final_CB805/_experiment_results/ablation/round2_lag/regression_tests.log)、[验收复算日志](F:/01JDX_code/python_code/seismic_well_tying/true_datacode_student_t_final_CB805/_experiment_results/ablation/round2_lag/lag_acceptance_checks.log)
- [源码差异](F:/01JDX_code/python_code/seismic_well_tying/true_datacode_student_t_final_CB805/_experiment_results/ablation/round2_lag/diagnostics_changes.patch)

四份配置位于原 `configs/ablation` 目录，以 L00–L03 开头。原 A00–A06 配置与结果保留。

```powershell
$env:MPLBACKEND = 'Agg'
$env:PYTHONIOENCODING = 'utf-8'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
& D:\miniconda\envs\myenv\python.exe -B tests/check_wavelet_stage_snapshots.py
& D:\miniconda\envs\myenv\python.exe -B experiments/check_lag_ablation.py
```

完整实验继续使用 `experiments/run_config_matrix.py` 顺序传入 L00–L03，输出汇总指向 `round2_lag/lag_summary.csv`；复跑前使用新的结果目录以保留本轮记录。
