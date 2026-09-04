# true_datacode — Student-t 时变子波反演最终版

该版本在原井震标定流程中保留 DTW、平稳先验、滑动窗口时变子波、物理 QC、Q 候选与局部回退，仅为时变子波数据项增加 Student-t IRLS，并提供完整诊断、固定自由度筛选和最终异构误差压力测试。

开始使用前依次阅读：

1. `CODE_LOGIC_REVIEW.md`
2. `FINAL_STAGE_GUIDE.md`
3. `FINAL_VALIDATION_REPORT.md`
4. `configs/cb803_center_smooth_student_t.yaml`
5. `configs/synthetic_student_t_phase5.yaml`
6. `configs/final_student_t_stress_test.yaml`

快速验证：

```bash
python -m pip install -r requirements.txt
python experiments/run_final_self_check.py --run-tests
```
