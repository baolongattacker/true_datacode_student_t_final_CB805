# -*- coding: utf-8 -*-
"""第四轮消融实验准备脚本：生成 M00–M04 配置文件及运行前冻结计划。

严格以第三轮工作基准 T00 为模板，仅改变 tv_wavelet.mu1。
"""

import copy
import datetime
import hashlib
import json
import sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
ROUND3_DIR = ROOT / "_experiment_results/ablation/round3_time"
OUT_DIR = ROOT / "_experiment_results/ablation/round4_mu1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

T00_YAML_PATH = ROUND3_DIR / "T00_time_on_gaussian_on.yaml"
assert T00_YAML_PATH.exists(), f"未找到基准文件: {T00_YAML_PATH}"

SWEEP_SPECS = [
    {"id": "M00", "mu1": 0.00, "filename": "M00_mu1_0p00.yaml", "desc": "能量无约束极值点"},
    {"id": "M01", "mu1": 0.05, "filename": "M01_mu1_0p05.yaml", "desc": "弱能量约束 (0.25x)"},
    {"id": "M02", "mu1": 0.10, "filename": "M02_mu1_0p10.yaml", "desc": "中弱能量约束 (0.50x)"},
    {"id": "M03", "mu1": 0.20, "filename": "M03_mu1_0p20.yaml", "desc": "基准复现参照点 (1.00x，与 T00 一致)"},
    {"id": "M04", "mu1": 0.40, "filename": "M04_mu1_0p40.yaml", "desc": "强能量约束 (2.00x)"},
]

def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()

def prepare_round4():
    raw_text = T00_YAML_PATH.read_text(encoding="utf-8")
    base_cfg = yaml.safe_load(raw_text)

    # 验证基准配置确实符合 T00 冻结要求
    assert base_cfg["tv_wavelet"]["mu1"] == 0.2
    assert base_cfg["tv_wavelet"]["mu2"] == 6.0
    assert base_cfg["tv_wavelet"]["mu_dc"] == 0.0
    assert base_cfg["tv_wavelet"]["mu_time"] == 8.0
    assert base_cfg["tv_wavelet"]["mu_edge"] == 1.0
    assert base_cfg["tv_wavelet"]["time_smooth_ms"] == 30.0
    assert base_cfg["tv_wavelet"]["wavelet_smooth_sigma"] == 0.0
    assert base_cfg["tv_wavelet"].get("store_stage_wavelets") is True

    configs_meta = []

    for spec in SWEEP_SPECS:
        cfg = copy.deepcopy(base_cfg)
        name = f"cb805_ablation_{spec['id']}_{spec['filename'].replace('.yaml', '')}"
        cfg["experiment"]["name"] = name
        result_dir_rel = f"_experiment_results/ablation/round4_mu1/{spec['id']}_{spec['filename'].replace('.yaml', '')}"
        cfg["paths"]["result_dir"] = result_dir_rel
        cfg["tv_wavelet"]["mu1"] = spec["mu1"]

        out_yaml_path = OUT_DIR / spec["filename"]
        out_yaml_path.write_text(yaml.dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
        yaml_digest = sha256_file(out_yaml_path)

        configs_meta.append({
            "id": spec["id"],
            "config_path": f"_experiment_results/ablation/round4_mu1/{spec['filename']}",
            "sha256": yaml_digest,
            "mu1": spec["mu1"],
            "description": spec["desc"],
            "result_dir": result_dir_rel,
        })
        print(f"已生成配置: {spec['filename']} (mu1={spec['mu1']}) -> {yaml_digest[:12]}")

    # 读取 round3_plan 中的 source 列表，计算最新 sha256
    plan3 = json.loads((ROUND3_DIR / "round3_plan.json").read_text(encoding="utf-8"))
    source_sha256 = {}
    for rel_path in sorted(plan3["source_sha256"].keys()):
        p = ROOT / rel_path
        if p.exists():
            source_sha256[rel_path] = sha256_file(p)
        else:
            source_sha256[rel_path] = plan3["source_sha256"][rel_path]

    input_npz = ROOT / base_cfg["paths"]["input_npz"]
    input_sha = sha256_file(input_npz)

    criteria = {
        "delta_CC_final_floor": -0.003,
        "raw_W_pass_no_regression": True,
        "hybrid_W_pass_no_regression": True,
        "peak_abs_p90_ms_hard_gate": 17.0,
        "peak_abs_p90_ms_preferred": 15.0,
        "normalized_Et_increase_limit_pct": 15.0,
        "normalization_epsilon": 1e-12,
        "valid_centers_allowed_drop": 0,
        "edge_p90_preferred": 0.15,
        "edge_p90_severe": 0.30,
        "side_lobe_reliable": 0.75,
        "side_lobe_fallback": 0.95,
        "irls_fallback_allowed_increase": 0,
        "ill_conditioned_allowed_increase": 0,
        "absolute_Et_is_veto": False,
        "pareto_objectives": [
            "maximize CC_final",
            "minimize common-centers normalized Et",
            "minimize amplitude log error: median(|log(R_amp_scaled)|)"
        ]
    }

    plan = {
        "status": "planned",
        "prepared_at": datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        "baseline": "T00_time_on_gaussian_on",
        "python_executable": "D:\\miniconda\\envs\\myenv\\python.exe",
        "input_file": str(input_npz),
        "input_sha256": input_sha,
        "source_sha256": source_sha256,
        "module_paths": {
            "solver": str(ROOT / "utils/wavelet_inversion_robust.py"),
            "stage": str(ROOT / "stages/stage_tv_wavelet.py"),
            "experiment": str(ROOT / "experiments/run_real_experiment.py"),
        },
        "configs": configs_meta,
        "criteria": criteria,
        "scope": "Round 4: mu1 strength sweep (0, 0.05, 0.10, 0.20, 0.40) under frozen T00 baseline",
    }

    plan_path = OUT_DIR / "round4_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"已生成运行前冻结计划: {plan_path} (哈希锁定完成)")

if __name__ == "__main__":
    prepare_round4()
