# -*- coding: utf-8 -*-
"""第 5.5 轮消融实验准备脚本：核验 I10–I12 配置文件，并生成运行前冻结计划 round5p5_plan.json。"""

import datetime
import hashlib
import json
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
ROUND5P5_DIR = ROOT / "_experiment_results/ablation/round5p5_mu1_mu2"
ROUND5_DIR = ROOT / "_experiment_results/ablation/round5_mu2"

I_CONFIGS = [
    {"id": "I10", "mu1": 0.05, "mu2": 1.5, "filename": "I10_mu1_0p05_mu2_1p5.yaml"},
    {"id": "I11", "mu1": 0.05, "mu2": 3.0, "filename": "I11_mu1_0p05_mu2_3p0.yaml"},
    {"id": "I12", "mu1": 0.05, "mu2": 6.0, "filename": "I12_mu1_0p05_mu2_6p0.yaml"},
]

def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()

def prepare_round5p5():
    configs_meta = []
    for spec in I_CONFIGS:
        yaml_path = ROUND5P5_DIR / spec["filename"]
        assert yaml_path.exists(), f"未找到文件: {yaml_path}"
        cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        tv = cfg["tv_wavelet"]

        assert tv["mu1"] == spec["mu1"]
        assert tv["mu2"] == spec["mu2"]
        assert tv["mu_dc"] == 0.0
        assert tv["mu_time"] == 8.0
        assert tv["mu_edge"] == 1.0
        assert tv["time_smooth_ms"] == 30.0
        assert tv["wavelet_smooth_sigma"] == 0.0
        assert tv.get("store_stage_wavelets") is True

        digest = sha256_file(yaml_path)
        configs_meta.append({
            "id": spec["id"],
            "config_path": f"_experiment_results/ablation/round5p5_mu1_mu2/{spec['filename']}",
            "sha256": digest,
            "mu1": spec["mu1"],
            "mu2": spec["mu2"],
            "result_dir": cfg["paths"]["result_dir"],
        })
        print(f"已核对配置: {spec['filename']} (mu1={spec['mu1']}, mu2={spec['mu2']}) -> {digest[:12]}")

    plan5 = json.loads((ROUND5_DIR / "round5_plan.json").read_text(encoding="utf-8"))
    source_sha256 = {}
    for rel_path in sorted(plan5["source_sha256"].keys()):
        p = ROOT / rel_path
        if p.exists():
            source_sha256[rel_path] = sha256_file(p)
        else:
            source_sha256[rel_path] = plan5["source_sha256"][rel_path]

    input_npz = ROOT / "_experiment_data/initial_model_data/initial_real_model_data_CB805.npz"
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
        "interaction_goals": [
            "Verify non-inferiority of mu1=0.0 across mu2 in (1.5, 3.0, 6.0)",
            "Detect any regularization compensation between mu1 and mu2",
            "Evaluate hard prior takeover ratio across the 2x3 grid"
        ]
    }

    plan = {
        "status": "planned",
        "prepared_at": datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        "baseline": "C03_mu2_6p0 (M00)",
        "anchor_case": "I12_mu1_0p05_mu2_6p0 (M01)",
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
        "reused_configs": [
            {"id": "C01", "mu1": 0.0, "mu2": 1.5, "result_dir": "_experiment_results/ablation/round5_mu2/C01_mu2_1p5"},
            {"id": "C02", "mu1": 0.0, "mu2": 3.0, "result_dir": "_experiment_results/ablation/round5_mu2/C02_mu2_3p0"},
            {"id": "C03", "mu1": 0.0, "mu2": 6.0, "result_dir": "_experiment_results/ablation/round5_mu2/C03_mu2_6p0"},
        ],
        "criteria": criteria,
        "scope": "Round 5.5: mu1 x mu2 (2x3) interaction check across mu1 in (0, 0.05) and mu2 in (1.5, 3.0, 6.0)",
    }

    plan_path = ROUND5P5_DIR / "round5p5_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"已生成运行前冻结计划: {plan_path} (哈希锁定完成)")

if __name__ == "__main__":
    prepare_round5p5()
