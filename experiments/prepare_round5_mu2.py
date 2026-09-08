# -*- coding: utf-8 -*-
"""第五轮消融实验准备脚本：核验已有的 C00–C04 配置文件，并生成运行前冻结计划 round5_plan.json。"""

import datetime
import hashlib
import json
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
ROUND5_DIR = ROOT / "_experiment_results/ablation/round5_mu2"
ROUND4_DIR = ROOT / "_experiment_results/ablation/round4_mu1"

C_CONFIGS = [
    {"id": "C00", "mu2": 0.0, "filename": "C00_mu2_0p0.yaml"},
    {"id": "C01", "mu2": 1.5, "filename": "C01_mu2_1p5.yaml"},
    {"id": "C02", "mu2": 3.0, "filename": "C02_mu2_3p0.yaml"},
    {"id": "C03", "mu2": 6.0, "filename": "C03_mu2_6p0.yaml"},
    {"id": "C04", "mu2": 12.0, "filename": "C04_mu2_12p0.yaml"},
]

def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()

def prepare_round5():
    configs_meta = []
    for spec in C_CONFIGS:
        yaml_path = ROUND5_DIR / spec["filename"]
        assert yaml_path.exists(), f"未找到文件: {yaml_path}"
        cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        tv = cfg["tv_wavelet"]

        # 严格核对基线与待扫描变量
        assert tv["mu1"] == 0.0, f"{spec['filename']} mu1 必须为 0.0"
        assert tv["mu2"] == spec["mu2"], f"{spec['filename']} mu2 期望 {spec['mu2']}, 实际 {tv['mu2']}"
        assert tv["mu_dc"] == 0.0, f"{spec['filename']} mu_dc 必须为 0.0"
        assert tv["mu_time"] == 8.0, f"{spec['filename']} mu_time 必须为 8.0"
        assert tv["mu_edge"] == 1.0, f"{spec['filename']} mu_edge 必须为 1.0"
        assert tv["time_smooth_ms"] == 30.0, f"{spec['filename']} time_smooth_ms 必须为 30.0"
        assert tv["wavelet_smooth_sigma"] == 0.0, f"{spec['filename']} wavelet_smooth_sigma 必须为 0.0"
        assert tv.get("store_stage_wavelets") is True, f"{spec['filename']} store_stage_wavelets 必须为 True"

        digest = sha256_file(yaml_path)
        configs_meta.append({
            "id": spec["id"],
            "config_path": f"_experiment_results/ablation/round5_mu2/{spec['filename']}",
            "sha256": digest,
            "mu2": spec["mu2"],
            "result_dir": cfg["paths"]["result_dir"],
        })
        print(f"已核对配置: {spec['filename']} (mu2={spec['mu2']}) -> {digest[:12]}")

    plan4 = json.loads((ROUND4_DIR / "round4_plan.json").read_text(encoding="utf-8"))
    source_sha256 = {}
    for rel_path in sorted(plan4["source_sha256"].keys()):
        p = ROOT / rel_path
        if p.exists():
            source_sha256[rel_path] = sha256_file(p)
        else:
            source_sha256[rel_path] = plan4["source_sha256"][rel_path]

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
        "pareto_objectives": [
            "minimize normalized lag-curvature P90",
            "preserve CC_final non-inferior (>= baseline - 0.003)",
            "minimize amplitude log error: median(|log(R_amp_scaled)|)"
        ]
    }

    plan = {
        "status": "planned",
        "prepared_at": datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        "baseline": "M00_mu1_0p00",
        "anchor_case": "C03_mu2_6p0",
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
        "scope": "Round 5: mu2 strength sweep (0, 1.5, 3.0, 6.0, 12.0) under frozen M00 baseline (mu1=0.0)",
    }

    plan_path = ROUND5_DIR / "round5_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"已生成运行前冻结计划: {plan_path} (哈希锁定完成)")

if __name__ == "__main__":
    prepare_round5()
