# -*- coding: utf-8 -*-
"""第六轮消融实验准备脚本：核验已有的 S00–S03 配置文件，并生成运行前冻结计划 round6_plan.json。"""

import datetime
import hashlib
import json
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
ROUND6_DIR = ROOT / "_experiment_results/ablation/round6_time_smooth"
ROUND5_DIR = ROOT / "_experiment_results/ablation/round5_mu2"

S_CONFIGS = [
    {"id": "S00", "time_smooth_ms": 0.0, "filename": "S00_time_smooth_0ms.yaml"},
    {"id": "S01", "time_smooth_ms": 10.0, "filename": "S01_time_smooth_10ms.yaml"},
    {"id": "S02", "time_smooth_ms": 20.0, "filename": "S02_time_smooth_20ms.yaml"},
    {"id": "S03", "time_smooth_ms": 30.0, "filename": "S03_time_smooth_30ms.yaml"},
]

def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()

def prepare_round6():
    configs_meta = []
    for spec in S_CONFIGS:
        yaml_path = ROUND6_DIR / spec["filename"]
        assert yaml_path.exists(), f"未找到文件: {yaml_path}"
        cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        tv = cfg["tv_wavelet"]

        # 严格核对基线与待扫描变量
        assert tv["mu1"] == 0.0, f"{spec['filename']} mu1 必须为 0.0"
        assert tv["mu2"] == 3.0, f"{spec['filename']} mu2 必须为 3.0"
        assert tv["mu_dc"] == 0.0, f"{spec['filename']} mu_dc 必须为 0.0"
        assert tv["mu_time"] == 8.0, f"{spec['filename']} mu_time 必须为 8.0"
        assert tv["mu_edge"] == 1.0, f"{spec['filename']} mu_edge 必须为 1.0"
        assert tv["time_smooth_ms"] == spec["time_smooth_ms"], (
            f"{spec['filename']} time_smooth_ms 期望 {spec['time_smooth_ms']}, 实际 {tv['time_smooth_ms']}"
        )
        assert tv["wavelet_smooth_sigma"] == 0.0, f"{spec['filename']} wavelet_smooth_sigma 必须为 0.0"
        assert tv.get("store_stage_wavelets") is True, f"{spec['filename']} store_stage_wavelets 必须为 True"

        digest = sha256_file(yaml_path)
        configs_meta.append({
            "id": spec["id"],
            "config_path": f"_experiment_results/ablation/round6_time_smooth/{spec['filename']}",
            "sha256": digest,
            "time_smooth_ms": spec["time_smooth_ms"],
            "result_dir": cfg["paths"]["result_dir"],
        })
        print(f"已核对配置: {spec['filename']} (time_smooth_ms={spec['time_smooth_ms']}) -> {digest[:12]}")

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
        "valid_centers_allowed_drop": 0,
        "irls_fallback_allowed_increase": 0,
        "ill_conditioned_allowed_increase": 0,
        "edge_p90_preferred": 0.15,
        "edge_p90_severe": 0.30,
        "side_lobe_reliable": 0.75,
        "side_lobe_fallback": 0.95,
        "regression_invariants": {
            "W_direct_exact_match": True,
            "W_preGaussian_exact_match": True,
            "tolerance": 1e-12
        },
        "pareto_objectives": [
            "minimize time_smooth_ms",
            "maximize continuity benefit ratio B(sigma) = (E_0 - E_sigma) / (E_0 - E_30)",
            "minimize Gaussian modification cost D_W(sigma) = median_t(||w_postG - w_preG|| / (||w_preG|| + eps))",
            "monitor hybrid dependency (hard_prior_ratio, tv_support_ratio, blend_ratio, shape_fallback_ratio)"
        ]
    }

    plan = {
        "status": "planned",
        "prepared_at": datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        "baseline": "C02_mu2_3p0",
        "anchor_case": "S03_time_smooth_30ms",
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
        "scope": "Round 6: time_smooth_ms sweep (0, 10, 20, 30 ms) under frozen (mu1=0.0, mu2=3.0) baseline",
    }

    plan_path = ROUND6_DIR / "round6_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"已生成运行前冻结计划: {plan_path} (哈希锁定完成)")

if __name__ == "__main__":
    prepare_round6()
