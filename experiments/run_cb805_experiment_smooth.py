# -*- coding: utf-8 -*-
"""
Compatibility wrapper for the historical CB805 runner name.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import os

# 解决import搜索路径问题，使得在当前目录下运行脚本时能够正确导入模块
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.run_real_experiment import choose_final_model, main  # noqa: F401,E402
  
# 解决文件读取路径问题
def _default_config_path() -> str:
    return str(
        Path(__file__).resolve().parents[1]
        / "configs"
        / "cb805_center_student_t_rickerprior.yaml"
    )
print("__file__ =", __file__)
print("resolve =", Path(__file__).resolve())
print("cwd =", os.getcwd())
   
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=_default_config_path(), # 没有提供--config参数时，使用默认路径  
        help="Path to experiment config yaml.",
    )
    args = parser.parse_args()
    main(args.config)
   
