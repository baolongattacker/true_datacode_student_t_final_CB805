# -*- coding: utf-8 -*-
"""
core/result_types.py

兼容导出层。

真实的阶段返回类型定义在 `stages/*` 模块内，本文件不再重复定义一套
dataclass。这样可以避免字段名随阶段实现变化后，旧的 result_types.py
继续保留过期字段，造成主流程和类型说明漂移。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from stages.stage_dtw import DtwResult
from stages.stage_q_constraint import QCandidateResult
from stages.stage_stationary import StationaryResult
from stages.stage_tv_wavelet import TvWaveletResult
from stages.stage_wavelet_prior import WaveletPriorResult


# 保留历史导入名，指向当前真实 stage 返回类型。
# 注意：这些别名只改变类型入口，不改变任何反演算法、阈值或流程顺序。
DTWResult = DtwResult
StationaryPriorResult = WaveletPriorResult
TVWaveletResult = TvWaveletResult
QResult = QCandidateResult


@dataclass
class FinalResult:
    """
    最终模型选择结果。

    输入：主流程中已经计算完成的候选子波矩阵、合成记录和 QC 指标。
    输出：被选中的最终子波矩阵、最终合成记录和模型类型说明。
    单位：W_final 与输入子波矩阵一致，s_syn_final 与 obs_work 振幅标定一致。
    物理意义：记录最终采用的是 Q 约束、时变子波，还是平稳子波兜底。
    数学作用：只保存选择结果，不重新计算反射系数、TWT、子波或 Q。
    """

    W_final: np.ndarray
    s_syn_final: np.ndarray
    cc_final: float
    final_model_type: str
    W_pass: bool
    q_pass: bool
    q_attempted: bool
    acceptance_reasons: list[str]
    q_reason: str
    extra: dict[str, Any] | None = None


__all__ = [
    "StationaryResult",
    "DtwResult",
    "DTWResult",
    "WaveletPriorResult",
    "StationaryPriorResult",
    "TvWaveletResult",
    "TVWaveletResult",
    "QCandidateResult",
    "QResult",
    "FinalResult",
]
