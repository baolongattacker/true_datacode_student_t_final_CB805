# -*- coding: utf-8 -*-
"""
stages/stage_q_constraint.py

Q-constrained candidate stage.

This stage builds and evaluates a Q-constrained W candidate. It does not
choose the final model, save files, plot, or mutate the input TV candidate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np



from utils.q_estimation import estimate_global_Q_and_rebuild_wavelets

from core.acceptance import QAcceptanceConfig, evaluate_q_candidate
from core.forward_operator import nonstationary_convolution
from core.metrics import (
    corrcoef_safe,
    envelope_cc,
    max_lag_cc,
    tie_similarity_score,
)
from core.signal_utils import match_rms


@dataclass
class QCandidateResult:
    attempted: bool
    accepted: bool
    reason: str
    reasons: list[str]

    Q_global: float

    W_q: np.ndarray | None
    s_syn_q: np.ndarray | None

    cc_q: float
    env_q: float
    maxlag_q: float
    lag_q_samples: int
    lag_q_ms: float

    cc_tv_reference: float
    env_tv_reference: float

    acceptance_details: dict[str, Any]
    params: dict[str, Any]


def _not_attempted_result(
    *,
    reason: str,
    cc_tv_direct: float,
    env_tv: float,
    params: dict[str, Any],
) -> QCandidateResult:
    return QCandidateResult(
        attempted=False,
        accepted=False,
        reason=reason,
        reasons=[reason],
        Q_global=np.nan,
        W_q=None,
        s_syn_q=None,
        cc_q=np.nan,
        env_q=np.nan,
        maxlag_q=np.nan,
        lag_q_samples=0,
        lag_q_ms=np.nan,
        cc_tv_reference=float(cc_tv_direct),
        env_tv_reference=float(env_tv),
        acceptance_details={},
        params=params,
    )


def try_q_constraint_stage(
    *,
    W_est_best: np.ndarray,
    r_time: np.ndarray,
    obs_work: np.ndarray,
    dt: float,
    f_dom: float,
    alignment: str,
    W_pass: bool,
    enable_q: bool,
    cc_tv_direct: float,
    env_tv: float,
    f_min: float = 10.0,
    f_max: float = 50.0,
    ref_range: tuple[float, float] = (0.1, 0.2),
    deep_range: tuple[float, float] = (0.6, 0.8),
    Q_min: float = 20.0,
    Q_max: float = 150.0,
    freeze_above_ref: bool = True,
    max_cc_drop_vs_tv: float = 0.005,
    max_env_drop_vs_tv: float = 0.03,
    max_lag_ms: float = 8.0,
    q_rebuild_func: Callable | None = None,
    verbose: bool = True,
) -> QCandidateResult:
    """
    Try to build a Q-constrained candidate.

    If Q is disabled or the TV-wavelet candidate failed, this returns an
    unattempted result and leaves final selection to the caller.
    """
    W_est_best = np.asarray(W_est_best, dtype=float)
    r_time = np.asarray(r_time, dtype=float).ravel()
    obs_work = np.asarray(obs_work, dtype=float).ravel()

    cc_q_direct = np.nan
    q_similarity_details = {}

    params = {
        "enable_q": enable_q,
        "f_min": f_min,
        "f_max": f_max,
        "ref_range": ref_range,
        "deep_range": deep_range,
        "Q_min": Q_min,
        "Q_max": Q_max,
        "freeze_above_ref": freeze_above_ref,
        "max_cc_drop_vs_tv": max_cc_drop_vs_tv,
        "max_env_drop_vs_tv": max_env_drop_vs_tv,
        "max_lag_ms": max_lag_ms,
        "cc_q_direct": cc_q_direct,
        "tie_similarity_q": q_similarity_details,
    }

    if not enable_q:
        return _not_attempted_result(
            reason="q_disabled",
            cc_tv_direct=cc_tv_direct,
            env_tv=env_tv,
            params=params,
        )

    if not W_pass:
        return _not_attempted_result(
            reason="skip_q_because_W_pass_false",
            cc_tv_direct=cc_tv_direct,
            env_tv=env_tv,
            params=params,
        )

    if q_rebuild_func is None:
        q_rebuild_func = estimate_global_Q_and_rebuild_wavelets

    if verbose:
        print("\n--- Step 4: Q constrained candidate reconstruction ---")

    try:
        Q_global, W_q = q_rebuild_func(
            W=W_est_best,
            dt=dt,
            f_dom=f_dom,
            f_min=f_min,
            f_max=f_max,
            ref_range=ref_range,
            deep_range=deep_range,
            Q_min=Q_min,
            Q_max=Q_max,
            freeze_above_ref=freeze_above_ref,
        )

        s_q = nonstationary_convolution(
            r_time=r_time,
            W=W_q,
            alignment=alignment,
        )
        s_q = match_rms(s_q, obs_work)

        # 原始零时移相关，仅作为诊断量
        cc_q_direct = corrcoef_safe(obs_work, s_q)
        env_q = envelope_cc(obs_work, s_q)

        maxlag_samples = int(round(max_lag_ms / (dt * 1000.0)))
        maxlag_q, lag_q = max_lag_cc(
            obs_work,
            s_q,
            max_lag_samples=maxlag_samples,
        )
        lag_q_ms = lag_q * dt * 1000.0

        # Q 候选也使用同一套井震标定综合相似性
        cc_q, q_similarity_details = tie_similarity_score(
            s_obs=obs_work,
            s_syn=s_q,
            dt=dt,
            max_lag_ms=max_lag_ms,
        )

        params["cc_q_direct"] = cc_q_direct
        params["tie_similarity_q"] = q_similarity_details

        acceptance = evaluate_q_candidate(
            cc_q=cc_q,
            cc_tv=cc_tv_direct,
            env_q=env_q,
            env_tv=env_tv,
            lag_q_ms=lag_q_ms,
            config=QAcceptanceConfig(
                max_cc_drop_vs_tv=max_cc_drop_vs_tv,
                max_env_drop_vs_tv=max_env_drop_vs_tv,
                max_lag_ms=max_lag_ms,
            ),
        )

        if verbose:
            print(f"  Q_global          = {Q_global:.2f}")
            print(f"  CC_tv_direct      = {cc_tv_direct:.4f}")
            print(f"  CC_Q              = {cc_q:.4f}")
            print(f"  Env_tv            = {env_tv:.4f}")
            print(f"  Env_Q             = {env_q:.4f}")
            print(f"  Maxlag_Q          = {maxlag_q:.4f}, lag={lag_q_ms:.1f} ms")
            print(f"  q_pass            = {acceptance.passed}")
            print(f"  q_reasons         = {acceptance.reasons}")

        return QCandidateResult(
            attempted=True,
            accepted=bool(acceptance.passed),
            reason="accepted" if acceptance.passed else "rejected_by_q_gate",
            reasons=acceptance.reasons,
            Q_global=float(Q_global),
            W_q=W_q,
            s_syn_q=s_q,
            cc_q=float(cc_q),
            env_q=float(env_q),
            maxlag_q=float(maxlag_q),
            lag_q_samples=int(lag_q),
            lag_q_ms=float(lag_q_ms),
            cc_tv_reference=float(cc_tv_direct),
            env_tv_reference=float(env_tv),
            acceptance_details=acceptance.details,
            params=params,
        )

    except Exception as exc:
        if verbose:
            print(f"  [Q] failed: {exc}")

        reason = f"q_failed: {exc}"
        return QCandidateResult(
            attempted=True,
            accepted=False,
            reason=reason,
            reasons=[reason],
            Q_global=np.nan,
            W_q=None,
            s_syn_q=None,
            cc_q=np.nan,
            env_q=np.nan,
            maxlag_q=np.nan,
            lag_q_samples=0,
            lag_q_ms=np.nan,
            cc_tv_reference=float(cc_tv_direct),
            env_tv_reference=float(env_tv),
            acceptance_details={},
            params=params,
        )
