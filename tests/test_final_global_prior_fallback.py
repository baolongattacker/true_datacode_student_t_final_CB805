import sys
from pathlib import Path

import numpy as np

# 确保直接以 python 脚本方式运行时也能找到工程根目录
_project_root = str(Path(__file__).resolve().parents[1])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from experiments.run_real_experiment import (
    FINAL_SOURCE_LOCAL_BLEND,
    FINAL_SOURCE_LOCAL_FALLBACK_PRIOR,
    FINAL_SOURCE_Q_CONSTRAINED,
    FINAL_SOURCE_STATIONARY_PRIOR,
    FINAL_SOURCE_TV_CANDIDATE,
    build_final_wavelet_source_codes,
    choose_final_model,
)


def test_global_prior_fallback_model_type_is_supported():
    n_time = 64
    wavelet_length = 129

    w_prior = np.zeros(
        wavelet_length,
        dtype=float,
    )
    w_prior[wavelet_length // 2] = 1.0

    s_syn_prior = np.linspace(
        -1.0,
        1.0,
        n_time,
        dtype=float,
    )

    W_tv = np.zeros(
        (n_time, wavelet_length),
        dtype=float,
    )

    s_syn_tv = np.zeros(
        n_time,
        dtype=float,
    )

    final = choose_final_model(
        alignment="center",
        W_pass=False,
        q_result=None,
        W_est_best=W_tv,
        s_syn_tv_direct=s_syn_tv,
        w_prior=w_prior,
        s_syn_prior=s_syn_prior,
        prior_source="constant_phase",
    )

    assert (
        final["model_type"]
        == "prior_after_DTW_global_fallback"
    )

    assert final["q_pass"] is False

    expected_W = np.tile(
        w_prior[None, :],
        (n_time, 1),
    )

    np.testing.assert_allclose(
        final["W_final"],
        expected_W,
        rtol=0.0,
        atol=0.0,
    )

    np.testing.assert_allclose(
        final["s_syn_final"],
        s_syn_prior,
        rtol=0.0,
        atol=0.0,
    )


def test_global_prior_fallback_source_codes_do_not_crash():
    n_time = 64

    final_source_alpha = np.ones(
        n_time,
        dtype=float,
    )

    (
        source_code,
        source_masks,
    ) = build_final_wavelet_source_codes(
        base_model_type=(
            "prior_after_DTW_global_fallback"
        ),
        local_fallback_alpha=(
            final_source_alpha
        ),
    )

    assert source_code.shape == (n_time,)

    assert np.all(
        source_code
        == FINAL_SOURCE_STATIONARY_PRIOR
    )

    assert np.all(
        source_masks[
            "stationary_prior_mask"
        ]
    )

    assert not np.any(
        source_masks[
            "tv_candidate_mask"
        ]
    )

    assert not np.any(
        source_masks[
            "q_constrained_mask"
        ]
    )

    assert not np.any(
        source_masks[
            "local_fallback_prior_mask"
        ]
    )

    assert not np.any(
        source_masks[
            "local_blend_mask"
        ]
    )


def test_all_final_source_codes_remain_mutually_exclusive():
    n_time = 32

    final_source_alpha = np.ones(
        n_time,
        dtype=float,
    )

    (
        source_code,
        source_masks,
    ) = build_final_wavelet_source_codes(
        base_model_type=(
            "prior_after_DTW_global_fallback"
        ),
        local_fallback_alpha=(
            final_source_alpha
        ),
    )

    mask_sum = np.zeros(
        n_time,
        dtype=int,
    )

    for source_mask in source_masks.values():
        mask_sum += np.asarray(
            source_mask,
            dtype=int,
        )

    np.testing.assert_array_equal(
        mask_sum,
        np.ones(
            n_time,
            dtype=int,
        ),
    )

    allowed_codes = {
        FINAL_SOURCE_TV_CANDIDATE,
        FINAL_SOURCE_Q_CONSTRAINED,
        FINAL_SOURCE_STATIONARY_PRIOR,
        FINAL_SOURCE_LOCAL_FALLBACK_PRIOR,
        FINAL_SOURCE_LOCAL_BLEND,
    }

    assert set(
        np.unique(source_code).tolist()
    ).issubset(allowed_codes)


if __name__ == "__main__":
    test_global_prior_fallback_model_type_is_supported()
    test_global_prior_fallback_source_codes_do_not_crash()
    test_all_final_source_codes_remain_mutually_exclusive()
    print("✅ All 3 global prior fallback tests PASSED successfully!")
