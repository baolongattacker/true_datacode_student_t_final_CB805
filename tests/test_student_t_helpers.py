# -*- coding: utf-8 -*-
"""Student-t 基础函数的无第三方测试框架单元测试。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.wavelet_inversion_robust import (
    _effective_sample_size,
    _mad_scale,
    _student_t_data_loss,
    _student_t_raw_weights,
    _student_t_weights,
)


def main() -> None:
    nu = 10.0
    sigma = 2.0

    residual = np.array([0.0, 1.0, -1.0, 4.0, -4.0])
    raw = _student_t_raw_weights(residual, sigma=sigma, nu=nu)

    assert np.isclose(raw[0], 1.0)
    assert np.isclose(raw[1], raw[2])
    assert np.isclose(raw[3], raw[4])
    assert raw[0] > raw[1] > raw[3]

    huge = _student_t_raw_weights(
        np.array([1.0e12]),
        sigma=1.0,
        nu=nu,
    )[0]
    assert huge < 1.0e-20

    clipped = _student_t_weights(
        np.array([1.0e12]),
        sigma=1.0,
        nu=nu,
        weight_floor=1.0e-3,
    )[0]
    assert np.isclose(clipped, 1.0e-3)

    scale = _mad_scale(
        np.array([-1.0, 0.0, 1.0]),
        scale_floor=0.1,
    )
    assert np.isclose(scale, 1.4826)

    scale_zero = _mad_scale(
        np.zeros(10),
        scale_floor=0.25,
    )
    assert np.isclose(scale_zero, 0.25)

    n_eff = _effective_sample_size(np.ones(7))
    assert np.isclose(n_eff, 7.0)

    e = np.array([-2.0, -0.5, 0.5, 2.0])
    loss_pos = _student_t_data_loss(e, sigma=1.5, nu=nu)
    loss_neg = _student_t_data_loss(-e, sigma=1.5, nu=nu)
    assert np.isclose(loss_pos, loss_neg)

    # nu -> infinity 时趋近 0.5 * ||e||^2。
    l2_half = 0.5 * float(np.sum(e ** 2))
    st_large_nu = _student_t_data_loss(e, sigma=1.5, nu=1.0e9)
    assert np.isclose(st_large_nu, l2_half, rtol=1.0e-8, atol=1.0e-10)

    print("Student-t helper tests PASSED")


if __name__ == "__main__":
    main()
