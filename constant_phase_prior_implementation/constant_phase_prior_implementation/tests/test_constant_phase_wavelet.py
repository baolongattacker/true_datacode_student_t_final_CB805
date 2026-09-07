# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np

from utils.constant_phase_wavelet import (
    build_constant_phase_wavelet,
    default_phase_grid_deg,
    estimate_statistical_amplitude_spectrum,
)


def _ricker(f0: float, dt: float, length: int) -> np.ndarray:
    t = (np.arange(length) - length // 2) * dt
    a = (np.pi * f0 * t) ** 2
    w = (1.0 - 2.0 * a) * np.exp(-a)
    return w / np.max(np.abs(w))


def test_phase_grid_covers_complete_cycle_without_duplicate_endpoint():
    grid = default_phase_grid_deg()
    assert grid[0] == -180.0
    assert grid[-1] == 179.0
    assert grid.size == 360


def test_statistical_spectrum_and_wavelet_are_finite_and_centered():
    rng = np.random.default_rng(7)
    dt = 0.001
    n = 4000
    f0 = 30.0

    reflectivity = rng.normal(size=n)
    reflectivity *= rng.random(n) < 0.08

    true_wavelet = _ricker(f0=f0, dt=dt, length=129)
    seismic = np.convolve(reflectivity, true_wavelet, mode="same")
    seismic += 0.05 * np.std(seismic) * rng.normal(size=n)

    spectrum = estimate_statistical_amplitude_spectrum(
        s_obs=seismic,
        dt=dt,
        f_dom=f0,
    )

    w0 = build_constant_phase_wavelet(
        spectrum=spectrum,
        dt=dt,
        wavelet_length=129,
        phase_deg=0.0,
    )

    assert w0.shape == (129,)
    assert np.all(np.isfinite(w0))
    assert np.isclose(np.max(np.abs(w0)), 1.0)
    assert abs(int(np.argmax(np.abs(w0))) - 64) <= 2
    assert spectrum.band_high_hz > spectrum.band_low_hz
    assert 0.4 * f0 <= spectrum.spectral_peak_hz <= 1.8 * f0


def test_phase_rotation_changes_shape_but_preserves_finite_support():
    rng = np.random.default_rng(19)
    dt = 0.001
    n = 3000
    f0 = 25.0

    seismic = rng.normal(size=n)
    seismic = np.convolve(
        seismic,
        _ricker(f0=f0, dt=dt, length=101),
        mode="same",
    )

    spectrum = estimate_statistical_amplitude_spectrum(
        s_obs=seismic,
        dt=dt,
        f_dom=f0,
    )

    w0 = build_constant_phase_wavelet(
        spectrum=spectrum,
        dt=dt,
        wavelet_length=101,
        phase_deg=0.0,
    )
    w45 = build_constant_phase_wavelet(
        spectrum=spectrum,
        dt=dt,
        wavelet_length=101,
        phase_deg=45.0,
    )

    assert np.all(np.isfinite(w45))
    assert np.isclose(np.max(np.abs(w45)), 1.0)
    assert not np.allclose(w0, w45)
    assert abs(w0[0]) < 0.2
    assert abs(w0[-1]) < 0.2
    assert abs(w45[0]) < 0.2
    assert abs(w45[-1]) < 0.2
