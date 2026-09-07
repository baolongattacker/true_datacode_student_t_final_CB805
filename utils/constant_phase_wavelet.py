# -*- coding: utf-8 -*-
"""
utils/constant_phase_wavelet.py

Statistical constant-phase wavelet construction.

This module estimates a smooth wavelet-amplitude spectrum from the observed
seismic trace and constructs finite-support, center-aligned constant-phase
wavelets. It intentionally introduces no experiment-level mu regularization
parameter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import welch
from scipy.signal.windows import tukey


EPS = 1e-12


@dataclass(frozen=True)
class StatisticalAmplitudeSpectrum:
    frequencies_hz: np.ndarray
    amplitude: np.ndarray
    spectral_peak_hz: float
    band_low_hz: float
    band_high_hz: float
    smooth_hz: float # 平滑核宽度
    nperseg: int # Welch 谱估计时每个滑动汉宁窗的分段点数


def _positive_scalar(value: float, name: str) -> float:
    array = np.asarray(value, dtype=float)
    if array.ndim != 0 or not np.isfinite(array) or float(array) <= 0.0:
        raise ValueError(f"{name} 必须是有限正数。")
    return float(array)


def _finite_1d(x: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(x, dtype=float).ravel()
    if array.size < 8:
        raise ValueError(f"{name} 长度至少需要 8 个采样点。")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} 包含 NaN 或 Inf。")
    return array


def _make_odd(length: int) -> int:
    length = int(length)
    if length < 3:
        raise ValueError("wavelet_length 必须 >= 3。")
    return length if length % 2 == 1 else length + 1


def _next_pow2(n: int) -> int:
    n = int(max(2, n))
    return 1 << (n - 1).bit_length()


def _cosine_band_mask(
    frequencies_hz: np.ndarray,
    pass_low_hz: float,
    pass_high_hz: float,
    transition_hz: float,
) -> np.ndarray:
    f = np.asarray(frequencies_hz, dtype=float)
    pass_low_hz = max(0.0, float(pass_low_hz))
    pass_high_hz = max(pass_low_hz, float(pass_high_hz))
    transition_hz = max(float(transition_hz), EPS)

    low_stop = max(0.0, pass_low_hz - transition_hz)
    high_stop = pass_high_hz + transition_hz

    mask = np.zeros_like(f)

    flat = (f >= pass_low_hz) & (f <= pass_high_hz)
    mask[flat] = 1.0

    rising = (f > low_stop) & (f < pass_low_hz)
    if np.any(rising):
        x = (f[rising] - low_stop) / max(pass_low_hz - low_stop, EPS)
        mask[rising] = 0.5 - 0.5 * np.cos(np.pi * x)

    falling = (f > pass_high_hz) & (f < high_stop)
    if np.any(falling):
        x = (f[falling] - pass_high_hz) / transition_hz
        mask[falling] = 0.5 + 0.5 * np.cos(np.pi * x)

    return np.clip(mask, 0.0, 1.0)


def estimate_statistical_amplitude_spectrum(
    s_obs: np.ndarray,
    dt: float,
    f_dom: float,
) -> StatisticalAmplitudeSpectrum:
    """
    Estimate a smooth statistical wavelet-amplitude spectrum from seismic.

    Assumption:
        reflectivity is sufficiently broadband that the smoothed seismic
        amplitude spectrum is a useful first-order proxy for wavelet amplitude.

    Automatic choices:
        * Welch segment length from dt and trace length.
        * smoothing bandwidth tied to f_dom.
        * useful spectral band detected around the dominant lobe.
    """
    s = _finite_1d(s_obs, "s_obs")
    dt = _positive_scalar(dt, "dt")
    f_dom = _positive_scalar(f_dom, "f_dom")

    fs = 1.0 / dt
    nyquist = 0.5 * fs
    s = s - np.mean(s)

    target_samples = int(round(1.0 / dt))
    nperseg = min(s.size, max(256, target_samples))
    nperseg = max(32, int(nperseg))
    noverlap = nperseg // 2

    frequencies_hz, power = welch(
        s,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
        scaling="spectrum",
        return_onesided=True,
    )

    amplitude_raw = np.sqrt(np.maximum(power, 0.0))

    if frequencies_hz.size < 4:
        raise ValueError("Welch 频谱采样点过少，无法估计 constant-phase prior。")

    df = float(np.median(np.diff(frequencies_hz)))
    smooth_hz = max(1.0, 0.08 * f_dom)
    sigma_bins = max(0.5, smooth_hz / max(df, EPS))

    amplitude_smooth = gaussian_filter1d(
        amplitude_raw,
        sigma=sigma_bins,
        mode="nearest",
    )

    peak_search_low = max(df, 0.45 * f_dom)
    peak_search_high = min(0.95 * nyquist, 1.80 * f_dom)
    peak_search_mask = (
        (frequencies_hz >= peak_search_low)
        & (frequencies_hz <= peak_search_high)
    )

    if not np.any(peak_search_mask):
        peak_search_mask = (
            (frequencies_hz > 0.0)
            & (frequencies_hz < 0.95 * nyquist)
        )

    candidate_indices = np.flatnonzero(peak_search_mask)
    if candidate_indices.size == 0:
        raise ValueError("无法找到可用频率范围。")

    local_peak_offset = int(np.argmax(amplitude_smooth[candidate_indices]))
    peak_index = int(candidate_indices[local_peak_offset])
    spectral_peak_hz = float(frequencies_hz[peak_index])
    peak_amplitude = float(amplitude_smooth[peak_index])

    if not np.isfinite(peak_amplitude) or peak_amplitude <= EPS:
        raise ValueError("观测地震频谱能量过低，无法估计统计子波。")

    high_noise_mask = frequencies_hz >= 0.75 * nyquist
    noise_floor = (
        float(np.median(amplitude_smooth[high_noise_mask]))
        if np.any(high_noise_mask)
        else 0.0
    )

    threshold = max(0.05 * peak_amplitude, 1.5 * noise_floor)
    threshold = min(threshold, 0.25 * peak_amplitude)

    left = peak_index
    while left > 1 and amplitude_smooth[left - 1] >= threshold:
        left -= 1

    right = peak_index
    while (
        right < amplitude_smooth.size - 2
        and amplitude_smooth[right + 1] >= threshold
    ):
        right += 1

    band_low_hz = float(frequencies_hz[left])
    band_high_hz = float(frequencies_hz[right])

    minimum_bandwidth = max(0.50 * f_dom, 4.0 * df)
    if band_high_hz - band_low_hz < minimum_bandwidth:
        band_low_hz = max(df, 0.20 * f_dom)
        band_high_hz = min(0.95 * nyquist, 3.00 * f_dom)

    transition_hz = max(2.0 * smooth_hz, 0.15 * f_dom)
    band_mask = _cosine_band_mask(
        frequencies_hz=frequencies_hz,
        pass_low_hz=band_low_hz,
        pass_high_hz=band_high_hz,
        transition_hz=transition_hz,
    )

    amplitude = amplitude_smooth * band_mask
    amplitude[0] = 0.0

    max_amp = float(np.max(amplitude))
    if max_amp <= EPS:
        raise ValueError("统计频谱经过自动频带检测后为空。")

    amplitude = amplitude / max_amp

    return StatisticalAmplitudeSpectrum(
        frequencies_hz=np.asarray(frequencies_hz, dtype=float),
        amplitude=np.asarray(amplitude, dtype=float),
        spectral_peak_hz=spectral_peak_hz,
        band_low_hz=band_low_hz,
        band_high_hz=band_high_hz,
        smooth_hz=float(smooth_hz),
        nperseg=int(nperseg),
    )


def build_constant_phase_wavelet(
    spectrum: StatisticalAmplitudeSpectrum,
    dt: float,
    wavelet_length: int,
    phase_deg: float,
) -> np.ndarray:
    """
    Construct a real finite-support center-aligned constant-phase wavelet.

    Positive frequencies receive +phase. Negative frequencies receive the
    conjugate phase implicitly through irfft, so the time-domain wavelet is real.

    A fixed Tukey taper only represents finite support cleanly; it is not a
    regularization penalty and has no experiment-level tuning parameter.
    """
    dt = _positive_scalar(dt, "dt")
    wavelet_length = _make_odd(wavelet_length)

    phase_array = np.asarray(phase_deg, dtype=float)
    if phase_array.ndim != 0 or not np.isfinite(phase_array):
        raise ValueError("phase_deg 必须是有限标量。")
    phase_deg = float(phase_array)

    nfft = _next_pow2(max(4096, 16 * wavelet_length))
    fft_frequencies = np.fft.rfftfreq(nfft, d=dt)

    amplitude = np.interp(
        fft_frequencies,
        spectrum.frequencies_hz,
        spectrum.amplitude,
        left=0.0,
        right=0.0,
    )

    phase_rad = np.deg2rad(phase_deg)
    positive_spectrum = amplitude.astype(np.complex128)
    positive_spectrum *= np.exp(1j * phase_rad)

    positive_spectrum[0] = 0.0
    if nfft % 2 == 0:
        positive_spectrum[-1] = 0.0

    w_full = np.fft.irfft(positive_spectrum, n=nfft)
    w_full = np.fft.fftshift(w_full)

    center = nfft // 2
    half = wavelet_length // 2
    w = w_full[center - half : center + half + 1].copy()

    if w.size != wavelet_length:
        raise RuntimeError("constant-phase 子波裁剪长度异常。")

    w *= tukey(wavelet_length, alpha=0.25)
    w -= np.mean(w)

    scale = float(np.max(np.abs(w)))
    if not np.isfinite(scale) or scale <= EPS:
        raise ValueError("constant-phase 子波振幅退化。")

    return w / scale


def default_phase_grid_deg() -> np.ndarray:
    """Fixed 1-degree grid over one complete phase cycle."""
    return np.arange(-180.0, 180.0, 1.0, dtype=float)
