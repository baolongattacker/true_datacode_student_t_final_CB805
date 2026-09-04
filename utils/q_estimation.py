import numpy as np


import numpy as np


def estimate_global_Q_and_rebuild_wavelets(
    W,
    dt,
    f_dom,
    f_min=10.0,
    f_max=50.0,
    ref_range=(0.1, 0.2),
    deep_range=(0.6, 0.8),
    Q_min=20.0,
    Q_max=150.0,
    freeze_above_ref=True,
):
    """
    Estimate one global Q value from inverted time-varying wavelets
    and rebuild a Q-constrained wavelet matrix.

    W: shape = (N, L)
    dt: sampling interval in seconds
    f_dom: reference/dominant frequency
    """

    W = np.asarray(W, dtype=float)

    if W.ndim != 2:
        raise ValueError("W 必须是二维矩阵，shape = (N, L)。")

    if dt <= 0:
        raise ValueError("dt 必须大于 0。")

    if f_dom <= 0:
        raise ValueError("f_dom 必须大于 0。")

    N, L = W.shape
    freqs = np.fft.rfftfreq(L, dt)
    nyquist = 0.5 / dt

    f_min = max(float(f_min), freqs[1] if len(freqs) > 1 else 0.0)
    f_max = min(float(f_max), nyquist)

    if f_max <= f_min:
        raise ValueError("f_max 必须大于 f_min，且不能超过 Nyquist。")

    ref_start = int(N * ref_range[0])
    ref_end = int(N * ref_range[1])
    deep_start = int(N * deep_range[0])
    deep_end = int(N * deep_range[1])

    ref_start = max(0, min(ref_start, N - 1))
    ref_end = max(ref_start + 1, min(ref_end, N))
    deep_start = max(0, min(deep_start, N - 1))
    deep_end = max(deep_start + 1, min(deep_end, N))

    ref_idx = int(0.5 * (ref_start + ref_end))
    deep_idx = int(0.5 * (deep_start + deep_end))
    delta_t = (deep_idx - ref_idx) * dt

    if delta_t <= 0:
        raise ValueError("deep_range 必须晚于 ref_range。")

    w_ref = np.mean(W[ref_start:ref_end], axis=0)
    w_deep = np.mean(W[deep_start:deep_end], axis=0)

    if not np.all(np.isfinite(w_ref)) or not np.all(np.isfinite(w_deep)):
        raise ValueError("参考子波或深层子波中存在 NaN/Inf。")

    if np.linalg.norm(w_ref) < 1e-12 or np.linalg.norm(w_deep) < 1e-12:
        raise ValueError("参考子波或深层子波能量过低，无法估计 Q。")

    taper = np.hanning(L)
    s_ref = np.abs(np.fft.rfft(w_ref * taper))
    s_deep = np.abs(np.fft.rfft(w_deep * taper))

    eps = 1e-12
    log_ratio = np.log((s_deep + eps) / (s_ref + eps))

    mask = (freqs >= f_min) & (freqs <= f_max)

    f_valid = freqs[mask]
    y_valid = log_ratio[mask]

    valid = np.isfinite(y_valid)

    f_valid = f_valid[valid]
    y_valid = y_valid[valid]

    if len(f_valid) < 3:
        raise ValueError("有效频率采样点太少，无法稳定估计 Q。")

    slope, intercept = np.polyfit(f_valid, y_valid, 1)

    if slope >= 0:
        raise ValueError(
            "谱比异常：深层高频未衰减，无法得到正 Q。"
        )

    Q_raw = -np.pi * delta_t / slope
    Q_global = float(np.clip(Q_raw, Q_min, Q_max))

    print(f"Estimated Global Q = {Q_raw:.2f}, clipped = {Q_global:.2f}")

    W_q = np.zeros_like(W)
    w_ref_spec = np.fft.rfft(w_ref)

    for i in range(N):
        tau_i = (i - ref_idx) * dt

        if freeze_above_ref and tau_i <= 0:
            W_q[i, :] = w_ref.copy()
            continue

        tau_i = max(0.0, tau_i)

        atten = np.exp(-np.pi * freqs * tau_i / Q_global)

        phase = np.exp(
            -1j * (2.0 * freqs * tau_i / Q_global)
            * np.log((freqs + 1e-6) / f_dom)
        )

        w_spec = w_ref_spec * atten * phase
        w_rebuilt = np.fft.irfft(w_spec, n=L)

        # Normalize energy of rebuilt wavelet to match the original W[i, :] energy.
        # This preserves the shape and phase effects of Q attenuation/dispersion,
        # while matching the gained (AGC/compensated) amplitude of processed seismic data.
        norm_orig = np.linalg.norm(W[i, :])
        norm_rebuilt = np.linalg.norm(w_rebuilt)
        if norm_rebuilt > 1e-12:
            w_rebuilt = w_rebuilt * (norm_orig / norm_rebuilt)

        W_q[i, :] = w_rebuilt

    return Q_global, W_q
