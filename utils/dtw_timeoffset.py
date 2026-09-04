import numpy as np
try:
    from dtaidistance import dtw as _dtw_backend
except ModuleNotFoundError:  # Allow non-DTW tests and tools to import the package.
    _dtw_backend = None
from scipy.ndimage import gaussian_filter1d
import scipy.sparse as sp




def compute_dtw_window_scalar(
    twt: np.ndarray,
    dt: float,
    f_dom: float,
    mode: str = "waveform",
    base_ms: float = 8.0,
    vel_err: float = 0.03,
    T0_ms: float = 20.0,
    min_samples: int = 2,
    max_samples: int | None = None
) -> int:
    """
    为 dtaidistance 的标量 window 参数计算全局 DTW 搜索窗口。

    设计逻辑：
    1. max_phys_error 估计全记录范围内最大可能累计时差；
    2. W_cycle 根据阶段限制最大搜索半径，防止跳周；
    3. final_W = min(max_phys_error, W_cycle)；
    4. 返回整数采样点数，直接传给 dtw.warping_path(..., window=current_window)。

    mode:
        envelope : 包络 DTW，允许较大窗口，用于大尺度时移；
        waveform : 波形 DTW，限制在半周期内；
        fine     : 联合微调，限制更小。
    """
    if dt <= 0:
        raise ValueError("dt 必须大于 0")
    if f_dom <= 0:
        raise ValueError("f_dom 必须大于 0")
    if len(twt) == 0:
        raise ValueError("twt 不能为空")

    base = base_ms / 1000.0
    T0 = T0_ms / 1000.0

    max_twt = float(np.nanmax(twt))
    max_phys_error = base + vel_err * (max_twt + T0)

    if mode == "envelope":
        W_cycle = 0.9 / f_dom
    elif mode == "waveform":
        W_cycle = 0.45 / f_dom
    elif mode == "fine":
        W_cycle = 0.25 / f_dom
    else:
        raise ValueError("mode 必须是 'envelope', 'waveform', 或 'fine'")

    final_W = min(max_phys_error, W_cycle)

    W_samples = int(round(final_W / dt))
    W_samples = max(W_samples, min_samples)

    if max_samples is not None:
        W_samples = min(W_samples, int(max_samples))

    return W_samples

def calculate_nonstationary_time_shift(s_seismic, s_syn, dt, window=None):
    """
    计算真实地震道与合成地震道之间的非平稳时间偏移 f(t)。
    【架构降级】：该函数不再计算非物理的数学导数 df/dt，只提供观测残差。
    """
    s_seismic = np.array(s_seismic, dtype=np.double)
    s_syn = np.array(s_syn, dtype=np.double)
    N = len(s_syn)

    # 1. 获取 DTW 路径
    if _dtw_backend is None:
        raise ModuleNotFoundError(
            "DTW 阶段需要 dtaidistance。请在当前环境执行："
            "pip install dtaidistance"
        )
    path = _dtw_backend.warping_path(s_seismic, s_syn, window=window)
    shifts_at_j = {j: [] for j in range(N)}
    for i, j in path:
        if j < N:
            shifts_at_j[j].append((i - j) * dt)

    # 2. 提取有效节点并用线性插值填补空洞
    valid_j = []
    valid_shifts = []
    for j in range(N):
        if len(shifts_at_j[j]) > 0:
            valid_j.append(j)
            valid_shifts.append(np.mean(shifts_at_j[j]))
            
    f_t_raw = np.interp(np.arange(N), valid_j, valid_shifts)

    # 3. 强平滑（提取极低频宏观运动学趋势）
    sigma_val = max(5.0, N * 0.01) 
    f_t_smoothed = gaussian_filter1d(f_t_raw, sigma=sigma_val, mode='nearest')

    # 4. 波形对齐 (用于验证/计算CC)
    t_axis = np.arange(N) * dt
    t_warped = t_axis - f_t_smoothed 
    aligned_trace = np.interp(
        t_warped,              
        t_axis,                
        s_syn,       
        left=s_syn[0], 
        right=s_syn[-1]
    )

    # 【注意】不再返回 df_dt，只返回时移量和对齐波形
    return f_t_smoothed, aligned_trace

def traveltime_tomography_1d(
    v_k,
    depth,
    dt_depth,
    lambda_reg=5.0,
    smooth_sigma_m=25.0,
    max_dv_fraction=0.03,
    v_min=1000.0,
    v_max=6000.0
):
    """
    快速一维走时更新版本。

    物理近似：
        TWT(z) = 2 ∫ s(z) dz
        ΔT(z) = 2 ∫ Δs(z) dz
        所以：
        Δs(z) ≈ 0.5 * d(ΔT)/dz

    优点：
        1. 不构造 N×N 矩阵；
        2. 计算复杂度约 O(N)；
        3. 更适合井震标定中的低频速度修正；
        4. 通过平滑和限幅防止速度更新发散。
    """

    import numpy as np
    from scipy.ndimage import gaussian_filter1d

    v_k = np.asarray(v_k, dtype=float)
    depth = np.asarray(depth, dtype=float)
    dt_depth = np.asarray(dt_depth, dtype=float)

    if len(v_k) != len(depth) or len(v_k) != len(dt_depth):
        raise ValueError("v_k, depth, dt_depth 长度必须一致。")

    if np.any(np.diff(depth) <= 0):
        raise ValueError("depth 必须严格递增。")

    dz_med = np.median(np.diff(depth))

    # smooth_sigma_m 是米，换算成采样点
    smooth_sigma_samples = max(1.0, smooth_sigma_m / dz_med)

    # 1. 先平滑 DTW 给出的走时扰动，避免直接微分放大噪声
    dt_smooth = gaussian_filter1d(
        dt_depth,
        sigma=smooth_sigma_samples,
        mode="nearest"
    )

    # 2. 由 ΔT(z) 反推出 Δs(z)
    dT_dz = np.gradient(dt_smooth, depth)
    delta_s = 0.5 * dT_dz

    # 3. 慢度扰动继续平滑，只保留低频速度更新
    delta_s = gaussian_filter1d(
        delta_s,
        sigma=smooth_sigma_samples,
        mode="nearest"
    )

    # 4. 当前慢度
    s_k = 1.0 / v_k

    # 5. 对慢度扰动限幅，避免单轮速度剧烈变化
    #    max_dv_fraction=0.03 表示单轮速度变化大约不超过 3%
    max_delta_s = max_dv_fraction * np.abs(s_k)
    delta_s = np.clip(delta_s, -max_delta_s, max_delta_s)

    # 6. 更新慢度
    s_new = s_k + delta_s

    # 7. 慢度物理边界
    s_new = np.clip(s_new, 1.0 / v_max, 1.0 / v_min)

    # 8. 回到速度
    v_updated = 1.0 / s_new

    return v_updated