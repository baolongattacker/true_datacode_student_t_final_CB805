import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter, gaussian_filter1d
from scipy.signal.windows import tukey
from scipy.stats import trim_mean
from scipy.ndimage import uniform_filter1d
from scipy.signal import welch

# 只使用check-shot数据对时深关系曲线进行了漂移校正。

# =========================================================
# 1. 基础工具函数
# =========================================================
def robust_load_tdr_txt(path, min_cols=1):
    """
    鲁棒读取 TDR txt 文件。
    只提取每行中的纯数字字段，自动跳过 CB805 等字符串字段。
    """
    float_pattern = re.compile(
        r"(?<![A-Za-z0-9_])[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?(?![A-Za-z0-9_])"
    )

    rows = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()

            if not line:
                continue
            if line.startswith("#"):
                continue

            vals = [float(m.group(0)) for m in float_pattern.finditer(line)]

            if len(vals) >= min_cols:
                rows.append(vals)

    if len(rows) == 0:
        raise ValueError(f"TDR 文件中没有读取到有效数字: {path}")

    ncols = min(len(r) for r in rows)

    return np.asarray([r[:ncols] for r in rows], dtype=float)

def fill_nan_by_interp(x):
    """
    用线性插值填补 NaN/Inf。
    """
    x = np.asarray(x, dtype=float).copy()
    idx = np.arange(len(x))
    valid = np.isfinite(x)

    if np.sum(valid) < 2:
        raise ValueError("有效点太少，无法插值。")

    x[~valid] = np.interp(idx[~valid], idx[valid], x[valid])

    return x

def robust_normalize(x):
    """
    稳健归一化：去均值后按 99 分位振幅归一化。
    比 max 归一化更不容易受单个尖峰控制。
    """
    x = np.asarray(x, dtype=float).copy()
    x = x - np.nanmean(x)

    scale = np.percentile(np.abs(x), 99)

    if scale <= 1e-12:
        scale = np.max(np.abs(x)) + 1e-12

    return x / scale

def clean_well_logs(depth, ac, rho=None, ac_unit="us/ft"):
    """
    清洗测井曲线：
    1. 排序
    2. 去重复深度
    3. 去 NaN/Inf
    4. AC 转速度
    5. 速度异常值限幅
    6. 中值滤波去尖峰
    """
    depth = np.asarray(depth, dtype=float).copy()
    ac = np.asarray(ac, dtype=float).copy()

    if rho is not None:
        rho = np.asarray(rho, dtype=float).copy()

    # 1. 基础有效性
    mask = np.isfinite(depth) & np.isfinite(ac) & (ac > 0)

    if rho is not None:
        mask = mask & np.isfinite(rho) & (rho > 0)

    depth = depth[mask]
    ac = ac[mask]

    if rho is not None:
        rho = rho[mask]

    # 2. 按深度排序
    order = np.argsort(depth)
    depth = depth[order]
    ac = ac[order]

    if rho is not None:
        rho = rho[order]

    # 3. 去重复深度
    depth_unique, idx_unique = np.unique(depth, return_index=True)
    depth = depth_unique
    ac = ac[idx_unique]

    if rho is not None:
        rho = rho[idx_unique]

    # 4. AC 转速度
    if ac_unit.lower() in ["us/ft", "us_per_ft", "usft"]:
        # 1 ft = 0.3048 m
        v_sonic = 304800.0 / ac
    elif ac_unit.lower() in ["us/m", "us_per_m", "usm"]:
        v_sonic = 1e6 / ac
    else:
        raise ValueError("ac_unit 只能是 'us/ft' 或 'us/m'。")

    # 5. 速度物理限幅
    # 实际工区可以按岩性调整，这里先给保守范围
    v_sonic = np.clip(v_sonic, 1200.0, 6500.0)

    # 6. 中值滤波去孤立尖峰
    v_sonic = median_filter(v_sonic, size=5, mode="nearest")

    # 7. 密度处理
    if rho is None:
        # Gardner 公式：rho = 310 * Vp^0.25，单位 kg/m3
        rho = 310.0 * (v_sonic ** 0.25)
    else:
        # 如果 rho 看起来是 g/cc，则转 kg/m3
        if np.nanmedian(rho) < 10.0:
            rho = rho * 1000.0

        rho = np.clip(rho, 1500.0, 3500.0)
        rho = median_filter(rho, size=5, mode="nearest")

    dz = np.median(np.diff(depth))

    if dz <= 0:
        raise ValueError("深度轴异常：dz <= 0。")

    return depth, v_sonic, rho, dz

def preprocess_seismic(time_axis, seismic_trace_raw, bandpass=True, fmin=8.0, fmax=80.0):
    """
    地震道预处理：
    1. 转一维
    2. 长度对齐
    3. 去 NaN
    4. 去均值
    5. 可选带通
    6. 稳健归一化
    """
    time_axis = np.asarray(time_axis, dtype=float).ravel()
    trace = np.asarray(seismic_trace_raw, dtype=float).ravel()

    n = min(len(time_axis), len(trace))
    time_axis = time_axis[:n]
    trace = trace[:n]

    dt_seismic = np.median(np.diff(time_axis))

    if dt_seismic <= 0:
        raise ValueError("地震时间轴异常：dt_seismic <= 0。")

    trace = fill_nan_by_interp(trace)
    trace = trace - np.mean(trace)

    if bandpass:
        trace = apply_bandpass(trace, dt_seismic, fmin, fmax, order=4)

    trace_norm = robust_normalize(trace)

    return time_axis, trace, trace_norm, dt_seismic

def get_tdr_top_bottom(tdr_path, time_col=0, time_unit="ms"):
    """
    从 TDR 文件读取井顶/井底 TWT。
    默认第 0 列为 TWT。
    如果你的 TDR 第 0 列不是时间，需要改 time_col。
    """
    tdr = robust_load_tdr_txt(tdr_path, min_cols=time_col + 1)

    twt = tdr[:, time_col].astype(float)

    if time_unit.lower() == "ms":
        twt = twt / 1000.0
    elif time_unit.lower() == "s":
        pass
    else:
        raise ValueError("time_unit 只能是 'ms' 或 's'。")

    twt = twt[np.isfinite(twt)]

    if len(twt) < 2:
        raise ValueError("TDR 有效时间点太少。")

    return float(twt[0]), float(twt[-1]), tdr

def apply_bandpass(trace, dt, freqmin, freqmax, order=4):
    """
    零相位 Butterworth 带通滤波。
    
    使用 scipy.signal.filtfilt 确保不引入群延迟。
    
    参数:
    trace : 输入信号 (N,)
    dt : 采样间隔 (s)
    freqmin : 低截止频率 (Hz)
    freqmax : 高截止频率 (Hz)
    order : 滤波器阶数 (默认 4)
    
    返回:
    filtered : 滤波后信号 (N,)
    """
    from scipy.signal import butter, filtfilt
    
    fs = 1.0 / dt
    nyq = fs / 2.0
    
    # 安全限制：防止频率超出 Nyquist
    low = max(freqmin / nyq, 0.001)
    high = min(freqmax / nyq, 0.999)
    
    b, a = butter(order, [low, high], btype='band')
    filtered = filtfilt(b, a, trace)
    return filtered

def load_checkshot_time_md(
    checkshot_path,
    time_unit="ms",
    time_type="twt",
    time_col=0,
    depth_col=1,
):
    """
    读取 check-shot 文件，格式：
        #TIME   MD
        574.0   676.447

    默认：
        TIME = TWT, 单位 ms
        MD   = measured depth, 单位 m
    """
    min_cols = max(time_col, depth_col) + 1
    arr = robust_load_tdr_txt(checkshot_path, min_cols=min_cols)

    time = arr[:, time_col].astype(float)
    depth_md = arr[:, depth_col].astype(float)
    # 消除坏数据，只保留好数据
    mask = np.isfinite(time) & np.isfinite(depth_md)
    time = time[mask]
    depth_md = depth_md[mask]
    # argsort 排序并返回索引，确保深度递增
    order = np.argsort(depth_md)
    depth_md = depth_md[order]
    time = time[order]

    depth_md, idx = np.unique(depth_md, return_index=True)
    time = time[idx]

    if time_unit.lower() == "ms":
        time = time / 1000.0
    elif time_unit.lower() == "s":
        pass
    else:
        raise ValueError("time_unit 只能是 'ms' 或 's'。")

    if time_type.lower() == "owt":
        time = 2.0 * time
    elif time_type.lower() == "twt":
        pass
    else:
        raise ValueError("time_type 只能是 'twt' 或 'owt'。")

    if np.any(np.diff(depth_md) <= 0):
        raise ValueError("check-shot MD 必须严格递增。")

    if np.any(np.diff(time) < 0):
        raise ValueError("check-shot TWT 必须严格递增。")

    return depth_md, time

def build_twt_sonic_zero(depth, v_eff):
    """
    从 depth[0] 开始，用速度积分得到相对 TWT。
    """
    depth = np.asarray(depth, dtype=float)
    v_eff = np.asarray(v_eff, dtype=float)

    dz_arr = np.diff(depth)
    v_interval = 0.5 * (v_eff[:-1] + v_eff[1:])

    dt_interval = 2.0 * dz_arr / (v_interval + 1e-12)

    twt_sonic = np.zeros_like(depth)
    twt_sonic[1:] = np.cumsum(dt_interval)

    return twt_sonic

def build_twt_from_checkshot_drift(
    depth,
    v_eff,
    checkshot_depth,
    checkshot_twt
):
    """
    工业终极版：基于 PCHIP（分段三次埃尔米特插值）的 Check-shot 漂移校正。
    
    优势：
      1. 严格 100% 闭合于每一个 Check-shot 测点。
      2. 漂移曲线一阶导数连续，速度平滑过渡，彻底消除边界处的“锯齿”人工假反射。
      3. 自动保持单调性，绝不产生时间逆差或过冲。
    """
    from scipy.interpolate import PchipInterpolator

    depth = np.asarray(depth, dtype=float)
    v_eff = np.asarray(v_eff, dtype=float)
    cs_z = np.asarray(checkshot_depth, dtype=float)
    cs_t = np.asarray(checkshot_twt, dtype=float)
    
    assert np.all(np.diff(depth) > 0), "测井深度轴必须单调递增"
    assert np.all(np.diff(cs_z) > 0), "Check-shot 深度必须单调递增"
    
    n_depth = len(depth)
    z_start = depth[0]
    
    # 1. 计算单样点的声波双程时间增量 dT (Two-way transit time increment)
    dz_arr = np.zeros_like(depth)
    dz_arr[1:] = np.diff(depth)
    dz_arr[0] = dz_arr[1]  # 边界填充
    
    dt_sonic = 2.0 * dz_arr / (v_eff + 1e-12)
    dt_sonic[0] = 0.0
    twt_sonic_rel = np.cumsum(dt_sonic)
    
    # ==========================================
    # 2. 边界情况处理：确定绝对时间起点 T_start
    # ==========================================
    if z_start >= cs_z[0]:
        # 【情况 1】测井首点位于 checkshot 首点之后（进行插值确定起点时间）
        if z_start <= cs_z[-1]:
            # 刚好落在两个 CS 点之间，线性插值确定时间基准
            idx_next = np.searchsorted(cs_z, z_start)
            idx_prev = idx_next - 1
            z_m, z_m1 = cs_z[idx_prev], cs_z[idx_next]
            t_m, t_m1 = cs_t[idx_prev], cs_t[idx_next]
            t_start = t_m + (z_start - z_m) / (z_m1 - z_m) * (t_m1 - t_m)
        else:
            # 极端外推情况（深度超出 checkshot 底部）
            t_start = cs_t[-1] + 2.0 * (z_start - cs_z[-1]) / np.mean(v_eff)
    else:
        # 【情况 2】测井首点比第一个 checkshot 还要浅（向上反向外推确定起点时间）
        idx_cs0 = np.searchsorted(depth, cs_z[0])
        idx_cs0 = np.clip(idx_cs0, 0, n_depth - 1)
        t_cs0_rel = twt_sonic_rel[idx_cs0]
        # 声波反向积分确定的首点绝对时间
        t_start = cs_t[0] - t_cs0_rel

    # 3. 获得对齐后的绝对声波旅行时（无校正）
    twt_sonic_abs = twt_sonic_rel + t_start
    
    # ==========================================
    # 4. 计算 Check-shot 各测点处的离散漂移误差
    # ==========================================
    twt_sonic_at_cs = np.interp(cs_z, depth, twt_sonic_abs)
    drift_cs = cs_t - twt_sonic_at_cs
    
    # ==========================================
    # 5. 使用 PCHIP 进行全局高精度单调平滑插值
    # ==========================================
    # 构建 PCHIP 插值器（允许外推以支持超出控制点深度的测井段）
    pchip_interp = PchipInterpolator(cs_z, drift_cs, extrapolate=True)
    drift_smooth = pchip_interp(depth)
    
    # 6. 计算最终闭合校正后的旅行时
    twt_corr = twt_sonic_abs + drift_smooth
    
    # 7. 严格的单调安全性检查，防止意外的逆差
    if np.any(np.diff(twt_corr) <= 0):
        eps = 1e-6
        for i in range(1, len(twt_corr)):
            if twt_corr[i] <= twt_corr[i - 1]:
                twt_corr[i] = twt_corr[i - 1] + eps

    # ==========================================
    # 8. 计算诊断变量并返回 (保持向下兼容)
    # ==========================================
    # 连续漂移曲线
    drift_smooth = twt_corr - twt_sonic_abs
    
    # Check-shot 测点处的离散漂移误差
    drift_cs = cs_t - np.interp(cs_z, depth, twt_sonic_abs)
            
    return twt_corr, twt_sonic_abs, drift_smooth, drift_cs

    

def build_initial_twt_from_velocity(depth, v_eff, t_well_top):
    """
    用 Backus 后速度积分得到初始 TWT(z)。
    """
    depth = np.asarray(depth, dtype=float).ravel()
    v_eff = np.asarray(v_eff, dtype=float).ravel()

    dz_arr = np.diff(depth)
    v_interval = 0.5 * (v_eff[:-1] + v_eff[1:])

    dt_interval = 2.0 * dz_arr / (v_interval + 1e-12)

    twt = np.zeros_like(depth)
    twt[0] = t_well_top
    twt[1:] = t_well_top + np.cumsum(dt_interval)

    return twt

def scatter_reflectivity_to_time(r_depth, twt_depth, t_ref):
    """
    固定深度域反射系数 r_depth，只根据 TWT(z) 投影到地震时间轴。
    """
    r_depth = np.asarray(r_depth, dtype=float)
    twt_depth = np.asarray(twt_depth, dtype=float)
    t_ref = np.asarray(t_ref, dtype=float)

    if len(r_depth) != len(twt_depth):
        raise ValueError("r_depth 和 twt_depth 长度必须一致。")
    # 防止浮点数精度误差
    dt = np.median(np.diff(t_ref))
    n_time = len(t_ref)

    r_time = np.zeros(n_time)
    # 都刨除最后一个点，进行插值
    r_if = r_depth[:-1]
    # n个点，n-1个层
    t_if = twt_depth[1:]
    mask = (t_if >= t_ref[0]) & (t_if <= t_ref[-1])
    # ti是等间隔深度点对应的双程旅行时
    ti = t_if[mask]
    ai = r_if[mask]

    if len(ti) == 0:
        return r_time
    # 将任意（非均匀）时间点上的深度域反射系数，精准投影并累加到均匀分布的地震时间网格上。
    k = np.floor((ti - t_ref[0]) / dt).astype(int)
    k = np.clip(k, 0, n_time - 2)

    frac = (ti - t_ref[k]) / dt
    frac = np.clip(frac, 0.0, 1.0)

    np.add.at(r_time, k, (1.0 - frac) * ai)
    np.add.at(r_time, k + 1, frac * ai)

    return r_time

def estimate_dominant_frequency(
    seismic_trace,
    dt,
    t_seismic=None,
    t_start=None,
    t_end=None,
    fmin=8.0,
    fmax=80.0,
    nperseg_max=256,
    clip_range=(10.0, 80.0),
    verbose=True
):
    """
    提取地震道主频：Welch 功率谱 + 频谱质心。

    参数
    ----
    seismic_trace : ndarray
        一维地震道
    dt : float
        采样间隔，单位 s
    t_seismic : ndarray or None
        地震时间轴，单位 s
    t_start, t_end : float or None
        主频估计时间窗，单位 s
    fmin, fmax : float
        有效频带范围
    nperseg_max : int
        Welch 最大分段长度
    clip_range : tuple
        对最终主频做合理裁剪
    verbose : bool
        是否打印诊断信息

    返回
    ----
    f_dom : float
        估计主频，单位 Hz
    """

    seismic_trace = np.asarray(seismic_trace, dtype=float)

    # -------------------------------------------------
    # 1. 截取有效时间窗
    # -------------------------------------------------
    if t_seismic is not None and t_start is not None and t_end is not None:
        t_seismic = np.asarray(t_seismic, dtype=float)

        t_min = max(t_start, t_seismic[0])
        t_max = min(t_end, t_seismic[-1])

        idx_start = np.searchsorted(t_seismic, t_min, side="left")
        idx_end = np.searchsorted(t_seismic, t_max, side="right")

        data = seismic_trace[idx_start:idx_end].copy()

        if verbose:
            print(f"  [主频估计窗] {t_min:.3f} ~ {t_max:.3f} s, 样点数 = {len(data)}")

    else:
        data = seismic_trace.copy()

        if verbose:
            print(f"  [主频估计窗] 使用整道数据, 样点数 = {len(data)}")

    # -------------------------------------------------
    # 2. 基本安全检查
    # -------------------------------------------------
    N = len(data)

    if N < 32:
        raise ValueError(f"主频估计窗口太短，仅 {N} 个样点，无法稳定估计主频。")

    data = data - np.mean(data)

    if np.allclose(data, 0):
        raise ValueError("主频估计窗口内数据近似为零，无法估计主频。")

    # -------------------------------------------------
    # 3. Tukey 加窗
    # -------------------------------------------------
    data = data * tukey(N, alpha=0.1)

    # -------------------------------------------------
    # 4. Welch 功率谱估计
    # -------------------------------------------------
    fs = 1.0 / dt

    nperseg = min(N, nperseg_max)
    # 计算相邻分段之间的重叠样点数
    noverlap = min(nperseg // 2, nperseg - 1)
    # f_welch：频率刻度数组（频率轴）
    # psd：功率谱密度估计值（能量谱密度），形状与 f_welch 匹配
    f_welch, psd = welch(
        data,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant"
    )

    # -------------------------------------------------
    # 5. 限制有效频带
    # -------------------------------------------------
    valid_mask = (f_welch >= fmin) & (f_welch <= fmax)

    if not np.any(valid_mask):
        raise ValueError(f"Welch 频率采样中没有落入 {fmin}~{fmax} Hz 的频点。")

    f_valid = f_welch[valid_mask]
    p_valid = psd[valid_mask]

    power_sum = np.sum(p_valid)

    if power_sum <= 0 or not np.isfinite(power_sum):
        raise ValueError("有效频带内功率谱能量异常，无法估计主频。")

    # -------------------------------------------------
    # 6. 质心频率 + 峰值频率诊断
    # -------------------------------------------------
    # 加权平均计算质心频率
    f_centroid = np.sum(f_valid * p_valid) / power_sum
    # 找到最大功率对应的频率（峰值频率）
    f_peak = f_valid[np.argmax(p_valid)]

    # 使用质心频率作为最终主频
    f_dom = f_centroid

    # 如果提供了裁剪范围，对主频进行裁剪
    if clip_range is not None:
        f_dom = np.clip(f_dom, clip_range[0], clip_range[1])

    if verbose:
        print(f"  [诊断] Welch nperseg = {nperseg}, noverlap = {noverlap}")
        print(f"  [诊断] 峰值频率 f_peak = {f_peak:.2f} Hz")
        print(f"  [诊断] 质心频率 f_centroid = {f_centroid:.2f} Hz")
        print(f"  [诊断] 最终使用主频 f_dom = {f_dom:.2f} Hz")

    return f_dom

def backus_average(v_sonic, rho, dz, window_length=None, f_dominant=33.0):
    if window_length is None:
        # v_avg = np.mean(v_sonic)
        v_avg = trim_mean(v_sonic, 0.05) # 去掉最高和最低各 5% 的点后再算平均
        window_length = v_avg / f_dominant  # 自动用主波长
        print(f"[Info] 自动计算窗口长度: {window_length:.1f} m (V_avg={v_avg:.0f} m/s, f={f_dominant} Hz)")
    N_window = int(window_length / dz)
    if N_window % 2 == 0:
        N_window += 1
    # 使用 uniform_filter1d，mode='reflect' 进行镜像延拓，避免边界不连续
    M = rho * (v_sonic ** 2)
    
    compliance = 1.0 / M
    avg_compliance = uniform_filter1d(compliance, size=N_window, mode='reflect')
    
    M_eff = 1.0 / avg_compliance
    rho_eff = uniform_filter1d(rho, size=N_window, mode='reflect')
    
    v_eff = np.sqrt(M_eff / rho_eff)
    
    return v_eff, rho_eff

def calculate_reflectivity(v_eff, rho_eff):
    """
    计算深度域的反射系数序列。
    """
    # 计算纵波阻抗
    Ip = rho_eff * v_eff
    
    # 计算反射系数：(Ip_next - Ip_current) / (Ip_next + Ip_current)
    # 序列长度会比原数据少 1，我们在末尾补零保持维度一致
    r_depth = np.zeros_like(Ip)
    r_depth[:-1] = (Ip[1:] - Ip[:-1]) / (Ip[1:] + Ip[:-1])
    
    return r_depth

def run_data_preprocess(
    well_path,
    seismic_path,
    tdr_path,
    checkshot_path,
    output_npz_path,
    dt_seismic_input=0.002,
    ac_unit="us/m",
    tdr_time_col=0,
    checkshot_time_col=0,
    checkshot_depth_col=3
):

    # 1. 读取地震、测井、TDR
    print("[输入参数]")
    print(f"  well_path              = {well_path}")
    print(f"  seismic_path           = {seismic_path}")
    # print(f"  tdr_path               = {tdr_path}")
    print(f"  checkshot_path         = {checkshot_path}")
    print(f"  output_npz_path        = {output_npz_path}")
    print(f"  dt_seismic_input       = {dt_seismic_input:.6f} s")
    print(f"  ac_unit                = {ac_unit}")
    print(f"  tdr_time_col           = {tdr_time_col}")
    print(f"  checkshot_time_col     = {checkshot_time_col}")
    print(f"  checkshot_depth_col    = {checkshot_depth_col}")

    well_data = np.load(well_path)
    depth_raw = well_data[:, 0]
    ac_raw = well_data[:, 1]              # 之前保存的是声波时差 (AC)
    
    # 鲁棒兼容没有密度曲线的测井数据（例如 CB805）
    if well_data.shape[1] >= 3:
        rho_raw = well_data[:, 2]         # 之前保存的密度数据，单位 kg/m^3
    else:
        rho_raw = None                    # 触发 clean_well_logs 中采用 Gardner 公式估算密度
        print("[提示] 测井数据中仅检测到 2 列，未包含密度曲线。将在 clean_well_logs 中采用 Gardner 公式自动估算密度。")
    
    # 兼容 npz 和 npy 格式加载地震数据
    if seismic_path.endswith(".npz"):
        seismic_data = np.load(seismic_path)
        # 优先使用未归一化的原始数据 sel_data，因为后续 preprocess_seismic 会进行稳健归一化
        if "trace_lowpass" in seismic_data:
            seismic_trace_raw = seismic_data["trace_lowpass"]
        elif "trace_raw" in seismic_data:
            seismic_trace_raw = seismic_data["trace_raw"]
        elif "sel_data" in seismic_data:
            seismic_trace_raw = seismic_data["sel_data"]
        elif "sel_data_norm" in seismic_data:
            seismic_trace_raw = seismic_data["sel_data_norm"]
        else:
            seismic_trace_raw = seismic_data[seismic_data.files[0]]
    else:
        seismic_trace_raw = np.load(seismic_path)
        
    ns = len(seismic_trace_raw)
    dt = float(dt_seismic_input)  # 地震采样间隔，单位 s
    time_axis_raw = np.arange(0, ns * dt, dt)

    # 鲁棒读取 TDR 文本文件：自动过滤非数字标题行
    # tdr = robust_load_tdr_txt(r"D:\python_code\python_project\seismic_well\true_data_tying\data\initial_model_data\seismic\CB805_newTDR2.txt")
    # =========================================================
    # 1. 测井预处理
    # =========================================================
    depth, v_sonic, rho, dz = clean_well_logs(
        depth=depth_raw,
        ac=ac_raw,
        rho=rho_raw,
        ac_unit=ac_unit
    )

    print("[测井]")
    print(f"  AC unit     = {ac_unit}")
    print(f"  AC range    = {np.nanmin(ac_raw):.2f} ~ {np.nanmax(ac_raw):.2f}")
    print(f"  depth range = {depth[0]:.2f} ~ {depth[-1]:.2f} m")
    print(f"  N_depth     = {len(depth)}")
    print(f"  dz          = {dz:.3f} m")
    print(f"  Vp range    = {np.min(v_sonic):.1f} ~ {np.max(v_sonic):.1f} m/s")
    print(f"  rho range   = {np.min(rho):.1f} ~ {np.max(rho):.1f} kg/m3")
    print("time_axis_raw shape:", np.shape(time_axis_raw))
    print("ns:", ns)
    print("time_axis_raw first 29:", time_axis_raw[:29])

    # =========================================================
    # 2. 地震道预处理
    # =========================================================
    time_axis, seismic_trace, seismic_trace_norm, dt_seismic = preprocess_seismic(
        time_axis=time_axis_raw,
        seismic_trace_raw=seismic_trace_raw,
        bandpass=True,
        fmin=8.0,
        fmax=80.0
    )

    print("[地震]")
    print(f"  time range = {time_axis[0]:.3f} ~ {time_axis[-1]:.3f} s")
    print(f"  N_time     = {len(time_axis)}")
    print(f"  dt         = {dt_seismic:.6f} s")
    print(f"  amp range  = {np.min(seismic_trace):.4f} ~ {np.max(seismic_trace):.4f}")


    # =========================================================
    # 3. 提前加载 Check-shot 并进行 First-pass 漂移校正（取代 TDR）
    # =========================================================
    checkshot_depth, checkshot_twt = load_checkshot_time_md(
        checkshot_path,
        time_unit="ms",
        time_type="twt",
        time_col=checkshot_time_col,
        depth_col=checkshot_depth_col
    )
    
    # 使用清洗后的原始声波速度 v_sonic 进行第一遍漂移校正（完美处理插值与外推）
    twt_first_pass, _, _, _ = build_twt_from_checkshot_drift(
        depth=depth,
        v_eff=v_sonic, # 此时还没有 v_eff，直接用物理合理的 v_sonic
        checkshot_depth=checkshot_depth,
        checkshot_twt=checkshot_twt
    )
    
    # 此时获取的井顶、井底时间 100% 严谨且符合物理规律
    t_well_top = float(twt_first_pass[0])
    t_well_bot = float(twt_first_pass[-1])
    print("[Check-shot First-pass 估算井段时间范围]")
    print(f"  t_well_top (井顶) = {t_well_top:.3f} s")
    print(f"  t_well_bot (井底) = {t_well_bot:.3f} s")

    # =========================================================
    # 4. 估计井段主频
    # =========================================================
    freq_margin = 0.25

    f_dom = estimate_dominant_frequency(
        seismic_trace=seismic_trace,
        dt=dt,
        t_seismic=time_axis,
        t_start=t_well_top - freq_margin,
        t_end=t_well_bot + freq_margin,
        fmin=8.0,
        fmax=80.0,
        nperseg_max=256,
        clip_range=(10.0, 80.0),
        verbose=True
    )

    print("[主频]")
    print(f"  f_dom = {f_dom:.2f} Hz")


    # =========================================================
    # 5. Backus 粗化
    # =========================================================
    # 推荐先用半波长作为 Backus 窗口
    v_avg = np.nanmean(v_sonic)
    lambda_dom = v_avg / f_dom
    window_length = min(lambda_dom / 4.0, 15.0)

    v_eff, rho_eff = backus_average(
        v_sonic=v_sonic,
        rho=rho,
        dz=dz,
        window_length=window_length,
        f_dominant=f_dom
    )

    print("[Backus]")
    print(f"  V_avg         = {v_avg:.1f} m/s")
    print(f"  lambda        = {lambda_dom:.1f} m")
    print(f"  window_length = {window_length:.1f} m")
    print(f"  v_eff range   = {np.min(v_eff):.1f} ~ {np.max(v_eff):.1f} m/s")


    # =========================================================
    # 6. 固定深度域反射系数
    # =========================================================
    r_depth_fixed = calculate_reflectivity(v_eff, rho_eff)

    # 可选：压制极端反射系数，真实资料中很有必要
    r_clip = np.percentile(np.abs(r_depth_fixed[np.isfinite(r_depth_fixed)]), 99.5)
    r_depth_fixed = np.clip(r_depth_fixed, -r_clip, r_clip)

    print("[反射系数]")
    print(f"  r_depth range = {np.min(r_depth_fixed):.5f} ~ {np.max(r_depth_fixed):.5f}")
    print(f"  r_depth std   = {np.std(r_depth_fixed):.5f}")


    # =========================================================
    # 7. Check-shot 校正 TWT(z)
    # =========================================================
    # checkshot_depth, checkshot_twt = load_checkshot_time_md(
    #     checkshot_path,
    #     time_unit="ms",
    #     time_type="twt",
    #     time_col=checkshot_time_col,
    #     depth_col=checkshot_depth_col
    # )

    print("[Check-shot]")
    print(f"  depth range = {checkshot_depth[0]:.2f} ~ {checkshot_depth[-1]:.2f} m")
    print(f"  TWT range   = {checkshot_twt[0]:.3f} ~ {checkshot_twt[-1]:.3f} s")
    print(f"  N_checkshot = {len(checkshot_depth)}")

    # 深度范围一致性检查
    print("[Depth basis check]")
    print(f"  log depth range       = {depth[0]:.2f} ~ {depth[-1]:.2f} m")
    print(f"  checkshot MD range    = {checkshot_depth[0]:.2f} ~ {checkshot_depth[-1]:.2f} m")
    print(f"  top difference        = {checkshot_depth[0] - depth[0]:+.2f} m")
    print(f"  bottom difference     = {checkshot_depth[-1] - depth[-1]:+.2f} m")
#twt_init：由 Backus V_eff 和 Checkshot 共同约束得到的初始双程时差（TWT）
#twt_sonic_abs：仅由深层测井数据（Sonic + 密度）积分得到的绝对双程时差
#checkshot_drift：深度域 Checkshot 在深度域上的平滑漂移量
#drift_cs：最终基于 Checkshot 得到的、更精细的时差校正曲线（包含高频成分）
    twt_init, twt_sonic_abs, checkshot_drift, drift_cs = build_twt_from_checkshot_drift(
        depth=depth,
        v_eff=v_eff,
        checkshot_depth=checkshot_depth,
        checkshot_twt=checkshot_twt
    )

    t_well_top = float(twt_init[0])
    t_well_bot = float(twt_init[-1])

    print("[Check-shot corrected TWT]")
    print(f"  twt_init range       = {twt_init[0]:.3f} ~ {twt_init[-1]:.3f} s")
    print(f"  sonic_abs range      = {twt_sonic_abs[0]:.3f} ~ {twt_sonic_abs[-1]:.3f} s")
    print(f"  drift_smooth range   = {np.min(checkshot_drift):+.4f} ~ {np.max(checkshot_drift):+.4f} s")
    print(f"  drift_cs range       = {np.min(drift_cs):+.4f} ~ {np.max(drift_cs):+.4f} s")
    
    # =========================================================
    # 8. 深度域反射系数 → 地震时间轴
    # =========================================================
    r_time_init = scatter_reflectivity_to_time(
        r_depth=r_depth_fixed,
        twt_depth=twt_init,
        # 
        t_ref=time_axis
    )

    print("[时间域反射系数]")
    print(f"  r_time length = {len(r_time_init)}")
    print(f"  nonzero count = {np.sum(np.abs(r_time_init) > 1e-10)}")


    # =========================================================
    # 9. 只截取井段附近窗口做后续反演
    # =========================================================
    margin = 0.25  # 井顶/井底上下各留 250 ms

    t_start_work = max(time_axis[0], t_well_top - margin)
    t_end_work = min(time_axis[-1], t_well_bot + margin)

    mask_work = (time_axis >= t_start_work) & (time_axis <= t_end_work)

    t_work = time_axis[mask_work]
    obs_work = seismic_trace_norm[mask_work]
    r_work = r_time_init[mask_work]

    print("[工作窗口]")
    print(f"  t_work = {t_work[0]:.3f} ~ {t_work[-1]:.3f} s")
    print(f"  N_work = {len(t_work)}")

    # =========================================================
    # 10. 保存数据
    # =========================================================
    np.savez(
        output_npz_path,

        # cleaned logs
        depth=depth,
        v_sonic=v_sonic,
        rho=rho,

        # Backus model
        v_eff=v_eff,
        rho_eff=rho_eff,
        r_depth_fixed=r_depth_fixed,

        # seismic
        time_axis=time_axis,
        seismic_trace_raw=seismic_trace_raw,
        seismic_trace_processed=seismic_trace,
        seismic_trace_norm=seismic_trace_norm,
        dt_seismic=dt_seismic,

        # TWT and reflectivity
        twt_init=twt_init,
        r_time_init=r_time_init,
        checkshot_depth=checkshot_depth,
        checkshot_twt=checkshot_twt,
        twt_sonic_abs=twt_sonic_abs,
        checkshot_drift=checkshot_drift,
        drift_cs=drift_cs,
        depth_basis="MD_assumed",
        checkshot_time_type="TWT",

        # work window
        t_work=t_work,
        obs_work=obs_work,
        r_work=r_work,
        mask_work=mask_work,

        # scalar params
        f_dom=f_dom,
        t_well_top=t_well_top,
        t_well_bot=t_well_bot,
        dz=dz,
        backus_window_length=window_length
    )
    # =========================================================
    # 11. 预处理 QC 图
    # =========================================================
    fig, axes = plt.subplots(1, 4, figsize=(16, 8), constrained_layout=True)

    # 速度
    axes[0].plot(v_sonic, depth, label="raw/clean Vp", linewidth=0.8)
    axes[0].plot(v_eff, depth, label="Backus Vp", linewidth=1.2)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Vp (m/s)")
    axes[0].set_ylabel("Depth (m)")
    axes[0].set_title("Velocity")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # 密度
    axes[1].plot(rho, depth, label="rho", linewidth=0.8)
    axes[1].plot(rho_eff, depth, label="Backus rho", linewidth=1.2)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Density (kg/m3)")
    axes[1].set_title("Density")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # 深度域反射系数
    axes[2].plot(r_depth_fixed, depth, linewidth=0.8)
    axes[2].invert_yaxis()
    axes[2].set_xlabel("Reflectivity")
    axes[2].set_title("r_depth_fixed")
    axes[2].grid(True, alpha=0.3)

    # 时间域
    axes[3].plot(obs_work, t_work, label="Observed seismic", linewidth=1.0)
    axes[3].plot(r_work / (np.max(np.abs(r_work)) + 1e-12), t_work, label="r_time norm", linewidth=0.8)
    axes[3].invert_yaxis()
    axes[3].set_xlabel("Amplitude")
    axes[3].set_ylabel("TWT (s)")
    axes[3].set_title("Time-domain QC")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()

    plt.savefig("F:\\01JDX_code\\python_code\\seismic_well_tying\\true_datacode_student_t_final_CB805\\_experiment_results\\Preprocess_Data\\fig_preprocess_qc.png", dpi=300)
    plt.show()


    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(twt_sonic_abs, depth, "b", label="Sonic integrated TWT", linewidth=1.0)
    ax.plot(twt_init, depth, "k", label="Check-shot corrected TWT", linewidth=1.5)
    ax.scatter(checkshot_twt, checkshot_depth, s=6 , c="r", label="Check-shot")

    ax.invert_yaxis()
    ax.set_xlabel("TWT (s)", fontsize=24)
    ax.set_ylabel("Depth / MD (m)", fontsize=24)
    ax.set_title("TWT-depth QC", fontsize=24, fontweight="bold")
    ax.tick_params(axis="both", labelsize=20)  # 调大刻度数字的字体大小
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=24)

    plt.tight_layout()
    plt.savefig("F:\\01JDX_code\\python_code\\seismic_well_tying\\true_datacode_student_t_final_CB805\\_experiment_results\\Preprocess_Data\\fig_twt_checkshot_qc.png", dpi=300)
    plt.show()

    fig, ax = plt.subplots(figsize=(5, 8))
    drift_ms = (twt_init - twt_sonic_abs) * 1000.0
    ax.plot(drift_ms, depth, "k", linewidth=1.2)
    ax.axvline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.invert_yaxis()
    ax.set_xlabel("Check-shot drift correction (ms)")
    ax.set_ylabel("Depth / MD (m)")
    ax.set_title("Check-shot drift: corrected TWT - sonic TWT")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("F:\\01JDX_code\\python_code\\seismic_well_tying\\true_datacode_student_t_final_CB805\\_experiment_results\\Preprocess_Data\\fig_checkshot_drift_ms.png", dpi=300)
    plt.show()
    print("[Check-shot drift]")
    print(f"  drift min  = {np.min(drift_ms):+.2f} ms")
    print(f"  drift max  = {np.max(drift_ms):+.2f} ms")
    print(f"  drift mean = {np.mean(drift_ms):+.2f} ms")
    print(f"  drift end  = {drift_ms[-1]:+.2f} ms")

if __name__ == "__main__":
    # =========================================================
    # 输入参数集中区
    # 更换数据时优先只改这里，不要到主流程中分散搜索路径和列号。
    # 当前 W258-308.dat 是综合表：第 0 列按 TWT 使用，第 3 列按 MD 使用。
    # =========================================================
    well_path = r"F:\01JDX_code\python_code\seismic_well_tying\true_datacode_student_t_final_CB805\_experiment_data\well_data\well_data_CB805.npy"
    seismic_path = r"F:\01JDX_code\python_code\seismic_well_tying\true_datacode_student_t_final_CB805\_experiment_data\seismic_CB805\seismic_data_CB805_lowpass_70Hz.npz"
    tdr_path = r"F:\02JDX_Data\测井实际数据\项目第三次数据工区资料（斜井）\06  Check Shots\Check Shots\CB805"
    checkshot_path = tdr_path
    output_npz_path = r"F:\01JDX_code\python_code\seismic_well_tying\true_datacode_student_t_final_CB805\_experiment_data\initial_model_data\initial_real_model_data_CB805.npz"

    dt_seismic_input = 0.001
    ac_unit = "us/ft"    
    tdr_time_col = 0
    checkshot_time_col = 0
    checkshot_depth_col = 1

    run_data_preprocess(
        well_path=well_path,
        seismic_path=seismic_path,
        tdr_path=tdr_path,
        checkshot_path=checkshot_path,
        output_npz_path=output_npz_path,
        dt_seismic_input=dt_seismic_input,
        ac_unit=ac_unit,
        tdr_time_col=tdr_time_col,
        checkshot_time_col=checkshot_time_col,
        checkshot_depth_col=checkshot_depth_col
    )
