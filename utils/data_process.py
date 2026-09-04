import numpy as np
from scipy.signal.windows import tukey
from scipy.stats import kurtosis
from scipy.ndimage import uniform_filter1d
from scipy.stats import trim_mean
from scipy.signal import welch

# def backus_average(v_sonic, rho, dz, window_length):
#     """
#     使用 Backus 平均对高频测井数据进行粗化处理。
    
#     参数:
#     v_sonic (np.array): 原始高频声波测井速度 (m/s)
#     rho (np.array): 原始测井密度 (kg/m^3)
#     dz (float): 测井数据的深度采样间隔 (m)
#     window_length (float): 对应地震波长的滑动窗口长度 (m)
    
#     返回:
#     v_eff (np.array): 粗化后的等效地震频带速度
#     rho_eff (np.array): 粗化后的等效密度
#     """
    # # 计算滑动窗口内的采样点数
    # N_window = int(window_length / dz) #一个平滑窗口包含多少测井采样点
    # if N_window % 2 == 0:
    #     N_window += 1 # 保证窗口长度为奇数，方便对称平滑
        
    # # 构造均值滤波器核
    # kernel = np.ones(N_window) / N_window #确保卷积滤波后数量级保持一致
    
    # # 1. 计算纵波模量 M
    # M = rho * (v_sonic ** 2)
    
    # # 2. 计算柔度 (M 的倒数) 并进行滑动平均
    # compliance = 1.0 / M
    # avg_compliance = np.convolve(compliance, kernel, mode='same')
    
    # # 3. 计算等效纵波模量
    # M_eff = 1.0 / avg_compliance
    
    # # 4. 计算等效密度 (直接算术平均)
    # rho_eff = np.convolve(rho, kernel, mode='same')
    
    # # 5. 计算最终的等效速度
    # v_eff = np.sqrt(M_eff / rho_eff)
    
    # # 处理边界效应（使用原始数据填充由于卷积产生的边界空缺）
    # half_win = N_window // 2
    # v_eff[:half_win] = v_sonic[:half_win]
    # v_eff[-half_win:] = v_sonic[-half_win:]
    # rho_eff[:half_win] = rho[:half_win]
    # rho_eff[-half_win:] = rho[-half_win:]
    #  # return v_eff, rho_eff
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

def depth_to_time_conversion(depth, v_eff, rho_eff, t_seismic, dt_seismic, t_well_top=0.0):
    """
    将深度域的反射系数转换到时间域，与给定的地震时间轴对齐。
    
    参照 MATLAB 实现逻辑：
    1. 计算声阻抗 Z = v * rho，再计算界面反射系数 r_if (N-1 个界面)
    2. 每个界面的 TWT 取层底时间 t_if = TWT(2:end)
    3. 输出 R 的长度 = len(t_seismic)（与地震道等长）
    4. 使用线性权重将界面反射系数分摊到左右两个相邻采样点，保留脉冲能量
    5. t_well_top 由 TDR 文件提供，确保反射系数在地震时间轴上的正确位置
    
    参数:
    depth (np.array): 测井深度序列 (m)
    v_eff (np.array): 粗化后的等效速度 (m/s)
    rho_eff (np.array): 粗化后的等效密度 (kg/m^3)
    t_seismic (np.array): 地震时间轴 (s)，决定输出 R 的长度和时间范围
    dt_seismic (float): 地震数据的时间采样间隔 (s)
    t_well_top (float): 测井顶部深度对应的 TWT (s)，来自 TDR 文件的第一个时间值
    
    返回:
    R (np.array): 时间域反射系数，长度与 t_seismic 一致
    t_z (np.array): 每个深度点对应的双程旅行时 (TWT，绝对时间)
    """
    dz = depth[1] - depth[0]
    
    # 1. 计算每个深度点的双程旅行时 TWT(z)
    # t_well_top 是测井顶部的绝对 TWT，由 TDR 文件提供
    dt_step = 2.0 * dz / v_eff
    t_z = t_well_top + np.concatenate(([0], np.cumsum(dt_step[:-1])))
    
    # 2. 计算声阻抗和界面反射系数 (N-1 个界面)
    Z_depth = v_eff * rho_eff
    Z1 = Z_depth[:-1]
    Z2 = Z_depth[1:]
    r_if = (Z2 - Z1) / (Z2 + Z1)
    
    # 3. 界面对应的 TWT 位于地层底部，即后一层的顶界面
    t_if = t_z[1:]  # 对应 MATLAB 的 t_if = TWT1(2:end)
    
    # 4. 初始化输出：与地震时间轴等长
    N_t = len(t_seismic)
    R = np.zeros(N_t)
    
    # 5. 仅保留落在地震时间范围内的界面
    mask = (t_if >= t_seismic[0]) & (t_if <= t_seismic[-1])
    ti = t_if[mask]
    ai = r_if[mask]
    
    # 6. 线性权重分摊到相邻两个采样点
    # 对每个界面，计算其左侧采样点索引 k 及线性权重 frac
    k = np.floor((ti - t_seismic[0]) / dt_seismic).astype(int)
    k = np.clip(k, 0, N_t - 2)  # 保证 0 <= k <= N_t-2
    frac = (ti - t_seismic[k]) / dt_seismic  # 线性权重 [0, 1)
    
    # 将反射系数按权重分摊到 k 和 k+1（使用 np.add.at 支持重复索引累加）
    np.add.at(R, k, (1.0 - frac) * ai)
    np.add.at(R, k + 1, frac * ai)
    
    return R, t_z

def estimate_dominant_frequency(seismic_trace, dt, t_seismic=None, t_start=None, t_end=None):
    """提取地震道主频（稳健版） - 使用质心频率 + Welch 谱估计"""
    # 1. 截取有效窗口
    if t_seismic is not None and t_start is not None and t_end is not None:
        idx_start = max(0, int((t_start - t_seismic[0]) / dt))
        idx_end   = min(len(seismic_trace) - 1, int((t_end - t_seismic[0]) / dt))
        data = seismic_trace[idx_start:idx_end+1].copy()
    else:
        data = seismic_trace.copy()
    # 2. 去直流 + Tukey 加窗
    data -= np.mean(data)
    data *= tukey(len(data), alpha=0.1)
    # 3. Welch 功率谱估计（比单次 FFT 更稳健）
    N = len(data)
    f_welch, psd = welch(data, fs=1.0/dt, nperseg=min(N, 128), noverlap=64)
    # 4. 限制在有效地震频带内
    valid_mask = (f_welch >= 8.0) & (f_welch <= 80.0)
    f_valid = f_welch[valid_mask]
    p_valid = psd[valid_mask]
    # 5. 功率谱加权质心频率（比 argmax 更稳健）
    f_dom = np.sum(f_valid * p_valid) / np.sum(p_valid)
    print(f"  [诊断] 质心主频估计: {f_dom:.2f} Hz")
    return np.clip(f_dom, 10.0, 80.0)
# 0. 辅助函数：生成一个雷克子波作为初始盲猜子波
def ricker_wavelet(f_dom, dt, length):
    """
    生成一个雷克(Ricker)子波
    f_dom: 主频 (Hz)
    dt: 采样间隔 (s)
    length: 子波长度 (s)
    """
    t = np.arange(-length/2, length/2, dt)
    y = (1.0 - 2.0 * (np.pi**2) * (f_dom**2) * (t**2)) * np.exp(-(np.pi**2) * (f_dom**2) * (t**2))
    return y

def estimate_wavelet(seismic_trace, dt_seismic, t_seismic, t_well_top, t_well_bot,
                     wavelet_length=0.2, phase_search_step=1, wavelet_type='zero'):
    """
    从实际地震道统计估计初始子波。
    
    方法：振幅谱提取 + 峰度(Kurtosis)匹配相位旋转
    1. 从井段对应的地震时间窗口截取地震数据
    2. 用 FFT 提取该窗口的振幅谱（代表地震子波的频率特征）
    3. 遍历 0°~179° 的恒相位旋转角度，对每个角度：
       将振幅谱与该相位组合 → IFFT → 得到候选子波 → 与反射系数褶积
       → 计算结果的峰度（Kurtosis）
    4. 选择峰度最大的相位角（峰度越大 → 脉冲越尖锐 → 反褶积效果越好）
    5. 用最优相位 + 振幅谱构造最终子波，截取到指定长度
    
    参数:
    seismic_trace (np.array): 完整的实际地震道
    dt_seismic (float): 地震采样间隔 (s)
    t_seismic (np.array): 地震时间轴 (s)
    t_well_top (float): 测井顶部的 TWT (s)，来自 TDR
    t_well_bot (float): 测井底部的 TWT (s)，来自 TDR
    wavelet_length (float): 期望的子波时间长度 (s)，默认 0.2s
    phase_search_step (int): 相位搜索步长 (度)，默认 1 度
    
    返回:
    wavelet (np.array): 估计出的子波
    best_phase (float): 最优相位角 (度)
    amp_spectrum (np.array): 提取的振幅谱 (用于诊断)
    """
    # ---- 1. 截取井段对应的地震时间窗口 ----
    # 在井段两端各留一点余量 (半个子波长度)
    margin = wavelet_length / 2.0
    t_start = max(t_well_top - margin, t_seismic[0])
    t_end = min(t_well_bot + margin, t_seismic[-1])
    
    idx_start = int((t_start - t_seismic[0]) / dt_seismic)
    idx_end = int((t_end - t_seismic[0]) / dt_seismic)
    idx_start = max(0, idx_start)
    idx_end = min(len(seismic_trace) - 1, idx_end)
    
    windowed_trace = seismic_trace[idx_start:idx_end + 1].copy()
    n_window = len(windowed_trace)
    
    # 施加 Tukey 窗（边缘 10% 渐变）减少截断引起的频谱泄漏
    taper = tukey(n_window, alpha=0.1)
    windowed_trace *= taper
    
    print(f"  子波估计窗口: {t_start:.3f}s ~ {t_end:.3f}s ({n_window} 个采样点)")
    
    # ---- 2. 提取振幅谱 ----
    N_fft = n_window
    freq_axis = np.fft.rfftfreq(N_fft, d=dt_seismic)
    spectrum = np.fft.rfft(windowed_trace)
    amp_spectrum = np.abs(spectrum)
    
    # 轻微平滑振幅谱 (5点滑动平均)，减少噪声毛刺
    smooth_kernel = np.ones(5) / 5.0
    amp_smoothed = np.convolve(amp_spectrum, smooth_kernel, mode='same')
    amp_smoothed = np.maximum(amp_smoothed, 0)  # 确保非负
    
    # ---- 3. 峰度匹配：搜索最优恒相位旋转角 ----
    best_phase = 0.0
    best_kurtosis = -np.inf
    
    for phase_deg in range(0, 180, phase_search_step):
        phase_rad = np.deg2rad(phase_deg)
        
        # 构造带相位旋转的频谱：A(f) * exp(j * phase)
        test_spectrum = amp_smoothed * np.exp(1j * phase_rad)
        
        # IFFT 得到候选子波
        candidate = np.fft.irfft(test_spectrum, n=N_fft)
        
        # 计算峰度 (Fisher 定义: 正态分布 kurtosis=0, 越大越尖锐)
        k = kurtosis(candidate, fisher=True)
        
        if k > best_kurtosis:
            best_kurtosis = k
            best_phase = phase_deg
    
    print(f"  最优相位角: {best_phase}°, 对应峰度: {best_kurtosis:.4f}")
    
    # ---- 4. 用最优相位构造最终子波 ----
    best_phase_rad = np.deg2rad(best_phase)
    final_spectrum = amp_smoothed * np.exp(1j * best_phase_rad)
    full_wavelet = np.fft.irfft(final_spectrum, n=N_fft)
    
    # ---- 5. 截取到指定长度，并根据相位类型对齐 ----
    n_len = int(wavelet_length / dt_seismic)
    if n_len % 2 == 0:
        n_len += 1
    n_half = n_len // 2
    
    if wavelet_type.lower() == 'zero':
        # 零相位要求严格对称，频移使其中心化
        full_wavelet = np.fft.fftshift(full_wavelet)
        center = np.argmax(np.abs(full_wavelet))
        # 从中心向两边截取对称窗口
        start = max(0, center - n_half)
        end = min(len(full_wavelet), center + n_half + 1)
        wavelet = full_wavelet[start:end]
    else:
        # 最小相位默认因果（能量集中在前端），IFFT 的零起点即前沿
        # 也可以为了平滑加一个小余量但在纯理论下直接截取起点
        wavelet = full_wavelet[:n_len]
        
        # 为了保证它能在非平稳褶积时对齐原点，往往在褶积时其相对原点在最左侧
        # 本实现由于后续非平稳褶积默认以中间对齐，为了不破坏统一性，将其补齐为同样长度并中心偏移
        
    # 归一化：使子波峰值为 1
    wavelet = wavelet / np.max(np.abs(wavelet))
    
    print(f"  子波类型: {wavelet_type.capitalize()}-Phase, 长度: {len(wavelet)} 采样点 ({len(wavelet) * dt_seismic * 1000:.1f} ms)")
    
    return wavelet, best_phase, amp_smoothed


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


def get_dtw_features(trace, dt):
    """
    构造 DTW 匹配用的混合特征信号。
    
    feature = 0.7 * envelope + 0.3 * normalized_trace
    
    包络提供稳定的大尺度对齐能力，
    原始信号保留相位信息以精确匹配。
    
    参数:
    trace : 输入信号 (N,)
    dt : 采样间隔 (s)，保留接口扩展性
    
    返回:
    feature : 混合特征信号 (N,)
    """
    from scipy.signal import hilbert
    
    # 计算解析信号的包络
    analytic = hilbert(trace)
    envelope = np.abs(analytic)
    
    # 归一化
    env_norm = envelope / (np.max(np.abs(envelope)) + 1e-12)
    trace_norm = trace / (np.max(np.abs(trace)) + 1e-12)
    
    feature = 0.7 * env_norm + 0.3 * trace_norm
    return feature


