import numpy as np
import matplotlib.pyplot as plt

def calculate_fwe_spectrum(f, n, alpha, C=1.0):
    """
    计算 FWE (Fractional Wavelet Envelope) 函数的振幅谱值。
    
    数学公式: A(f) = C * f^n * e^(-alpha * f)
    
    输入参数:
    f     : np.ndarray, 频率数组，单位: Hz, Shape: (N_freq,)
    n     : float, 对称性指数或形状参数，控制低频段上升速度
    alpha : float, 带宽控制参数或指数衰减参数，控制高频段衰减速度
    C     : float, 归一化或尺度系数，缺省为 1.0
    
    输出结果:
    A     : np.ndarray, 频率 f 对应的振幅值, Shape: (N_freq,)
    """
    # 强制输入检查
    assert isinstance(f, np.ndarray), "输入频率 f 必须是 NumPy 数组"
    assert n >= 0, "形状参数 n 必须非负"
    assert alpha >= 0, "衰减参数 alpha 必须非负"
    
    # 1. 计算频率幂函数项 (低频上升段)
    # Shape of term1: (N_freq,)
    term1 = C * (f ** n)
    
    # 2. 计算指数衰减项 (高频衰减段)
    # Shape of term2: (N_freq,)
    term2 = np.exp(-alpha * f)
    
    # 3. 两个相互竞争的项相乘，得到单峰振幅谱
    # Shape of A: (N_freq,)
    A = term1 * term2
    
    return A

def plot_fwe_demonstration():
    """
    绘制 FWE 函数的图像，展示不同参数对单峰振幅谱形状的影响。
    """
    # 定义频率范围 0 到 100 Hz
    f_hz = np.linspace(0, 100, 1000) # shape: (1000,)
    
    plt.figure(figsize=(10, 6))
    
    # 案例 1：基准参数 (n=2, alpha=0.1)
    # n=2 控制低频以二次方上升，alpha=0.1 控制高频以 e^(-0.1*f) 衰减
    n1, alpha1 = 2.0, 0.1
    A1 = calculate_fwe_spectrum(f_hz, n1, alpha1) # shape: (1000,)
    plt.plot(f_hz, A1, label=f'Base: n={n1}, alpha={alpha1}', linewidth=2)
    
    # 案例 2：增大 n (n=3, alpha=0.1)
    # n 变大，低频段上升变缓，单峰向右移动，低频压制更强
    n2, alpha2 = 3.0, 0.1
    A2 = calculate_fwe_spectrum(f_hz, n2, alpha2) # shape: (1000,)
    plt.plot(f_hz, A2, label=f'Larger n: n={n2}, alpha={alpha2} (Slower rise)', linewidth=2)
    
    # 案例 3：增大 alpha (n=2, alpha=0.15)
    # alpha 变大，高频衰减更快，带宽变窄，单峰向左移动
    n3, alpha3 = 2.0, 0.15
    A3 = calculate_fwe_spectrum(f_hz, n3, alpha3) # shape: (1000,)
    plt.plot(f_hz, A3, label=f'Larger alpha: n={n3}, alpha={alpha3} (Faster decay)', linewidth=2)
    
    # 标注和美化
    plt.title('FWE (Fractional Wavelet Envelope) Amplitude Spectrum', fontsize=14)
    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('Amplitude A(f)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=10)
    
    # 保存图片以供验证
    plt.savefig('fwe_spectrum_demo.png', dpi=300, bbox_inches='tight')
    print("图像已成功保存为 fwe_spectrum_demo.png")
    
    plt.show()

if __name__ == '__main__':
    plot_fwe_demonstration()
