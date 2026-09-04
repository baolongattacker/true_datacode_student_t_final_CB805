import numpy as np
import matplotlib.pyplot as plt

def xwigb(seis, t=None, x=None, scale=0.75, linewidth=1.0, 
          mode='vertical', wiggle_fill='peak_fill', wigb_color='k', ax=None):
    """
    绘制 2D 地震变面积波形图 (Wiggle Traces with fill).
    
    参数:
        seis        : 2D 地震数据矩阵 (nt x ntr)
        t           : 时间/深度轴 (nt,)
        x           : 地震道位置轴 (ntr,)
        scale       : 振幅缩放因子 (默认: 0.75)
        linewidth   : 波形线宽 (默认: 1.0)
        mode        : 'vertical' (垂直显示) 或 'horizontal' (水平显示)
        wiggle_fill : 'no_fill', 'peak_fill' (波峰填充), 'trough_fill' (波谷填充)
        wigb_color  : 线条和填充颜色 (默认: 'k' 黑色)
        ax          : matplotlib 坐标轴句柄 (默认: 当前坐标轴)
    """
    # 转换为 numpy 数组并处理单道数据
    seis = np.atleast_2d(seis)
    if seis.shape[0] == 1 and seis.shape[1] > 1:
        seis = seis.T  # 转置为列向量
    nt, ntr = seis.shape

    # 默认时间向量和道位置向量
    if t is None:
        t = np.arange(1, nt + 1)
    else:
        t = np.asarray(t).flatten()
        
    if x is None:
        x = np.arange(1, ntr + 1)
    else:
        x = np.asarray(x).flatten()

    # 如果没有提供坐标轴，则创建一个新的
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 10))

    # 计算时间/空间采样间隔
    dt = np.mean(np.diff(t)) if len(t) > 1 else 1.0
    dtr = np.mean(np.diff(x)) if len(x) > 1 else 1.0

    # 振幅归一化与缩放
    max_val = np.max(np.abs(seis))
    if max_val != 0:
        seis_scaled = scale * dtr * (seis / max_val)
    else:
        seis_scaled = seis

    # 开始逐道绘制
    for itr in range(ntr):
        xpos = x[itr]
        trace = seis_scaled[:, itr] + xpos

        if mode == 'vertical':
            # 神奇的 fill_betweenx: interpolate=True 会自动寻找零交叉点并完美填充
            if wiggle_fill == 'peak_fill':
                ax.fill_betweenx(t, xpos, trace, where=(trace >= xpos), 
                                 facecolor=wigb_color, interpolate=True)
            elif wiggle_fill == 'trough_fill':
                ax.fill_betweenx(t, xpos, trace, where=(trace <= xpos), 
                                 facecolor=wigb_color, interpolate=True)
            
            # 绘制波形外轮廓线
            ax.plot(trace, t, color=wigb_color, linewidth=linewidth)

        elif mode == 'horizontal':
            if wiggle_fill == 'peak_fill':
                ax.fill_between(t, xpos, trace, where=(trace >= xpos), 
                                facecolor=wigb_color, interpolate=True)
            elif wiggle_fill == 'trough_fill':
                ax.fill_between(t, xpos, trace, where=(trace <= xpos), 
                                facecolor=wigb_color, interpolate=True)
            
            # 绘制波形外轮廓线
            ax.plot(t, trace, color=wigb_color, linewidth=linewidth)

    # 坐标轴格式化设置
    if mode == 'vertical':
        ax.set_xlim([x[0] - dtr, x[-1] + dtr])
        ax.set_ylim([t[0], t[-1]])
        if not ax.yaxis_inverted():
            ax.invert_yaxis()  # 反转 Y 轴 (深度/时间向下增加)
        ax.set_xlabel('Trace Number')
        ax.set_ylabel('Time / ms')
        ax.xaxis.tick_top()    # X 轴刻度移到顶部
        ax.xaxis.set_label_position('top')
    elif mode == 'horizontal':
        ax.set_xlim([t[0], t[-1]])
        ax.set_ylim([x[0] - dtr, x[-1] + dtr])
        ax.set_xlabel('Time / ms')
        ax.set_ylabel('Trace Number')

    ax.tick_params(direction='out')
    
    # 调整坐标轴边框线宽
    for spine in ax.spines.values():
        spine.set_linewidth(linewidth + 0.5)

    try:
        fig = ax.get_figure()
        annot = ax.annotate(
            "", 
            xy=(0, 0), 
            xytext=(15, 15), 
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7, ec="gray"),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0")
        )
        annot.set_visible(False)

        def hover(event):
            if event.inaxes == ax:
                x_val, y_val = event.xdata, event.ydata
                if x_val is not None and y_val is not None:
                    annot.xy = (x_val, y_val)
                    ylabel_str = ax.get_ylabel() if mode == "vertical" else ax.get_xlabel()
                    unit = "s" if "s" in ylabel_str.lower() else "ms"
                    if mode == "vertical":
                        annot.set_text(f"Trace: {x_val:.2f}\nTWT: {y_val:.4f} {unit}")
                    else:
                        annot.set_text(f"TWT: {x_val:.4f} {unit}\nTrace: {y_val:.2f}")
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
            else:
                if annot.get_visible():
                    annot.set_visible(False)
                    fig.canvas.draw_idle()

        fig.canvas.mpl_connect("motion_notify_event", hover)
    except Exception as e:
        print(f"警告: 绑定鼠标悬浮坐标显示失败: {e}")

    return ax

# ==========================================
# 测试与展示脚本
# ==========================================
if __name__ == "__main__":
    # 1. 生成模拟地震数据 (雷克子波组成的合成记录)
    nt = 500
    ntr = 20
    dt = 0.002
    t = np.arange(nt) * dt
    x = np.arange(1, ntr + 1)
    
    # 初始化数据矩阵
    seis_data = np.zeros((nt, ntr))
    
    # 放置几个地震同相轴 (Events)
    f_dom = 30.0 # 主频 30Hz
    wavelet = (1.0 - 2.0 * (np.pi**2) * (f_dom**2) * (t**2)) * np.exp(-(np.pi**2) * (f_dom**2) * (t**2))
    
    for i in range(ntr):
        # 模拟一个平层和一个倾斜层，并加入少量随机噪声
        shift1 = 100
        shift2 = 250 + i * 5
        seis_data[shift1:shift1+len(wavelet), i] += wavelet[:nt-shift1] * 0.8
        seis_data[shift2:shift2+len(wavelet), i] += wavelet[:nt-shift2]
        seis_data[:, i] += 0.05 * np.random.randn(nt) # 添加噪声

    # 2. 调用 xwigb 进行绘图
    fig, ax = plt.subplots(figsize=(6, 8))
    
    # 使用波峰填充 (peak_fill), 垂直显示 (vertical)
    xwigb(seis_data, t=t, x=x, scale=1.5, linewidth=0.8, 
          mode='vertical', wiggle_fill='peak_fill', wigb_color='k', ax=ax)
    
    plt.title("Python version of XWIGB")
    plt.show()