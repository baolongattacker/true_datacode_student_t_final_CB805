import numpy as np
import matplotlib.pyplot as plt


def xwigb(
    seis,
    t=None,
    x=None,
    scale=0.75,
    linewidth=1.0,
    mode="vertical",
    wiggle_fill="peak_fill",
    wigb_color="k",
    ax=None,
    normalization="global",
    time_unit="s",
):
    """
    绘制地震变面积波形图。

    Parameters
    ----------
    seis : array_like
        地震数据，推荐形状为 (nt, ntr)。

    t : array_like or None
        时间或深度轴，长度为 nt。

    x : array_like or None
        道位置，长度为 ntr。

    scale : float
        最大摆幅相对于典型道间距的比例。

    normalization : {"global", "trace", "none"}
        global：全剖面统一归一化；
        trace：逐道归一化；
        none：不归一化。

    time_unit : str
        仅用于坐标轴标签，例如 "s"、"ms" 或 "m"。
    """
    seis = np.asarray(seis, dtype=np.float64)

    if seis.ndim == 1:
        seis = seis[:, None]
    elif seis.ndim != 2:
        raise ValueError(
            "seis must be a 1D or 2D array"
        )

    if not np.all(np.isfinite(seis)):
        raise ValueError(
            "seis contains NaN or Inf"
        )

    # 根据 t 自动识别数据方向
    if t is not None:
        t = np.asarray(t, dtype=np.float64).reshape(-1)

        if seis.shape[0] != t.size:
            if seis.shape[1] == t.size:
                seis = seis.T
            else:
                raise ValueError(
                    f"t length ({t.size}) does not match "
                    f"seis shape {seis.shape}"
                )

    nt, ntr = seis.shape

    if t is None:
        t = np.arange(nt, dtype=np.float64)
    else:
        if t.size != nt:
            raise ValueError(
                f"t length is {t.size}, expected {nt}"
            )

        if not np.all(np.isfinite(t)):
            raise ValueError(
                "t contains NaN or Inf"
            )

        if nt > 1 and not np.all(np.diff(t) > 0):
            raise ValueError(
                "t must be strictly increasing"
            )

    if x is None:
        x = np.arange(1, ntr + 1, dtype=np.float64)
    else:
        x = np.asarray(x, dtype=np.float64).reshape(-1)

        if x.size != ntr:
            raise ValueError(
                f"x length is {x.size}, expected {ntr}"
            )

        if not np.all(np.isfinite(x)):
            raise ValueError(
                "x contains NaN or Inf"
            )

        if ntr > 1 and not np.all(np.diff(x) > 0):
            raise ValueError(
                "x must be strictly increasing"
            )

    if mode not in {"vertical", "horizontal"}:
        raise ValueError(
            "mode must be 'vertical' or 'horizontal'"
        )

    if wiggle_fill not in {
        "no_fill",
        "peak_fill",
        "trough_fill",
    }:
        raise ValueError(
            "wiggle_fill must be 'no_fill', "
            "'peak_fill', or 'trough_fill'"
        )

    if scale <= 0:
        raise ValueError("scale must be positive")

    if ntr > 1:
        trace_spacing = float(np.median(np.diff(x)))
    else:
        trace_spacing = 1.0

    # 归一化
    if normalization == "global":
        denominator = float(np.max(np.abs(seis)))

        if denominator > 0:
            seis_normalized = seis / denominator
        else:
            seis_normalized = seis.copy()

    elif normalization == "trace":
        denominator = np.max(
            np.abs(seis),
            axis=0,
            keepdims=True,
        )

        denominator[denominator == 0] = 1.0
        seis_normalized = seis / denominator

    elif normalization == "none":
        seis_normalized = seis.copy()

    else:
        raise ValueError(
            "normalization must be "
            "'global', 'trace', or 'none'"
        )

    seis_scaled = (
        scale
        * trace_spacing
        * seis_normalized
    )

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 10))

    for itr in range(ntr):
        baseline = x[itr]
        trace_position = baseline + seis_scaled[:, itr]

        if mode == "vertical":
            if wiggle_fill == "peak_fill":
                ax.fill_betweenx(
                    t,
                    baseline,
                    trace_position,
                    where=(trace_position >= baseline),
                    facecolor=wigb_color,
                    interpolate=True,
                    linewidth=0,
                )

            elif wiggle_fill == "trough_fill":
                ax.fill_betweenx(
                    t,
                    baseline,
                    trace_position,
                    where=(trace_position <= baseline),
                    facecolor=wigb_color,
                    interpolate=True,
                    linewidth=0,
                )

            ax.plot(
                trace_position,
                t,
                color=wigb_color,
                linewidth=linewidth,
                antialiased=True,
                solid_joinstyle="round",
                solid_capstyle="round",
            )

        else:
            if wiggle_fill == "peak_fill":
                ax.fill_between(
                    t,
                    baseline,
                    trace_position,
                    where=(trace_position >= baseline),
                    facecolor=wigb_color,
                    interpolate=True,
                    linewidth=0,
                )

            elif wiggle_fill == "trough_fill":
                ax.fill_between(
                    t,
                    baseline,
                    trace_position,
                    where=(trace_position <= baseline),
                    facecolor=wigb_color,
                    interpolate=True,
                    linewidth=0,
                )

            ax.plot(
                t,
                trace_position,
                color=wigb_color,
                linewidth=linewidth,
                antialiased=True,
                solid_joinstyle="round",
                solid_capstyle="round",
            )

    if mode == "vertical":
        ax.set_xlim(
            x[0] - trace_spacing,
            x[-1] + trace_spacing,
        )
        ax.set_ylim(t[0], t[-1])

        if not ax.yaxis_inverted():
            ax.invert_yaxis()

        ax.set_xlabel("Trace position")
        ax.set_ylabel(f"Time / {time_unit}")
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")

    else:
        ax.set_xlim(t[0], t[-1])
        ax.set_ylim(
            x[0] - trace_spacing,
            x[-1] + trace_spacing,
        )
        ax.set_xlabel(f"Time / {time_unit}")
        ax.set_ylabel("Trace position")

    ax.tick_params(direction="out")

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