# -*- coding: utf-8 -*-
"""
从普通文本测井文件中提取有效声波时差 AC，并用 Gardner 公式计算密度。

设计原则
--------
1. 只以 Depth 和 AC 的有效性决定是否保留采样点；
2. 默认不修改有效 AC 原始值；
3. 剔除缺测值、非有限值及超出给定物理范围的数据；
4. 可选检测孤立尖峰，但尽量不误删真实地层突变；
5. 默认直接删除异常点，不跨越长缺失段插值；
6. 密度全部由 Gardner 公式根据有效 AC 计算，不读取实测密度。
"""

from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _find_exact_curve(
    curve_names: list[str],
    candidates: tuple[str, ...],
) -> str | None:
    """按照候选名称优先级查找曲线，名称必须完全匹配。"""
    for candidate in candidates:
        pattern = re.compile(rf"^{re.escape(candidate)}$", re.IGNORECASE)
        for name in curve_names:
            if pattern.match(name):
                return name
    return None


def _interpolate_short_internal_gaps(
    values: pd.Series,
    max_gap_samples: int,
) -> tuple[pd.Series, int]:
    """
    只插值位于曲线内部、且长度不超过 max_gap_samples 的短缺失段。

    长缺失段以及曲线首尾缺失段保持为 NaN，随后由主函数删除。
    """
    if max_gap_samples < 1:
        return values.copy(), 0

    result = values.copy()
    missing = result.isna()

    if not missing.any():
        return result, 0

    group_id = missing.ne(missing.shift(fill_value=False)).cumsum()
    group_size = missing.groupby(group_id).transform("sum")

    short_gap_mask = missing & (group_size <= max_gap_samples)
    interpolated = result.interpolate(
        method="linear",
        limit_area="inside",
    )

    fill_mask = short_gap_mask & interpolated.notna()
    result.loc[fill_mask] = interpolated.loc[fill_mask]

    return result, int(fill_mask.sum())


def extract_curves_from_txt(
    txt_file_path,
    output_file_path=None,
    extract_ac=True,
    extract_rhob=True,
    null_values=(-999.25, -999.0, -999, -9999.0, -9999),
    ac_min=40.0,
    ac_max=200.0,
    detect_isolated_spikes=True,
    max_ac_deviation=30.0,
    spike_window=5,
    handle_ac_anomalies="drop",
    max_interp_gap=2,
    ac_unit="us/ft",
    gardner_coefficient=0.23,
):
    """
    提取有效的 Depth、AC，并可用 Gardner 公式计算 RHOB。

    Parameters
    ----------
    txt_file_path : str or pathlib.Path
        普通文本测井文件。文件中应包含深度列和声波时差列。

    output_file_path : str or pathlib.Path or None
        输出路径：
        - .npy：保存数值数组；
        - .txt：Tab 分隔文本；
        - .csv：逗号分隔文本。

    extract_ac : bool
        为兼容旧调用保留。该函数必须依赖 AC，因此必须为 True。

    extract_rhob : bool
        True 时，不读取实测密度，而是全部利用 Gardner 公式生成 RHOB。

    null_values : tuple
        缺测或填充值。

    ac_min, ac_max : float
        AC 有效范围。默认 40～200 us/ft。
        这是宽松筛选范围，应根据工区和 AC 单位调整。

    detect_isolated_spikes : bool
        是否检测孤立尖峰。只处理明显偏离局部中值、且前后邻点彼此接近的点，
        尽量避免将真实地层阶跃误判为噪声。

    max_ac_deviation : float or None
        AC 相对局部中值的最大允许偏差。

    spike_window : int
        局部滚动中值窗口，必须为不小于 3 的奇数。

    handle_ac_anomalies : {"drop", "interpolate"}
        drop：
            默认方案。直接删除异常采样点，最大程度保留真实原始数据。
        interpolate：
            仅插值长度不超过 max_interp_gap 的内部短缺失段；
            不会跨越长井段插值。

    max_interp_gap : int
        handle_ac_anomalies="interpolate" 时允许修复的最大连续缺失样点数。

    ac_unit : {"us/ft", "us/m"}
        AC 单位。

    gardner_coefficient : float
        Gardner 公式系数。Vp 使用 ft/s 时，经典值为 0.23：
        RHOB(g/cm3) = 0.23 * Vp(ft/s)^0.25

    Returns
    -------
    pandas.DataFrame
        默认列为 Depth、AC、RHOB。
        其中 AC 为筛选后保留的原始值；只有启用短缺口插值时，少量点会被修复。
    """
    if not extract_ac:
        raise ValueError("该函数以 AC 为筛选核心，extract_ac 必须为 True。")

    if not np.isfinite(ac_min) or not np.isfinite(ac_max) or ac_min >= ac_max:
        raise ValueError("必须满足有限的 ac_min < ac_max。")

    if spike_window < 3 or spike_window % 2 == 0:
        raise ValueError("spike_window 必须是不小于 3 的奇数。")

    if handle_ac_anomalies not in {"drop", "interpolate"}:
        raise ValueError(
            "handle_ac_anomalies 只能是 'drop' 或 'interpolate'。"
        )

    ac_unit_normalized = str(ac_unit).strip().lower().replace("μ", "u")
    if ac_unit_normalized not in {"us/ft", "us/m"}:
        raise ValueError("ac_unit 仅支持 'us/ft' 或 'us/m'。")

    txt_path = Path(txt_file_path)
    if not txt_path.is_file():
        raise FileNotFoundError(f"测井文本文件不存在: {txt_path}")

    # ------------------------------------------------------------
    # 1. 自动定位表头
    # ------------------------------------------------------------
    with txt_path.open("r", encoding="utf-8", errors="ignore") as file:
        lines = file.readlines()

    header_line_index = None
    for index, line in enumerate(lines):
        tokens = re.split(r"\s+", line.strip())
        if any(
            re.fullmatch(r"(DEPTH|DEPT|MD)", token, re.IGNORECASE)
            for token in tokens
        ):
            header_line_index = index
            break

    if header_line_index is None:
        raise ValueError(
            "文本文件中没有找到深度表头，例如 DEPTH、DEPT 或 MD。"
        )

    # ------------------------------------------------------------
    # 2. 读取数据并识别 Depth、AC
    # ------------------------------------------------------------
    raw_df = pd.read_csv(
        txt_path,
        sep=r"\s+",
        header=0,
        skiprows=header_line_index,
        engine="python",
    )
    raw_df.columns = [str(column).strip() for column in raw_df.columns]
    curve_names = list(raw_df.columns)

    depth_curve = _find_exact_curve(
        curve_names,
        ("DEPTH", "DEPT", "MD"),
    )
    ac_curve = _find_exact_curve(
        curve_names,
        ("AC_QYZ", "AC", "DTCO", "DTC", "DT", "DTSM"),
    )

    if depth_curve is None:
        raise ValueError("没有找到深度曲线 DEPTH、DEPT 或 MD。")
    if ac_curve is None:
        raise ValueError("没有找到声波时差曲线 AC、DT、DTCO 或 DTC。")

    print(f"找到深度曲线: {depth_curve}")
    print(f"找到声波时差曲线: {ac_curve}")

    df = pd.DataFrame(
        {
            "Depth": pd.to_numeric(raw_df[depth_curve], errors="coerce"),
            "AC": pd.to_numeric(raw_df[ac_curve], errors="coerce"),
        }
    )

    original_len = len(df)

    # 保存原值，只用于质控计数和判断；有效样点的 AC 不做平滑。
    df["AC_raw"] = df["AC"]

    # ------------------------------------------------------------
    # 3. 基础数据整理
    # ------------------------------------------------------------
    for null_value in null_values:
        df.loc[np.isclose(df["AC"], null_value, equal_nan=False), "AC"] = np.nan
        df.loc[
            np.isclose(df["Depth"], null_value, equal_nan=False),
            "Depth",
        ] = np.nan

    nonfinite_depth_mask = ~np.isfinite(df["Depth"])
    nonfinite_ac_mask = ~np.isfinite(df["AC"])

    df.loc[nonfinite_depth_mask, "Depth"] = np.nan
    df.loc[nonfinite_ac_mask, "AC"] = np.nan

    # 深度排序并删除重复深度，避免后续积分和插值出现问题。
    df = df.dropna(subset=["Depth"]).sort_values("Depth")
    duplicate_depth_count = int(df["Depth"].duplicated(keep="first").sum())
    df = df.drop_duplicates(subset=["Depth"], keep="first").reset_index(drop=True)

    # ------------------------------------------------------------
    # 4. AC 物理范围筛选
    # ------------------------------------------------------------
    range_invalid_mask = (
        df["AC"].notna()
        & ((df["AC"] < ac_min) | (df["AC"] > ac_max))
    )
    range_invalid_count = int(range_invalid_mask.sum())
    df.loc[range_invalid_mask, "AC"] = np.nan

    # ------------------------------------------------------------
    # 5. 可选：检测孤立尖峰
    # ------------------------------------------------------------
    isolated_spike_mask = pd.Series(False, index=df.index)

    if (
        detect_isolated_spikes
        and max_ac_deviation is not None
        and max_ac_deviation > 0
    ):
        local_median = df["AC"].rolling(
            window=spike_window,
            center=True,
            min_periods=3,
        ).median()

        previous_ac = df["AC"].shift(1)
        next_ac = df["AC"].shift(-1)

        deviates_from_local = (
            df["AC"].notna()
            & local_median.notna()
            & ((df["AC"] - local_median).abs() > max_ac_deviation)
        )

        # 只有当前后邻点彼此较接近时，才认为中间点是孤立尖峰。
        neighbors_are_consistent = (
            previous_ac.notna()
            & next_ac.notna()
            & ((previous_ac - next_ac).abs() <= max_ac_deviation)
        )

        isolated_spike_mask = deviates_from_local & neighbors_are_consistent
        df.loc[isolated_spike_mask, "AC"] = np.nan

    isolated_spike_count = int(isolated_spike_mask.sum())

    # ------------------------------------------------------------
    # 6. 异常点处理
    # ------------------------------------------------------------
    interpolated_count = 0
    if handle_ac_anomalies == "interpolate":
        df["AC"], interpolated_count = _interpolate_short_internal_gaps(
            df["AC"],
            max_gap_samples=max_interp_gap,
        )

    # 默认删除仍然无效的 AC 点，不用长距离插值制造“伪原始数据”。
    before_drop = len(df)
    df = df.dropna(subset=["AC"]).copy()
    dropped_after_qc_count = before_drop - len(df)

    if df.empty:
        raise ValueError(
            "筛选后没有有效 AC 数据。请检查 AC 单位、有效范围和原始文件。"
        )

    # ------------------------------------------------------------
    # 7. Gardner 密度
    # ------------------------------------------------------------
    if ac_unit_normalized == "us/ft":
        vp_ft_s = 1.0e6 / df["AC"].to_numpy(dtype=float)
    else:
        vp_m_s = 1.0e6 / df["AC"].to_numpy(dtype=float)
        vp_ft_s = vp_m_s * 3.280839895013123

    if extract_rhob:
        df["RHOB"] = gardner_coefficient * np.power(vp_ft_s, 0.25)

    # 只输出实际需要的列，避免改变下游程序的数据结构。
    output_columns = ["Depth", "AC"]
    if extract_rhob:
        output_columns.append("RHOB")
    result = df[output_columns].reset_index(drop=True)

    # ------------------------------------------------------------
    # 8. 输出质控摘要
    # ------------------------------------------------------------
    null_or_nonfinite_count = int(
        pd.to_numeric(raw_df[ac_curve], errors="coerce").isna().sum()
    )
    retained_ratio = len(result) / max(original_len, 1)

    print("\nAC 有效性筛选结果")
    print("-" * 48)
    print(f"原始行数                  : {original_len}")
    print(f"缺测或非数值 AC           : {null_or_nonfinite_count}")
    print(f"超出 [{ac_min}, {ac_max}] : {range_invalid_count}")
    print(f"检测到的孤立尖峰          : {isolated_spike_count}")
    print(f"短缺口插值点数            : {interpolated_count}")
    print(f"重复深度点                : {duplicate_depth_count}")
    print(f"最终保留点数              : {len(result)}")
    print(f"保留比例                  : {retained_ratio:.2%}")
    print(f"异常处理策略              : {handle_ac_anomalies}")
    print("-" * 48)

    # ------------------------------------------------------------
    # 9. 保存结果
    # ------------------------------------------------------------
    if output_file_path is not None:
        output_path = Path(output_file_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        suffix = output_path.suffix.lower()

        if suffix == ".csv":
            result.to_csv(output_path, index=False, encoding="utf-8-sig")
        elif suffix == ".txt":
            result.to_csv(
                output_path,
                sep="\t",
                index=False,
                float_format="%.6f",
            )
        else:
            if suffix != ".npy":
                output_path = output_path.with_suffix(".npy")
            np.save(output_path, result.to_numpy(dtype=float))

        print(f"结果已保存到: {output_path}")
        print(f"列顺序: {list(result.columns)}")

    return result


def plot_well_preview(
    well_data: pd.DataFrame,
    source_file_path,
    output_image_path=None,
    show=True,
):
    """绘制 AC 与 Gardner 密度预览图。"""
    if well_data.empty:
        raise ValueError("well_data 为空，无法绘图。")

    has_rhob = "RHOB" in well_data.columns
    ncols = 2 if has_rhob else 1

    fig, axes = plt.subplots(
        1,
        ncols,
        figsize=(10 if has_rhob else 5, 8),
        sharey=True,
    )
    axes = np.atleast_1d(axes)

    axes[0].plot(
        well_data["AC"],
        well_data["Depth"],
        linewidth=1.0,
    )
    axes[0].set_xlabel("AC")
    axes[0].set_title("Valid Sonic Slowness")
    axes[0].set_ylabel("Depth")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].invert_yaxis()

    if has_rhob:
        axes[1].plot(
            well_data["RHOB"],
            well_data["Depth"],
            linewidth=1.0,
        )
        axes[1].set_xlabel("RHOB (g/cm³)")
        axes[1].set_title("Gardner Density")
        axes[1].grid(True, linestyle="--", alpha=0.6)

    source_name = Path(source_file_path).name
    fig.suptitle(f"Well Logging Preview: {source_name}")
    fig.tight_layout()

    if output_image_path is not None:
        image_path = Path(output_image_path)
        image_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(image_path, dpi=200, bbox_inches="tight")
        print(f"预览图已保存到: {image_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    txt_file_path = Path(
        r"F:\02JDX_Data\测井实际数据\项目第三次数据工区资料（斜井）\03  Well Logs\Well Logs\CB805.txt"
    )

    output_file_path = Path(
        r"F:\01JDX_code\python_code\seismic_well_tying\true_datacode_student_t_final_CB805\_experiment_data\well_data\well_data_CB805.npy"
    )

    # 推荐默认方案：
    # 1. AC 只保留 40～200 us/ft；
    # 2. 检测明显孤立尖峰；
    # 3. 不插值，直接保留原始有效采样点；
    # 4. RHOB 全部由 Gardner 公式计算。
    well_data = extract_curves_from_txt(
        txt_file_path=txt_file_path,
        output_file_path=output_file_path,
        extract_ac=True,
        extract_rhob=True,
        ac_min=40.0,
        ac_max=200.0,
        detect_isolated_spikes=True,
        max_ac_deviation=30.0,
        handle_ac_anomalies="drop",
        ac_unit="us/ft",
        gardner_coefficient=0.23,
    )

    print("\n提取结果预览:")
    print(well_data.head())

    plot_well_preview(
        well_data=well_data,
        source_file_path=txt_file_path,
        output_image_path=output_file_path.with_suffix(".png"),
        show=True,
    )