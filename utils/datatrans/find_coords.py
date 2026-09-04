import numpy as np
import segyio
from pathlib import Path

def main():
    # 1. 已知控制点
    wells = {
        'CB323': {'X': 674300.00, 'Y': 4241700.00, 'IL': 1973, 'XL': 1549},
        'CB803': {'X': 669116.47, 'Y': 4240146.46, 'IL': 1766, 'XL': 1487},
        'CB826': {'X': 669848.76, 'Y': 4241239.47, 'IL': 1795, 'XL': 1531},
    }
    
    # 2. 求解仿射变换矩阵
    M = np.array([
        [wells['CB323']['X'], wells['CB323']['Y'], 1],
        [wells['CB803']['X'], wells['CB803']['Y'], 1],
        [wells['CB826']['X'], wells['CB826']['Y'], 1],
    ])
    
    B_il = np.array([wells['CB323']['IL'], wells['CB803']['IL'], wells['CB826']['IL']])
    B_xl = np.array([wells['CB323']['XL'], wells['CB803']['XL'], wells['CB826']['XL']])
    
    coef_il = np.linalg.solve(M, B_il)
    coef_xl = np.linalg.solve(M, B_xl)
    
    # 3. 预测 CB805 (X=666688.00, Y=4241888.70)
    x_805, y_805 = 666688.00, 4241888.70
    pred_il = coef_il[0] * x_805 + coef_il[1] * y_805 + coef_il[2]
    pred_xl = coef_xl[0] * x_805 + coef_xl[1] * y_805 + coef_xl[2]
    
    print(f"CB805 坐标估算值:")
    print(f"  X = {x_805}, Y = {y_805}")
    print(f"  预测 Inline = {pred_il:.2f} (建议取整: {round(pred_il)})")
    print(f"  预测 Crossline = {pred_xl:.2f} (建议取整: {round(pred_xl)})")
    
    # 4. 在实际的 SEGY 文件中验证
    segy_file = Path(r"F:\02JDX_Data\测井实际数据\项目第三次数据工区资料（斜井）\seismicdata.segy")
    if not segy_file.exists():
        print(f"\n警告：未在 {segy_file} 找到 SEGY 文件，无法进行 trace 验证。")
        return
        
    print(f"\n正在打开 SEGY 文件进行匹配度验证...")
    target_il_int = int(round(pred_il))
    target_xl_int = int(round(pred_xl))
    
    iline_byte = 9
    xline_byte = 21
    
    with segyio.open(str(segy_file), "r", ignore_geometry=True) as src:
        inline_all = np.asarray(src.attributes(iline_byte)[:])
        crossline_all = np.asarray(src.attributes(xline_byte)[:])
        
        dist = (inline_all - target_il_int)**2 + (crossline_all - target_xl_int)**2
        min_idx = np.argmin(dist)
        best_il = inline_all[min_idx]
        best_xl = crossline_all[min_idx]
        
        print(f"SEGY 文件中匹配结果:")
        print(f"  最近 Trace 索引: {min_idx}")
        print(f"  对应的 Inline   : {best_il}")
        print(f"  对应的 Crossline : {best_xl}")
        print(f"  与估算值的距离   : {np.sqrt(dist[min_idx]):.4f}")

if __name__ == "__main__":
    main()
