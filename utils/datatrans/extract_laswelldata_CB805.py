import lasio
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt


def extract_curves(las_file_path, output_file_path=None, extract_ac=True, extract_rhob=True):
    """
    从 LAS 文件中提取多种曲线数据
    """
    try:
        # 读取 LAS 文件
        las = lasio.read(las_file_path)

        # 获取所有可用的曲线名称
        all_curve_names = [curve.mnemonic for curve in las.curves]
        print(f"LAS 文件中的所有曲线名称: {all_curve_names}")

        # 查找深度曲线
        depth_patterns = ['DEPTH', 'DEPT', r'DEPT\.', r'DEPT\..*']
        depth_curve = None

        for pattern in depth_patterns:
            regex = re.compile(f'^{pattern}', re.IGNORECASE)
            matching_curves = [name for name in all_curve_names if regex.match(name)]
            if matching_curves:
                depth_curve = matching_curves[0]
                print(f"找到深度曲线: {depth_curve}")
                break

        if not depth_curve:
            raise ValueError("LAS 文件中没有找到深度曲线")

        # 初始化数据字典
        data_dict = {'Depth': las[depth_curve]}
        extracted_curves = []

        # 检查并提取 AC 曲线
        if extract_ac:
            ac_patterns = ['AC_QYZ', 'AC', 'DT', 'DTCO', 'DTSM']
            ac_curve = None
            for pattern in ac_patterns:
                regex = re.compile(f'^{pattern}', re.IGNORECASE)
                matching_curves = [name for name in all_curve_names if regex.match(name)]
                if matching_curves:
                    ac_curve = matching_curves[0]
                    print(f"找到声波曲线: {ac_curve}")
                    data_dict['AC'] = las[ac_curve]
                    extracted_curves.append('AC')
                    break
            if not ac_curve:
                print("警告: LAS 文件中没有找到声波速度曲线(AC/DT)")

        # 恢复并检查提取密度曲线 (RHOB)
        if extract_rhob:
            rhob_patterns = ['RHOB_QYZ', 'RHOB', 'DEN', 'DENS', 'ZDEN']
            rhob_curve = None
            for pattern in rhob_patterns:
                regex = re.compile(f'^{pattern}', re.IGNORECASE)
                matching_curves = [name for name in all_curve_names if regex.match(name)]
                if matching_curves:
                    rhob_curve = matching_curves[0]
                    print(f"找到密度曲线: {rhob_curve}")
                    data_dict['RHOB'] = las[rhob_curve]
                    extracted_curves.append('RHOB')
                    break
            if not rhob_curve:
                print("警告: LAS 文件中没有找到密度曲线(RHOB/DEN)")

        if not extracted_curves:
            raise ValueError("没有找到任何需要提取的测井曲线")

        # 创建 DataFrame
        df = pd.DataFrame(data_dict)

        # 直接使用 dropna() 剔除无效值，因为 lasio 已经自动将 NULL 转为 NaN
        original_len = len(df)
        df = df.dropna()
        print(f"有效数据点数: {len(df)} (剔除了 {original_len - len(df)} 个包含无效值的深度点)")

        # 保存到文件
        if output_file_path:
            # 根据实际提取到的曲线，动态组装保存数组
            columns = [df['Depth'].values]
            col_names = ['Depth']
            
            if 'AC' in extracted_curves:
                columns.append(df['AC'].values)
                col_names.append('AC')
            if 'RHOB' in extracted_curves:
                columns.append(df['RHOB'].values)
                col_names.append('RHOB')
            
            save_array = np.column_stack(columns)
            header_str = '  '.join(col_names)
            
            if output_file_path.lower().endswith('.txt') or output_file_path.lower().endswith('.csv'):
                # 可选：保存为文本文件
                np.savetxt(output_file_path, save_array, 
                           header=header_str, comments='', fmt='%.6f', delimiter='\t')
                print(f"文本数据已保存到 {output_file_path}")
            else:
                # 默认保存为 npy 格式
                if not output_file_path.lower().endswith('.npy'):
                    output_file_path = output_file_path + '.npy'
                np.save(output_file_path, save_array)
                print(f"npy 数据已保存到 {output_file_path}, shape={save_array.shape}")
            
            print(f"列顺序: {col_names}")

        return df

    except Exception as e:
        print(f"处理 LAS 文件时出错: {str(e)}")
        return None

# 使用示例
if __name__ == "__main__":
    # 输入LAS文件路径
    las_file_path = r"F:\Project\基于随钻测井数据的高精度井震标定及速度建模\实际数据\工区资料\03Well Logs\Well Logs\CB803.Las"

    # 输出文件路径(可选)
    output_file_path = r"D:\python_code\python_project\seismic_well\true_data_tyingCB803\true_datacode\_experiment_data\well_data\CB803_well.npy"

    # 提取曲线 (可以自由选择要提取的曲线)
    well_data = extract_curves(
        las_file_path,
        output_file_path,
        extract_ac=True,
        extract_rhob=True
    )

    if well_data is not None:
        print("\n提取的曲线数据预览:")
        print(well_data.head())

        # =========================================================
        # 可视化预览
        # =========================================================
        fig, axes = plt.subplots(1, 2, figsize=(10, 8), sharey=True)
        
        # 1. 声波时差 (AC)
        if 'AC' in well_data.columns:
            axes[0].plot(well_data['AC'], well_data['Depth'], color='blue', linewidth=1.0)
            axes[0].set_xlabel('AC (us/ft)')
            axes[0].set_title('Sonic (AC)')
            axes[0].grid(True, linestyle='--', alpha=0.6)
            axes[0].invert_yaxis()
        
        # 2. 密度 (RHOB)
        if 'RHOB' in well_data.columns:
            axes[1].plot(well_data['RHOB'], well_data['Depth'], color='red', linewidth=1.0)
            axes[1].set_xlabel('RHOB (g/cm³)')
            axes[1].set_title('Density (RHOB)')
            axes[1].grid(True, linestyle='--', alpha=0.6)
        
        axes[0].set_ylabel('Depth (m)')
        plt.suptitle(f"Well Logging Preview: {las_file_path.split('\\')[-1]}", fontsize=14)
        plt.tight_layout()
        
        # 保存预览图
        vis_save_path = output_file_path.replace('.npy', '.png') if output_file_path else "well_preview.png"
        plt.savefig(vis_save_path, dpi=200)
        print(f"\n预览图已保存至: {vis_save_path}")
        plt.show()
