import pandas as pd
import numpy as np

def process_results_to_excel(pandas_file, output_excel):
    # 读取.pandas文件
    df = pd.read_pickle(pandas_file)
    
    # 创建新的DataFrame来存储结果
    results = pd.DataFrame(columns=['编号', 'vocals', 'drums', 'bass', 'other', 'F1 score'])
    
    # 获取所有唯一的track名称
    unique_tracks = df['track'].unique()
    
    # 为每个track创建一行数据
    for i, track_name in enumerate(unique_tracks):
        # 获取当前track的所有数据
        track_data = df[df['track'] == track_name]
        
        # 初始化结果行
        results.at[i, '编号'] = i + 1
        
        # 提取每个音轨的SDR值
        for target in ['vocals', 'drums', 'bass', 'other']:
            # 获取该音轨的所有SDR值
            sdr_values = track_data[(track_data['target'] == target) & (track_data['metric'] == 'SDR')]['score']
            if not sdr_values.empty:
                # 使用中位数作为该音轨的SDR值
                results.at[i, target] = sdr_values.median()
            else:
                results.at[i, target] = np.nan
        
        # 提取bass的F1 score
        bass_f1 = track_data[(track_data['target'] == 'bass') & (track_data['metric'] == 'F1')]['score']
        if not bass_f1.empty:
            results.at[i, 'F1 score'] = bass_f1.median()
        else:
            results.at[i, 'F1 score'] = np.nan
    
    # 保存到Excel文件
    results.to_excel(output_excel, index=False)
    print(f"结果已保存到 {output_excel}")
    
    # 打印一些统计信息
    print("\n统计信息:")
    print(f"处理了 {len(unique_tracks)} 首歌曲")
    print("\n各音轨SDR中位数:")
    for target in ['vocals', 'drums', 'bass', 'other']:
        print(f"{target}: {results[target].median():.2f}")
    print(f"\nF1 score中位数: {results['F1 score'].median():.4f}")

if __name__ == "__main__":
    # 指定输入和输出文件路径
    pandas_file = "umxl.pandas"  # 根据实际文件名修改
    output_excel = "evaluation_results.xlsx"
    
    process_results_to_excel(pandas_file, output_excel) 