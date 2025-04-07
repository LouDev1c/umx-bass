import pandas as pd
import numpy as np


def process_results_to_excel(pandas_file, output_excel):
    # 读取.pandas文件
    df = pd.read_pickle(pandas_file)

    # 打印数据结构以便调试
    print("DataFrame结构:")
    print(df.head())
    print("\n数据类型:")
    print(df.dtypes)

    # 创建新的DataFrame来存储结果
    results = pd.DataFrame(columns=['编号', 'vocals', 'drums', 'bass', 'other', 'F1 score'])

    # 添加编号列
    results['编号'] = range(1, 51)

    # 提取每个音轨的SDR值
    for i in range(50):
        track_data = df.iloc[i]
        # 打印当前track_data的结构
        print(f"\n处理第{i + 1}条数据:")
        print(track_data)

        # 检查数据结构并提取数据
        if isinstance(track_data, dict):
            # 如果数据是字典格式
            results.at[i, 'vocals'] = track_data.get('vocals', {}).get('SDR', np.nan)
            results.at[i, 'drums'] = track_data.get('drums', {}).get('SDR', np.nan)
            results.at[i, 'bass'] = track_data.get('bass', {}).get('SDR', np.nan)
            results.at[i, 'other'] = track_data.get('other', {}).get('SDR', np.nan)
            results.at[i, 'F1 score'] = track_data.get('bass_f1', np.nan)
        else:
            # 如果数据是其他格式，尝试不同的访问方式
            try:
                results.at[i, 'vocals'] = track_data.vocals.SDR
                results.at[i, 'drums'] = track_data.drums.SDR
                results.at[i, 'bass'] = track_data.bass.SDR
                results.at[i, 'other'] = track_data.other.SDR
                results.at[i, 'F1 score'] = track_data.bass_f1
            except AttributeError:
                print(f"无法访问第{i + 1}条数据的属性")
                continue

    # 保存到Excel文件
    results.to_excel(output_excel, index=False)
    print(f"\n结果已保存到 {output_excel}")


if __name__ == "__main__":
    # 指定输入和输出文件路径
    pandas_file = "umxl.pandas"  # 根据实际文件名修改
    output_excel = "evaluation_results.xlsx"

    process_results_to_excel(pandas_file, output_excel)
