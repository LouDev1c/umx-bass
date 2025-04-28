import pandas as pd
import numpy as np


def extract_f1_scores_from_log(log_file, tracks):
    """从日志文件中按照9x+3行规律提取F1 scores"""
    f1_scores = {}
    with open(log_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines()]

    # 生成所有包含F1 score的行号 (3,12,21,...)
    f1_line_numbers = [3 + 9 * x for x in range(len(tracks))]

    for i, track_name in enumerate(tracks):
        line_num = f1_line_numbers[i] - 1  # 转换为0-based索引

        if line_num < len(lines):
            line = lines[line_num]
            if line.startswith("Bass F1 Score:"):
                try:
                    f1_score = float(line.split(":")[1].strip())
                    f1_scores[track_name] = f1_score
                except (IndexError, ValueError):
                    print(f"警告：无法解析行{line_num + 1}的F1 score: '{line}'")
                    f1_scores[track_name] = np.nan
            else:
                print(f"警告：行{line_num + 1}不是F1 score行（预期以'Bass F1 Score:'开头）: '{line}'")
                f1_scores[track_name] = np.nan
        else:
            print(f"错误：日志文件只有{len(lines)}行，无法读取行{line_num + 1}")
            f1_scores[track_name] = np.nan

    return f1_scores


def process_results_to_excel(log_file, output_excel):
    # 从日志文件提取F1 scores
    tracks = [f"track_{i+1}" for i in range(50)]  # 假设有50个track
    f1_scores = extract_f1_scores_from_log(log_file, tracks)

    # 创建结果DataFrame
    results = pd.DataFrame(columns=['编号', 'F1 score'])

    # 处理每个track
    for i, track_name in enumerate(tracks):
        results.at[i, '编号'] = i + 1
        results.at[i, 'F1 score'] = f1_scores.get(track_name, np.nan)

    # 添加统计行（平均值、中位数和方差）
    results.loc[len(tracks), '编号'] = '平均值'
    results.loc[len(tracks) + 1, '编号'] = '中位数'
    results.loc[len(tracks) + 2, '编号'] = '方差'

    results.loc[len(tracks), 'F1 score'] = results['F1 score'].mean()
    results.loc[len(tracks) + 1, 'F1 score'] = results['F1 score'].median()
    results.loc[len(tracks) + 2, 'F1 score'] = results['F1 score'].var()

    # 保存结果
    results.to_excel(output_excel, index=False)
    print(f"结果已保存到 {output_excel}")


if __name__ == "__main__":
    log_file = r"E:\open-unmix\umx-bass\umx-bass-1_evaluation"
    output_excel = "umx-bass-1_evaluation_results.xlsx"

    process_results_to_excel(log_file, output_excel)
