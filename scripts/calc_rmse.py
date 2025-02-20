### calc_rmse

import pandas as pd
from sklearn.metrics import mean_squared_error
import math

# 读取xlsx文件为DataFrame
df = pd.read_excel('scripts/last_step_stats_250102.xlsx')

# 6-10
filtered_df = df[(df['seq_len'] >= 6) & (df['seq_len'] <= 10) & (df['t_true'] <= 500)]
filtered_df = filtered_df.dropna(subset=['t_true', 't_pred'])
rmse = math.sqrt(mean_squared_error(filtered_df['t_true'], filtered_df['t_pred']))
print(f"做题量在区间 (5, 10] 内的学生数量: {filtered_df.shape[0]}")
print("rmse为:")
print(rmse)

filtered_df = df[(df['seq_len'] > 10) & (df['seq_len'] <= 50) & (df['t_true'] <= 500)]
filtered_df = filtered_df.dropna(subset=['t_true', 't_pred'])
rmse = math.sqrt(mean_squared_error(filtered_df['t_true'], filtered_df['t_pred']))
print(f"做题量在区间 (10, 50] 内的学生数量: {filtered_df.shape[0]}")
print("rmse为:")
print(rmse)

filtered_df = df[(df['seq_len'] > 50) & (df['seq_len'] <= 100) & (df['t_true'] <= 500)]
filtered_df = filtered_df.dropna(subset=['t_true', 't_pred'])
rmse = math.sqrt(mean_squared_error(filtered_df['t_true'], filtered_df['t_pred']))
print(f"做题量在区间 (50, 100] 内的学生数量: {filtered_df.shape[0]}")
print("rmse为:")
print(rmse)

###
"""添加sid"""
import pandas as pd


def read_and_split_txt(file_path):
    result = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            # 检查行号，如果是1的倍数则进行处理
            if i % 5 == 0:
                # 去除换行符并以逗号为间隔split
                splitted_line = line.strip().split(',')
                result.append(int(splitted_line[-1]))
    return result


def write_list_to_xlsx(file_path, data_list):
    # 尝试读取现有xlsx文件，如果没有则创建一个新的DataFrame
    try:
        df = pd.read_excel(file_path)
    except FileNotFoundError:
        df = pd.DataFrame()

    # 确保有足够的行以容纳数据列表
    if len(df) == len(data_list):
        # 将列表内容作为新的列sid写入
        df['sid'] = data_list
        # 保存文件
        df.to_excel(file_path, index=False)
    else:
        raise ValueError


# 调用
file_path = 'data/arithmetic_retag/test.txt'
sids = read_and_split_txt(file_path)

file_path = 'scripts/last_step_stats_250102.xlsx'
write_list_to_xlsx(file_path, sids)
print(f"已将列表数据写入文件 {file_path} 的列sid")



