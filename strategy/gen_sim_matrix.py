import pandas as pd
import ast
import numpy as np


"""生成topic相似性的稀疏矩阵"""


# 读取 Excel 文件的指定列
df = pd.read_excel('PROC_topic_sim.xlsx', usecols=['id','最相似的前5个节点id及相似度'])

# 使用 ast.literal_eval 转换单元格内容
ind_cols = df['id'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
info_raws = df['最相似的前5个节点id及相似度'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# 将整列数据存为一个列表
ind_cols = ind_cols.tolist()
info_raws = info_raws.tolist()

# 生成相似度稀疏矩阵
sim_matrix = np.zeros((len(ind_cols)+1, len(ind_cols)+1))  # +1 是因为 id 从 1 开始，(479,479)
for ind_col, info_raw in zip(ind_cols, info_raws):
    for ind_raw, val in info_raw:
        sim_matrix[ind_raw, ind_col] = val

# 确认一下是否需要对对角线进行赋值（应该不用，对角线已经在正常流程中进行了调整）
pass
# 最终输出矩阵的索引0列和行均为0，因为 id 从 1 开始
pass
