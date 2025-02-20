"""
prompt:
现有字典形式的学生做题数据样例，形式如下：
data = {
    "pid": [50, 34, 45, 67, 89],
    "q": ["5_18_11", "5_16_10_101", "5_13_99_100", "5_12_18_101", "5_16_10_101"],
    "s": [1, 1, 1, 1, 1],
    "diff": [1, 2, 3, 2, 3],
    "new_pid": [30, 28, 45, 3],
    "new_q": ["5_18_11", "16_10_1", "13_5_100", "5_15_103"],
    "new_s": [1, 1, 1, 1],
    "new_diff": [1, 3, 2, 3],
}
其中训练集数据包含以下四项：pid为题目ID，q为对应题目设涉及的多个知识点（每道题涉及的所有知识点id以下划线为间隔拼合为一个字符串的形式表征），
s为作答结果（1答对，0答错），diff为题目对应难度（由易到难取值1-3）。相应的测试集数据为四个以'new_'为前缀的键值对。
请你仿照上面的数据格式，生成新的模拟题库和学生作答数据，要求：
1. 题库：包含30道题，每道题的难度取值属于集合{1,2,3}，数值越大难度越高；
2. 知识点库：包含8个知识点，题库中的每道题涉及且仅涉及其中的三个知识点；
3. 学生：学生对知识点库中每个知识点的掌握程度以集合{1, 2, 3, 4}中的一个数值表征，1代表完全不会，4代表完全掌握；假设学生对知识点0-1的掌握度为1，2-3的掌握度为2，4-5的掌握度为3，6-7的掌握度为4；
4. 作答：具有上述知识点掌握度的学生对题库中每道题的作答正确概率 = 对这道题目涉及3个知识点掌握度的平均值/4，在此基础上通过random.random()采样随机数，若大于作答正确率，则即为1（答对），否则即为0（答错）；
在此基础上，为题库中的题目按照以下规则分配难度和知识点：使得按ID正序遍历题库时，上述要求3中的学生对前一题涉及知识点的平均掌握度大于等于后一题的（若大于，则前后两题难度相等；若等于，则要求前一题难度大于后一题），且ID相邻的两道题的涉及知识点至少有两个相同（相同知识点数量越多越好，允许出现涉及知识点完全一致但难度不同的题，但这样题目的占比应不超过题库总数的20%）。在此基础上，生成按ID正序遍历题库的做题序列，并按照上述样例数据分割为训练集和测试集两个部分。除此之外新增学生对于每道题知识点平均掌握度，和对于所有题知识点平均掌握度的返回。
"""


import itertools
import random
import numpy as np


def map_interval(x, a=4/9, b=11/9, c=1/3, d=1.0, inverse=False):
    """将数据从[a,b]映射至[c,d]区间"""
    # if not (0 < c < a < b < d):
    #     raise ValueError("Constraints 0 < c < a < b < d are not satisfied")
    if not inverse:
        z = c + (d - c) * (x - a) / (b - a)
        slope = (d - c) / (b - a)
        intercept = c - a * slope
    else:
        z = a + (b - a) * (x - c) / (d - c)
        slope = (b - a) / (d - c)
        intercept = a - c * slope
    return z, (slope, intercept)


def get_proj_diff2slope(n_diff_=3, cmax=11/4, cr_max=0.8, slope_=0.9642857142857141, intercept_=-0.17857142857142844):  # , cmin=4/9
    """
    slope = (a*diff_norm+b)^-1
        a, b = get_proj_diff2slope(slope_=slope, intercept_=intercept)
    """
    _a = 1/(1-1/n_diff_) * (cmax/(cr_max-intercept_) - 1/slope_)
    _b = cmax/(cr_max-intercept_) - _a
    return _a, _b


def generate_proportions(n=3):
    # 随机生成n份比例
    proportions = [random.random() for _ in range(n)]
    total = sum(proportions)
    # 将每一份比例规范化为总和为1
    normalized_proportions = [p / total for p in proportions]
    return normalized_proportions


def get_solution(coefficient=None, constant=None):
    """求解线性方程组（diff实际为离散的二次项）"""
    if coefficient is None:
        coefficient = np.array([[4/9, -1*(1/3)**2, 1],
                                [11/9,-1*(3/3)**2, 1],
                                [8/9, -1*(2/3)**2, 1]])
    if constant is None:
        constant = np.array([0, 1.0, 0.7])

    _x, _y, _z = np.linalg.solve(coefficient, constant)
    return _x, _y, _z


def flatten(nested_list):
    """递归展开嵌套列表"""
    for item in nested_list:
        if isinstance(item, list):
            yield from flatten(item)
        else:
            yield item


def gen_sample(n_train=40):
    """
    240911: 新增 weight 和 new_weight，形如：
            "weight": ["0.3_0.5_0.2", "0.3_0.5_0.2", "0.3_0.5_0.2", "0.3_0.5_0.2", "0.3_0.5_0.2"]
            "new_weight": ["0.3_0.5_0.2", "0.3_0.5_0.2", "0.3_0.5_0.2", "0.3_0.5_0.2"]
    生成如下形式的数据：
    data2 = {
        "pid": [50, 34, 45, 67, 89],
        "q": ["5_18_11", "5_16_10_101", "5_13_99_100", "5_12_18_101", "5_16_10_101"],
        "s": [1, 1, 1, 1, 1],
        "diff": [1, 2, 3, 2, 3],
        "new_pid": [30, 28, 45, 3],
        "new_q": ["5_18_11", "16_10_1", "13_5_100", "5_15_103"],
        "new_s": [1, 1, 1, 1],
        "new_diff": [1, 3, 2, 3],
    }
    """
    # 生成所有长度为3的子集
    subsets = list(itertools.combinations(range(1, 9), 3))

    # 按照子集内元素平均值从小到大排列
    sorted_subsets = sorted(subsets, key=lambda x: sum(x) / len(x))

    """（deprecated）要求前后相邻两个知识点至少有两个一致"""
    valid_subsets = sorted_subsets  # 240910：直接使用所有子集（并打乱顺序），不再添加限定条件
    random.shuffle(valid_subsets)   # in-place
    # valid_subsets = []
    # for subset in sorted_subsets:
    #     if len(valid_subsets) == 0:
    #         valid_subsets.append(subset)
    #     else:
    #         last_subset = valid_subsets[-1]
    #         if len(set(last_subset) & set(subset)) >= 2:
    #             valid_subsets.append(subset)
    #         else:
    #             continue

    """学生对知识点的掌握度"""
    student_mastery = {
        1: 1, 2: 1,
        3: 2, 4: 2,
        5: 3, 6: 3,
        7: 4, 8: 4
    }

    list_qids = list(flatten(
        [['_'.join(map(str, subset))] * 3 for subset in valid_subsets]
    ))
    list_c = list(flatten(
        [['_'.join(map(str, map(lambda x: student_mastery.get(x), subset)))] * 3 for subset in valid_subsets]
    ))
    """240912：新增权重"""
    list_weight = list(flatten(
        [['_'.join(map(str, generate_proportions()))] * 3 for _ in valid_subsets]
    ))


    data = {
        "pid": list(range(1, len(valid_subsets) * 3 + 1)),
        "q": list_qids,
        "c": list_c,
        "weight": list_weight,
        "diff": [1, 2, 3] * len(valid_subsets),
        "s": [],
    }

    """
    计算s（随机采样 or 直接用0.5作为阈值进行判断）
    # 240911: 函数化计算超参（方法1）
    # _, (slo, inter) = map_interval(x=4/9, a=4/9, b=11/9, c=1/3, d=1.2, inverse=False)
    # param_a, param_b = get_proj_diff2slope(n_diff_=3, cmax=11/4, cr_max=1.2, slope_=slo, intercept_=inter)
    """
    # 240911: 函数化计算超参（方法2）
    param_a, param_b, param_c = get_solution()
    # 遍历每个题目的知识点
    for ind, elem in enumerate(data["q"]):
        diff = data["diff"][ind]            # 获取对应难度
        # corr = (np.mean([int(x) for x in data["c"][ind].split("_")]) / 4) / (0.675 * (diff/3) + 0.575)  # 计算正确率
        """
        240910: 测试新公式
        # corr = -0.5714285714285713 + (np.mean([int(_x) for _x in data["c"][ind].split("_")]) / (4-1)) / (0.17013888888888906 * (diff/3) + 0.7210648148148149)      # 映射到[0,1]
        # 240911: 方法1（函数化超参计算流程）
        # corr = inter + (np.mean([int(_x) for _x in data["c"][ind].split("_")]) / (4-1)) / (param_a/1.25 * (diff/3) + param_b)      # 映射到[0,1]
        """
        # 240911: 方法2（diff项与c分离）
        # corr = param_a * (np.mean([int(_x) for _x in data["c"][ind].split("_")]) / (4-1)) - param_b * (diff/3)**2 + param_c
        # 240912：新增weight加权
        capa = [float(_x) for _x in data["c"][ind].split("_")]
        weig = [float(_x) for _x in data["weight"][ind].split("_")]  # 权重已经归一化
        weigAvg_c = sum(_c * _w for _c, _w in zip(capa, weig))
        corr = param_a * (weigAvg_c / (4-1)) - param_b * (diff/3)**2 + param_c

        # 生成s（随机采样判断大小）
        # data["s"].append(1 if random.random() > corr else 0)  # 极限情况下正确率略大于1，但不影响s的生成（平均后取值1.33-3.66）
        # 生成s（阈值=0.5）
        data["s"].append(1 if corr > 0.5 else 0)

    # 拆分训练、测试集
    # data_split = data  # 暂不划分
    # _n_train = round(2*len(data['pid'])/3)  # 240912：训练集占比2/3
    _n_train = n_train                        # 240912：训练集固定前_n_train个
    data_split = {
        "pid": data['pid'][:_n_train],
        "q": data['q'][:_n_train],
        "c": data['c'][:_n_train],
        "diff": data['diff'][:_n_train],
        "s": data['s'][:_n_train],
        "weight": data['weight'][:_n_train],
        "new_pid": data['pid'][_n_train:],
        "new_q": data['q'][_n_train:],
        "new_c": data['c'][_n_train:],
        "new_diff": data['diff'][_n_train:],
        "new_s": data['s'][_n_train:],
        "new_weight": data['weight'][_n_train:],
    }

    return data_split


if __name__ == "__main__":
    data_out = gen_sample()

    # 打印结果
    # for key, value in data_split.items():
    #     print(f"{key}: {value}")

    for elem_ in data_out['c']:
        print(sum(map(lambda x: float(x), elem_.split("_")))/len(elem_.split("_")))

    # 学生在知识点总集上的掌握度情况
    student_mastery_total = {
        1: 1, 2: 1,
        3: 2, 4: 2,
        5: 3, 6: 3,
        7: 4, 8: 4,
        9: 1, 10: 2,
        11: 3, 12: 4,
    }

    # 13x13的相似度矩阵（不包含对角线）
    sim_matrix = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                           [0., 0.,0.8, 0., 0., 0., 0., 0., 0.,0.8, 0., 0., 0.],
                           [0.,0.8, 0., 0., 0., 0., 0., 0., 0.,0.8, 0., 0., 0.],
                           [0., 0., 0., 0.,0.8, 0., 0., 0., 0., 0.,0.8, 0., 0.],
                           [0., 0., 0.,0.8, 0., 0., 0., 0., 0., 0.,0.8, 0., 0.],
                           [0., 0., 0., 0., 0., 0.,0.8, 0., 0., 0., 0.,0.8, 0.],
                           [0., 0., 0., 0., 0.,0.8, 0., 0., 0., 0., 0.,0.8, 0.],
                           [0., 0., 0., 0., 0., 0., 0., 0.,0.8, 0., 0., 0.,0.8],
                           [0., 0., 0., 0., 0., 0., 0.,0.8, 0., 0., 0., 0.,0.8],
                           [0.,0.8,0.8, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                           [0., 0., 0.,0.8,0.8, 0., 0., 0., 0., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 0.,0.8,0.8, 0., 0., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 0., 0., 0.,0.8,0.8, 0., 0., 0., 0.]])

