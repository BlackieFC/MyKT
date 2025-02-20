# import math
import sys
import os
cur_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(cur_dir)))
sys.path.append(root_dir)
# from internal.common.utils import configer  # logger,
import numpy as np
from scipy.sparse import lil_matrix
# from script import gen_sample
from gen_test_case import gen_sample, get_solution, map_interval


class ConceptPredictStrategy:
    """
    self.history 学生在知识点上历史答题记录的统计，可以包括该知识点历史做题数，平均难度，平均正确率
    self.knowledge_points 学生的知识点掌握度，由历史答题记录初始化，每次做题后更新
    """
    def __init__(self, n_diff=3, history=None, valid_range=None, list_kp=None, similarity=None):
        """
        初始化
        :param history: dict, 学生的历史答题记录
        :param valid_range: tuple, 知识点掌握度取值范围，（1,4）
        :param list_kp: list, 知识点id列表（暂时要求必须为全集！！！）
        :param similarity: np.ndarray, 相似度稀疏矩阵
        """
        # self.n_diff = int(configer.getParams('Data','n_diff'))  # 3,难度分级数
        self.n_diff = n_diff

        if valid_range is None:
            self.valid_range = (1, 4)   # 知识点掌握度取值范围
        else:
            self.valid_range = valid_range
        if history is None:
            self.history = {}           # 学生的历史答题记录
        else:
            self.history = history      # dict，item形如   ind_kp: (n_p, avg_diff, avg_correctness)

        self.knowledge_points = {}      # 学生的知识点掌握度
        self.ALPHA = 0                           # 映射方法1超参数
        self.BETA = 0
        self.A, self.B, self.C = get_solution()  # 映射方法2超参数
        self.list_kp = list_kp                   # 知识点id列表（冷启动初始化用）
        # 240912：相似度稀疏矩阵
        if similarity is None:
            self.similarity = None
        else:
            self.similarity = similarity  # 13x13 np.ndarray

        self.reset_and_init_from_history()

    def map_difficulty(self, diff):
        """难度取值scaling至[0,1]"""
        return diff / self.n_diff

    def reset_and_init_from_history(self, ALPHA=0.675, BETA=0.575):
        """
        使用历史做题记录初始化self.knowledge_points（取值范围[0,4]，连续含小数）
            用线性模型，假设对于最高难度（self.map_difficulty标准化后=1），正确率达到0.8即可认为完全掌握，
            同时最低难度（self.map_difficulty标准化后=0.33），满正确率时认为掌握度=0.8，则权重系数如下：
                self.valid_range[1]            # 4
                avg_corr                       # [0,1]
                self.map_difficulty(avg_diff)  # [0,3]->[0,1]
                ALPHA                          # 27/40
                BETA                           # 23/40
        """
        self.ALPHA = ALPHA
        self.BETA = BETA
        self.knowledge_points = {}    # reset
        if len(self.history) > 0:     # 使用历史做题记录初始化self.knowledge_points
            for ind_kp, (n_p, avg_diff, avg_corr) in self.history.items():
                # 240910-11方法1: 新增corr正确率线性映射
                # self.knowledge_points[ind_kp] = (self.valid_range[1] - self.valid_range[0]) * (ALPHA * self.map_difficulty(avg_diff) + BETA) * map_interval(avg_corr)
                # 240911 方法2: 新公式求解三元一次方程组获得系数，分离diff二次项和c一次项
                self.knowledge_points[ind_kp] = ((self.valid_range[1] - self.valid_range[0])/self.A) * (avg_corr - self.C + self.B * (self.map_difficulty(avg_diff))**2)
                # clip
                self.knowledge_points[ind_kp] = max(self.valid_range[0], min(self.knowledge_points[ind_kp], self.valid_range[1]))  # clip
                """240912: 注意在这种情况下，history中未涉及到的知识点不会被初始化
                （暂时强制要求将知识点总集传入self.list_kp，再对初始化结果进行查漏）
                """
                for _elem in self.list_kp:
                    if _elem not in self.knowledge_points.keys():
                        self.knowledge_points[str(_elem)] = 0.5 * (self.valid_range[1] + self.valid_range[0])  # 2.5
        elif self.list_kp is not None:
            # 若提供了知识点id列表，则使用其进行初始化
            for _elem in self.list_kp:
                self.knowledge_points[str(_elem)] = (self.valid_range[1] + self.valid_range[0]) / 2  # 2.5

    def update_knowledge_points(self,pids, qs, ss, diffs, weights=None, last_n=10, threshold=0.05):
        """
        传入当前session做题序列，更新并返回知识点掌握度
        同时判断是否满足停止条件

        :param pids: list, 题目id序列
        :param qs: list, 知识点序列
        :param ss: list, 答题结果序列
        :param diffs: list, 难度序列
        :param weights: None or list of float, 涉及知识点对于题目的相对重要性
        :param last_n: int, 根据最近last_n道题的平均知识点掌握度调整量判断是否停止
        :param threshold: float, 停止条件阈值，平均调整量小于threshold则停止
        :return:
        """
        learning_rate = 1.0
        decay_factor_1 = 0.99                          # 衰减因子1:session内遗忘(弱化)
        decay_factor_2 = 0.8                           # 衰减因子2:session内练习量(强化)
        n_ = len(pids)                                 # 题目序列长度
        list_adj = []                                  # 记录调整量，用于判断停止当前session的条件
        kp_count = {}                                  # 记录当前session已做过的涉及某知识点的题数，如'1': 3

        # 240912: 涉及知识点对于题目的相对重要性
        if weights is None:
            weights = ["0.333_0.333_0.333"] * n_        # 默认均匀权重

        # reset & 根据history初始化知识点掌握度
        self.reset_and_init_from_history()             # 新session输入，重置并初始化了self.knowledge_points
        print('-------------------------------')
        print(self.knowledge_points)
        print(len(self.history))
        print('-------------------------------')

        """遍历题目序列"""
        for _i, (pid, q, s, diff, weight) in enumerate(zip(pids, qs, ss, diffs, weights)):
            _list_adj = []                             # 记录当前题目的调整量（append至list_adj）
            knowledge_ids = q.split('_')               # 解出题目知识点list（元素为str）
            diff_norm = self.map_difficulty(diff)      # 难度取值scaling至[0,1]

            # 240911: 每个题目涉及3个知识点，它们的相对重要度加权（转为数值，并解除权重的归一化）
            weight = list(map(lambda _x: float(_x)*len(knowledge_ids), weight.split('_')))  # ["0.3","0.5","0.2"] -> [0.3, 0.5, 0.2] -> [0.9, 1.5, 0.6]

            """遍历题目涉及的知识点"""
            for _ii, k_id in enumerate(knowledge_ids):
                if k_id not in self.knowledge_points.keys():  # 对于未涉及的知识点，初始化掌握度为取值范围中心（对于[1,4]，初始化为2.5）
                    self.knowledge_points[k_id] = 0.5 * (self.valid_range[1] + self.valid_range[0])
                """
                （直接涉及）知识点的掌握度调整量：考虑时间衰减和题目难度
                调整量 = 学习率 * 
                        指数形式的时间衰减（遗忘，如果是题目在材料中的顺序的话比较合理）*
                        题目难度（越难调整越多，这个不尽然，难题作对了奖励多，但做错了的话惩罚应该相对温和，反而是容易的题做错了惩罚大，作对了奖励小）
                ==============================================================================================================
                新增：从历史统计中获取当前kp的历史做题数，用于调整量的指数衰减（以使其稳定）
                注意：暂未考虑session间的遗忘，即session间的知识点掌握度不会互相影响！！！
                ==============================================================================================================
                """
                # # 当前知识点的历史做题数（旧方案）
                # n_hist = self.history[k_id][0] if k_id in self.history.keys() else 1
                """
                240910: 当前session已做过的涉及该知识点的题数
                """
                if k_id in kp_count.keys():
                    n_sess = kp_count[k_id]
                    kp_count[k_id] += 1
                else:
                    n_sess = 0
                    kp_count[k_id] = 1
                """
                # 两部分指数衰减，一是距做完当前遍历题目&知识点已过去多少道题（session内遗忘），二是对当前遍历知识点的历史做题数
                # adjustment = learning_rate * (decay_factor_1 ** (n_-_i)) * (decay_factor_2 ** n_hist)  # 调整量
                # adjustment = learning_rate * (decay_factor_2 ** n_hist)                                # 调整量(历史总做题数衰减)
                """
                adjustment = learning_rate * (decay_factor_2 ** n_sess)    # 调整量（当前session内衰减）
                # 240912：涉及知识点对于题目的相对重要性
                adjustment = adjustment * weight[_ii]                      # 调整量（解除归一化后的加权）
                if s == 1:
                    # 答对(容易题奖励小，难题奖励大)
                    adjustment = adjustment * diff_norm
                else:
                    # 答错(容易题惩罚大，难题惩罚小)
                    adjustment = -1 * adjustment * ((1/self.n_diff) + 1-diff_norm)

                # # check（优化结构）
                # print('---------------------------')
                # print(_i)
                # print(adjustment)
                # print('---------------------------')
                self.knowledge_points[k_id] += adjustment                           # 直接调整
                self.knowledge_points[k_id] = max(self.valid_range[0],
                                                  min(self.knowledge_points[k_id], self.valid_range[1])
                                                  )                                 # clip
                # self.knowledge_points[k_id] = round(self.knowledge_points[k_id])  # 取整
                """240912: 新增基于相似度的邻域调整"""
                if self.similarity is not None:
                    for _ind, _val in enumerate(self.similarity[:, int(k_id)]):     # 取相似度矩阵的对应列进行遍历
                        if _val > 0:                                                # 暂不考虑相似的知识点位于同一题目的情况，
                            self.knowledge_points[str(_ind)] += adjustment * _val   # （以_val作为倍率，直接多调整一次）
                            self.knowledge_points[str(_ind)] = max(self.valid_range[0],
                                                                   min(self.knowledge_points[str(_ind)], self.valid_range[1])
                                                                   )                                      # clip
                            # self.knowledge_points[str(_ind)] = round(self.knowledge_points[str(_ind)])  # 取整

                """
                更新self.history
                240912：暂不将单道题的重要度纳入考量，因history只用于提供初始化，因此无需精确记录每道题的重要度
                240912：新增邻域调整项的更新（暂不考虑间接调整对于历史统计的影响）
                """
                if k_id not in self.history.keys():    # 第一次做题
                    self.history[k_id] = (1, diff, s)  # 注意：这里的diff是未经过scaling的
                else:                                  # 否则更新历史统计
                    n_p, avg_diff, avg_corr = self.history[k_id]    # 读取
                    n_p += 1                                        # 更新
                    avg_diff = (avg_diff * (n_p - 1) + diff) / n_p
                    avg_corr = (avg_corr * (n_p - 1) + s) / n_p
                    self.history[k_id] = (n_p, avg_diff, avg_corr)  # 写回

                # 记录当前遍历题目涉及的各个知识点（掌握度）的调整量至 _list_adj
                _list_adj.append(adjustment)

            # 当前题目遍历完成
            # learning_rate *= decay_factor  # 设置学习率衰减（完成了一道题的遍历，但序列不长，知识点覆盖度不高的情况下不需要）
            list_adj.append(_list_adj)       # 将当前题目的直接调整量（长度3的list）整合至 list_adj

        """
        当前session的整个做题序列遍历完毕
        240912：调整停止条件，之前的条件太宽松了（加上绝对值），导致冷启动情况下调整不充分（也有可能是需要进一步调整session内做题次数衰减）
        """
        # （1）舍入self.knowledge_points
        # self.knowledge_points = {k: math.ceil(v) for k, v in self.knowledge_points.items()}  # 向上取整
        self.knowledge_points = {k: round(v) for k, v in self.knowledge_points.items()}        # 四舍五入
        # （2）判断是否满足停止条件
        if n_ > last_n:
            list_adj = list_adj[-last_n:]
        list_adj = sum(list_adj, [])             # flatten
        list_adj = [abs(_x) for _x in list_adj]  # 取绝对值
        avg_adj = np.mean( list_adj )            # 最后last_n道题的平均调整量
        # 返回掌握度和是否满足停止条件
        return self.knowledge_points, avg_adj < threshold

    def predict_answer_probability(self, new_question_knowledge_ids, diff, weight=None):
        """
        根据新题目和掌握情况预测答对概率
        :param new_question_knowledge_ids: str, 新题目的知识点序列  "5_18_11"
        :param diff: float, 新题目的难度                            1
        :param weight: list of float, 涉及知识点对于题目的相对重要性   "0.2_0.3_0.5"
        典型调用形式：
        result_ = self.predict_answer_probability(new_q[i],
                                                  new_diff[i],
                                                  weight=new_weight[i] if new_weight is not None else None
                                                  )
        # 输入为 "5_18_11" 和 1 （i=0时），返回float答对概率
        """
        knowledge_ids = new_question_knowledge_ids.split('_')     # ["5","18","11"]
        count = len(knowledge_ids)                                # 3
        if weight is None:
            weight = [1/count] * count                            # 默认均匀权重 [0.333, 0.333, 0.333]
        else:
            weight = list(map(float, weight.split('_')))          # ["0.2","0.3","0.5"] -> [0.2, 0.3, 0.5]

        # 计算平均掌握程度
        avg_knowledge_level = 0
        for _ind, k_id in enumerate(knowledge_ids):
            mastery_level = self.knowledge_points.get(k_id, 0)    # 获取掌握度，默认值为0
            avg_knowledge_level += mastery_level * weight[_ind]   # 加权求和

        # 根据reset_and_init_from_history中的公式，反向计算正确率
        # # old
        # corr_rate = avg_knowledge_level / (self.valid_range[1] - self.valid_range[0])
        # corr_rate = corr_rate / (self.ALPHA * self.map_difficulty(diff) + self.BETA)
        # 240912: 对应于方法2的逆映射
        corr_rate = avg_knowledge_level / ((self.valid_range[1] - self.valid_range[0])/self.A)
        corr_rate = corr_rate + self.C - self.B * (self.map_difficulty(diff))**2

        return max(0, min(corr_rate, 1))  # clip and return

    def predict_knowledge_mastery(self, data_, weighted=False):
        """预测学生对知识点的掌握情况&是否可以停止推题（读取session数据并调用update_knowledge_points）"""
        pid = data_.get('pid')
        q = data_.get('q')
        s = data_.get('s')
        diff = data_.get('diff')
        if weighted:  # 涉及知识点对于题目的相对重要性
            weights = data_.get('weight')
        else:
            weights = None
        # 更新知识点掌握情况
        updated_knowledge_points, stop_session = self.update_knowledge_points(pid, q, s, diff, weights=weights)
        return updated_knowledge_points, stop_session

    def predict_answer_accuracy(self, data_, weighted=False, update_knowledge=False):
        """
        根据新题目和掌握情况预测答对概率，于batch_process中调用
        :param data_: dict, 包括当前session的历史做题序列和未做题目序列
        :param weighted: bool, 是否考虑知识点对于题目的相对重要性
        :param update_knowledge: bool, 是否更新知识点掌握情况
        形如：
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
            240911: 新增 weight 和 new_weight，形如：
            "weight": ["0.3_0.5_0.2", "0.3_0.5_0.2", "0.3_0.5_0.2", "0.3_0.5_0.2", "0.3_0.5_0.2"]
            "new_weight": ["0.3_0.5_0.2", "0.3_0.5_0.2", "0.3_0.5_0.2", "0.3_0.5_0.2"]
        """
        # 当前session的历史做题序列
        pid = data_.get('pid')
        q = data_.get('q')
        s = data_.get('s')
        diff = data_.get('diff')
        if weighted:   # 240912: 涉及知识点对于题目的相对重要性
            weights = data_.get('weight')
        else:
            weights = None

        # 当前session的未做题目序列
        new_pid = data_.get('new_pid')
        new_q = data_.get('new_q')
        new_diff = data_.get('new_diff')
        if weighted:
            new_weight = data_.get('new_weight')  # list
        else:
            new_weight = None

        # 调用update_knowledge_points，更新知识点掌握情况 self.knowledge_points
        if update_knowledge:
            _, _ = self.update_knowledge_points(pid, q, s, diff, weights=weights)  # 无需返回值

        # 调用predict_answer_probability，预测答对概率
        answer_prob = {}
        for i in range(len(new_pid)):
            # 输入为 "5_18_11" 和 1 （i=0时） 和 "0.2_0.3_0.5" or None，返回float答对概率
            result_ = self.predict_answer_probability(new_q[i], new_diff[i], weight=new_weight[i] if new_weight is not None else None)
            answer_prob[new_pid[i]] = result_  # 以new_pid为key，答对概率为value
        return answer_prob  # dict

    @staticmethod
    def calculate_loss(predictions, targets):
        """使用均方误差（MSE）作为损失函数"""
        return np.mean((np.array(predictions) - np.array(targets)) ** 2)

    def batch_process(self, data_batches):
        """对于list形式的batch输入，遍历并调用predict_answer_accuracy，统计平均损失"""
        total_loss = 0
        total_len = 0
        # 遍历batch中的每个session/序列
        for _data in data_batches:
            predictions_dict = self.predict_answer_accuracy(_data)           # 返回字典，key为new_pid，value为答对概率
            new_pid = _data.get('new_pid')                                   # [30, 28, 45, 3]
            predictions_list = [predictions_dict[_key] for _key in new_pid]  # 按new_pid的顺序，获取答对概率
            targets = _data.get('new_s')                                     # 真实的答对情况
            loss = self.calculate_loss(predictions_list, targets)            # 计算MSE损失
            total_loss += loss                                               # 总损失
            total_len += len(new_pid)                                        # 题目总数
        return total_loss / total_len                                        # 返回平均损失


def create_sparse_matrix(_n, _correlation_dict):
    """
    创建特征维度相关的稀疏矩阵。

    :param _n: 特征向量的长度
    :param _correlation_dict: 字典，键为维度索引，值为长度为5的list，元素为形如(相关维度索引，相关系数)的元组
    :return: 稀疏矩阵
    """
    _sparse_matrix = lil_matrix((_n, _n))

    for i, correlations in _correlation_dict.items():
        for j, coeff in correlations:
            _sparse_matrix[i, j] = coeff

    return _sparse_matrix


def adjust_vector(_V, i, _delta_Vi, _sparse_matrix):
    """
    调整特定索引i的向量值，并根据稀疏矩阵更新相关维度的值。

    :param _V: 特征向量
    :param i: 被调整的维度索引
    :param _delta_Vi: 调整值
    :param _sparse_matrix: 稀疏矩阵
    :return: 更新后的特征向量
    """
    # 调整第i个维度的值
    _V[i] += _delta_Vi

    # 获取与第i个维度相关的所有非零元素及其对应系数
    related_indices = _sparse_matrix.rows[i]
    related_coeffs = _sparse_matrix.data[i]

    # 更新与第i个维度相关的其他维度的值
    for idx, coeff in zip(related_indices, related_coeffs):
        _V[idx] += coeff * _delta_Vi

    return _V


if __name__ == "__main__":

    """（预设）学生对知识点的掌握度"""
    # student_mastery = {
    #     1: 1, 2: 1,
    #     3: 2, 4: 2,
    #     5: 3, 6: 3,
    #     7: 4, 8: 4
    # }
    # 学生在知识点总集上的掌握度情况
    student_mastery = {
        1: 1, 2: 1,
        3: 2, 4: 2,
        5: 3, 6: 3,
        7: 4, 8: 4,
        9: 1, 10: 2,
        11: 3, 12: 4,
    }
    list_q = list(map(str, student_mastery.keys()))  # key为str
    # list_q = list(student_mastery.keys())            # key为int
    list_q.sort()

    """
    240912: 新增完整知识点库的相似度矩阵（对角线 & 第0行/列 置空）
    根据实际xlsx数据生成的相似度矩阵的代码参见script_240912.py
    """
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
    
    # 实例化
    strategy = ConceptPredictStrategy(list_kp=list_q, similarity=sim_matrix)

    # # 示例输入数据1
    # data = {
    #     "pid": [50, 34, 45, 67, 89],
    #     "q": ["5_18_11", "5_16_10_101", "5_13_99_100", "5_12_18_101", "5_16_10_101"],
    #     "s": [1, 1, 1, 1, 1],
    #     "diff": [1, 2, 3, 2, 3]
    # }
    # # 预测学生对知识点的掌握情况
    # result, stop_sess = strategy.predict_knowledge_mastery(data)
    # print(result)

    # # 示例输入数据2（batch形式）
    # data2 = [{
    #     "pid": [50, 34, 45, 67, 89],
    #     "q": ["5_18_11", "5_16_10_101", "5_13_99_100", "5_12_18_101", "5_16_10_101"],
    #     "s": [1, 1, 1, 1, 1],
    #     "diff": [1, 2, 3, 2, 3],
    #     "new_pid": [30, 28, 45, 3],
    #     "new_q": ["5_18_11", "16_10_1", "13_5_100", "5_15_103"],
    #     "new_s": [1, 1, 1, 1],
    #     "new_diff": [1, 3, 2, 3],
    # }]
    # # # 预测学生对新题目的答对概率
    # # result2, stop_sess2 = strategy.predict_answer_accuracy(data2)
    # # print(result2)
    # result3 = strategy.batch_process(data2)
    # print(result3)

    # 示例输入数据3（batch形式，通过gen_sample生成的测试样例）
    # data3 = gen_sample()
    # # 预测学生对新题目的答对概率
    # result3 = strategy.predict_answer_accuracy(data3)
    # print(result3)
    # print(data3['new_s'])
    data3 = gen_sample()
    result3, stop_sess3 = strategy.predict_knowledge_mastery(data3)
    for key in sorted(result3.keys()):
        print(f"{key}: {result3[key]}")
    print(stop_sess3)
    data3 = gen_sample()
    result3, stop_sess3 = strategy.predict_knowledge_mastery(data3)
    for key in sorted(result3.keys()):
        print(f"{key}: {result3[key]}")
    print(stop_sess3)
    data3 = gen_sample()
    result3, stop_sess3 = strategy.predict_knowledge_mastery(data3)
    for key in sorted(result3.keys()):
        print(f"{key}: {result3[key]}")
    print(stop_sess3)
    print('=============================================')
    for key in sorted(strategy.history.keys()):
        print(f"{key}: {strategy.history[key]}")
    print('=============================================')

    # # 示例输入数据4（batch形式，通过gen_sample生成的测试样例）
    # data4 = []
    # for i in range(10):
    #     data4.append(gen_sample())
    # result4 = strategy.batch_process(data4)
    # print(result4)
    # # 预测学生对新题目的答对概率
    # result34 = strategy.predict_answer_accuracy(data4[0])
    # print(result4)
    # print(data4[0]['new_s'])
    # result4, stop_sess4 = strategy.predict_knowledge_mastery(data4[0])
    # print(result4)
    # print(stop_sess4)
    # print('=============================================')

    # # 示例输入数据4（batch形式，通过gen_sample生成的测试样例）
    # data4 = []
    # for i in range(10):
    #     data4.append(gen_sample())
    # # 初始化合并后的字典
    # merged_data = {key: [] for d in data4 for key in d.keys()}
    # # 合并数据
    # for d in data4:
    #     for key in d:
    #         merged_data[key].extend(d[key])
    # # 预测学生对新题目的答对概率
    # result4 = strategy.predict_answer_accuracy(merged_data)
    # print(result4)
    # print(merged_data['new_s'])
    # result4, stop_sess4 = strategy.predict_knowledge_mastery(merged_data)
    # print(result4)
    # print(stop_sess4)


    """特征向量相关稀疏矩阵调用"""
    # n = 10  # 特征向量长度
    # correlation_dict = {
    #     0: [(1, 0.1), (2, 0.2), (3, 0.3), (4, 0.4), (5, 0.5)],
    #     1: [(0, 0.1), (2, 0.2), (3, 0.3), (4, 0.4), (5, 0.5)],
    #     2: [(0, 0.1), (1, 0.2), (3, 0.3), (4, 0.4), (5, 0.5)],
    #     3: [(0, 0.1), (1, 0.2), (2, 0.3), (4, 0.4), (5, 0.5)],
    #     4: [(0, 0.1), (1, 0.2), (2, 0.3), (3, 0.4), (5, 0.5)],
    #     5: [(0, 0.1), (1, 0.2), (2, 0.3), (3, 0.4), (4, 0.5)],
    #     6: [(1, 0.1), (2, 0.2), (3, 0.3), (4, 0.4), (5, 0.5)],
    #     7: [(2, 0.1), (3, 0.2), (4, 0.3), (5, 0.4), (6, 0.5)],
    #     8: [(3, 0.1), (4, 0.2), (5, 0.3), (6, 0.4), (7, 0.5)],
    #     9: [(4, 0.1), (5, 0.2), (6, 0.3), (7, 0.4), (8, 0.5)],
    # }
    # # 创建初始特征向量V和稀疏矩阵
    # V = np.zeros(n)
    # print("Initial vector V:", list(V))
    #
    # sparse_matrix = create_sparse_matrix(n, correlation_dict)
    #
    # # 调整第0个维度的值，并更新相关维度的值
    # delta_Vi = 1.0
    # V = adjust_vector(V, 0, delta_Vi, sparse_matrix)
    #
    # print("Updated vector V:", list(V))
    #
    # """
    # 输出：
    # Initial vector V: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # Updated vector V: [1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.0, 0.0, 0.0, 0.0]
    # """
