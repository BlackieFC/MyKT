import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error, mean_squared_error, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
matplotlib.use('Agg')


class Evaluator_with_t:
    """
    (1120改动版，记得去git上抓一个原版备份下来)
    评估器，实现以下方法：
        evaluate: 将预报值&真实值处理为适用于sparse损失函数的形式（作答预测模型）
        report: 返回相应的统计指标（作答预测模型）
        evaluate_know: 同上，对于知识点映射模型
        report_know:   同上，对于知识点映射模型
    """
    def __init__(self, t_ratio=1, t_weight=0.5):
        self.y_true = []
        self.y_pred = []
        self.t_true = []
        self.t_pred = []
        self.t_ratio = t_ratio
        self.t_weight = t_weight
        self.max_len = 0

    def evaluate(self, y_true, y_pred, t_true=None, t_pred=None):
        mask = y_true >= 0  # 作答s的padding值为-1（采用sparse损失后所有padding均为-1）
        y_true = y_true[mask].float()
        y_pred = y_pred[mask].float()
        self.y_true.extend(y_true.cpu().tolist())
        self.y_pred.extend(y_pred.cpu().tolist())
        if t_true is not None:
            t_true = t_true[mask].float()
            t_pred = t_pred[mask].float()
            self.t_true.extend(t_true.cpu().tolist())
            self.t_pred.extend(t_pred.cpu().tolist())

    def report(self):
        """返回相应的统计指标"""
        # 还原至0-1区间
        y_t = [ele/100 for ele in self.y_true]
        y_p = [ele/100 for ele in self.y_pred]
        t_t = [ele/self.t_ratio for ele in self.t_true]
        t_p = [ele/self.t_ratio for ele in self.t_pred]

        try:
            # # 二分类指标
            # acc = accuracy_score(self.y_true, np.asarray(self.y_pred).round())
            # auc = roc_auc_score(self.y_true, self.y_pred)

            # 回归指标
            y_mae = mean_absolute_error(y_t, y_p)
            y_rmse = mean_squared_error(y_t, y_p) ** 0.5
            if len(self.t_true)==0:
                mae = y_mae
                rmse = y_rmse
            else:
                """
                # 合并在一起算
                # val_t = self.y_true + [ele/self.t_ratio for ele in self.t_true]
                # val_p = self.y_pred + [ele/self.t_ratio for ele in self.t_pred]
                # mae = mean_absolute_error(val_t, val_p)
                # rmse = mean_squared_error(val_t, val_p) ** 0.5
                """
                
                """分开算然后倍率组合"""
                t_mae = mean_absolute_error(t_t, t_p)
                t_rmse = mean_squared_error(t_t, t_p) ** 0.5
                mae = self.t_weight * t_mae + (1-self.t_weight) * y_mae
                rmse = self.t_weight * t_rmse + (1-self.t_weight) * y_rmse

            return {
                # "acc": acc,
                # "auc": auc,
                "mae": mae,
                "rmse": rmse,
            }
        except Exception as e:
            print("Error in evaluation: ", e)
            print("self.y_pred: ", self.y_pred)

    def statistics(self, y_true, y_pred, t_true=None, t_pred=None):
        """该extend为append便于分组"""
        mask = y_true >= 0  # 作答s的padding值为-1（采用sparse损失后所有padding均为-1）
        y_true = y_true[mask].float()
        y_pred = y_pred[mask].float()
        self.y_true.append(y_true.cpu().tolist())
        self.y_pred.append(y_pred.cpu().tolist())
        if t_true is not None:
            t_true = t_true[mask].float()
            t_pred = t_pred[mask].float()
            self.t_true.append(t_true.cpu().tolist())
            self.t_pred.append(t_pred.cpu().tolist())
        # 其他需要传入的统计数据
        pass

    def last_step_stats(self, file_out):
        """最终修改好的状态"""
        # data_out = pd.DataFrame(columns=["sid","seq_len","p_true","p_pred","t_true","t_pred","p_error", "t_error"])
        data_out = pd.DataFrame(columns=["seq_len","p_true","p_pred","t_true","t_pred","p_error", "t_error"])
        for ind in range(len(self.y_true)):
            y_t = self.y_true[ind][-1]  # batch_size=1
            y_p = self.y_pred[ind][-1]
            t_t = self.t_true[ind][-1]
            t_p = self.t_pred[ind][-1]
            len_seq = len(self.y_true[ind])  # len(list):seqlen

            # # 当前样本的回归指标
            # y_mae = mean_absolute_error(y_t, y_p)
            # y_rmse = mean_squared_error(y_t, y_p) ** 0.5
            # t_mae = mean_absolute_error(t_t, t_p)
            # t_rmse = mean_squared_error(t_t, t_p) ** 0.5

            # 记录
            data_out.loc[ind] = [len_seq, y_t, y_p, t_t, t_p, y_p-y_t, t_p-t_t]
        
        # 保存 & 返回
        data_out.to_excel(file_out, index=False)
        return data_out
    
    def result_statistics(self):
        """
        误差大小统计和展示，无需再归一化
            # # 还原至0-1区间
            # y_t = [ele/100 for ele in self.y_true]
            # y_p = [ele/100 for ele in self.y_pred]
            # t_t = [ele/self.t_ratio for ele in self.t_true]
            # t_p = [ele/self.t_ratio for ele in self.t_pred]
        """
        # self.max_len = max([len(elem) for elem in self.y_true])  # 计算最大有效序列长度
        data_out = pd.DataFrame(columns=["seq_len","p_mae","p_rmse","t_mae","t_rmse"])

        for ind in range(len(self.y_true)):
            y_t = self.y_true[ind]
            y_p = self.y_pred[ind]
            t_t = self.t_true[ind]
            t_p = self.t_pred[ind]
            len_seq = len(y_t)  # len(list):seqlen

            # 当前样本的回归指标
            y_mae = mean_absolute_error(y_t, y_p)
            y_rmse = mean_squared_error(y_t, y_p) ** 0.5
            t_mae = mean_absolute_error(t_t, t_p)
            t_rmse = mean_squared_error(t_t, t_p) ** 0.5

            # 记录
            data_out.loc[ind] = [len_seq, y_mae, y_rmse, t_mae, t_rmse]
        
        # 分类统计
        filtered_df = data_out[(data_out['seq_len'] > 5) & (data_out['seq_len'] <= 10)]
        # 计算筛选后的行的数量
        num_rows = filtered_df.shape[0]
        # 计算这些行的各列平均值
        mean_values = filtered_df.mean()
        print(f"做题量在区间 [6, 10] 内的学生数量: {num_rows}")
        print("平均值为:")
        print(mean_values)

        # 分类统计
        filtered_df = data_out[(data_out['seq_len'] > 10) & (data_out['seq_len'] <= 50)]
        # 计算筛选后的行的数量
        num_rows = filtered_df.shape[0]
        # 计算这些行的各列平均值
        mean_values = filtered_df.mean()
        print(f"做题量在区间 [11, 50] 内的学生数量: {num_rows}")
        print("平均值为:")
        print(mean_values)

        # 分类统计
        filtered_df = data_out[(data_out['seq_len'] > 50) & (data_out['seq_len'] <= 100)]
        # 计算筛选后的行的数量
        num_rows = filtered_df.shape[0]
        # 计算这些行的各列平均值
        mean_values = filtered_df.mean()
        print(f"做题量在区间 [51, 100] 内的学生数量: {num_rows}")
        print("平均值为:")
        print(mean_values)

        # 保存结果
        data_out.to_excel('result_stats_241121.xlsx', index=False)
        return data_out


class Evaluator_1119:
    """
    评估器，实现以下方法：
        evaluate: 将预报值&真实值处理为适用于sparse损失函数的形式（作答预测模型）
        report: 返回相应的统计指标（作答预测模型）
        evaluate_know: 同上，对于知识点映射模型
        report_know:   同上，对于知识点映射模型
    """
    def __init__(self, t_ratio=1):
        self.y_true = []
        self.y_pred = []
        self.t_true = []
        self.t_pred = []
        self.t_ratio = t_ratio
        self.min_len = 20  # 最小len(一般为静态)
        self.max_len = 30  # 最大len(动态)
        self.DiffByError_i = [[] for _ in range(100)]  # 记录统计量(题目涉及kp)
        self.DiffByError_f = [[] for _ in range(100)]  # 记录统计量(题目未涉及kp)
        self.Error_i = [[] for _ in range(100)]  # 记录统计量(题目涉及kp)
        self.Error_f = [[] for _ in range(100)]  # 记录统计量(题目未涉及kp)

    def evaluate(self, y_true, y_pred, t_true=None, t_pred=None):
        mask = y_true >= 0  # 作答s的padding值为-1（采用sparse损失后所有padding均为-1）
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        self.y_true.extend(y_true.cpu().tolist())
        self.y_pred.extend(y_pred.cpu().tolist())
        if t_true is not None:
            t_true = t_true[mask]
            t_pred = t_pred[mask]
            self.t_true.extend(t_true.cpu().tolist())
            self.t_pred.extend(t_pred.cpu().tolist())

    def report(self):
        """返回相应的统计指标"""
        try:
            # 二分类指标
            acc = accuracy_score(self.y_true, np.asarray(self.y_pred).round())
            auc = roc_auc_score(self.y_true, self.y_pred)
            # 回归指标
            if len(self.t_true)==0:
                mae = mean_absolute_error(self.y_true, self.y_pred)
                rmse = mean_squared_error(self.y_true, self.y_pred) ** 0.5
            else:
                val_t = self.y_true + [ele/self.t_ratio for ele in self.t_true]
                val_p = self.y_pred + [ele/self.t_ratio for ele in self.t_pred]
                mae = mean_absolute_error(val_t, val_p)
                rmse = mean_squared_error(val_t, val_p) ** 0.5
            return {
                "acc": acc,
                "auc": auc,
                "mae": mae,
                "rmse": rmse,
            }
        except Exception as e:
            print("Error in evaluation: ", e)
            print("self.y_pred: ", self.y_pred)

    def evaluate_know(self, y_true, y_pred, filter_label=True):
        """
        将预报值&真实值处理为适用于sparse损失函数的形式
        :param y_true: (tensor) True labels of shape (bs, n_know)
        :param y_pred: (tensor) Predicted labels of shape (bs, n_know)
        :param filter_label: (bool) Whether to filter out the labels that are -1
        :return:
        """
        y_true = y_true.flatten()                # 将标签展平为一维数组
        y_pred = y_pred.cpu().numpy().flatten()  # 将预测展平为一维数组
        if filter_label:
            # 如果使用sparse损失，则需要使用mask剔除掉值为-1的部分
            labeled_mask = (y_true != -1)
            y_pred = y_pred[labeled_mask]
            y_true = y_true[labeled_mask]
        self.y_true.extend(y_true.tolist())
        self.y_pred.extend(y_pred.tolist())

    def report_know(self):
        """
        Evaluate the performance of a model using common metrics.
        :return: (dict) A dictionary containing accuracy, precision, recall, and F1 score
        """
        accuracy = accuracy_score(self.y_true, self.y_pred)  # sklearn库函数
        precision = precision_score(self.y_true, self.y_pred, average='weighted', zero_division=0)
        recall = recall_score(self.y_true, self.y_pred, average='weighted', zero_division=0)
        f1 = f1_score(self.y_true, self.y_pred, average='weighted', zero_division=0)
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

    def evaluate_know_new(self, y_true, y_pred, kp4pid, padding_val=-1):
        """
        :param y_true: (array) True labels of shape       (n_concepts,)               (89,)
        :param y_pred: (array) Predicted labels of shape  (bs, n_concepts)            (33-20+1, 89)
        :param kp4pid: (array) kp for pid sequence        (seq_len, padding_len=20)   (33, 20)
        :param padding_val: kp的padding值（一般为-1）
        :return:
        """

        """根据输入算出所有必要的属性"""
        # valid_len = (kp4pid != padding_val).sum(dim=-1)  # array(33,20) -> array(33,)，处理kp4pid，先get每道题的有效kp数
        valid_len = np.sum((kp4pid != padding_val), axis=-1)  # array(33,20) -> array(33,)，处理kp4pid，先get每道题的有效kp数
        n_concepts = y_true.shape[0] - 1                 # 88 (0不包含在内)
        seq_len, pad_len = kp4pid.shape                  # (33,20)
        min_len = seq_len - y_pred.shape[0] + 1          # 20

        """处理y_pred，生成差分y_diff，size应为(33-20+1-1, 89) = (13,89), 即能力的知识点映射相较于上一时刻的变化"""
        y_diff = y_pred[:-1] - y_pred[1:]  # 注意输入数据的首个维度排序为seq_len道题递减至min_len道题
        y_diff = y_diff[::-1, ...]         # 倒序，从第21题的效果——第33题的效果
        y_error = y_pred - y_true      # array (14,89)-(89,) = (14,89), 可以触发broadcast
        y_error_final = y_error[0]     # (89,)  做第33题后的最终误差
        y_error_pre = y_error[1:]      # (13,89)
        y_error_pre = y_error_pre[::-1, ...]   # 倒序，从做第21题前的误差 ，... ，做第33题前的误差
        y_error_post = y_error[:-1]    # (13,89)
        y_error_post = y_error_post[::-1, ...]  # 倒序，从做第21题后的误差 ，... ，做第33题后的误差
        y_error_diff = np.abs(y_error_post) - np.abs(y_error_pre)  # (13,89)
        # 注意这里不应该是直接相减（理论上误差越接近0越好，也就是说abs越小越好，
        # 要考虑到error在0点的左右震荡收敛的情况 —— abs相减的结果中，正值代表劣化，负值代表优化）

        """做题序列（只需要考虑21-33题）的kp划分：涉及or未涉及"""
        kp_all = list(range(1, n_concepts+1))            # [1,2,...,88]
        kp_involve = []
        kp_free = []
        for _i in range(min_len, seq_len):  # 只需要考虑21-33题
            _temp = kp4pid[_i, :valid_len[_i]].tolist()
            _temp.sort()
            # 做题序列中每道题涉及到的kp list
            kp_involve.append(_temp)                                                # (n_kps,), list{seq_len-min_len=13}
            # 做题序列中每道题未涉及到的kp list
            kp_free.append([_item for _item in kp_all if _item not in set(_temp)])  # (n_concepts-n_kps,), list{13}
            # 题目序列本身是正序的，无需进行倒序变换（结果从第21题-第33题）

        """将几个kp空间表征拆分为：当前题目相关kp & 当前题目无关kp 两个互补的部分"""
        y_diff_involve = []
        y_diff_free = []
        y_error_involve = []
        y_error_free = []
        for _i, (_kp_i, _kp_f) in enumerate(zip(kp_involve, kp_free)):  # 索引0-12, ([1,2,3],[4,5,...,88])
            """这里append元组是为了给后续按照kp进行统计留接口"""
            y_diff_involve.append((_kp_i, y_diff[_i, _kp_i]))           # (n_kps,), (n_kps,)
            y_diff_free.append((_kp_f, y_diff[_i, _kp_f]))              # (n_concepts-n_kps,), (n_concepts-n_kps,)
            # y_error_involve.append((_kp_i, y_error_pre[_i, _kp_i]))   # 同上, for error_pre
            # y_error_free.append((_kp_f, y_error_pre[_i, _kp_f]))
            y_error_involve.append((_kp_i, y_error_diff[_i, _kp_i]))    # 同上, for error_diff
            y_error_free.append((_kp_f, y_error_diff[_i, _kp_f]))
            # 最终得到的四个list长度均为seq_len-min_len=13

        """统计y_diff（能力差分/变化）在两种情况下的水平"""
        def _temp_func(_list_a, _list_b, axis=-1, use_abs=True):
            """用于把上述操作的数值部分拆出来"""
            _out = []
            for (a_elem, b_elem) in zip(_list_a, _list_b):
                a_list = a_elem[axis]  # or 1 as index
                b_list = b_elem[axis]
                # if isinstance(a_list, np.ndarray) and isinstance(b_list, np.ndarray) and len(a_list) == len(b_list):
                #     # 若均为list且等长,则对位相除
                #     _res = [a / (b+1e-3) for a, b in zip(a_list, b_list)]
                #     # 添加至新列表c中
                #     _out.append(_res)
                if use_abs:
                    _out.append([np.abs(_a) for _a in a_list])
                else:
                    _out.append([_a for _a in a_list])  # 不需要取绝对值时
            return _out

        DiffByError_i = _temp_func(y_diff_involve, y_error_involve)  # list{13}, elem as list{n_kps,}
        DiffByError_f = _temp_func(y_diff_free, y_error_free)        # list{13}, elem as list{n_concepts-n_kps,}
        Error_i = _temp_func(y_error_involve, y_error_involve, use_abs=False)
        Error_f = _temp_func(y_error_free, y_error_free, use_abs=False)

        """将上述生成的内容保存到指定的self属性中"""
        if min_len < self.min_len:
            self.min_len = min_len            # 最小len(一般为静态)
        if seq_len > self.max_len:
            self.max_len = seq_len            # 最大len(动态)
        for _ind, _val in enumerate(range(min_len+1, seq_len+1)):     # 这里得对应上统计量具体表征的对象，是第21题至第33题的学习效果，因此均+1
            self.DiffByError_i[_val].extend(DiffByError_i[_ind])  # 记录统计量(题目涉及kp)
            self.DiffByError_f[_val].extend(DiffByError_f[_ind])  # 记录统计量(题目未涉及kp)
            self.Error_i[_val].extend(Error_i[_ind])
            self.Error_f[_val].extend(Error_f[_ind])

    def report_know_new(self, out_path='./script.png'):
        """
        TBD
        """
        # 清洗数据（不确定效果是否好，总之先留着这一步骤）
        DiffByError_i = [[] for _ in range(100)]
        Error_i = [[] for _ in range(100)]
        for _ind in range(len(self.DiffByError_i)):
            for _ii in range(len(self.DiffByError_i[_ind])):
                if self.DiffByError_i[_ind][_ii] != 0 or self.Error_i[_ind][_ii] != 0:
                    DiffByError_i[_ind].append(self.DiffByError_i[_ind][_ii])
                    Error_i[_ind].append(self.Error_i[_ind][_ii])

        DiffByError_f = [[] for _ in range(100)]
        Error_f = [[] for _ in range(100)]
        for _ind in range(len(self.DiffByError_f)):
            for _ii in range(len(self.DiffByError_f[_ind])):
                if self.DiffByError_f[_ind][_ii] != 0 or self.Error_f[_ind][_ii] != 0:
                    DiffByError_f[_ind].append(self.DiffByError_f[_ind][_ii])
                    Error_f[_ind].append(self.Error_f[_ind][_ii])

        """计算均值&标准差"""
        def calculate_mean_std(_data):
            return [(np.mean(lst), np.std(lst)) if len(lst) > 0 else None for lst in _data]

        def calculate_percentile(_data):
            """check用：实际数据中，由于是离散取值，且0的数量压倒性的多"""
            return [(np.min(lst), np.percentile(lst, 25), np.percentile(lst, 75), np.max(lst)) if len(lst) > 0 else None for lst in _data]

        # 注意pred diff是单边的
        DiffByError_i = calculate_mean_std(DiffByError_i)  # list{100}，元素为list([]或元素为np.int64)
        DiffByError_f = calculate_mean_std(DiffByError_f)  # 同上
        # error diff是双边的（看看会不会有比较严重的抵消问题）
        Error_i = calculate_mean_std(Error_i)
        Error_f = calculate_mean_std(Error_f)

        # slicing
        x = [_ind for _ind, elem in enumerate(DiffByError_i) if elem is not None]  # 获取非None元素的索引，作为data_x
        DiffByError_i = [DiffByError_i[_ind] for _ind in x]
        DiffByError_f = [DiffByError_f[_ind] for _ind in x]
        Error_i = [Error_i[_ind] for _ind in x]
        Error_f = [Error_f[_ind] for _ind in x]

        """将上述两组数据以折线的形式统一绘制到一张图中"""
        # # delta pred
        # one_tailed = True
        # a_means, a_stds = zip(*DiffByError_i)
        # b_means, b_stds = zip(*DiffByError_f)

        # delta error
        one_tailed = False
        a_means, a_stds = zip(*Error_i)
        b_means, b_stds = zip(*Error_f)

        # Plot the data
        plt.figure(figsize=(10, 6))

        # 根据单双边两种情况处理（注意这里的单边仅适用于均值趋近于0的情况）
        contourf_a = np.array(a_means) - np.array(a_stds)
        contourf_b = np.array(b_means) - np.array(b_stds)
        if one_tailed:
            contourf_a = np.where(contourf_a < 0, 0, contourf_a)
            contourf_b = np.where(contourf_b < 0, 0, contourf_b)

        # Plot data a
        plt.plot(x, a_means, label='involved kp', color='blue')
        plt.fill_between(x,
                         contourf_a,                            # 下界
                         np.array(a_means)+np.array(a_stds),    # 上界
                         color='blue',
                         alpha=0.2
                         )
        # Plot data b
        plt.plot(x, b_means, label='free kp', color='red')
        plt.fill_between(x,
                         contourf_b,
                         np.array(b_means) + np.array(b_stds),
                         color='red',
                         alpha=0.2
                         )
        # Customize the plot
        plt.xlabel('valid sequence length')
        plt.ylabel('delta error')
        plt.title('')
        plt.legend()
        plt.grid(True)
        plt.savefig(out_path)    # Save the plot as a PNG file
        plt.close()              # Close the plot to avoid displaying it

        # 还要有按kp统计的做题后学习效率图（看看不同kp之是否一致）
        pass


class Evaluator:
    """
    评估器，实现以下方法：
        evaluate: 将预报值&真实值处理为适用于sparse损失函数的形式（作答预测模型）
        report: 返回相应的统计指标（作答预测模型）
        evaluate_know: 同上，对于知识点映射模型
        report_know:   同上，对于知识点映射模型
    """
    def __init__(self):
        self.y_true = []
        self.y_pred = []
        self.min_len = 20  # 最小len(一般为静态)
        self.max_len = 30  # 最大len(动态)
        self.DiffByError_i = [[] for _ in range(100)]  # 记录统计量(题目涉及kp)
        self.DiffByError_f = [[] for _ in range(100)]  # 记录统计量(题目未涉及kp)
        self.Error_i = [[] for _ in range(100)]  # 记录统计量(题目涉及kp)
        self.Error_f = [[] for _ in range(100)]  # 记录统计量(题目未涉及kp)

    def evaluate(self, y_true, y_pred):
        mask = y_true >= 0  # 作答s的padding值为-1（采用sparse损失后所有padding均为-1）
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        self.y_true.extend(y_true.cpu().tolist())
        self.y_pred.extend(y_pred.cpu().tolist())

    def report(self):
        """返回相应的统计指标"""
        try:
            acc = accuracy_score(self.y_true, np.asarray(self.y_pred).round())
            auc = roc_auc_score(self.y_true, self.y_pred)
            mae = mean_absolute_error(self.y_true, self.y_pred)
            rmse = mean_squared_error(self.y_true, self.y_pred) ** 0.5
            return {
                "acc": acc,
                "auc": auc,
                "mae": mae,
                "rmse": rmse,
            }
        except Exception as e:
            print("Error in evaluation: ", e)
            print("self.y_pred: ", self.y_pred)

    def evaluate_know(self, y_true, y_pred, filter_label=True):
        """
        将预报值&真实值处理为适用于sparse损失函数的形式
        :param y_true: (tensor) True labels of shape (bs, n_know)
        :param y_pred: (tensor) Predicted labels of shape (bs, n_know)
        :param filter_label: (bool) Whether to filter out the labels that are -1
        :return:
        """
        y_true = y_true.flatten()                # 将标签展平为一维数组
        y_pred = y_pred.cpu().numpy().flatten()  # 将预测展平为一维数组
        if filter_label:
            # 如果使用sparse损失，则需要使用mask剔除掉值为-1的部分
            labeled_mask = (y_true != -1)
            y_pred = y_pred[labeled_mask]
            y_true = y_true[labeled_mask]
        self.y_true.extend(y_true.tolist())
        self.y_pred.extend(y_pred.tolist())

    def report_know(self):
        """
        Evaluate the performance of a model using common metrics.
        :return: (dict) A dictionary containing accuracy, precision, recall, and F1 score
        """
        accuracy = accuracy_score(self.y_true, self.y_pred)  # sklearn库函数
        precision = precision_score(self.y_true, self.y_pred, average='weighted', zero_division=0)
        recall = recall_score(self.y_true, self.y_pred, average='weighted', zero_division=0)
        f1 = f1_score(self.y_true, self.y_pred, average='weighted', zero_division=0)
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

    def evaluate_know_new(self, y_true, y_pred, kp4pid, padding_val=-1):
        """
        :param y_true: (array) True labels of shape       (n_concepts,)               (89,)
        :param y_pred: (array) Predicted labels of shape  (bs, n_concepts)            (33-20+1, 89)
        :param kp4pid: (array) kp for pid sequence        (seq_len, padding_len=20)   (33, 20)
        :param padding_val: kp的padding值（一般为-1）
        :return:
        """

        """根据输入算出所有必要的属性"""
        # valid_len = (kp4pid != padding_val).sum(dim=-1)  # array(33,20) -> array(33,)，处理kp4pid，先get每道题的有效kp数
        valid_len = np.sum((kp4pid != padding_val), axis=-1)  # array(33,20) -> array(33,)，处理kp4pid，先get每道题的有效kp数
        n_concepts = y_true.shape[0] - 1                 # 88 (0不包含在内)
        seq_len, pad_len = kp4pid.shape                  # (33,20)
        min_len = seq_len - y_pred.shape[0] + 1          # 20

        """处理y_pred，生成差分y_diff，size应为(33-20+1-1, 89) = (13,89), 即能力的知识点映射相较于上一时刻的变化"""
        y_diff = y_pred[:-1] - y_pred[1:]  # 注意输入数据的首个维度排序为seq_len道题递减至min_len道题
        y_diff = y_diff[::-1, ...]         # 倒序，从第21题的效果——第33题的效果
        y_error = y_pred - y_true      # array (14,89)-(89,) = (14,89), 可以触发broadcast
        y_error_final = y_error[0]     # (89,)  做第33题后的最终误差
        y_error_pre = y_error[1:]      # (13,89)
        y_error_pre = y_error_pre[::-1, ...]   # 倒序，从做第21题前的误差 ，... ，做第33题前的误差
        y_error_post = y_error[:-1]    # (13,89)
        y_error_post = y_error_post[::-1, ...]  # 倒序，从做第21题后的误差 ，... ，做第33题后的误差
        y_error_diff = np.abs(y_error_post) - np.abs(y_error_pre)  # (13,89)
        # 注意这里不应该是直接相减（理论上误差越接近0越好，也就是说abs越小越好，
        # 要考虑到error在0点的左右震荡收敛的情况 —— abs相减的结果中，正值代表劣化，负值代表优化）

        """做题序列（只需要考虑21-33题）的kp划分：涉及or未涉及"""
        kp_all = list(range(1, n_concepts+1))            # [1,2,...,88]
        kp_involve = []
        kp_free = []
        for _i in range(min_len, seq_len):  # 只需要考虑21-33题
            _temp = kp4pid[_i, :valid_len[_i]].tolist()
            _temp.sort()
            # 做题序列中每道题涉及到的kp list
            kp_involve.append(_temp)                                                # (n_kps,), list{seq_len-min_len=13}
            # 做题序列中每道题未涉及到的kp list
            kp_free.append([_item for _item in kp_all if _item not in set(_temp)])  # (n_concepts-n_kps,), list{13}
            # 题目序列本身是正序的，无需进行倒序变换（结果从第21题-第33题）

        """将几个kp空间表征拆分为：当前题目相关kp & 当前题目无关kp 两个互补的部分"""
        y_diff_involve = []
        y_diff_free = []
        y_error_involve = []
        y_error_free = []
        for _i, (_kp_i, _kp_f) in enumerate(zip(kp_involve, kp_free)):  # 索引0-12, ([1,2,3],[4,5,...,88])
            """这里append元组是为了给后续按照kp进行统计留接口"""
            y_diff_involve.append((_kp_i, y_diff[_i, _kp_i]))           # (n_kps,), (n_kps,)
            y_diff_free.append((_kp_f, y_diff[_i, _kp_f]))              # (n_concepts-n_kps,), (n_concepts-n_kps,)
            # y_error_involve.append((_kp_i, y_error_pre[_i, _kp_i]))   # 同上, for error_pre
            # y_error_free.append((_kp_f, y_error_pre[_i, _kp_f]))
            y_error_involve.append((_kp_i, y_error_diff[_i, _kp_i]))    # 同上, for error_diff
            y_error_free.append((_kp_f, y_error_diff[_i, _kp_f]))
            # 最终得到的四个list长度均为seq_len-min_len=13

        """统计y_diff（能力差分/变化）在两种情况下的水平"""
        def _temp_func(_list_a, _list_b, axis=-1, use_abs=True):
            """用于把上述操作的数值部分拆出来"""
            _out = []
            for (a_elem, b_elem) in zip(_list_a, _list_b):
                a_list = a_elem[axis]  # or 1 as index
                b_list = b_elem[axis]
                # if isinstance(a_list, np.ndarray) and isinstance(b_list, np.ndarray) and len(a_list) == len(b_list):
                #     # 若均为list且等长,则对位相除
                #     _res = [a / (b+1e-3) for a, b in zip(a_list, b_list)]
                #     # 添加至新列表c中
                #     _out.append(_res)
                if use_abs:
                    _out.append([np.abs(_a) for _a in a_list])
                else:
                    _out.append([_a for _a in a_list])  # 不需要取绝对值时
            return _out

        DiffByError_i = _temp_func(y_diff_involve, y_error_involve)  # list{13}, elem as list{n_kps,}
        DiffByError_f = _temp_func(y_diff_free, y_error_free)        # list{13}, elem as list{n_concepts-n_kps,}
        Error_i = _temp_func(y_error_involve, y_error_involve, use_abs=False)
        Error_f = _temp_func(y_error_free, y_error_free, use_abs=False)

        """将上述生成的内容保存到指定的self属性中"""
        if min_len < self.min_len:
            self.min_len = min_len            # 最小len(一般为静态)
        if seq_len > self.max_len:
            self.max_len = seq_len            # 最大len(动态)
        for _ind, _val in enumerate(range(min_len+1, seq_len+1)):     # 这里得对应上统计量具体表征的对象，是第21题至第33题的学习效果，因此均+1
            self.DiffByError_i[_val].extend(DiffByError_i[_ind])  # 记录统计量(题目涉及kp)
            self.DiffByError_f[_val].extend(DiffByError_f[_ind])  # 记录统计量(题目未涉及kp)
            self.Error_i[_val].extend(Error_i[_ind])
            self.Error_f[_val].extend(Error_f[_ind])

    def report_know_new(self, out_path='./script.png'):
        """
        TBD
        """
        # 清洗数据（不确定效果是否好，总之先留着这一步骤）
        DiffByError_i = [[] for _ in range(100)]
        Error_i = [[] for _ in range(100)]
        for _ind in range(len(self.DiffByError_i)):
            for _ii in range(len(self.DiffByError_i[_ind])):
                if self.DiffByError_i[_ind][_ii] != 0 or self.Error_i[_ind][_ii] != 0:
                    DiffByError_i[_ind].append(self.DiffByError_i[_ind][_ii])
                    Error_i[_ind].append(self.Error_i[_ind][_ii])

        DiffByError_f = [[] for _ in range(100)]
        Error_f = [[] for _ in range(100)]
        for _ind in range(len(self.DiffByError_f)):
            for _ii in range(len(self.DiffByError_f[_ind])):
                if self.DiffByError_f[_ind][_ii] != 0 or self.Error_f[_ind][_ii] != 0:
                    DiffByError_f[_ind].append(self.DiffByError_f[_ind][_ii])
                    Error_f[_ind].append(self.Error_f[_ind][_ii])

        """计算均值&标准差"""
        def calculate_mean_std(_data):
            return [(np.mean(lst), np.std(lst)) if len(lst) > 0 else None for lst in _data]

        def calculate_percentile(_data):
            """check用：实际数据中，由于是离散取值，且0的数量压倒性的多"""
            return [(np.min(lst), np.percentile(lst, 25), np.percentile(lst, 75), np.max(lst)) if len(lst) > 0 else None for lst in _data]

        # 注意pred diff是单边的
        DiffByError_i = calculate_mean_std(DiffByError_i)  # list{100}，元素为list([]或元素为np.int64)
        DiffByError_f = calculate_mean_std(DiffByError_f)  # 同上
        # error diff是双边的（看看会不会有比较严重的抵消问题）
        Error_i = calculate_mean_std(Error_i)
        Error_f = calculate_mean_std(Error_f)

        # slicing
        x = [_ind for _ind, elem in enumerate(DiffByError_i) if elem is not None]  # 获取非None元素的索引，作为data_x
        DiffByError_i = [DiffByError_i[_ind] for _ind in x]
        DiffByError_f = [DiffByError_f[_ind] for _ind in x]
        Error_i = [Error_i[_ind] for _ind in x]
        Error_f = [Error_f[_ind] for _ind in x]

        """将上述两组数据以折线的形式统一绘制到一张图中"""
        # # delta pred
        # one_tailed = True
        # a_means, a_stds = zip(*DiffByError_i)
        # b_means, b_stds = zip(*DiffByError_f)

        # delta error
        one_tailed = False
        a_means, a_stds = zip(*Error_i)
        b_means, b_stds = zip(*Error_f)

        # Plot the data
        plt.figure(figsize=(10, 6))

        # 根据单双边两种情况处理（注意这里的单边仅适用于均值趋近于0的情况）
        contourf_a = np.array(a_means) - np.array(a_stds)
        contourf_b = np.array(b_means) - np.array(b_stds)
        if one_tailed:
            contourf_a = np.where(contourf_a < 0, 0, contourf_a)
            contourf_b = np.where(contourf_b < 0, 0, contourf_b)

        # Plot data a
        plt.plot(x, a_means, label='involved kp', color='blue')
        plt.fill_between(x,
                         contourf_a,                            # 下界
                         np.array(a_means)+np.array(a_stds),    # 上界
                         color='blue',
                         alpha=0.2
                         )
        # Plot data b
        plt.plot(x, b_means, label='free kp', color='red')
        plt.fill_between(x,
                         contourf_b,
                         np.array(b_means) + np.array(b_stds),
                         color='red',
                         alpha=0.2
                         )
        # Customize the plot
        plt.xlabel('valid sequence length')
        plt.ylabel('delta error')
        plt.title('')
        plt.legend()
        plt.grid(True)
        plt.savefig(out_path)    # Save the plot as a PNG file
        plt.close()              # Close the plot to avoid displaying it

        # 还要有按kp统计的做题后学习效率图（看看不同kp之是否一致）
        pass
