import linecache
import subprocess
import sys
import json
import torch
from torch.utils.data import DataLoader


def list_to_comma_string(data_in):
    # 将列表中的每个元素转换为字符串
    string_elements = map(str, data_in)
    # 使用逗号连接所有字符串元素
    result = ",".join(string_elements)
    return result


class Batch:
    def __init__(self, data, fields, seq_len=None, kp_len=None, know=False):
        """
        240722: 无效化做题序列长度参数seq_len（暂时应该不需要截断序列长度），引入知识点padding长度参数kp_len；
        240730: 新增know参数，用于适配需要返回学生知识状态的情况.
        240813: 具体调用示例：
        :param data:      tensor_data,    # (tensor(14,150), tensor(8,), tensor(605,))
        :param fields:    self.inputs,    # ["pid", "topic", "tense", "q", "s"]
        :param seq_len:   self.seq_len,   # None(deprecated)
        :param kp_len:    self.kp_len,    # [[0,1], [1,2], [2,3], [3,13], [13,14]]
        :param know:      know=self.know  # True
        """
        self.know = know
        """判断是否需要返回学生能力/掌握特征"""
        if self.know:
            self.data = data[0]  # tensor(14,150)
            self.data_k = list(data[1:])  # [tensor(8,), tensor(605,)]
        else:
            self.data = data  # tensor(14,150)
        """题目特征相关参数"""
        self.fields = fields  # ["pid", "topic", "tense", "q", "s"]
        self.field_index = {f: i for i, f in enumerate(fields)}  # {'pid':0, 'topic':1, 'tense':2, 'q':3, 's':4}
        self.seq_len = seq_len  # None(deprecated)  若切分长序列，则指定长度为
        self.kp_len = kp_len  # [[0,1], [1,2], [2,3], [3,13], [13,14]] (沿self.data第一维拆分题目fields的索引)
        if self.kp_len is not None:
            # 生成“field：起止索引”字典，形如 {'pid':[0,1], 'topic':[1,2], 'tense':[2,3], 'q':[3,13], 's':[13,14]}
            self.kp_index = {f: i for f, i in zip(list(self.fields), list(self.kp_len))}

    def get(self, *fields):
        """
        需要实现多种调用形式：
            topic, tense, q, s, pid, know_tense, know_q = batch.get("topic", "tense", "q", "s", "pid", "know_tense", "know_q")
            q, s, pid, know = batch.get("q", "s", "pid", "know")
            q, s, pid = batch.get("q", "s", "pid")
        """
        # L = len(self.data[0])  # 截断长序列时需要
        # 先判断是否需要沿指定的self.data第一维拆分题目fields的索引
        if self.kp_len is None:
            _data_out = [self.data[self.field_index[f]] for f in fields]
        else:
            _data_out = [self.data[self.kp_index[f][0]:self.kp_index[f][1]] for f in fields]
        # 若要返回题目+学生能力/掌握特征
        if self.know:
            _data_out = _data_out + self.data_k
        # 返回 _data_out
        return _data_out


class KTData:
    """
    240821：新增功能，实现对输入数据有效取值范围的检查，剔除掉非法取值
            修改参数num_kps，新增参数num_know（问题和能力特征的取值范围，元组形式）
    """

    def __init__(
            self,
            data_path,         # v04_0807/train.txt
            inputs,            # ["pid", "q", "s"]
            num_kps,           # (dataset["n_pid"], dataset["n_questions"], None)  240821修改，新增判定有效取值范围功能
            group=None,        # dataset['n_features']=5(v04), 用于指示读取txt文件
            batch_size=1,
            seq_len=None,      # deprecated
            kp_len=None,       # [[0,1], [1, 51], [51,52]]
            shuffle=False,
            num_workers=0,
            max_len_kp=None,   # kp list的padding长度（无传入时自动获取）
            name_know=None,    # 学生能力键值对应的名称list（若需要返回学生能力）  ["ind_c", "val_c"]
            num_know=None,     # 类比num_kps，若有传入，则为各特征有效最大取值组成的元组  (dataset["n_questions"], None)  --暂时无效化
            rand_init=True,    # 能力表征填充方式（否则统一padding为-1）
            type_in='json',    # 输入数据文件格式
            n_classes=5,       # 知识点能力表征类别数
    ):
        self.n_classes = n_classes
        self.num_know = num_know
        self.type_in = type_in
        self.rand_init = rand_init
        self.seq_len = seq_len
        self.kp_len = kp_len
        if max_len_kp is None:
            if kp_len is None:
                self.max_len_kp = 1
            else:
                # self.max_len_kp = kp_len[-2][1] - kp_len[-2][0]  # 51-1 or 13-3
                self.max_len_kp = kp_len[1][1] - kp_len[1][0]  # 51-1 or 13-3
        self.num_kps = num_kps    # 240821: (dataset["n_pid"], dataset["n_questions"], None)
        if name_know is None:
            self.know = False
            self.name_know = []
        else:
            self.know = True
            self.name_know = name_know
        self.inputs = inputs
        if group is None:
            self.group = len(self.inputs)
        else:
            self.group = group
        # Lines类似于Dataset封装，__getitem__返回指定条目对应的group行内容，以list的形式
        if self.type_in == 'json':
            self.data = LinesFromJson(data_path, keys=self.inputs + self.name_know)
        else:
            self.data = Lines(data_path, group=self.group + 1)
        self.loader = DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=transform_batch,
            num_workers=num_workers,
        )

    def __iter__(self):
        return iter(self.loader)  # 将可迭代对象转换为迭代器（从而使用next()函数访问元素）

    def __len__(self):
        return len(self.data)  # 调用Lines类的__len__方法

    @staticmethod
    def pad_nested_list(nested_list, length, padding_value=-1):
        """
        将嵌套列表中的每个子列表填充到指定长度。

        :param nested_list: 需要填充的嵌套列表
        :param length: 填充后的指定长度
        :param padding_value: 用于填充的值，默认为0
        :return: 填充后的嵌套列表
        """
        padded_list = []
        for sublist in nested_list:
            # 如果为int，变为列表
            if isinstance(sublist, int):
                sublist = [sublist]
            # 如果子列表长度小于指定长度，则进行填充
            if len(sublist) < length:
                sublist = sublist + [padding_value] * (length - len(sublist))
            # 如果子列表长度大于指定长度，则进行截断
            elif len(sublist) > length:
                sublist = sublist[:length]
            padded_list.append(sublist)

        return padded_list

    @staticmethod
    def get_onehot(keys_in, values_in, n_kps, rand_init=True, rand_range=(0, 5)):
        """
        用于根据学生知识点id和掌握度，生成学生能力状态的one-hot表示
        """
        _dict = {int(keys_in[_i]): float(values_in[_i]) for _i in range(len(keys_in))}
        if rand_init:
            data_out = torch.randint(rand_range[0], rand_range[1], (n_kps,))  # (n_kps,)
        else:
            data_out = torch.zeros(n_kps) - 1  # 240821: 引入sparse损失，padding值相应地统一调整为-1
            # data_out = torch.zeros(n_kps)
        for _ind, _val in _dict.items():
            data_out[_ind] = _val
        return data_out

    def __getitem__(self, index):
        processed_data = []  # 保存读取后的题目特征
        know_line = []  # 保存读取后的学生能力/掌握特征

        """
        （1）读取数据
        self.data为Lines实例化对象，Lines[index]调用__getitem__获取条目，然后剔除第一行（做题计数）后遍历
        """
        if self.type_in == 'json':
            ind_start = 0
        else:
            ind_start = 1
        for _ind, line in enumerate(self.data[index][ind_start:]):
            """处理当前遍历行数据"""
            parsed_line = []  # 声明保存当前行处理后数据的list

            if self.type_in == 'json':
                # =================================================================================
                # (1)线上推理阶段，原始传入json字符串转为的dict或其组成的list, Line_from_json处理后为嵌套列表
                # =================================================================================
                for x in line:
                    # 将包含'_'的字符串（kp相关）切分并转换为整数列表，然后展平
                    if isinstance(x, str):
                        split_values = [int(ii) for ii in x.split('_') if ii]  # 注意要剔除掉可能出现的空字符串
                        parsed_line.append(split_values)  # list：元素为当前条目/学生做题数个list
                    else:
                        parsed_line.append(x)  # 否则直接添加当前行

            else:
                # =================================================================================
                # (2)线下训练阶段，使用txt数据文件，当前遍历行line为字符串形式，按逗号拆分出单个题目
                # =================================================================================
                for x in line.strip().split(","):
                    # 将包含'_'的字符串（kp相关）切分并转换为整数列表，然后展平
                    if '_' in x:
                        split_values = [int(i) for i in x.split('_')]
                        parsed_line.append(split_values)  # list：元素为当前条目/学生做题数个list
                    elif len(x)>0:  # else:
                        parsed_line.append(int(x))  # 否则直接将单个元素进行添加

            """判断当前行为题目特征or学生能力掌握特征，append至对应list"""
            if _ind < len(self.inputs):  # 若为题目特征（索引0 至 self.inputs 列表长度 之间）
                processed_data.append(parsed_line)  # list：元素为len(inputs)个list
            elif self.know:  # 若为学生掌握/能力表征（self.inputs 列表长度 至 group=9=dataset['n_features']之间）
                know_line.append(parsed_line[0])  # 学生能力不变因此仅一个元素无逗号分隔，只需处理下划线，parsed_line形如[[1,2,3]]

        """
        （2）240821: 新增功能，判断取值是否有效
        当前输入为 processed_data 和 know_line，均为长度等于inputs和name_know的list，元素为int或者int列表
        """
        ind_invalid = []  # 初始化需要删除的索引
        # ======================= 问题特征 =======================
        for _i, _max in enumerate(self.num_kps):  # 逐个特征进行遍历
            if _max is None:
                continue  # 无需对当前行进行有效值筛选
            else:   # 合法取值范围为[1, _max]
                # 遍历当前行processed_data[_i](list)中的所有元素(int或int列表)，检查是否存在非法值
                for _ii, _elem in enumerate(processed_data[_i]):
                    if isinstance(_elem, int):
                        if _elem > _max:  # 检测到非法值，将当前索引记录至待删除索引list
                            ind_invalid.append(_ii)
                    else:  # 当前元素为list(如题目对应的知识点可能有多个)
                        processed_data[_i][_ii] = [_x for _x in _elem if 1 <= _x <= _max]  # 直接更新原始数据
                        if not len(processed_data[_i][_ii]):
                            # 若剔除非法值后无有效知识点（列表为空），则记录当前索引至待删除索引list
                            ind_invalid.append(_ii)  # 注意是二级索引！！！
        # 遍历完成，执行剔除
        ind_invalid = list(set(ind_invalid))  # 去重
        ind_invalid = sorted(ind_invalid)     # 排序
        # 对所有list进行一致的剔除（按照非法索引list）
        for _ind, line in enumerate(processed_data):
            processed_data[_ind] = [val for ind, val in enumerate(line) if ind not in ind_invalid]

        # ======================= 能力特征 =======================
        """
        首先得逐两行读取（一项能力表征的ind和val，都是int列表，不同能力的表征很可能不同）
        首先检查能否在one-hot生成的步骤解决这个问题（如果可以则无需在此处进行）
        """
        if self.name_know is not None:  # 多一层判断是否需求返回能力相关的
            for _i in range(0, len(know_line), 2):  # 两两一组
                line_index = know_line[_i]
                line_value = know_line[_i+1]   # 二者均为元素为 int 的 list
                max_ind = self.num_know[_i]
                max_val = self.num_know[_i+1]  # 获取对应的合法最大取值
                temp_invalid = []              # 当前能力的剔除索引list
                for _ii in range(len(line_index)):
                    if max_ind is not None and line_index[_ii] > max_ind:
                        temp_invalid.append(_ii)  # 键值任一不满足时，记录非法索引
                    if max_val is not None and line_value[_ii] > max_val:
                        temp_invalid.append(_ii)  # 键值任一不满足时，记录非法索引
                # 遍历完毕后执行剔除
                temp_invalid = list(set(temp_invalid))  # 去重
                temp_invalid = sorted(temp_invalid)     # 排序
                know_line[_i] = [val for ind, val in enumerate(know_line[_i]) if ind not in temp_invalid]  # 更新know_line
                know_line[_i+1] = [val for ind, val in enumerate(know_line[_i+1]) if ind not in temp_invalid]

        """
        （3）处理题目特征(processed_data)
        max_outer_length = seq_len（当前 条目/学生 的 做题数/序列长度）
        max_inner_length = 10（特征的padding长度）
        """
        # # 最大外长
        # max_outer_length = max(len(sublist) for sublist in processed_data)
        # # 最大内长
        # max_inner_length = max(
        #     len(inner) if isinstance(inner, list) else 1 for sublist in processed_data for inner in sublist)
        temp_list = []  # 用于收集预处理+张量化后的前条目的前len(inputs)行数据
        for elem in processed_data:  # 遍历processed_data中的len(inputs)个list形式的元素（即当前条目的前len(inputs)行）
            if not any(isinstance(eee, list) for eee in elem):  # 若当前行所有元素均为单个数值而非list
                temp_list.append(torch.tensor(elem).unsqueeze(1))  # 张量化(max_outer_length,1)
            else:  # 若当前行元素中存在list，则padding至统一长度（目前只有词汇维度需要进行操作，暂不升级self.max_len_kp参数）
                for ind, xxx in enumerate(elem):
                    if isinstance(xxx, int):  # 先把所有int元素转为长度1的list
                        elem[ind] = [xxx]
                # 由于作答s的取值为0or1，因此padding值应为-1
                _temp = self.pad_nested_list(elem, self.max_len_kp, padding_value=-1)  # 再将当前行元素padding至统一长度
                temp_list.append(torch.tensor(_temp))  # 张量化(max_outer_length, max_inner_length)
        # temp_list 中顺序与 inputs 一致，可以直接cat（注意是沿着特征维），然后转置
        tensor_data = torch.cat(temp_list, dim=1).T  # (kp_len[-1][-1]-1, max_outer_length=seq_len)

        """
        （4）处理学生能力/掌握特征：两两一组，生成one-hot向量
        """
        if self.know:  # 判断是否需要返回学生能力/掌握特征
            tensor_data = [tensor_data]  # 将题目特征张量加入list
            # 遍历学生能力/掌握特征，生成one-hot张量并加入list
            for _i in range(0, len(know_line), 2):  # 两两一组
                line_index = know_line[_i]
                line_value = know_line[_i + 1]   # 二者均为元素为 int 的 list
                """
                self.num_kps 原来为 dataset["n_questions"]，适用于单个能力（知识点掌握度）的情况
                240821:
                 (dataset["n_pid"], dataset["n_questions"], None) —— 已经转为问题相关特征的有效最大取值
                新增传参 self.num_know，取值为 (dataset["n_questions"], None) —— 能力相关特征的有效最大取值
                """
                # n_kps = self.num_kps if isinstance(self.num_kps, int) else self.num_kps[_i // 2]  # self.num_kps=1827 or (7,604)
                n_kps = self.num_know if isinstance(self.num_kps, int) else self.num_know[_i]  # self.num_know=(dataset["n_questions"], None)
                # 注意随机初始化参数至对于学生能力/掌握特征生效！！！题目特征的padding在line 186，均为-1（考虑s取值0或1）
                tensor_know = self.get_onehot(
                    line_index, line_value, n_kps + 1, rand_init=self.rand_init, rand_range=(0, self.n_classes)
                )  # 返回形状为(n_kps+1,)的张量
                tensor_data.append(tensor_know)
            # list转tuple，准备进行Batch封装
            tensor_data = tuple(tensor_data)  # (tensor(14,150), tensor(8,), tensor(605,))

        """
        （5）Batch封装并返回
        """
        return Batch(
            tensor_data,  # (tensor(14,150), tensor(8,), tensor(605,))
            self.inputs,  # ["pid", "topic", "tense", "q", "s"]
            self.seq_len,  # None(deprecated)
            self.kp_len,  # [[0,1], [1,2], [2,3], [3,13], [13,14]]
            know=self.know  # True
        )


def transform_batch(batch):
    """
    作为collate_fn使用:
    :param batch: list，元素为调用KTData的__getitem__返回的Batch实例化对象
    对于：
    topic, tense, q, s, pid, know_tense, know_q = batch.get("topic", "tense", "q", "s", "pid", "know_tense", "know_q")
    此时 batch 为:
    n_batch个单样本Batch实例化对象构成的list
    此函数的任务是聚合batch中所有元素的对应属性
    其中每个元素的[tensor(1,150), tensor(1,150), tensor(10,150), tensor(1,150), tensor(1,150), tensor(8,), tensor(605,)]
    """
    # collect & merge configs
    fields, seq_len = batch[0].fields, batch[0].seq_len  # ["pid", "topic", "tense", "q", "s"] and None
    kp_len = batch[0].kp_len  # [[0,1], [1,2], [2,3], [3,13], [13,14]] (沿self.data第一维拆分题目fields的索引)
    know = batch[0].know  # True
    # collect data
    batch_data = [b.data for b in batch]  # list of tensor(14,150)
    # transpose to separate sequences
    batch_ques = list(zip(*batch_data))
    """此时batch为长度14的list，元素为长度32的tuple，再其中的元素为tensor(<=150,)"""
    # pad sequences
    batch_ques = [
        torch.nn.utils.rnn.pad_sequence(
            seqs,
            batch_first=True,
            padding_value=-1,  # 对于题目特征，统一padding值为-1
        )
        for seqs in batch_ques
    ]
    """此时batch为长度14的list，元素为tensor(32, 150)"""
    # 若还需要返回学生能力/掌握特征
    if know:
        # collect 学生能力/掌握特征 data
        batch_data_k = [b.data_k for b in batch]  # list of [tensor(8,)] 或 list of [tensor(8,), tensor(605,)]
        batch_data_k = [torch.stack(_temp, dim=0) for _temp in zip(*batch_data_k)]  # [tensor(32,8), tensor(32,605)]
        # know为True时，重新组装为list后传入Batch
        batch_ques = [batch_ques] + batch_data_k
    # 返回重新Batch封装后的结果
    return Batch(batch_ques, fields, seq_len, kp_len, know=know)


class Lines:
    """主要作用是从txt文件中读取数据条目/item对应的group行，并以list的形式返回"""

    def __init__(self, filename, skip=0, group=1, preserve_newline=False):
        self.filename = filename  # txt数据文件
        with open(filename):
            pass
        if sys.platform == "win32":
            linecount = sum(1 for _ in open(filename))
        else:
            # python subprocess 模块的 check_output 函数可以用于执行一个shell命令，并返回命令的输出内容。
            output = subprocess.check_output(("wc -l " + filename).split())  # wc -l 计算文件的行数
            linecount = int(output.split()[0])  # 只保留行数，舍弃后续输出（文件名）
        self.length = (linecount - skip) // group  # 计算数据集大小/条目数
        self.skip = skip
        self.group = group
        self.preserve_newline = preserve_newline

    def __len__(self):
        """返回数据集总条目数"""
        return self.length

    def __iter__(self):
        """按照条目进行遍历"""
        for i in range(len(self)):
            yield self[i]  # 此语法会调用__getitem__

    def __getitem__(self, item):
        d = self.skip + 1  # linechche读取从1开始
        if isinstance(item, int):
            if item < len(self):
                if self.group == 1:
                    # linechche 与传统的f = open('./test.txt','r')相比，
                    # 当所需读取的文件比较大时，linecache将所需读取的文件加载到缓存中，从而提高了读取的效率。
                    line = linecache.getline(self.filename, item + d)
                    if not self.preserve_newline:
                        line = line.strip("\r\n")
                else:
                    line = [
                        linecache.getline(self.filename, d + item * self.group + k)
                        for k in range(self.group)
                    ]
                    if not self.preserve_newline:
                        line = [l.strip("\r\n") for l in line]
                return line  # 返回的是一个当前item的group行内容为元素的list

        elif isinstance(item, slice):
            low = 0 if item.start is None else item.start
            low = _clip(low, -len(self), len(self) - 1)
            if low < 0:
                low += len(self)
            high = len(self) if item.stop is None else item.stop
            high = _clip(high, -len(self), len(self))
            if high < 0:
                high += len(self)
            ls = []
            for i in range(low, high):
                if self.group == 1:
                    line = linecache.getline(self.filename, i + d)
                    if not self.preserve_newline:
                        line = line.strip("\r\n")
                else:
                    line = [
                        linecache.getline(self.filename, d + i * self.group + k)
                        for k in range(self.group)
                    ]
                    if not self.preserve_newline:
                        line = [l.strip("\r\n") for l in line]
                ls.append(line)

            return ls

        raise IndexError


def _clip(v, low, high):
    if v < low:
        v = low
    if v > high:
        v = high
    return v


class LinesFromJson:
    """
    修改自Lines，主要作用是从json文件中读取数据条目/item对应的group行，并以list的形式返回。
    在原项目的KTData中调用形式为(读取txt文件保存的数据)：
        self.data = Lines(data_path, group=self.group + 1)
    根据接口文档，传入json文件的格式为：（一个学生一个json文件）
        {
           "pid": [50,34,45,...,newpid1,newpid2,...],     //一个学生的所有做题题目id序列，按时间排序
            "q": [5_18_11,7_16_10_101,5_13_99_100,...,],  //题目对应的知识点id,一道题目包含多个知识点则用下划线拼接
            "s": [0,1,1,...],                             //学生历史每道题对应是否答对，1对正确，0为错误
            "diff": [2,3,4,...],                          //每道题对应的难度
            "new_pid": [30,87,50,...],                    //需要预测的新题
            "new_q": [3_5_10,24_56_85_9,...]              //需要预测的新题对应的知识点
            "new_diff": [2,3,4,...],                      //需要预测的新题对应的难度
        }
    """

    def __init__(self, filenames, keys, preserve_newline=False):
        """
        :param filenames: 学生数据的json文件 或 文件list
        :param keys: 需要读取的键列表，形如["pid", "topic", "tense", "q", "s"] + ["ind_te","val_te","ind_vo","val_vo"]
                     由于字典无序，因此必须给定该参数
        :param preserve_newline: 清除数据头尾的换行符
        """
        self.is_path = True
        # 数据文件路径list
        if isinstance(filenames, list):
            self.filenames = filenames
            self.length = len(self.filenames)  # 计算数据集大小/条目数：一个学生一个json文件or字典
            if isinstance(filenames[0], dict):  # 若为dict的列表
                self.is_path = False
        else:
            self.filenames = [filenames]  # 传入单个路径or字典时
            self.length = 1
            if isinstance(filenames, dict):  # 若为单个dict
                self.is_path = False
        # 其他参数
        self.keys = keys
        self.group = len(keys)
        self.preserve_newline = preserve_newline

    def __len__(self):
        """返回数据集总条目数"""
        return self.length

    def __iter__(self):
        """按照条目进行遍历"""
        for i in range(len(self)):
            # 此语法会调用__getitem__
            yield self[i]

    def __getitem__(self, item):
        """
        slice支持完成
        :param item: 与txt文件不同，不再需要根据group计算具体行数，而是直接从数据文件路径list中读取对应索引的文件即可
        """
        # （1）若item为int索引
        if isinstance(item, int):
            if item < len(self):  # 判断传入的索引合法
                if self.is_path:
                    # 读取索引item的json文件，遍历self.keys读取对应的值
                    with open(self.filenames[item], 'r', encoding='utf-8') as file:
                        _data = json.load(file)
                    # lines = [list_to_comma_string(_data[_key]) for _key in self.keys]
                    lines = [_data[_key] for _key in self.keys]  # 240815
                    # # 清除数据头尾的换行符
                    # if not self.preserve_newline:
                    #     lines = [l.strip("\r\n") for l in lines]
                    # 返回索引item数据文件内group个键的值作为元素的list
                    return lines
                else:
                    _data = self.filenames[item]  # 找到索引为item的字典
                    # lines = [list_to_comma_string(_data[_key]) for _key in self.keys]
                    lines = [_data[_key] for _key in self.keys]  # 240815
                    # if not self.preserve_newline:
                    #     lines = [l.strip("\r\n") for l in lines]
                    return lines

        # （2）若item为切片
        elif isinstance(item, slice):
            # 起
            low = 0 if item.start is None else item.start  # 未指定时从0起
            low = _clip(low, -len(self), len(self) - 1)  # 裁剪至valid范围
            if low < 0:
                low += len(self)  # 转换为>=0表示
            # 止
            high = len(self) if item.stop is None else item.stop  # 未指定时从-1起
            high = _clip(high, -len(self), len(self))  # 裁剪至valid范围
            if high < 0:
                high += len(self)  # 转换为>=0表示
            # 记录并返回slice结果
            ls = []
            for i in range(low, high):
                if self.is_path:
                    # 读取索引i的json文件，遍历self.keys读取对应的值
                    with open(self.filenames[i], 'r', encoding='utf-8') as file:
                        _data = json.load(file)
                    # lines = [list_to_comma_string(_data[_key]) for _key in self.keys]
                    lines = [_data[_key] for _key in self.keys]  # 240815
                    # # 清除数据头尾的换行符
                    # if not self.preserve_newline:
                    #     lines = [l.strip("\r\n") for l in lines]
                    # 记录至ls
                    ls.append(lines)
                else:
                    _data = self.filenames[i]  # 找到索引为i的字典
                    # lines = [list_to_comma_string(_data[_key]) for _key in self.keys]
                    lines = [_data[_key] for _key in self.keys]  # 240815
                    # if not self.preserve_newline:
                    #     lines = [l.strip("\r\n") for l in lines]
                    ls.append(lines)
            # 返回ls，为单个索引__getitem__结果为元素的list
            return ls

        # （3）若非int索引或索引切片则报错
        else:
            raise IndexError
