import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # 确保你的图表在后台生成且不显示图形窗口
import matplotlib.pyplot as plt
import numpy as np
import torch
import tomlkit
import wandb
from tqdm import tqdm
from datetime import datetime
from dotenv import load_dotenv
from argparse import ArgumentParser
from models.data import KTData
from models.eval import Evaluator
from models.CDMTransformer_1120 import CDMTransformer      # 注意模型版本！
from models.ConceptPredictor import ConceptPredictor_GCN as ConceptPredictor
from collections import Counter


load_dotenv()
DATA_DIR = os.getenv('DATA_DIR')
parser = ArgumentParser()


"""输入传参设置"""


# general options
parser.add_argument("--device", help="device to run network on", default="cpu")
parser.add_argument("-bs", "--batch_size", help="batch size", default=8, type=int)
parser.add_argument("--type", help="which type of concept to predict: word or tense", default="word")

# dataset options
datasets = tomlkit.load(open(os.path.join(DATA_DIR, "datasets.toml")))
parser.add_argument(
    "-d",
    "--dataset",
    help="choose from a dataset",
    choices=datasets.keys(),
    default="",
    required=True,  # 240903: 知识点映射向量随dataset传入！！！
)

# model options
# CDM model
parser.add_argument("--d_model", help="CDMTransformer model hidden size", type=int, default=128)
parser.add_argument("--n_layers", help="number of layers", type=int, default=3)
parser.add_argument("--n_heads", help="number of heads", type=int, default=8)
parser.add_argument("--n_know", help="dimension of CDMTransformer model knowledge parameter", type=int, default=32)
parser.add_argument("--proj", help="projection layer before CL", action="store_true")
parser.add_argument("-f_cdm", "--cdm_from_file", help="CDMTransformer model file", required=True)
# ConceptPredictor model
parser.add_argument("--node_dim", help="concept graph node embedding size", type=float, default=128)
parser.add_argument("--h_dim", help="model hidden size", type=float, default=128)
parser.add_argument("-f_concept", "--concept_from_file", help="concept predictor model file", required=True)


"""evaluate"""


CAT_PROP = {3: 0.45922587397241776, 
            2: 0.1357168177444675, 
            1: 0.1315305278438919, 
            0: 0.11321550952887359, 
            -1: 0.16031127091034925}
for key, val in CAT_PROP.items():
    CAT_PROP[key] = 1/val
print(CAT_PROP)

def mastery_level(score):
    """
    原始能力映射函数
    """
    if score == 1:
        return 3   # "精通"
    elif score >= 0.5:
        return 2   # "熟练"
    elif score > 0:
        return 1   # "模糊"
    elif score == 0:
        return 0   # "未掌握"
    else:
        # return -1   # 与padding值对齐
        return score  # -1


def calc_kp_weight(score):
    """
    批量对应相应的权重项
    """
    output = score.clone()
    output = torch.where(score==1.0,   # "精通"
                         torch.ones_like(score)*CAT_PROP[3],  # 0.46
                         output)
    output = torch.where(score==0.5,   # "熟练"
                         torch.ones_like(score)*CAT_PROP[2],  # 0.14
                         output)
    output = torch.where(score==0.1,   # "模糊"
                         torch.ones_like(score)*CAT_PROP[1],  # 0.13
                         output)
    output = torch.where(score==0,   # "未掌握"
                         torch.ones_like(score)*CAT_PROP[0],  # 0.11
                         output)
    return output


def mastery_level_reverse(score):
    """
    (原始能力映射函数)的逆函数，并使其适用于pytorch tensor
    """
    output = score.clone()           # padding值不处理
    output = torch.where(score==3,   # "精通"
                         torch.ones_like(score)*1.0,
                         output)
    output = torch.where(score==2,   # "熟练"
                         torch.ones_like(score)*0.5,
                         output)
    output = torch.where(score==1,   # "模糊"
                         torch.ones_like(score)*0.1,
                         output)
    output = torch.where(score==0,   # "未掌握"
                         torch.zeros_like(score),
                         output)
    return output


def torch_nanmean(tensor, weight=None, padding_value=None, dim=None):
    """
    定义pytorch框架下的类numpy.nanmean操作
    """
    if padding_value is None:
        # 创建一个 mask 来标识 NaN 元素
        mask = ~ torch.isnan(tensor)
    else:
        # 同上，标识padding值
        mask = ~ (tensor == padding_value)
    
    if weight is None:
        # 将 NaN/padding值 替换为 0
        tensor_with_nan_as_zero = torch.where(mask, tensor, torch.zeros_like(tensor))
        # 计算每个维度上的有值元素个数
        count = torch.sum(mask, dim=dim, keepdim=True)
        # 计算和并除以有值元素个数，避免除以0，因此用 torch.where 来处理这种情况
        sum = torch.sum(tensor_with_nan_as_zero, dim=dim, keepdim=True)
        mean = sum / torch.where(count == 0, torch.ones_like(count), count)  # 防止分母为0
    else:
        weighted_caps = torch.mul(weight, tensor)
        mean = torch.sum(weighted_caps, dim=dim, keepdim=True)
    
    # 去掉 keepdim=True 带来的额外维度
    if dim is not None:
        mean = mean.squeeze(dim)
    
    return mean


def plot_histogram(data_in, text=None, file_out=None):
    """
    绘制直方图
    :param data_in: 1d array
    :param text: str or None
    :param file_out: str or None
    """
    # if data_in is None:
    #     # 生成正态分布示例数据
    #     np.random.seed(42)
    #     data_in = np.random.normal(loc=0, scale=1, size=1000)
    if text is None:
        text = ''

    # 使用 seaborn 绘制概率密度函数图
    plt.figure(figsize=(8, 6))

    bins = np.linspace(-1,1,31).tolist()
    plt.hist(data_in, bins=bins, color='b', alpha=0.7, edgecolor='black')  # 设定箱子数量为30

    plt.xlim([-1,1])
    # plt.ylim([0,20])  # 不设置ylim

    # 计算文本显示位置
    ax = plt.gca()
    x_limits = ax.get_xlim()
    y_limits = ax.get_ylim()
    x_text = x_limits[1] + (x_limits[1] - x_limits[0]) * 0.05  # 超出横坐标最大值的5%
    y_text = y_limits[1] + (y_limits[1] - y_limits[0]) * 0.1  # 超出纵坐标最大值的10%
    plt.text(x_text, y_text, text, fontsize=12, ha='right', va='top')  # 显示文本

    # 添加标题和标签
    plt.title('Histogram')
    plt.xlabel('Value')
    plt.ylabel('Density')

    # 显示网格
    plt.grid(True)

    # 保存
    now = datetime.now()
    now = now.strftime("%Y-%m-%d_%H:%M:%S.%f")  # 默认格式：YYYY-MM-DD HH:MM:SS.mmmmmm
    if file_out is None:
        file_out = 'pics/histogram_{}.png'.format(now)  # 默认保存文件名
    plt.savefig(file_out)
    plt.close()


def load_graph_data(node_feature_file, edge_file):
    """加载知识点图的节点特征和边"""
     # 加载节点特征
    node_features_df = pd.read_csv(node_feature_file, header=0, index_col=0)
    node_features = torch.FloatTensor(node_features_df.values)

    # 加载边信息
    edges = np.loadtxt(edge_file, dtype=float)
    if edges.shape[1] == 2:
         # 没有权重列
        edges = edges.astype(int)  # 转换为整数类型
        edge_index = torch.LongTensor(edges.T)
        edge_weight = None
    elif edges.shape[1] == 3:
        # 有权重列
        edge_index = torch.LongTensor(edges[:, :2].T.astype(int))
        edge_weight = torch.FloatTensor(edges[:, 2])
    return node_features, edge_index, edge_weight


class strategy_component:
    """
    策略的 <能力to正确率/作答对错> 功能组件
    """
    def __init__(self, 
                 datasets, 
                 args, 
                 valid_range=(0,1), 
                 padding_value=-1, 
                 hyperparams=(2/3, 0, 0.25), 
                 n_hist=150, 
                 unknown_cap=0.5
                 ):
        self.args = args
        self.padding_value = padding_value       # 填充值(未接触到的KC掌握度赋值为-1)
        self.A, self.B, self.C = hyperparams     # 直接指定超参数
        self.valid_range = valid_range
        self.n_hist = n_hist                     # 切分序列时，历史序列长度
        self.dataset = datasets[args.dataset]
        self.unknown_cap = unknown_cap

    def split_sequence(self, seq_in, n_hist=150):
        """
        用于切分序列为历史和待预报两个部分
        调用形如：
            data_hist, data_new = self.split_sequence(seq_in=(q, s, pid))
        输入为元组，元素分别为: 
            q:   (bs, max_seqlen, kp_len)
            s:   (bs, max_seqlen)
            pid: (bs, max_seqlen)
        输出为字典，形如：
            data_hist = {
                "q":   (bs, 150, kp_len),
                "s":   (bs, 150),
                "pid": (bs,150),
            }
            data_new = ...
        注意：异常情况的判断：
            若 max_seqlen < n_hist, 返回 None, None
            其他情况下，可能返回的data_new中存在全为padding值的情况，这个放到预报中处理
            此函数无需进行batch化，因为基本只tensor的切片操作
        """
        _q, _s, _pid = seq_in
        _bs, _seqlen = _s.shape
        # 特殊情况，返回None值
        if _seqlen <= n_hist:
            return None, None
        # 正常情况，直接拆分
        _data_hist = {
            "q": _q[:, :n_hist, ...],
            "s": _s[..., :n_hist],
            "pid": _pid[..., :n_hist],
        }
        _data_new = {
            "q": _q[:, n_hist:, ...],
            "s": _s[..., n_hist:],
            "pid": _pid[..., n_hist:],
        }
        return _data_hist, _data_new  # dict, dict

    def predict_answer_accuracy(self, cap_in, seq_in, unknown_cap=0.5):
        """
        (考虑将该函数batch化，同时暂不考虑diff和weight)
        根据topic掌握情况预测新题目的答对概率，于外部传入数据并调用
        输入形如：
            cap_in: (bs, n_concept)
            seq_in: dict, split_sequence 的输出，切片出来的序列末尾部分
        输出dict，包含各项需要的内容，形如：
        {
            "mastery":   (bs, n_concept),          # 输入
            "s_true":    (bs, max_seqlen-n_hist),  # 输入
            "s_pred":    (bs, max_seqlen-n_hist),  # binary, 注意padding值的处理
            "y_pred":    (bs, max_seqlen-n_hist),  # probabilistic
            "acc":       float                     # 统计准确率
            "num_valid": (bs,)                     # batch内每个样本有效序列长度
        }
        """
        # 处理输入
        new_pid = seq_in["pid"]                       # (bs, max_seqlen-n_hist)
        new_s = seq_in["s"]
        new_q = seq_in["q"]                           # (bs, max_seqlen-n_hist, kp_len)
        bs, seq_len, kp_len = new_q.shape
        n_caps = cap_in.shape[-1]                     # (bs, n_concepts)
        # 若知识点掌握度未知，则置为0.5（默认）
        _cap_in = cap_in * 1.0
        cap_in[cap_in==self.padding_value] = unknown_cap

        # 计算有效长度
        max_len = new_s!=self.padding_value
        max_len = max_len.to(torch.int).cpu().numpy()
        max_len = np.sum(max_len, axis=-1)            # numpy.ndarray: (bs,)

        # 使用 torch.gather 根据 new_q 中的索引从 cap_in 中查找值
        mask_clamp = (new_q<0) & (new_q>=n_caps)      # 掩码，后续处理padding值用
        q_clamped = new_q.clamp(min=0, max=n_caps-1)  # 截断，保证gather正常运行
        q_clamped = q_clamped.view(bs, -1)            # (bs, seq_len, kp_len) -> (bs, seq_len * kp_len)
        val_cap = cap_in.gather(1, q_clamped)         # 主操作
        val_cap = val_cap.view(bs, seq_len, kp_len)   # 还原shape
        val_cap[mask_clamp] = self.padding_value      # 应用掩膜，置为padding值（-1）—— 注意要与能力未知的情况进行区分（已提前处理）

        # 计算权重
        w_cap = calc_kp_weight(val_cap)                  # (bs, seq_len, kp_len)
        w_cap = torch.where(w_cap==self.padding_value,   # 处理padding值，权重置为0
                            torch.zeros_like(w_cap),
                            w_cap)
        # 沿着dim=-1归一化
        w_sum = torch.sum(w_cap, dim=-1)  # (bs, seq_len)
        w_sum = w_sum.unsqueeze(-1)       # (bs, seq_len, 1)
        w_cap = w_cap / w_sum

        # 求能力平均值
        mask_s = new_pid==self.padding_value          # (bs, seq_len)，用于处理序列中占位的题目
        avg_cap = torch_nanmean(val_cap, w_cap, padding_value=self.padding_value, dim=-1)  # (bs, seq_len)，全空时为0

        # 逆映射至作答正确率：对应于init_from_history中映射公式
        corr_rate = avg_cap / ((self.valid_range[1] - self.valid_range[0])/self.A)
        corr_rate = corr_rate + self.C - self.B * 0       # 暂时无效化难度项，(self.map_difficulty(diff))**2
        corr_rate = corr_rate.clamp(min=0, max=1)         # clip至有效范围     
        s_pred = torch.where(corr_rate>0.6,               # 二值化
                             torch.ones_like(corr_rate),  # 1
                             torch.zeros_like(corr_rate)  # 0
                             )
        
        # 应用掩膜，将不存在的题目的结果置为-1
        s_pred = torch.where(mask_s,                      # 二值化
                             torch.ones_like(corr_rate) * self.padding_value,  # -1
                             s_pred)
        corr_rate = torch.where(mask_s,
                                torch.ones_like(corr_rate) * self.padding_value,
                                corr_rate)
        
        # 计算ACC
        acc = (new_s==s_pred).to(torch.float)
        total_num = torch.sum(torch.ones_like(mask_s).to(torch.float) - mask_s.to(torch.float))
        total_correct_num = torch.sum(acc - mask_s.to(torch.float))  # padding值的位置必定相等，相减后为0
        acc = float(total_correct_num / total_num)

        # 计算答对占比
        pass

        # 整合为输出格式并返回
        output = {
            "mastery": _cap_in,         # 输入
            "s_true": new_s,            # 输入
            "s_pred": s_pred.to(torch.int),  # binary, padding值默认-1
            "y_pred": corr_rate,        # probabilistic
            "acc": acc,                 # batch统计准确率（total）
            "num_valid": max_len        # batch内每个样本有效序列长度
        }
        return output

    def preprocess_data(self, args=None):
        """
        根据dataset获取toml文件中对应信息，并进行dataloader封装
        """
        if args == None:
            args = self.args
        # 根据dataset获取toml文件中对应信息
        dataset = self.dataset
        seq_len = dataset["seq_len"] if "seq_len" in dataset else None
        kp_len = dataset["kp_len"] if "kp_len" in dataset else None  # 新增知识点长度参数，同时将其转换为嵌套列表
        if kp_len is not None:
            kp_len = list(kp_len)
            for ind, temp in enumerate(kp_len):
                if isinstance(temp, int):
                    continue
                else:
                    kp_len[ind] = list(kp_len[ind])
        # dataloader封装
        test_data = KTData(
            data_path=os.path.join(DATA_DIR, dataset["test"]),
            inputs=dataset["inputs"],
            num_kps=(dataset["n_pid"], dataset["n_questions"], None),  # 240821: 适用于新功能，对输入数据的取值范围进行检查
            group=dataset['n_features'],
            seq_len=seq_len,                          # deprecated
            kp_len=kp_len,                            # [[0,1], [1, 51], [51,52]]
            batch_size=args.batch_size,
            shuffle=False,
            name_know=dataset["inputs_know"],         # 同上，适配调整后的KTData
            num_know=(dataset["n_questions"], None),  # 240821: 适用于新功能，对输入数据的取值范围进行检查（若有多个能力指标，则效仿num_kps传入元组）
            rand_init=False,                          # 同上
            type_in='txt',
        )
        return test_data

    def get_pretrained_models(self, args=None):
        """
        读取预训练模型
        """
        if args == None:
            args = self.args
        dataset = self.dataset

        # 读取知识点映射向量（与train.sh保持一致，作为不可训练的固定参数，作为qemb的初始化使用）
        q_emb = dataset['q_emb'] if 'q_emb' in dataset else None
        if q_emb is not None:
            q_emb = np.load(q_emb)                                                          # np.ndarray{n_questions,128(d_model)}
            q_emb = np.concatenate((np.zeros((1, q_emb.shape[-1])), q_emb), axis=0)         # np.ndarray{n_questions,128}
            q_emb = torch.from_numpy(q_emb).float().to(args.device)
        
        # 读取预训练的CDM模型
        cdm_model = CDMTransformer(
            n_concepts=dataset["n_questions"],
            n_pid=dataset["n_pid"],
            path_diff=None,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_attn_heads=args.n_heads,
            n_know=args.n_know,
            # lambda_cl=args.lambda_cl,      # 同上，eval模式下无用
            # dropout=args.dropout,
            proj=args.proj,
            # hard_neg=args.hard_neg,
            # window=args.window,
            qemb_matrix=q_emb,               # 240903: 新增知识点映射向量（与train.sh保持一致即可，作为不可训练的固定参数，作为qemb的初始化使用）
        )
        cdm_model.load_state_dict(torch.load(args.cdm_from_file, map_location=lambda s, _: s))
        cdm_model.to(args.device)
        cdm_model.eval()

        """读取预训练的知识点映射模型"""
        z2concept_model = ConceptPredictor(
            input_dim=args.d_model, 
            n_classes=dataset['n_classes'], 
            node_feature_dim=args.node_dim,  # 新增gcn相关参数
            in_channels=args.h_dim,
        )
        z2concept_model.load_state_dict(torch.load(args.concept_from_file, map_location=lambda ss, _: ss))
        z2concept_model.to(args.device)
        z2concept_model.eval()               # 所有模型均置于eval模式

        return cdm_model, z2concept_model
    
    def eval(self, args=None):
        """
        测试使用模型预测的能力能否准确预测答对概率
        """
        if args == None:
            args = self.args
        dataset = self.dataset
        
        # 获取dataloader
        test_data = self.preprocess_data(args)

        # 读取预训练模型，置于eval模式和相同device
        cdm_model, z2concept_model = self.get_pretrained_models(args)
        cdm_model.to(args.device)
        cdm_model.eval()
        z2concept_model.to(args.device)
        z2concept_model.eval()

        # 获取知识点图的节点特征和边
        node_feature_file = os.path.join(DATA_DIR, dataset['node_feature'])
        edge_file = os.path.join(DATA_DIR, dataset['edge'])
        node_features, edge_index, edge_weight = load_graph_data(node_feature_file, edge_file)  # edge weight为None
        node_features = node_features.to(args.device)                                           # (94, 128)
        edge_index = edge_index.to(args.device)                                                 # (2, 172)

        # 主要流程：拆分数据，先用前150预报能力，再用后续的预报正确率
        count_total = 0
        count_correct = 0
        with torch.no_grad():
            it = tqdm(iter(test_data))
            for batch in it:
                # 提取batch数据（注意填充值默认为-1，存在于q中，以及对齐batch时较短的序列数据的末尾）
                q, s, pid, concept = batch.get("q", "s", "pid")  # 注意 q 是长度为 max_len_kp 的 list
                if not isinstance(q, torch.Tensor):
                    q = torch.stack(q, dim=-1)                   # 形如：50x(bs,36) -> (bs,36,50)
                    q = q.to(args.device)
                    s = s[0].to(args.device)
                    pid = pid[0].to(args.device)
                    # concept = concept.to(args.device)          # (bs,1828) -> (bs,1,1828)
                    concept = concept.cpu().numpy()
                else:  # 适用于single-kc的情况（默认不会触发）
                    q = q.to(args.device)
                    s = s.to(args.device)
                    pid = pid.to(args.device)
                    # concept = concept.to(args.device)
                    concept = concept.cpu().numpy()
                
                # 拆分数据
                data_hist, data_new = self.split_sequence(seq_in=(q, s, pid), n_hist=self.n_hist)
                # 若batch内所有数据均太短，则跳过
                if data_hist is None:
                    continue
                # (历史数据) CDM模型正向传播，获得稠密能力表征z
                _, z, *_ = cdm_model.predict(data_hist["q"], data_hist["s"], data_hist["pid"])
                # 调用CDM模型的concept_map方法，获得知识点映射
                q_, k_, v_ = cdm_model.concept_map_new(z, data_hist["s"], n=1)
                # 知识点映射模型正向传播，获得知识点层面的能力预测（241011: 需要相应地传入gcn参数）
                y = z2concept_model.predict(q_, k_, v_, node_features, edge_index, edge_weight=None)  # (bs, n_concepts)

                # 将模型预测的分类结果重新映射至0-1区间
                y = mastery_level_reverse(y)  # 0-1映射值

                # (末尾数据) 调用相应方法，预测作答正确率
                result_batch = self.predict_answer_accuracy(y, data_new, unknown_cap=self.unknown_cap)
                valid_batch = np.sum(result_batch["num_valid"])  # numpy.ndarray
                count_total += int(valid_batch)
                count_correct += int(valid_batch * result_batch["acc"])

                print("========================================================================")
                print(result_batch["y_pred"][2])
                print(result_batch["s_true"][2])
                print(result_batch["s_pred"][2])
                print("========================================================================")

        # 计算并print总体acc
        print("===========================================================================")
        print("total acc is {:.2f}%".format(count_correct/count_total * 100))
        print("===========================================================================")
        

def main(args):
    
    """根据dataset获取toml文件中对应信息"""
    dataset = datasets[args.dataset]
    seq_len = dataset["seq_len"] if "seq_len" in dataset else None
    kp_len = dataset["kp_len"] if "kp_len" in dataset else None  # 新增知识点长度参数，同时将其转换为嵌套列表
    if kp_len is not None:
        kp_len = list(kp_len)
        for ind, temp in enumerate(kp_len):
            if isinstance(temp, int):
                continue
            else:
                kp_len[ind] = list(kp_len[ind])
    
    # 240903新增: 读取知识点映射向量（与train.sh保持一致即可，作为不可训练的固定参数，作为qemb的初始化使用）
    q_emb = dataset['q_emb'] if 'q_emb' in dataset else None
    if q_emb is not None:
        q_emb = np.load(q_emb)                                                          # np.ndarray{n_questions,128(d_model)}
        q_emb = np.concatenate((np.zeros((1, q_emb.shape[-1])), q_emb), axis=0)         # np.ndarray{n_questions,128}
        q_emb = torch.from_numpy(q_emb).float().to(args.device)

    """dataloader封装"""
    test_data = KTData(
        data_path=os.path.join(DATA_DIR, dataset["test"]),
        inputs=dataset["inputs"],
        num_kps=(dataset["n_pid"], dataset["n_questions"], None),  # 240821: 适用于新功能，对输入数据的取值范围进行检查
        group=dataset['n_features'],
        seq_len=seq_len,  # deprecated
        kp_len=kp_len,  # [[0,1], [1, 51], [51,52]]
        batch_size=args.batch_size,
        shuffle=False,
        name_know=dataset["inputs_know"],         # 同上，适配调整后的KTData
        num_know=(dataset["n_questions"], None),  # 240821: 适用于新功能，对输入数据的取值范围进行检查（若有多个能力指标，则效仿num_kps传入元组）
        rand_init=False,                          # 同上
        type_in='txt',
    )

    """读取预训练的CDM模型"""
    cdm_model = CDMTransformer(
        n_concepts=dataset["n_questions"],
        n_pid=dataset["n_pid"],
        path_diff=None,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_attn_heads=args.n_heads,
        n_know=args.n_know,
        # lambda_cl=args.lambda_cl,                                # 同上，eval模式下无用
        # dropout=args.dropout,
        proj=args.proj,
        # hard_neg=args.hard_neg,
        # window=args.window,
        qemb_matrix=q_emb,                                         # 240903: 新增知识点映射向量（与train.sh保持一致即可，作为不可训练的固定参数，作为qemb的初始化使用）
    )
    cdm_model.load_state_dict(torch.load(args.cdm_from_file, map_location=lambda s, _: s))
    cdm_model.to(args.device)
    cdm_model.eval()

    """读取预训练的知识点映射模型"""
    z2concept_model = ConceptPredictor(
        input_dim=args.d_model, 
        n_classes=dataset['n_classes'], 
        node_feature_dim=args.node_dim,  # 新增gcn相关参数
        in_channels=args.h_dim,
    )
    z2concept_model.load_state_dict(torch.load(args.concept_from_file, map_location=lambda ss, _: ss))
    z2concept_model.to(args.device)
    z2concept_model.eval()  # 所有模型均置于eval模式

    # 获取知识点图的节点特征和边
    node_feature_file = os.path.join(DATA_DIR, dataset['node_feature'])
    edge_file = os.path.join(DATA_DIR, dataset['edge'])
    node_features, edge_index, edge_weight = load_graph_data(node_feature_file, edge_file)  # edge weight为None
    node_features = node_features.to(args.device)  # (94, 128)
    edge_index = edge_index.to(args.device)        # (2, 172)

    """评估测试集"""
    evaluator = Evaluator()
    with torch.no_grad():
        it = tqdm(iter(test_data))
        for batch in it:
            # 提取batch数据
            q, s, pid, concept = batch.get("q", "s", "pid")  # q比较特殊，是长度为 max_len_kp 的 list
            if not isinstance(q, torch.Tensor):
                q = torch.stack(q, dim=-1)  # 50x(32,36) -> (32,36,50)
                q = q.to(args.device)
                s = s[0].to(args.device)
                pid = pid[0].to(args.device)
                concept = concept.to(args.device)  # (bs,1828) -> (bs,1,1828)
            else:
                q = q.to(args.device)
                s = s.to(args.device)
                pid = pid.to(args.device)
                concept = concept.to(args.device)
            
            # CDM模型正向传播，获得稠密能力表征z
            _, z, *_ = cdm_model.predict(q, s, pid)

            """240825"""
            # 调用CDM模型的concept_map方法，获得知识点映射
            # concept_embed = cdm_model.concept_map(z, s, n=3)
            q_, k_, v_ = cdm_model.concept_map_new(z, s, n=1)
            # 知识点映射模型正向传播，获得知识点层面的能力预测（241011: 需要相应地传入gcn参数）
            y = z2concept_model.predict(q_, k_, v_, node_features, edge_index, edge_weight=None)  # (bs, n_concepts)
            # Evaluator.evaluate_know将当前batch数据的预报值&真实值处理为适用于sparse损失的形式，保存至其属性中
            concept = concept.cpu().numpy()
            evaluator.evaluate_know(concept, y)

    # 遍历完整个测试集后，直接从evaluator的y_pred和y_true属性中读取已经mask好的结果（均为int列表）
    counter_pred = Counter(evaluator.y_pred)
    total_pred = sum(counter_pred.values())
    counter_pred = dict(sorted(counter_pred.items()))
    prec_pred = {item: (count / total_pred) * 100 for item, count in sorted(counter_pred.items())}

    counter_true = Counter(evaluator.y_true)
    total_true = sum(counter_true.values())
    counter_true = dict(sorted(counter_true.items()))
    prec_true = {item: (count / total_true) * 100 for item, count in sorted(counter_true.items())}

    counter_diff = Counter([_a-_b for _a, _b in zip(evaluator.y_pred, evaluator.y_true)])
    total_diff = sum(counter_diff.values())
    counter_diff = dict(sorted(counter_diff.items()))
    prec_diff = {item: (count / total_diff) * 100 for item, count in sorted(counter_diff.items())}

    # print
    print('---------------------------------------------------------------------------------------------')
    print('y_pred:{}'.format(counter_pred))
    print('y_pred in percentage:{}'.format(prec_pred))
    print('---------------------------------------------------------------------------------------------')
    print('y_true:{}'.format(counter_true))
    print('y_true in percentage:{}'.format(prec_true))
    print('---------------------------------------------------------------------------------------------')
    print('y_pred-y_true:{}'.format(counter_diff))
    print('y_pred-y_true in percentage:{}'.format(prec_diff))
    print('---------------------------------------------------------------------------------------------')

    # 调用Evaluator.report_know方法，计算并保存评估指标
    r = evaluator.report_know()
    print(r)
    """{
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }"""


if __name__ == "__main__":

    args_in = parser.parse_args()
    print(args_in)
    
    # main(args_in)  # print(r)，输出多项评估指标的字典
    # print(args_in)

    # 验证实验
    eval_exp = strategy_component(datasets, args_in, unknown_cap=0.0)  # 其他参数默认
    eval_exp.eval()
