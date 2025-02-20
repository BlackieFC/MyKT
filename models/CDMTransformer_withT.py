import torch
from torch import nn
import math
import random
import torch.nn.functional as F
# from utils import read_p_diff
from torch.nn import init
import gc
import csv


MIN_SEQ_LEN = 5


# ==========================================================================================
# 关于t的引入，索性不正则化了，直接搞一个类似DDPM里T的嵌入层
# ==========================================================================================


class Swish(nn.Module):
    """
    Swish激活函数
    """
    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    """
    Create sinusoidal timestep embeddings.
    Args:
        T        : 1000           指定整个空间对应t的范围（还真就大概等于1000）
        d_model  : ch = 128       三角函数嵌入特征维度
        dim      : ch * 4 = 512   MLP ratio
    输入：(n,)
    输出：(n, dim)
    """
    def __init__(self, T, d_model, dim):
        assert d_model % 2 == 0  # 要求网络各深度通道数为偶数
        super().__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)  # (64,)，取值从0至9.06
        emb = torch.exp(-emb)                                               # (64,)，取值从1至1.15e-4
        pos = torch.arange(T).float()          # (T,)，取值从0至T-1
        emb = pos[:, None] * emb[None, :]      # (T,1) * (1,64) = (T,64)  先将标量t映射至64维矢量
        assert list(emb.shape) == [T, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)   # (T,64)和(T,64)新增最后一维堆叠-->(T,64,2)
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view(T, d_model)  # reshape： (T,64,2) --> (T,128)

        """
        nn.Embedding.from_pretrained：使用预训练好的词向量初始化，参数 freeze=True(默认使用固定编码)
        参考：https://zhuanlan.zhihu.com/p/607515931
        """
        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),  # 输入(N,) --> 输出(N, dim=d_model)
            nn.Linear(d_model, dim),            # (N, d_model) --> (N, dim=512)
            Swish(),                            # x * torch.sigmoid(x)
            nn.Linear(dim, dim),                # (N, dim) --> (N, dim)
        )
        """调用initialize()初始化参数"""
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                """初始化FC层的权重和bias"""
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, t):
        """前向传播"""
        emb = self.timembedding(t)  # 输入(N,) --> 输出(N, dim)
        return emb


class ParametricSigmoid(nn.Module):
    """
    含参的sigmoid，主要用于缓解正确率集中于1一端的问题，使用示例：
        logits = torch.tensor([0.5, 1.0, 1.5, 2.0], requires_grad=True)
        parametric_sigmoid = ParametricSigmoid()
        probabilities = parametric_sigmoid(logits)  # 前向传播
    """
    def __init__(self, freeze=False):
        super(ParametricSigmoid, self).__init__()
        # 初始化缩放参数a和偏移参数b
        self.freeze = freeze
        self.a = None  # nn.Parameter(torch.tensor(1.0))
        self.b = None  # nn.Parameter(torch.tensor(0.0))
        self.initialize()

    def forward(self, x):
        # 计算含参的 Sigmoid 函数
        return torch.sigmoid(self.a * x + self.b)

    def initialize(self, init_a=1.0, init_b=0.0):
        """
        初始化缩放参数a和偏移参数b（手动调用防止被覆盖）
        :param init_a: 缩放参数a的初始值
        :param init_b: 偏移参数b的初始值
        """
        self.a = nn.Parameter(torch.tensor(init_a), requires_grad=not self.freeze)
        self.b = nn.Parameter(torch.tensor(init_b), requires_grad=not self.freeze)


class exer_embedding(nn.Module):
    """
    用于：题目embedding（题目ID -> 归一化的TF-IDF）
        250102：需要移除ID转索引的对齐步骤 —— 在读取数组时进行
    """
    def __init__(self,
                 emb_table,
                 freeze=True
                 ):
        super(exer_embedding, self).__init__()
        dim_in, dim_out = emb_table.shape  # 比如(n_stu, d_model)
        self.emb_table = emb_table         # 权重本体
        self.freeze = freeze               # 不可训练
        # 声明嵌入层
        self.embed_layer = nn.Embedding(dim_in, dim_out)
        # 调用权重初始化方法
        self.initialize()

    def initialize(self):
        """
        需要注意在模型中调用时，不要被其他初始化方法覆盖掉
        """
        # 将预训练的权重复制到Embedding层的权重中
        self.embed_layer.weight.data.copy_(self.emb_table)
        # 确保不需要更新预训练的嵌入层参数
        if self.freeze:
            self.embed_layer.weight.requires_grad = False

    def forward(self, x_in):
        return self.embed_layer(x_in)


class kc_embedding(nn.Module):
    """
    用于：KC embedding（KC ID -> KC 文本嵌入）
    区别主要在于forward方法，传入multi-hot，返回加权平均
        250102：需要移除ID转索引的对齐步骤 —— 在读取数组时进行
    """
    def __init__(self,
                 emb_table,
                 freeze=True
                 ):
        super(kc_embedding, self).__init__()
        num_embeddings, embedding_dim = emb_table.shape
        self.emb_table = emb_table
        self.freeze = freeze
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.initialize()

    def initialize(self):
        self.embedding.weight.data.copy_(self.emb_table)  # 将预训练的权重复制到Embedding层的权重中
        if self.freeze:
            self.embedding.weight.requires_grad = False   # 确保不需要更新预训练的嵌入层参数

    def forward(self, weights):
        """
        weights: batch_size x num_embeddings (normalized weights associated with each embedding)
        """
        # # # Ensure weights are normalized (assert 表达式 ，"错误信息")
        # # assert torch.allclose(weights.sum(dim=1), torch.ones_like(weights.sum(dim=1))), "Weights must be normalized to 1"
        # # Get the embeddings for all possible indices
        # all_embeddings = self.embedding.weight              # num_embeddings, embedding_dim
        # # Perform weighted average of the embeddings
        # weighted_avg = torch.mm(weights, all_embeddings)    # n_batch, embedding_dim
        # return weighted_avg
        return self.embedding(weights)


# ==========================================================================================
# DTransformer相关类（只保留最终迭代版本）
# ==========================================================================================


class CDMTransformer_0102(nn.Module):
    """
    注意：全部的嵌入层不进行ID的平移，索引值极为ID值！！！
        1120相较于1119:
            输入s从0-1变为0-100的百分整数，对应地也换用TimeEmbedding，
            输出仍然为正确率的logit，且loss换用回归的（但要注意还原为百分数后计算）
        1125相较于1120:
            简化代码，
            验证 embedding 优化的效果.
        0102相较于1125:
            从完全的 ID-based 转为 文本嵌入based（目前只有 KC的文本嵌入，题目的 KC-wise 归一化 TF-IDF 权重）
    """
    def __init__(self,
                 # n_concepts,      # kc总数
                 # n_pid,           # 题库大小
                 emb_table_exer,    # 题目嵌入层freeze权重: (exer_n, knowledge_n)
                 emb_table_kc,      # KC 嵌入层freeze权重: (knowledge_n, d_model)
                 d_model=128,       # 特征维度
                 d_ff=256,          # 隐藏层维度
                 n_attn_heads=8,    # mha头数
                 n_know=16,         # know_param大小
                 n_layers=3,        # attention层堆叠数量
                 dropout=0.3,       # dropout
                 window=1,          # 向前预报时的窗口大小
                 n_y=100,           # 做题正确率倍率（百分数）
                 n_t=500,           # 做题时间上限
                 p_ratio=0.1,       # loss中 performance 项的倍率
                 use_param_sigmoid=True  # 使用含参sigmoid函数
                 ):
        super().__init__()
        # 需要进行嵌入的特征总数
        # self.n_concepts = n_concepts                 # 70
        self.n_y = n_y                                 # 100
        self.n_t = n_t                                 # 500 or 1000(s)
        self.p_ratio = p_ratio                         # 0.1
        self.emb_dim = emb_table_kc.shape[1]           # 1024
        n_pid, self.n_concepts = emb_table_exer.shape  # 677, 70

        """KC嵌入层：迭代"""
        # self.q_embed = nn.Embedding(n_concepts + 1, d_model)
        # self.q_diff_embed = nn.Embedding(self.n_concepts + 1, d_model)  # 区分度嵌入：d_ct summarizes the variations of questions within this concept
        self.q_embed = kc_embedding(emb_table_kc, freeze=True)            # BGE-large文本嵌入长度为1024
        self.q_embed_proj = nn.Linear(self.emb_dim, d_model)              # KC嵌入：后接全连接层(+sigmoid函数)，映射至d_model(128)维
        self.q_diff_embed = nn.Linear(self.emb_dim, d_model)              # KC难度表征：后接全连接层(+sigmoid函数)，映射至d_model(128)维

        """题目嵌入：迭代"""
        self.p_embed = exer_embedding(emb_table_exer, freeze=True)        # 题目的KC-wise 归一化 TF-IDF向量（长度70）
        # self.p_diff_embed = nn.Embedding(n_pid + 1, 1)                  # u_q 当前题目难度/区分度嵌入
        self.p_diff_embed = nn.Linear(self.n_concepts, 1)      # 题目难度表征：后接全连接层(+sigmoid函数)，映射至1维标量

        """响应嵌入层：迭代"""
        # self.s_embed = nn.Embedding(2, d_model)                         # binary对错信息
        # self.s_diff_embed = nn.Embedding(2, d_model)                    # 区分度嵌入
        self.s_embed = TimeEmbedding(101, d_model, d_model // 2)       # 正确率百分数（只占一般的特征维度，后续和用时的嵌入cat起来）
        self.s_diff_embed = TimeEmbedding(101, d_model, d_model // 2)  # 区分度嵌入

        """作答用时嵌入层：迭代"""
        # self.t_embed = nn.Embedding(self.n_t + 1, d_model)              # 普通嵌入层
        # self.t_diff_embed = nn.Embedding(self.n_t + 1, d_model)         # 区分度嵌入
        self.t_embed = TimeEmbedding(self.n_t + 1, d_model, d_model//2)   # 正弦编码
        self.t_diff_embed = TimeEmbedding(self.n_t + 1, d_model, d_model//2)  # 同上

        """MHA layers"""
        self.n_heads = n_attn_heads                                                             # 8
        # self.block1 = CDMTransformerLayer(d_model, n_attn_heads, dropout=dropout)             # q,q,q
        self.block1 = CDMTransformerLayers(d_model, n_attn_heads, n_layers=4, dropout=dropout)  # 堆叠SA层数
        # self.block2 = CDMTransformerLayer(d_model, n_attn_heads, dropout=dropout)             # s,s,s
        self.block2 = CDMTransformerLayers(d_model, n_attn_heads, n_layers=4, dropout=dropout)  # 堆叠SA层数
        self.block3 = CDMTransformerLayer(d_model, n_attn_heads, dropout)                       # hq,hq,hs，得到问题层面掌握度
        self.block4 = CDMTransformerLayer(d_model, n_attn_heads, dropout, kq_same=False)        # know_param, hq, 问题层面掌握度
        # self.block5 = SelfAttentionLayer(d_model, n_attn_heads, dropout)                      # c_emb的自注意力
        self.block5 = SelfAttentionLayer(self.emb_dim, n_attn_heads, dropout)                   # 250102：c_emb使用了文本嵌入
        self.block6 = SelfAttentionLayer(d_model, n_attn_heads, dropout, kq_same=False)         # readout —— 正确率
        self.block7 = SelfAttentionLayer(d_model, n_attn_heads, dropout, kq_same=False)         # readout —— 用时

        """
        know_param 两种定义方式：
            （1）直接声明
            （2）通过c_emb 获取
        """
        self.n_know = n_know  # 16
        # （1）直接声明
        # self.know_params = nn.Parameter(torch.randn(n_know, d_model))
        # torch.nn.init.uniform_(self.know_params, -1.0, 1.0)
        # （2）后续通过c_emb 获取（相较前者强化了know_param作为稠密知识向量与KC嵌入之间的联系）
        self.know_params = None
        self.c_k_linear = nn.Sequential(                        # 250102：文本嵌入based模型中实际调用时输入为(1,1024)
            # nn.Linear(d_model, n_know * d_model * 2),         # 128 -> 128*16*2
            nn.Linear(self.emb_dim, n_know * d_model * 2),      # 250102：1024 -> 128*16*2
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(n_know * d_model * 2, n_know * d_model),  # 128*16*2 -> 128*16，方便后续拆成(n_know, d_model)的形式
        )

        """输出层(y_pred)"""
        self.out = nn.Sequential(
            nn.Linear(d_model * 2, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_ff // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff // 2, 1),
        )

        """输出层(t_pred)"""
        self.out_t = nn.Sequential(
            nn.Linear(d_model * 2, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_ff // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff // 2, 1),
        )

        """含参的sigmoid（新增开关选项）"""
        self.param_sigmoid_y = ParametricSigmoid(freeze=not use_param_sigmoid)
        self.param_sigmoid_t = ParametricSigmoid(freeze=not use_param_sigmoid)

        """其他参数"""
        self.dropout_rate = dropout   # 0.3
        self.n_layers = n_layers      # 3
        self.window = window          # 1

    def device(self):
        return next(self.parameters()).device

    def embedding(self, q, s, pid, t):
        """
        基于Rasch 模型计算question embeddings and question-response embeddings
        :param q:   (bs, seq_len, ks_len)
        :param s:   (bs, seq_len)
        :param pid: (bs, seq_len)
        :param t:   (bs, seq_len)
        :return:
        """
        if q.ndim == 2:                                           # 检查张量维度
            q = q.unsqueeze(-1)                                   # bs, seqlen -> bs, seqlen, ks_len=1
        lens = (s >= 0).sum(dim=1)                                # (bs, seqlen) -> (bs,)，batch中每个学生的实际序列长度
        q = q.masked_fill(q < 0, 0)                               # 将padding值置为0
        s = s.masked_fill(s < 0, 0)
        pid = pid.masked_fill(pid < 0, 0)
        t = t.masked_fill(t < 0, 0)                               # 241125：对t进行相同处理 —— 将所有padding值（-1）置为0 —— 要求0在duration的真实数据中不出现

        """c_ct is the embedding of the concept this question covers"""
        _q_emb = self.q_embed(q)                                  # (bs, seqlen, kp_len) -> (bs, seqlen, kp_len, 1024)
        q_emb = self.q_embed_proj(_q_emb)                         # (bs, seqlen, kp_len, 1024) -> (bs, seqlen, kp_len, d_model)
        q_emb = q_emb.mean(dim=-2, keepdim=False)                 # multi-KCs，对KC维度求平均，返回 (bs, seqlen, d_model)

        """e_(ct,rt) = g_rt + c_ct  e_(ct,rt): concept-response embedding"""
        t_emb = self.t_embed(t)                                   # (bs, seq_len) -> (bs, seqlen, d_model//2)
        s_emb = self.s_embed(s)                                   # 同上
        """241125: 响应由作答正确率&用时两部分的嵌入concat而成（在此基础上再加上q_emb，即以kc嵌入为中心）"""
        s_emb = torch.cat((s_emb, t_emb), dim=-1) + q_emb  # (bs, seqlen, d_model)

        # u_q 当前problem难度
        p_diff = self.p_embed(pid)                                 # (bs, seqlen) -> (bs, seqlen, 70+1)
        p_diff = self.p_diff_embed(p_diff)                         # (bs, seqlen, 70+1) -> (bs, seqlen, 1)

        # d_ct 总结了包含当前question（concept）的problems（questions）的变化
        q_diff_emb = self.q_diff_embed(_q_emb)                     # (bs, seqlen, kp_len, 1024) -> (bs, seqlen, kp_len, d_model)
        q_diff_emb = q_diff_emb.mean(dim=-2, keepdim=False)        # multi-KCs，对KC维度求平均，返回 (bs, seqlen, d_model)

        """题目嵌入方程：q_t = c_ct + uq *d_ct"""
        q_emb += q_diff_emb * p_diff                               # (bs, seqlen, d_model) * (bs, seqlen, 1)

        """
        响应嵌入方程子项：
            f = s_diff_embed + d_ct
            f summarizes the variations of learning activities within this concept
        """
        t_diff_emb = self.t_diff_embed(t)                          # (bs, seq_len) -> (bs, seqlen, d_model//2)
        s_diff_emb = self.s_diff_embed(s)                          # 同上
        s_diff_emb = torch.cat((s_diff_emb, t_diff_emb), dim=-1) + q_diff_emb  # (bs, seqlen, d_model)
        """响应嵌入方程：s_t = e_(ct,rt) + u_q * f"""
        s_emb += s_diff_emb * p_diff

        return q_emb, s_emb, lens, p_diff

    def forward(self, q_emb, s_emb, lens):
        """
        forward 一般在其他方法，如 predict 中进行调用，会先于其调用embedding，然后将嵌入好的内容作为q,s,pid传入
            q_emb: (bs, seqlen, d_model)
            s_emb: (bs, seqlen, d_model)
            lens:  (bs,)，batch内各样本的有效长度
        """
        # （1）前置的几个MHA层，获取题目层面掌握度
        if self.n_layers == 1:
            hq = q_emb
            p, q_scores = self.block1(q_emb, q_emb, s_emb, lens, peek_cur=True)
        elif self.n_layers == 2:
            hq = q_emb
            hs, _ = self.block2(s_emb, s_emb, s_emb, lens, peek_cur=True)
            p, q_scores = self.block3(hq, hq, hs, lens, peek_cur=True)
        else:  # self.n_layers == 3:
            hq, _ = self.block1(q_emb, q_emb, q_emb, lens, peek_cur=True)  # (bs, seqlen, d_model)
            hs, _ = self.block2(s_emb, s_emb, s_emb, lens, peek_cur=True)  # (bs, seqlen, d_model)
            p, q_scores = self.block3(hq, hq, hs, lens, peek_cur=True)     # (bs, seqlen, d_model), (bs, n_heads, seqlen, seqlen)

        """
        （2）采用 know_param 定义（2）：计算c_emb，并通过其 self attention 计算得到 know_params
        250102：
            在文本嵌入based模型中，self.q_embed 的权重形状为(71, 1024)，即(self.n_concept+1, self.emb_dim)
        """
        bs, seqlen, d_model = p.size()
        n_know = self.n_know                                                        # 16
        c_emb = self.q_embed.embedding.weight[1:].unsqueeze(0)                      # (1, n_concepts=70, 1024) 排除第一个未使用的嵌入（id=0）
        c, _ = self.block5(c_emb, c_emb, c_emb)                                     # kc嵌入的自注意力，(1, 70, 1024)
        self.know_params = self.c_k_linear(c.mean(dim=1)).reshape(n_know, d_model)  # 输出 (1, n_know * d_model) -> (n_know, d_model)

        # （3）获取稠密能力表征z（知识层面掌握表征）
        query = (
            self.know_params[None, :, None, :]                                      # (1, n_know, 1, d_model)
            .expand(bs, -1, seqlen, -1)                                             # (bs, n_know, seqlen, d_model)
            .contiguous()
            .view(bs * n_know, seqlen, d_model)                                     # (bs * n_know, seqlen, d_model)
        )
        hq = hq.unsqueeze(1).expand(-1, n_know, -1, -1).reshape_as(query)           # (bs, seqlen, d_model) -> (bs*n_know, seqlen, d_model)
        p = p.unsqueeze(1).expand(-1, n_know, -1, -1).reshape_as(query)             # 同上
        z, k_scores = self.block4(query, hq, p,
                                  torch.repeat_interleave(lens, n_know),            # 用于在指定维度上按给定次数重复张量中的元素（对应于bs -> bs*n_know），指示扩展后的样本有效长度
                                  peek_cur=False                                    # MHA层的参数
                                  )                                                 # (bs*n_know, seqlen, d_model)

        # （4）reshape and return
        z = (
            z.view(bs, n_know, seqlen, d_model)
            .transpose(1, 2)                                                        # (bs, seqlen, n_know, d_model)
            .contiguous()
            .view(bs, seqlen, -1)
        )                                                                           # (bs, seqlen, n_know * d_model)
        k_scores = (
            k_scores.view(bs, n_know, self.n_heads, seqlen, seqlen)                 # unpack dimensions
            .permute(0, 2, 3, 1, 4)                                                 # (bs, n_heads, seqlen, n_know, seqlen)
            .contiguous()
        )
        return z, q_scores, k_scores

    def readout_attention_y(self, z, query):
        """
        作答 正确率 预测端的readout计算（调用block6）
        """
        bs, seqlen, _ = query.size()
        q = query.reshape(bs * seqlen, 1, -1)            # (bs * seq_len, 1, d_model)
        key = (
            self.know_params[None, None, :, :]
            .expand(bs, seqlen, -1, -1)
            .view(bs * seqlen, self.n_know, -1)
        )                                                # (bs * seqlen, n_know, d_model)
        value = z.reshape(bs * seqlen, self.n_know, -1)  # (bs * seqlen, n_know, d_model)
        h, _ = self.block6(q, key, value)                # (bs * seqlen, 1, d_model)
        return h.view(bs, seqlen, -1)                    # (bs * seqlen, 1, d_model) -> (bs, seqlen, d_model)

    def readout_attention_t(self, z, query):
        """
        作答 时间 预测端的readout计算（调用block7）
        """
        bs, seqlen, _ = query.size()
        q = query.reshape(bs * seqlen, 1, -1)            # (bs * seq_len, 1, d_model)
        key = (
            self.know_params[None, None, :, :]
            .expand(bs, seqlen, -1, -1)
            .view(bs * seqlen, self.n_know, -1)
        )                                                # (bs * seqlen, n_know, d_model)
        value = z.reshape(bs * seqlen, self.n_know, -1)  # (bs * seqlen, n_know, d_model)
        h, _ = self.block7(q, key, value)                # (bs * seqlen, 1, d_model)
        return h.view(bs, seqlen, -1)                    # (bs * seqlen, 1, d_model) -> (bs, seqlen, d_model)

    def predict(self, q, s, pid, t, n=1):
        """
        向前N步，预测作答正确率和用时
        :param q:   (bs, seq_len, ks_len)
        :param s:   (bs, seq_len)
        :param pid: (bs, seq_len)
        :param t:   (bs, seq_len)
        :param n:   向前预报步数，直接取1即可
        """
        # 利用Rasch embedding获取编码后的问题和问题-答案对
        q_emb, s_emb, lens, p_diff = self.embedding(q, s, pid, t)         # (bs, seqlen, d_model), (bs, seqlen, d_model), (bs,), (bs, seqlen, 1)

        # 调用forward，模型推理获取知识层级的掌握情况 z
        z, _, _ = self(q_emb, s_emb, lens)                                # (bs, seqlen, n_know * d_model)

        # predict T+N
        query = q_emb[:, n - 1:, :]                                       # (bs, seqlen-n+1, d_model)

        # 预测作答正确率的logits
        h_y = self.readout_attention_y(z[:, : query.size(1), :], query)   # (bs, seqlen-n+1, d_model)
        h_t = self.readout_attention_t(z[:, : query.size(1), :], query)   # (bs, seqlen-n+1, d_model)

        # 正确率预测输出（从 输出logits 改为 输出概率）
        y = self.out(torch.cat([query, h_y], dim=-1)).squeeze(-1)  # (bs, seqlen-n+1, d_model*2) -> (bs, seqlen-n+1, 1) -> (bs, seqlen-n+1)
        y = self.param_sigmoid_y(y)                                        # 过含参的sigmoid函数映射至[0,1]内
        y = y * self.n_y                                                   # 还原为百分数（便于与输入计算loss）

        # 作答时间t预测
        t_pred = self.out_t(torch.cat([query, h_t], dim=-1)).squeeze(-1)
        t_pred = self.param_sigmoid_t(t_pred)
        t_pred = t_pred * self.n_t                                         # 同上，(bs, seqlen-n+1)

        """
        最终返回：
            (bs,seq_len2), (bs,seq_len,n_know*d_model), (bs,seq_len,d_model), (bs,seq_len,1), (bs,seq_len2)
            其中y是正确率百分数预测，z是稠密知识表征，q_emb是知识点嵌入，(p_diff**2).mean() * 1e-3为损失正则项，t_pred为作答时间预测
        """
        reg_l2 = (p_diff ** 2).mean() * 1e-3
        return y, z, q_emb, reg_l2, t_pred

    def get_loss(self, q, s, pid, t):
        """
        loss计算（直接调用predict方法）
        """
        logits, _, _, reg_loss, t_pred = self.predict(q, s, pid, t)  # 返回shape见上面
        # 挑选出有作答的记录，-1代表未作答
        masked_labels = s[s >= 0].float()                            # 转为float便于计算loss
        masked_logits = logits[s >= 0]
        masked_t_true = t[s >= 0].float()                            # 同上
        masked_t_pred = t_pred[s >= 0]
        # combined loss
        loss_y = F.mse_loss(masked_logits, masked_labels, reduction="mean") / (self.n_y ** 2)
        loss_t = F.mse_loss(masked_t_pred, masked_t_true, reduction="mean") / (self.n_t ** 2)
        loss = (
                loss_y * self.p_ratio                                # 正确率预测的MSEloss（权重系数）
                + reg_loss                                           # 正则化损失
                + loss_t * (1 - self.p_ratio)                        # 作答时间预测的MSE损失
        )
        return loss, loss_y, loss_t


class CDMTransformerLayers(nn.Module):
    def __init__(self, d_model, n_heads, n_layers,dropout, kq_same=True):
        super(CDMTransformerLayers, self).__init__()
        self.positional_encoding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList(
            [CDMTransformerLayer(d_model, n_heads, dropout, kq_same) for _ in range(n_layers)]
        )
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, q, k, v, lens, peek_cur=False):
        # q = self.positional_encoding(q)
        for layer in self.layers:
            q, q_scores = layer(q, q, q, lens, peek_cur)
        return self.layer_norm(q), q_scores


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class CDMTransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout, kq_same=True):
        super().__init__()
        self.kq_same = kq_same
        self.masked_attn_head = MultiHeadAttention(d_model, n_heads, kq_same)
        self.dropout_rate = dropout
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        # self.ffn = FeedForward(d_model, dropout)
        # self.norm1 = nn.LayerNorm(d_model)
        # self.norm2 = nn.LayerNorm(d_model)

    def device(self):
        return next(self.parameters()).device

    def forward(self, q, k, v, lens, peek_cur=False):
        # construct mask
        seqlen = q.size(1)
        # 当 peek_cur 为 True 时，允许当前位置关注当前及之前的位置；当 peek_cur 为 False 时，只允许当前位置关注之前的位置
        mask = torch.ones(seqlen, seqlen).tril(0 if peek_cur else -1)
        mask = mask.bool()[None, None, :, :].to(self.device())

        if self.training:
            mask = mask.expand(q.size(0), -1, -1, -1).contiguous()
            # 随机mask
            for b in range(q.size(0)):
                # sample for each batch
                if lens[b] < MIN_SEQ_LEN:
                    # skip for short sequences
                    continue
                idx = random.sample(
                    range(lens[b] - 1), max(1, int(lens[b] * self.dropout_rate))
                )
                for i in idx:
                    mask[b, :, i + 1:, i] = 0
        query_, scores = self.masked_attn_head(q, k, v, mask, maxout=not peek_cur)

        query = q + self.dropout(query_)
        return self.layer_norm(query), scores


class SelfAttentionLayers(nn.Module):
    """
    堆叠指定数量的 SelfAttentionLayer（用于验证增大模型规模的效果）
    """
    def __init__(self, d_model, n_heads, n_layers,dropout, kq_same=True):
        super().__init__()
        self.positional_encoding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList(
            [SelfAttentionLayer(d_model, n_heads, dropout, kq_same) for _ in range(n_layers)]
        )
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, q, k, v, lens, peek_cur=False):
        # q = self.positional_encoding(q)
        for layer in self.layers:
            q,q_scores = layer(q, q, q, lens, peek_cur)
        return self.layer_norm(q), q_scores


class SelfAttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout, kq_same=True):
        super().__init__()
        self.kq_same = kq_same
        self.self_attn_head = MultiHeadAttention(d_model, n_heads, kq_same, gammas=False)
        self.dropout_rate = dropout
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def device(self):
        return next(self.parameters()).device

    def forward(self, q, k, v):
        query_, scores = self.self_attn_head(q, k, v, mask=None, maxout=False)
        query = q + self.dropout(query_)
        return self.layer_norm(query), scores


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, kq_same=True, bias=True, gammas=True):
        super().__init__()
        self.d_model = d_model
        self.d_head = d_model // n_heads
        self.h = n_heads

        self.q_linear = nn.Linear(d_model, d_model, bias=bias)
        if kq_same:
            self.k_linear = self.q_linear
        else:
            self.k_linear = nn.Linear(d_model, d_model, bias=bias)
        self.v_linear = nn.Linear(d_model, d_model, bias=bias)
        self.out_linear = nn.Linear(d_model, d_model, bias=bias)
        if gammas:
            self.gammas = nn.Parameter(torch.zeros(n_heads, 1, 1))
            torch.nn.init.xavier_uniform_(self.gammas)
        else:
            self.gammas = None

    def forward(self, q, k, v, mask, maxout=False):
        bs = q.size(0)

        q = self.q_linear(q).view(bs, -1, self.h, self.d_head).transpose(1, 2)
        k = self.k_linear(k).view(bs, -1, self.h, self.d_head).transpose(1, 2)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_head).transpose(1, 2)

        output, scores = self.Attention(q, k, v, mask, gamma=self.gammas, maxout=maxout)
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out_linear(output)
        return output, scores

    def Attention(self, q, k, v, mask=None, gamma=None, maxout=False):
        d_k = k.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        bs, head, seqlen, _ = scores.size()
        # e^(-gamma * d(▲t))
        if gamma is not None:
            x1 = torch.arange(seqlen).float().expand(seqlen, -1).to(gamma.device)
            x2 = x1.transpose(0, 1).contiguous()

            with torch.no_grad():
                if mask is not None:
                    scores_ = scores.masked_fill(mask == 0, -1e32)
                else:
                    scores_ = scores
                scores_ = F.softmax(scores_, dim=-1)
                # 累计和
                distcum_scores = torch.cumsum(scores_, dim=-1)
                # 总和
                disttotal_scores = torch.sum(scores_, dim=-1, keepdim=True)

                position_effect = torch.abs(x1 - x2)[None, None, :, :]
                # mask的数据dist_scores为 0，距离越远dist_scores越高
                dist_scores = torch.clamp(
                    (disttotal_scores - distcum_scores) * position_effect, min=0.0
                )
                dist_scores = dist_scores.sqrt().detach()
            gamma = -1.0 * gamma.abs().unsqueeze(0)
            # 距离越远total_effect越小
            total_effect = torch.clamp((dist_scores * gamma).exp(), min=1e-5, max=1e5)
            scores *= total_effect
        if mask is not None:
            # scores.masked_fill(mask == 0, -1e32)
            scores = scores.masked_fill(mask == 0, -1e32)
        scores = F.softmax(scores, dim=-1)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, 0)
        # max-out scores (bs, n_heads, seqlen, seqlen)
        if maxout:
            scale = torch.clamp(1.0 / scores.max(dim=-1, keepdim=True)[0], max=5.0)
            scores *= scale
        output = torch.matmul(scores, v)
        return output, scores
