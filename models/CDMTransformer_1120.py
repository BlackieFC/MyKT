import torch
from torch import nn
import math
import random
import torch.nn.functional as F
from torch.nn import init

MIN_SEQ_LEN = 5


# ==========================================================================================
# 关于t的引入，索性不正则化了，直接搞一个类似DDPM里T的嵌入层
# ==========================================================================================


class Swish(nn.Module):
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
    # 使用示例
    logits = torch.tensor([0.5, 1.0, 1.5, 2.0], requires_grad=True)
    parametric_sigmoid = ParametricSigmoid()
    # 前向传播
    probabilities = parametric_sigmoid(logits)
    """
    def __init__(self):
        super(ParametricSigmoid, self).__init__()
        # 初始化缩放参数a和偏移参数b
        self.a = nn.Parameter(torch.tensor(1.0))
        self.b = nn.Parameter(torch.tensor(0.0))
        
    def forward(self, x):
        # 计算含参的 Sigmoid 函数
        return torch.sigmoid(self.a * x + self.b)


class CDMTransformer_1125(nn.Module):
    """
    1120相较于1119:
        输入s从0-1变为0-100的百分整数，对应地也换用TimeEmbedding，
        输出仍然为正确率的logit，且loss换用回归的（但要注意还原为百分数后计算）
    1125相较于1120:
        简化代码，
        验证 embedding 优化的效果.
    """
    def __init__(self,
                 n_concepts,       # kc总数
                 n_pid,            # 题库大小
                 d_model=128,      # 特征维度
                 d_ff=256,         # 隐藏层维度
                 n_attn_heads=8,   # mha头数
                 n_know=16,        # know_param大小
                 n_layers=3,       # attention层堆叠数量
                 dropout=0.3,      # dropout
                 window=1,         # 向前预报时的窗口大小
                 n_t=500,          # 做题时间上限
                 p_ratio=1,        # loss中performance项的倍率
                 ):
        super().__init__()
        # 需要进行嵌入的特征总数
        self.n_concepts = n_concepts
        self.n_t = n_t
        self.p_ratio = p_ratio

        # kc编码
        self.q_embed = nn.Embedding(n_concepts + 1, d_model)

        # response编码  self.s_embed = nn.Embedding(2, d_model)
        self.s_embed = TimeEmbedding(101, d_model, d_model//2)

        # duration编码  self.t_embed = nn.Embedding(self.n_t+1, d_model)
        self.t_embed = TimeEmbedding(self.n_t+1, d_model, d_model//2)

        """Rasch Embedding 区分度嵌入"""
        # d_ct summarizes the variations of questions within this concept,
        self.q_diff_embed = nn.Embedding(n_concepts + 1, d_model)

        # response区分度嵌入  self.s_diff_embed = nn.Embedding(2, d_model)
        self.s_diff_embed = TimeEmbedding(101, d_model, d_model//2)

        # duration区分度嵌入  self.t_diff_embed = nn.Embedding(self.n_t+1, d_model)
        self.t_diff_embed = TimeEmbedding(self.n_t+1, d_model, d_model//2)

        # u_q 当前problem难度/题目区分度嵌入
        self.p_diff_embed = nn.Embedding(n_pid + 1, 1)

        # MHA layers
        self.n_heads = n_attn_heads  # 8
        # self.block1 = CDMTransformerLayer(d_model, n_attn_heads, dropout=dropout)              # q,q,q
        self.block1 = CDMTransformerLayers(d_model, n_attn_heads,n_layers=4, dropout=dropout)  # 堆叠层数
        # self.block2 = CDMTransformerLayer(d_model, n_attn_heads, dropout=dropout)              # s,s,s
        self.block2 = CDMTransformerLayers(d_model, n_attn_heads,n_layers=4, dropout=dropout)  # 堆叠层数
        self.block3 = CDMTransformerLayer(d_model, n_attn_heads, dropout)                      # hq,hq,hs，得到问题层面掌握度
        self.block4 = CDMTransformerLayer(d_model, n_attn_heads, dropout, kq_same=False)       # know_param, hq, 问题层面掌握度
        self.block5 = SelfAttentionLayer(d_model, n_attn_heads, dropout)                       # c_emb的自注意力
        self.block6 = SelfAttentionLayer(d_model, n_attn_heads, dropout, kq_same=False)        # readout —— 正确率
        self.block7 = SelfAttentionLayer(d_model, n_attn_heads, dropout, kq_same=False)        # readout —— 用时

        """
        know_param 两种定义方式：
        （1）直接声明；
        （2）通过c_emb 获取。
        """
        self.n_know = n_know
        # （1）直接声明
        # self.know_params = nn.Parameter(torch.randn(n_know, d_model))
        # torch.nn.init.uniform_(self.know_params, -1.0, 1.0)
        
        # （2）后续通过c_emb 获取
        self.know_params = None
        self.c_k_linear = nn.Sequential(
            nn.Linear(d_model, n_know * d_model*2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(n_know * d_model * 2, n_know * d_model),
        )  # 某种意义上强化了know_param作为稠密知识向量与kc嵌入之间的联系，但是否有效有待考察

        # 输出层(y_pred)
        self.out = nn.Sequential(
            nn.Linear(d_model * 2, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_ff // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff // 2, 1),
        )

        # 输出层(t_pred)
        self.out_t = nn.Sequential(
            nn.Linear(d_model * 2, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_ff // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff // 2, 1),
        )

        # 含参的sigmoid(通过训练让sigmoid具备更好的性质)
        self.param_sigmoid_y = ParametricSigmoid()
        self.param_sigmoid_t = ParametricSigmoid()

        # 其他参数
        self.dropout_rate = dropout
        self.n_layers = n_layers     # 默认为3
        self.window = window

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
        if q.ndim == 2:               # 检查张量维度
            q = q.unsqueeze(-1)       # bs, seqlen -> bs, seqlen, ks_len=1
        
        bs, seqlen, kp_len = q.shape  # 各维度大小
        lens = (s >= 0).sum(dim=1)    # (bs, seqlen) -> (bs,)，batch中每个学生的实际序列长度

        q = q.masked_fill(q < 0, 0)   # 将padding值置为0
        s = s.masked_fill(s < 0, 0)
        t = t.masked_fill(t < 0, 0)   # 241125:对t进行相同处理 —— 将所有padding值（-1）置为0 —— 要求0在duration的真实数据中不出现

        # c_ct is the embedding of the concept this question covers
        q_emb = self.q_embed(q)                     # (bs,seqlen,kp_len) -> (bs,seqlen,kp_len,d_model)
        q_emb = q_emb.mean(dim=-2, keepdim=False)   # multi-KCs，对KC维度求平均，返回 (bs, seqlen, d_model)

        # e_(ct,rt) = g_rt + c_ct  e_(ct,rt): concept-response embedding
        t_emb = self.t_embed(t)                     # (bs, seq_len) -> (bs, seqlen, d_model//2)
        s_emb = self.s_embed(s)                     # 同上
        """241125: response改为由correct rate和duration的嵌入concat而来（在此基础上再加上q_emb，以kc嵌入为中心）"""
        s_emb = torch.cat((s_emb, t_emb), dim=-1) + q_emb  # (bs, seqlen, d_model)

        # 难度嵌入为scalar
        pid = pid.masked_fill(pid < 0, 0)           # (bs, seqlen)
        p_diff = self.p_diff_embed(pid)             # u_q 当前problem难度，(bs, seqlen, 1)

        # d_ct 总结了包含当前question（concept）的problems（questions）的变化
        q_diff_emb = self.q_diff_embed(q)           # (bs,seqlen,kp_len) -> (bs,seqlen,kp_len,d_model)
        q_diff_emb = q_diff_emb.mean(dim=-2, keepdim=False)  # 同上，(bs, seqlen, d_model)

        # question embedding q_t = c_ct + uq *d_ct
        q_emb += q_diff_emb * p_diff                # (bs, seqlen, d_model) * (bs, seqlen, 1)
        
        # f = s_diff_embed + d_ct f summarizes the variations of learning activities within this concept
        t_diff_emb = self.t_diff_embed(t)           # (bs, seq_len) -> (bs, seqlen, d_model//2)
        s_diff_emb = self.s_diff_embed(s)           # 同上
        """241125:同上进行修改"""
        s_diff_emb = torch.cat((s_diff_emb, t_diff_emb), dim=-1) + q_diff_emb  # (bs, seqlen, d_model)

        # s_t = e_(ct,rt) + u_q * f
        s_emb += s_diff_emb * p_diff

        return q_emb, s_emb, lens, p_diff

    def forward(self, q_emb, s_emb, lens):
        """
        forward 一般在其他方法，如 predict 中进行调用，会先于其调用embedding，然后将嵌入好的内容作为q,s,pid传入
        """
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
            p, q_scores = self.block3(hq, hq, hs, lens, peek_cur=True)     # (bs, seqlen, d_model)  (bs, n_heads, seqlen, seqlen)

        """
        know_param 两种定义方式：
        （2）计算c_emb，并通过其 self attention 计算得到 know_params
        """
        bs, seqlen, d_model = p.size()
        n_know = self.n_know
        c_emb = self.q_embed.weight[1:].unsqueeze(0)        # (1, n_questions, d_model) 排除第一个未使用的嵌入（id=0）
        c, _ = self.block5(c_emb, c_emb, c_emb)             # kc嵌入的自注意力，(1, n_questions, d_model)
        self.know_params = self.c_k_linear(c.mean(dim=1)).reshape(n_know, d_model)  # Shape: (1, n_know * d_model)，再扩展到 (n_know, d_model) 的形状 n_konw=1

        # 获取稠密能力表征z
        query = (
            self.know_params[None, :, None, :]              # (1, n_know, 1, d_model)
            .expand(bs, -1, seqlen, -1)                     # (bs, n_know, seqlen, d_model)
            .contiguous()
            .view(bs * n_know, seqlen, d_model)             # (bs * n_know, seqlen, d_model)
        )
        hq = hq.unsqueeze(1).expand(-1, n_know, -1, -1).reshape_as(query)  # (bs, seqlen, d_model) -> (bs*n_know, seqlen, d_model)
        p = p.unsqueeze(1).expand(-1, n_know, -1, -1).reshape_as(query)    # 同上
        z, k_scores = self.block4(query, hq, p, torch.repeat_interleave(lens, n_know), peek_cur=False)  # (bs*n_know, seqlen, d_model)

        # reshape and return
        z = (
            z.view(bs, n_know, seqlen, d_model)
            .transpose(1, 2)  # (bs, seqlen, n_know, d_model)
            .contiguous()
            .view(bs, seqlen, -1)
        )                     # (bs, seqlen, n_know * d_model)
        k_scores = (
            k_scores.view(bs, n_know, self.n_heads, seqlen, seqlen)  # unpack dimensions
            .permute(0, 2, 3, 1, 4)  # (bs, n_heads, seqlen, n_know, seqlen)
            .contiguous()
        )
        return z, q_scores, k_scores

    def readout(self, z, query):
        """
        仍在CDM的作答结果预报中使用，简化的attention计算
        softmax(know * query) * z
        """
        bs, seqlen, _ = query.size()
        key = (
            self.know_params[None, None, :, :]
            .expand(bs, seqlen, -1, -1)
            .view(bs * seqlen, self.n_know, -1)
        )                                                       # (bs * seqlen, n_know, d_model)
        value = z.reshape(bs * seqlen, self.n_know, -1)         # (bs * seqlen, n_know, d_model)
        beta = torch.matmul(
            key,
            query.reshape(bs * seqlen, -1, 1)                   # (bs, seqlen, d_model)->(bs * seqlen, d_model, 1)
        ).view(bs*seqlen, 1, self.n_know)                       # (bs * seqlen, n_know, 1) -> (bs * seqlen, 1, n_know)
        alpha = torch.softmax(beta, dim=-1)                     # (bs * seqlen, 1, n_know)
        return torch.matmul(alpha, value).view(bs, seqlen, -1)  # (bs * seqlen, 1, d_model) -> (bs, seqlen, d_model)
    
    def readout_attention(self, z, query):
        """
        作答 正确率 预测端的readout计算（调用block6）
        """
        bs, seqlen, _ = query.size()
        q = query.reshape(bs * seqlen, 1, -1)                   # (bs * seq_len, 1, d_model)
        key = (
            self.know_params[None, None, :, :]
            .expand(bs, seqlen, -1, -1)
            .view(bs * seqlen, self.n_know, -1)
        )                                                       # (bs * seqlen, n_know, d_model)
        value = z.reshape(bs * seqlen, self.n_know, -1)         # (bs * seqlen, n_know, d_model)
        # 将矩阵乘法替换为attention操作
        h,_ = self.block6(q, key, value)                        # (bs * seqlen, 1, d_model)
        return h.view(bs, seqlen, -1)                           # (bs * seqlen, 1, d_model) -> (bs, seqlen, d_model)
    
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
        # 将矩阵乘法替换为attention操作
        h,_ = self.block7(q, key, value)                 # (bs * seqlen, 1, d_model)
        return h.view(bs, seqlen, -1)                    # (bs * seqlen, 1, d_model) -> (bs, seqlen, d_model)

    def predict(self, q, s, pid, t, n=1):
        """
        向前N步，预测作答正确率和用时
        """
        # 利用Rasch embedding获取编码后的问题和问题-答案对
        q_emb, s_emb, lens, p_diff = self.embedding(q, s, pid, t)
        # 调用forward，模型推理获取知识层级的掌握情况 z (bs, seqlen, n_know * d_model)
        z, _, _ = self(q_emb, s_emb, lens)

        # predict T+N
        query = q_emb[:, n-1:, :]                                           # (bs, seqlen2, d_model)
        # h = self.readout(z[:, : query.size(1), :], query)                 # 预测作答正确率的logits（readout 或 readout attention）
        h = self.readout_attention(z[:, : query.size(1), :], query)         # (bs, seqlen2, d_model)
        h_t = self.readout_attention_t(z[:, : query.size(1), :], query)     # 预测作答用时（readout attention）

        # 正确率预测输出（从输出logits改为输出概率）
        y = self.out(torch.cat([query, h], dim=-1)).squeeze(-1)             # concat (bs, seqlen2, d_model*2), then (bs, seqlen2, 1) -> (bs, seqlen2)
        y = self.param_sigmoid_y(y)                                         # 过含参的sigmoid函数映射至[0,1]内
        y = y * 100                                                         # 还原为百分数（便于与输入计算loss）
        
        # 作答时间t预测（注意还原倍率）
        t_pred = self.out_t(torch.cat([query, h_t], dim=-1)).squeeze(-1)    # 同上，(bs, seqlen2)
        t_pred = self.param_sigmoid_t(t_pred)                               # 过含参的sigmoid函数映射至[0,1]内
        t_pred = t_pred * self.n_t                                          # 还原倍率（self.out_t的输出应该是0-1的，避免权重过大）

        """
        最终返回：
            (bs,seq_len2), (bs,seq_len,n_know*d_model), (bs,seq_len,d_model), (bs,seq_len,1), (bs,seq_len2)
            其中y是正确率百分数，z是稠密知识表征，q_emb是知识点嵌入，(p_diff**2).mean() * 1e-3为损失正则项，t_pred为作答时间预测
        """
        return y, z, q_emb, (p_diff**2).mean() * 1e-3, t_pred

    def get_loss(self, q, s, pid, t):
        """
        loss计算（直接调用predict方法）
        """
        logits, _, _, reg_loss, t_pred = self.predict(q, s, pid, t)
        # 挑选出有作答的记录，-1代表未作答
        masked_labels = s[s >= 0].float()  # 转为float便于计算loss
        masked_logits = logits[s >= 0]
        masked_t_true = t[s >= 0].float()  # 同上
        masked_t_pred = t_pred[s >= 0]

        loss = (
            F.mse_loss(masked_logits, masked_labels, reduction="mean") / (100**2) * self.p_ratio  # 正确率预测的MSEloss
            + reg_loss                                                                            # 正则化损失
            + F.mse_loss(masked_t_pred, masked_t_true, reduction="mean") / (self.n_t**2)          # 作答时间预测的MSE损失
        )

        return loss


class CDMTransformer_1120(nn.Module):
    """
    相较于1119:
        输入s从0-1变为0-100的百分整数，对应地也换用TimeEmbedding，
        输出仍然为正确率的logit，且loss换用回归的（但要注意还原为百分数后计算）
    """
    def __init__(self,
                 n_concepts,
                 n_pid,
                 path_diff=None,  # 新增难度数据路径
                 d_model=128,
                 d_ff=256,
                 n_attn_heads=8,
                 n_know=16,
                 n_topic=None,
                 n_tense=None,
                 n_layers=3,
                 dropout=0.3,
                 lambda_cl=0.1,
                 proj=False,
                 hard_neg=False,
                 window=1,
                 qemb_matrix=None,  # 龙哥的知识点映射向量
                 n_t=1000,          # 最大做题时间
                 p_ratio=1,         # loss中正确率的倍率
                 ):
        """
        :param n_concepts: 知识点数量
        :param n_pid: 问题数量
        :param d_model: 嵌入维度
        :param d_ff: 隐藏层维度
        :param n_attn_heads: 注意力头数
        :param n_know: latent知识点数量
        :param n_topic: 主题数量
        :param n_tense: 时态数量
        :param n_layers: 层数
        :param dropout: dropout
        :param lambda_cl: 对比损失权重
        :param proj: 是否使用投影层
        :param hard_neg: 是否使用hard negative
        :param window: 滑动窗口大小
        :param qemb_matrix: 龙哥的知识点映射向量
        """
        super().__init__()
        # 需要进行嵌入的特征总数
        self.n_concepts = n_concepts
        self.n_topic = n_topic
        self.n_tense = n_tense
        self.n_t = n_t
        self.p_ratio = p_ratio

        # 对知识点进行编码
        self.q_embed = nn.Embedding(n_concepts + 1, d_model)

        # 对回答进行编码（1120换用TimeEmbedding）
        # self.s_embed = nn.Embedding(2, d_model)
        self.s_embed = TimeEmbedding(101, d_model, d_model)

        # 对作答时间进行编码
        # self.t_embed = nn.Embedding(self.n_t+1, d_model)
        self.t_embed = TimeEmbedding(self.n_t+1, d_model, d_model)

        # d_ct summarizes the variations of questions within this concept,
        self.q_diff_embed = nn.Embedding(n_concepts + 1, d_model)

        # Rasch Embedding 相关参数（1120换用TimeEmbedding）
        # self.s_diff_embed = nn.Embedding(2, d_model)
        self.s_diff_embed = TimeEmbedding(101, d_model, d_model)

        # 类似地，作答时间相关嵌入
        # self.t_diff_embed = nn.Embedding(self.n_t+1, d_model)
        self.t_diff_embed = TimeEmbedding(self.n_t+1, d_model, d_model)

        # u_q 当前problem难度
        self.p_diff_embed = nn.Embedding(n_pid + 1, 1)

        # topic 话题嵌入
        if self.n_topic is not None:
            self.topic_embed = nn.Embedding(self.n_topic + 1, d_model)

        # tense 时态嵌入
        if self.n_tense is not None:
            self.tense_embed = nn.Embedding(self.n_tense + 1, d_model)

        # MHA layers
        self.n_heads = n_attn_heads
        self.block1 = CDMTransformerLayers(d_model, n_attn_heads,n_layers=4, dropout=dropout)
        self.block2 = CDMTransformerLayer(d_model, n_attn_heads, dropout=dropout)
        self.block3 = CDMTransformerLayer(d_model, n_attn_heads, dropout)
        self.block4 = CDMTransformerLayer(d_model, n_attn_heads, dropout, kq_same=False)
        self.block5 = SelfAttentionLayer(d_model, n_attn_heads, dropout)
        self.block6 = SelfAttentionLayer(d_model, n_attn_heads, dropout, kq_same=False)

        """know_param 的生成改为新的形式（从c_emb中获取）"""
        self.n_know = n_know
        
        # self.know_params = nn.Parameter(torch.randn(n_know, d_model))
        # torch.nn.init.uniform_(self.know_params, -1.0, 1.0)
        
        self.know_params = None
        self.c_k_linear = nn.Sequential(
            nn.Linear(d_model, n_know * d_model*2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(n_know * d_model * 2, n_know * d_model),
        )

        # 输出层(y_pred)
        self.out = nn.Sequential(
            nn.Linear(d_model * 2, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_ff // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff // 2, 1),
        )

        # 输出层(t_pred)
        self.out_t = nn.Sequential(
            nn.Linear(d_model * 2, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_ff // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff // 2, 1),
        )

        # 含参的sigmoid(通过训练让sigmoid具备更好的性质)
        self.param_sigmoid_y = ParametricSigmoid()
        self.param_sigmoid_t = ParametricSigmoid()

        # 重映射层(用于cl loss的sim相似度计算中)
        if proj:
            self.proj = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU())
        else:
            self.proj = None

        # 其他参数
        self.dropout_rate = dropout
        self.lambda_cl = lambda_cl      # cl_loss的权重
        self.hard_neg = hard_neg        # 作为判断是否翻转分数
        self.n_layers = n_layers
        self.window = window

        # 声明知识点映射向量（freezed，默认全0）
        if qemb_matrix is None:
            qemb_matrix = torch.zeros_like(self.q_embed.weight)  # 无指定输入时全0
        else:
            qemb_matrix = qemb_matrix.to(self.device())      # 龙哥的知识点映射向量
        self.qemb_matrix = nn.Parameter(qemb_matrix, requires_grad=False)

    def device(self):
        return next(self.parameters()).device

    def embedding(self, q, s, pid, t, topic=None, tense=None):
        """
        基于Rasch 模型计算question embeddings and question-response embeddings
        :param q:   (bs, seq_len, ks_len)
        :param s:   (bs, seq_len)
        :param pid: (bs, seq_len)
        :param t: (bs, seq_len)
        :param topic:
        :param tense:
        :return:
        """
        if q.ndim == 2:
            q = q.unsqueeze(-1)      # bs, seqlen -> bs, seqlen, 1
        bs, seqlen, kp_len = q.shape
        lens = (s >= 0).sum(dim=1)   # (bs, seq_len) -> (bs,)，batch中每个学生的实际序列长度
        q = q.masked_fill(q < 0, 0)
        s = s.masked_fill(s < 0, 0)
        t = t.view(bs*seqlen, -1).squeeze()  # flatten（输入前已clamp，取值范围符合要求）
        if topic is not None:
            topic = topic.masked_fill(topic < 0, 0)
        if tense is not None:
            tense = tense.masked_fill(tense < 0, 0)

        # c_ct is the embedding of the concept this question covers
        q = q.permute(0, 2, 1).reshape(-1, seqlen)  # (bs*, seq_len), 嵌入层标准输入格式
        q_emb = self.q_embed(q)                     # 输出为(bs*, seq_len, d_model)
        """将知识点映射向量加至question embeddings（无输入时freezed全零，不变化）"""
        q_emb = q_emb + self.qemb_matrix[q]         # (bs*, seq_len, d_model)
        q_emb = q_emb.view(bs, kp_len, seqlen, -1)  # 还原shape
        q_emb = q_emb.mean(dim=1, keepdim=False)    # 如果是一道题对应多个知识点，这里直接进行求平均操作

        # e_(ct,rt) = g_rt + c_ct  e_(ct,rt): concept-response embedding
        t_emb = self.t_embed(t)
        t_emb = t_emb.view(bs, seqlen, -1)
        s_emb = self.s_embed(s) + t_emb + q_emb

        # p_diff = 0.0
        pid = pid.masked_fill(pid < 0, 0)
        p_diff = self.p_diff_embed(pid)  # u_q 当前problem难度

        # q无需再次reshape
        q_diff_emb = self.q_diff_embed(q)  # d_ct 总结了包含当前question（concept）的problems（questions）的变化
        q_diff_emb = q_diff_emb.view(bs, kp_len, seqlen, -1)
        q_diff_emb = q_diff_emb.mean(dim=1, keepdim=False)

        # question embedding q_t = c_ct + uq *d_ct
        q_emb += q_diff_emb * p_diff
        if self.n_topic is not None:
            q_emb += self.topic_embed(topic)
        if self.n_tense is not None:
            q_emb += self.tense_embed(tense)
        
        # f = s_diff_embed + d_ct f summarizes the variations of learning activities within this concept
        t_diff_emb = self.t_diff_embed(t)
        t_diff_emb = t_diff_emb.view(bs, seqlen, -1)
        s_diff_emb = self.s_diff_embed(s) + t_diff_emb + q_diff_emb
        # s_t = e_(ct,rt) + u_q * f
        s_emb += s_diff_emb * p_diff

        # 还原t
        t = t.view(bs, seqlen)
        return q_emb, s_emb, lens, p_diff  # , c_emb

    def forward(self, q_emb, s_emb, lens):  # , c_emb
        """
        forward 一般在其他方法，如 predict 中进行调用，会先于其调用embedding，然后将嵌入好的内容作为q,s,pid传入
        """
        if self.n_layers == 1:
            hq = q_emb
            p, q_scores = self.block1(q_emb, q_emb, s_emb, lens, peek_cur=True)
        elif self.n_layers == 2:
            hq = q_emb
            hs, _ = self.block2(s_emb, s_emb, s_emb, lens, peek_cur=True)
            p, q_scores = self.block3(hq, hq, hs, lens, peek_cur=True)
        else:  # self.n_layers == 3:
            hq, _ = self.block1(q_emb, q_emb, q_emb, lens, peek_cur=True)
            hs, _ = self.block2(s_emb, s_emb, s_emb, lens, peek_cur=True)
            p, q_scores = self.block3(hq, hq, hs, lens, peek_cur=True)  # (bs, len, d_model)  (bs, h, len, len)

        """计算c_emb，并通过其 self attention 计算得到 know_params"""
        bs, seqlen, d_model = p.size()
        n_know = self.n_know

        c_emb = self.q_embed.weight[1:].unsqueeze(0)        # (n_questions, dim) 排除第一个未使用的嵌入
        c_emb = c_emb + self.qemb_matrix[1:].unsqueeze(0)   # 240903: 加上知识点映射向量！！！
        c, _ = self.block5(c_emb, c_emb, c_emb)             # 自注意力机制
        self.know_params = self.c_k_linear(c.mean(dim=1)).reshape(n_know, d_model)  # Shape: (1, n_know * dim)，再扩展到 (n_know, dim) 的形状 n_konw=1

        # 获取稠密能力表征z
        query = (
            self.know_params[None, :, None, :]
            .expand(bs, -1, seqlen, -1)
            .contiguous()
            .view(bs * n_know, seqlen, d_model)
        )
        hq = hq.unsqueeze(1).expand(-1, n_know, -1, -1).reshape_as(query)
        p = p.unsqueeze(1).expand(-1, n_know, -1, -1).reshape_as(query)
        z, k_scores = self.block4(query, hq, p, torch.repeat_interleave(lens, n_know), peek_cur=False)

        # reshape and return
        z = (
            z.view(bs, n_know, seqlen, d_model)
            .transpose(1, 2)  # (bs, seqlen, n_know, d_model)
            .contiguous()
            .view(bs, seqlen, -1)
        )  # (bs, seqlen, n_know * d_model)
        k_scores = (
            k_scores.view(bs, n_know, self.n_heads, seqlen, seqlen)  # unpack dimensions
            .permute(0, 2, 3, 1, 4)  # (bs, n_heads, seqlen, n_know, seqlen)
            .contiguous()
        )
        return z, q_scores, k_scores

    def readout(self, z, query):
        """
        仍在CDM的作答结果预报中使用，知识点映射模型中已无需使用
        softmax(know * query) * z
        """
        bs, seqlen, _ = query.size()
        key = (
            self.know_params[None, None, :, :]
            .expand(bs, seqlen, -1, -1)
            .view(bs * seqlen, self.n_know, -1)
        )  # (bs * seqlen, n_know, d_model)
        value = z.reshape(bs * seqlen, self.n_know, -1)  # (bs * seqlen, n_know, d_model)
        beta = torch.matmul(
            key,
            query.reshape(bs * seqlen, -1, 1)  # (bs, seqlen, d_model)->(bs * seqlen, d_model, 1)
        ).view(bs*seqlen, 1, self.n_know)  # (bs * seqlen, n_know, 1) -> (bs * seqlen, 1, n_know)
        alpha = torch.softmax(beta, dim=-1)  # (bs * seqlen, 1, n_know)
        return torch.matmul(alpha, value).view(bs, seqlen, -1)  # (bs * seqlen, 1, d_model) -> (bs, seqlen, d_model)
    
    def readout_attention(self, z, query):
        """
        用于作答时间t的输出
        """
        bs, seqlen, _ = query.size()
        q = query.reshape(bs * seqlen, 1, -1) # (bs * seq_len, 1, d_model)
        key = (
            self.know_params[None, None, :, :]
            .expand(bs, seqlen, -1, -1)
            .view(bs * seqlen, self.n_know, -1)
        )  # (bs * seqlen, n_know, d_model)
        value = z.reshape(bs * seqlen, self.n_know, -1)  # (bs * seqlen, n_know, d_model)
        # 将矩阵乘法替换为attention操作
        h,_ = self.block6(q, key, value)                 # (bs * seqlen, 1, d_model)
        return h.view(bs, seqlen, -1)                    # (bs * seqlen, 1, d_model) -> (bs, seqlen, d_model)

    def predict(self, q, s, pid, t, topic=None, tense=None, n=1):
        """
        整体添加topic和tense参数，用于传入embedding方法
        # T+N预测是否答对
        """
        # 利用Rasch embedding获取编码后的问题和问题-答案对
        q_emb, s_emb, lens, p_diff = self.embedding(q, s, pid, t, topic, tense)
        # 模型推理获取知识层级的掌握情况 z (bs, seqlen, n_know * d_model)
        # z (bs, seqlen, n_know * d_model), q_scores (bs, n_heads, seqlen, seqlen), k_scores (bs, n_know, n_heads, seqlen, seqlen)
        z, _, _ = self(q_emb, s_emb, lens)  # 调用forward，返回 z (bs, seqlen, n_know * d_model)

        # predict T+N
        query = q_emb[:, n-1:, :]  # (bs, seqlen2, d_model)
        # h (bs, seqlen2, d_model)
        h = self.readout(z[:, : query.size(1), :], query)                 # 预测y的logits
        h_t = self.readout_attention(z[:, : query.size(1), :], query)     # 预测t

        # 正确率预测输出（从输出logits改为输出概率）
        # concat (bs, seqlen2, d_model*2), then (bs, seqlen2, 1) -> (bs, seqlen2)
        y = self.out(torch.cat([query, h], dim=-1)).squeeze(-1)
        y = self.param_sigmoid_y(y)  # 过含参的sigmoid函数
        y = y * 100                  # 还原倍率
        
        # 作答时间t预测（注意还原倍率）
        t_pred = self.out_t(torch.cat([query, h_t], dim=-1)).squeeze(-1)  # 同上，(bs, seqlen2)
        t_pred = self.param_sigmoid_t(t_pred)  # 过含参的sigmoid函数映射至[0,1]内
        t_pred = t_pred * self.n_t  # 还原倍率（self.out_t的输出应该是0-1的，避免权重过大）

        """
        最终返回：
            (bs,seq_len2), (bs,seq_len,n_know*d_model),(bs,seq_len,d_model),(bs,seq_len,1)
            其中y是正确率百分数，z是稠密知识表征，q_emb是知识点嵌入，(p_diff**2).mean() * 1e-3为损失正则项，t_pred为作答时间预测
        """
        return y, z, q_emb, (p_diff**2).mean() * 1e-3, t_pred             # (q_scores, k_scores)

    def get_loss(self, q, s, pid, t, topic=None, tense=None):
        """
        用于传入predict方法
        将正确率的输出从logits变为百分数，然后将其和输入之间的回归指标作为loss项
        """
        logits, _, _, reg_loss, t_pred = self.predict(q, s, pid, t, topic=topic, tense=tense)
        # 挑选出有作答的记录，-1代表未作答
        masked_labels = s[s >= 0].float()
        masked_logits = logits[s >= 0]
        masked_t_true = t[s >= 0].float()
        masked_t_pred = t_pred[s >= 0]

        """
        # 旧版
        loss = (
            F.binary_cross_entropy_with_logits(      # BCE损失
                masked_logits, masked_labels, reduction="mean"
            )
            + reg_loss                               # 正则化损失
            + F.l1_loss(
                t_pred[s >= 0], t[s >= 0], reduction="mean"
                ) / self.n_t                         # mae损失(除以倍率)
        )"""

        loss = (
            F.mse_loss(masked_logits, masked_labels, reduction="mean") / (100**2) * self.p_ratio  # 正确率预测的MSEloss
            + reg_loss                                                                            # 正则化损失
            + F.mse_loss(masked_t_pred, masked_t_true, reduction="mean") / (self.n_t**2)          # 作答时间预测的MSE损失
        )

        return loss


class CDMTransformer(nn.Module):
    def __init__(self,
                 n_concepts,
                 n_pid,
                 path_diff=None,  # 新增难度数据路径
                 d_model=128,
                 d_ff=256,
                 n_attn_heads=8,
                 n_know=16,
                 n_topic=None,
                 n_tense=None,
                 n_layers=3,
                 dropout=0.3,
                 lambda_cl=0.1,
                 proj=False,
                 hard_neg=False,
                 window=1,
                 qemb_matrix=None,  # 龙哥的知识点映射向量
                 ):
        """
        :param n_concepts: 知识点数量
        :param n_pid: 问题数量
        :param d_model: 嵌入维度
        :param d_ff: 隐藏层维度
        :param n_attn_heads: 注意力头数
        :param n_know: latent知识点数量
        :param n_topic: 主题数量
        :param n_tense: 时态数量
        :param n_layers: 层数
        :param dropout: dropout
        :param lambda_cl: 对比损失权重
        :param proj: 是否使用投影层
        :param hard_neg: 是否使用hard negative
        :param window: 滑动窗口大小
        :param qemb_matrix: 龙哥的知识点映射向量
        """
        super().__init__()
        # 需要进行嵌入的特征总数
        self.n_concepts = n_concepts
        self.n_topic = n_topic
        self.n_tense = n_tense

        # 对知识点进行编码
        self.q_embed = nn.Embedding(n_concepts + 1, d_model)

        # 对回答进行编码
        self.s_embed = nn.Embedding(2, d_model)

        # d_ct summarizes the variations of questions within this concept,
        self.q_diff_embed = nn.Embedding(n_concepts + 1, d_model)

        # Rasch Embedding 相关参数
        self.s_diff_embed = nn.Embedding(2, d_model)

        # u_q 当前problem难度
        self.p_diff_embed = nn.Embedding(n_pid + 1, 1)

        # topic 话题嵌入
        if self.n_topic is not None:
            self.topic_embed = nn.Embedding(self.n_topic + 1, d_model)

        # tense 时态嵌入
        if self.n_tense is not None:
            self.tense_embed = nn.Embedding(self.n_tense + 1, d_model)

        # MHA layers
        self.n_heads = n_attn_heads
        self.block1 = CDMTransformerLayers(d_model, n_attn_heads,n_layers=4, dropout=dropout)
        self.block2 = CDMTransformerLayer(d_model, n_attn_heads, dropout=dropout)
        self.block3 = CDMTransformerLayer(d_model, n_attn_heads, dropout)
        self.block4 = CDMTransformerLayer(d_model, n_attn_heads, dropout, kq_same=False)
        self.block5 = SelfAttentionLayer(d_model, n_attn_heads, dropout)
        self.block6 = SelfAttentionLayer(d_model, n_attn_heads, dropout, kq_same=False)

        """know_param 的生成改为新的形式（从c_emb中获取）"""
        self.n_know = n_know
        self.know_params = None
        # self.know_params = nn.Parameter(torch.randn(n_know, d_model))
        # torch.nn.init.uniform_(self.know_params, -1.0, 1.0)
        self.c_k_linear = nn.Sequential(
            nn.Linear(d_model, n_know * d_model*2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(n_know * d_model * 2, n_know * d_model),
        )

        # 输出层
        self.out = nn.Sequential(
            nn.Linear(d_model * 2, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_ff // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff // 2, 1),
        )

        # 重映射层
        if proj:
            self.proj = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU())
        else:
            self.proj = None

        # 其他参数
        self.dropout_rate = dropout
        self.lambda_cl = lambda_cl      # cl_loss的权重
        self.hard_neg = hard_neg        # 作为判断是否翻转分数
        self.n_layers = n_layers
        self.window = window
        # 声明知识点映射向量
        if qemb_matrix is None:
            qemb_matrix = torch.zeros_like(self.q_embed.weight)  # 无指定输入时全0
        else:
            qemb_matrix = qemb_matrix.to(self.device())      # 龙哥的知识点映射向量
        self.qemb_matrix = nn.Parameter(qemb_matrix, requires_grad=False)

    def device(self):
        return next(self.parameters()).device

    # def q_embedding(self, q):
    #     """deprecated"""
    #     bs, seqlen = q.size()
    #     q_emb_list = []
    #     for i in range(bs):
    #         for j in range(seqlen):
    #             q_str = str(q[i, j].item())
    #             if '_' in q_str:
    #                 parts = [int(x) for x in q_str.split('_')]
    #                 part_embeddings = self.q_embed(torch.tensor(parts).to(q.device))
    #                 mean_embedding = part_embeddings.mean(dim=0)
    #                 q_emb_list.append(mean_embedding)
    #             else:
    #                 single_embedding = self.q_embed(torch.tensor([int(q_str)]).to(q.device))
    #                 q_emb_list.append(single_embedding.squeeze(0))
    #     q_emb = torch.stack(q_emb_list).view(bs, seqlen, -1)
    #     return q_emb

    def embedding(self, q, s, pid, topic=None, tense=None):
        """
        基于Rasch 模型计算question embeddings and question-response embeddings
        :param q:   (bs, seq_len, ks_len)
        :param s:   (bs, seq_len)
        :param pid: (bs, seq_len)
        :param topic:
        :param tense:
        :return:
        """
        if q.ndim == 2:
            q = q.unsqueeze(-1)      # bs, seqlen -> bs, seqlen, 1
        bs, seqlen, kp_len = q.shape
        lens = (s >= 0).sum(dim=1)   # (bs, seq_len) -> (bs,)，batch中每个学生的实际序列长度
        q = q.masked_fill(q < 0, 0)
        s = s.masked_fill(s < 0, 0)
        if topic is not None:
            topic = topic.masked_fill(topic < 0, 0)
        if tense is not None:
            tense = tense.masked_fill(tense < 0, 0)

        # c_ct is the embedding of the concept this question covers
        q = q.permute(0, 2, 1).reshape(-1, seqlen)  # (bs*, seq_len), 嵌入层标准输入格式
        q_emb = self.q_embed(q)                     # 输出为(bs*, seq_len, d_model)
        # q_emb = self.q_embedding(q)               # old
        """将知识点映射向量加至question embeddings"""
        q_emb = q_emb + self.qemb_matrix[q]         # (bs*, seq_len, d_model)

        q_emb = q_emb.view(bs, kp_len, seqlen, -1)  # 还原shape
        q_emb = q_emb.mean(dim=1, keepdim=False)    # 如果是一道题对应多个知识点，这里直接进行求平均操作
        # c_emb = q_emb

        # e_(ct,rt) = g_rt + c_ct  e_(ct,rt): concept-response embedding
        s_emb = self.s_embed(s) + q_emb

        # p_diff = 0.0
        pid = pid.masked_fill(pid < 0, 0)
        p_diff = self.p_diff_embed(pid)  # u_q 当前problem难度

        # q无需再次reshape
        q_diff_emb = self.q_diff_embed(q)  # d_ct 总结了包含当前question（concept）的problems（questions）的变化
        q_diff_emb = q_diff_emb.view(bs, kp_len, seqlen, -1)
        q_diff_emb = q_diff_emb.mean(dim=1, keepdim=False)

        # question embedding q_t = c_ct + uq *d_ct
        q_emb += q_diff_emb * p_diff
        if self.n_topic is not None:
            q_emb += self.topic_embed(topic)
        if self.n_tense is not None:
            q_emb += self.tense_embed(tense)
        # f = s_diff_embed + d_ct f summarizes the variations of learning activities within this concept
        s_diff_emb = self.s_diff_embed(s) + q_diff_emb
        # s_t = e_(ct,rt) + u_q * f
        s_emb += s_diff_emb * p_diff
        return q_emb, s_emb, lens, p_diff  # , c_emb

    def forward(self, q_emb, s_emb, lens):  # , c_emb
        """
        forward 一般在其他方法，如 predict 中进行调用，会先于其调用embedding，然后将嵌入好的内容作为q,s,pid传入
        """
        if self.n_layers == 1:
            hq = q_emb
            p, q_scores = self.block1(q_emb, q_emb, s_emb, lens, peek_cur=True)
        elif self.n_layers == 2:
            hq = q_emb
            hs, _ = self.block2(s_emb, s_emb, s_emb, lens, peek_cur=True)
            p, q_scores = self.block3(hq, hq, hs, lens, peek_cur=True)
        else:  # self.n_layers == 3:
            hq, _ = self.block1(q_emb, q_emb, q_emb, lens, peek_cur=True)
            hs, _ = self.block2(s_emb, s_emb, s_emb, lens, peek_cur=True)
            p, q_scores = self.block3(hq, hq, hs, lens, peek_cur=True)  # (bs, len, d_model)  (bs, h, len, len)

        """计算c_emb，并通过其 self attention 计算得到 know_params"""
        bs, seqlen, d_model = p.size()
        n_know = self.n_know
        c_emb = self.q_embed.weight[1:].unsqueeze(0)        # (n_questions, dim) 排除第一个未使用的嵌入
        c_emb = c_emb + self.qemb_matrix[1:].unsqueeze(0)   # 240903: 加上知识点映射向量！！！
        c, _ = self.block5(c_emb, c_emb, c_emb)             # 自注意力机制
        self.know_params = self.c_k_linear(c.mean(dim=1)).reshape(n_know, d_model)  # Shape: (1, n_know * dim)，再扩展到 (n_know, dim) 的形状 n_konw=1

        # 获取稠密能力表征z
        query = (
            self.know_params[None, :, None, :]
            .expand(bs, -1, seqlen, -1)
            .contiguous()
            .view(bs * n_know, seqlen, d_model)
        )
        hq = hq.unsqueeze(1).expand(-1, n_know, -1, -1).reshape_as(query)
        p = p.unsqueeze(1).expand(-1, n_know, -1, -1).reshape_as(query)
        z, k_scores = self.block4(query, hq, p, torch.repeat_interleave(lens, n_know), peek_cur=False)

        # reshape and return
        z = (
            z.view(bs, n_know, seqlen, d_model)
            .transpose(1, 2)  # (bs, seqlen, n_know, d_model)
            .contiguous()
            .view(bs, seqlen, -1)
        )  # (bs, seqlen, n_know * d_model)
        k_scores = (
            k_scores.view(bs, n_know, self.n_heads, seqlen, seqlen)  # unpack dimensions
            .permute(0, 2, 3, 1, 4)  # (bs, n_heads, seqlen, n_know, seqlen)
            .contiguous()
        )
        return z, q_scores, k_scores

    def readout(self, z, query):
        """
        仍在CDM的作答结果预报中使用，知识点映射模型中已无需使用
        softmax(know * query) * z
        """
        bs, seqlen, _ = query.size()
        key = (
            self.know_params[None, None, :, :]
            .expand(bs, seqlen, -1, -1)
            .view(bs * seqlen, self.n_know, -1)
        )  # (bs * seqlen, n_know, d_model)

        value = z.reshape(bs * seqlen, self.n_know, -1)  # (bs * seqlen, n_know, d_model)

        beta = torch.matmul(
            key,
            query.reshape(bs * seqlen, -1, 1)  # (bs, seqlen, d_model)->(bs * seqlen, d_model, 1)
        ).view(bs*seqlen, 1, self.n_know)  # (bs * seqlen, n_know, 1) -> (bs * seqlen, 1, n_know)

        alpha = torch.softmax(beta, dim=-1)  # (bs * seqlen, 1, n_know)
        return torch.matmul(alpha, value).view(bs, seqlen, -1)  # (bs * seqlen, 1, d_model) -> (bs, seqlen, d_model)

    def readout_attention(self, z, query):
        bs, seqlen, _ = query.size()
        q = query.reshape(bs * seqlen, 1, -1) # (bs * seq_len, 1, d_model)
        key = (
            self.know_params[None, None, :, :]
            .expand(bs, seqlen, -1, -1)
            .view(bs * seqlen, self.n_know, -1)
        )  # (bs * seqlen, n_know, d_model)

        value = z.reshape(bs * seqlen, self.n_know, -1)  # (bs * seqlen, n_know, d_model)

        h,_ = self.block6(q, key, value) # (bs * seqlen, 1, d_model)
        return h.view(bs, seqlen, -1)  # (bs * seqlen, 1, d_model) -> (bs, seqlen, d_model)

    def concept_attention(self, z, query):
        """将该功能整合至conceptpredictor"""
        bs, *_ = z.size()
        key = (
            self.know_params[None, :, :]
            .expand(bs,  -1, -1)
            .view(bs, self.n_know, -1)
        )  # (bs , n_know, d_model)
        # value = z.reshape(bs, self.n_know, -1)  # (bs , n_know, d_model)
        d_k = key.size(-1)
        # (bs, n_c, d_model) * (bs, d_model, n_know) -> (bs, n_c, n_know)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        scores = F.softmax(scores, dim=-1)
        # (bs, n_c, n_know) * (bs, n_know, d_model) -> (bs, n_c, d_model)
        output = torch.matmul(scores, z)
        return output
    # 知识点映射c_embed为 query，know_params 为 key, 最后时刻 z为value, 通过注意力计算得到 z到知识点的映射

    def get_last_z(self, z, s, n):
        # 获取最后一个 z
        # 找到 s 中每行最后一个非 padding 值的位置
        indices = (s >= 0).sum(dim=1) - 1
        # batch_indices = torch.arange(s.size(0))
        last_z = torch.zeros(s.size(0), n, z.size(-1)).to(self.device())
        for i in range(s.size(0)):
            last_z[i] = z[i, indices[i] - n:indices[i], :]
        # (bs, n_know * d_model)
        last_z = torch.mean(last_z, dim=1, keepdim=False)
        return last_z

    def concept_map(self, z, s, n=1):
        """
        获取能力映射知识点注意力
        """
        last_z = self.get_last_z(z, s, n)
        bs = s.size(0)

        # z_embed = last_z.unsequeeze(1) # value (bs, 1, n_know*d_model)
        # c_embed = (
        #     self.q_embed.weight
        #     .expand(bs, -1, -1) # (bs, n_know, d_model)
        #     .unsqueeze(1) #  (bs,1, n_know, d_model)
        #     .view(bs,1,-1)
        # ) # query (bs,1, n_know * d_model)

        z_embed = last_z.reshape(bs, self.n_know, -1)  # value (bs, n_know, d_model)
        c_embed = self.q_embed.weight.expand(bs, -1, -1)  # query (bs, n_c, d_model)
        # (bs, n_c, d_model)
        output = self.concept_attention(z_embed, c_embed)
        return output

    def concept_map_new(self, z, s, n=1):
        last_z = self.get_last_z(z, s, n)
        bs = s.size(0)

        z_embed = last_z.reshape(bs, self.n_know, -1)                                    # value (bs, n_know, d_model)
        c_embed = self.q_embed.weight.expand(bs, -1, -1)                                 # query (bs, n_c, d_model)
        key = self.know_params[None, :, :].expand(bs, -1, -1).view(bs, self.n_know, -1)  # key   (bs ,n_know, d_model)
        return c_embed, key, z_embed

    def tense_map(self, z, s, n=1):
        last_z = self.get_last_z(z, s, n)
        bs = s.size(0)
        z_embed = last_z.reshape(bs, self.n_know, -1)  # value (bs, n_know, d_model)
        t_embed = self.tense_embed.weight.expand(bs, -1, -1)  # query (bs, n_tense, d_model)
        # (bs, n_tense, d_model)
        output = self.concept_attention(z_embed, t_embed)
        return output

    def predict(self, q, s, pid, topic=None, tense=None, n=1):
        """
        整体添加topic和tense参数，用于传入embedding方法
        # T+N预测是否答对
        """
        # 利用Rasch embedding获取编码后的问题和问题-答案对
        q_emb, s_emb, lens, p_diff = self.embedding(q, s, pid, topic, tense)
        # 模型推理获取知识层级的掌握情况 z (bs, seqlen, n_know * d_model)
        # z (bs, seqlen, n_know * d_model), q_scores (bs, n_heads, seqlen, seqlen), k_scores (bs, n_know, n_heads, seqlen, seqlen)
        z, q_scores, k_scores = self(q_emb, s_emb, lens)  # 调用forward，返回 z (bs, seqlen, n_know * d_model)
        # 要预测的 query
        # predict T+N
        query = q_emb[:, n-1:, :]  # (bs, seqlen2, d_model)
        # h (bs, seqlen2, d_model)
        h = self.readout(z[:, : query.size(1), :], query)
        # h = self.readout_attention(z[:, : query.size(1), :], query)
        # concat (bs, seqlen2, d_model*2)
        # y (bs, seqlen2, 1) -> (bs, seqlen2)
        y = self.out(torch.cat([query, h], dim=-1)).squeeze(-1)
        # (bs,seq_len2), (bs,seq_len,n_know*d_model),(bs,seq_len,d_model),(bs,seq_len,1)
        return y, z, q_emb, (p_diff**2).mean() * 1e-3, (q_scores, k_scores)

    def get_loss(self, q, s, pid, topic=None, tense=None):
        """
        整体添加topic和tense参数，用于传入predict方法
        """
        logits, _, _, reg_loss, _ = self.predict(q, s, pid, topic=topic, tense=tense)
        # 挑选出有作答的记录，-1代表未作答
        masked_labels = s[s >= 0].float()
        masked_logits = logits[s >= 0]
        return (
            F.binary_cross_entropy_with_logits(
                masked_logits, masked_labels, reduction="mean"
            )
            + reg_loss  # 正则化损失
        )

    # 对比损失
    def get_cl_loss(self, q, s, pid=None, topic=None, tense=None):
        """
        整体添加topic和tense参数，用于传入predict方法
        """
        bs = s.size(0)

        # skip CL for batches that are too short
        lens = (s >= 0).sum(dim=1)
        minlen = lens.min().item()
        if minlen < MIN_SEQ_LEN:
            return self.get_loss(q, s, pid, topic=topic, tense=tense)

        q_ = q.clone()
        s_ = s.clone()
        pid_ = pid.clone()
        topic_ = None
        tense_ = None
        if topic is not None:
            topic_ = topic.clone()
        if tense is not None:
            tense_ = tense.clone()

        # 交换顺序
        for b in range(bs):
            idx = random.sample(
                range(lens[b]-1), max(1, int(lens[b] * self.dropout_rate))
            )
            for i in idx:
                q_[b, i], q_[b, i + 1] = q_[b, i + 1], q_[b, i]
                s_[b, i], s_[b, i + 1] = s_[b, i + 1], s_[b, i]
                pid_[b, i], pid_[b, i + 1] = pid_[b, i + 1], pid_[b, i]
                if topic is not None:
                    topic_[b, i], topic_[b, i + 1] = topic_[b, i + 1], topic_[b, i]
                if tense is not None:
                    tense_[b, i], tense_[b, i + 1] = tense_[b, i + 1], tense_[b, i]

        # 翻转score
        s_flip = s.clone() if self.hard_neg else s_
        for b in range(bs):
            # manipulate score
            idx = random.sample(
                range(lens[b]), max(1, int(lens[b] * self.dropout_rate))
            )
            for i in idx:
                s_flip[b, i] = 1 - s_flip[b, i]
        if not self.hard_neg:
            s_ = s_flip

        # model predict
        logits, z_1, q_emb, reg_loss, _ = self.predict(q, s, pid, topic=topic, tense=tense)
        masked_logits = logits[s >= 0]

        _, z_2, *_ = self.predict(q_, s_, pid_, topic=topic_, tense=tense_)
        if self.hard_neg:
            _, z_3, *_ = self.predict(q, s_flip, pid, topic=topic, tense=tense)

        # 计算对比损失
        _input = self.sim(z_1[:, :minlen, :], z_2[:, :minlen, :])
        if self.hard_neg:
            hard_neg = self.sim(z_1[:, :minlen, :], z_3[:, :minlen, :])
            _input = torch.cat([_input, hard_neg], dim=1)
        target = (
            torch.arange(s.size(0))[:, None]
            .to(self.know_params.device)
            .expand(-1, minlen)
        )

        cl_loss = F.cross_entropy(_input, target)

        # 利用前面已计算的logits，计算prediction loss
        # 初始预测损失计算
        masked_labels = s[s >= 0].float()
        pred_loss = F.binary_cross_entropy_with_logits(
            masked_logits, masked_labels, reduction="mean"
        )
        # 后续预测损失计算
        for i in range(1, self.window):
            label = s[:, i:]
            query = q_emb[:, i:, :]
            h = self.readout(z_1[:, : query.size(1), :], query)
            # h = self.readout_attention(z_1[:, : query.size(1), :], query)
            y = self.out(torch.cat([query, h], dim=-1)).squeeze(-1)
            pred_loss += F.binary_cross_entropy_with_logits(
                y[label >= 0], label[label >= 0].float()
            )
        pred_loss /= self.window

        return pred_loss + cl_loss * self.lambda_cl + reg_loss, pred_loss, cl_loss

    def sim(self, z1, z2):
        bs, seqlen, _ = z1.size()  # (bs,seq_len,n_know*d_model)
        z1 = z1.unsqueeze(1).view(bs, 1, seqlen, self.n_know, -1)
        z2 = z2.unsqueeze(0).view(1, bs, seqlen, self.n_know, -1)
        if self.proj is not None:
            z1 = self.proj(z1)
            z2 = self.proj(z2)
        return F.cosine_similarity(z1.mean(-2), z2.mean(-2), dim=-1) / 0.05  # 0.05为温度超参数

    # def tracing(self, q, s, pid):
    #     # add fake q, s, pid to generate the last tracing result (bs=1,seq_len+1)
    #     pad = torch.tensor([0]).to(self.know_params.device)
    #     q = torch.cat([q, pad], dim=0).unsqueeze(0)
    #     s = torch.cat([s, pad], dim=0).unsqueeze(0)
    #     pid = torch.cat([pid, pad], dim=0).unsqueeze(0)
    #
    #     with torch.no_grad():
    #         # q_emb: (bs, seq_len, d_model) bs=1
    #         # z: (bs, seq_len, n_know * d_model)
    #         # know_params: (n_know, d_model)->(n_know, 1, d_model)
    #         q_emb, s_emb, lens, _ = self.embedding(q, s, pid)
    #         z, _, _ = self(q_emb, s_emb, lens)
    #         # (n_know, seq_len, d_model)
    #         query = self.know_params.unsqueeze(1).expand(-1, z.size(1), -1).contiguous()
    #         # (n_know, seq_len, n_know * d_model)
    #         z = z.expand(self.n_know, -1, -1).contiguous()
    #         # (n_know, seq_len, d_model)
    #         h = self.readout(z, query)
    #         y = self.out(torch.cat([query, h], dim=-1)).squeeze(-1)
    #         # (n_know, seq_len)
    #         y = torch.sigmoid(y)
    #
    #     return y


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
            q,q_scores = layer(q, q, q, lens, peek_cur)
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
