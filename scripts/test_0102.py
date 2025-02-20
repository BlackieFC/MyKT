import os
import sys
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from argparse import ArgumentParser
import torch
import tomlkit
from tqdm import tqdm
from models.data import KTData
from models.eval import Evaluator_with_t as Evaluator
from models.CDMTransformer_withT import CDMTransformer_0102

from dotenv import load_dotenv
load_dotenv()
DATA_DIR = os.getenv('DATA_DIR')

# import wandb
# wandb.login(key=os.getenv('WANDB_API_KEY'))


def prepare_data():
    """
    读取文本嵌入和tf-idf信息
    """
    # （1）tf-idf信息
    df = pd.read_excel('../data/arithmetic_retag/arithmetic_tfidf_norm.xlsx')
    exer_n = df['point_id'].max()                           # 677
    knowledge_n = df['skillId'].max()                       # 70
    # 声明输出
    tfidf_norm = torch.zeros(exer_n+1, knowledge_n+1)       # (677+1, 70+1)
    for ind, row in df.iterrows():                          # 不再进行ID与索引的对齐（直接引入空的0索引——可以用来学习padding）
        idx_row = row['point_id']  # - 1
        idx_col = row['skillId']   # - 1
        tfidf_norm[idx_row, idx_col] = row['TF-IDF_norm']

    # （2）KC文本嵌入
    kc_embeds = np.load('../data/arithmetic_retag/kc_emb.npy')  # (70, 1024)
    # 拼接0索引（空置——用于学习padding）
    kc_embeds = np.concatenate((np.zeros((1, kc_embeds.shape[1])), kc_embeds), axis=0)  # (70, 1024) -> (71, 1024)
    kc_embeds = torch.tensor(kc_embeds)

    return tfidf_norm, kc_embeds  # (677+1, 70+1), (70+1, 1024)


"""输入传参设置"""


parser = ArgumentParser()

# general options
parser.add_argument("--device", help="device to run network on", default="cpu")
parser.add_argument("-bs", "--batch_size", help="batch size", default=8, type=int)
parser.add_argument("-tbs", "--test_batch_size", help="test batch size", default=64, type=int)

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
parser.add_argument("--d_model", help="model hidden size", type=int, default=128)
parser.add_argument("--n_layers", help="number of layers", type=int, default=3)
parser.add_argument("--n_heads", help="number of heads", type=int, default=8)
parser.add_argument("--n_know", help="dimension of knowledge parameter", type=int, default=32)
parser.add_argument("--dropout", help="dropout rate", type=float, default=0.2)
parser.add_argument("--proj", help="projection layer before CL", action="store_true")
parser.add_argument("--hard_neg", help="use hard negative samples in CL", action="store_true")
parser.add_argument("-di", "--diff", help="whether to initialize p_diff", default=False)

# 1119新增
parser.add_argument("--max_t", help="maximum duration (s)", default=500)
# parser.add_argument("--max_t", help="maximum duration (s)", default=1000)
# parser.add_argument("--max_seqlen", help="maximum sequence length", default=200)
parser.add_argument("--p_ratio", help="ratio of correctness in loss", default=0.1)
parser.add_argument("--padding_value", help="padding value", default=-1)

# training setup
parser.add_argument("-n", "--n_epochs", help="training epochs", type=int, default=100)
parser.add_argument("-es", "--early_stop", help="early stop after N epochs of no improvements",
                    type=int, default=10)
parser.add_argument("-lr", "--learning_rate", help="learning rate", type=float, default=1e-3)
parser.add_argument("-l2", help="L2 regularization", type=float, default=1e-5)
parser.add_argument("-cl", "--cl_loss", help="use contrastive learning loss", action="store_true")
parser.add_argument("--lambda", help="CL loss weight", type=float, default=0.1, dest="lambda_cl")
parser.add_argument("--window", help="prediction window", type=int, default=1)

# snapshot setup
parser.add_argument("-o", "--output_dir", help="directory to save model files and logs")
parser.add_argument("-f", "--from_file", help="resume training from existing model file", default=None)



def main(args):
    # run = wandb.init(project=f"CAT-{args.dataset}", config=args)

    """根据dataset获取toml文件中对应信息"""

    max_t = args.max_t                  # 最大单题作答时间
    # max_len = args.max_seqlen         # 最大序列长度
    p_ratio = args.p_ratio
    padding_value = args.padding_value  # padding值设置

    dataset = datasets[args.dataset]
    # 新增知识点长度参数，同时将其转换为嵌套列表
    seq_len = dataset["seq_len"] if "seq_len" in dataset else None
    kp_len = dataset["kp_len"] if "kp_len" in dataset else None
    if kp_len is not None:
        kp_len = list(kp_len)
        for ind, temp in enumerate(kp_len):
            if isinstance(temp, int):
                continue
            else:
                kp_len[ind] = list(kp_len[ind])

    """封装dataloader"""
    valid_data = KTData(
        data_path=os.path.join(DATA_DIR, dataset["valid"] if "valid" in dataset else dataset["test"]),
        inputs=dataset["inputs"],
        num_kps=(dataset["n_pid"], dataset["n_questions"], None),
        group=dataset['n_features'],
        seq_len=seq_len,
        kp_len=kp_len,
        batch_size=args.test_batch_size,
        shuffle=False,
        name_know=None,
        num_know=None,
        rand_init=True,
        type_in='txt',
    )

    """实例化模型"""
    # 0102: 文本嵌入based
    emb_tab_exer, emb_tab_kc = prepare_data()
    model = CDMTransformer_0102(
        emb_tab_exer,               # 题目嵌入层freeze权重: (677+1, 70+1)
        emb_tab_kc,                 # KC 嵌入层freeze权重: (70+1, 1024)
        d_model=args.d_model,       # 特征维度
        d_ff=256,                   # 隐藏层维度
        n_attn_heads=args.n_heads,  # mha头数
        n_know=args.n_know,         # know_param大小
        n_layers=args.n_layers,     # attention层堆叠数量
        dropout=args.dropout,       # dropout
        window=args.window,         # 向前预报时的窗口大小
        n_y=100,                    # 做题正确率倍率（百分数）
        n_t=max_t,                  # 做题时间上限
        p_ratio=p_ratio,            # loss中作答正确率项的倍率
        use_param_sigmoid=True      # 使用含参sigmoid函数
    )

    # 是否读取预训练权重
    if args.from_file:
        model.load_state_dict(torch.load(args.from_file, map_location=lambda _s, _: _s))

    # 放置于GPU
    model.to(args.device)
    # wandb.watch追踪模型的参数变化/梯度分布等
    # wandb.watch(model, log="all", log_freq=10)

    """evaluate"""
    model.eval()
    # evaluator = Evaluator(t_ratio=max_t)                        # 需要传入t的截断/倍率参数
    evaluator = Evaluator(t_ratio=max_t, t_weight=1-p_ratio)      # 0102：修正权重项

    with torch.no_grad():
        it = tqdm(iter(valid_data))
        for batch in it:
            q, s, pid, t = batch.get("q", "s", "pid", "t")
            if not isinstance(q, torch.Tensor):
                q = torch.stack(q, dim=-1)  # 50x(32,36) -> (32,36,50)
                q = q.to(args.device)
                s = s[0].to(args.device)
                pid = pid[0].to(args.device)
                t = t[0].to(args.device)
            else:
                q = q.to(args.device)
                s = s.to(args.device)
                pid = pid.to(args.device)
                t = t.to(args.device)
            # 根据max_t对t进行处理
            t = torch.clamp(t, min=padding_value, max=max_t)  # 241125: 注意min的选取不要让padding失效！！！

            s_pred, _, _, _, t_pred = model.predict(q, s, pid, t)      # 现在predict返回的是百分数和作答时间
            """核心变化"""
            # evaluator.evaluate(s, s_pred, t, t_pred)
            evaluator.statistics(s, s_pred, t, t_pred)
    
    # print 统计结果
    # evaluator.result_statistics()
    evaluator.last_step_stats('last_step_stats_250102.xlsx')

    return None


if __name__ == "__main__":

    args_in = parser.parse_args()
    print(args_in)

    main(args_in)
