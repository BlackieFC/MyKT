import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import numpy as np
from argparse import ArgumentParser
from datetime import datetime
import torch
import tomlkit
from tqdm import tqdm
from models.data import KTData
from models.eval import Evaluator_with_t as Evaluator
# from models.CDMTransformer import CDMTransformer  # 选取需要的具体模型架构！！！
from models.CDMTransformer_1120 import CDMTransformer_1120, CDMTransformer_1125

# import wandb
from dotenv import load_dotenv
load_dotenv()
# wandb.login(key=os.getenv('WANDB_API_KEY'))
DATA_DIR = os.getenv('DATA_DIR')


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
# parser.add_argument("--max_seqlen", help="maximum sequence length", default=200)
parser.add_argument("--p_ratio", help="ratio of correctness in loss", default=1)
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

    max_t = args.max_t  # 最大单题作答时间
    # max_len = args.max_seqlen  # 最大序列长度
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
    # 读取知识点映射向量（tomlkit中不存在时为None，不生效）
    q_emb = dataset['q_emb'] if 'q_emb' in dataset else None
    if q_emb is not None:
        q_emb = np.load(q_emb)                                                          # np.ndarray{478,128}
        q_emb = np.concatenate((np.zeros((1, q_emb.shape[-1])), q_emb), axis=0)   # np.ndarray{479,128}
        q_emb = torch.from_numpy(q_emb).float().to(args.device)
    # 题目难度初始化(除非输入，否则不进行)
    if args.diff:
        diff_path = os.path.join(DATA_DIR, dataset["diff_pid"])
    else:
        diff_path = None

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
    # model = CDMTransformer_1120(
    #     dataset["n_questions"],
    #     dataset["n_pid"],
    #     path_diff=diff_path,  # 新增diff
    #     d_model=args.d_model,
    #     n_layers=args.n_layers,
    #     n_attn_heads=args.n_heads,
    #     n_know=args.n_know,
    #     lambda_cl=args.lambda_cl,
    #     dropout=args.dropout,
    #     proj=args.proj,
    #     hard_neg=args.hard_neg,
    #     window=args.window,
    #     qemb_matrix=q_emb,   # 240903: 新增知识点映射向量(默认为None)
    #     n_t=max_t,           # 最大做题时间
    # )

    # 1125: 模型结构修改验证
    model = CDMTransformer_1125(
        dataset["n_questions"],
        dataset["n_pid"],
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_attn_heads=args.n_heads,
        n_know=args.n_know,
        dropout=args.dropout,
        window=args.window,
        n_t=max_t,           # 最大做题时间
        p_ratio=p_ratio,     # loss中正确率的倍率
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
    evaluator = Evaluator(t_ratio=max_t, t_weight=1/(p_ratio+1))  # 1125：修正权重项

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
    evaluator.last_step_stats('last_step_stats_241125.xlsx')

    return None


if __name__ == "__main__":

    args_in = parser.parse_args()
    print(args_in)

    main(args_in)
