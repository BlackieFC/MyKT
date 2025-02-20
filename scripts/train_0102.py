import os
import sys
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import numpy as np
from argparse import ArgumentParser
from datetime import datetime
import torch
import tomlkit
from tqdm import tqdm
from models.data import KTData
from models.eval import Evaluator
# from models.CDMTransformer import CDMTransformer  # 选取需要的具体模型架构！！！
from models.CDMTransformer_withT import CDMTransformer_0102

import wandb
from dotenv import load_dotenv
load_dotenv()
wandb.login(key=os.getenv('WANDB_API_KEY'))
DATA_DIR = os.getenv('DATA_DIR')


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
                    type=int, default=5)
parser.add_argument("-lr", "--learning_rate", help="learning rate", type=float, default=1e-3)
parser.add_argument("-l2", help="L2 regularization", type=float, default=1e-5)
parser.add_argument("-cl", "--cl_loss", help="use contrastive learning loss", action="store_true")
parser.add_argument("--lambda", help="CL loss weight", type=float, default=0.1, dest="lambda_cl")
parser.add_argument("--window", help="prediction window", type=int, default=1)

# snapshot setup
parser.add_argument("-o", "--output_dir", help="directory to save model files and logs")
parser.add_argument("-f", "--from_file", help="resume training from existing model file", default=None)


"""training"""


def main(args):
    run = wandb.init(project=f"CAT-{args.dataset}", config=args)

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
    train_data = KTData(
        data_path=os.path.join(DATA_DIR, dataset["train"]),        # arithmetic_pointWise/train.txt
        inputs=dataset["inputs"],                                  # ["pid", "q", "s", "t"]
        num_kps=(dataset["n_pid"], dataset["n_questions"], None),  # 240821: 适用于新功能，对输入数据的取值范围进行检查
        group=dataset['n_features'],                               # 4
        seq_len=seq_len,                                           # None(deprecated)
        kp_len=kp_len,                                             # [[0,1], [1,11], [11,12], [12,13]]
        batch_size=args.batch_size,                                # [[0,1], [1,51], [51,52], [52,53]]
        shuffle=True,
        name_know=None,
        num_know=None,
        rand_init=True,
        type_in='txt',
    )
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

    # prepare logger and output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    dt = datetime.now().strftime('%Y%m%d')
    if args.output_dir:
        os.makedirs(os.path.join(args.output_dir, dt+f"_{args.d_model}_{args.n_know}"), exist_ok=True)
        # os.makedirs(args.output_dir, exist_ok=True)
        config_path = os.path.join(args.output_dir, dt+f"_{args.d_model}_{args.n_know}", f"config.json")
        json.dump(vars(args), open(config_path, "w"), indent=2)
    else:
        print("No persistency warning: No output directory provided. Data will not be saved.")

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

    # 优化器
    optim = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.l2
    )
    # optim = torch.optim.Adam(
    #     model.parameters(), lr=args.learning_rate, weight_decay=0
    # )

    # 放置于GPU
    model.to(args.device)
    # wandb.watch追踪模型的参数变化/梯度分布等
    # wandb.watch(model, log="all", log_freq=10)

    """training"""
    best_info = {"mae": float('inf'), "rmse": float('inf')}  # "auc": 0,
    best_epoch = 0
    for epoch in range(1, args.n_epochs + 1):
        print("start epoch", epoch)
        model.train()
        it = tqdm(iter(train_data))
        total_loss = 0.0
        total_cnt = 0
        for batch in it:
            q, s, pid, t = batch.get("q", "s", "pid", "t")
            
            if not isinstance(q, torch.Tensor):
                q = torch.stack(q, dim=-1)  # kp_len x (32,36) -> (32,36,kp_len)
                q = q.to(args.device)       # q比较特殊，是长度为 kp_len 的 list(但该情况下也为1)
                s = s[0].to(args.device)
                pid = pid[0].to(args.device)
                t = t[0].to(args.device)
            else:
                q = q.to(args.device)
                s = s.to(args.device)
                pid = pid.to(args.device)
                t = t.to(args.device)  
            # 根据max_t对t进行截断处理（注意min的选取不要让padding失效！！！）
            t = torch.clamp(t, min=padding_value, max=max_t)

            # BP优化
            loss, loss_y, loss_t = model.get_loss(q, s, pid, t)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
            optim.step()

            # 结果记录
            total_loss += loss.item()
            total_cnt += 1                                                  # (s >= 0).sum().item()
            postfix = {"loss": total_loss / total_cnt, "epoch": epoch}
            it.set_postfix(postfix)
            wandb.log(postfix)                                              # 向wandb传入想要记录的键值对的字典

        """validating"""
        model.eval()
        evaluator = Evaluator(t_ratio=max_t, t_weight=1-p_ratio)            # 需要传入t的截断/倍率参数，0102：修正权重项

        with torch.no_grad():
            it = tqdm(iter(valid_data))
            for batch in it:
                q, s, pid, t = batch.get("q", "s", "pid", "t")
                if not isinstance(q, torch.Tensor):
                    q = torch.stack(q, dim=-1)
                    q = q.to(args.device)
                    s = s[0].to(args.device)
                    pid = pid[0].to(args.device)
                    t = t[0].to(args.device)
                else:
                    q = q.to(args.device)
                    s = s.to(args.device)
                    pid = pid.to(args.device)
                    t = t.to(args.device)
                # 根据max_t对t进行处理（注意min的选取不要让padding失效！！！）
                t = torch.clamp(t, min=padding_value, max=max_t)

                s_pred, _, _, _, t_pred = model.predict(q, s, pid, t)      # 现在predict返回的是正确率百分数和作答时间
                # evaluator.evaluate(s, torch.sigmoid(y), t, t_pred)
                evaluator.evaluate(s, s_pred, t, t_pred)
        
        r = evaluator.report()
        r['epoch'] = epoch
        print(r)
        wandb.log(r)  # 传入 evaluator.report()
        if args.output_dir:
            config_path = os.path.join(args.output_dir, dt+f"_{args.d_model}_{args.n_know}", f"report.json")
            with open(config_path, "a") as f:
                f.write(json.dumps(r, indent=4) + "\n")

        if r["rmse"] < best_info["rmse"]:
            best_info = r
            best_epoch = epoch

            if args.output_dir:
                model_path = os.path.join(
                    args.output_dir, dt+f"_{args.d_model}_{args.n_know}", f"model.pt"  # _{epoch:03d}
                )
                print("saving snapshot to:", model_path)
                torch.save(model.state_dict(), model_path)

        if 0 < args.early_stop < epoch - best_epoch:
            print(f"did not improve for {args.early_stop} epochs, stop early")
            break
    wandb.finish()
    best_info['best_epoch'] = best_epoch
    if args_in.output_dir:
        config_path = os.path.join(args_in.output_dir, dt+f"_{args.d_model}_{args.n_know}", f"report.json")
        with open(config_path, "a") as f:
            f.write(json.dumps(best_info, indent=4))
    return best_info


if __name__ == "__main__":

    args_in = parser.parse_args()
    print(args_in)

    best = main(args_in)
    print(args_in)

    print("best result", {k: f"{v:.4f}" for k, v in best.items()})

