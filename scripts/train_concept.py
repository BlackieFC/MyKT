import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import torch
import wandb
import tomlkit
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from dotenv import load_dotenv
from argparse import ArgumentParser
from models.data import KTData
from models.eval import Evaluator
from models.CDMTransformer_1120 import CDMTransformer  # 注意模型版本！
from models.ConceptPredictor import ConceptPredictor_GCN as ConceptPredictor


load_dotenv()
wandb.login(key=os.getenv('WANDB_API_KEY'))
DATA_DIR = os.getenv('DATA_DIR')
parser = ArgumentParser()
# general options
parser.add_argument("--device", help="device to run network on", default="cpu")
parser.add_argument("-bs", "--batch_size", help="batch size", default=8, type=int)
parser.add_argument(
    "-tbs", "--test_batch_size", help="test batch size", default=16, type=int
)
parser.add_argument("--type", help="which type of concept to predict: word or tense", default="word")

# dataset options
datasets = tomlkit.load(open(os.path.join(DATA_DIR, "datasets.toml")))
parser.add_argument(
    "-d",
    "--dataset",
    help="choose from a dataset",
    choices=datasets.keys(),
    default="",
    required=True,
)

# model options
# CDM model
parser.add_argument("--d_model", help="CDMTransformer model hidden size", type=int, default=128)
parser.add_argument("--n_layers", help="number of layers", type=int, default=3)
parser.add_argument("--n_heads", help="number of heads", type=int, default=8)
parser.add_argument("--n_know", help="dimension of CDMTransformer model knowledge parameter", type=int, default=32)
parser.add_argument("--proj", help="projection layer before CL", action="store_true")
parser.add_argument("-f", "--from_file", help="CDMTransformer model file", required=True)
# ConceptPredictor model
parser.add_argument("--dropout", help="dropout rate", type=float, default=0.2)
parser.add_argument("--h_dim", help="gnn hidden size", type=float, default=128)
parser.add_argument("--node_dim", help="concept graph node embedding size", type=float, default=128)


# training setup
parser.add_argument("-n", "--n_epochs", help="training epochs", type=int, default=100)
parser.add_argument(
    "-es",
    "--early_stop",
    help="early stop after N epochs of no improvements",
    type=int,
    default=10,
)
parser.add_argument(
    "-lr", "--learning_rate", help="learning rate", type=float, default=2e-3
)
parser.add_argument("-l2", help="L2 regularization", type=float, default=1e-5)

# snapshot setup
parser.add_argument("-o", "--output_dir", help="directory to save concept predict model files and logs")

# 加载知识点图的节点特征和边
def load_graph_data(node_feature_file, edge_file):
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

def main(args):
    """
    根据答题记录获取能力评估，将能力评估 z作为输入，通过KnowledgePredictor获取已掌握的知识点，
    能力评估模型，需要输入历史做题数据，从而得到能力评估 z，在后续模型训练中保持能力评估模型参数固定
    :param args:
    :return:
    """
    run = wandb.init(project=f"Z2Concept-{args.dataset}", config=args)

    # 根据dataset获取toml文件中对应信息，
    dataset = datasets[args.dataset]
    seq_len = dataset["seq_len"] if "seq_len" in dataset else None  # seq_len和kp_len影响KTData，还是要保留
    kp_len = dataset["kp_len"] if "kp_len" in dataset else None     # 新增知识点长度参数，同时将其转换为嵌套列表
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

    # 训练&验证集的dataloader封装
    train_data = KTData(
        data_path=os.path.join(DATA_DIR, dataset["train"]),
        inputs=dataset["inputs"],
        num_kps=(dataset["n_pid"], dataset["n_questions"], None),  # 240821: 适用于新功能，对输入数据的取值范围进行检查
        group=dataset['n_features'],
        seq_len=seq_len,  # deprecated
        kp_len=kp_len,  # [[0,1], [1, 51], [51,52]]
        batch_size=args.batch_size,
        shuffle=True,
        name_know=dataset['inputs_know'],  # 适配调整后的KTData，此处为["ind_c","val_c"]
        num_know=(dataset["n_questions"], None),  # 240821: 适用于新功能，对输入数据的取值范围进行检查（若有多个能力指标，则效仿num_kps传入元组）
        rand_init=False,                   # 240821:换用sparse损失后，需要设置为False（此时会padding为-1）
        type_in='txt',
    )
    valid_data = KTData(
        data_path=os.path.join(
            DATA_DIR, dataset["valid"] if "valid" in dataset else dataset["test"]
        ),
        inputs=dataset["inputs"],
        num_kps=(dataset["n_pid"], dataset["n_questions"], None),  # 240821: 适用于新功能，对输入数据的取值范围进行检查
        group=dataset['n_features'],
        seq_len=seq_len,  # deprecated
        kp_len=kp_len,  # [[0,1], [1, 51], [51,52]]
        batch_size=args.test_batch_size,
        shuffle=False,
        name_know=dataset['inputs_know'],  # 同上，适配调整后的KTData
        num_know=(dataset["n_questions"], None),  # 240821: 适用于新功能，对输入数据的取值范围进行检查（若有多个能力指标，则效仿num_kps传入元组）
        rand_init=False,                   # 同上
        type_in='txt',
    )

    # 输出路径
    dt = datetime.now().strftime('%Y%m%d')
    if args.output_dir:
        os.makedirs(os.path.join(args.output_dir, dt), exist_ok=True)
        config_path = os.path.join(args.output_dir, dt, f"config.json")
        json.dump(vars(args), open(config_path, "w"), indent=2)
    else:
        print("No persistency warning: No output directory provided. Data will not be saved.")

    # 加载预训练的CDM模型
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
    cdm_model.load_state_dict(torch.load(args.from_file, map_location=lambda _s, _: _s))
    cdm_model.to(args.device)
    cdm_model.eval()

    # 实例化知识点映射模型
    z2concept_model = ConceptPredictor(
        input_dim=args.d_model, 
        n_classes=dataset['n_classes'], 
        node_feature_dim=args.node_dim,
        in_channels=args.h_dim,
    )
    z2concept_model.to(args.device)

    # 设置优化器
    optim = torch.optim.AdamW(
        z2concept_model.parameters(), lr=args.learning_rate, weight_decay=args.l2
    )
    # 获取知识点图的节点特征和边
    node_feature_file = os.path.join(DATA_DIR, dataset['node_feature'])
    edge_file = os.path.join(DATA_DIR, dataset['edge'])
    node_features, edge_index, edge_weight = load_graph_data(node_feature_file, edge_file)  # edge weight为None
    node_features = node_features.to(args.device)  # (94, 128)
    edge_index = edge_index.to(args.device)        # (2, 172)

    """执行训练"""
    best = {"accuracy": 0}
    best_epoch = 0
    for epoch in range(1, args.n_epochs + 1):
        print("start epoch", epoch)
        z2concept_model.train()
        it = tqdm(iter(train_data))
        total_loss = 0.0
        total_cnt = 0
        for batch in it:
            # 获取batch数据
            q, s, pid, concept = batch.get("q", "s", "pid")  # q 比较特殊，是长度为 max_len_kp 的 list
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
            _, z, *_ = cdm_model.predict(q, s, pid)

            # # check
            # print("================================================================================")
            # print(q.shape)
            # print(s.shape)
            # print(pid.shape)
            # print(concept.shape)
            # print("================================================================================")
            # raise ValueError("debug")

            """240825: 调整返回值"""
            # if args.type == 'word':  # 词汇作为知识点
            #     concept_embed = cdm_model.concept_map(z, s, n=1)  # (bs, n_c, d_model)
            # else:                    # 时态作为知识点
            #     concept_embed = cdm_model.tense_map(z, s, n=3)
            q_, k_, v_ = cdm_model.concept_map_new(z, s, n=1)  # (bs, n_c, d_model)

            """
            240820: sparse损失函数
            240825: 
            """
            # # loss = z2concept_model.get_focal_loss(concept_embed, concept)
            # loss = z2concept_model.get_semi_sparse_loss(concept_embed, concept)
            loss = z2concept_model.get_semi_sparse_loss(q_, k_, v_,node_features, edge_index, concept,edge_weight)

            # 计算梯度 & 反向传播
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(z2concept_model.parameters(), 1.0)
            optim.step()

            # 可视化
            total_loss += loss.item()
            total_cnt += 1
            postfix = {"loss": total_loss / total_cnt}
            postfix["epoch"] = epoch
            it.set_postfix(postfix)
            wandb.log(postfix)  # 传入想要记录的键值对的字典

        # 当前epoch训练效果评估
        z2concept_model.eval()
        evaluator = Evaluator()
        with torch.no_grad():
            it = tqdm(iter(valid_data))
            for batch in it:
                q, s, pid, concept = batch.get("q", "s", "pid")
                if not isinstance(q, torch.Tensor):
                    q = torch.stack(q, dim=-1)  # 50x(32,36) -> (32,36,50)
                    q = q.to(args.device)
                    s = s[0].to(args.device)
                    pid = pid[0].to(args.device)
                    concept = concept.to(args.device)  # (bs, n_concepts)
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
                # 调用Evaluator.evaluate_know方法，将预报值&真实值处理为适用于sparse损失函数的形式
                concept = concept.cpu().numpy()
                evaluator.evaluate_know(concept, y)

        # 调用Evaluator.report_know方法，计算并保存评估指标
        r = evaluator.report_know()
        print(r)
        wandb.log(r)
        if args.output_dir:
            config_path = os.path.join(args.output_dir, dt, f"report.json")
            with open(config_path, "a") as f:
                f.write(json.dumps(r, indent=4) + "\n")
        if r["accuracy"] > best["accuracy"]:
            best = r
            best_epoch = epoch
            if args.output_dir:
                model_path = os.path.join(
                    args.output_dir, dt, f"concept_model.pt"  # _{epoch:03d}
                )
                print("saving snapshot to:", model_path)
                torch.save(z2concept_model.state_dict(), model_path)

        # 根据验证集评估结果判断是否触发早停
        if 0 < args.early_stop < epoch - best_epoch:
            print(f"did not improve for {args.early_stop} epochs, stop early")
            break


    wandb.finish()
    return best_epoch, best


if __name__ == "__main__":

    args_in = parser.parse_args()
    print(args_in)

    best_epoch_, best_ = main(args_in)
    print(args_in)
