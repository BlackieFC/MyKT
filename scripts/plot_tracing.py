import os
from argparse import ArgumentParser
from dotenv import load_dotenv

import torch
import tomlkit
import matplotlib.pyplot as plt

from models.data import KTData
from models.eval import Evaluator
from models.visualize import trace_map
from models.CDMTransformer import CDMTransformer

load_dotenv()  # 加载 .env 文件中的环境变量
DATA_DIR = os.getenv('DATA_DIR')

parser = ArgumentParser()
# general options
parser.add_argument("--device", help="device to run network on", default="cpu")
parser.add_argument("-s", "--seq_id", help="select a sequence index", default=0, type=int)

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
parser.add_argument("--d_model", help="model hidden size", type=int, default=128)
parser.add_argument("--n_layers", help="number of layers", type=int, default=3)
parser.add_argument("--n_heads", help="number of heads", type=int, default=8)
parser.add_argument("--n_know", help="dimension of knowledge parameter", type=int, default=32)
parser.add_argument("--proj", help="projection layer before CL", action="store_true")

# test setup
parser.add_argument("-f", "--from_file", help="test existing model file", required=True)

def main(args):
    dataset = datasets[args.dataset]
    seq_len = dataset["seq_len"] if "seq_len" in dataset else None

    # 新增知识点长度参数，同时将其转换为嵌套列表
    kp_len = dataset["kp_len"] if "kp_len" in dataset else None
    if kp_len is not None:
        kp_len = list(kp_len)
        for ind, temp in enumerate(kp_len):
            if isinstance(temp, int):
                continue
            else:
                kp_len[ind] = list(kp_len[ind])

    test_data = KTData(
        os.path.join(DATA_DIR, dataset["test"]),
        dataset["inputs"],
        dataset['n_questions'],
        group=dataset['n_features'],
        seq_len=seq_len,
        kp_len=kp_len,
        shuffle=True,
    )

    model = CDMTransformer(
        dataset["n_questions"],
        dataset["n_pid"],
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_attn_heads=args.n_heads,
        n_know=args.n_know,
        proj=args.proj,
    )

    model.load_state_dict(torch.load(args.from_file, map_location=lambda s, _: s))
    model.to(args.device)
    model.eval()

    # get one sequence
    data = test_data[args.seq_id]
    q, s, pid = data.get("q", "s", "pid")
    # q比较特殊，是长度为 max_len_kp 的 list
    if not isinstance(q, torch.Tensor):
        q = torch.stack(q, dim=-1)  # 50x(32,36) -> (32,36,50)
        q = q.to(args.device)
        s = s[0].to(args.device)
        pid = pid[0].to(args.device)
    else:
        q = q.to(args.device)
        s = s.to(args.device)
        pid = pid.to(args.device)
    y = model.tracing(q, s, pid).cpu()

    # knowledge tracing on a specific knowledge set
    ind_k = [0, 1, 3, 5, 6]
    span = range(0, 8)
    fig = trace_map(y[ind_k, :], q, s, span, text_label=True)

    plt.show()
    fig.savefig('tracing.pdf', bbox_inches='tight')

if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    main(args)







