import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import time
import torch
import tomlkit
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import torch.nn.functional as F
from argparse import ArgumentParser
from models.data import KTData
from models.eval import Evaluator
from models.CDMTransformer import CDMTransformer

load_dotenv()
DATA_DIR = os.getenv('DATA_DIR')

parser = ArgumentParser()
# general options
parser.add_argument("--device", help="device to run network on", default="cpu")
parser.add_argument("-bs", "--batch_size", help="batch size", default=8, type=int)

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
parser.add_argument(
    "--n_know", help="dimension of knowledge parameter", type=int, default=32
)
parser.add_argument("--proj", help="projection layer before CL", action="store_true")
# test setup
parser.add_argument("-f", "--from_file", help="test existing model file", required=True)
parser.add_argument("-N", help="T+N prediction window size", type=int, default=1)

def simple_moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def sliding_window_analysis(data, window_size):
    means = []
    stds = []
    for i in range(len(data) - window_size + 1):
        window = data[i:i+window_size]
        means.append(np.mean(window))
        stds.append(np.std(window))
    return means, stds


def moving_average(data, window_size):
    """
    对数据进行移动平均平滑处理。

    参数:
    data: 输入数据，形状为 (batch_size, seq_len)
    window_size: 移动窗口大小

    返回:
    平滑处理后的数据，形状与输入相同
    """
    smoothed_data = np.zeros_like(data)

    for i in range(data.shape[0]):
        smoothed_data[i] = pd.Series(data[i]).rolling(window=window_size, min_periods=1).mean().to_numpy()

    return smoothed_data

def main(args):
    # 根据dataset获取toml文件中对应信息
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
        batch_size=args.batch_size,
        # shuffle=True,
        know=False,
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

    evaluator = Evaluator()

    with torch.no_grad():
        it = tqdm(iter(test_data))
        for index,batch in enumerate(it):
            q, s, pid = batch.get("q", "s", "pid")
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

            # (n_know, seq_len) (seq_len, n_know*d_model)
            # y,h = model.tracing(q,s,pid)
            # n_know, seq_len, d_model = h.size()
            #
            # # 定义前 n 个时刻
            # n = 1
            # # 计算每个时刻与前 n 个时刻的 MSE
            # mse_values = np.zeros((n_know, seq_len))
            # for i in range(n_know):
            #     for j in range(n, seq_len):
            #         mse_values[i,j] = torch.mean((h[i,j] - h[i,j - n]) ** 2)
            #         # mse_values[i, j] = torch.mean(h[i, j])
            # # 绘制每个学生 MSE 随时间变化的图表
            # # for i in range(n_know):
            # #     plt.plot(range(seq_len), mse_values[i], label=f'Student {i+1}')
            # plt.plot(range(seq_len), (mse_values[0]+mse_values[i])/2, label=f'Student {1}')
            # plt.xlabel('Sequence (Question) Index')
            # plt.ylabel('MSE')
            # plt.title('Student Ability Change Over Time (MSE)')
            # plt.legend()
            # plt.show()

            # n_know, seq_len = y.size()
            # y_np = y.detach().cpu().numpy()
            # for i in range(n_know):
            #     plt.plot(range(seq_len), y_np[i], label=f'Knowledge Point {i + 1}')
            # plt.title('Knowledge Points Trend Across Questions')
            # plt.xlabel('Question Number')
            # plt.ylabel('Value')
            # plt.legend()
            # plt.show()


            y, z, *_ = model.predict(q, s, pid, args.N)
            # evaluator.evaluate(s[:, (args.N - 1):], torch.sigmoid(y))
            batch_size, seq_len, dim = z.size()
            mse_values = np.zeros((batch_size, seq_len))
            # 定义前 n 个时刻
            n = 3
            # 计算每个时刻与前 n 个时刻的 MSE
            for i in range(batch_size):
                for j in range(n, seq_len):
                    mse_values[i, j] = torch.mean((z[i, j] - z[i, j - n]) )

            window_size = 5  # 可以调整窗口大小

            # 计算滑动窗口标准差
            size = seq_len - window_size + 1
            moving_std = np.zeros((batch_size, size, dim))
            for i in range(size):
                # 提取当前窗口的数据
                window = z[:, i:i + window_size, :]
                # 计算当前窗口内的标准差
                std = np.std(window.cpu().numpy(), axis=1)
                moving_std[:, i] = std
            # 计算每个时间点上所有特征维度的标准差均值
            mean_std = np.mean(moving_std, axis=2)
            # 绘制每个学生 mean_std 随时间变化的图表
            for i in range(mean_std.shape[0]):
                plt.plot(range(len(mean_std[0])), mean_std[i], label=f'Student {index + i + 1}')
            plt.xlabel('Sequence (Question) Index')
            plt.ylabel('Standard Deviation')
            plt.title('Ability Variation Over Time')
            plt.legend()
            plt.show()

            # 计算移动平均
            smoothed_mse_values =moving_average(mse_values, window_size)

            # 计算每个位置的二元交叉熵损失
            loss_per_position = F.binary_cross_entropy(torch.sigmoid(y), s.float(), reduction='none').cpu()  # (batch_size, seq_len)

            # 绘制每个学生 MSE 随时间变化的图表
            for i in range(batch_size):
                # plt.plot(range(seq_len), mse_values[i], label=f'Student {i + 1}')
                plt.plot(range(seq_len), smoothed_mse_values[i], label=f'Student {index + i + 1}')
            plt.xlabel('Sequence (Question) Index')
            plt.ylabel('MSE')
            plt.title('Student Ability Change Over Time (MSE)')
            plt.legend()
            plt.show()




            # for i in range(batch_size):
            #     plt.plot(range(seq_len), loss_per_position[i], label=f'Student {index + 1}')
            # plt.xlabel('Sequence (Question) Index')
            # plt.ylabel('loss')
            # plt.title('Student Predict Loss Over Time ')
            # plt.legend()
            # plt.show()

            # time.sleep(1)

            # average_ability = torch.mean(z, dim=2).detach().numpy()
            # for i in range(batch_size):
            #     plt.plot(average_ability[i], label=f'Student {i + 1}')
            #
            # # plt.xlabel('Sequence Length')
            # # plt.ylabel('Average Ability')
            # # plt.title('Student Ability Over Time')
            # # plt.legend()
            # # plt.show()
            #
            # window_size = 5  # 可以调整窗口大小
            # # smoothed_ability = [simple_moving_average(average_ability[i], window_size) for i in range(batch_size)]
            # # for i in range(batch_size):
            # #     plt.plot(smoothed_ability[i], label=f'Student {i + 1}')
            # #
            # # plt.xlabel('Sequence Length')
            # # plt.ylabel('Smoothed Average Ability')
            # # plt.title('Smoothed Student Ability Over Time')
            # # plt.legend()
            # # plt.show()
            # for i in range(batch_size):
            #     means, stds = sliding_window_analysis(average_ability[i], window_size)
            # plt.plot(range(len(means)), means, label='Mean')
            # plt.fill_between(range(len(means)), np.array(means) - np.array(stds), np.array(means) + np.array(stds),
            #                  alpha=0.2)
            # plt.xlabel('Question Number')
            # plt.ylabel('Ability Value')
            # plt.title('Student Ability Value Changes Over Questions (Sliding Window)')
            # plt.legend()
            # plt.show()


    # output_path = os.path.dirname(args.from_file)
    # result_path = os.path.join(output_path, f"test_result.json")
    # output = {"args": vars(args), "metrics": {}}
    # with open(result_path, "a") as f:
    #     output["metrics"][args.N] = evaluator.report()
    #     f.write(json.dumps(output, indent=2) + "\n")

if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    main(args)