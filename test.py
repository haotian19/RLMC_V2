# -*- coding: utf-8 -*-
"""
test.py — 独立复评：加载 main.py 保存的 checkpoint，在指定 seed 的测试集上评估

用法:
  python test.py --ckpt output_fixed/attn_feature/checkpoint_attn_feature_seed26.pth \
                 --mode attn --count_mode feature --seed 26
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rlmc import ResidualLinealModelClassifier
from utils import binary_metrics
from main import prepare_one_seed


def test(ckpt_path, mode, count_mode, seed, data="clean_data.csv", output_dir="."):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ck = torch.load(str(ckpt_path), map_location=device, weights_only=False)

    # 从 state_dict 推断输入维度
    n_features = ck["state_dict"]["classifier.weight"].shape[1]
    model = ResidualLinealModelClassifier(n_features, 2, mode=mode).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    # 重建该 seed 的测试集（与训练时完全相同的分组划分与预处理）
    class A:  # 复用 prepare_one_seed 所需的最小参数集
        pass
    a = A()
    a.count_mode = count_mode
    df = pd.read_csv(data)
    tensors = prepare_one_seed(df, a, seed)

    X_te, y_te = tensors["X_te"], tensors["y_te"]
    probs = []
    with torch.no_grad():
        for i in range(0, len(X_te), 4096):
            logits = model(X_te[i:i + 4096].to(device))
            probs.append(torch.softmax(logits, dim=1)[:, 1].cpu())
    prob = torch.cat(probs).numpy()

    m = binary_metrics(y_te.numpy(), prob)
    scores = pd.DataFrame([{"ckpt": str(ckpt_path), "seed": seed,
                            **{k: round(v, 4) for k, v in m.items()}}])
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out / f"retest_{Path(ckpt_path).stem}.csv", index=False)
    print(scores.to_string(index=False))
    return scores


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--mode', default='attn', choices=['base', 'conv', 'attn'])
    p.add_argument('--count_mode', default='feature')
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--data', default='clean_data.csv')
    p.add_argument('--output_dir', default='.')
    args = p.parse_args()
    test(args.ckpt, args.mode, args.count_mode, args.seed, args.data, args.output_dir)
