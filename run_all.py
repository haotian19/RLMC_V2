# -*- coding: utf-8 -*-
"""
run_all.py — 一键跑完论文所需的全部实验并汇总

默认流程 (全部使用 --n_seeds 3):
  1) preprocess.py            (若 clean_data.csv 不存在)
  2) main.py  attn/base/conv  (count_mode=feature; attn 附带 --run_attn_viz --run_shap)
  3) count 消融: attn×{drop,weight} + main_ml×{feature,drop}
  4) main_ml.py 基线 (feature)
  5) merge_results.py → paper_data.xlsx

用法:
  python run_all.py                    # 完整流程
  python run_all.py --skip_ablation    # 跳过 count 消融(省一半时间)
  python run_all.py --skip_shap        # 跳过 SHAP(最耗时的解释计算)
  python run_all.py --quick            # 冒烟自检: 1 epoch 小参数, 验证流程
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def run(cmd, desc):
    print("\n" + "=" * 70)
    print(f"[{time.strftime('%H:%M:%S')}] {desc}")
    print("  $ " + " ".join(cmd))
    print("=" * 70, flush=True)
    subprocess.run([sys.executable] + cmd, cwd=HERE, check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n_seeds', type=int, default=3)
    p.add_argument('--modes', type=str, default='attn base conv',
                   help='空格分隔, 只跑指定模式, 如 --modes "attn base" 跳过conv')
    p.add_argument('--resume', action='store_true',
                   help='断点续跑: 每个seed从已有checkpoint的epoch继续')
    p.add_argument('--num_epochs', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=512)
    p.add_argument('--output_dir', type=str, default='output_fixed')
    p.add_argument('--skip_ablation', action='store_true', help='跳过 count 消融')
    p.add_argument('--skip_shap', action='store_true', help='跳过 SHAP 计算')
    p.add_argument('--quick', action='store_true', help='冒烟自检(1 epoch)')
    p.add_argument('--tune_threshold', action='store_true',
                   help='验证集选阈值(默认固定0.5; 开启后所有深度/ML模型统一口径)')
    args = p.parse_args()

    if args.quick:
        args.num_epochs, args.batch_size = 1, 256

    seeds, ep, bs, out = args.n_seeds, args.num_epochs, args.batch_size, args.output_dir

    # 1. 预处理
    if not os.path.exists(os.path.join(HERE, "clean_data.csv")):
        run(["preprocess.py"], "步骤1/5 数据预处理 risk.txt → clean_data.csv")
    else:
        print("clean_data.csv 已存在, 跳过预处理")

    # 2. RLMC v2 三模式（可用 --modes 选择/跳过）
    for mode in args.modes.split():
        cmd = ["main.py", "--mode", mode, "--n_seeds", str(seeds),
               "--num_epochs", str(ep), "--batch_size", str(bs),
               "--count_mode", "feature", "--output_dir", out]
        if args.tune_threshold:
            cmd.append("--tune_threshold")
        if args.resume:
            cmd.append("--resume")
        if mode == "attn":
            cmd.append("--run_attn_viz")
            if not args.skip_shap:
                cmd.append("--run_shap")
        run(cmd, f"步骤2/5 RLMC v2 [{mode}]  ({seeds} seeds × ≤{ep} epochs)")

    # 3. count 消融
    if not args.skip_ablation:
        run((["main.py", "--mode", "attn", "--n_seeds", str(seeds),
              "--num_epochs", str(ep), "--batch_size", str(bs),
              "--count_mode", "drop", "--output_dir", out]
             + (["--tune_threshold"] if args.tune_threshold else [])
             + (["--resume"] if args.resume else [])),
            "步骤3/5 count消融 [attn + count=drop]")
        run(["main.py", "--mode", "attn", "--n_seeds", str(seeds),
             "--num_epochs", str(ep), "--batch_size", str(bs),
             "--count_mode", "weight", "--output_dir", out]
            + (["--tune_threshold"] if args.tune_threshold else [])
            + (["--resume"] if args.resume else []),
            "步骤3/5 count消融 [attn + count=weight]")
        run(["main_ml.py", "--n_seeds", str(seeds), "--count_mode", "drop",
             "--output_dir", out] + (["--tune_threshold"] if args.tune_threshold else []),
            "步骤3/5 count消融 [ML + count=drop]")

    # 4. ML 基线
    run(["main_ml.py", "--n_seeds", str(seeds), "--count_mode", "feature",
         "--output_dir", out] + (["--tune_threshold"] if args.tune_threshold else []),
        f"步骤4/5 ML 基线 ({seeds} seeds)")

    # 5. 汇总
    run(["merge_results.py", "--root", out, "--out", "paper_data.xlsx"],
        "步骤5/5 汇总 → paper_data.xlsx")

    print("\n" + "=" * 70)
    print(f"全部完成。下一步: Rscript plots.R paper_data.xlsx  (预览图件)")
    print("=" * 70)


if __name__ == "__main__":
    main()
