# -*- coding: utf-8 -*-
"""
merge_results.py — 把 main.py / main_ml.py 各次运行的输出汇总成一个 Excel 文件

用法:
  python merge_results.py --root output_fixed --out paper_data.xlsx

Excel 工作表（与论文图表对应）:
  Table2_deep_per_seed   深度模型逐 seed 测试指标（Table 2 上半素材）
  Table2_deep_summary    深度模型 mean±std（长表, tidyr 友好）
  Table2_ml_per_seed / Table2_ml_summary / Table2_ml_CI   基线部分
  Fig2_training_curves   训练/验证损失曲线数据（图2）
  Fig3_roc_curves        ROC/PR曲线坐标（图3, 每模型约200点, 体积小约400倍）
  Fig4_attention         注意力矩阵长表（图4）
  Fig5_shap              SHAP tidy 数据（图5）+ Fig5_shap_importance
  configs                各次运行的完整配置
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def collect(root: Path, include_predictions: bool = False):
    deep_per, deep_sum, curves, preds, attn, shap_tidy, shap_imp, configs =         [], [], [], [], [], [], [], []
    ml_per, ml_sum, ml_ci = [], [], []

    for run in sorted(root.rglob("config.json")):
        rdir = run.parent
        cfg = json.load(open(run))
        # 识别运行类型: 目录名形如 {mode}_{count_mode} 或 ml_{count_mode}
        tag = rdir.name
        parts = tag.split("_")
        cfg_row = {"run_dir": str(rdir), "tag": tag, **{k: str(v) for k, v in cfg.items()}}
        configs.append(pd.DataFrame([cfg_row]))

        f = rdir / "per_seed_scores.csv"
        if f.exists():
            df = pd.read_csv(f)
            if tag.startswith("ml_"):
                ml_per.append(df.assign(count_mode=parts[1]))
            else:
                df = df.assign(mode=parts[0], count_mode=parts[1] if len(parts) > 1 else "?")
                deep_per.append(df)

        f = rdir / "summary_tidy.csv"
        if f.exists():                                   # ML: tidy 长表
            ml_sum.append(pd.read_csv(f).assign(count_mode=parts[1]))
        f = rdir / "summary_mean_std.csv"
        if f.exists() and not tag.startswith("ml_"):     # deep: index=metric
            s = pd.read_csv(f, index_col=0)
            s.index.name = "metric"
            s = s.reset_index()
            deep_sum.append(s.assign(mode=parts[0], count_mode=parts[1] if len(parts) > 1 else "?"))

        f = rdir / "bootstrap_ci_last_seed.csv"
        if f.exists():
            ml_ci.append(pd.read_csv(f).assign(count_mode=parts[1]))

        f = rdir / "training_curves.csv"
        if f.exists():
            curves.append(pd.read_csv(f).assign(mode=parts[0],
                                                count_mode=parts[1] if len(parts) > 1 else "?"))
        for f in sorted(rdir.glob("roc_points_*.csv")):  # 曲线坐标(小, 默认并入)
            d = pd.read_csv(f)
            if tag.startswith("ml_"):
                preds.append(d.assign(count_mode=parts[1] if len(parts) > 1 else "?",
                                      source="ml", sheet_kind="roc_points"))
            else:
                d = d.assign(mode=parts[0], count_mode=parts[1] if len(parts) > 1 else "?",
                             source="deep", sheet_kind="roc_points")
                preds.append(d)
        if include_predictions:                                   # 可选: 原始预测(大)
            for f in sorted(rdir.glob("test_pred_*.csv")):
                d = pd.read_csv(f)
                if tag.startswith("ml_"):
                    preds.append(d.assign(source="ml", sheet_kind="raw"))
                else:
                    d = d.assign(mode=parts[0], count_mode=parts[1] if len(parts) > 1 else "?",
                                 source="deep", sheet_kind="raw")
                    preds.append(d)

        for f in sorted(rdir.glob("attention_long_*.csv")):
            attn.append(pd.read_csv(f).assign(mode=parts[0]))
        for f in sorted(rdir.glob("shap_tidy_*.csv")):
            shap_tidy.append(pd.read_csv(f).assign(mode=parts[0],
                                                   count_mode=parts[1] if len(parts) > 1 else "?"))
        for f in sorted(rdir.glob("shap_importance_*.csv")):
            shap_imp.append(pd.read_csv(f).assign(mode=parts[0],
                                                  count_mode=parts[1] if len(parts) > 1 else "?"))
    return dict(deep_per=deep_per, deep_sum=deep_sum, ml_per=ml_per, ml_sum=ml_sum, ml_ci=ml_ci,
                curves=curves, preds=preds, attn=attn, shap_tidy=shap_tidy,
                shap_imp=shap_imp, configs=configs)


def drop_apei(df):
    """丢弃旧输出中残留的 APEI 列/行（稿件已删除该指标）"""
    if df is None:
        return df
    df = df.drop(columns=[c for c in df.columns if "apei" in str(c).lower()], errors="ignore")
    if "metric" in df.columns:
        df = df[~df["metric"].astype(str).str.lower().eq("apei")]
    return df


def concat_or_none(lst):
    lst = [drop_apei(d) for d in lst]
    lst = [d for d in lst if d is not None and len(d)]
    return pd.concat(lst, ignore_index=True) if lst else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=str, default="output_fixed",help='输出根目录(含各次运行子目录)')
    ap.add_argument('--out', type=str, default='paper_data.xlsx')
    ap.add_argument('--include_predictions', action='store_true',
                    help='把逐样本预测也并入Excel(文件会很大, 默认只并曲线坐标)')
    args = ap.parse_args()

    d = collect(Path(args.root), include_predictions=args.include_predictions)
    sheets = {
        "Table2_deep_per_seed": concat_or_none(d["deep_per"]),
        "Table2_deep_summary": concat_or_none(d["deep_sum"]),
        "Table2_ml_per_seed": concat_or_none(d["ml_per"]),
        "Table2_ml_summary": concat_or_none(d["ml_sum"]),
        "Table2_ml_CI": concat_or_none(d["ml_ci"]),
        "Fig2_training_curves": concat_or_none(d["curves"]),
        "Fig3_roc_curves": concat_or_none(d["preds"]),
        "Fig4_attention": concat_or_none(d["attn"]),
        "Fig5_shap": concat_or_none(d["shap_tidy"]),
        "Fig5_shap_importance": concat_or_none(d["shap_imp"]),
        "configs": concat_or_none(d["configs"]),
    }
    with pd.ExcelWriter(args.out, engine="openpyxl") as w:
        for name, df in sheets.items():
            if df is not None:
                df.to_excel(w, sheet_name=name[:31], index=False)
                print(f"  sheet [{name[:31]}]  {df.shape}")
            else:
                print(f"  sheet [{name[:31]}]  (无数据, 跳过)")
    print(f"\n已生成: {args.out}")


if __name__ == "__main__":
    main()
