# -*- coding: utf-8 -*-
"""
main_ml.py — 传统机器学习基线（修正版）

相对旧版(main_ml(1).py)的修复:
  [泄漏]  与主脚本一致的分组三路划分 80/10/10
  [方法]  SMOTE -> SMOTENC
  [基线]  SVC(rbf, max_iter=1000 在45万样本上必然欠收敛 -> AUC 0.53) 换成可扩展的线性 SVC；
          新增 LR 作为可解释参照基线
  [统计]  --n_seeds 多种子重复 mean±std + 测试集 bootstrap 95% CI
"""
import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
import lightgbm as lgb
from imblearn.over_sampling import SMOTENC

from utils import binary_metrics

CAT_COLS = ["menopaus", "agegrp", "density", "race", "Hispanic",
            "nrelbc", "brstproc", "surgmeno", "hrt"]
CONT_COLS = ["bmi", "agefirst", "lastmamm"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def grouped_split(X, y, seed):
    """按风险因子组合分组的三路划分 80/10/10"""
    group_cols = [c for c in CAT_COLS + CONT_COLS if c in X.columns]
    groups = X[group_cols].astype(str).agg("|".join, axis=1)
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=seed)
    tr_va_idx, te_idx = next(gss1.split(X, y, groups))
    X_tr_va, y_tr_va = X.iloc[tr_va_idx], y.iloc[tr_va_idx]
    g_tr_va = groups.iloc[tr_va_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=1/9, random_state=seed)
    tr_idx, va_idx = next(gss2.split(X_tr_va, y_tr_va, g_tr_va))
    return (X_tr_va.iloc[tr_idx], y_tr_va.iloc[tr_idx],
            X_tr_va.iloc[va_idx], y_tr_va.iloc[va_idx],
            X.iloc[te_idx], y.iloc[te_idx])


def tune_threshold(y_va, prob_va):
    """验证集上按最大 F1 选阈值"""
    from sklearn.metrics import f1_score
    grid = np.unique(np.quantile(prob_va, np.linspace(0.01, 0.99, 197)))
    f1s = [f1_score(y_va, (prob_va >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(f1s))])


def bootstrap_ci(y_true, y_prob, n_boot=2000, seed=0, threshold=0.5):
    """百分位 bootstrap 95% CI"""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    keys = ["accuracy", "recall", "precision", "f1", "roc_auc"]
    boots = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, yp = y_true[idx], y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        m = binary_metrics(yt, yp, threshold)
        for k in keys:
            boots[k].append(m[k])
    return {k: (float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975)))
            for k, v in boots.items()}


def get_models(seed):
    models = {
        "DT_SMOTE":    DecisionTreeClassifier(random_state=seed),
        "SVC_SMOTE":   LinearSVC(max_iter=5000, random_state=seed),   # 修复: rbf+max_iter=1000 欠收敛
        "LGBM_SMOTE":  lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, max_depth=5,
                                          subsample=0.8, colsample_bytree=0.8,
                                          n_jobs=-1, random_state=seed),
        "LR_SMOTE":    LogisticRegression(max_iter=2000, random_state=seed),   # 新增: 可解释参照
    }
    try:
        from xgboost import XGBClassifier
        models["XGB_SMOTE"] = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=5,
                                            subsample=0.8, colsample_bytree=0.8,
                                            eval_metric="logloss", n_jobs=-1, random_state=seed)
    except ImportError:
        print("警告: 未安装 xgboost, 跳过 XGB_SMOTE")
    return models


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=str, default='clean_data.csv')
    p.add_argument('--n_seeds', type=int, default=3)
    p.add_argument('--seed0', type=int, default=22)
    p.add_argument('--count_mode', type=str, default='feature', choices=['feature', 'drop'])
    p.add_argument('--tune_threshold', action='store_true',
                   help='验证集上按最大F1选阈值(与 main.py 口径一致)')
    p.add_argument('--output_dir', type=str, default='output_ml_fixed')
    args = p.parse_args()

    out = Path(args.output_dir) / f"ml_{args.count_mode}"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "config.json", "w") as f:
        json.dump(vars(args), f, indent=4)

    df = pd.read_csv(args.data)
    X_all = df.drop(columns=["invasive", "cancer"])
    if args.count_mode == "drop":
        X_all = X_all.drop(columns=["count"])
    y_all = df["cancer"].astype(int)
    cat_idx = [X_all.columns.get_loc(c) for c in CAT_COLS if c in X_all.columns]

    rows, last_pred = [], {}
    for i in range(args.n_seeds):
        seed = args.seed0 + i
        set_seed(seed)
        X_tr, y_tr, X_va, y_va, X_te, y_te = grouped_split(X_all, y_all, seed)
        scaler = StandardScaler().fit(X_tr)
        X_tr_s, X_va_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_va), scaler.transform(X_te)
        sm = SMOTENC(categorical_features=cat_idx, k_neighbors=5, random_state=seed)
        X_tr_r, y_tr_r = sm.fit_resample(X_tr_s, y_tr)

        for name, model in get_models(seed).items():
            t0 = time.time()
            model.fit(X_tr_r, y_tr_r)
            if hasattr(model, "predict_proba"):
                prob_va = model.predict_proba(X_va_s)[:, 1]
                prob = model.predict_proba(X_te_s)[:, 1]
            else:  # LinearSVC
                d_va = model.decision_function(X_va_s)
                prob_va = 1 / (1 + np.exp(-d_va))
                d = model.decision_function(X_te_s)
                prob = 1 / (1 + np.exp(-d))
            thr = 0.5
            if args.tune_threshold:
                thr = tune_threshold(y_va.to_numpy(), prob_va)
            m = binary_metrics(y_te.to_numpy(), prob, threshold=thr)
            m.update({"seed": seed, "model": name, "threshold": thr,
                      "minutes": (time.time() - t0) / 60})
            rows.append(m)
            print(f"seed{seed} {name:10s} acc={m['accuracy']:.4f} rec={m['recall']:.4f} "
                  f"prec={m['precision']:.4f} f1={m['f1']:.4f} auc={m['roc_auc']:.4f}")
            if i == args.n_seeds - 1:
                last_pred[name] = (y_te.to_numpy(), prob)

    res = pd.DataFrame(rows)
    res.to_csv(out / "per_seed_scores.csv", index=False)

    # ---- 逐样本预测留档 + 曲线坐标(小文件, 进Excel) ----
    pred_rows = []
    curve_rows = []
    for name, (yt, yp) in last_pred.items():
        for t, p in zip(yt, yp):
            pred_rows.append((name, int(t), float(p)))
        from utils import export_curve_points
        curve_rows.append(export_curve_points(yt, yp).assign(model=name))
    pd.DataFrame(pred_rows, columns=["model", "y_true", "y_prob"]).to_csv(
        out / "test_pred_ml_last_seed.csv", index=False)   # 原始预测(留档, 不进Excel)
    pd.concat(curve_rows, ignore_index=True).to_csv(
        out / "roc_points_ml_last_seed.csv", index=False)  # 图3曲线坐标
    print("曲线坐标已保存: roc_points_ml_last_seed.csv")
    summary = res.groupby("model")[["accuracy", "recall", "precision", "f1", "roc_auc"]]\
                 .agg(["mean", "std"])
    summary.to_csv(out / "summary_mean_std.csv")
    summary_tidy = summary.stack(level=0).reset_index()  # tidy 长表: model, metric, mean, std
    summary_tidy.columns = ["model", "metric", "mean", "std"]
    summary_tidy.to_csv(out / "summary_tidy.csv", index=False)
    print("\n===== 跨 seed 汇总 (mean ± std) =====")
    print(summary.round(4))

    # bootstrap 95% CI（基于最后一个 seed 的测试集预测）
    print("\n===== bootstrap 95% CI (最后一个 seed) =====")
    ci_rows = []
    for name, (yt, yp) in last_pred.items():
        ci = bootstrap_ci(yt, yp, n_boot=2000, seed=args.seed0 + args.n_seeds - 1)
        row = {"model": name}
        for k, (lo, hi) in ci.items():
            row[f"{k}_lo"], row[f"{k}_hi"] = lo, hi
        ci_rows.append(row)
        print(f"{name:10s} " + "  ".join(f"{k}=[{ci[k][0]:.4f},{ci[k][1]:.4f}]" for k in ci))
    pd.DataFrame(ci_rows).to_csv(out / "bootstrap_ci_last_seed.csv", index=False)


if __name__ == "__main__":
    main()
