# -*- coding: utf-8 -*-
"""
utils.py — 早停（修复版）、评估指标、注意力矩阵导出（供 R 绘图，不再直接出图）
"""
import datetime
import sys

import numpy as np
import pandas as pd


class EarlyStopping:
    """监控验证损失，连续 patience 次无改善(min_delta)则触发。计数器跨 epoch 持久化。"""

    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss):
        if self.best is None:
            self.best = val_loss
            return False
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.counter = 0
        else:
            self.counter += 1
            print(f"早停计数: 连续 {self.counter}/{self.patience} 次验证损失未改善")
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class Logger:
    """终端输出同步写日志文件"""

    def __init__(self, filename, stream=None, time_format="%Y-%m-%d %H:%M:%S"):
        self.terminal = stream if stream else sys.stdout
        self.log = open(filename, 'a', encoding='utf-8')
        self.time_format = time_format
        self.previous_msg = None
        sys.stdout = self

    def write(self, message):
        if self.previous_msg is None or "\n" in self.previous_msg:
            timestamp = datetime.datetime.now().strftime(self.time_format) + " : "
            self.terminal.write(timestamp)
            self.log.write(timestamp)
        self.previous_msg = message if isinstance(message, str) else ""
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass

    def close(self):
        if not self.log.closed:
            self.log.close()


def binary_metrics(y_true, y_prob, threshold=0.5):
    """从概率计算全部指标"""
    from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                                 f1_score, roc_auc_score, confusion_matrix)
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc = roc_auc_score(y_true, y_prob)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"accuracy": acc, "recall": rec, "precision": prec, "f1": f1,
            "roc_auc": roc, "specificity": tn / (tn + fp) if tn + fp else np.nan}


def export_curve_points(y_true, y_prob, n_points=200):
    """导出 ROC/PR 曲线坐标（n_points 个分位阈值），替代逐样本预测进入 Excel，文件缩小约两个数量级"""
    from sklearn.metrics import roc_curve
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    ths = np.unique(np.quantile(y_prob, np.linspace(0, 1, n_points + 1))[1:-1])

    fpr, tpr, thr = roc_curve(y_true, y_prob)        # thr 降序
    o = np.argsort(thr)
    fpr_i = np.interp(ths, thr[o], fpr[o])
    tpr_i = np.interp(ths, thr[o], tpr[o])

    P = (y_prob[None, :] >= ths[:, None])            # (G, N) 布尔
    tp = (P & (y_true[None, :] == 1)).sum(axis=1)
    fp = (P & (y_true[None, :] == 0)).sum(axis=1)
    prec = np.divide(tp, tp + fp, out=np.ones_like(tp, dtype=float), where=(tp + fp) > 0)
    rec = tp / max(1, int((y_true == 1).sum()))

    return pd.DataFrame({"threshold": ths, "fpr": fpr_i, "tpr": tpr_i,
                         "recall": rec, "precision": prec})


def export_attention_long(attn_maps, feature_names, csv_path):
    """将各层注意力矩阵导出为长表 CSV（供 R/ggplot 画热图）。

    列: layer, row_label, col_label, weight
    仅当该层维度 == 特征数时给出特征名（第1/5层），中间层为扩展通道（ch0..chN）。
    """
    n_feat = len(feature_names)
    rows = []
    for li, attn in enumerate(attn_maps):
        a = attn.detach().cpu().numpy().mean(axis=0)   # (F_l, F_l) 对样本取均值
        d = a.shape[0]

        def lab(i):
            return feature_names[i] if (d == n_feat and i < n_feat) else f"ch{i}"

        for i in range(d):
            for j in range(d):
                rows.append((li + 1, lab(i), lab(j), float(a[i, j])))
    df = pd.DataFrame(rows, columns=["layer", "row_label", "col_label", "weight"])
    df.to_csv(csv_path, index=False)
    print(f"Attention long table saved: {csv_path}")
    return df
