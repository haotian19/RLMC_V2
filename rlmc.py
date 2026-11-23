# -*- coding: utf-8 -*-
"""
rlmc.py — RLMC v2 模型定义
Residual Linear Block (BN + 双线性层 + ReLU + Dropout + 残差 + He初始化)
Conv1D Block (Conv1d + BN + ReLU) / Attention Block (单头逐特征自注意力)
三种模式: base / conv / attn
"""
import math
import torch
from torch import nn
import torch.nn.functional as F


class ResidualLinealModelBlocker(nn.Module):
    """双线性层 + 残差连接 + Dropout（BatchNorm + ReLU + He 初始化）"""

    def __init__(self, in_features, out_features, dropout=0.3):
        super().__init__()
        self.norm1 = nn.BatchNorm1d(in_features)
        self.fc1 = nn.Linear(in_features, 128)
        self.act1 = nn.ReLU()

        self.norm2 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, in_features)
        self.act2 = nn.ReLU()

        self.dropout = nn.Dropout(p=dropout)
        self.out = nn.Linear(in_features, out_features)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.norm2(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.dropout(x)
        x = x + residual          # (B, F_in)
        x = self.out(x)           # (B, F_out)
        return x


class Conv1dBlocker(nn.Module):
    """1D 卷积块（注意：特征位于通道维，序列长度恒为 1，属逐点特征变换，无邻域混合）"""

    def __init__(self, in_features):
        super().__init__()
        self.conv1 = nn.Conv1d(in_features, in_features, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_features, in_features, kernel_size=3, padding=1)
        self.act = nn.ReLU()
        self.norm = nn.BatchNorm1d(in_features)

    def forward(self, x):
        x = x.unsqueeze(2)   # (B, F, 1)
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm(x)
        x = self.act(x)
        x = x.squeeze(2)     # (B, F)
        return x


class Attention1dBlocker(nn.Module):
    """单头自注意力：每个特征视为一个 token（Q/K/V 由标量特征线性投影），输出注意力矩阵"""

    def __init__(self, in_channels, emb_dim=64):
        super().__init__()
        self.emb_dim = emb_dim
        self.Q = nn.Linear(1, emb_dim)
        self.K = nn.Linear(1, emb_dim)
        self.V = nn.Linear(1, emb_dim)
        self.out = nn.Linear(emb_dim, 1)
        self.norm = nn.BatchNorm1d(in_channels)

    def forward(self, x):
        residual = x
        b, f = x.shape
        x = x.view(b, f, 1)
        q = self.Q(x)
        k = self.K(x)
        v = self.V(x)
        score = q @ k.transpose(1, 2) / math.sqrt(self.emb_dim)
        attn = F.softmax(score, dim=-1)           # (B, F, F)
        context = attn @ v
        out = self.out(context).view(b, f)
        out = self.norm(out + residual)
        return out, attn


class ResidualLinealModelClassifier(nn.Module):
    """RLMC v2 主模型，支持 base / conv / attn 三种模式"""

    def __init__(self, in_features, num_classes, mode="base"):
        super().__init__()
        assert mode in ["base", "conv", "attn"], f"不支持的模式: {mode}"
        self.mode = mode

        self.cb1 = Conv1dBlocker(in_features)
        self.cb2 = Conv1dBlocker(2 * in_features)
        self.cb3 = Conv1dBlocker(4 * in_features)
        self.cb4 = Conv1dBlocker(2 * in_features)

        self.rlmb1 = ResidualLinealModelBlocker(in_features, 2 * in_features)
        self.rlmb2 = ResidualLinealModelBlocker(2 * in_features, 4 * in_features)
        self.rlmb3 = ResidualLinealModelBlocker(4 * in_features, 2 * in_features)
        self.rlmb4 = ResidualLinealModelBlocker(2 * in_features, in_features)

        self.attn1 = Attention1dBlocker(in_features)
        self.attn2 = Attention1dBlocker(in_features * 2)
        self.attn3 = Attention1dBlocker(in_features * 4)
        self.attn4 = Attention1dBlocker(in_features * 2)
        self.attn5 = Attention1dBlocker(in_features)

        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x, return_attn=False):
        attn_maps = []

        if self.mode == "base":
            x = self.rlmb1(x)   # (B, 2F)
            x = self.rlmb2(x)   # (B, 4F)
            x = self.rlmb3(x)   # (B, 2F)
            x = self.rlmb4(x)   # (B, F)
            out = self.classifier(x)

        elif self.mode == "conv":
            x = self.cb1(x)
            x = self.rlmb1(x)
            x = self.cb2(x)
            x = self.rlmb2(x)
            x = self.cb3(x)
            x = self.rlmb3(x)
            x = self.cb4(x)
            x = self.rlmb4(x)
            out = self.classifier(x)

        elif self.mode == "attn":
            x, a1 = self.attn1(x); attn_maps.append(a1.detach().cpu())
            x = self.rlmb1(x)
            x, a2 = self.attn2(x); attn_maps.append(a2.detach().cpu())
            x = self.rlmb2(x)
            x, a3 = self.attn3(x); attn_maps.append(a3.detach().cpu())
            x = self.rlmb3(x)
            x, a4 = self.attn4(x); attn_maps.append(a4.detach().cpu())
            x = self.rlmb4(x)
            x, a5 = self.attn5(x); attn_maps.append(a5.detach().cpu())
            out = self.classifier(x)

        return (out, attn_maps) if return_attn else out
