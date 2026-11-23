# -*- coding: utf-8 -*-
"""
evaluate.py — SHAP 计算
"""
import os
import numpy as np


def compute_shap(model, X_train, X_test, device=None,
                 n_background=20, n_explain=50):
    """计算 KernelSHAP，返回 (shap_values, base_value, xtest, background)"""
    import shap
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)

    import torch
    def predict_fn(X):
        X_t = torch.tensor(np.asarray(X), dtype=torch.float32).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(X_t), dim=1).cpu().numpy()
        return probs[:, 1]

    rng = np.random.default_rng(42)
    background = X_train[rng.choice(X_train.shape[0], min(n_background, X_train.shape[0]), replace=False)]
    xtest = X_test[rng.choice(X_test.shape[0], min(n_explain, X_test.shape[0]), replace=False)]
    print(f"SHAP background: {background.shape}, explained: {xtest.shape}")

    explainer = shap.KernelExplainer(predict_fn, background)
    shap_values = np.asarray(explainer.shap_values(xtest))
    if shap_values.ndim == 3:
        shap_values = shap_values[..., -1]

    importance = np.abs(shap_values).mean(axis=0)
    print("\nMean |SHAP| importance:")
    for i in np.argsort(-importance):
        print(f"  feature{i}  {importance[i]:.5f}")
    return shap_values, explainer.expected_value, xtest, background
