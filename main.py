# ============================== main.py ==============================
import argparse
import os
import sys
import datetime
import random
import time
from pathlib import Path
import json 
import torch.nn.functional as F 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score, roc_auc_score 


# 内部模块
from breast_cancer2.RLMC import ResidualLinealModelClassifier
from breast_cancer2.utils import Logger, get_early_stop_signal, visualize_attention
from breast_cancer2.test import test
from breast_cancer2.evaluate import evaluate


# ============================================================
# 🔧 工具函数
# ============================================================
def set_seed(seed: int = 22, deterministic: bool = False):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


# ============================================================
# ⚙️ 参数解析
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='clean_data.csv')
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--warmup_epochs', type=int, default=5)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--mode', type=str, default='attn', choices=['base', 'conv', 'attn'])
parser.add_argument('--resume', action='store_true'
                    , help='若存在 checkpoint 则继续训练')
parser.add_argument('--seed', type=int, default=22)
parser.add_argument('--output_dir', type=str, default='output')
parser.add_argument('--use_amp', action='store_true')
parser.add_argument('--deterministic', action='store_true')
parser.add_argument('--grad_clip', type=float, default=1.0, help='梯度裁剪的最大范数') 
parser.add_argument('--weight_decay', type=float, default=1e-4, help='L2 正则化系数 (Adam优化器)')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
set_seed(args.seed, deterministic=args.deterministic)

BATCH_SIZE, LR, warmup_epochs, num_epochs, mode = args.batch_size, args.lr, args.warmup_epochs, args.num_epochs, args.mode
resume_flag, use_amp = args.resume, args.use_amp

# 保存配置到JSON文件
config_path = Path(args.output_dir) / f'config_{mode}.json'
with open(config_path, 'w') as f:
    # 将 args 命名空间转换为字典并写入文件
    json.dump(vars(args), f, indent=4)
print(f"✅ 配置已保存到: {config_path}\n")

# ============================================================
# 📊 数据准备
# ============================================================
def prepare_data(csv_path: str):
    """加载并预处理数据"""
    df_bc = pd.read_csv(csv_path)
    X = df_bc.drop(columns=["invasive", "cancer"])
    y = df_bc["cancer"]

    x_train_full, x_test_full, y_train_full, y_test_full = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=args.seed
    )
    
    # 1. 计算原始训练集的类别权重 (用于 CrossEntropyLoss)
    class_counts = y_train_full.value_counts()
    total_samples = len(y_train_full)
    # 权重与频率成反比，用于平衡损失函数
    class_weights = total_samples / (class_counts * 2) 
    
    # 确保权重顺序与类别标签 (0, 1, ...) 对应
    weight_0 = class_weights.get(0, 1.0)
    weight_1 = class_weights.get(1, 1.0)
    weights = torch.tensor([weight_0, weight_1], dtype=torch.float32)

    scaler = StandardScaler()
    x_train_full = scaler.fit_transform(x_train_full)
    x_test_full = scaler.transform(x_test_full)

    # 2. SMOTE 过采样 (用于平衡训练集样本数)
    smote = SMOTE(random_state=args.seed, k_neighbors=5)
    x_res, y_res = smote.fit_resample(x_train_full, y_train_full)

    x_train = torch.tensor(x_res, dtype=torch.float32)
    y_train = torch.tensor(np.asarray(y_res).ravel(), dtype=torch.long)
    x_test = torch.tensor(x_test_full, dtype=torch.float32)
    y_test = torch.tensor(np.asarray(y_test_full).ravel(), dtype=torch.long)

    x_valid, x_test_final, y_valid, y_test_final = train_test_split(
        x_test, y_test, test_size=0.5, stratify=y_test, random_state=args.seed
    )

    train_dataloader = DataLoader(TensorDataset(x_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    valid_dataloader = DataLoader(TensorDataset(x_valid, y_valid), batch_size=BATCH_SIZE, shuffle=False)
    test_dataloader = DataLoader(TensorDataset(x_test_final, y_test_final), batch_size=BATCH_SIZE, shuffle=False)

    dataloaders = {"train": train_dataloader, "valid": valid_dataloader}
    
    return dataloaders, test_dataloader, X.columns, weights


dataloaders, test_dataloader, feature_names, class_weights = prepare_data(args.data)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ============================================================
# 🧠 模型定义
# ============================================================
sample_batch = next(iter(dataloaders['train']))[0]
num_features = sample_batch.shape[1]
num_classes = int(torch.unique(next(iter(dataloaders['train']))[1]).numel())

model = ResidualLinealModelClassifier(
    in_features=num_features,
    num_classes=num_classes,
    mode=mode 
).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=args.weight_decay) 
scheduler_cos = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, num_epochs - warmup_epochs))


# ============================================================
# 📜 日志系统
# ============================================================
fileName = datetime.datetime.today().strftime("%Y-%m-%d")
log_path = Path(args.output_dir) / f"{fileName}_{mode}.log"
sys.stdout = Logger(str(log_path))
sys.stderr = Logger(str(log_path))

print(f"🔥 模型模式: {mode}")
print(f"⚙️ 参数设置: batch={BATCH_SIZE}, lr={LR}, warmup={warmup_epochs}, epochs={num_epochs}, AMP={use_amp}, GradClip={args.grad_clip}, WeightDecay={args.weight_decay}")
print(f"⚖️ 损失权重: {class_weights.cpu().numpy()}")
print(f"📁 输出路径: {args.output_dir}\n")


# ============================================================
# 🧩 辅助函数
# ============================================================
def append_csv_row(fname, row, header=None):
    write_header = not os.path.exists(fname)
    if write_header:
        header=['time', 'epoch', 'phase', 'lr', 'loss', 'acc', 'f1', 'auc']
    else:
        header = False
    pd.DataFrame([row]).to_csv(fname, mode='a', header=header, index=False)


def warmup_scheduler(epoch, base_lr=LR):
    """线性warmup"""
    if warmup_epochs <= 0:
        return
    if epoch < warmup_epochs:
        lr_scale = float(epoch + 1) / max(1, warmup_epochs)
        new_lr = lr_scale * base_lr
        for pg in optimizer.param_groups:
            pg["lr"] = new_lr


# ============================================================
# 📈 学习率可视化 + 报告保存
# ============================================================
def save_training_report(train_losses, valid_losses, train_accs, val_accs, val_f1s, val_aucs, lr_history, mode):
    """绘制学习率与loss曲线，并分别保存为不同PDF"""
    report_path = Path(args.output_dir) / f"training_report_{mode}.csv"
    loss_pdf_path = Path(args.output_dir) / f"loss_curve_{mode}.pdf"
    lr_pdf_path = Path(args.output_dir) / f"lr_curve_{mode}.pdf"

    epochs = list(range(1, len(train_losses) + 1))

    # 保存数据表格
    df = pd.DataFrame({
        "epoch": epochs,
        "train_loss": train_losses,
        "valid_loss": valid_losses,
        "train_acc": train_accs,
        "valid_acc": val_accs,
        "valid_f1": val_f1s,   
        "valid_auc": val_aucs, 
        "lr": lr_history[:len(epochs)]
    })
    df.to_csv(report_path, index=False)
    print(f"✅ 报告已保存: {report_path}")

    # ---- 绘制并保存 Loss 曲线 ----
    plt.figure()
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, valid_losses, label="Valid Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f"Training Loss Curve - {mode}")
    plt.tight_layout()
    plt.savefig(loss_pdf_path)
    plt.close()
    print(f"✅ Loss 曲线已保存: {loss_pdf_path}")

    # ---- 绘制并保存 Learning Rate 曲线 ----
    plt.figure()
    plt.plot(epochs, lr_history[:len(epochs)], color='orange', label="Learning Rate")
    plt.xlabel("Epoch")
    plt.ylabel("LR")
    plt.legend()
    plt.title(f"Learning Rate Curve - {mode}")
    plt.tight_layout()
    plt.savefig(lr_pdf_path)
    plt.close()
    print(f"✅ 学习率曲线已保存: {lr_pdf_path}")


# ============================================================
# 🚀 训练主函数
# ============================================================
def train(mode: str = "base", resume: bool = False):
    model.mode = mode
    ckpt_path = Path(args.output_dir) / f'checkpoint_{mode}.pth'

    best_f1, best_acc, start_epoch = 0.0, 0.0, 0
    if resume and ckpt_path.exists():
        print(f"🔄 加载断点: {ckpt_path}")
        checkpoint = torch.load(str(ckpt_path), map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint.get('optimizer', {}))
        best_f1 = checkpoint.get('best_f1', 0.0)
        best_acc = checkpoint.get('best_acc', 0.0)
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"✅ 从第 {start_epoch} epoch 继续, 最佳验证 F1={best_f1:.4f}")

    since = time.time()
    train_losses, valid_losses, train_accs, val_accs, valid_f1s, valid_aucs, lr_history = [], [], [], [], [], [], []
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for epoch in range(start_epoch, num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}\n' + '-' * 10)
        warmup_scheduler(epoch)
        lr_history.append(optimizer.param_groups[0]["lr"])

        for phase in ['train', 'valid']:
            model.train() if phase == 'train' else model.eval()
            
            # 评估指标累加器
            running_loss, running_corrects = 0.0, 0
            all_preds, all_labels, all_probas = [], [], []
            dataset_len = len(dataloaders[phase].dataset)

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        # 确保在 Attention 模式下，不返回注意力权重
                        outputs = model(inputs, return_attn=False)
                        loss = criterion(outputs, labels)
                        
                        # Softmax 后的概率（用于 AUC）
                        probas = F.softmax(outputs, dim=1) 
                        _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        scaler.scale(loss).backward()
                        
                        # 梯度裁剪 (防止梯度爆炸)
                        if args.grad_clip > 0.0:
                            scaler.unscale_(optimizer) # AMP模式下，需要先unscale
                            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

                        scaler.step(optimizer)
                        scaler.update()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels).item()

                # 收集预测结果用于 F1/AUC 计算 (移到 CPU)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
                # 只取阳性类（类别 1）的概率
                all_probas.extend(probas[:, 1].cpu().tolist()) 

            epoch_loss = running_loss / dataset_len
            epoch_acc = running_corrects / dataset_len
            
            # 计算 F1 和 AUC
            epoch_f1 = f1_score(all_labels, all_preds) if dataset_len > 0 else 0.0
            try:
                epoch_auc = roc_auc_score(all_labels, all_probas) if dataset_len > 0 else 0.0
            except ValueError:
                # 如果只有一类标签，roc_auc_score 会报错
                epoch_auc = 0.0
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} F1: {epoch_f1:.4f} AUC: {epoch_auc:.4f}')

            # 记录更多指标
            append_csv_row(
                str(Path(args.output_dir) / f'loss_acc_{mode}.csv'),
                [datetime.datetime.now(), f"Epoch[{epoch}]", phase, optimizer.param_groups[0]['lr'], epoch_loss, epoch_acc, epoch_f1, epoch_auc]
            )

            if phase == 'valid':
                valid_losses.append(epoch_loss)
                val_accs.append(epoch_acc)
                valid_f1s.append(epoch_f1)   
                valid_aucs.append(epoch_auc) 
                
                # 最佳模型保存逻辑基于 F1 Score
                if epoch_f1 > best_f1: 
                    best_f1 = epoch_f1
                    best_acc = epoch_acc 
                    torch.save({
                        'state_dict': model.state_dict(),
                        'best_f1': best_f1, 
                        'best_acc': best_acc,
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch,
                        'config': vars(args) # 保存参数配置
                    }, str(ckpt_path))
                    print(f"💾 保存新最佳模型 (基于 F1 Score={best_f1:.4f})")
            else:
                train_losses.append(epoch_loss)
                train_accs.append(epoch_acc)

        if epoch >= warmup_epochs:
            scheduler_cos.step()
        if get_early_stop_signal(train_losses, valid_losses, patience=10, min_delta=1e-4):
            print("⛔ 早停触发，结束训练")
            break

    time_elapsed = time.time() - since
    print(f'训练完成，用时 {time_elapsed // 60:.0f}分{time_elapsed % 60:.0f}秒, 最佳F1={best_f1:.4f}')

    # 传递 F1 和 AUC 历史
    save_training_report(train_losses, valid_losses, train_accs, val_accs, valid_f1s, valid_aucs, lr_history, mode)
    return best_f1


# ============================================================
# 🧩 主流程
# ============================================================
if __name__ == "__main__":
    best = train(mode=mode, resume=resume_flag)

    x_test_all = torch.cat([batch[0] for batch in test_dataloader]).cpu()
    y_test_all = torch.cat([batch[1] for batch in test_dataloader]).cpu()

    ckpt_path = Path(args.output_dir) / f'checkpoint_{mode}.pth'
    if ckpt_path.exists():
        checkpoint = torch.load(str(ckpt_path), map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        print(f"已加载最佳模型进行评估 (Best F1={checkpoint.get('best_f1', -1):.4f})")

    if mode == "attn":
        print("🔍 提取 Attention 权重...")
        model.eval()
        model.mode=mode
        background_idx = np.random.choice(x_test_all.shape[0], min(100, x_test_all.shape[0]), replace=False)
        background = x_test_all[background_idx]
        with torch.no_grad():
            _, attn_maps = model(background.to(device), return_attn=True)
        attention_pdf = Path(args.output_dir) / f"attention_{mode}.pdf"
        visualize_attention(attn_maps, feature_names, draw_samples=False, pdf_filename=str(attention_pdf))
        print(f"✅ Attention PDF 已保存: {attention_pdf}")

    # test(x_test_all, y_test_all, mode=mode, output_dir=args.output_dir)
    # evaluate(model, dataloaders['train'], test_dataloader, feature_names, mode=mode, output_dir=args.output_dir)
    print("🎉 全部流程完成！")