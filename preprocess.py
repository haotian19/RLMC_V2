# -*- coding: utf-8 -*-
"""
输入: risk.txt（BCSC Risk Estimation 数据集，空白分隔，无表头，16 列）
输出: clean_data.csv（删除 training 列；保留 count，count_mode 由 main.py/main_ml.py 决定）
"""
import pandas as pd

COL_NAMES = [
    "menopaus", "agegrp", "density", "race", "Hispanic",
    "bmi", "agefirst", "nrelbc", "brstproc", "lastmamm",
    "surgmeno", "hrt", "invasive", "cancer", "training", "count"
]

# 1. 读取原始数据
df = pd.read_csv("risk.txt", delim_whitespace=True, header=None, names=COL_NAMES)

# 2. 数据体检（不符合预期直接报错，避免脏数据静默进入管线）
assert df.shape[1] == 16, f"列数应为 16，实际 {df.shape[1]}"
n_na = int(df.isna().sum().sum())
assert n_na == 0, f"存在 {n_na} 个缺失值，请先处理"
assert set(df["cancer"].unique()) <= {0, 1}, "cancer 应为 0/1"
assert set(df["training"].unique()) <= {0, 1}, "training 应为 0/1"

# 3. 删除 training 列
df = df.drop(columns=["training"])

# 4. 保存
df.to_csv("clean_data.csv", index=False)

# 5. 体检报告（论文 2.1/2.2 节可用）
n = len(df)
pos = int((df["cancer"] == 1).sum())
print("✅ 数据预处理完成，已保存为 clean_data.csv")
print(f"   记录数: {n}  特征数(含count): {df.shape[1] - 1}")
print(f"   cancer=1 记录: {pos} ({pos / n:.4f})   cancer=0 记录: {n - pos} ({1 - pos / n:.4f})")
print(f"   阳性率 = {pos / n:.4f}   全阴基线 accuracy = {1 - pos / n:.4f}")
print(f"   count: min={df['count'].min()}, median={df['count'].median():.0f}, "
      f"max={df['count'].max()}, sum={int(df['count'].sum())} (对应筛查人次)")
print(f"   独立风险组合数: {df.drop(columns=['cancer', 'invasive']).drop_duplicates().shape[0]}")
