# -*- coding: utf-8 -*-
"""多时相分类器验证：评估样本量 + 训练李子vs柑橘 / 落叶vs常绿分类器"""
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

BASE = r"e:\工作相关\2026年\0624 待测试数据"
ORCH = os.path.join(BASE, "xuyong_orchard_monthly.csv")
CTRL = os.path.join(BASE, "xuyong_control_monthly.csv")

MONTHS = [f"M{i:02d}" for i in range(1, 13)]
BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']

orc = pd.read_csv(ORCH, encoding="utf-8-sig")
ctrl = pd.read_csv(CTRL, encoding="utf-8-sig")
ndvi_cols = [f"{m}_NDVI" for m in MONTHS]
ndvi_evi = ndvi_cols + [f"{m}_EVI" for m in MONTHS]
all_bands = [f"{m}_{b}" for m in MONTHS for b in BANDS]


def run_binary(df, y_vals, feat_cols, name):
    sub = df[df["_y"].isin(y_vals)].copy()
    y = (sub["_y"] == y_vals[0]).astype(int).values
    X = sub[feat_cols].values.astype(float)
    mask = ~np.isnan(X).any(axis=1)
    X, y = X[mask], y[mask]
    print(f"\n=== {name} ===")
    print(f"  正={int((y == 1).sum())} 负={int((y == 0).sum())}  特征维数={X.shape[1]}")
    if min((y == 1).sum(), (y == 0).sum()) < 5:
        print("  样本过少，跳过")
        return
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, f1s = [], []
    for tr, te in skf.split(X, y):
        m = lgb.LGBMClassifier(n_estimators=100, max_depth=3, num_leaves=15,
                               learning_rate=0.05, min_child_samples=10,
                               random_state=42, verbose=-1)
        m.fit(X[tr], y[tr])
        p = m.predict(X[te])
        accs.append(accuracy_score(y[te], p))
        f1s.append(f1_score(y[te], p))
    print(f"  5折CV Acc={np.mean(accs):.3f} (±{np.std(accs):.3f})  F1={np.mean(f1s):.3f}")


# 任务1：李子 vs 柑橘
orc = orc.copy()
orc["_y"] = orc["ZWMC"]
run_binary(orc, ["李子", "柑橘"], ndvi_cols, "李子 vs 柑橘（12月 NDVI）")
run_binary(orc, ["李子", "柑橘"], ndvi_evi, "李子 vs 柑橘（12月 NDVI+EVI）")
run_binary(orc, ["李子", "柑橘"], all_bands, "李子 vs 柑橘（120维光谱）")

# 任务2：落叶(李子) vs 常绿(柑橘+林地)
df2 = pd.concat([orc, ctrl], ignore_index=True)
df2["_y"] = np.where(df2["ZWMC"] == "李子", "落叶",
             np.where(df2["ZWMC"] == "柑橘", "常绿",
             np.where(df2["landtype"] == "林地", "常绿", None)))
run_binary(df2, ["落叶", "常绿"], ndvi_cols, "落叶(李) vs 常绿(柑+林)（12月 NDVI）")

# 任务3：五分类（李子/柑橘/林地/草地/农田）
df3 = pd.concat([orc, ctrl], ignore_index=True)
df3["_y"] = df3["ZWMC"].fillna(df3["landtype"])
sub = df3[df3["_y"].isin(["李子", "柑橘", "林地", "草地", "农田"])].copy()
X = sub[ndvi_cols].values.astype(float)
y = sub["_y"].astype(str).values
mask = ~np.isnan(X).any(axis=1)
X, y = X[mask], y[mask]
print(f"\n=== 五分类（12月 NDVI）===")
print(f"  样本={len(y)}  类别分布={dict(pd.Series(y).value_counts())}")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
accs, f1s = [], []
for tr, te in skf.split(X, y):
    m = lgb.LGBMClassifier(n_estimators=100, max_depth=3, num_leaves=15,
                           learning_rate=0.05, min_child_samples=10,
                           random_state=42, verbose=-1)
    m.fit(X[tr], y[tr])
    p = m.predict(X[te])
    accs.append(accuracy_score(y[te], p))
    f1s.append(f1_score(y[te], p, average="weighted"))
print(f"  5折CV Acc={np.mean(accs):.3f} (±{np.std(accs):.3f})  F1(weighted)={np.mean(f1s):.3f}")
