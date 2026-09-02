# -*- coding: utf-8 -*-
"""train_dachun_18d_select.py — 数据驱动的时相自选验证

读 18 旬原始波段（180 列），用 LightGBM 三分类：
1. 特征重要性按旬聚合，看模型自选了哪些旬（关键物候期）
2. 前向选择：按旬重要性从高到低累加，看 Acc 饱和点
"""
import os, sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import lightgbm as lgb

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(BASE_DIR, "gee_dachun_5counties_18d.csv")

BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
CLASSES = ["水稻", "玉米", "高粱"]
BAND_MAP = {'B2': 'B02', 'B3': 'B03', 'B4': 'B04', 'B5': 'B05', 'B6': 'B06',
            'B7': 'B07', 'B8': 'B08', 'B8A': 'B8A', 'B11': 'B11', 'B12': 'B12'}
DEKADS = [f"D{i:02d}" for i in range(1, 19)]

# 旬 -> 日期标签（便于读图）
DEKAD_LABEL = {
    'D01': '4月上', 'D02': '4月中', 'D03': '4月下',
    'D04': '5月上', 'D05': '5月中', 'D06': '5月下',
    'D07': '6月上', 'D08': '6月中', 'D09': '6月下',
    'D10': '7月上', 'D11': '7月中', 'D12': '7月下',
    'D13': '8月上', 'D14': '8月中', 'D15': '8月下',
    'D16': '9月上', 'D17': '9月中', 'D18': '9月下',
}


def make_model():
    return lgb.LGBMClassifier(
        n_estimators=300, max_depth=7, num_leaves=63, learning_rate=0.03,
        subsample=0.7, colsample_bytree=0.6, reg_alpha=1.0, reg_lambda=2.0,
        min_child_samples=20, class_weight="balanced",
        objective="multiclass", num_class=len(CLASSES),
        random_state=42, n_jobs=-1, verbose=-1)


def load():
    df = pd.read_csv(CSV, encoding="utf-8-sig")
    rename = {}
    for c in df.columns:
        if "_" in c:
            p, b = c.rsplit("_", 1)
            if b in BAND_MAP and p in DEKADS:
                rename[c] = f"{p}_{BAND_MAP[b]}"
    df = df.rename(columns=rename)
    df = df.rename(columns={"ZWMC": "类别"})
    return df


def cv_acc(X, y):
    # LightGBM 原生处理 NaN，无需 dropna
    X = np.asarray(X, dtype=np.float32)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs = []
    for tr, te in skf.split(X, y):
        m = make_model(); m.fit(X[tr], y[tr]); p = m.predict(X[te])
        accs.append(accuracy_score(y[te], p))
    return np.mean(accs)


def main():
    df = load()
    y = df["类别"].values
    feat_cols = [f"{d}_{b}" for d in DEKADS for b in BANDS]
    X = df[feat_cols].values
    print(f"数据: {X.shape}，类别分布:")
    print(pd.Series(y).value_counts().to_string())

    # 诊断：每旬 NaN 比例（判断哪些旬缺数据）
    print("\n各旬 NaN 比例（样本级，任一列缺失即计）:")
    for d in DEKADS:
        cols = [f"{d}_{b}" for b in BANDS]
        nan_ratio = df[cols].isna().any(axis=1).mean()
        print(f"  {d} ({DEKAD_LABEL[d]}): {nan_ratio*100:5.1f}% 样本含 NaN")

    # 1) 全量训练 -> 特征重要性按旬聚合（LightGBM 原生处理 NaN）
    Xc = np.asarray(X, dtype=np.float32)
    m = make_model()
    m.fit(Xc, y)
    imp = m.booster_.feature_importance(importance_type="gain")

    dekad_imp = {}
    for i, d in enumerate(DEKADS):
        s = slice(i * len(BANDS), (i + 1) * len(BANDS))
        dekad_imp[d] = float(imp[s].sum())
    total = sum(dekad_imp.values())
    ranked = sorted(dekad_imp.items(), key=lambda kv: -kv[1])

    print("\n" + "=" * 60)
    print("旬重要性排名（模型自选的关键物候期）")
    print("=" * 60)
    for d, v in ranked:
        bar = "#" * int(round(v / total * 60))
        print(f"{d} ({DEKAD_LABEL[d]}): {v/total*100:5.1f}%  {bar}")

    # 2) 前向选择：按旬重要性从高到低累加
    print("\n" + "=" * 60)
    print("前向选择（按旬重要性累加，看 Acc 饱和点）")
    print("=" * 60)
    order = [d for d, _ in ranked]
    selected_cols = []
    for k in range(1, len(order) + 1):
        selected_cols += [f"{order[k-1]}_{b}" for b in BANDS]
        acc = cv_acc(df[selected_cols].values, y)
        mark = " <<<" if k in (3, 5, 7, 9) else ""
        print(f"  top{k:2d} 旬 -> Acc={acc:.4f}{mark}")


if __name__ == "__main__":
    main()
