# -*- coding: utf-8 -*-
"""train_gaoliang_5phase.py — 大春 7期细时相：时相选择验证 + 高粱早收特征(A+B)

用 gee_dachun_5counties_7phase.csv（4/5/6/7/8上/8下/9 月），做两件事：
1. 时相 ablation：对比不同时相组合的三分类精度，验证 5~8 月是否最优
2. 方案A+B：加「8月上旬」时相 + 「NDVI 骤降」早收特征，看高粱能否提升
"""
import os, sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
import lightgbm as lgb

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from common import compute_feature_matrix

CSV = os.path.join(BASE_DIR, "gee_dachun_5counties_7phase.csv")

BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
CLASSES = ["水稻", "玉米", "高粱"]
BAND_MAP = {'B2': 'B02', 'B3': 'B03', 'B4': 'B04', 'B5': 'B05', 'B6': 'B06',
            'B7': 'B07', 'B8': 'B08', 'B8A': 'B8A', 'B11': 'B11', 'B12': 'B12'}

# 时相组合（P1=4月 P2=5月 P3=6月 P4=7月 P5=8上 P6=8下 P7=9月）
COMBOS = [
    ("5/6/7/8上 (4期)", ["P2", "P3", "P4", "P5"], False),
    ("5/6/7/8上/8下 (5期)", ["P2", "P3", "P4", "P5", "P6"], False),
    ("+4月 (6期)", ["P1", "P2", "P3", "P4", "P5", "P6"], False),
    ("+9月 (7期, 4~9全)", ["P1", "P2", "P3", "P4", "P5", "P6", "P7"], False),
    ("5期 + 早收特征(A+B)", ["P2", "P3", "P4", "P5", "P6"], True),
]


def load():
    df = pd.read_csv(CSV, encoding="utf-8-sig")
    rename = {}
    for c in df.columns:
        if "_" in c:
            p, b = c.rsplit("_", 1)
            if b in BAND_MAP and p.startswith("P"):
                rename[c] = f"{p}_{BAND_MAP[b]}"
    df = df.rename(columns=rename)
    df = df.rename(columns={"ZWMC": "类别"})
    return df


def ndvi_of(df, p):
    nir = df[f"{p}_B08"].values
    red = df[f"{p}_B04"].values
    d = nir + red
    return np.where(d > 0, (nir - red) / d, 0.0)


def build_features(df, phases, add_early):
    band_values = {}
    for p in phases:
        for b in BANDS:
            col = f"{p}_{b}"
            if col in df.columns:
                band_values[col] = df[col].values
    X, names = compute_feature_matrix(band_values, phases, available_bands=BANDS)
    if add_early:
        # 早收特征：7月 NDVI - 8月上旬 NDVI（高粱骤降 / 玉米缓降）
        drop = ndvi_of(df, "P4") - ndvi_of(df, "P5")
        X = np.column_stack([X, drop])
        names.append("EARLY_NDVI_drop_7_8up")
        # 8月上旬 NDVI 绝对值（高粱收割后接近裸土）
        ndvi_8up = ndvi_of(df, "P5")
        X = np.column_stack([X, ndvi_8up])
        names.append("EARLY_NDVI_8up")
    return X, names


def make_model():
    return lgb.LGBMClassifier(
        n_estimators=300, max_depth=6, num_leaves=31, learning_rate=0.03,
        subsample=0.7, colsample_bytree=0.6, reg_alpha=1.0, reg_lambda=2.0,
        min_child_samples=20, class_weight="balanced",
        objective="multiclass", num_class=len(CLASSES),
        random_state=42, n_jobs=-1, verbose=-1)


def clean(X, y):
    X = np.asarray(X, dtype=np.float32)
    mask = ~np.isnan(X).any(axis=1)
    cur = np.where(mask)[0]
    Xc = X[mask]
    q1, q3 = np.percentile(Xc[:, :Xc.shape[1] // 2], [25, 75], axis=0)
    iqr = q3 - q1
    outlier = np.any((Xc[:, :Xc.shape[1] // 2] > q3 + 3 * iqr) |
                     (Xc[:, :Xc.shape[1] // 2] < q1 - 3 * iqr), axis=1)
    mask[cur[outlier]] = False
    return X[mask], y[mask]


def evaluate(X, y):
    X, y = clean(X, y)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, yt_all, yp_all = [], [], []
    for tr, te in skf.split(X, y):
        m = make_model()
        m.fit(X[tr], y[tr])
        p = m.predict(X[te])
        accs.append(accuracy_score(y[te], p))
        yt_all.append(y[te]); yp_all.append(p)
    yt = np.concatenate(yt_all); yp = np.concatenate(yp_all)
    rep = classification_report(yt, yp, labels=CLASSES, digits=3, output_dict=True)
    return np.mean(accs), rep


def main():
    df = load()
    print(f"数据: {df.shape}")
    print(df["类别"].value_counts().to_string())
    y = df["类别"].values

    print("\n" + "=" * 70)
    print("时相组合对比（5折分层CV，三分类）")
    print("=" * 70)
    print(f"{'组合':<24} {'Acc':>7} {'水稻F1':>7} {'玉米F1':>7} {'高粱F1':>7} {'高粱R':>6} {'高粱P':>6}")
    for name, phases, add_early in COMBOS:
        X, _ = build_features(df, phases, add_early)
        acc, rep = evaluate(X, y)
        r = lambda c: rep[c]
        print(f"{name:<24} {acc:>7.4f} {r('水稻')['f1-score']:>7.3f} {r('玉米')['f1-score']:>7.3f} "
              f"{r('高粱')['f1-score']:>7.3f} {r('高粱')['recall']:>6.3f} {r('高粱')['precision']:>6.3f}")


if __name__ == "__main__":
    main()
