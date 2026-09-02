# -*- coding: utf-8 -*-
"""train_dachun_binary_3dekad.py — 关键 3 旬重训水稻/玉米二分类

基于 13.8 结论（模型自选关键旬 = 5月上 + 7月中 + 8月中），
用旬级 18d 数据验证：关键 3 旬是否足以支撑水稻/玉米二分类（对比 4期/7期 + 全18旬上界）。
"""
import os, sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from sklearn.metrics import accuracy_score, f1_score
import lightgbm as lgb

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from common import compute_feature_matrix

CSV = os.path.join(BASE_DIR, "gee_dachun_5counties_18d.csv")

BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
CLASSES = ["水稻", "玉米"]
BAND_MAP = {'B2': 'B02', 'B3': 'B03', 'B4': 'B04', 'B5': 'B05', 'B6': 'B06',
            'B7': 'B07', 'B8': 'B08', 'B8A': 'B8A', 'B11': 'B11', 'B12': 'B12'}

DEKAD_LABEL = {
    'D04': '5月上', 'D10': '7月上', 'D11': '7月中', 'D14': '8月中', 'D15': '8月下',
}

COMBOS = [
    ("关键3旬 5月上+7月中+8月中", ["D04", "D11", "D14"]),
    ("关键3旬 +7月上", ["D04", "D10", "D11", "D14"]),
    ("关键3旬 +8月下", ["D04", "D11", "D14", "D15"]),
    ("全18旬(上界)", [f"D{i:02d}" for i in range(1, 19)]),
]


def load():
    df = pd.read_csv(CSV, encoding="utf-8-sig")
    rename = {}
    for c in df.columns:
        if "_" in c:
            p, b = c.rsplit("_", 1)
            if b in BAND_MAP and p.startswith("D"):
                rename[c] = f"{p}_{BAND_MAP[b]}"
    df = df.rename(columns=rename)
    df = df.rename(columns={"ZWMC": "类别"})
    df = df[df["类别"].isin(CLASSES)].copy()
    return df


def make_model():
    return lgb.LGBMClassifier(
        n_estimators=300, max_depth=5, num_leaves=31, learning_rate=0.03,
        subsample=0.7, colsample_bytree=0.6, reg_alpha=1.0, reg_lambda=2.0,
        min_child_samples=20, class_weight="balanced",
        objective="binary", random_state=42, n_jobs=-1, verbose=-1)


def clean(X, y, groups):
    X = np.asarray(X, dtype=np.float32)
    mask = ~np.isnan(X).any(axis=1)
    cur = np.where(mask)[0]
    Xc = X[mask]
    q1, q3 = np.percentile(Xc[:, :Xc.shape[1] // 2], [25, 75], axis=0)
    iqr = q3 - q1
    outlier = np.any((Xc[:, :Xc.shape[1] // 2] > q3 + 3 * iqr) |
                     (Xc[:, :Xc.shape[1] // 2] < q1 - 3 * iqr), axis=1)
    mask[cur[outlier]] = False
    return X[mask], y[mask], groups[mask]


def evaluate(df):
    y = (df["类别"] == "玉米").astype(int).values  # 1=玉米 0=水稻
    groups = df["QXMC"].astype(str).values

    print(f"{'组合':<26} {'5折Acc':>8} {'5折F1':>8} {'留一县Acc':>9} {'特征数':>7}")
    print("-" * 62)
    for name, phases in COMBOS:
        band_values = {}
        for p in phases:
            for b in BANDS:
                col = f"{p}_{b}"
                if col in df.columns:
                    band_values[col] = df[col].values
        X, fnames = compute_feature_matrix(band_values, phases, available_bands=BANDS)
        X, y_, g_ = clean(X, y, groups)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        accs, f1s = [], []
        for tr, te in skf.split(X, y_):
            m = make_model(); m.fit(X[tr], y_[tr]); p = m.predict(X[te])
            accs.append(accuracy_score(y_[te], p))
            f1s.append(f1_score(y_[te], p, average="weighted"))

        logo = LeaveOneGroupOut()
        logo_acc = []
        for tr, te in logo.split(X, y_, g_):
            m = make_model(); m.fit(X[tr], y_[tr]); p = m.predict(X[te])
            logo_acc.append(accuracy_score(y_[te], p))

        print(f"{name:<26} {np.mean(accs):>8.4f} {np.mean(f1s):>8.4f} "
              f"{np.mean(logo_acc):>9.4f} {X.shape[1]:>7d}")


def main():
    df = load()
    print(f"二分类样本: {len(df)}")
    print(df["类别"].value_counts().to_string())
    print()
    evaluate(df)


if __name__ == "__main__":
    main()
