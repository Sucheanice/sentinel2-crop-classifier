# -*- coding: utf-8 -*-
"""train_dachun_18d_mc.py — 多景去云合成（扩窗） vs 严格窗口 对比

对比严格 10 天窗口 与 扩窗多景去云合成 的：
1. 各旬 NaN 比例
2. 全量 18 旬三分类 Acc
"""
import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import lightgbm as lgb

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_RAW = os.path.join(BASE_DIR, "gee_dachun_5counties_18d.csv")
CSV_MC = os.path.join(BASE_DIR, "gee_dachun_5counties_18d_mc.csv")

BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
DEKADS = [f"D{i:02d}" for i in range(1, 19)]
DEKAD_LABEL = {
    'D01': '4月上', 'D02': '4月中', 'D03': '4月下',
    'D04': '5月上', 'D05': '5月中', 'D06': '5月下',
    'D07': '6月上', 'D08': '6月中', 'D09': '6月下',
    'D10': '7月上', 'D11': '7月中', 'D12': '7月下',
    'D13': '8月上', 'D14': '8月中', 'D15': '8月下',
    'D16': '9月上', 'D17': '9月中', 'D18': '9月下',
}
BAND_MAP = {'B2': 'B02', 'B3': 'B03', 'B4': 'B04', 'B5': 'B05', 'B6': 'B06',
            'B7': 'B07', 'B8': 'B08', 'B8A': 'B8A', 'B11': 'B11', 'B12': 'B12'}


def make_model():
    return lgb.LGBMClassifier(
        n_estimators=300, max_depth=7, num_leaves=63, learning_rate=0.03,
        subsample=0.7, colsample_bytree=0.6, reg_alpha=1.0, reg_lambda=2.0,
        min_child_samples=20, class_weight="balanced",
        objective="multiclass", num_class=3,
        random_state=42, n_jobs=-1, verbose=-1)


def load(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    rename = {}
    for c in df.columns:
        if "_" in c:
            p, b = c.rsplit("_", 1)
            if b in BAND_MAP and p in DEKADS:
                rename[c] = f"{p}_{BAND_MAP[b]}"
    df = df.rename(columns=rename)
    df = df.rename(columns={"ZWMC": "类别"})
    return df


def feat_cols():
    return [f"{d}_{b}" for d in DEKADS for b in BANDS]


def cv_acc(X, y):
    X = np.asarray(X, dtype=np.float32)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs = []
    for tr, te in skf.split(X, y):
        m = make_model(); m.fit(X[tr], y[tr]); p = m.predict(X[te])
        accs.append(accuracy_score(y[te], p))
    return np.mean(accs)


def nan_by_dekad(df):
    return {d: df[[f"{d}_{b}" for b in BANDS]].isna().any(axis=1).mean()
            for d in DEKADS}


def main():
    df_raw = load(CSV_RAW)
    df_mc = load(CSV_MC)
    y = df_raw["类别"].values
    cols = feat_cols()

    nan_raw = nan_by_dekad(df_raw)
    nan_mc = nan_by_dekad(df_mc)

    print(f"{'旬':<4} {'日期':<6} {'严格窗口NaN':>10} {'扩窗NaN':>9} {'降幅':>7}")
    print("-" * 44)
    for d in DEKADS:
        r, m = nan_raw[d], nan_mc[d]
        print(f"{d:<4} {DEKAD_LABEL[d]:<6} {r*100:9.1f}% {m*100:8.1f}% {(r-m)*100:6.1f}%")

    print("\n全量 18 旬三分类 Acc（5折）:")
    acc_raw = cv_acc(df_raw[cols].values, y)
    acc_mc = cv_acc(df_mc[cols].values, y)
    print(f"  严格 10 天窗口      : {acc_raw:.4f}")
    print(f"  扩窗多景去云合成    : {acc_mc:.4f}  (Δ {acc_mc-acc_raw:+.4f})")


if __name__ == "__main__":
    main()
