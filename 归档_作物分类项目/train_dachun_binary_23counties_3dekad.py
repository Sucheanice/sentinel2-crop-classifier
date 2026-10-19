# -*- coding: utf-8 -*-
"""train_dachun_binary_23counties_3dekad.py — 23县关键3旬重训水稻/玉米二分类（保存模型）

基于 gee_dachun_23counties_3dekad.csv（D04/D11/D14 × 10波段），
用 22 县全量样本重训 LightGBM 二分类（水稻/玉米），
输出 5折CV + 留一县(LOGO) 指标，并保存最终全量模型 pkl，
供 predict_dachun_gee.py 推理使用。
"""
import os, sys, pickle
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

CSV = os.path.join(BASE_DIR, "gee_dachun_23counties_3dekad.csv")
OUT_PKL = os.path.join(BASE_DIR, "dachun_binary_23counties_3dekad.pkl")

BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
CLASSES = ["水稻", "玉米"]
PHASES = ["D04", "D11", "D14"]

# GEE 输出波段名（B2/B3...） -> 特征工程口径（B02/B03...）
BAND_MAP = {'B2': 'B02', 'B3': 'B03', 'B4': 'B04', 'B5': 'B05', 'B6': 'B06',
            'B7': 'B07', 'B8': 'B08', 'B8A': 'B8A', 'B11': 'B11', 'B12': 'B12'}


def load():
    df = pd.read_csv(CSV, encoding="utf-8-sig")
    rename = {}
    for c in df.columns:
        if "_" in c:
            p, b = c.rsplit("_", 1)
            if b in BAND_MAP and p in PHASES:
                rename[c] = f"{p}_{BAND_MAP[b]}"
    df = df.rename(columns=rename)
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


def build_features(df):
    y = (df["类别"] == "玉米").astype(int).values  # 1=玉米 0=水稻
    groups = df["QXMC"].astype(str).values
    band_values = {}
    for p in PHASES:
        for b in BANDS:
            col = f"{p}_{b}"
            if col in df.columns:
                band_values[col] = df[col].values
    X, fnames = compute_feature_matrix(band_values, PHASES, available_bands=BANDS)
    return X, y, groups, fnames


def main():
    df = load()
    print(f"二分类样本: {len(df)}")
    print(df["类别"].value_counts().to_string())
    print(f"县数: {df['QXMC'].nunique()}\n")

    X, y, groups, fnames = build_features(df)
    X, y, groups = clean(X, y, groups)
    print(f"清洗后样本: {len(y)}  (去除 NaN/异常 {len(df) - len(y)})\n")

    # 5 折分层 CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, f1s = [], []
    for tr, te in skf.split(X, y):
        m = make_model(); m.fit(X[tr], y[tr]); p = m.predict(X[te])
        accs.append(accuracy_score(y[te], p))
        f1s.append(f1_score(y[te], p, average="weighted"))
    print(f"5折CV:  Acc={np.mean(accs):.4f} (±{np.std(accs):.4f})  "
          f"F1(weighted)={np.mean(f1s):.4f}")

    # 留一县 LOGO
    logo = LeaveOneGroupOut()
    logo_acc = []
    for tr, te in logo.split(X, y, groups):
        m = make_model(); m.fit(X[tr], y[tr]); p = m.predict(X[te])
        logo_acc.append(accuracy_score(y[te], p))
    print(f"留一县:  Acc={np.mean(logo_acc):.4f}  (共 {len(logo_acc)} 县)\n")

    # 最终模型：全量训练 + 保存
    model = make_model()
    model.fit(X, y)
    bundle = {
        "model": model,
        "model_type": "dachun_binary_23counties_3dekad",
        "class_names": CLASSES,           # 0=水稻 1=玉米
        "label_encoder": None,
        "selected_features": fnames,      # 全特征
        "phases": PHASES,
        "bands": BANDS,
    }
    with open(OUT_PKL, "wb") as f:
        pickle.dump(bundle, f)
    print(f"[saved] {OUT_PKL}")
    print(f"  特征数={len(fnames)}, 类别={CLASSES}")


if __name__ == "__main__":
    main()
