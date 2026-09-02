# -*- coding: utf-8 -*-
"""train_dachun_gee_vs_local.py — 大春三分类：GEE 特征 vs 本地影像特征 对比

同一批地块（射洪/泸县5县 水稻/玉米/高粱），分别用：
  - 本地：features_23counties.csv(水稻/玉米) + gaoliang_features.csv(高粱)，单景采样
  - GEE ：gee_dachun_5counties.csv，SCL云掩膜 + median 合成
跑相同的 5折分层CV + 阈值扫描 + 留一县，输出并排对比。
"""
import os, sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import lightgbm as lgb

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from common import compute_feature_matrix

SCENE_LABELS = ["P1", "P2", "P3", "P4"]
AVAILABLE_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07",
                   "B08", "B8A", "B11", "B12"]
CLASSES = ["水稻", "玉米", "高粱"]
COUNTIES = ["射洪市", "泸县", "龙马潭区", "船山区", "纳溪区"]

BAND_MAP = {'B2': 'B02', 'B3': 'B03', 'B4': 'B04', 'B5': 'B05',
            'B6': 'B06', 'B7': 'B07', 'B8': 'B08', 'B8A': 'B8A',
            'B11': 'B11', 'B12': 'B12'}


def prep_local():
    df_rc = pd.read_csv(os.path.join(BASE_DIR, "待训练数据大春", "features_23counties.csv"),
                        encoding="utf-8-sig")
    df_rc = df_rc[df_rc["类别"].isin(["水稻", "玉米"])].copy()
    df_rc = df_rc[df_rc["QXMC"].isin(COUNTIES)].copy()
    df_gao = pd.read_csv(os.path.join(BASE_DIR, "待训练数据大春", "gaoliang_features.csv"),
                         encoding="utf-8-sig")
    return pd.concat([df_rc, df_gao], ignore_index=True)


def prep_gee():
    df = pd.read_csv(os.path.join(BASE_DIR, "gee_dachun_5counties.csv"), encoding="utf-8-sig")
    # 波段列 B2->B02 对齐本地
    rename = {}
    for c in df.columns:
        if "_" in c:
            p, b = c.rsplit("_", 1)
            if b in BAND_MAP and p in SCENE_LABELS:
                rename[c] = f"{p}_{BAND_MAP[b]}"
    df = df.rename(columns=rename)
    df = df.rename(columns={"ZWMC": "类别"})
    return df


def make_model():
    return lgb.LGBMClassifier(
        n_estimators=300, max_depth=6, num_leaves=31, learning_rate=0.03,
        subsample=0.7, colsample_bytree=0.6, reg_alpha=1.0, reg_lambda=2.0,
        min_child_samples=20, class_weight="balanced",
        objective="multiclass", num_class=len(CLASSES),
        random_state=42, n_jobs=-1, verbose=-1)


def train_and_report(df, name):
    print("\n" + "#" * 64)
    print(f"# {name}")
    print("#" * 64)
    band_values = {}
    for lbl in SCENE_LABELS:
        for band in AVAILABLE_BANDS:
            col = f"{lbl}_{band}"
            if col in df.columns:
                band_values[col] = df[col].values
    X, feat_names = compute_feature_matrix(band_values, SCENE_LABELS, available_bands=AVAILABLE_BANDS)

    y = df["类别"].values
    groups = df["QXMC"].astype(str).values

    X = np.asarray(X, dtype=np.float32)
    mask = ~np.isnan(X).any(axis=1)
    cur = np.where(mask)[0]
    Xc = X[mask]
    q1, q3 = np.percentile(Xc[:, :Xc.shape[1] // 2], [25, 75], axis=0)
    iqr = q3 - q1
    outlier = np.any((Xc[:, :Xc.shape[1] // 2] > q3 + 3 * iqr) |
                     (Xc[:, :Xc.shape[1] // 2] < q1 - 3 * iqr), axis=1)
    mask[cur[outlier]] = False
    X, y, groups = X[mask], y[mask], groups[mask]
    print(f"清洗后样本: {len(X)}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, yt_all, yp_all, yprob_all = [], [], [], []
    for tr, te in skf.split(X, y):
        m = make_model()
        m.fit(X[tr], y[tr])
        p = m.predict(X[te])
        pb = m.predict_proba(X[te])
        accs.append(accuracy_score(y[te], p))
        yt_all.append(y[te]); yp_all.append(p); yprob_all.append(pb)
    yt = np.concatenate(yt_all); yp = np.concatenate(yp_all)
    yprob = np.concatenate(yprob_all)

    print(f"5折 Acc: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    rep = classification_report(yt, yp, labels=CLASSES, digits=3, output_dict=True)
    for c in CLASSES:
        r = rep[c]
        print(f"  {c}: recall={r['recall']:.3f}  precision={r['precision']:.3f}  f1={r['f1-score']:.3f}")

    # 高粱阈值扫描
    gao_prob = yprob[:, CLASSES.index("高粱")]
    gao_true = (yt == "高粱")
    print("  高粱阈值: ", end="")
    for th in [0.5, 0.6, 0.7, 0.8]:
        pred = gao_prob >= th
        tp = (pred & gao_true).sum(); fp = (pred & ~gao_true).sum()
        fn = (~pred & gao_true).sum()
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        print(f"@{th}:R{rec:.2f}/P{prec:.2f}  ", end="")
    print()

    # 留一县
    logo = LeaveOneGroupOut()
    logo_acc = {}
    for tr, te in logo.split(X, y, groups):
        g = groups[te][0]
        m = make_model(); m.fit(X[tr], y[tr]); p = m.predict(X[te])
        logo_acc[g] = accuracy_score(y[te], p)
    print(f"  留一县: " + "  ".join(f"{g}={logo_acc[g]:.3f}" for g in sorted(logo_acc)))
    print(f"  留一县平均 Acc: {np.mean(list(logo_acc.values())):.4f}")
    return {"acc": float(np.mean(accs)), "logo_acc": float(np.mean(list(logo_acc.values())))}


def main():
    r_local = train_and_report(prep_local(), "本地影像特征（单景采样）")
    r_gee = train_and_report(prep_gee(), "GEE 特征（median 合成）")
    print("\n" + "=" * 64)
    print(f"本地  5折Acc={r_local['acc']:.4f}  留一县={r_local['logo_acc']:.4f}")
    print(f"GEE   5折Acc={r_gee['acc']:.4f}  留一县={r_gee['logo_acc']:.4f}")


if __name__ == "__main__":
    main()
