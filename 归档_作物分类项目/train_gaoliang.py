# -*- coding: utf-8 -*-
"""train_gaoliang.py — 射洪/泸县等5县「水稻/玉米/高粱」三分类试点

目的：验证高粱在现有哨兵特征下能否被识别，重点看高粱 recall/precision。
口径：5县（射洪市/泸县/龙马潭区/船山区/纳溪区，即高粱标注分布的县）。
"""
import os, sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import lightgbm as lgb

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from common import compute_feature_matrix

FEATURES_CSV = os.path.join(BASE_DIR, "待训练数据大春", "features_23counties.csv")
GAOLIANG_CSV = os.path.join(BASE_DIR, "待训练数据大春", "gaoliang_features.csv")

SCENE_LABELS = ["P1", "P2", "P3", "P4"]
AVAILABLE_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07",
                   "B08", "B8A", "B11", "B12"]
CLASSES = ["水稻", "玉米", "高粱"]
COUNTIES = ["射洪市", "泸县", "龙马潭区", "船山区", "纳溪区"]


def build_features(df):
    band_values = {}
    for lbl in SCENE_LABELS:
        for band in AVAILABLE_BANDS:
            col = f"{lbl}_{band}"
            if col in df.columns:
                band_values[col] = df[col].values
    X, feat_names = compute_feature_matrix(
        band_values, SCENE_LABELS, available_bands=AVAILABLE_BANDS)
    return X, feat_names


def make_model():
    return lgb.LGBMClassifier(
        n_estimators=300, max_depth=6, num_leaves=31, learning_rate=0.03,
        subsample=0.7, colsample_bytree=0.6, reg_alpha=1.0, reg_lambda=2.0,
        min_child_samples=20, class_weight="balanced",
        objective="multiclass", num_class=len(CLASSES),
        random_state=42, n_jobs=-1, verbose=-1)


def main():
    df_rc = pd.read_csv(FEATURES_CSV, encoding="utf-8-sig")
    df_rc = df_rc[df_rc["类别"].isin(["水稻", "玉米"])].copy()
    df_rc = df_rc[df_rc["QXMC"].isin(COUNTIES)].copy()
    df_gao = pd.read_csv(GAOLIANG_CSV, encoding="utf-8-sig")
    df = pd.concat([df_rc, df_gao], ignore_index=True)
    print(f"合并样本: {len(df)}")
    print(df["类别"].value_counts().to_string())
    print(df["QXMC"].value_counts().to_string())
    print()

    X, feat_names = build_features(df)
    print(f"特征维度: {X.shape}")

    y = df["类别"].values
    groups = df["QXMC"].astype(str).values

    # 清洗：NaN + IQR 异常值
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
    print(f"清洗后: {len(X)} 样本")
    print(pd.Series(y).value_counts().to_string())
    print()

    # ===== 5折分层CV =====
    print("=" * 60)
    print("5折分层CV（随机划分，反映同县内区分能力）")
    print("=" * 60)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs = []
    y_true_all, y_pred_all, y_prob_all = [], [], []
    for tr, te in skf.split(X, y):
        m = make_model()
        m.fit(X[tr], y[tr])
        p = m.predict(X[te])
        pb = m.predict_proba(X[te])
        accs.append(accuracy_score(y[te], p))
        y_true_all.append(y[te])
        y_pred_all.append(p)
        y_prob_all.append(pb)
    yt = np.concatenate(y_true_all)
    yp = np.concatenate(y_pred_all)
    yprob = np.concatenate(y_prob_all)
    gao_prob = yprob[:, CLASSES.index("高粱")]
    gao_true = (yt == "高粱")
    print(f"5折 Acc: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print("\n分类报告（5折聚合）:")
    print(classification_report(yt, yp, labels=CLASSES, digits=3))
    cm = confusion_matrix(yt, yp, labels=CLASSES)
    print("混淆矩阵（行=真值，列=预测）:")
    print(pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_string())
    print()

    print("=" * 60)
    print("高粱概率阈值扫描（降低误报）")
    print("=" * 60)
    print("阈值   高粱recall   高粱precision   高粱F1")
    for th in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        pred_gao = gao_prob >= th
        tp = (pred_gao & gao_true).sum()
        fp = (pred_gao & ~gao_true).sum()
        fn = (~pred_gao & gao_true).sum()
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        print(f"{th:.1f}   {recall:.3f}         {precision:.3f}           {f1:.3f}")
    print()

    # ===== 留一县 =====
    print("=" * 60)
    print("留一县验证（跨县泛化，看高粱能否迁移）")
    print("=" * 60)
    logo = LeaveOneGroupOut()
    logo_acc = {}
    for tr, te in logo.split(X, y, groups):
        g = groups[te][0]
        m = make_model()
        m.fit(X[tr], y[tr])
        p = m.predict(X[te])
        logo_acc[g] = accuracy_score(y[te], p)
        # 若该县有高粱，单独看高粱 recall
        gao_te = (y[te] == "高粱")
        if gao_te.any():
            gao_recall = (p[gao_te] == "高粱").mean()
            print(f"  留出[{g}] Acc={logo_acc[g]:.4f} (n={len(te)})  高粱recall={gao_recall:.3f} (n={gao_te.sum()})")
        else:
            print(f"  留出[{g}] Acc={logo_acc[g]:.4f} (n={len(te)})  无高粱")
    print(f"  >>> 留一县平均 Acc: {np.mean(list(logo_acc.values())):.4f}")


if __name__ == "__main__":
    main()
