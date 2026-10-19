# -*- coding: utf-8 -*-
"""train_dachun_binary.py — 大春「水稻 vs 玉米」二分类训练（空间分组验证版）

相对 v3 的改进：
  1. 二分类 objective='binary'（v3 用的是 multiclass）
  2. 空间分组验证：GroupKFold + 留一县/留一市，避免同区域样本泄漏
  3. 波段/分组字段/标签字段可配置（遂宁用 district+ZWMC；23县用 QXMC/SZMC+类别）

用法：
  python train_dachun_binary.py
"""
import os, sys, pickle, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import lightgbm as lgb
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from common import compute_feature_matrix

# ================= 配置 =================
FEATURES_CSV = os.path.join(BASE_DIR, "待训练数据大春", "features_23counties.csv")
GROUP_COL = "QXMC"              # 空间分组字段（遂宁=区县；23县用 QXMC 县 / SZMC 市）
LABEL_COL = "类别"              # 标签字段（水稻/玉米）
SCENE_LABELS = ["P1", "P2", "P3", "P4"]
AVAILABLE_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]

OUT_DIR = os.path.join(BASE_DIR, "dachun_binary_23counties")
MODEL_OUT = os.path.join(OUT_DIR, "crop_model_binary.pkl")
REPORT_DIR = os.path.join(OUT_DIR, "report")

CLASSES = ["水稻", "玉米"]       # 二分类目标（顺序无关，LabelEncoder 会处理）


def preprocess(X, y, groups):
    """NaN / IQR 异常值 / 重复样本 清洗，同步过滤 y 和 groups。"""
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    groups = np.asarray(groups)

    valid = ~np.isnan(X).any(axis=1)
    X, y, groups = X[valid], y[valid], groups[valid]

    # 原始波段列做 IQR 异常值检测（沿用 v3）
    q1, q3 = np.percentile(X[:, :X.shape[1] // 2], [25, 75], axis=0)
    iqr = q3 - q1
    outlier = np.any((X[:, :X.shape[1] // 2] > q3 + 3 * iqr) |
                     (X[:, :X.shape[1] // 2] < q1 - 3 * iqr), axis=1)
    X, y, groups = X[~outlier], y[~outlier], groups[~outlier]

    # 重复样本
    _, uid = np.unique(np.round(X, 2), axis=0, return_index=True)
    X, y, groups = X[uid], y[uid], groups[uid]
    return X, y, groups


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    print("=" * 64)
    print("大春水稻/玉米二分类（空间分组验证）")
    print("=" * 64)

    df = pd.read_csv(FEATURES_CSV, encoding="utf-8-sig")
    # 县名归一化：合并笔误/别名（大英市 -> 大英县）
    df[GROUP_COL] = df[GROUP_COL].astype(str).str.strip()
    df[GROUP_COL] = df[GROUP_COL].replace({"大英市": "大英县"})
    print(f"样本: {len(df)}")
    print(f"分组字段 [{GROUP_COL}] 取值: {sorted(df[GROUP_COL].astype(str).unique())}")

    # 只保留水稻/玉米
    df = df[df[LABEL_COL].isin(CLASSES)].copy()
    print(f"二分类样本: {len(df)}")
    print(df[LABEL_COL].value_counts().to_string())

    # 构建 band_values + 特征矩阵
    band_values = {}
    for lbl in SCENE_LABELS:
        for band in AVAILABLE_BANDS:
            col = f"{lbl}_{band}"
            if col in df.columns:
                band_values[col] = df[col].values
    print(f"波段列: {len(band_values)} 个（期望 {len(SCENE_LABELS) * len(AVAILABLE_BANDS)}）")

    X, feat_names = compute_feature_matrix(
        band_values, SCENE_LABELS, available_bands=AVAILABLE_BANDS)
    print(f"特征维度: {X.shape}")

    le = LabelEncoder()
    y = le.fit_transform(df[LABEL_COL].values)
    groups = df[GROUP_COL].astype(str).values
    print(f"类别: {le.classes_.tolist()} -> {list(le.transform(le.classes_))}")

    X, y, groups = preprocess(X, y, groups)
    print(f"预处理后: {len(X)} 样本, {len(set(groups))} 个分组")
    for g in sorted(set(groups)):
        print(f"  {g}: {len(groups[groups == g])}")

    # ============ 特征选择（训练集整体，仅用于缩减维度）============
    sel = lgb.LGBMClassifier(
        n_estimators=200, max_depth=5, num_leaves=31, learning_rate=0.05,
        subsample=0.7, colsample_bytree=0.6, reg_alpha=0.3, reg_lambda=0.3,
        objective="binary", random_state=42, n_jobs=-1, verbose=-1)
    sel.fit(X, y)
    imp = sel.booster_.feature_importance(importance_type="gain")
    top_n = min(40, X.shape[1])
    top_idx = np.argsort(imp)[::-1][:top_n]
    X_sel = X[:, top_idx]
    feat_sel = [feat_names[i] for i in top_idx]
    print(f"\n选择 {len(feat_sel)} 特征, Top5: {feat_sel[:5]}")

    def make_model():
        return lgb.LGBMClassifier(
            n_estimators=300, max_depth=5, num_leaves=31, learning_rate=0.03,
            subsample=0.7, colsample_bytree=0.6, reg_alpha=1.0, reg_lambda=2.0,
            min_child_samples=20, min_child_weight=0.001,
            objective="binary", random_state=42, n_jobs=-1, verbose=-1)

    # ============ 1) GroupKFold（空间分组交叉验证）============
    print("\n[GroupKFold 空间分组 CV]")
    gkf = GroupKFold(n_splits=min(5, len(set(groups))))
    accs, f1s = [], []
    for tr, te in gkf.split(X_sel, y, groups):
        m = make_model()
        m.fit(X_sel[tr], y[tr])
        p = m.predict(X_sel[te])
        accs.append(accuracy_score(y[te], p))
        f1s.append(f1_score(y[te], p, average="weighted"))
    print(f"  Acc: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"  F1 : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

    # ============ 2) 留一县/留一市（迁移代理验证）============
    print("\n[留一组验证 Leave-One-Group-Out]（每组单独当测试，其余训练）")
    logo = LeaveOneGroupOut()
    logo_acc, logo_f1 = {}, {}
    for tr, te in logo.split(X_sel, y, groups):
        g = groups[te][0]
        m = make_model()
        m.fit(X_sel[tr], y[tr])
        p = m.predict(X_sel[te])
        logo_acc[g] = accuracy_score(y[te], p)
        logo_f1[g] = f1_score(y[te], p, average="weighted")
        print(f"  留出 [{g}]: Acc={logo_acc[g]:.4f} F1={logo_f1[g]:.4f} (n={len(te)})")
    print(f"  >>> 留一组平均 Acc: {np.mean(list(logo_acc.values())):.4f}")

    # ============ 3) 全量训练 + 保存 ============
    model = make_model()
    model.fit(X_sel, y)
    bundle = {
        "model": model,
        "label_encoder": le,
        "selected_features": feat_sel,
        "all_feature_names": feat_names,
        "scene_labels": SCENE_LABELS,
        "available_bands": AVAILABLE_BANDS,
        "group_col": GROUP_COL,
        "label_col": LABEL_COL,
        "gkf_acc": float(np.mean(accs)),
        "gkf_f1": float(np.mean(f1s)),
        "logo_acc": logo_acc,
        "logo_f1": logo_f1,
        "class_names": le.classes_.tolist(),
        "model_type": "lightgbm_binary_spatial",
    }
    with open(MODEL_OUT, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\n模型已保存: {MODEL_OUT}")

    # 特征重要性
    imp_df = pd.DataFrame({"feature": feat_sel, "importance": imp[top_idx]})
    imp_df = imp_df.sort_values("importance", ascending=False)
    imp_df.to_csv(os.path.join(REPORT_DIR, "feature_importance.csv"),
                  index=False, encoding="utf-8-sig")
    print(f"特征重要性已保存: {REPORT_DIR}/feature_importance.csv")


if __name__ == "__main__":
    main()
