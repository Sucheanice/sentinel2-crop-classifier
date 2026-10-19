# -*- coding: utf-8 -*-
"""train_dachun_23counties.py — 23县库「水稻 vs 玉米」二分类训练（空间分组验证）

【状态】待影像到位 + 特征提取完成后再运行（依赖 features_23counties.csv）。

相对遂宁版 train_dachun_binary.py 的改动：
  1. 波段 7 -> 10（AVAILABLE_BANDS 补 B05/B06/B07）
  2. 时序列名 date -> P1..P4（对应 5/6/7/8 月）
  3. 分组字段 district -> QXMC(县)；新增 SZMC(市) 做「留一市」迁移代理验证
  4. 标签字段 ZWMC -> 类别

核心验证：
  - GroupKFold（按县分组）
  - 留一县（QXMC）
  - 留一市（SZMC）—— 判断「23县训练 -> 绵阳迁移」可行性的关键证据

用法（特征 CSV 就绪后）：
  python train_dachun_23counties.py
"""
import os, sys, pickle, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
import lightgbm as lgb
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from common import compute_feature_matrix

# ================= 配置 =================
FEATURES_CSV = os.path.join(BASE_DIR, "待训练数据大春", "features_23counties.csv")
LABEL_COL = "类别"               # 标签字段（水稻/玉米）
GROUP_COL = "QXMC"               # 县分组（留一县）
CITY_COL = "SZMC"                # 市分组（留一市，迁移代理验证）
SCENE_LABELS = ["P1", "P2", "P3", "P4"]   # 对应 5/6/7/8 月
AVAILABLE_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07",
                   "B08", "B8A", "B11", "B12"]   # 10 波段

OUT_DIR = os.path.join(BASE_DIR, "dachun_23counties")
MODEL_OUT = os.path.join(OUT_DIR, "crop_model_binary_23counties.pkl")
REPORT_DIR = os.path.join(OUT_DIR, "report")

CLASSES = ["水稻", "玉米"]


def clean_mask(X, y):
    """返回最终保留样本的布尔掩码（NaN / IQR 异常值 / 重复样本）。

    掩码长度 = 输入 X 的行数，供 main 里统一过滤 X/y/groups/cities。
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    n = len(X)

    # 1) NaN
    mask = ~np.isnan(X).any(axis=1)

    # 2) IQR 异常值（原始波段前一半，沿用 v3/遂宁版）
    cur = np.where(mask)[0]
    Xc = X[mask]
    q1, q3 = np.percentile(Xc[:, :Xc.shape[1] // 2], [25, 75], axis=0)
    iqr = q3 - q1
    outlier = np.any((Xc[:, :Xc.shape[1] // 2] > q3 + 3 * iqr) |
                     (Xc[:, :Xc.shape[1] // 2] < q1 - 3 * iqr), axis=1)
    mask[cur[outlier]] = False

    # 3) 重复样本
    cur = np.where(mask)[0]
    Xr = np.round(X[mask], 2)
    _, first = np.unique(Xr, axis=0, return_index=True)
    keep = np.zeros(len(Xr), dtype=bool)
    keep[first] = True
    mask[cur[~keep]] = False

    return mask


def make_model():
    return lgb.LGBMClassifier(
        n_estimators=300, max_depth=5, num_leaves=31, learning_rate=0.03,
        subsample=0.7, colsample_bytree=0.6, reg_alpha=1.0, reg_lambda=2.0,
        min_child_samples=20, min_child_weight=0.001,
        objective="binary", random_state=42, n_jobs=-1, verbose=-1)


def evaluate_spatial(X_sel, y, groups, title):
    """对给定分组跑 GroupKFold + 留一组验证，返回结果字典。"""
    print(f"\n[{'=' * 20} {title} {'=' * 20}]")
    print(f"分组数: {len(set(groups))}")

    # 1) GroupKFold
    gkf = GroupKFold(n_splits=min(5, len(set(groups))))
    accs, f1s = [], []
    for tr, te in gkf.split(X_sel, y, groups):
        m = make_model()
        m.fit(X_sel[tr], y[tr])
        p = m.predict(X_sel[te])
        accs.append(accuracy_score(y[te], p))
        f1s.append(f1_score(y[te], p, average="weighted"))
    print(f"  GroupKFold Acc: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"  GroupKFold F1 : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

    # 2) Leave-One-Group-Out
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

    return {
        "gkf_acc_mean": float(np.mean(accs)),
        "gkf_acc_std": float(np.std(accs)),
        "gkf_f1_mean": float(np.mean(f1s)),
        "logo_acc": logo_acc,
        "logo_f1": logo_f1,
        "logo_acc_mean": float(np.mean(list(logo_acc.values()))),
    }


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    print("=" * 64)
    print("23县库 水稻/玉米二分类（空间分组验证）")
    print("=" * 64)

    df = pd.read_csv(FEATURES_CSV, encoding="utf-8-sig")
    print(f"样本: {len(df)}")

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

    # 标签 / 分组（县、市）派生（原始长度）
    le = LabelEncoder()
    y = le.fit_transform(df[LABEL_COL].values)
    groups = df[GROUP_COL].astype(str).values
    cities = df[CITY_COL].astype(str).values if CITY_COL in df.columns else None
    print(f"类别: {le.classes_.tolist()} -> {list(le.transform(le.classes_))}")

    # 统一清洗过滤（X/y/groups/cities 同步）
    mask = clean_mask(X, y)
    X, y, groups = X[mask], y[mask], groups[mask]
    if cities is not None:
        cities = cities[mask]
    print(f"预处理后: {len(X)} 样本, {len(set(groups))} 个县"
          + (f", {len(set(cities))} 个市" if cities is not None else ""))

    # 特征选择（训练集整体，仅用于降维）
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

    results = {}
    results["county"] = evaluate_spatial(X_sel, y, groups, "留一县 (QXMC)")
    if cities is not None:
        results["city"] = evaluate_spatial(X_sel, y, cities, "留一市 (SZMC)")

    # 全量训练 + 保存
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
        "city_col": CITY_COL,
        "label_col": LABEL_COL,
        "class_names": le.classes_.tolist(),
        "model_type": "lightgbm_binary_spatial_23counties",
        "results": results,
    }
    with open(MODEL_OUT, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\n模型已保存: {MODEL_OUT}")

    imp_df = pd.DataFrame({"feature": feat_sel, "importance": imp[top_idx]})
    imp_df = imp_df.sort_values("importance", ascending=False)
    imp_df.to_csv(os.path.join(REPORT_DIR, "feature_importance.csv"),
                  index=False, encoding="utf-8-sig")
    print(f"特征重要性已保存: {REPORT_DIR}/feature_importance.csv")


if __name__ == "__main__":
    main()
