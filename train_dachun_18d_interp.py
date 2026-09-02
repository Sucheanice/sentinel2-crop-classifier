# -*- coding: utf-8 -*-
"""train_dachun_18d_interp.py — 18 旬雨季缺失的时序插值补齐 + 上限试探

对每个地块、每个波段，沿 18 旬时间轴做线性插值（首尾用最近邻外推），
补齐雨季（6月）等几乎全被云覆盖造成的 NaN，对比补齐前后 LightGBM 三分类上限。
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
CSV = os.path.join(BASE_DIR, "gee_dachun_5counties_18d.csv")
OUT = os.path.join(BASE_DIR, "gee_dachun_5counties_18d_interp.csv")

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


def load(path=CSV):
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


def interp(df):
    """沿旬轴（每波段独立）做线性插值 + 首尾最近邻外推"""
    df = df.copy()
    cols = feat_cols()
    # 每波段 18 个旬值排成一排做时间插值
    for b in BANDS:
        bcols = [f"{d}_{b}" for d in DEKADS]
        sub = df[bcols]
        # 记录插值前的缺失情况
        df[bcols] = sub.interpolate(axis=1, method="linear", limit_direction="both")
    return df


def forward_selection(df, y, ranked):
    order = [d for d, _ in ranked]
    selected = []
    print(f"{'旬数':>4} | Acc")
    print("-" * 24)
    for k in range(1, len(order) + 1):
        selected += [f"{order[k-1]}_{b}" for b in BANDS]
        acc = cv_acc(df[selected].values, y)
        mark = " <<<" if k in (3, 5, 7, 9, 12, 15) else ""
        print(f"  top{k:2d} | {acc:.4f}{mark}")
    return


def main():
    df = load()
    y = df["类别"].values
    cols = feat_cols()

    # 补齐前 NaN 统计
    nan_before = df[cols].isna().any(axis=1).mean()
    print(f"补齐前：任一样本含 NaN 比例 = {nan_before*100:.1f}%")

    # 原始（NaN 交 LightGBM）18 旬全量 Acc
    acc_raw = cv_acc(df[cols].values, y)
    print(f"原始 18 旬（NaN 交 LightGBM）三分类 Acc = {acc_raw:.4f}")

    # 时序插值补齐
    df_i = interp(df)
    nan_after = df_i[cols].isna().any(axis=1).mean()
    print(f"补齐后：任一样本含 NaN 比例 = {nan_after*100:.1f}%")
    df_i.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"已写出 {os.path.basename(OUT)}")

    # 补齐后 18 旬全量 Acc
    acc_int = cv_acc(df_i[cols].values, y)
    print(f"补齐后 18 旬三分类 Acc = {acc_int:.4f}  (Δ {acc_int-acc_raw:+.4f})")

    # 补齐后重新算旬重要性 + 前向选择
    Xc = np.asarray(df_i[cols].values, dtype=np.float32)
    m = make_model(); m.fit(Xc, y)
    imp = m.booster_.feature_importance(importance_type="gain")
    dekad_imp = {}
    for i, d in enumerate(DEKADS):
        s = slice(i * len(BANDS), (i + 1) * len(BANDS))
        dekad_imp[d] = float(imp[s].sum())
    ranked = sorted(dekad_imp.items(), key=lambda kv: -kv[1])

    print("\n补齐后旬重要性排名:")
    total = sum(dekad_imp.values())
    for d, v in ranked:
        print(f"  {d} ({DEKAD_LABEL[d]}): {v/total*100:5.1f}%")

    print("\n补齐后前向选择（看 Acc 饱和点）:")
    forward_selection(df_i, y, ranked)


if __name__ == "__main__":
    main()
