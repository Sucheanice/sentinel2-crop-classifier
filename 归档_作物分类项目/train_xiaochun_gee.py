# -*- coding: utf-8 -*-
"""
train_xiaochun_gee.py — 用 GEE 提取的特征训练小春分类（小麦 vs 油菜）

输入：gee_*.csv（列 fid, ZWMC, P1_B2 ... P4_B12，共 40 波段列）
流程：波段名映射(B2->B02) + 日期映射(P1->2024-12) -> compute_feature_matrix(172维)
      -> LightGBM 二分类（类别平衡 + 分层CV + 留一县）

目标：诊断小麦/油菜可分离性，重点关注油菜召回（此前仅 56%）。
"""
import os, pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, LeaveOneGroupOut
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

BASE = r'e:\工作相关\2026年\0624 待测试数据'
from common import compute_feature_matrix

# GEE 波段名 -> 本地波段名（B2 -> B02 等）
BAND_MAP = {
    'B2': 'B02', 'B3': 'B03', 'B4': 'B04', 'B5': 'B05', 'B6': 'B06',
    'B7': 'B07', 'B8': 'B08', 'B8A': 'B8A', 'B11': 'B11', 'B12': 'B12',
}
# GEE 期次 -> 日期标签（仅作 key 前缀，不参与数值）
PERIOD_MAP = {'P1': '2024-12', 'P2': '2025-01', 'P3': '2025-03', 'P4': '2025-05'}
BANDS_ALL = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12']


def df_to_band_values(df):
    band_values = {}
    for col in df.columns:
        if '_B' not in col or col.startswith('ZWMC') or col == 'fid':
            continue
        period, band = col.split('_', 1)  # P1 / B2
        date = PERIOD_MAP[period]
        band_local = BAND_MAP[band]
        band_values[f'{date}_{band_local}'] = df[col].values
    return band_values


def load_gee(csv_path, district=None):
    """读 GEE 特征 CSV，映射到 {日期}_{波段} 格式，返回 (df, band_values)"""
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    if 'QXMC' in df.columns:
        df['district'] = df['QXMC']
    elif district is not None:
        df['district'] = district
    return df, df_to_band_values(df)


def build_X(df, band_values):
    scene_labels = list(PERIOD_MAP.values())
    X, names = compute_feature_matrix(band_values, scene_labels, available_bands=BANDS_ALL)
    return X, names


def train_and_eval(df_all, X, names, group_col=None, class_weight='balanced'):
    le = LabelEncoder()
    y = le.fit_transform(df_all['ZWMC'].values)
    cls = le.classes_.tolist()
    print(f'类别: {cls} -> {dict(zip(range(len(cls)), cls))}')
    print(f'分布: {pd.Series(y).value_counts().to_dict()}')

    # 缺失值剔除
    valid = ~np.isnan(X).any(axis=1)
    X = X[valid]; y = y[valid]; df_all = df_all.iloc[valid].reset_index(drop=True)
    print(f'有效样本: {len(y)} (剔除 {int((~valid).sum())} 个含NaN)')

    model = lgb.LGBMClassifier(
        n_estimators=300, max_depth=5, num_leaves=31,
        learning_rate=0.03, subsample=0.7, colsample_bytree=0.6,
        reg_alpha=1.0, reg_lambda=2.0, min_child_samples=20,
        class_weight=class_weight, random_state=42, n_jobs=-1, verbose=-1,
    )

    # 分层 5 折 CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acc = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
    f1w = cross_val_score(model, X, y, cv=skf, scoring='f1_weighted')
    print(f'\n[分层5折CV] Acc={acc.mean():.4f}±{acc.std():.4f}  F1w={f1w.mean():.4f}±{f1w.std():.4f}')

    # 留一县验证（若有县名）
    if group_col and group_col in df_all.columns:
        groups = df_all[group_col].values
        logo = LeaveOneGroupOut()
        acc_l = cross_val_score(model, X, y, cv=logo, groups=groups, scoring='accuracy')
        f1_l = cross_val_score(model, X, y, cv=logo, groups=groups, scoring='f1_weighted')
        print(f'[留一县] Acc={acc_l.mean():.4f}  F1w={f1_l.mean():.4f}')
        for g, a in zip(np.unique(groups), acc_l):
            print(f'   留出 {g}: Acc={a:.4f}')

    # 全量训练 + 每类 recall（用分层CV的交叉预测更稳）
    from sklearn.model_selection import cross_val_predict
    y_pred = cross_val_predict(model, X, y, cv=skf)
    print(f'\n[交叉预测] 混淆矩阵:')
    cm = confusion_matrix(y, y_pred)
    print(cm)
    for i, c in enumerate(cls):
        rec = cm[i, i] / cm[i].sum() * 100
        print(f'  {c} recall = {rec:.1f}% ({(y==i).sum()} 块)')

    # 训练最终模型
    model.fit(X, y)
    imp = model.booster_.feature_importance(importance_type='gain')
    top = np.argsort(imp)[::-1][:15]
    print('\nTop15 特征:')
    for i in top:
        print(f'  {names[i]}: {imp[i]:.0f}')

    return model, le, names


if __name__ == '__main__':
    # 遂宁 3 区 + 广安 5 区合并
    dfs = []
    for csv in ['gee_遂宁3区_特征.csv', 'gee_广安5区_特征.csv']:
        df, _ = load_gee(os.path.join(BASE, csv))
        dfs.append(df)
    df_all = pd.concat(dfs, ignore_index=True)
    print(f'合并样本: {len(df_all)}')
    print(f'作物分布:\n{df_all["ZWMC"].value_counts().to_string()}')
    print(f'县分布:\n{df_all["district"].value_counts().to_string()}')

    bv = df_to_band_values(df_all)
    X, names = build_X(df_all, bv)
    print(f'特征矩阵: {X.shape}')
    train_and_eval(df_all, X, names, group_col='district')
