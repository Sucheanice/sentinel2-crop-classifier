# -*- coding: utf-8 -*-
"""
花敏感指数验证：GRVI / NDYI / VARI / NDPI
========================================
油菜 3 月开黄花，靠可见光颜色信号区分小麦（返青绿）vs 油菜（开花黄）。
现有 172 维特征只有 NDVI/EVI/NDWI 等生物量指数，缺「花敏感指数」。

本脚本对比：
  A. 原 172 维特征（基线）
  B. 172 维 + 花敏感指数（每期4指数 + 时序统计 + 首尾差 = 40 维）
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

BASE = r'e:\工作相关\2026年\0624 待测试数据'
from common import compute_feature_matrix, compute_temporal_stats

BAND_MAP = {'B2': 'B02', 'B3': 'B03', 'B4': 'B04', 'B5': 'B05', 'B6': 'B06',
            'B7': 'B07', 'B8': 'B08', 'B8A': 'B8A', 'B11': 'B11', 'B12': 'B12'}
PERIOD_MAP = {'P1': '2024-12', 'P2': '2025-01', 'P3': '2025-03', 'P4': '2025-05'}
BANDS_ALL = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12']
EPS = 1e-10


def make_model():
    return lgb.LGBMClassifier(
        n_estimators=300, max_depth=5, num_leaves=31,
        learning_rate=0.03, subsample=0.7, colsample_bytree=0.6,
        reg_alpha=1.0, reg_lambda=2.0, min_child_samples=20,
        class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1,
    )


def df_to_band_values(df):
    bv = {}
    for col in df.columns:
        if '_B' not in col or col.startswith('ZWMC') or col == 'fid':
            continue
        period, band = col.split('_', 1)
        if period not in PERIOD_MAP:
            continue
        bv[f'{PERIOD_MAP[period]}_{BAND_MAP[band]}'] = df[col].values
    return bv


def flower_features(bv, scene_labels):
    """计算 4 个花敏感指数的每期值 + 时序统计 + 首尾差，返回 (X, names)。"""
    n = len(bv[f'{scene_labels[0]}_B03'])
    idx_names = ['GRVI', 'NDYI', 'VARI', 'NDPI', 'WRI1', 'WRI2']
    per_scene = {nm: [] for nm in idx_names}

    for lbl in scene_labels:
        blue = bv[f'{lbl}_B02']
        green = bv[f'{lbl}_B03']
        red = bv[f'{lbl}_B04']
        nir = bv[f'{lbl}_B08']
        swir1 = bv[f'{lbl}_B11']

        grvi = np.where(np.abs(green + red) > EPS, (green - red) / (green + red), 0.0)
        ndyi = np.where(np.abs(green + blue) > EPS, (green - blue) / (green + blue), 0.0)
        vari = np.where(np.abs(green + red - blue) > EPS,
                        (green - red) / (green + red - blue), 0.0)
        # NDPI (Wang et al. 2017): (NIR - (0.74R + 0.26SWIR)) / (NIR + (0.74R + 0.26SWIR))
        mix = 0.74 * red + 0.26 * swir1
        ndpi = np.where(np.abs(nir + mix) > EPS, (nir - mix) / (nir + mix), 0.0)
        # WRI (Tao et al. 2019): (NIR-Green)/(NIR+Green) * Blue/(Green+Red)
        # WRI1 = CARM30/Tao 原始; WRI2 = Agrosystems 2026 变体(Blue/(Green+Blue))
        denom = nir + green + EPS
        wri1 = np.where(np.abs(green + red) > EPS,
                        (nir - green) / denom * blue / (green + red), 0.0)
        wri2 = np.where(np.abs(green + blue) > EPS,
                        (nir - green) / denom * blue / (green + blue), 0.0)

        for nm, arr in zip(idx_names, [grvi, ndyi, vari, ndpi, wri1, wri2]):
            per_scene[nm].append(arr)

    cols, data = [], []
    # 每期原始值
    for nm in idx_names:
        for li, lbl in enumerate(scene_labels):
            data.append(per_scene[nm][li])
            cols.append(f'{nm}_{lbl}')
    # 时序统计
    for nm in idx_names:
        across = np.column_stack(per_scene[nm])
        stats = compute_temporal_stats(np.where(across == 0, np.nan, across))
        for sn in ['min', 'max', 'mean', 'std', 'range']:
            data.append(np.nan_to_num(stats[sn], nan=0.0))
            cols.append(f'TSTAT_{nm}_{sn}')
    # 首尾差
    for nm in idx_names:
        delta = per_scene[nm][-1] - per_scene[nm][0]
        data.append(delta)
        cols.append(f'DELTA_{nm}')

    return np.column_stack(data).astype(np.float32), cols


def eval_X(X, y, le, tag):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(make_model(), X, y, cv=skf)
    cm = confusion_matrix(y, y_pred)
    acc = (y == y_pred).mean()
    print(f'\n=== [{tag}] 特征={X.shape[1]} ===')
    print(f'Acc = {acc:.4f}')
    print('混淆矩阵 (行=真实[0小麦,1油菜], 列=预测):')
    print(cm)
    for i, c in enumerate(le.classes_):
        rec = cm[i, i] / cm[i].sum() * 100
        prec = cm[i, i] / cm[:, i].sum() * 100 if cm[:, i].sum() > 0 else 0
        print(f'  {c}: recall={rec:.1f}%  precision={prec:.1f}%  (n={cm[i].sum()})')
    return acc


def main():
    dfs = []
    for c in ['gee_遂宁3区_特征.csv', 'gee_广安5区_特征.csv']:
        dfs.append(pd.read_csv(os.path.join(BASE, c), encoding='utf-8-sig'))
    df = pd.concat(dfs, ignore_index=True)

    le = LabelEncoder()
    y = le.fit_transform(df['ZWMC'].values)
    bv = df_to_band_values(df)
    scene_labels = list(PERIOD_MAP.values())

    X_base, names_base = compute_feature_matrix(bv, scene_labels, available_bands=BANDS_ALL)
    X_flower, names_flower = flower_features(bv, scene_labels)

    valid = ~np.isnan(X_base).any(axis=1)
    X_base, y = X_base[valid], y[valid]
    X_flower = X_flower[valid]

    eval_X(X_base, y, le, 'A. 原172维(基线)')
    eval_X(np.hstack([X_base, X_flower]), y, le, 'B. 172+花敏感指数40维')

    # 只看花敏感指数单独能分多少
    eval_X(X_flower, y, le, 'C. 仅花敏感指数40维')

    # 特征重要性（含花敏感指数的模型）
    X_full = np.hstack([X_base, X_flower])
    model = make_model().fit(X_full, y)
    imp = model.booster_.feature_importance(importance_type='gain')
    all_names = list(names_base) + list(names_flower)
    top = np.argsort(imp)[::-1][:20]
    print('\n[含花敏感指数模型 Top20 特征]')
    for i in top:
        print(f'  {all_names[i]}: {imp[i]:.0f}')


if __name__ == '__main__':
    main()
