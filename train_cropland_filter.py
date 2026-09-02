# -*- coding: utf-8 -*-
"""train_cropland_filter.py — 纯哨兵「耕地/非耕地」过滤器训练

用前进村 dltb 真实地类 + SLIC 超像素哨兵特征，训练 LightGBM 二分类。
耕地=旱地/水田/水浇地/后备耕地；非耕地=林地/园地/草地/建筑/道路/水域。

产出：cropland_filter_qianjincun.pkl（供 SLIC 链路过滤非耕地对象）
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from common import compute_feature_matrix

import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

BANDS_FEAT = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
PHASES = ["D04", "D11", "D14"]
BAND_MAP = {'B2': 'B02', 'B3': 'B03', 'B4': 'B04', 'B5': 'B05', 'B6': 'B06',
            'B7': 'B07', 'B8': 'B08', 'B8A': 'B8A', 'B11': 'B11', 'B12': 'B12'}
CROPLAND = ['旱地', '水田', '水浇地', '后备耕地']
OUT_PKL = os.path.join(BASE_DIR, 'cropland_filter_qianjincun.pkl')


def make_model():
    return lgb.LGBMClassifier(
        n_estimators=300, max_depth=5, num_leaves=31, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.7, min_child_samples=20,
        class_weight='balanced', objective='binary', random_state=42,
        n_jobs=-1, verbose=-1)


def main():
    slic = gpd.read_file(os.path.join(BASE_DIR, 'slic_objects_qianjincun.gpkg'))
    dltb = gpd.read_file(os.path.join(BASE_DIR, '待测试数据前进0806', '前进0806.gdb'), layer='dltb')
    dltb = dltb[dltb['ZLDWMC'] == '前进村'].copy()
    dltb['is_crop'] = dltb['DLMC'].isin(CROPLAND).astype(int)

    # 超像素质心 -> dltb 图斑，打标签
    sc = slic.to_crs(4523).copy()
    sc['geometry'] = sc.geometry.centroid
    j = gpd.sjoin(sc[['label', 'geometry']], dltb[['is_crop', 'geometry']],
                  how='left', predicate='within')
    j = j[~j.index.duplicated(keep='first')]
    slic['is_crop'] = j['is_crop'].reindex(slic.index).values
    slic = slic.dropna(subset=['is_crop']).copy()
    slic['is_crop'] = slic['is_crop'].astype(int)
    print(f'有标签超像素: 耕地={int((slic["is_crop"] == 1).sum())}, '
          f'非耕地={int((slic["is_crop"] == 0).sum())}')

    # 波段名归一化 D04_B2 -> D04_B02
    rename = {}
    for c in slic.columns:
        if '_' in c:
            p, b = c.rsplit('_', 1)
            if b in BAND_MAP and p in PHASES:
                rename[c] = f'{p}_{BAND_MAP[b]}'
    slic = slic.rename(columns=rename)

    band_values = {}
    for p in PHASES:
        for b in BANDS_FEAT:
            col = f'{p}_{b}'
            if col in slic.columns:
                band_values[col] = slic[col].values
    X, fnames = compute_feature_matrix(band_values, PHASES, available_bands=BANDS_FEAT)
    y = slic['is_crop'].values
    print(f'特征数={len(fnames)}')

    # 5 折 CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, f1s = [], []
    y_true_all, y_pred_all = [], []
    for tr, te in skf.split(X, y):
        m = make_model(); m.fit(X[tr], y[tr]); p = m.predict(X[te])
        accs.append(accuracy_score(y[te], p))
        f1s.append(f1_score(y[te], p, average='weighted'))
        y_true_all.append(y[te]); y_pred_all.append(p)
    yt = np.concatenate(y_true_all); yp = np.concatenate(y_pred_all)
    cm = confusion_matrix(yt, yp)  # 行=真实(0非耕地,1耕地), 列=预测
    print(f'\n5折CV: Acc={np.mean(accs):.4f}  F1(weighted)={np.mean(f1s):.4f}')
    print('混淆矩阵 [行=真实(0非耕地,1耕地) 列=预测]:')
    print(cm)
    rec_crop = cm[1, 1] / cm[1].sum()
    prec_crop = cm[1, 1] / cm[:, 1].sum()
    rec_nc = cm[0, 0] / cm[0].sum()
    prec_nc = cm[0, 0] / cm[:, 0].sum()
    print(f'  耕地   recall={rec_crop:.3f} precision={prec_crop:.3f} (n={cm[1].sum()})')
    print(f'  非耕地 recall={rec_nc:.3f} precision={prec_nc:.3f} (n={cm[0].sum()})')

    # 全量训练 + 保存
    model = make_model()
    model.fit(X, y)
    bundle = {
        'model': model,
        'model_type': 'cropland_filter_qianjincun',
        'feature_names': fnames,
        'phases': PHASES,
        'bands': BANDS_FEAT,
        'class_names': ['非耕地', '耕地'],
    }
    with open(OUT_PKL, 'wb') as f:
        pickle.dump(bundle, f)
    print(f'\n[saved] {OUT_PKL}')

    # 特征重要性 Top15
    imp = sorted(zip(fnames, model.feature_importances_), key=lambda x: -x[1])[:15]
    print('\nTop15 特征重要性:')
    for name, v in imp:
        print(f'  {name:<22} {v:.4f}')


if __name__ == '__main__':
    main()
