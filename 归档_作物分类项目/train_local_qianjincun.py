# -*- coding: utf-8 -*-
"""train_local_qianjincun.py — 前进村本地水稻/玉米二分类训练（补江油样本，消除域差）

用前进村 dltb 真实地类（水田=水稻 / 旱地=玉米）在 SLIC 超像素对象上打标签，
复用新版特征工程（含 MNDWI + 移栽期联合淹水信号），训练 LightGBM 二分类。

产出：dachun_binary_qianjincun_local.pkl（供 predict_slic_overlay.py 推理）
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
CLASSES = ['水稻', '玉米']  # 0=水稻(水田) 1=玉米(旱地)
OUT_PKL = os.path.join(BASE_DIR, 'dachun_binary_qianjincun_local.pkl')


def make_model():
    return lgb.LGBMClassifier(
        n_estimators=300, max_depth=4, num_leaves=15, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.6, reg_alpha=1.0, reg_lambda=2.0,
        min_child_samples=15, class_weight='balanced',
        objective='binary', random_state=42, n_jobs=-1, verbose=-1)


def main():
    slic = gpd.read_file(os.path.join(BASE_DIR, 'slic_objects_qianjincun.gpkg'))
    dltb = gpd.read_file(os.path.join(BASE_DIR, '待测试数据前进0806', '前进0806.gdb'), layer='dltb')
    dltb = dltb[dltb['ZLDWMC'] == '前进村'].copy()
    # 0=水稻(水田) 1=玉米(旱地)，其余地类剔除
    dltb = dltb[dltb['DLMC'].isin(['水田', '旱地'])].copy()
    dltb['y'] = (dltb['DLMC'] == '旱地').astype(int)

    # 超像素质心 -> dltb 图斑，打标签
    sc = slic.to_crs(4523).copy()
    sc['geometry'] = sc.geometry.centroid
    j = gpd.sjoin(sc[['label', 'geometry']], dltb[['y', 'geometry']],
                  how='left', predicate='within')
    j = j[~j.index.duplicated(keep='first')]
    slic['y'] = j['y'].reindex(slic.index).values
    slic = slic.dropna(subset=['y']).copy()
    slic['y'] = slic['y'].astype(int)
    print(f'本地样本超像素: 水稻(水田)={int((slic["y"] == 0).sum())}, '
          f'玉米(旱地)={int((slic["y"] == 1).sum())}')

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
    y = slic['y'].values
    print(f'特征数={len(fnames)}')

    # 5 折 CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, f1s = [], []
    yt_all, yp_all = [], []
    for tr, te in skf.split(X, y):
        m = make_model(); m.fit(X[tr], y[tr]); p = m.predict(X[te])
        accs.append(accuracy_score(y[te], p))
        f1s.append(f1_score(y[te], p, average='weighted'))
        yt_all.append(y[te]); yp_all.append(p)
    yt = np.concatenate(yt_all); yp = np.concatenate(yp_all)
    cm = confusion_matrix(yt, yp)  # 行=真实(0水稻,1玉米)
    print(f'\n5折CV: Acc={np.mean(accs):.4f}  F1(weighted)={np.mean(f1s):.4f}')
    print('混淆矩阵 [行=真实(0水稻,1玉米) 列=预测]:')
    print(cm)
    rec_rice = cm[0, 0] / cm[0].sum() if cm[0].sum() > 0 else 0
    prec_rice = cm[0, 0] / cm[:, 0].sum() if cm[:, 0].sum() > 0 else 0
    rec_corn = cm[1, 1] / cm[1].sum() if cm[1].sum() > 0 else 0
    prec_corn = cm[1, 1] / cm[:, 1].sum() if cm[:, 1].sum() > 0 else 0
    print(f'  水稻 recall={rec_rice:.3f} precision={prec_rice:.3f} (n={cm[0].sum()})')
    print(f'  玉米 recall={rec_corn:.3f} precision={prec_corn:.3f} (n={cm[1].sum()})')

    # 全量训练 + 保存
    model = make_model()
    model.fit(X, y)
    bundle = {
        'model': model,
        'model_type': 'dachun_binary_qianjincun_local',
        'class_names': CLASSES,
        'label_encoder': None,
        'selected_features': fnames,
        'phases': PHASES,
        'bands': BANDS_FEAT,
    }
    with open(OUT_PKL, 'wb') as f:
        pickle.dump(bundle, f)
    print(f'\n[saved] {OUT_PKL}')

    # 特征重要性 Top20
    imp = sorted(zip(fnames, model.feature_importances_), key=lambda x: -x[1])[:20]
    print('\nTop20 特征重要性:')
    for name, v in imp:
        print(f'  {name:<24} {v:.4f}')


if __name__ == '__main__':
    main()
