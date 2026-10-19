# -*- coding: utf-8 -*-
"""
train_cropland_filter_general.py — 泛化「作物 / 非作物」过滤器训练
================================================================
正样本（作物=1）：22县作物(水稻/玉米) + 9村作物(险标的)
负样本（非作物=0）：9村非作物(林地/建筑/水面/草地等)
负样本按地类降采样林地，避免林地(90%)淹没建筑/水面。

产出：cropland_filter_general.pkl
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd

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
# 林地类（负样本中占绝对多数，需降采样）
FOREST = ['A0300', '乔木林地', '灌木林地', '其他林地', '竹林地']
FOREST_MAX = 30000
OUT_PKL = os.path.join(BASE_DIR, 'cropland_filter_general.pkl')


def make_model():
    return lgb.LGBMClassifier(
        n_estimators=300, max_depth=5, num_leaves=31, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.7, min_child_samples=20,
        class_weight='balanced', objective='binary', random_state=42,
        n_jobs=-1, verbose=-1)


def load_and_rename(path):
    df = pd.read_csv(path)
    rename = {}
    for c in df.columns:
        if '_' in c:
            p, b = c.rsplit('_', 1)
            if b in BAND_MAP and p in PHASES:
                rename[c] = f'{p}_{BAND_MAP[b]}'
    return df.rename(columns=rename)


def to_features(df):
    band_values = {}
    for p in PHASES:
        for b in BANDS_FEAT:
            col = f'{p}_{b}'
            if col in df.columns:
                band_values[col] = df[col].values
    X, fnames = compute_feature_matrix(band_values, PHASES, available_bands=BANDS_FEAT)
    return X, fnames


def main():
    # 正样本
    pos22 = load_and_rename(os.path.join(BASE_DIR, 'gee_dachun_23counties_3dekad.csv'))
    pos9 = load_and_rename(os.path.join(BASE_DIR, '正样本集_作物_9村.csv'))
    # 负样本
    neg9 = load_and_rename(os.path.join(BASE_DIR, '负样本集_非作物_9村.csv'))

    # 负样本按地类降采样林地
    neg_forest = neg9[neg9['地类'].isin(FOREST)]
    neg_other = neg9[~neg9['地类'].isin(FOREST)]
    if len(neg_forest) > FOREST_MAX:
        neg_forest = neg_forest.sample(FOREST_MAX, random_state=42)
    neg = pd.concat([neg_forest, neg_other], ignore_index=True)

    pos = pd.concat([pos22, pos9], ignore_index=True)
    print(f'正样本: 22县={len(pos22)} + 9村={len(pos9)} = {len(pos)}')
    print(f'负样本: 林地={len(neg_forest)}(降采样) + 其他={len(neg_other)} = {len(neg)}')

    Xp, fnames = to_features(pos)
    Xn, _ = to_features(neg)
    X = np.vstack([Xp, Xn])
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))]).astype(int)
    print(f'特征数={len(fnames)}, 总样本={len(y)} (正={int(y.sum())}, 负={int((y == 0).sum())})')

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
    cm = confusion_matrix(yt, yp)  # 行=真实(0非作物,1作物) 列=预测
    print(f'\n5折CV: Acc={np.mean(accs):.4f}  F1(weighted)={np.mean(f1s):.4f}')
    print('混淆矩阵 [行=真实(0非作物,1作物) 列=预测]:')
    print(cm)
    rec_crop = cm[1, 1] / cm[1].sum()
    prec_crop = cm[1, 1] / cm[:, 1].sum()
    rec_nc = cm[0, 0] / cm[0].sum()
    prec_nc = cm[0, 0] / cm[:, 0].sum()
    print(f'  作物   recall={rec_crop:.3f} precision={prec_crop:.3f} (n={cm[1].sum()})')
    print(f'  非作物 recall={rec_nc:.3f} precision={prec_nc:.3f} (n={cm[0].sum()})')

    # 全量训练 + 保存
    model = make_model()
    model.fit(X, y)
    bundle = {
        'model': model,
        'model_type': 'cropland_filter_general',
        'feature_names': fnames,
        'phases': PHASES,
        'bands': BANDS_FEAT,
        'class_names': ['非作物', '作物'],
    }
    with open(OUT_PKL, 'wb') as f:
        pickle.dump(bundle, f)
    print(f'\n[saved] {OUT_PKL}')

    imp = sorted(zip(fnames, model.feature_importances_), key=lambda x: -x[1])[:15]
    print('\nTop15 特征重要性:')
    for name, v in imp:
        print(f'  {name:<24} {v:.4f}')


if __name__ == '__main__':
    main()
