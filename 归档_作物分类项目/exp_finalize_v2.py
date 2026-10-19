# -*- coding: utf-8 -*-
"""
小春成果固化 v2：172维 + WRI花敏感指数 + 安居区标签清洗
====================================================
在 v1(exp_finalize.py) 基础上加入 WRI/GRVI/NDYI/VARI/NDPI 花敏感指数，
整合标签清洗，输出最终模型 + 可疑清单 + 性能报告。
"""
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

BASE = r'e:\工作相关\2026年\0624 待测试数据'
from common import compute_feature_matrix
from exp_flower_indices import flower_features

BAND_MAP = {'B2': 'B02', 'B3': 'B03', 'B4': 'B04', 'B5': 'B05', 'B6': 'B06',
            'B7': 'B07', 'B8': 'B08', 'B8A': 'B8A', 'B11': 'B11', 'B12': 'B12'}
PERIOD_MAP = {'P1': '2024-12', 'P2': '2025-01', 'P3': '2025-03', 'P4': '2025-05'}
BANDS_ALL = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12']
THRESH = 0.7


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


def build_X_full(df):
    bv = df_to_band_values(df)
    scenes = list(PERIOD_MAP.values())
    X_base, names_base = compute_feature_matrix(bv, scenes, available_bands=BANDS_ALL)
    X_flower, names_flower = flower_features(bv, scenes)
    return np.hstack([X_base, X_flower]), list(names_base) + list(names_flower)


def main():
    dfs = []
    for c in ['gee_遂宁3区_特征.csv', 'gee_广安5区_特征.csv']:
        dfs.append(pd.read_csv(os.path.join(BASE, c), encoding='utf-8-sig'))
    df = pd.concat(dfs, ignore_index=True)

    le = LabelEncoder()
    le.fit(df['ZWMC'].values)

    X, names = build_X_full(df)
    valid = ~np.isnan(X).any(axis=1)
    df = df.iloc[valid].reset_index(drop=True)
    X = X[valid]
    y = le.transform(df['ZWMC'].values)
    print(f'特征矩阵: {X.shape}')

    # 非安居区训练 -> 预测安居区概率
    is_aj = df['QXMC'].values == '安居区'
    Xtr, _ = build_X_full(df[~is_aj])
    ytr = le.transform(df[~is_aj]['ZWMC'].values)
    cleaner = make_model().fit(Xtr, ytr)
    Xaj, _ = build_X_full(df[is_aj])
    yaj = le.transform(df[is_aj]['ZWMC'].values)
    prob_aj = cleaner.predict_proba(Xaj)[:, 1]

    flip_w2r = (yaj == 0) & (prob_aj > THRESH)
    flip_r2w = (yaj == 1) & (prob_aj < 1 - THRESH)

    # 导出可疑清单
    aj_df = df[is_aj].reset_index(drop=True)
    aj_rows = np.where(is_aj)[0]
    sus_rows = []
    for k in range(len(aj_df)):
        if flip_w2r[k] or flip_r2w[k]:
            sus_rows.append({
                'fid': aj_df.iloc[k]['fid'], 'QXMC': aj_df.iloc[k]['QXMC'],
                '原标签': aj_df.iloc[k]['ZWMC'],
                '建议标签': '油菜' if flip_w2r[k] else '小麦',
                'P_油菜': round(float(prob_aj[k]), 4),
                '置信度': '高' if abs(prob_aj[k] - 0.5) > 0.4 else '中',
            })
    sus_df = pd.DataFrame(sus_rows)
    sus_path = os.path.join(BASE, '安居区_可疑标签清单.csv')
    sus_df.to_csv(sus_path, index=False, encoding='utf-8-sig')
    print(f'[可疑清单] {sus_path}: {len(sus_df)} 条 (小麦→油菜 {int(flip_w2r.sum())}, 油菜→小麦 {int(flip_r2w.sum())})')

    # 清洗后标签
    y_cleaned = y.copy()
    for k, ridx in enumerate(aj_rows):
        if flip_w2r[k]:
            y_cleaned[ridx] = 1
        elif flip_r2w[k]:
            y_cleaned[ridx] = 0

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(make_model(), X, y_cleaned, cv=skf)
    cm = confusion_matrix(y_cleaned, y_pred)
    acc = (y_cleaned == y_pred).mean()
    print(f'\n[WRI+清洗 5折CV] Acc = {acc:.4f}')
    print('混淆矩阵 (行=真实[0小麦,1油菜], 列=预测):')
    print(cm)
    for i, c in enumerate(le.classes_):
        rec = cm[i, i] / cm[i].sum() * 100
        prec = cm[i, i] / cm[:, i].sum() * 100 if cm[:, i].sum() > 0 else 0
        print(f'  {c}: recall={rec:.1f}%  precision={prec:.1f}%  (n={cm[i].sum()})')

    # 最终模型 + Top 特征
    final_model = make_model().fit(X, y_cleaned)
    imp = final_model.booster_.feature_importance(importance_type='gain')
    top = np.argsort(imp)[::-1][:15]
    print('\n[最终模型 Top15 特征]')
    for i in top:
        print(f'  {names[i]}: {imp[i]:.0f}')

    pkg = {
        'model': final_model, 'label_encoder': le, 'feature_names': names,
        'period_map': PERIOD_MAP, 'band_map': BAND_MAP, 'thresh': THRESH,
        'n_flip': int((y_cleaned != y).sum()),
        'has_flower_indices': True,
    }
    pkg_path = os.path.join(BASE, '小春_小麦油菜_清洗后模型.pkl')
    with open(pkg_path, 'wb') as f:
        pickle.dump(pkg, f)
    print(f'\n[模型已保存] {pkg_path}')


if __name__ == '__main__':
    main()
