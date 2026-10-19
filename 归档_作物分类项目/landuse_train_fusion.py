# -*- coding: utf-8 -*-
"""
融合训练：无人机 RGB + 哨兵 NIR/红边/SWIR 波段，重训用地四大类
================================================================
对比特征组合：
  A 纯 RGB（baseline 19列）
  B RGB + 哨兵 8 波段原始值
  C RGB + 哨兵指数（NDVI/NDRE/NDWI）
  D RGB + 哨兵波段 + 指数（全量）
评估：四大类 5折CV + 按村留一（跨村泛化）
"""
import os
import glob
from collections import defaultdict
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import box
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_predict, LeaveOneGroupOut
from sklearn.metrics import confusion_matrix, accuracy_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

import landuse_train_multi as L

SENT_CSV = r'e:\工作相关\2026年\0624 待测试数据\gee_6村_哨兵波段_202507.csv'


def extract_rgb(gdf):
    """复用 L 的 assign + extract_sub，返回带 fid 的 RGB 特征表。"""
    imgs = sorted(glob.glob(os.path.join(L.IMG_DIR, '*', '*.tif')))
    img_boxes = []
    for p in imgs:
        with rasterio.open(p) as src:
            b = src.bounds
        img_boxes.append((p, box(b.left, b.bottom, b.right, b.top)))
    img_bounds = [bx.bounds for _, bx in img_boxes]
    assigned = {}
    for i, geom in enumerate(gdf.geometry):
        if geom is None or geom.is_empty:
            continue
        try:
            area = geom.area
        except Exception:
            continue
        if area <= 0:
            continue
        gb = geom.bounds
        best = None
        for (p, bx), ib in zip(img_boxes, img_bounds):
            if gb[2] <= ib[0] or gb[0] >= ib[2] or gb[3] <= ib[1] or gb[1] >= ib[3]:
                continue
            try:
                ratio = geom.intersection(bx).area / area
            except Exception:
                continue
            if ratio >= L.COVER and (best is None or ratio > best[1]):
                best = (p, ratio)
        if best:
            assigned[i] = best[0]

    by_img = defaultdict(list)
    for i, p in assigned.items():
        by_img[p].append(i)
    feat_parts, fid_parts = [], []
    for p, idxs in sorted(by_img.items()):
        res = L.extract_sub(gdf.loc[idxs], p)
        if res is None:
            continue
        feats, fids = res
        feat_parts.append(feats)
        fid_parts.append(fids)
    X = pd.concat(feat_parts, ignore_index=True)
    X['fid'] = np.concatenate(fid_parts)
    return X


def eval_combo(model_fn, Xm, y, groups, skf, logo):
    """返回 (acc5, cm5, acc_logo, cm_logo)。"""
    yp = cross_val_predict(model_fn(), Xm, y, cv=skf)
    cm5 = confusion_matrix(y, yp)
    yp2 = cross_val_predict(model_fn(), Xm, y, cv=logo, groups=groups)
    cml = confusion_matrix(y, yp2)
    return cm5.diagonal().sum() / cm5.sum(), cm5, cml.diagonal().sum() / cml.sum(), cml


def main():
    gdf = L.load_all()
    print('提取无人机 RGB 特征...')
    X_rgb = extract_rgb(gdf)
    print(f'  RGB 特征表: {X_rgb.shape}')

    sent = pd.read_csv(SENT_CSV)
    print(f'  哨兵波段表: {sent.shape}')
    # 计算指数（DN ×10000，比值抵消）
    eps = 1e-6
    sent['NDVI'] = (sent['JUL_B8'] - sent['JUL_B4']) / (sent['JUL_B8'] + sent['JUL_B4'] + eps)
    sent['NDRE'] = (sent['JUL_B8'] - sent['JUL_B5']) / (sent['JUL_B8'] + sent['JUL_B5'] + eps)
    sent['NDWI'] = (sent['JUL_B8'] - sent['JUL_B11']) / (sent['JUL_B8'] + sent['JUL_B11'] + eps)

    # 合并
    X = X_rgb.merge(sent, on='fid', how='inner')
    fids = X['fid'].values
    gdf = gdf.set_index('fid').loc[fids].reset_index(drop=True)
    X = X.reset_index(drop=True)
    print(f'  合并后: {X.shape}')

    # 哨兵数值列 fillna（小图斑可能无有效哨兵像素）
    sent_cols = [c for c in X.columns if c.startswith('JUL')]
    idx_cols = ['NDVI', 'NDRE', 'NDWI']
    for c in sent_cols + idx_cols:
        X[c] = X[c].fillna(X[c].median())

    valid = (X['_count'] >= 10).to_numpy()
    X = X[valid].reset_index(drop=True)
    gdf = gdf[valid].reset_index(drop=True)

    vc = gdf['YDFLEJ'].value_counts()
    keep = vc[vc >= L.MIN_SAMPLES].index.tolist()
    m = gdf['YDFLEJ'].isin(keep).to_numpy()
    X = X[m].reset_index(drop=True)
    gdf = gdf[m].reset_index(drop=True)
    print(f'  训练图斑: {len(gdf)} (类≥{L.MIN_SAMPLES})')

    # 四大类标签
    gdf['大类'] = gdf['YDFLEJ'].map(L.LARGE)
    le2 = LabelEncoder()
    y = le2.fit_transform(gdf['大类'].values)
    le_v = LabelEncoder()
    groups = le_v.fit_transform(gdf['村'].values)

    drop = ['fid', '村', '_count', 'YDFLEJ', '大类']
    rgb_cols = [c for c in X.columns if c not in drop
                and not c.startswith('JUL') and c not in idx_cols]
    sent_raw = [c for c in X.columns if c.startswith('JUL')]

    def model_fn():
        return lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                                  min_child_samples=10, class_weight='balanced',
                                  random_state=42, verbose=-1)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    logo = LeaveOneGroupOut()

    combos = {
        'A 纯RGB': rgb_cols,
        'B RGB+哨兵8波段': rgb_cols + sent_raw,
        'C RGB+哨兵指数': rgb_cols + idx_cols,
        'D RGB+波段+指数': rgb_cols + sent_raw + idx_cols,
    }

    print(f'\n===== 四大类对比 (类别={list(le2.classes_)}) =====')
    print(f'{"组合":<20} {"5折Acc":>8} {"跨村Acc":>8}  各组合跨村 recall')
    results = {}
    for name, cols in combos.items():
        Xm = X[cols].values
        acc5, cm5, accl, cml = eval_combo(model_fn, Xm, y, groups, skf, logo)
        results[name] = (acc5, accl, cml)
        rec = [cml[i, i] / cml[i].sum() * 100 if cml[i].sum() > 0 else 0 for i in range(len(le2.classes_))]
        rec_s = ' '.join(f'{le2.classes_[i][:2]}={rec[i]:.0f}%' for i in range(len(le2.classes_)))
        print(f'{name:<20} {acc5*100:>7.1f}% {accl*100:>7.1f}%  {rec_s}')

    # 打印 D 组合（全量）的完整混淆矩阵
    best_name = max(results, key=lambda k: results[k][1])
    print(f'\n===== 最优跨村组合: {best_name} 按村留一混淆矩阵 =====')
    cml = results[best_name][2]
    names = list(le2.classes_)
    print('        ' + ' '.join(f'{n[:4]:>6}' for n in names))
    for i, row in enumerate(cml):
        print(f'{names[i][:4]:>8}' + ' '.join(f'{v:>6}' for v in row))
    for i, c in enumerate(names):
        rec = cml[i, i] / cml[i].sum() * 100 if cml[i].sum() > 0 else 0
        prec = cml[i, i] / cml[:, i].sum() * 100 if cml[:, i].sum() > 0 else 0
        print(f'  {c:<10} recall={rec:5.1f}%  precision={prec:5.1f}%  (n={cml[i].sum()})')


if __name__ == '__main__':
    main()
