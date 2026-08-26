# -*- coding: utf-8 -*-
"""
前进村用地分类试点训练：高分 RGB 影像 -> 面向对象图斑分类
================================================================
数据：F:\YXX\江油市影像\0715江油马角镇-1.tif (8.9cm RGB)
标注：F:\0421给yxx\提交成果\前进村\矢量\2前进.shp (YDFLEJ 三调用地分类)
方法：降采样到 0.5m -> 图斑 zonal 统计(RGB/可见光指数/纹理) -> LightGBM 多分类
"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.windows import from_bounds
from rasterio.features import rasterize
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

IMG = r'F:\YXX\江油市影像\0715江油马角镇-1\0715江油马角镇-1.tif'
SHP = r'F:\0421给yxx\提交成果\前进村\矢量\2前进.shp'
SCALE = 6          # 0.089m × 6 ≈ 0.53m
MIN_SAMPLES = 40   # 少于该样本的类别剔除

# 类别名映射
NAME = {
    'A0101': '水田', 'A0103': '旱地', 'A0200': '园地', 'A0300': '林地',
    'A0400': '草地', 'B1199': '其他水面', 'D0702': '农村宅基地',
    'D1000': '交通运输', 'D9900': '其他建设用地',
}


def extract_features(gdf):
    """对图斑提取 RGB 影像 zonal 统计特征。"""
    minx, miny, maxx, maxy = gdf.total_bounds
    with rasterio.open(IMG) as src:
        win = from_bounds(minx, miny, maxx, maxy, transform=src.transform)
        out_h = int(win.height) // SCALE
        out_w = int(win.width) // SCALE
        data = src.read([1, 2, 3], window=win,
                        out_shape=(3, out_h, out_w),
                        resampling=Resampling.average).astype(np.float32)
        lb, bb, rb, tb = rasterio.windows.bounds(win, src.transform)
        transform = from_origin(lb, tb, src.res[0] * SCALE, src.res[1] * SCALE)

    R, G, B = data[0], data[1], data[2]
    eps = 1e-6
    # 可见光指数
    ExG = 2 * G - R - B
    NGRDI = (G - R) / (G + R + eps)
    VARI = (G - R) / (G + R - B + eps)
    GLI = (2 * G - R - B) / (2 * G + R + B + eps)
    gray = 0.299 * R + 0.587 * G + 0.114 * B
    # 边缘密度（梯度）
    grad = (np.abs(np.diff(gray, axis=0, append=gray[-1:])) +
            np.abs(np.diff(gray, axis=1, append=gray[:, -1:])))

    # rasterize 图斑（fid+1 作为值）
    shapes = [(g, i + 1) for i, g in enumerate(gdf.geometry) if g is not None and not g.is_empty]
    label_img = rasterize(shapes, out_shape=data.shape[1:], transform=transform, fill=0, dtype='uint32')
    flat_lab = label_img.ravel()
    valid = flat_lab > 0
    lab = flat_lab[valid]
    n = len(gdf)
    count = np.bincount(lab, minlength=n + 1)[1:n + 1]

    feats = {}
    feats['_count'] = count
    for name, arr in [('R', R), ('G', G), ('B', B), ('ExG', ExG), ('NGRDI', NGRDI),
                      ('VARI', VARI), ('GLI', GLI), ('gray', gray), ('grad', grad)]:
        f = arr.ravel()[valid]
        s1 = np.bincount(lab, weights=f, minlength=n + 1)[1:n + 1]
        s2 = np.bincount(lab, weights=f * f, minlength=n + 1)[1:n + 1]
        mean = s1 / np.where(count > 0, count, 1)
        var = np.maximum(s2 / np.where(count > 0, count, 1) - mean ** 2, 0)
        std = np.sqrt(var)
        feats[name + '_mean'] = mean
        feats[name + '_std'] = std
    return pd.DataFrame(feats)


def main():
    gdf = gpd.read_file(SHP)
    print(f'前进村图斑: {len(gdf)}, 类别: {gdf["YDFLEJ"].nunique()} 类')

    # 过滤样本不足的类
    vc = gdf['YDFLEJ'].value_counts()
    keep = vc[vc >= MIN_SAMPLES].index.tolist()
    gdf = gdf[gdf['YDFLEJ'].isin(keep)].copy()
    print(f'保留 {len(keep)} 类, {len(gdf)} 图斑: {dict(vc[vc>=MIN_SAMPLES])}')

    # 提取特征
    print('提取特征中...')
    X = extract_features(gdf)
    valid = (X['_count'] >= 10).to_numpy()  # 图斑至少覆盖 10 个像元
    X = X[valid].reset_index(drop=True)
    gdf = gdf[valid].reset_index(drop=True)
    print(f'有效图斑: {len(gdf)} (剔除空图斑 {int((~valid).sum())})')

    le = LabelEncoder()
    y = le.fit_transform(gdf['YDFLEJ'].values)
    feat_cols = [c for c in X.columns if not c.startswith('_')]
    Xm = X[feat_cols].values

    # 训练 + 分层 CV
    model = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                               min_child_samples=10, class_weight='balanced',
                               random_state=42, verbose=-1)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(model, Xm, y, cv=skf)
    acc = accuracy_score(y, y_pred)
    print(f'\n===== 前进村用地分类 5折CV =====')
    print(f'Acc = {acc:.4f}')
    cm = confusion_matrix(y, y_pred)
    print('混淆矩阵 (行=真实, 列=预测):')
    names = [NAME.get(c, c) for c in le.classes_]
    print('        ' + ' '.join(f'{n[:4]:>6}' for n in names))
    for i, row in enumerate(cm):
        print(f'{names[i][:4]:>8}' + ' '.join(f'{v:>6}' for v in row))
    print('\n各类 recall / precision:')
    for i, c in enumerate(le.classes_):
        rec = cm[i, i] / cm[i].sum() * 100 if cm[i].sum() > 0 else 0
        prec = cm[i, i] / cm[:, i].sum() * 100 if cm[:, i].sum() > 0 else 0
        print(f'  {NAME.get(c, c):<10} recall={rec:5.1f}%  precision={prec:5.1f}%  (n={cm[i].sum()})')

    # 大类合并评估（4类：农用地/林地/水域/建设用地）
    LARGE = {
        'A0101': '农用地', 'A0103': '农用地', 'A0200': '农用地', 'A0400': '农用地',
        'A0300': '林地', 'B1199': '水域',
        'D0702': '建设用地', 'D1000': '建设用地', 'D9900': '建设用地',
    }
    gdf['大类'] = gdf['YDFLEJ'].map(LARGE)
    le2 = LabelEncoder()
    y2 = le2.fit_transform(gdf['大类'].values)
    model2 = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                                min_child_samples=10, class_weight='balanced',
                                random_state=42, verbose=-1)
    y_pred2 = cross_val_predict(model2, Xm, y2, cv=skf)
    acc2 = accuracy_score(y2, y_pred2)
    cm2 = confusion_matrix(y2, y_pred2)
    print(f'\n===== 4大类合并 5折CV =====')
    print(f'大类 Acc = {acc2:.4f}')
    print('混淆矩阵 (行=真实, 列=预测):')
    print('        ' + ' '.join(f'{n[:4]:>6}' for n in le2.classes_))
    for i, row in enumerate(cm2):
        print(f'{le2.classes_[i][:4]:>8}' + ' '.join(f'{v:>6}' for v in row))
    for i, c in enumerate(le2.classes_):
        rec = cm2[i, i] / cm2[i].sum() * 100 if cm2[i].sum() > 0 else 0
        print(f'  {c:<8} recall={rec:5.1f}%  (n={cm2[i].sum()})')

    # 特征重要性
    model.fit(Xm, y)
    imp = sorted(zip(feat_cols, model.feature_importances_), key=lambda x: -x[1])
    print('\nTop10 特征重要性:')
    for name, v in imp[:10]:
        print(f'  {name:<14} {v}')

    # 保存
    import pickle
    out = {'model': model, 'label_encoder': le, 'feature_names': feat_cols}
    with open('landuse_qianjin_model.pkl', 'wb') as f:
        pickle.dump(out, f)
    print('\n[保存] landuse_qianjin_model.pkl')


if __name__ == '__main__':
    main()
