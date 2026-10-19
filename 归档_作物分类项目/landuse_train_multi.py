# -*- coding: utf-8 -*-
"""
6 村合并用地分类试点训练（多村 + 多景高分影像）
================================================================
影像：F:\YXX\江油市影像（14 景，8.9cm RGB，EPSG 未写入但坐标=CGCS2000 高斯）
标注：F:\0421给yxx\提交成果\{村}\矢量\2{村}.shp（YDFLEJ 用地分类）
方法：逐影像 -> 图斑归属 -> 降采样 0.53m zonal 统计 -> LightGBM
评估：11 类细分类 / 四大类(农用地·其他水域·建设用地) / 按村留一跨村泛化
"""
import os
import glob
from collections import defaultdict
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.windows import from_bounds
from rasterio.features import rasterize
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from shapely.geometry import box
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_predict, LeaveOneGroupOut
from sklearn.metrics import confusion_matrix, accuracy_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

BASE = r'F:\0421给yxx\提交成果'
IMG_DIR = r'F:\YXX\江油市影像'
SCALE = 6
MIN_SAMPLES = 40
COVER = 0.8

NAME = {
    'A0101': '水田', 'A0103': '旱地', 'A0200': '园地', 'A0300': '林地',
    'A0400': '草地', 'A1107': '沟渠', 'A1202': '设施农地',
    'B1199': '其他水面', 'D0702': '农村宅基地', 'D1000': '交通运输', 'D9900': '其他建设',
}

# 《用地识别要求》四大类（一级类）
LARGE = {
    'A0101': '农用地', 'A0102': '农用地', 'A0103': '农用地', 'A0200': '农用地',
    'A0300': '农用地', 'A0400': '农用地', 'A0500': '农用地',
    'A1104': '农用地', 'A1107': '农用地', 'A1202': '农用地', 'C1203': '农用地',
    'B1199': '其他水域', 'B0506': '其他水域',
    'C1201': '其他土地', 'C1299': '其他土地',
    'D0702': '建设用地', 'D1000': '建设用地', 'D9900': '建设用地',
}

VILLAGES = ['前进村', '印坪村', '大岳村', '沉水村', '马阁寺村', '龙宫村']
FILES = ['2前进.shp', '2印坪.shp', '2大岳.shp', '2沉水.shp', '2马阁寺.shp', '2龙宫.shp']


def load_all():
    parts = []
    for v, f in zip(VILLAGES, FILES):
        shp = os.path.join(BASE, v, '矢量', f)
        if not os.path.exists(shp):
            print(f'[缺失] {v}/{f}')
            continue
        g = gpd.read_file(shp)
        g['村'] = v
        parts.append(g)
    gdf = pd.concat(parts, ignore_index=True)
    gdf['fid'] = range(len(gdf))
    return gdf


def extract_sub(gdf_sub, img_path):
    """提取子集图斑特征，返回 (DataFrame, fid数组)；窗口裁剪到影像内。"""
    sub = gdf_sub.reset_index(drop=True)
    with rasterio.open(img_path) as src:
        minx, miny, maxx, maxy = sub.total_bounds
        minx = max(minx, src.bounds.left)
        miny = max(miny, src.bounds.bottom)
        maxx = min(maxx, src.bounds.right)
        maxy = min(maxy, src.bounds.top)
        if maxx <= minx or maxy <= miny:
            return None
        win = from_bounds(minx, miny, maxx, maxy, transform=src.transform)
        out_h = max(int(win.height) // SCALE, 1)
        out_w = max(int(win.width) // SCALE, 1)
        data = src.read([1, 2, 3], window=win, out_shape=(3, out_h, out_w),
                        resampling=Resampling.average).astype(np.float32)
        lb, bb, rb, tb = rasterio.windows.bounds(win, src.transform)
        transform = from_origin(lb, tb, src.res[0] * SCALE, src.res[1] * SCALE)

    R, G, B = data[0], data[1], data[2]
    eps = 1e-6
    ExG = 2 * G - R - B
    NGRDI = (G - R) / (G + R + eps)
    VARI = (G - R) / (G + R - B + eps)
    GLI = (2 * G - R - B) / (2 * G + R + B + eps)
    gray = 0.299 * R + 0.587 * G + 0.114 * B
    grad = (np.abs(np.diff(gray, axis=0, append=gray[-1:])) +
            np.abs(np.diff(gray, axis=1, append=gray[:, -1:])))

    shapes = [(g, i + 1) for i, g in enumerate(sub.geometry)
              if g is not None and not g.is_empty]
    label_img = rasterize(shapes, out_shape=data.shape[1:], transform=transform,
                          fill=0, dtype='uint32')
    flat_lab = label_img.ravel()
    valid = flat_lab > 0
    lab = flat_lab[valid]
    n = len(sub)
    count = np.bincount(lab, minlength=n + 1)[1:n + 1]

    feats = {'_count': count}
    for name, arr in [('R', R), ('G', G), ('B', B), ('ExG', ExG), ('NGRDI', NGRDI),
                      ('VARI', VARI), ('GLI', GLI), ('gray', gray), ('grad', grad)]:
        f = arr.ravel()[valid]
        s1 = np.bincount(lab, weights=f, minlength=n + 1)[1:n + 1]
        s2 = np.bincount(lab, weights=f * f, minlength=n + 1)[1:n + 1]
        mean = s1 / np.where(count > 0, count, 1)
        var = np.maximum(s2 / np.where(count > 0, count, 1) - mean ** 2, 0)
        feats[name + '_mean'] = mean
        feats[name + '_std'] = np.sqrt(var)
    return pd.DataFrame(feats), sub['fid'].values


def report(cm, names, title, show_prec=True):
    print(f'\n===== {title} =====')
    print(f'Acc = {cm.diagonal().sum() / cm.sum():.4f}')
    print('混淆矩阵 (行=真实, 列=预测):')
    print('        ' + ' '.join(f'{n[:4]:>6}' for n in names))
    for i, row in enumerate(cm):
        print(f'{names[i][:4]:>8}' + ' '.join(f'{v:>6}' for v in row))
    print('各类 recall / precision:')
    for i, c in enumerate(names):
        rec = cm[i, i] / cm[i].sum() * 100 if cm[i].sum() > 0 else 0
        prec = cm[i, i] / cm[:, i].sum() * 100 if cm[:, i].sum() > 0 else 0
        if show_prec:
            print(f'  {c:<12} recall={rec:5.1f}%  precision={prec:5.1f}%  (n={cm[i].sum()})')
        else:
            print(f'  {c:<12} recall={rec:5.1f}%  (n={cm[i].sum()})')


def main():
    gdf = load_all()
    print(f'合并图斑: {len(gdf)}')
    vc = gdf['YDFLEJ'].value_counts()
    print('类别分布 (YDFLEJ):')
    print(vc.to_string())

    # 影像 bbox
    imgs = sorted(glob.glob(os.path.join(IMG_DIR, '*', '*.tif')))
    img_boxes = []
    for p in imgs:
        with rasterio.open(p) as src:
            b = src.bounds
        img_boxes.append((p, box(b.left, b.bottom, b.right, b.top)))

    # 图斑归属影像（bbox 粗筛 -> 精确 intersection，覆盖面积最大且 >= COVER）
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
            # bbox 快速排除
            if gb[2] <= ib[0] or gb[0] >= ib[2] or gb[3] <= ib[1] or gb[1] >= ib[3]:
                continue
            try:
                ratio = geom.intersection(bx).area / area
            except Exception:
                continue
            if ratio >= COVER and (best is None or ratio > best[1]):
                best = (p, ratio)
        if best:
            assigned[i] = best[0]

    print(f'\n可提取图斑: {len(assigned)} / {len(gdf)} (覆盖>={COVER*100:.0f}%)', flush=True)

    by_img = defaultdict(list)
    for i, p in assigned.items():
        by_img[p].append(i)

    feat_parts, fid_parts = [], []
    for p, idxs in sorted(by_img.items()):
        res = extract_sub(gdf.loc[idxs], p)
        if res is None:
            continue
        feats, fids = res
        feat_parts.append(feats)
        fid_parts.append(fids)
        print(f'  {os.path.basename(os.path.dirname(p))}: {len(idxs)} 图斑')

    X = pd.concat(feat_parts, ignore_index=True)
    fids = np.concatenate(fid_parts)
    X['fid'] = fids
    gdf = gdf.loc[fids].reset_index(drop=True)

    valid = (X['_count'] >= 10).to_numpy()
    X = X[valid].reset_index(drop=True)
    gdf = gdf[valid].reset_index(drop=True)
    print(f'有效图斑: {len(gdf)} (剔除空图斑 {int((~valid).sum())})')

    # 过滤样本不足的类
    vc = gdf['YDFLEJ'].value_counts()
    keep = vc[vc >= MIN_SAMPLES].index.tolist()
    m = gdf['YDFLEJ'].isin(keep).to_numpy()
    X = X[m].reset_index(drop=True)
    gdf = gdf[m].reset_index(drop=True)
    print(f'训练类 ({MIN_SAMPLES}+): {len(keep)} 类, {len(gdf)} 图斑')

    feat_cols = [c for c in X.columns if not c.startswith('_') and c != 'fid']
    Xm = X[feat_cols].values

    def make_model():
        return lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                                  min_child_samples=10, class_weight='balanced',
                                  random_state=42, verbose=-1)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 1) 细分类
    le = LabelEncoder()
    y = le.fit_transform(gdf['YDFLEJ'].values)
    y_pred = cross_val_predict(make_model(), Xm, y, cv=skf)
    cm = confusion_matrix(y, y_pred)
    report(cm, [NAME.get(c, c) for c in le.classes_], '11类细分类 5折CV')

    # 2) 四大类
    gdf['大类'] = gdf['YDFLEJ'].map(LARGE)
    le2 = LabelEncoder()
    y2 = le2.fit_transform(gdf['大类'].values)
    y_pred2 = cross_val_predict(make_model(), Xm, y2, cv=skf)
    cm2 = confusion_matrix(y2, y_pred2)
    report(cm2, le2.classes_, '四大类 5折CV')

    # 3) 四大类 按村留一（跨村泛化）
    le_v = LabelEncoder()
    groups = le_v.fit_transform(gdf['村'].values)
    logo = LeaveOneGroupOut()
    y_pred3 = cross_val_predict(make_model(), Xm, y2, cv=logo, groups=groups)
    cm3 = confusion_matrix(y2, y_pred3)
    report(cm3, le2.classes_, '四大类 按村留一(跨村泛化)')

    # 特征重要性
    model = make_model()
    model.fit(Xm, y)
    imp = sorted(zip(feat_cols, model.feature_importances_), key=lambda x: -x[1])
    print('\nTop10 特征重要性:')
    for name, v in imp[:10]:
        print(f'  {name:<14} {v}')

    # 保存
    import pickle
    out = {'model': model, 'label_encoder': le, 'large_encoder': le2,
           'feature_names': feat_cols, 'name_map': NAME, 'large_map': LARGE}
    with open('landuse_6village_model.pkl', 'wb') as f:
        pickle.dump(out, f)
    print('\n[保存] landuse_6village_model.pkl')


if __name__ == '__main__':
    main()
