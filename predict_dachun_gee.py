# -*- coding: utf-8 -*-
"""predict_dachun_gee.py — 大春 GEE 预测（水稻/玉米二分类，关键3旬 23县模型）

与训练完全一致的 GEE 特征口径：S2_SR_HARMONIZED + SCL 云掩膜 + median 合成，
提取关键 3 旬（D04=5月上 / D11=7月中 / D14=8月中）× 10 波段 zonal mean，
并用 23 县关键3旬模型做推理。

输入：地块边界 shp（外部给边界，任意 CRS，自动转 WGS84 供 GEE）
输出：写回 shp/gpkg/csv，保留原始字段 + 以下 11 个质量字段（人工复核参考）：
  parcel_id / area_m2 / ndvi_max / ndvi_flag / predicted / prdct_lbl /
  max_prob / conf_flag / prdct_cf / buf_pixel / px_statu

用法：
  python predict_dachun_gee.py --input 地块.shp --output 结果目录 --model dachun_binary_23counties_3dekad.pkl
"""
import ee
import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd

ee.Initialize()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from common import compute_feature_matrix

# 关键 3 旬（与训练一致）
WINDOWS = [
    ('D04', '2025-05-01', '2025-05-10'),
    ('D11', '2025-07-11', '2025-07-20'),
    ('D14', '2025-08-11', '2025-08-20'),
]
PHASES = [p for p, _, _ in WINDOWS]

# GEE 输出波段名（B2/B3...）与特征工程口径（B02/B03...）映射
BANDS_GEE = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
BANDS_FEAT = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
BAND_MAP = dict(zip(BANDS_GEE, BANDS_FEAT))

BATCH_SIZE = 3000

# 三阈值（口径见工作记录 8.13 / 设计说明）
NDVI_VEG = 0.30
CONF_LOW = 0.60
CONF_HIGH = 0.70
PX_MIN = 3


def mask_s2_clouds(img):
    scl = img.select('SCL')
    cloud = scl.eq(3).Or(scl.eq(8)).Or(scl.eq(9)).Or(scl.eq(10))
    return img.updateMask(cloud.Not())


def build_composite(roi):
    s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(roi)
    composite = None
    for prefix, start, end in WINDOWS:
        col = s2.filterDate(start, end).map(mask_s2_clouds)
        med = col.select(BANDS_GEE).median().rename([f'{prefix}_{b}' for b in BANDS_GEE])
        composite = med if composite is None else composite.addBands(med)
    return composite


def extract_features(gdf_wgs84):
    """GEE 提取关键3旬 zonal mean + 有效像元计数，返回 DataFrame（fid + 波段 + buf_pixel）。"""
    band_names = [f'{p}_{b}' for p, _, _ in WINDOWS for b in BANDS_GEE]
    n = len(gdf_wgs84)

    rows = []
    nb = (n + BATCH_SIZE - 1) // BATCH_SIZE
    print(f'[extract] 共 {nb} 批，每批 {BATCH_SIZE}')
    for bi in range(nb):
        sub = gdf_wgs84.iloc[bi * BATCH_SIZE:(bi + 1) * BATCH_SIZE]
        sub_fc = ee.FeatureCollection(sub[['fid', 'geometry']].__geo_interface__['features'])
        composite = build_composite(sub_fc.geometry())

        reduced = composite.reduceRegions(
            collection=sub_fc, reducer=ee.Reducer.mean(), scale=10, tileScale=4,
        ).select(['fid'] + band_names)

        # 有效像元计数：三旬 B8 掩膜并集（任一旬有有效像元即算覆盖），sum=有效10m像素数
        valid_any = (composite.select(f'{WINDOWS[0][0]}_B8').mask()
                     .Or(composite.select(f'{WINDOWS[1][0]}_B8').mask())
                     .Or(composite.select(f'{WINDOWS[2][0]}_B8').mask()))
        reduced_cnt = valid_any.reduceRegions(
            collection=sub_fc, reducer=ee.Reducer.sum().unweighted(), scale=10, tileScale=4,
        )

        feats = reduced.getInfo()['features']
        cnts = {f['properties']['fid']: f['properties'].get('sum', 0)
                for f in reduced_cnt.getInfo()['features']}
        for f in feats:
            props = f['properties']
            fid = props['fid']
            row = {k: props.get(k) for k in ['fid'] + band_names}
            row['buf_pixel'] = cnts.get(fid, 0)
            rows.append(row)
        print(f'  batch {bi + 1}/{nb} done (累计 {len(rows)}/{n})')

    return pd.DataFrame(rows)


def area_m2(gdf):
    """计算地块面积（㎡）：投影 CRS 直接用，地理 CRS 转 UTM 再算。"""
    if gdf.crs is None:
        return pd.Series([np.nan] * len(gdf), index=gdf.index)
    if gdf.crs.is_projected:
        return gdf.geometry.area
    lon = gdf.geometry.centroid.x.mean()
    zone = int((lon + 180) // 6) + 1
    epsg = 32600 + zone  # 北半球 UTM
    return gdf.to_crs(epsg=epsg).geometry.area


def run_predict(gdf_orig, feat_df, bundle):
    """组装 11 个质量字段，返回带预测结果的 GeoDataFrame。"""
    df = feat_df.copy()

    # 1) 波段名归一化 D04_B2 -> D04_B02
    rename = {}
    for c in df.columns:
        if '_' in c:
            p, b = c.rsplit('_', 1)
            if b in BAND_MAP and p in PHASES:
                rename[c] = f'{p}_{BAND_MAP[b]}'
    df = df.rename(columns=rename)

    # 2) 特征工程（与训练一致）
    band_values = {}
    for p in PHASES:
        for b in BANDS_FEAT:
            col = f'{p}_{b}'
            if col in df.columns:
                band_values[col] = df[col].values
    X, fnames = compute_feature_matrix(band_values, PHASES, available_bands=BANDS_FEAT)
    X = pd.DataFrame(X, columns=fnames)

    selected = bundle.get('selected_features', list(X.columns))
    for m in [f for f in selected if f not in X.columns]:
        X[m] = 0.0
    X = X[selected].fillna(0).values.astype(np.float32)

    # 3) 推理
    model = bundle['model']
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    max_prob = np.max(y_proba, axis=1)
    class_names = bundle.get('class_names', ['水稻', '玉米'])
    prdct_lbl = np.array([class_names[i] for i in y_pred])

    # 4) 三阈值联合判断链
    buf_pixel = df['buf_pixel'].values
    # NDVI 三期最大值（NDVI=(B08-B04)/(B08+B04)）
    ndvi = []
    for p in PHASES:
        b8 = df.get(f'{p}_B08', pd.Series(np.zeros(len(df))))
        b4 = df.get(f'{p}_B04', pd.Series(np.zeros(len(df))))
        denom = b8.values.astype(float) + b4.values.astype(float)
        ndvi.append(np.where(denom > 1e-10, (b8.values.astype(float) - b4.values.astype(float)) / denom, 0.0))
    ndvi_max = np.nanmax(np.vstack(ndvi), axis=0)

    ndvi_flag = np.where(ndvi_max >= NDVI_VEG, '植被', '非植被')
    conf_flag = np.where(
        ndvi_max < NDVI_VEG, '非植被',
        np.where(max_prob >= CONF_HIGH, '高',
                 np.where(max_prob >= CONF_LOW, '中', '低')))
    prdct_cf = np.where(
        buf_pixel < 1, '无覆盖',
        np.where(ndvi_max < NDVI_VEG, '非植被(建筑/道路/裸土)',
                 np.where(max_prob < CONF_LOW, '不确定', prdct_lbl)))
    px_statu = np.where(buf_pixel < PX_MIN, '过小(<3px)', 'OK')

    # 5) 回填原 gdf（按 fid 对齐）
    out = gdf_orig.copy()
    fid_map = {fid: i for i, fid in enumerate(df['fid'].values)}
    idx = out['fid'].map(fid_map).values  # 每个原行对应 feat_df 行

    out['parcel_id'] = out['fid'] + 1
    out['area_m2'] = area_m2(out).values
    out['ndvi_max'] = np.round(ndvi_max[idx], 4)
    out['ndvi_flag'] = ndvi_flag[idx]
    out['predicted'] = y_pred[idx]
    out['prdct_lbl'] = prdct_lbl[idx]
    out['max_prob'] = np.round(max_prob[idx], 4)
    out['conf_flag'] = conf_flag[idx]
    out['prdct_cf'] = prdct_cf[idx]
    out['buf_pixel'] = buf_pixel[idx].astype(int)
    out['px_statu'] = px_statu[idx]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='地块边界 shp/gpkg 路径')
    ap.add_argument('--output', required=True, help='输出目录')
    ap.add_argument('--model', required=True, help='模型 pkl 路径')
    ap.add_argument('--name', default='dachun_prediction', help='输出文件前缀')
    args = ap.parse_args()

    with open(args.model, 'rb') as f:
        bundle = pickle.load(f)
    print(f"[model] {bundle.get('model_type')} 类别={bundle.get('class_names')} "
          f"特征数={len(bundle.get('selected_features', []))}")

    gdf = gpd.read_file(args.input)
    print(f'[input] {args.input}: {len(gdf)} 块, CRS={gdf.crs}')
    print(f'  字段: {list(gdf.columns)}')

    # 原始 CRS 保留（面积/写回用），工作副本转 WGS84 供 GEE
    gdf = gdf.reset_index(drop=True)
    gdf['fid'] = range(len(gdf))
    gdf_wgs84 = gdf.to_crs('EPSG:4326')
    gdf_wgs84['geometry'] = gdf_wgs84['geometry'].simplify(tolerance=0.0001, preserve_topology=True)

    feat_df = extract_features(gdf_wgs84)
    if feat_df.empty:
        print('!! 无有效特征')
        return

    out = run_predict(gdf, feat_df, bundle)

    os.makedirs(args.output, exist_ok=True)
    base = os.path.join(args.output, args.name)

    # CSV（去掉 geometry 保留属性）
    out.drop(columns='geometry').to_csv(base + '.csv', index=False, encoding='utf-8-sig')
    # gpkg / shp
    out.to_file(base + '.gpkg', driver='GPKG')
    try:
        out.to_file(base + '.shp', driver='ESRI Shapefile', encoding='utf-8')
    except Exception as e:
        print(f'  (shp 写入失败: {e}; gpkg/csv 已就绪)')

    print(f'[done] 输出: {base}.gpkg / .csv')
    print(f'\n[prdct_cf 分布]')
    print(out['prdct_cf'].value_counts().to_string())
    print(f'\n[conf_flag 分布]')
    print(out['conf_flag'].value_counts().to_string())


if __name__ == '__main__':
    main()
