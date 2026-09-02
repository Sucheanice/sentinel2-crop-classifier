# -*- coding: utf-8 -*-
"""predict_slic_overlay.py — SLIC 对象预测 + 套合外部 shp（输出A赋值 + 输出B质检）

面向对象预测第二、三步：
  1. 加载 slic_objects.gpkg（SNIC 超像素对象 + 30波段特征）
  2. 特征工程 + 23县关键3旬模型预测 + 三阈值判定 → 超像素预测结果
  3. 套合外部 shp：
     - 输出A（主交付）：外部 shp + 作物标签 + 11 质量字段（面积多数投票）
     - 输出B（质检附件）：超像素对象 + 套合状态字段（标注三类问题）

用法：
  python predict_slic_overlay.py --slic slic_objects.gpkg --external 外部地块.shp \
      --model dachun_binary_23counties_3dekad.pkl --output 结果目录
"""
import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from predict_dachun_gee import (run_predict, BANDS_GEE, BANDS_FEAT, BAND_MAP, PHASES,
                                NDVI_VEG, CONF_LOW, CONF_HIGH, PX_MIN)
from common import compute_feature_matrix


def load_slic(path):
    gdf = gpd.read_file(path)
    gdf = gdf.reset_index(drop=True)
    gdf['fid'] = range(len(gdf))
    return gdf


def apply_cropland_filter(slic_gdf, filter_bundle):
    """用纯哨兵「耕地/非耕地」过滤器预测每个对象是否耕地（1=耕地, 0=非耕地）。"""
    df = slic_gdf.copy()
    rename = {}
    for c in df.columns:
        if '_' in c:
            p, b = c.rsplit('_', 1)
            if b in BAND_MAP and p in PHASES:
                rename[c] = f'{p}_{BAND_MAP[b]}'
    df = df.rename(columns=rename)

    band_values = {}
    for p in PHASES:
        for b in BANDS_FEAT:
            col = f'{p}_{b}'
            if col in df.columns:
                band_values[col] = df[col].values
    X, fnames = compute_feature_matrix(band_values, PHASES, available_bands=BANDS_FEAT)
    X = pd.DataFrame(X, columns=fnames)

    fsel = filter_bundle['feature_names']
    for m in [f for f in fsel if f not in X.columns]:
        X[m] = 0.0
    X = X[fsel].fillna(0).values.astype(np.float32)
    return filter_bundle['model'].predict(X)


def predict_slic(slic_gdf, bundle):
    """对每个超像素对象做水稻/玉米二分类（无过滤），返回带 11 字段的 GeoDataFrame。"""
    band_names = [f'{p}_{b}' for p in PHASES for b in BANDS_GEE]
    feat_df = slic_gdf[['fid', 'buf_pixel'] + band_names].copy()
    out = run_predict(slic_gdf, feat_df, bundle)
    # parcel_id 用干净序号（fid+1）；SNIC 原始 cluster id 可能为负/超 int32，不作 parcel_id
    out['parcel_id'] = out['fid'] + 1
    return out


def utm_epsg(gdf):
    lon = gdf.to_crs(4326).geometry.centroid.x.mean()
    zone = int((lon + 180) // 6) + 1
    return 32600 + zone


def overlay_assign(slic_pred, external_gdf):
    """套合：返回 (输出A 外部地块结果, 输出B 超像素套合状态)。"""
    epsg = utm_epsg(external_gdf)

    s = slic_pred.to_crs(epsg)[
        ['fid', 'label', 'prdct_cf', 'prdct_lbl', 'max_prob', 'ndvi_max', 'buf_pixel', 'geometry']
    ].copy()
    s['slic_area'] = s.geometry.area

    e = external_gdf.to_crs(epsg).copy()
    e['fid_ext'] = range(len(e))
    e['ext_area'] = e.geometry.area

    inter = gpd.overlay(
        s[['fid', 'label', 'prdct_cf', 'prdct_lbl', 'max_prob', 'ndvi_max', 'buf_pixel', 'geometry']],
        e[['fid_ext', 'geometry']], how='intersection')
    inter['ov_area'] = inter.geometry.area

    # ===== 输出A：外部地块面积多数投票 =====
    rows_a = []
    for fid_ext, grp in inter.groupby('fid_ext'):
        agg = grp.groupby('prdct_cf').agg(
            ov_area=('ov_area', 'sum'),
            max_prob=('max_prob', lambda x: float(np.average(x, weights=grp.loc[x.index, 'ov_area']))),
            ndvi_max=('ndvi_max', 'max'),
            buf_pixel=('buf_pixel', 'sum'),
        ).reset_index()
        main = agg.loc[agg['ov_area'].idxmax()]

        main_cf = main['prdct_cf']
        max_prob = main['max_prob']
        ndvi_max = main['ndvi_max']
        buf_pixel = main['buf_pixel']

        ndvi_flag = '植被' if ndvi_max >= NDVI_VEG else '非植被'
        conf_flag = ('高' if max_prob >= CONF_HIGH else
                     '中' if max_prob >= CONF_LOW else '低')
        px_statu = '过小(<3px)' if buf_pixel < PX_MIN else 'OK'
        predicted = 0 if main_cf == '水稻' else (1 if main_cf == '玉米' else -1)
        prdct_lbl = main_cf if main_cf in ('水稻', '玉米') else main_cf

        rows_a.append({
            'fid_ext': fid_ext,
            'area_m2': float(e.loc[e['fid_ext'] == fid_ext, 'ext_area'].iloc[0]),
            'ndvi_max': round(float(ndvi_max), 4),
            'ndvi_flag': ndvi_flag,
            'predicted': predicted,
            'prdct_lbl': prdct_lbl,
            'max_prob': round(float(max_prob), 4),
            'conf_flag': conf_flag,
            'prdct_cf': main_cf,
            'buf_pixel': int(buf_pixel),
            'px_statu': px_statu,
        })

    out_a = e.copy()
    df_a = pd.DataFrame(rows_a).set_index('fid_ext')
    for col in ['area_m2', 'ndvi_max', 'ndvi_flag', 'predicted', 'prdct_lbl',
                'max_prob', 'conf_flag', 'prdct_cf', 'buf_pixel', 'px_statu']:
        out_a[col] = out_a['fid_ext'].map(df_a[col])
    out_a['parcel_id'] = out_a['fid_ext'] + 1

    # ===== 输出B：超像素套合状态（三类问题） =====
    # 每个超像素最大重叠外部地块
    best = inter.sort_values('ov_area').groupby('fid', as_index=False).first()
    best_map = best.set_index('fid')
    main_cf_map = df_a['prdct_cf'].to_dict()
    s_area = s.set_index('fid')['slic_area']

    def judge(r):
        fid = r['fid']
        cf = r['prdct_cf']
        if fid not in best_map.index:
            if cf in ('水稻', '玉米'):
                # 有过滤器时：只有过滤器也判为耕地，才算"真漏给"；否则归为未匹配(非作物)
                if 'is_crop' in r.index and int(r['is_crop']) == 0:
                    return '未匹配(非作物)'
                return '未匹配-疑似甲方漏给'
            return '未匹配(非作物)'
        fe = best_map.loc[fid, 'fid_ext']
        if cf in ('水稻', '玉米') and main_cf_map.get(fe, cf) != cf:
            return '地块内冲突'
        return '正常'

    out_b = slic_pred.copy()
    # 只保留作物相关对象：物理过滤掉非耕地（水体/林地/建筑等）
    if 'is_crop' in out_b.columns:
        out_b = out_b[out_b['is_crop'] == 1].copy()
    out_b['match_fid_ext'] = out_b['fid'].map(
        lambda f: int(best_map.loc[f, 'fid_ext']) if f in best_map.index else -1)
    out_b['ov_area'] = out_b['fid'].map(
        lambda f: float(best_map.loc[f, 'ov_area']) if f in best_map.index else 0.0)
    out_b['slic_area'] = out_b['fid'].map(s_area)
    out_b['ov_ratio'] = (out_b['ov_area'] / out_b['slic_area']).replace([np.inf, -np.inf], 0).fillna(0)
    out_b['suit_status'] = out_b.apply(judge, axis=1)

    return out_a, out_b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--slic', required=True, help='slic_objects.gpkg 路径')
    ap.add_argument('--external', required=True, help='外部地块 shp/gpkg')
    ap.add_argument('--model', required=True, help='模型 pkl')
    ap.add_argument('--filter', default=None, help='耕地/非耕地过滤器 pkl（可选）')
    ap.add_argument('--output', required=True, help='输出目录')
    ap.add_argument('--name', default='slic_overlay', help='输出前缀')
    args = ap.parse_args()

    with open(args.model, 'rb') as f:
        bundle = pickle.load(f)
    print(f"[model] {bundle.get('model_type')} 类别={bundle.get('class_names')}")

    filter_bundle = None
    if args.filter:
        with open(args.filter, 'rb') as f:
            filter_bundle = pickle.load(f)
        print(f"[filter] {filter_bundle.get('model_type')} 类别={filter_bundle.get('class_names')}")

    slic_gdf = load_slic(args.slic)
    print(f'[slic] {len(slic_gdf)} 个超像素对象')

    external_gdf = gpd.read_file(args.external)
    print(f'[external] {len(external_gdf)} 个外部地块')

    # 1) 预测（无过滤，输出A 用纯二分类结果）
    slic_pred = predict_slic(slic_gdf, bundle)
    # 过滤器只用于输出B 质检：附加 is_crop 列（1=耕地, 0=非耕地）
    if filter_bundle is not None:
        slic_pred['is_crop'] = apply_cropland_filter(slic_gdf, filter_bundle)
    print(f'\n[超像素 prdct_cf 分布]')
    print(slic_pred['prdct_cf'].value_counts().to_string())

    # 2) 套合
    out_a, out_b = overlay_assign(slic_pred, external_gdf)

    os.makedirs(args.output, exist_ok=True)
    a_base = os.path.join(args.output, args.name + '_A_赋值')
    b_base = os.path.join(args.output, args.name + '_B_质检')

    out_a.drop(columns=['geometry']).to_csv(a_base + '.csv', index=False, encoding='utf-8-sig')
    out_a.to_file(a_base + '.gpkg', driver='GPKG')
    out_b.drop(columns=['geometry']).to_csv(b_base + '.csv', index=False, encoding='utf-8-sig')
    out_b.to_file(b_base + '.gpkg', driver='GPKG')
    # 附加 shp 输出（QGIS 可直接打开；字段名超 10 字符会被自动截断）
    try:
        out_a.to_file(a_base + '.shp', driver='ESRI Shapefile', encoding='utf-8')
        out_b.to_file(b_base + '.shp', driver='ESRI Shapefile', encoding='utf-8')
    except Exception as e:
        print(f'  (shp 写入失败: {e}; gpkg/csv 已就绪)')

    print(f'\n[done] 输出A: {a_base}.gpkg / .csv')
    print(f'      输出B: {b_base}.gpkg / .csv')
    print(f'\n[输出A prdct_cf 分布]')
    print(out_a['prdct_cf'].value_counts().to_string())
    print(f'\n[输出B 套合状态分布]')
    print(out_b['suit_status'].value_counts().to_string())


if __name__ == '__main__':
    main()
