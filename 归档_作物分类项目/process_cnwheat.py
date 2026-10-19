# -*- coding: utf-8 -*-
"""
CN_Wheat10 小麦复验（高效版）：窗口一次读取 + rasterize + bincount
==================================================================
对比梓潼模型预测 vs CN_Wheat10 2024 冬小麦图的一致率。
"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.windows import from_bounds
from rasterio.features import rasterize

BASE = r'e:\工作相关\2026年\0624 待测试数据'
TIF = os.path.join(BASE, 'CN-Wheat10_2024', 'CN-Wheat10_2024_WW_H.tif')
TIF_P = os.path.join(BASE, 'CN-Wheat10_2024', 'CN-Wheat10_2024_WW_P.tif')
SHP = os.path.join(BASE, '待训练数据绵阳市', '绵阳市', '梓潼县', '矢量', '梓潼县.shp')
PRED = os.path.join(BASE, '梓潼县_小春预测.csv')
THRESH = 0.3  # 地块内小麦像元占比 > 该阈值 -> 判小麦


def zonal_wheat_ratio(tif, gdf):
    """返回每块的 (小麦像元占比, 有效像元数)。编码: 1=小麦, 0=非小麦, 255=nodata"""
    gdf = gdf.to_crs('EPSG:4326')  # 成品图是 WGS84
    with rasterio.open(tif) as src:
        minx, miny, maxx, maxy = gdf.total_bounds
        win = from_bounds(minx, miny, maxx, maxy, transform=src.transform)
        data = src.read(1, window=win)
        transform = src.window_transform(win)
    u, c = np.unique(data, return_counts=True)
    print(f'    窗口形状={data.shape}, 值分布={dict(zip(u.tolist(), c.tolist()))}')

    shapes = [(g, fid + 1) for fid, g in enumerate(gdf.geometry) if g is not None and not g.is_empty]
    label_img = rasterize(shapes, out_shape=data.shape, transform=transform, fill=0, dtype='uint32')

    flat = data.ravel()
    flat_lab = label_img.ravel()
    n = len(gdf)
    in_parcel = flat_lab > 0
    valid_px = (flat != 255)  # 非 nodata（即 0 或 1）
    wheat_px = (flat == 1)    # 小麦

    total = np.bincount(flat_lab[in_parcel & valid_px], minlength=n + 1)
    wheat = np.bincount(flat_lab[in_parcel & wheat_px], minlength=n + 1)
    ratio = wheat[1:n + 1] / np.where(total[1:n + 1] > 0, total[1:n + 1], np.nan)
    return ratio, total[1:n + 1]


def compare(tif, tag):
    gdf = gpd.read_file(SHP)
    gdf['fid'] = range(len(gdf))
    pred = pd.read_csv(PRED, encoding='utf-8-sig')
    gdf = gdf.merge(pred[['fid', '预测作物', 'P_小麦', 'P_油菜']], on='fid', how='left')
    print(f'\n[处理 {tag}] {tif}')
    ratio, count = zonal_wheat_ratio(tif, gdf)
    gdf['小麦占比'] = ratio
    gdf['有效像元'] = count

    valid = count > 0
    pred_wheat = (gdf['预测作物'].values == '小麦')
    gt_wheat = ratio > THRESH
    both = valid & (~np.isnan(ratio))

    agree = (gt_wheat == pred_wheat) & both
    acc = agree[both].mean() if both.sum() > 0 else np.nan

    gt_pos = gt_wheat & both
    rec = (pred_wheat & gt_pos).sum() / gt_pos.sum() if gt_pos.sum() > 0 else np.nan
    prec = (pred_wheat & gt_pos).sum() / (pred_wheat & both).sum() if (pred_wheat & both).sum() > 0 else np.nan

    print(f'  有效块={both.sum()} (无有效像元 {int((~valid).sum())} 块)')
    print(f'  模型判小麦={pred_wheat.sum()}, 成品图判小麦={int(gt_wheat[both].sum())}')
    print(f'  一致率(overall)={acc:.4f}')
    print(f'  小麦 recall(成品图为真)={rec:.4f}  precision={prec:.4f}')

    # 油菜地块里成品图的小麦占比
    pred_rape = gdf['预测作物'].values == '油菜'
    r_r = ratio[pred_rape & both]
    print(f'  模型判油菜地块的成品图小麦占比: 中位数={np.nanmedian(r_r):.4f}, 均值={np.nanmean(r_r):.4f}')
    return gdf


if __name__ == '__main__':
    gh = compare(TIF, '冬小麦收获 WW_H')
    gp = compare(TIF_P, '冬小麦种植 WW_P')
    # 合并保存
    out = gh[['fid', '预测作物', 'P_小麦', 'P_油菜', '小麦占比', '有效像元']].copy()
    out.columns = ['fid', '预测作物', 'P_小麦', 'P_油菜', 'WW_H_小麦占比', '有效像元']
    out['WW_P_小麦占比'] = gp['小麦占比'].values
    out.to_csv(os.path.join(BASE, '梓潼县_复验_CNWheat10.csv'), index=False, encoding='utf-8-sig')
    print('\n[保存] 梓潼县_复验_CNWheat10.csv')
