# -*- coding: utf-8 -*-
"""make_abc_sam3.py — SAM3 耕地 shp → GEE 特征 → 作物判断 → 产出 A/B/C 三件套。

流程（路径 A 直接串）：
  SAM3 耕地 shp（精确边界坐标）→ GEE 提取哨兵3旬特征 → LightGBM 判断水稻/玉米
  → 赋值回 SAM3 shp → 拆出 A（主交付）/ B（全量质检明细）/ C（交付标准9列）。

几何统一 CGCS2000/EPSG:4523（SAM3 原始即 4523）。

用法：
  python make_abc_sam3.py --input result_baozhucun_v10_fix2/baozhucun_v10_fix2.shp \
      --model dachun_binary_23counties_3dekad_wide.pkl \
      --output 宝珠村结果 --name 宝珠村
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
from predict_dachun_gee import extract_features, run_predict

# 作物 → 编码（full.md 附表三，6 位）
CROP_CODE = {
    '水稻': 'AA0000',
    '玉米': 'AA0007',
    '小麦': 'AA0001',
    '油菜': 'AB0100',
    '蔬菜': 'AC1400',
    '柑橘': 'AD0400',
    '柑桔': 'AD0400',
    '蜜柚': 'AD0402',
    '柚': 'AD0402',
    '李子': 'AD0102',
    '李': 'AD0102',
    '中草药': 'AB0428',
}

# 期数（时间）默认：2025 大春（特征窗口即 2025 年 D04/D11/D14）
QISHU = '2025大春'

# 亩换算
MU_PER_M2 = 1.0 / 666.667


def build_c(out):
    """交付标准 9 列；村编码/村名称从 village_code/village_name 透传，无则留空。"""
    c = out.copy()
    c['序号'] = range(1, len(c) + 1)
    for src, dst in [('village_code', '村编码'), ('village_name', '村名称')]:
        if src in c.columns:
            c[dst] = c[src].astype(str).replace({'nan': '', 'None': '', '<NA>': ''})
        else:
            c[dst] = ''
    c['期数时间'] = QISHU
    c['五级标的编码'] = c['prdct_cf'].map(CROP_CODE).fillna('')
    c['五级标的名称'] = c['prdct_cf'].astype(str)
    c['地块唯一编码'] = 'G' + c['序号'].astype(str).str.zfill(6)
    c['实际面积亩'] = (pd.to_numeric(c['area_m2'], errors='coerce') * MU_PER_M2).round(4)
    cols = ['序号', '村编码', '村名称', '期数时间', '五级标的编码',
            '五级标的名称', '地块唯一编码', '实际面积亩', 'geometry']
    return c[cols]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='SAM3 耕地 shp/gpkg 路径')
    ap.add_argument('--model', required=True, help='模型 pkl 路径')
    ap.add_argument('--output', required=True, help='输出目录')
    ap.add_argument('--name', default='宝珠村', help='输出文件前缀')
    args = ap.parse_args()

    with open(args.model, 'rb') as f:
        bundle = pickle.load(f)
    print(f"[model] {bundle.get('model_type')} 类别={bundle.get('class_names')}")

    gdf = gpd.read_file(args.input)
    print(f'[input] {args.input}: {len(gdf)} 块, CRS={gdf.crs}')

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

    # A：主交付（地块 + 作物标签 + 核心质量字段）
    A = out[['id', 'class', 'area_m2', 'prdct_cf', 'conf_flag', 'max_prob', 'px_statu', 'geometry']].copy()
    A.to_file(base + '_A_赋值.gpkg', driver='GPKG')

    # B：全量质检明细（所有字段，不筛选）
    out.to_file(base + '_B_质检.gpkg', driver='GPKG')

    # C：交付标准 9 列
    C = build_c(out)
    C.to_file(base + '_C_交付.gpkg', driver='GPKG')

    print(f'[done] 输出: {args.output}\\')
    print(f'  {args.name}_A_赋值.gpkg  ({len(A)} 块)')
    print(f'  {args.name}_B_质检.gpkg  ({len(out)} 块)')
    print(f'  {args.name}_C_交付.gpkg  ({len(C)} 块)')
    print('\n[prdct_cf 分布]')
    print(out['prdct_cf'].value_counts().to_string())


if __name__ == '__main__':
    main()
