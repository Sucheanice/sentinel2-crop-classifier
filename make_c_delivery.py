# -*- coding: utf-8 -*-
"""make_c_delivery.py — 从 A 结果生成 C 交付文件。

对齐交付标准 9 列（样例「大春作物.xlsx」结构），作物编码参照
full.md 附表三（6 位）。几何统一 CGCS2000/EPSG:4523。

已确认决策（09-07）：
  1. C 只含确定作物（水稻/玉米），不确定/无覆盖/云覆盖缺失/非植被 不进 C。
  2. 实际面积/亩 用 TBMJ（投保面积，单位亩）。
  3. 输出 gpkg，几何 EPSG:4523。
  4. 编码 6 位（水稻 AA0000、玉米 AA0007），小麦取冬小麦 AA0001。

遍历所有 *_wide_v4_A_赋值.gpkg，逐个输出到 C交付/ 目录。
"""
import os
import glob
import geopandas as gpd
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, 'C交付')
os.makedirs(OUT_DIR, exist_ok=True)

# 作物 → 编码（full.md 附表三，6 位）
CROP_CODE = {
    '水稻': 'AA0000',
    '玉米': 'AA0007',
    '小麦': 'AA0001',   # 冬小麦
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

# 当前大春项目只导出确定作物
VALID_CROPS = {'水稻', '玉米'}


def _to_str(s):
    """数值型村编码转干净字符串（去 .0）。"""
    s = pd.to_numeric(s, errors='coerce')
    if pd.notna(s):
        if float(s).is_integer():
            return str(int(s))
        return str(s)
    return str(s)


def make_c(src):
    gdf = gpd.read_file(src)
    if 'prdct_cf' not in gdf.columns:
        print(f'[跳过] 无 prdct_cf: {os.path.basename(src)}')
        return None

    sub = gdf[gdf['prdct_cf'].isin(VALID_CROPS)].copy()
    if len(sub) == 0:
        print(f'[跳过] 无确定作物: {os.path.basename(src)}')
        return None

    # 期数（时间）= 承保年份 + 承保期次，如 2025大春
    year = sub['YEAR_'].iloc[0] if 'YEAR_' in sub.columns else '2025'
    season = sub['TBJJ'].iloc[0] if 'TBJJ' in sub.columns else '大春'
    qishu = f'{_to_str(year)}{season}'

    # 几何统一转到 CGCS2000/EPSG:4523
    sub = sub.to_crs('EPSG:4523')

    # 统一村编码为字符串，按 村编码+地块编号 排序，村内编号
    sub['_cjdm'] = sub['CJDM'].apply(_to_str)
    sub = sub.sort_values(['_cjdm', 'DKBH']).reset_index(drop=True)
    sub['_seq_in_village'] = sub.groupby('_cjdm').cumcount() + 1

    out = gpd.GeoDataFrame({
        '序号': sub['_seq_in_village'],
        '村编码': sub['_cjdm'],
        '村名称': sub['CJMC'].astype(str),
        '期数时间': qishu,
        '五级标的编码': sub['prdct_cf'].map(CROP_CODE),
        '五级标的名称': sub['prdct_cf'].astype(str),
        '地块唯一编码': 'G' + sub['_cjdm'] + sub['_seq_in_village'].astype(str).str.zfill(6),
        '实际面积亩': pd.to_numeric(sub['TBMJ'], errors='coerce'),
        'geometry': sub['geometry'].values,
    }, crs='EPSG:4523')

    return out


def main():
    srcs = sorted(glob.glob(os.path.join(BASE, '**', '*_wide_v4_A_赋值.gpkg'), recursive=True))
    print(f'找到 {len(srcs)} 个 wide_v4 A 结果')
    total = 0
    for src in srcs:
        rel = os.path.relpath(src, BASE)
        name = os.path.basename(src).replace('_wide_v4_A_赋值', '_C')
        out_path = os.path.join(OUT_DIR, name)
        c = make_c(src)
        if c is None:
            continue
        c.to_file(out_path, driver='GPKG', encoding='utf-8')
        total += len(c)
        print(f'  {rel} -> C交付/{name}  ({len(c)} 块)')
    print(f'[done] 共 {total} 块，输出目录 C交付/')


if __name__ == '__main__':
    main()
