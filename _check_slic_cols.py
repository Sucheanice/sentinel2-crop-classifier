# -*- coding: utf-8 -*-
"""临时检查：盐亭3乡镇 slic 对象 gpkg 关键列是否齐全。"""
import geopandas as gpd

files = {
    '富驿镇': r'e:\工作相关\2026年\0624 待测试数据\slic_富驿镇.gpkg',
    '莲花湖乡': r'e:\工作相关\2026年\0624 待测试数据\slic_莲花湖乡.gpkg',
    '大兴回族乡': r'e:\工作相关\2026年\0624 待测试数据\slic_大兴回族乡.gpkg',
}

PHASES = ['D04', 'D11', 'D14']
BANDS_GEE = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
need = [f'{p}_{b}' for p in PHASES for b in BANDS_GEE] + ['fid', 'buf_pixel']

for name, path in files.items():
    g = gpd.read_file(path)
    cols = set(g.columns)
    missing = [c for c in need if c not in cols]
    print(name, '| 行数=', len(g), '| 总列数=', len(g.columns))
    print('  缺失关键列:', missing if missing else '无')
    print('  有 label 列:', 'label' in cols)
    if 'D11_B8' in cols:
        print('  D11_B8 NaN 占比:', round(g['D11_B8'].isna().mean(), 3))
