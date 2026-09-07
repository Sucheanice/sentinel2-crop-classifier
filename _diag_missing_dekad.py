# -*- coding: utf-8 -*-
"""临时诊断：盐亭3乡镇 slic 对象三旬 B8 缺值情况，定位「云覆盖缺失」来源。"""
import geopandas as gpd

files = {
    '富驿镇': r'e:\工作相关\2026年\0624 待测试数据\slic_富驿镇.gpkg',
    '莲花湖乡': r'e:\工作相关\2026年\0624 待测试数据\slic_莲花湖乡.gpkg',
    '大兴回族乡': r'e:\工作相关\2026年\0624 待测试数据\slic_大兴回族乡.gpkg',
}

for name, path in files.items():
    g = gpd.read_file(path)
    print('====', name, '====')
    cols = [c for c in g.columns if '_B8' in c or c in ('fid', 'buf_pixel', 'label')]
    print('  相关列:', cols)
    for p in ['D04', 'D11', 'D14']:
        col = f'{p}_B8'
        if col in g.columns:
            print(f'  {col} NaN={int(g[col].isna().sum())} 占比={round(g[col].isna().mean(), 4)}')
        else:
            print(f'  {col} 列不存在!')
    b8cols = [f'{p}_B8' for p in ['D04', 'D11', 'D14'] if f'{p}_B8' in g.columns]
    if b8cols:
        miss = g[b8cols].isna().any(axis=1)
        print('  三旬B8任一NaN(≈云覆盖缺失)对象数:', int(miss.sum()), '占比:', round(miss.mean(), 4))
