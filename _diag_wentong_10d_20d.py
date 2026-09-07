# -*- coding: utf-8 -*-
"""临时诊断：文通镇 10天 vs 20天窗口 特征缺值对比。"""
import geopandas as gpd

files = {
    '文通镇_10天(slic_文通镇)': r'e:\工作相关\2026年\0624 待测试数据\slic_文通镇.gpkg',
    '文通镇_20天(slic_文通镇_wide)': r'e:\工作相关\2026年\0624 待测试数据\slic_文通镇_wide.gpkg',
}

for name, path in files.items():
    g = gpd.read_file(path)
    print('====', name, '| 对象数=', len(g), '====')
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
