# -*- coding: utf-8 -*-
"""
build_negative_samples_9villages.py — 9 村地类打标签，合并「非作物」负样本集
================================================================
口径（种植险标的 = 作物，其余 = 非作物）：
  DLMC 中文：旱地/水田/水浇地/后备耕地/果园/其他园地/设施农用地
  YDFLEJ 编码：A0101(水田)/A0103(旱地)/A0200(园地)/A1202(设施农用地)

流程：SLIC 对象质心 -> 各村地类图斑（within）打标签 -> 排除云污染(波段NaN)
输出：负样本集_非作物_9村.csv / .gpkg，正样本集_作物_9村.csv / .gpkg
"""
import os
import geopandas as gpd
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
PHASES = ['D04', 'D11', 'D14']
BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
BAND_COLS = [f'{p}_{b}' for p in PHASES for b in BANDS]

CROP_DLMC = {'旱地', '水田', '水浇地', '后备耕地', '果园', '其他园地', '设施农用地'}
CROP_YDFLEJ = {'A0101', 'A0102', 'A0103', 'A0200', 'A1202'}

# 村名, SLIC gpkg, 地类类型(gdb/shp), 地类路径, 字段口径(DLMC/YDFLEJ)
VILLAGES = [
    ('前进村', 'slic_前进村.gpkg', 'gdb', r'待测试数据前进0806\前进0806.gdb', 'DLMC'),
    ('印坪村', 'slic_印坪村.gpkg', 'shp', r'F:\0421给yxx\提交成果\印坪村\矢量\2印坪.shp', 'YDFLEJ'),
    ('大岳村', 'slic_大岳村.gpkg', 'shp', r'F:\0421给yxx\提交成果\大岳村\矢量\2大岳.shp', 'YDFLEJ'),
    ('沉水村', 'slic_沉水村.gpkg', 'shp', r'F:\0421给yxx\提交成果\沉水村\矢量\2沉水.shp', 'YDFLEJ'),
    ('马阁寺村', 'slic_马阁寺村.gpkg', 'shp', r'F:\0421给yxx\提交成果\马阁寺村\矢量\2马阁寺.shp', 'YDFLEJ'),
    ('龙宫村', 'slic_龙宫村.gpkg', 'shp', r'F:\0421给yxx\提交成果\龙宫村\矢量\2龙宫.shp', 'YDFLEJ'),
    ('宝珠村', 'slic_宝珠村.gpkg', 'shp', r'20260902绵阳(项目任务区)\20260901绵阳对应影像和矢量\宝珠村地类.shp', 'DLMC'),
    ('朝天村', 'slic_朝天村.gpkg', 'shp', r'20260902绵阳(项目任务区)\20260901绵阳对应影像和矢量\朝天村地类.shp', 'DLMC'),
    ('统一村', 'slic_统一村.gpkg', 'shp', r'20260902绵阳(项目任务区)\20260901绵阳对应影像和矢量\统一村地类.shp', 'DLMC'),
]


def read_landuse(kind, path, field):
    if kind == 'gdb':
        g = gpd.read_file(os.path.join(BASE, path), layer='dltb')
        g = g[g['ZLDWMC'] == '前进村'].copy()
    else:
        g = gpd.read_file(path if os.path.isabs(path) else os.path.join(BASE, path))
    if field == 'DLMC':
        g['is_crop'] = g['DLMC'].isin(CROP_DLMC).astype(int)
        g['DLMC_name'] = g['DLMC']
    else:
        g['is_crop'] = g['YDFLEJ'].isin(CROP_YDFLEJ).astype(int)
        g['DLMC_name'] = g['YDFLEJ']
    return g[['is_crop', 'DLMC_name', 'geometry']]


def main():
    neg_parts, pos_parts = [], []
    print(f'{"村":<6} {"对象":>7} {"有效":>7} {"作物":>6} {"非作物":>7} {"无标签":>7} {"云NaN":>7}')
    for name, slic_fn, kind, lu_path, field in VILLAGES:
        slic = gpd.read_file(os.path.join(BASE, slic_fn))
        lu = read_landuse(kind, lu_path, field)

        valid_band = slic[BAND_COLS].notna().all(axis=1)
        sc = slic.to_crs(lu.crs).copy()
        sc['geometry'] = sc.geometry.centroid
        j = gpd.sjoin(sc[['label', 'geometry']], lu, how='left', predicate='within')
        j = j[~j.index.duplicated(keep='first')]
        slic['is_crop'] = j['is_crop'].reindex(slic.index).values
        slic['DLMC_name'] = j['DLMC_name'].reindex(slic.index).values

        n_crop = int((slic['is_crop'] == 1).sum())
        n_neg = int((slic['is_crop'] == 0).sum())
        n_nan = int(slic['is_crop'].isna().sum())
        n_cloud = int((~valid_band).sum())
        print(f'{name:<6} {len(slic):>7} {valid_band.sum():>7} {n_crop:>6} {n_neg:>7} {n_nan:>7} {n_cloud:>7}')

        neg = slic[(slic['is_crop'] == 0) & valid_band].copy()
        neg['村'] = name
        neg_parts.append(neg)
        pos = slic[(slic['is_crop'] == 1) & valid_band].copy()
        pos['村'] = name
        pos_parts.append(pos)

    neg_all = pd.concat(neg_parts, ignore_index=True).reset_index(drop=True)
    pos_all = pd.concat(pos_parts, ignore_index=True).reset_index(drop=True)
    # SNIC label 跨村重复，重新生成全局唯一 fid
    neg_all['fid'] = range(1, len(neg_all) + 1)
    pos_all['fid'] = range(1, len(pos_all) + 1)

    print('\n=== 负样本（非作物）地类分布 ===')
    print(neg_all['DLMC_name'].value_counts().to_string())
    print(f'\n负样本合计: {len(neg_all)}, 正样本(作物)合计: {len(pos_all)}')

    out_cols = ['fid'] + BAND_COLS + ['buf_pixel', 'DLMC_name', '村']
    neg_csv = os.path.join(BASE, '负样本集_非作物_9村.csv')
    neg_all[out_cols].rename(columns={'DLMC_name': '地类'}).to_csv(
        neg_csv, index=False, encoding='utf-8-sig')
    print(f'\n已导出 CSV: {neg_csv} ({len(neg_all)} 行)')

    pos_csv = os.path.join(BASE, '正样本集_作物_9村.csv')
    pos_all[out_cols].rename(columns={'DLMC_name': '地类'}).to_csv(
        pos_csv, index=False, encoding='utf-8-sig')
    print(f'已导出 CSV: {pos_csv} ({len(pos_all)} 行)')

    neg_all[out_cols + ['geometry']].rename(columns={'DLMC_name': '地类'}).to_file(
        os.path.join(BASE, '负样本集_非作物_9村.gpkg'), driver='GPKG')
    pos_all[out_cols + ['geometry']].rename(columns={'DLMC_name': '地类'}).to_file(
        os.path.join(BASE, '正样本集_作物_9村.gpkg'), driver='GPKG')
    print('已导出 GPKG')


if __name__ == '__main__':
    main()
