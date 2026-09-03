# -*- coding: utf-8 -*-
"""
构建「非作物」负样本集（前进村，位于江油马角镇）
================================================================
目的：为「作物 vs 非作物」泛化过滤器提供非作物负样本。
数据：前进村 SLIC 对象（3旬30波段）+ 前进村 dltb 完整地类。
口径（种植险标的 = 作物，其余 = 非作物）：
  作物：旱地/水田/水浇地/后备耕地/果园/其他园地/设施农用地
  非作物：林地/草地/建设用地/水面/道路/沟渠/采矿用地 等
输出：非作物负样本 CSV（对齐正样本 3旬30波段口径）+ gpkg（含 geometry 供查看）
"""
import os
import geopandas as gpd

BASE = r'e:\工作相关\2026年\0624 待测试数据'
SLIC_GPKG = os.path.join(BASE, 'slic_objects_qianjincun.gpkg')
DLTB_GDB = os.path.join(BASE, '待测试数据前进0806', '前进0806.gdb')
PHASES = ['D04', 'D11', 'D14']
BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
BAND_COLS = [f'{p}_{b}' for p in PHASES for b in BANDS]

# 种植险标的（作物）地类
CROP_DLMC = ['旱地', '水田', '水浇地', '后备耕地', '果园', '其他园地', '设施农用地']


def main():
    slic = gpd.read_file(SLIC_GPKG)
    dltb = gpd.read_file(DLTB_GDB, layer='dltb')
    dltb = dltb[dltb['ZLDWMC'] == '前进村'].copy()
    dltb['is_crop'] = dltb['DLMC'].isin(CROP_DLMC).astype(int)

    # SLIC 质心 -> dltb 打标签
    sc = slic.to_crs(4523).copy()
    sc['geometry'] = sc.geometry.centroid
    j = gpd.sjoin(sc[['label', 'geometry']], dltb[['is_crop', 'DLMC', 'geometry']],
                  how='left', predicate='within')
    slic['is_crop'] = j['is_crop'].values
    slic['DLMC'] = j['DLMC'].values

    # 负样本 = 明确落在非耕地（非作物）dltb 图斑内的 SLIC 对象
    neg = slic[slic['is_crop'] == 0].copy()
    print(f'SLIC 对象总数: {len(slic)}')
    print(f'有地类标签: {slic["is_crop"].notna().sum()}')
    print(f'  作物(险标的): {(slic["is_crop"] == 1).sum()}')
    print(f'  非作物(负样本): {len(neg)}')
    print(f'  无标签(NaN): {slic["is_crop"].isna().sum()}')
    print('\n负样本地类分布:')
    print(neg['DLMC'].value_counts().to_string())

    # 导出 CSV（训练用，对齐正样本口径）
    out_csv = os.path.join(BASE, '负样本集_非作物_前进村.csv')
    csv_df = neg[['label'] + BAND_COLS + ['buf_pixel', 'DLMC']].copy()
    csv_df = csv_df.rename(columns={'label': 'fid'})
    csv_df['类别'] = '非作物'
    csv_df.to_csv(out_csv, index=False, encoding='utf-8-sig')
    print(f'\n已导出 CSV: {out_csv} ({len(csv_df)} 行)')

    # 导出 gpkg（含 geometry 供 QGIS 查看）
    out_gpkg = os.path.join(BASE, '负样本集_非作物_前进村.gpkg')
    neg_out = neg[['label'] + BAND_COLS + ['buf_pixel', 'DLMC', 'is_crop', 'geometry']].copy()
    neg_out = neg_out.rename(columns={'label': 'fid'})
    neg_out.to_file(out_gpkg, driver='GPKG')
    print(f'已导出 GPKG: {out_gpkg} ({len(neg_out)} 行)')


if __name__ == '__main__':
    main()
