# -*- coding: utf-8 -*-
"""临时诊断：两旬模型对「云覆盖缺失」对象的实际预测分布（定位为何未覆盖）。"""
import os, sys, pickle
import pandas as pd
import geopandas as gpd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predict_dachun_gee import run_predict, BANDS_GEE, PHASES

BASE = os.path.dirname(os.path.abspath(__file__))

slic = gpd.read_file(os.path.join(BASE, 'slic_文通镇_wide.gpkg')).reset_index(drop=True)
slic['fid'] = range(len(slic))

with open(os.path.join(BASE, 'dachun_binary_23counties_3dekad.pkl'), 'rb') as f:
    b3 = pickle.load(f)
with open(os.path.join(BASE, 'dachun_binary_2dekad.pkl'), 'rb') as f:
    b2 = pickle.load(f)

band_names = [f'{p}_{b}' for p in PHASES for b in BANDS_GEE]
feat_df = slic[['fid', 'buf_pixel'] + band_names].copy()

out3 = run_predict(slic, feat_df, b3)
mask = (out3['prdct_cf'].values == '云覆盖缺失')
print(f'三旬「云覆盖缺失」对象数: {mask.sum()}/{len(slic)}')

out2 = run_predict(slic, feat_df, b2)
print('\n两旬模型全量 prdct_cf 分布:')
print(out2['prdct_cf'].value_counts().to_string())
print('\n两旬模型对「云覆盖缺失」对象的 prdct_cf 分布:')
print(out2.loc[mask, 'prdct_cf'].value_counts().to_string())
print('\n两旬模型对「云覆盖缺失」对象的 max_prob 分箱:')
print(pd.cut(out2.loc[mask, 'max_prob'], [0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]).value_counts().to_string())
print('\n这些对象 D04_B08 / D14_B08 是否 NaN:')
sub = slic.loc[mask, ['D04_B8', 'D14_B8', 'D11_B8']]
print('D04_B8 NaN:', sub['D04_B8'].isna().sum(), ' D14_B8 NaN:', sub['D14_B8'].isna().sum(),
      ' D11_B8 NaN:', sub['D11_B8'].isna().sum())

# 复现 _apply_2dekad 的赋值逻辑，定位为何未覆盖
print('\n--- 复现 _apply_2dekad ---')
m2 = (out3['prdct_cf'].values == '云覆盖缺失')
upd = m2 & out2['prdct_cf'].isin(['水稻', '玉米']).values
print('mask.sum()=', m2.sum(), ' upd.sum()=', upd.sum())
out_test = out3.copy()
print('out_test index 范围:', out_test.index.min(), out_test.index.max(), ' len=', len(out_test))
print('out2 index 范围:', out2.index.min(), out2.index.max(), ' len=', len(out2))
out_test.loc[upd, 'prdct_cf'] = out2.loc[upd, 'prdct_cf'].values
print('赋值后 out_test prdct_cf 分布:')
print(out_test['prdct_cf'].value_counts().to_string())

# 直接调用真实函数，验证 predict_slic / _apply_2dekad 是否生效
print('\n--- 调用真实 predict_slic(bundle2=b2) ---')
from predict_slic_overlay import predict_slic
out_slic = predict_slic(slic, b3, bundle2=b2)
print('predict_slic 结果 prdct_cf 分布:')
print(out_slic['prdct_cf'].value_counts().to_string())


