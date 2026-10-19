# -*- coding: utf-8 -*-
"""
梓潼县小春推理：89.89% 小麦/油菜模型 -> 预测图（无真值）
======================================================
流程：加载模型 pkl -> 读梓潼小春特征 CSV -> 特征工程(232维，与训练一致)
      -> 推理 -> 输出预测 CSV + 写回 SHP(加预测字段)
"""
import os
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
import warnings
warnings.filterwarnings('ignore')

BASE = r'e:\工作相关\2026年\0624 待测试数据'
from exp_finalize_v2 import build_X_full


def main():
    # 1. 加载模型
    pkg_path = os.path.join(BASE, '小春_小麦油菜_清洗后模型.pkl')
    with open(pkg_path, 'rb') as f:
        pkg = pickle.load(f)
    model = pkg['model']
    le = pkg['label_encoder']
    feat_names = pkg['feature_names']
    print(f'[模型] 特征数={len(feat_names)}, 类别={le.classes_.tolist()}')

    # 2. 读梓潼特征
    csv_path = os.path.join(BASE, 'gee_梓潼县_小春特征.csv')
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    print(f'[特征] {csv_path}: {df.shape}')

    # 3. 特征工程（与训练一致）
    X, names = build_X_full(df)
    print(f'[特征工程] X={X.shape}, 列名一致={names == feat_names}')
    valid = ~np.isnan(X).any(axis=1)
    n_drop = int((~valid).sum())
    print(f'[清洗] 含NaN样本 {n_drop} 个（可能云掩膜后无有效像元）')
    X_valid = X[valid]

    # 4. 推理
    prob = model.predict_proba(X_valid)  # (n, 2) [P(小麦), P(油菜)]
    pred_idx = prob.argmax(axis=1)
    pred_label = le.inverse_transform(pred_idx)
    prob_rape = prob[:, 1]
    prob_wheat = prob[:, 0]

    # 5. 组装结果
    out = df.iloc[valid].reset_index(drop=True).copy()
    out['预测作物'] = pred_label
    out['P_小麦'] = np.round(prob_wheat, 4)
    out['P_油菜'] = np.round(prob_rape, 4)
    out['置信度'] = np.round(np.max(prob, axis=1), 4)

    res_csv = os.path.join(BASE, '梓潼县_小春预测.csv')
    out.to_csv(res_csv, index=False, encoding='utf-8-sig')
    print(f'[结果] {res_csv}: {out.shape}')

    print('\n[预测统计]')
    print(out['预测作物'].value_counts().to_string())
    print(f'\n高置信(>0.9): {(out["置信度"] > 0.9).sum()} / {len(out)}')
    print(f'中置信(0.7~0.9): {((out["置信度"] > 0.7) & (out["置信度"] <= 0.9)).sum()}')
    print(f'低置信(<0.7): {(out["置信度"] < 0.7).sum()}')

    # 6. 写回 SHP（加预测字段）
    shp_path = os.path.join(BASE, '待训练数据绵阳市', '绵阳市', '梓潼县', '矢量', '梓潼县.shp')
    gdf = gpd.read_file(shp_path)
    # 用 fid 匹配（提取时 fid=range(len(gdf))，简化后顺序不变）
    pred_map = dict(zip(out['fid'], out['预测作物']))
    prob_map = dict(zip(out['fid'], out['置信度']))
    gdf['小春预测'] = [pred_map.get(i, '无') for i in range(len(gdf))]
    gdf['小春置信度'] = [prob_map.get(i, np.nan) for i in range(len(gdf))]
    out_shp = os.path.join(BASE, '梓潼县_小春预测.shp')
    gdf.to_file(out_shp, encoding='utf-8')
    print(f'[写回] {out_shp}: {len(gdf)} 块')


if __name__ == '__main__':
    main()
