# -*- coding: utf-8 -*-
"""gee_extract_xuyong_texture.py — 柑橘 vs 林地 GLCM 纹理特征提取

针对物候盲区「常绿果园(柑橘) vs 常绿林地」，用 GLCM 纹理特征验证可分性。
公开参数（多篇文献一致）：
  Gray = 0.3*B8 + 0.59*B4 + 0.11*B3（灰度图）
  窗口 3x3，特征 asm/contrast/corr/var/idm/savg/ent/diss

时相：8 月 median（生长季冠层纹理最明显）。
输出：xuyong_texture_aug.csv（柑橘/林地斑块 × [category + B8纹理8 + Gray纹理8 + B8_mean + NDVI]）
"""
import ee
import geopandas as gpd
import pandas as pd
import os

ee.Initialize()

BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
TEX_BANDS = ['asm', 'contrast', 'corr', 'var', 'idm', 'savg', 'ent', 'diss']

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GPKG = os.path.join(BASE_DIR, '待训练数据水果', '叙永水果成果数据', 'result.shp')
OUT_CSV = os.path.join(BASE_DIR, 'xuyong_texture_aug.csv')

UTM = 'EPSG:32648'


def mask_s2_clouds(img):
    scl = img.select('SCL')
    cloud = scl.eq(3).Or(scl.eq(8)).Or(scl.eq(9)).Or(scl.eq(10))
    return img.updateMask(cloud.Not())


def main():
    gdf = gpd.read_file(GPKG)
    gdf['geometry'] = gdf['geometry'].simplify(tolerance=0.0001, preserve_topology=True)

    # 柑橘果园斑块（centroid 15m 圆斑块）
    ganju = gdf[gdf['ZWMC'] == '柑橘'].copy()
    ganju_fc = ee.FeatureCollection(ganju[['geometry']].__geo_interface__['features'])
    ganju_patches = ganju_fc.map(lambda f: ee.Feature(
        f.geometry().centroid().transform(UTM, 10).buffer(15), {'category': '柑橘'}))

    # 林地斑块（果园 100m 缓冲内 WorldCover tree cover 采样点 -> 15m 圆斑块）
    orchards = ee.FeatureCollection(gdf[['geometry']].__geo_interface__['features'])
    buffers = orchards.map(lambda f: ee.Feature(f.geometry().transform(UTM, 10).buffer(100)))
    region = buffers.union()
    wc = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map')
    lin_points = wc.updateMask(wc.eq(10)).sample(
        region=region, scale=10, numPixels=500, seed=42,
        geometries=True, tileScale=2, dropNulls=True)
    lin_patches = lin_points.map(lambda f: ee.Feature(
        f.geometry().transform(UTM, 10).buffer(15), {'category': '林地'}))

    patches = ee.FeatureCollection([lin_patches, ganju_patches]).flatten()

    # 8 月 median 影像
    s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(region) \
        .filterDate('2025-08-01', '2025-09-01').map(mask_s2_clouds)
    med = s2.select(BANDS).median()

    # B8 纹理
    b8 = med.select('B8').unitScale(0, 5000).multiply(255).toUint8().rename('b8gray')
    b8_glcm = b8.glcmTexture(size=3)

    # Gray 灰度图纹理
    gray = med.expression(
        '0.3 * NIR + 0.59 * RED + 0.11 * GREEN',
        {'NIR': med.select('B8'), 'RED': med.select('B4'), 'GREEN': med.select('B3')},
    ).unitScale(0, 5000).multiply(255).toUint8().rename('gray')
    gray_glcm = gray.glcmTexture(size=3)

    # 组装纹理 + 光谱
    texture = b8_glcm.select([f'b8gray_{t}' for t in TEX_BANDS], [f'B8_{t}' for t in TEX_BANDS])
    texture = texture.addBands(
        gray_glcm.select([f'gray_{t}' for t in TEX_BANDS], [f'Gray_{t}' for t in TEX_BANDS]))
    ndvi = med.normalizedDifference(['B8', 'B4']).rename('NDVI')
    texture = texture.addBands(med.select('B8').rename('B8_mean')).addBands(ndvi)

    out_cols = ['category'] + [f'B8_{t}' for t in TEX_BANDS] + [f'Gray_{t}' for t in TEX_BANDS] + ['B8_mean', 'NDVI']
    reduced = texture.reduceRegions(
        collection=patches, reducer=ee.Reducer.mean(), scale=10, tileScale=4,
    ).select(out_cols)

    feats = reduced.getInfo()['features']
    rows = [{k: f['properties'].get(k) for k in out_cols} for f in feats]
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')
    print(f'[done] {OUT_CSV}: {df.shape}')
    print(f'  类别分布: {dict(df["category"].value_counts())}')


if __name__ == '__main__':
    main()
