# -*- coding: utf-8 -*-
"""gee_extract_xuyong_control_monthly.py — 果园周边 100m 缓冲内对照地类 12 个月特征提取

用 ESA WorldCover 在叙永 276 块果园 100m 缓冲区内采样三类对照地类：
  林地(tree cover=10) / 草地(grassland=30) / 农田(cropland=40)
再提取与果园完全相同的 12 个月月级特征，用于「果园 vs 周边植被」物候对比。

输出：xuyong_control_monthly.csv（对照斑块 × [fid/landtype + 12月×(10波段+NDVI+EVI)]）
"""
import ee
import geopandas as gpd
import pandas as pd
import os

ee.Initialize()

BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
MONTHS = [
    ('M01', '2025-01-01', '2025-02-01'),
    ('M02', '2025-02-01', '2025-03-01'),
    ('M03', '2025-03-01', '2025-04-01'),
    ('M04', '2025-04-01', '2025-05-01'),
    ('M05', '2025-05-01', '2025-06-01'),
    ('M06', '2025-06-01', '2025-07-01'),
    ('M07', '2025-07-01', '2025-08-01'),
    ('M08', '2025-08-01', '2025-09-01'),
    ('M09', '2025-09-01', '2025-10-01'),
    ('M10', '2025-10-01', '2025-11-01'),
    ('M11', '2025-11-01', '2025-12-01'),
    ('M12', '2025-12-01', '2026-01-01'),
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GPKG = os.path.join(BASE_DIR, '待训练数据水果', '叙永水果成果数据', 'result.shp')
OUT_CSV = os.path.join(BASE_DIR, 'xuyong_control_monthly.csv')

UTM = 'EPSG:32648'
CLASSES = [(10, '林地'), (30, '草地'), (40, '农田')]
N_PER_CLASS = 500


def mask_s2_clouds(img):
    scl = img.select('SCL')
    cloud = scl.eq(3).Or(scl.eq(8)).Or(scl.eq(9)).Or(scl.eq(10))
    return img.updateMask(cloud.Not())


def build_monthly(roi):
    s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(roi)
    composite = None
    for prefix, start, end in MONTHS:
        col = s2.filterDate(start, end).map(mask_s2_clouds)
        med = col.select(BANDS).median().rename([f'{prefix}_{b}' for b in BANDS])
        ndvi = med.normalizedDifference([f'{prefix}_B8', f'{prefix}_B4']).rename(f'{prefix}_NDVI')
        evi = med.expression(
            '2.5 * (NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1)',
            {'NIR': med.select(f'{prefix}_B8'),
             'RED': med.select(f'{prefix}_B4'),
             'BLUE': med.select(f'{prefix}_B2')},
        ).rename(f'{prefix}_EVI')
        med = med.addBands(ndvi).addBands(evi)
        composite = med if composite is None else composite.addBands(med)
    return composite


def main():
    gdf = gpd.read_file(GPKG)
    gdf['geometry'] = gdf['geometry'].simplify(tolerance=0.0001, preserve_topology=True)
    orchards = ee.FeatureCollection(gdf[['geometry']].__geo_interface__['features'])
    print(f'[input] 果园 {len(gdf)} 块')

    # 果园 100m 缓冲（UTM 米制）
    def buffered(f):
        g = f.geometry().transform(UTM, 10).buffer(100)
        return ee.Feature(g)
    buffers = orchards.map(buffered)
    region = buffers.union()

    # WorldCover 分层采样
    wc = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map')
    sample_list = []
    for val, name in CLASSES:
        masked = wc.updateMask(wc.eq(val))
        pts = masked.sample(
            region=region, scale=10, numPixels=N_PER_CLASS, seed=42,
            geometries=True, tileScale=2, dropNulls=True,
        ).map(lambda f: f.set('landtype', name))
        sample_list.append(pts)
    samples = ee.FeatureCollection(sample_list).flatten()

    # 采样点 buffer 15m 成小斑块（约 7 个 10m 像元），降噪声
    def to_patch(f):
        g = f.geometry().transform(UTM, 10).buffer(15)
        return ee.Feature(g, {'landtype': f.get('landtype')})
    patches = samples.map(to_patch)

    band_names = []
    for p, _, _ in MONTHS:
        band_names += [f'{p}_{b}' for b in BANDS]
        band_names += [f'{p}_NDVI', f'{p}_EVI']

    composite = build_monthly(region)
    reduced = composite.reduceRegions(
        collection=patches, reducer=ee.Reducer.mean(), scale=10, tileScale=2,
    ).select(['landtype'] + band_names)

    feats = reduced.getInfo()['features']
    rows = [{k: f['properties'].get(k) for k in ['landtype'] + band_names}
            for f in feats]
    df = pd.DataFrame(rows)
    df.insert(0, 'fid', range(len(df)))
    df.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')
    print(f'[done] {OUT_CSV}: {df.shape}')
    print(f'  地类分布: {dict(df["landtype"].value_counts())}')


if __name__ == '__main__':
    main()
