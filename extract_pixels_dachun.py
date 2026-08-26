# -*- coding: utf-8 -*-
"""extract_pixels_dachun.py — 从遂宁5区 SHP + 大春 S2 提取像素级训练数据 (水稻 vs 玉米)。"""
import os, sys, time
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.ops import unary_union
import rasterio
from rasterio.features import rasterize
from rasterio.windows import from_bounds, Window
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# === 五区 SHP ===
SHP_BASE = os.path.join(BASE_DIR, "蓬溪县数据", "遂宁市矢量数据原始")
DISTRICTS = {
    "安居区": os.path.join(SHP_BASE, "安居区", "矢量", "安居区.shp"),
    "大英县": os.path.join(SHP_BASE, "大英县", "矢量", "大英县.shp"),
    "射洪市": os.path.join(SHP_BASE, "射洪市", "矢量", "射洪市.shp"),
    "船山区": os.path.join(SHP_BASE, "船山区", "矢量", "船山区.shp"),
    "蓬溪县": os.path.join(SHP_BASE, "蓬溪县", "蓬溪县矢量数据", "蓬溪县.shp"),
}

# === 大春 S2 ===
S2_DIR = os.path.join(BASE_DIR, "蓬溪县数据", "s2_pengnan")
SCENE_DATES = ["2025-05-20", "2025-06-26", "2025-07-16", "2025-08-03"]
ALL_BANDS = ["B02","B03","B04","B05","B06","B07","B08","B8A","B11","B12"]
BANDS_20M = {"B05","B06","B07","B8A","B11","B12"}

OUTPUT_DIR = os.path.join(BASE_DIR, "dachun_pixel")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "pixels_dachun.csv")

BUFFER_DIST = -5.0
MIN_PX = 3

sys.path.insert(0, BASE_DIR)
from common import compute_feature_matrix


def find_s2_scenes():
    """查找大春 S2 场景目录。"""
    scenes = {}
    for d in os.listdir(S2_DIR):
        dp = os.path.join(S2_DIR, d)
        if not os.path.isdir(dp): continue
        b02 = os.path.join(dp, "B02.tif")
        if os.path.exists(b02):
            parts = d.split('_')
            date = parts[0] if parts else ''
            if len(date) == 10 and date in SCENE_DATES:
                scenes[date] = dp
    return scenes


def main():
    t0 = time.time()
    print("=" * 70)
    print("大春 Pixel-Level 训练数据提取 (水稻 vs 玉米)")
    print("=" * 70)

    # 1. 查找 S2
    print("\n[1] 查找大春 S2 场景...")
    s2_scenes = find_s2_scenes()
    for d in SCENE_DATES:
        status = "✓" if d in s2_scenes else "✗"
        print(f"  {d}: {status}")
    missing = [d for d in SCENE_DATES if d not in s2_scenes]
    if missing:
        print(f"ERROR: 缺失场景 {missing}")
        return

    # 获取参考 CRS
    ref_path = os.path.join(s2_scenes[SCENE_DATES[0]], "B02.tif")
    with rasterio.open(ref_path) as r:
        s2_crs = r.crs
    print(f"  S2 CRS: {s2_crs}")

    # 2. 加载所有区县 SHP
    print(f"\n[2] 加载五区 SHP (筛选水稻/玉米)...")
    all_rows = []
    for name, shp_path in DISTRICTS.items():
        if not os.path.exists(shp_path):
            print(f"  {name}: MISSING!")
            continue
        gdf = gpd.read_file(shp_path)
        if 'ZWMC' not in gdf.columns:
            print(f"  {name}: 无 ZWMC 字段, 跳过")
            continue
        sub = gdf[gdf['ZWMC'].isin(['水稻', '玉米'])].copy()
        sub['district'] = name
        print(f"  {name}: {len(sub)} 地块 (水稻 {(sub['ZWMC'] == '水稻').sum()}, 玉米 {(sub['ZWMC'] == '玉米').sum()})")
        all_rows.append(sub)

    gdf_all = pd.concat(all_rows, ignore_index=True)
    print(f"  总计: {len(gdf_all)} 地块")

    # 3. 投影 + 缓冲
    print(f"\n[3] 投影 + 缓冲 (buffer={BUFFER_DIST}m)...")
    gdf_proj = gdf_all.to_crs(s2_crs)
    bufs = []
    for i, geom in enumerate(gdf_proj.geometry):
        if geom is None or geom.is_empty: continue
        buf = geom.buffer(BUFFER_DIST)
        if buf.is_empty or buf.area <= 0: continue
        bufs.append((i, buf))
    print(f"  有效地块: {len(bufs)}/{len(gdf_all)}")

    # 4. 逐场景提取像素
    print(f"\n[4] 逐场景逐波段提取像素值...")
    union_g = unary_union([b for _, b in bufs])
    minx, miny, maxx, maxy = union_g.bounds

    # 先建 ref window
    with rasterio.open(ref_path) as ref_src:
        win_full = from_bounds(minx - 200, miny - 200, maxx + 200, maxy + 200,
                               transform=ref_src.transform)
        win_full = Window(
            max(0, int(win_full.col_off)), max(0, int(win_full.row_off)),
            min(ref_src.width - int(win_full.col_off), int(win_full.width)),
            min(ref_src.height - int(win_full.row_off), int(win_full.height)))
        win_t = ref_src.transform * ref_src.transform.translation(
            win_full.col_off, win_full.row_off)

    # Rasterize 全窗口
    print(f"  窗口: {win_full.width}x{win_full.height}")
    pids = [i for i, _ in bufs]
    shapes = [(bufs[j][1], bufs[j][0] + 1) for j in range(len(bufs))]
    label_img = rasterize(shapes, out_shape=(win_full.height, win_full.width),
                          transform=win_t, fill=0, dtype=np.int32)

    px_counts = np.bincount(label_img.ravel(), minlength=len(gdf_all) + 2)
    valid_ids = set(pid for pid in range(1, len(gdf_all) + 1)
                    if px_counts[pid] >= MIN_PX)
    valid_mask = np.isin(label_img, list(valid_ids))
    n_valid_px = valid_mask.sum()
    print(f"  有效像素: {n_valid_px} (来自 {len(valid_ids)} 个地块)")

    # 提取各波段
    band_vals = {}
    for date in SCENE_DATES:
        scene_dir = s2_scenes[date]
        print(f"  [{date}] 提取...")
        for band in ALL_BANDS:
            bp = os.path.join(scene_dir, f"{band}.tif")
            key = f"{date}_{band}"
            if not os.path.exists(bp):
                band_vals[key] = np.zeros(n_valid_px, dtype=np.float32)
                continue
            with rasterio.open(bp) as src:
                data = src.read(1, window=win_full).astype(np.float32)
                if band in BANDS_20M:
                    data = np.repeat(np.repeat(data, 2, axis=0), 2, axis=1)[
                        :win_full.height, :win_full.width]
                band_vals[key] = data[valid_mask]
        print(f"    {len(band_vals)} 个波段")

    # 5. 像素→地块 ID + 作物标签
    print(f"\n[5] 组装标签...")
    flat_labels = label_img[valid_mask]
    parcel_ids = []
    crop_names = []
    district_names = []
    for px_label in flat_labels:
        idx = px_label - 1
        parcel_ids.append(idx)
        crop_names.append(gdf_all.iloc[idx]['ZWMC'])
        district_names.append(gdf_all.iloc[idx]['district'])

    # 6. 计算特征矩阵
    print(f"\n[6] 计算特征矩阵 ({n_valid_px} 像素)...")
    features_arr, feat_names = compute_feature_matrix(
        band_vals, SCENE_DATES, available_bands=ALL_BANDS)
    print(f"  特征维度: {features_arr.shape}")

    # 7. 保存
    print(f"\n[7] 保存 CSV...")
    df_out = pd.DataFrame(features_arr, columns=feat_names)
    df_out.insert(0, 'district', district_names)
    df_out.insert(0, 'ZWMC', crop_names)
    df_out.insert(0, 'parcel_id', parcel_ids)

    df_out.to_csv(OUTPUT_CSV, index=False)
    size_mb = os.path.getsize(OUTPUT_CSV) / 1e6
    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"完成: {elapsed:.0f}s, {size_mb:.1f}MB")
    print(f"样本: {len(df_out)} 像素, {len(valid_ids)} 地块")
    print(f"水稻: {(df_out['ZWMC'] == '水稻').sum()}, 玉米: {(df_out['ZWMC'] == '玉米').sum()}")
    print(f"输出: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
