# -*- coding: utf-8 -*-
import geopandas as gpd
import rasterio
from rasterio import features
from rasterio.warp import Resampling
import numpy as np
import pandas as pd
import os, time

SHP_SEARCH_DIR = r"E:\工作相关\2026年\0624 待测试数据\待训练数据\矢量\人保-仅作物面"
DATA_DIR = r"E:\工作相关\2026年\0624 待测试数据\待训练数据4"
BANDS = ["B02", "B03", "B04", "B08"]
OUTPUT = r"E:\工作相关\2026年\0624 待测试数据\待训练数据\features_anju_correct.csv"

SCL_CLOUD = {0, 1, 3, 8, 9, 10, 11}

def find_correct_shp(search_dir):
    candidates = []
    for f in os.listdir(search_dir):
        if f.endswith('.shp'):
            fp = os.path.join(search_dir, f)
            gdf = gpd.read_file(fp, encoding='utf-8')
            b = gdf.total_bounds
            center_lat = (b[1] + b[3]) / 2
            candidates.append((fp, len(gdf), b, center_lat, os.path.getsize(fp)))
    for fp, n, b, clat, sz in sorted(candidates, key=lambda x: x[3]):
        in_range = 30.0 < clat < 30.6
        print("  %s: %d parcels, lat=%.2f, %.0f MB, in_48RWU=%s" % (
            os.path.basename(fp), n, clat, sz/1024/1024, in_range))
    in_rwu = [c for c in candidates if 30.0 < c[3] < 30.6]
    if not in_rwu:
        raise RuntimeError("No SHP found in 48RWU range (30.0-30.6N)")
    best = max(in_rwu, key=lambda x: x[1])
    print("  -> Selected: %s (%d parcels)" % (os.path.basename(best[0]), best[1]))
    return best[0]

def get_scene_dirs(data_dir):
    dirs = sorted([d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d)) and d[0].isdigit()])
    result = []
    for d in dirs:
        dp = os.path.join(data_dir, d)
        parts = d.split("_")
        label = parts[1] if len(parts) >= 2 else d
        result.append((d, dp, label))
    return result

def rasterize_labels(gdf, transform, width, height):
    shapes = ((geom, i + 1) for i, geom in enumerate(gdf.geometry))
    labels = features.rasterize(
        shapes=shapes, out_shape=(height, width),
        transform=transform, fill=0, dtype=np.int32, all_touched=True
    )
    return labels

def zonal_mean(feature_2d, labels_2d, n_labels):
    valid = (labels_2d > 0) & (feature_2d > 0)
    labels_valid = labels_2d[valid]
    values_valid = feature_2d[valid]
    if len(values_valid) == 0:
        return np.full(n_labels, np.nan, dtype=np.float32)
    sums = np.bincount(labels_valid, weights=values_valid, minlength=n_labels + 1)[1:]
    counts = np.bincount(labels_valid, minlength=n_labels + 1)[1:].astype(np.float32)
    means = np.full(n_labels, np.nan, dtype=np.float32)
    valid_labels = counts > 0
    means[valid_labels] = sums[valid_labels] / counts[valid_labels]
    return means

def zonal_mean_with_cloud_mask(feature_2d, labels_2d, cloud_mask_2d, n_labels):
    valid = (labels_2d > 0) & (feature_2d > 0) & (~cloud_mask_2d)
    labels_valid = labels_2d[valid]
    values_valid = feature_2d[valid]
    if len(values_valid) == 0:
        return np.full(n_labels, np.nan, dtype=np.float32), np.ones(n_labels, dtype=np.float32)

    all_labels_valid = labels_2d > 0
    all_labs = labels_2d[all_labels_valid]
    total_counts = np.bincount(all_labs, minlength=n_labels + 1)[1:].astype(np.float32)

    sums = np.bincount(labels_valid, weights=values_valid, minlength=n_labels + 1)[1:]
    counts = np.bincount(labels_valid, minlength=n_labels + 1)[1:].astype(np.float32)
    means = np.full(n_labels, np.nan, dtype=np.float32)
    valid_labels = counts > 0
    means[valid_labels] = sums[valid_labels] / counts[valid_labels]
    cloud_frac = np.where(total_counts > 0, 1.0 - counts / total_counts, 1.0)
    return means, cloud_frac

def main():
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

    print("Finding correct SHP in 48RWU range...")
    shp_path = find_correct_shp(SHP_SEARCH_DIR)

    print("\nLoading SHP...")
    gdf = gpd.read_file(shp_path, encoding='utf-8')
    n = len(gdf)
    b = gdf.total_bounds
    print("  %d parcels, bounds: (%.4f,%.4f)-(%.4f,%.4f)" % (n, b[0], b[1], b[2], b[3]))
    print("  CRS: %s" % gdf.crs)
    print("  Columns: %s" % list(gdf.columns)[:15])

    crop_col = None
    for c in ['bz', 'BZ', 'ZWMC']:
        if c in gdf.columns:
            crop_col = c
            break
    if crop_col:
        print("  Crop column [%s]: %d unique types" % (crop_col, gdf[crop_col].nunique()))
        for ct, cnt in gdf[crop_col].value_counts().head(10).items():
            print("    %s: %d" % (ct, cnt))

    scenes = get_scene_dirs(DATA_DIR)
    n_scenes = len(scenes)
    n_feats = n_scenes * len(BANDS)
    print("\nScenes: %d x %d bands = %d features" % (n_scenes, len(BANDS), n_feats))
    for i, (d, dp, label) in enumerate(scenes):
        print("  [%d] %s -> %s" % (i+1, d, label))

    features = np.full((n, n_feats), np.nan, dtype=np.float32)
    cloud_fracs = np.full((n, n_scenes), np.nan, dtype=np.float32)

    first_band_path = os.path.join(scenes[0][1], "B02.tif")
    with rasterio.open(first_band_path) as ref_ds:
        ref_transform = ref_ds.transform
        ref_width = ref_ds.width
        ref_height = ref_ds.height
        ref_crs = ref_ds.crs

    print("\nBand size: %d x %d, CRS: %s" % (ref_width, ref_height, ref_crs))

    if gdf.crs != ref_crs:
        print("Reprojecting SHP from %s to %s..." % (gdf.crs, ref_crs), end="", flush=True)
        t_rep = time.time()
        gdf = gdf.to_crs(ref_crs)
        print(" done (%.1fs)" % (time.time() - t_rep))

    print("Rasterizing %d parcels..." % n, end="", flush=True)
    t0 = time.time()
    labels_full = rasterize_labels(gdf, ref_transform, ref_width, ref_height)
    print(" done (%.1fs)" % (time.time() - t0))
    print("Labels range: %d-%d, parcels hit: %d" % (
        labels_full.min(), labels_full.max(), len(np.unique(labels_full)) - 1))

    col_names = []

    for si, (dir_name, dp, label) in enumerate(scenes):
        t_scene = time.time()
        print("\n[%d/%d] %s" % (si + 1, n_scenes, label))

        scl_path = os.path.join(dp, "SCL.tif")
        cloud_mask_10m = None

        if os.path.exists(scl_path):
            print("  SCL: reading...", end="", flush=True)
            with rasterio.open(scl_path) as scl_ds:
                scl_data = scl_ds.read(
                    1,
                    out_shape=(ref_height, ref_width),
                    resampling=Resampling.nearest
                )
                cloud_mask_10m = np.isin(scl_data, list(SCL_CLOUD))
                cloud_pct = cloud_mask_10m.mean() * 100
            print(" %.1f%% cloud" % cloud_pct)

        for bi, band in enumerate(BANDS):
            feat_idx = si * len(BANDS) + bi
            feat_name = "%s_%s" % (label, band)
            col_names.append(feat_name)

            tif_path = os.path.join(dp, "%s.tif" % band)
            if not os.path.exists(tif_path):
                print("    %s: MISSING" % band)
                continue

            print("    %s: " % band, end="", flush=True)
            t_band = time.time()
            with rasterio.open(tif_path) as band_ds:
                band_data = band_ds.read(1)

            if band_data.shape != labels_full.shape:
                with rasterio.open(tif_path) as band_ds:
                    band_data = band_ds.read(
                        1, out_shape=(ref_height, ref_width),
                        resampling=Resampling.bilinear
                    )

            if cloud_mask_10m is not None:
                means, cfs = zonal_mean_with_cloud_mask(
                    band_data, labels_full, cloud_mask_10m, n
                )
                if bi == 0:
                    cloud_fracs[:, si] = cfs
            else:
                means = zonal_mean(band_data, labels_full, n)

            features[:, feat_idx] = means
            done_cnt = np.sum(~np.isnan(means))
            print("%d/%d OK (%.1fs)" % (done_cnt, n, time.time() - t_band))

        print("  scene done (%.1fs)" % (time.time() - t_scene))

    print("\nComputing NDVI...")
    ndvi_cols = []
    ndvi_data = []
    for si in range(n_scenes):
        label = scenes[si][2]
        nir_idx = si * len(BANDS) + 3
        red_idx = si * len(BANDS) + 2
        ndvi_n = features[:, nir_idx]
        ndvi_r = features[:, red_idx]
        ndvi = np.where((ndvi_n + ndvi_r) > 0, (ndvi_n - ndvi_r) / (ndvi_n + ndvi_r), np.nan)
        ndvi_cols.append("NDVI_%s" % label)
        ndvi_data.append(ndvi)

    all_features = np.column_stack([features] + ndvi_data)
    all_cols = col_names + ndvi_cols

    print("Building DataFrame...")
    df = pd.DataFrame(all_features, columns=all_cols)
    df.insert(0, "parcel_id", range(n))
    if crop_col:
        df.insert(1, "crop_type", gdf[crop_col].values)
    df.insert(2, "area_m2", gdf.geometry.area.values)

    for si in range(n_scenes):
        label = scenes[si][2]
        df["cloud_frac_%s" % label] = cloud_fracs[:, si]

    df.to_csv(OUTPUT, index=False, encoding='utf-8-sig')
    print("\nSaved: %s" % OUTPUT)
    print("Shape: %s" % str(df.shape))

    feats_only = all_features
    complete = ~np.isnan(feats_only).any(axis=1)
    print("Complete records: %d/%d (%.1f%%)" % (complete.sum(), n, 100*complete.sum()/n))

    if crop_col:
        print("Crop types: %d" % df['crop_type'].nunique())
        print("Top crop types:")
        for ct, cnt in df["crop_type"].value_counts().head(15).items():
            complete_ct = df[df["crop_type"] == ct]
            complete_in_ct = complete_ct.dropna(subset=all_cols).shape[0]
            print("  %s: %d total, %d complete" % (ct, cnt, complete_in_ct))

if __name__ == "__main__":
    main()
