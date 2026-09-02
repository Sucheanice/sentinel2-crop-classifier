# -*- coding: utf-8 -*-
"""extract_gaoliang.py — 提取「高粱」地块的 S2 10波段特征（复用 23 县库本地瓦片）

标签源：大春训练标注库_完整字段.gpkg（ZWMC='高粱'）
影像源：E:\迅雷下载（与 extract_dachun_23counties.py 相同的本地瓦片）
输出：待训练数据大春\gaoliang_features.csv（列结构与 features_23counties.csv 对齐）
"""
import os
import random
import numpy as np
import numpy.ma as ma
import pandas as pd

for _p in [
    r"C:\Users\lenovo\AppData\Roaming\Python\Python313\site-packages\rasterio\proj_data",
    r"C:\Users\lenovo\AppData\Roaming\Python\Python312\site-packages\rasterio\proj_data",
    r"C:\Users\lenovo\AppData\Roaming\Python\Python311\site-packages\rasterio\proj_data",
]:
    if os.path.isdir(_p):
        os.environ["PROJ_LIB"] = _p
        break

import geopandas as gpd
import rasterio
from rasterio.sample import sample_gen
from rasterio.warp import transform_bounds
from shapely.geometry import Point

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

LABEL_GPKG = os.path.join(BASE_DIR, "大春训练标注库_完整字段.gpkg")
DL_DIR = r"E:\迅雷下载"
OUT_CSV = os.path.join(BASE_DIR, "待训练数据大春", "gaoliang_features.csv")

BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
PHASES = ["05", "06", "07", "08"]
PREFIX = ["P1", "P2", "P3", "P4"]
N_SAMPLES = 3


def scan_downloaded_tiles():
    tile_base, tile_bounds = {}, {}
    if not os.path.isdir(DL_DIR):
        return tile_base, tile_bounds
    for d in os.listdir(DL_DIR):
        if not d.endswith("-5-8"):
            continue
        parts = d.split("-")
        if len(parts) < 2:
            continue
        tile = parts[1]
        b02 = os.path.join(DL_DIR, d, PHASES[0], "B02.tif")
        if not os.path.exists(b02):
            continue
        tile_base[tile] = os.path.join(DL_DIR, d)
        try:
            with rasterio.open(b02) as src:
                b = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
            tile_bounds[tile] = b
        except Exception as e:
            print(f"[scan] 读取 {d} 失败: {e}")
    return tile_base, tile_bounds


def sample_parcel_points(geom, n, rng):
    pts = []
    minx, miny, maxx, maxy = geom.bounds
    attempts = 0
    while len(pts) < n and attempts < n * 40:
        x = rng.uniform(minx, maxx)
        y = rng.uniform(miny, maxy)
        if geom.contains(Point(x, y)):
            pts.append((x, y))
        attempts += 1
    while len(pts) < n:
        c = geom.centroid
        pts.append((c.x, c.y))
    return pts


def sample_band(dataset, points):
    vals = []
    for arr in sample_gen(dataset, points):
        v = arr[0]
        if ma.is_masked(v) or v is None:
            vals.append(np.nan)
        else:
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                vals.append(np.nan)
    return vals


def extract_tile(tile, base, parcel_idxs, gdf, rng):
    first_b02 = os.path.join(base, PHASES[0], "B02.tif")
    if not os.path.exists(first_b02):
        return None
    with rasterio.open(first_b02) as ref:
        crs = ref.crs

    sub = gdf.loc[parcel_idxs].to_crs(crs)
    parcel_points = {}
    for idx in sub.index:
        geom = sub.loc[idx].geometry
        if geom is None or geom.is_empty:
            continue
        parcel_points[idx] = sample_parcel_points(geom, N_SAMPLES, rng)
    if not parcel_points:
        return None

    flat = []
    for pts in parcel_points.values():
        flat.extend(pts)

    rows = {idx: {} for idx in parcel_points}
    for pi, phase in enumerate(PHASES):
        col_prefix = PREFIX[pi]
        ph_dir = os.path.join(base, phase)
        for band in BANDS:
            bp = os.path.join(ph_dir, f"{band}.tif")
            col = f"{col_prefix}_{band}"
            if not os.path.exists(bp):
                for idx in parcel_points:
                    rows[idx][col] = np.nan
                continue
            with rasterio.open(bp) as ds:
                vals = sample_band(ds, flat)
            off = 0
            for idx, pts in parcel_points.items():
                pv = vals[off:off + len(pts)]
                pv = [v for v in pv if not np.isnan(v)]
                rows[idx][col] = float(np.mean(pv)) if pv else np.nan
                off += len(pts)

    recs = []
    for idx in parcel_points:
        r = dict(rows[idx])
        r["tile"] = tile
        r["SZMC"] = gdf.loc[idx, "SZMC"]
        r["QXMC"] = gdf.loc[idx, "QXMC"]
        r["TBMJ"] = gdf.loc[idx, "TBMJ"] if "TBMJ" in gdf.columns else np.nan
        r["类别"] = "高粱"
        recs.append(r)
    return pd.DataFrame(recs)


def main():
    rng = random.Random(42)
    tile_base, tile_bounds = scan_downloaded_tiles()
    print(f"已下载瓦片(磁盘): {len(tile_base)} 个")

    gdf = gpd.read_file(LABEL_GPKG)
    gdf = gdf[gdf["ZWMC"].astype(str).str.strip() == "高粱"].copy()
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)
    print(f"高粱地块: {len(gdf)}")
    print(gdf["QXMC"].value_counts().to_string())

    tile_to_idxs = {}
    uncovered = []
    for idx, geom in gdf.geometry.items():
        if geom is None or geom.is_empty:
            continue
        c = geom.centroid
        x, y = c.x, c.y
        hit = None
        for t, (minx, miny, maxx, maxy) in tile_bounds.items():
            if minx <= x <= maxx and miny <= y <= maxy:
                hit = t
                break
        if hit is None or hit not in tile_base:
            uncovered.append(hit)
            continue
        tile_to_idxs.setdefault(hit, []).append(idx)

    covered = sum(len(v) for v in tile_to_idxs.values())
    print(f"\n分配到已下载瓦片的地块: {covered} / {len(gdf)}")
    if uncovered:
        from collections import Counter
        print(f"  未覆盖: {len(uncovered)} -> {Counter(['无瓦片' if t is None else t for t in uncovered])}")

    all_dfs = []
    for tile in sorted(tile_to_idxs):
        print(f"处理瓦片 [{tile}]，地块 {len(tile_to_idxs[tile])}")
        df = extract_tile(tile, tile_base[tile], tile_to_idxs[tile], gdf, rng)
        if df is not None and len(df) > 0:
            all_dfs.append(df)

    if not all_dfs:
        print("ERROR: 无有效数据")
        return

    merged = pd.concat(all_dfs, ignore_index=True)
    band_cols = [f"{p}_{b}" for p in PREFIX for b in BANDS]
    n_before = len(merged)
    merged = merged.dropna(subset=band_cols, how="any")
    print(f"\n合并: {n_before} -> 剔除缺波段后 {len(merged)}")

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    merged.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"输出: {OUT_CSV}")
    print(f"县分布:\n{merged['QXMC'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
