# -*- coding: utf-8 -*-
"""extract_dachun_23counties.py — 23县库「水稻/玉米」二分类地块 S2 10波段特征提取

【状态】待影像到位后再运行（当前影像仍在下载中，缺数据跑不出结果）。

流程：
  1. 读下载清单 CSV，建立 tile -> 4期(05/06/07/08)实际日期 映射
  2. 读标签 gpkg（水稻+玉米，WGS84），按地块质心算 MGRS 瓦片归属
  3. 按瓦片批处理：一次只打开一个 tile 的 4期×10波段，批量采样地块内点
  4. 输出 features_23counties.csv（时序列名 P1..P4_{band}）+ tile_dates.json

与遂宁版 batch_extract_5districts.py 的区别：
  - 波段 7 -> 10（补 B05/B06/B07 红边）
  - 场景由「固定4个日期」改为「按 tile 从清单读各期日期」（A方案跨瓦片日期不同）
  - 列名用 P1..P4 表示时序（5/6/7/8月），避免不同瓦片日期不一致导致列错位
  - 标签字段用 QXMC(县)/SZMC(市)/类别，适配 23 县 gpkg

用法（影像下载完成后）：
  python extract_dachun_23counties.py
"""
import os, sys, json, random
import numpy as np
import numpy.ma as ma
import pandas as pd

# rasterio/geopandas 找 PROJ 数据
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

# ================= 配置 =================
LABEL_GPKG = os.path.join(BASE_DIR, "待训练数据大春", "大春标注_水稻玉米.gpkg")
LINKS_CSV  = os.path.join(BASE_DIR, "哨兵影像下载清单_大春.csv")
DL_DIR     = r"E:\迅雷下载"          # 下载根目录（若搬到服务器需改这里）
OUT_CSV    = os.path.join(BASE_DIR, "待训练数据大春", "features_23counties.csv")
OUT_JSON   = os.path.join(BASE_DIR, "待训练数据大春", "tile_dates.json")

BANDS   = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
PHASES  = ["05", "06", "07", "08"]     # 5/6/7/8 月，时序顺序固定
PREFIX  = ["P1", "P2", "P3", "P4"]     # 时序列名前缀（与训练脚本 SCENE_LABELS 对应）
N_SAMPLES = 3                           # 每个地块随机采点数
REGION = "训练侧_23县库"               # 只处理训练侧瓦片

# 遂宁 48RWU/48RWV 已按训练侧 A 方案重下完整 10 波段（见 哨兵影像下载地址_遂宁补充.txt），
# 与其他 18 县瓦片一致，无需跳过。
LOCAL_7BAND_TILES = set()


def scan_downloaded_tiles():
    """扫描 DL_DIR，返回 {tile: base_dir} 与 {tile: (minx,miny,maxx,maxy)}。

    不再用经纬度推算 MGRS 瓦片代码（此前的 mgrs_tile 对纬度带 S 的行字母
    偏移计算有误，会把地块误判到不存在的瓦片），而是直接读取每个已下载
    瓦片 B02 影像的真实范围（转 EPSG:4326），用「地块质心落在哪个瓦片范围内」
    来分配，保证与磁盘上实际下载的影像一一对应。
    """
    tile_base = {}
    tile_bounds = {}
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
            tile_bounds[tile] = b  # (minx, miny, maxx, maxy)
        except Exception as e:
            print(f"[scan] 读取 {d} 失败: {e}")
    return tile_base, tile_bounds


def load_tile_dates():
    """读下载清单，返回 tile -> {phase: date} 映射（仅训练侧）。"""
    df = pd.read_csv(LINKS_CSV, encoding="utf-8-sig")
    df = df[df["region"] == REGION]
    tile_dates = {}
    for _, row in df.iterrows():
        tile = str(row["tile"])
        phase = str(row["window"]).split("_")[0]
        if phase in PHASES:
            tile_dates.setdefault(tile, {})[phase] = row["date"]
    return tile_dates


def sample_parcel_points(geom, n, rng):
    """在地块多边形内随机采 n 个点，返回 [(x, y), ...]（输入 geom 需已是影像 CRS）。"""
    points = []
    minx, miny, maxx, maxy = geom.bounds
    attempts = 0
    while len(points) < n and attempts < n * 40:
        x = rng.uniform(minx, maxx)
        y = rng.uniform(miny, maxy)
        if geom.contains(Point(x, y)):
            points.append((x, y))
        attempts += 1
    # 随机采样不足时用质心补
    while len(points) < n:
        c = geom.centroid
        points.append((c.x, c.y))
    return points


def sample_band(dataset, points):
    """批量采样单波段所有点，越界/nodata 用 np.nan，返回与 points 等长的 list。"""
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
    """对一个瓦片内所有地块提取 4期×10波段 zonal mean 特征。

    Args:
        tile: MGRS 瓦片代码
        base: 该瓦片的本地目录（由 scan_downloaded_tiles 发现，前缀可能是 0814 或 0817）
        parcel_idxs: 落在该瓦片的地块（gdf 原始 index）列表
        gdf: 标签 GeoDataFrame（WGS84）
    Returns:
        DataFrame（每行一个地块，列 P1_B02..P4_B12 + 元数据），或 None
    """
    first_b02 = os.path.join(base, PHASES[0], "B02.tif")
    if not os.path.exists(first_b02):
        print(f"[{tile}] 影像目录不存在: {base}（跳过）")
        return None

    with rasterio.open(first_b02) as ref:
        crs = ref.crs

    # 投影该瓦片地块到影像 CRS 并生成采样点
    sub = gdf.loc[parcel_idxs].to_crs(crs)
    parcel_points = {}
    for idx in sub.index:
        geom = sub.loc[idx].geometry
        if geom is None or geom.is_empty:
            continue
        parcel_points[idx] = sample_parcel_points(geom, N_SAMPLES, rng)
    if not parcel_points:
        return None

    # 展平所有采样点（顺序与 parcel_points 一致，便于后续聚合）
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
            # 按地块聚合取均值
            off = 0
            for idx, pts in parcel_points.items():
                pv = vals[off:off + len(pts)]
                pv = [v for v in pv if not np.isnan(v)]
                rows[idx][col] = float(np.mean(pv)) if pv else np.nan
                off += len(pts)

    # 组装 DataFrame
    recs = []
    for idx in parcel_points:
        r = dict(rows[idx])
        r["tile"] = tile
        r["SZMC"] = gdf.loc[idx, "SZMC"]
        r["QXMC"] = gdf.loc[idx, "QXMC"]
        r["TBMJ"] = gdf.loc[idx, "TBMJ"] if "TBMJ" in gdf.columns else np.nan
        r["类别"] = gdf.loc[idx, "类别"]
        recs.append(r)
    return pd.DataFrame(recs)


def main():
    rng = random.Random(42)
    tile_dates = load_tile_dates()
    print(f"训练侧瓦片(清单): {len(tile_dates)} 个 -> {sorted(tile_dates)}")

    tile_base, tile_bounds = scan_downloaded_tiles()
    print(f"已下载瓦片(磁盘): {len(tile_base)} 个 -> {sorted(tile_base)}")

    gdf = gpd.read_file(LABEL_GPKG)
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)
    print(f"标签地块: {len(gdf)}")

    # 地块 -> 瓦片归属：用「质心是否落在已下载影像真实范围」分配（不再用 mgrs_tile）
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
        if hit is None or hit not in tile_dates:
            uncovered.append(hit)
            continue
        tile_to_idxs.setdefault(hit, []).append(idx)

    print(f"分配到训练侧瓦片的地块: {sum(len(v) for v in tile_to_idxs.values())}")
    if uncovered:
        from collections import Counter
        print(f"  未覆盖或非训练侧的地块: {len(uncovered)}")
        print(f"  分布: {Counter(['无覆盖' if t is None else t for t in uncovered])}")

    all_dfs = []
    for tile in sorted(tile_to_idxs):
        if tile in LOCAL_7BAND_TILES:
            print(f"[{tile}] 本地仅 7 波段（缺 B05/B06/B07），跳过并提示补数据")
            continue
        if tile not in tile_base:
            print(f"[{tile}] 无本地目录，跳过")
            continue
        print(f"\n处理瓦片 [{tile}]，地块 {len(tile_to_idxs[tile])}，日期 {tile_dates[tile]}")
        df = extract_tile(tile, tile_base[tile], tile_to_idxs[tile], gdf, rng)
        if df is not None and len(df) > 0:
            all_dfs.append(df)

    if not all_dfs:
        print("ERROR: 无任何有效数据（影像是否已到位？）")
        return

    merged = pd.concat(all_dfs, ignore_index=True)
    band_cols = [f"{p}_{b}" for p in PREFIX for b in BANDS]
    n_before = len(merged)
    merged = merged.dropna(subset=band_cols, how="any")  # 任一波段缺失则丢弃该地块
    print(f"\n合并地块: {n_before} -> 剔除含缺失波段后 {len(merged)}")

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    merged.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    # 保存 tile -> 各期日期映射（供追溯/复现）
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(tile_dates, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(f"输出: {OUT_CSV}")
    print(f"日期映射: {OUT_JSON}")
    print(f"样本: {len(merged)}")
    print(f"类别分布:\n{merged['类别'].value_counts().to_string()}")
    print(f"县分布（前20）:\n{merged['QXMC'].value_counts().head(20).to_string()}")


if __name__ == "__main__":
    main()
