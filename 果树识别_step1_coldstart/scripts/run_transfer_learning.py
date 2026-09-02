# -*- coding: utf-8 -*-
"""
果树识别 Step1 转移学习版:
  叙永(Xuyong)SHP标注训练 → 遂宁(Suining)推理

训练区: 泸州市叙永县 (S2 48RWR) — 使用农业保险果树地块(李子+柑橘)作标签
推理区: 遂宁市 (S2 48RWU)   — 使用本地已有S2数据

标签来源: 叙永SHP (result.shp) — 276个果树地块, 李子+柑橘, 2026大春承保数据
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
import warnings

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.warp import reproject, calculate_default_transform, Resampling
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from rasterio.windows import Window
from shapely.ops import unary_union
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

# ============================================================
# 0. 路径与常量
# ============================================================
WORK_DIR = Path(r"E:\工作相关\2026年\0624 待测试数据")
PROJ_DIR = WORK_DIR / "果树识别_step1_coldstart"
DATA_DIR = PROJ_DIR / "data"
OUTPUT_DIR = PROJ_DIR / "outputs"
(XY_S2_DIR := DATA_DIR / "xuyong_s2").mkdir(parents=True, exist_ok=True)
(XY_FEAT_DIR := DATA_DIR / "xuyong_features").mkdir(parents=True, exist_ok=True)
(XY_MODEL_DIR := PROJ_DIR / "models").mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SHP_PATH = WORK_DIR / "待训练数据水果" / "叙永水果成果数据" / "result.shp"

# 遂宁本地S2路径
SN_S2_DIRS = {
    "main":    WORK_DIR / "小春_s2_48RWU",
    "extra":   WORK_DIR / "小春_s2_48RWU_extra",
}
SN_WC = WORK_DIR / "ESA_WorldCover_10m_2021_v200_N30E105_Map.tif"
SN_S2_CACHE = DATA_DIR / "suining_features" / "feature_cube.npy"
SN_TRANSFORM_CACHE = DATA_DIR / "suining_features" / "transform.json"

# 遂宁区域（与 run_suining_binary.py 一致）
SUINING_BOUNDS_WGS84 = (105.25, 30.25, 105.85, 30.75)  # 约60km×56km

# 叙永区域（SHP覆盖范围 + 少许buffer）
XUYONG_BOUNDS_WGS84 = (105.37, 27.69, 105.67, 27.80)   # 约30km×12km

# S2场景 (48RWR) — 4季 (含夏季, 与遂宁对齐)
SCENES_48RWR = {
    "late_autumn":  {"date": "2024-10-29", "id": "S2B_MSIL2A_20241029T032749_R018_T48RWR_20241029T062836"},
    "winter":       {"date": "2025-01-05", "id": "S2A_MSIL2A_20250105T034131_R061_T48RWR_20250105T080351"},
    "spring":       {"date": "2025-03-26", "id": "S2C_MSIL2A_20250326T033601_R061_T48RWR_20250326T085914"},
    "late_spring":  {"date": "2025-05-12", "id": "S2C_MSIL2A_20250512T032531_R018_T48RWR_20250512T075838"},
    "summer":       {"date": "2025-07-16", "id": "S2B_MSIL2A_20250716T032519_R018_T48RWR_20250716T055526"},
}

# 叙永 SAR 后向散射 dB (已对齐到叙永 S2 网格 2963x1232, nodata=-9999)
# label 与遂宁侧 SAR_SCENES_SUINING 一一对应 (9月→autumn, 11-12月→late_autumn)
SAR_SCENES_XUYONG = {
    "autumn":      (WORK_DIR / "叙永_s1_48RWR" / "2025-09-17_S1_asc" / "vv_db.tif",
                    WORK_DIR / "叙永_s1_48RWR" / "2025-09-17_S1_asc" / "vh_db.tif"),
    "late_autumn": (WORK_DIR / "叙永_s1_48RWR" / "2024-12-03_S1_asc" / "vv_db.tif",
                    WORK_DIR / "叙永_s1_48RWR" / "2024-12-03_S1_asc" / "vh_db.tif"),
}

# 10个波段 (S2-L2A)
S2_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
S2_BANDS_RES10 = ["B02", "B03", "B04", "B08"]       # 原生10m
S2_BANDS_RES20 = ["B05", "B06", "B07", "B8A", "B11", "B12"]  # 原生20m

# 遂宁已有S2日期
SN_DATES = ["2025-01-20", "2025-03-26", "2025-04-17", "2024-12-11"]
SN_SEASON_NAMES = ["winter", "spring", "early_summer", "late_autumn"]

# 遂宁S2场景ID格式 (用于匹配48RWU目录下文件名)
SN_SCENE_IDS = {
    "2025-01-20": "S2B_MSIL2A_20250120T033959",
    "2025-03-26": "S2C_MSIL2A_20250326T033601",
    "2025-04-17": "S2B_MSIL2A_20250417T032519",
    "2024-12-11": "S2B_MSIL2A_20241211T034049",
}

LGB_PARAMS = dict(
    n_estimators=300, learning_rate=0.03, max_depth=5,
    num_leaves=31, subsample=0.7, colsample_bytree=0.6,
    reg_alpha=1.0, reg_lambda=2.0, min_child_samples=50,
    class_weight="balanced", random_state=42, n_jobs=-1,
    verbose=-1,
)


# ============================================================
# 1. 下载叙永S2影像
# ============================================================
def _search_stac_post(daterange, bbox_wgs84, max_items=3):
    """Fallback: 直接用 AWS S3 URL 构建场景列表（无需网络搜索）"""
    # 不需要网络搜索, 直接用预定义的场景ID
    return [{}]  # 仅作占位, 实际场景通过 _build_s3_urls 获取


def _build_s3_urls(scene_id, date_str):
    """根据 Sentinel-2 场景ID构建 AWS S3 公开 URL"""
    # 场景ID示例: S2B_MSIL2A_20241029T032749_R018_T48RWR_20241029T062836
    parts = scene_id.split("_")
    utm_zone = parts[4][1:3]    # 48 (从 T48RWR 提取)
    lat_band = parts[4][3]      # R
    square = parts[4][4:6]      # WR
    date_part = parts[2][:8]    # 20241029
    year = date_part[:4]
    month = str(int(date_part[4:6]))   # 去前导零 (01→1, 10→10)
    day = str(int(date_part[6:8]))     # 去前导零 (05→5, 29→29)
    seq = "0"  # 通常为0

    base = f"https://sentinel-s2-l2a.s3.eu-central-1.amazonaws.com/tiles/{utm_zone}/{lat_band}/{square}/{year}/{month}/{day}/{seq}"

    band_urls = {}
    for band in S2_BANDS:
        if band in S2_BANDS_RES10:
            band_urls[band] = f"{base}/R10m/{band}.jp2"
        else:
            band_urls[band] = f"{base}/R20m/{band}.jp2"
    return band_urls


def download_xuyong_s2():
    """从 AWS S3 下载叙永4季S2 (48RWR), 裁剪到Xuyong范围, 输出10m 10波段GeoTIFF
    JP2远程解码有兼容性问题，改用本地下载后读取。
    """
    import math
    import tempfile
    import shutil

    bbox_wgs = list(XUYONG_BOUNDS_WGS84)
    dst_crs = "EPSG:32648"

    from rasterio.warp import transform_bounds
    bbox_utm = transform_bounds("EPSG:4326", dst_crs, *bbox_wgs)
    res = 10.0
    xmin = math.floor(bbox_utm[0] / res) * res
    ymax = math.ceil(bbox_utm[3] / res) * res
    xmax = math.ceil(bbox_utm[2] / res) * res
    ymin = math.floor(bbox_utm[1] / res) * res
    width = int((xmax - xmin) / res)
    height = int((ymax - ymin) / res)
    dst_transform = rasterio.transform.from_origin(xmin, ymax, res, res)

    log(f"叙永目标范围 (UTM 48N): {width}×{height} px @ 10m")

    all_cached = all((XY_S2_DIR / f"{s}_10m_10band.tif").exists() for s in SCENES_48RWR)
    if all_cached:
        log("叙永S2: 全部4季已缓存，跳过下载")
        return {s: XY_S2_DIR / f"{s}_10m_10band.tif" for s in SCENES_48RWR}

    result = {}
    tmp_dir = Path(tempfile.mkdtemp(prefix="xuyong_s2_"))

    for season, info in SCENES_48RWR.items():
        out_path = XY_S2_DIR / f"{season}_10m_10band.tif"
        if out_path.exists():
            log(f"  [{season}] 已缓存")
            result[season] = out_path
            continue

        date_str = info["date"]
        scene_id = info["id"]
        log(f"  [{season}] {date_str} 下载10波段 (本地中转)...")

        band_urls = _build_s3_urls(scene_id, date_str)
        bands_data = []

        for band in S2_BANDS:
            url = band_urls.get(band)
            if not url:
                continue
            local_jp2 = tmp_dir / f"{season}_{band}.jp2"

            # 带重试的下载 (处理DNS间歇性失败)
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    if not local_jp2.exists() or local_jp2.stat().st_size < 1000:
                        import requests as req_dl
                        r = req_dl.get(url, timeout=120, stream=True)
                        r.raise_for_status()
                        with open(local_jp2, "wb") as f:
                            for chunk in r.iter_content(chunk_size=8192):
                                f.write(chunk)
                    # 本地读取+重投影
                    with rasterio.open(local_jp2) as src:
                        band_arr = np.zeros((height, width), dtype="float32")
                        reproject(
                            source=src.read(1),
                            destination=band_arr,
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=dst_transform,
                            dst_crs=dst_crs,
                            resampling=Resampling.bilinear,
                        )
                    bands_data.append(band_arr)
                    break  # 成功, 跳出重试
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait_s = 2 ** attempt
                        if attempt == 0:
                            log(f"    {band} 失败, {wait_s}s后重试... ({str(e)[:40]})")
                        import time
                        time.sleep(wait_s)
                    else:
                        log(f"    [WARN] 波段 {band} 失败 (重试{max_retries}次): {str(e)[:60]}")
                        if local_jp2.exists():
                            local_jp2.unlink()
                        break

        if len(bands_data) != len(S2_BANDS):
            log(f"    [WARN] 仅获取 {len(bands_data)}/{len(S2_BANDS)} 波段，跳过 {season}")
            continue

        # 写出多波段GeoTIFF
        raster = np.stack(bands_data, axis=0).astype("float32")
        profile = {
            "driver": "GTiff", "height": height, "width": width,
            "count": len(S2_BANDS), "dtype": "float32",
            "crs": dst_crs, "transform": dst_transform,
            "compress": "deflate", "tiled": True, "blockxsize": 256, "blockysize": 256,
        }
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(raster)
            for i, b in enumerate(S2_BANDS):
                dst.set_band_description(i + 1, b)
        log(f"    -> {out_path.name} ({raster.nbytes/1e6:.0f} MB)")

        # 清理该季临时文件
        for f in tmp_dir.glob(f"{season}_*.jp2"):
            f.unlink()
        result[season] = out_path

    # 清理临时目录
    shutil.rmtree(tmp_dir, ignore_errors=True)
    log(f"叙永S2下载完成: {len(result)}/4 季")
    return result if result else None


# ============================================================
# 2. 构建特征
# ============================================================
def compute_indices(bands):
    """bands: [B, H, W] float32"""
    eps = 1e-8
    b2, b3, b4, b5, b6, b7, b8, b8a, b11, b12 = [
        bands[i] for i in range(len(S2_BANDS))
    ]
    ndvi = (b8 - b4) / (b8 + b4 + eps)
    ndre = (b8a - b5) / (b8a + b5 + eps)   # NDRE: RedEdge
    ndmi = (b8a - b11) / (b8a + b11 + eps)  # NDMI: moisture
    evi = 2.5 * (b8 - b4) / (b8 + 6 * b4 - 7.5 * b2 + 1 + eps)
    return ndvi, ndre, ndmi, evi


def compute_texture(img, size=3):
    """局部标准差纹理"""
    from scipy.ndimage import uniform_filter, uniform_filter1d
    mean = uniform_filter(img.astype("float64"), size=size, mode="reflect")
    sq_mean = uniform_filter(img.astype("float64") ** 2, size=size, mode="reflect")
    return np.sqrt(np.maximum(sq_mean - mean ** 2, 0)).astype("float32")


def pad_to_match(*arrays):
    """确保所有数组形状一致 (填充或裁剪到最小公共形状)"""
    shapes = [a.shape for a in arrays]
    min_h = min(s[0] for s in shapes)
    min_w = min(s[1] for s in shapes)
    return tuple(a[:min_h, :min_w] for a in arrays)


def _read_sar_band(path, H, W):
    """读取已对齐到目标网格的 SAR dB 波段, nodata(-9999)→NaN"""
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
    arr[arr <= -9000] = np.nan
    return arr[:H, :W]


def build_feature_cube(s2_paths, dst_area_label, sar_scenes=None):
    """
    构建特征立方体。
    s2_paths: {season: tif_path}, season名如 'late_autumn','winter','spring','peak_summer'
    sar_scenes: {label: (vv_path, vh_path)}, 已对齐到目标网格的 SAR dB
    返回: (cube, band_names) — cube shape=(F, H, W), band_names 用于对齐
    """
    cache_file = XY_FEAT_DIR / f"features_{dst_area_label}.npy"
    names_file = XY_FEAT_DIR / f"band_names_{dst_area_label}.json"
    if cache_file.exists() and names_file.exists():
        log(f"特征: 已有缓存 {cache_file.name}，跳过构建")
        with open(names_file) as f:
            band_names = json.load(f)
        return np.load(cache_file), band_names

    # 如果没有缓存但有旧版(无names), 删除重建
    if cache_file.exists():
        cache_file.unlink()

    log(f"构建特征立方体 ({dst_area_label})...")

    seasons = sorted(s2_paths.keys())  # 固定顺序
    s2_data = {}
    for s in seasons:
        p = s2_paths[s]
        with rasterio.open(p) as src:
            arr = src.read().astype("float32")
            s2_data[s] = arr
            log(f"  [{s}] {p.name}: {arr.shape}")

    # 对齐形状
    ref_h = min(a.shape[1] for a in s2_data.values())
    ref_w = min(a.shape[2] for a in s2_data.values())
    for s in seasons:
        s2_data[s] = s2_data[s][:, :ref_h, :ref_w]

    H, W = ref_h, ref_w
    features = []
    band_names = []

    # 1. 多时相光谱
    for s in seasons:
        arr = s2_data[s]
        features.append(arr.reshape(len(S2_BANDS), H, W))
        for b in S2_BANDS:
            band_names.append(f"s2_{s}_{b}")
    log(f"  光谱: {len(seasons)}×{len(S2_BANDS)}={len(seasons)*len(S2_BANDS)}")

    # 2. 植被指数
    indices = {}
    for s in seasons:
        ndvi, ndre, ndmi, evi = compute_indices(s2_data[s])
        indices[s] = {"ndvi": ndvi, "ndre": ndre, "ndmi": ndmi, "evi": evi}
        features.extend([ndvi[None], ndre[None], ndmi[None], evi[None]])
        for vi in ["ndvi", "ndre", "ndmi", "evi"]:
            band_names.append(f"vi_{s}_{vi}")
    log(f"  指数: {len(seasons)}×4={len(seasons)*4}")

    # 3. 物候差值 (相邻季NDVI/NDRE差)
    for i in range(len(seasons) - 1):
        s1, s2 = seasons[i], seasons[i + 1]
        diff_ndvi = indices[s2]["ndvi"] - indices[s1]["ndvi"]
        diff_ndre = indices[s2]["ndre"] - indices[s1]["ndre"]
        features.extend([diff_ndvi[None], diff_ndre[None]])
        band_names.append(f"phen_diff_ndvi_{s1}_to_{s2}")
        band_names.append(f"phen_diff_ndre_{s1}_to_{s2}")
    log(f"  差值: {(len(seasons)-1)}×2={(len(seasons)-1)*2}")

    # 4. 盛夏纹理
    # 选最后季节(peak_summer或early_summer)的NDVI和NIR
    texture_season = seasons[-1]
    ps_bands = s2_data[texture_season]
    ps_ndvi = indices[texture_season]["ndvi"]
    ps_nir = ps_bands[S2_BANDS.index("B08")]

    for win in [3, 7]:
        tex_ndvi = compute_texture(ps_ndvi, win)
        tex_nir = compute_texture(ps_nir, win)
        features.extend([tex_ndvi[None], tex_nir[None]])
        band_names.append(f"tex_{texture_season}_ndvi_{win}x{win}")
        band_names.append(f"tex_{texture_season}_nir_{win}x{win}")
    log(f"  纹理: 4")

    # 5. SAR 后向散射 (VV/VH, 与遂宁侧 label 一致)
    if sar_scenes:
        for label in sorted(sar_scenes.keys()):
            vv_path, vh_path = sar_scenes[label]
            vv = _read_sar_band(vv_path, H, W)
            vh = _read_sar_band(vh_path, H, W)
            features.extend([vv[None], vh[None]])
            band_names.append(f"s1_{label}_vv")
            band_names.append(f"s1_{label}_vh")
        log(f"  SAR: {len(sar_scenes)}x2={len(sar_scenes)*2}")

    cube = np.concatenate(features, axis=0)
    log(f"  总特征: {cube.shape[0]} bands, {H}×{W}")

    np.save(cache_file, cube)
    with open(names_file, "w") as f:
        json.dump(band_names, f)
    log(f"  已缓存 -> {cache_file.name}")
    return cube, band_names


# ============================================================
# 3. DEM
# ============================================================
def load_dem_for_xuyong():
    """下载或加载叙永区域DEM (Copernicus 30m)"""
    cache = DATA_DIR / "dem_xuyong_slope.tif"
    elev_cache = DATA_DIR / "dem_xuyong_elevation.tif"

    if cache.exists() and elev_cache.exists():
        log("DEM: 已有缓存")
        with rasterio.open(cache) as f:
            slope = f.read(1).astype("float32")
        with rasterio.open(elev_cache) as f:
            elev = f.read(1).astype("float32")
        return elev, slope

    log("DEM: 从 Planetary Computer 获取 Copernicus DEM...")
    try:
        import planetary_computer
        import requests as req
    except ImportError:
        log("[WARN] planetary-computer 未安装")
        return None, None

    bbox_wgs = list(XUYONG_BOUNDS_WGS84)
    payload = {"collections": ["cop-dem-glo-30"], "bbox": bbox_wgs, "limit": 1}
    try:
        resp = req.post(
            "https://planetarycomputer.microsoft.com/api/stac/v1/search",
            json=payload, timeout=60,
        )
        resp.raise_for_status()
        features = resp.json().get("features", [])
    except Exception as e:
        log(f"[WARN] DEM 搜索失败: {e}")
        return None, None

    if not features:
        log("[WARN] 无DEM数据")
        return None, None

    import math
    from rasterio.warp import transform_bounds

    dst_crs = "EPSG:32648"
    bbox_utm = transform_bounds("EPSG:4326", dst_crs, *bbox_wgs)
    res = 10.0
    xmin = math.floor(bbox_utm[0] / res) * res
    ymax = math.ceil(bbox_utm[3] / res) * res
    xmax = math.ceil(bbox_utm[2] / res) * res
    ymin = math.floor(bbox_utm[1] / res) * res
    w = int((xmax - xmin) / res)
    h = int((ymax - ymin) / res)
    d_transform = rasterio.transform.from_origin(xmin, ymax, res, res)

    href = planetary_computer.sign_url(features[0]["assets"]["data"]["href"])
    with rasterio.Env(**{"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", "GDAL_HTTP_TIMEOUT": "300"}):
        dem = np.zeros((h, w), dtype="float32")
        with rasterio.open(href) as src:
            reproject(
                source=src.read(1),
                destination=dem,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=d_transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear,
            )

    gy, gx = np.gradient(dem, 10.0)
    slope_rad = np.arctan(np.sqrt(gx**2 + gy**2))
    slope_deg = np.degrees(slope_rad).astype("float32")

    for arr, path in [(dem, elev_cache), (slope_deg, cache)]:
        prof = {"driver": "GTiff", "height": h, "width": w, "count": 1,
                "dtype": "float32", "crs": dst_crs, "transform": d_transform,
                "compress": "deflate"}
        with rasterio.open(path, "w", **prof) as dst:
            dst.write(arr, 1)
    log(f"DEM: 已缓存")
    return dem, slope_deg


# ============================================================
# 4. 标签构建
# ============================================================
def build_training_labels(feature_shape, s2_transform, s2_crs):
    """
    从叙永SHP栅格化果树标签(正样本) + WorldCover构建负样本。
    feature_shape: (bands, H, W)
    """
    H, W = feature_shape[1], feature_shape[2]
    cache = DATA_DIR / "xuyong_labels.npz"

    if cache.exists():
        log("标签: 已有缓存")
        d = np.load(cache)
        return d["y"], d["mask"]

    log("构建训练标签...")

    # --- 正样本: 栅格化SHP ---
    gdf = gpd.read_file(SHP_PATH)
    # 确保全部是果树（我们已确认 ZWMC 只有李子和柑橘）
    # 转换到UTM 48N
    gdf_utm = gdf.to_crs(s2_crs)
    geoms = [(g, 1) for g in gdf_utm.geometry if g.is_valid and not g.is_empty]
    if not geoms:
        log("[ERROR] SHP无有效几何")
        return None, None

    positive = rasterize(
        geoms, out_shape=(H, W), transform=s2_transform,
        fill=0, dtype="uint8", all_touched=True,
    )
    # 膨胀+腐蚀去边界混合像元
    from scipy.ndimage import binary_erosion, binary_dilation
    pos_clean = binary_erosion(binary_dilation(positive > 0, iterations=2), iterations=2)
    pos_count = int(pos_clean.sum())
    log(f"  正样本(果树): {pos_count:,} pixels")

    # --- 负样本: WorldCover非树类别 ---
    wc_cache = DATA_DIR / "worldcover_xuyong_10m.tif"
    if not wc_cache.exists():
        log("  下载WorldCover for Xuyong...")
        import planetary_computer
        bbox_wgs = list(XUYONG_BOUNDS_WGS84)
        try:
            wc_url = (
                "https://esa-worldcover.s3.eu-central-1.amazonaws.com"
                "/v200/2021/map/ESA_WorldCover_10m_2021_v200_N27E105_Map.tif"
            )
            with rasterio.Env(**{"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", "GDAL_HTTP_TIMEOUT": "120"}):
                with rasterio.open(wc_url) as src:
                    wc_data = np.zeros((H, W), dtype="uint8")
                    reproject(
                        source=src.read(1),
                        destination=wc_data,
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=s2_transform,
                        dst_crs=s2_crs,
                        resampling=Resampling.nearest,
                    )
        except Exception as e:
            log(f"  [WARN] WorldCover下载失败: {e}")
            # 尝试 PC
            try:
                import requests as req
                payload = {"collections": ["esa-worldcover"], "bbox": bbox_wgs, "limit": 1}
                resp = req.post(
                    "https://planetarycomputer.microsoft.com/api/stac/v1/search",
                    json=payload, timeout=60,
                )
                resp.raise_for_status()
                features = resp.json().get("features", [])
                if features:
                    href = planetary_computer.sign_url(features[0]["assets"]["map"]["href"])
                    with rasterio.Env(**{"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", "GDAL_HTTP_TIMEOUT": "120"}):
                        with rasterio.open(href) as src:
                            wc_data = np.zeros((H, W), dtype="uint8")
                            reproject(
                                source=src.read(1),
                                destination=wc_data,
                                src_transform=src.transform,
                                src_crs=src.crs,
                                dst_transform=s2_transform,
                                dst_crs=s2_crs,
                                resampling=Resampling.nearest,
                            )
                else:
                    wc_data = None
            except Exception as e2:
                log(f"  [WARN] PC WorldCover也失败: {e2}")
                wc_data = None

        if wc_data is not None:
            prof = {"driver": "GTiff", "height": H, "width": W, "count": 1,
                    "dtype": "uint8", "crs": s2_crs, "transform": s2_transform,
                    "compress": "deflate"}
            with rasterio.open(wc_cache, "w", **prof) as dst:
                dst.write(wc_data, 1)
    else:
        with rasterio.open(wc_cache) as f:
            wc_data = f.read(1)

    # WorldCover类别: 10=Tree, 20=Shrub, 30=Grass, 40=Crop, 50=Built, 60=Bare,
    # 70=Snow, 80=Water, 90=Wetland
    neg_cats = [40, 50, 60, 70, 80, 90]  # 非树类别作负样本
    neg_mask = np.isin(wc_data, neg_cats)
    neg_count = int(neg_mask.sum())
    log(f"  负样本(WC非树): {neg_count:,} pixels")

    if pos_count < 100:
        log(f"[WARN] 正样本太少({pos_count}), 可能影响训练")

    # 合并标签 & 有效掩码
    y = np.full((H, W), -1, dtype="int8")
    y[pos_clean] = 1
    y[neg_mask & ~pos_clean] = 0
    valid = y >= 0

    np.savez_compressed(cache, y=y, mask=valid)
    log(f"  总标签: 正={pos_count:,}, 负={neg_count:,}, 有效={int(valid.sum()):,}")
    return y, valid


# ============================================================
# 5. 训练LightGBM
# ============================================================
def train_lgb(features, y, valid, s2_transform):
    """训练LightGBM, 空间分组验证"""
    H, W = features.shape[1], features.shape[2]
    F = features.shape[0]

    log("提取训练样本...")
    valid_idx = np.where(valid.ravel())[0]
    X = features.reshape(F, -1)[:, valid_idx].T  # [N, F]
    y_true = y.ravel()[valid_idx]

    # 空间分组 (800m块)
    block_size = 80  # 80px @ 10m = 800m

    rows, cols = np.unravel_index(valid_idx, (H, W))
    block_r = rows // block_size
    block_c = cols // block_size
    block_ids = block_r * (W // block_size + 1) + block_c
    unique_blocks = np.unique(block_ids)

    log(f"  样本: {len(y_true):,}, 正={int((y_true==1).sum()):,}, 负={int((y_true==0).sum()):,}")
    log(f"  空间块: {len(unique_blocks)}, 块大小={block_size}px (800m)")

    # 块分组4折
    n_splits = 4
    gkf = GroupKFold(n_splits=n_splits)
    all_f1s = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(valid_idx, y_true, block_ids)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y_true[train_idx], y_true[test_idx]

        X_tr = np.nan_to_num(X_tr, nan=0)
        X_te = np.nan_to_num(X_te, nan=0)

        lgb = LGBMClassifier(**LGB_PARAMS)
        lgb.fit(X_tr, y_tr)
        y_pred = lgb.predict(X_te)
        f1 = f1_score(y_te, y_pred, average="binary")
        all_f1s.append(f1)
        log(f"  Fold {fold+1}: F1={f1:.4f}, train_n={len(y_tr):,}, test_n={len(y_te):,}")

    log(f"  平均F1: {np.mean(all_f1s):.4f} ± {np.std(all_f1s):.4f}")

    # 全量训练
    log("全量训练...")
    X_all = np.nan_to_num(X, nan=0)
    lgb_final = LGBMClassifier(**LGB_PARAMS)
    lgb_final.fit(X_all, y_true)

    # 保存模型
    import joblib
    model_path = XY_MODEL_DIR / "lgb_xuyong_fruit.pkl"
    joblib.dump(lgb_final, model_path)
    log(f"模型已保存: {model_path}")

    # 特征重要性TOP10
    top_idx = np.argsort(lgb_final.feature_importances_)[-10:][::-1]
    log("特征重要性TOP10:")
    for i, idx in enumerate(top_idx):
        log(f"  {i+1}. band_{idx}: {lgb_final.feature_importances_[idx]:.4f}")

    return lgb_final


# ============================================================
# 6. 加载遂宁特征
# ============================================================
def load_or_build_suining_features(xy_seasons=None):
    """加载遂宁特征立方体。波段名直接从重建脚本保存的 band_names.json 读取。"""
    cache = DATA_DIR / "suining_features" / "feature_cube.npy"
    transform_cache = DATA_DIR / "suining_features" / "transform.json"
    names_cache = DATA_DIR / "suining_features" / "band_names.json"

    if not cache.exists():
        log("[ERROR] 遂宁特征缓存不存在")
        return None, None, None

    log("遂宁特征: 已有缓存")
    cube = np.load(cache)

    with open(transform_cache) as f:
        tinfo = json.load(f)
    transform = rasterio.transform.Affine(*tinfo["affine"])

    # 直接读取重建脚本保存的波段名（避免季节名推断错误）
    if names_cache.exists():
        with open(names_cache) as f:
            band_names = json.load(f)
    else:
        # 兜底: 从 cube 形状推断季节数
        F = cube.shape[0]
        Ns = (F - 4) // 16
        all_seasons = ["late_autumn", "spring", "summer", "winter"]
        seasons = sorted(xy_seasons) if xy_seasons else all_seasons[:Ns]
        bnames = []
        for s in seasons:
            for b in S2_BANDS:
                bnames.append(f"s2_{s}_{b}")
        for s in seasons:
            for vi in ["ndvi", "ndre", "ndmi", "evi"]:
                bnames.append(f"vi_{s}_{vi}")
        for i in range(len(seasons) - 1):
            s1, s2 = seasons[i], seasons[i + 1]
            bnames.append(f"phen_diff_ndvi_{s1}_to_{s2}")
            bnames.append(f"phen_diff_ndre_{s1}_to_{s2}")
        tex_s = seasons[-1]
        for win in [3, 7]:
            bnames.append(f"tex_{tex_s}_ndvi_{win}x{win}")
            bnames.append(f"tex_{tex_s}_nir_{win}x{win}")
        bnames.extend(["dem_elevation", "dem_slope"])
        band_names = bnames[:F]

    log(f"  波段名: {len(band_names)} 波段 - 直接读取 band_names.json")
    log(f"  特征形状: {cube.shape}")
    return cube, transform, band_names


# ============================================================
# 7. 遂宁推理
# ============================================================
def predict_suining(lgb_model, features, transform, sn_band_names, train_band_names):
    """对遂宁区进行全图推理，自动对齐波段"""
    F, H, W = features.shape

    # 匹配波段
    sn_name_to_idx = {n: i for i, n in enumerate(sn_band_names)}
    feature_indices = [sn_name_to_idx[bn] for bn in train_band_names if bn in sn_name_to_idx]
    missing = [bn for bn in train_band_names if bn not in sn_name_to_idx]
    if missing:
        log(f"  [WARN] {len(missing)} 个训练波段在遂宁中缺失 (共{len(train_band_names)}训练波段)")

    log(f"遂宁推理: {H}×{W} px, 选择 {len(feature_indices)}/{F} bands")

    # 分块推理以避免内存爆炸
    chunk_h = 500
    prob = np.zeros((H, W), dtype="float32")
    pred = np.zeros((H, W), dtype="uint8")

    for row_start in range(0, H, chunk_h):
        row_end = min(row_start + chunk_h, H)
        chunk = features[feature_indices, row_start:row_end, :].reshape(len(feature_indices), -1).T
        chunk = np.nan_to_num(chunk, nan=0)
        chunk_prob = lgb_model.predict_proba(chunk)[:, 1]
        chunk_pred = lgb_model.predict(chunk).astype("uint8")
        col_sz = W
        prob[row_start:row_end, :] = chunk_prob.reshape(row_end - row_start, col_sz)
        pred[row_start:row_end, :] = chunk_pred.reshape(row_end - row_start, col_sz)
        if row_start % 1000 == 0:
            log(f"  推理进度: {row_start}/{H}")

    # 写出栅格
    # 二值分类
    bin_path = OUTPUT_DIR / "orchard_xuyong_model_suining.tif"
    prob_path = OUTPUT_DIR / "orchard_xuyong_model_suining_prob.tif"

    prof = {"driver": "GTiff", "height": H, "width": W, "count": 1,
            "dtype": "uint8", "crs": "EPSG:32648", "transform": transform,
            "compress": "deflate", "tiled": True}
    with rasterio.open(bin_path, "w", **prof) as dst:
        dst.write(pred, 1)
        dst.set_band_description(1, "orchard_binary_1=orchard")

    prof["dtype"] = "float32"
    with rasterio.open(prob_path, "w", **prof) as dst:
        dst.write(prob, 1)
        dst.set_band_description(1, "orchard_probability")

    # 统计
    total_px = int(np.sum(pred > 0))
    total_ha = total_px * 100 / 10000  # 10m像素=100m²=0.01ha
    log(f"推理完成: 果树={total_px:,} px = {total_ha:.0f} ha")

    # 矢量斑块
    try:
        shp_dir = OUTPUT_DIR / "shapefile_xuyong_model"
        shp_dir.mkdir(exist_ok=True)
        from rasterio.features import shapes
        mask = pred > 0
        results = list(shapes(mask.astype("int16"), transform=transform))
        geoms = [{"properties": {"area_ha": v * 100 / 10000}, "geometry": g}
                 for g, v in results if v > 0]
        if geoms:
            gdf_out = gpd.GeoDataFrame.from_features(geoms, crs="EPSG:32648")
            gdf_out.to_file(shp_dir / "orchard_patches.shp")
            log(f"矢量: {len(gdf_out)} 个斑块 -> {shp_dir}")
    except Exception as e:
        log(f"[WARN] 矢量化失败: {e}")

    # 评估报告
    report = {
        "model": "LightGBM (Xuyong-trained, 4-season)",
        "training_samples": {
            "positive": "叙永SHP果树地块(李子+柑橘)",
            "negative": "WorldCover非树类别",
        },
        "suining_results": {
            "total_orchard_pixels": int(total_px),
            "total_orchard_ha": round(total_ha, 1),
            "outputs": {
                "binary": str(bin_path),
                "probability": str(prob_path),
            },
        },
    }
    report_path = OUTPUT_DIR / "transfer_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    log(f"报告: {report_path}")

    return pred, prob


# ============================================================
# 8. 主流程
# ============================================================
def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def main():
    log("=" * 60)
    log("果树识别 转移学习: 叙永训练 → 遂宁推理")
    log("=" * 60)

    # ---- 清理旧缓存(无band_names的旧格式) ----
    stale_cache = XY_FEAT_DIR / "features_xuyong.npy"
    stale_names = XY_FEAT_DIR / "band_names_xuyong.json"
    if stale_cache.exists() and not stale_names.exists():
        log("清理旧版Xuyong特征缓存(无band_names)...")
        stale_cache.unlink()

    # ---- Step 1: 下载叙永S2 ----
    log("\nStep 1: 获取叙永 Sentinel-2")
    xy_s2 = download_xuyong_s2()
    if xy_s2 is None or len(xy_s2) < 2:
        log(f"[FATAL] 叙永S2仅获取 {len(xy_s2) if xy_s2 else 0} 季，至少需要2季")
        return

    # ---- Step 2: 构建叙永特征 ----
    log("\nStep 2: 构建叙永特征立方体")
    xy_features, xy_band_names = build_feature_cube(xy_s2, "xuyong", SAR_SCENES_XUYONG)
    log(f"  特征形状: {xy_features.shape}, {len(xy_band_names)} 波段")

    # ---- Step 3: DEM (尝试, 但不强求) ----
    log("\nStep 3: 获取DEM")
    elev, slope = load_dem_for_xuyong()
    if elev is not None and slope is not None:
        H, W = xy_features.shape[1], xy_features.shape[2]
        elev = elev[:H, :W]
        slope = slope[:H, :W]
        xy_features = np.concatenate([xy_features, elev[None], slope[None]], axis=0)
        xy_band_names.extend(["dem_elevation", "dem_slope"])
        log(f"  +DEM: 特征形状={xy_features.shape}")
    else:
        log("  DEM不可用，跳过")

    # ---- Step 4: 构建标签 ----
    log("\nStep 4: 构建训练标签")
    with rasterio.open(list(xy_s2.values())[0]) as src:
        xy_transform = src.transform
        xy_crs = src.crs

    y_labels, valid_mask = build_training_labels(xy_features.shape, xy_transform, xy_crs)
    if y_labels is None:
        log("[FATAL] 标签构建失败")
        return

    # ---- Step 5: 训练LightGBM ----
    log("\nStep 5: 训练LightGBM模型")
    lgb_model = train_lgb(xy_features, y_labels, valid_mask, xy_transform)

    # ---- Step 6: 加载遂宁特征 ----
    log("\nStep 6: 加载遂宁特征")
    sn_features, sn_transform, sn_band_names = load_or_build_suining_features()
    if sn_features is None:
        log("[ERROR] 遂宁特征不可用，跳过推理")
        return

    # ---- Step 7: 遂宁推理 ----
    log("\nStep 7: 遂宁推理")
    pred, prob = predict_suining(lgb_model, sn_features, sn_transform, sn_band_names, xy_band_names)

    log("\n" + "=" * 60)
    log("全部完成！")
    log("=" * 60)


if __name__ == "__main__":
    main()
