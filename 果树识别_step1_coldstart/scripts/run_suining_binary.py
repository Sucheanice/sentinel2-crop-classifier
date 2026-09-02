# -*- coding: utf-8 -*-
"""
遂宁区域果树识别 - Step 1: AOMC冷启动 (本地数据版)
===================================================
数据源:
  - Sentinel-2: 本地 小春_s2_48RWU + 小春_s2_48RWU_extra (10m, UTM 48N)
  - WorldCover: 本地 ESA_WorldCover_10m_2021_v200_N30E105_Map.tif
  - AOMC 苹果园标签: figshare 下载 (https://doi.org/10.6084/m9.figshare.28113302)
  - Copernicus DEM: Planetary Computer 下载

不依赖项目已有代码，独立从头编写。
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import shapes as raster_shapes
from rasterio.warp import reproject, Resampling as WarpResampling
from rasterio.windows import from_bounds as window_from_bounds
from scipy.ndimage import binary_erosion, uniform_filter
from shapely.geometry import shape as shapely_shape
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# 0. 路径与基础配置
# ============================================================
ROOT = Path(__file__).resolve().parents[1]                        # 果树识别_step1_coldstart/
WORK_DIR = ROOT.parent                                             # 0624 待测试数据/
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
for d in [DATA_DIR, OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---- 本地数据路径 ----
S2_MAIN_DIR = WORK_DIR / "小春_s2_48RWU"                           # 主时相目录
S2_EXTRA_DIR = WORK_DIR / "小春_s2_48RWU_extra"                   # 补充时相目录
WC_LOCAL_PATH = WORK_DIR / "ESA_WorldCover_10m_2021_v200_N30E105_Map.tif"
AOMC_SICHUAN = WORK_DIR / "待测试数据AOMC" / "2021" / "AOMC_30_Sichuan_2021.tif"  # AOMC四川苹果园

# ---- Sentinel-2 时相映射（用已有数据，不做下载）----
# 各文件夹下为按波段分的 .tif 文件 (B02.tif ~ B12.tif)
S2_BANDS_10 = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
SEASON_DIRS = {
    "winter":       S2_MAIN_DIR / "2025-01-20_48RWU_cloud1.2",     # 云量 1.2%
    "spring":       S2_MAIN_DIR / "2025-03-26_48RWU_cloud0.0",     # 云量 0.0%
    "early_summer": S2_EXTRA_DIR / "2025-04-17_48RWU",             # 云量低
    "late_autumn":  S2_MAIN_DIR / "2024-12-11_48RWU_cloud9.5",     # 云量 9.5%
}

# ---- 遂宁市目标区（WGS84）----
SUINING_BBOX_WGS84 = [105.00, 30.15, 106.05, 31.10]   # [min_lon, min_lat, max_lon, max_lat]

# ---- 数据参考 ----
# 本地 S2 为 UTM 48N (EPSG:32648)，10m波段 10980×10980，20m波段 5490×5490
# 遂宁全境在该景内
TARGET_CRS = "EPSG:32648"
RESOLUTION = 10  # 米
S2_10M_BANDS = {"B02", "B03", "B04", "B08"}
S2_20M_BANDS = {"B05", "B06", "B07", "B8A", "B11", "B12"}

# ---- 随机森林参数 ----
RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = 18
RF_MIN_SAMPLES_LEAF = 3
RANDOM_STATE = 42

# ---- 空间验证 ----
BLOCK_SIZE_M = 800
TEST_SIZE = 0.25
MAX_SAMPLES_PER_CLASS = 5000

# ---- 日志 ----
def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================
# 1. 本地数据加载（替代网络下载）
# ============================================================

def get_local_s2_bounds() -> tuple:
    """获取本地 S2 影像的投影范围和栅格参数"""
    first_season = next(iter(SEASON_DIRS.values()))
    sample = next(first_season.glob("*.tif"))
    with rasterio.open(sample) as src:
        return src.crs, src.transform, src.width, src.height


def compute_suining_windows() -> dict:
    """
    计算遂宁 bbox 在本地 S2 10m和20m波段中的窗口。
    返回: {"10m": (row_start, row_end, col_start, col_end),
           "20m": (row_start, row_end, col_start, col_end)}
    """
    s2_crs, s2_transform, _, _ = get_local_s2_bounds()

    from pyproj import Transformer
    transformer = Transformer.from_crs("EPSG:4326", s2_crs, always_xy=True)
    xmin, ymin = transformer.transform(SUINING_BBOX_WGS84[0], SUINING_BBOX_WGS84[1])
    xmax, ymax = transformer.transform(SUINING_BBOX_WGS84[2], SUINING_BBOX_WGS84[3])

    windows = {}
    for res, label in [(10.0, "10m"), (20.0, "20m")]:
        col_start = max(0, int((xmin - s2_transform.c) / res))
        col_end = min(int(10980 * 10 / res), int(np.ceil((xmax - s2_transform.c) / res)))
        row_start = max(0, int((s2_transform.f - ymax) / res))
        row_end = min(int(10980 * 10 / res), int(np.ceil((s2_transform.f - ymin) / res)))
        windows[label] = (row_start, row_end, col_start, col_end)

    return windows


def load_s2_season(season: str) -> Optional[tuple]:
    """
    加载本地单季 Sentinel-2 数据，所有波段统一重采样到10m并裁剪到遂宁区。
    返回: (stack: (bands, H_10m, W_10m), band_names: list)
    若波段不足则返回 None。
    """
    season_dir = SEASON_DIRS[season]
    if not season_dir.exists():
        log(f"  [WARN] {season}: 目录不存在 -> {season_dir}")
        return None

    windows = compute_suining_windows()
    w10 = windows["10m"]
    w20 = windows["20m"]

    # 10m 目标尺寸
    h10 = w10[1] - w10[0]
    w10_width = w10[3] - w10[2]
    log(f"  {season}: 裁剪窗口 10m=({h10}x{w10_width}), 20m=({w20[1]-w20[0]}x{w20[3]-w20[2]})")

    band_arrays = []
    loaded_bands = []

    for band_name in S2_BANDS_10:
        band_file = season_dir / f"{band_name}.tif"
        if not band_file.exists():
            continue

        with rasterio.open(band_file) as src:
            is_20m = band_name in S2_20M_BANDS
            if is_20m:
                # 20m波段：用20m窗口读，out_shape 自动重采样到10m
                win = rasterio.windows.Window(w20[2], w20[0],
                                              w20[3] - w20[2], w20[1] - w20[0])
                data = src.read(1, window=win, out_shape=(h10, w10_width)).astype("float32")
            else:
                # 10m波段：直接用10m窗口
                win = rasterio.windows.Window(w10[2], w10[0],
                                              w10_width, h10)
                data = src.read(1, window=win).astype("float32")
            data[data <= 0] = np.nan
            band_arrays.append(data)
            loaded_bands.append(band_name)

    if len(band_arrays) < 4:
        log(f"  [WARN] {season}: 波段不足 ({len(band_arrays)} 个)")
        return None

    cube = np.stack(band_arrays)
    log(f"  {season}: {cube.shape[1]}x{cube.shape[2]} @ 10m, {len(loaded_bands)} 波段 ({', '.join(loaded_bands)})")
    return cube, loaded_bands


def load_worldcover(width: int, height: int, crs: str, transform) -> np.ndarray:
    """
    加载本地 WorldCover，重投影到目标栅格。
    返回: (height, width) uint8 数组
    """
    if not WC_LOCAL_PATH.exists():
        log(f"  [ERROR] WorldCover 不存在: {WC_LOCAL_PATH}")
        sys.exit(1)

    log("  重投影 WorldCover 到目标范围...")
    wc_out = np.zeros((height, width), dtype="uint8")

    with rasterio.open(WC_LOCAL_PATH) as src:
        reproject(
            source=src.read(1),
            destination=wc_out,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=crs,
            resampling=WarpResampling.nearest,
        )

    log(f"  WorldCover: {width}x{height}, 各类别像元数:")
    for cls_val in [10, 20, 30, 40, 50, 60, 80, 90, 95, 100]:
        cnt = int(np.sum(wc_out == cls_val))
        cls_name = {
            10: "Tree cover", 20: "Shrubland", 30: "Grassland",
            40: "Cropland", 50: "Built-up", 60: "Bare",
            80: "Water", 90: "Wetland", 95: "Mangrove", 100: "Moss",
        }.get(cls_val, "Unknown")
        if cnt > 0:
            log(f"    {cls_val:3d} {cls_name:15s}: {cnt:>10,}")

    return wc_out


# ============================================================
# 2. AOMC (本地) + DEM (网络)
# ============================================================

def load_aomc_label(target_crs: str, target_transform, width: int, height: int) -> Optional[np.ndarray]:
    """
    加载本地 AOMC 苹果园标签（30m, Albers等积投影），重投影/重采样到遂宁区10m。
    来源: AOMC Sichuan 2021 (本地下载)
    """
    aomc_cache = DATA_DIR / "aomc_sichuan_10m.tif"
    if aomc_cache.exists():
        log("  AOMC: 已有本地缓存，读取中...")
        with rasterio.open(aomc_cache) as src:
            return src.read(1).astype("uint8")

    if not AOMC_SICHUAN.exists():
        log(f"  [WARN] AOMC 本地文件不存在: {AOMC_SICHUAN}")
        log("  [INFO] 请将 AOMC Sichuan TIF 放入待测试数据AOMC/2021/ 目录")
        return None

    log(f"  AOMC: 读取本地 {AOMC_SICHUAN.name}，重投影到 UTM 48N 10m...")
    aomc_out = np.zeros((height, width), dtype="uint8")

    with rasterio.open(AOMC_SICHUAN) as src:
        # Albers Equal Area → UTM 48N, 30m → 10m (nearest)
        reproject(
            source=src.read(1),
            destination=aomc_out,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=target_transform,
            dst_crs=target_crs,
            resampling=WarpResampling.nearest,
        )

    apple_count = int(np.sum(aomc_out > 0))
    log(f"  AOMC: 遂宁区苹果园像元={apple_count:,} (10m)")

    # 缓存
    profile = {
        "driver": "GTiff", "height": height, "width": width, "count": 1,
        "dtype": "uint8", "crs": target_crs, "transform": target_transform,
        "compress": "deflate", "tiled": True, "nodata": 0,
    }
    with rasterio.open(aomc_cache, "w", **profile) as dst:
        dst.write(aomc_out, 1)
        dst.set_band_description(1, "AOMC_Sichuan_2021_apple_orchard")
    log(f"  AOMC: 已缓存 -> {aomc_cache.name}")

    return aomc_out


def download_dem(target_crs: str, target_transform, width: int, height: int) -> Optional[np.ndarray]:
    """
    从 Planetary Computer 下载 Copernicus DEM 30m，重采样到目标栅格。
    返回: (2, height, width) — band0=elevation, band1=slope(deg)
    """
    dem_cache = DATA_DIR / "dem_suining_slope.tif"
    if dem_cache.exists():
        log("  DEM: 已有本地缓存，读取中...")
        with rasterio.open(dem_cache) as src:
            return src.read().astype("float32")

    log("  DEM: 从 Planetary Computer 获取 Copernicus DEM...")
    try:
        import planetary_computer
        import requests as req
    except ImportError:
        log("  [WARN] planetary-computer 未安装，无法下载 DEM")
        return None

    payload = {
        "collections": ["cop-dem-glo-30"],
        "bbox": SUINING_BBOX_WGS84,
        "limit": 1,
    }
    try:
        resp = req.post(
            "https://planetarycomputer.microsoft.com/api/stac/v1/search",
            json=payload, timeout=60,
        )
        resp.raise_for_status()
        features = resp.json().get("features", [])
    except Exception as e:
        log(f"  [WARN] DEM 搜索失败: {e}")
        return None

    if not features:
        log("  [WARN] 无 DEM 数据")
        return None

    item = features[0]
    href = planetary_computer.sign_url(item["assets"]["data"]["href"])

    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_TIMEOUT="300"):
        dem_30m = np.zeros((height, width), dtype="float32")
        with rasterio.open(href) as src:
            reproject(
                source=src.read(1),
                destination=dem_30m,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=target_transform,
                dst_crs=target_crs,
                resampling=WarpResampling.bilinear,
            )

    # 计算坡度
    dem = dem_30m
    dy, dx = np.gradient(dem, RESOLUTION)
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2) / RESOLUTION)
    slope_deg = np.degrees(slope_rad)

    # 保存
    dem_stack = np.stack([dem, slope_deg]).astype("float32")
    profile = {
        "driver": "GTiff", "height": height, "width": width, "count": 2,
        "dtype": "float32", "crs": target_crs, "transform": target_transform,
        "compress": "deflate", "tiled": True, "nodata": np.nan,
    }
    with rasterio.open(dem_cache, "w", **profile) as dst:
        dst.write(dem_stack)
        dst.set_band_description(1, "elevation_m")
        dst.set_band_description(2, "slope_deg")

    log(f"  DEM: 已缓存 ({width}x{height}, 2波段)")
    return dem_stack


# ============================================================
# 3. 特征构建
# ============================================================

def compute_indices(stack: np.ndarray, band_names: list[str]) -> dict[str, np.ndarray]:
    """计算植被指数"""
    indices = {}
    band_map = {name: stack[i] for i, name in enumerate(band_names)}
    eps = 1e-6

    if "B08" in band_map and "B04" in band_map:
        nir, red = band_map["B08"], band_map["B04"]
        indices["ndvi"] = (nir - red) / (nir + red + eps)

    if "B8A" in band_map and "B05" in band_map:
        indices["ndre"] = (band_map["B8A"] - band_map["B05"]) / (band_map["B8A"] + band_map["B05"] + eps)

    for swir in ["B11", "B12"]:
        if "B08" in band_map and swir in band_map:
            indices["ndmi"] = (band_map["B08"] - band_map[swir]) / (band_map["B08"] + band_map[swir] + eps)
            break

    if all(b in band_map for b in ["B08", "B04", "B02"]):
        nir, red, blue = band_map["B08"], band_map["B04"], band_map["B02"]
        indices["evi"] = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1 + eps)

    return indices


def local_std(img: np.ndarray, window: int) -> np.ndarray:
    """局部标准差纹理"""
    mean = uniform_filter(img, size=window)
    sq_mean = uniform_filter(img**2, size=window)
    var = np.maximum(sq_mean - mean**2, 0)
    return np.sqrt(var)


def build_feature_cube(
    s2_data: dict[str, tuple[np.ndarray, list[str]]],
    dem_arr: Optional[np.ndarray],
) -> tuple[np.ndarray, list[str]]:
    """
    构建全特征立方体。
    s2_data: {season: (cube_bands_h_w, band_names)}
    返回: (features: n_bands x H x W, feature_names)
    """
    feature_list = []
    feature_names = []

    # ---- 多时相光谱 + 指数 ----
    for season, (stack, band_names) in s2_data.items():
        for i, bn in enumerate(band_names):
            feature_list.append(stack[i])
            feature_names.append(f"s2_{season}_{bn.lower()}")
        indices = compute_indices(stack, band_names)
        for idx_name, idx_arr in indices.items():
            feature_list.append(idx_arr)
            feature_names.append(f"s2_{season}_{idx_name}")

    # ---- 跨季节物候差值 ----
    seasons_list = list(s2_data.keys())
    for i in range(len(seasons_list) - 1):
        for j in range(i + 1, len(seasons_list)):
            ndvi_k1 = f"s2_{seasons_list[i]}_ndvi"
            ndvi_k2 = f"s2_{seasons_list[j]}_ndvi"
            if ndvi_k1 in feature_names and ndvi_k2 in feature_names:
                idx1 = feature_names.index(ndvi_k1)
                idx2 = feature_names.index(ndvi_k2)
                feature_list.append(feature_list[idx2] - feature_list[idx1])
                feature_names.append(f"phen_ndvi_diff_{seasons_list[i]}_to_{seasons_list[j]}")

    # ---- 纹理特征（盛夏/early_summer 的 NIR + NDVI）----
    if "early_summer" in s2_data:
        es_stack, es_bands = s2_data["early_summer"]
        es_indices = compute_indices(es_stack, es_bands)
        texture_sources = {}
        if "B08" in es_bands:
            nir_idx = es_bands.index("B08")
            texture_sources["nir"] = es_stack[nir_idx]
        if "ndvi" in es_indices:
            texture_sources["ndvi"] = es_indices["ndvi"]
        for base_name, arr in texture_sources.items():
            valid = np.isfinite(arr)
            arr_filled = np.nan_to_num(arr, nan=0)
            for win in [3, 7]:
                feat = local_std(arr_filled, win)
                feat[~valid] = np.nan
                feature_list.append(feat)
                feature_names.append(f"texture_{base_name}_std_{win}x{win}")

    # ---- 地形特征 ----
    if dem_arr is not None:
        terrain_labels = ["elevation", "slope"]
        for i in range(min(dem_arr.shape[0], len(terrain_labels))):
            feature_list.append(dem_arr[i])
            feature_names.append(f"terrain_{terrain_labels[i]}")

    cube = np.stack(feature_list).astype("float32")
    log(f"  特征立方体: {cube.shape[0]} 波段, {cube.shape[1]}x{cube.shape[2]} 像元")
    return cube, feature_names


# ============================================================
# 4. 标签构建
# ============================================================

def build_binary_labels(
    cube: np.ndarray,
    aomc: Optional[np.ndarray],
    wc: np.ndarray,
) -> np.ndarray:
    """
    构建二分类标签: 1=果树, 0=非果树, 255=无效。
    """
    _, height, width = cube.shape
    labels = np.zeros((height, width), dtype="uint8")
    finite_mask = np.all(np.isfinite(cube), axis=0)

    # --- 正样本：AOMC苹果园 ---
    has_aomc = aomc is not None and np.sum(aomc > 0) > 0
    if has_aomc:
        labels[aomc > 0] = 1
        log(f"  AOMC正样本(果树): {np.sum(labels==1):,} 像元")
    else:
        # 回退：WorldCover Tree cover (10)
        labels[wc == 10] = 1
        log(f"  [回退] WorldCover Tree cover 作为果树代理: {np.sum(labels==1):,} 像元")

    # --- 负样本 ----
    negative_classes = [20, 30, 40, 50, 60, 80, 90, 95, 100]
    if has_aomc:
        # AOMC已标出苹果园，剩余Tree cover大概率是天然林 → 有效负样本
        negative_classes.append(10)

    for nc in negative_classes:
        labels[(wc == nc) & (labels == 0)] = 0

    # 腐蚀边界
    clean = np.zeros_like(labels)
    for class_val in [0, 1]:
        class_mask = labels == class_val
        if np.sum(class_mask) == 0:
            continue
        eroded = binary_erosion(class_mask, structure=np.ones((3, 3)), border_value=0)
        clean[eroded & finite_mask] = class_val

    # 无效区域
    valid_negative = np.zeros_like(labels, dtype=bool)
    for nc in negative_classes:
        valid_negative |= (wc == nc)
    clean[(labels == 0) & ~valid_negative & (labels != 1)] = 255

    log(f"  清洗后: 果树={np.sum(clean==1):,}, 非果树={np.sum(clean==0):,}, 无效={np.sum(clean==255):,}")

    return clean


# ============================================================
# 5. 模型训练
# ============================================================

def train_binary_rf(
    cube: np.ndarray,
    labels: np.ndarray,
    feature_names: list[str],
    transform,
) -> dict:
    """训练二分类随机森林，空间分块验证，全图推理。"""
    height, width = labels.shape
    n_features = cube.shape[0]

    # ---- 采样 ----
    rng = np.random.default_rng(RANDOM_STATE)
    sampled = []
    for class_val in [0, 1]:
        idx = np.flatnonzero((labels == class_val).ravel())
        if len(idx) > MAX_SAMPLES_PER_CLASS:
            idx = rng.choice(idx, MAX_SAMPLES_PER_CLASS, replace=False)
        sampled.append(idx)
    all_idx = np.concatenate(sampled)
    rng.shuffle(all_idx)

    rows, cols = np.unravel_index(all_idx, (height, width))
    y = labels.ravel()[all_idx]
    X = cube[:, rows, cols].T

    # NaN 中位数填充
    median_vals = np.nanmedian(X, axis=0)
    for j in range(X.shape[1]):
        col = X[:, j]
        col[np.isnan(col)] = median_vals[j]

    # 空间分组
    block_pixels = int(BLOCK_SIZE_M / RESOLUTION)
    groups = (rows // block_pixels) * 10000 + cols // block_pixels

    # 空间留出
    splitter = GroupShuffleSplit(n_splits=200, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    split = None
    for train_idx, test_idx in splitter.split(X, y, groups):
        if set(np.unique(y[train_idx])) == {0, 1} and set(np.unique(y[test_idx])) == {0, 1}:
            counts_train = np.bincount(y[train_idx], minlength=2)
            counts_test = np.bincount(y[test_idx], minlength=2)
            if np.min(counts_train) >= 50 and np.min(counts_test) >= 20:
                split = (train_idx, test_idx)
                break
    if split is None:
        raise RuntimeError("无法构建有效的空间分割，请检查标签覆盖范围")

    train_idx, test_idx = split
    log(f"  训练: {len(train_idx):,} 样本 (果树 {np.sum(y[train_idx]==1):,}, 非果树 {np.sum(y[train_idx]==0):,})")
    log(f"  测试: {len(test_idx):,} 样本 (果树 {np.sum(y[test_idx]==1):,}, 非果树 {np.sum(y[test_idx]==0):,})")

    # ---- 训练 ----
    log("  训练随机森林...")
    rf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X[train_idx], y[train_idx])

    # ---- 评估 ----
    y_test = y[test_idx]
    y_pred = rf.predict(X[test_idx])
    metrics = {
        "accuracy": float(np.mean(y_test == y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "n_train_pos": int(np.sum(y[train_idx] == 1)),
        "n_train_neg": int(np.sum(y[train_idx] == 0)),
        "n_test_pos": int(np.sum(y_test == 1)),
        "n_test_neg": int(np.sum(y_test == 0)),
    }

    log(f"  测试: Acc={metrics['accuracy']:.4f} P={metrics['precision']:.4f} "
        f"R={metrics['recall']:.4f} F1={metrics['f1']:.4f}")
    cm = metrics["confusion_matrix"]
    log(f"  混淆: TN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]} TP={cm[1][1]}")

    # ---- 全图推理 ----
    log("  全图推理...")
    flat = cube.reshape(n_features, -1).T
    valid_mask = np.all(np.isfinite(flat), axis=1)
    valid_pos = np.flatnonzero(valid_mask)

    for j in range(flat.shape[1]):
        col = flat[:, j]
        col[np.isnan(col)] = median_vals[j]

    proba = np.full(flat.shape[0], np.nan, dtype="float32")
    batch_size = 100000
    for start in range(0, len(valid_pos), batch_size):
        end = min(start + batch_size, len(valid_pos))
        pos = valid_pos[start:end]
        proba[pos] = rf.predict_proba(flat[pos])[:, 1]
    proba = proba.reshape(height, width)

    pred = np.zeros((height, width), dtype="uint8")
    v2d = valid_mask.reshape(height, width)
    pred[v2d & (proba >= 0.5)] = 1
    pred[v2d & (proba < 0.5)] = 0

    # 特征重要性
    topf = sorted(
        [(feature_names[i], float(rf.feature_importances_[i])) for i in range(len(feature_names))],
        key=lambda x: -x[1],
    )

    return {
        "model": rf,
        "metrics": metrics,
        "probability_map": proba,
        "prediction_map": pred,
        "top_features": topf[:20],
        "all_features": feature_names,
        "median_imputation": median_vals.tolist(),
    }


# ============================================================
# 6. 成果输出
# ============================================================

def save_outputs(result: dict, transform, crs: str) -> dict:
    """保存分类结果并返回面积统计"""
    pred = result["prediction_map"]
    proba = result["probability_map"]
    height, width = pred.shape

    base_profile = {
        "driver": "GTiff", "height": height, "width": width,
        "crs": crs, "transform": transform,
        "compress": "deflate", "tiled": True,
    }

    # 分类栅格
    out_cls = np.where(np.isfinite(proba), pred, 255).astype("uint8")
    profile_cls = {**base_profile, "dtype": "uint8", "count": 1, "nodata": 255}
    with rasterio.open(OUTPUT_DIR / "orchard_binary_10m.tif", "w", **profile_cls) as dst:
        dst.write(out_cls, 1)
        dst.set_band_description(1, "0=non_orchard 1=orchard 255=nodata")
    log(f"  分类栅格: orchard_binary_10m.tif")

    # 概率栅格
    out_prob = np.where(np.isfinite(proba), proba, -9999).astype("float32")
    with rasterio.open(OUTPUT_DIR / "orchard_probability_10m.tif", "w",
                       **{**base_profile, "dtype": "float32", "count": 1, "nodata": -9999.0}) as dst:
        dst.write(out_prob, 1)
        dst.set_band_description(1, "orchard_probability")
    log(f"  概率栅格: orchard_probability_10m.tif")

    # 面积
    pixel_area_ha = RESOLUTION**2 / 10000
    valid = np.isfinite(proba)
    pv = pred[valid]
    area_stats = {
        "total_valid_ha": float(np.sum(valid) * pixel_area_ha),
        "orchard_ha": float(np.sum(pv == 1) * pixel_area_ha),
        "non_orchard_ha": float(np.sum(pv == 0) * pixel_area_ha),
        "orchard_px": int(np.sum(pv == 1)),
        "non_orchard_px": int(np.sum(pv == 0)),
    }
    log(f"  面积: 果树={area_stats['orchard_ha']:.0f}ha, 非果树={area_stats['non_orchard_ha']:.0f}ha")

    # SHP 矢量
    try:
        import geopandas as gpd
        polygons = []
        om = (pred == 1).astype("uint8")
        for geom, value in raster_shapes(om, mask=om > 0, transform=transform):
            if value != 1:
                continue
            poly = shapely_shape(geom)
            if poly.is_empty or poly.area < 1000:
                continue
            polygons.append({
                "class": "orchard",
                "area_ha": poly.area / 10000,
                "geometry": poly.simplify(5),
            })
        if polygons:
            gdf = gpd.GeoDataFrame(polygons, crs=crs)
            shp_dir = OUTPUT_DIR / "shapefile"
            shp_dir.mkdir(exist_ok=True)
            gdf.to_file(shp_dir / "orchard_candidates.shp", driver="ESRI Shapefile", encoding="UTF-8")
            log(f"  矢量: {len(gdf):,} 个果树斑块")
    except ImportError:
        log("  [WARN] geopandas 未安装，跳过矢量")

    # 报告
    report = {
        "pipeline": "遂宁果树识别 Step1 (本地S2版)",
        "target": "遂宁市",
        "crs": str(crs), "resolution_m": RESOLUTION,
        "seasons": list(SEASON_DIRS.keys()),
        "positive_source": "AOMC Sichuan 2021 (本地)" if (DATA_DIR / "aomc_sichuan_10m.tif").exists() else "WorldCover fallback",
        "model": "RandomForest(binary)",
        "model_params": {"n_estimators": RF_N_ESTIMATORS, "max_depth": RF_MAX_DEPTH},
        "validation": {"method": f"{BLOCK_SIZE_M}m block {TEST_SIZE:.0%} holdout", **result["metrics"]},
        "area": area_stats,
        "top20_features": result["top_features"],
    }
    (OUTPUT_DIR / "evaluation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"  报告: evaluation_report.json")

    log("  特征重要性 Top 10:")
    for i, (n, imp) in enumerate(result["top_features"][:10], 1):
        log(f"    {i:2d}. {n}: {imp:.4f}")

    return area_stats


# ============================================================
# 7. 主流程
# ============================================================

def main():
    log("=" * 60)
    log("遂宁区域果树识别 Step1: AOMC冷启动 (本地数据版)")
    log("=" * 60)

    # ---- Step 1: 加载本地 Sentinel-2 ----
    log("Step 1: 加载本地 Sentinel-2 影像")
    s2_data = {}
    for season in SEASON_DIRS:
        result = load_s2_season(season)
        if result:
            s2_data[season] = result

    if len(s2_data) < 2:
        log("[ERROR] Sentinel-2 时相不足")
        sys.exit(1)
    log(f"  Sentinel-2: 成功加载 {len(s2_data)} 个时相")

    # 取第一季影像的 grid 参数
    ref_stack, _ = next(iter(s2_data.values()))
    _, height, width = ref_stack.shape
    s2_crs, s2_transform, _, _ = get_local_s2_bounds()

    # 计算遂宁子区域 transform（对应已裁剪的窗口）
    windows = compute_suining_windows()
    w10 = windows["10m"]
    suining_transform = rasterio.transform.from_origin(
        s2_transform.c + w10[2] * s2_transform.a,
        s2_transform.f + w10[0] * s2_transform.e,
        abs(s2_transform.a),
        abs(s2_transform.e),
    )

    log(f"  遂宁裁剪区: {width}x{height} 像元 @ 10m = {width*10/1000:.1f}x{height*10/1000:.1f} km")
    log(f"  CRS: {s2_crs}")

    # ---- Step 2: 加载 WorldCover ----
    log("Step 2: 加载本地 WorldCover")
    wc = load_worldcover(width, height, s2_crs, suining_transform)

    # ---- Step 3: 获取 AOMC ----
    log("Step 3: 加载 AOMC 苹果园标签 (四川2021)")
    aomc = load_aomc_label(s2_crs, suining_transform, width, height)

    # ---- Step 4: 获取 DEM ----
    log("Step 4: 获取 Copernicus DEM")
    dem = download_dem(s2_crs, suining_transform, width, height)
    if dem is not None:
        log("  DEM: OK")
    else:
        log("  [WARN] DEM 获取失败，在无地形特征下运行")

    # ---- Step 5: 构建特征 ----
    log("Step 5: 构建特征立方体")
    cube, feature_names = build_feature_cube(s2_data, dem)

    # 缓存特征立方体供转移学习使用
    sn_feat_dir = DATA_DIR / "suining_features"
    sn_feat_dir.mkdir(exist_ok=True)
    np.save(sn_feat_dir / "feature_cube.npy", cube)
    with open(sn_feat_dir / "transform.json", "w") as f:
        json.dump({"affine": list(suining_transform)[:6]}, f)
    log(f"  特征立方体已缓存 -> suining_features/ ({cube.shape})")

    # ---- Step 6: 构建标签 ----
    log("Step 6: 构建二分类标签")
    labels = build_binary_labels(cube, aomc, wc)

    # ---- Step 7: 训练 ----
    log("Step 7: 训练随机森林二分类")
    result = train_binary_rf(cube, labels, feature_names, suining_transform)

    # ---- Step 8: 输出 ----
    log("Step 8: 保存成果")
    area = save_outputs(result, suining_transform, s2_crs)

    log("=" * 60)
    log("完成！")
    log(f"  F1={result['metrics']['f1']:.4f}")
    log(f"  果树={area['orchard_ha']:.0f} ha")
    log(f"  成果目录: {OUTPUT_DIR}")
    log("=" * 60)
    log("")
    log("===== 后续步骤 =====")
    log("1. GIS中打开 outputs/orchard_binary_10m.tif 查看分类")
    log("2. 对比 outputs/orchard_probability_10m.tif 可靠性")
    log("3. 如果AOMC未获取到（回退到WorldCover），精度偏低，需:")
    log("   - 手动下载AOMC数据集放入 data/")
    log("   - 或用F盘SHP标注树替换正样本")
    log("4. 结合F盘DOM影像目视评估，确定是否需要补充本地样本")


if __name__ == "__main__":
    main()
