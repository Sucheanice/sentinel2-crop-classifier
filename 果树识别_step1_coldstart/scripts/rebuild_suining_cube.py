# -*- coding: utf-8 -*-
"""重建遂宁特征立方体 - 修复 NaN 问题"""
import json, sys, time, math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from scipy.ndimage import uniform_filter

WORK_DIR = Path(r"E:\工作相关\2026年\0624 待测试数据")
PROJ_DIR = WORK_DIR / "果树识别_step1_coldstart"
DATA_DIR = PROJ_DIR / "data"

S2_MAIN = WORK_DIR / "小春_s2_48RWU"
S2_EXTRA = WORK_DIR / "小春_s2_48RWU_extra"
DEM_PATH = DATA_DIR / "xuyong_dem" / "xuyong_dem_10m.tif"

S2_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
S2_10M = {"B02", "B03", "B04", "B08"}
S2_20M = {"B05", "B06", "B07", "B8A", "B11", "B12"}

SEASONS = {
    "winter":       S2_MAIN / "2025-01-20_48RWU_cloud1.2",
    "spring":       S2_MAIN / "2025-03-26_48RWU_cloud0.0",
    "late_spring":  WORK_DIR / "小春_s2_48RWU_summer" / "late_spring_10m_10band.tif",  # 5月
    "summer":       WORK_DIR / "小春_s2_48RWU_summer" / "summer_10m_10band.tif",  # 7月, 迅雷JP2转
    "late_autumn":  S2_MAIN / "2024-12-11_48RWU_cloud9.5",
}

# 遂宁 SAR 后向散射 dB (已对齐到遂宁 S2 网格 10015x10576, nodata=-9999)
# label 与叙永侧 SAR_SCENES_XUYONG 一一对应 (9月→autumn, 11-12月→late_autumn)
SAR_SCENES_SUINING = {
    "autumn":      (WORK_DIR / "小春_s1_48RWU" / "2025-09-29_S1_asc" / "vv_db.tif",
                    WORK_DIR / "小春_s1_48RWU" / "2025-09-29_S1_asc" / "vh_db.tif"),
    "late_autumn": (WORK_DIR / "小春_s1_48RWU" / "2024-11-21_S1_asc" / "vv_db.tif",
                    WORK_DIR / "小春_s1_48RWU" / "2024-11-21_S1_asc" / "vh_db.tif"),
}

SUINING_BBOX = [105.00, 30.15, 106.05, 31.10]  # WGS84
TARGET_CRS = "EPSG:32648"
RES = 10.0

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def compute_target_grid():
    """计算遂宁区域在 UTM 48N 10m 下的目标网格"""
    from pyproj import Transformer
    t = Transformer.from_crs("EPSG:4326", TARGET_CRS, always_xy=True)
    xmin, ymin = t.transform(SUINING_BBOX[0], SUINING_BBOX[1])
    xmax, ymax = t.transform(SUINING_BBOX[2], SUINING_BBOX[3])
    xmin = math.floor(xmin / RES) * RES
    ymax = math.ceil(ymax / RES) * RES
    xmax = math.ceil(xmax / RES) * RES
    ymin = math.floor(ymin / RES) * RES
    width = int((xmax - xmin) / RES)
    height = int((ymax - ymin) / RES)
    transform = rasterio.transform.from_origin(xmin, ymax, RES, RES)
    return width, height, transform

def load_season_bands(season_path):
    """加载单个季节的10波段，统一到10m，重投影到目标网格
    支持: (a) 目录内单波段tif  (b) 多波段GeoTIFF文件
    """
    width, height, dst_transform = compute_target_grid()
    dst_crs = TARGET_CRS

    bands = []
    if season_path.is_file():
        # 多波段GeoTIFF
        with rasterio.open(season_path) as src:
            for i in range(1, src.count + 1):
                data = src.read(i).astype("float32")
                out = np.full((height, width), np.nan, dtype="float32")
                reproject(
                    source=data, destination=out,
                    src_transform=src.transform, src_crs=src.crs,
                    dst_transform=dst_transform, dst_crs=dst_crs,
                    resampling=Resampling.bilinear,
                )
                out[out <= 0] = np.nan
                bands.append(out)
    else:
        # 目录内单波段tif
        for bn in S2_BANDS:
            fpath = season_path / f"{bn}.tif"
            if not fpath.exists():
                log(f"  [WARN] {bn}.tif 不存在")
                return None
            with rasterio.open(fpath) as src:
                data = src.read(1).astype("float32")
                out = np.full((height, width), np.nan, dtype="float32")
                reproject(
                    source=data, destination=out,
                    src_transform=src.transform, src_crs=src.crs,
                    dst_transform=dst_transform, dst_crs=dst_crs,
                    resampling=Resampling.bilinear,
                )
                out[out <= 0] = np.nan
                bands.append(out)
    return np.stack(bands), width, height, dst_transform

def compute_indices(bands):
    """bands: [10, H, W]"""
    eps = 1e-8
    b2, b3, b4, b5, b6, b7, b8, b8a, b11, b12 = [bands[i] for i in range(10)]
    ndvi = (b8 - b4) / (b8 + b4 + eps)
    ndre = (b8a - b5) / (b8a + b5 + eps)
    ndmi = (b8a - b11) / (b8a + b11 + eps)
    evi = 2.5 * (b8 - b4) / (b8 + 6 * b4 - 7.5 * b2 + 1 + eps)
    return ndvi, ndre, ndmi, evi

def compute_texture(img, size=3):
    mean = uniform_filter(np.nan_to_num(img, nan=0).astype("float64"), size=size, mode="reflect")
    sq_mean = uniform_filter(np.nan_to_num(img, nan=0).astype("float64") ** 2, size=size, mode="reflect")
    return np.sqrt(np.maximum(sq_mean - mean ** 2, 0)).astype("float32")

def load_dem(H, W, dst_transform):
    """加载 DEM 并重投影到目标网格 (H=行数, W=列数)"""
    if not DEM_PATH.exists():
        log("  [WARN] DEM 文件不存在，使用全零")
        return np.zeros((2, H, W), dtype="float32")

    with rasterio.open(DEM_PATH) as src:
        elev = np.full((H, W), np.nan, dtype="float32")
        reproject(
            source=src.read(1),
            destination=elev,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=TARGET_CRS,
            resampling=Resampling.bilinear,
        )

    # 计算坡度
    from numpy import gradient
    gy, gx = gradient(elev, RES)
    slope = np.arctan(np.sqrt(gx**2 + gy**2))  # 弧度

    return np.stack([np.nan_to_num(elev, nan=0), np.nan_to_num(slope, nan=0)])

def _read_sar_band(path, H, W):
    """读取已对齐到目标网格的 SAR dB 波段, nodata(-9999)→NaN"""
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
    arr[arr <= -9000] = np.nan
    return arr[:H, :W]

def main():
    log("=" * 60)
    log("重建遂宁特征立方体")
    log("=" * 60)

    # 计算目标网格
    width, height, dst_transform = compute_target_grid()
    log(f"目标网格: {width}x{height} @ 10m ({(width*RES/1000):.1f}x{(height*RES/1000):.1f} km)")

    # 加载各季节
    s2_data = {}
    for season, sdir in SEASONS.items():
        log(f"\n加载 {season}: {sdir.name}")
        result = load_season_bands(sdir)
        if result is None:
            log(f"  [WARN] {season} 加载失败，跳过")
            continue
        arr, w, h, tr = result
        nan_pct = np.isnan(arr).sum() / arr.size * 100
        log(f"  -> shape={arr.shape}, NaN={nan_pct:.1f}%")
        if nan_pct > 50:
            log(f"  [WARN] {season} NaN 比例过高 ({nan_pct:.1f}%)，跳过")
            continue
        s2_data[season] = arr

    if len(s2_data) < 2:
        log("[FATAL] 有效季节不足")
        sys.exit(1)
    log(f"\n有效季节: {list(s2_data.keys())}")

    # 确定公共有效区域
    H, W = height, width
    log(f"最终尺寸: {H}x{W}")

    # 构建特征（与 run_suining_binary.py 中 build_feature_cube 一致）
    # 季节顺序: 字母序与叙永对齐 (late_autumn, spring, summer, winter)
    season_order = sorted(s2_data.keys())
    active_seasons = [s for s in season_order if s in s2_data]

    features = []
    band_names = []

    # 1. 光谱
    for s in active_seasons:
        arr = s2_data[s]
        features.append(arr.reshape(10, H, W))
        for b in S2_BANDS:
            band_names.append(f"s2_{s}_{b}")
    log(f"光谱: {len(active_seasons)}x10={len(active_seasons)*10}")

    # 2. 植被指数
    indices = {}
    for s in active_seasons:
        ndvi, ndre, ndmi, evi = compute_indices(s2_data[s])
        indices[s] = {"ndvi": ndvi, "ndre": ndre, "ndmi": ndmi, "evi": evi}
        features.extend([ndvi[None], ndre[None], ndmi[None], evi[None]])
        for vi in ["ndvi", "ndre", "ndmi", "evi"]:
            band_names.append(f"vi_{s}_{vi}")
    log(f"指数: {len(active_seasons)}x4={len(active_seasons)*4}")

    # 3. 物候差值
    for i in range(len(active_seasons) - 1):
        s1, s2 = active_seasons[i], active_seasons[i + 1]
        diff_ndvi = indices[s2]["ndvi"] - indices[s1]["ndvi"]
        diff_ndre = indices[s2]["ndre"] - indices[s1]["ndre"]
        features.extend([diff_ndvi[None], diff_ndre[None]])
        band_names.append(f"phen_diff_ndvi_{s1}_to_{s2}")
        band_names.append(f"phen_diff_ndre_{s1}_to_{s2}")
    log(f"差值: {(len(active_seasons)-1)}x2={(len(active_seasons)-1)*2}")

    # 4. 纹理（用最后一季）
    tex_s = active_seasons[-1]
    ps_bands = s2_data[tex_s]
    ps_ndvi = indices[tex_s]["ndvi"]
    ps_nir = ps_bands[S2_BANDS.index("B08")]
    for win in [3, 7]:
        tex_ndvi = compute_texture(ps_ndvi, win)
        tex_nir = compute_texture(ps_nir, win)
        features.extend([tex_ndvi[None], tex_nir[None]])
        band_names.append(f"tex_{tex_s}_ndvi_{win}x{win}")
        band_names.append(f"tex_{tex_s}_nir_{win}x{win}")
    log(f"纹理: 4")

    # 5. DEM
    dem_arr = load_dem(H, W, dst_transform)
    features.extend([dem_arr[0:1], dem_arr[1:2]])
    band_names.extend(["dem_elevation", "dem_slope"])
    log(f"DEM: 2")

    # 6. SAR 后向散射 (VV/VH, 与叙永侧 label 一致)
    for label in sorted(SAR_SCENES_SUINING):
        vv_path, vh_path = SAR_SCENES_SUINING[label]
        vv = _read_sar_band(vv_path, H, W)
        vh = _read_sar_band(vh_path, H, W)
        features.extend([vv[None], vh[None]])
        band_names.append(f"s1_{label}_vv")
        band_names.append(f"s1_{label}_vh")
    log(f"SAR: {len(SAR_SCENES_SUINING)}x2={len(SAR_SCENES_SUINING)*2}")

    cube = np.concatenate(features, axis=0)
    log(f"\n总特征: {cube.shape[0]} bands, {H}x{W}")
    log(f"NaN 总数: {np.isnan(cube).sum():,} / {cube.size:,} ({np.isnan(cube).sum()/cube.size*100:.1f}%)")

    # 保存
    out_dir = DATA_DIR / "suining_features"
    out_dir.mkdir(exist_ok=True)
    np.save(out_dir / "feature_cube.npy", cube)
    with open(out_dir / "transform.json", "w") as f:
        json.dump({"affine": list(dst_transform)[:6]}, f)
    with open(out_dir / "band_names.json", "w") as f:
        json.dump(band_names, f)
    log(f"\n已保存 -> {out_dir / 'feature_cube.npy'}")
    log(f"波段名 -> {out_dir / 'band_names.json'}")
    log("完成!")

if __name__ == "__main__":
    main()
