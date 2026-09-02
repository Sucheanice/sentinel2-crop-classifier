# -*- coding: utf-8 -*-
"""处理遂宁 2024-11-21 / 2025-09-29 S1 GRD → 后向散射 dB → 遂宁 S2 网格 GeoTIFF

v2 修复:
1. nodata bug: reproject 默认 init_dest_nodata=True 会把预填 NaN 覆盖成 0。
2. 地理精度: 单仿射拟合 S1 轨道残差 ~350m。改用 GCP 反距离插值 (LinearNDInterpolator)
   做 (lon,lat)->(row,col) 逆变换，精度 ~50m (element84 COG 稀疏 GCP 网格的信息极限)。

用法:
  python process_s1_db.py
"""
import time, math
from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import from_origin
from pyproj import Transformer
from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage import map_coordinates

WORK_DIR = Path(r"E:\工作相关\2026年\0624 待测试数据")
OUT_DIR = WORK_DIR / "小春_s1_48RWU"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RES = 10.0
SUINING_BBOX_WGS = [105.00, 30.15, 106.05, 31.10]
NODATA = -9999.0

SCENES = {
    "2024-11-21": Path(r"E:\迅雷下载\08131435"),
    "2025-09-29": Path(r"E:\迅雷下载\08131435-2"),
}

# 目标网格四周留的纬度余量 (度)，超出 GCP 覆盖即 nodata
LAT_MARGIN = 0.02


def compute_target_grid():
    t = Transformer.from_crs("EPSG:4326", "EPSG:32648", always_xy=True)
    xmin, ymin = t.transform(SUINING_BBOX_WGS[0], SUINING_BBOX_WGS[1])
    xmax, ymax = t.transform(SUINING_BBOX_WGS[2], SUINING_BBOX_WGS[3])
    xmin = math.floor(xmin / RES) * RES
    ymax = math.ceil(ymax / RES) * RES
    xmax = math.ceil(xmax / RES) * RES
    ymin = math.floor(ymin / RES) * RES
    width = int((xmax - xmin) / RES)
    height = int((ymax - ymin) / RES)
    return width, height, from_origin(xmin, ymax, RES, RES)


def grd_to_db(dn):
    dn = dn.astype(np.float32)
    dn = np.where(dn <= 0, np.nan, dn)
    return 10.0 * np.log10(np.square(dn))


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_inverse_interpolators(gcps):
    """用 GCP 建立 (lon,lat)->(col,row) 的逆变换插值器。"""
    cols = np.array([g.col for g in gcps], dtype=np.float64)
    rows = np.array([g.row for g in gcps], dtype=np.float64)
    lons = np.array([g.x for g in gcps], dtype=np.float64)
    lats = np.array([g.y for g in gcps], dtype=np.float64)
    pts = np.column_stack([lons, lats])
    f_col = LinearNDInterpolator(pts, cols)
    f_row = LinearNDInterpolator(pts, rows)
    return f_col, f_row, lats.max()


def warp_band(db, f_col, f_row, width, height, dst_transform, to_wgs, r_start):
    """把源 dB 数组 warp 到目标 UTM 网格，返回 (height,width) float32 数组。"""
    out = np.full((height, width), np.nan, dtype=np.float32)
    a, b, c, d, e, f = (dst_transform.a, dst_transform.b, dst_transform.c,
                        dst_transform.d, dst_transform.e, dst_transform.f)

    block = 400
    cols = np.arange(width, dtype=np.float64)
    for r0 in range(r_start, height, block):
        r1 = min(r0 + block, height)
        rows = np.arange(r0, r1, dtype=np.float64)
        C, R = np.meshgrid(cols, rows)  # (nrows, width)
        x = a * C + b * R + c
        y = d * C + e * R + f
        lon, lat = to_wgs.transform(x, y)

        col_src = f_col(lon, lat)
        row_src = f_row(lon, lat)
        valid = np.isfinite(col_src) & np.isfinite(row_src)
        col_src = np.where(valid, col_src, 0.0)
        row_src = np.where(valid, row_src, 0.0)

        coords = np.vstack([row_src.ravel(), col_src.ravel()])
        sampled = map_coordinates(db, coords, order=1, mode="constant",
                                  cval=0.0, prefilter=False).reshape(col_src.shape)
        sampled = np.where(valid, sampled, np.nan)
        out[r0:r1] = sampled.astype(np.float32)
    return out


def main():
    t0 = time.time()
    width, height, dst_transform = compute_target_grid()
    log(f"遂宁目标网格: {width}x{height} @ 10m, UTM48N")

    to_wgs = Transformer.from_crs("EPSG:32648", "EPSG:4326", always_xy=True)

    for date, scene_dir in SCENES.items():
        log(f"\n[{date}]")
        out_scene = OUT_DIR / f"{date}_S1_asc"
        out_scene.mkdir(parents=True, exist_ok=True)

        # 先读 vv 拿 GCP (两个波段 GCP 相同)
        vv_path = scene_dir / "iw-vv.tiff"
        vh_path = scene_dir / "iw-vh.tiff"
        if not vv_path.exists():
            log(f"  [WARN] iw-vv.tiff 不存在")
            continue

        with rasterio.open(vv_path) as src:
            gcps, gcp_crs = src.gcps
            vv_dn = src.read(1)
        n_gcp = len(gcps)
        log(f"  GCP 数量={n_gcp}, crs={gcp_crs}")

        f_col, f_row, max_lat = build_inverse_interpolators(gcps)
        log(f"  GCP 覆盖最大纬度={max_lat:.4f}")

        # 计算目标网格中纬度 <= max_lat+LAT_MARGIN 的起始行 (行从上到下纬度递减)
        # 目标网格顶部纬度 = dst_transform.f
        top_lat_t = Transformer.from_crs("EPSG:32648", "EPSG:4326", always_xy=True)
        _, top_lat = top_lat_t.transform(dst_transform.c, dst_transform.f)
        lat_limit = max_lat + LAT_MARGIN
        # 每行纬度递减 (度) = RES 米 / 111320 米每度纬度
        deg_per_row = RES / 111320.0
        r_start = int((top_lat - lat_limit) / deg_per_row) + 1
        r_start = max(0, min(r_start, height))
        log(f"  warp 起始行 r_start={r_start} (纬度<=~{lat_limit:.3f})")

        vv_db = grd_to_db(vv_dn)
        log(f"  VV 有效像素={np.isfinite(vv_db).sum()/vv_db.size*100:.1f}%")
        vv_out = warp_band(vv_db, f_col, f_row, width, height, dst_transform, to_wgs, r_start)
        log(f"  VV warp 后有效像素={np.isfinite(vv_out).sum()/vv_out.size*100:.1f}%")

        if vh_path.exists():
            with rasterio.open(vh_path) as src:
                vh_dn = src.read(1)
            vh_db = grd_to_db(vh_dn)
            log(f"  VH 有效像素={np.isfinite(vh_db).sum()/vh_db.size*100:.1f}%")
            vh_out = warp_band(vh_db, f_col, f_row, width, height, dst_transform, to_wgs, r_start)
            log(f"  VH warp 后有效像素={np.isfinite(vh_out).sum()/vh_out.size*100:.1f}%")
        else:
            log(f"  [WARN] iw-vh.tiff 不存在")
            vh_out = None

        profile = {
            "driver": "GTiff", "height": height, "width": width,
            "count": 1, "dtype": "float32",
            "crs": "EPSG:32648", "transform": dst_transform,
            "compress": "deflate", "nodata": NODATA,
        }
        for band, arr in [("vv", vv_out), ("vh", vh_out)]:
            if arr is None:
                continue
            out_nan = np.nan_to_num(arr, nan=NODATA)
            out_path = out_scene / f"{band}_db.tif"
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(out_nan, 1)
            log(f"  -> {out_path.name}")

    log(f"\n完成! {(time.time()-t0)/60:.1f}分")


if __name__ == "__main__":
    main()
