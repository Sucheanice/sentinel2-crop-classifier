# -*- coding: utf-8 -*-
"""处理叙永 S1 GRD → 后向散射 dB → 叙永 S2 特征网格 (10m UTM48N) GeoTIFF

与 process_s1_db.py (遂宁) 相同思路:
  - GRD DN → σ⁰_dB = 10*log10(DN²), DN<=0 置 NaN
  - GCP (lon,lat)->(row,col) LinearNDInterpolator 逆变换 (精度 ~50m)
  - 目标网格与叙永缓存 S2 完全一致 (2963x1232 @10m, EPSG:32648, 原点 536440,3075230)

用法:
  python process_xuyong_s1_db.py
"""
import time
from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import from_origin
from pyproj import Transformer
from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage import map_coordinates

WORK_DIR = Path(r"E:\工作相关\2026年\0624 待测试数据")
OUT_DIR = WORK_DIR / "叙永_s1_48RWR"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RES = 10.0
NODATA = -9999.0

# 与 data/xuyong_s2/*_10m_10band.tif 完全对齐的网格
XMIN, YMAX = 536440.0, 3075230.0
WIDTH, HEIGHT = 2963, 1232

SCENES = {
    "2025-09-17": Path(r"E:\迅雷下载\08131435-叙永"),
    "2024-12-03": Path(r"E:\迅雷下载\08131435-叙永2"),
}


def grd_to_db(dn):
    dn = dn.astype(np.float32)
    dn = np.where(dn <= 0, np.nan, dn)
    return 10.0 * np.log10(np.square(dn))


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_inverse_interpolators(gcps):
    cols = np.array([g.col for g in gcps], dtype=np.float64)
    rows = np.array([g.row for g in gcps], dtype=np.float64)
    lons = np.array([g.x for g in gcps], dtype=np.float64)
    lats = np.array([g.y for g in gcps], dtype=np.float64)
    pts = np.column_stack([lons, lats])
    f_col = LinearNDInterpolator(pts, cols)
    f_row = LinearNDInterpolator(pts, rows)
    return f_col, f_row


def warp_band(db, f_col, f_row, dst_transform, to_wgs):
    out = np.full((HEIGHT, WIDTH), np.nan, dtype=np.float32)
    a, b, c, d, e, f = (dst_transform.a, dst_transform.b, dst_transform.c,
                        dst_transform.d, dst_transform.e, dst_transform.f)
    cols = np.arange(WIDTH, dtype=np.float64)
    block = 200
    for r0 in range(0, HEIGHT, block):
        r1 = min(r0 + block, HEIGHT)
        rows = np.arange(r0, r1, dtype=np.float64)
        C, R = np.meshgrid(cols, rows)
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
    dst_transform = from_origin(XMIN, YMAX, RES, RES)
    log(f"叙永目标网格: {WIDTH}x{HEIGHT} @ 10m, UTM48N (与缓存S2对齐)")
    to_wgs = Transformer.from_crs("EPSG:32648", "EPSG:4326", always_xy=True)

    for date, scene_dir in SCENES.items():
        log(f"\n[{date}]")
        out_scene = OUT_DIR / f"{date}_S1_asc"
        out_scene.mkdir(parents=True, exist_ok=True)

        vv_path = scene_dir / "iw-vv.tiff"
        vh_path = scene_dir / "iw-vh.tiff"
        if not vv_path.exists():
            log(f"  [WARN] iw-vv.tiff 不存在")
            continue

        with rasterio.open(vv_path) as src:
            gcps, gcp_crs = src.gcps
            vv_dn = src.read(1)
        log(f"  GCP 数量={len(gcps)}, crs={gcp_crs}")

        f_col, f_row = build_inverse_interpolators(gcps)

        vv_db = grd_to_db(vv_dn)
        log(f"  VV 有效像素={np.isfinite(vv_db).sum()/vv_db.size*100:.1f}%")
        vv_out = warp_band(vv_db, f_col, f_row, dst_transform, to_wgs)
        log(f"  VV warp 后有效像素={np.isfinite(vv_out).sum()/vv_out.size*100:.1f}%")

        if vh_path.exists():
            with rasterio.open(vh_path) as src:
                vh_dn = src.read(1)
            vh_db = grd_to_db(vh_dn)
            log(f"  VH 有效像素={np.isfinite(vh_db).sum()/vh_db.size*100:.1f}%")
            vh_out = warp_band(vh_db, f_col, f_row, dst_transform, to_wgs)
            log(f"  VH warp 后有效像素={np.isfinite(vh_out).sum()/vh_out.size*100:.1f}%")
        else:
            vh_out = None

        profile = {
            "driver": "GTiff", "height": HEIGHT, "width": WIDTH,
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
