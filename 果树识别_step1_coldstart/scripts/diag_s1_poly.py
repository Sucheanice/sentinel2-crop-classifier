# -*- coding: utf-8 -*-
"""检查 GDAL 可用性 + 多项式(2/3阶)拟合残差，决定用哪种 warping。"""
import numpy as np
import rasterio

# 1) GDAL 可用性
try:
    from osgeo import gdal
    print(f"osgeo.gdal 可用, 版本={gdal.VersionInfo()}")
except ImportError as e:
    print(f"osgeo.gdal 不可用: {e}")

# 2) 多项式拟合残差
def poly_features(cols, rows, order):
    feats = []
    for i in range(order + 1):
        for j in range(order + 1 - i):
            feats.append((cols ** i) * (rows ** j))
    return np.column_stack(feats)

for label, p in [("2024-11-21", r"E:\迅雷下载\08131435\iw-vv.tiff"),
                 ("2025-09-29", r"E:\迅雷下载\08131435-2\iw-vv.tiff")]:
    with rasterio.open(p) as src:
        gcps, _ = src.gcps
        cols = np.array([g.col for g in gcps], dtype=np.float64)
        rows = np.array([g.row for g in gcps], dtype=np.float64)
        lons = np.array([g.x for g in gcps], dtype=np.float64)
        lats = np.array([g.y for g in gcps], dtype=np.float64)

    # 归一化坐标避免病态
    cols_n = (cols - cols.mean()) / cols.std()
    rows_n = (rows - rows.mean()) / rows.std()

    print(f"\n===== {label} 多项式残差 =====")
    for order in [1, 2, 3]:
        A = poly_features(cols_n, rows_n, order)
        coef_lon, *_ = np.linalg.lstsq(A, lons, rcond=None)
        coef_lat, *_ = np.linalg.lstsq(A, lats, rcond=None)
        lon_err = lons - A @ coef_lon
        lat_err = lats - A @ coef_lat
        err_m = np.hypot(lon_err * 95000 * np.cos(np.radians(30)), lat_err * 111000)
        print(f"  阶数{order}: mean={err_m.mean():.1f}m, max={err_m.max():.1f}m, "
              f"RMSE={np.sqrt(np.mean(err_m**2)):.1f}m")
