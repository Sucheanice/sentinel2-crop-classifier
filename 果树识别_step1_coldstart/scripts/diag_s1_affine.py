# -*- coding: utf-8 -*-
"""诊断 S1 GCP 仿射拟合: 自己用 numpy 计算，检查残差是否可接受。"""
import numpy as np
import rasterio

for label, p in [("2024-11-21", r"E:\迅雷下载\08131435\iw-vv.tiff"),
                 ("2025-09-29", r"E:\迅雷下载\08131435-2\iw-vv.tiff")]:
    with rasterio.open(p) as src:
        gcps, _ = src.gcps
        cols = np.array([g.col for g in gcps], dtype=np.float64)
        rows = np.array([g.row for g in gcps], dtype=np.float64)
        lons = np.array([g.x for g in gcps], dtype=np.float64)
        lats = np.array([g.y for g in gcps], dtype=np.float64)

    print(f"\n===== {label} =====")
    print(f"  shape={src.shape}, GCP数={len(gcps)}")

    # 仿射拟合: lon = a*col + b*row + c ; lat = d*col + e*row + f
    A = np.column_stack([cols, rows, np.ones_like(cols)])
    coef_lon, res_lon, *_ = np.linalg.lstsq(A, lons, rcond=None)
    coef_lat, res_lat, *_ = np.linalg.lstsq(A, lats, rcond=None)

    a, b, c = coef_lon
    d, e, f = coef_lat
    print(f"  仿射(像素→经纬度):")
    print(f"    lon = {a:.8e}*col + {b:.8e}*row + {c:.6f}")
    print(f"    lat = {d:.8e}*col + {e:.8e}*row + {f:.6f}")

    # 残差
    lon_pred = A @ coef_lon
    lat_pred = A @ coef_lat
    lon_err = (lons - lon_pred)
    lat_err = (lats - lat_pred)
    # 误差转米 (1度经度≈95km@30N, 1度纬度≈111km)
    err_m = np.hypot(lon_err * 95000 * np.cos(np.radians(30)), lat_err * 111000)
    print(f"  残差: lon RMSE={np.sqrt(np.mean(lon_err**2))*1e5:.0f}m, "
          f"lat RMSE={np.sqrt(np.mean(lat_err**2))*1e5:.0f}m")
    print(f"  位置误差: mean={err_m.mean():.0f}m, max={err_m.max():.0f}m")

    # 检查像素分辨率 (米/像素)
    # 用 GCP 对角点
    pix_m_col = (np.hypot((lons[209]-lons[0])*95000*np.cos(np.radians(30)),
                          (lats[209]-lats[0])*111000) /
                 np.hypot(cols[209]-cols[0], rows[209]-rows[0]))
    print(f"  平均像素尺寸≈{pix_m_col:.1f} m (GRD应为10m)")
