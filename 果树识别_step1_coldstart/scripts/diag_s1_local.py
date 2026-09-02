# -*- coding: utf-8 -*-
"""测试: 仅用遂宁附近(lat>=29.9)的 GCP 做局部拟合，检查残差。"""
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

    # 遂宁 bbox: lon 104.9-106.2, lat 30.0-30.6 (S1只覆盖到30.54)
    mask = (lats >= 29.9) & (lons >= 104.5) & (lons <= 106.5)
    c, r, lo, la = cols[mask], rows[mask], lons[mask], lats[mask]
    print(f"\n===== {label} =====")
    print(f"  遂宁附近 GCP 数 = {mask.sum()}")

    # 局部仿射 (1阶) + 2阶多项式
    def feats(col, row, order):
        out = []
        for i in range(order + 1):
            for j in range(order + 1 - i):
                out.append((col ** i) * (row ** j))
        return np.column_stack(out)

    for order in [1, 2, 3]:
        cn = (c - c.mean()) / c.std()
        rn = (r - r.mean()) / r.std()
        A = feats(cn, rn, order)
        clo, *_ = np.linalg.lstsq(A, lo, rcond=None)
        cla, *_ = np.linalg.lstsq(A, la, rcond=None)
        lo_err = lo - A @ clo
        la_err = la - A @ cla
        err = np.hypot(lo_err * 95000 * np.cos(np.radians(30)), la_err * 111000)
        print(f"  局部{order}阶: mean={err.mean():.1f}m, max={err.max():.1f}m")
