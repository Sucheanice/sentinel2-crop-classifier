# -*- coding: utf-8 -*-
"""诊断 S1 源 tiff 的 GCP 覆盖范围和 from_gcps 变换精度。"""
import numpy as np
import rasterio
from rasterio.transform import from_gcps, rowcol
from pyproj import Transformer

SRC = r"E:\迅雷下载\08131435\iw-vv.tiff"
SRC2 = r"E:\迅雷下载\08131435-2\iw-vv.tiff"

to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32648", always_xy=True)

for label, p in [("2024-11-21", SRC), ("2025-09-29", SRC2)]:
    with rasterio.open(p) as src:
        print(f"\n===== {label} =====")
        print(f"  shape={src.shape}, crs={src.crs}, transform={src.transform}")
        gcps, gcp_crs = src.gcps
        print(f"  GCP数量={len(gcps)}, crs={gcp_crs}")
        # GCP 的经纬度范围
        lons = [g.x for g in gcps]
        lats = [g.y for g in gcps]
        rows = [g.row for g in gcps]
        cols = [g.col for g in gcps]
        print(f"  GCP lat 范围: {min(lats):.4f} ~ {max(lats):.4f}")
        print(f"  GCP lon 范围: {min(lons):.4f} ~ {max(lons):.4f}")
        print(f"  GCP row 范围: {min(rows)} ~ {max(rows)}")
        print(f"  GCP col 范围: {min(cols)} ~ {max(cols)}")
        # 按 row 排序，看纬度沿 row 的分布
        order = np.argsort(rows)
        for i in [0, len(gcps)//4, len(gcps)//2, 3*len(gcps)//4, len(gcps)-1]:
            g = gcps[order[i]]
            print(f"    GCP[{i}]: row={g.row:.0f} col={g.col:.0f} -> lon={g.x:.4f} lat={g.y:.4f}")

        # from_gcps 变换
        tr = from_gcps(gcps)
        print(f"  from_gcps transform: {tr}")

        # 检查 安居区北部 (105.6, 30.47) 在源图像中的行列
        x, y = to_utm.transform(105.6, 30.47)
        # 用 GCP 直接: 把 lon/lat 反投影到源行列
        inv = ~tr
        c, r = inv * (x, y)
        print(f"  安居区北部(105.6,30.47) UTM=({x:.0f},{y:.0f}) -> 源 col={c:.0f} row={r:.0f} (shape={src.shape})")
        # 安居区中心 (105.4,30.3)
        x2, y2 = to_utm.transform(105.4, 30.3)
        c2, r2 = inv * (x2, y2)
        print(f"  安居区中心(105.4,30.3) -> 源 col={c2:.0f} row={r2:.0f}")
