# -*- coding: utf-8 -*-
"""验证 S1 dB reproject 结果: 覆盖范围 + 数值合理性。"""
import numpy as np
import rasterio
from rasterio.transform import rowcol
from pyproj import Transformer

WORK = r"E:\工作相关\2026年\0624 待测试数据\小春_s1_48RWU"

# 要检查的点 (lon, lat, 说明)
POINTS = [
    (105.40, 30.30, "安居区中心"),
    (105.20, 30.18, "安居区南部"),
    (105.60, 30.47, "安居区北部"),
    (105.50, 30.55, "遂宁中部(边界附近)"),
    (105.50, 30.80, "遂宁北部(大英县)"),
    (105.30, 31.00, "遂宁最北(应nodata)"),
]

to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32648", always_xy=True)

for date in ["2024-11-21", "2025-09-29"]:
    for band in ["vv", "vh"]:
        p = rf"{WORK}\{date}_S1_asc\{band}_db.tif"
        with rasterio.open(p) as src:
            data = src.read(1)
            h, w = data.shape
            valid = data != src.nodata
            print(f"\n===== {date} {band} =====")
            print(f"  尺寸 {w}x{h}, nodata={src.nodata}, CRS={src.crs}")
            print(f"  有效像素占比: {valid.sum()/valid.size*100:.2f}%")

            # 按行(纬度)统计有效占比
            row_valid = valid.sum(axis=1) / w
            step = max(1, h // 20)
            print(f"  纬度带有效占比 (row: northing -> valid%):")
            for r in range(0, h, step):
                x, y = src.transform * (0, r)  # (col=0, row=r) -> (easting, northing)
                print(f"    row {r:5d}: y={y:10.0f} -> {row_valid[r]*100:5.1f}%")

            # 采样点
            print(f"  采样点:")
            for lon, lat, label in POINTS:
                x, y = to_utm.transform(lon, lat)
                r, c = rowcol(src.transform, x, y)
                if 0 <= r < h and 0 <= c < w:
                    val = data[r, c]
                    print(f"    {label} ({lon},{lat}): r={r} c={c} dB={val}")
                else:
                    print(f"    {label} ({lon},{lat}): 越界 r={r} c={c}")
