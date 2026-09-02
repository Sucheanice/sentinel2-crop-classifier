# -*- coding: utf-8 -*-
"""识别用户下载的 S1 measurement tiff 属于哪个场景（通过 GCP id）"""
import rasterio, sys

for f in sys.argv[1:]:
    print("=" * 70)
    print("FILE:", f)
    with rasterio.open(f) as d:
        print("crs:", d.crs)
        print("transform:", d.transform)
        print("size:", d.width, "x", d.height)
        print("tags:", d.tags())
        gcps, gcps_crs = d.gcps
        print("n_gcp:", len(gcps), "crs:", gcps_crs)
        for gp in gcps[:6]:
            print("  ", gp.id, "row=", round(gp.row, 1), "col=", round(gp.col, 1),
                  "lon=", round(gp.x, 6), "lat=", round(gp.y, 6))
        print("  ...")
        for gp in gcps[-3:]:
            print("  ", gp.id, "row=", round(gp.row, 1), "col=", round(gp.col, 1),
                  "lon=", round(gp.x, 6), "lat=", round(gp.y, 6))
