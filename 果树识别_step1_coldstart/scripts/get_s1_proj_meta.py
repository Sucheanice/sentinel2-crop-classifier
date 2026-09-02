# -*- coding: utf-8 -*-
"""获取遂宁 2024-11-21 / 2025-09-29 S1 场景的 proj 元数据，验证与 measurement tiff 匹配"""
from pystac_client import Client

STAC_URL = "https://earth-search.aws.element84.com/v1"
BBOX = [105.0, 29.7, 106.2, 30.8]

TARGETS = {
    "2024-11-21": r"E:\迅雷下载\08131435\iw-vv.tiff",
    "2025-09-29": r"E:\迅雷下载\08131435-2\iw-vv.tiff",
}

import rasterio

cat = Client.open(STAC_URL)

for date, tif in TARGETS.items():
    print("=" * 70)
    print(f"[{date}]")
    with rasterio.open(tif) as src:
        print(f"  tiff shape: {src.shape}")

    search = cat.search(
        collections=["sentinel-1-grd"],
        bbox=BBOX,
        datetime=f"{date}/{date}",
        max_items=50,
    )
    for item in search.items():
        pols = item.properties.get("sar:polarizations", [])
        orbit = item.properties.get("sat:orbit_state", "")
        if "VV" not in pols or "VH" not in pols or orbit != "ascending":
            continue
        epsg = item.properties.get("proj:epsg")
        shape = item.properties.get("proj:shape")
        transform = item.properties.get("proj:transform")
        print(f"  场景: {item.id}")
        print(f"  proj:epsg = {epsg}")
        print(f"  proj:shape = {shape}")
        print(f"  proj:transform = {transform}")
        # 验证 vv asset 的 shape
        print(f"  vv asset href = {item.assets['vv'].href[:80]}...")
        break
    print()
