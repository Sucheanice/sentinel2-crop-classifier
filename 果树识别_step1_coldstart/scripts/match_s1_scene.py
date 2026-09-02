# -*- coding: utf-8 -*-
"""精确列出遂宁 2024-11-21 / 2025-09-29 所有 S1 子带场景，匹配用户下载的 tiff shape"""
from pystac_client import Client

STAC_URL = "https://earth-search.aws.element84.com/v1"
BBOX = [105.0, 29.7, 106.2, 30.8]

TARGETS = {
    "2024-11-21": (16732, 25758),  # 用户下载 tiff 的 (height, width)
    "2025-09-29": (16756, 25382),
}

cat = Client.open(STAC_URL)

for date, (h, w) in TARGETS.items():
    print("=" * 70)
    print(f"[{date}] 用户 tiff shape = ({h}, {w}), 期望 proj:shape = [{w}, {h}]")
    search = cat.search(
        collections=["sentinel-1-grd"],
        bbox=BBOX,
        datetime=f"{date}/{date}",
        max_items=100,
    )
    for item in search.items():
        pols = item.properties.get("sar:polarizations", [])
        orbit = item.properties.get("sat:orbit_state", "")
        if "VV" not in pols or "VH" not in pols or orbit != "ascending":
            continue
        shape = item.properties.get("proj:shape")
        epsg = item.properties.get("proj:epsg")
        transform = item.properties.get("proj:transform")
        match = "★匹配" if shape == [w, h] else ""
        print(f"  {item.id}  {match}")
        print(f"    proj:shape={shape}  epsg={epsg}  transform={transform}")
        print(f"    vv href 哈希: {item.assets['vv'].href.split('_')[-1]}")
        print()
