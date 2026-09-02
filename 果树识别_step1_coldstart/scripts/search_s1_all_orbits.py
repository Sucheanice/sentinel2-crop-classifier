# -*- coding: utf-8 -*-
"""搜索 2024-11-21 / 2025-09-29 所有 S1 场景(升/降轨, 全部子带)，看安居区覆盖。"""
from pystac_client import Client

STAC_URL = "https://earth-search.aws.element84.com/v1"
# 安居区 bbox
BBOX = [105.08, 30.17, 105.71, 30.47]

cat = Client.open(STAC_URL)

for date in ["2024-11-21", "2025-09-29"]:
    print("=" * 70)
    print(f"[{date}]")
    search = cat.search(
        collections=["sentinel-1-grd"],
        bbox=BBOX,
        datetime=f"{date}/{date}",
        max_items=100,
    )
    items = list(search.items())
    print(f"  相交场景数={len(items)}")
    for item in items:
        orbit = item.properties.get("sat:orbit_state", "")
        pols = item.properties.get("sar:polarizations", [])
        shape = item.properties.get("proj:shape")
        geom = item.geometry
        # footprint bbox (lon/lat)
        import json
        coords = json.loads(json.dumps(geom))["coordinates"]
        print(f"  {item.id}")
        print(f"    轨道={orbit}, 极化={pols}, shape={shape}")
