# -*- coding: utf-8 -*-
"""搜索遂宁 48RWU 夏季 S2 影像 (2025年6-8月)"""
import planetary_computer, pystac_client, json

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
BBOX_48RWU = [105.0, 30.2, 106.2, 31.1]

cat = pystac_client.Client.open(STAC_URL)

for month_name, (start, end) in [
    ("6月", ("2025-06-01", "2025-06-30")),
    ("7月", ("2025-07-01", "2025-07-31")),
    ("8月", ("2025-08-01", "2025-08-31")),
]:
    print(f"\n=== {month_name} ===")
    search = cat.search(
        collections=["sentinel-2-l2a"],
        bbox=BBOX_48RWU,
        datetime=f"{start}/{end}",
        max_items=20,
    )
    candidates = []
    for item in search.items():
        item_id = item.id
        if "T48RWU" not in item_id:
            continue
        cloud = item.properties.get("eo:cloud_cover", 100)
        dt = item.properties['datetime'][:10]
        candidates.append((cloud, dt, item_id))
    
    for cloud, dt, item_id in sorted(candidates):
        print(f"  {dt}  cloud={cloud:.1f}%  {item_id}")
