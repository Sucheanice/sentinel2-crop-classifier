# -*- coding: utf-8 -*-
"""搜索遂宁 48RWU 在 2024-11 / 2025-05 / 2025-09 的可用 S2 场景"""
import pystac_client

cat = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1"
)

# 遂宁 bbox (覆盖 48RWU)
BBOX = [105.0, 30.2, 106.2, 31.1]

TARGETS = [
    ("2024-11", "2024-11-01/2024-11-30"),
    ("2025-05", "2025-05-01/2025-05-31"),
    ("2025-09", "2025-09-01/2025-09-30"),
]

for label, dtrange in TARGETS:
    print("=" * 70)
    print(f"[{label}] 搜索 48RWU 场景...")
    search = cat.search(
        collections=["sentinel-2-l2a"],
        bbox=BBOX,
        datetime=dtrange,
        max_items=100,
    )
    n = 0
    for item in search.items():
        if "T48RWU" not in item.id:
            continue
        n += 1
        cloud = item.properties.get("eo:cloud_cover", 0)
        date = item.properties.get("datetime", "")[:10]
        platform = item.properties.get("platform", "")
        print(f"  {item.id}  云量={cloud:.1f}%  日期={date}  平台={platform}")
    if n == 0:
        print("  未找到场景")
    print()
