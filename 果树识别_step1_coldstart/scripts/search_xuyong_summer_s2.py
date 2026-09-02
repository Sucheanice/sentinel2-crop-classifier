# -*- coding: utf-8 -*-
"""搜索叙永 48RWR 夏季 S2 (需要匹配遂宁的3个夏季窗口)"""
import pystac_client
import time

cat = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1"
)

# 叙永范围 (48RWR)
XUYONG_BBOX = [105.37, 27.69, 105.67, 27.80]

# 搜索2025年6-8月, 每个窗口找最低云量
windows = [
    ("2025-06-15/2025-06-30", "early_jun"),
    ("2025-07-15/2025-07-31", "mid_jul"),
    ("2025-08-15/2025-08-31", "late_aug"),
]

print("=" * 60)
print("叙永 48RWR 夏季 S2 搜索")
print("=" * 60)

for daterange, label in windows:
    print(f"\n[{label}] {daterange}")
    search = cat.search(
        collections=["sentinel-2-l2a"],
        bbox=XUYONG_BBOX,
        datetime=daterange,
        query={"eo:cloud_cover": {"lt": 30}},
        max_items=10,
    )
    items = list(search.items())
    # 筛选 48RWR
    rwr_items = [it for it in items if "T48RWR" in it.id]
    if not rwr_items:
        print(f"  无 48RWR 场景 (共 {len(items)} 个其他tile)")
        continue

    # 按云量排序
    rwr_items.sort(key=lambda it: it.properties.get("eo:cloud_cover", 100))
    for it in rwr_items[:3]:
        cloud = it.properties.get("eo:cloud_cover", "?")
        print(f"  {it.id} | cloud={cloud}% | {it.datetime}")
    time.sleep(0.5)

print("\n完成!")
