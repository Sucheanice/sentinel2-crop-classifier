# -*- coding: utf-8 -*-
"""搜索遂宁 2024-11 / 2025-09 的 Sentinel-1 GRD (ascending VV+VH) 场景，输出下载链接"""
from pystac_client import Client

STAC_URL = "https://earth-search.aws.element84.com/v1"
BBOX = [105.0, 29.7, 106.2, 30.8]  # 遂宁

TARGETS = [
    ("2024-11", "2024-11-01/2024-11-30"),
    ("2025-09", "2025-09-01/2025-09-30"),
]


def check_coverage(item, bbox):
    geom = item.geometry
    if geom is None:
        return 0.0
    coords = geom["coordinates"][0]
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    olon = min(bbox[2], max(lons)) - max(bbox[0], min(lons))
    olat = min(bbox[3], max(lats)) - max(bbox[1], min(lats))
    if olon <= 0 or olat <= 0:
        return 0.0
    return olon * olat / ((bbox[2]-bbox[0])*(bbox[3]-bbox[1]))


def s3_to_https(href):
    if href.startswith("s3://"):
        href = href.replace("s3://", "https://", 1)
        href = href.replace("sentinel-s1-l1c/", "sentinel-s1-l1c.s3.eu-central-1.amazonaws.com/")
    return href


cat = Client.open(STAC_URL)

for label, dtrange in TARGETS:
    print("=" * 70)
    print(f"[{label}] 搜索遂宁 S1 GRD...")
    search = cat.search(
        collections=["sentinel-1-grd"],
        bbox=BBOX,
        datetime=dtrange,
        max_items=100,
    )
    found = []
    for item in search.items():
        pols = item.properties.get("sar:polarizations", [])
        orbit = item.properties.get("sat:orbit_state", "")
        if "VV" not in pols or "VH" not in pols or orbit != "ascending":
            continue
        if "vv" not in item.assets or "vh" not in item.assets:
            continue
        ratio = check_coverage(item, BBOX)
        if ratio < 0.1:
            continue
        date = item.datetime.strftime("%Y-%m-%d")
        found.append((ratio, date, item.id, item))

    found.sort(key=lambda x: -x[0])
    print(f"  找到 {len(found)} 个可用场景 (按覆盖排序):\n")
    for ratio, date, sid, item in found[:5]:
        print(f"  日期={date}  覆盖={ratio:.1%}")
        print(f"    场景ID: {sid}")
        print(f"    VV: {s3_to_https(item.assets['vv'].href)}")
        print(f"    VH: {s3_to_https(item.assets['vh'].href)}")
        print()
    if not found:
        print("  未找到场景")
    print()
