# -*- coding: utf-8 -*-
"""搜索叙永 5月S2 + 9/11月S1 场景，输出下载链接。"""
from pystac_client import Client

STAC_URL = "https://earth-search.aws.element84.com/v1"
# 叙永 bbox (带buffer)
XUYONG_BBOX = [105.20, 27.60, 105.80, 27.95]

cat = Client.open(STAC_URL)

def s3_to_https(url):
    return url.replace("s3://sentinel-s1-l1c", "https://sentinel-s1-l1c.s3.amazonaws.com", 1)

# ============ 1. 叙永 5月 S2 ============
print("=" * 70)
print("[叙永 5月 S2 L2A] 2025-05-10 ~ 2025-06-05")
print("=" * 70)
s2 = cat.search(
    collections=["sentinel-2-l2a"],
    bbox=XUYONG_BBOX,
    datetime="2025-05-10/2025-06-05",
    query={"eo:cloud_cover": {"lt": 30}},
    max_items=20,
)
items = sorted(s2.items(), key=lambda i: i.properties.get("eo:cloud_cover", 100))
for it in items[:8]:
    cc = it.properties.get("eo:cloud_cover")
    dt = it.properties.get("datetime")
    print(f"  {it.id}  云量={cc:.1f}%  时间={dt}")
    # 输出资产
    for k in ["B02", "B03", "B04", "B08", "visual"]:
        if k in it.assets:
            print(f"      {k}: {it.assets[k].href[:100]}")

# ============ 2. 叙永 9月 S1 ============
print()
print("=" * 70)
print("[叙永 9月 S1 GRD] 2025-09-20 ~ 2025-10-05")
print("=" * 70)
s1_9 = cat.search(
    collections=["sentinel-1-grd"],
    bbox=XUYONG_BBOX,
    datetime="2025-09-20/2025-10-05",
    max_items=100,
)
for it in s1_9.items():
    pols = it.properties.get("sar:polarizations", [])
    orbit = it.properties.get("sat:orbit_state", "")
    if "VV" not in pols or "VH" not in pols:
        continue
    shape = it.properties.get("proj:shape")
    print(f"  {it.id}  轨道={orbit} shape={shape}")
    for k in ["vv", "vh"]:
        if k in it.assets:
            print(f"      {k}: {s3_to_https(it.assets[k].href)}")

# ============ 3. 叙永 11月 S1 ============
print()
print("=" * 70)
print("[叙永 11月 S1 GRD] 2024-11-15 ~ 2024-11-30")
print("=" * 70)
s1_11 = cat.search(
    collections=["sentinel-1-grd"],
    bbox=XUYONG_BBOX,
    datetime="2024-11-15/2024-11-30",
    max_items=100,
)
for it in s1_11.items():
    pols = it.properties.get("sar:polarizations", [])
    orbit = it.properties.get("sat:orbit_state", "")
    if "VV" not in pols or "VH" not in pols:
        continue
    shape = it.properties.get("proj:shape")
    print(f"  {it.id}  轨道={orbit} shape={shape}")
    for k in ["vv", "vh"]:
        if k in it.assets:
            print(f"      {k}: {s3_to_https(it.assets[k].href)}")
