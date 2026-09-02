# -*- coding: utf-8 -*-
"""获取叙永 v4 数据的确切下载链接 (S2 5/20 + S1 9/29 + S1 11/21)。"""
from pystac_client import Client

STAC_URL = "https://earth-search.aws.element84.com/v1"
XUYONG_BBOX = [105.20, 27.60, 105.80, 27.95]
cat = Client.open(STAC_URL)

def s3_to_https(url, bucket):
    return url.replace(f"s3://{bucket}", f"https://{bucket}.s3.amazonaws.com", 1)

# ===== S2 5/20 =====
print("=" * 70)
print("叙永 5月 S2 (推荐 S2B_48RWR_20250520)")
print("=" * 70)
s2 = cat.search(collections=["sentinel-2-l2a"], bbox=XUYONG_BBOX,
                datetime="2025-05-20/2025-05-21", max_items=10)
for it in s2.items():
    if "48RWR" not in it.id:
        continue
    print(f"item id: {it.id}")
    print(f"  datetime: {it.properties.get('datetime')}")
    print(f"  cloud: {it.properties.get('eo:cloud_cover')}%")
    print(f"  s2:scene_id(ESA): {it.properties.get('s2:product_uri', 'N/A')}")
    print("  assets:")
    for k, a in it.assets.items():
        if k in ["B02","B03","B04","B05","B06","B07","B08","B8A","B11","B12","SCL"]:
            print(f"    {k}: {a.href}")
    break

# ===== S1 9/29 =====
print()
print("=" * 70)
print("叙永 9月 S1 (推荐 20250929T110005)")
print("=" * 70)
s1 = cat.search(collections=["sentinel-1-grd"], bbox=XUYONG_BBOX,
                datetime="2025-09-29/2025-09-30", max_items=20)
for it in s1.items():
    pols = it.properties.get("sar:polarizations", [])
    if "VV" not in pols or "VH" not in pols:
        continue
    print(f"item id: {it.id}")
    print(f"  orbit: {it.properties.get('sat:orbit_state')}")
    print(f"  shape: {it.properties.get('proj:shape')}")
    for k in ["vv", "vh"]:
        print(f"    {k}: {s3_to_https(it.assets[k].href, 'sentinel-s1-l1c')}")

# ===== S1 11/21 =====
print()
print("=" * 70)
print("叙永 11月 S1 (推荐 20241121T110013)")
print("=" * 70)
s1b = cat.search(collections=["sentinel-1-grd"], bbox=XUYONG_BBOX,
                 datetime="2024-11-21/2024-11-22", max_items=20)
for it in s1b.items():
    pols = it.properties.get("sar:polarizations", [])
    if "VV" not in pols or "VH" not in pols:
        continue
    print(f"item id: {it.id}")
    print(f"  orbit: {it.properties.get('sat:orbit_state')}")
    print(f"  shape: {it.properties.get('proj:shape')}")
    for k in ["vv", "vh"]:
        print(f"    {k}: {s3_to_https(it.assets[k].href, 'sentinel-s1-l1c')}")
