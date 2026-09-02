# -*- coding: utf-8 -*-
"""搜索 5月 S2 完整场景 ID (叙永 48RWR + 遂宁 48RWU)，输出 S3 下载 URL。"""
from pystac_client import Client

cat = Client.open("https://earth-search.aws.element84.com/v1")

S2_BANDS = ["B02","B03","B04","B05","B06","B07","B08","B8A","B11","B12"]
RES10 = ["B02","B03","B04","B08"]
RES20 = ["B05","B06","B07","B8A","B11","B12"]

def s3_urls(scene_id):
    # scene_id 形如 S2B_MSIL2A_20250520T033539_R061_T48RWR_20250520T060217
    p = scene_id.split("_")
    tile = p[4]           # T48RWR (compact scene_id: S2B_MSIL2A_20250520T033539_R061_T48RWR_20250520T060217)
    utm = tile[1:3]       # 48
    lat = tile[3]         # R
    sq = tile[4:6]        # WR
    date = p[2][:8]       # 20250520
    y = date[:4]; m = str(int(date[4:6])); d = str(int(date[6:8]))
    base = f"https://sentinel-s2-l2a.s3.eu-central-1.amazonaws.com/tiles/{utm}/{lat}/{sq}/{y}/{m}/{d}/0"
    urls = {}
    for b in S2_BANDS:
        sub = "R10m" if b in RES10 else "R20m"
        urls[b] = f"{base}/{sub}/{b}.jp2"
    return urls

for label, bbox, tile in [
    ("叙永 48RWR", [105.20, 27.60, 105.80, 27.95], "48RWR"),
    ("遂宁 48RWU", [105.25, 30.25, 105.85, 30.75], "48RWU"),
]:
    print("=" * 70)
    print(f"[{label}] 5月 S2 L2A  2025-05-01/2025-06-10")
    print("=" * 70)
    s2 = cat.search(
        collections=["sentinel-2-l2a"], bbox=bbox,
        datetime="2025-05-01/2025-06-10",
        query={"eo:cloud_cover": {"lt": 30}}, max_items=20,
    )
    items = sorted(s2.items(), key=lambda i: i.properties.get("eo:cloud_cover", 100))
    for it in items[:4]:
        cc = it.properties.get("eo:cloud_cover")
        dt = it.properties.get("datetime")
        uri = it.properties.get("s2:product_uri") or it.properties.get("product_uri") or ""
        scene_id = uri.replace(".SAFE", "").replace("_N0511", "").replace("_N0500", "")
        print(f"\n  {it.id}  云量={cc:.1f}% 时间={dt}")
        print(f"  ESA: {uri}")
        print(f"  场景ID(compact): {scene_id}")
        if scene_id and "_MSIL2A_" in scene_id:
            urls = s3_urls(scene_id)
            for b in S2_BANDS:
                print(f"    {b}: {urls[b]}")
