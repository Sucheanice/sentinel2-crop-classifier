# -*- coding: utf-8 -*-
"""诊断 + 重新搜索：找到真正覆盖叙永(27.69-27.80N)的 S1 GRD 场景。
之前 search_xuyong_v4_data.py 搜到的 9/29、11/21 场景 footprint 实际是 28.62-30.54N（遂宁轨道），
并不覆盖叙永。这里打印每个 item 的真实 footprint bbox，并筛出真正覆盖叙永的场景。
"""
from pystac_client import Client

STAC_URL = "https://earth-search.aws.element84.com/v1"
XUYONG_BBOX = [105.20, 27.60, 105.80, 27.95]

cat = Client.open(STAC_URL)

def s3_to_https(url):
    return url.replace("s3://sentinel-s1-l1c", "https://sentinel-s1-l1c.s3.amazonaws.com", 1)

def covers_xuyong(bbox):
    """判断 bbox 是否覆盖叙永核心区 27.69-27.80"""
    if bbox is None:
        return False
    lon0, lat0, lon1, lat1 = bbox
    # 纬度区间有交集且经度区间有交集
    return lat0 <= 27.80 and lat1 >= 27.69 and lon0 <= 105.67 and lon1 >= 105.37

for label, daterange in [("9月", "2025-09-15/2025-10-10"), ("11月", "2024-11-15/2024-12-05")]:
    print("=" * 70)
    print(f"[S1 GRD {label}] {daterange}  搜索bbox={XUYONG_BBOX}")
    print("=" * 70)
    s1 = cat.search(
        collections=["sentinel-1-grd"],
        bbox=XUYONG_BBOX,
        datetime=daterange,
        max_items=100,
    )
    items = list(s1.items())
    print(f"  返回 {len(items)} 个 item\n")
    for it in items:
        pols = it.properties.get("sar:polarizations", [])
        if "VV" not in pols or "VH" not in pols:
            continue
        bbox = it.bbox
        geom_bbox = None
        if it.geometry:
            import shapely.geometry
            try:
                geom_bbox = list(shapely.geometry.shape(it.geometry).bounds)
            except Exception:
                geom_bbox = None
        ok = covers_xuyong(geom_bbox or bbox)
        print(f"  {it.id}")
        print(f"    轨道={it.properties.get('sat:orbit_state')} rel_orbit={it.properties.get('sat:relative_orbit')}")
        print(f"    item.bbox={[round(x,3) for x in bbox] if bbox else None}")
        print(f"    geom.bounds={[round(x,3) for x in geom_bbox] if geom_bbox else None}")
        print(f"    {'==> 覆盖叙永 ✓' if ok else '    不覆盖叙永 ✗'}")
        for k in ["vv", "vh"]:
            if k in it.assets:
                print(f"      {k}: {s3_to_https(it.assets[k].href)}")
        print()
