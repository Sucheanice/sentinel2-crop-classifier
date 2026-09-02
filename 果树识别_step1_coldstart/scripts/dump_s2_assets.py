# -*- coding: utf-8 -*-
from pystac_client import Client

STAC_URL = "https://earth-search.aws.element84.com/v1"
XUYONG_BBOX = [105.20, 27.60, 105.80, 27.95]
cat = Client.open(STAC_URL)

s2 = cat.search(collections=["sentinel-2-l2a"], bbox=XUYONG_BBOX,
                datetime="2025-05-20/2025-05-21", max_items=10)
for it in s2.items():
    if "48RWR" not in it.id:
        continue
    print(f"item id: {it.id}")
    print(f"ESA scene: {it.properties.get('s2:product_uri')}")
    print(f"资产键: {list(it.assets.keys())}")
    for k, a in it.assets.items():
        print(f"  {k}: {a.href}")
    break
