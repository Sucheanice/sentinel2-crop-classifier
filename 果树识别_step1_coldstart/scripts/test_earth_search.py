import pystac_client, requests

# Test Earth Search
cat = pystac_client.Client.open("https://earth-search.aws.element84.com/v1")
items = list(cat.search(
    collections=["sentinel-2-l2a"],
    bbox=[105, 30.2, 106.2, 31.1],
    datetime="2025-06-01/2025-07-01",
    max_items=10,
).items())

rwr = [i for i in items if "T48RWU" in i.id]
print(f"Total items: {len(items)}, T48RWU: {len(rwr)}")
for i in rwr[:5]:
    cloud = i.properties.get("eo:cloud_cover", "?")
    print(f"  {i.id} cloud={cloud}%")
    b04 = i.assets.get("B04", {})
    if b04:
        href = b04.href
        print(f"    B04: {href[:80]}...")
        # Quick check if COG is accessible
        try:
            r = requests.head(href, timeout=10)
            print(f"    HEAD: {r.status_code}")
        except Exception as e:
            print(f"    HEAD failed: {e}")
