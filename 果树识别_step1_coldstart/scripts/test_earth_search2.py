import pystac_client

cat = pystac_client.Client.open("https://earth-search.aws.element84.com/v1")

for dt in ["2025-06-15/2025-06-30", "2025-07-15/2025-07-31"]:
    items = list(cat.search(
        collections=["sentinel-2-l2a"],
        bbox=[104, 29, 107, 32],
        datetime=dt,
        max_items=30,
    ).items())
    rwr = [i for i in items if "T48RWU" in i.id]
    print(f"{dt}: total={len(items)}, T48RWU={len(rwr)}")
    for i in rwr[:3]:
        cloud = i.properties.get("eo:cloud_cover", "?")
        print(f"  {i.id} cloud={cloud}%")
