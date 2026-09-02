"""测试 planetary_computer.sign() 是否能成功"""
import time, sys
import planetary_computer
import pystac_client

cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
search = cat.search(
    collections=["sentinel-2-l2a"],
    bbox=[105.0, 30.2, 106.2, 31.1],
    datetime="2025-07-26",
    max_items=10,
)
for it in search.items():
    if "T48RWU" in it.id:
        print(f"Found: {it.id}", flush=True)
        t0 = time.time()
        try:
            signed = planetary_computer.sign(it)
            elapsed = time.time() - t0
            print(f"Sign succeeded in {elapsed:.0f}s", flush=True)
            # 测试 B02 URL
            import requests
            r = requests.get(signed.assets["B02"].href, timeout=10, stream=True)
            print(f"Download test: status={r.status_code}", flush=True)
        except Exception as e:
            elapsed = time.time() - t0
            print(f"Sign failed after {elapsed:.0f}s: {e}", flush=True)
        break
