"""测试 PC Azure URL 是否免 SAS token 可用"""
import pystac_client, requests

cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
search = cat.search(
    collections=["sentinel-2-l2a"],
    bbox=[105.0, 30.2, 106.2, 31.1],
    datetime="2025-07-26",
    max_items=10,
)
for it in search.items():
    if "T48RWU" in it.id:
        href = it.assets["B02"].href
        print(f"URL: {href[:120]}...")
        try:
            r = requests.get(href, timeout=15, stream=True)
            print(f"Status: {r.status_code}")
            print(f"Content-Type: {r.headers.get('content-type', '?')}")
            print(f"Content-Length: {r.headers.get('content-length', '?')}")
            if r.status_code == 200:
                # 测试下载速度
                chunk = r.iter_content(chunk_size=10*1024*1024)
                data = next(chunk)
                print(f"Downloaded: {len(data)/1024:.0f} KB of chunk")
        except Exception as e:
            print(f"Error: {e}")
        break
