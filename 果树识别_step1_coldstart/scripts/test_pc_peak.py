import planetary_computer, pystac_client, time, requests

cat = pystac_client.Client.open('https://planetarycomputer.microsoft.com/api/stac/v1')
items = list(cat.search(
    collections=['sentinel-2-l2a'],
    bbox=[105.0, 30.2, 106.2, 31.1],
    datetime='2025-06-26',
    max_items=20,
).items())
rwr = [it for it in items if 'T48RWU' in it.id]
print(f'Found {len(rwr)} scenes')
it = rwr[0]
print(f'ID: {it.id}')
print(f'Cloud: {it.properties.get("eo:cloud_cover")}%')

t0 = time.time()
signed = planetary_computer.sign(it)
print(f'Sign: {time.time()-t0:.1f}s')

href = signed.assets['B04'].href
print(f'B04 URL: {href[:80]}...')
r = requests.head(href, timeout=30)
print(f'HEAD: {r.status_code}')
