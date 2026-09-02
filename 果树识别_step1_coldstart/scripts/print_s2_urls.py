# -*- coding: utf-8 -*-
"""输出遂宁 48RWU 夏季 S2 下载链接"""
import planetary_computer, pystac_client

cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")

# mid_summer 2025-07-26 (cloud=2.2%, S2B)
items = list(cat.search(
    collections=["sentinel-2-l2a"],
    bbox=[105, 30.2, 106.2, 31.1],
    datetime="2025-07-26",
    max_items=20,
).items())

it = [i for i in items if "T48RWU" in i.id][0]
print(f"场景: {it.id}")
print(f"云量: {it.properties.get('eo:cloud_cover')}%")
print(f"日期: 2025-07-26")
print()

signed = planetary_computer.sign(it)
bands = ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12"]

print("=" * 80)
print("Planetary Computer COG 下载链接 (带SAS token, 浏览器可直接打开)")
print("文件格式: Cloud Optimized GeoTIFF, 约15-20MB/波段")
print("=" * 80)
for b in bands:
    href = signed.assets[b].href
    print(f"\n{b}:")
    print(href)

print()
print()
print("=" * 80)
print("AWS S3 JP2 下载链接 (公开直连, 国内下载快)")
print("文件格式: JPEG2000, 约4-15MB/波段")
print("注意: rasterio解码可能有90%零值问题, 需用专业JP2解码器")
print("=" * 80)

scene_id = "S2B_MSIL2A_20250726T032519_R018_T48RWU_20250726T055341"
parts = scene_id.split("_")
utm_zone = parts[4][1:3]
lat_band = parts[4][3]
square = parts[4][4:6]
date_part = parts[2][:8]
year = date_part[:4]
month = str(int(date_part[4:6]))
day = str(int(date_part[6:8]))

base = f"https://sentinel-s2-l2a.s3.eu-central-1.amazonaws.com/tiles/{utm_zone}/{lat_band}/{square}/{year}/{month}/{day}/0"
for b in bands:
    res = "R10m" if b in ["B02", "B03", "B04", "B08"] else "R20m"
    print(f"\n{b}:")
    print(f"{base}/{res}/{b}.jp2")
