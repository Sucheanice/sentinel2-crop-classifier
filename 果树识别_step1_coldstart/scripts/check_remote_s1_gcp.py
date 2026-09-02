# -*- coding: utf-8 -*-
"""远程读取 S1 GRD measurement tiff 的 GCP（只读 GeoTIFF 头，不下载整景），
确认候选场景的真实覆盖范围（因为 element84 STAC bbox 与实测 GCP 有 ~1.5° 偏差）。

候选场景（rel_orbit 55 已确认=遂宁带 28.62-30.54，需验证 rel_orbit 128 是否覆盖叙永 27.70-27.78）:
  9月:  9/22, 10/4
  11月: 11/26
"""
import rasterio
import sys

XUYONG = dict(lat0=27.69, lat1=27.80, lon0=105.37, lon1=105.67)

URLS = {
    "9月 9/22  rel128": "https://sentinel-s1-l1c.s3.amazonaws.com/GRD/2025/9/22/IW/DV/S1A_IW_GRDH_1SDV_20250922T110805_20250922T110830_061100_079DAF_2E34/measurement/iw-vv.tiff",
    "9月 10/4  rel128": "https://sentinel-s1-l1c.s3.amazonaws.com/GRD/2025/10/4/IW/DV/S1A_IW_GRDH_1SDV_20251004T110805_20251004T110830_061275_07A4BF_6582/measurement/iw-vv.tiff",
    "11月 11/26 rel128": "https://sentinel-s1-l1c.s3.amazonaws.com/GRD/2024/11/26/IW/DV/S1A_IW_GRDH_1SDV_20241126T110813_20241126T110838_056725_06F61E_4191/measurement/iw-vv.tiff",
    "11月 12/3  rel55": "https://sentinel-s1-l1c.s3.amazonaws.com/GRD/2024/12/3/IW/DV/S1A_IW_GRDH_1SDV_20241203T110012_20241203T110037_056827_06FA1F_9DC9/measurement/iw-vv.tiff",
}

for name, url in URLS.items():
    print("=" * 70)
    print(name)
    print("=" * 70)
    try:
        with rasterio.open(url) as src:
            gcps, gcp_crs = src.gcps
            lats = [g.y for g in gcps]
            lons = [g.x for g in gcps]
            rows = [g.row for g in gcps]
            cols = [g.col for g in gcps]
            print(f"  size: {src.width}x{src.height}")
            print(f"  n_gcp={len(gcps)} crs={gcp_crs}")
            print(f"  lon: {min(lons):.3f} ~ {max(lons):.3f}")
            print(f"  lat: {min(lats):.3f} ~ {max(lats):.3f}")
            print(f"  row: {min(rows):.0f} ~ {max(rows):.0f}  col: {min(cols):.0f} ~ {max(cols):.0f}")
            # 判断是否覆盖叙永
            covers = min(lats) <= XUYONG["lat1"] and max(lats) >= XUYONG["lat0"] and \
                     min(lons) <= XUYONG["lon1"] and max(lons) >= XUYONG["lon0"]
            print(f"  ==> 覆盖叙永: {'是 ✓' if covers else '否 ✗'}")
    except Exception as e:
        print(f"  [ERROR] {type(e).__name__}: {e}")
