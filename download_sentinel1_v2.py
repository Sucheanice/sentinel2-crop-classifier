# -*- coding: utf-8 -*-
"""download_sentinel1_v2.py — 修复版 S1 下载: 通过 STAC geometry 预检实际覆盖范围。

问题: 原版只靠 bbox 搜索, 但 S1 的 STAC bbox 是轨道全长 BBOX, 实际子带可能不覆盖目标。
修复: 下载前检查 STAC item geometry, 确保实际覆盖目标区域。

用法:
  python download_sentinel1_v2.py                    # 下载遂宁 S1 (48RWU)
  python download_sentinel1_v2.py --area jiangyou    # 下载江油 S1
"""
import os, sys, time, argparse
import requests
from pystac_client import Client
import datetime

STAC_URL = "https://earth-search.aws.element84.com/v1"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 区域配置
AREAS = {
    "suining": {
        "name": "遂宁",
        "bbox": [105.0, 29.7, 106.2, 30.8],
        "out_dir": os.path.join(BASE_DIR, "小春_s1_48RWU"),
        "s2_dates": {
            "2024-12-11": "2024-12-15",
            "2025-01-20": "2025-01-27",
            "2025-03-26": "2025-03-28",
            "2025-05-05": "2025-05-03",
        },
    },
    "jiangyou": {
        "name": "江油",
        "bbox": [104.90, 31.90, 105.20, 32.20],
        "out_dir": os.path.join(BASE_DIR, "江油_s1"),
        "s2_dates": {
            "2024-12-01": None,  # S1 search needed
            "2025-01-05": None,
            "2025-03-26": None,
            "2025-05-20": None,
        },
    },
}

# S2→S1 日期偏移 (如果未指定)
S1_DATE_OFFSETS = list(range(-10, 11))


def check_geometry_coverage(item, target_bbox, min_overlap_deg=0.3):
    """检查 STAC item 的实际 footprint 是否充分覆盖目标 bbox。

    返回 (overlap_ratio, max_overlap_deg) 或 (0, 0)。
    """
    geom = item.geometry
    if geom is None:
        # 无 geometry, 回退到 bbox 检查
        item_bbox = item.bbox  # [min_lon, min_lat, max_lon, max_lat]
        if item_bbox is None:
            return 0.0, 0.0
        overlap_lon = min(target_bbox[2], item_bbox[2]) - max(target_bbox[0], item_bbox[0])
        overlap_lat = min(target_bbox[3], item_bbox[3]) - max(target_bbox[1], item_bbox[1])
        if overlap_lon <= 0 or overlap_lat <= 0:
            return 0.0, 0.0
        target_area = (target_bbox[2] - target_bbox[0]) * (target_bbox[3] - target_bbox[1])
        overlap_area = overlap_lon * overlap_lat
        return overlap_area / target_area, overlap_lon

    # 从 geometry 多边形提取实际边界
    coords = geom['coordinates'][0]
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    g_min_lon, g_max_lon = min(lons), max(lons)
    g_min_lat, g_max_lat = min(lats), max(lats)

    # 计算与目标 bbox 的重叠
    overlap_lon = min(target_bbox[2], g_max_lon) - max(target_bbox[0], g_min_lon)
    overlap_lat = min(target_bbox[3], g_max_lat) - max(target_bbox[1], g_min_lat)

    if overlap_lon <= 0 or overlap_lat <= 0:
        return 0.0, 0.0

    target_area = (target_bbox[2] - target_bbox[0]) * (target_bbox[3] - target_bbox[1])
    overlap_area = overlap_lon * overlap_lat
    ratio = overlap_area / target_area

    return ratio, overlap_lon


def download_band(item, band_key, out_path, session, max_retries=5):
    if os.path.exists(out_path):
        # 检查文件大小是否合理 (> 10MB)
        if os.path.getsize(out_path) > 10 * 1024 * 1024:
            print(f"    {band_key}.tif exists, skip")
            return True
        else:
            print(f"    {band_key}.tif exists but too small, re-downloading")
            os.remove(out_path)

    raw_href = item.assets[band_key].href
    if raw_href.startswith("s3://"):
        href = raw_href.replace("s3://", "https://", 1)
        href = href.replace("sentinel-s1-l1c/", "sentinel-s1-l1c.s3.eu-central-1.amazonaws.com/")
    else:
        href = raw_href
    
    # 获取远程文件大小
    remote_size = 0
    try:
        head_resp = session.head(href, timeout=30)
        remote_size = int(head_resp.headers.get('Content-Length', 0))
        supports_range = head_resp.headers.get('Accept-Ranges', '') == 'bytes'
    except:
        supports_range = False
    
    tmp_path = out_path + ".tmp"
    downloaded = os.path.getsize(tmp_path) if os.path.exists(tmp_path) else 0
    
    print(f"    downloading {band_key}.tif ({remote_size/1024/1024:.0f} MB total)")
    
    for attempt in range(1, max_retries + 1):
        try:
            headers = {}
            if downloaded > 0 and supports_range:
                headers['Range'] = f'bytes={downloaded}-'
                print(f"    attempt {attempt}/{max_retries}: resuming from {downloaded/1024/1024:.0f} MB...")
            else:
                downloaded = 0
                print(f"    attempt {attempt}/{max_retries}: starting...")
            
            resp = session.get(href, headers=headers, stream=True, timeout=(30, 600))
            resp.raise_for_status()
            
            mode = 'ab' if downloaded > 0 else 'wb'
            with open(tmp_path, mode) as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
            
            # 检查完整性
            total_size = os.path.getsize(tmp_path)
            if remote_size > 0 and total_size < remote_size * 0.95:
                raise Exception(f"Incomplete: {total_size}/{remote_size} bytes")
            
            os.rename(tmp_path, out_path)
            size_mb = downloaded / (1024 * 1024)
            print(f"    {band_key}.tif  {size_mb:.1f} MB  OK")
            return True
        except Exception as e:
            print(f"    {band_key}.tif attempt {attempt}/{max_retries} FAILED: {e}")
            if os.path.exists(out_path + ".tmp"):
                os.remove(out_path + ".tmp")
            if attempt < max_retries:
                wait = min(30, 5 * attempt)
                print(f"    waiting {wait}s...")
                time.sleep(wait)
    return False


def search_s1_for_date(catalog, target_date, target_bbox, area_name=""):
    """搜索覆盖目标区域的最佳 S1 ascending VV+VH 场景。

    返回 (item, actual_date) 或 (None, None)。
    """
    d = datetime.datetime.strptime(target_date, "%Y-%m-%d")

    for delta in S1_DATE_OFFSETS:
        search_date = d + datetime.timedelta(days=delta)
        start = search_date.strftime("%Y-%m-%d")
        end = (search_date + datetime.timedelta(days=1)).strftime("%Y-%m-%d")

        search = catalog.search(
            collections=["sentinel-1-grd"],
            bbox=target_bbox,
            datetime=f"{start}/{end}",
            max_items=20,
        )
        items = list(search.items())
        if not items:
            continue

        for it in items:
            pols = it.properties.get("sar:polarizations", [])
            orbit = it.properties.get("sat:orbit_state", "")
            if "VV" not in pols or "VH" not in pols or orbit != "ascending":
                continue
            if "vv" not in it.assets or "vh" not in it.assets:
                continue

            # 关键修复: 检查实际覆盖
            ratio, overlap_lon = check_geometry_coverage(it, target_bbox)
            if ratio < 0.1 or overlap_lon < 0.3:
                gstr = f"geometry覆盖={ratio:.2f}, 经度重叠={overlap_lon:.2f}°"
                # 不打印太多, 但记录被跳过的场景
                continue

            # 通过覆盖检查!
            item_date = it.datetime.strftime("%Y-%m-%d")
            offset = abs((search_date - d).days)
            print(f"    Found: {it.id}")
            print(f"    Date: {item_date} (offset={offset}d), Coverage: {ratio:.1%}, lon_overlap={overlap_lon:.2f}°")
            return it, item_date

    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--area", default="suining", choices=["suining", "jiangyou"],
                        help="目标区域")
    args = parser.parse_args()

    cfg = AREAS[args.area]
    OUT_DIR = cfg["out_dir"]
    bbox = cfg["bbox"]

    t_total = time.time()
    print("=" * 70)
    print(f"下载 {cfg['name']} Sentinel-1 GRD VV+VH (v2: 覆盖预检)")
    print(f"BBOX: {bbox}")
    print("=" * 70)

    session = requests.Session()
    catalog = Client.open(STAC_URL)
    os.makedirs(OUT_DIR, exist_ok=True)

    downloaded = 0
    skipped_exists = 0
    failed = 0

    for s2_date, preset_s1_date in cfg["s2_dates"].items():
        target_s1 = preset_s1_date if preset_s1_date else s2_date
        print(f"\n=== S2: {s2_date} -> Target S1: {target_s1} ===")

        item, actual_date = search_s1_for_date(catalog, target_s1, bbox, cfg['name'])
        if item is None:
            print(f"  ERROR: No covering S1 scene found!")
            failed += 1
            continue

        scene_name = f"{actual_date}_S1_asc"
        scene_dir = os.path.join(OUT_DIR, scene_name)

        # 检查是否已存在且完整
        vv_path = os.path.join(scene_dir, "vv.tif")
        vh_path = os.path.join(scene_dir, "vh.tif")
        if (os.path.exists(vv_path) and os.path.getsize(vv_path) > 100 * 1024 * 1024 and
                os.path.exists(vh_path) and os.path.getsize(vh_path) > 100 * 1024 * 1024):
            print(f"  All bands exist, skip")
            skipped_exists += 1
            continue

        os.makedirs(scene_dir, exist_ok=True)

        all_ok = True
        for band_key in ["vv", "vh"]:
            out_path = os.path.join(scene_dir, f"{band_key}.tif")
            if not download_band(item, band_key, out_path, session):
                all_ok = False

        if all_ok:
            downloaded += 1
            print(f"  {scene_name}: COMPLETE")
        else:
            failed += 1
            print(f"  {scene_name}: PARTIAL FAILURE")

    elapsed = time.time() - t_total
    print("\n" + "=" * 70)
    print(f"下载汇总: 成功={downloaded}, 跳过={skipped_exists}, 失败={failed}")
    print(f"总耗时: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"输出: {OUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
