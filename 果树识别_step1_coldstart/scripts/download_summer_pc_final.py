# -*- coding: utf-8 -*-
"""从 Planetary Computer 下载遂宁 48RWU 夏季全瓦片 - 美国VPN节点版"""
import os, sys, time, json
from pathlib import Path
import planetary_computer
import pystac_client
import requests

WORK_DIR = Path(r"E:\工作相关\2026年\0624 待测试数据")
OUT_DIR = WORK_DIR / "小春_s2_48RWU_summer" / "2025-06-26_48RWU"
OUT_DIR.mkdir(parents=True, exist_ok=True)

S2_BANDS = ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12"]

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def main():
    t0 = time.time()
    log("=" * 60)
    log("遂宁 48RWU 夏季全瓦片下载 (PC + 美国VPN)")
    log(f"目标目录: {OUT_DIR}")
    log("=" * 60)

    # ---- Step 1: 搜索 ----
    log("搜索 2025-06-26 T48RWU...")
    cat = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    search = cat.search(
        collections=["sentinel-2-l2a"],
        bbox=[105.0, 30.2, 106.2, 31.1],
        datetime="2025-06-26",
        max_items=20,
    )

    item = None
    for it in search.items():
        if "T48RWU" in it.id:
            item = it
            break

    if item is None:
        log("[FATAL] 未找到 48RWU 场景!")
        return

    cloud = item.properties.get("eo:cloud_cover", "?")
    log(f"  场景: {item.id}")
    log(f"  云量: {cloud}%")

    # ---- Step 2: 逐个波段下载 ----
    total_mb = 0
    ok_count = 0

    for band in S2_BANDS:
        out_path = OUT_DIR / f"{band}.tif"
        tmp_path = str(out_path) + ".tmp"

        if out_path.exists():
            sz_mb = out_path.stat().st_size / 1e6
            log(f"  [{band}] 已存在 ({sz_mb:.0f}MB), 跳过")
            total_mb += sz_mb
            ok_count += 1
            continue

        if band not in item.assets:
            log(f"  [{band}] 不在assets中, 跳过")
            continue

        href = item.assets[band].href
        log(f"  [{band}] 下载中...")

        max_retries = 5
        success = False
        for attempt in range(max_retries):
            try:
                # 清除残留的tmp文件
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

                r = requests.get(href, timeout=(60, 600), stream=True)
                r.raise_for_status()

                with open(tmp_path, "wb") as f:
                    downloaded = 0
                    for chunk in r.iter_content(chunk_size=262144):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)

                # 验证文件大小
                file_sz = os.path.getsize(tmp_path) / 1e6
                if file_sz < 0.1:  # 至少100KB
                    raise Exception(f"文件太小 ({file_sz:.2f}MB)")

                os.rename(tmp_path, str(out_path))
                total_mb += file_sz
                ok_count += 1
                log(f"  [{band}] {file_sz:.1f}MB OK")
                success = True
                break

            except Exception as e:
                wait_s = min(2 ** attempt, 30)
                if attempt < max_retries - 1:
                    log(f"  [{band}] 重试 {attempt+1}/{max_retries} ({wait_s}s后): {str(e)[:50]}")
                    time.sleep(wait_s)
                else:
                    log(f"  [{band}] 失败: {str(e)[:60]}")

    elapsed = time.time() - t0
    log(f"\n{'=' * 60}")
    log(f"完成: {elapsed/60:.1f}分")
    log(f"成功: {ok_count}/{len(S2_BANDS)} 波段, 共 {total_mb:.0f}MB")
    log(f"输出: {OUT_DIR}")

    # 验证
    if ok_count == len(S2_BANDS):
        log("\n全部10个波段下载完成! 可以重建特征立方体了。")
    else:
        log(f"\n警告: {len(S2_BANDS)-ok_count} 个波段未下载")


if __name__ == "__main__":
    main()
