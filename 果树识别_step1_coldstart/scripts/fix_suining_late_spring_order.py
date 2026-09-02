# -*- coding: utf-8 -*-
"""修复遂宁 late_spring 波段顺序:
原顺序 [B02,B03,B04,B08,B05,B06,B07,B8A,B11,B12] (B08/B05 位置错)
→ 正确 [B02,B03,B04,B05,B06,B07,B08,B8A,B11,B12] (与叙永及 S2_BANDS 一致)
"""
import os
import time
from pathlib import Path

os.environ["PROJ_LIB"] = r"C:\Users\lenovo\AppData\Roaming\Python\Python313\site-packages\rasterio\proj_data"

import numpy as np
import rasterio

SRC = Path(r"E:\工作相关\2026年\0624 待测试数据\小春_s2_48RWU_summer\late_spring_10m_10band.tif")
TMP = SRC.with_suffix(".tmp.tif")

CORRECT = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
# 源文件当前描述对应的正确顺序下的源索引 (见上面注释)
PERM = [0, 1, 2, 4, 5, 6, 3, 7, 8, 9]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    log("修复遂宁 late_spring 波段顺序...")
    with rasterio.open(SRC) as src:
        cur = [src.descriptions[i] for i in range(src.count)]
        log(f"当前: {cur}")
        profile = src.profile.copy()
        profile.update(count=len(CORRECT), tiled=True, blockxsize=256, blockysize=256, compress="deflate")
        with rasterio.open(TMP, "w", **profile) as dst:
            for i in range(len(CORRECT)):
                data = src.read(PERM[i] + 1).astype("float32")
                dst.write(data, i + 1)
                dst.set_band_description(i + 1, CORRECT[i])
                log(f"  写波段 {i + 1}: {CORRECT[i]} (源 {PERM[i] + 1})")

    # 原子替换
    SRC.unlink()
    TMP.rename(SRC)
    log(f"完成 -> {SRC.name}")


if __name__ == "__main__":
    main()
