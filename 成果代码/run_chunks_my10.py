#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""分块跑 my10（与 baozhucun_v10_fix2 同一套代码/权重/参数）

为什么要分块：整幅跑时每个实例都常驻整窗掩膜，按面积外推约 1.2 TB，超过服务器 755 GB 内存，
聚合阶段必然 OOM。分块后每块实例量降到 1/N。

分块口径：
  * 块核心边界对齐 ALIGN=1008 px（= LCM(粗窗口步长 504, 细窗口步长 336)），
    保证块内窗口网格与原图窗口网格完全对齐，不产生错位伪影；
  * 每块额外留 HALO=1008 px 重叠带（也是 1008 的整数倍，维持对齐），
    让跨块地块在"质心所属块"里能被完整看到；
  * 合并时按「质心落在本块核心范围」归属，再按重叠比例去重（保留 score 高的）。

可断点续跑：已存在结果的块会跳过。
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window, bounds as win_bounds
from rasterio.transform import Affine
from shapely.geometry import shape, mapping, box
from shapely.strtree import STRtree

IMG = "/data3/0918my/my/510725zt/my10.tif"
ROOT = Path("/data3/0918my/my/510725zt/result_my10_v10_fix2_chunks")
TIF, OUT = ROOT / "tif", ROOT / "out"
SERVICE = Path("/data4/sam3_project/sam3_train/Code/sam3-main/sam3_service")
CODE = Path("/data4/sam3_project/sam3_train/Code/sam3-main")
PY = "/root/anaconda3/envs/sam3/bin/python"
CKPT = "/data4/sam3_project/sam3_train/data/runs/finetune_v3/checkpoints/checkpoint_25.pt"
GPUS = "3"
ALIGN = 1008
HALO = 1008
NX, NY = 4, 2
MIN_FREE_GB = 200

PARAMS = ["--prompt", "farmland", "--gpus", GPUS, "--checkpoint", CKPT,
          "--geometric-refine", "0", "--confidence-threshold", "0.3",
          "--dark-p10-threshold", "25", "--ndvi-threshold", "0",
          "--simplify-tolerance", "1.0", "--mask-smooth-sigma", "1.0", "--max-vertices", "200",
          "--min-area-m2", "67", "--merge-touching", "1", "--merge-fill-hole-m2", "0.0",
          "--stretch", "0", "--slic-points", "1", "--slic-min-brightness", "0"]


def log(msg):
    print("[%s] %s" % (time.strftime("%m-%d %H:%M:%S"), msg), flush=True)


def free_gb():
    try:
        out = subprocess.run(["free", "-g"], capture_output=True, text=True).stdout.splitlines()[1].split()
        return int(out[6])
    except Exception:
        return 999


def build_grid(W, H):
    cw = int(np.ceil(np.ceil(W / NX) / ALIGN)) * ALIGN
    ch = int(np.ceil(np.ceil(H / NY) / ALIGN)) * ALIGN
    return [(x, y, min(cw, W - x), min(ch, H - y))
            for y in range(0, H, ch) for x in range(0, W, cw)]


def run_chunks(grid, W, H):
    TIF.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    for i, (x, y, w, h) in enumerate(grid):
        ctif = TIF / ("chunk_%02d.tif" % i)
        cshp = OUT / ("chunk_%02d.shp" % i)
        x0, y0 = max(0, x - HALO), max(0, y - HALO)
        x1, y1 = min(W, x + w + HALO), min(H, y + h + HALO)
        if not ctif.exists():
            subprocess.run(["gdal_translate", "-q", "-srcwin", str(x0), str(y0), str(x1 - x0), str(y1 - y0),
                            "-co", "TILED=YES", "-co", "COMPRESS=DEFLATE", "-co", "BIGTIFF=IF_SAFER",
                            IMG, str(ctif)], check=True)
        if cshp.exists():
            log("[%d/%d] 已有结果，跳过" % (i + 1, len(grid)))
            continue
        for _ in range(120):
            fg = free_gb()
            if fg >= MIN_FREE_GB:
                break
            log("内存可用仅 %d GB，等 5 分钟再继续…" % fg)
            time.sleep(300)
        log("[%d/%d] 推理开始（可用内存 %d GB）" % (i + 1, len(grid), free_gb()))
        env = dict(os.environ, SAM3_ROOT=str(CODE), PYTHONPATH="%s:%s" % (SERVICE, CODE))
        cmd = [PY, "geo_infer_v10_multigpu.py", "--input", str(ctif),
               "--output", str(cshp.with_suffix(""))] + PARAMS
        t0 = time.time()
        with open(ROOT / ("chunk_%02d.log" % i), "w") as lf:
            subprocess.run(cmd, cwd=str(SERVICE), env=env, stdout=lf, stderr=subprocess.STDOUT)
        if not cshp.exists():
            log("[%d/%d] 失败，中止（见 chunk_%02d.log）" % (i + 1, len(grid), i))
            return False
        import fiona
        n = sum(1 for _ in fiona.open(cshp))
        log("[%d/%d] 完成，用时 %.0f s，产出 %d 个多边形" % (i + 1, len(grid), time.time() - t0, n))
    return True


def merge(grid, W, H):
    import fiona
    where = rasterio.open(IMG)
    tr, crs = where.transform, where.crs
    where.close()
    kept, dropped = [], 0
    for i, (x, y, w, h) in enumerate(grid):
        p = OUT / ("chunk_%02d.shp" % i)
        if not p.exists():
            log("缺块 %d 结果，跳过" % i)
            continue
        b = win_bounds(Window(x, y, w, h), tr)          # 本块核心范围 (left, bottom, right, top)
        core = box(*b)                                   # rasterio 各版本返回元组/namedtuple，统一按序解包
        eps = 1e-6
        n_in = 0
        with fiona.open(p) as src:
            for f in src:
                g = shape(f["geometry"])
                c = g.centroid
                if core.buffer(eps).contains(c):
                    kept.append((g, f["properties"], i))
                    n_in += 1
        log("[块 %d] 保留 %d 个（质心在本块核心内）" % (i, n_in))

    # 跨块重叠去重：按 score 降序处理，若自身 >50% 面积压在"已保留且不更低分"的邻居上则丢弃
    n = len(kept)
    geoms = [t[0] for t in kept]
    tree = STRtree(geoms)
    order = sorted(range(n), key=lambda i: -float(kept[i][1].get("score") or 0))
    alive = [True] * n
    dropped = 0
    for i in order:
        if not alive[i]:
            continue
        gi = geoms[i]
        if gi.area <= 0:
            alive[i] = False
            dropped += 1
            continue
        for j in tree.query(gi):
            if j == i or not alive[j]:
                continue
            if gi.intersection(geoms[j]).area / gi.area > 0.5:
                alive[i] = False
                dropped += 1
                break
    final = [kept[i] for i in order if alive[i]]
    log("跨块去重：丢弃 %d 个重叠块，保留 %d 个" % (dropped, len(final)))

    out = ROOT / "my10_v10_fix2_merged.shp"
    schema = {"geometry": "Polygon",
              "properties": {"id": "int", "area_px": "int", "area_m2": "float",
                             "score": "float", "class": "str", "src_chunk": "int"}}
    px_area = abs(tr.a * tr.e)
    with fiona.open(out, "w", driver="ESRI Shapefile", schema=schema, crs=crs.to_wkt(), encoding="utf-8") as dst:
        for k, (g, props, ci) in enumerate(final, 1):
            dst.write({"geometry": mapping(g), "properties": {
                "id": k, "area_px": int(round(g.area / px_area)),
                "area_m2": round(g.area, 2), "score": float(props.get("score") or 0),
                "class": "farmland", "src_chunk": int(ci)}})
    total_m2 = sum(g.area for g, _, _ in final)
    log("合并完成: %s" % out)
    log("  %d 个多边形 / %.1f 亩 / 最大 %.1f 亩" % (
        len(final), total_m2 * 0.0015, max((g.area for g, _, _ in final), default=0) * 0.0015))
    return True


def main():
    with rasterio.open(IMG) as s:
        W, H = s.width, s.height
    grid = build_grid(W, H)
    (ROOT).mkdir(parents=True, exist_ok=True)
    (ROOT / "grid.json").write_text(json.dumps(
        {"W": W, "H": H, "align": ALIGN, "halo": HALO,
         "chunks": [{"i": i, "x": x, "y": y, "w": w, "h": h} for i, (x, y, w, h) in enumerate(grid)]}))
    log("影像 %dx%d -> %d 块；块核心 %dx%d，重叠带 %d px" % (W, H, len(grid), grid[0][2], grid[0][3], HALO))
    ok = run_chunks(grid, W, H)
    if not ok:
        log("有块失败，未进入合并")
        return 1
    return 0 if merge(grid, W, H) else 1


if __name__ == "__main__":
    sys.exit(main())
