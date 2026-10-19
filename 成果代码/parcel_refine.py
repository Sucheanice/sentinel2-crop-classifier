#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""parcel_refine.py — 边界精化：把 SAM 出的地块边界"吸附"到影像真实地物边缘上。

原理
----
SAM 的掩膜在边界处常带缺角、掐腰、毛刺（尤其在低对比度的田埂/林缘上）。
本模块对每个图斑做一次**局部 GrabCut**：
  确定前景 = 原多边形向内腐蚀 `--erode-m` 米（这部分肯定是地块）
  确定背景 = 原多边形向外扩 `--dilate-m` 米之外（这部分肯定不是）
  未知区   = 中间那圈 → 由 GrabCut 按 RGB 颜色分布决定归属
于是边界会被拉到**颜色反差最强的地方**（= 真实地物边缘），而不是停在 SAM 的掩膜噪声上。

安全阀（任一不满足就退回原图斑）
  面积变化超过 `--skip-frac`；结果为空；结果碎成多块且主块占比 < `--keep-ratio`。

用法
----
  python parcel_refine.py --shp in.shp --tif img.tif --out out.shp \
      --erode-m 1.5 --dilate-m 2.5 --iters 3 --simplify-m 0.6
"""
import argparse
import sys

import cv2
import geopandas as gpd
import numpy as np
import rasterio
from affine import Affine
from rasterio.features import geometry_mask, shapes
from rasterio.windows import from_bounds
from scipy import ndimage
from shapely.geometry import shape

sys.stdout.reconfigure(encoding="utf-8")
T = "EPSG:4523"


def refine_one(img, m_poly, erode_px, dilate_px, iters, smooth_px):
    """img: HxWx3 uint8 局部裁剪; m_poly: 同尺寸 bool 多边形掩膜。返回精化后 bool 掩膜或 None"""
    h, w = m_poly.shape
    if h < 6 or w < 6:
        return None
    fg = ndimage.binary_erosion(m_poly, iterations=erode_px)
    bg = ~ndimage.binary_dilation(m_poly, iterations=dilate_px)
    if fg.sum() < 4 or bg.sum() < 4:
        return None
    gm = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    gm[bg] = cv2.GC_BGD
    gm[m_poly] = cv2.GC_PR_FGD
    gm[fg] = cv2.GC_FGD
    try:
        cv2.setRNGSeed(0)
        bgm = np.zeros((1, 65), np.float64)
        fgm = np.zeros((1, 65), np.float64)
        cv2.grabCut(img, gm, None, bgm, fgm, iters, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return None
    out = (gm == cv2.GC_FGD) | (gm == cv2.GC_PR_FGD)
    # 只保留与原图斑重叠最多的那一块
    lab, n = ndimage.label(out)
    if n == 0:
        return None
    best, best_ov = 0, -1
    for k in range(1, n + 1):
        mm = lab == k
        ov = int((mm & m_poly).sum())
        if ov > best_ov:
            best_ov, best = ov, k
    out = lab == best
    # 去毛刺：高斯平滑后重取阈值（这一步很关键，否则矢量化的锯齿会全部带出来）
    if smooth_px > 0:
        out = ndimage.gaussian_filter(out.astype(np.float32), smooth_px) > 0.5
    out = ndimage.binary_closing(out, structure=np.ones((3, 3)), iterations=1)
    out = ndimage.binary_fill_holes(out)
    return out


def run(shp, tif, out, erode_m, dilate_m, iters, min_area_m2, skip_frac,
        keep_ratio, simplify_m, smooth_m, roundtrip_m, id_offset=0, chunk=0):
    g = gpd.read_file(shp).to_crs(T).reset_index(drop=True)
    src = rasterio.open(tif)
    tr = src.transform
    px = abs(tr.a)
    img = src.read(indexes=[1, 2, 3]).astype(np.uint8).transpose(1, 2, 0)
    H, W = img.shape[:2]
    erode_px = max(1, int(round(erode_m / px)))
    dilate_px = max(2, int(round(dilate_m / px)))
    smooth_px = smooth_m / px
    margin_px = dilate_px + max(2, int(round(2.0 / px)))
    core_m = erode_m

    new_geoms, flags = [], []
    for geom in g.geometry:
        if geom.area < min_area_m2 or geom.geom_type != "Polygon":
            new_geoms.append(geom); flags.append("skip_small" if geom.geom_type == "Polygon" else "skip_multi")
            continue
        win = from_bounds(*geom.bounds, transform=tr)
        r0 = max(0, int(win.row_off) - margin_px)
        c0 = max(0, int(win.col_off) - margin_px)
        r1 = min(H, int(win.row_off) + int(win.height) + margin_px)
        c1 = min(W, int(win.col_off) + int(win.width) + margin_px)
        if r1 - r0 < 8 or c1 - c0 < 8:
            new_geoms.append(geom); flags.append("skip_edge"); continue
        crop = np.ascontiguousarray(img[r0:r1, c0:c1])
        wt = tr * Affine.translation(c0, r0)
        m_poly = geometry_mask([geom], out_shape=crop.shape[:2], transform=wt, invert=True)
        if m_poly.sum() < 8:
            new_geoms.append(geom); flags.append("skip_mask"); continue
        ref = refine_one(crop, m_poly, erode_px, dilate_px, iters, smooth_px)
        if ref is None:
            new_geoms.append(geom); flags.append("revert_gc"); continue
        try:
            polys = [shape(s) for s, v in shapes(ref.astype(np.uint8), mask=ref, transform=wt)
                     if v == 1]
            if not polys:
                new_geoms.append(geom); flags.append("revert_empty"); continue
            polys.sort(key=lambda p: p.area, reverse=True)
            main = polys[0]
            if main.area / max(sum(p.area for p in polys), 1e-9) < keep_ratio:
                new_geoms.append(geom); flags.append("revert_split"); continue
            main = main.buffer(0)
            if main.is_empty or main.geom_type != "Polygon":
                new_geoms.append(geom); flags.append("revert_type"); continue
            if not (1 - skip_frac) <= main.area / geom.area <= (1 + skip_frac):
                new_geoms.append(geom); flags.append("revert_area"); continue
            # 去掉栅格锯齿：先 buffer 往返磨圆，再 Douglas-Peucker 减点
            if roundtrip_m > 0:
                main = main.buffer(roundtrip_m, join_style=1,
                                   mitre_limit=1.0).buffer(-roundtrip_m, join_style=1,
                                                           mitre_limit=1.0).buffer(0)
                if main.is_empty or main.geom_type != "Polygon":
                    new_geoms.append(geom); flags.append("revert_rt"); continue
            if simplify_m > 0:
                main = main.simplify(simplify_m)
            # 安全阀：精化后必须仍包含原图斑的"内核"，否则等于切掉了地块
            core = geom.buffer(-core_m)
            if core.area > 0 and main.intersection(core).area / core.area < 0.90:
                new_geoms.append(geom); flags.append("revert_core"); continue
            if main.is_empty or main.geom_type != "Polygon" or main.area <= 0:
                new_geoms.append(geom); flags.append("revert_simp"); continue
            new_geoms.append(main); flags.append("refined")
        except Exception:
            new_geoms.append(geom); flags.append("revert_err")

    out_g = g.copy()
    out_g["geometry"] = new_geoms
    out_g["refine"] = flags
    out_g["orig_m2"] = g.geometry.area.round(2)
    out_g["area_m2"] = out_g.geometry.area.round(2)
    if id_offset:
        out_g["id"] = id_offset + np.arange(1, len(out_g) + 1)
    if chunk:
        out_g["chunk"] = int(chunk)
    out_g.to_file(out, driver="ESRI Shapefile", encoding="UTF-8")
    s = out_g["refine"].value_counts()
    print(f"[精化] {len(out_g)} 块 → 实际精化 {int(s.get('refined',0))}, "
          f"回退/跳过 {len(out_g)-int(s.get('refined',0))}")
    for k, v in s.items():
        print(f"    {k}: {v}")
    a0, a1 = g.geometry.area.sum(), out_g.geometry.area.sum()
    print(f"[面积] {a0*0.0015:.1f} → {a1*0.0015:.1f} 亩 ({(a1/a0-1)*100:+.1f}%) → {out}")
    return out_g


def main():
    ap = argparse.ArgumentParser(description="边界精化：GrabCut 吸附到影像边缘")
    ap.add_argument("--shp", required=True)
    ap.add_argument("--tif", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--erode-m", type=float, default=1.5, help="确定前景内缩(米)")
    ap.add_argument("--dilate-m", type=float, default=2.5, help="确定背景外扩(米)")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--min-area-m2", type=float, default=400, help="小于此面积不精化")
    ap.add_argument("--skip-frac", type=float, default=0.35, help="面积变化超过则回退")
    ap.add_argument("--keep-ratio", type=float, default=0.8, help="主块占比下限")
    ap.add_argument("--simplify-m", type=float, default=2.0, help="Douglas-Peucker 容差(米)")
    ap.add_argument("--smooth-m", type=float, default=1.0, help="掩膜高斯平滑尺度(米)，去栅格锯齿")
    ap.add_argument("--roundtrip-m", type=float, default=0.7, help="buffer 往返磨圆半径(米)")
    ap.add_argument("--id-offset", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=0)
    a = ap.parse_args()
    run(a.shp, a.tif, a.out, a.erode_m, a.dilate_m, a.iters, a.min_area_m2,
        a.skip_frac, a.keep_ratio, a.simplify_m, a.smooth_m, a.roundtrip_m,
        a.id_offset, a.chunk)


if __name__ == "__main__":
    main()
