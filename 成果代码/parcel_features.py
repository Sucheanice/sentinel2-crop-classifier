#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""parcel_features.py — 给 SAM 输出的地块 shp 追加「几何(A) + RGB 纹理(B)」特征值。

设计
----
独立后处理，**不改动 geo_infer_v10 推理代码**：读推理产出的 shp + 对应影像，
按每个多边形统计特征后写出新 shp。好处是可反复重算、结果与基线一一对应。

A 组（几何，纯 shapely，零影像 I/O）
    perim_m  周长(m)            n_vert  顶点数            vdens  顶点密度(个/100m)
    cmpt     紧致度 4πA/P²      sqr     方正度 A/最小外接矩形
    aspect   长宽比              mrr_az  长边方位角(°,0=北顺时针)
    cvx      面积/凸包面积       holes   孔洞数             edge_mu 平均边长(m)

B 组（RGB 统计/纹理，按对象在原始影像上统计）
    lum_mu   亮度均值(归一化0-255)   lum_sd 亮度标准差      color_sd 三通道标准差均值
    lum_p10  亮度P10(深色门同口径)   darkf  深色像素占比(归一化亮度<25)
    glcm_con GLCM对比度          glcm_hom GLCM同质性       glcm_ent GLCM熵(32级,4方向)
    edge_den 边缘密度(Sobel>20占比)  grain 颗粒感(局部std均值)

注意：B 组是**光学外观**特征，只在同景同季内可比，跨影像/跨时相阈值会漂移。

用法
----
  python parcel_features.py --shp chunk_00.shp --tif chunk_00.tif --out chunk_00_feat.shp
  python parcel_features.py --shp in.shp --tif in.tif --out out.shp --no-rgb
"""
import argparse
import math
import sys

import numpy as np
import fiona
import rasterio
from affine import Affine
from rasterio.features import geometry_mask
from rasterio.windows import Window, from_bounds
from shapely.geometry import mapping, shape

sys.stdout.reconfigure(encoding="utf-8")

GEOM_FIELDS = [("perim_m", "float"), ("n_vert", "int"), ("vdens", "float"),
               ("cmpt", "float"), ("sqr", "float"), ("aspect", "float"),
               ("mrr_az", "float"), ("cvx", "float"), ("holes", "int"),
               ("edge_mu", "float")]
RGB_FIELDS = [("lum_mu", "float"), ("lum_sd", "float"), ("color_sd", "float"),
              ("lum_p10", "float"), ("darkf", "float"),
              ("glcm_con", "float"), ("glcm_hom", "float"), ("glcm_ent", "float"),
              ("edge_den", "float"), ("grain", "float"),
              ("bgrad", "float"), ("bgrad_pct", "float")]

DARK_LEVEL = 25.0      # 与生产 --dark-p10-threshold 25 同口径：归一化亮度 < 该值视为深色
EDGE_SOBEL = 20.0      # Sobel 幅值阈值
GLCM_LEVELS = 32


# --------------------------------------------------------------------------- #
# A 组：几何
# --------------------------------------------------------------------------- #
def geom_feats(poly):
    d = {"perim_m": 0.0, "n_vert": 0, "vdens": 0.0, "cmpt": 0.0, "sqr": 0.0,
         "aspect": 0.0, "mrr_az": 0.0, "cvx": 0.0, "holes": 0, "edge_mu": 0.0}
    try:
        a = float(poly.area)
        p = float(poly.length)
        geoms = [poly] if poly.geom_type == "Polygon" else \
            [g for g in getattr(poly, "geoms", []) if g.geom_type == "Polygon"]
        nv = sum(max(0, len(g.exterior.coords) - 1) for g in geoms)
        nh = sum(len(g.interiors) for g in geoms)
        d["perim_m"] = round(p, 2)
        d["n_vert"] = int(nv)
        d["vdens"] = round(nv / (p / 100.0), 3) if p > 0 else 0.0
        d["cmpt"] = round(4 * math.pi * a / (p * p), 4) if p > 0 else 0.0
        d["holes"] = int(nh)
        d["edge_mu"] = round(p / nv, 2) if nv > 0 else 0.0
        try:
            hull = poly.convex_hull.area
            d["cvx"] = round(min(a / hull, 1.0), 4) if hull > 0 else 0.0
        except Exception:
            pass
        try:
            mrr = poly.minimum_rotated_rectangle
            ma = float(mrr.area)
            d["sqr"] = round(min(a / ma, 1.0), 4) if ma > 0 else 0.0
            c = list(mrr.exterior.coords)
            if len(c) >= 5:
                s0 = math.hypot(c[1][0] - c[0][0], c[1][1] - c[0][1])
                s1 = math.hypot(c[2][0] - c[1][0], c[2][1] - c[1][1])
                lo, hi = min(s0, s1), max(s0, s1)
                d["aspect"] = round(hi / lo, 3) if lo > 0 else 0.0
                dx, dy = (c[1][0] - c[0][0], c[1][1] - c[0][1]) if s0 >= s1 else \
                         (c[2][0] - c[1][0], c[2][1] - c[1][1])
                d["mrr_az"] = round(math.degrees(math.atan2(dx, dy)) % 180.0, 1)
        except Exception:
            pass
    except Exception:
        pass
    return d


# --------------------------------------------------------------------------- #
# B 组：RGB 统计 / 纹理
# --------------------------------------------------------------------------- #
def rgb_feats(poly, src, lo, hi, max_side=512):
    d = {k: 0.0 for k, _ in RGB_FIELDS}
    try:
        win = from_bounds(*poly.bounds, transform=src.transform)
        win = win.intersection(Window(0, 0, src.width, src.height))
        if win.width < 4 or win.height < 4:
            return d
        scale = min(1.0, float(max_side) / max(win.width, win.height))
        sh = max(1, int(round(win.height * scale)))
        sw = max(1, int(round(win.width * scale)))
        data = src.read(indexes=[1, 2, 3], window=win,
                        out_shape=(3, sh, sw)).astype(np.float32)
        wtrans = src.window_transform(win)
        m = geometry_mask([mapping(poly)], out_shape=(sh, sw),
                          transform=wtrans * Affine.scale(win.width / sw, win.height / sh),
                          invert=True)
        if int(m.sum()) < 16:
            return d
        if data.shape[0] >= 3:
            lum = 0.299 * data[0] + 0.587 * data[1] + 0.114 * data[2]
        else:
            lum = data[0]
        lum255 = np.clip((lum - lo) * 255.0 / max(hi - lo, 1e-6), 0, 255)
        v = lum255[m]
        d["lum_mu"] = round(float(v.mean()), 2)
        d["lum_sd"] = round(float(v.std()), 2)
        d["color_sd"] = round(float(np.mean([data[i][m].std()
                                             for i in range(min(3, data.shape[0]))])), 2)
        d["lum_p10"] = round(float(np.percentile(v, 10)), 2)
        d["darkf"] = round(float((v < DARK_LEVEL).mean()), 4)

        from scipy.ndimage import binary_erosion, sobel, uniform_filter
        gm = np.hypot(sobel(lum), sobel(lum, 1))
        d["edge_den"] = round(float((gm[m] > EDGE_SOBEL).mean()), 4)
        # bgrad：边界带(掩膜内缘1px)上的平均梯度 —— 越大说明边界越压在影像真实边缘上
        band = m & ~binary_erosion(m)
        if band.sum() >= 8 and gm.size > 0:
            bv = float(gm[band].mean())
            d["bgrad"] = round(bv, 2)
            d["bgrad_pct"] = round(float((gm < bv).mean() * 100), 1)
        else:
            bv = float(gm[m].mean())
            d["bgrad"] = round(bv, 2)
            d["bgrad_pct"] = round(float((gm < bv).mean() * 100), 1)
        mu = uniform_filter(lum, 3)
        sd = np.sqrt(np.maximum(uniform_filter(lum * lum, 3) - mu * mu, 0.0))
        d["grain"] = round(float(sd[m].mean()), 3)

        # GLCM：裁到对象 bbox 内，减少背景稀释
        try:
            rows = np.where(m.any(axis=1))[0]
            cols = np.where(m.any(axis=0))[0]
            q = lum255[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
            if q.size >= 64:
                from skimage.feature import graycomatrix, graycoprops
                q8 = np.clip(q / 256.0 * GLCM_LEVELS, 0, GLCM_LEVELS - 1).astype(np.uint8)
                g = graycomatrix(q8, distances=[1],
                                 angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                                 levels=GLCM_LEVELS, symmetric=True, normed=True)
                d["glcm_con"] = round(float(graycoprops(g, "contrast").mean()), 3)
                d["glcm_hom"] = round(float(graycoprops(g, "homogeneity").mean()), 4)
                p = g.mean(axis=(2, 3)).ravel()
                p = p / max(p.sum(), 1e-9)
                nz = p[p > 0]
                d["glcm_ent"] = round(float(-(nz * np.log2(nz)).sum()), 3)
        except Exception:
            pass
    except Exception:
        pass
    return d


def norm_range(src, fixed_lo=0.0, fixed_hi=0.0):
    """归一化区间：固定值优先，否则取 1/16 概览的 2/98 百分位（与深色过滤一致）。"""
    if fixed_lo > 0 and fixed_hi > fixed_lo:
        return float(fixed_lo), float(fixed_hi), "固定值"
    ov = src.read(indexes=[1, 2, 3],
                  out_shape=(3, max(1, src.height // 16), max(1, src.width // 16))).astype(np.float32)
    lum = 0.299 * ov[0] + 0.587 * ov[1] + 0.114 * ov[2] if ov.shape[0] >= 3 else ov[0]
    lo, hi = [float(x) for x in np.percentile(lum, (2, 98))]
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi, "概览2/98"


def main():
    ap = argparse.ArgumentParser(description="给地块 shp 追加几何(A)+RGB纹理(B)特征")
    ap.add_argument("--shp", required=True)
    ap.add_argument("--tif", required=True, help="统计特征用的原始影像")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-rgb", action="store_true", help="只算 A 组几何特征")
    ap.add_argument("--norm-lo", type=float, default=0.0)
    ap.add_argument("--norm-hi", type=float, default=0.0)
    ap.add_argument("--id-offset", type=int, default=0,
                    help="id 重编号的基数：输出 id = id-offset + 1..N（多块实验时保证全局唯一）")
    ap.add_argument("--chunk", type=int, default=0, help="写入 chunk 字段（块号）")
    args = ap.parse_args()

    with fiona.open(args.shp, encoding="UTF-8") as src_shp:
        schema_in = src_shp.schema
        crs = src_shp.crs
        recs = list(src_shp)
    print(f"[输入] {args.shp}: {len(recs)} 个要素, 原字段={list(schema_in['properties'])}")

    props_out = dict(schema_in["properties"])
    for k, t in GEOM_FIELDS:
        props_out[k] = t
    if not args.no_rgb:
        for k, t in RGB_FIELDS:
            props_out[k] = t
    if args.chunk:
        props_out["chunk"] = "int"
    schema_out = {"geometry": schema_in["geometry"], "properties": props_out}

    lo = hi = 0.0
    with rasterio.open(args.tif) as src:
        if not args.no_rgb:
            lo, hi, how = norm_range(src, args.norm_lo, args.norm_hi)
            print(f"[归一化] lo={lo:.2f} hi={hi:.2f} ({how})")
        n = 0
        with fiona.open(args.out, "w", driver="ESRI Shapefile", crs=crs,
                        schema=schema_out, encoding="UTF-8") as sink:
            for rec in recs:
                geom = shape(rec["geometry"])
                pr = {k: rec["properties"].get(k) for k in schema_in["properties"]}
                pr.update(geom_feats(geom))
                if not args.no_rgb:
                    pr.update(rgb_feats(geom, src, lo, hi))
                n += 1
                pr["id"] = args.id_offset + n          # 全局唯一编号
                if args.chunk:
                    pr["chunk"] = int(args.chunk)
                sink.write({"geometry": mapping(geom), "properties": pr})
                if n % 200 == 0:
                    print(f"  {n}/{len(recs)}", flush=True)
    print(f"[完成] {n} 个要素 → {args.out}")
    print(f"[id] {args.id_offset + 1} ~ {args.id_offset + n}"
          + (f"  chunk={args.chunk}" if args.chunk else ""))
    print(f"[新增字段] {[k for k, _ in GEOM_FIELDS] + ([] if args.no_rgb else [k for k, _ in RGB_FIELDS])}")


if __name__ == "__main__":
    main()
