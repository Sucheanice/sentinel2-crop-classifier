# -*- coding: utf-8 -*-
"""pixel_fruit_detect.py —— 逐像元水果识别（**只用自己的画库**），输出我们自己的水果面

与「地块打分」的区别：
  * 不再依赖交付地块，也不需要「同镇水稻/玉米对照库」（那是同事的前序数据）；
  * 判据 = 本库 98 维典型区间(P10~P90) + MAD 衰减 + **常绿物理约束**；
  * 逐像元打分 → 阈值 → 矢量化 → 输出我们自己的 polygon（shp + KML）。

用法：
    python pixel_fruit_detect.py --crop citrus --town 水晶镇 --out "…\水果面"
    python pixel_fruit_detect.py --crop pomelo --town 黎雅镇 --county 梓潼县 --out "…\水果面"
"""
import argparse
import os
import sys
import urllib.error
import urllib.request

import ee
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
import shapely.geometry as sgeom

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from citrus_pomelo_extract import build_stack, ee_init  # noqa: E402
from citrus_pomelo_score import SEASON, USE_IDX, build_fingerprint, fingerprints, score_one  # noqa: E402

ROOT = r"e:\工作相关\2026年\0624 待测试数据"
LIB = os.path.join(ROOT, r"成果数据\柑橘蜜柚识别\自净")
VILL = os.path.join(ROOT, r"训练与标注数据\0907村界等数据\绵阳\四川省村界\四川省村界\四川省_村界.shp")
MU = 0.0015          # m² -> 亩


def fp_image(stack, names):
    """540 波段 → 98 维逐像元指纹（与 citrus_pomelo_score.fingerprints 同口径）"""
    mm = {n: int(n[:2]) for n in names}
    out = []
    for p in USE_IDX:
        def dek(ms):
            return stack.select([f"{p}_{n}" for n in names if mm[n] in ms])

        mon = {m: dek([m]).reduce(ee.Reducer.mean()).rename(f"{p}_m{m}") for m in range(1, 13)}
        for s, ms in SEASON.items():
            out.append(ee.Image.cat([mon[m] for m in ms]).reduce(ee.Reducer.mean())
                       .rename(f"{s}_{p}"))
        out.append(ee.Image.cat([mon[7], mon[8]]).reduce(ee.Reducer.mean())
                   .subtract(ee.Image.cat([mon[12], mon[1], mon[2]]).reduce(ee.Reducer.mean()))
                   .rename(f"振幅_{p}"))
        out.append(dek([12]).reduce(ee.Reducer.stdDev()).rename(f"冬稳定_{p}"))
        out.append(dek([10, 11]).reduce(ee.Reducer.mean()).rename(f"成熟期_{p}"))
    return ee.Image.cat(out)


def score_image(fimg, tab, dims):
    """逐像元打分：区间内 1 分，区间外按 MAD 距离高斯衰减（同 score_one）"""
    scs = []
    for c in dims:
        lo, hi = float(tab.loc[c, "P10"]), float(tab.loc[c, "P90"])
        mad = max(float(tab.loc[c, "MAD"]), 1e-3)
        v = fimg.select(c)
        z = v.subtract(hi).max(0).subtract(v.subtract(lo).min(0)).divide(mad)
        scs.append(z.divide(1.5).pow(2).multiply(-0.5).exp().rename(c))
    # 注意：不能对「波段名各不相同」的 ImageCollection 求 mean（会变成 98 波段影像），
    # 必须先 cat 成 98 波段、再用 reducer 压成单波段。
    return ee.Image.cat(scs).reduce(ee.Reducer.mean())


def evergreen_mask(stack, names, thr):
    """全年 12 个月月度 NDVI 的最小值 ≥ thr（柑橘/蜜柚等常绿果树的物理约束）"""
    mm = {n: int(n[:2]) for n in names}
    mon = [stack.select([f"NDVI_{n}" for n in names if mm[n] == m])
           .reduce(ee.Reducer.mean()) for m in range(1, 13)]
    return ee.Image.cat(mon).reduce(ee.Reducer.min()).gte(thr)


def town_geom(town, county=None):
    g = gpd.read_file(VILL)
    s = g[g["乡镇级"].astype(str) == town]
    if county:
        s = s[s["区县级"].astype(str) == county]
    if s.empty:
        raise SystemExit(f"[错误] 村界库找不到乡镇：{county or ''}/{town}")
    return s.to_crs(4326).geometry.union_all(), len(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crop", choices=["citrus", "pomelo"], required=True)
    ap.add_argument("--town", required=True)
    ap.add_argument("--county", help="区县名（同名乡镇时用于消歧）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--threshold", type=float, default=0.60,
                    help="分值阈值（默认 0.60，与地块版「高」档同量级；先看打印的分布再调）")
    ap.add_argument("--min-mu", type=float, default=0.5, help="最小成图面积（亩，默认 0.5）")
    ap.add_argument("--evergreen-min", type=float, default=0.40,
                    help="全年最低月 NDVI 下限（常绿约束；0=关闭）")
    ap.add_argument("--scale", type=int, default=10)
    ap.add_argument("--mask-shp", help="只当空间掩膜用的地块 shp（如交付件）；只取其农地范围，不参与特征")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    # 1) 特征库（只来自我们画的 polygon）
    lib_p = os.path.join(LIB, f"来源产区_{a.crop}_{a.year}_自净.csv")
    if not os.path.exists(lib_p):
        raise SystemExit(f"[错误] 缺库: {lib_p}")
    tab, pos_fp, _ = build_fingerprint(lib_p, a.year)
    dims = list(pos_fp.columns)
    self_sc, _ = score_one(pos_fp, pos_fp, dims)
    print(f"[库] {os.path.basename(lib_p)} | {len(pos_fp)} 点 × {len(dims)} 维"
          f"（库内自评分中位 {np.nanmedian(self_sc):.3f}）")

    # 2) AOI
    geom, nv = town_geom(a.town, a.county)
    b = geom.bounds
    print(f"[AOI] {a.county or ''}{a.town}：{nv} 个村，bbox {b[0]:.4f},{b[1]:.4f},{b[2]:.4f},{b[3]:.4f}")

    ee_init()
    aoi = ee.Geometry(geom.__geo_interface__)

    # 3) 540 → 98 → 分值
    print("[GEE] 构建 36 旬 × 15 指数堆叠 …")
    stack, names = build_stack(aoi, a.year)
    print("[GEE] 派生 98 维指纹 …")
    fimg = fp_image(stack, names)
    print("[GEE] 逐像元打分 …")
    sc = score_image(fimg, tab, dims)
    if a.evergreen_min > 0:
        print(f"[GEE] 常绿约束：全年最低月 NDVI ≥ {a.evergreen_min}")
        sc = sc.updateMask(evergreen_mask(stack, names, a.evergreen_min))

    # 硬物候规则（阈值全部取自我们自己的库，不引入任何同事特征）
    amp_p90 = float(tab.loc["振幅_NDVI", "P90"])
    mat_lo, mat_hi = float(tab.loc["成熟期_NDVI", "P10"]), float(tab.loc["成熟期_NDVI", "P90"])
    print(f"[规则] 振幅_NDVI ≤ {amp_p90:.3f}；成熟期_NDVI ∈ [{mat_lo:.3f}, {mat_hi:.3f}]")
    rule = (fimg.select("振幅_NDVI").lte(amp_p90)
            .And(fimg.select("成熟期_NDVI").gte(mat_lo)
                 .And(fimg.select("成熟期_NDVI").lte(mat_hi))))
    sc = sc.updateMask(rule)
    sc = sc.clip(aoi)

    # 4) 下载分值栅格
    tif = os.path.join(a.out, f"{a.town}_{a.crop}_分值.tif")
    rect = ee.Geometry.Rectangle([b[0], b[1], b[2], b[3]])
    for scale in (a.scale, a.scale * 2, a.scale * 4):
        try:
            print(f"[下载] scale={scale} …")
            url = sc.getDownloadURL({"region": rect, "scale": scale, "crs": "EPSG:32648",
                                     "format": "GEO_TIFF"})
            urllib.request.urlretrieve(url, tif)
            print(f"        -> {tif}  {os.path.getsize(tif)/1024/1024:.2f} MB")
            break
        except urllib.error.HTTPError as e:
            print(f"        ! scale={scale} HTTP {e.code}: {e.read().decode('utf-8','ignore')[:400]}")
        except Exception as e:
            print(f"        ! scale={scale} 失败：{type(e).__name__}: {str(e)[:160]}")
    else:
        raise SystemExit("[错误] 三次下载均失败")

    # 5) 矢量化
    with rasterio.open(tif) as ds:
        band = ds.read(1)
        tr = ds.transform
        crs = ds.crs
        grid = (ds.height, ds.width)
    valid = np.isfinite(band) & (band > 0)
    if a.mask_shp:
        mg = gpd.read_file(a.mask_shp).to_crs(crs)
        if "AG_BZBM" in mg.columns:
            mg = mg[mg["AG_BZBM"].astype(str).isin(["A01", ""])]
        if len(mg):
            pm = rasterio.features.rasterize(
                [(gm, 1) for gm in mg.geometry if gm is not None and not gm.is_empty],
                out_shape=grid, transform=tr, fill=0, dtype="uint8")
            before = int(valid.sum())
            valid &= (pm == 1)
            print(f"[掩膜] 农地块 {len(mg)} 个：有效像元 {before} -> {int(valid.sum())}")
    if valid.any():
        q = np.nanpercentile(band[valid], [50, 90, 95, 99])
        print(f"[分值分布] 有效像元 {int(valid.sum())} | P50={q[0]:.3f} P90={q[1]:.3f} "
              f"P95={q[2]:.3f} P99={q[3]:.3f} max={band[valid].max():.3f}")
    thr = a.threshold
    m = valid & (band >= thr)
    print(f"[阈值] {thr:.3f} -> 命中 {int(m.sum())} 像元 "
          f"({m.sum()/max(valid.sum(),1)*100:.2f}% of 有效)")
    polys = []
    for shp, val in rasterio.features.shapes(band.astype("float32"), mask=m, transform=tr):
        pg = sgeom.shape(shp)
        if pg.area * MU >= a.min_mu:
            polys.append((pg, float(val)))
    if not polys:
        raise SystemExit("[错误] 无面积达标的图斑")
    g = gpd.GeoDataFrame({"CROP": a.crop, "SCORE": [v for _, v in polys]},
                         geometry=[p for p, _ in polys], crs=crs)
    g["SJMJ_MU"] = (g.geometry.area * MU).round(3)
    shp_p = os.path.join(a.out, f"{a.town}_{a.crop}_水果面.shp")
    g.to_file(shp_p, driver="ESRI Shapefile", encoding="UTF-8")
    kml_p = os.path.join(a.out, f"{a.town}_{a.crop}_水果面.kml")
    g.to_crs(4326).to_file(kml_p, driver="KML", encoding="UTF-8")
    print(f"[输出] {len(g)} 个面 / {g['SJMJ_MU'].sum():.0f} 亩（中位 {g['SJMJ_MU'].median():.2f} 亩）")
    print(f"       {shp_p}")
    print(f"       {kml_p}")


if __name__ == "__main__":
    main()
