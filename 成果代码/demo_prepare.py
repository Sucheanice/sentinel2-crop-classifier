# -*- coding: utf-8 -*-
"""demo_prepare.py — 演示数据准备（宝珠村底图 + 矢量 + 特征库 JSON）

产出（成果数据\\演示\\）：
    底图\\宝珠村.jpg            影像（4326，约 4000 px 宽，8bit）
    底图\\bounds.json           四角经纬度，供 Leaflet imageOverlay 使用
    矢量\\赋值.geojson          我们的赋值面（4,137 块，按作物着色）
    矢量\\村界.geojson          宝珠村村界
    特征库\\library.json        7 套库：月度 NDVI 曲线 + 98 维指纹 + 可分性 + 索引

用法：python demo_prepare.py [--width 4000]
"""
import argparse
import glob
import json
import os
import sys

# 本机 PostgreSQL 自带的旧 proj.db 会干扰 rasterio，先指到 rasterio 自己的 proj_data
import rasterio as _rio
os.environ["PROJ_LIB"] = os.path.join(os.path.dirname(_rio.__file__), "proj_data")

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = r"e:\工作相关\2026年\0624 待测试数据"
TIF = os.path.join(ROOT, r"原始影像\20260902绵阳(项目任务区)\20260901绵阳对应影像和矢量\宝珠村.tif")
ASSIGN = os.path.join(ROOT, r"成果数据\1103 宝珠村赋值\我们的赋值\JY_OUT\510781125\交付\方水镇_交付.shp")
VILL = os.path.join(ROOT, r"原始影像\20260902绵阳(项目任务区)\20260901绵阳对应影像和矢量\宝珠村村界.shp")
OUT = os.path.join(ROOT, r"成果数据\演示")

FRUIT_DIR = os.path.join(ROOT, r"成果数据\柑橘蜜柚识别")
CROP_DIR = os.path.join(ROOT, r"成果数据\作物特征库")
LIBS = [("citrus", "柑橘", "水果", FRUIT_DIR), ("pomelo", "蜜柚", "水果", FRUIT_DIR),
        ("peach", "桃", "水果", FRUIT_DIR), ("rice", "水稻", "作物", CROP_DIR),
        ("corn", "玉米", "作物", CROP_DIR), ("wheat", "小麦", "作物", CROP_DIR),
        ("rape", "油菜", "作物", CROP_DIR)]


def pct_stretch(src, band, lo=2, hi=98, step=17):
    """按 2%~98% 分位做线性拉伸到 0~255（抽样统计，省时）"""
    s = src.read(band, out_shape=(src.height // step, src.width // step)).astype(np.float64)
    s = s[s > 0]
    if s.size == 0:
        return 0.0, 1.0
    a, b = np.percentile(s, [lo, hi])
    return float(a), float(max(b, a + 1e-6))


def make_base(width):
    out_dir = os.path.join(OUT, "底图")
    os.makedirs(out_dir, exist_ok=True)
    with rasterio.open(TIF) as src:
        gains = [pct_stretch(src, b) for b in (1, 2, 3)]
        dst_crs = "EPSG:4326"
        _, w0, h0 = calculate_default_transform(src.crs, dst_crs, src.width, src.height, *src.bounds)
        sc = width / w0
        w2, h2 = int(width), max(1, int(round(h0 * sc)))
        transform2, w2, h2 = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds,
            dst_width=w2, dst_height=h2)
        left, bottom, right, top = rasterio.warp.transform_bounds(
            src.crs, dst_crs, *src.bounds)

        arr = np.zeros((3, h2, w2), dtype=np.uint8)
        for i, b in enumerate((1, 2, 3)):
            tmp = np.zeros((h2, w2), dtype=np.float32)
            reproject(source=rasterio.band(src, b), destination=tmp,
                      src_transform=src.transform, src_crs=src.crs,
                      dst_transform=transform2, dst_crs=dst_crs,
                      resampling=Resampling.bilinear, src_nodata=0)
            a, c = gains[i]
            v = np.clip((tmp - a) / (c - a), 0, 1) * 255
            arr[i] = v.astype(np.uint8)
        jpg = os.path.join(out_dir, "宝珠村.jpg")
        with rasterio.open(jpg, "w", driver="JPEG", height=h2, width=w2, count=3,
                           dtype="uint8", crs=dst_crs, transform=transform2,
                           photometric="YCBCR", quality=88) as dst:
            dst.write(arr)
        w, s, e, n = rasterio.transform.array_bounds(h2, w2, transform2)
        b = {"bounds": [[s, w], [n, e]], "width": w2, "height": h2, "crs": "EPSG:4326"}
        with open(os.path.join(out_dir, "bounds.json"), "w", encoding="utf-8") as f:
            json.dump(b, f, ensure_ascii=False, indent=2)
        print(f"[底图] {jpg}  {w2}x{h2}  bounds={b['bounds']}")


def make_vectors():
    out_dir = os.path.join(OUT, "矢量")
    os.makedirs(out_dir, exist_ok=True)
    g = gpd.read_file(ASSIGN).to_crs("EPSG:4326").reset_index(drop=True)
    g["id"] = range(len(g))
    keep = ["id", "XH", "CJBM", "CJMC", "SJMJ", "CROP_BZBM", "CROP_BZMC",
            "LAND_BZBM", "LAND_BZMC", "geometry"]
    g = g[keep].copy()
    g["SJMJ"] = pd.to_numeric(g["SJMJ"], errors="coerce").round(2)
    g["geometry"] = g.geometry.simplify(0.000003, preserve_topology=True)   # ~0.3 m
    p = os.path.join(out_dir, "赋值.geojson")
    g.to_file(p, driver="GeoJSON", encoding="utf-8")
    print(f"[矢量] {p}  {len(g)} 块  {round(os.path.getsize(p)/1024/1024, 2)} MB")

    v = gpd.read_file(VILL).to_crs("EPSG:4326")
    v["geometry"] = v.geometry.simplify(0.00001, preserve_topology=True)
    p2 = os.path.join(out_dir, "村界.geojson")
    v.to_file(p2, driver="GeoJSON", encoding="utf-8")
    print(f"[矢量] {p2}  {len(v)} 面")


def monthly_from(path):
    cols = ["NDVI_%02d%d" % (m, k) for m in range(1, 13) for k in (1, 2, 3)]
    df = pd.read_csv(path, usecols=lambda c: c in cols)
    med, p25, p75 = [], [], []
    for m in range(12):
        v = df[cols[m * 3:m * 3 + 3]].values.ravel()
        v = v[np.isfinite(v) & (v > -1)]
        if v.size < 200:
            med.append(None); p25.append(None); p75.append(None)
        else:
            med.append(round(float(np.median(v)), 3))
            p25.append(round(float(np.percentile(v, 25)), 3))
            p75.append(round(float(np.percentile(v, 75)), 3))
    return {"median": med, "p25": p25, "p75": p75, "n": int(df.shape[0])}


def make_library():
    out_dir = os.path.join(OUT, "特征库")
    os.makedirs(out_dir, exist_ok=True)
    libs = []
    for key, name, cat, root in LIBS:
        src = os.path.join(root, "自净", f"来源产区_{key}_2025_自净.csv")
        fp = os.path.join(root, "指纹库", f"指纹_{key}_2025.csv")
        if not (os.path.exists(src) and os.path.exists(fp)):
            print(f"  !! 缺 {key}，跳过")
            continue
        tab = pd.read_csv(fp)
        dims = tab.iloc[:, 0].astype(str).tolist()
        stats = {c: [None if pd.isna(x) else float(x) for x in tab[c]]
                 for c in ("P10", "P25", "P50", "P75", "P90", "MAD")}
        libs.append({"key": key, "name": name, "category": cat,
                     "n": int(np.nanmax(tab["N"])) if "N" in tab else None,
                     "dims": dims, "stats": stats, "monthly": monthly_from(src)})
        print(f"[库] {name}: {len(dims)} 维 × {libs[-1]['n']} 点")

    comps = []
    for p in sorted(glob.glob(os.path.join(FRUIT_DIR, "指纹库", "指纹对比_*.csv")) +
                    glob.glob(os.path.join(CROP_DIR, "指纹库", "指纹对比_*.csv"))):
        t = pd.read_csv(p)
        comps.append({"pair": os.path.basename(p)[5:-4].replace("vs", " vs "),
                      "total": int(len(t)),
                      "ok": int((t["可分辨"] == "是").sum()),
                      "top": [{"dim": r["指标"], "d": float(r["CohenD"])}
                              for _, r in t.head(8).iterrows()]})

    idx = pd.read_csv(os.path.join(ROOT, r"成果数据\特征库索引.csv"))
    index = idx.fillna("").astype(str).to_dict(orient="records")

    out = {"generated": "2026-11-09", "libraries": libs, "comparisons": comps, "index": index}
    p = os.path.join(out_dir, "library.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"[库] {p}  {round(os.path.getsize(p)/1024, 1)} KB")


def make_blocks(batch=400, year=2025, shp=None, out_path=None, lib_keys=("rice", "corn")):
    """给每个赋值地块提 98 维指纹 + 到指定作物库的距离，供演示点击联动。
    演示时纯静态读取；GEE 只在此刻用一次。
    shp / out_path 缺省为宝珠村；lib_keys 可含 rice/corn/citrus/pomelo。"""
    import ee
    import citrus_pomelo_extract as EX
    from citrus_pomelo_score import fingerprints
    from merge_fruit_into_delivery import block_fit_dist

    shp = shp or ASSIGN
    out_path = out_path or os.path.join(OUT, "地块特征", "blocks.json")
    EX.ee_init()
    g = gpd.read_file(shp).to_crs("EPSG:4326").reset_index(drop=True)
    g["fid"] = range(len(g))
    g["geometry"] = g.geometry.simplify(0.0001, preserve_topology=True)
    b = g.total_bounds
    aoi = ee.Geometry.Rectangle([float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                                proj="EPSG:4326", geodesic=False)
    stack, names = EX.build_stack(aoi, year)
    cols = EX.band_names(names)
    rows = []
    nb = (len(g) + batch - 1) // batch
    for bi in range(nb):
        sub = g.iloc[bi * batch:(bi + 1) * batch]
        fc = ee.FeatureCollection(sub[["fid", "geometry"]].__geo_interface__["features"])
        red = stack.reduceRegions(collection=fc, reducer=ee.Reducer.mean(),
                                 scale=10, tileScale=4).select(["fid"] + cols)
        for f in EX.getinfo(red)["features"]:
            pr = f["properties"]
            rows.append({"fid": pr["fid"], **{c: pr.get(c) for c in cols}})
        print(f"    batch {bi+1}/{nb} → {len(rows)} 块", flush=True)

    df = pd.DataFrame(rows)
    tfp = fingerprints(df)
    lib_root = {key: root for key, _, _, root in LIBS}
    libs = {}
    for k in lib_keys:
        root = lib_root[k]
        libs[k] = fingerprints(pd.read_csv(os.path.join(root, "自净",
                                                        f"来源产区_{k}_2025_自净.csv")))
    common = [c for c in libs[lib_keys[0]].columns
              if all(c in libs[k].columns for k in lib_keys)]
    dist = {k: block_fit_dist(libs[k], tfp, common) for k in lib_keys}
    truth = g.set_index("fid").apply(
        lambda r: (str(r.get("FRT_BZMC") or "").strip() or str(r.get("CROP_BZMC") or "").strip()),
        axis=1).to_dict()

    blocks = {}
    v = tfp[common].round(3)

    def num(x, nd=3):
        """NaN / inf → None（JSON 不接受 NaN）"""
        x = float(x)
        return None if not np.isfinite(x) else round(x, nd)

    for i, fid in enumerate(df["fid"].values):
        rec = {
            "v": [None if pd.isna(x) else float(x) for x in v.iloc[i]],
            "d": {k: num(dist[k].values[i]) for k in lib_keys},
            "truth": truth.get(fid) or "",
        }
        if "rice" in lib_keys:
            rec["dr"] = rec["d"]["rice"]
        if "corn" in lib_keys:
            rec["dc"] = rec["d"]["corn"]
        blocks[str(int(fid))] = rec
    out = {"generated": "2026-11-09", "dims": common, "libs": list(lib_keys),
           "blocks": blocks}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    print(f"[地块] {out_path}  {len(blocks)} 块 × {len(common)} 维  "
          f"{round(os.path.getsize(out_path)/1024/1024, 2)} MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=4000)
    ap.add_argument("--skip-base", action="store_true")
    ap.add_argument("--blocks", action="store_true", help="提取每个地块的 98 维（调 GEE）")
    ap.add_argument("--blocks-shp", help="要提特征的赋值 shp（默认宝珠村）")
    ap.add_argument("--blocks-out", help="输出 json 路径（默认 成果数据\\演示\\地块特征\\blocks.json）")
    ap.add_argument("--blocks-libs", default="rice,corn",
                    help="对比的作物库，逗号分隔，如 rice,corn,citrus")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if not a.skip_base:
        make_base(a.width)
    make_vectors()
    make_library()
    if a.blocks:
        libs = tuple(x.strip() for x in a.blocks_libs.split(",") if x.strip())
        make_blocks(shp=a.blocks_shp, out_path=a.blocks_out, lib_keys=libs)


if __name__ == "__main__":
    main()
