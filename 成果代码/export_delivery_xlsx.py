# -*- coding: utf-8 -*-
"""交付标准导出（合并为单一表）：一行一地块，三类编码并排

按 `成果数据/交付标准数据格式/马角镇前进村*.xlsx` 的口径复刻，公共列 + 三套五级标的：

    序号 | 村编码 | 村名称 | 期数（时间） | 地块唯一编码 | 地块边界（WGS） | 实际面积/亩
    | 农业生产分类编码 | 农业生产分类名称 | 大春作物编码 | 大春作物名称 | 用地分类编码 | 用地分类名称

原来的三类表（地块边界 / 大春作物 / 用地分类）已合并为同一张表，各自的口径如下：

  * 农业生产分类 —— A01 耕种地块、B03 鱼塘（来自耕地层 + 水塘层）；河流/水库等水体不入此口径，留空
  * 大春作物     —— 标的编码（8 位）：AA000000 水稻、AA000700 玉米；潜在裸土留空
  * 用地分类     —— 国土空间用地分类：A0101 水田、A0103 旱地、A1104 养殖水面、B1199 其他水面、
                    C1299 其他未利用地（非植被 / 潜在裸土）

「每块必判」口径（2026-10-15 用户拍定）：模型三阈值链产生的「不确定 / 无覆盖 / 云覆盖缺失」
一律按模型最高概率类（水稻 / 玉米）强制赋值，不再留空；「非植被(建筑/道路/裸土)」判为
**潜在裸土**，写入 用地分类 = C1299 其他未利用地（大春作物列留空）。

水塘层会先按「与耕地重叠」裁剪（`clip_water_to_farmland`）：同事前序水体层与耕地层
重复出图（泡田期水田被 SAM 圈成鱼塘），耕地判为作物保留后，重复的水体图斑要同步削掉，
否则同一块地既记水田又记养殖水面、面积重复计算。

其他口径（对齐参照文件）：
  * 村编码：12 位村级码（来自村界库「村级码」，与参照文件一致）
  * 村名称：村级名 + "委会"（参照文件写「前进村委会」）
  * 期数（时间）：默认 2026-08-20（与参照文件一致，可用 --period 覆盖）
  * 地块唯一编码：G + 村编码 + 6 位序号，**按村独立编号**
  * 地块边界（WGS）：MULTIPOLYGON WKT 文本，WGS84 全精度
  * 实际面积/亩：area_m2 × 0.0015（与参照文件的数值口径一致）

用法：
    python export_delivery_xlsx.py --farmland 太平镇_B_质检.gpkg --pond 绵阳_pond_merged.shp \
        --village-lib 四川省_村界.shp --out-dir 交付 --prefix 太平镇
"""
import argparse
import os
import re
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

sys.stdout.reconfigure(encoding="utf-8")

TARGET_CRS = "EPSG:4523"
MU_FACTOR = 0.0015  # 亩 = m² × 0.0015（等价于 /666.6667）

# 农业生产分类（地块边界口径）：水体（河流/水库等）不属于种植/养殖/林业地块，留空
AG_CODE = {"farmland": ("A01", "耕种地块"), "pond": ("B03", "鱼塘")}
# 标的编码（大春作物）
CROP_CODE = {"水稻": ("AA000000", "水稻"), "玉米": ("AA000700", "玉米")}
# 国土空间用地分类（用地分类口径）
LAND_CODE = {
    "水田": ("A0101", "水田"),
    "旱地": ("A0103", "旱地"),
    "养殖水面": ("A1104", "养殖水面"),
    "其他水面": ("B1199", "其他水面"),
    "其他未利用地": ("C1299", "其他未利用地"),   # 非植被 / 潜在裸土
}

COLUMNS = ["序号", "村编码", "村名称", "期数（时间）", "地块唯一编码",
           "地块边界（WGS）", "实际面积/亩",
           "农业生产分类编码", "农业生产分类名称",
           "大春作物编码", "大春作物名称",
           "用地分类编码", "用地分类名称"]


def load_villages(lib_path, bbox=None):
    v = gpd.read_file(lib_path, bbox=bbox)  # 省库很大，按范围裁剪读取
    cols = {c: c for c in v.columns}
    for src, dst in [("村级码", "village_code"), ("村级名", "village_name")]:
        if src not in v.columns:
            raise SystemExit(f"[错误] 村界库缺少字段 {src}，现有字段: {list(v.columns)}")
        cols[src] = dst
    v = v.rename(columns=cols)[["village_code", "village_name", "geometry"]]
    return v.to_crs(TARGET_CRS)


def assign_village(gdf, villages):
    """按「地块 ∩ 村界」最大重叠面积归属村"""
    vi = villages.sindex
    codes, names = [], []
    for g in gdf.geometry:
        best, bi = None, None
        for j in vi.query(g):
            inter = g.intersection(villages.geometry.iloc[j]).area
            if bi is None or inter > best:
                best, bi = inter, j
        codes.append(str(villages["village_code"].iloc[bi]) if bi is not None else "")
        names.append(villages["village_name"].iloc[bi] if bi is not None else "")
    return codes, names


def village_display(name):
    """村界库里是「前进村」，交付参照里写「前进村委会」"""
    name = "" if name is None else str(name)
    return name + "委会" if name.endswith("村") else name


def _wkt_wgs84(geom):
    """参照文件口径：MULTIPOLYGON，类型名后与坐标逗号后都不带空格，WGS84 全精度"""
    g = geom if geom.geom_type == "MultiPolygon" else MultiPolygon([geom])
    return re.sub(r"^([A-Za-z]+)\s+", r"\1", g.wkt).replace(", ", ",")


def _polygonal_parts(geom):
    """difference 结果可能是 GeometryCollection，只保留面状部分"""
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    return [p for p in getattr(geom, "geoms", []) if isinstance(p, Polygon)]


def clip_water_to_farmland(water, farmland, ratio=0.20, min_m2=100.0):
    """水体多边形减去耕地覆盖；剩余不足（< ratio×原面积 或 < min_m2）则整块剔除。

    背景：同事前序「水体层」与「耕地层」在同一批地块上重复出图 —— SAM 在 5 月上旬
    泡田期把水田圈成鱼塘、在河谷把河滩耕地圈成水面。耕地被我们判为作物保留后，
    水体层里那条重复图斑必须同步削掉，否则同一块地在交付件里既记「水田」又记
    「养殖水面/其他水面」，面积被重复计算。

    返回 (裁剪后的水体层, 整块剔除数, 剔除面积m², 被部分裁剪数)。
    """
    si = farmland.sindex
    fg = list(farmland.geometry)
    keep_idx, keep_geom = [], []
    n_drop = n_clip = 0
    a_drop = 0.0
    for i, g in enumerate(water.geometry):
        if g is None or g.is_empty or g.area <= 0:
            continue
        idx = si.query(g)
        rem = g if len(idx) == 0 else g.difference(unary_union([fg[j] for j in idx]))
        parts = _polygonal_parts(rem)
        rem = parts[0] if len(parts) == 1 else (MultiPolygon(parts) if parts else None)
        if rem is None or rem.area < max(min_m2, ratio * g.area):
            n_drop += 1
            a_drop += g.area
            continue
        if len(idx):
            n_clip += 1
        keep_idx.append(i)
        keep_geom.append(rem)
    out = water.iloc[keep_idx].copy()
    out["geometry"] = keep_geom
    return gpd.GeoDataFrame(out, geometry="geometry", crs=water.crs), n_drop, a_drop, n_clip


def build_rows(gdf, period):
    """把全部地块整理成交付列（序号与地块唯一编码按村独立编号）"""
    out = gdf.copy().reset_index(drop=True)
    out["_mu"] = out.geometry.area * MU_FACTOR
    out["_wkt"] = out.geometry.to_crs("EPSG:4326").apply(_wkt_wgs84)

    code = out["village_code"].astype(str).values
    is_pond = out["_kind"].eq("pond").values
    wkind = out["_wkind"].astype(str).values

    # 「每块必判」：非植被 → 潜在裸土；其余（含不确定 / 无覆盖 / 云覆盖缺失）
    # 一律按模型最高概率类（水稻 / 玉米）赋值，不再留空。
    raw = out["prdct_cf"].astype(str).values
    lbl = (out["prdct_lbl"].fillna("").astype(str).values
           if "prdct_lbl" in out.columns else raw)
    crop = np.where(raw == "非植被(建筑/道路/裸土)", "潜在裸土",
                    np.where(np.isin(raw, ["水稻", "玉米"]), raw, lbl))
    is_rice, is_corn = crop == "水稻", crop == "玉米"
    is_bare = crop == "潜在裸土"

    # 按村独立连续编号 → G + 村编码 + 6 位序号
    seq = pd.DataFrame({"c": code}).groupby("c", sort=False).cumcount().values + 1
    uniq = np.char.add(np.char.add("G", code), np.char.zfill(seq.astype(str), 6))

    # 农业生产分类：水体不入此口径 → 留空
    ag_bz = np.where(is_pond, np.where(wkind == "鱼塘", "B03", ""), "A01")
    ag_mc = np.where(is_pond, np.where(wkind == "鱼塘", "鱼塘", ""), "耕种地块")
    # 大春作物：仅水稻/玉米有值，潜在裸土留空
    crop_bz = np.where(is_rice, "AA000000", np.where(is_corn, "AA000700", ""))
    crop_mc = np.where(is_rice, "水稻", np.where(is_corn, "玉米", ""))
    # 用地分类：水体层按细分落 养殖水面/其他水面；潜在裸土→其他未利用地；
    #           耕地 水稻→水田，其余→旱地
    land_bz = np.where(is_pond, np.where(wkind == "水体", "B1199", "A1104"),
                       np.where(is_bare, "C1299",
                                np.where(is_rice, "A0101", "A0103")))
    land_mc = np.where(is_pond, np.where(wkind == "水体", "其他水面", "养殖水面"),
                       np.where(is_bare, "其他未利用地",
                                np.where(is_rice, "水田", "旱地")))

    df = pd.DataFrame({
        "序号": np.arange(1, len(out) + 1),
        "村编码": code.astype(np.int64),
        "村名称": [village_display(x) for x in out["village_name"].values],
        "期数（时间）": period,
        "地块唯一编码": uniq,
        "地块边界（WGS）": out["_wkt"].values,
        "实际面积/亩": out["_mu"].values,
        "农业生产分类编码": ag_bz,
        "农业生产分类名称": ag_mc,
        "大春作物编码": crop_bz,
        "大春作物名称": crop_mc,
        "用地分类编码": land_bz,
        "用地分类名称": land_mc,
    })
    return df[COLUMNS]


def main():
    ap = argparse.ArgumentParser(description="按交付标准导出合并后的单一表（xlsx + gpkg 镜像）")
    ap.add_argument("--farmland", required=True, help="耕地层（需含 village_code/village_name，作物层还需 prdct_cf）")
    ap.add_argument("--pond", help="水塘层（可选，会一并归入地块边界与用地分类）")
    ap.add_argument("--village-lib", required=True, help="村界库 shp（含 村级码/村级名）")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prefix", default="成果")
    ap.add_argument("--period", default="2026-08-20", help="期数（时间），默认对齐参照文件")
    ap.add_argument("--also-gpkg", action="store_true",
                    help="同时输出 gpkg 镜像（带真实几何，便于在 GIS 里核对；正式交付仍是 xlsx）")
    ap.add_argument("--clip-pond-ratio", type=float, default=0.20,
                    help="水体块减去耕地后，剩余不足「原面积×该比例」即整块剔除，默认 0.20")
    ap.add_argument("--clip-pond-min-m2", type=float, default=100.0,
                    help="水体块减去耕地后，剩余不足该面积（㎡）即整块剔除，默认 100")
    args = ap.parse_args()

    fa = gpd.read_file(args.farmland)
    print(f"[耕地层] {args.farmland}: {len(fa)} 块, 字段={list(fa.columns)}")
    bbox = tuple(round(x, 6) for x in fa.to_crs("EPSG:4326").total_bounds)
    villages = load_villages(args.village_lib, bbox)

    if "village_code" not in fa.columns:
        print("[提示] 耕地层无村界字段，现按村界库归属")
        fa["village_code"], fa["village_name"] = assign_village(fa.to_crs(TARGET_CRS), villages)
    fa = fa.to_crs(TARGET_CRS)
    fa["_kind"] = "farmland"

    parts = [fa]
    if args.pond:
        po = gpd.read_file(args.pond).to_crs(TARGET_CRS)
        po["village_code"], po["village_name"] = assign_village(po, villages)
        po["_kind"] = "pond"
        po["prdct_cf"] = ""  # 水塘不做作物赋值
        n0, a0 = len(po), po.geometry.area.sum() * MU_FACTOR
        po, n_drop, a_drop, n_clip = clip_water_to_farmland(
            po, fa, ratio=args.clip_pond_ratio, min_m2=args.clip_pond_min_m2)
        print(f"[水塘层] {args.pond}: {n0} 块 / {a0:.1f} 亩 "
              f"→ 裁剪后 {len(po)} 块 / {po.geometry.area.sum()*MU_FACTOR:.1f} 亩"
              f"（整块剔除 {n_drop} 块 / {a_drop*MU_FACTOR:.1f} 亩，压边裁剪 {n_clip} 块）")
        cols = ["village_code", "village_name", "prdct_cf", "_kind", "geometry"]
        if "kind" in po.columns:  # 前序检查细分出的 鱼塘/水体
            cols.insert(4, "kind")
        parts.append(po[cols])

    allp = pd.concat(parts, ignore_index=True)
    allp = gpd.GeoDataFrame(allp, geometry="geometry", crs=TARGET_CRS)
    missing = allp["village_code"].astype(str).eq("").sum()
    print(f"合并地块 {len(allp)} 块；村界无归属 {missing} 块")

    # 用地分类里的水体细分：有 kind 列按 kind，否则一律按鱼塘（无水体层时不涉及）
    allp["_wkind"] = ""
    is_pond = allp["_kind"] == "pond"
    if is_pond.any():
        if "kind" in allp.columns:
            allp.loc[is_pond, "_wkind"] = allp.loc[is_pond, "kind"].fillna("鱼塘")
        else:
            allp.loc[is_pond, "_wkind"] = "鱼塘"
            print("[提示] 水体层无 kind 列（未做前序细分），一律按鱼塘处理")

    os.makedirs(args.out_dir, exist_ok=True)
    sub = allp.reset_index(drop=True)
    df = build_rows(sub, args.period)
    path = os.path.join(args.out_dir, f"{args.prefix}_交付.xlsx")
    try:
        df.to_excel(path, index=False)
        print(f"[交付] {len(df)} 行 -> {path}")
    except PermissionError:
        print(f"[警告] {path} 被占用（请在 Excel/GIS 中关闭后重跑），本次跳过 xlsx")
    for col in ["农业生产分类名称", "大春作物名称", "用地分类名称"]:
        print(f"[{col}]")
        print(df.groupby(col)["实际面积/亩"].agg(["count", "sum"]).round(1).to_string())

    if args.also_gpkg:
        mir = gpd.GeoDataFrame(
            df.drop(columns=["地块边界（WGS）"]), geometry=sub.geometry.values, crs=TARGET_CRS)
        gpath = os.path.join(args.out_dir, f"{args.prefix}_交付.gpkg")
        try:
            mir.to_file(gpath, driver="GPKG", encoding="utf-8")
            print(f"        镜像 -> {gpath}")
        except PermissionError:
            print(f"[警告] {gpath} 被占用，本次跳过")


if __name__ == "__main__":
    main()
