#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""citrus_pomelo_extract.py — 柑橘 / 蜜柚：GEE 全年多指数旬序列提取

两种模式：
  --mode source   来源产区：按 `四川省_村界.shp` 的乡镇范围随机点采样
  --mode target   目标地块：按 0922 交付 shp 的地块做 zonal mean

口径（与 orchard_explore.py 一致，指数扩到 9 个）：
  S2_SR_HARMONIZED + SCL 云掩膜 + median 旬合成，2025 全年 36 旬
  9 指数 × 36 旬 = 324 维

用法：
  python citrus_pomelo_extract.py --mode source --crop citrus  --out DIR [--points 400] [--limit 1]
  python citrus_pomelo_extract.py --mode source --crop pomelo  --out DIR
  python citrus_pomelo_extract.py --mode target --crop citrus  --out DIR
  python citrus_pomelo_extract.py --mode target --crop pomelo  --out DIR
"""
import argparse
import calendar
import os
import sys
import time

import ee
import geopandas as gpd
import pandas as pd
from shapely.ops import unary_union

sys.stdout.reconfigure(encoding="utf-8")
sys.setrecursionlimit(100000)   # 540 波段时 ee 序列化表达式树很深，默认 1000 会 RecursionError


def ee_init(tries=20, wait=15):
    """经代理刷 OAuth token 偶发被重置（链路间歇性通断），重试到成功。
    默认最多等约 15 分钟（15+30+...+15*19 秒）。"""
    for i in range(tries):
        try:
            ee.Initialize()
            if i:
                print(f"[ee.Initialize 第 {i + 1} 次成功]", flush=True)
            return
        except Exception as e:
            print(f"[ee.Initialize 重试 {i + 1}/{tries}: {type(e).__name__}]", flush=True)
            if i == tries - 1:
                raise
            time.sleep(wait * (i + 1))


def getinfo(obj, tries=12, wait=10):
    """GEE 经代理偶发断连，包一层重试"""
    for i in range(tries):
        try:
            return obj.getInfo()
        except Exception as e:
            if i == tries - 1:
                raise
            print(f"    (getInfo 重试 {i + 1}/{tries - 1}: {type(e).__name__})", flush=True)
            time.sleep(wait * (i + 1))


ROOT = r"e:\工作相关\2026年\0624 待测试数据"
VILL = os.path.join(ROOT, r"训练与标注数据\0907村界等数据\绵阳\四川省村界\四川省村界\四川省_村界.shp")
DELIV = os.path.join(ROOT, r"成果数据\0922 前序环节数据\MY第一批_我们的赋值")

S2 = "COPERNICUS/S2_SR_HARMONIZED"
CLOUD_SCL = [3, 8, 9, 10, 11]
# 15 个指数用到：B2 B3 B4 B5(红边1) B6(红边2) B7(红边3) B8(NIR) B11(SWIR1)
BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B11"]
YEAR = 2025
TARGET_CRS = "EPSG:4523"
BATCH = 400

# 指标前缀（15 个；NDII 与 LSWI 公式相同，按用户点名一并保留）
IDX = ["NDVI", "EVI", "SAVI", "RVI", "NDWI", "MNDWI", "LSWI", "NDRE", "NDI45",
       "NDII", "CVI", "TCARI", "IRECI", "MTCI", "GCVI"]

# ---- 来源产区（乡镇级按名称「包含」匹配村界库；给 village 则下沉到村级）----
BB_CITRUS = [(103.28, 30.08, 103.62, 30.32), (103.30, 29.90, 103.60, 30.12),
             (103.55, 29.85, 104.00, 30.15), (104.55, 29.60, 105.05, 29.95)]
BB_JY = [(103.28, 30.08, 103.62, 30.32)]
BB_DL = [(103.30, 29.90, 103.60, 30.12)]
BB_DP = [(103.55, 29.85, 104.00, 30.15)]
BB_ZZ = [(104.55, 29.60, 105.05, 29.95)]
BB_ZI = [(104.90, 31.40, 105.50, 31.95)]
BB_LZ = [(105.30, 28.80, 105.60, 29.05)]
BB_NC = [(105.90, 31.45, 106.20, 31.80)]
BB_BZ = [(106.45, 31.65, 106.80, 31.95)]
BB_WS = [(105.85, 30.10, 106.65, 30.62)]      # 武胜县 511622
BB_JY2 = [(103.75, 29.35, 104.45, 29.92)]     # 井研县 511124
BB_NB = [(105.50, 31.00, 106.50, 31.70)]      # 南部县 511321

REGIONS = {
    "citrus": [
        # ★ 主产区（园区级）—— 用户 Google Earth 实地确认的连片柑橘园，纯度最高
        {"county": "梓潼县", "garden": "卧龙镇牛头山柑橘产业园", "center": [105.0194, 31.5731], "radius": 700},
        {"county": "梓潼县", "garden": "观义镇回銮村耙耙柑基地", "center": [105.1383, 31.5002], "radius": 700},
        {"county": "梓潼县", "garden": "自强镇错季柑连片基地5000亩", "center": [105.2642, 31.5275], "radius": 900},
        {"county": "蒲江县", "garden": "大兴镇连片耙耙柑核心区", "center": [103.4220, 30.1611], "radius": 900},
        {"county": "蒲江县", "garden": "朝阳湖镇不知火丑柑基地", "center": [103.3247, 30.1235], "radius": 700},
        {"county": "蒲江县", "garden": "西来镇爱媛橙连片基地", "center": [103.5482, 30.2167], "radius": 700},
        # 以下为镇/县级补充产区（粒度较粗，靠「聚类自净」筛常绿群）
        # 蒲江县（全省耙耙柑主产区，约 18 万亩）
        *[{"county": "蒲江县", "town": t, "bbox": BB_JY[0]} for t in
          ["西来镇", "鹤山街道", "朝阳湖镇", "寿安街道", "大塘镇", "大兴镇", "成佳镇"]],
        # 丹棱县（中国桔橙之乡，晚熟柑橘约 18 万亩）
        *[{"county": "丹棱县", "town": t, "bbox": BB_DL[0]} for t in
          ["齐乐镇", "仁美镇", "杨场镇", "顺龙乡", "张场镇"]],
        # 东坡区（中国晚熟柑橘之乡，春见/大雅主产）
        *[{"county": "东坡区", "town": t, "bbox": BB_DP[0]} for t in
          ["尚义镇", "思蒙镇", "复兴镇", "崇礼镇", "富牛镇", "太和镇",
           "龙正镇", "永寿镇", "三苏镇", "万胜镇", "松江镇", "修文镇"]],
        # 资中县（全国最大血橙基地县，作异质交叉验证）
        *[{"county": "资中县", "town": t, "bbox": BB_ZZ[0]} for t in
          ["重龙镇", "水南镇", "银山镇", "鱼溪镇", "球溪镇"]],
        # 新增：武胜县（大雅柑 20 万亩）/ 井研县 / 南部县 —— 按村级码圈整县，
        # 县级整体采样粒度较粗，靠后面的「聚类自净」把常绿群（柑橘）筛出来
        {"county": "武胜县", "code": "511622", "bbox": BB_WS[0]},
        {"county": "井研县", "code": "511124", "bbox": BB_JY2[0]},
        {"county": "南部县", "code": "511321", "bbox": BB_NB[0]},
    ],
    "pomelo": [
        # ★ 主产区：天宝蜜柚博览园（2,500 亩连片，梓潼县许州镇天宝村，
        #   20 万亩蜜柚产业核心示范区 / 国家地理标志 / 3A 景区）
        #   中心点由用户实地确认（Google Earth）→ 园区级采样，纯度最高
        {"county": "梓潼县", "garden": "天宝蜜柚博览园", "center": [105.166984, 31.800474],
         "radius": 600, "epsg": "EPSG:4523"},
        # ★ 天宝村万亩蜜柚主产区（用户实地确认）
        {"county": "梓潼县", "garden": "天宝村万亩蜜柚区", "center": [105.163900, 31.820990],
         "radius": 900, "epsg": "EPSG:4523"},
        # 梓潼县（四川省最大蜜柚基地，全县 18.2 万亩 / 16 个乡镇）—— 镇级作补充
        {"county": "梓潼县", "town": "许州镇", "bbox": BB_ZI[0]},            # 含天宝村
        {"county": "梓潼县", "town": "演武镇", "bbox": BB_ZI[0]},            # 含小亭村
        # 阆中市（文成镇大桥村优质蜜柚基地）
        {"county": "阆中市", "town": "文成镇", "village": "大桥村", "bbox": BB_NC[0]},
        # 泸州龙马潭区（胡市镇黄桷村三红蜜柚基地，仅 11 亩，村级采样会被稀释）
        {"county": "泸州市龙马潭区", "town": "胡市", "village": "黄桷", "bbox": BB_LZ[0]},
        # 注：巴中恩阳司城平桥村已剔除 —— 自净实测常绿群仅 4.3%、冬 NDVI 0.261，
        #     判定为「该村不是蜜柚主导」，保留会污染指纹库
    ],
}

# ---- 目标地块：0922 交付件（AOI 目录 → 镇名）----
TARGETS = {
    "citrus": [("PW_OUT", "510727108", "水晶镇"), ("PW_OUT", "510727216", "豆叩羌族乡"),
               ("PW_OUT", "510727107", "大桥镇"), ("PW_OUT", "510727208", "旧堡羌族乡"),
               ("JY_OUT", "33", "新安镇"), ("JY_OUT", "38", "河口镇"),
               ("JY_OUT", "26", "太平镇"), ("JY_OUT", "42", "马角镇")],
    "pomelo": [("ZT_OUT", "510725103", "黎雅镇"), ("ZT_OUT", "510725112", "金龙镇")],
}


def dekad_list(year):
    out = []
    for m in range(1, 13):
        last = calendar.monthrange(year, m)[1]
        for k, (d0, d1) in enumerate([(1, 10), (11, 20), (21, last)], 1):
            out.append((f"{m:02d}{k}", f"{year}-{m:02d}-{d0:02d}", f"{year}-{m:02d}-{d1:02d}T23:59:59"))
    return out


def mask_s2(img):
    scl = img.select("SCL")
    m = None
    for c in CLOUD_SCL:
        m = scl.eq(c) if m is None else m.Or(scl.eq(c))
    return img.updateMask(m.Not())


def build_stack(aoi, year):
    """36 旬 × 15 指数 的堆叠影像（540 波段）+ 旬标签列表"""
    s2 = ee.ImageCollection(S2).filterBounds(aoi)
    imgs, names, per_dekad = [], [], []
    for nm, st, en in dekad_list(year):
        c = s2.filterDate(st, en).map(mask_s2).select(BANDS).median()
        b2, b3, b4 = c.select("B2"), c.select("B3"), c.select("B4")
        b5, b6, b7 = c.select("B5"), c.select("B6"), c.select("B7")
        b8, b11 = c.select("B8"), c.select("B11")
        idx = [
            ("NDVI", c.normalizedDifference(["B8", "B4"])),
            ("EVI", c.expression("2.5*(N-R)/(N+6*R-7.5*B+1)", {"N": b8, "R": b4, "B": b2})),
            ("SAVI", c.expression("1.5*(N-R)/(N+R+0.5)", {"N": b8, "R": b4})),
            ("RVI", b8.divide(b4)),
            ("NDWI", c.normalizedDifference(["B3", "B8"])),
            ("MNDWI", c.normalizedDifference(["B3", "B11"])),
            ("LSWI", c.normalizedDifference(["B8", "B11"])),
            ("NDRE", c.normalizedDifference(["B8", "B5"])),
            ("NDI45", c.normalizedDifference(["B5", "B4"])),
            # NDII 与 LSWI 同式（(NIR-SWIR)/(NIR+SWIR)），按点名保留
            ("NDII", c.normalizedDifference(["B8", "B11"])),
            # 叶绿素类
            ("CVI", c.expression("N*R/(G*G)", {"N": b8, "R": b4, "G": b3})),
            ("TCARI", c.expression("3*((RE-R)-0.2*(RE-G)*(RE/R))",
                                   {"RE": b5, "R": b4, "G": b3})),
            ("IRECI", c.expression("(B7-B4)/(B5/B6)", {"B7": b7, "B4": b4, "B5": b5, "B6": b6})),
            ("MTCI", c.expression("(B6-B5)/(B5-B4)", {"B6": b6, "B5": b5, "B4": b4})),
            ("GCVI", c.expression("N/G-1", {"N": b8, "G": b3})),
        ]
        for pre, img in idx:
            imgs.append(img.rename(f"{pre}_{nm}"))
        names.append(nm)
        # 关键：每旬先合成一个 15 波段影像再拼，嵌套深度 36 而非 540（避免 RecursionError）
        per_dekad.append(ee.Image.cat(imgs[-len(idx):]))
    return ee.Image.cat(per_dekad), names


def band_names(names):
    return [f"{p}_{n}" for n in names for p in IDX]


def towns_geometry(crop, limit=None, gardens_only=False):
    """按 REGIONS 取每个产区的联合范围；给 village 则下沉到村级。
    gardens_only=True 时只保留「园区级」（center+radius）产区。
    返回 [(key, geometry, 含村数, 级)]"""
    out, cache = [], {}
    for spec in REGIONS[crop]:
        if gardens_only and not spec.get("center"):
            continue
        # 园区级：直接用「已知连片果园中心 + 半径」圈样区，不依赖村镇边界
        # （村镇级撒点必然混入大量耕地；冬小麦/油菜/蔬菜冬季也绿，会被误当常绿果园）
        if spec.get("center"):
            lon, lat = spec["center"]
            r = spec.get("radius", 600)
            epsg = spec.get("epsg", "EPSG:4523")
            g = (gpd.GeoSeries([gpd.points_from_xy([lon], [lat])[0]], crs="EPSG:4326")
                 .to_crs(epsg).buffer(r).to_crs("EPSG:4326").iloc[0])
            key = f'{spec["county"]}_{spec["garden"]}'
            if any(k == key for k, *_ in out):
                continue
            out.append((key, g, 1, f"园{r}m"))
            continue
        bb = spec["bbox"]
        if bb not in cache:
            cache[bb] = gpd.read_file(VILL, bbox=bb)
        v = cache[bb]
        if spec.get("code"):        # 按村级码前 6 位圈整县
            sel = v[v["村级码"].astype(str).str[:6] == spec["code"]]
            lv, tag = "县", spec["county"]
        else:
            sel = v[v["乡镇级"].astype(str).str.contains(spec["town"], na=False)]
            lv, tag = "镇", spec["town"]
            if spec.get("village"):
                sel = sel[sel["村级名"].astype(str).str.contains(spec["village"], na=False)]
                lv, tag = "村", f'{spec["town"]}{spec["village"]}'
        key = f'{spec["county"]}_{tag}'
        if len(sel) == 0:
            print(f"  !! 未匹配: {key}")
            continue
        if any(k == key for k, *_ in out):
            continue
        g = sel.geometry.union_all() if hasattr(sel.geometry, "union_all") \
            else sel.geometry.unary_union
        out.append((key, g, len(sel), lv))
    if limit:
        out = out[:limit]
    return out


def truth_regions(path):
    """从「真值地块」矢量读产区（人工在 Google Earth 手绘、确知是目标作物的地块）。
    返回 [(key, geometry, 建议采样点数, '真值')] —— 采样点数按面积 1 点/10m 像元估。"""
    g = gpd.read_file(path).to_crs("EPSG:4326")
    gm = g.to_crs(TARGET_CRS)
    out = []
    for i, (_, r) in enumerate(g.iterrows()):
        a_m2 = float(gm.geometry.iloc[i].area)
        npts = int(min(9000, max(60, a_m2 / 100)))     # 1 点/像元，60~9000 截断
        nm = str(r.get("名称", "真值"))
        out.append((f"{nm}_{i:02d}", r.geometry, npts, "真值"))
    return out


SAMPLE_MAX = 4500      # GEE sample() 单次硬上限 5000 要素，留余量


def sample_region(stack, reg, npts, seed0=42):
    """按 numPixels 抽样；单次超 5000 会被 GEE 中止，故分多次不同 seed 抽后合并。"""
    feats, got, s = [], 0, seed0
    while got < npts and s < seed0 + 6:
        k = min(SAMPLE_MAX, npts - got)
        fc = stack.sample(region=reg, scale=10, numPixels=k, seed=s,
                          geometries=True, tileScale=4, dropNulls=False)
        f = getinfo(fc)["features"]
        feats.extend(f)
        got += len(f)
        if len(f) < k:          # 该区域像元已抽尽
            break
        s += 1
    return feats


def run_source(crop, out_dir, points, limit=None, year=YEAR, shard=None, gardens_only=False,
               truth_shp=None, tag_extra=""):
    """一次构建堆叠（覆盖该 shard 全部产区的联合范围），各产区共用 → 省掉逐产区重建 36 旬合成。
    shard=(i, n) 时只跑第 i 片（0-based），用于多进程并行。
    truth_shp 给定时，产区 = 该矢量里的真值地块（此时元组第 3 位即采样点数）。"""
    truth = bool(truth_shp)
    towns = truth_regions(truth_shp) if truth else towns_geometry(crop, limit, gardens_only)
    if shard:
        i, n = shard
        towns = towns[i::n]
        print(f"[shard {i}/{n}] 本片 {len(towns)} 个产区", flush=True)
    if not towns:
        print("!! 未匹配到任何产区")
        return None
    union = unary_union([g for _, g, *_ in towns])
    aoi = ee.Geometry(union.__geo_interface__, proj="EPSG:4326", geodesic=False)
    print(f"[source] {len(towns)} 个产区，构建 36 旬 × 15 指数（540 波段）堆叠 …", flush=True)
    stack, names = build_stack(aoi, year)
    cols = band_names(names)
    tag = (f"_s{shard[0]}of{shard[1]}" if shard
           else ("_真值" if truth_shp else ("_园" if gardens_only else ""))) + tag_extra
    p = os.path.join(out_dir, f"来源产区_{crop}_{year}{tag}.csv")
    rows = []
    for key, geom, nv, lv in towns:
        npts = nv if truth else points
        print(f"    {key}（{lv}，{nv}{'点' if truth else ' 个村'}）采样 {npts} 点 …", flush=True)
        reg = ee.Geometry(geom.__geo_interface__, proj="EPSG:4326", geodesic=False)
        feats = sample_region(stack, reg, npts)
        for f in feats:
            p2 = f["properties"]
            g = f.get("geometry")
            r = {"region": key, "crop": crop, "kind": "source"}
            if g:
                r["lon"], r["lat"] = g["coordinates"][0], g["coordinates"][1]
            r.update({c: p2.get(c) for c in cols})
            rows.append(r)
        print(f"          → {len(rows)} 点累计", flush=True)
        # 分步落盘：中途失败也不至于全丢
        pd.DataFrame(rows).to_csv(p, index=False, encoding="utf-8-sig")
    df = pd.DataFrame(rows)
    df.to_csv(p, index=False, encoding="utf-8-sig")
    print(f"[输出] {p}  ({len(df)} 行 × {len(df.columns)} 列)")
    return p


def _read_target(aoi, code, town):
    """交付件在 DELIV/{OUT}/{AOI编码}/交付/{镇名}_交付.shp。

    兼容两种批次：0922 用 AOI 编码（如 33/38），1014/1016/1019 用镇编码（如 510781109）。
    配置的 code 目录不存在时，按镇名在该 OUT 下自动查找。"""
    d = os.path.join(DELIV, aoi, code, "交付")
    if not os.path.isdir(d):
        import glob as _glob
        cands = _glob.glob(os.path.join(DELIV, aoi, "*", "交付", f"{town}_交付.shp"))
        if not cands:
            raise FileNotFoundError(
                f"找不到 {town} 交付件：{DELIV}\\{aoi}\\*\\交付\\{town}_交付.shp")
        d = os.path.dirname(cands[0])
    f = [x for x in os.listdir(d) if x.endswith("_交付.shp")][0]
    g = gpd.read_file(os.path.join(d, f)).to_crs(TARGET_CRS)
    g = g[g.AG_BZBM == "A01"].copy().reset_index(drop=True)
    g["fid"] = range(len(g))
    return g


def run_target(crop, out_dir, year=YEAR, limit=None, towns=None):
    names = [d[0] for d in dekad_list(year)]
    cols = band_names(names)
    targets = TARGETS[crop]
    if towns:
        keys = [t.strip() for t in towns.split(",") if t.strip()]
        targets = [x for x in targets if x[2] in keys]
    if limit:
        targets = targets[:limit]
    for aoi, code, town in targets:
        g = _read_target(aoi, code, town)
        gw = g.to_crs("EPSG:4326")
        gw["geometry"] = gw.geometry.simplify(tolerance=0.0001, preserve_topology=True)
        print(f"[target] {town} {len(g)} 块 × {len(cols)} 波段 …", flush=True)
        # 堆叠只建一次（用全镇外接矩形，避免对上千地块做 union 触发拓扑异常）
        b = gw.total_bounds
        town_geom = ee.Geometry.Rectangle([float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                                          proj="EPSG:4326", geodesic=False)
        stack, _ = build_stack(town_geom, year)
        rows = []
        nb = (len(gw) + BATCH - 1) // BATCH
        for bi in range(nb):
            sub = gw.iloc[bi * BATCH:(bi + 1) * BATCH]
            fc = ee.FeatureCollection(sub[["fid", "geometry"]].__geo_interface__["features"])
            red = stack.reduceRegions(collection=fc, reducer=ee.Reducer.mean(),
                                      scale=10, tileScale=4).select(["fid"] + cols)
            for f in getinfo(red)["features"]:
                p = f["properties"]
                rows.append({"fid": p["fid"], **{c: p.get(c) for c in cols}})
            print(f"          batch {bi+1}/{nb} → {len(rows)}", flush=True)
        df = pd.DataFrame(rows)
        df["town"] = town
        df["aoi"] = f"{aoi}/{code}"
        out = os.path.join(out_dir, f"目标地块_{crop}_{town}_{year}.csv")
        df.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"[输出] {out}  ({len(df)} 行)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["source", "target"], required=True)
    ap.add_argument("--crop", choices=["citrus", "pomelo"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--points", type=int, default=400, help="来源产区每镇采样点数")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 个产区/目标镇（试点用）")
    ap.add_argument("--towns", help="只跑指定镇（逗号分隔，target 模式用）")
    ap.add_argument("--year", type=int, default=YEAR)
    ap.add_argument("--dry-run", action="store_true", help="只列匹配到的产区，不调 GEE")
    ap.add_argument("--shard", help="分片并行，形如 0/3（第 0 片，共 3 片）")
    ap.add_argument("--gardens-only", action="store_true",
                    help="只跑园区级产区（center+radius，用户实地确认的连片果园）")
    ap.add_argument("--truth-shp", help="真值地块矢量（人工手绘确知作物的地块）→ 产区=这些地块")
    ap.add_argument("--tag", help="输出文件名后缀（如 真值v2），便于分批跑后合并")
    ap.add_argument("--deliv", help="交付件根目录（默认 0922 批次；1014 批次需指定）")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.deliv:
        global DELIV
        DELIV = a.deliv
        print(f"[交付根] {DELIV}")
    if a.dry_run:
        for crop in ["citrus", "pomelo"]:
            t = towns_geometry(crop)
            n_pts = sum(1 for _ in t)
            print(f"\n[{crop}] 匹配 {len(t)} 个产区：")
            for key, g, nv, lv in t:
                print(f"   {key:<34} {lv}级 {nv:>3} 个村  bbox(4326)="
                      f"{[round(x, 4) for x in g.bounds]}")
        return
    ee_init()
    shard = None
    if a.shard:
        i, n = a.shard.split("/")
        shard = (int(i), int(n))
    if a.mode == "source":
        run_source(a.crop, a.out, a.points, a.limit or None, a.year, shard, a.gardens_only,
                   a.truth_shp, a.tag or "")
    else:
        run_target(a.crop, a.out, a.year, a.limit or None, a.towns)


if __name__ == "__main__":
    main()
