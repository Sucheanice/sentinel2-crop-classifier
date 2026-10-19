# -*- coding: utf-8 -*-
"""predict_dachun_binary.py — 用 23县二分类模型（水稻/玉米）做推理。

三种模式（--mode）：
  mianyang    绵阳迁移验证：推绵阳标注库，对比真实标签算准确率（有标签）
  qianjincun  前进村预测：推江油前进村耕地地块，输出 shp/csv（无标签）
  anju        安居区预测：推大春标注库安居区地块，输出 shp/csv + 精度（有标签）

与训练完全一致：按瓦片读 4 期影像（05/06/07/08 -> P1/P2/P3/P4），每地块
随机采 3 点取 zonal mean，再用 common.compute_feature_matrix 构建 172 维特征。
"""
import os, sys, json, random, pickle, argparse
import numpy as np
import pandas as pd

for _p in [
    r"C:\Users\lenovo\AppData\Roaming\Python\Python313\site-packages\rasterio\proj_data",
    r"C:\Users\lenovo\AppData\Roaming\Python\Python312\site-packages\rasterio\proj_data",
    r"C:\Users\lenovo\AppData\Roaming\Python\Python311\site-packages\rasterio\proj_data",
]:
    if os.path.isdir(_p):
        os.environ["PROJ_LIB"] = _p
        break

import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.warp import transform_bounds

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "dachun_binary_23counties", "crop_model_binary.pkl")
TILE_JSON = os.path.join(BASE_DIR, "待训练数据大春", "tile_dates.json")
DL_DIR = r"E:\迅雷下载"
OUT_DIR = os.path.join(BASE_DIR, "dachun_prediction")
os.makedirs(OUT_DIR, exist_ok=True)

BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
PHASES = ["05", "06", "07", "08"]
PREFIX = ["P1", "P2", "P3", "P4"]
N_SAMPLES = 3

# 绵阳 6 瓦片 4 期日期（推理侧，不在 tile_dates.json 里）
MIANYANG_DATES = {
    "48SUA": {"05": "2025-05-20", "06": "2025-06-12", "07": "2025-07-16", "08": "2025-08-31"},
    "48SUB": {"05": "2025-05-28", "06": "2025-06-17", "07": "2025-07-17", "08": "2025-08-31"},
    "48SVA": {"05": "2025-05-20", "06": "2025-06-12", "07": "2025-07-22", "08": "2025-08-31"},
    "48SVB": {"05": "2025-05-20", "06": "2025-06-12", "07": "2025-07-16", "08": "2025-08-31"},
    "48SWA": {"05": "2025-05-20", "06": "2025-06-04", "07": "2025-07-16", "08": "2025-08-03"},
    "48SWB": {"05": "2025-05-10", "06": "2025-06-26", "07": "2025-07-16", "08": "2025-08-13"},
}

from common import compute_feature_matrix


def scan_downloaded_tiles():
    """扫描 DL_DIR，返回 {tile: base_dir} 与 {tile: (minx,miny,maxx,maxy)}。"""
    tile_base, tile_bounds = {}, {}
    if not os.path.isdir(DL_DIR):
        return tile_base, tile_bounds
    for d in os.listdir(DL_DIR):
        if not d.endswith("-5-8"):
            continue
        parts = d.split("-")
        if len(parts) < 2:
            continue
        tile = parts[1]
        b02 = os.path.join(DL_DIR, d, PHASES[0], "B02.tif")
        if not os.path.exists(b02):
            continue
        tile_base[tile] = os.path.join(DL_DIR, d)
        try:
            with rasterio.open(b02) as src:
                b = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
            tile_bounds[tile] = b
        except Exception as e:
            print(f"[scan] 读取 {d} 失败: {e}")
    return tile_base, tile_bounds


def load_tile_dates():
    """合并训练侧(tile_dates.json) + 绵阳 S 带日期。"""
    td = {}
    if os.path.exists(TILE_JSON):
        with open(TILE_JSON, "r", encoding="utf-8") as f:
            td.update(json.load(f))
    td.update(MIANYANG_DATES)
    return td


BUFFER_DIST = 0.0
MIN_PX = 1
BANDS_20M = {"B05", "B06", "B07", "B8A", "B11", "B12"}


def extract_features(gdf, tile_dates, cloud_mask_threshold=None, cloud_mask_scl=False):
    """对 gdf（WGS84）按瓦片提取 4期×10波段 zonal mean（栅格化整地块均值）。

    cloud_mask_threshold: 若不为 None，用每期 B02 亮像元(B02>threshold)做云掩膜（近似）。
    cloud_mask_scl: 若为 True，优先用每期 SCL.tif（若存在）做精确云掩膜
                   （SCL 3=云影/8=中云/9=高云/10=薄卷云），仅剔除真实云像元。
    两者可叠加；剔除云像元后再算 zonal mean（分母=掩膜后有效像素数）。

    返回 (feat_df, uncovered_idx)，feat_df 列 = P1_B02..P4_B12 + tile + _idx。
    """
    tile_base, tile_bounds = scan_downloaded_tiles()

    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)

    tile_to_idxs = {}
    uncovered = []
    for idx, geom in gdf.geometry.items():
        if geom is None or geom.is_empty:
            uncovered.append(idx)
            continue
        c = geom.centroid
        x, y = c.x, c.y
        hit = None
        for t, (minx, miny, maxx, maxy) in tile_bounds.items():
            if minx <= x <= maxx and miny <= y <= maxy:
                hit = t
                break
        if hit is None or hit not in tile_dates or hit not in tile_base:
            uncovered.append(idx)
            continue
        tile_to_idxs.setdefault(hit, []).append(idx)

    print(f"  分配瓦片: {len(tile_to_idxs)} 个, 覆盖地块 {sum(len(v) for v in tile_to_idxs.values())}, "
          f"未覆盖 {len(uncovered)}")

    all_dfs = []
    for tile in sorted(tile_to_idxs):
        base = tile_base[tile]
        idxs = tile_to_idxs[tile]
        print(f"  [{tile}] {len(idxs)} 地块, 提取特征...", flush=True)
        ref = os.path.join(base, PHASES[0], "B02.tif")
        with rasterio.open(ref) as r:
            crs = r.crs
            h, w = r.height, r.width
            t = r.transform

        sub = gpd.GeoDataFrame(gdf.loc[idxs], geometry="geometry").to_crs(crs)
        shapes = []
        valid_idx = []
        for i, idx in enumerate(sub.index):
            geom = sub.loc[idx].geometry
            if geom is None or geom.is_empty:
                continue
            buf = geom.buffer(BUFFER_DIST)
            if buf.is_empty or buf.area <= 0:
                continue
            # label 必须用 valid_idx 的连续序号（而非 enumerate 位置 i），
            # 否则前面有地块被 continue 跳过时，label 与 valid_idx 错位，
            # 导致后续所有地块特征张冠李戴。
            shapes.append((buf, len(valid_idx) + 1))
            valid_idx.append(idx)
        if not shapes:
            continue

        label = rasterize(shapes, out_shape=(h, w), transform=t, fill=0, dtype=np.int32)
        px_counts = np.bincount(label.ravel(), minlength=len(valid_idx) + 2)

        rows = {idx: {} for idx in valid_idx}
        for pi, phase in enumerate(PHASES):
            col_prefix = PREFIX[pi]
            ph_dir = os.path.join(base, phase)

            # 云掩膜：优先 SCL（精确）> B02 阈值（近似）；True=云，剔除后再算均值
            cloud_mask = None
            valid_counts = None
            if cloud_mask_scl:
                sclp = os.path.join(ph_dir, "SCL.tif")
                if os.path.exists(sclp):
                    with rasterio.open(sclp) as ssrc:
                        scl = ssrc.read(1)
                    if scl.shape[0] != h:  # 20m -> 10m 上采样
                        scl = np.repeat(np.repeat(scl, 2, axis=0), 2, axis=1)[:h, :w]
                    cloud_mask = np.isin(scl, [3, 8, 9, 10])
            elif cloud_mask_threshold is not None:
                b02p = os.path.join(ph_dir, "B02.tif")
                if os.path.exists(b02p):
                    with rasterio.open(b02p) as bsrc:
                        b02 = bsrc.read(1)
                    cloud_mask = b02 > cloud_mask_threshold
            if cloud_mask is not None:
                valid = (~cloud_mask).astype(np.float64)
                valid_counts = np.bincount(label.ravel(), weights=valid.ravel(),
                                           minlength=len(valid_idx) + 2)

            for band in BANDS:
                bp = os.path.join(ph_dir, f"{band}.tif")
                col = f"{col_prefix}_{band}"
                if not os.path.exists(bp):
                    for idx in valid_idx:
                        rows[idx][col] = np.nan
                    continue
                with rasterio.open(bp) as src:
                    data = src.read(1)
                    if band in BANDS_20M:
                        data = np.repeat(np.repeat(data, 2, axis=0), 2, axis=1)[:h, :w]
                    if cloud_mask is not None:
                        data = np.where(cloud_mask, 0.0, data)
                    sums = np.bincount(label.ravel(), weights=data.ravel().astype(np.float64),
                                       minlength=len(valid_idx) + 2)
                for i, idx in enumerate(valid_idx):
                    pc = valid_counts[i + 1] if valid_counts is not None else px_counts[i + 1]
                    rows[idx][col] = sums[i + 1] / pc if pc >= MIN_PX else np.nan

        for idx in valid_idx:
            rows[idx]["tile"] = tile
            rows[idx]["_idx"] = idx
        all_dfs.append(pd.DataFrame.from_dict(rows, orient="index"))
        print(f"  [{tile}] 特征完成", flush=True)

    if not all_dfs:
        return None, uncovered
    feat_df = pd.concat(all_dfs, ignore_index=True)
    return feat_df, uncovered


def run_predict(gdf, feat_df, bundle, label_col=None):
    """用模型预测，返回带预测字段的 GeoDataFrame。"""
    # 按 P1..P4 × band 收集原始波段 zonal mean
    band_values = {}
    for p in PREFIX:
        for b in BANDS:
            c = f"{p}_{b}"
            if c in feat_df.columns:
                band_values[c] = feat_df[c].values

    features_arr, feat_names = compute_feature_matrix(
        band_values, PREFIX, available_bands=BANDS)
    feat_df2 = pd.DataFrame(features_arr, columns=feat_names)

    selected = bundle.get("selected_features", list(feat_df2.columns))
    for m in [f for f in selected if f not in feat_df2.columns]:
        feat_df2[m] = 0.0
    X = feat_df2[selected].fillna(0).values.astype(np.float32)

    model = bundle["model"]
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    max_prob = np.max(y_proba, axis=1)

    le = bundle.get("label_encoder")
    class_names = bundle.get("class_names", ["水稻", "玉米"])
    if le is not None:
        pred_labels = le.inverse_transform(y_pred)
    else:
        pred_labels = [class_names[i] for i in y_pred]

    out = gdf.loc[feat_df["_idx"].values].copy()
    out["pred"] = y_pred
    out["crop"] = pred_labels
    out["prob"] = max_prob
    out["conf"] = ["高" if p >= 0.70 else ("中" if p >= 0.60 else "低")
                   for p in max_prob]

    if label_col is not None and label_col in out.columns:
        out["true"] = out[label_col].astype(str)
        out["correct"] = (out["crop"] == out["true"]).astype(int)
    return out


def save_outputs(out, name):
    csv_p = os.path.join(OUT_DIR, name + ".csv")
    gpkg_p = os.path.join(OUT_DIR, name + ".gpkg")
    shp_p = os.path.join(OUT_DIR, name + ".shp")
    # CSV 去掉 geometry 保留属性
    out.drop(columns="geometry").to_csv(csv_p, index=False, encoding="utf-8-sig")
    out.to_file(gpkg_p, driver="GPKG")
    try:
        out.to_file(shp_p, driver="ESRI Shapefile", encoding="utf-8")
    except Exception as e:
        print(f"  (shp 写入失败: {e}; gpkg/csv 已就绪)")
    return csv_p, gpkg_p, shp_p


def report_accuracy(out):
    if "correct" not in out.columns:
        return
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    acc = accuracy_score(out["true"], out["crop"])
    f1 = f1_score(out["true"], out["crop"], average="weighted")
    cm = confusion_matrix(out["true"], out["crop"], labels=["水稻", "玉米"])
    print(f"  总体 Acc: {acc:.4f}  F1(weighted): {f1:.4f}")
    print(f"  混淆矩阵 (行=真实, 列=预测):\n    {cm.tolist()}")
    print(f"  预测分布: {dict(out['crop'].value_counts())}")
    # 分县精度
    for gcol in ["QXMC", "XZMC"]:
        if gcol in out.columns:
            print(f"  分县 Acc:")
            for g, sub in out.groupby(gcol):
                if len(sub) < 5:
                    continue
                a = accuracy_score(sub["true"], sub["crop"])
                print(f"    {g}: {a:.4f} (n={len(sub)})")
            break


# ============ 各模式 ============
def mode_mianyang(bundle):
    print("=" * 70)
    print("[绵阳迁移验证] 推绵阳标注库, 对比真实标签")
    print("=" * 70)
    gpkg = os.path.join(BASE_DIR, "待训练数据绵阳市", "绵阳标注_水稻玉米.gpkg")
    gdf = gpd.read_file(gpkg)
    gdf = gdf[gdf["类别"].isin(["水稻", "玉米"])].reset_index(drop=True)
    print(f"  绵阳标注: {len(gdf)} 块")

    tile_dates = load_tile_dates()
    feat_df, uncovered = extract_features(gdf, tile_dates)
    if feat_df is None:
        print("  !! 无有效特征")
        return
    out = run_predict(gdf, feat_df, bundle, label_col="类别")
    save_outputs(out, "mianyang_transfer")
    report_accuracy(out)
    print(f"  未覆盖地块: {len(uncovered)}")
    return out


def mode_qianjincun(bundle):
    print("=" * 70)
    print("[前进村预测] 推江油前进村耕地地块 (无标签)")
    print("=" * 70)
    gdb = os.path.join(BASE_DIR, "待测试数据前进0806", "前进0806.gdb")
    gdf = gpd.read_file(gdb, layer="dltb")
    gdf = gdf[gdf["ZLDWMC"] == "前进村"].copy()
    gdf = gdf[gdf["DLMC"].isin(["旱地", "水田", "水浇地", "后备耕地"])].reset_index(drop=True)
    print(f"  前进村耕地地块: {len(gdf)} 块")

    tile_dates = load_tile_dates()
    feat_df, uncovered = extract_features(gdf, tile_dates)
    if feat_df is None:
        print("  !! 无有效特征")
        return
    out = run_predict(gdf, feat_df, bundle, label_col=None)
    save_outputs(out, "qianjincun_prediction")
    print(f"  预测分布: {dict(out['crop'].value_counts())}")
    print(f"  置信度: {dict(out['conf'].value_counts())}")
    print(f"  未覆盖地块: {len(uncovered)}")
    return out


def mode_anju(bundle):
    print("=" * 70)
    print("[安居区预测] 推大春标注库安居区地块 (有标签)")
    print("=" * 70)
    gpkg = os.path.join(BASE_DIR, "待训练数据大春", "大春标注_水稻玉米.gpkg")
    gdf = gpd.read_file(gpkg)
    gdf = gdf[gdf["QXMC"] == "安居区"].reset_index(drop=True)
    gdf = gdf[gdf["类别"].isin(["水稻", "玉米"])].reset_index(drop=True)
    print(f"  安居区地块: {len(gdf)} 块")

    tile_dates = load_tile_dates()
    feat_df, uncovered = extract_features(gdf, tile_dates)
    if feat_df is None:
        print("  !! 无有效特征")
        return
    out = run_predict(gdf, feat_df, bundle, label_col="类别")
    save_outputs(out, "anju_prediction")
    report_accuracy(out)
    print(f"  未覆盖地块: {len(uncovered)}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["mianyang", "qianjincun", "anju"])
    args = ap.parse_args()

    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    print(f"模型: {bundle.get('model_type')}, 类别 {bundle.get('class_names')}, "
          f"selected_features {len(bundle.get('selected_features', []))}")

    if args.mode == "mianyang":
        mode_mianyang(bundle)
    elif args.mode == "qianjincun":
        mode_qianjincun(bundle)
    elif args.mode == "anju":
        mode_anju(bundle)


if __name__ == "__main__":
    main()
