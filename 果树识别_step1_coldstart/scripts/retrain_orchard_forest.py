# -*- coding: utf-8 -*-
"""
果园 vs 林地重训脚本 (v1)
==========================
目标: 修复叙永果园模型"负样本漏天然林"的致命短板, 让模型能区分果园和林地。

改动点:
  1. 负样本补林地: WorldCover Tree cover(10) 加入负样本
     (正样本栅格化覆盖果园后, 剩余 Tree cover 即天然林)
  2. 复用已缓存的特征立方体 (features_xuyong.npy / feature_cube.npy), 不重新下载哨兵影像
  3. 训练 LightGBM + 遂宁推理 + 面积对比

输出:
  - outputs/orchard_forest_model.pkl
  - outputs/orchard_vs_forest_suining.tif / _prob.tif
  - outputs/retrain_report.json
"""
from __future__ import annotations

import os

# 修复 PROJ 数据库版本冲突: 全局 PROJ_LIB 被指向 PostgreSQL/PostGIS, 强制用 rasterio 自带 proj_data
os.environ["PROJ_LIB"] = r"C:\Users\lenovo\AppData\Roaming\Python\Python313\site-packages\rasterio\proj_data"

import json
import math
import time
from pathlib import Path

import numpy as np
import rasterio
import geopandas as gpd
from rasterio.warp import reproject, transform_bounds, Resampling
from rasterio.windows import from_bounds as window_from_bounds
from rasterio.features import rasterize
from rasterio.transform import from_origin
from scipy.ndimage import binary_erosion, binary_dilation
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import GroupKFold

WORK_DIR = Path(r"E:\工作相关\2026年\0624 待测试数据")
PROJ_DIR = WORK_DIR / "果树识别_step1_coldstart"
DATA_DIR = PROJ_DIR / "data"
OUTPUT_DIR = PROJ_DIR / "outputs"

SHP_PATH = WORK_DIR / "待训练数据水果" / "叙永水果成果数据" / "result.shp"
XY_FEAT = DATA_DIR / "xuyong_features" / "features_xuyong.npy"
XY_NAMES = DATA_DIR / "xuyong_features" / "band_names_xuyong.json"
SN_FEAT = DATA_DIR / "suining_features" / "feature_cube.npy"
SN_NAMES = DATA_DIR / "suining_features" / "band_names.json"
SN_TRANSFORM = DATA_DIR / "suining_features" / "transform.json"
WC_CACHE = DATA_DIR / "worldcover_xuyong_10m.tif"

XUYONG_BOUNDS_WGS84 = (105.37, 27.69, 105.67, 27.80)
WC_URL = ("https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
          "ESA_WorldCover_10m_2021_v200_N27E105_Map.tif")

LGB_PARAMS = dict(
    n_estimators=300, learning_rate=0.03, max_depth=5,
    num_leaves=31, subsample=0.7, colsample_bytree=0.6,
    reg_alpha=1.0, reg_lambda=2.0, min_child_samples=50,
    class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1,
)

# 负样本最大像元数 (控制正负比, 避免林地负样本过多拖慢训练)
MAX_NEG_SAMPLES = 300000


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def xy_transform_and_shape():
    """根据叙永 bbox 重新计算 UTM48N 网格的 transform 和 (h, w), 应与特征立方体一致。"""
    bbox_utm = transform_bounds("EPSG:4326", "EPSG:32648", *XUYONG_BOUNDS_WGS84)
    res = 10.0
    xmin = math.floor(bbox_utm[0] / res) * res
    ymax = math.ceil(bbox_utm[3] / res) * res
    xmax = math.ceil(bbox_utm[2] / res) * res
    ymin = math.floor(bbox_utm[1] / res) * res
    w = int((xmax - xmin) / res)
    h = int((ymax - ymin) / res)
    return from_origin(xmin, ymax, res, res), h, w


def download_worldcover() -> np.ndarray:
    """下载/读取叙永 WorldCover (N27E105) 并裁剪到 UTM48N 网格, 返回 (h, w) uint8。"""
    if WC_CACHE.exists():
        log("WorldCover: 已有缓存, 读取中...")
        with rasterio.open(WC_CACHE) as src:
            return src.read(1).astype("uint8")

    transform, h, w = xy_transform_and_shape()
    log(f"WorldCover: 下载 N27E105 并裁剪到 {h}x{w} @ UTM48N ...")
    out = np.zeros((h, w), dtype="uint8")
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_TIMEOUT="120"):
        with rasterio.open(WC_URL) as src:
            win = window_from_bounds(*XUYONG_BOUNDS_WGS84, transform=src.transform)
            win = win.round_lengths().round_offsets()
            wc_sub = src.read(1, window=win)
            win_transform = src.window_transform(win)
            reproject(
                source=wc_sub, destination=out,
                src_transform=win_transform, src_crs=src.crs,
                dst_transform=transform, dst_crs="EPSG:32648",
                resampling=Resampling.nearest,
            )
    profile = {
        "driver": "GTiff", "height": h, "width": w, "count": 1,
        "dtype": "uint8", "crs": "EPSG:32648", "transform": transform,
        "compress": "deflate", "tiled": True,
    }
    with rasterio.open(WC_CACHE, "w", **profile) as dst:
        dst.write(out, 1)
    log(f"WorldCover: 已缓存 -> {WC_CACHE.name}")
    return out


def rebuild_labels(wc: np.ndarray, transform, h: int, w: int):
    """重建标签: 正=叙永SHP果树, 负=WorldCover非树类别 + Tree cover(10)天然林。"""
    log("重建标签 (负样本补 Tree cover 林地)...")

    gdf = gpd.read_file(SHP_PATH)
    gdf_utm = gdf.to_crs("EPSG:32648")
    geoms = [(g, 1) for g in gdf_utm.geometry if g.is_valid and not g.is_empty]
    positive = rasterize(geoms, out_shape=(h, w), transform=transform,
                         fill=0, dtype="uint8", all_touched=True)
    pos_clean = binary_erosion(binary_dilation(positive > 0, iterations=2), iterations=2)
    pos_count = int(pos_clean.sum())
    log(f"  正样本(果园): {pos_count:,} px")

    # 关键改动: 负样本加入 Tree cover(10) —— 天然林
    # WorldCover: 10=Tree, 20=Shrub, 30=Grass, 40=Crop, 50=Built, 60=Bare, 70=Snow, 80=Water, 90=Wetland
    neg_cats = [10, 40, 50, 60, 70, 80, 90]  # 之前不含 10, 现补上
    neg_mask = np.isin(wc, neg_cats)
    neg_count = int(neg_mask.sum())
    log(f"  负样本(含Tree cover林地): {neg_count:,} px")

    y = np.full((h, w), -1, dtype="int8")
    y[pos_clean] = 1
    y[neg_mask & ~pos_clean] = 0
    valid = y >= 0
    log(f"  标签: 正={int((y == 1).sum()):,}, 负={int((y == 0).sum()):,}, 有效={int(valid.sum()):,}")
    return y, valid


def load_features():
    """加载叙永特征立方体 + 波段名。"""
    cube = np.load(XY_FEAT, mmap_mode="r")
    with open(XY_NAMES) as f:
        names = json.load(f)
    log(f"叙永特征: {cube.shape}, {len(names)} 波段")
    return cube, names


def train_lgb(features, y, valid):
    """训练 LightGBM, 空间分组 4 折验证 + 全量训练。"""
    H, W = y.shape
    F = features.shape[0]
    valid_idx = np.where(valid.ravel())[0]
    X_all = features.reshape(F, -1)[:, valid_idx].T
    y_true = y.ravel()[valid_idx]

    # 负样本下采样 (正样本全用)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    if len(neg_idx) > MAX_NEG_SAMPLES:
        rng = np.random.default_rng(42)
        neg_idx = rng.choice(neg_idx, MAX_NEG_SAMPLES, replace=False)
    sel = np.sort(np.concatenate([pos_idx, neg_idx]))
    X = X_all[sel]
    yt = y_true[sel]
    log(f"采样后: 正={int((yt == 1).sum()):,}, 负={int((yt == 0).sum()):,} (负样本cap={MAX_NEG_SAMPLES:,})")

    X = np.nan_to_num(X, nan=0)

    # 空间分组
    block_size = 80
    rows, cols = np.unravel_index(valid_idx[sel], (H, W))
    block_ids = (rows // block_size) * (W // block_size + 1) + (cols // block_size)

    gkf = GroupKFold(n_splits=4)
    f1s = []
    for fold, (tr, te) in enumerate(gkf.split(X, yt, block_ids)):
        m = LGBMClassifier(**LGB_PARAMS)
        m.fit(X[tr], yt[tr])
        pred = m.predict(X[te])
        f1 = f1_score(yt[te], pred, average="binary")
        f1s.append(f1)
        log(f"  Fold{fold + 1}: F1={f1:.4f}, train={len(tr):,}, test={len(te):,}")
    log(f"  平均F1: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

    log("全量训练...")
    final = LGBMClassifier(**LGB_PARAMS)
    final.fit(X, yt)

    import joblib
    model_path = OUTPUT_DIR / "orchard_forest_model.pkl"
    joblib.dump(final, model_path)
    log(f"模型已保存: {model_path}")

    top = np.argsort(final.feature_importances_)[-10:][::-1]
    log("特征重要性TOP10:")
    names = None
    with open(XY_NAMES) as f:
        names = json.load(f)
    for i, idx in enumerate(top):
        log(f"  {i + 1}. {names[idx]}: {final.feature_importances_[idx]:.4f}")

    return final


def predict_suining(model):
    """对遂宁全境推理, 返回 (pred, prob) 和面积。"""
    cube = np.load(SN_FEAT, mmap_mode="r")
    with open(SN_NAMES) as f:
        sn_names = json.load(f)
    with open(XY_NAMES) as f:
        xy_names = json.load(f)
    with open(SN_TRANSFORM) as f:
        tinfo = json.load(f)
    sn_transform = rasterio.transform.Affine(*tinfo["affine"])

    F, H, W = cube.shape
    sn_idx = {n: i for i, n in enumerate(sn_names)}
    feat_idx = [sn_idx[bn] for bn in xy_names if bn in sn_idx]
    missing = [bn for bn in xy_names if bn not in sn_idx]
    if missing:
        log(f"[WARN] {len(missing)} 个训练波段在遂宁缺失: {missing[:5]}")
    log(f"遂宁推理: {H}x{W}, 使用 {len(feat_idx)}/{F} 波段")

    prob = np.zeros((H, W), dtype="float32")
    pred = np.zeros((H, W), dtype="uint8")
    chunk_h = 500
    for rs in range(0, H, chunk_h):
        re = min(rs + chunk_h, H)
        chunk = cube[feat_idx, rs:re, :].reshape(len(feat_idx), -1).T
        chunk = np.nan_to_num(chunk, nan=0)
        prob[rs:re, :] = model.predict_proba(chunk)[:, 1].reshape(re - rs, W)
        pred[rs:re, :] = model.predict(chunk).reshape(re - rs, W)
        if rs % 1000 == 0:
            log(f"  推理进度 {rs}/{H}")

    bin_path = OUTPUT_DIR / "orchard_vs_forest_suining.tif"
    prob_path = OUTPUT_DIR / "orchard_vs_forest_suining_prob.tif"
    prof = {"driver": "GTiff", "height": H, "width": W, "count": 1,
            "dtype": "uint8", "crs": "EPSG:32648", "transform": sn_transform,
            "compress": "deflate", "tiled": True}
    with rasterio.open(bin_path, "w", **prof) as dst:
        dst.write(pred, 1)
        dst.set_band_description(1, "orchard_binary_1=orchard")
    prof["dtype"] = "float32"
    with rasterio.open(prob_path, "w", **prof) as dst:
        dst.write(prob, 1)
    log(f"推理栅格已保存: {bin_path.name}")

    total_px = int((pred > 0).sum())
    total_ha = total_px * 100 / 10000
    log(f"遂宁果园面积: {total_px:,} px = {total_ha:,.0f} ha")
    return pred, prob, total_ha


def main():
    log("=" * 60)
    log("果园 vs 林地重训 (补林地负样本)")
    log("=" * 60)

    # 1. WorldCover
    wc = download_worldcover()
    transform, h, w = xy_transform_and_shape()
    feat, names = load_features()
    if (h, w) != (feat.shape[1], feat.shape[2]):
        log(f"[FATAL] 网格不一致: 重算=({h},{w}), 特征=({feat.shape[1]},{feat.shape[2]})")
        return

    # 2. 标签
    y, valid = rebuild_labels(wc, transform, h, w)

    # 3. 训练
    model = train_lgb(feat, y, valid)

    # 4. 遂宁推理
    pred, prob, total_ha = predict_suining(model)

    # 5. 报告
    old_ha = 350625.4  # 旧值 (transfer_report.json)
    report = {
        "model": "LightGBM (补林地负样本)",
        "negative_classes": "WorldCover [10(Tree),40,50,60,70,80,90]",
        "old_suining_orchard_ha": old_ha,
        "new_suining_orchard_ha": round(total_ha, 1),
        "ratio": round(total_ha / old_ha, 4),
        "outputs": {
            "binary": str(OUTPUT_DIR / "orchard_vs_forest_suining.tif"),
            "probability": str(OUTPUT_DIR / "orchard_vs_forest_suining_prob.tif"),
        },
    }
    with open(OUTPUT_DIR / "retrain_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    log(f"报告: retrain_report.json")
    log(f"面积变化: {old_ha:,.0f} ha -> {total_ha:,.0f} ha (比值 {total_ha / old_ha:.3f})")
    log("完成")


if __name__ == "__main__":
    main()
