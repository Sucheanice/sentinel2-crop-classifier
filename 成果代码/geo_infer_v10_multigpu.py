#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生产 v10 多 GPU 并行推理命令行入口（不改动现有 main.py 服务）。

原理：把 Context-Core 粗尺度窗口 + Fine-Tile 细尺度窗口按轮询均分给多张 GPU，
每张 GPU 各自加载一次模型、独立处理自己分到的窗口（纯数据并行），
结果写各自 pickle 中间文件，主进程汇总后做全局去重 + 矢量化（与 run_sliding_window 完全一致）。

用法示例（GPU 0 + 6）：
  python geo_infer_v10_multigpu.py \
    --input /data3/sam3_project/test_data/tif/baozhucun.tif \
    --output /data3/sam3_project/test_data/result_baozhucun_v10/baozhucun_v10 \
    --prompt farmland \
    --gpus 0,6 \
    --checkpoint /data4/sam3_project/sam3_train/data/runs/finetune_v3/checkpoints/checkpoint_25.pt

依赖：与 geo_infer_v10.py 同目录；SAM3 源码在 CODE_ROOT（可用环境变量 SAM3_CODE_ROOT 覆盖）。
"""
import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

CODE_ROOT = Path(os.environ.get("SAM3_CODE_ROOT", "/data4/sam3_project/sam3_train/Code/sam3-main"))
SERVICE_DIR = CODE_ROOT / "sam3_service"
sys.path.insert(0, str(SERVICE_DIR))
sys.path.insert(0, str(CODE_ROOT))

from geo_infer_v10 import (  # noqa: E402
    _infer_tile,
    clean_and_vectorize_instances,
    deduplicate_instances,
)


# ════════════════════════════════════════════
# 窗口生成（对齐 _process_context_windows / _process_fine_tiles 的窗口遍历逻辑）
# ════════════════════════════════════════════

def build_coarse_windows(img_h, img_w, tile_size, core_size, core_overlap):
    core_pad = (tile_size - core_size) // 2
    core_step = max(1, core_size - core_overlap)
    wins = []
    for row_off in range(0, img_h, core_step):
        for col_off in range(0, img_w, core_step):
            ctx_col = max(0, col_off - core_pad)
            ctx_row = max(0, row_off - core_pad)
            ctx_w = min(tile_size, img_w - ctx_col)
            ctx_h = min(tile_size, img_h - ctx_row)
            wins.append({
                "scale": "coarse",
                "ctx_col": ctx_col, "ctx_row": ctx_row,
                "ctx_w": ctx_w, "ctx_h": ctx_h,
                "tile_size": tile_size,
                "core_c0": col_off - ctx_col,
                "core_r0": row_off - ctx_row,
                "core_c1": (col_off - ctx_col) + min(core_size, img_w - col_off),
                "core_r1": (row_off - ctx_row) + min(core_size, img_h - row_off),
            })
    return wins


def build_fine_windows(img_h, img_w, fine_tile_size, fine_core_size, fine_core_overlap):
    if fine_tile_size <= 0:
        return []
    if fine_core_size > fine_tile_size:
        fine_core_size = fine_tile_size
    core_pad = (fine_tile_size - fine_core_size) // 2
    core_step = max(1, fine_core_size - fine_core_overlap)
    wins = []
    for row_off in range(0, img_h, core_step):
        for col_off in range(0, img_w, core_step):
            ctx_col = max(0, col_off - core_pad)
            ctx_row = max(0, row_off - core_pad)
            ctx_w = min(fine_tile_size, img_w - ctx_col)
            ctx_h = min(fine_tile_size, img_h - ctx_row)
            wins.append({
                "scale": "fine",
                "ctx_col": ctx_col, "ctx_row": ctx_row,
                "ctx_w": ctx_w, "ctx_h": ctx_h,
                "tile_size": fine_tile_size,
                "core_c0": col_off - ctx_col,
                "core_r0": row_off - ctx_row,
                "core_c1": (col_off - ctx_col) + min(fine_core_size, img_w - col_off),
                "core_r1": (row_off - ctx_row) + min(fine_core_size, img_h - row_off),
            })
    return wins


def _prefilter_windows(src, win_metas, min_valid_ratio=0.05, scale=16):
    """跳过几乎无有效内容的窗口（nodata 或全黑），省掉无效窗口的完整推理。

    读取下采样有效区掩膜（dataset_mask）与下采样亮度图，按窗口统计
    「有数据且非黑」的像素占比，低于阈值直接剔除，不送进 worker。
    """
    ds_h, ds_w = max(1, src.height // scale), max(1, src.width // scale)
    valid_low = src.dataset_mask(out_shape=(ds_h, ds_w)) > 0
    # 亮度：RGB 均值（uint16 原始值，全黑≈0）
    lum_low = None
    try:
        arr = src.read(indexes=list(range(1, min(src.count, 3) + 1)),
                       out_shape=(min(src.count, 3), ds_h, ds_w)).astype(np.float32)
        if arr.shape[0] == 1:
            lum_low = arr[0]
        else:
            lum_low = 0.299 * arr[0] + 0.587 * arr[1] + 0.114 * arr[2]
    except Exception:
        lum_low = None

    kept, skipped = [], 0
    for wm in win_metas:
        r0 = max(0, wm["ctx_row"] // scale)
        c0 = max(0, wm["ctx_col"] // scale)
        r1 = min(ds_h, -(-(wm["ctx_row"] + wm["ctx_h"]) // scale))
        c1 = min(ds_w, -(-(wm["ctx_col"] + wm["ctx_w"]) // scale))
        if r1 <= r0 or c1 <= c0:
            skipped += 1
            continue
        sub_valid = valid_low[r0:r1, c0:c1]
        if float(sub_valid.mean()) < min_valid_ratio:
            skipped += 1
            continue
        if lum_low is not None:
            sub_lum = lum_low[r0:r1, c0:c1][sub_valid]
            # 有效区内亮度 > 20（uint16）的像素占比
            if sub_lum.size and float((sub_lum > 20).mean()) < min_valid_ratio:
                skipped += 1
                continue
        kept.append(wm)
    return kept, skipped


def filter_window_instances(tile_insts, wm, params):
    """对 _infer_tile 返回的实例做 core ownership + 尺度特定过滤（与 v10 逐窗口逻辑一致）。"""
    kept = []
    for inst in tile_insts:
        mask = inst["mask"]
        yy, xx = np.where(mask)
        if len(yy) == 0:
            continue
        if wm["scale"] == "fine":
            if int(mask.sum()) < 10:
                continue
        cy, cx = yy.mean(), xx.mean()
        if not (wm["core_c0"] <= cx < wm["core_c1"] and wm["core_r0"] <= cy < wm["core_r1"]):
            continue
        if wm["scale"] == "coarse":
            nz = int(mask.sum())
            if nz / (wm["ctx_w"] * wm["ctx_h"]) > params["max_context_area_ratio"]:
                continue
            min_x, max_x = xx.min(), xx.max()
            min_y, max_y = yy.min(), yy.max()
            margin = 2
            gx0 = wm["ctx_col"] + min_x
            gx1 = wm["ctx_col"] + max_x
            gy0 = wm["ctx_row"] + min_y
            gy1 = wm["ctx_row"] + max_y
            edge_touches = sum([
                gx0 <= margin, gx1 >= params["img_w"] - margin,
                gy0 <= margin, gy1 >= params["img_h"] - margin,
            ])
            if edge_touches > params["max_context_edge_touches"]:
                continue
        kept.append(inst)
    return kept


# ════════════════════════════════════════════
# 子进程入口（每 GPU 一个）
# ════════════════════════════════════════════

def _worker(gpu_id, input_path, checkpoint, win_metas, params, out_path):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    out_path = Path(out_path)
    sys.path.insert(0, str(SERVICE_DIR))
    sys.path.insert(0, str(CODE_ROOT))

    import torch
    from geo_infer_v10 import _infer_tile as infer_tile
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model = build_sam3_image_model(
        checkpoint_path=str(checkpoint),
        load_from_HF=False,
        device="cuda",
        eval_mode=True,
    )
    processor_kwargs = {}
    if params.get("resolution"):
        processor_kwargs["resolution"] = int(params["resolution"])
    if params.get("confidence_threshold"):
        processor_kwargs["confidence_threshold"] = float(params["confidence_threshold"])
    processor = Sam3Processor(model, **processor_kwargs)

    coarse_insts = []
    fine_insts = []

    with rasterio.open(input_path) as src:
        transform = src.transform
        px_w, px_h = abs(transform.a), abs(transform.e)
        mean_px = (px_w + px_h) / 2.0
        crs = src.crs
        if crs is not None and crs.is_projected and mean_px > 0:
            slic_min_distance_px = max(1.0, float(params["slic_min_distance_m"]) / mean_px)
        else:
            slic_min_distance_px = max(1.0, params["tile_size"] * 0.02)

        coarse_n = max(4, int(round(params["slic_segments"] * params["tile_size"] ** 2 / (1008 * 1008))))
        fine_n = max(4, int(round(params["slic_segments"] * params["fine_tile_size"] ** 2 / (1008 * 1008)))) \
            if params["fine_tile_size"] > 0 else 0

        for wm in win_metas:
            win = Window(wm["ctx_col"], wm["ctx_row"], wm["ctx_w"], wm["ctx_h"])
            if wm["scale"] == "coarse":
                slic_enabled = bool(params["slic_points"])
                slic_n = coarse_n
            else:
                slic_enabled = bool(params["slic_points_fine"])
                slic_n = fine_n

            tile_insts = infer_tile(
                processor, src, win, wm["tile_size"], params["prompt"],
                params["confidence_threshold"],
                stretch=params["stretch"],
                geometric_refine=params["geometric_refine"],
                refine_low=params["refine_low"],
                refine_high=params["refine_high"],
                max_refine_boxes=params["max_refine_boxes"],
                slic_enabled=slic_enabled,
                slic_n_segments=slic_n,
                slic_compactness=params["slic_compactness"],
                slic_min_distance_px=slic_min_distance_px,
                slic_max_points=params["slic_max_points"],
                slic_min_brightness=params["slic_min_brightness"],
                farm_poly=None,
            )
            kept = filter_window_instances(tile_insts, wm, params)
            if wm["scale"] == "coarse":
                coarse_insts.extend(kept)
            else:
                fine_insts.extend(kept)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({"coarse": coarse_insts, "fine": fine_insts}, f, protocol=4)
    print(f"[GPU {gpu_id}] done: coarse={len(coarse_insts)} fine={len(fine_insts)}", flush=True)


# ════════════════════════════════════════════
# 主流程
# ════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="SAM3 生产 v10 多 GPU 并行推理")
    ap.add_argument("--input", required=True, help="输入 GeoTIFF 路径")
    ap.add_argument("--output", required=True, help="输出 shp 路径（不含 .shp 后缀）")
    ap.add_argument("--prompt", default="farmland")
    ap.add_argument("--gpus", default="0,6", help="逗号分隔的 GPU 编号，如 0,6")
    ap.add_argument("--skip-empty-windows", type=int, default=1, help="跳过 nodata/全黑无效窗口（1=开）")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--confidence-threshold", type=float, default=0.3)
    ap.add_argument("--resolution", type=int, default=1008)
    # 尺度参数
    ap.add_argument("--tile-size", type=int, default=1008)
    ap.add_argument("--core-size", type=int, default=672)
    ap.add_argument("--core-overlap", type=int, default=168)
    ap.add_argument("--fine-tile-size", type=int, default=588)
    ap.add_argument("--fine-overlap", type=float, default=0.25)
    ap.add_argument("--fine-core-size", type=int, default=448)
    ap.add_argument("--fine-core-overlap", type=int, default=112)
    ap.add_argument("--stretch", type=int, default=0)
    ap.add_argument("--max-context-edge-touches", type=int, default=2)
    ap.add_argument("--max-context-area-ratio", type=float, default=0.25)
    # 精修
    ap.add_argument("--geometric-refine", type=int, default=1)
    ap.add_argument("--refine-low", type=float, default=0.23)
    ap.add_argument("--refine-high", type=float, default=0.60)
    ap.add_argument("--max-refine-boxes", type=int, default=20)
    # SLIC 点提示
    ap.add_argument("--slic-points", type=int, default=1)
    ap.add_argument("--slic-segments", type=int, default=64)
    ap.add_argument("--slic-compactness", type=float, default=10.0)
    ap.add_argument("--slic-min-distance-m", type=float, default=3.0)
    ap.add_argument("--slic-points-fine", type=int, default=0)
    ap.add_argument("--slic-max-points", type=int, default=128)
    ap.add_argument("--slic-min-brightness", type=float, default=70.0)
    # 矢量化
    ap.add_argument("--fill-holes", type=int, default=1)
    ap.add_argument("--hole-area-pixels", type=int, default=268)
    ap.add_argument("--simplify-tolerance", type=float, default=1.0)
    ap.add_argument("--mask-smooth-sigma", type=float, default=1.0)
    ap.add_argument("--remove-overlap", type=int, default=1)
    ap.add_argument("--min-area-m2", type=float, default=67.0)
    ap.add_argument("--dark-p10-threshold", type=float, default=25.0, help="深色林地过滤阈值（归一化亮度 p10，>0 启用）")
    ap.add_argument("--merge-touching", type=int, default=1)
    ap.add_argument("--merge-fill-hole-m2", type=float, default=0.0)
    ap.add_argument("--ndvi-threshold", type=float, default=0.0, help="NDVI 林地过滤阈值（>0 启用）")
    ap.add_argument("--max-vertices", type=int, default=200, help="顶点上限（>0 启用；仅对超过上限的多边形做共线点合并）")
    ap.add_argument("--regularize", type=int, default=0, help="边界正交化规则化（1=开；保拓扑，只规整直角/合并共线点）")
    ap.add_argument("--regularize-tolerance", type=float, default=2.0, help="规则化前 DP 简化容差(米)")
    args = ap.parse_args()

    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    input_path = args.input
    out_shp = Path(args.output)
    output_dir = out_shp.parent
    output_name = out_shp.name
    output_dir.mkdir(parents=True, exist_ok=True)

    params = {
        "prompt": args.prompt,
        "confidence_threshold": args.confidence_threshold,
        "resolution": args.resolution,
        "tile_size": args.tile_size,
        "fine_tile_size": args.fine_tile_size,
        "stretch": bool(args.stretch),
        "geometric_refine": bool(args.geometric_refine),
        "refine_low": args.refine_low,
        "refine_high": args.refine_high,
        "max_refine_boxes": args.max_refine_boxes,
        "slic_points": args.slic_points,
        "slic_segments": args.slic_segments,
        "slic_compactness": args.slic_compactness,
        "slic_min_distance_m": args.slic_min_distance_m,
        "slic_points_fine": args.slic_points_fine,
        "slic_max_points": args.slic_max_points,
        "slic_min_brightness": args.slic_min_brightness,
        "max_context_edge_touches": args.max_context_edge_touches,
        "max_context_area_ratio": args.max_context_area_ratio,
    }

    started = time.perf_counter()
    with rasterio.open(input_path) as src:
        profile = src.profile
        img_w, img_h = src.width, src.height
    params["img_w"] = img_w
    params["img_h"] = img_h

    coarse_wins = build_coarse_windows(img_h, img_w, args.tile_size, args.core_size, args.core_overlap)
    fine_wins = build_fine_windows(img_h, img_w, args.fine_tile_size, args.fine_core_size, args.fine_core_overlap)
    print(f"窗口统计: coarse={len(coarse_wins)} fine={len(fine_wins)} 影像={img_w}x{img_h}", flush=True)

    # 跳过无效窗口（nodata/全黑），省掉空窗口推理（问题1 提速方案4）
    if bool(args.skip_empty_windows):
        with rasterio.open(input_path) as src:
            coarse_wins, sk_c = _prefilter_windows(src, coarse_wins)
            fine_wins, sk_f = _prefilter_windows(src, fine_wins)
        if sk_c or sk_f:
            print(f"跳过无效窗口: coarse -{sk_c} fine -{sk_f} "
                  f"→ coarse={len(coarse_wins)} fine={len(fine_wins)}", flush=True)

    # 轮询分配窗口给各 GPU
    coarse_split = [[] for _ in gpus]
    fine_split = [[] for _ in gpus]
    for i, w in enumerate(coarse_wins):
        coarse_split[i % len(gpus)].append(w)
    for i, w in enumerate(fine_wins):
        fine_split[i % len(gpus)].append(w)

    tmp_dir = output_dir / ".tmp_mp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    if len(gpus) == 1:
        # 单卡：直接主进程跑，避免 spawn 开销
        out_files = []
        for idx, gid in enumerate(gpus):
            op = tmp_dir / f"worker_{idx}.pkl"
            win_metas = coarse_split[idx] + fine_split[idx]
            _worker(gid, input_path, args.checkpoint, win_metas, params, op)
            out_files.append(op)
    else:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        procs = []
        out_files = []
        for idx, gid in enumerate(gpus):
            op = tmp_dir / f"worker_{idx}.pkl"
            out_files.append(op)
            win_metas = coarse_split[idx] + fine_split[idx]
            p = ctx.Process(
                target=_worker,
                args=(gid, input_path, args.checkpoint, win_metas, params, str(op)),
            )
            p.start()
            procs.append(p)
            print(f"启动 worker: GPU {gid} -> {len(win_metas)} 窗口", flush=True)
        for p in procs:
            p.join()

    # 汇总各 worker 结果
    coarse_all, fine_all = [], []
    for op in out_files:
        with open(op, "rb") as f:
            d = pickle.load(f)
        coarse_all.extend(d["coarse"])
        fine_all.extend(d["fine"])
    print(f"汇总: coarse={len(coarse_all)} fine={len(fine_all)}", flush=True)

    # 尺度内去重（与 run_sliding_window 一致）
    if fine_all:
        coarse_ded = deduplicate_instances(coarse_all, match_thresh=0.50)
        fine_ded = deduplicate_instances(fine_all, match_thresh=0.35)
        combined = coarse_ded + fine_ded
        meta = {"coarse": coarse_ded, "fine": fine_ded}
    else:
        coarse_ded = deduplicate_instances(coarse_all, match_thresh=0.50)
        combined = coarse_ded
        meta = None
    print(f"去重后: coarse={len(coarse_ded)} fine={len(fine_ded) if fine_all else 0} combined={len(combined)}", flush=True)

    shp_path = clean_and_vectorize_instances(
        combined, profile,
        output_dir=output_dir,
        output_name=output_name,
        min_area_m2=args.min_area_m2,
        fill_holes=bool(args.fill_holes),
        hole_area_pixels=args.hole_area_pixels,
        simplify_tolerance=args.simplify_tolerance,
        remove_overlap=bool(args.remove_overlap),
        mask_smooth_sigma=args.mask_smooth_sigma,
        coarse_instances=meta["coarse"] if meta else None,
        fine_instances=meta["fine"] if meta else None,
        raster_path=input_path,
        dark_p10_threshold=args.dark_p10_threshold,
        clip_shp=None,
        merge_touching=bool(args.merge_touching),
        merge_fill_hole_m2=args.merge_fill_hole_m2,
        ndvi_threshold=args.ndvi_threshold,
        max_vertices=args.max_vertices,
        regularize=args.regularize,
        regularize_tolerance=args.regularize_tolerance,
    )

    print(f"完成: {shp_path}", flush=True)
    print(f"总耗时: {time.perf_counter() - started:.2f}s", flush=True)


if __name__ == "__main__":
    main()
