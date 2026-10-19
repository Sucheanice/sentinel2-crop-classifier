import shutil
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from config import (
    ALLOWED_ORIGINS,
    CHECKPOINT_PATH,
    MAX_UPLOAD_BYTES,
    RESULT_DIR,
    UPLOAD_DIR,
)
from inference import sam3_service

# 实验记录用，切换版本时改下面两行
_GEO_INFER_VERSION = "geo_infer_v10"
from geo_infer_v10 import (
    auto_tune_parameters,
    clean_and_vectorize_instances,
    run_sliding_window,
)


for directory in (UPLOAD_DIR, RESULT_DIR):
    directory.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(_: FastAPI):
    sam3_service.load()
    yield


app = FastAPI(
    title="SAM3 Image Segmentation API",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=ALLOWED_ORIGINS != ["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)
app.mount("/files/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
app.mount("/files/results", StaticFiles(directory=RESULT_DIR), name="results")


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_ready": sam3_service.is_ready,
        "checkpoint": str(CHECKPOINT_PATH),
    }


@app.post("/api/segment")
async def segment(
    request: Request,
    image: UploadFile = File(...),
    prompt: str = Form(...),
    display_mode: str = Form("mask"),
) -> dict:
    prompt = prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    if display_mode not in {"mask", "detail"}:
        raise HTTPException(status_code=400, detail="Invalid display mode")

    suffix = Path(image.filename or "").suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}:
        raise HTTPException(status_code=400, detail="Unsupported image format")

    upload_name = f"{uuid.uuid4().hex}{suffix}"
    upload_path = UPLOAD_DIR / upload_name
    size = 0

    try:
        with upload_path.open("wb") as target:
            while chunk := await image.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail="Image exceeds the upload size limit",
                    )
                target.write(chunk)

        with Image.open(upload_path) as uploaded_image:
            uploaded_image.verify()

        inference_path = upload_path
        preview_name = upload_name
        if suffix in {".tif", ".tiff"}:
            preview_name = f"{Path(upload_name).stem}_preview.png"
            inference_path = UPLOAD_DIR / preview_name
            # 用 rasterio 正确读取 TIFF，百分位拉伸到 uint8 RGB
            import rasterio  # noqa: E402
            with rasterio.open(upload_path) as src:
                bands = min(src.count, 3)
                data = src.read(indexes=list(range(1, bands + 1)))
            data = np.nan_to_num(data, nan=0.0)
            if bands == 1:
                data = np.repeat(data, 3, axis=0)
            rgb = np.empty((data.shape[1], data.shape[2], 3), dtype=np.uint8)
            for b in range(3):
                band = data[b].astype(np.float64)
                v = band[np.isfinite(band)]
                if v.size == 0:
                    rgb[:, :, b] = 0; continue
                lo, hi = np.percentile(v, (2, 98))
                if hi <= lo:
                    hi, lo = float(v.max()), float(v.min())
                if hi <= lo:
                    rgb[:, :, b] = 0; continue
                rgb[:, :, b] = np.clip((band - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
            Image.fromarray(rgb).save(inference_path, format="PNG")

        result = sam3_service.infer(
            inference_path,
            prompt,
            image.filename or upload_name,
            display_mode,
        )
    except HTTPException:
        upload_path.unlink(missing_ok=True)
        raise
    except (UnidentifiedImageError, OSError) as exc:
        upload_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Invalid image file") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc
    finally:
        await image.close()

    base_url = str(request.base_url).rstrip("/")
    result_path = result.pop("result_path")
    result.pop("masks", None)  # tensor 无法序列化，evaluate 端点单独处理
    return {
        "success": True,
        "prompt": prompt,
        "display_mode": display_mode,
        "result_filename": result_path.name,
        "original_image_url": f"{base_url}/files/uploads/{preview_name}",
        "result_image_url": f"{base_url}/files/results/{result_path.name}",
        **result,
    }


@app.post("/api/evaluate")
async def evaluate(
    request: Request,
    image: UploadFile = File(...),
    annotation: UploadFile = File(...),
    prompt: str = Form(...),
    display_mode: str = Form("mask"),
) -> dict:
    """
    推理 + 精度评估一体。
    image:  待识别的图像
    annotation: COCO JSON 标注
    """
    import json
    from collections import defaultdict
    from pycocotools import mask as mask_util

    prompt = prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    suffix = Path(image.filename or "").suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}:
        raise HTTPException(status_code=400, detail="Unsupported image format")

    upload_name = f"{uuid.uuid4().hex}{suffix}"
    upload_path = UPLOAD_DIR / upload_name
    try:
        with upload_path.open("wb") as target:
            while chunk := await image.read(1024 * 1024):
                if len(chunk) > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413)
                target.write(chunk)
    except HTTPException:
        raise

    ann_bytes = await annotation.read()
    ann_data = json.loads(ann_bytes)

    inference_path = upload_path
    preview_name = upload_name
    if suffix in {".tif", ".tiff"}:
        preview_name = f"{Path(upload_name).stem}_preview.png"
        inference_path = UPLOAD_DIR / preview_name
        import rasterio
        with rasterio.open(upload_path) as src:
            bands = min(src.count, 3)
            data = src.read(indexes=list(range(1, bands + 1)))
        data = np.nan_to_num(data, nan=0.0)
        if bands == 1:
            data = np.repeat(data, 3, axis=0)
        rgb = np.empty((data.shape[1], data.shape[2], 3), dtype=np.uint8)
        for b in range(3):
            band = data[b].astype(np.float64)
            v = band[np.isfinite(band)]
            if v.size == 0: rgb[:, :, b] = 0; continue
            lo, hi = np.percentile(v, (2, 98))
            if hi <= lo: hi, lo = float(v.max()), float(v.min())
            if hi <= lo: rgb[:, :, b] = 0; continue
            rgb[:, :, b] = np.clip((band - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
        Image.fromarray(rgb).save(inference_path, format="PNG")

    try:
        result = sam3_service.infer(inference_path, prompt, image.filename or upload_name, display_mode)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    cat_map = {c["id"]: c["name"] for c in ann_data.get("categories", [])}
    img_anns = []
    for img_info in ann_data.get("images", []):
        for ann in ann_data.get("annotations", []):
            if ann["image_id"] == img_info["id"]:
                img_anns.append(ann)

    pred_masks_raw = result.pop("masks", None)
    per_class = defaultdict(lambda: {"tp": 0, "fn": 0, "iou_sum": 0.0})

    for ann in img_anns:
        rle = ann["segmentation"]
        if isinstance(rle["counts"], str):
            rle["counts"] = rle["counts"].encode("utf-8")
        gt_mask = mask_util.decode(rle)
        cat_name = cat_map.get(ann["category_id"], str(ann["category_id"]))

        if pred_masks_raw is None or len(pred_masks_raw) == 0:
            per_class[cat_name]["fn"] += 1
            continue

        best_iou = 0.0
        for pm in pred_masks_raw:
            pm_np = pm.squeeze().detach().cpu().numpy() if hasattr(pm, "detach") else np.array(pm)
            pm_np = pm_np.squeeze()
            if pm_np.shape != gt_mask.shape:
                mi = Image.fromarray((pm_np > 0.5).astype(np.uint8) * 255)
                mi = mi.resize((gt_mask.shape[1], gt_mask.shape[0]), Image.Resampling.NEAREST)
                pm_np = (np.array(mi) > 0).astype(np.uint8)
            else:
                pm_np = (pm_np > 0.5).astype(np.uint8)
            inter = np.logical_and(pm_np, gt_mask).sum()
            union = np.logical_or(pm_np, gt_mask).sum()
            iou = float(inter / union) if union > 0 else 0.0
            if iou > best_iou:
                best_iou = iou

        if best_iou >= 0.5:
            per_class[cat_name]["tp"] += 1
            per_class[cat_name]["iou_sum"] += best_iou
        else:
            per_class[cat_name]["fn"] += 1

    metrics = {}
    for cn, v in sorted(per_class.items()):
        tp, fn = v["tp"], v["fn"]
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        avg_iou = v["iou_sum"] / tp if tp > 0 else 0.0
        metrics[cn] = {"tp": tp, "fn": fn, "avg_iou": round(avg_iou, 4), "recall": round(recall, 4)}

    base_url = str(request.base_url).rstrip("/")
    result_path = result.pop("result_path")
    return {
        "success": True, "prompt": prompt,
        "result_filename": result_path.name,
        "original_image_url": f"{base_url}/files/uploads/{preview_name}",
        "result_image_url": f"{base_url}/files/results/{result_path.name}",
        "object_count": result.get("object_count", 0),
        "evaluation": metrics,
        "elapsed_seconds": result.get("elapsed_seconds", 0),
        "geo_infer_version": _GEO_INFER_VERSION,
    }

@app.post("/api/segment_geo")
async def segment_geo(
    request: Request,
    image: UploadFile = File(...),
    prompt: str = Form(...),
    display_mode: str = Form("mask"),
    use_sliding: int = Form(1),
    tile_size: int = Form(1008),
    overlap: int = Form(0),
    core_size: int = Form(672),
    stretch: int = Form(1),
    max_context_area_ratio: float = Form(0.25),
    fine_tile_size: int = Form(0),
    fine_overlap: float = Form(0.75),
    fine_core_size: int = Form(384),
    fine_core_overlap: int = Form(96),
    core_overlap: int = Form(168),
    geometric_refine: int = Form(1),
    refine_low: float = Form(0.23),
    refine_high: float = Form(0.60),
    confidence_threshold: float = Form(-1),
    max_refine_boxes: int = Form(20),
    mask_smooth_sigma: float = Form(1.0),
    min_area_pixels: int = Form(0),
    min_area_m2: float = Form(0),
    fill_holes: int = Form(1),
    hole_area_pixels: int = Form(256),
    simplify_tolerance: float = Form(0.5),
    remove_overlap: int = Form(1),
    max_context_edge_touches: int = Form(2),
    slic_points: int = Form(1),
    slic_segments: int = Form(64),
    slic_compactness: float = Form(10),
    slic_min_distance_m: float = Form(3.0),
    slic_points_fine: int = Form(0),
    slic_max_points: int = Form(128),
    slic_min_brightness: float = Form(0),
    dark_p10_threshold: float = Form(0),
    restrict_shp: str = Form(""),
    merge_touching: int = Form(1),
    merge_fill_hole_m2: float = Form(0),
    auto_tune: int = Form(0),
) -> dict:
    """
    大图滑动窗口识别 + Shapefile 导出。

    Form 参数:
      - use_sliding:        是否启用滑动窗口（1=自动/强制, 0=禁用）
      - tile_size:          窗口大小（默认 1008）
      - overlap:            窗口重叠像素（默认 128）
      - min_area_pixels:    最小面积阈值（默认 0=不剔除）
      - fill_holes:         是否填补内部空洞（1=是，0=否）
      - hole_area_pixels:   空洞填补面积阈值（默认 256）
      - simplify_tolerance: 多边形简化容差，CRS单位如米（默认 0.5，0=不平滑）
      - remove_overlap:     是否去除实例间重叠（1=是，0=否）
      - max_context_edge_touches: 允许 mask 触碰大图边界的条数上限（默认 2，只删贴 ≥3 条的病态 mask）
      - max_refine_boxes:    每窗口 box 精修数量上限（默认 20，0=不限制；每个框一次完整解码，低置信阈值下框数可能上百）
      - confidence_threshold: 覆盖置信度阈值（默认 -1 = 用服务启动默认值；>0 仅本次请求生效，请求结束自动恢复）
      - slic_points:        是否在粗尺度启用 SLIC 超像素自适应点提示（1=是, 0=否, 默认 1）
      - slic_segments:      SLIC 超像素基准数量（相对 1008² 瓦片，默认 64，按瓦片面积缩放）
      - slic_compactness:   SLIC 紧凑度（默认 10）
      - slic_min_distance_m: 超像素内点距边界最小距离（米，默认 3.0；非投影 CRS 回退像素启发式）
      - slic_points_fine:   是否在细尺度也启用 SLIC 点提示（默认 0）
      - slic_max_points:    每窗口点提示数量上限（默认 128）
      - slic_min_brightness: SLIC 超像素亮度门（原始影像亮度 0-255，默认 0=关；低于此值的深色超像素
                            不放点，针对深色树林误检，建议 70）
      - dark_p10_threshold: 深色 polygon 过滤（原始影像亮度 0-255，默认 0=关；最终 polygon 内
                            亮度 p10 低于此值删除，针对深色树林误检，建议 69）
      - restrict_shp:       农田范围约束 shp 路径（服务器上的路径，空=不约束）。slic/context/fine
                            只在此范围内识别：范围外窗口跳过、不放点、mask 过滤，其他判定不变
      - merge_touching:     相接/相交 polygon 按连通组合并（默认 1=开；0=关=v9 行为）
      - merge_fill_hole_m2: 合并结果补洞（<0 不补；0=补全部洞；>0 只补 ≤ 该面积 m² 的洞）
      - auto_tune:          是否自动搜参（1=是, 0=否, 默认 0）。注意：每组合=整幅图一次完整
                            滑窗推理，总耗时 ×组合数，仅建议小图/服务器使用；强制启用细尺度
    """
    svc = sam3_service
    prompt = prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    suffix = Path(image.filename or "").suffix.lower()
    if suffix not in {".tif", ".tiff"}:
        raise HTTPException(status_code=400, detail="Only GeoTIFF is supported for this endpoint")
    if restrict_shp and not Path(restrict_shp).exists():
        raise HTTPException(status_code=400, detail=f"restrict_shp 文件不存在: {restrict_shp}")

    # 保存上传
    upload_name = f"{uuid.uuid4().hex}{suffix}"
    upload_path = UPLOAD_DIR / upload_name
    size = 0
    try:
        with upload_path.open("wb") as target:
            while chunk := await image.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413)
                target.write(chunk)
    except HTTPException:
        upload_path.unlink(missing_ok=True)
        raise

    default_thresh = None
    try:
        svc.load()
        # 每次请求可覆盖置信度阈值（-1 = 服务启动默认），请求结束 finally 恢复，避免跨请求串扰
        default_thresh = svc.processor.confidence_threshold
        if confidence_threshold > 0:
            svc.processor.set_confidence_threshold(confidence_threshold)
            print(f"请求覆盖置信度阈值: {default_thresh} → {confidence_threshold}", flush=True)
        started_at = time.time()

        import rasterio
        with rasterio.open(upload_path) as src:
            img_w, img_h = src.width, src.height
            profile = src.profile

        needs_sliding = bool(use_sliding) and (img_w > tile_size or img_h > tile_size)
        thresh = svc.processor.confidence_threshold

        if needs_sliding:
            print(f"滑动窗口模式: {img_w}×{img_h}, tile={tile_size}, overlap={overlap}")
            if bool(auto_tune):
                tune = auto_tune_parameters(
                    svc.processor, upload_path, prompt,
                    fine_tile_size=fine_tile_size if fine_tile_size > 0 else 512,
                    slic_points=slic_points, slic_segments=slic_segments,
                    slic_compactness=slic_compactness,
                    slic_min_distance_m=slic_min_distance_m,
                    slic_max_points=slic_max_points,
                    slic_min_brightness=slic_min_brightness,
                    restrict_shp=restrict_shp or None,
                )
                if tune is None:
                    raise HTTPException(status_code=500, detail="Auto-tune failed (all combos errored)")
                instances, profile, meta = tune["instances"], tune["profile"], tune["meta"]
                tune_report = tune["report"]
            else:
                instances, profile, meta = run_sliding_window(
                    svc.processor, upload_path, prompt,
                    tile_size=tile_size,
                    confidence_threshold=thresh, core_size=core_size, stretch=bool(stretch),
                    max_context_edge_touches=max_context_edge_touches,
                    max_context_area_ratio=max_context_area_ratio,
                    fine_tile_size=fine_tile_size, fine_overlap=fine_overlap,
                    fine_core_size=fine_core_size, fine_core_overlap=fine_core_overlap,
                    geometric_refine=bool(geometric_refine), refine_low=refine_low, refine_high=refine_high,
                    max_refine_boxes=max_refine_boxes,
                    core_overlap=core_overlap, mask_smooth_sigma=mask_smooth_sigma,
                    slic_points=slic_points, slic_segments=slic_segments,
                    slic_compactness=slic_compactness, slic_min_distance_m=slic_min_distance_m,
                    slic_points_fine=slic_points_fine, slic_max_points=slic_max_points,
                    slic_min_brightness=slic_min_brightness,
                    restrict_shp=restrict_shp or None,
                )
                tune_report = None
            total_pixels = sum(inst["mask"].sum() for inst in instances)
        else:
            if bool(auto_tune):
                print("auto_tune 仅滑窗模式生效，已忽略", flush=True)
            meta = None
            tune_report = None
            # 单次推理，保留实例级 mask
            inference_path = UPLOAD_DIR / f"{Path(upload_name).stem}_preview.png"
            with rasterio.open(upload_path) as src:
                bands = min(src.count, 3)
                data = src.read(indexes=list(range(1, bands + 1)))
            data = np.nan_to_num(data, nan=0.0)
            if bands == 1:
                data = np.repeat(data, 3, axis=0)
            rgb = np.empty((data.shape[1], data.shape[2], 3), dtype=np.uint8)
            for b in range(3):
                band = data[b].astype(np.float64)
                v = band[np.isfinite(band)]
                if v.size == 0: rgb[:, :, b] = 0; continue
                lo, hi = np.percentile(v, (2, 98))
                if hi <= lo: hi, lo = float(v.max()), float(v.min())
                if hi <= lo: rgb[:, :, b] = 0; continue
                rgb[:, :, b] = np.clip((band - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
            Image.fromarray(rgb).save(inference_path, format="PNG")

            result = svc.infer(inference_path, prompt, image.filename or upload_name, display_mode)
            masks_raw = result.pop("masks", None)
            scores_raw = result.pop("scores", None)
            instances = []
            if masks_raw is not None and len(masks_raw) > 0:
                if hasattr(masks_raw, "detach"):
                    mask_arr = masks_raw.detach().float().cpu().numpy()
                else:
                    mask_arr = np.asarray(masks_raw, dtype=np.float32)
                mask_arr = np.squeeze(mask_arr)
                if mask_arr.ndim == 2:
                    mask_arr = mask_arr[np.newaxis, ...]

                sc_arr = np.array(scores_raw).flatten() if scores_raw is not None else np.ones(mask_arr.shape[0])

                for m in range(mask_arr.shape[0]):
                    if m < len(sc_arr) and sc_arr[m] < thresh:
                        continue
                    bin_mask = (mask_arr[m] > thresh).astype(np.uint8)
                    if bin_mask.sum() < 10:
                        continue
                    mi = Image.fromarray(bin_mask * 255)
                    mi = mi.resize((img_w, img_h), Image.Resampling.NEAREST)
                    instances.append({
                        "row": 0, "col": 0, "h": img_h, "w": img_w,
                        "mask": (np.array(mi) > 0).astype(np.uint8),
                        "box": None,
                        "score": float(sc_arr[m]) if m < len(sc_arr) else 0.0,
                    })
            total_pixels = sum(inst["mask"].sum() for inst in instances)

        # 构建描述性文件名 + 独立输出文件夹
        tif_stem = Path(image.filename or upload_name).stem
        conf_val = thresh  # 本请求生效的阈值（可能是 -F confidence_threshold 覆盖后的值）
        ts = time.strftime("%Y%m%d_%H%M%S")
        params_tag = (
            f"conf{conf_val}"
            f"_tile{tile_size}"
            f"_ov{overlap}"
            f"_mina{min_area_pixels}"
            f"_holes{hole_area_pixels}"
            f"_simp{simplify_tolerance}"
            f"_nov{1 if remove_overlap else 0}"
            f"_slic{1 if slic_points else 0}"
        )
        if float(slic_min_brightness) > 0:
            params_tag += f"_sb{float(slic_min_brightness):.0f}"
        if float(dark_p10_threshold) > 0:
            params_tag += f"_p10{float(dark_p10_threshold):.0f}"
        if bool(merge_touching):
            params_tag += "_mt1"
        if float(merge_fill_hole_m2) != 0:
            params_tag += f"_mfh{float(merge_fill_hole_m2):g}"
        folder_name = f"{tif_stem}_{params_tag}_{ts}"
        result_folder = RESULT_DIR / folder_name
        result_folder.mkdir(parents=True, exist_ok=True)
        base_name = tif_stem

        # 合并所有实例生成可视化结果图
        merged = np.zeros((img_h, img_w), dtype=np.uint8)
        for inst in instances:
            r, c, h_i, w_i = inst["row"], inst["col"], inst["h"], inst["w"]
            merged[r:r + h_i, c:c + w_i] = np.maximum(
                merged[r:r + h_i, c:c + w_i], inst["mask"]
            )
        vis_name = f"{base_name}_result.png"
        vis_path = result_folder / vis_name
        Image.fromarray(merged * 255).save(vis_path, format="PNG")

        # 后处理 + 导出 Shapefile
        shp_name = base_name
        shp_path = clean_and_vectorize_instances(
            instances, profile,
            output_dir=result_folder,
            output_name=shp_name,
            min_area_pixels=min_area_pixels,
            min_area_m2=min_area_m2,
            fill_holes=bool(fill_holes),
            hole_area_pixels=hole_area_pixels,
            simplify_tolerance=simplify_tolerance,
            remove_overlap=bool(remove_overlap),
            mask_smooth_sigma=mask_smooth_sigma,
            coarse_instances=(meta or {}).get("coarse") if meta else None,
            fine_instances=(meta or {}).get("fine") if meta else None,
            raster_path=str(upload_path),
            dark_p10_threshold=dark_p10_threshold,
            clip_shp=restrict_shp or None,
            merge_touching=bool(merge_touching),
            merge_fill_hole_m2=merge_fill_hole_m2,
        )

    except Exception as exc:
        upload_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Geo inference failed: {exc}") from exc
    finally:
        if default_thresh is not None:
            svc.processor.set_confidence_threshold(default_thresh)

    base_url = str(request.base_url).rstrip("/")
    elapsed = round(time.time() - started_at, 2)

    # 收集同名的 .shp/.shx/.dbf/.prj
    shp_files = {}
    for ext in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
        p = shp_path.with_suffix(ext)
        if p.exists():
            shp_files[ext.lstrip(".")] = f"{base_url}/files/results/{p.name}"

    # ── 写实验记录 ──
    import json as _json
    exp_log = result_folder / "_experiment.json"
    entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "geo_infer": _GEO_INFER_VERSION, "image": image.filename, "prompt": prompt,
        "tile_size": tile_size, "core_size": core_size, "core_overlap": core_overlap,
        "fine_tile_size": fine_tile_size, "fine_overlap": fine_overlap,
        "stretch": bool(stretch), "confidence_threshold": thresh, "simplify_tolerance": float(simplify_tolerance),
        "mask_smooth_sigma": mask_smooth_sigma, "min_area_m2": min_area_m2,
        "fill_holes": bool(fill_holes), "hole_area_pixels": hole_area_pixels,
        "remove_overlap": bool(remove_overlap), "max_context_area_ratio": max_context_area_ratio,
        "max_context_edge_touches": max_context_edge_touches,
        "max_refine_boxes": max_refine_boxes,
        "slic_points": slic_points, "slic_segments": slic_segments,
        "slic_compactness": slic_compactness, "slic_min_distance_m": slic_min_distance_m,
        "slic_points_fine": slic_points_fine, "slic_max_points": slic_max_points,
        "slic_min_brightness": slic_min_brightness, "dark_p10_threshold": dark_p10_threshold,
        "restrict_shp": restrict_shp,
        "merge_touching": bool(merge_touching), "merge_fill_hole_m2": merge_fill_hole_m2,
        "auto_tune": bool(auto_tune), "auto_tune_report": tune_report,
        "instances": len(instances), "elapsed_seconds": elapsed,
        "image_size": f"{img_w}×{img_h}",
    }
    exp_log.write_text(_json.dumps(entry, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "success": True,
        "prompt": prompt,
        "image_size": f"{img_w}×{img_h}",
        "result_image_url": f"{base_url}/files/results/{folder_name}/{vis_name}",
        "output_folder": str(result_folder),
        "geo_infer_version": _GEO_INFER_VERSION,
        "sliding_window": needs_sliding,
        "tile_size": tile_size,
        "overlap": overlap,
        "instances": len(instances),
        "mask_pixels": int(total_pixels),
        "shapefile_base": f"{base_url}/files/results/{folder_name}/{shp_path.stem}",
        "shapefile_files": shp_files,
        "params": {
            "geo_infer": _GEO_INFER_VERSION,
            "min_area_m2": min_area_m2,
            "min_area_pixels": min_area_pixels,
            "fill_holes": bool(fill_holes),
            "hole_area_pixels": hole_area_pixels,
            "simplify_tolerance": simplify_tolerance,
            "mask_smooth_sigma": mask_smooth_sigma,
            "core_overlap": core_overlap,
            "max_context_edge_touches": max_context_edge_touches,
            "max_refine_boxes": max_refine_boxes,
            "slic_points": slic_points,
            "slic_segments": slic_segments,
            "slic_compactness": slic_compactness,
            "slic_min_distance_m": slic_min_distance_m,
            "slic_points_fine": slic_points_fine,
            "slic_max_points": slic_max_points,
            "slic_min_brightness": slic_min_brightness,
            "dark_p10_threshold": dark_p10_threshold,
            "restrict_shp": restrict_shp,
            "merge_touching": bool(merge_touching),
            "merge_fill_hole_m2": merge_fill_hole_m2,
            "auto_tune": bool(auto_tune),
        },
        "geo_infer_version": _GEO_INFER_VERSION,
        "experiment": {
            "prompt": prompt,
            "tile_size": tile_size,
            "core_size": core_size,
            "core_overlap": core_overlap,
            "fine_tile_size": fine_tile_size,
            "fine_overlap": fine_overlap,
            "stretch": bool(stretch),
            "confidence_threshold": thresh,
            "simplify_tolerance": float(simplify_tolerance),
            "mask_smooth_sigma": mask_smooth_sigma,
            "min_area_m2": min_area_m2,
            "fill_holes": bool(fill_holes),
            "hole_area_pixels": hole_area_pixels,
            "remove_overlap": bool(remove_overlap),
            "max_context_area_ratio": max_context_area_ratio,
            "max_context_edge_touches": max_context_edge_touches,
            "max_refine_boxes": max_refine_boxes,
            "slic_points": slic_points,
            "slic_segments": slic_segments,
            "slic_compactness": slic_compactness,
            "slic_min_distance_m": slic_min_distance_m,
            "slic_points_fine": slic_points_fine,
            "slic_max_points": slic_max_points,
            "slic_min_brightness": slic_min_brightness,
            "dark_p10_threshold": dark_p10_threshold,
            "restrict_shp": restrict_shp,
            "merge_touching": bool(merge_touching),
            "merge_fill_hole_m2": merge_fill_hole_m2,
            "auto_tune": bool(auto_tune),
            "auto_tune_report": tune_report,
            "geo_infer_version": _GEO_INFER_VERSION,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "instances": len(instances),
        },
        "elapsed_seconds": elapsed,
    }


