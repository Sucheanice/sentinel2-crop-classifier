"""
v4: 对齐 geotiff_farmland_parcels_context.py 架构

核心:
  Context-Core: 大窗口推理，小 core 认领所有权（core 不重叠 → 无重复）
  质心所有权:   用 mask 质心（非 box 中心）判定归属
  边界触碰剔除: 用 mask bounds（非 centroid）判定是否被窗口切碎
  STRtree 去重: shapely R-tree 空间索引，O(n log n)

v10 = v9 + 相接对象合并 + 多边形补洞（矢量化层，范围切割之后）:
  - merge_touching: 相接/相交的 polygon 按连通组合并（每组保留最高分属性）
  - merge_fill_hole_m2: 合并结果补洞（<0 不补；=0 补全部；>0 只补 ≤ 该面积的洞）
"""

import re
from pathlib import Path

import numpy as np
import rasterio
import torch
from PIL import Image
from rasterio.features import shapes
from rasterio.windows import Window
from scipy.ndimage import distance_transform_edt, find_objects
from skimage.morphology import remove_small_holes
from skimage.segmentation import slic


# ════════════════════════════════════════════
# 图像处理
# ════════════════════════════════════════════

def _tile_to_uint8_rgb(tile: np.ndarray) -> np.ndarray:
    """rasterio (C,H,W) → (H,W,3) uint8 RGB，2-98% 百分位拉伸"""
    if tile.ndim != 3:
        raise ValueError(f"Expected (C,H,W), got {tile.shape}")
    bands = tile.shape[0]
    if bands == 1:
        tile = np.repeat(tile, 3, axis=0)
    elif bands >= 3:
        tile = tile[:3]
    else:
        tile = np.tile(tile, (3 // bands + 1, 1, 1))[:3]
    tile = np.nan_to_num(tile, nan=0.0, posinf=0.0, neginf=0.0)
    rgb = np.empty((tile.shape[1], tile.shape[2], 3), dtype=np.uint8)
    for i in range(3):
        band = tile[i].astype(np.float64)
        valid = band[np.isfinite(band)]
        if valid.size == 0: rgb[:, :, i] = 0; continue
        lo, hi = np.percentile(valid, (2, 98))
        if hi <= lo: hi, lo = float(valid.max()), float(valid.min())
        if hi <= lo: rgb[:, :, i] = 0; continue
        rgb[:, :, i] = np.clip((band - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
    return rgb


def _tile_to_uint8_rgb_no_stretch(tile: np.ndarray) -> np.ndarray:
    """不拉伸，直接 min-max 映射"""
    rgb_raw = np.nan_to_num(tile, nan=0.0)
    if rgb_raw.shape[0] == 1: rgb_raw = np.repeat(rgb_raw, 3, axis=0)
    elif rgb_raw.shape[0] > 3: rgb_raw = rgb_raw[:3]
    h, w = rgb_raw.shape[1], rgb_raw.shape[2]
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    n_bands = min(rgb_raw.shape[0], 3)
    for b in range(n_bands):
        band = rgb_raw[b].astype(np.float64)
        v = band[np.isfinite(band)]
        lo, hi = float(v.min()), float(v.max())
        if hi <= lo: hi = lo + 1
        rgb[:, :, b] = np.clip((band - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
    if n_bands == 1:
        rgb[:, :, 1:] = rgb[:, :, :1]
    return rgb


def _pad_rgb(rgb: np.ndarray, tile_size: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    if h == tile_size and w == tile_size:
        return rgb
    padded = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
    padded[:h, :w] = rgb
    return padded


def _resize_mask(mask: np.ndarray, w: int, h: int) -> np.ndarray:
    mi = Image.fromarray((mask * 255).astype(np.uint8))
    mi = mi.resize((w, h), Image.Resampling.NEAREST)
    return (np.array(mi) > 127).astype(np.uint8)


def extract_slic_prompt_points(rgb: np.ndarray, n_segments: int, compactness: float,
                               min_distance_px: float, max_points: int = 128,
                               min_brightness: float = 0.0, raw_lum: np.ndarray = None,
                               valid_mask: np.ndarray = None):
    """
    论文模块①: SLIC 超像素 → 距离变换 → 距超像素边界最远点作为 SAM 点提示。

    rgb: (H, W, 3) uint8，即 pad 后送入模型的那张图（与模型所见一致）
    min_distance_px: 超像素内点距边界的最小距离（像素），低于则丢弃（论文 3 m 阈值换算）
    min_brightness: 超像素均值亮度门（原始影像亮度 0-255，>0 启用；低于则不放点，
                    针对深色树林误检。stretch 后的 rgb 是逐瓦片 min-max，单位对不上，
                    所以必须用 raw_lum 的原始值算）
    raw_lum: (H, W) float32 原始亮度图（未拉伸、与 rgb 同尺寸，含 padding），
             min_brightness > 0 时必须提供
    valid_mask: (H, W) bool 有效区掩膜（与 rgb 同尺寸，含 padding）；点落在无效
                （nodata）区则丢弃，防止影像有效区非规整矩形时在图像外检出
    返回: (points, dists)
      points: List[[x_norm, y_norm]]，归一化 [0,1]（除以 pad 后的 H/W）
      dists:  np.ndarray (N,)，各点的最大内切距离（像素）
    """
    h, w = rgb.shape[:2]
    if h < 8 or w < 8 or float(rgb.std()) < 1e-3:
        return [], np.array([], dtype=np.float32)  # 极小窗口 / 纯黑 nodata 窗口

    if min_brightness > 0 and raw_lum is None:
        raise ValueError("min_brightness 需要 raw_lum（原始亮度图）")

    seg = slic(rgb, n_segments=max(4, int(n_segments)), compactness=compactness,
               channel_axis=-1, convert2lab=True, start_label=0,
               enforce_connectivity=False, sigma=0)

    objs = find_objects(seg)  # 一次扫描拿到每个 label 的 bbox slice
    if objs is None:
        return [], np.array([], dtype=np.float32)

    cand = []  # (distance, y, x)
    n_dark = 0
    for lab, sl in enumerate(objs):
        if sl is None:
            continue
        # 跳过贴 pad 后 tile 外边缘的超像素（含右下黑边 padding 区、被窗口截断的地块；
        # 相邻 Context 窗口会完整看到该地块，最外缘靠文本解码兜底）
        if sl[0].start == 0 or sl[1].start == 0 or sl[0].stop == h or sl[1].stop == w:
            continue
        sub = seg[sl] == lab
        if int(sub.sum()) < 4:
            continue
        # ── 亮度门：深色超像素（树林等）不放点 ──
        if min_brightness > 0 and float(raw_lum[sl][sub].mean()) < min_brightness:
            n_dark += 1
            continue
        dist = distance_transform_edt(sub)  # 到超像素边界的距离
        dmax = float(dist.max())
        if dmax < min_distance_px:
            continue
        iy, ix = np.unravel_index(int(np.argmax(dist)), dist.shape)
        gy = sl[0].start + iy
        gx = sl[1].start + ix
        if valid_mask is not None and not bool(valid_mask[gy, gx]):
            continue  # 点落在 nodata 区，不放
        cand.append((dmax, gy, gx))

    if n_dark > 0:
        print(f"  [SLIC] 亮度门砍掉 {n_dark} 个深色超像素 (<{min_brightness:.0f})", flush=True)

    if not cand:
        return [], np.array([], dtype=np.float32)

    cand.sort(key=lambda t: t[0], reverse=True)  # 最内点优先
    cand = cand[:max_points]  # 上限保护
    points, dists = [], []
    for dmax, gy, gx in cand:
        xn = min(max(float(gx) / w, 0.0), 1.0 - 1e-4)
        yn = min(max(float(gy) / h, 0.0), 1.0 - 1e-4)
        points.append([xn, yn])
        dists.append(dmax)
    return points, np.array(dists, dtype=np.float32)


def _tensor_to_np(t):
    if t is None:
        return None
    return t.detach().float().cpu().numpy() if hasattr(t, "detach") else np.asarray(t, dtype=np.float32)


def _scores_np(t):
    """scores tensor → 1-D float 数组（None/空 → 空数组；空数组不能直接用 or 判真）"""
    a = _tensor_to_np(t)
    return np.array([] if a is None else a).flatten()


def _masks3d(a):
    if a is None:
        return None
    a = np.squeeze(a)
    if a.ndim == 2:
        a = a[np.newaxis, ...]
    return a


def _extract_instances(mask_arr, sc_arr, box_arr, row, col, wh, ww, tile_size,
                       confidence_threshold):
    """从 mask/score/box 数组构建实例列表（文本解码与点解码共用）。"""
    instances = []
    for m in range(mask_arr.shape[0]):
        if m < len(sc_arr) and sc_arr[m] < confidence_threshold: continue
        inst_mask = mask_arr[m]
        if inst_mask.max() < confidence_threshold: continue
        bin_mask = (inst_mask > confidence_threshold).astype(np.uint8)
        if bin_mask.sum() < 10: continue
        bin_resized = _resize_mask(bin_mask.astype(np.float32), ww, wh)

        box = None
        if box_arr is not None and m < box_arr.shape[0]:
            bx = box_arr[m].copy()
            bx[0] = bx[0] * ww / tile_size; bx[1] = bx[1] * wh / tile_size
            bx[2] = bx[2] * ww / tile_size; bx[3] = bx[3] * wh / tile_size
            box = bx

        score = float(sc_arr[m]) if m < len(sc_arr) else 0.0
        instances.append({"row": row, "col": col, "h": wh, "w": ww,
                          "mask": bin_resized, "box": box, "score": score})
    return instances


# ════════════════════════════════════════════
# 推理
# ════════════════════════════════════════════

def _infer_tile(processor, src, win, tile_size, prompt, confidence_threshold, stretch=True,
                geometric_refine=True, refine_low=0.23, refine_high=0.60,
                max_refine_boxes=20,
                slic_enabled=False, slic_n_segments=64, slic_compactness=10.0,
                slic_min_distance_px=3.0, slic_max_points=128,
                slic_min_brightness=0.0, farm_poly=None):
    """
    混合提示推理：文本初筛 → 对中间置信度对象做 box 几何精修 → SLIC 自适应点提示二次解码。

    - score >= refine_high: 高置信，直接信
    - refine_low <= score < refine_high: 中间地带，用几何 box 提示精修
    - score < refine_low: 丢弃（低于置信度阈值也不进结果）
    - SLIC 点提示: 超像素内距边界最远点作为正样本点，一次批量解码，与文本结果合并
    """
    row = int(win.row_off); col = int(win.col_off)
    wh = int(win.height); ww = int(win.width)

    tile = src.read(indexes=list(range(1, min(src.count, 3) + 1)), window=win)
    # ── nodata 掩膜：影像有效区非规整矩形时，无效区不能出检出 ──
    valid_win = src.dataset_mask(window=win) > 0  # (wh, ww) bool
    # ── 农田范围约束：只在 farmland shp 范围内识别（slic/context/fine 共用此掩膜）──
    if farm_poly is not None:
        from rasterio.features import geometry_mask
        farm_win = geometry_mask(
            [farm_poly], out_shape=valid_win.shape,
            transform=src.window_transform(win), invert=True)
        valid_win = valid_win & farm_win
    if float(valid_win.mean()) < 0.05:
        return []  # 几乎全是 nodata / 范围外 的窗口：跳过（省一次完整解码）
    tile = np.where(valid_win, tile, 0)
    if stretch:
        rgb = _pad_rgb(_tile_to_uint8_rgb(tile), tile_size)
    else:
        rgb = _pad_rgb(_tile_to_uint8_rgb_no_stretch(tile), tile_size)

    # ── 原始亮度图（亮度门用；stretch 后的 rgb 是逐瓦片 min-max，阈值单位对不上）──
    raw_lum = None
    if slic_enabled and slic_min_brightness > 0:
        t = tile[:3].astype(np.float32)
        if t.shape[0] == 1:
            lum = t[0]
        else:
            lum = 0.299 * t[0] + 0.587 * t[1] + 0.114 * t[2]
        padded = np.zeros((tile_size, tile_size), dtype=np.float32)
        padded[:lum.shape[0], :lum.shape[1]] = lum
        raw_lum = padded

    # ── SLIC 有效掩膜（pad 与 rgb 一致；无效区不放点）──
    valid_pad = None
    if slic_enabled:
        valid_pad = np.zeros((tile_size, tile_size), dtype=bool)
        valid_pad[:valid_win.shape[0], :valid_win.shape[1]] = valid_win

    # ── ⓪ SLIC 超像素自适应点提取（纯 CPU；失败降级为纯文本）──
    slic_pts = []
    if slic_enabled:
        try:
            slic_pts, _ = extract_slic_prompt_points(
                rgb, n_segments=slic_n_segments, compactness=slic_compactness,
                min_distance_px=slic_min_distance_px, max_points=slic_max_points,
                min_brightness=slic_min_brightness, raw_lum=raw_lum,
                valid_mask=valid_pad)
        except Exception as exc:
            print(f"  [SLIC] ({col},{row}) 点提取失败: {exc}", flush=True)

    image = Image.fromarray(rgb)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = processor.set_image(image)
        out = processor.set_text_prompt(state=state, prompt=prompt)

    # ── ① 文本初筛结果（0 mask 也不提前返回：SLIC 点解码正是兜底文本漏检的）──
    mask_arr = _masks3d(_tensor_to_np(out.get("masks")))
    if mask_arr is None:
        mask_arr = np.empty((0, 1, 1), dtype=np.float32)
    sc_arr = _scores_np(out.get("scores"))
    box_arr = _tensor_to_np(out.get("boxes"))

    # ── ② 几何精修：对中间置信度对象加 box 提示 ──
    # 每个框 = 一次完整解码且框累加，低置信阈值下框数可能上百 → 加数量上限 + 进度日志
    if geometric_refine:
        refine_indices = [
            m for m in range(mask_arr.shape[0])
            if m < len(sc_arr) and refine_low <= sc_arr[m] < refine_high
        ]
        if refine_indices:
            if int(max_refine_boxes) > 0 and len(refine_indices) > int(max_refine_boxes):
                # 按分数从高到低截断（高分的精修收益更大）
                refine_indices = sorted(
                    refine_indices, key=lambda m: sc_arr[m], reverse=True,
                )[:int(max_refine_boxes)]
            print(f"  [Refine] ({col},{row}) {len(refine_indices)} 个框精修中", flush=True)
            try:
                for n, m in enumerate(refine_indices):
                    if box_arr is None or m >= box_arr.shape[0]: continue
                    bx = box_arr[m].copy()
                    box_prompt = [float(bx[0]), float(bx[1]), float(bx[2]), float(bx[3])]
                    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                        state = processor.add_geometric_prompt(state=state, box=box_prompt, label=True)
                    if (n + 1) % 10 == 0:
                        print(f"  [Refine] ({col},{row}) {n+1}/{len(refine_indices)}", flush=True)
                # 精修后重新取结果
                r_masks = _masks3d(_tensor_to_np(state.get("masks")))
                if r_masks is not None and len(r_masks) > 0:
                    mask_arr = r_masks
                    sc_arr = _scores_np(state.get("scores"))
                    box_arr = _tensor_to_np(state.get("boxes"))
            except Exception:
                pass  # 精修失败则用文本初筛结果

    # ── ③ 文本(+精修)实例快照（点解码会覆盖 state 输出）──
    instances = _extract_instances(mask_arr, sc_arr, box_arr, row, col, wh, ww,
                                   tile_size, confidence_threshold)

    # ── ④ SLIC 点提示二次解码（一次批量解码）──
    if slic_pts:
        try:
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                state = processor.add_geometric_points(slic_pts, [1] * len(slic_pts), state)
            p_masks = _masks3d(_tensor_to_np(state.get("masks")))
            p_insts = []
            if p_masks is not None and len(p_masks) > 0:
                p_sc = _scores_np(state.get("scores"))
                p_bx = _tensor_to_np(state.get("boxes"))
                p_insts = _extract_instances(p_masks, p_sc, p_bx, row, col, wh, ww,
                                             tile_size, confidence_threshold)
            instances.extend(p_insts)
            print(f"  [SLIC] ({col},{row}) {len(slic_pts)} 点 → {len(p_insts)} 实例", flush=True)
        except Exception as exc:
            print(f"  [SLIC] ({col},{row}) 点解码失败，回退文本结果: {exc}", flush=True)

    # ── ⑤ nodata 过滤：mask 过半像素落在无效区则丢弃（图像外检出的兜底）──
    if not bool(valid_win.all()):
        kept = []
        for inst in instances:
            nz = int(inst["mask"].sum())
            if nz > 0 and int((inst["mask"] & ~valid_win).sum()) / nz > 0.5:
                continue
            kept.append(inst)
        instances = kept

    return instances


# ════════════════════════════════════════════
# Context-Core 滑动窗口
# ════════════════════════════════════════════

def _process_context_windows(processor, src, img_w, img_h, tile_size, core_size,
                              prompt, confidence_threshold, stretch, max_context_edge_touches,
                              max_context_area_ratio, core_step,
                              geometric_refine=True, refine_low=0.23, refine_high=0.60,
                              max_refine_boxes=20,
                              slic_enabled=False, slic_n_segments=64, slic_compactness=10.0,
                              slic_min_distance_px=3.0, slic_max_points=128,
                              slic_min_brightness=0.0, farm_poly=None):
    """粗尺度 Context-Core 推理"""
    core_pad = (tile_size - core_size) // 2
    instances = []
    total_tiles = 0

    for row_off in range(0, img_h, core_step):
        for col_off in range(0, img_w, core_step):
            ctx_col = max(0, col_off - core_pad)
            ctx_row = max(0, row_off - core_pad)
            ctx_w = min(tile_size, img_w - ctx_col)
            ctx_h = min(tile_size, img_h - ctx_row)

            win = Window(ctx_col, ctx_row, ctx_w, ctx_h)
            tile_insts = _infer_tile(processor, src, win, tile_size, prompt, confidence_threshold,
                                     stretch=stretch, geometric_refine=geometric_refine,
                                     refine_low=refine_low, refine_high=refine_high,
                                     max_refine_boxes=max_refine_boxes,
                                     slic_enabled=slic_enabled, slic_n_segments=slic_n_segments,
                                     slic_compactness=slic_compactness,
                                     slic_min_distance_px=slic_min_distance_px,
                                     slic_max_points=slic_max_points,
                                     slic_min_brightness=slic_min_brightness,
                                     farm_poly=farm_poly)

            core_c0 = col_off - ctx_col; core_r0 = row_off - ctx_row
            core_c1 = core_c0 + min(core_size, img_w - col_off)
            core_r1 = core_r0 + min(core_size, img_h - row_off)

            for inst in tile_insts:
                mask = inst["mask"]
                yy, xx = np.where(mask)
                if len(yy) == 0: continue
                cy, cx = yy.mean(), xx.mean()
                if not (core_c0 <= cx < core_c1 and core_r0 <= cy < core_r1):
                    continue
                nz = mask.sum()
                if nz / (ctx_w * ctx_h) > max_context_area_ratio:
                    continue
                min_x, max_x = xx.min(), xx.max()
                min_y, max_y = yy.min(), yy.max()
                margin = 2
                # 用图像全局坐标检查大图边缘
                gx0 = ctx_col + min_x; gx1 = ctx_col + max_x
                gy0 = ctx_row + min_y; gy1 = ctx_row + max_y
                edge_touches = sum([
                    gx0 <= margin, gx1 >= img_w - margin,
                    gy0 <= margin, gy1 >= img_h - margin,
                ])
                if edge_touches > max_context_edge_touches:
                    continue
                instances.append(inst)

            total_tiles += 1
            if total_tiles % 20 == 0:
                print(f"  [Context {total_tiles}] {len(instances)} 实例", flush=True)

    print(f"Context-Core: {total_tiles} 窗口, {len(instances)} 实例", flush=True)
    return instances


def _process_fine_tiles(processor, src, img_w, img_h, fine_tile_size, fine_overlap,
                         prompt, confidence_threshold, stretch, fine_core_size=384,
                         fine_core_overlap=96,
                         geometric_refine=True, refine_low=0.23, refine_high=0.60,
                         max_refine_boxes=20,
                         slic_enabled=False, slic_n_segments=64, slic_compactness=10.0,
                         slic_min_distance_px=3.0, slic_max_points=128,
                         slic_min_brightness=0.0, farm_poly=None):
    """细尺度 Context-Core：Core 间有重叠，消除边界漏检"""
    if fine_core_size > fine_tile_size:
        fine_core_size = fine_tile_size
    core_pad = (fine_tile_size - fine_core_size) // 2
    core_step = max(1, fine_core_size - fine_core_overlap)
    instances = []
    total_tiles = 0

    for row_off in range(0, img_h, core_step):
        for col_off in range(0, img_w, core_step):
            ctx_col = max(0, col_off - core_pad)
            ctx_row = max(0, row_off - core_pad)
            ctx_w = min(fine_tile_size, img_w - ctx_col)
            ctx_h = min(fine_tile_size, img_h - ctx_row)

            win = Window(ctx_col, ctx_row, ctx_w, ctx_h)
            tile_insts = _infer_tile(processor, src, win, fine_tile_size, prompt, confidence_threshold,
                                     stretch=stretch, geometric_refine=geometric_refine,
                                     refine_low=refine_low, refine_high=refine_high,
                                     max_refine_boxes=max_refine_boxes,
                                     slic_enabled=slic_enabled, slic_n_segments=slic_n_segments,
                                     slic_compactness=slic_compactness,
                                     slic_min_distance_px=slic_min_distance_px,
                                     slic_max_points=slic_max_points,
                                     slic_min_brightness=slic_min_brightness,
                                     farm_poly=farm_poly)

            core_c0 = col_off - ctx_col; core_r0 = row_off - ctx_row
            core_c1 = core_c0 + min(fine_core_size, img_w - col_off)
            core_r1 = core_r0 + min(fine_core_size, img_h - row_off)

            for inst in tile_insts:
                mask = inst["mask"]
                yy, xx = np.where(mask)
                if len(yy) == 0: continue
                nz = mask.sum()
                if nz < 10: continue
                cy, cx = yy.mean(), xx.mean()
                if not (core_c0 <= cx < core_c1 and core_r0 <= cy < core_r1):
                    continue
                instances.append(inst)

            total_tiles += 1
            if total_tiles % 50 == 0:
                print(f"  [Fine {total_tiles}] {len(instances)} 实例", flush=True)

    print(f"Fine Context-Core: {total_tiles} 窗口 (ctx={fine_tile_size}, core={fine_core_size}), {len(instances)} 实例", flush=True)
    return instances


def _load_restrict_poly(restrict_shp, tif_crs):
    """加载农田范围 shp → 转换到影像 CRS 的 shapely polygon。

    返回: (farm_poly, 多边形个数)；shp 无 CRS 时按与影像同 CRS 处理。
    """
    import fiona
    from shapely.geometry import shape as _shape
    from shapely.ops import unary_union
    shp_crs = None
    parts = []
    with fiona.open(restrict_shp) as fs:
        shp_crs = fs.crs
        for feat in fs:
            g = feat["geometry"]
            if g is None or g["type"] not in ("Polygon", "MultiPolygon"):
                continue
            parts.append(_shape(g))
    if not parts:
        raise ValueError(f"restrict_shp 无有效多边形: {restrict_shp}")
    farm_poly = unary_union(parts)
    if shp_crs is not None and tif_crs is not None and shp_crs != tif_crs:
        # 用 fiona/GDAL 做 CRS 转换（服务器无 pyproj；fiona 已随环境安装）
        from fiona.transform import transform_geom
        from shapely.geometry import mapping as _mapping
        farm_poly = _shape(transform_geom(shp_crs, tif_crs, _mapping(farm_poly)))
    elif shp_crs is None:
        print("  [范围约束] shp 无 CRS，按与影像同 CRS 处理", flush=True)
    return farm_poly, len(parts)


def run_sliding_window(
    processor, input_path: Path, prompt: str,
    tile_size: int = 1008, core_size: int = 672,
    confidence_threshold: float = 0.5, stretch: bool = True,
    max_context_edge_touches: int = 2, max_context_area_ratio: float = 0.25,
    fine_tile_size: int = 0, fine_overlap: float = 0.25,
    fine_core_size: int = 384, fine_core_overlap: int = 96,
    core_overlap: int = 168, mask_smooth_sigma: float = 1.0,
    geometric_refine: bool = True, refine_low: float = 0.23, refine_high: float = 0.60,
    max_refine_boxes: int = 20,
    slic_points: int = 1, slic_segments: int = 64, slic_compactness: float = 10.0,
    slic_min_distance_m: float = 3.0, slic_points_fine: int = 0,
    slic_max_points: int = 128,
    slic_min_brightness: float = 0.0,
    restrict_shp: str = None,
) -> tuple[list, dict]:
    """
    Context-Core + Fine-Tile 双尺度推理（含论文模块① SLIC 自适应点提示）。

    粗尺度（Context-Core）：大窗口 + 质心所有权 → 消除重复
    细尺度（Fine-Tile）：512 重叠瓦片 → 捕捉小地块
    尺度内去重 → STRtree 双向确认（mutual-best）
    跨尺度合并 → 矢量化层 _cross_scale_merge（mask 级不做跨尺度去重）

    返回: (instances, profile, meta)；meta = {"coarse": …, "fine": …}，无细尺度时为 None
    """
    if core_size > tile_size: core_size = tile_size
    core_step = max(1, core_size - core_overlap)

    with rasterio.open(input_path) as src:
        profile = src.profile
        img_w, img_h = src.width, src.height

        # ── 农田范围约束：只在 farmland shp 范围内识别（slic/context/fine 之前生效）──
        farm_poly = None
        if restrict_shp:
            farm_poly, n_parts = _load_restrict_poly(restrict_shp, src.crs)
            print(f"  范围约束: 仅在 farmland shp 内识别（{restrict_shp}，{n_parts} 个多边形，"
                  f"总面积 {farm_poly.area / 1e6:.1f} km²）", flush=True)

        # ── SLIC 点提示配置：米 → 像素（利用 raster 分辨率；非投影 CRS 用启发式）──
        slic_coarse_enabled = bool(slic_points)
        slic_fine_enabled = bool(slic_points_fine) and fine_tile_size > 0
        if slic_coarse_enabled or slic_fine_enabled:
            transform = src.transform
            px_w, px_h = abs(transform.a), abs(transform.e)
            mean_px = (px_w + px_h) / 2.0
            crs = src.crs
            if crs is not None and crs.is_projected and mean_px > 0:
                slic_min_distance_px = max(1.0, float(slic_min_distance_m) / mean_px)
            else:
                slic_min_distance_px = max(1.0, tile_size * 0.02)
                print(f"  [SLIC] CRS 非投影或未知，min_distance 用像素启发式 {slic_min_distance_px:.1f}px", flush=True)
            # n_segments 按瓦片面积相对 1008² 缩放
            slic_coarse_n = max(4, int(round(slic_segments * tile_size * tile_size / (1008 * 1008))))
            slic_fine_n = max(4, int(round(slic_segments * fine_tile_size * fine_tile_size / (1008 * 1008)))) \
                if fine_tile_size > 0 else 0
            print(f"SLIC 点提示: 粗={slic_coarse_enabled}(n={slic_coarse_n},compact={slic_compactness}) "
                  f"细={slic_fine_enabled}(n={slic_fine_n}) min_dist={slic_min_distance_px:.2f}px "
                  f"亮度门={slic_min_brightness:.0f}(0=关)", flush=True)
        else:
            slic_min_distance_px = 3.0; slic_coarse_n = 64; slic_fine_n = 0

        # ── 粗尺度 ──
        coarse = _process_context_windows(
            processor, src, img_w, img_h, tile_size, core_size,
            prompt, confidence_threshold, stretch,
            max_context_edge_touches, max_context_area_ratio,
            core_step=core_step,
            geometric_refine=geometric_refine, refine_low=refine_low, refine_high=refine_high,
            max_refine_boxes=max_refine_boxes,
            slic_enabled=slic_coarse_enabled, slic_n_segments=slic_coarse_n,
            slic_compactness=slic_compactness, slic_min_distance_px=slic_min_distance_px,
            slic_max_points=slic_max_points,
            slic_min_brightness=slic_min_brightness,
            farm_poly=farm_poly)

        # ── 细尺度（可选）──
        fine = []
        if fine_tile_size > 0:
            fine = _process_fine_tiles(
                processor, src, img_w, img_h, fine_tile_size, fine_overlap,
                prompt, confidence_threshold, stretch, fine_core_size=fine_core_size,
                fine_core_overlap=fine_core_overlap,
                geometric_refine=geometric_refine, refine_low=refine_low, refine_high=refine_high,
                max_refine_boxes=max_refine_boxes,
                slic_enabled=slic_fine_enabled, slic_n_segments=slic_fine_n,
                slic_compactness=slic_compactness, slic_min_distance_px=slic_min_distance_px,
                slic_max_points=slic_max_points,
                slic_min_brightness=slic_min_brightness,
                farm_poly=farm_poly)

        if fine:
            # ── 尺度内去重（跨尺度合并交给矢量化层 _cross_scale_merge）──
            coarse = deduplicate_instances(coarse, match_thresh=0.50)
            fine = deduplicate_instances(fine, match_thresh=0.35)
            combined = coarse + fine
            print(f"双尺度: coarse={len(coarse)} fine={len(fine)} → combined={len(combined)}", flush=True)
            return combined, profile, {"coarse": coarse, "fine": fine}

        coarse = deduplicate_instances(coarse, match_thresh=0.50)
        return coarse, profile, None


# ════════════════════════════════════════════
# STRtree 空间索引去重
# ════════════════════════════════════════════

def deduplicate_instances(instances: list, match_thresh: float = 0.50) -> list:
    """
    真·双向确认去重（mutual-best，对齐 geotiff_farmland_parcels.py 的语义）。

    ① 重叠系数 = 交集 / min(A面积, B面积)（比 IoU 对不同尺寸更友好）
    ② STRtree 只比较空间相邻的对；overlap >= match_thresh 才进入候选
    ③ 每 (i, 对方窗口) 只保留 overlap 最大的候选 j
    ④ 双向确认: i 对 j 所在窗口的最佳匹配是 j，且 j 对 i 所在窗口的最佳匹配也是 i → 才合并
       （防止链式错误合并；同窗口的文本/点两轮解码重复也走同一确认路径）
    ⑤ Union-Find 聚成组，每组保留分数最高的
    """
    if len(instances) <= 1:
        return instances

    from shapely.geometry import box
    from shapely.strtree import STRtree

    n = len(instances)
    mask_areas = [float(inst["mask"].sum()) for inst in instances]
    scores = [inst.get("score", 0.0) for inst in instances]
    win_ids = [(inst["row"], inst["col"], inst["h"], inst["w"]) for inst in instances]

    # ── bbox → polygon，建 R-tree ──
    bboxes = []
    for inst in instances:
        m = inst["mask"]
        yy, xx = np.where(m)
        if len(yy) == 0:
            bboxes.append(box(0, 0, 1, 1))
            continue
        bboxes.append(box(
            inst["col"] + xx.min(), inst["row"] + yy.min(),
            inst["col"] + xx.max(), inst["row"] + yy.max(),
        ))

    tree = STRtree(bboxes)

    # ── 每 (i, 对方窗口) 只保留 overlap 最大的候选 ──
    best = {}  # (i, win_ids[j]) → (overlap, j)
    for i in range(n):
        for j in tree.query(bboxes[i]):
            j = int(j)
            if j == i: continue
            ri, ci = instances[i]["row"], instances[i]["col"]
            rj, cj = instances[j]["row"], instances[j]["col"]
            r0 = max(ri, rj); c0 = max(ci, cj)
            r1 = min(ri + instances[i]["h"], rj + instances[j]["h"])
            c1 = min(ci + instances[i]["w"], cj + instances[j]["w"])
            if r0 >= r1 or c0 >= c1: continue

            mi = instances[i]["mask"][r0 - ri:r1 - ri, c0 - ci:c1 - ci]
            mj = instances[j]["mask"][r0 - rj:r1 - rj, c0 - cj:c1 - cj]
            inter = np.logical_and(mi, mj).sum()
            if inter <= 0: continue
            overlap = inter / min(mask_areas[i], mask_areas[j])
            if overlap < match_thresh: continue
            key = (i, win_ids[j])
            if key not in best or overlap > best[key][0]:
                best[key] = (overlap, j)

    if not best:
        return instances

    # ── 双向确认 ──
    pairs = set()
    for (i, _), (_, j) in best.items():
        reverse = best.get((j, win_ids[i]))
        if reverse is not None and reverse[1] == i:
            pairs.add((min(i, j), max(i, j)))

    if not pairs:
        return instances

    # ── Union-Find 聚组 ──
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[rb] = ra

    for (i, j) in pairs:
        union(i, j)

    # ── 每组保留 score 最高的 ──
    groups = {}
    for idx in range(n):
        root = find(idx)
        if root not in groups:
            groups[root] = []
        groups[root].append(idx)

    result = []
    for group in groups.values():
        best = max(group, key=lambda i: scores[i])
        result.append(instances[best])

    if len(result) < n:
        print(f"  去重: {n} → {len(result)}", flush=True)

    return result


# ════════════════════════════════════════════
# 层级后处理（文章第 3-5 层）
# ════════════════════════════════════════════

def _suppress_coarse_parents(items, parent_min_children=3, child_size_ratio=0.60,
                              containment=0.80, min_child_coverage=0.10):
    """
    第 3 层：粗父对象抑制。

    一个大 polygon 内部覆盖了 ≥parent_min_children 个小 polygon，
    且子对象总面积 ≥ 父面积的 min_child_coverage → 删大留小。
    返回: (保留的 items, 被删的详情)
    """
    if len(items) < parent_min_children + 1:
        return items, []

    from shapely.strtree import STRtree

    polys, props = zip(*items)
    tree = STRtree(polys)
    rejected = []

    for parent_idx in sorted(range(len(polys)), key=lambda i: polys[i].area, reverse=True):
        parent = polys[parent_idx]
        if parent.area <= 0: continue
        children = []
        for child_idx in tree.query(parent):
            child_idx = int(child_idx)
            if child_idx == parent_idx: continue
            child = polys[child_idx]
            if child.area >= parent.area * child_size_ratio: continue
            inter = parent.intersection(child)
            if inter.is_empty: continue
            if inter.area / child.area < containment: continue
            children.append(inter)

        if len(children) < parent_min_children: continue
        from shapely.ops import unary_union
        coverage = unary_union(children).area / parent.area
        if coverage >= min_child_coverage:
            rejected.append(parent_idx)

    if rejected:
        rejected_set = set(rejected)
        items = [item for i, item in enumerate(items) if i not in rejected_set]
        print(f"  父对象抑制: 删除 {len(rejected)} 个粗框", flush=True)

    return items, rejected


def _cross_scale_merge(coarse_items, fine_items, overlap_thresh=0.70, size_ratio=0.60):
    """
    第 4 层：跨尺度合并。

    细尺度对象与粗尺度对象 overlap ≥70% 且面积比 ≥60% → 用细尺度版替换粗尺度版
      （细尺度分辨率更高，边界更精——修复原实现只丢细对象、保留粗对象的 bug）
    细尺度对象没有对应粗尺度对象 → 作为补充保留（小地块，粗尺度漏检）
    返回: (merged_items, replaced_count)
    """
    if not fine_items:
        return list(coarse_items), 0

    from shapely.strtree import STRtree

    coarse_polys = [p for p, _ in coarse_items]
    tree = STRtree(coarse_polys) if coarse_polys else None
    replaced = set()          # 被细尺度版替换的粗对象下标
    replacements = []         # 替换进来的细尺度 (poly, props)
    additions = []            # 无对应粗对象的细尺度补充

    for fine_poly, fine_props in fine_items:
        matched = None
        if tree is not None:
            for ci in tree.query(fine_poly):
                ci = int(ci)
                cp = coarse_polys[ci]
                inter = fine_poly.intersection(cp)
                if inter.is_empty: continue
                overlap = inter.area / min(fine_poly.area, cp.area)
                sr = min(fine_poly.area, cp.area) / max(fine_poly.area, cp.area)
                if overlap >= overlap_thresh and sr >= size_ratio:
                    matched = ci
                    break
        if matched is not None:
            # 每个粗对象只被替换一次（多个细对象命中同一粗对象时忽略后续）
            if matched not in replaced:
                replaced.add(matched)
                replacements.append((fine_poly, fine_props))
        else:
            additions.append((fine_poly, fine_props))

    merged = [item for i, item in enumerate(coarse_items) if i not in replaced]
    merged.extend(replacements)
    merged.extend(additions)
    return merged, len(replacements)


def _remove_invisible(items):
    """
    删除无效/不渲染的 polygon：
    - 空几何 / 面积 ≤ 0
    - 非 Polygon 类型
    - 修复后仍然无效
    """
    from shapely.geometry import Polygon

    clean = []
    removed = 0
    for poly, props in items:
        if poly.is_empty or poly.area <= 0: removed += 1; continue
        if poly.geom_type == "Polygon":
            if poly.is_valid:
                clean.append((poly, props))
            else:
                # 最后尝试修复
                try:
                    fixed = _fix_geometry(poly)
                    if fixed is not None and fixed.is_valid and fixed.area > 0:
                        clean.append((fixed, props))
                    else:
                        removed += 1
                except Exception:
                    removed += 1
        elif poly.geom_type == "MultiPolygon":
            # 取最大的 polygon
            largest = max(poly.geoms, key=lambda p: p.area) if len(poly.geoms) > 0 else None
            if largest is not None and isinstance(largest, Polygon) and largest.is_valid:
                clean.append((largest, props))
            else:
                removed += 1
        else:
            removed += 1

    if removed:
        print(f"  删除无效几何: {removed} 个", flush=True)
    return clean


# ════════════════════════════════════════════
# Shapefile 导出
# ════════════════════════════════════════════

def _fix_geometry(poly):
    """修复无效几何体 + 规范环方向（ArcGIS 要求外环逆时针）"""
    from shapely.geometry import Polygon, MultiPolygon
    from shapely.validation import make_valid
    from shapely.ops import unary_union, orient

    # 1. 修复拓扑
    if not poly.is_valid:
        poly = make_valid(poly)

    # 2. buffer(0) 清理微小自交
    poly = poly.buffer(0)

    # 3. 提取最大的 Polygon
    if poly.geom_type == "Polygon":
        result = poly
    elif poly.geom_type == "MultiPolygon":
        result = max(poly.geoms, key=lambda p: p.area) if len(poly.geoms) > 0 else None
    elif poly.geom_type == "GeometryCollection":
        polys = [g for g in poly.geoms if isinstance(g, (Polygon, MultiPolygon))]
        if not polys:
            return None
        merged = unary_union(polys)
        result = merged if merged.geom_type == "Polygon" else (
            max(merged.geoms, key=lambda p: p.area) if merged.geom_type == "MultiPolygon" else None
        )
    else:
        return None

    if result is None or not isinstance(result, Polygon):
        return None

    # 4. 规范环方向：外环逆时针，内环顺时针（ArcGIS 格式）
    return orient(result)


def _smooth_polygon(geom: dict, tolerance: float) -> dict:
    from shapely.geometry import shape, mapping
    poly = shape(geom)

    # 1. 修复并归一化
    poly = _fix_geometry(poly)
    if poly is None:
        return geom  # 无法修复，回退原始

    # 2. 简化
    if float(tolerance) > 0:
        simplified = poly.simplify(float(tolerance), preserve_topology=True)
        if not simplified.is_empty:
            simplified = _fix_geometry(simplified)
            if simplified is not None:
                poly = simplified

    # 3. 最终兜底
    if poly.is_empty:
        return geom
    fixed = _fix_geometry(poly)
    return mapping(fixed) if fixed is not None else geom


def _count_vertices(poly):
    """统计 polygon 顶点数（外环+内环）。"""
    if poly.geom_type == "Polygon":
        return len(poly.exterior.coords) + sum(len(r.coords) for r in poly.interiors)
    if poly.geom_type == "MultiPolygon":
        return sum(_count_vertices(p) for p in poly.geoms)
    return 0


def _final_smooth_polygon(poly, simplify_tolerance, max_vertices=0):
    """最终共线点合并（问题7）：仅用 Douglas-Peucker 把近似共线的点合并成直线。

    不再做 buffer 回缩（会造成整体内收）、不再做 Chaikin 圆角（会把地块磨成 6/7 边形）。
    max_vertices>0 时，仅对超过上限的极端锯齿多边形渐进加大 tolerance 简化，
    其余多边形保持原 simplify 结果不变，从而沿用原版描边。
    """
    if float(simplify_tolerance) > 0:
        try:
            s = poly.simplify(float(simplify_tolerance), preserve_topology=True)
            if not s.is_empty:
                poly = s
        except Exception:
            pass

    if max_vertices and int(max_vertices) > 0:
        tol = float(simplify_tolerance) or 1.0
        for _ in range(12):
            if _count_vertices(poly) <= int(max_vertices):
                break
            tol *= 1.5
            try:
                s = poly.simplify(tol, preserve_topology=True)
                if not s.is_empty:
                    poly = s
            except Exception:
                break

    fixed = _fix_geometry(poly)
    return fixed if fixed is not None else poly


def _filter_dark_polygons(items, raster_path, threshold):
    """polygon 在原始影像上的亮度 p10 低于阈值 → 删除（针对深色树林误检）。

    items: [(shapely polygon, props), ...]，polygon 为 tif CRS 坐标
    亮度用原始值（0-255，与诊断脚本 analyze_classes_spectra.py 单位一致）；
    大 polygon 的窗口降采样到 ≤256 边（p10 亮度对降采样不敏感）。
    返回: (保留的 items, 删除数量)
    """
    from rasterio.windows import from_bounds, Window
    from rasterio.features import geometry_mask
    from affine import Affine

    keep = []
    n_dark = 0
    with rasterio.open(raster_path) as src:
        # 全局亮度归一化到 0-255（修复 16bit 输入量纲 bug）：读降采样概览算 2/98 百分位
        try:
            ov = src.read(indexes=[1, 2, 3],
                          out_shape=(3, max(1, src.height // 16), max(1, src.width // 16))).astype(np.float32)
            ov_lum = ov[0] if ov.shape[0] == 1 else 0.299 * ov[0] + 0.587 * ov[1] + 0.114 * ov[2]
            lo, hi = np.percentile(ov_lum, (2, 98))
            if hi <= lo:
                hi = lo + 1.0
        except Exception:
            lo, hi = 0.0, 255.0
        for poly, props in items:
            try:
                win = from_bounds(*poly.bounds, transform=src.transform)
                win = win.intersection(Window(0, 0, src.width, src.height))
                if win.width < 4 or win.height < 4:
                    keep.append((poly, props)); continue
                scale = min(1.0, 256.0 / max(win.width, win.height))
                sh = max(1, int(round(win.height * scale)))
                sw = max(1, int(round(win.width * scale)))
                data = src.read(indexes=[1, 2, 3], window=win,
                                out_shape=(3, sh, sw)).astype(np.float32)
                wtrans = src.window_transform(win)
                m = geometry_mask(
                    [poly], out_shape=(sh, sw),
                    transform=wtrans * Affine.scale(win.width / sw, win.height / sh),
                    invert=True)
                if int(m.sum()) < 16:
                    keep.append((poly, props)); continue
                if data.shape[0] == 1:
                    lum = data[0]
                else:
                    lum = 0.299 * data[0] + 0.587 * data[1] + 0.114 * data[2]
                lum255 = (lum - lo) * 255.0 / (hi - lo)
                if float(np.percentile(lum255[m], 10)) < threshold:
                    n_dark += 1
                    continue
            except Exception:
                pass  # 读不到就保留（宁漏删不错删）
            keep.append((poly, props))
    return keep, n_dark


def _filter_forest_by_ndvi(items, raster_path, ndvi_threshold):
    """用 NIR(第4波段)/红(第3波段) 计算 NDVI，polygon 内 NDVI 均值 > 阈值 → 判林地删除（问题3）。

    需影像 >= 4 波段且第 4 波段为 NIR。NDVI 为归一化差值，与 8/16bit 量纲无关，
    对本影像林地(NDVI≈0.09)与耕地(NDVI≈-0.04)分离良好，阈值建议 0.04~0.06。
    返回: (保留的 items, 删除数量)
    """
    from rasterio.windows import from_bounds, Window
    from rasterio.features import geometry_mask
    from affine import Affine

    keep = []
    n_forest = 0
    with rasterio.open(raster_path) as src:
        has_nir = src.count >= 4
        if not has_nir:
            return items, 0
        for poly, props in items:
            try:
                win = from_bounds(*poly.bounds, transform=src.transform)
                win = win.intersection(Window(0, 0, src.width, src.height))
                if win.width < 4 or win.height < 4:
                    keep.append((poly, props)); continue
                scale = min(1.0, 256.0 / max(win.width, win.height))
                sh = max(1, int(round(win.height * scale)))
                sw = max(1, int(round(win.width * scale)))
                data = src.read(indexes=[3, 4], window=win,
                                out_shape=(2, sh, sw)).astype(np.float32)
                wtrans = src.window_transform(win)
                m = geometry_mask(
                    [poly], out_shape=(sh, sw),
                    transform=wtrans * Affine.scale(win.width / sw, win.height / sh),
                    invert=True)
                if int(m.sum()) < 16:
                    keep.append((poly, props)); continue
                red = data[0]
                nir = data[1]
                ndvi = (nir - red) / (nir + red + 1e-6)
                if float(ndvi[m].mean()) > float(ndvi_threshold):
                    n_forest += 1
                    continue
            except Exception:
                pass  # 读不到就保留（宁漏删不错删）
            keep.append((poly, props))
    return keep, n_forest


def _fill_holes_geom(geom, max_hole_m2):
    """多边形补洞（v10）：max_hole_m2<0 不补；=0 补全部洞；>0 只补面积 ≤ 该值的洞。"""
    from shapely.geometry import Polygon, MultiPolygon
    if max_hole_m2 < 0:
        return geom
    if geom.geom_type == "Polygon":
        if len(geom.interiors) == 0:
            return geom
        if max_hole_m2 == 0:
            return Polygon(geom.exterior)
        kept = [r for r in geom.interiors if Polygon(r).area > max_hole_m2]
        try:
            return Polygon(geom.exterior, kept)
        except Exception:
            return geom
    if geom.geom_type == "MultiPolygon":
        return MultiPolygon([_fill_holes_geom(p, max_hole_m2) for p in geom.geoms])
    return geom


def _merge_touching_and_fill(items, max_hole_m2=0.0):
    """相接/相交对象按连通组合并取并集，再对合并结果补洞（v10）。

    相接判定: 几何相交（touches 或 overlap）即同组。
    属性: 每组保留分数最高者的 props。
    max_hole_m2: <0 不补洞；=0 补全部；>0 只补小洞。
    返回: [(poly, props), ...]
    """
    from shapely.strtree import STRtree
    from shapely.ops import unary_union

    polys = [p for p, _ in items]
    n = len(polys)
    if n == 0:
        return []
    if n == 1:
        g = _fill_holes_geom(polys[0], max_hole_m2)
        return [(g, dict(items[0][1]))]

    tree = STRtree(polys)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in tree.query(polys[i]):
            j = int(j)
            if j <= i:
                continue
            try:
                if polys[i].intersects(polys[j]):
                    union(i, j)
            except Exception:
                continue

    groups = {}
    for k in range(n):
        groups.setdefault(find(k), []).append(k)

    out = []
    for g in groups.values():
        try:
            geom = unary_union([polys[k] for k in g])
        except Exception:
            geom = polys[g[0]]
        if geom.is_empty:
            continue
        best = max(g, key=lambda k: float(items[k][1].get("score", 0.0) or 0.0))
        props = dict(items[best][1])
        geom = _fill_holes_geom(geom, max_hole_m2)
        if geom.geom_type == "MultiPolygon":
            for p in geom.geoms:
                if not p.is_empty:
                    out.append((p, dict(props)))
        elif geom.geom_type == "Polygon" and not geom.is_empty:
            out.append((geom, props))
        else:
            out.append((geom, props))  # 罕见几何兜底
    return out


def clean_and_vectorize_instances(
    instances, profile, output_dir, output_name,
    min_area_pixels=0, min_area_m2=0, fill_holes=True, hole_area_pixels=256,
    simplify_tolerance=0.5, remove_overlap=True, mask_smooth_sigma=1.0,
    coarse_instances=None, fine_instances=None,
    raster_path=None, dark_p10_threshold=0.0,
    clip_shp=None,
    merge_touching=1, merge_fill_hole_m2=0.0,
    ndvi_threshold=0.0,
    max_vertices=0,
) -> Path:
    """
    mask 实例 → 矢量化 → 层级后处理 → Shapefile 导出。

    coarse_instances / fine_instances: 粗/细尺度实例分开传入时（run_sliding_window 的 meta），
    先各自矢量化，再走第 4 层 _cross_scale_merge（细尺度版替换粗尺度版）。
    raster_path + dark_p10_threshold: 深色过滤（>0 启用）——最终 polygon 在原始影像上的
    亮度 p10 低于阈值则删除，针对深色树林误检（建议 69；已归一化到 0-255）。
    clip_shp: 农田范围 shp 路径（>0 个多边形时启用）——最终 polygon 超出范围的部分直接裁掉，
    完全在范围外的丢弃；裁出的碎片由后续深色过滤/面积过滤兜底。
    merge_touching: 相接/相交 polygon 按连通组合并（默认 1=开；0=关，行为同 v9）
    merge_fill_hole_m2: 合并结果补洞（<0 不补；0=补全部；>0 只补 ≤ 该面积 m² 的洞）
    ndvi_threshold: NDVI 林地过滤（>0 启用）——polygon 内 NDVI 均值 > 阈值判林地删除（建议 0.04~0.06）。
    max_vertices: 最终顶点上限（>0 启用）——仅对超过上限的多边形做共线点合并（Douglas-Peucker）。
    """
    import fiona
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"{output_name}.shp"

    crs = profile.get("crs"); transform = profile.get("transform")
    px_w, px_h = abs(transform.a), abs(transform.e)
    px_area = px_w * px_h

    # m² → 像素
    if float(min_area_m2) > 0:
        min_area_pixels = max(1, int(float(min_area_m2) / px_area))

    # 逐实例类型转换，防止 numpy 布尔判断报错
    def _normalize(insts):
        return [{**i,
            "row": int(i["row"]), "col": int(i["col"]),
            "h": int(i["h"]), "w": int(i["w"]),
            "score": float(i.get("score", 0)),
            "mask": i["mask"].astype(bool),
        } for i in insts]

    schema = {
        "geometry": "Polygon",
        "properties": {"id": "int", "area_px": "int", "area_m2": "float",
                       "score": "float", "class": "str:32"},
    }
    layer_name = re.sub(r"[^0-9A-Za-z_]", "_", out.stem).strip("_")[:64]
    feature_count = 0

    # ── 预生成所有 polygon ──
    def _vectorize(insts):
        items = []  # [(shapely_polygon, properties_dict), ...]
        for inst in _normalize(insts):
            try:
                m = inst["mask"].astype(bool)
            except Exception:
                continue
            if bool(fill_holes) and int(hole_area_pixels) > 0:
                m = remove_small_holes(m, max_size=hole_area_pixels)
            m_u8 = m.astype(np.uint8)
            if m_u8.sum() == 0: continue
            # 高斯模糊 → 边界平滑（在 shapes() 前做一次，消除像素锯齿）
            if float(mask_smooth_sigma) > 0:
                from scipy.ndimage import gaussian_filter
                blurred = gaussian_filter(m_u8.astype(np.float32), sigma=float(mask_smooth_sigma))
                m_u8 = (blurred > 0.5).astype(np.uint8)
                if m_u8.sum() == 0: continue
            score = inst.get("score", 0.0)
            for geom, val in shapes(m_u8, mask=m_u8 == 1,
                                    transform=transform * transform.translation(int(inst["col"]), int(inst["row"]))):
                if int(val) != 1: continue
                try:
                    smoothed = _smooth_polygon(geom, float(simplify_tolerance))
                except Exception:
                    smoothed = geom
                from shapely.geometry import shape
                poly = shape(smoothed)
                if poly.is_empty: continue
                # 用最终 polygon 的几何面积（不再用原始 mask 像素数）
                final_m2 = poly.area
                final_px = max(1, int(final_m2 / px_area))
                if int(min_area_pixels) > 0 and final_px < int(min_area_pixels):
                    continue
                items.append((poly, {"id": 0, "area_px": final_px,
                                     "area_m2": round(final_m2, 2),
                                     "score": round(score, 4), "class": "farmland"}))
        return items

    # ── 矢量化（粗/细分离传入时走第 4 层跨尺度合并）──
    if coarse_instances is not None and fine_instances is not None:
        coarse_items = _vectorize(coarse_instances)
        fine_items = _vectorize(fine_instances)
        items, cross_replaced = _cross_scale_merge(coarse_items, fine_items)
        print(f"  跨尺度合并: coarse={len(coarse_items)} fine={len(fine_items)} "
              f"→ {len(items)} (细替换粗 {cross_replaced})", flush=True)
    else:
        items = _vectorize(instances)

    # ── 重叠裁剪：STRtree 找重叠对，小对象减掉重叠区 ──
    if remove_overlap and len(items) >= 2:
        from shapely.strtree import STRtree
        polys = [p for p, _ in items]
        tree = STRtree(polys)
        clipped = [p for p, _ in items]
        keep = [True] * len(items)
        removed = 0

        for i in range(len(items)):
            if not keep[i]: continue
            for j in tree.query(polys[i]):
                j = int(j)
                if j <= i or not keep[j]: continue
                pi, pj = clipped[i], clipped[j]
                if not pi.intersects(pj): continue
                inter = pi.intersection(pj)
                if inter.is_empty or inter.area <= 0: continue

                # 重叠 > 80% 中小对象 → 直接删小对象（同一地块的重复检出）
                containment = inter.area / min(pi.area, pj.area)
                if containment > 0.80:
                    if pi.area >= pj.area:
                        keep[j] = False  # j 是小对象
                    else:
                        keep[i] = False  # i 是小对象
                    removed += 1
                    continue

                # 否则：按分数优先权，低分者减掉重叠区（高分者保留完整边界，避免凹口）
                si = float(items[i][1].get("score", 0.0) or 0.0)
                sj = float(items[j][1].get("score", 0.0) or 0.0)
                if si >= sj:
                    new_pj = pj.difference(pi)
                    if not new_pj.is_empty and new_pj.geom_type in ("Polygon", "MultiPolygon"):
                        clipped[j] = new_pj
                else:
                    new_pi = pi.difference(pj)
                    if not new_pi.is_empty and new_pi.geom_type in ("Polygon", "MultiPolygon"):
                        clipped[i] = new_pi

        # 过滤被删的，同步 clip
        items = [(clipped[k], items[k][1]) for k in range(len(items)) if keep[k]]
        clipped = [clipped[k] for k in range(len(clipped)) if keep[k]]
        if removed:
            print(f"  重叠删除: {removed} 个小对象（大框套小框）", flush=True)

        # 展开 MultiPolygon
        final_polys = []
        final_props = []
        for idx, poly in enumerate(clipped):
            if idx >= len(items): break
            props = items[idx][1]
            if poly.geom_type == "MultiPolygon":
                for p in poly.geoms:
                    if not p.is_empty:
                        final_polys.append(p)
                        final_props.append(dict(props))
            elif poly.geom_type == "Polygon" and not poly.is_empty:
                final_polys.append(poly)
                final_props.append(dict(props))
        items = list(zip(final_polys, final_props))

    # ── 第 3 层：粗父对象抑制 ──
    items, _ = _suppress_coarse_parents(items)

    # ── 删除无效/闪烁 polygon ──
    items = _remove_invisible(items)

    # ── 范围切割：polygon 超出农田范围的部分直接裁掉（完全在外的丢弃）──
    if clip_shp is not None and items:
        clip_poly, n_parts = _load_restrict_poly(clip_shp, crs)
        new_items = []
        n_clipped = 0
        for poly, props in items:
            try:
                if not poly.intersects(clip_poly):
                    continue  # 完全在范围外
                inter = poly.intersection(clip_poly)
            except Exception:
                inter = poly  # 切割失败保留原样
            if inter.is_empty or inter.area <= 0:
                continue
            new_items.append((inter, props))
        print(f"  范围切割: {len(items)} → {len(new_items)} 个 polygon（超出范围的部分已裁掉）", flush=True)
        items = new_items

    # ── 相接对象合并 + 补洞（v10 新增；合并后的碎片交给后续面积过滤兜底）──
    if bool(merge_touching):
        before = len(items)
        items = _merge_touching_and_fill(items, float(merge_fill_hole_m2))
        if len(items) != before:
            print(f"  相接合并: {before} → {len(items)} 个 polygon（含补洞）", flush=True)

    # ── 深色过滤：最终 polygon 在原始影像上的亮度 p10 低于阈值 → 删除（树林误检门）──
    if float(dark_p10_threshold) > 0 and raster_path is not None and items:
        items, n_dark = _filter_dark_polygons(items, str(raster_path), float(dark_p10_threshold))
        if n_dark:
            print(f"  深色过滤(p10亮度<{float(dark_p10_threshold):.0f}): 删除 {n_dark} 个树林类误检", flush=True)

    # ── NDVI 林地过滤：polygon 内 NDVI 均值 > 阈值 → 删除（林地误检门，利用第4波段NIR）──
    if float(ndvi_threshold) > 0 and raster_path is not None and items:
        items, n_forest = _filter_forest_by_ndvi(items, str(raster_path), float(ndvi_threshold))
        if n_forest:
            print(f"  NDVI林地过滤(NDVI>{float(ndvi_threshold):.2f}): 删除 {n_forest} 个林地误检", flush=True)

    # ── 末尾面积过滤：重叠裁剪/父抑制可能切出碎片，用最终几何面积再筛一遍 ──
    min_area_m2_eff = float(min_area_pixels) * px_area if int(min_area_pixels) > 0 else 0.0
    if min_area_m2_eff > 0:
        before = len(items)
        items = [(poly, props) for poly, props in items if poly.area >= min_area_m2_eff]
        if len(items) < before:
            print(f"  末尾面积过滤: 删除 {before - len(items)} 个碎片 (<{min_area_m2_eff:.1f} m²)", flush=True)

    # ── 写 Shapefile ──
    with fiona.open(out, "w", driver="ESRI Shapefile", crs=crs,
                    schema=schema, layer=layer_name, encoding="UTF-8") as sink:
        for poly, props in items:
            # 展开 MultiPolygon 并分别写入
            geoms_to_write = []
            if poly.geom_type == "MultiPolygon":
                geoms_to_write.extend([g for g in poly.geoms if not g.is_empty])
            elif poly.geom_type == "Polygon" and not poly.is_empty:
                geoms_to_write.append(poly)

            for g in geoms_to_write:
                if min_area_m2_eff > 0 and g.area < min_area_m2_eff:
                    continue  # MultiPolygon 里的小碎片部件
                # 最终共线点合并：仅对超过 max_vertices 的极端多边形简化（问题7）
                if int(max_vertices) > 0:
                    g = _final_smooth_polygon(g, float(simplify_tolerance), max_vertices=int(max_vertices))
                    if g is None or g.is_empty:
                        continue
                    if g.geom_type == "MultiPolygon":
                        g = max(g.geoms, key=lambda p: p.area) if len(g.geoms) else None
                        if g is None or g.is_empty:
                            continue
                from shapely.geometry import mapping
                feature_count += 1
                props["id"] = feature_count
                sink.write({"geometry": mapping(g), "properties": props})

    print(f"  导出 {feature_count} 个多边形", flush=True)
    return out


# ════════════════════════════════════════════
# 六维无监督质量评分 + 自适应参数优选
# ════════════════════════════════════════════

def score_instances(instances, score_img, img_w, img_h, max_samples=150):
    """
    六维无监督质量评分（mask 级，供 auto_tune 选参；全部维度逐实例计算）。

    几何:
      ① 边缘平滑度: 4π·area/perimeter²（compactness，越高越平滑，0-1）
      ② 凸度:       area / convex_area（regionprops，0-1）
    光谱（score_img 上 mask bbox 裁剪区）:
      ③ 平均亮度:   mask 内像素均值（适中 128 最好，0-1 距离）
      ④ 光谱方差:   mask 内像素 std（越高越好，/64 归一化）
    纹理（bbox 裁剪区 GLCM，32 灰度级）:
      ⑤ 同质性:     GLCM 同质性（越高越好）
      ⑥ 对比度:     GLCM 对比度（越低越好，1 - c/200）

    返回: 0-1 质量综合分（实例数补偿项由 auto_tune_parameters 加权）
    """
    if not instances or score_img is None:
        return 0.0

    from skimage.measure import regionprops
    try:
        from skimage.feature import graycomatrix, graycoprops
    except ImportError:
        graycomatrix = None

    img = np.asarray(score_img)
    gray_full = img.mean(axis=2).astype(np.uint8) if img.ndim == 3 else np.asarray(img).astype(np.uint8)
    H, W = gray_full.shape
    fx = W / max(img_w, 1)
    fy = H / max(img_h, 1)

    n = len(instances)
    idx = np.linspace(0, n - 1, min(n, max_samples)).astype(int)  # 确定性抽样

    smooths, convexes, brights, variances, homos, contrasts = [], [], [], [], [], []
    for k in idx:
        inst = instances[int(k)]
        m = inst["mask"].astype(bool)
        if m.sum() < 10: continue
        sw = max(1, int(round(inst["w"] * fx)))
        sh = max(1, int(round(inst["h"] * fy)))
        m_s = _resize_mask(m.astype(np.uint8), sw, sh).astype(bool)

        props = regionprops(m_s.astype(np.uint8))
        if props:
            p = props[0]
            if p.area >= 4:
                if p.convex_area > 0:
                    convexes.append(min(p.area / p.convex_area, 1.0))
                if p.perimeter > 0:
                    comp = 4 * np.pi * p.area / (p.perimeter * p.perimeter)
                    smooths.append(min(comp, 1.0))

        y0 = int(inst["row"] * fy); x0 = int(inst["col"] * fx)
        crop = gray_full[y0:y0 + sh, x0:x0 + sw]
        sel = crop[m_s]
        if sel.size >= 4:
            brights.append(float(sel.mean()))
            variances.append(float(sel.std()))
        if graycomatrix is not None and crop.size >= 64:
            try:
                g = (crop // 8).astype(np.uint8)  # 32 灰度级
                glcm = graycomatrix(g, distances=[1], angles=[0], levels=32,
                                    symmetric=True, normed=True)
                homos.append(float(graycoprops(glcm, "homogeneity")[0, 0]))
                contrasts.append(1.0 - min(float(graycoprops(glcm, "contrast")[0, 0]) / 200.0, 1.0))
            except Exception:
                pass

    def _mean(xs):
        return float(np.mean(xs)) if xs else 0.0

    smoothness = _mean(smooths)
    convexity = _mean(convexes)
    brightness = 1.0 - min(abs(_mean(brights) - 128) / 128, 1.0) if brights else 0.0
    variance = min(_mean(variances) / 64.0, 1.0)
    homogeneity = _mean(homos)
    contrast = _mean(contrasts)

    # 加权综合: 几何 40%, 光谱 30%, 纹理 30%
    geometry = 0.6 * smoothness + 0.4 * convexity
    spectrum = 0.5 * brightness + 0.5 * variance
    texture = 0.5 * homogeneity + 0.5 * contrast
    return float(0.4 * geometry + 0.3 * spectrum + 0.3 * texture)


def auto_tune_parameters(processor, input_path, prompt, param_grid=None, max_combinations=9,
                         fine_tile_size=512,
                         slic_points=1, slic_segments=64, slic_compactness=10.0,
                         slic_min_distance_m=3.0, slic_max_points=128,
                         slic_min_brightness=0.0, restrict_shp=None):
    """
    自适应参数优选：在参数网格上搜索，用六维 mask 级评分选最优组合。

    每个组合 = 整幅图一次完整滑窗推理（粗+细），总耗时 ×组合数，大图慎用。
    param_grid 只放推理参数（mask 级评分感知不到后处理参数）。

    返回: {"params", "instances", "profile", "meta", "report"}；全部组合失败时返回 None
      report: [{params, quality, score, instances}, ...] 按综合分降序
    """
    import itertools
    if param_grid is None:
        param_grid = {
            "fine_core_size": [384, 448],
            "fine_core_overlap": [96, 128],
        }

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combos = list(itertools.product(*values))[:max_combinations]

    # ── 评分参考图：整幅降采样 RGB 预览（与模型所见一致的 2-98% 拉伸）──
    with rasterio.open(input_path) as src:
        img_w, img_h = src.width, src.height
        f = min(1.0, 4096.0 / max(img_w, img_h))
        sh, sw = max(1, int(round(img_h * f))), max(1, int(round(img_w * f)))
        data = src.read(indexes=list(range(1, min(src.count, 3) + 1)),
                        out_shape=(min(src.count, 3), sh, sw))
    score_img = _tile_to_uint8_rgb(data)  # (sh, sw, 3) uint8

    print(f"参数优选: {len(combos)} 种组合 (评分图 {sw}×{sh})", flush=True)
    results = []
    for idx, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        try:
            instances, profile, meta = run_sliding_window(
                processor, input_path, prompt,
                tile_size=1008, core_size=672, core_overlap=168,
                confidence_threshold=processor.confidence_threshold, stretch=False,
                fine_tile_size=fine_tile_size,
                fine_core_size=params.get("fine_core_size", 384),
                fine_core_overlap=params.get("fine_core_overlap", 96),
                mask_smooth_sigma=1.0,
                slic_points=slic_points, slic_segments=slic_segments,
                slic_compactness=slic_compactness,
                slic_min_distance_m=slic_min_distance_m,
                slic_max_points=slic_max_points,
                slic_min_brightness=slic_min_brightness,
                restrict_shp=restrict_shp)
            quality = score_instances(instances, score_img, img_w, img_h)
            results.append({"params": params, "quality": quality,
                            "instances": instances, "profile": profile, "meta": meta})
            print(f"  [{idx+1}/{len(combos)}] {params} → quality={quality:.3f} (n={len(instances)})", flush=True)
        except Exception as exc:
            print(f"  [{idx+1}/{len(combos)}] {params} → 失败: {exc}", flush=True)

    if not results:
        return None

    # 综合分 = 质量分 + 实例数补偿（防止选参坍缩到"检出最少"）
    max_count = max(len(r["instances"]) for r in results)
    for r in results:
        r["score"] = 0.7 * r["quality"] + 0.3 * (len(r["instances"]) / max_count if max_count else 0.0)
    results.sort(key=lambda r: r["score"], reverse=True)
    best = results[0]
    report = [{"params": r["params"], "quality": round(r["quality"], 4),
               "score": round(r["score"], 4), "instances": len(r["instances"])}
              for r in results]
    print(f"最优参数: {best['params']} (score={best['score']:.3f})", flush=True)
    return {"params": best["params"], "instances": best["instances"],
            "profile": best["profile"], "meta": best["meta"], "report": report}
