# -*- coding: utf-8 -*-
"""sam_pipeline.py — SAM3 耕地分割产品「核心层」（cmd 版）。

这是 SAM 前置环节的统一入口，职责：
  1. 资源检测（GPU / CPU）
  2. 村界加载（内置村界库按名勾选，或用户上传村界 shp）
  3. 村界 ↔ 影像匹配校验（影像范围可 ≥ 村界，不可 < 村界）
  4. SAM3 分割 + 耕地/水体识别（调用 geo_infer_v10_multigpu.py，GPU 推理）
  5. 地块归属村（最大重叠面积）
  6. 输出：耕地 shp + 水体 shp + 合并 shp（class 区分，均带村编码/村名称）

设计：核心逻辑与 UI 解耦，本脚本可独立 cmd 运行，也可被桌面 UI 调用。

用法：
  # 完整跑（用户机 GPU + SAM3 环境）
  python sam_pipeline.py --image 宝珠村.tif \
      --village-upload 宝珠村村界.shp \
      --checkpoint checkpoint_25.pt --gpus 0 --output 输出目录 --name 宝珠村

  # 内置村界库勾选
  python sam_pipeline.py --image 宝珠村.tif \
      --village-library 四川省_村界.shp --village-names 宝珠村,朝天村 \
      --checkpoint checkpoint_25.pt --output 输出目录 --name 宝珠村

  # 跳过 SAM3 推理（用已有结果，本地测试后处理）
  python sam_pipeline.py --image 宝珠村.tif \
      --village-upload 宝珠村村界.shp --skip-inference \
      --farmland-shp result_baozhucun_v10_fix2/baozhucun_v10_fix2.shp \
      --water-shp result_baozhucun_water/baozhucun_water.shp \
      --output 输出目录 --name 宝珠村
"""
import os
import sys
import argparse
import subprocess
import multiprocessing

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from shapely.ops import unary_union

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 目标统一坐标系：CGCS2000 / 3° GK Zone 35（项目统一，影像与上传村界即此）
TARGET_CRS = 'EPSG:4523'

# 村编码 / 村名称 字段候选（内置库与上传村界字段名不同，自动识别）
CODE_COL_CANDIDATES = ['村级码', 'ZLDWDM', '村编码', 'village_code', 'XZQDM', 'CJDM']
NAME_COL_CANDIDATES = ['村级名', 'ZLDWMC', '村名称', 'village_name', 'XZQMC', 'CJMC']


# ════════════════════════════════════════════
# 1. 资源检测
# ════════════════════════════════════════════
def detect_resources():
    """检测 GPU（NVIDIA）与 CPU。无 torch / 无 CUDA 时降级，不报错。"""
    res = {'gpus': [], 'cpu_count': None, 'gpu_error': None}
    try:
        res['cpu_count'] = os.cpu_count() or multiprocessing.cpu_count()
    except Exception:
        res['cpu_count'] = None
    try:
        import torch
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            for i in range(n):
                try:
                    name = torch.cuda.get_device_name(i)
                    mem = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
                except Exception:
                    name, mem = f'GPU {i}', 0.0
                res['gpus'].append({'id': i, 'name': name, 'memory_gb': round(mem, 1)})
        else:
            res['gpu_error'] = '未检测到可用 CUDA GPU'
    except Exception as e:
        res['gpu_error'] = f'torch 不可用: {e}'
    return res


def format_resources(res):
    lines = [f"CPU 核数: {res['cpu_count']}"]
    if res['gpus']:
        lines.append(f"可用 GPU {len(res['gpus'])} 张:")
        for g in res['gpus']:
            lines.append(f"  [GPU {g['id']}] {g['name']}  {g['memory_gb']} GB")
    else:
        lines.append(f"GPU: {res.get('gpu_error') or '无'}")
    return '\n'.join(lines)


# ════════════════════════════════════════════
# 2. 村界加载
# ════════════════════════════════════════════
def _detect_village_cols(gdf):
    """自动识别村编码 / 村名称字段。"""
    cols = list(gdf.columns)
    code = next((c for c in CODE_COL_CANDIDATES if c in cols), None)
    name = next((c for c in NAME_COL_CANDIDATES if c in cols), None)
    return code, name


def load_villages(library_path=None, village_names=None, upload_path=None, image_bounds_4523=None):
    """加载村界，统一为 [geometry, village_code, village_name]（CRS=TARGET_CRS）。

    上传模式：直接读用户上传的村界（精确，不做影像范围过滤，交由匹配校验判断）。
    内置库模式：先按影像范围过滤（只保留与影像相交的村，规避全省重名村），再按名勾选。
    """
    if upload_path:
        gdf = gpd.read_file(upload_path)
        code_col, name_col = _detect_village_cols(gdf)
        if code_col is None or name_col is None:
            raise ValueError(f'上传村界缺少村编码/村名称字段，现有字段: {list(gdf.columns)}')
        gdf = gdf[[code_col, name_col, 'geometry']].rename(
            columns={code_col: 'village_code', name_col: 'village_name'})
        source = 'upload'
    elif library_path:
        gdf = gpd.read_file(library_path)
        code_col, name_col = _detect_village_cols(gdf)
        if code_col is None or name_col is None:
            raise ValueError(f'内置村界库缺少村编码/村名称字段，现有字段: {list(gdf.columns)}')
        gdf = gdf[[code_col, name_col, 'geometry']].rename(
            columns={code_col: 'village_code', name_col: 'village_name'})
        gdf = gdf.to_crs(TARGET_CRS)
        # 先按影像范围过滤（bbox 粗筛 + intersects 精筛），只保留影像覆盖的村
        if image_bounds_4523 is not None:
            minx, miny, maxx, maxy = image_bounds_4523
            img_box = box(minx, miny, maxx, maxy)
            gdf = gdf.cx[minx:maxx, miny:maxy]
            gdf = gdf[gdf.geometry.intersects(img_box)]
        # 再按名勾选（在影像覆盖的候选村内）
        if village_names:
            names = [n.strip() for n in village_names.split(',') if n.strip()]
            gdf = gdf[gdf['village_name'].isin(names)]
        if len(gdf) == 0:
            raise ValueError('影像范围内未找到匹配村')
        source = 'library'
    else:
        raise ValueError('必须提供 --village-upload 或 --village-library')

    gdf = gdf.to_crs(TARGET_CRS)
    gdf = gdf.reset_index(drop=True)
    return gdf, source


# ════════════════════════════════════════════
# 3. 村界 ↔ 影像匹配校验
# ════════════════════════════════════════════
def check_village_image_match(village_gdf, image_bounds, tolerance_m=15.0):
    """校验村界是否落在影像范围内。

    image_bounds: (minx, miny, maxx, maxy) 在 TARGET_CRS 下。
    返回 (status, message)，status ∈ {'ok', 'partial', 'outside'}。
    规则：影像范围可 ≥ 村界（ok）；不可 < 村界（村界超出影像 → 报错）。
    """
    vminx, vminy, vmaxx, vmaxy = village_gdf.total_bounds
    iminx, iminy, imaxx, imaxy = image_bounds
    t = tolerance_m

    inside_x = vminx >= iminx - t and vmaxx <= imaxx + t
    inside_y = vminy >= iminy - t and vmaxy <= imaxy + t
    if inside_x and inside_y:
        return 'ok', '村界完全在影像范围内 ✓'
    # 完全在外
    outside_x = vmaxx < iminx - t or vminx > imaxx + t
    outside_y = vmaxy < iminy - t or vminy > imaxy + t
    if outside_x or outside_y:
        return 'outside', '村界在影像范围之外 ✗（影像可能不对）'
    # 部分超出
    msg = '村界部分超出影像范围 ✗（影像覆盖范围小于村界）'
    if not inside_x:
        msg += f'\n  村界 X [{vminx:.1f},{vmaxx:.1f}] vs 影像 X [{iminx:.1f},{imaxx:.1f}]'
    if not inside_y:
        msg += f'\n  村界 Y [{vminy:.1f},{vmaxy:.1f}] vs 影像 Y [{iminy:.1f},{imaxy:.1f}]'
    return 'partial', msg


# ════════════════════════════════════════════
# 4. SAM3 推理接口
# ════════════════════════════════════════════
def run_sam3(image, checkpoint, gpus, prompt, out_shp, extra_args=None):
    """调用 geo_infer_v10_multigpu.py 做一次 SAM3 推理（GPU）。"""
    cmd = [
        sys.executable, os.path.join(BASE_DIR, 'geo_infer_v10_multigpu.py'),
        '--input', image,
        '--output', out_shp,
        '--prompt', prompt,
        '--gpus', gpus,
        '--checkpoint', checkpoint,
    ]
    if extra_args:
        cmd += extra_args
    print(f'[sam3] 启动推理 prompt={prompt} gpus={gpus}')
    subprocess.run(cmd, check=True)
    return out_shp + '.shp'


def _read_sam3_result(shp, class_label):
    """读 SAM3 结果 shp，强制统一 class 字段（规避矢量化阶段 class 误标）。"""
    gdf = gpd.read_file(shp)
    gdf['class'] = class_label
    return gdf


# ════════════════════════════════════════════
# 5. 地块归属村（最大重叠面积）
# ════════════════════════════════════════════
def assign_villages(parcel_gdf, village_gdf):
    """按「地块 ∩ 村界」最大重叠面积归属村，写入 village_code/village_name。"""
    village_gdf = village_gdf.to_crs(TARGET_CRS)
    parcel_gdf = parcel_gdf.to_crs(TARGET_CRS)
    vi = village_gdf.sindex
    codes, names = [], []
    for geom in parcel_gdf.geometry:
        if geom is None or geom.is_empty:
            codes.append(''); names.append(''); continue
        cand = list(vi.query(geom, predicate='intersects'))
        best_idx, best_area = None, -1.0
        for idx in cand:
            inter = geom.intersection(village_gdf.geometry.iloc[idx])
            if inter.is_empty:
                continue
            if inter.area > best_area:
                best_area, best_idx = inter.area, idx
        if best_idx is None:
            codes.append(''); names.append('')
        else:
            codes.append(str(village_gdf.village_code.iloc[best_idx]))
            names.append(str(village_gdf.village_name.iloc[best_idx]))
    out = parcel_gdf.copy()
    out['village_code'] = codes
    out['village_name'] = names
    return out


# ════════════════════════════════════════════
# 6. 输出（按村界范围裁剪）
# ════════════════════════════════════════════
def _clip_to_villages(gdf, village_gdf):
    """把地块裁剪到村界联合范围内（去掉村界外部分）。"""
    village_union = unary_union(village_gdf.geometry.tolist())
    gdf = gdf.copy()
    gdf['geometry'] = gdf.geometry.apply(
        lambda g: g.intersection(village_union) if g is not None else g)
    # 去掉裁剪后为空的
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()]
    return gdf


def output_shps(merged_gdf, village_gdf, out_dir, name):
    """输出三类 shp（gpkg）：耕地 / 水体 / 合并，均带村字段并按村界裁剪。"""
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, name)

    # 统一字段顺序：id/class/area 等原始 + village_code/village_name + geometry
    merged = merged_gdf.copy()
    farmland = merged[merged['class'] == 'farmland'].copy()
    water = merged[merged['class'] == 'water'].copy()

    # 按村界裁剪
    farmland = _clip_to_villages(farmland, village_gdf)
    water = _clip_to_villages(water, village_gdf)
    merged = _clip_to_villages(merged, village_gdf)

    f_path = base + '_耕地.gpkg'
    w_path = base + '_水体.gpkg'
    m_path = base + '_合并.gpkg'
    farmland.to_file(f_path, driver='GPKG')
    water.to_file(w_path, driver='GPKG')
    merged.to_file(m_path, driver='GPKG')

    print(f'[output] 耕地 {len(farmland)} / 水体 {len(water)} / 合并 {len(merged)}')
    print(f'  {name}_耕地.gpkg')
    print(f'  {name}_水体.gpkg')
    print(f'  {name}_合并.gpkg')
    return {'farmland': f_path, 'water': w_path, 'merged': m_path}


# ════════════════════════════════════════════
# 主流程
# ════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description='SAM3 耕地分割产品核心层')
    ap.add_argument('--image', required=True, help='输入 DOM 影像 tif')
    ap.add_argument('--village-library', help='内置村界库 shp（配合 --village-names）')
    ap.add_argument('--village-names', help='勾选村名（逗号分隔，村级名）')
    ap.add_argument('--village-upload', help='用户上传村界 shp')
    ap.add_argument('--checkpoint', help='SAM3 checkpoint .pt（不跳过推理时必填）')
    ap.add_argument('--gpus', default='0', help='GPU 编号，逗号分隔，如 0,6')
    ap.add_argument('--output', required=True, help='输出目录')
    ap.add_argument('--name', required=True, help='输出文件前缀（村名）')
    ap.add_argument('--skip-inference', action='store_true', help='跳过 SAM3 推理（用已有结果）')
    ap.add_argument('--farmland-shp', help='已有耕地 shp（配合 --skip-inference）')
    ap.add_argument('--water-shp', help='已有水体 shp（配合 --skip-inference）')
    args = ap.parse_args()

    # 1. 资源检测
    res = detect_resources()
    print('[资源检测]')
    print(format_resources(res))
    print()

    # 2. 影像 bounds（转 TARGET_CRS，用 pyproj 避免 rasterio PROJ_LIB 冲突）
    import rasterio
    from pyproj import Transformer
    with rasterio.open(args.image) as ds:
        img_crs_wkt = ds.crs.to_wkt() if hasattr(ds.crs, 'to_wkt') else str(ds.crs)
        img_bounds = ds.bounds
    tr = Transformer.from_crs(img_crs_wkt, TARGET_CRS, always_xy=True)
    corners = [(img_bounds[0], img_bounds[1]), (img_bounds[0], img_bounds[3]),
               (img_bounds[2], img_bounds[1]), (img_bounds[2], img_bounds[3])]
    tx, ty = [], []
    for x, y in corners:
        nx, ny = tr.transform(x, y)
        tx.append(nx); ty.append(ny)
    img_bounds_4523 = (min(tx), min(ty), max(tx), max(ty))

    # 3. 加载村界（内置库模式按影像范围过滤，规避重名村）
    village_gdf, src = load_villages(args.village_library, args.village_names,
                                     args.village_upload, img_bounds_4523)
    print(f'[村界] 来源={src} 候选村 {len(village_gdf)} 个: '
          f'{list(village_gdf["village_name"])}')

    # 4. 匹配校验
    status, msg = check_village_image_match(village_gdf, img_bounds_4523)
    print(f'[匹配校验] {status}: {msg}')
    if status == 'outside':
        print('!! 终止：村界在影像外，影像可能不对')
        return
    if status == 'partial':
        print('!! 警告：影像覆盖范围小于村界，请确认影像是否正确')
    print()

    # 4. SAM3 推理（或跳过）
    if args.skip_inference:
        if not args.farmland_shp or not args.water_shp:
            raise SystemExit('--skip-inference 需同时提供 --farmland-shp 和 --water-shp')
        farm_gdf = _read_sam3_result(args.farmland_shp, 'farmland')
        water_gdf = _read_sam3_result(args.water_shp, 'water')
    else:
        if not args.checkpoint:
            raise SystemExit('未跳过推理时必须提供 --checkpoint')
        tmp_farm = os.path.join(args.output, f'{args.name}_farmland_tmp')
        tmp_water = os.path.join(args.output, f'{args.name}_water_tmp')
        run_sam3(args.image, args.checkpoint, args.gpus, 'farmland', tmp_farm,
                 sam3_code_root=args.sam3_code_root)
        run_sam3(args.image, args.checkpoint, args.gpus, 'water', tmp_water,
                 sam3_code_root=args.sam3_code_root)
        farm_gdf = _read_sam3_result(tmp_farm + '.shp', 'farmland')
        water_gdf = _read_sam3_result(tmp_water + '.shp', 'water')

    print(f'[推理结果] 耕地 {len(farm_gdf)} / 水体 {len(water_gdf)}')

    # 5. 合并 + 归属村
    merged = pd.concat([farm_gdf, water_gdf], ignore_index=True)
    merged = assign_villages(merged, village_gdf)
    n_assigned = int((merged['village_code'].astype(str) != '').sum())
    print(f'[村归属] 已归属 {n_assigned}/{len(merged)} 个地块（最大重叠面积）')

    # 6. 输出三类 shp
    output_shps(merged, village_gdf, args.output, args.name)
    print('[done]')


if __name__ == '__main__':
    main()
