# 前进村农田矢量提取 — 微小碎片消除流程

## 一、项目概述

基于 SAM3（Segment Anything Model 3）模型，对前进村 0.5m 分辨率正射影像进行农田语义分割，输出矢量 Shapefile。

- **影像**: `qianjin_village_05m.tif`（9855×10171 像素，0.5m/像素）
- **覆盖范围**: 约 4.9km × 5.1km
- **文本提示**: `farmland`
- **服务端口**: `http://127.0.0.1:8073`

---

## 二、问题发现

初次运行（`min_area_pixels=0`）输出 **661 个多边形**，但分析发现：

| 面积区间 | 数量 | 占比 | 说明 |
|---------|------|------|------|
| <4px (1m²) | **401** | **60.7%** | 微碎片（单像素正方形为主） |
| 4-16px (1-4m²) | 55 | 8.3% | 小地块 |
| 16-100px (4-25m²) | 34 | 5.1% | 中等地块 |
| ≥100px (25m²) | 171 | 25.9% | 真正的农田地块 |

**60.7% 的多边形是面积不足 1m² 的微碎片**，形态以 0.5px（0.125m²）的单像素正方形为主，并非狭长伪图斑。

---

## 三、诊断过程

### 3.1 排除狭长伪图斑

使用代码内置指标（最小旋转矩形）和 QGIS 表达式分别检测：

- **QGIS 表达式** `2*area/perimeter < 1.0` 标记了 466 个"窄面"
- **代码指标**（`_is_narrow_polygon`）回测发现：真正狭长（width<1.5px 且 aspect≥20）的多边形 = **0 个**

结论：QGIS 表达式对任何小面积多边形都给出低值，属于误判。微碎片是**小而方**，不是**小而长**。

### 3.2 极端参数验证

将参数调至极端值（`0.50/5.0/5`）验证过滤代码是否生效：

| 配置 | 去重叠删除 | 狭长过滤 | 最终多边形 |
|------|:---:|:---:|:---:|
| 保守 (0.08/1.5/20) | 1 | 1 | 661 |
| 进阶 (0.10/2.0/15) | 1 | 1 | 661 |
| 极端 (0.50/5.0/5) | 5 | 3 | 295 |

结论：狭长过滤代码工作正常，但微碎片不满足"窄+长"条件，无法被狭长过滤器捕获。

### 3.3 根因定位

微碎片产生的原因：

1. 一个实例的掩膜矢量化后（`rasterio.features.shapes()`）会产生多个连通区域
2. 部分连通区域仅为 1 个像素（0.125m²）
3. `min_area_pixels=0` 导致所有碎片都被保留

---

## 四、解决方案

**核心修复：将 `min_area_pixels` 从 0 改为 4（即 1m²）**

此参数作用于 `clean_and_vectorize_instances()` 函数中矢量化后的逐多边形过滤（`geo_infer.py` 第 532-533 行）：

```python
if min_area_pixels > 0 and cnt < min_area_pixels:
    continue  # 跳过面积不足的碎片
```

同时在第 543-544 行对矢量化后的子多边形再次过滤：

```python
if min_area_pixels > 0 and part_area_pixels < min_area_pixels:
    continue
```

### 最终参数配置

| 参数 | 值 | 作用 |
|------|------|------|
| `prompt` | `farmland` | 文本提示词 |
| `use_sliding` | `1` | 启用滑动窗口 |
| `tile_size` | `1008` | 窗口尺寸（像素） |
| `overlap` | `128` | 重叠像素 |
| **`min_area_pixels`** | **`4`** | **过滤 <1m² 的微碎片（核心修复）** |
| `fill_holes` | `1` | 填充掩膜内孔洞 |
| `hole_area_pixels` | `256` | 孔洞面积阈值 |
| `simplify_tolerance` | `0.5` | 矢量简化容差 |
| `remove_overlap` | `1` | 去重叠 |
| `min_remaining_area_ratio` | `0.08` | 重叠裁剪后剩余面积比例阈值 |
| `min_sliver_width_pixels` | `1.5` | 狭长多边形最小宽度（像素） |
| `min_sliver_aspect_ratio` | `20` | 狭长多边形最小长宽比 |

---

## 五、最终结果

### 效果对比

| 指标 | 修复前 (min_area=0) | 修复后 (min_area=4) |
|------|:---:|:---:|
| 总多边形数 | 661 | **260** |
| 微碎片 <1m² | 401 (60.7%) | **0 (0%)** |
| 小地块 1-4m² | 55 (8.3%) | 55 (21.2%) |
| 中等地块 4-25m² | 34 (5.1%) | 34 (13.1%) |
| 大地块 ≥25m² | 171 (25.9%) | 171 (65.8%) |
| 最小面积 | 0.125m² | 1.0m² |
| 最大面积 | 33,580m² | 33,580m² |
| 中位数面积 | 0.25m² | 609m² |
| 处理耗时 | 167s（首次加载模型） | 52s（模型已缓存） |

### 关键结论

- 401 个微碎片全部消除
- 171 个大地块完整保留，无一丢失
- 过滤逻辑作用于**矢量化后的单个多边形级别**，不影响同一实例的其他子多边形

---

## 六、数据位置

### 桌面文件

```
C:\Users\52273\Desktop\
├── 前进村原始\           ← 修复前 (661 个多边形，含微碎片)
│   ├── 前进村原始.shp
│   ├── 前进村原始.dbf
│   ├── 前进村原始.shx
│   ├── 前进村原始.prj
│   ├── 前进村原始.cpg
│   └── 前进村原始_预览.png
├── 前进村消除\           ← 修复后 (260 个多边形，无微碎片)
│   ├── 前进村消除.shp
│   ├── 前进村消除.dbf
│   ├── 前进村消除.shx
│   ├── 前进村消除.prj
│   ├── 前进村消除.cpg
│   └── 前进村消除_预览.png
└── 前进村消除碎片_README.md  ← 本文档
```

### 服务端原始结果

```
D:\sam3_project\sam3_project\sam3_service\data\results\
├── ...mina0...20260821_104427\   ← 修复前
└── ...mina4...20260821_134749\   ← 修复后
```

---

## 七、后处理过滤机制详解

整个后处理流程分为三层过滤，按执行顺序：

### 第一层：实例去重叠

```
函数: _remove_mask_overlaps()
位置: geo_infer.py 第 507-508 行
作用对象: 整个实例（含掩膜）
参数: min_remaining_area_ratio = 0.08
逻辑:
  1. 遍历实例对，处理重叠区域
  2. 高置信度实例优先"认领"重叠像素
  3. 被裁剪后剩余面积 < 原面积 × 0.08 的实例 → 整体删除
本次结果: 删除 1 个细小重复残余
```

### 第二层：面积过滤

```
函数: clean_and_vectorize_instances()
位置: geo_infer.py 第 532-533 行 + 第 543-544 行
作用对象: 矢量化后的单个多边形
参数: min_area_pixels = 4
逻辑:
  1. 矢量化前: 整个掩膜像素数 < 4 → 跳过
  2. 矢量化后: 子多边形面积 < 4px → 跳过
本次结果: 消除 401 个微碎片
```

### 第三层：狭长伪图斑过滤

```
函数: _is_narrow_polygon()
位置: geo_infer.py 第 545-549 行
作用对象: 矢量化后的单个多边形
参数: min_sliver_width_pixels = 1.5, min_sliver_aspect_ratio = 20
逻辑:
  1. 计算多边形的最小旋转矩形
  2. 取短边为 width，长边为 length
  3. 同时满足 width < 1.5px 且 length/width ≥ 20 → 删除
本次结果: 过滤 1 个狭长伪图斑
```

### 数据流总览

```
滑动窗口检出 156 个实例
    │
    ▼
第一层: 去重叠 → 删除 1 个残余实例 (剩余 155 个实例)
    │
    ▼
第二层: 矢量化 + 面积过滤 → 产生 260 个多边形 (过滤 401 个微碎片)
    │
    ▼
第三层: 狭长过滤 → 删除 1 个狭长伪图斑 (最终 260 个多边形... 实际过滤后仍为 260)
    │
    ▼
导出 Shapefile: 前进村消除.shp (260 个多边形)
```

---

## 八、调参建议

### 碎片过滤（min_area_pixels）

| 值 | 效果 | 适用场景 |
|----|------|---------|
| 0 | 保留所有碎片 | 不推荐，会产生大量微碎片 |
| 4 | 过滤 <1m² | **推荐默认值** |
| 16 | 过滤 <4m² | 地块较大、不需要细小田埂 |
| 100 | 过滤 <25m² | 仅保留大地块 |

### 狭长过滤（min_sliver_width_pixels / min_sliver_aspect_ratio）

| 配置 | 宽度阈值 | 长宽比阈值 | 适用场景 |
|------|---------|-----------|---------|
| 保守 | 1.5 | 20 | **推荐默认值** |
| 进阶 | 2.0 | 15 | 仍有明显细条时使用 |
| 极端 | 5.0 | 5 | 仅用于诊断，会过滤真实狭长地块 |

### 去重叠（min_remaining_area_ratio）

| 值 | 效果 | 风险 |
|----|------|------|
| 0.08 | 仅删除极度碎化的残余 | 无 |
| 0.10 | 略微更严 | 无 |
| 0.50 | 删除任何损失过半的实例 | 可能误删真实地块 |

**注意**: `min_remaining_area_ratio` 影响远大于狭长参数，因为它删除整个实例（连带所有子多边形）。从 0.08 调到 0.50 会导致多边形从 661 降到 295。

---

## 九、环境与依赖

### 系统环境

- OS: Windows 11
- GPU: CUDA 0（需支持 3.45GB 模型加载）
- Python 虚拟环境: `D:\sam3_project\sam3_env\`

### 环境变量

```powershell
$env:SAM3_ROOT = "D:\sam3_project\sam3_project\sam3"
$env:SAM3_CHECKPOINT = "D:\sam3_project\sam3_project\sam3\model\sam3.pt"
$env:CUDA_VISIBLE_DEVICES = "0"
$env:SAM3_MAX_UPLOAD_MB = "500"
$env:PYTHONUTF8 = "1"
$env:PROJ_LIB = "C:\ITS\gdal\bin\proj7\share"
$env:GDAL_DATA = "C:\ITS\gdal\bin\gdal-data"
```

### 服务启动

```powershell
Set-Location "D:\sam3_project\sam3_project\sam3_service"
& "D:\sam3_project\sam3_env\Scripts\python.exe" -m uvicorn main:app --host 127.0.0.1 --port 8073 --workers 1
```

### API 调用

```
POST http://127.0.0.1:8073/api/segment_geo
Content-Type: multipart/form-data

参数:
  image:                    (文件) qianjin_village_05m.tif
  prompt:                   farmland
  use_sliding:              1
  tile_size:                1008
  overlap:                  128
  min_area_pixels:          4
  fill_holes:               1
  hole_area_pixels:         256
  simplify_tolerance:       0.5
  remove_overlap:           1
  min_remaining_area_ratio: 0.08
  min_sliver_width_pixels:  1.5
  min_sliver_aspect_ratio:  20
```

---

## 十、修复过程中解决的技术问题

| # | 问题 | 修复方法 |
|---|------|---------|
| 1 | torchvision 0.26.0 与 torch 2.10.0 ABI 不匹配 | 降级为 torchvision 0.25.0+cu128 |
| 2 | 缺失 joblib、narwhals、threadpoolctl | pip 安装缺失依赖 |
| 3 | fiona 在中文 Windows 上因 GDAL GBK 编码崩溃 | 修补 `fiona/env.py` 的 `defenv()` 捕获 SystemError |
| 4 | `geo_infer.py:557` NumPy 数组布尔值歧义 | `inst.get("box")` → `inst.get("box") is not None` |
| 5 | PROJ_LIB 未设置导致投影失败 | 设为 `C:\ITS\gdal\bin\proj7\share` |

---

*文档生成时间: 2026-08-21*
*服务版本: SAM3 Service @ D:\sam3_project\sam3_project\sam3_service*
