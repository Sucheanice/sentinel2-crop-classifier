# GEE 直连与影像使用操作文档

> 本文记录本项目如何**直接连接 Google Earth Engine（GEE）**，以及如何**按地方（区县/村地块）取用 Sentinel-2 影像**并提取特征。
> 核心思路：**影像不下载到本地**，直接在 GEE 云端按「区域 + 日期 + 云量」选景、合成、提取每个地块的统计特征，只把几 MB 的特征 CSV 拉回本地训练。

---

## 一、总体流程（一张图看懂）

```
本地 SHP（区县/村地块边界）
        │  geopandas 读取，转成 ee.FeatureCollection
        ▼
GEE 云端：filterBounds(区域) + filterDate(时相窗口)
        │  SCL 云掩膜 → median 中值合成 → 4~6 期 × 10 波段
        ▼
reduceRegions 提取每个地块的 zonal mean
        │  getInfo() 拉回
        ▼
本地特征 CSV → 特征工程 → LightGBM 训练 / 推理
```

关键点：**影像全程留在 GEE，不下载**；唯一导出的是特征表（每地块 40~60 列光谱均值），体量极小。

---

## 二、直连 GEE 的环境准备（一次性）

### 2.1 账号注册

- 需要一个 **Google 账号 + 可用的代理**。
- 注册入口：`signup.earthengine.google.com`，类型选 **Non-commercial（非商用）**。
- 审批通常几分钟到 2 天。

### 2.2 安装 Python API

```powershell
pip install earthengine-api
```

本项目已验证版本：`earthengine-api 1.7.40`。

### 2.3 绑定项目并认证（一次固化）

```powershell
# 1) 浏览器登录认证
earthengine authenticate

# 2) 指定 GEE 项目（把项目 ID 固化到本机）
earthengine set_project neat-shell-506206-q0
```

本项目 GEE 项目 ID：**`neat-shell-506206-q0`**（显示名 "My Project 27848"，但真实 ID 是 `neat-shell-506206-q0`）。

> 执行 `set_project` 之后，Python 脚本里只需要 `ee.Initialize()`，**不用再传 credentials 参数**。

---

## 三、代码里怎么"直连"（三行打通）

三个 GEE 脚本开头都是同一套：

```python
import ee

ee.Initialize()   # 已用 earthengine set_project 固化项目，无需参数
```

涉及脚本：

| 脚本 | 用途 | 时相 |
|------|------|------|
| [gee_extract_features.py](file:///e:/工作相关/2026年/0624 待测试数据/gee_extract_features.py) | 小春训练特征提取（小麦/油菜） | 4 期 |
| [gee_extract_features_v2.py](file:///e:/工作相关/2026年/0624 待测试数据/gee_extract_features_v2.py) | 加密时相验证（6 期） | 6 期 |
| [gee_extract_zitong.py](file:///e:/工作相关/2026年/0624 待测试数据/gee_extract_zitong.py) | 无真值推理用特征提取（不过滤作物） | 4 期 |

---

## 四、用哪个影像数据集

统一使用 Sentinel-2 L2A（大气校正后的地表反射率）在 GEE 的镜像集合：

```python
ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
```

- 与本地 Element84 COG 同源，含 **10 个光谱波段 + SCL（场景分类）+ QA60**。
- 取值是 **×10000 的 DN**（反射率 × 10000），特征工程时如需 0–1 反射率要除以 10000。

用到的 10 个波段：

```
B2  B3  B4  B5  B6  B7  B8  B8A  B11  B12
```

本地代码里统一映射成 `B02…B12`（见各脚本的 `BAND_MAP`）。

---

## 五、怎么"按相应地方"选影像（核心）

影像的"地方"由**本地 SHP 的几何范围**决定，不是手填瓦片号。流程如下：

### 5.1 读本地 SHP → 转成 GEE 矢量

```python
import geopandas as gpd
gdf = gpd.read_file(shp_path)                    # 读区县/村地块 SHP
gdf['geometry'] = gdf['geometry'].simplify(tolerance=0.0001, preserve_topology=True)  # 简化顶点，约10m
gdf['fid'] = range(len(gdf))                     # 加自增 fid
fc = ee.FeatureCollection(gdf[keep_cols].__geo_interface__['features'])
```

### 5.2 用区域范围选影像 + 按日期窗口合成

```python
def mask_s2_clouds(img):
    scl = img.select('SCL')
    cloud = scl.eq(3).Or(scl.eq(8)).Or(scl.eq(9)).Or(scl.eq(10))  # 云影/中云/高云/薄卷云
    return img.updateMask(cloud.Not())

def build_composite(roi):
    s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(roi)
    composite = None
    for prefix, start, end in WINDOWS:
        col = s2.filterDate(start, end).map(mask_s2_clouds)
        med = col.select(BANDS).median().rename([f'{prefix}_{b}' for b in BANDS])
        composite = med if composite is None else composite.addBands(med)
    return composite
```

- `filterBounds(roi)`：**"相应地方"** —— 自动筛出覆盖该地块范围的影像。
- `filterDate(start, end)`：限定物候时相窗口。
- `mask_s2_clouds`：用 SCL 去掉云/云影（类别 3/8/9/10）。
- `.median()`：对窗口内多景做**中值合成**（抗云、抗异常）。

### 5.3 时相窗口（本项目小春 2024–2025 生长季）

**4 期窗口**（`gee_extract_features.py` / `gee_extract_zitong.py`）：

| 期次 | 含义 | 日期窗口 |
|------|------|----------|
| P1 | 越冬 | 2024-12-01 ~ 2025-01-05 |
| P2 | 休眠 | 2025-01-06 ~ 2025-02-10 |
| P3 | 返青/油菜开花 | 2025-03-01 ~ 2025-04-05 |
| P4 | 收获前 | 2025-04-25 ~ 2025-05-31 |

**6 期窗口**（`gee_extract_features_v2.py`，在 4 期基础上补 2 月、4 月）：

```
P1 2024-12  P2 2025-01  P3 2025-02  P4 2025-03  P5 2025-04  P6 2025-05
```

> 若要换季节/换作物，只需改 `WINDOWS` 里的日期即可；换影像类型则改 `ee.ImageCollection(...)` 集合 ID。

---

## 六、提取地块特征（reduceRegions）

对合成好的多期影像，逐地块取"均值"作为该地块的光谱特征：

```python
reduced = composite.reduceRegions(
    collection=fc,
    reducer=ee.Reducer.mean(),
    scale=10,          # 对应 S2 10m 分辨率
    tileScale=4,
).select(['fid', 'ZWMC', 'QXMC'] + band_names)

feats = reduced.getInfo()['features']   # 拉回本地
```

输出每地块 **期次 × 波段 = 40 列（4 期）或 60 列（6 期）** 的 zonal mean。

---

## 七、大县分批（踩坑必读）

直接传整县地块会超 GEE 的 10MB payload，且每批重算整个县的中值会很慢。解决办法：

1. **简化几何**：`simplify(tolerance=0.0001)`，约 10m 容差，对 zonal mean 影响可忽略。
2. **大县分批**：`n > 5000` 时按 **3000 块/批** 分批，每批用**该批子集的范围**独立合成（避免每批重算全县中值）。

```python
if batch_size is None and n > 5000:
    batch_size = 3000
```

---

## 八、完整操作步骤（从零到产出特征 CSV）

1. **准备本地 SHP**：区县/村地块边界（含 `ZWMC`=作物名、`QXMC`=区县名等字段）。
2. **确认 GEE 已打通**：跑一遍 `ee.Initialize()` 不报错。
3. **改脚本入口**：把 `__main__` 里的 `shp_path` / `out_csv` 换成目标区县。
4. **运行**：
   ```powershell
   python gee_extract_features.py
   ```
5. **得到特征 CSV**，进入本地特征工程 + 训练 / 推理。

下游使用链路：

- 训练：`gee_*.csv` → [train_xiaochun_gee.py](file:///e:/工作相关/2026年/0624 待测试数据/train_xiaochun_gee.py) / [exp_finalize_v2.py](file:///e:/工作相关/2026年/0624 待测试数据/exp_finalize_v2.py) → 模型 `.pkl`
- 推理：`gee_梓潼县_小春特征.csv` → [predict_zitong.py](file:///e:/工作相关/2026年/0624 待测试数据/predict_zitong.py) → 预测 CSV + SHP

---

## 九、常见坑速查

| 问题 | 原因 | 处理 |
|------|------|------|
| `ee.Initialize()` 报错 | 未 `set_project` 或未认证 | `earthengine authenticate` + `earthengine set_project <项目ID>` |
| payload 超 10MB | 地块顶点过多 / 一次性传太多 | 简化几何 tolerance=0.0001 + 大县分批 |
| `getInfo()` 极慢 | 每批重算整个县中值合成 | 每批用子集 roi 独立合成 |
| EVI/SAVI 数值异常 | GEE 是 ×10000 DN，常数 +1.0/+0.5 按 0–1 反射率设计 | 树模型对单调缩放不敏感，实际影响可忽略；确需时除以 10000 |

---

## 十、关键结论（本项目已闭环）

- **免下载影像**：云端选景 → 合成 → 提特征 → 本地训练，一套流程可复用到任意区县/作物。
- **"相应地方"的影像由 SHP 几何自动决定**（`filterBounds`），不再需要手工确定下载哪个瓦片。
- 小春小麦/油菜：GEE 4 期特征 + WRI 花敏感指数 + 安居区标签清洗，最终固化模型 **Acc 89.89%**（10m 哨兵光学天花板约 89%，95% 需高分影像补边界）。
