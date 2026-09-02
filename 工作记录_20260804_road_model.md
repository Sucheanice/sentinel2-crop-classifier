# 工作记录 — 蓬南镇作物分类全流程

**日期**: 2026-08-04

---

## 一、道路分割模型：D-LinkNet / U-Net / DeepLabV3+ 对比

### 背景

蓬南镇全流程测试中，WorldCover 10m 耕地面具过滤不掉田间土路，导致道路被错分作物。尝试了三种传统方法均失败：

| 方法 | 失败原因 |
|------|----------|
| S2 NDVI 时序 | 10m 混合像素，路被作物信号淹没 |
| DOM RGB 颜色 | 土路 vs 田内裸土颜色相同 |
| 几何形态过滤 | 土路弯弯曲曲，非规则长条 |

结论：田间土路只能用深度学习做像素级语义分割。

---

## 数据集

**WHU-RuR**（武汉大学）
- 下载链接：https://www.scidb.cn/en/detail?dataSetId=9824c6ce552244e087dd5d4f7ad23883
- 约 3 万对 1024×1024 高分影像 + 二值道路标注
- 7 省农村道路（含四川），免费学术许可
- 特征：专门针对田间细碎机耕路，论文基准为 D-LinkNet

**LoveDA**（武汉大学）— 下载因 Zenodo SSL 中断，暂未获取

---

## 模型架构

**D-LinkNet**（DeepGlobe 道路提取冠军）
- 结构：ResNet-34 编码器 + 空洞卷积（Dilated Conv）中间层 + 反卷积解码器
- 专为细长道路结构优化，扩大感受野
- 参数量：34.5M

---

## 训练配置

| 项目 | 值 |
|------|-----|
| 服务器 | 10.62.11.15, root / Ccyy123@ |
| GPU | 5 × RTX 4090 (GPU 2,3,4,5,6)，避开 Xinference 占用卡 |
| 框架 | PyTorch 2.6.0 + CUDA 12.4, DDP 多卡 |
| 训练/验证 | 14,688 / 13,082 对 |
| 批量 | 40 (8×5卡) |
| Epochs | 30 |
| 输入 | 512×512 随机裁剪 |
| 损失函数 | BCE + Dice (组合) |
| 数据目录 | /home/data/datasets/ |
| 输出目录 | /home/data/road_model/ |
| 日志 | /home/data/road_model/train.log |
| 训练脚本 | train_road_model.py（项目目录）|

---

## 服务器检查命令

```bash
tail -f /home/data/road_model/train.log      # 查看训练进度
watch -n 2 nvidia-smi                         # GPU 实时状态
ps aux | grep train_road                      # 进程状态
```

---

## 待办

- [x] 训练完成下载最优模型
- [x] 用蓬南镇 DOM 测试道路分割效果
- [x] 将道路掩膜集成到 segment_parcels.py
- [x] 端到端用道路过滤重新测试蓬南镇

---

## 二、道路模型三架构对比（WHU-RuR 数据集，服务器 6×RTX 4090）

| 模型 | 参数 | 最佳 Val IoU | 最佳 Val F1 | 最佳轮次 |
|------|------|-------------|-------------|----------|
| **U-Net (ResNet-34)** | 24.8M | **0.3648** | ~0.51 | E19 |
| DeepLabV3+ (ResNet-34) | 20.5M | 0.3461 | 0.5008 | E25 |
| D-LinkNet (ResNet-34) | 34.5M | 0.284 | 0.427 | E30 |

### 结论
- **U-Net 效果最好**，IoU 比 D-LinkNet 高 28%。DeepLabV3+ 次之。
- U-Net 训练约 80 分钟（30 epochs, 3×4090），DeepLabV3+ 约 75 分钟。
- D-LinkNet 表现最弱，可能因 WHU-RuR 数据多样性不足导致孔洞卷积退化。
- **后期建议：用 U-Net 作为主力道路模型，可考虑更长时间训练或更大输入尺寸。**

---

## 三、作物分类模型 v2 升级：多光谱特征工程 + LightGBM

### 动机
原有作物分类仅用 4 个波段 (B02/B03/B04/B08) 的 zonal mean + 5 种 VI，没有利用：
- 红边波段 (B05/B06/B07) 对植被叶绿素的诊断性信息
- 时序统计（跨日期 min/max/std/range 捕捉物候动态）
- 物候指标（NDVI 振幅、生长/衰老速率）

### v2 特征体系（5层）
```
Layer 1 - 原始波段 zonal mean: 7波段×4日期 = 28维
Layer 2 - 植被指数: NDVI/EVI/NDWI/SAVI/LSWI/NDMI/NDRE/RVI ×4日期 = 32维
Layer 3 - 时序统计: 每波段/VI 的 min/max/mean/std/range = 75维
Layer 4 - 物候指标: 振幅/生长斜率/衰老率/峰值/积分/淹水 = 6维
Layer 5 - 首尾差值: NDVI/EVI/NDWI/SAVI delta = 4维
────────────────────────────────────────────────
总计: 145维 → LightGBM gain 选前40
```

### 训练设置
- 训练数据: `待训练数据6_anju/features_anju.csv`（蓬南镇 S2 + 安居标注 SHP）
- 14,810 样本（7,431 水稻 + 7,379 玉米）
- 5-fold 分层交叉验证

### 结果对比

| 指标 | 旧模型 (XGBoost) | v2 模型 (LightGBM) |
|------|------------------|---------------------|
| CV Accuracy | 0.8799 | **0.8942** (+1.4%) |
| CV F1-w | — | **0.8941** |
| 特征数 | 30/44 | 40/145 |
| 分类器 | XGBoost | LightGBM |

### Top 10 重要特征
```
1. 2025-05-20_B11         — 5月 SWIR（早期土壤/水体信号）
2. LSWI_2025-08-03        — 8月地表水指数（水稻季末淹水）
3. 2025-07-16_B11         — 7月 SWIR
4. EVI_2025-08-03         — 8月增强植被指数
5. NDWI_2025-05-20        — 5月水体指数（水稻田早期淹水）
6. DELTA_NDWI             — NDWI 首尾变化
7. PHENO_senescence_rate  — NDVI 衰老速率
8. EVI_2025-05-20         — 5月 EVI
9. TSTAT_B11_mean         — SWIR 跨季节均值
10. 2025-05-20_B08        — 5月近红外
```
→ LSWI/NDWI/B11 等水相关指标占据高位，恰好是区分水稻（淹水）和玉米（旱地）的关键。

### 蓬南镇预测结果
- v2 (LightGBM): 126 玉米 (0.2726 km²) + 74 水稻 (0.1680 km²) = 0.4406 km²
- 旧 (XGBoost): 126 玉米 (0.2716 km²) + 74 水稻 (0.1690 km²) = 0.4406 km²
- 两模型对蓬南镇 200 个地块的分类高度一致（面积差异 < 0.1%）

### 待改进
- 红边波段 (B05/B06/B07): element84 STAC 已不可访问旧场景，后续需用新 API 重下全量数据
- 可进一步增加 Sentinel-1 SAR 数据（VV/VH）用于水稻淹水检测
- 添加纹理特征(GLCM)改善地块内部异质性地块的分类

---

## 代码文件变更

| 文件 | 变更 |
|------|------|
| `common.py` | 重写：新增 compute_feature_matrix / select_features_lightgbm / compute_temporal_stats / compute_phenology |
| `download_sentinel_v2.py` | 增加 B05/B06/B07 红边波段到下载列表 |
| `segment_parcels.py` | 集成 D-LinkNet 道路掩膜过滤 (--road-mask 参数) |
| `run_pengnan_pipeline.py` | 升级 predict_parcel() 使用 compute_feature_matrix + LightGBM |
| `predict_parcel.py` | 同步升级特征计算 |
| `train_save_model_v2.py` | 新建：LightGBM + 多光谱特征训练脚本 |
| `detect_roads.py` | D-LinkNet 道路检测推理脚本 |
| `extract_pixels_v2.py` | 扩大波段列表至 B02-B12 |

### 输出文件
- `蓬溪县数据/crop_pengnan_parcels_v2.shp` — v2 LightGBM 多光谱特征分类结果
- `蓬溪县数据/crop_pengnan_parcels_roadless.shp` — 旧模型 + 道路掩膜过滤
- `road_mask_pengnan.tif` — D-LinkNet 道路检测结果
- `待训练数据6_anju/crop_model_anju_v2.pkl` — v2 LightGBM 模型
- 服务器: `/home/data/unet_model/road_model_best.pth` — U-Net 最佳模型
- 服务器: `/home/data/deeplabv3p_model/road_model_best.pth` — DeepLabV3+ 最佳模型

---

## 五、安居区全流程结果

### 运行参数
- 地块分割: `segmented_parcels_anju_1m_wc.shp` (WorldCover掩膜, 3177 parcels)
- S2数据: `待训练数据6_v2_48RWU_anju_cropped/` (已裁剪, 4场景×7波段)
- 模型: `crop_model_anju_v2.pkl` (LightGBM v2)
- 脚本: `run_anju_pipeline.py`

### 预测结果
| 作物 | 地块数 | 面积 | 占比 |
|------|--------|------|------|
| 水稻 | 1838 | 4.2251 km² | 57.9% |
| 玉米 | 1338 | 3.1138 km² | 42.1% |
| **合计** | **3176** | **7.34 km²** | |

### 与蓬南镇对比
| | 蓬南镇 | 安居区 |
|--|--------|--------|
| 影像分辨率 | 0.075m | 1m |
| 地块数 | 200 | 3176 |
| 水稻占比 | 37% | **58%** |
| 玉米占比 | 63% | 42% |

> 安居区水稻占比远高于蓬南镇，与安居区地势低平、水网密集一致。

---

## 六、小春作物分类 v4：小麦 vs 油菜 (LightGBM + 多光谱特征)

### 背景
- 大春模型 (v3) 完成水稻/玉米二分类，5区合并 93.58% 准确率
- 按 [作物识别要求.md](file:///e:/工作相关/2026年/0624 待测试数据/作物识别要求.md)，还需识别小麦、油菜
- 小春生育期：播种/出苗(11-12月) → 越冬(1月) → 返青/油菜花(3月) → 抽穗/收获前(4-5月)

### 数据
| 类型 | 详情 |
|------|------|
| S2影像 | 小春_s2_48RWU/ 4景×10波段，tile 48RWU，~4.5GB |
| 标注SHP | 2024小春/遂宁市/{安居,大英,船山}.shp |
| 场景 | 2024-12-11(cloud 9.5%), 2025-01-20(cloud 1.2%), 2025-03-26(cloud 0.0%), 2025-05-05(cloud 49.4%) |

### 区县样本
| 区县 | 地块数 | 小麦 | 油菜 |
|------|--------|------|------|
| 安居区 | 13,317 | 8,692 | 4,625 |
| 大英县 | 4,387 | 2,715 | 1,672 |
| 船山区 | 2,101 | 523 | 1,578 |
| **合计** | **19,805** | **11,930** | **7,875** |

### 训练配置
- 训练脚本: `train_xiaochun_v4.py`
- 预处理后样本: 17,550 (剔除 521 异常值)
- 训练/测试: 14,040 / 3,510 (80/20 分层)
- 特征工程: 5层172维 → LightGBM gain 选 Top 40
- 5-fold StratifiedKFold 交叉验证
- 分类器: LightGBM (300 trees, max_depth=5, learning_rate=0.03)

### 结果
| 指标 | 值 |
|------|-----|
| CV Accuracy | 0.8152 ± 0.0051 |
| Test Accuracy | 0.8157 |
| Test F1-w | 0.8154 |

**混淆矩阵：**

| | 预测小麦 | 预测油菜 | Recall |
|--|---------|---------|--------|
| 实际小麦 | 1841 | 313 | 85.5% |
| 实际油菜 | 334 | 1022 | 75.4% |

**分类报告：**
| 类别 | Precision | Recall | F1 | Support |
|------|-----------|--------|----|---------|
| 小麦 | 0.85 | 0.85 | 0.85 | 2154 |
| 油菜 | 0.77 | 0.75 | 0.76 | 1356 |

### Top 10 重要特征
```
1. NDWI_2025-03-26         — 3月水体指数（油菜花/小麦返青水分差异）
2. 2025-03-26_B03          — 3月绿色波段（油菜花期绿色反射特征）
3. EVI_2025-01-20          — 1月增强植被指数（越冬期覆盖度差异）
4. EVI_2025-03-26          — 3月EVI（油菜花导致EVI下降）
5. TSTAT_B08_mean          — 近红外跨季节均值
6. LSWI_2025-03-26         — 3月地表水指数
7. 2025-01-20_B02          — 1月蓝色波段
8. PHENO_growth_rate       — NDVI生长斜率（小麦spring green-up更快）
9. NDWI_2025-01-20         — 1月水体指数
10. 2025-03-26_B04         — 3月红色波段
```
→ 3月（油菜花期）是关键区分窗口：油菜开花时绿色反射减弱、NDWI变化明显

### 各层贡献度
| 特征层 | Gain占比 | 特征数 |
|--------|----------|--------|
| Layer2_植被指数 | 47.0% | 32 |
| Layer1_原始波段 | 26.6% | 40 |
| Layer3_时序统计 | 23.0% | 90 |
| Layer4_物候 | 2.6% | 6 |
| Layer5_首尾差 | 0.8% | 4 |

### 分析与待改进
- **准确率 81.6% 低于大春 93.6%**，原因是 小麦/油菜 同为越冬作物，物候重叠度高
- 3月油菜花期是主要区分窗口，但 S2 10m 分辨率无法精确捕捉黄色花信号
- **改进方向**:
  1. 增加 Sentinel-1 SAR 数据（VV/VH极化），油菜角果期 vs 小麦抽穗期后向散射差异大
  2. 增加4月油菜花盛期的 S2 影像（当前5月 cloud 49.4% 质量差）
  3. 纹理特征 (GLCM)：油菜花田 vs 麦田纹理差异
  4. 尝试数据增强/类别平衡 (小麦:油菜 ≈ 3:2，欠采样或 class_weight)
  5. 增加更多区县训练数据（当前只有遂宁3区）

### 输出文件
- `小春_v4_output/crop_model_xiaochun_v4.pkl` — v4 LightGBM 模型
- `小春_v4_output/xiaochun_zonal_mean.csv` — zonal mean 原始数据
- `小春_v4_output/xiaochun_features_full.csv` — 172维完整特征矩阵
- `小春_v4_output/xiaochun_features_importance.csv` — 特征重要性排名
- `小春_v4_output/xiaochun_test_predictions.csv` — 测试集预测
