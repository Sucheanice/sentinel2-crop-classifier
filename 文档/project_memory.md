# 项目记忆 — 遂宁/广安/江油 作物遥感分类

## 项目概述
基于 Sentinel-2 多光谱影像 + Sentinel-1 SAR + 标注地块 SHP，使用 LightGBM 对四川地区进行作物分类。

## 目标
- 作物识别准确率 ≥ 95%（详见 `作物识别要求.md`）
- 大春：水稻 vs 玉米 (二分类)
- 小春：小麦 vs 油菜 (二分类)
- 后续扩展：蔬菜、柑橘、蜜柚等多类

## 技术路线
- **分类器**: LightGBM (梯度提升树)
- **特征工程**: 5层145维多光谱特征 (原始波段/植被指数/时序统计/物候/首尾差)
- **数据源**: Sentinel-2 L2A (10波段, element84 STAC API), Sentinel-1 GRD (VV+VH)
- **标注**: 各区县 SHP (字段 ZWMC=作物名称)
- **代码模块**: `common.py` (特征计算), `train_xiaochun_v4.py` (小春训练), `train_save_model_v2.py` (大春训练)

## 当前进度

### 已完成
- [x] 大春 v3 模型: 5区合并, 水稻vs玉米, Test Acc=93.58%
- [x] 小春 v4 基础模型: 3区合并, 小麦vs油菜, Test Acc=81.57%
- [x] 小春 v4_cw: class_weight='balanced', 油菜Recall 75.4%→84.5%
- [x] ExG+WorldCover 联合耕地面具
- [x] 道路模型: U-Net/DeepLabV3+/D-LinkNet
- [x] 遂宁 S1 下载: 4景全部完成 (vv+vh)
- [x] 江油前进村 S2+S1 全部下载完成
- [x] S1 覆盖 bug 诊断: 后3期不覆盖遂宁, 根因确认
- [x] `download_sentinel1_v2.py` 修复: geometry 预检 + 断点续传
- [x] `predict_jiangyou.py` 预测脚本就绪
- [x] `工作记录_20260807_s1_pipeline.md` 工作记录
- [x] `代码架构整理.md` (本文档附录)

### 进行中
- [ ] 遂宁 S1 重新下载中 (v2 geometry 过滤, 断点续传)
  - 2024-12-15: 完成 (保持)
  - 2025-01-20 (替换 01-27): vv完成, vh下载中
  - 2025-03-26 (替换 03-28): 待搜索
  - 2025-05 期: 待搜索

### 待做 (优先级排序)
1. [ ] S1 下载完成后自动运行: `run_xiaochun_s1_pipeline.py` (遂宁 S1+S2 重新训练)
2. [ ] 运行 `predict_jiangyou.py` (江油前进村预测)
3. [ ] 重构为统一预测管线 `run_prediction_pipeline.py` (去掉硬编码)
4. [ ] 广安 5 区 S2 下载 (tiles 48RXV/48RXU/48RWV)
5. [ ] SAM3 地块边界集成
6. [ ] 多类作物扩展 (蔬菜/柑橘/蜜柚)

## 关键文件

### 核心模块
| 文件 | 功能 | 状态 |
|------|------|------|
| `common.py` | 5层145维多光谱特征计算 | 稳定 |
| `extract_sar_features.py` | S1 zonal mean + S1+S2 合并 | 稳定 (已修 GCP 对齐) |
| `extract_pixels_v2.py` | 像素级特征提取 | 稳定 |
| `download_sentinel1_v2.py` | **S1 下载 (推荐, 带覆盖预检+续传)** | 最新 |
| `download_sentinel_v2.py` | S2 10波段下载 | 稳定 |

### 训练
| 文件 | 作物 | 备注 |
|------|------|------|
| `train_xiaochun_v4.py` | 小麦vs油菜 | S2-only |
| `train_xiaochun_v4_sar.py` | 小麦vs油菜 | S1+S2 |
| `train_save_model_v2.py` | 水稻vs玉米 | 大春 |
| `train_save_model_v3.py` | 水稻vs玉米 | 大春 v3 |

### 管线
| 文件 | 功能 |
|------|------|
| `run_xiaochun_s1_pipeline.py` | 一键: 检查→提取→训练→对比 |
| `run_anju_pipeline.py` | 安居区流程 |
| `run_pengnan_pipeline.py` | 蓬南镇流程 |

### 预测
| 文件 | 功能 |
|------|------|
| `predict_jiangyou.py` | 江油前进村预测 |
| `predict_parcel.py` | 地块预测通用 |

### 工具
| 文件 | 功能 |
|------|------|
| `download_jiangyou.py` | 江油 S2+S1 下载 (已使用完毕) |
| `fix_missing_vv.py` | 修复缺失 vv.tif |
| `check_jiangyou_gdb.py` | 检查江油 GDB |
| `monitor_downloads.ps1` | 下载监控 |

## 数据路径
- S2 大春: `待训练数据6_v2_48RWU_anju_cropped/`
- S2 小春 (遂宁): `小春_s2_48RWU/`
- S1 小春 (遂宁): `小春_s1_48RWU/`
- S2 江油: `江油_s2/` (tile 48SWA/SVA)
- S1 江油: `江油_s1/` (tile 48SWA/SVA)
- 大春 SHP: `蓬溪县数据/遂宁市矢量数据原始/`
- 小春 SHP: `2024小春/遂宁市/` (安居区/大英县/船山区)
- 江油 SHP: `待测试数据前进0806/前进0806.gdb` (dltb 层, 1503 地块)
- 广安 SHP: `2024小春/广安市/`
- 模型 (大春): `待训练数据6_anju/crop_model_5districts_v3.pkl`
- 模型 (小春 v4_cw): `小春_v4_output/crop_model_xiaochun_v4_cw.pkl` (最佳)
- 模型 (小春 v4_sar): `小春_v4_output/crop_model_xiaochun_v4_sar.pkl` (S1+S2)
- 模型 (小春 v4_sar_cw): `小春_v4_output/crop_model_xiaochun_v4_sar_cw.pkl`

## MGRS Tiles
- 48RWU: 遂宁 (安居/大英/蓬溪/船山/射洪)
- 48SWA: 江油 (马角镇前进村)
- 48SVA: 江油 (1月/5月 S2)
- 48RXV, 48RXU, 48RWV: 广安

## 工作记录
- `工作记录_20260807_s1_pipeline.md` ← 最新
- `工作记录_20260804_road_model.md` (道路模型)
- `工作记录_20260731_segment.md` (地块分割)

## 服务器
- 10.62.11.15, root/Ccyy123@
- GPU: 5×RTX 4090
- SAM3 项目: /data4/sam3_project
