# -*- coding: utf-8 -*-
"""公共模块：多光谱特征工程 + LightGBM 分类。

特征体系（v2 升级）：
  Layer 1 - 原始波段 zonal mean：B02/B03/B04/B05/B06/B07/B08/B8A/B11/B12
  Layer 2 - 植被指数时序：NDVI/EVI/NDWI/SAVI/LSWI/NDMI/NDRE 每景
  Layer 3 - 时序统计：每波段/VI 跨日期 min/max/mean/std/range
  Layer 4 - 物候指标：NDVI振幅、峰值日期、生长斜率
  Layer 5 - 波段比值：RVI(B08/B04) 等

用法：
  from common import compute_feature_matrix, select_features_lightgbm
"""
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Sentinel-2 波段全量（含红边）
BANDS_10M = ["B02", "B03", "B04", "B08"]
BANDS_20M = ["B05", "B06", "B07", "B8A", "B11", "B12"]
BANDS_ALL = BANDS_10M + BANDS_20M  # 全部10个波段

# 训练用波段（Anju数据只有这些，保持兼容）
BANDS_TRAIN = ["B02", "B03", "B04", "B08"]


def compute_indices(blue, green, red, nir, swir1=None, re1=None, re2=None):
    """逐像素计算 7 种植被/水体/红边指数。

    Returns: (evi, ndwi, savi, lswi, ndmi, ndre, rvi)
    """
    eps = 1e-10

    # 核心植被指数
    evi = np.where(np.abs(nir + 6.0 * red - 7.5 * blue + 1.0) > eps,
                   2.5 * (nir - red) / (nir + 6.0 * red - 7.5 * blue + 1.0), 0.0)
    ndwi = np.where(np.abs(green + nir) > eps, (green - nir) / (green + nir), 0.0)
    savi = np.where(np.abs(nir + red + 0.5) > eps, 1.5 * (nir - red) / (nir + red + 0.5), 0.0)

    # 水分/湿度指数
    if swir1 is not None:
        denom_lswi = nir + swir1
        lswi = np.where(np.abs(denom_lswi) > eps, (nir - swir1) / denom_lswi, 0.0)
        denom_ndmi = nir + swir1
        ndmi = np.where(np.abs(denom_ndmi) > eps, (nir - swir1) / denom_ndmi, 0.0)
    else:
        lswi = np.zeros_like(nir)
        ndmi = np.zeros_like(nir)

    # 红边指数 (需 B05/B06/B07)
    if re1 is not None:  # B05
        ndre = np.where(np.abs(nir + re1) > eps, (nir - re1) / (nir + re1), 0.0)
    else:
        ndre = np.zeros_like(nir)

    # 比值植被指数
    rvi = np.where(np.abs(red) > eps, nir / (red + eps), 0.0)

    return evi, ndwi, savi, lswi, ndmi, ndre, rvi


def compute_temporal_stats(values_2d):
    """跨日期时序统计。

    Args:
        values_2d: (n_samples, n_dates) 数组
    Returns:
        stats dict: {min, max, mean, std, range, first, last}
    """
    valid = np.where(~np.isnan(values_2d), values_2d, 0.0)
    return {
        'min': np.min(values_2d, axis=1),
        'max': np.max(values_2d, axis=1),
        'mean': np.mean(values_2d, axis=1),
        'std': np.nanstd(values_2d, axis=1),
        'range': np.max(values_2d, axis=1) - np.min(values_2d, axis=1),
    }


def compute_phenology(ndvi_series):
    """从 NDVI 时序计算物候指标。

    Args:
        ndvi_series: (n_samples, n_dates) NDVI 值
    Returns:
        dict: {amplitude, growth_rate, senescence_rate, peak_index, integral}
    """
    n = ndvi_series.shape[1]
    amplitude = np.max(ndvi_series, axis=1) - np.min(ndvi_series, axis=1)

    # 生长斜率：前半段 vs 后半段
    half = max(2, n // 2)
    growth_rate = (np.mean(ndvi_series[:, :half], axis=1) -
                   ndvi_series[:, 0]) / (half - 1 + 1e-8)
    senescence_rate = (ndvi_series[:, -1] -
                       np.mean(ndvi_series[:, half:], axis=1)) / (n - half + 1e-8)

    peak_index = np.argmax(ndvi_series, axis=1).astype(np.float32)
    integral = np.trapz(ndvi_series, axis=1)

    # LSWI 淹水检测：如果早期 LSWI > NDVI，说明是水田
    # 这在水稻识别中特别有用

    return {
        'amplitude': np.nan_to_num(amplitude, nan=0.0),
        'growth_rate': np.nan_to_num(growth_rate, nan=0.0),
        'senescence_rate': np.nan_to_num(senescence_rate, nan=0.0),
        'peak_index': np.nan_to_num(peak_index, nan=0.0),
        'integral': np.nan_to_num(integral, nan=0.0),
    }


def compute_feature_matrix(band_values, scene_labels, available_bands=None):
    """从逐景波段 zonal mean 构建完整多光谱特征矩阵。

    Args:
        band_values: dict, key="{label}_{band}" -> (n_samples,) array
        scene_labels: list of str, 场景标签（如 ['2025-05-20', '2025-06-26', ...]）
        available_bands: list of str, 可用波段列表（默认 BANDS_ALL）

    Returns:
        features: (n_samples, n_features) np.ndarray
        feature_names: list of str
    """
    if available_bands is None:
        available_bands = BANDS_ALL

    n_scenes = len(scene_labels)
    # 先确定样本数
    first_key = list(band_values.keys())[0]
    n_samples = len(band_values[first_key])

    # 过滤实际存在的波段
    active_bands = []
    for b in available_bands:
        key = '%s_%s' % (scene_labels[0], b)
        if key in band_values:
            active_bands.append(b)

    if not active_bands:
        raise ValueError("No valid bands found in band_values!")

    feature_cols = []
    feature_data = []

    # === Layer 1: 原始波段 zonal mean ===
    for band in active_bands:
        for lbl in scene_labels:
            key = '%s_%s' % (lbl, band)
            if key in band_values:
                val = np.nan_to_num(band_values[key], nan=0.0)
                feature_data.append(val)
                feature_cols.append(key)

    # === Layer 2: 植被指数（每景）===
    nir_idx = {lbl: i for i, lbl in enumerate(scene_labels)}
    vi_names = ['NDVI', 'EVI', 'NDWI', 'SAVI', 'LSWI', 'NDMI', 'NDRE', 'RVI']

    # 收集每个场景的 VI
    vi_per_scene = {vn: [] for vn in vi_names}
    for li, lbl in enumerate(scene_labels):
        blue = band_values.get('%s_B02' % lbl, np.zeros(n_samples))
        green = band_values.get('%s_B03' % lbl, np.zeros(n_samples))
        red = band_values.get('%s_B04' % lbl, np.zeros(n_samples))
        nir = band_values.get('%s_B08' % lbl, np.zeros(n_samples))
        swir1 = band_values.get('%s_B11' % lbl, None)
        re1 = band_values.get('%s_B05' % lbl, None)

        evi_arr, ndwi_arr, savi_arr, lswi_arr, ndmi_arr, ndre_arr, rvi_arr = \
            compute_indices(blue, green, red, nir,
                            swir1=swir1 if swir1 is not None else None,
                            re1=re1 if re1 is not None else None)

        # NDVI 单独算（更可靠）
        denom = nir + red
        ndvi_arr = np.where(denom > 1e-10, (nir - red) / denom, 0.0)

        indices = [ndvi_arr, evi_arr, ndwi_arr, savi_arr,
                   lswi_arr, ndmi_arr, ndre_arr, rvi_arr]
        for vn, arr in zip(vi_names, indices):
            vi_per_scene[vn].append(arr)

    for vn in vi_names:
        for li, lbl in enumerate(scene_labels):
            feature_data.append(np.nan_to_num(vi_per_scene[vn][li], nan=0.0))
            feature_cols.append('%s_%s' % (vn, lbl))

    # === Layer 3: 时序统计（每个波段/VI 跨日期）===
    stat_names = ['min', 'max', 'mean', 'std', 'range']

    # 波段时序统计
    for band in active_bands:
        band_across = np.column_stack([
            band_values.get('%s_%s' % (lbl, band), np.zeros(n_samples))
            for lbl in scene_labels
        ])
        stats = compute_temporal_stats(np.where(band_across == 0, np.nan, band_across))
        for sn in stat_names:
            feature_data.append(np.nan_to_num(stats[sn], nan=0.0))
            feature_cols.append('TSTAT_%s_%s' % (band, sn))

    # VI 时序统计
    for vn in vi_names:
        vi_across = np.column_stack(vi_per_scene[vn])
        stats = compute_temporal_stats(vi_across)
        for sn in stat_names:
            feature_data.append(np.nan_to_num(stats[sn], nan=0.0))
            feature_cols.append('TSTAT_%s_%s' % (vn, sn))

    # === Layer 4: 物候指标 ===
    ndvi_across = np.column_stack(vi_per_scene['NDVI'])
    pheno = compute_phenology(ndvi_across)
    for pk in ['amplitude', 'growth_rate', 'senescence_rate', 'peak_index', 'integral']:
        feature_data.append(np.nan_to_num(pheno[pk], nan=0.0))
        feature_cols.append('PHENO_%s' % pk)

    # 淹水检测（水稻特征）
    if 'LSWI' in vi_per_scene and len(vi_per_scene['LSWI']) >= 2:
        lswi_early = vi_per_scene['LSWI'][0]
        ndvi_early = vi_per_scene['NDVI'][0]
        flooding = np.where(lswi_early > ndvi_early, 1.0, 0.0)
        feature_data.append(flooding)
        feature_cols.append('PHENO_flooding')

    # === Layer 5: 首尾差值（变化方向）===
    for vn in vi_names[:4]:  # NDVI, EVI, NDWI, SAVI
        delta = vi_per_scene[vn][-1] - vi_per_scene[vn][0]
        feature_data.append(np.nan_to_num(delta, nan=0.0))
        feature_cols.append('DELTA_%s' % vn)

    features = np.column_stack(feature_data).astype(np.float32)
    return features, feature_cols


def select_features_lightgbm(X, y, feature_names, top_n=40, random_state=42):
    """用 LightGBM 特征重要性选择 top_n 个特征。

    Returns: (selected_names, X_selected, selected_indices)
    """
    import lightgbm as lgb
    model = lgb.LGBMClassifier(
        n_estimators=150, max_depth=6, num_leaves=31,
        learning_rate=0.05, subsample=0.8, colsample_bytree=0.7,
        reg_alpha=0.5, reg_lambda=0.5,
        random_state=random_state, n_jobs=-1, verbose=-1,
    )
    model.fit(X, y)

    # LightGBM feature_importances_ 是 split 次数，用 gain 更准确
    importance = model.booster_.feature_importance(importance_type='gain')
    keep = min(top_n, X.shape[1])
    top_idx = np.argsort(importance)[::-1][:keep]
    selected_names = [feature_names[i] for i in top_idx]
    return selected_names, X[:, top_idx], top_idx


# 向后兼容：保留旧接口名
def select_features(X, y, feature_names, top_n=30, random_state=42):
    """[兼容旧接口] 用 LightGBM 重要性选择特征。"""
    import lightgbm as lgb
    model = lgb.LGBMClassifier(
        n_estimators=150, max_depth=6, num_leaves=31,
        learning_rate=0.05, subsample=0.8, colsample_bytree=0.7,
        reg_alpha=0.5, reg_lambda=0.5,
        num_class=len(set(y)),
        random_state=random_state, n_jobs=-1, verbose=-1,
    )
    model.fit(X, y)
    importance = model.booster_.feature_importance(importance_type='gain')
    keep = min(top_n, X.shape[1])
    top_idx = np.argsort(importance)[::-1][:keep]
    selected_names = [feature_names[i] for i in top_idx]
    return selected_names, X[:, top_idx], top_idx
