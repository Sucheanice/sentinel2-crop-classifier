# -*- coding: utf-8 -*-
import pandas as pd, numpy as np
import pickle, os, time, re, math, sys
import rasterio, geopandas as gpd
from rasterio.features import rasterize
from rasterio.windows import from_bounds
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from pyproj import Transformer

AREA = "shehong"

PROJ_DIR = r"E:\工作相关\2026年\0624 待测试数据"
SHP_BASE = os.path.join(PROJ_DIR, r"待训练数据5")
S2_DIR = os.path.join(PROJ_DIR, r"待训练数据6")
BUFFER = 100
BANDS = ["B02", "B03", "B04", "B08"]
MIN_SAMPLES = 10
N_FOLDS = 3

SHEHONG_TARGET_CLASSES = ["玉米", "水稻", "高粱"]

def find_shp(area):
    for folder in os.listdir(SHP_BASE):
        full = os.path.join(SHP_BASE, folder)
        if not os.path.isdir(full):
            continue
        if (area == "anju" and "安居" in folder) or (area == "shehong" and "射洪" in folder):
            for f in os.listdir(full):
                if f.endswith('.shp'):
                    return os.path.join(full, f)
    return None

def compute_indices(blue, green, red, nir):
    eps = 1e-10
    evi = np.where(np.abs(nir + 6.0*red - 7.5*blue + 1.0) > eps,
                   2.5*(nir - red)/(nir + 6.0*red - 7.5*blue + 1.0), 0)
    ndwi = np.where(np.abs(green + nir) > eps, (green - nir)/(green + nir), 0)
    savi = np.where(np.abs(nir + red + 0.5) > eps, 1.5*(nir - red)/(nir + red + 0.5), 0)
    return evi, ndwi, savi

def clean_labels(df, area):
    col = "ZWMC"
    if col not in df.columns:
        print("  WARN: ZWMC column not found, trying BZ")
        col = "BZ"
    vals = df[col].astype(str).str.strip()
    df["crop_clean"] = vals
    if area == "shehong":
        df["crop_clean"] = df["crop_clean"].apply(
            lambda x: x if x in SHEHONG_TARGET_CLASSES else "其他"
        )
    return df

def get_scene_info(s2_cropped_dir):
    dirs = sorted([d for d in os.listdir(s2_cropped_dir)
                   if os.path.isdir(os.path.join(s2_cropped_dir, d)) and d[0].isdigit()])
    labels = []
    for d in dirs:
        parts = d.split("_")
        label = parts[0] if len(parts) >= 1 else d
        labels.append(label)
    return dirs, labels

def step0_crop(area, shp_path):
    print("=" * 60)
    print("STEP 0: Crop S2 to SHP extent")
    print("=" * 60)

    cropped_dir = os.path.join(PROJ_DIR, f"待训练数据6_{area}_cropped")
    if os.path.exists(cropped_dir):
        existing = [d for d in os.listdir(cropped_dir)
                    if os.path.isdir(os.path.join(cropped_dir, d))]
        if len(existing) >= 4:
            print("  Already cropped (%d scene dirs), skip" % len(existing))
            return cropped_dir

    gdf = gpd.read_file(shp_path)
    s2_dirs = sorted([d for d in os.listdir(S2_DIR)
                      if os.path.isdir(os.path.join(S2_DIR, d)) and d[0].isdigit()])
    with rasterio.open(os.path.join(S2_DIR, s2_dirs[0], "B02.tif")) as ref:
        s2_crs = ref.crs

    if gdf.crs != s2_crs:
        gdf = gdf.to_crs(s2_crs)

    bounds = gdf.total_bounds
    crop_bounds = (bounds[0] - BUFFER, bounds[1] - BUFFER,
                   bounds[2] + BUFFER, bounds[3] + BUFFER)
    print("  SHP bounds + %dm buffer: [%.0f, %.0f, %.0f, %.0f]" %
          (BUFFER, crop_bounds[0], crop_bounds[1], crop_bounds[2], crop_bounds[3]))
    print("  %d S2 scenes to crop" % len(s2_dirs))

    t0 = time.time()
    for di, dname in enumerate(s2_dirs):
        scene_out = os.path.join(cropped_dir, dname)
        os.makedirs(scene_out, exist_ok=True)

        for band in BANDS:
            src_path = os.path.join(S2_DIR, dname, band + ".tif")
            dst_path = os.path.join(scene_out, band + ".tif")
            if os.path.exists(dst_path):
                continue
            with rasterio.open(src_path) as src:
                window = from_bounds(*crop_bounds, transform=src.transform)
                window = window.round_offsets().round_shape()
                data = src.read(window=window)
                transform = src.window_transform(window)
                profile = src.profile.copy()
                profile.update(height=data.shape[1], width=data.shape[2],
                               transform=transform, compress="lzw", tiled=True,
                               blockxsize=256, blockysize=256)
                with rasterio.open(dst_path, "w", **profile) as dst:
                    dst.write(data)
            print("  [%d/%d] %s/%s (%d x %d)" % (
                di + 1, len(s2_dirs), dname, band, data.shape[2], data.shape[1]))

    print("  Crop done in %.1fs" % (time.time() - t0))
    return cropped_dir

def step1_extract(area, shp_path, s2_cropped_dir):
    print("\n" + "=" * 60)
    print("STEP 1: Pixel extraction (%s)" % area)
    print("=" * 60)

    gdf = gpd.read_file(shp_path)
    print("  SHP: %d parcels" % len(gdf))

    scene_dirs, scene_labels = get_scene_info(s2_cropped_dir)
    n_scenes = len(scene_dirs)
    print("  Scenes: %d" % n_scenes)

    with rasterio.open(os.path.join(s2_cropped_dir, scene_dirs[0], "B02.tif")) as ref:
        s2_crs = ref.crs
        height = ref.height
        width = ref.width
        transform = ref.transform

    if gdf.crs != s2_crs:
        gdf = gdf.to_crs(s2_crs)

    valid_indices = []
    shapes = []
    for idx, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        parcel_id = len(shapes) + 1
        shapes.append((geom, parcel_id))
        valid_indices.append(idx)

    n_valid = len(shapes)
    print("  Valid parcels (non-empty geom): %d" % n_valid)

    label_img = rasterize(shapes, out_shape=(height, width), transform=transform,
                           fill=0, dtype='uint32')
    flat_labels = label_img.ravel()
    pixel_counts = np.bincount(flat_labels, minlength=n_valid + 1)
    valid_mask = pixel_counts >= 3
    valid_mask[0] = False

    base_info = {}
    for i, idx in enumerate(valid_indices):
        parcel_id = i + 1
        if valid_mask[parcel_id]:
            row = gdf.iloc[idx]
            base_info[parcel_id] = {
                'parcel_id': idx, 'ZWMC': row.get('ZWMC', ''), 'BZ': row.get('BZ', '')
            }

    rows = {}
    for si in range(n_scenes):
        dp = os.path.join(s2_cropped_dir, scene_dirs[si])
        label = scene_labels[si]
        band_data = {}
        for bn in BANDS:
            band_data[bn] = rasterio.open(os.path.join(dp, bn + '.tif'))

        for bn in BANDS:
            data = band_data[bn].read(1).ravel().astype(np.float32)
            sums = np.bincount(flat_labels, weights=data, minlength=n_valid + 1)
            counts = pixel_counts
            means = np.divide(sums, counts, out=np.full_like(sums, np.nan, dtype=np.float64),
                             where=(counts >= 3))
            col_name = '%s_%s' % (label, bn)
            for parcel_id, info in base_info.items():
                rows.setdefault(parcel_id, info)
                rows[parcel_id][col_name] = means[parcel_id] if not np.isnan(means[parcel_id]) else np.nan

        for ds in band_data.values():
            ds.close()
        hit_count = sum(1 for pid in base_info if not np.isnan(rows.get(pid, {}).get(col_name, np.nan)))
        print("  Scene %d/%d done, %d hits" % (si + 1, n_scenes, hit_count))

    rows_list = list(rows.values())
    df = pd.DataFrame(rows_list)
    out_dir = os.path.join(PROJ_DIR, "待训练数据6_%s" % area)
    os.makedirs(out_dir, exist_ok=True)
    features_csv = os.path.join(out_dir, "features_%s.csv" % area)
    df.to_csv(features_csv, index=False, encoding='utf-8-sig')
    print("  Saved: %s (%d records)" % (features_csv, len(df)))
    return features_csv, out_dir

def step2_train(area, features_csv, s2_cropped_dir, out_dir):
    print("\n" + "=" * 60)
    print("STEP 2: Training (%s)" % area)
    print("=" * 60)

    df = pd.read_csv(features_csv, encoding='utf-8-sig')
    df = clean_labels(df, area)
    print("  Labels after cleaning: %d unique" % df['crop_clean'].nunique())

    scene_dirs, scene_labels = get_scene_info(s2_cropped_dir)
    n_scenes = len(scene_labels)

    all_features = []
    for si in range(n_scenes):
        label = scene_labels[si]
        for bn in BANDS:
            col = '%s_%s' % (label, bn)
            if col not in df.columns:
                continue
            all_features.append(col)

    if len(all_features) == 0:
        print("  ERROR: No feature columns found")
        return None

    print("  Raw features: %d" % len(all_features))

    for si in range(n_scenes):
        label = scene_labels[si]
        blue = df.get('%s_B02' % label)
        green = df.get('%s_B03' % label)
        red = df.get('%s_B04' % label)
        nir = df.get('%s_B08' % label)
        if blue is None:
            continue
        blue, green, red, nir = blue.values, green.values, red.values, nir.values

        denom = nir + red
        ndvi = np.where(denom > 0, (nir - red) / denom, np.nan)
        evi, ndwi, savi = compute_indices(blue, green, red, nir)

        df['NDVI_%s' % label] = ndvi
        df['EVI_%s' % label] = evi
        df['NDWI_%s' % label] = ndwi
        df['SAVI_%s' % label] = savi
        all_features += ['NDVI_%s' % label, 'EVI_%s' % label, 'NDWI_%s' % label, 'SAVI_%s' % label]

    df_clean = df.dropna(subset=all_features + ['crop_clean']).copy()
    df_clean = df_clean[df_clean['crop_clean'].notna() &
                        (df_clean['crop_clean'] != '') &
                        (df_clean['crop_clean'] != 'nan')]
    vc = df_clean["crop_clean"].value_counts()
    valid_classes = vc[vc >= MIN_SAMPLES].index.tolist()
    df_clean = df_clean[df_clean["crop_clean"].isin(valid_classes)]
    n_classes = len(valid_classes)

    le = LabelEncoder()
    y = le.fit_transform(df_clean["crop_clean"])
    X = df_clean[all_features].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0)

    selector = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    selector.fit(X, y)
    feat_imp = selector.feature_importances_
    top_idx = np.argsort(feat_imp)[::-1][:min(30, X.shape[1])]
    selected_features = [all_features[i] for i in top_idx]
    X = X[:, top_idx]

    print("  Records: %d, Classes: %d, Features: %d" % (len(df_clean), n_classes, X.shape[1]))
    for i, name in enumerate(le.classes_):
        print("    [%d] %s: %d" % (i, name, (y == i).sum()))

    model = XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.5, reg_lambda=1.0,
        min_child_weight=3, gamma=0.01,
        objective='multi:softmax', num_class=n_classes,
        random_state=42, n_jobs=-1, verbosity=0
    )

    if n_classes >= 3 and len(df_clean) >= n_classes * 10:
        min_per_class = np.bincount(y[y>=0]).min()
        skf = StratifiedKFold(n_splits=min(N_FOLDS, int(min_per_class)), shuffle=True, random_state=42)
        try:
            cv_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
            cv_f1 = cross_val_score(model, X, y, cv=skf, scoring='f1_weighted')
            print("  CV Accuracy: %.4f +/- %.4f" % (cv_scores.mean(), cv_scores.std()))
            print("  CV F1-w:     %.4f +/- %.4f" % (cv_f1.mean(), cv_f1.std()))
        except Exception:
            cv_scores = np.array([0.0])
            cv_f1 = np.array([0.0])
            print("  CV failed (too few samples), skipping")
    else:
        print("  Too few classes/samples for CV, training on all data")
        cv_scores = np.array([0.0])
        cv_f1 = np.array([0.0])

    model.fit(X, y)

    bundle = {
        'model': model,
        'label_encoder': le,
        'selected_features': selected_features,
        'scene_labels': scene_labels,
        'scene_dirs': scene_dirs,
        'all_feature_names': all_features,
        'cv_accuracy': cv_scores.mean(),
        'cv_f1': cv_f1.mean(),
        'class_names': le.classes_.tolist(),
    }
    model_out = os.path.join(out_dir, "crop_model_%s.pkl" % area)
    with open(model_out, 'wb') as f:
        pickle.dump(bundle, f)

    print("\n  Model saved: %s" % model_out)
    print("  CV Accuracy: %.4f +/- %.4f" % (cv_scores.mean(), cv_scores.std()))
    print("  CV F1-w:     %.4f +/- %.4f" % (cv_f1.mean(), cv_f1.std()))
    return model_out

def step3_predict(area, model_path, out_dir, s2_cropped_dir):
    print("\n" + "=" * 60)
    print("STEP 3: Prediction (%s)" % area)
    print("=" * 60)

    with open(model_path, 'rb') as f:
        bundle = pickle.load(f)
    model = bundle['model']
    selected_features = bundle['selected_features']
    all_feature_names = bundle['all_feature_names']
    scene_dirs = bundle['scene_dirs']
    n_scenes = len(scene_dirs)

    feat_idx_map = [all_feature_names.index(f) for f in selected_features
                    if f in all_feature_names]

    with rasterio.open(os.path.join(s2_cropped_dir, scene_dirs[0], 'B02.tif')) as ref:
        width, height = ref.width, ref.height
        profile = ref.profile.copy()
    profile.update(dtype='int16', count=1, compress='lzw', nodata=-1)
    print("  %d x %d, %d scenes, %d features" % (width, height, n_scenes, len(feat_idx_map)))

    readers = []
    for si in range(n_scenes):
        dp = os.path.join(s2_cropped_dir, scene_dirs[si])
        scene_readers = []
        for band in BANDS:
            scene_readers.append(rasterio.open(os.path.join(dp, band + '.tif')))
        readers.append(scene_readers)

    t0 = time.time()
    crop_map = os.path.join(out_dir, "crop_map_%s.tif" % area)
    with rasterio.open(crop_map, 'w', **profile) as dst:
        feature_blocks = []
        for si in range(n_scenes):
            blue = readers[si][0].read(1).astype(np.float32)
            green = readers[si][1].read(1).astype(np.float32)
            red = readers[si][2].read(1).astype(np.float32)
            nir = readers[si][3].read(1).astype(np.float32)

            denom = nir + red
            ndvi = np.where(denom > 1e-10, (nir - red) / denom, 0)
            evi, ndwi, savi = compute_indices(blue, green, red, nir)
            feature_blocks.extend([blue, green, red, nir, ndvi, evi, ndwi, savi])

        X_block = np.stack(feature_blocks, axis=-1)
        X_flat = X_block.reshape(-1, 8 * n_scenes)
        X_flat = np.nan_to_num(X_flat, nan=0.0)
        X_selected = X_flat[:, feat_idx_map].astype(np.float32)

        y_pred = model.predict(X_selected).astype(np.int16)
        y_pred = y_pred.reshape(height, width)
        dst.write(y_pred, 1)

    for si in range(n_scenes):
        for r in readers[si]:
            r.close()

    elapsed = time.time() - t0
    print("  Done in %.1fs" % elapsed)
    print("  Saved: %s" % crop_map)
    return crop_map

def main():
    t0 = time.time()
    area = AREA.lower()
    if area not in ("anju", "shehong"):
        print("ERROR: AREA must be 'anju' or 'shehong'")
        return

    print("Pipeline v2 - Area: %s" % area)

    shp_path = find_shp(area)
    if shp_path is None:
        print("ERROR: SHP not found for area '%s'" % area)
        return
    print("SHP: %s" % shp_path)

    s2_cropped_dir = step0_crop(area, shp_path)
    features_csv, out_dir = step1_extract(area, shp_path, s2_cropped_dir)
    model_path = step2_train(area, features_csv, s2_cropped_dir, out_dir)
    if model_path:
        step3_predict(area, model_path, out_dir, s2_cropped_dir)

    total_min = (time.time() - t0) / 60
    print("\n" + "=" * 60)
    print("Pipeline v2 (%s) complete in %.1f min" % (area, total_min))
    print("Output dir: %s" % out_dir)
    print("=" * 60)

if __name__ == "__main__":
    main()
