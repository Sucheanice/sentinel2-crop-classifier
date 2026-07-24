# -*- coding: utf-8 -*-
import pandas as pd, numpy as np
import pickle, os, time, re, math
import rasterio, geopandas as gpd
from rasterio.features import rasterize
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from pyproj import Transformer

SHP_PATH = r"E:\工作相关\2026年\0624 待测试数据\待训练数据\矢量\人保-仅作物面"
S2_CROPPED_DIR = r"E:\工作相关\2026年\0624 待测试数据\待训练数据4_cropped"
OUT_DIR = r"E:\工作相关\2026年\0624 待测试数据\待训练数据"
FEATURES_CSV = os.path.join(OUT_DIR, "features_anju_cropped.csv")
MODEL_OUT = os.path.join(OUT_DIR, "crop_model_cropped.pkl")
CROP_MAP = os.path.join(OUT_DIR, "crop_map_cropped.tif")
MIN_SAMPLES = 10
N_FOLDS = 3

def compute_indices(blue, green, red, nir):
    eps = 1e-10
    evi = np.where(np.abs(nir + 6.0*red - 7.5*blue + 1.0) > eps,
                   2.5*(nir - red)/(nir + 6.0*red - 7.5*blue + 1.0), 0)
    ndwi = np.where(np.abs(green + nir) > eps, (green - nir)/(green + nir), 0)
    savi = np.where(np.abs(nir + red + 0.5) > eps, 1.5*(nir - red)/(nir + red + 0.5), 0)
    return evi, ndwi, savi

def clean_labels(df):
    col = 'bz'
    vals = df[col].values.copy()
    cleaned = []
    for v in vals:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            cleaned.append(v)
            continue
        s = str(v).strip()
        s_norm = s.replace('（', '(').replace('）', ')').replace('，', ',').replace(' ', '')
        s_norm = re.sub(r'[\\(（][^)）]*未种[^)）]*[\\)）]', '(未种)', s_norm)
        s_norm = re.sub(r',未种.*$', '(未种)', s_norm)
        s_norm = re.sub(r',未种植.*$', '(未种)', s_norm)
        if '(未种)' in s_norm and not s_norm.endswith('(未种)'):
            s_norm = s_norm.replace('(未种)', '')
        cleaned.append(s_norm.strip())
    df['crop_clean'] = cleaned
    return df

def get_scene_info():
    dirs = sorted([d for d in os.listdir(S2_CROPPED_DIR)
                   if os.path.isdir(os.path.join(S2_CROPPED_DIR, d)) and d[0].isdigit()])
    labels = []
    for d in dirs:
        parts = d.split("_")
        label = parts[1] if len(parts) >= 2 else d
        labels.append(label)
    return dirs, labels

def step1_extract():
    print("=" * 60)
    print("STEP 1: Pixel extraction")
    print("=" * 60)

    shp_files = [f for f in os.listdir(SHP_PATH) if f.endswith('.shp')]
    gdf = gpd.read_file(os.path.join(SHP_PATH, shp_files[0]))
    print("  SHP: %d parcels" % len(gdf))

    scene_dirs, scene_labels = get_scene_info()
    n_scenes = len(scene_dirs)
    print("  Scenes: %d" % n_scenes)

    with rasterio.open(os.path.join(S2_CROPPED_DIR, scene_dirs[0], "B02.tif")) as ref:
        s2_crs = ref.crs

    if gdf.crs != s2_crs:
        gdf = gdf.to_crs(s2_crs)

    rows = {}
    for si in range(n_scenes):
        dp = os.path.join(S2_CROPPED_DIR, scene_dirs[si])
        label = scene_labels[si]
        band_data = {}
        for bn in ['B02', 'B03', 'B04', 'B08']:
            band_data[bn] = rasterio.open(os.path.join(dp, bn + '.tif'))

        hit_count = 0
        for idx, row in gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            try:
                mask = rasterize([(geom, 1)], out_shape=(band_data['B02'].height, band_data['B02'].width),
                                 transform=band_data['B02'].transform, fill=0, dtype='uint8')
                px_count = np.count_nonzero(mask)
                if px_count < 3:
                    continue
                hit_count += 1
                for bn in ['B02', 'B03', 'B04', 'B08']:
                    vals = band_data[bn].read(1)[mask == 1].astype(np.float32)
                    col_name = '%s_%s' % (label, bn)
                    rows.setdefault(idx, {'parcel_id': idx, 'bz': row.get('bz', ''),
                                          'area_m2': row.get('scmj', 0)})
                    rows[idx][col_name] = vals.mean()
            except Exception as e:
                if si == 0:
                    print("  WARN parcel %d: %s" % (idx, e))
                continue

        for ds in band_data.values():
            ds.close()
        print("  Scene %d/%d done, %d hits" % (si + 1, n_scenes, hit_count))

    rows_list = list(rows.values())
    df = pd.DataFrame(rows_list)
    df.to_csv(FEATURES_CSV, index=False, encoding='utf-8-sig')
    print("  Saved: %s (%d records)" % (FEATURES_CSV, len(df)))

def step2_train():
    print("\n" + "=" * 60)
    print("STEP 2: Training")
    print("=" * 60)

    df = pd.read_csv(FEATURES_CSV, encoding='utf-8-sig')
    df = clean_labels(df)
    print("  Labels after cleaning: %d unique" % df['crop_clean'].nunique())

    scene_dirs, scene_labels = get_scene_info()
    n_scenes = len(scene_labels)

    all_features = []
    for si in range(n_scenes):
        label = scene_labels[si]
        for bn in ['B02', 'B03', 'B04', 'B08']:
            col = '%s_%s' % (label, bn)
            if col not in df.columns:
                continue
            all_features.append(col)

    if len(all_features) == 0:
        print("  ERROR: No feature columns found. Check scene_labels match CSV columns.")
        return

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

    print("  Records: %d, Classes: %d, Features: %d" % (len(df_clean), len(valid_classes), X.shape[1]))
    for i, name in enumerate(le.classes_):
        print("    [%d] %s: %d" % (i, name, (y == i).sum()))

    model = XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.5, reg_lambda=1.0,
        min_child_weight=3, gamma=0.01,
        objective='multi:softmax',
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
            cv_scores = [0.0]
            cv_f1 = [0.0]
            print("  CV failed (too few samples), skipping")
    else:
        print("  Too few classes/samples for CV, training on all data")
        cv_scores = [0.0]
        cv_f1 = [0.0]

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
    with open(MODEL_OUT, 'wb') as f:
        pickle.dump(bundle, f)

    print("\n  Model saved: %s" % MODEL_OUT)
    print("  CV Accuracy: %.4f +/- %.4f" % (cv_scores.mean(), cv_scores.std()))
    print("  CV F1-w:     %.4f +/- %.4f" % (cv_f1.mean(), cv_f1.std()))

def step3_predict():
    print("\n" + "=" * 60)
    print("STEP 3: Prediction")
    print("=" * 60)

    with open(MODEL_OUT, 'rb') as f:
        bundle = pickle.load(f)
    model = bundle['model']
    selected_features = bundle['selected_features']
    all_feature_names = bundle['all_feature_names']
    scene_dirs = bundle['scene_dirs']
    n_scenes = len(scene_dirs)

    feat_idx_map = [all_feature_names.index(f) for f in selected_features
                    if f in all_feature_names]

    with rasterio.open(os.path.join(S2_CROPPED_DIR, scene_dirs[0], 'B02.tif')) as ref:
        width, height = ref.width, ref.height
        profile = ref.profile.copy()
    profile.update(dtype='int16', count=1, compress='lzw', nodata=-1)
    print("  %d x %d, %d scenes, %d features" % (width, height, n_scenes, len(feat_idx_map)))

    readers = []
    for si in range(n_scenes):
        dp = os.path.join(S2_CROPPED_DIR, scene_dirs[si])
        scene_readers = []
        for band in ['B02', 'B03', 'B04', 'B08']:
            scene_readers.append(rasterio.open(os.path.join(dp, band + '.tif')))
        readers.append(scene_readers)

    t0 = time.time()
    with rasterio.open(CROP_MAP, 'w', **profile) as dst:
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
    print("  Saved: %s" % CROP_MAP)

def main():
    t0 = time.time()
    step1_extract()
    step2_train()
    step3_predict()
    print("\nTotal: %.1f min" % ((time.time() - t0) / 60))

if __name__ == "__main__":
    main()
