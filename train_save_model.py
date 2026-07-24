# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import pickle, re, math
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

FEATURES_CSV = r"E:\工作相关\2026年\0624 待测试数据\待训练数据\features_anju_correct.csv"
MODEL_OUT = r"E:\工作相关\2026年\0624 待测试数据\待训练数据\crop_model.pkl"
MIN_SAMPLES = 30

def clean_labels(df):
    col = 'crop_type'
    vals = df[col].values.copy()
    cleaned = []
    for v in vals:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            cleaned.append(v)
            continue
        s = str(v).strip()
        s_norm = s.replace('（', '(').replace('）', ')').replace('，', ',').replace(' ', '')
        s_norm = re.sub(r'[\(（][^)）]*未种[^)）]*[\)）]', '(未种)', s_norm)
        s_norm = re.sub(r',未种.*$', '(未种)', s_norm)
        s_norm = re.sub(r',未种植.*$', '(未种)', s_norm)
        if '(未种)' in s_norm and not s_norm.endswith('(未种)'):
            s_norm = s_norm.replace('(未种)', '')
        cleaned.append(s_norm.strip())
    df['crop_clean'] = cleaned
    return df

def get_scene_labels():
    import os
    base = r"E:\工作相关\2026年\0624 待测试数据\待训练数据4"
    dirs = sorted([d for d in os.listdir(base)
                   if os.path.isdir(os.path.join(base, d)) and d[0].isdigit()])
    labels = []
    for d in dirs:
        parts = d.split("_")
        label = parts[1] if len(parts) >= 2 else d
        labels.append(label)
    return dirs, labels

def compute_indices(blue, green, red, nir):
    denom_evi = nir + 6.0 * red - 7.5 * blue + 1.0
    evi = np.where(np.abs(denom_evi) > 1e-10, 2.5 * (nir - red) / denom_evi, 0)
    denom_ndwi = green + nir
    ndwi = np.where(np.abs(denom_ndwi) > 1e-10, (green - nir) / denom_ndwi, 0)
    denom_savi = nir + red + 0.5
    savi = np.where(np.abs(denom_savi) > 1e-10, 1.5 * (nir - red) / denom_savi, 0)
    return evi, ndwi, savi

def main():
    df = pd.read_csv(FEATURES_CSV, encoding='utf-8-sig')
    df = clean_labels(df)

    scene_dirs, scene_labels = get_scene_labels()
    n_scenes = len(scene_labels)

    all_features = []
    for si in range(n_scenes):
        label = scene_labels[si]
        blue  = df['%s_B02' % label].values
        green = df['%s_B03' % label].values
        red   = df['%s_B04' % label].values
        nir   = df['%s_B08' % label].values

        denom = nir + red
        ndvi = np.where(denom > 0, (nir - red) / denom, np.nan)

        evi, ndwi, savi = compute_indices(blue, green, red, nir)

        df['NDVI_%s' % label] = ndvi
        df['EVI_%s' % label] = evi
        df['NDWI_%s' % label] = ndwi
        df['SAVI_%s' % label] = savi
        all_features += ['%s_B02' % label, '%s_B03' % label, '%s_B04' % label, '%s_B08' % label,
                         'NDVI_%s' % label, 'EVI_%s' % label, 'NDWI_%s' % label, 'SAVI_%s' % label]

    df_clean = df.dropna(subset=all_features + ['crop_clean']).copy()
    df_clean = df_clean[df_clean['crop_clean'].notna() &
                        (df_clean['crop_clean'] != '') &
                        (df_clean['crop_clean'] != 'nan')]
    vc = df_clean["crop_clean"].value_counts()
    valid_classes = vc[vc >= MIN_SAMPLES].index.tolist()
    df_clean = df_clean[df_clean["crop_clean"].isin(valid_classes)]

    le = LabelEncoder()
    y = le.fit_transform(df_clean["crop_clean"])
    X = df_clean[all_features].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0)

    selector = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    selector.fit(X, y)
    feat_imp = selector.feature_importances_
    top_idx = np.argsort(feat_imp)[::-1][:30]
    selected_features = [all_features[i] for i in top_idx]
    X = X[:, top_idx]

    model = XGBClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.03,
        subsample=0.7, colsample_bytree=0.6,
        reg_alpha=1.0, reg_lambda=2.0,
        min_child_weight=5, gamma=0.1,
        objective='multi:softmax',
        random_state=42, n_jobs=-1, verbosity=0
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
    cv_f1 = cross_val_score(model, X, y, cv=skf, scoring='f1_weighted')

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

    print("Model saved: %s" % MODEL_OUT)
    print("CV Accuracy: %.4f +/- %.4f" % (cv_scores.mean(), cv_scores.std()))
    print("CV F1-w:     %.4f +/- %.4f" % (cv_f1.mean(), cv_f1.std()))
    print("Classes: %d" % len(le.classes_))
    for i, name in enumerate(le.classes_):
        print("  %d: %s (%d)" % (i, name, (y == i).sum()))
    print("Features: %d selected / %d total" % (len(selected_features), len(all_features)))

if __name__ == "__main__":
    main()
