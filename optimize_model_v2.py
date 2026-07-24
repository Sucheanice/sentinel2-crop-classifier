# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

FEATURES_CSV = r"E:\工作相关\2026年\0624 待测试数据\待训练数据\features_anju_correct.csv"
MIN_SAMPLES = 30
N_FOLDS = 5

def clean_labels_strict(df):
    col = 'crop_type'
    vals = df[col].values.copy()

    import re, math
    cleaned = []
    for v in vals:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            cleaned.append(v)
            continue
        s = str(v).strip()
        s_norm = s.replace('（', '(').replace('）', ')').replace('，', ',').replace(' ','')
        s_norm = re.sub(r'[\(（][^)）]*未种[^)）]*[\)）]', '(未种)', s_norm)
        s_norm = re.sub(r',未种.*$', '(未种)', s_norm)
        s_norm = re.sub(r',未种植.*$', '(未种)', s_norm)
        if '(未种)' in s_norm and not s_norm.endswith('(未种)'):
            s_norm = s_norm.replace('(未种)', '')
        cleaned.append(s_norm.strip())
    df['crop_clean'] = cleaned

    print("  Mapping changes:")
    changes = df[df['crop_type'] != df['crop_clean']][['crop_type', 'crop_clean']].drop_duplicates()
    for _, row in changes.iterrows():
        c_before = df[df['crop_type'] == row['crop_type']].shape[0]
        print("    [%s] (%d) -> [%s]" % (row['crop_type'], c_before, row['crop_clean']))
    return df

def add_veg_indices(df, feature_cols):
    scene_labels = set()
    for c in feature_cols:
        for suffix in ['_B02', '_B03', '_B04', '_B08']:
            if c.endswith(suffix):
                scene_labels.add(c[:-len(suffix)])
    scene_labels = sorted(scene_labels)

    new_features = []
    for label in scene_labels:
        blue  = df[label + '_B02'].values
        green = df[label + '_B03'].values
        red   = df[label + '_B04'].values
        nir   = df[label + '_B08'].values

        denom_evi = nir + 6.0*red - 7.5*blue + 1.0
        evi = np.where(np.abs(denom_evi) > 1e-10, 2.5 * (nir - red) / denom_evi, 0)

        denom_ndwi = green + nir
        ndwi = np.where(np.abs(denom_ndwi) > 1e-10, (green - nir) / denom_ndwi, 0)

        denom_savi = nir + red + 0.5
        savi = np.where(np.abs(denom_savi) > 1e-10, 1.5 * (nir - red) / denom_savi, 0)

        df['EVI_' + label] = evi
        df['NDWI_' + label] = ndwi
        df['SAVI_' + label] = savi
        new_features.extend(['EVI_' + label, 'NDWI_' + label, 'SAVI_' + label])
    return df, new_features

def main():
    print("=" * 60)
    print("STEP 1: Label Cleaning")
    print("=" * 60)
    df = pd.read_csv(FEATURES_CSV, encoding='utf-8-sig')
    df = clean_labels_strict(df)

    print("\n  Before: %d classes" % df['crop_type'].nunique())
    print("  After:  %d classes" % df['crop_clean'].nunique())

    feature_cols = [c for c in df.columns
                    if any(c.endswith(b) for b in ['_B02', '_B03', '_B04', '_B08'])
                    or c.startswith('NDVI_')]

    print("\n" + "=" * 60)
    print("STEP 2: Vegetation Indices + Feature Selection")
    print("=" * 60)
    df, extra_cols = add_veg_indices(df, feature_cols)
    feature_cols = feature_cols + extra_cols
    print("  Total features: %d" % len(feature_cols))

    df_clean = df.dropna(subset=feature_cols + ['crop_clean']).copy()
    df_clean = df_clean[df_clean['crop_clean'].notna() & (df_clean['crop_clean'] != '') & (df_clean['crop_clean'] != 'nan')]
    vc = df_clean["crop_clean"].value_counts()
    valid_classes = vc[vc >= MIN_SAMPLES].index.tolist()
    df_clean = df_clean[df_clean["crop_clean"].isin(valid_classes)]

    le = LabelEncoder()
    y = le.fit_transform(df_clean["crop_clean"])
    X = df_clean[feature_cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0)

    print("  Records: %d, Classes: %d, Features: %d" % (len(df_clean), len(valid_classes), X.shape[1]))

    selector = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    selector.fit(X, y)
    feat_imp = selector.feature_importances_
    keep_n = min(30, X.shape[1])
    top_feat_idx = np.argsort(feat_imp)[::-1][:keep_n]
    selected_cols = [feature_cols[i] for i in top_feat_idx]
    X = X[:, top_feat_idx]
    feature_cols = selected_cols
    print("  Feature selection: %d -> %d features" % (len(feat_imp), keep_n))
    for i, name in enumerate(le.classes_):
        print("    [%d] n=%-5d %s" % (i, (y == i).sum(), name))

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    print("\n" + "=" * 60)
    print("STEP 3: Models comparison")
    print("=" * 60)

    print("\n--- RF baseline ---")
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=12, min_samples_leaf=8,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    rf_scores = cross_val_score(rf, X, y, cv=skf, scoring='accuracy')
    rf_f1 = cross_val_score(rf, X, y, cv=skf, scoring='f1_weighted')
    print("  Accuracy: %.4f +/- %.4f" % (rf_scores.mean(), rf_scores.std()))
    print("  F1-w:     %.4f +/- %.4f" % (rf_f1.mean(), rf_f1.std()))

    rf.fit(X, y)
    rf_train = accuracy_score(y, rf.predict(X))
    print("  Train acc: %.4f" % rf_train)

    print("\n--- XGBoost v1 (conservative) ---")
    xgb1 = XGBClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.03,
        subsample=0.7, colsample_bytree=0.6,
        reg_alpha=1.0, reg_lambda=2.0,
        min_child_weight=5,
        gamma=0.1,
        objective='multi:softmax',
        random_state=42, n_jobs=-1,
        verbosity=0
    )
    xgb1_scores = cross_val_score(xgb1, X, y, cv=skf, scoring='accuracy')
    xgb1_f1 = cross_val_score(xgb1, X, y, cv=skf, scoring='f1_weighted')
    print("  Accuracy: %.4f +/- %.4f" % (xgb1_scores.mean(), xgb1_scores.std()))
    print("  F1-w:     %.4f +/- %.4f" % (xgb1_f1.mean(), xgb1_f1.std()))

    xgb1.fit(X, y)
    xgb1_train = accuracy_score(y, xgb1.predict(X))
    print("  Train acc: %.4f (gap: %.1f%%)" % (xgb1_train, (xgb1_train - xgb1_scores.mean())*100))

    print("\n--- XGBoost v2 (aggressive) ---")
    xgb2 = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7,
        reg_alpha=0.5, reg_lambda=1.5,
        min_child_weight=3,
        objective='multi:softmax',
        random_state=42, n_jobs=-1,
        verbosity=0
    )
    xgb2_scores = cross_val_score(xgb2, X, y, cv=skf, scoring='accuracy')
    xgb2_f1 = cross_val_score(xgb2, X, y, cv=skf, scoring='f1_weighted')
    print("  Accuracy: %.4f +/- %.4f" % (xgb2_scores.mean(), xgb2_scores.std()))
    print("  F1-w:     %.4f +/- %.4f" % (xgb2_f1.mean(), xgb2_f1.std()))

    xgb2.fit(X, y)
    xgb2_train = accuracy_score(y, xgb2.predict(X))
    print("  Train acc: %.4f (gap: %.1f%%)" % (xgb2_train, (xgb2_train - xgb2_scores.mean())*100))

    print("\n" + "=" * 60)
    print("STEP 4: Best model detailed report (XGBoost v1)")
    print("=" * 60)

    best = xgb1 if xgb1_scores.mean() > xgb2_scores.mean() else xgb2
    best_name = "v1" if xgb1_scores.mean() > xgb2_scores.mean() else "v2"

    y_pred = best.predict(X)
    print("\n--- Classification Report ---")
    rpt = classification_report(y, y_pred, target_names=le.classes_, zero_division=0)
    print(rpt)

    print("--- Per-class accuracy ---")
    cm = confusion_matrix(y, y_pred)
    for i, name in enumerate(le.classes_):
        row = cm[i]
        class_acc = row[i] / row.sum() if row.sum() > 0 else 0
        print("  %s: %.3f (n=%d)" % (name, class_acc, row.sum()))

    importances = best.feature_importances_
    top_idx = np.argsort(importances)[::-1][:15]
    print("\n--- Top 15 features ---")
    for idx in top_idx:
        print("  %-25s %.4f" % (feature_cols[idx], importances[idx]))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("  Configuration: %d classes, %d features, %d samples" % (len(valid_classes), len(feature_cols), len(df_clean)))
    print("  RF        CV: %.4f (train: %.4f, gap: %.1f%%)" % (rf_scores.mean(), rf_train, (rf_train - rf_scores.mean())*100))
    print("  XGBoost %s CV: %.4f (train: %.4f, gap: %.1f%%)" % (best_name, best.cv_avg if hasattr(best, 'cv_avg') else xgb1_scores.mean() if best_name=='v1' else xgb2_scores.mean(), xgb1_train if best_name=='v1' else xgb2_train, (xgb1_train - xgb1_scores.mean())*100 if best_name=='v1' else (xgb2_train - xgb2_scores.mean())*100))

if __name__ == "__main__":
    main()
