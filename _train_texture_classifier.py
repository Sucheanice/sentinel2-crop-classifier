# -*- coding: utf-8 -*-
"""纹理特征验证：柑橘 vs 林地（物候盲区）"""
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

BASE = r"e:\工作相关\2026年\0624 待测试数据"
TEX = os.path.join(BASE, "xuyong_texture_aug.csv")
TEX_BANDS = ['asm', 'contrast', 'corr', 'var', 'idm', 'savg', 'ent', 'diss']

df = pd.read_csv(TEX, encoding="utf-8-sig")
b8_tex = [f'B8_{t}' for t in TEX_BANDS]
gray_tex = [f'Gray_{t}' for t in TEX_BANDS]

sub = df[df['category'].isin(['柑橘', '林地'])].copy()
sub = sub.dropna(subset=['NDVI'] + b8_tex + gray_tex)
y = (sub['category'] == '柑橘').astype(int).values
print(f"柑橘={int(y.sum())} 林地={int((y == 0).sum())}\n")


def cv_eval(X, y, name, cols=None):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, f1s, aucs = [], [], []
    imp = np.zeros(X.shape[1])
    for tr, te in skf.split(X, y):
        m = lgb.LGBMClassifier(n_estimators=100, max_depth=3, num_leaves=15,
                               learning_rate=0.05, min_child_samples=10,
                               random_state=42, verbose=-1)
        m.fit(X[tr], y[tr])
        p = m.predict(X[te])
        accs.append(accuracy_score(y[te], p))
        f1s.append(f1_score(y[te], p))
        aucs.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
        imp += m.feature_importances_
    print(f"  {name:<22} Acc={np.mean(accs):.3f}  F1={np.mean(f1s):.3f}  AUC={np.mean(aucs):.3f}")
    return imp / 5


cv_eval(sub[['NDVI']].values, y, '纯 NDVI（物候基准）')
cv_eval(sub[b8_tex].values, y, 'B8 纹理(8)')
cv_eval(sub[gray_tex].values, y, 'Gray 纹理(8)')
cv_eval(sub[b8_tex + gray_tex].values, y, 'B8+Gray 纹理(16)')
imp = cv_eval(sub[b8_tex + gray_tex + ['NDVI']].values, y, '纹理(16)+NDVI',
              cols=b8_tex + gray_tex + ['NDVI'])

# 特征重要性
print("\n纹理特征重要性 Top8（纹理+NDVI 方案）:")
cols = b8_tex + gray_tex + ['NDVI']
order = np.argsort(-imp)
for i in order[:8]:
    print(f"  {cols[i]:<12} {imp[i]:.3f}")
