# -*- coding: utf-8 -*-
"""对比 10天 vs 20天 窗口训练样本缺值率，评估重训样本保留率变化"""
import pandas as pd
import os

BASE = r"e:\工作相关\2026年\0624 待测试数据"
CSV10 = os.path.join(BASE, "gee_dachun_23counties_3dekad.csv")
CSV20 = os.path.join(BASE, "gee_dachun_23counties_3dekad_wide.csv")


def stats(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    d04 = df['D04_B8'].isna()
    d11 = df['D11_B8'].isna()
    d14 = df['D14_B8'].isna()
    return {
        '样本数': len(df),
        'D04缺': d04.sum(),
        'D11缺': d11.sum(),
        'D14缺': d14.sum(),
        '任一缺(clean丢弃)': (d04 | d11 | d14).sum(),
        '仅D04缺': (d04 & ~d11 & ~d14).sum(),
        '仅D11缺': (d11 & ~d04 & ~d14).sum(),
        '仅D14缺': (d14 & ~d04 & ~d11).sum(),
        'D04或D14缺(两旬救不了)': (d04 | d14).sum(),
    }


s10 = stats(CSV10)
s20 = stats(CSV20)
n10, n20 = s10['样本数'], s20['样本数']

print(f"{'指标':<28}{'10天':>10}{'20天':>10}")
print("-" * 50)
for k in s10:
    v10 = f"{s10[k]} ({s10[k]/n10:.1%})" if '样本' not in k else s10[k]
    v20 = f"{s20[k]} ({s20[k]/n20:.1%})" if '样本' not in k else s20[k]
    print(f"{k:<28}{v10:>14}{v20:>14}")

print("\n=== 关键结论 ===")
print(f"clean 后保留样本: 10天={n10 - s10['任一缺(clean丢弃)']}  "
      f"20天={n20 - s20['任一缺(clean丢弃)']}  "
      f"(保留率 {(n20 - s20['任一缺(clean丢弃)'])/n20:.1%})")
