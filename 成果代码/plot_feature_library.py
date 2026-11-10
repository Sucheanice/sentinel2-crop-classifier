# -*- coding: utf-8 -*-
"""特征库可视化：月度 NDVI 物候曲线（按实际有效像元数断线，不跨缺口连线）

两张图：
  1. 水果库：柑橘 / 蜜柚 / 桃（附常绿判据线 NDVI=0.40）
  2. 作物库：水稻 / 小麦 / 油菜
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = r"e:\工作相关\2026年\0624 待测试数据\成果数据"

FRUIT = {
    "base": os.path.join(ROOT, "柑橘蜜柚识别", "自净"),
    "out": os.path.join(ROOT, "柑橘蜜柚识别", "指纹库", "特征库_月度NDVI曲线.png"),
    "title": "水果特征库 · 2025 年月度 NDVI 物候曲线（阴影为 P25–P75）",
    "sets": [("柑橘", "来源产区_citrus_2025_自净.csv", "#E8833A"),
             ("蜜柚", "来源产区_pomelo_2025_自净.csv", "#2E8B57"),
             ("桃", "来源产区_peach_2025_自净.csv", "#C0392B")],
    "winter_text": "冬季（落叶果树无叶期）",
    "ref": ("常绿判据 NDVI=0.40（柑橘/蜜柚需全年 ≥0.40）", 0.40, 0.415),
}

CROP = {
    "base": os.path.join(ROOT, "作物特征库", "自净"),
    "out": os.path.join(ROOT, "作物特征库", "指纹库", "特征库_月度NDVI曲线.png"),
    "title": "作物特征库 · 2025 年月度 NDVI 物候曲线（阴影为 P25–P75）",
    "sets": [("水稻", "来源产区_rice_2025_自净.csv", "#1F6FB2"),
             ("玉米", "来源产区_corn_2025_自净.csv", "#D35400"),
             ("小麦", "来源产区_wheat_2025_自净.csv", "#C9A227"),
             ("油菜", "来源产区_rape_2025_自净.csv", "#8E44AD")],
    "winter_text": "冬季（小麦/油菜越冬期）",
    "ref": None,
}

MIN_N = 200  # 有效像元少于此值的月份不画，视作该时相无有效观测


def monthly(path):
    cols = ["NDVI_%02d%d" % (m, k) for m in range(1, 13) for k in (1, 2, 3)]
    df = pd.read_csv(path, usecols=lambda c: c in cols)
    nrow = df.shape[0]
    med = np.full(12, np.nan)
    p25 = np.full(12, np.nan)
    p75 = np.full(12, np.nan)
    cnt = np.zeros(12, int)
    for m in range(12):
        c = cols[m * 3:m * 3 + 3]
        v = df[c].values.ravel()
        v = v[np.isfinite(v) & (v > -1)]
        cnt[m] = v.size
        if v.size >= MIN_N:
            med[m] = np.median(v)
            p25[m], p75[m] = np.percentile(v, [25, 75])
    return med, p25, p75, cnt, nrow


def draw(cfg):
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(11, 5.6), dpi=150)

    ax.axvspan(0.5, 2.5, color="#DCE6F1", alpha=0.55, zorder=0)
    ax.text(1.5, 0.045, cfg["winter_text"], ha="center", fontsize=9,
            color="#6C82A6", zorder=1)

    notes = []
    for name, fn, color in cfg["sets"]:
        med, p25, p75, cnt, nrow = monthly(os.path.join(cfg["base"], fn))
        x = np.arange(1, 13, dtype=float)
        ax.fill_between(x, p25, p75, color=color, alpha=0.16, linewidth=0)
        ax.plot(x, med, "-o", color=color, lw=2.0, ms=5,
                label="%s（%s 点）" % (name, format(nrow, ",")))
        miss = [f"{m + 1}月" for m in range(12) if np.isnan(med[m])]
        if miss:
            notes.append("%s无有效观测：%s" % (name, "、".join(miss)))
            ax.plot([m + 1 for m in range(12) if np.isnan(med[m])], [0.02] * len(miss),
                    "x", color=color, ms=7, mew=2)

    if cfg["ref"]:
        txt, val, ty = cfg["ref"]
        ax.axhline(val, ls="--", lw=1.3, color="#7F8C8D")
        ax.text(0.6, ty, txt, fontsize=9, color="#5D6D7E")

    ax.set_xlim(0.5, 12.5)
    ax.set_ylim(0, 1.0)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["%d月" % m for m in range(1, 13)])
    ax.set_ylabel("NDVI（月度中位）")
    ax.set_title(cfg["title"], fontsize=13, pad=12)
    ax.grid(alpha=0.25, ls=":")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="lower right", frameon=False, fontsize=10)

    if notes:
        ax.text(0.5, -0.13, "× 标记为无有效观测月份（云缺失/样本采集时相）　" + "；".join(notes),
                transform=ax.transAxes, ha="center", fontsize=8.5, color="#7F8C8D")
    fig.tight_layout()
    fig.savefig(cfg["out"], bbox_inches="tight")
    print("saved:", cfg["out"])


if __name__ == "__main__":
    draw(FRUIT)
    draw(CROP)
