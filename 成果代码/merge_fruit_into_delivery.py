# -*- coding: utf-8 -*-
"""merge_fruit_into_delivery.py — 把柑橘/蜜柚判定并入既有交付件（1 shp + 1 xlsx）

范围约束（对齐第一批次任务区）：
  * 蜜柚 —— 仅 梓潼 黎雅镇、金龙镇
  * 柑橘 —— 仅 平武 水晶镇/豆叩羌族乡/大桥镇/旧堡羌族乡、江油 河口镇/马角镇/新安镇/太平镇
  * 其余镇只判大春作物，不做水果

流程：在「大春作物赋值」已完成的交付件基础上，对**耕地地块**（AG_BZBM=A01）叠加水果判定：
  * 用 citrus_pomelo_extract.py --mode target 已提好的 540 波段特征打分
  * 打分口径 = **双库对照（同镇自校准）**：
      - 正库 = 来源产区（自净）水果指纹
      - 对照库 = **本镇内已被判为水稻/玉米的耕地块**（同镇同地理、同云况）
      - 逐块算到两库的归一化偏离 d_pos / d_neg，`score = d_neg / (d_pos + d_neg)`
        （>0.5 = 比本镇大春作物更像该水果）
      - 这样「冬稳定 / MNDWI」等**地理/云况维度在两库间自动抵消**，
        避免同县取样时全部地块都拿高分的伪结果（见 2026-10-15 记录）
  * 只把「高 / 中」嫌疑写入编码列（低嫌疑留空），另存分值便于排序复核

产出：原地更新 `{镇名}_交付.shp` 与 `{镇名}_交付.xlsx`，各追加 4 列：
  shp   : FRT_BZBM / FRT_BZMC / FRT_SCORE / FRT_LEVEL
  xlsx  : 水果标的编码 / 水果标的名称 / 水果分值 / 水果档位

阈值默认（用蜜柚库样本锚定：库样本自己 score 中位 ≈0.69~0.72、P10≈0.54~0.58）：
  `--hi 0.62`（≈库样本下四分位）/ `--mid 0.50`（≈等距分界）

用法：
    python merge_fruit_into_delivery.py --crop pomelo --town 黎雅镇 \
        --shp "…\\交付\\黎雅镇_交付.shp" --xlsx "…\\交付\\黎雅镇_交付.xlsx" \
        --feat "…\\目标地块_pomelo_黎雅镇_2025.csv" --lib-dir "…\\自净"
"""
import argparse
import os
import sys

import geopandas as gpd
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from citrus_pomelo_score import fingerprints, NAME  # noqa: E402


def block_fit_dist(lib_fp, tgt_fp, dims):
    """逐块到「库」的归一化偏离：|tgt − 库中位| / 库MAD，逐维算后取均值。越小 = 越像该库。"""
    Z = {}
    for c in dims:
        s = lib_fp[c].dropna()
        if len(s) < 20:
            continue
        m = s.median()
        mad = max((s - m).abs().median() * 1.4826, 1e-3)
        Z[c] = np.abs(pd.to_numeric(tgt_fp[c], errors="coerce").values - m) / mad
    return pd.DataFrame(Z, index=tgt_fp.index).mean(axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crop", choices=["citrus", "pomelo", "plum"], required=True)
    ap.add_argument("--town", required=True)
    ap.add_argument("--shp", required=True)
    ap.add_argument("--xlsx", required=True)
    ap.add_argument("--feat", required=True, help="目标地块特征 CSV")
    ap.add_argument("--lib-dir", required=True, help="自净库目录（含 来源产区_{crop}_2025_自净.csv）")
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--hi", type=float, default=0.62)
    ap.add_argument("--mid", type=float, default=0.50)
    ap.add_argument("--min-ref", type=int, default=50,
                    help="对照库（本镇水稻/玉米）最少块数，不足则报错")
    a = ap.parse_args()

    lib_p = os.path.join(a.lib_dir, f"来源产区_{a.crop}_{a.year}_自净.csv")
    if not os.path.exists(lib_p):
        raise SystemExit(f"[错误] 缺库: {lib_p}")
    pos_fp = fingerprints(pd.read_csv(lib_p), a.year)
    tgt = pd.read_csv(a.feat)
    tgt_fp = fingerprints(tgt, a.year)

    g = gpd.read_file(a.shp)
    a01 = np.flatnonzero(g["AG_BZBM"].astype(str).values == "A01")
    if len(a01) != len(tgt):
        raise SystemExit(f"[错误] 耕地块数对不上：交付件 {len(a01)} vs 特征 {len(tgt)}")

    # 对照库 = 本镇被判为水稻/玉米的耕地块（同镇同地理 → 地理/云况维度自动抵消）
    crop = g["CROP_BZBM"].astype(str).values[a01]
    is_ref = np.isin(crop, ["AA000000", "AA000700"])
    if int(is_ref.sum()) < a.min_ref:
        raise SystemExit(f"[错误] 对照库块数不足：{int(is_ref.sum())} < {a.min_ref}")

    dims = [c for c in pos_fp.columns if c in tgt_fp.columns]
    d_pos = block_fit_dist(pos_fp, tgt_fp, dims)
    d_neg = block_fit_dist(tgt_fp[is_ref], tgt_fp, dims)
    score = (d_neg / (d_pos + d_neg)).values          # >0.5 = 更像该水果
    lv = np.where(score >= a.hi, "高", np.where(score >= a.mid, "中", "低"))
    label, code = NAME[a.crop]
    print(f"[打分] {a.town} {len(tgt)} 块（对照库 {int(is_ref.sum())} 块）；分数中位 {np.nanmedian(score):.3f}；"
          f"高 {int((lv=='高').sum())} / 中 {int((lv=='中').sum())} / 低 {int((lv=='低').sum())}")

    # 与 _read_target 同样按 A01 子集顺序（reset_index）对齐
    g["FRT_SCORE"] = np.nan
    g["FRT_LEVEL"] = ""
    g["FRT_BZBM"] = ""
    g["FRT_BZMC"] = ""
    g.loc[g.index[a01], "FRT_SCORE"] = np.round(score, 4)
    g.loc[g.index[a01], "FRT_LEVEL"] = lv
    keep = np.isin(lv, ["高", "中"])
    g.loc[g.index[a01[keep]], "FRT_BZBM"] = code
    g.loc[g.index[a01[keep]], "FRT_BZMC"] = label

    g.to_file(a.shp, driver="ESRI Shapefile", encoding="UTF-8")
    print(f"[shp] 已更新 {a.shp}（新增 FRT_BZBM/FRT_BZMC/FRT_SCORE/FRT_LEVEL）")

    df = pd.read_excel(a.xlsx, dtype=str) if a.xlsx.endswith((".xlsx", ".xls")) \
        else pd.read_csv(a.xlsx, dtype=str, encoding="utf-8-sig")
    if len(df) != len(g):
        raise SystemExit(f"[错误] xlsx 行数 {len(df)} vs shp {len(g)} 不一致")
    add = pd.DataFrame({
        "水果标的编码": g["FRT_BZBM"].values,
        "水果标的名称": g["FRT_BZMC"].values,
        "水果分值": g["FRT_SCORE"].values,
        "水果档位": g["FRT_LEVEL"].values,
    })
    for c in add.columns:                 # 重跑时避免重复追加
        if c in df.columns:
            df = df.drop(columns=c)
    df = pd.concat([df, add], axis=1)
    df.to_excel(a.xlsx, index=False)
    n_ok = int((df["水果标的编码"].astype(str).str.strip() != "").sum())
    print(f"[xlsx] 已更新 {a.xlsx}（新增 4 列；标为{label}的 {n_ok} 块）")


if __name__ == "__main__":
    main()
