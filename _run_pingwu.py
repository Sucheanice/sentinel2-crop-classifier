# -*- coding: utf-8 -*-
"""批量跑平武已有 3 乡镇大春预测（20天 wide 口径 + D11/D14 兜底）"""
import subprocess
import os

BASE = r"e:\工作相关\2026年\0624 待测试数据"
MODEL = "dachun_binary_23counties_3dekad_wide.pkl"
MODEL2 = "dachun_binary_2dekad_wide.pkl"
MODEL3 = "dachun_binary_2dekad_d11d14_wide.pkl"
FILTER = "cropland_filter_general.pkl"

TOWNS = ['大桥镇', '土城藏族乡', '旧堡羌族乡']


def run(cmd):
    print(f'\n>>> {cmd}', flush=True)
    return subprocess.run(cmd, shell=True, cwd=BASE).returncode


def main():
    os.chdir(BASE)
    out_dir = os.path.join(BASE, '平武结果')
    os.makedirs(out_dir, exist_ok=True)
    for town in TOWNS:
        shp = f'平武_{town}_地块.shp'
        slic = f'slic_平武_{town}_wide.gpkg'
        name = f'平武_{town}_wide_v4'
        print(f'\n{"="*70}\n[平武/{town}]', flush=True)

        if not os.path.exists(slic):
            rc = run(f'python gee_slic_extract_wide.py --input "{shp}" --output "{slic}"')
            if rc != 0:
                print(f'!! 分割失败 {town}，跳过', flush=True)
                continue
        else:
            print(f'[skip] {slic} 已存在', flush=True)

        rc = run(f'python predict_slic_overlay.py --slic "{slic}" --external "{shp}" '
                 f'--model {MODEL} --model2 {MODEL2} --model3 {MODEL3} --filter {FILTER} '
                 f'--output "{out_dir}" --name {name}')
        if rc != 0:
            print(f'!! 预测失败 {town}', flush=True)


if __name__ == '__main__':
    main()
