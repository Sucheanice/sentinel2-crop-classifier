# -*- coding: utf-8 -*-
"""批量跑梓潼/三台/江油 11 乡镇大春预测（20天 wide 口径）"""
import subprocess
import os

BASE = r"e:\工作相关\2026年\0624 待测试数据"
MODEL = "dachun_binary_23counties_3dekad_wide.pkl"
MODEL2 = "dachun_binary_2dekad_wide.pkl"
FILTER = "cropland_filter_general.pkl"

TOWNS = [
    ('梓潼', ['黎雅镇', '石牛镇', '金龙镇', '玛瑙镇']),
    ('三台', ['金石镇', '新鲁镇', '三元镇']),
    ('江油', ['新安镇', '太平镇', '马角镇', '河口镇']),
]


def run(cmd):
    print(f'\n>>> {cmd}', flush=True)
    return subprocess.run(cmd, shell=True, cwd=BASE).returncode


def main():
    os.chdir(BASE)
    for county, towns in TOWNS:
        out_dir = os.path.join(BASE, f'{county}结果')
        os.makedirs(out_dir, exist_ok=True)
        for town in towns:
            shp = f'{county}_{town}_地块.shp'
            slic = f'slic_{county}_{town}_wide.gpkg'
            name = f'{county}_{town}_wide_v4'
            print(f'\n{"="*70}\n[{county}/{town}]', flush=True)

            if not os.path.exists(slic):
                rc = run(f'python gee_slic_extract_wide.py --input "{shp}" --output "{slic}"')
                if rc != 0:
                    print(f'!! 分割失败 {town}，跳过', flush=True)
                    continue
            else:
                print(f'[skip] {slic} 已存在', flush=True)

            rc = run(f'python predict_slic_overlay.py --slic "{slic}" --external "{shp}" '
                     f'--model {MODEL} --model2 {MODEL2} --filter {FILTER} '
                     f'--output "{out_dir}" --name {name}')
            if rc != 0:
                print(f'!! 预测失败 {town}', flush=True)


if __name__ == '__main__':
    main()
