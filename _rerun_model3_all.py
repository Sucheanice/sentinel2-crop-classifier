# -*- coding: utf-8 -*-
"""统一用 model2(D04+D14)+model3(D11+D14) 双两旬兜底，重跑所有已跑任务区乡镇。

slic_*_wide.gpkg 均已存在，只重跑 predict_slic_overlay.py（补 --model3）。
"""
import subprocess
import os

BASE = r"e:\工作相关\2026年\0624 待测试数据"
MODEL = "dachun_binary_23counties_3dekad_wide.pkl"
MODEL2 = "dachun_binary_2dekad_wide.pkl"
MODEL3 = "dachun_binary_2dekad_d11d14_wide.pkl"
FILTER = "cropland_filter_general.pkl"

# (输出目录, slic, 外部地块shp, 输出前缀name)
TASKS = [
    # 梓潼 4 镇
    ('梓潼结果', 'slic_梓潼_黎雅镇_wide.gpkg', '梓潼_黎雅镇_地块.shp', '梓潼_黎雅镇_wide_v4'),
    ('梓潼结果', 'slic_梓潼_石牛镇_wide.gpkg', '梓潼_石牛镇_地块.shp', '梓潼_石牛镇_wide_v4'),
    ('梓潼结果', 'slic_梓潼_金龙镇_wide.gpkg', '梓潼_金龙镇_地块.shp', '梓潼_金龙镇_wide_v4'),
    ('梓潼结果', 'slic_梓潼_玛瑙镇_wide.gpkg', '梓潼_玛瑙镇_地块.shp', '梓潼_玛瑙镇_wide_v4'),
    # 三台 3 镇
    ('三台结果', 'slic_三台_金石镇_wide.gpkg', '三台_金石镇_地块.shp', '三台_金石镇_wide_v4'),
    ('三台结果', 'slic_三台_新鲁镇_wide.gpkg', '三台_新鲁镇_地块.shp', '三台_新鲁镇_wide_v4'),
    ('三台结果', 'slic_三台_三元镇_wide.gpkg', '三台_三元镇_地块.shp', '三台_三元镇_wide_v4'),
    # 江油 3 镇（新安镇已重跑）
    ('江油结果', 'slic_江油_太平镇_wide.gpkg', '江油_太平镇_地块.shp', '江油_太平镇_wide_v4'),
    ('江油结果', 'slic_江油_马角镇_wide.gpkg', '江油_马角镇_地块.shp', '江油_马角镇_wide_v4'),
    ('江油结果', 'slic_江油_河口镇_wide.gpkg', '江油_河口镇_地块.shp', '江油_河口镇_wide_v4'),
    # 盐亭 4 乡镇
    ('盐亭结果', 'slic_大兴回族乡_wide.gpkg', '盐亭_大兴回族乡_地块.shp', '大兴回族乡_wide_v4'),
    ('盐亭结果', 'slic_富驿镇_wide.gpkg', '盐亭_富驿镇_地块.shp', '富驿镇_wide_v4'),
    ('盐亭结果', 'slic_莲花湖乡_wide.gpkg', '盐亭_莲花湖乡_地块.shp', '莲花湖乡_wide_v4'),
    ('盐亭结果', 'slic_文通镇_wide.gpkg', '盐亭_文通镇_地块.shp', '文通镇_wide_v4'),
]


def run(cmd):
    print(f'\n>>> {cmd}', flush=True)
    return subprocess.run(cmd, shell=True, cwd=BASE).returncode


def main():
    os.chdir(BASE)
    for out_dir, slic, external, name in TASKS:
        out = os.path.join(BASE, out_dir)
        os.makedirs(out, exist_ok=True)
        if not os.path.exists(slic):
            print(f'!! 缺少 slic {slic}，跳过 {name}', flush=True)
            continue
        if not os.path.exists(external):
            print(f'!! 缺少地块 {external}，跳过 {name}', flush=True)
            continue
        print(f'\n{"="*70}\n[{name}]', flush=True)
        rc = run(f'python predict_slic_overlay.py --slic "{slic}" --external "{external}" '
                 f'--model {MODEL} --model2 {MODEL2} --model3 {MODEL3} --filter {FILTER} '
                 f'--output "{out}" --name {name}')
        if rc != 0:
            print(f'!! 预测失败 {name}', flush=True)


if __name__ == '__main__':
    main()
