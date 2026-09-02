# -*- coding: utf-8 -*-
"""果园识别全流程一键跑: JP2处理 → 特征重建 → LightGBM训练+预测"""
import os, sys, time, subprocess
from pathlib import Path

PROJ_DIR = Path(r"E:\工作相关\2026年\0624 待测试数据\果树识别_step1_coldstart")
SCRIPTS = PROJ_DIR / "scripts"
DATA = PROJ_DIR / "data"

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def run_step(name, script, env=None):
    log(f"\n{'='*60}")
    log(f"STEP: {name}")
    log(f"{'='*60}")
    result = subprocess.run(
        ["python", str(script)],
        cwd=str(SCRIPTS.parent.parent),
        env=env,
        capture_output=False,
    )
    if result.returncode != 0:
        log(f"[FAIL] {name} 退出码={result.returncode}")
        return False
    log(f"[OK] {name}")
    return True

def main():
    t0 = time.time()
    log("=" * 70)
    log("果园识别全流程: JP2处理 → 特征重建 → LightGBM训练+预测")
    log("=" * 70)

    # 设置 PROJ_LIB
    proj_lib = r"C:\Users\lenovo\AppData\Roaming\Python\Python313\site-packages\rasterio\proj_data"
    env = os.environ.copy()
    env["PROJ_LIB"] = proj_lib

    jp2_dir = Path(r"E:\迅雷下载\08121721")
    bands = ["B02","B03","B04","B05","B06","B07","B08","B8A","B11","B12"]

    # 1. 检查 JP2 是否下载完毕
    log("\n检查 JP2 下载...")
    missing = [b for b in bands if not (jp2_dir / f"{b}.jp2").exists()]
    if missing:
        log(f"[WAITING] 缺失: {missing}")
        log("等待60秒后重试...")
        time.sleep(60)
        missing = [b for b in bands if not (jp2_dir / f"{b}.jp2").exists()]
        if missing:
            log(f"[FATAL] 仍有 {len(missing)} 个文件缺失: {missing}")
            return
    skips = 0
    while True:
        missing = [b for b in bands if not (jp2_dir / f"{b}.jp2").exists()]
        if not missing:
            break
        skips += 1
        if skips > 10:
            log(f"[FATAL] 等待超时, 缺失: {missing}")
            return
        log(f"  等待下载完成... ({len(missing)}/10) [{skips*30}s]")
        time.sleep(30)
    log("全部10波段已就绪")

    # 2. 处理 JP2 → 多波段 GeoTIFF
    if not run_step("JP2处理", SCRIPTS / "process_xunlei_jp2.py", env):
        return

    # 3. 清除遂宁特征缓存
    suining_cache = DATA / "suining_features"
    for f in suining_cache.glob("*"):
        if f.suffix in [".npy", ".json"]:
            f.unlink()
    log("遂宁特征缓存已清除")

    # 4. 重建遂宁特征立方体（4季）
    if not run_step("遂宁特征重建", SCRIPTS / "rebuild_suining_cube.py", env):
        return

    # 5. 清除叙永特征缓存
    xuyong_cache = DATA / "xuyong_features"
    for f in xuyong_cache.glob("*"):
        if f.suffix in [".npy", ".json"]:
            f.unlink()
    log("叙永特征缓存已清除")

    # 6. 转移学习（LightGBM版）
    if not run_step("转移学习训练+预测", SCRIPTS / "run_transfer_learning.py", env):
        return

    elapsed = time.time() - t0
    log(f"\n{'='*70}")
    log(f"全流程完成! 总耗时: {elapsed/60:.1f} 分钟")
    log(f"{'='*70}")


if __name__ == "__main__":
    main()
