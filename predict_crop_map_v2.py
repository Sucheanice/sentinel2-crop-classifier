# -*- coding: utf-8 -*-
import numpy as np
import pickle, os, time, contextlib
import rasterio
from rasterio.windows import Window

MODEL_PATH = r"E:\工作相关\2026年\0624 待测试数据\待训练数据\crop_model.pkl"
DATA_DIR = r"E:\工作相关\2026年\0624 待测试数据\待训练数据4"
OUTPUT = r"E:\工作相关\2026年\0624 待测试数据\待训练数据\crop_map_anju.tif"
BLOCK_SIZE = 1024

def compute_indices(blue, green, red, nir):
    eps = 1e-10
    denom_evi = nir + 6.0 * red - 7.5 * blue + 1.0
    evi = 2.5 * (nir - red) / np.where(np.abs(denom_evi) < eps, eps, denom_evi)
    denom_ndwi = green + nir
    ndwi = (green - nir) / np.where(np.abs(denom_ndwi) < eps, eps, denom_ndwi)
    denom_savi = nir + red + 0.5
    savi = 1.5 * (nir - red) / np.where(np.abs(denom_savi) < eps, eps, denom_savi)
    return evi, ndwi, savi

def main():
    print("Loading model...")
    with open(MODEL_PATH, 'rb') as f:
        bundle = pickle.load(f)
    model = bundle['model']
    selected_features = bundle['selected_features']
    all_feature_names = bundle['all_feature_names']
    scene_labels = bundle['scene_labels']
    scene_dirs = bundle['scene_dirs']
    n_scenes = len(scene_dirs)

    feat_idx_map = [all_feature_names.index(f) for f in selected_features
                    if f in all_feature_names]

    first_dir = os.path.join(DATA_DIR, scene_dirs[0])
    with rasterio.open(os.path.join(first_dir, 'B02.tif')) as ref:
        width = ref.width
        height = ref.height
        profile = ref.profile.copy()
    profile.update(dtype='int16', count=1, compress='lzw', nodata=-1)
    print("  %d x %d, %d scenes, %d features" % (width, height, n_scenes, len(feat_idx_map)))

    readers = []
    for si in range(n_scenes):
        dp = os.path.join(DATA_DIR, scene_dirs[si])
        scene_readers = []
        for band in ['B02', 'B03', 'B04', 'B08']:
            scene_readers.append(rasterio.open(os.path.join(dp, band + '.tif')))
        readers.append(scene_readers)

    n_blocks_y = (height + BLOCK_SIZE - 1) // BLOCK_SIZE
    n_blocks_x = (width + BLOCK_SIZE - 1) // BLOCK_SIZE
    total_blocks = n_blocks_y * n_blocks_x

    print("Predicting %d blocks (%d x %d)..." % (total_blocks, n_blocks_x, n_blocks_y))
    t0 = time.time()

    with rasterio.open(OUTPUT, 'w', **profile) as dst:
        block_num = 0
        for by in range(n_blocks_y):
            y0 = by * BLOCK_SIZE
            y1 = min(y0 + BLOCK_SIZE, height)
            bh = y1 - y0

            for bx in range(n_blocks_x):
                x0 = bx * BLOCK_SIZE
                x1 = min(x0 + BLOCK_SIZE, width)
                bw = x1 - x0
                window = Window(x0, y0, bw, bh)
                block_num += 1

                feature_blocks = []
                for si in range(n_scenes):
                    blue  = readers[si][0].read(1, window=window).astype(np.float32)
                    green = readers[si][1].read(1, window=window).astype(np.float32)
                    red   = readers[si][2].read(1, window=window).astype(np.float32)
                    nir   = readers[si][3].read(1, window=window).astype(np.float32)

                    denom = nir + red
                    ndvi = (nir - red) / np.where(denom < 1e-10, 1e-10, denom)
                    evi, ndwi, savi = compute_indices(blue, green, red, nir)

                    feature_blocks.extend([blue, green, red, nir, ndvi, evi, ndwi, savi])

                X_block = np.stack(feature_blocks, axis=-1)
                X_flat = X_block.reshape(-1, 8 * n_scenes)
                X_flat = np.nan_to_num(X_flat, nan=0.0)
                X_selected = X_flat[:, feat_idx_map].astype(np.float32)

                y_pred = model.predict(X_selected).astype(np.int16)
                y_pred = y_pred.reshape(bh, bw)
                dst.write(y_pred, 1, window=window)

                if block_num % 20 == 0 or block_num == total_blocks:
                    elapsed = time.time() - t0
                    eta = elapsed / block_num * (total_blocks - block_num)
                    print("  [%d/%d] %.1fs elapsed, ETA %.1fs" % (block_num, total_blocks, elapsed, eta))

    for si in range(n_scenes):
        for r in readers[si]:
            r.close()

    elapsed = time.time() - t0
    print("Done in %.1f min" % (elapsed / 60))
    print("Saved: %s" % OUTPUT)

if __name__ == "__main__":
    main()
