# -*- coding: utf-8 -*-
import rasterio, os, time
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds
from pyproj import Transformer

DOM_PATH = r"E:\工作相关\2026年\0624 待测试数据\待训练数据\DOM\人保-安居区DOM.img"
S2_DIR = r"E:\工作相关\2026年\0624 待测试数据\待训练数据4"
OUT_DIR = r"E:\工作相关\2026年\0624 待测试数据\待训练数据4_cropped"
BUFFER = 200
BANDS = ["B02", "B03", "B04", "B08"]

def main():
    with rasterio.open(DOM_PATH) as dom:
        dom_crs = dom.crs
        dom_bounds = dom.bounds

    s2_dirs = sorted([d for d in os.listdir(S2_DIR)
                      if os.path.isdir(os.path.join(S2_DIR, d)) and d[0].isdigit()])

    with rasterio.open(os.path.join(S2_DIR, s2_dirs[0], "B02.tif")) as ref:
        s2_crs = ref.crs

    transformer = Transformer.from_crs(dom_crs, s2_crs, always_xy=True)
    dl, db = transformer.transform(dom_bounds.left, dom_bounds.bottom)
    dr, dt = transformer.transform(dom_bounds.right, dom_bounds.top)

    crop_bounds = (dl - BUFFER, db - BUFFER, dr + BUFFER, dt + BUFFER)
    print("DOM bounds (UTM): [%.0f, %.0f, %.0f, %.0f]" % crop_bounds)
    print("%d scenes, bands: %s" % (len(s2_dirs), BANDS))

    t0 = time.time()
    for di, dname in enumerate(s2_dirs):
        scene_out = os.path.join(OUT_DIR, dname)
        os.makedirs(scene_out, exist_ok=True)

        for band in BANDS:
            src_path = os.path.join(S2_DIR, dname, band + ".tif")
            dst_path = os.path.join(scene_out, band + ".tif")

            if os.path.exists(dst_path):
                continue

            with rasterio.open(src_path) as src:
                window = from_bounds(*crop_bounds, transform=src.transform)
                window = window.round_offsets().round_shape()
                data = src.read(window=window)
                transform = src.window_transform(window)

                profile = src.profile.copy()
                profile.update(
                    height=data.shape[1], width=data.shape[2],
                    transform=transform, compress="lzw", tiled=True,
                    blockxsize=256, blockysize=256
                )

                with rasterio.open(dst_path, "w", **profile) as dst:
                    dst.write(data)

            print("  [%d/%d] %s/%s done (%d x %d)" % (
                di + 1, len(s2_dirs), dname, band, data.shape[2], data.shape[1]))

    print("Done in %.1fs" % (time.time() - t0))

if __name__ == "__main__":
    main()
