#!/bin/bash
# 三查：代码指纹（对比本地 fix2 副本）、窗口数、占用 GPU 的进程
set -u
HOST="${HOST:-10.62.11.15}"
RUSER="${RUSER:-root}"
AP="$HOME/.askpass_srv"
MANUAL="/mnt/e/工作相关/2026年/0624 待测试数据/文档/10.62.11.15_服务器恢复手册.md"
if [ ! -x "$AP" ]; then
  PW=$(grep -m1 -E '^# *密码' "$MANUAL" 2>/dev/null | sed -E 's/^# *密码 *[:：] *//' | tr -d '\r\n')
  printf '#!/bin/sh\nprintf "%%s\\n" %q\n' "$PW" > "$AP"; chmod 700 "$AP"
fi
export SSH_ASKPASS="$AP" SSH_ASKPASS_REQUIRE=force DISPLAY=:0
ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 "$RUSER@$HOST" 'bash -s' <<'EOF'
D=/data4/sam3_project/sam3_train/Code/sam3-main
echo "=========== 1. 代码指纹 ==========="
md5sum $D/sam3_service/geo_infer_v10_multigpu.py $D/sam3_service/geo_infer_v10.py \
       /data3/sam3_project/sam3_service/geo_infer_v10.py 2>&1

echo
echo "=========== 2. 新影像窗口数（用生产代码的窗口函数算）==========="
cd $D/sam3_service
/root/anaconda3/envs/sam3/bin/python - <<'PY'
import sys, rasterio
sys.path.insert(0, '/data4/sam3_project/sam3_train/Code/sam3-main/sam3_service')
sys.path.insert(0, '/data4/sam3_project/sam3_train/Code/sam3-main')
import geo_infer_v10_multigpu as G
for tag, img in [('my10 梓潼', '/data3/0918my/my/510725zt/my10.tif'),
                 ('baozhucun 对照', '/data3/sam3_project/test_data/tif/baozhucun.tif')]:
    with rasterio.open(img) as s:
        h, w, c = s.height, s.width, s.count
        co = G.build_coarse_windows(h, w, 1008, 672, 168)
        fi = G.build_fine_windows(h, w, 588, 448, 112)
        kept_c, sk_c = G._prefilter_windows(s, co)
        print('%s: %dx%d px, %d band -> coarse %d(有效%d) fine %d | 合计 %d 有效窗口' % (
            tag, w, h, c, len(co), len(kept_c), len(fi), len(kept_c)+len(fi)))
PY

echo
echo "=========== 3. 占显存的进程详情 ==========="
for p in 803579 1645749; do
  echo "--- pid $p ---"
  ps -o pid,etimes,lstart,cmd -p $p 2>&1 | tail -2
done
echo "--- 全部 python 进程（最近启动 10 个）---"
ps -eo pid,etimes,cmd --sort=-etimes | grep -E '[p]ython' | head -10
EOF
