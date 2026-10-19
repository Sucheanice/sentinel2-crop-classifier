#!/bin/bash
# 用 fix2 那套代码+权重跑新影像：/data3/0918my/my/510725zt/my10.tif
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
set -u
D=/data4/sam3_project/sam3_train/Code/sam3-main
SERVICE=$D/sam3_service
PY=/root/anaconda3/envs/sam3/bin/python
IMG=/data3/0918my/my/510725zt/my10.tif
OUTDIR=/data3/0918my/my/510725zt/result_my10_v10_fix2
OUT=$OUTDIR/my10_v10_fix2
CKPT=/data4/sam3_project/sam3_train/data/runs/finetune_v3/checkpoints/checkpoint_25.pt

echo "=========== 启动前自检 ==========="
for m in _filter_dark_polygons max_vertices _merge_touching_and_fill _cross_scale_merge _final_smooth_polygon; do
  n=$(grep -c "$m" $SERVICE/geo_infer_v10.py)
  echo "  geo_infer_v10.py 含 $m : $n 处"
done
echo "  geo_infer_v10.py 含 '归一化到 0-255' : $(grep -c '归一化到 0-255' $SERVICE/geo_infer_v10.py) 处"
echo "  权重: $(ls -l $CKPT | awk '{print $5" bytes"}')"
echo "  影像: $(ls -l $IMG | awk '{print $5" bytes"}')"
[ -f "$IMG" ] && [ -f "$CKPT" ] || { echo "[中止] 影像或权重缺失"; exit 1; }

mkdir -p $OUTDIR
cd $SERVICE
export SAM3_ROOT=$D
export PYTHONPATH=$SERVICE:$D
echo
echo "=========== 启动 ==========="
nohup $PY geo_infer_v10_multigpu.py \
  --input $IMG \
  --output $OUT \
  --prompt farmland \
  --gpus 0,1,6,7 \
  --checkpoint $CKPT \
  --geometric-refine 0 --confidence-threshold 0.3 \
  --dark-p10-threshold 25 --ndvi-threshold 0 \
  --simplify-tolerance 1.0 --mask-smooth-sigma 1.0 \
  --max-vertices 200 \
  --min-area-m2 67 --merge-touching 1 --merge-fill-hole-m2 0.0 \
  --stretch 0 --slic-points 1 --slic-min-brightness 0 \
  > $OUTDIR/run.log 2>&1 &
echo "已后台启动 PID=$!"
echo "日志: $OUTDIR/run.log"

echo
echo "=========== 60 秒后看日志 ==========="
sleep 60
tail -c 1500 $OUTDIR/run.log
echo
echo "--- 显存 ---"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
EOF
