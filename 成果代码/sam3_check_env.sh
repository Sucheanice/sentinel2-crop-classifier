#!/bin/bash
# 在服务器上检查 SAM3 推理环境：影像 / 代码 / 权重 / 显卡 / 环境
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
echo "=========== 1. 目标影像 ==========="
ls -lh /data3/0918my/my/510725zt/ 2>&1 | head -40
IMG=/data3/0918my/my/510725zt/my10.tif
if [ -f "$IMG" ]; then
  echo "[存在] $IMG  $(stat -c '%s bytes, mtime=%y' "$IMG")"
  command -v gdalinfo >/dev/null && gdalinfo "$IMG" 2>&1 | head -35 || echo "(无 gdalinfo)"
else
  echo "[不存在] $IMG"
  echo "--- 在 /data3/0918my/my 下找同名 ---"
  find /data3/0918my/my -name 'my10*' -printf '%s\t%p\n' 2>/dev/null | head
fi

echo
echo "=========== 2. 推理代码 ==========="
for f in /data3/sam3_project/sam3_service/geo_infer_v10_multigpu.py \
         /data4/sam3_project/sam3_service/geo_infer_v10_multigpu.py; do
  [ -f "$f" ] && echo "[有] $f  $(stat -c '%s bytes, mtime=%y' "$f")" || echo "[无] $f"
done
echo "--- sam3_project 顶层 ---"
ls -d /data3/sam3_project/* 2>/dev/null | head -20

echo
echo "=========== 3. 微调权重 checkpoint_25.pt ==========="
find /data3/sam3_project /data4/sam3_project -name 'checkpoint_25.pt' -printf '%s\t%p\n' 2>/dev/null

echo
echo "=========== 4. 显卡 ==========="
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv 2>&1 | head -12

echo
echo "=========== 5. 磁盘 ==========="
df -h /data3 /data4 2>&1 | grep -v tmpfs

echo
echo "=========== 6. Python 环境 ==========="
ls /root/miniconda3/envs 2>/dev/null || ls /opt/conda/envs 2>/dev/null || echo "(未找到 conda envs)"
for p in /root/miniconda3/envs/*/bin/python; do
  [ -x "$p" ] && echo "$p -> $($p -c 'import sys;print(sys.version.split()[0])' 2>&1)"
done
python3 -c "import torch,sys;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.device_count())" 2>&1 | tail -1

echo
echo "=========== 7. 已有的 baozhucun fix2 产物（对照） ==========="
ls -l /data3/sam3_project/test_data/result_baozhucun_v10_fix2/ 2>&1 | head
EOF
