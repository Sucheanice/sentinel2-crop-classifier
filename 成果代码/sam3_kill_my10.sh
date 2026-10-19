#!/bin/bash
# 停掉当前 my10 推理，并确认显存释放
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
echo "=== 停前进程 ==="
ps -eo pid,cmd | grep -E '[g]eo_infer_v10_multigpu|[m]ultiprocessing' | head -6
pkill -f geo_infer_v10_multigpu.py
sleep 3
pkill -9 -f 'from multiprocessing' 2>/dev/null
sleep 5
echo "=== 停后进程 ==="
ps -eo pid,cmd | grep -E '[g]eo_infer_v10_multigpu' | head -3 || echo "  已无残留"
echo "=== 清中间产物 ==="
rm -rf /data3/0918my/my/510725zt/result_my10_v10_fix2/.tmp_mp
echo "=== 显存（应回落到只剩 vLLM 占用）==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo "=== 内存 ==="
free -g | head -2
EOF
