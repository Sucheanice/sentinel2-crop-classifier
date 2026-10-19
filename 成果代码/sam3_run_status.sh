#!/bin/bash
# 服务器资源 + my10 运行状态速查
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
echo "=== 内存（GB）==="; free -g
echo "=== CPU 核数 ==="; nproc
echo "=== 输出目录 ==="; ls -la /data3/0918my/my/510725zt/result_my10_v10_fix2/ 2>/dev/null
echo "=== 中间 pickle 目录 ==="; du -sh /data3/0918my/my/510725zt/result_my10_v10_fix2/.tmp_mp 2>/dev/null; ls -la /data3/0918my/my/510725zt/result_my10_v10_fix2/.tmp_mp 2>/dev/null | head
echo "=== 主进程内存占用 ==="; ps -eo pid,rss,etime,cmd --sort=-rss | grep -E '[g]eo_infer_v10_multigpu' | head -5 | awk '{printf "PID=%s RSS=%.1fGB 已运行=%s\n", $1, $2/1048576, $3}'
echo "=== 日志进度（最后 3 行）==="; tail -c 400 /data3/0918my/my/510725zt/result_my10_v10_fix2/run.log | tr '\r' '\n' | tail -3
echo "=== 已处理窗口数（SLIC 行计数）==="; grep -c '\[SLIC\]' /data3/0918my/my/510725zt/result_my10_v10_fix2/run.log
echo "=== 显存 ==="; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
EOF
