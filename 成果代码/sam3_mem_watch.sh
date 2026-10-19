#!/bin/bash
# 看 worker 进程的内存增长（判断聚合阶段会不会 OOM）
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
echo "=== 内存（GB）==="; free -g | head -2
echo "=== 相关进程 RSS ==="
ps -eo pid,ppid,rss,etime,cmd --sort=-rss | grep -E 'geo_infer_v10_multigpu' | grep -v grep | \
  awk '{printf "PID=%-8s PPID=%-8s RSS=%7.2f GB  已运行=%-10s %s\n", $1, $2, $3/1048576, $4, substr($0, index($0,$5), 40)}'
echo "=== 已处理粗窗口数（SLIC 行）==="
grep -c '\[SLIC\]' /data3/0918my/my/510725zt/result_my10_v10_fix2/run.log
echo "=== 日志时间跨度 ==="
stat -c '%y' /data3/0918my/my/510725zt/result_my10_v10_fix2/run.log
ls -l --time-style=+%F_%T /data3/0918my/my/510725zt/result_my10_v10_fix2/run.log
EOF
