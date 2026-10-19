#!/bin/bash
# 找 worker 进程（spawn 出来的 cmdline 不含脚本名）+ 内存增长 + 进度
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
echo "=== 占内存前 10 的进程 ==="
ps -eo pid,ppid,rss,etime,cmd --sort=-rss | head -11 | \
  awk 'NR==1{print; next} {printf "%-9s %-9s %7.2f GB %-10s %.60s\n", $1, $2, $3/1048576, $4, substr($0, index($0,$5))}'
echo
echo "=== 找 spawn 出来的 worker ==="
ps -eo pid,ppid,rss,etime,cmd | grep -E '[s]pawn_main|[m]ultiprocessing' | \
  awk '{printf "PID=%-9s PPID=%-9s RSS=%7.2f GB 已运行=%-10s\n", $1, $2, $3/1048576, $4}'
echo
echo "=== 进度 ==="
N=$(grep -c '\[SLIC\]' /data3/0918my/my/510725zt/result_my10_v10_fix2/run.log)
echo "已处理粗窗口 SLIC 行数 = $N / 8906"
head -3 /data3/0918my/my/510725zt/result_my10_v10_fix2/run.log
echo "run.log 最后修改: $(stat -c '%y' /data3/0918my/my/510725zt/result_my10_v10_fix2/run.log)"
EOF
