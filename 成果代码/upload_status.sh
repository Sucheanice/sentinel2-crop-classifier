#!/bin/bash
# 随时查看上传进度（在 WSL 里跑）
#
#   bash upload_status.sh [本地目录] [远程目录]
#   例：bash upload_status.sh F:/0918my /data3/0918my
set -u

SRC_WIN="${1:-F:/0918my}"
DST_DIR="${2:-/data3/0918my}"
HOST="${HOST:-10.62.11.15}"
RUSER="${RUSER:-root}"
LOG="${LOG:-$HOME/upload_$(basename "$DST_DIR").log}"
AP="$HOME/.askpass_$(basename "$DST_DIR")"
MANUAL="/mnt/e/工作相关/2026年/0624 待测试数据/文档/10.62.11.15_服务器恢复手册.md"

if [ ! -x "$AP" ]; then
  PW=$(grep -m1 -E '^# *密码' "$MANUAL" 2>/dev/null | sed -E 's/^# *密码 *[:：] *//' | tr -d '\r\n')
  [ -n "$PW" ] || { echo "[错误] 未能从手册读到密码"; exit 1; }
  printf '#!/bin/sh\nprintf "%%s\\n" %q\n' "$PW" > "$AP"; chmod 700 "$AP"
fi
export SSH_ASKPASS="$AP" SSH_ASKPASS_REQUIRE=force DISPLAY=:0
SSHOPT="-o StrictHostKeyChecking=accept-new -o ConnectTimeout=15"

SRC_WSL=$(printf '%s' "$SRC_WIN" | sed -E 's|^([A-Za-z]):|/mnt/\L\1|')

LOC_B=$(find "$SRC_WSL" -type f -printf '%s\n' 2>/dev/null | awk '{s+=$1} END{print s+0}')
LOC_N=$(find "$SRC_WSL" -type f 2>/dev/null | wc -l)

REM=$(ssh $SSHOPT "$RUSER@$HOST" "
  if [ -d '$DST_DIR' ]; then
    echo \"B \$(du -sb '$DST_DIR' 2>/dev/null | cut -f1)\"
    echo \"N \$(find '$DST_DIR' -type f -not -path '*/.rsync-partial/*' | wc -l)\"
    echo \"P \$(du -sb '$DST_DIR/.rsync-partial' 2>/dev/null | cut -f1)\"
    echo \"R \$(ps -eo etimes,cmd | grep '[r]sync' | awk '{print \$1}' | head -1 | tr -d ' ' | grep . || echo 0)\"
  else echo 'B 0'; echo 'N 0'; echo 'P 0'; echo 'R 0'; fi")

REM_B=$(printf '%s\n' "$REM" | awk '/^B /{print $2+0}')
REM_N=$(printf '%s\n' "$REM" | awk '/^N /{print $2+0}')
REM_P=$(printf '%s\n' "$REM" | awk '/^P /{print $2+0}')
RUN=$(printf '%s\n' "$REM" | awk '/^R /{print $2+0}')

fmt() { awk -v b="$1" 'BEGIN{ if(b>=1073741824) printf "%.2f GB", b/1073741824; else printf "%.1f MB", b/1048576 }'; }
PCT=$(awk -v a="$REM_B" -v b="$LOC_B" 'BEGIN{ if(b>0) printf "%.1f%%", 100*a/b; else print "n/a" }')

echo "=== 上传进度  $(date '+%F %T') ==="
printf "本地 %s 个文件 / %s\n" "$LOC_N" "$(fmt "$LOC_B")"
printf "远端 %s 个文件 / %s （含未完成分片 %s）\n" "$REM_N" "$(fmt "$REM_B")" "$(fmt "$REM_P")"
printf "进度 %s\n" "$PCT"
[ -n "$RUN" ] && [ "$RUN" -gt 0 ] 2>/dev/null && printf "rsync 已运行 %d 分 %d 秒\n" $((RUN/60)) $((RUN%60)) || echo "rsync 当前未在运行（中断过的话，重跑 upload_to_server.sh 即续传）"
echo "--- 日志末尾 ---"
[ -f "$LOG" ] && tail -c 400 "$LOG" | tr '\r' '\n' | tail -4 || echo "(暂无日志)"
# 机器可读行（便于脚本/agent 稳定解析）：RAW 远端字节 文件数 rsync秒 本地字节
echo "RAW $REM_B $REM_N ${RUN:-0} $LOC_B"
