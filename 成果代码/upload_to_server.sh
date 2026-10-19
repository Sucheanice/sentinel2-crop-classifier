#!/bin/bash
# 断点续传上传（Windows 目录 -> 远程服务器），在 WSL 里跑
#
#   bash upload_to_server.sh [本地目录] [远程目录] [--detach]
#   例：bash upload_to_server.sh F:/0918my /data3/0918my --detach
#
# 特性：
#   * rsync 断点续传：中断后重跑同一条命令即从中断处继续（半截文件放 .rsync-partial/）
#   * --detach：脱离终端在独立会话后台运行（关掉终端/IDE 也不断），启动日志在 ~/<脚本名>.boot.log
#   * 全程写日志，可随时用 upload_status.sh 查看进度
#   * 密码不写死：从《10.62.11.15_服务器恢复手册.md》里读取，写入临时 askpass 脚本（用完可删）
set -u

# --detach：先把 --detach 摘出来，再以独立会话重新拉起自己
DETACH=0
ARGS=()
for a in "$@"; do
  if [ "$a" = "--detach" ]; then DETACH=1; else ARGS+=("$a"); fi
done
if [ "$DETACH" = "1" ] && [ "${_DETACHED:-0}" != "1" ]; then
  BOOT="$HOME/$(basename "$0").boot.log"
  _DETACHED=1 setsid nohup env _DETACHED=1 bash "$0" "${ARGS[@]}" </dev/null >"$BOOT" 2>&1 &
  sleep 3
  echo "[已后台启动] PID=$!  启动日志: $BOOT"
  exit 0
fi
set -- "${ARGS[@]}"

SRC_WIN="${1:-F:/0918my}"
DST_DIR="${2:-/data3/0918my}"
HOST="${HOST:-10.62.11.15}"
RUSER="${RUSER:-root}"
LOG="${LOG:-$HOME/upload_$(basename "$DST_DIR").log}"
AP="$HOME/.askpass_$(basename "$DST_DIR")"
MANUAL="/mnt/e/工作相关/2026年/0624 待测试数据/文档/10.62.11.15_服务器恢复手册.md"

# 1) 取密码（不落盘到脚本里）
PW=$(grep -m1 -E '^# *密码' "$MANUAL" 2>/dev/null | sed -E 's/^# *密码 *[:：] *//' | tr -d '\r\n')
if [ -z "$PW" ]; then echo "[错误] 没能从 $MANUAL 读到密码，请检查该文件的『# 密码:』行"; exit 1; fi
printf '#!/bin/sh\nprintf "%%s\\n" %q\n' "$PW" > "$AP"
chmod 700 "$AP"
export SSH_ASKPASS="$AP" SSH_ASKPASS_REQUIRE=force DISPLAY=:0

# 2) Windows 路径 -> WSL 路径
SRC_WSL=$(printf '%s' "$SRC_WIN" | sed -E 's|^([A-Za-z]):|/mnt/\L\1|')
[ -d "$SRC_WSL" ] || { echo "[错误] 本地目录不存在: $SRC_WSL"; exit 1; }

SSHOPT="-o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=6"

echo "=== 上传 $SRC_WSL/  ->  $RUSER@$HOST:$DST_DIR/ ==="
echo "开始时间: $(date '+%F %T')"
echo "日志: $LOG"

# 3) 连通性 + 建目标目录
ssh $SSHOPT "$RUSER@$HOST" "mkdir -p '$DST_DIR' && echo '[远端就绪]' && df -h '$DST_DIR' | tail -1" || {
    echo "[错误] SSH 连接或建目录失败"; exit 1; }

# 4) 上传（-a 保留时间戳；--partial-dir 半截文件单独存放，重跑自动续传）
rsync -a --partial-dir=.rsync-partial --human-readable --info=progress2 --no-inc-recursive \
      -e "ssh $SSHOPT" "$SRC_WSL/" "$RUSER@$HOST:$DST_DIR/" 2>&1 | tee -a "$LOG"

RC=${PIPESTATUS[0]}
echo "结束时间: $(date '+%F %T')  退出码=$RC"
[ "$RC" -eq 0 ] && echo "[完成] 上传结束（重跑本脚本可校验/补齐）" || echo "[中断] 退出码 $RC，重跑本脚本即从中断处续传"
