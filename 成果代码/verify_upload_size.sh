#!/bin/bash
# 上传结果核对：本地盘数据量 vs 服务器文件夹数据量（只读，不改任何文件）
set -u
SRC_WIN="${1:-F:/0918my}"
DST_DIR="${2:-/data3/0918my}"
HOST="${HOST:-10.62.11.15}"
RUSER="${RUSER:-root}"
AP="$HOME/.askpass_$(basename "$DST_DIR")"
MANUAL="/mnt/e/工作相关/2026年/0624 待测试数据/文档/10.62.11.15_服务器恢复手册.md"
if [ ! -x "$AP" ]; then
  PW=$(grep -m1 -E '^# *密码' "$MANUAL" 2>/dev/null | sed -E 's/^# *密码 *[:：] *//' | tr -d '\r\n')
  printf '#!/bin/sh\nprintf "%%s\\n" %q\n' "$PW" > "$AP"; chmod 700 "$AP"
fi
export SSH_ASKPASS="$AP" SSH_ASKPASS_REQUIRE=force DISPLAY=:0
SSHOPT="-o StrictHostKeyChecking=accept-new -o ConnectTimeout=20"
SRC_WSL=$(printf '%s' "$SRC_WIN" | sed -E 's|^([A-Za-z]):|/mnt/\L\1|')

cd "$SRC_WSL" || { echo "[错误] 本地目录不存在: $SRC_WSL"; exit 1; }
find . -type f -printf '%s\t%p\n' > /tmp/_loc.list
ssh $SSHOPT "$RUSER@$HOST" "cd '$DST_DIR' && find . -type f -printf '%s\t%p\n'" > /tmp/_rem.list 2>/dev/null

sumf() { awk -F'\t' '{s+=$1} END{printf "%.0f", s+0}' "$1"; }
cntf() { wc -l < "$1"; }
fmt() { awk -v b="$1" 'BEGIN{printf "%.2f GB", b/1073741824}'; }
LB=$(sumf /tmp/_loc.list); RB=$(sumf /tmp/_rem.list)
LN=$(cntf /tmp/_loc.list); RN=$(cntf /tmp/_rem.list)

echo "=================== 总体核对 ==================="
printf "本地 %-8s : %12d 字节  (%s)  %d 个文件\n" "$SRC_WIN" "$LB" "$(fmt $LB)" "$LN"
printf "服务器 %-6s : %12d 字节  (%s)  %d 个文件\n" "$DST_DIR" "$RB" "$(fmt $RB)" "$RN"
if [ "$LB" = "$RB" ] && [ "$LN" = "$RN" ]; then
  echo "结果: [一致] 字节数完全相同，文件数相同"
else
  echo "结果: [不一致] 字节差 $((LB-RB))，文件数差 $((LN-RN))"
fi

echo
echo "=================== 按顶层目录核对 ==================="
breakdown() {
  awk -F'\t' '{split($2,a,"/"); d=(a[2]==""?"(根目录)":a[2]); s[d]+=$1; n[d]++}
              END{for(k in s) printf "  %-26s %14d 字节  %6d 文件\n", k, s[k], n[k]}' "$1" | sort
}
echo "本地 $SRC_WIN :"
breakdown /tmp/_loc.list
echo "服务器 $DST_DIR :"
breakdown /tmp/_rem.list
