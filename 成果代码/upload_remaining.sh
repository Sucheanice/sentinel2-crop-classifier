#!/bin/bash
# 列出远端尚未传完的文件（本地有、远端无或大小不一致）
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
SRC_WSL=$(printf '%s' "$SRC_WIN" | sed -E 's|^([A-Za-z]):|/mnt/\L\1|')

cd "$SRC_WSL" || exit 1
find . -type f -printf '%s\t%p\n' | sort -k2 > /tmp/_loc.txt
ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 "$RUSER@$HOST" \
    "cd '$DST_DIR' && find . -type f -not -path './.rsync-partial/*' -printf '%s\t%p\n' | sort -k2" > /tmp/_rem.txt

echo "=== 尚未传完的文件（按大小降序）==="
awk -F'\t' 'NR==FNR{r[$2]=$1; next} {if ($2 in r) {if (r[$2] != $1) print "  差异 " $1 " <- " r[$2] "  " $2; delete r[$2]} else print "  new  " $1 "  " $2} END {for (p in r) print "  缺   " r[p] "  " p}' /tmp/_rem.txt /tmp/_loc.txt \
  | sort -k3 -rn | head -30
echo "=== 未完成合计 ==="
awk -F'\t' 'NR==FNR{r[$2]=$1; next} {if ($2 in r) {if (r[$2] != $1) {s+=r[$2]; n++} delete r[$2]} else {s+=$1; n++}} END {for (p in r) {s+=r[p]; n++}; printf "  %d 个文件 / %.2f GB\n", n, s/1073741824}' /tmp/_rem.txt /tmp/_loc.txt
