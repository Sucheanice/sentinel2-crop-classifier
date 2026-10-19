#!/bin/bash
# 上传完整性校验（只读，不改动任何文件）
#   1) 全量比对 文件大小 + 修改时间（rsync --dry-run --itemize-changes，出空即完全一致）
#   2) 抽查若干文件做 sha256 一致性比对（含大文件）
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
SSHOPT="-o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -o ServerAliveInterval=30"
SRC_WSL=$(printf '%s' "$SRC_WIN" | sed -E 's|^([A-Za-z]):|/mnt/\L\1|')

echo "########## 步骤 1：全量比对大小与修改时间 ##########"
OUT=$(rsync -ain --out-format='DIFF|%n|%l|%M' -e "ssh $SSHOPT" "$SRC_WSL/" "$RUSER@$HOST:$DST_DIR/" 2>&1)
if [ -z "$OUT" ]; then
  echo "  [通过] 两侧文件大小与修改时间完全一致，无差异项"
else
  echo "  [有差异] 以下文件不一致："
  printf '%s\n' "$OUT" | head -40
  echo "  差异项数: $(printf '%s\n' "$OUT" | grep -c 'DIFF|')"
fi

echo
echo "########## 步骤 2：sha256 抽查 ##########"
cd "$SRC_WSL" || exit 1
FILES=$(find . -type f -printf '%s\t%p\n' | sort -rn | awk -F'\t' '
  {print $2}
  END{}' | awk 'NR==1 || NR==3 || /\.ovr$/ || /\.md$/ || /\.dbf$/ {print; c++} c>=6 {exit}')
echo "  抽查文件:"; printf '    %s\n' $FILES
LOC=$(sha256sum $FILES | awk '{print $1" "$2}')
REM=$(ssh $SSHOPT "$RUSER@$HOST" "cd '$DST_DIR' && sha256sum $(printf '%q ' $FILES)" 2>&1 | awk '{print $1" "$2}')
printf '%s\n' "$LOC" | sort -k2 > /tmp/_l.sha
printf '%s\n' "$REM" | sort -k2 > /tmp/_r.sha
if diff -q /tmp/_l.sha /tmp/_r.sha >/dev/null; then
  echo "  [通过] $(printf '%s\n' "$LOC" | wc -l) 个抽查文件 sha256 完全一致"
else
  echo "  [不一致] 本地 vs 远端 sha256："
  diff /tmp/_l.sha /tmp/_r.sha | head -20
fi

echo
echo "########## 步骤 3：远端残留检查 ##########"
ssh $SSHOPT "$RUSER@$HOST" "cd '$DST_DIR' && echo '  临时/分片残留:'; find . \( -name '.*.??????' -o -name '.rsync-partial' \) -print | head -10; echo '  目录数: '\$(find . -type d | wc -l)'; echo '  文件数: '\$(find . -type f | wc -l)" 2>&1
