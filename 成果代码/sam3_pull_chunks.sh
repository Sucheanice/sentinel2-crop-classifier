#!/bin/bash
# 把服务器上已完成的块成果下载到本地（供 QGIS 查看）
set -u
HOST="${HOST:-10.62.11.15}"
RUSER="${RUSER:-root}"
AP="$HOME/.askpass_srv"
MANUAL="/mnt/e/工作相关/2026年/0624 待测试数据/文档/10.62.11.15_服务器恢复手册.md"
DST="/mnt/e/工作相关/2026年/0624 待测试数据/成果数据/梓潼my10_SAM3_v10fix2"
if [ ! -x "$AP" ]; then
  PW=$(grep -m1 -E '^# *密码' "$MANUAL" 2>/dev/null | sed -E 's/^# *密码 *[:：] *//' | tr -d '\r\n')
  printf '#!/bin/sh\nprintf "%%s\\n" %q\n' "$PW" > "$AP"; chmod 700 "$AP"
fi
export SSH_ASKPASS="$AP" SSH_ASKPASS_REQUIRE=force DISPLAY=:0
SSHOPT="-o StrictHostKeyChecking=accept-new -o ConnectTimeout=20"
mkdir -p "$DST"

echo "=== 服务器上已完成的块 ==="
ssh $SSHOPT "$RUSER@$HOST" "ls -l /data3/0918my/my/510725zt/result_my10_v10_fix2_chunks/out/*.shp 2>/dev/null"

for f in $(ssh $SSHOPT "$RUSER@$HOST" "ls /data3/0918my/my/510725zt/result_my10_v10_fix2_chunks/out/*.shp 2>/dev/null"); do
  b=$(basename "$f" .shp)
  echo "--- 下载 $b ---"
  for ext in shp shx dbf prj cpg; do
    scp $SSHOPT "$RUSER@$HOST:$f" "$DST/$b.$ext" 2>/dev/null || true
    scp $SSHOPT "$RUSER@$HOST:${f%.shp}.$ext" "$DST/$b.$ext" 2>/dev/null || true
  done
done
echo
echo "=== 本地目录 ==="
ls -lh "$DST"
