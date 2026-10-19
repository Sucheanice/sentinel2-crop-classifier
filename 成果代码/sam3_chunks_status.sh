#!/bin/bash
# 分块任务进度速查
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
R=/data3/0918my/my/510725zt
echo "=== 驱动日志（末 12 行）==="
tail -12 $R/run_chunks.log 2>/dev/null
echo
echo "=== 各块状态 ==="
for f in $R/result_my10_v10_fix2_chunks/out/chunk_*.shp; do
  [ -f "$f" ] && echo "  $(basename $f) 有 ($(stat -c %y "$f" | cut -d. -f1))" || true
done
ls $R/result_my10_v10_fix2_chunks/tif/*.tif 2>/dev/null | wc -l | xargs echo "  已切块数:"
echo
echo "=== 当前块进度（含 SLIC 行数）==="
cur=$(ls -t $R/result_my10_v10_fix2_chunks/chunk_*.log 2>/dev/null | head -1)
[ -n "$cur" ] && echo "  当前: $(basename $cur)  SLIC 行=$(grep -c '\[SLIC\]' $cur)  末行: $(tail -c 200 $cur | tr '\r' '\n' | tail -1)"
echo
echo "=== 资源 ==="
free -g | head -2
ps -eo pid,rss,etime,cmd --sort=-rss | grep -E '[s]pawn_main' | head -5 | awk '{printf "  worker RSS=%7.2f GB 已运行=%s\n", $3/1048576, $4}'
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
EOF
