#!/bin/bash
# 把分块驱动脚本传到服务器并后台启动（脱离终端）
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
SSHOPT="-o StrictHostKeyChecking=accept-new -o ConnectTimeout=20"

cp "/mnt/e/工作相关/2026年/0624 待测试数据/成果代码/run_chunks_my10.py" "$HOME/run_chunks_my10.py"
scp $SSHOPT "$HOME/run_chunks_my10.py" "$RUSER@$HOST:/data3/0918my/my/510725zt/run_chunks_my10.py"
echo "[已上传] /data3/0918my/my/510725zt/run_chunks_my10.py"

ssh $SSHOPT "$RUSER@$HOST" 'bash -s' <<'EOF'
set -u
cd /data3/0918my/my/510725zt
pkill -f run_chunks_my10.py 2>/dev/null
sleep 1
setsid nohup /root/anaconda3/envs/sam3/bin/python run_chunks_my10.py \
    > /data3/0918my/my/510725zt/run_chunks.log 2>&1 &
sleep 8
echo "[已后台启动] PID=$(pgrep -f run_chunks_my10.py | head -1)"
echo "--- 日志 ---"
cat /data3/0918my/my/510725zt/run_chunks.log
EOF
