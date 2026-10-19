#!/bin/bash
# 二查：找推理代码、看 baozhucun 那次真实耗时、看显卡占用者、看影像波段
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
echo "=========== 1. 找 geo_infer 系列代码 ==========="
find /data3 /data4 /root -maxdepth 6 -name 'geo_infer*.py' -printf '%s\t%T@\t%p\n' 2>/dev/null | sort -k3
echo "--- sam3_service 目录 ---"
ls -l /data3/sam3_project/sam3_service/ 2>/dev/null | head -40

echo
echo "=========== 2. 影像波段/分辨率 ==========="
gdalinfo /data3/0918my/my/510725zt/my10.tif 2>/dev/null | grep -E 'Band |Type=|Pixel Size|NoData|Size is|Origin'

echo
echo "=========== 3. baozhucun fix2 那次运行日志 ==========="
head -25 /data3/sam3_project/test_data/result_baozhucun_v10_fix2/run.log 2>/dev/null
echo "  ...(尾部)"
tail -25 /data3/sam3_project/test_data/result_baozhucun_v10_fix2/run.log 2>/dev/null

echo
echo "=========== 4. 谁占着显存 ==========="
nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv 2>&1 | head -20
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader 2>&1

echo
echo "=========== 5. conda sam3 环境 ==========="
ls -d /root/miniconda3/envs/* 2>/dev/null
for p in /root/miniconda3/envs/sam3/bin/python /root/miniconda3/bin/python; do
  [ -x "$p" ] && echo "$p : $($p -c 'import sys,torch;print(sys.version.split()[0], "torch", torch.__version__, "cuda", torch.cuda.is_available())' 2>&1 | tail -1)"
done
EOF
