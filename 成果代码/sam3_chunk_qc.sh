#!/bin/bash
# 检查已完成块的结果质量（地块数/面积分布），用于判断识别是否正常
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
/root/anaconda3/envs/sam3/bin/python - <<'PY'
import fiona, numpy as np
from shapely.geometry import shape
p = "/data3/0918my/my/510725zt/result_my10_v10_fix2_chunks/out/chunk_00.shp"
areas, scores = [], []
with fiona.open(p) as s:
    for f in s:
        areas.append(shape(f["geometry"]).area)
        scores.append(float(f["properties"].get("score") or 0))
a = np.array(areas) * 0.0015  # 亩
print("chunk_00: %d 个地块" % len(a))
print("  面积(亩) 分位: p5=%.4f p25=%.4f p50=%.4f p75=%.4f p95=%.2f max=%.1f sum=%.0f" % (
    *np.percentile(a, [5,25,50,75,95]), a.max(), a.sum()))
print("  score 分位: p5=%.3f p50=%.3f p95=%.3f" % tuple(np.percentile(scores, [5,50,95])))
print("  地块密度 = %.1f 个/km2（块核心 16128x19152 px @0.5m = %.1f km2）" % (
    len(a), 16128*0.5*19152*0.5/1e6))
# 对照：baozhucun fix2
try:
    b = "/data3/sam3_project/test_data/result_baozhucun_v10_fix2/baozhucun_v10_fix2.shp"
    with fiona.open(b) as s:
        ba = np.array([shape(f["geometry"]).area for f in s]) * 0.0015
    print("对照 baozhucun fix2: %d 个, 面积中位 %.4f 亩, 密度(影像 5716x5430@0.8m=19.9km2) %.0f 个/km2" % (
        len(ba), np.median(ba), len(ba)/19.9))
except Exception as e:
    print("对照读取失败:", e)
PY
EOF
