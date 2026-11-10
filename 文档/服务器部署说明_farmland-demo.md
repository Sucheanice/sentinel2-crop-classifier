# 耕地智能识别演示 · 服务器部署说明

> 服务器：`10.62.11.15`　部署目录：`/data4/farmland-demo`
> 访问地址：**http://10.62.11.15:8770/**（特征库页：`/library.html`）
> 部署时间：2026-11-10　适用数据：宝珠村（江油方水镇）、统一村（江油河口镇）

---

## 1. 快速使用

```bash
cd /data4/farmland-demo

./start.sh     # 启动（若特征库未装入，会先提示输入口令解锁）
./lock.sh      # 停止服务并清除特征库明文（演示结束务必执行）
```

启动后本机与局域网都可访问 `http://10.62.11.15:8770/`。
特征库口令存在**本机**（演示电脑）`成果数据\_加密库\口令_勿删.txt`，服务器上不保存口令。

---

## 2. 目录结构

```
/data4/farmland-demo
├── app/                     # 程序（同事 farmland-demo 前端 + prepare 预处理）
│   ├── prepare.py           # 影像金字塔瓦片 + 矢量转像素坐标
│   ├── render_rgb.py        # 16bit → 8bit RGB 拉伸
│   ├── config.json          # 两个村的数据路径（本机已改成本服务器路径）
│   └── web/                 # 前端（index.html / library.html / app.js / library.js）
│       ├── vendor/echarts.min.js
│       └── data/            # prepare 生成的显示数据（约 49 MB，可重建）
│           ├── tiles_rgb_v2/…        宝珠村瓦片
│           ├── vectors.json / attrs.json / polyFeature.json
│           ├── library.json          ← 特征库（解锁后才有）
│           ├── features.json         ← 地块 98 维判据（解锁后才有）
│           └── village_b/…           统一村瓦片与元数据
├── data/
│   ├── imagery/baozhu.tif   # 68 MB，0.8 m，4 波段（服务器裁剪）
│   ├── imagery/tongyi.tif   # 81 MB，0.5 m，3 波段（服务器裁剪）
│   ├── vector/              # 两个村的赋值矢量（shp 全套）
│   └── *_cut.geojson        # 村界，用于从原始影像裁图
├── secure/demo_lib.flib     # 加密特征库（AES-256-GCM，1.2 MB）
├── tools/feature_lib_crypt.py  # 加解密工具
├── serve_linux.py           # 静态服务（0.0.0.0:8770）
├── start.sh / unlock.sh / lock.sh
├── logs/serve.log           # 运行日志
└── run/serve.pid            # 进程号
```

---

## 3. 数据来源

| 内容 | 来源 | 说明 |
|---|---|---|
| 影像 | 服务器 `/data3/0918my/my/510781jy/my1.tif`（宝珠）、`my9.tif`（统一） | 原始整幅 11.8 GB / 2.76 GB，已用村界 `gdalwarp -cutline` 裁村 |
| 赋值矢量 | 江油 `方水镇_交付.shp`（4,137 块）、河口镇交付件按统一村村界裁剪（915 块） | 字段：`XH / SJMJ / LAND_BZMC / CROP_BZMC / FRT_BZMC` |
| 特征库 | 7 套作物库的派生数据（`library.json` + 两份 `blocks.json`） | 加密存放，见第 4 节 |

---

## 4. 特征库加密

特征库是本项目核心资产（原始样本 313 MB、指纹 57 KB），**只保留在演示电脑**；
服务器上只有密文 `secure/demo_lib.flib`，演示时临时解密，结束即清除。

**解锁**（`./start.sh` 会自动调用）：

```bash
./unlock.sh
# 输入口令 → 解密到 runtime/ → 复制 3 个 json 到 app/web/data/ → 删除 runtime
```

**锁定**：

```bash
./lock.sh
# 停服务 + 删除 app/web/data/{library.json,features.json,village_b/features.json}
```

**重新打包**（换库或改口令时，在演示电脑上执行）：

```bash
python 成果代码\feature_lib_crypt.py pack --only-demo
# 输出 成果数据\_加密库\demo_lib.flib（约 1.2 MB），再上传到 secure/ 覆盖
```

**安全约束**：

- 口令不要写进脚本、`config.json`、日志或 git；服务器上不留口令文件
- 忘了口令无法恢复，重新 pack 即可
- 密文被改动一个字节就解不开（GCM tag 校验），错误口令会直接报 `InvalidTag`

---

## 5. 换数据 / 重建

```bash
# 1) 替换 app/config.json 里的 image / vector 路径（或放新文件到 data/）
# 2) 重新生成显示数据（约 1~3 分钟）
cd /data4/farmland-demo/app && python3 prepare.py
# 3) 重启
cd .. && ./lock.sh && ./start.sh
```

若新增第三个村，需要同步修改：
`prepare.py` 的 `main()` 里 `[('primary',''),('village_b','village_b')]`、
前端 `app.js` 的 `datasets` 映射与下拉项。

---

## 6. 端口与进程

- 监听 `0.0.0.0:8770`（8765 已被 docker-proxy 占用，故改用 8770）
- 查看进程：`pgrep -af serve_linux.py`
- 查看日志：`tail -f /data4/farmland-demo/logs/serve.log`
- 改端口：编辑 `serve_linux.py` 最后一行的 `8770`，再 `./lock.sh && ./start.sh`

---

## 7. 排障

| 现象 | 处理 |
|---|---|
| 页面空白 / 一直"正在准备本地数据" | 查 `logs/serve.log`；确认 `app/web/data/metadata.json` 存在，否则重跑 `prepare.py` |
| 特征库页图表为空 | 未执行 `unlock.sh`，或口令错误；`ls app/web/data/library.json` 确认 |
| 地图像素模糊 | 放大超过影像原生分辨率（宝珠 0.8 m / 统一 0.5 m），属正常插值 |
| 地块悬空、周围大片浅灰蓝 | 村界裁剪后的影像外围，非加载失败 |
| 端口被占 | `ss -ltnp | grep 8770`，或改端口后重启 |

---

## 8. 合规提醒

- 本目录含业务数据，**不要提交到 GitHub**；演示电脑上的仓库已用 `.gitignore` 排除
  （`成果数据/`、`*.flib`、`web/data/` 等）
- 演示结束执行 `./lock.sh`，并确认 `app/web/data/` 下没有 `library.json`、`features.json`
- 页面不依赖在线底图、CDN 或外部字体，断网可用（仅限局域网访问本机服务）
