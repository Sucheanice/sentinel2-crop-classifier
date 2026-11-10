# -*- coding: utf-8 -*-
"""feature_lib_crypt.py — 特征库打包加密 / 解密 / 入库检查

为什么需要：特征库是本项目最核心的资产（211 MB 样本 + 57 KB 指纹），
只应留在本机；拷到服务器或演示机时用密文包，演示时临时解密、用完即删；
同时确保任何特征库文件都不会被 git 收进去。

用法：
    # 1) 加密打包（默认把 7 处库打成 成果数据\\_加密库\\特征库_vYYYYMMDD.flib）
    python feature_lib_crypt.py pack
    python feature_lib_crypt.py pack --out D:\\临时\\特征库.flib

    # 2) 在服务器/演示机上解密到临时目录
    python feature_lib_crypt.py unpack 特征库_v20261110.flib
    python feature_lib_crypt.py unpack 特征库_v20261110.flib --dest D:\\demo\\lib --install-demo

    # 3) 用完删除明文（只删 unpack 产生的目录）
    python feature_lib_crypt.py clean --dest D:\\demo\\lib

    # 4) 检查 git 是否会把这些文件纳入版本库（应当全部 IGNORED）
    python feature_lib_crypt.py verify

格式：magic(6) + salt(16) + nonce(12) + 分块 AES-256-GCM（每块 1 MiB，含 16 字节 tag）
口令经 scrypt(n=2^15, r=8, p=1) 派生 32 字节密钥；不落任何明文临时文件在目标目录。
"""
import argparse
import datetime as dt
import getpass
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

sys.stdout.reconfigure(encoding="utf-8")

ROOT = r"e:\工作相关\2026年\0624 待测试数据"
MAGIC = b"FLIB1\n"
CHUNK = 1 << 20

# 需要保护的东西（相对 ROOT）
ITEMS = [
    r"成果数据\作物特征库\自净",
    r"成果数据\作物特征库\指纹库",
    r"成果数据\柑橘蜜柚识别\自净",
    r"成果数据\柑橘蜜柚识别\指纹库",
    r"成果数据\特征库索引.csv",
    r"成果数据\演示\特征库\library.json",
    r"成果数据\演示\地块特征\blocks.json",
    r"成果数据\演示\统一村\blocks.json",
]
DEFAULT_OUT_DIR = os.path.join(ROOT, r"成果数据\_加密库")

# 服务器演示只需要这三份派生数据（图表 + 逐块判读），体积小、便于随包分发
ITEMS_DEMO = [
    r"成果数据\演示\特征库\library.json",
    r"成果数据\演示\地块特征\blocks.json",
    r"成果数据\演示\统一村\blocks.json",
]


def derive(password, salt):
    # OpenSSL 默认 32 MB 内存上限，n=2^15/r=8 需要约 33.5 MB，显式放宽到 128 MB
    return hashlib.scrypt(password.encode("utf-8"), salt=salt,
                          n=2 ** 15, r=8, p=1, dklen=32, maxmem=128 * 1024 * 1024)


def iter_files(items):
    for it in items:
        p = it if os.path.isabs(it) else os.path.join(ROOT, it)
        if os.path.isfile(p):
            yield p, os.path.relpath(p, ROOT)
        elif os.path.isdir(p):
            for base, _, files in os.walk(p):
                for f in files:
                    fp = os.path.join(base, f)
                    yield fp, os.path.relpath(fp, ROOT)


def get_pwd(args):
    if getattr(args, "password", None):
        return args.password
    p1 = getpass.getpass("设置口令（打包）/ 输入口令（解密）：")
    if args.cmd == "pack":
        p2 = getpass.getpass("再输一次确认：")
        if p1 != p2:
            raise SystemExit("两次口令不一致")
    if len(p1) < 8:
        raise SystemExit("口令至少 8 位")
    return p1


def cmd_pack(a):
    out = a.out or os.path.join(
        DEFAULT_OUT_DIR, "特征库_v%s.flib" % dt.date.today().strftime("%Y%m%d"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    items = ITEMS_DEMO if getattr(a, "only_demo", False) else ITEMS
    files = list(iter_files(items))
    if not files:
        raise SystemExit("没找到任何特征库文件，检查 ITEMS 路径")
    total = sum(os.path.getsize(f) for f, _ in files)
    print(f"打包 {len(files)} 个文件，共 {total/1024/1024:.1f} MB …")

    tmp_zip = tempfile.NamedTemporaryFile(suffix=".zip", delete=False).name
    try:
        with zipfile.ZipFile(tmp_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for fp, rel in files:
                z.write(fp, rel)
        zsize = os.path.getsize(tmp_zip)
        print(f"压缩后 {zsize/1024/1024:.1f} MB，开始加密 …")

        pwd = get_pwd(a)
        salt, nonce = os.urandom(16), os.urandom(12)
        key = derive(pwd, salt)
        aes = AESGCM(key)

        with open(tmp_zip, "rb") as fin, open(out, "wb") as fout:
            fout.write(MAGIC + salt + nonce)
            n = 0
            while True:
                buf = fin.read(CHUNK)
                if not buf:
                    break
                nonce_i = nonce[:8] + n.to_bytes(4, "big")
                ct = aes.encrypt(nonce_i, buf, None)
                fout.write(len(ct).to_bytes(4, "big"))
                fout.write(ct)
                n += 1
        print(f"[完成] {out}  {os.path.getsize(out)/1024/1024:.1f} MB（AES-256-GCM）")
        print("提示：口令不要写进脚本或配置文件；丢失口令无法恢复。")
    finally:
        if os.path.exists(tmp_zip):
            os.remove(tmp_zip)


def cmd_unpack(a):
    src = a.archive
    if not os.path.isabs(src):
        cand = os.path.join(DEFAULT_OUT_DIR, src)
        src = cand if os.path.exists(cand) else os.path.join(os.getcwd(), src)
    if not os.path.exists(src):
        raise SystemExit("找不到密文包: " + src)
    dest = a.dest or os.path.join(tempfile.gettempdir(), "flib_" + dt.datetime.now().strftime("%H%M%S"))
    os.makedirs(dest, exist_ok=True)

    pwd = get_pwd(a)
    tmp_zip = os.path.join(tempfile.gettempdir(), "flib_dec.zip")
    try:
        with open(src, "rb") as fin:
            if fin.read(6) != MAGIC:
                raise SystemExit("不是本工具生成的密文包")
            salt, nonce = fin.read(16), fin.read(12)
            aes = AESGCM(derive(pwd, salt))
            with open(tmp_zip, "wb") as fout:
                n = 0
                while True:
                    hdr = fin.read(4)
                    if len(hdr) < 4:
                        break
                    ln = int.from_bytes(hdr, "big")
                    ct = fin.read(ln)
                    fout.write(aes.decrypt(nonce[:8] + n.to_bytes(4, "big"), ct, None))
                    n += 1
    except Exception as e:
        if os.path.exists(tmp_zip):
            os.remove(tmp_zip)
        raise SystemExit("解密失败：口令错误或密文包被篡改（%s）" % type(e).__name__)
    try:
        with zipfile.ZipFile(tmp_zip) as z:
            z.extractall(dest)
        print(f"[完成] 已解密到 {dest}")
    finally:
        os.remove(tmp_zip)

    if a.install_demo:
        repo = os.path.join(ROOT, "_colleague_repo", "web", "data")
        if not os.path.isdir(repo):
            print("!! 未找到 _colleague_repo\\web\\data，跳过程序安装")
            return
        pairs = [
            (os.path.join(dest, r"成果数据\演示\特征库\library.json"), os.path.join(repo, "library.json")),
            (os.path.join(dest, r"成果数据\演示\地块特征\blocks.json"), os.path.join(repo, "features.json")),
        ]
        for s, t in pairs:
            if os.path.exists(s):
                shutil.copyfile(s, t)
                print("   装入演示 →", t)
        b = os.path.join(dest, r"成果数据\演示\统一村\blocks.json")
        bdst = os.path.join(repo, "village_b", "features.json")
        if os.path.exists(b) and os.path.isdir(os.path.dirname(bdst)):
            shutil.copyfile(b, bdst)
            print("   装入演示 →", bdst)
    print("提醒：演示结束执行 clean 删除明文目录。")


def cmd_clean(a):
    dest = a.dest
    if not dest:
        raise SystemExit("必须指定 --dest（只删你指定的解密目录）")
    if not os.path.isdir(dest):
        print("目录不存在:", dest)
        return
    shutil.rmtree(dest)
    print("[已删除]", dest)


def cmd_verify(a):
    bad = 0
    for it in ITEMS:
        p = it if os.path.isabs(it) else os.path.join(ROOT, it)
        targets = [os.path.relpath(f, ROOT).replace("\\", "/") for f, _ in iter_files([p])][:3]
        for t in targets:
            r = subprocess.run(["git", "check-ignore", "-q", t], cwd=ROOT)
            flag = "IGNORED   " if r.returncode == 0 else "!! TRACKABLE"
            if r.returncode != 0:
                bad += 1
            print(f"{flag} {t}")
    print("\n结论:", "全部被 git 忽略，安全" if bad == 0 else f"有 {bad} 个文件可能被提交，请补 .gitignore")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("pack"); p1.add_argument("--out"); p1.add_argument("--password")
    p1.add_argument("--only-demo", action="store_true", help="只打包演示用的 3 个 json（约 3 MB）")
    p2 = sub.add_parser("unpack"); p2.add_argument("archive"); p2.add_argument("--dest")
    p2.add_argument("--install-demo", action="store_true"); p2.add_argument("--password")
    p3 = sub.add_parser("clean"); p3.add_argument("--dest", required=True)
    sub.add_parser("verify")
    a = ap.parse_args()
    {"pack": cmd_pack, "unpack": cmd_unpack, "clean": cmd_clean, "verify": cmd_verify}[a.cmd](a)


if __name__ == "__main__":
    main()
