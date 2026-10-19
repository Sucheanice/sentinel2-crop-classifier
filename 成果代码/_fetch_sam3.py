# -*- coding: utf-8 -*-
"""临时脚本：从服务器拉取 SAM3 源码 + 模型权重到本地（用完即删）。

用法：
  python _fetch_sam3.py --password <pwd> probe
  python _fetch_sam3.py --password <pwd> code --local <本地目录>
  python _fetch_sam3.py --password <pwd> weights --local <本地目录>
"""
import argparse
import os
import stat
import paramiko

HOST = '10.62.11.15'
USER = 'root'
REMOTE_CODE = '/data4/sam3_project/sam3_train/Code/sam3-main'
REMOTE_CKPT = '/data4/sam3_project/sam3_train/data/runs/finetune_v3/checkpoints/checkpoint_25.pt'
REMOTE_SAM3_PT = '/data4/sam3_project/sam3_train/Code/sam3-main/model/sam3.pt'


def connect(pwd):
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, username=USER, password=pwd, timeout=30)
    return c


def run(c, cmd):
    _, out, err = c.exec_command(cmd, timeout=600)
    o = out.read().decode('utf-8', 'replace')
    e = err.read().decode('utf-8', 'replace')
    return o, e


def probe(pwd):
    c = connect(pwd)
    cmds = [
        'grep -rl "sam3_backend" ' + REMOTE_CODE + '/sam3/ | head -20',
        'find ' + REMOTE_CODE + '/sam3_service -maxdepth 1 -name "*.py" | head -80',
        'sed -n "1,60p" ' + REMOTE_CODE + '/sam3/model_builder.py',
        'sed -n "1,40p" ' + REMOTE_CODE + '/sam3/model/sam3_image_processor.py',
    ]
    for cmd in cmds:
        print('$ ' + cmd)
        o, e = run(c, cmd)
        if o.strip():
            print(o)
        if e.strip():
            print('[stderr]', e)
        print('-' * 60)
    c.close()


def _walk(sftp, remote_dir, local_dir, excludes):
    os.makedirs(local_dir, exist_ok=True)
    for entry in sftp.listdir_attr(remote_dir):
        rp = remote_dir + '/' + entry.filename
        lp = os.path.join(local_dir, entry.filename)
        if any(entry.filename == x for x in excludes):
            print('skip', rp)
            continue
        if stat.S_ISDIR(entry.st_mode):
            _walk(sftp, rp, lp, excludes)
        else:
            os.makedirs(os.path.dirname(lp), exist_ok=True)
            print('download', rp, f'({entry.st_size/1e6:.1f}MB)')
            sftp.get(rp, lp)


CODE_ITEMS = [
    ('sam3', {'__pycache__'}),
    ('sam3_backend', {'__pycache__'}),
    ('model', {'sam3.pt', 'sam3.1_multiplex.pt'}),
]


def code(pwd, local):
    c = connect(pwd)
    sftp = c.open_sftp()
    for rel, excludes in CODE_ITEMS:
        _walk(sftp, REMOTE_CODE + '/' + rel, os.path.join(local, rel), excludes)
    sftp.close()
    c.close()
    print('[done] code ->', local)


def weights(pwd, local):
    os.makedirs(local, exist_ok=True)
    c = connect(pwd)
    sftp = c.open_sftp()
    for rp in [REMOTE_CKPT, REMOTE_SAM3_PT]:
        lp = os.path.join(local, os.path.basename(rp))
        st = sftp.stat(rp)
        print('download', rp, f'({st.st_size/1e6:.1f}MB)')
        sftp.get(rp, lp)
    sftp.close()
    c.close()
    print('[done] weights ->', local)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--password', required=True)
    ap.add_argument('--local', default=r'e:\工作相关\2026年\0624 待测试数据\sam3_win')
    ap.add_argument('mode', choices=['probe', 'code', 'weights'])
    args = ap.parse_args()

    if args.mode == 'probe':
        probe(args.password)
    elif args.mode == 'code':
        code(args.password, args.local)
    elif args.mode == 'weights':
        weights(args.password, args.local)
