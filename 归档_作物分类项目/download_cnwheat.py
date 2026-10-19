# -*- coding: utf-8 -*-
"""CN-Wheat10_2024.rar 断点续传下载"""
import os
import requests

BASE = r'e:\工作相关\2026年\0624 待测试数据'
URL = 'https://ndownloader.figshare.com/files/53955743'
PATH = os.path.join(BASE, 'CN-Wheat10_2024.rar')


def main():
    existing = os.path.getsize(PATH) if os.path.exists(PATH) else 0
    print(f'已有 {existing/1e6:.1f} MB，从断点续传')

    headers = {}
    if existing > 0:
        headers['Range'] = f'bytes={existing}-'

    with requests.get(URL, stream=True, headers=headers, timeout=120) as r:
        print('status', r.status_code)
        if r.status_code in (200, 206):
            total = int(r.headers.get('content-length', 0)) + (existing if r.status_code == 206 else 0)
            mode = 'ab' if r.status_code == 206 else 'wb'
            with open(PATH, mode) as f:
                done = existing
                for chunk in r.iter_content(1024 * 256):
                    f.write(chunk)
                    done += len(chunk)
                    if total > 0 and done % (20 * 1024 * 1024) < 256 * 1024:
                        print(f'  {done/1e6:.1f} / {total/1e6:.1f} MB ({done/total*100:.0f}%)', flush=True)
            print('DONE', round(done / 1e6, 1), 'MB')
        else:
            print('下载失败', r.status_code, r.text[:200])


if __name__ == '__main__':
    main()
