# -*- coding: utf-8 -*-
"""临时：从服务器下载文件（用完即删）。"""
import sys
import paramiko

HOST = "10.62.11.15"
USER = "root"
PWD = "Ccyy123@"

remote = sys.argv[1]
local = sys.argv[2]

t = paramiko.Transport((HOST, 22))
t.connect(username=USER, password=PWD)
sftp = paramiko.SFTPClient.from_transport(t)
sftp.get(remote, local)
sftp.close()
t.close()
print("downloaded", remote, "->", local)
