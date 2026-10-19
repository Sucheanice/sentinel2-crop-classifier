# -*- coding: utf-8 -*-
"""临时：上传文件到服务器（用完即删）。"""
import sys
import paramiko

HOST = "10.62.11.15"
USER = "root"
PWD = "Ccyy123@"

local = sys.argv[1]
remote = sys.argv[2]

t = paramiko.Transport((HOST, 22))
t.connect(username=USER, password=PWD)
sftp = paramiko.SFTPClient.from_transport(t)
sftp.put(local, remote)
sftp.close()
t.close()
print("uploaded", local, "->", remote)
