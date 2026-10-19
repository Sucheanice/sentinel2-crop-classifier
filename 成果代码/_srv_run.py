# -*- coding: utf-8 -*-
"""临时：SSH 到服务器执行命令（用完即删）。"""
import sys
import paramiko

HOST = "10.62.11.15"
USER = "root"
PWD = "Ccyy123@"

cmd = sys.argv[1]
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, username=USER, password=PWD, timeout=15)
_, stdout, stderr = c.exec_command(cmd)
print(stdout.read().decode("utf-8", "replace"))
err = stderr.read().decode("utf-8", "replace")
if err.strip():
    print("[stderr]", err, file=sys.stderr)
c.close()
