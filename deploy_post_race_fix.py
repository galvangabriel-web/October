"""
Deploy Post-Race Analysis Fix to Production
"""

import paramiko
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

SSH_HOST = os.getenv('SSH_HOST', '200.58.107.214')
SSH_PORT = int(os.getenv('SSH_PORT', '5197'))
SSH_USER = os.getenv('SSH_USER', 'tactical')
SSH_PASSWORD = os.getenv('SSH_PASSWORD')

REMOTE_BASE = "/home/tactical/racing_analytics"

FILES = [
    ("src/models/inference/simple_post_race_predictor.py", f"{REMOTE_BASE}/src/models/inference/simple_post_race_predictor.py"),
    ("src/dashboard/post_race_widget.py", f"{REMOTE_BASE}/src/dashboard/post_race_widget.py"),
]

print("="*70)
print("DEPLOYING POST-RACE FIX")
print("="*70)

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(SSH_HOST, SSH_PORT, SSH_USER, SSH_PASSWORD)
sftp = ssh.open_sftp()

for local, remote in FILES:
    print(f"Uploading {local}...")
    sftp.put(local, remote)
    print(f"  OK")

ssh.exec_command(f"cd {REMOTE_BASE} && find src -name '__pycache__' -exec rm -rf {{}} + 2>/dev/null")
ssh.exec_command(f"pkill -f dashboard && cd {REMOTE_BASE} && nohup venv/bin/python src/dashboard/app.py > dashboard.log 2>&1 &")

sftp.close()
ssh.close()

print("="*70)
print("DEPLOYMENT COMPLETE")
print(f"Dashboard: http://{SSH_HOST}:8050")
print("="*70)
