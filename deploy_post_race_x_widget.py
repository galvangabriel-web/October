#!/usr/bin/env python3
"""
Deploy post_race_x_widget.py to production server
"""

import paramiko
import os
from pathlib import Path
from dotenv import load_dotenv

# Load credentials
load_dotenv()

HOST = os.getenv('SSH_HOST')
PORT = int(os.getenv('SSH_PORT', 22))
USERNAME = os.getenv('SSH_USER')
PASSWORD = os.getenv('SSH_PASSWORD')
REMOTE_BASE = '/home/tactical/racing_analytics'

# File to upload
LOCAL_FILE = Path('src/dashboard/post_race_x_widget.py')
REMOTE_FILE = f'{REMOTE_BASE}/src/dashboard/post_race_x_widget.py'

print("=" * 80)
print("DEPLOYING POST-RACE-X WIDGET TO PRODUCTION")
print("=" * 80)

# Connect via SSH
print(f"\nConnecting to {HOST}:{PORT}...")
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, port=PORT, username=USERNAME, password=PASSWORD)

# Upload file
print(f"\nUploading {LOCAL_FILE}...")
sftp = ssh.open_sftp()
sftp.put(str(LOCAL_FILE), REMOTE_FILE)
sftp.close()

# Get file info
stdin, stdout, stderr = ssh.exec_command(f'ls -lh {REMOTE_FILE}')
print(f"Uploaded: {stdout.read().decode().strip()}")

# Clear Python cache
print("\nClearing Python cache...")
ssh.exec_command(f'cd {REMOTE_BASE} && find src/dashboard -type d -name "__pycache__" -exec rm -rf {{}} + 2>/dev/null || true')

# Restart dashboard
print("\nRestarting dashboard...")
ssh.exec_command(f'pkill -f "python.*src/dashboard/app.py"')

import time
time.sleep(2)

restart_cmd = f'cd {REMOTE_BASE} && nohup venv/bin/python src/dashboard/app.py > dashboard.log 2>&1 &'
ssh.exec_command(restart_cmd)

time.sleep(3)

# Verify dashboard is running
stdin, stdout, stderr = ssh.exec_command('ps aux | grep "python.*src/dashboard/app.py" | grep -v grep')
processes = stdout.read().decode().strip()

if processes:
    print("\nDashboard restarted successfully!")
    print(f"Processes: {len(processes.splitlines())}")
else:
    print("\nWARNING: Dashboard may not be running!")

# Check for recent errors
print("\nChecking logs for errors...")
stdin, stdout, stderr = ssh.exec_command(f'tail -20 {REMOTE_BASE}/dashboard.log | grep -i "error\\|exception\\|traceback" || echo "No errors found"')
log_check = stdout.read().decode().strip()
print(log_check)

ssh.close()

print("\n" + "=" * 80)
print("DEPLOYMENT COMPLETE!")
print("=" * 80)
print("\nNext steps:")
print("1. Open browser: http://200.58.107.214:8050/")
print("2. Go to Post-Race-X Analysis tab")
print("3. Select a track and vehicle")
print("4. Verify Pro Tip is now personalized and changes per vehicle")
print("=" * 80)
