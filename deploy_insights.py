#!/usr/bin/env python3
"""Deploy src/insights directory to production server"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import paramiko

# Fix Windows console encoding
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Load .env
env_path = Path(__file__).parent / '.env'
load_dotenv(env_path)

# Get credentials
ssh_host = os.getenv('SSH_HOST')
ssh_port = int(os.getenv('SSH_PORT', 22))
ssh_user = os.getenv('SSH_USER')
ssh_password = os.getenv('SSH_PASSWORD')
deploy_path = os.getenv('DEPLOY_PATH')

print(f"Connecting to {ssh_user}@{ssh_host}:{ssh_port}...")

# Connect
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(
    hostname=ssh_host,
    port=ssh_port,
    username=ssh_user,
    password=ssh_password
)

print("Connected successfully")

# Create remote directory if it doesn't exist
sftp = client.open_sftp()
remote_insights_dir = f"{deploy_path}/src/insights"

print(f"Ensuring remote directory exists: {remote_insights_dir}...")
try:
    sftp.stat(remote_insights_dir)
    print("Directory exists")
except FileNotFoundError:
    print("Creating directory...")
    stdin, stdout, stderr = client.exec_command(f"mkdir -p {remote_insights_dir}")
    stdout.channel.recv_exit_status()

# Get all Python files in src/insights
local_insights_dir = Path("src/insights")
py_files = list(local_insights_dir.glob("*.py"))

print(f"\nFound {len(py_files)} Python files to upload:")
for f in py_files:
    print(f"  - {f.name}")

# Upload each file
print("\nUploading files...")
for local_file in py_files:
    remote_file = f"{remote_insights_dir}/{local_file.name}"
    print(f"  Uploading {local_file.name}...", end=" ")
    try:
        sftp.put(str(local_file), remote_file)
        print("✓")
    except Exception as e:
        print(f"✗ Error: {e}")

print("\nAll files uploaded successfully!")

# Restart API
print("\nRestarting API...")
stdin, stdout, stderr = client.exec_command(
    f"pkill -f 'uvicorn.*api' && sleep 2 && cd {deploy_path} && nohup venv/bin/python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 1 --timeout-keep-alive 30 > api.log 2>&1 &"
)
exit_code = stdout.channel.recv_exit_status()

print("API restart command sent")

# Check if it's running
print("Waiting 5 seconds for API to start...")
import time
time.sleep(5)

stdin, stdout, stderr = client.exec_command("ps aux | grep '[u]vicorn'")
output = stdout.read().decode('utf-8')

if output:
    print("✓ API is running:")
    print(output)
else:
    print("WARNING: API may not have started. Checking logs...")
    stdin, stdout, stderr = client.exec_command(f"tail -30 {deploy_path}/api.log")
    print(stdout.read().decode('utf-8'))

sftp.close()
client.close()

print("\nDone! API should be accessible at http://200.58.107.214:8000")
print("Test feature extraction at: http://200.58.107.214:8050")
