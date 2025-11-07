#!/usr/bin/env python3
"""Quick upload of fixed chatbot_widget.py to server"""
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

# Upload chatbot_widget.py
sftp = client.open_sftp()
local_file = "src/dashboard/chatbot_widget.py"
remote_file = f"{deploy_path}/src/dashboard/chatbot_widget.py"

print(f"Uploading {local_file} to {remote_file}...")
sftp.put(local_file, remote_file)
print("Upload complete")

# Verify file size
stat = sftp.stat(remote_file)
print(f"Remote file size: {stat.st_size} bytes")

# Clear Python cache
print("Clearing Python cache...")
stdin, stdout, stderr = client.exec_command(
    f"find {deploy_path}/src -type d -name '__pycache__' -exec rm -rf {{}} + 2>/dev/null || true"
)
stdout.channel.recv_exit_status()

# Restart dashboard
print("Restarting dashboard...")
stdin, stdout, stderr = client.exec_command(
    f"pkill -f 'python.*src/dashboard/app.py' && sleep 2 && cd {deploy_path} && nohup venv/bin/python src/dashboard/app.py > dashboard.log 2>&1 &"
)
exit_code = stdout.channel.recv_exit_status()

print("Dashboard restart command sent")

# Check if it's running
print("Waiting 8 seconds for dashboard to start...")
import time
time.sleep(8)

stdin, stdout, stderr = client.exec_command("ps aux | grep '[p]ython.*src/dashboard/app.py'")
output = stdout.read().decode('utf-8')

if output:
    print("Dashboard is running:")
    print(output)

    # Check logs for chatbot messages
    print("\nChecking for chatbot initialization in logs...")
    stdin, stdout, stderr = client.exec_command(f"grep 'CHATBOT' {deploy_path}/dashboard.log | tail -10")
    log_output = stdout.read().decode('utf-8')
    if log_output:
        print(log_output)
    else:
        print("No chatbot messages found in logs yet")
else:
    print("WARNING: Dashboard may not have started. Checking logs...")
    stdin, stdout, stderr = client.exec_command(f"tail -30 {deploy_path}/dashboard.log")
    print(stdout.read().decode('utf-8'))

sftp.close()
client.close()

print("\nDone! Dashboard should be accessible at http://200.58.107.214:8050")
print("Try sending a chatbot message to test the fix.")
