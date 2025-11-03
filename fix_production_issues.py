#!/usr/bin/env python3
"""Fix production server issues: upload insights, clear cache, restart services"""
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

print("="*60)
print("PRODUCTION SERVER FIX")
print("="*60)

# Connect
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
try:
    client.connect(
        hostname=os.getenv('SSH_HOST'),
        port=int(os.getenv('SSH_PORT', 22)),
        username=os.getenv('SSH_USER'),
        password=os.getenv('SSH_PASSWORD'),
        timeout=10
    )
    print(f"✓ Connected to {os.getenv('SSH_HOST')}")
except Exception as e:
    print(f"✗ Connection failed: {e}")
    sys.exit(1)

sftp = client.open_sftp()
remote_base = "/home/tactical/racing_analytics"

# 1. Upload all src/insights files
print("\n[1/5] Uploading src/insights files...")
local_insights = Path("src/insights")
if local_insights.exists():
    py_files = list(local_insights.glob("*.py"))
    print(f"Found {len(py_files)} Python files to upload")

    for local_file in py_files:
        remote_file = f"{remote_base}/src/insights/{local_file.name}"
        try:
            sftp.put(str(local_file), remote_file)
            print(f"  ✓ {local_file.name}")
        except Exception as e:
            print(f"  ✗ {local_file.name}: {e}")
else:
    print("  ✗ src/insights directory not found locally")

# 2. Clear Python cache
print("\n[2/5] Clearing Python cache...")
cmd = f"cd {remote_base} && find src -type d -name '__pycache__' -exec rm -rf {{}} + 2>/dev/null || true"
stdin, stdout, stderr = client.exec_command(cmd)
stdout.read()
print("  ✓ Cache cleared")

# 3. Kill old API processes
print("\n[3/5] Stopping old API processes...")
cmd = "pkill -f 'uvicorn.*src.api.main'"
stdin, stdout, stderr = client.exec_command(cmd)
stdout.read()
print("  ✓ Old processes stopped")

# Wait a moment
import time
time.sleep(2)

# 4. Start new API
print("\n[4/5] Starting API server...")
cmd = f"cd {remote_base} && nohup venv/bin/python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 1 --timeout-keep-alive 30 > api.log 2>&1 &"
stdin, stdout, stderr = client.exec_command(cmd)
stdout.read()
print("  ✓ API started")

# Wait for startup
time.sleep(3)

# 5. Verify API is running
print("\n[5/5] Verifying API status...")
cmd = "ps aux | grep '[u]vicorn.*src.api.main'"
stdin, stdout, stderr = client.exec_command(cmd)
result = stdout.read().decode('utf-8')
if result.strip():
    print("  ✓ API is running")
    print(f"  Process: {result.strip()[:100]}...")
else:
    print("  ✗ API not found, checking logs...")
    cmd = f"cd {remote_base} && tail -20 api.log"
    stdin, stdout, stderr = client.exec_command(cmd)
    log_output = stdout.read().decode('utf-8', errors='replace')
    print("\n  Last 20 lines of api.log:")
    print(log_output)

# 6. Start dashboard if not running
print("\n[6/6] Checking dashboard status...")
cmd = "ps aux | grep '[p]ython.*dashboard/app.py'"
stdin, stdout, stderr = client.exec_command(cmd)
result = stdout.read().decode('utf-8')
if result.strip():
    print("  ✓ Dashboard is already running")
else:
    print("  ! Dashboard not running, starting it...")
    cmd = f"cd {remote_base} && nohup venv/bin/python src/dashboard/app.py > dashboard.log 2>&1 &"
    stdin, stdout, stderr = client.exec_command(cmd)
    stdout.read()
    print("  ✓ Dashboard started")

sftp.close()
client.close()

print("\n" + "="*60)
print("FIX COMPLETE")
print("="*60)
print(f"Dashboard: http://{os.getenv('SSH_HOST')}:8050")
print(f"API: http://{os.getenv('SSH_HOST')}:8000")
print("\nNext steps:")
print("1. Test dashboard at the URL above")
print("2. If issues persist, check logs:")
print(f"   python ssh_helper.py \"tail -50 {remote_base}/api.log\"")
print(f"   python ssh_helper.py \"tail -50 {remote_base}/dashboard.log\"")
