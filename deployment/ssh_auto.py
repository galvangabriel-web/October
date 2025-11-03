#!/usr/bin/env python3
"""
Automated SSH Helper with .env Password Management
===================================================

This script reads SSH credentials from .env file and automates common operations.

Usage:
    python ssh_auto.py command [args]

Commands:
    connect         - Open interactive SSH session
    exec <cmd>      - Execute command on server
    restart         - Restart dashboard
    logs            - View dashboard logs
    status          - Check dashboard status
    upload <file>   - Upload file to server

Examples:
    python ssh_auto.py connect
    python ssh_auto.py exec "ls -la"
    python ssh_auto.py restart
    python ssh_auto.py logs
"""

import os
import sys
import subprocess
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
env_file = Path(__file__).parent / ".env"
if not env_file.exists():
    print(f"ERROR: .env file not found at {env_file}")
    print("Please create .env file with SSH credentials")
    sys.exit(1)

load_dotenv(env_file)

# Get credentials from .env
SSH_HOST = os.getenv('SSH_HOST')
SSH_PORT = os.getenv('SSH_PORT')
SSH_USER = os.getenv('SSH_USER')
SSH_PASSWORD = os.getenv('SSH_PASSWORD')
DASHBOARD_PATH = os.getenv('DASHBOARD_PATH', '/home/tactical/racing-dashboard')
DASHBOARD_LOG = os.getenv('DASHBOARD_LOG', '/tmp/dashboard.log')

# Validate credentials
if not all([SSH_HOST, SSH_PORT, SSH_USER, SSH_PASSWORD]):
    print("ERROR: Missing SSH credentials in .env file")
    print("Required: SSH_HOST, SSH_PORT, SSH_USER, SSH_PASSWORD")
    sys.exit(1)


def ssh_connect():
    """Open interactive SSH session"""
    print(f"Connecting to {SSH_USER}@{SSH_HOST}:{SSH_PORT}...")
    print(f"Password: {SSH_PASSWORD}")
    print()

    # Use ssh with StrictHostKeyChecking=no to avoid prompts
    cmd = [
        'ssh',
        '-p', SSH_PORT,
        '-o', 'StrictHostKeyChecking=no',
        f'{SSH_USER}@{SSH_HOST}'
    ]

    subprocess.run(cmd)


def ssh_exec(command):
    """Execute command on server via SSH"""
    print(f"Executing: {command}")
    print()

    # Create SSH command with password echo
    ssh_cmd = f'sshpass -p "{SSH_PASSWORD}" ssh -p {SSH_PORT} -o StrictHostKeyChecking=no {SSH_USER}@{SSH_HOST} "{command}"'

    # If sshpass not available, fall back to regular ssh
    if subprocess.run(['which', 'sshpass'], capture_output=True).returncode != 0:
        print("Note: sshpass not installed - you'll need to enter password manually")
        ssh_cmd = f'ssh -p {SSH_PORT} -o StrictHostKeyChecking=no {SSH_USER}@{SSH_HOST} "{command}"'

    result = subprocess.run(ssh_cmd, shell=True, capture_output=False)
    return result.returncode


def restart_dashboard():
    """Restart dashboard on server"""
    print("=" * 60)
    print("Restarting Dashboard")
    print("=" * 60)
    print()

    commands = f'''
        cd {DASHBOARD_PATH} && \
        pkill -9 -f "src/dashboard/app.py" || true && \
        sleep 2 && \
        nohup venv/bin/python src/dashboard/app.py > {DASHBOARD_LOG} 2>&1 & \
        sleep 5 && \
        curl -s -o /dev/null -w "HTTP Status: %{{http_code}}\\n" http://localhost:8050/
    '''

    return ssh_exec(commands)


def view_logs(lines=50):
    """View dashboard logs"""
    print(f"Last {lines} lines of dashboard log:")
    print()

    command = f'tail -{lines} {DASHBOARD_LOG}'
    return ssh_exec(command)


def check_status():
    """Check dashboard status"""
    print("=" * 60)
    print("Dashboard Status")
    print("=" * 60)
    print()

    commands = f'''
        echo "Process:"
        ps aux | grep "src/dashboard/app.py" | grep -v grep || echo "  Not running"
        echo ""
        echo "HTTP Status:"
        curl -s -o /dev/null -w "  %{{http_code}}\\n" http://localhost:8050/
        echo ""
        echo "Port 8050:"
        netstat -tuln | grep 8050 || echo "  Not listening"
    '''

    return ssh_exec(commands)


def upload_file(local_file, remote_path=None):
    """Upload file to server"""
    if not os.path.exists(local_file):
        print(f"ERROR: File not found: {local_file}")
        return 1

    if remote_path is None:
        remote_path = f"{DASHBOARD_PATH}/{os.path.basename(local_file)}"

    print(f"Uploading {local_file} to {remote_path}...")

    scp_cmd = f'scp -P {SSH_PORT} -o StrictHostKeyChecking=no "{local_file}" {SSH_USER}@{SSH_HOST}:{remote_path}'

    # Try with sshpass first
    if subprocess.run(['which', 'sshpass'], capture_output=True).returncode == 0:
        scp_cmd = f'sshpass -p "{SSH_PASSWORD}" {scp_cmd}'
    else:
        print("Note: sshpass not installed - you'll need to enter password manually")

    result = subprocess.run(scp_cmd, shell=True)

    if result.returncode == 0:
        print(f"✓ Uploaded successfully to {remote_path}")
    else:
        print(f"✗ Upload failed")

    return result.returncode


def show_help():
    """Show help message"""
    print(__doc__)


def main():
    if len(sys.argv) < 2:
        show_help()
        sys.exit(0)

    command = sys.argv[1].lower()

    if command == 'connect':
        ssh_connect()

    elif command == 'exec':
        if len(sys.argv) < 3:
            print("ERROR: exec command requires an argument")
            print("Usage: python ssh_auto.py exec \"<command>\"")
            sys.exit(1)
        ssh_exec(sys.argv[2])

    elif command == 'restart':
        restart_dashboard()

    elif command == 'logs':
        lines = int(sys.argv[2]) if len(sys.argv) > 2 else 50
        view_logs(lines)

    elif command == 'status':
        check_status()

    elif command == 'upload':
        if len(sys.argv) < 3:
            print("ERROR: upload command requires a file path")
            print("Usage: python ssh_auto.py upload <file> [remote_path]")
            sys.exit(1)

        local_file = sys.argv[2]
        remote_path = sys.argv[3] if len(sys.argv) > 3 else None
        upload_file(local_file, remote_path)

    elif command in ['help', '--help', '-h']:
        show_help()

    else:
        print(f"ERROR: Unknown command: {command}")
        print()
        show_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
