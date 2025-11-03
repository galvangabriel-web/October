#!/usr/bin/env python3
"""Map production server directory structure - Fixed version"""
import os
from pathlib import Path
from dotenv import load_dotenv
import paramiko

env_path = Path(__file__).parent / '.env'
load_dotenv(env_path)

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(
    hostname=os.getenv('SSH_HOST'),
    port=int(os.getenv('SSH_PORT', 22)),
    username=os.getenv('SSH_USER'),
    password=os.getenv('SSH_PASSWORD')
)

commands = {
    "Directory tree": "cd /home/tactical/racing_analytics && find . -maxdepth 3 -type d | sort",
    "Key files in root": "cd /home/tactical/racing_analytics && ls -lh *.py *.log *.csv *.txt *.md 2>/dev/null | head -20",
    "Source code structure": "cd /home/tactical/racing_analytics/src && find . -type d | sort",
    "Data directory": "cd /home/tactical/racing_analytics && ls -lh data/ 2>/dev/null",
    "Venv info": "cd /home/tactical/racing_analytics && du -sh venv/ && echo '---' && ls venv/bin/ | grep -E '(python|pip|uvicorn)'",
    "Log files": "cd /home/tactical/racing_analytics && ls -lh *.log 2>/dev/null",
    "Disk usage by directory": "cd /home/tactical/racing_analytics && du -sh */ 2>/dev/null | sort -h",
    "Python packages": "cd /home/tactical/racing_analytics && venv/bin/pip list | head -30",
}

output_file = open("server_structure.txt", "w", encoding="utf-8")

for title, cmd in commands.items():
    output_file.write(f"\n{'='*60}\n")
    output_file.write(f"{title}\n")
    output_file.write(f"{'='*60}\n")

    stdin, stdout, stderr = client.exec_command(cmd)
    result = stdout.read().decode('utf-8', errors='replace')

    output_file.write(result + "\n")
    print(f"✓ Collected: {title}")  # Safe ASCII-only output

output_file.close()
client.close()

print(f"\n✓ Server structure saved to: server_structure.txt")
