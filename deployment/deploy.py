#!/usr/bin/env python3
"""
Automated Deployment Script for Racing Analytics Dashboard
Deploys to Linux server with full capabilities (RAM upgraded)
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import paramiko
import time

# Fix Windows console encoding
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Load environment variables from project root
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

class DashboardDeployer:
    def __init__(self):
        self.ssh_host = os.getenv('SSH_HOST')
        self.ssh_port = int(os.getenv('SSH_PORT', 22))
        self.ssh_user = os.getenv('SSH_USER')
        self.ssh_password = os.getenv('SSH_PASSWORD')
        self.ssh_key_path = os.getenv('SSH_KEY_PATH')
        self.deploy_path = os.getenv('DEPLOY_PATH')
        self.venv_path = os.getenv('VENV_PATH')

        self.client = None
        self.sftp = None

    def validate_config(self):
        """Validate required configuration"""
        required = ['SSH_HOST', 'SSH_USER', 'DEPLOY_PATH']
        missing = [var for var in required if not os.getenv(var)]

        if missing:
            print(f"❌ Missing required environment variables: {', '.join(missing)}")
            print("Please edit .env file with your server details")
            return False

        if not self.ssh_password and not self.ssh_key_path:
            print("❌ Either SSH_PASSWORD or SSH_KEY_PATH must be set")
            return False

        if self.ssh_password == "CHANGE_ME_TO_YOUR_PASSWORD":
            print("❌ Please update SSH_PASSWORD in .env file")
            return False

        return True

    def connect(self):
        """Establish SSH connection"""
        print(f"🔌 Connecting to {self.ssh_user}@{self.ssh_host}:{self.ssh_port}...")

        try:
            self.client = paramiko.SSHClient()
            self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

            if self.ssh_key_path:
                self.client.connect(
                    hostname=self.ssh_host,
                    port=self.ssh_port,
                    username=self.ssh_user,
                    key_filename=self.ssh_key_path
                )
            else:
                self.client.connect(
                    hostname=self.ssh_host,
                    port=self.ssh_port,
                    username=self.ssh_user,
                    password=self.ssh_password
                )

            self.sftp = self.client.open_sftp()
            print("✅ Connected successfully")
            return True

        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False

    def execute_command(self, command, show_output=True):
        """Execute command on remote server"""
        stdin, stdout, stderr = self.client.exec_command(command)
        exit_code = stdout.channel.recv_exit_status()

        output = stdout.read().decode('utf-8')
        error = stderr.read().decode('utf-8')

        if show_output and output:
            print(output)
        if error:
            print(f"⚠️  {error}")

        return exit_code == 0, output, error

    def create_remote_directory(self, path):
        """Create directory on remote server"""
        try:
            self.sftp.mkdir(path)
        except IOError:
            pass  # Directory already exists

    def upload_file(self, local_path, remote_path):
        """Upload file to server"""
        try:
            self.sftp.put(local_path, remote_path)
            return True
        except Exception as e:
            print(f"❌ Failed to upload {local_path}: {e}")
            return False

    def upload_directory(self, local_dir, remote_dir):
        """Recursively upload directory"""
        print(f"📤 Uploading {local_dir} to {remote_dir}...")

        local_path = Path(local_dir)
        if not local_path.exists():
            print(f"⚠️  {local_dir} does not exist, skipping")
            return

        # Create remote directory
        self.execute_command(f"mkdir -p {remote_dir}", show_output=False)

        # Upload files
        for item in local_path.rglob('*'):
            if item.is_file():
                # Skip certain files
                if any(skip in str(item) for skip in ['__pycache__', '.pyc', '.git', 'venv', 'myenv']):
                    continue

                relative_path = item.relative_to(local_path)
                remote_file = f"{remote_dir}/{relative_path}".replace('\\', '/')

                # Create parent directories
                remote_parent = '/'.join(remote_file.split('/')[:-1])
                self.execute_command(f"mkdir -p {remote_parent}", show_output=False)

                # Upload file
                self.upload_file(str(item), remote_file)

        print(f"✅ Uploaded {local_dir}")

    def setup_server(self):
        """Setup server environment"""
        print("\n🔧 Setting up server environment...")

        # Update system
        print("📦 Updating system packages...")
        self.execute_command("sudo apt-get update -qq", show_output=False)

        # Install Python and dependencies
        print("🐍 Installing Python and dependencies...")
        install_cmd = """
        sudo apt-get install -y python3 python3-pip python3-venv \\
            build-essential libssl-dev libffi-dev python3-dev \\
            git curl wget
        """
        self.execute_command(install_cmd, show_output=False)

        # Create deploy directory
        print(f"📁 Creating deployment directory: {self.deploy_path}")
        self.execute_command(f"mkdir -p {self.deploy_path}")

        print("✅ Server setup complete")

    def deploy_application(self):
        """Deploy application files"""
        print("\n📦 Deploying application files...")

        # Upload core files
        files_to_upload = [
            'data_loader.py',
            'analyze_all_data.py',
            'inventory_data.py',
            'requirements.txt',
            'README.md',
            'CLAUDE.md',
            '.env'
        ]

        for file in files_to_upload:
            if Path(file).exists():
                remote_path = f"{self.deploy_path}/{file}"
                self.upload_file(file, remote_path)
                print(f"✅ Uploaded {file}")

        # Upload directories
        directories = ['src', 'tests', 'deployment']
        for directory in directories:
            self.upload_directory(directory, f"{self.deploy_path}/{directory}")

        print("✅ Application files deployed")

    def setup_virtualenv(self):
        """Setup Python virtual environment"""
        print("\n🐍 Setting up Python virtual environment...")

        # Create venv
        self.execute_command(f"cd {self.deploy_path} && python3 -m venv venv")

        # Upgrade pip
        pip_cmd = f"cd {self.deploy_path} && source venv/bin/activate && pip install --upgrade pip"
        self.execute_command(pip_cmd, show_output=False)

        # Install requirements
        print("📦 Installing Python packages (this may take 5-10 minutes)...")
        install_cmd = f"cd {self.deploy_path} && source venv/bin/activate && pip install -r requirements.txt"
        success, output, error = self.execute_command(install_cmd, show_output=True)

        if success:
            print("✅ Virtual environment setup complete")
        else:
            print("⚠️  Some packages may have failed to install")

        return success

    def create_startup_scripts(self):
        """Create startup scripts on server"""
        print("\n📝 Creating startup scripts...")

        # Dashboard startup script
        dashboard_script = f"""#!/bin/bash
# Racing Analytics Dashboard Startup Script
# Full Capabilities Mode (RAM Upgraded)

cd {self.deploy_path}
source venv/bin/activate

export ENABLE_FULL_MEMORY=true
export LOAD_ALL_CHUNKS=true
export DASHBOARD_HOST={os.getenv('DASHBOARD_HOST', '0.0.0.0')}
export DASHBOARD_PORT={os.getenv('DASHBOARD_PORT', '8050')}

echo "🏁 Starting Racing Analytics Dashboard..."
echo "📊 Mode: Full Capabilities (All data chunks loaded)"
echo "🌐 Host: $DASHBOARD_HOST:$DASHBOARD_PORT"
echo ""

python src/dashboard/app.py
"""

        # API startup script
        api_script = f"""#!/bin/bash
# Racing Analytics API Startup Script
# Full Capabilities Mode

cd {self.deploy_path}
source venv/bin/activate

export ENABLE_FULL_MEMORY=true
export API_HOST={os.getenv('API_HOST', '0.0.0.0')}
export API_PORT={os.getenv('API_PORT', '8000')}
export API_WORKERS={os.getenv('API_WORKERS', '4')}

echo "🚀 Starting Racing Analytics API..."
echo "🌐 Host: $API_HOST:$API_PORT"
echo "👷 Workers: $API_WORKERS"
echo ""

python -m uvicorn src.api.main:app \\
    --host $API_HOST \\
    --port $API_PORT \\
    --workers $API_WORKERS \\
    --reload
"""

        # Stop all script
        stop_script = f"""#!/bin/bash
# Stop all Racing Analytics services

echo "🛑 Stopping all services..."
pkill -f "python src/dashboard/app.py"
pkill -f "uvicorn src.api.main:app"
echo "✅ All services stopped"
"""

        # Write scripts to temp files with UTF-8 encoding
        with open('temp_dashboard.sh', 'w', encoding='utf-8') as f:
            f.write(dashboard_script)
        with open('temp_api.sh', 'w', encoding='utf-8') as f:
            f.write(api_script)
        with open('temp_stop.sh', 'w', encoding='utf-8') as f:
            f.write(stop_script)

        # Upload and set permissions
        for script_name, temp_file in [
            ('start_dashboard.sh', 'temp_dashboard.sh'),
            ('start_api.sh', 'temp_api.sh'),
            ('stop_all.sh', 'temp_stop.sh')
        ]:
            remote_path = f"{self.deploy_path}/{script_name}"
            self.upload_file(temp_file, remote_path)
            self.execute_command(f"chmod +x {remote_path}")
            os.remove(temp_file)

        print("✅ Startup scripts created")

    def create_systemd_services(self):
        """Create systemd service files (optional)"""
        print("\n🔧 Creating systemd services...")

        dashboard_service = f"""[Unit]
Description=Racing Analytics Dashboard
After=network.target

[Service]
Type=simple
User={self.ssh_user}
WorkingDirectory={self.deploy_path}
ExecStart={self.deploy_path}/start_dashboard.sh
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""

        api_service = f"""[Unit]
Description=Racing Analytics API
After=network.target

[Service]
Type=simple
User={self.ssh_user}
WorkingDirectory={self.deploy_path}
ExecStart={self.deploy_path}/start_api.sh
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""

        # Write to temp files with UTF-8 encoding
        with open('temp_dashboard.service', 'w', encoding='utf-8') as f:
            f.write(dashboard_service)
        with open('temp_api.service', 'w', encoding='utf-8') as f:
            f.write(api_service)

        # Upload to /tmp and move to systemd directory
        self.upload_file('temp_dashboard.service', '/tmp/racing-dashboard.service')
        self.upload_file('temp_api.service', '/tmp/racing-api.service')

        self.execute_command("sudo mv /tmp/racing-dashboard.service /etc/systemd/system/", show_output=False)
        self.execute_command("sudo mv /tmp/racing-api.service /etc/systemd/system/", show_output=False)
        self.execute_command("sudo systemctl daemon-reload", show_output=False)

        os.remove('temp_dashboard.service')
        os.remove('temp_api.service')

        print("✅ Systemd services created")
        print("   To enable: sudo systemctl enable racing-dashboard racing-api")
        print("   To start: sudo systemctl start racing-dashboard racing-api")

    def test_deployment(self):
        """Test deployment"""
        print("\n🧪 Testing deployment...")

        # Check Python version
        success, output, _ = self.execute_command(
            f"cd {self.deploy_path} && source venv/bin/activate && python --version",
            show_output=False
        )
        if success:
            print(f"✅ Python: {output.strip()}")

        # Check key packages
        for package in ['pandas', 'dash', 'fastapi', 'lightgbm']:
            success, output, _ = self.execute_command(
                f"cd {self.deploy_path} && source venv/bin/activate && python -c 'import {package}; print({package}.__version__)'",
                show_output=False
            )
            if success:
                print(f"✅ {package}: {output.strip()}")
            else:
                print(f"❌ {package}: Not installed")

        print("✅ Deployment tests complete")

    def deploy(self):
        """Main deployment function"""
        print("=" * 60)
        print("🏁 Racing Analytics Dashboard - Automated Deployment")
        print("=" * 60)

        # Validate configuration
        if not self.validate_config():
            return False

        # Connect to server
        if not self.connect():
            return False

        try:
            # Setup server
            self.setup_server()

            # Deploy application
            self.deploy_application()

            # Setup virtual environment
            if not self.setup_virtualenv():
                print("⚠️  Continuing despite package installation warnings...")

            # Create startup scripts
            self.create_startup_scripts()

            # Optional: Create systemd services
            create_services = input("\n❓ Create systemd services for auto-start? (y/n): ").lower() == 'y'
            if create_services:
                self.create_systemd_services()

            # Test deployment
            self.test_deployment()

            print("\n" + "=" * 60)
            print("✅ DEPLOYMENT SUCCESSFUL!")
            print("=" * 60)
            print(f"\n📍 Application deployed to: {self.deploy_path}")
            print(f"\n🚀 To start the dashboard:")
            print(f"   ssh {self.ssh_user}@{self.ssh_host}")
            print(f"   cd {self.deploy_path}")
            print(f"   ./start_dashboard.sh")
            print(f"\n🔌 To start the API:")
            print(f"   ssh {self.ssh_user}@{self.ssh_host}")
            print(f"   cd {self.deploy_path}")
            print(f"   ./start_api.sh")
            print(f"\n🛑 To stop all services:")
            print(f"   ./stop_all.sh")
            print(f"\n🌐 Access dashboard at: http://{self.ssh_host}:{os.getenv('DASHBOARD_PORT', '8050')}")

            if create_services:
                print(f"\n🔧 Systemd services created:")
                print(f"   sudo systemctl start racing-dashboard")
                print(f"   sudo systemctl start racing-api")

            return True

        except Exception as e:
            print(f"\n❌ Deployment failed: {e}")
            import traceback
            traceback.print_exc()
            return False

        finally:
            if self.sftp:
                self.sftp.close()
            if self.client:
                self.client.close()

def main():
    """Main entry point"""
    if not Path('.env').exists():
        print("❌ .env file not found!")
        print("Please copy .env.example to .env and fill in your server details")
        return 1

    deployer = DashboardDeployer()
    success = deployer.deploy()

    return 0 if success else 1

if __name__ == '__main__':
    sys.exit(main())
