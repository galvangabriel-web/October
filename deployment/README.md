# Deployment Scripts - Racing Dashboard

This directory contains automated deployment scripts to deploy the Racing Telemetry Dashboard to a Linux server.

## Quick Start

### Prerequisites
- Windows machine with OpenSSH client
- Linux server with SSH access
- Server credentials

### Deployment Steps

1. **Run automated deployment (Recommended)**
   ```powershell
   cd C:\project\data_analisys_car\deployment
   .\deploy_to_server.ps1
   ```
   Enter password when prompted.

2. **Or run step-by-step**
   ```cmd
   cd C:\project\data_analisys_car\deployment
   deploy_windows.bat
   ```
   Follow the interactive prompts.

## Files in this Directory

### Automation Scripts (Run from Windows)

- **`deploy_to_server.ps1`** - PowerShell deployment automation
  - Runs all phases: audit → upload → deploy
  - Handles SSH connections and file transfers
  - Usage:
    ```powershell
    .\deploy_to_server.ps1                    # Full deployment
    .\deploy_to_server.ps1 -Phase audit       # Audit only
    .\deploy_to_server.ps1 -Phase upload      # Upload only
    .\deploy_to_server.ps1 -Phase deploy      # Deploy only
    ```

- **`deploy_windows.bat`** - Batch file for CMD
  - Interactive deployment guide
  - Copies commands to clipboard
  - Easier for Windows users unfamiliar with PowerShell

### Server-Side Scripts (Uploaded to Linux)

- **`01_audit_server.sh`** - Server requirements audit
  - Checks CPU, RAM, disk space
  - Verifies Python version
  - Tests network ports
  - Run on server: `./01_audit_server.sh`

- **`02_deploy_dashboard.sh`** - Automated deployment
  - Installs system packages
  - Creates virtual environment
  - Installs Python dependencies
  - Configures systemd services
  - Sets up Nginx reverse proxy
  - Starts all services
  - Run on server: `./02_deploy_dashboard.sh`

### Documentation

- **`DEPLOYMENT_PLAN.md`** - Comprehensive deployment guide
  - Server requirements
  - Step-by-step instructions
  - Troubleshooting guide
  - Security considerations
  - **READ THIS FIRST!**

- **`QUICK_REFERENCE.txt`** - Quick command reference
  - Common SSH commands
  - Service management
  - Log viewing
  - Emergency procedures

## Deployment Process

### Phase 1: Server Audit (5 minutes)
```powershell
.\deploy_to_server.ps1 -Phase audit
```
Verifies server meets requirements.

### Phase 2: File Upload (5-15 minutes)
```powershell
.\deploy_to_server.ps1 -Phase upload
```
Or use WinSCP for easier file management.

### Phase 3: Deployment (15-30 minutes)
```powershell
.\deploy_to_server.ps1 -Phase deploy
```
Installs packages, configures services, starts dashboard.

## Server Details

- **IP:** 200.58.107.214
- **SSH Port:** 5197
- **Username:** tactical
- **Remote Directory:** /home/tactical/racing-dashboard
- **Dashboard URL:** http://200.58.107.214
- **API URL:** http://200.58.107.214/api

## After Deployment

### Access Dashboard
Open browser: `http://200.58.107.214`

### Service Management
```bash
# SSH into server
ssh -p 5197 tactical@200.58.107.214

# Check status
sudo systemctl status racing-api racing-dashboard

# Restart services
sudo systemctl restart racing-dashboard

# View logs
sudo journalctl -u racing-dashboard -f
```

## Troubleshooting

### Can't connect to dashboard
1. Check services: `sudo systemctl status racing-dashboard nginx`
2. Check firewall: `sudo ufw status`
3. View logs: `sudo journalctl -u racing-dashboard -n 100`

### Import errors
1. Verify files uploaded: `ls -la /home/tactical/racing-dashboard/`
2. Reinstall deps: `source venv/bin/activate && pip install -r requirements.txt`
3. Restart service: `sudo systemctl restart racing-dashboard`

### High memory usage
1. Check with: `free -h`
2. Restart services: `sudo systemctl restart racing-api racing-dashboard`
3. Reduce API workers in `/etc/systemd/system/racing-api.service`

## Security Notes

⚠️ **Current deployment uses password authentication**

For production, implement:
1. SSH key authentication
2. HTTPS/SSL with Let's Encrypt
3. Dashboard authentication
4. Firewall restrictions

See `DEPLOYMENT_PLAN.md` for detailed security guidelines.

## Support

- Full deployment guide: `DEPLOYMENT_PLAN.md`
- Architecture reference: `../CLAUDE.md`
- Project documentation: `../README.md`

## Version

- Script Version: 1.0
- Created: 2025-11-02
- Platform: Windows → Linux (Ubuntu/CentOS)
