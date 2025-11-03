@echo off
REM Racing Dashboard - Windows Deployment Script (Simplified)
REM Purpose: Guide user through manual deployment steps
REM Date: 2025-11-02

echo ==========================================
echo Racing Dashboard - Deployment Guide
echo ==========================================
echo.

set SERVER_IP=200.58.107.214
set SERVER_PORT=5197
set USERNAME=tactical
set REMOTE_DIR=/home/tactical/racing-dashboard

echo Server: %USERNAME%@%SERVER_IP%:%SERVER_PORT%
echo Remote Directory: %REMOTE_DIR%
echo.

echo ==========================================
echo PHASE 1: SERVER AUDIT
echo ==========================================
echo.

echo Step 1: Upload audit script
echo Command:
echo   scp -P %SERVER_PORT% deployment\01_audit_server.sh %USERNAME%@%SERVER_IP%:/tmp/audit_server.sh
echo.

set /p continue="Press Enter to copy this command to clipboard..."
echo scp -P %SERVER_PORT% deployment\01_audit_server.sh %USERNAME%@%SERVER_IP%:/tmp/audit_server.sh | clip
echo [Command copied to clipboard - paste in terminal]
echo.

echo Step 2: Run audit script via SSH
echo Command:
echo   ssh -p %SERVER_PORT% %USERNAME%@%SERVER_IP% "chmod +x /tmp/audit_server.sh && /tmp/audit_server.sh"
echo.

set /p continue="Press Enter to copy this command to clipboard..."
echo ssh -p %SERVER_PORT% %USERNAME%@%SERVER_IP% "chmod +x /tmp/audit_server.sh && /tmp/audit_server.sh" | clip
echo [Command copied to clipboard - paste in terminal]
echo.

set /p audit_passed="Did the audit pass? (yes/no): "
if /i not "%audit_passed%"=="yes" (
    echo.
    echo [ERROR] Server audit failed. Please resolve issues before continuing.
    pause
    exit /b 1
)

echo.
echo ==========================================
echo PHASE 2: UPLOAD PROJECT FILES
echo ==========================================
echo.

echo Recommended: Use WinSCP for easy file upload
echo   Download: https://winscp.net/
echo   Connect to: sftp://%USERNAME%@%SERVER_IP%:%SERVER_PORT%
echo   Upload folder: %CD%
echo   To: %REMOTE_DIR%
echo   Exclude: organized_data/, venv/, myenv/, __pycache__/
echo.

echo Alternative: Use SCP command
echo Command:
echo   scp -P %SERVER_PORT% -r src requirements.txt setup.bat data_loader.py analyze_all_data.py inventory_data.py %USERNAME%@%SERVER_IP%:%REMOTE_DIR%/
echo.

set /p upload_method="Upload method? (winscp/scp): "

if /i "%upload_method%"=="scp" (
    set /p continue="Press Enter to copy SCP command to clipboard..."
    echo scp -P %SERVER_PORT% -r src requirements.txt setup.bat data_loader.py analyze_all_data.py inventory_data.py %USERNAME%@%SERVER_IP%:%REMOTE_DIR%/ | clip
    echo [Command copied to clipboard]
)

echo.
set /p upload_complete="Have you uploaded all project files? (yes/no): "
if /i not "%upload_complete%"=="yes" (
    echo.
    echo [WARNING] Please complete file upload before proceeding to deployment.
    pause
    exit /b 1
)

echo.
echo ==========================================
echo PHASE 3: DEPLOYMENT
echo ==========================================
echo.

echo Step 1: Upload deployment script
echo Command:
echo   scp -P %SERVER_PORT% deployment\02_deploy_dashboard.sh %USERNAME%@%SERVER_IP%:%REMOTE_DIR%/deploy.sh
echo.

set /p continue="Press Enter to copy this command to clipboard..."
echo scp -P %SERVER_PORT% deployment\02_deploy_dashboard.sh %USERNAME%@%SERVER_IP%:%REMOTE_DIR%/deploy.sh | clip
echo [Command copied to clipboard - paste in terminal]
echo.

echo Step 2: Run deployment script
echo Command:
echo   ssh -p %SERVER_PORT% %USERNAME%@%SERVER_IP% "cd %REMOTE_DIR% && chmod +x deploy.sh && ./deploy.sh"
echo.

set /p continue="Press Enter to copy this command to clipboard..."
echo ssh -p %SERVER_PORT% %USERNAME%@%SERVER_IP% "cd %REMOTE_DIR% && chmod +x deploy.sh && ./deploy.sh" | clip
echo [Command copied to clipboard - paste in terminal]
echo.

echo [INFO] The deployment script will:
echo   - Install system packages (Python, Nginx, etc.)
echo   - Create virtual environment
echo   - Install Python dependencies
echo   - Configure systemd services
echo   - Set up Nginx reverse proxy
echo   - Start all services
echo.
echo [WARNING] You will be prompted for sudo password during deployment
echo.

set /p deploy_complete="Did the deployment complete successfully? (yes/no): "
if /i not "%deploy_complete%"=="yes" (
    echo.
    echo [ERROR] Deployment failed. SSH into server to debug:
    echo   ssh -p %SERVER_PORT% %USERNAME%@%SERVER_IP%
    pause
    exit /b 1
)

echo.
echo ==========================================
echo DEPLOYMENT COMPLETE!
echo ==========================================
echo.
echo Dashboard URL: http://%SERVER_IP%
echo API URL: http://%SERVER_IP%/api
echo.
echo Next Steps:
echo   1. Open browser: http://%SERVER_IP%
echo   2. Upload master_racing_data.csv
echo   3. Test all features
echo.
echo Service Management:
echo   Check status: ssh -p %SERVER_PORT% %USERNAME%@%SERVER_IP% "sudo systemctl status racing-api racing-dashboard"
echo   View logs: ssh -p %SERVER_PORT% %USERNAME%@%SERVER_IP% "sudo journalctl -u racing-dashboard -f"
echo   Restart: ssh -p %SERVER_PORT% %USERNAME%@%SERVER_IP% "sudo systemctl restart racing-dashboard"
echo.

pause
