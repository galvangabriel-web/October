@echo off
REM Server Audit Automation for Windows
REM This script will upload and run the audit script on the server

setlocal EnableDelayedExpansion

set SERVER_IP=200.58.107.214
set SERVER_PORT=5197
set USERNAME=tactical
set PASSWORD=1253*1253*Win1

echo ========================================
echo Racing Dashboard - Server Audit
echo ========================================
echo.
echo Server: %USERNAME%@%SERVER_IP%:%SERVER_PORT%
echo.

REM Step 1: Upload audit script
echo [1/2] Uploading audit script...
echo.

REM Create a temporary file with SCP command
echo !PASSWORD!| scp -P %SERVER_PORT% -o StrictHostKeyChecking=no deployment\01_audit_server.sh %USERNAME%@%SERVER_IP%:/tmp/audit_server.sh

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Failed to upload audit script
    echo.
    echo Please run this command manually:
    echo   scp -P %SERVER_PORT% deployment\01_audit_server.sh %USERNAME%@%SERVER_IP%:/tmp/audit_server.sh
    echo.
    pause
    exit /b 1
)

echo.
echo Upload successful!
echo.

REM Step 2: Run audit script
echo [2/2] Running audit on server...
echo ========================================
echo.

REM Execute audit
echo !PASSWORD!| ssh -p %SERVER_PORT% -o StrictHostKeyChecking=no %USERNAME%@%SERVER_IP% "chmod +x /tmp/audit_server.sh && /tmp/audit_server.sh"

set AUDIT_EXIT=%ERRORLEVEL%

echo.
echo ========================================
echo Audit Complete
echo ========================================
echo.

if %AUDIT_EXIT% EQU 0 (
    echo [SUCCESS] Server is ready for deployment!
    echo.
    echo Next step: Upload project files
    echo   Use WinSCP or run: deployment\upload_files.bat
) else (
    echo [WARNING] Audit completed with exit code: %AUDIT_EXIT%
    echo.
    echo Review the output above for any issues.
)

echo.
pause
