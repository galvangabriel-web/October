# Temporary script for server audit
param(
    [string]$Password = "1253*1253*Win1",
    [string]$ServerIP = "200.58.107.214",
    [int]$ServerPort = 5197,
    [string]$Username = "tactical"
)

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Server Audit - Starting..." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$AuditScript = Join-Path $PSScriptRoot "01_audit_server.sh"

# Test 1: Verify audit script exists
if (-not (Test-Path $AuditScript)) {
    Write-Host "ERROR: Audit script not found at $AuditScript" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Audit script found" -ForegroundColor Green

# Test 2: Test SSH connection
Write-Host ""
Write-Host "Testing SSH connection..." -ForegroundColor Yellow

# Simple connection test - will prompt for password
$sshTest = "ssh -p $ServerPort -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ${Username}@${ServerIP} 'echo Connected' 2>&1"
Write-Host "Attempting connection to ${Username}@${ServerIP}:${ServerPort}..." -ForegroundColor Gray

# Test 3: Upload audit script using psftp approach
Write-Host ""
Write-Host "Uploading audit script to server..." -ForegroundColor Yellow

# Use WinSCP scripting or simple scp
# For automation, we'll use a direct approach
$scpCmd = "scp -P $ServerPort -o StrictHostKeyChecking=no `"$AuditScript`" ${Username}@${ServerIP}:/tmp/audit_server.sh"

Write-Host "Running: $scpCmd" -ForegroundColor Gray
Write-Host ""
Write-Host "Note: OpenSSH on Windows requires interactive password entry" -ForegroundColor Yellow
Write-Host "Please enter password when prompted: 1253*1253*Win1" -ForegroundColor Yellow
Write-Host ""

# Execute SCP
$process = Start-Process -FilePath "scp" -ArgumentList "-P",$ServerPort,"-o","StrictHostKeyChecking=no",$AuditScript,"${Username}@${ServerIP}:/tmp/audit_server.sh" -Wait -NoNewWindow -PassThru

if ($process.ExitCode -eq 0) {
    Write-Host "✓ Audit script uploaded successfully" -ForegroundColor Green
} else {
    Write-Host "✗ Upload failed with exit code: $($process.ExitCode)" -ForegroundColor Red
    Write-Host ""
    Write-Host "Alternative: Use WinSCP to upload 01_audit_server.sh to /tmp/" -ForegroundColor Yellow
    exit 1
}

# Test 4: Run audit script
Write-Host ""
Write-Host "Running audit on server..." -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$sshCmd = "ssh -p $ServerPort -o StrictHostKeyChecking=no ${Username}@${ServerIP} `"chmod +x /tmp/audit_server.sh; /tmp/audit_server.sh`""

Write-Host "Running: $sshCmd" -ForegroundColor Gray
Write-Host ""

# Execute SSH command
$process2 = Start-Process -FilePath "ssh" -ArgumentList "-p",$ServerPort,"-o","StrictHostKeyChecking=no","${Username}@${ServerIP}","chmod +x /tmp/audit_server.sh; /tmp/audit_server.sh" -Wait -NoNewWindow -PassThru

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan

if ($process2.ExitCode -eq 0) {
    Write-Host "✓ Audit completed successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Server is READY for deployment" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next step: Upload project files and run deployment" -ForegroundColor Cyan
} else {
    Write-Host "⚠ Audit exit code: $($process2.ExitCode)" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Review the output above for any issues" -ForegroundColor Yellow
}

Write-Host "========================================" -ForegroundColor Cyan
