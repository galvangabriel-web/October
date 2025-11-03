# Racing Dashboard - Windows Deployment Automation
# Purpose: Automate deployment from Windows to Linux server via SSH
# Date: 2025-11-02

param(
    [Parameter(Mandatory=$false)]
    [string]$Phase = "all",  # Options: audit, upload, deploy, all

    [Parameter(Mandatory=$false)]
    [string]$ServerIP = "200.58.107.214",

    [Parameter(Mandatory=$false)]
    [int]$ServerPort = 5197,

    [Parameter(Mandatory=$false)]
    [string]$Username = "tactical",

    [Parameter(Mandatory=$false)]
    [SecureString]$Password
)

# Color functions
function Write-ColorOutput {
    param(
        [string]$Message,
        [string]$Color = "White"
    )
    Write-Host $Message -ForegroundColor $Color
}

function Write-Success {
    param([string]$Message)
    Write-ColorOutput "✓ $Message" "Green"
}

function Write-Error {
    param([string]$Message)
    Write-ColorOutput "✗ $Message" "Red"
}

function Write-Warning {
    param([string]$Message)
    Write-ColorOutput "⚠ $Message" "Yellow"
}

function Write-Info {
    param([string]$Message)
    Write-ColorOutput "[INFO] $Message" "Cyan"
}

# Banner
Write-ColorOutput "==========================================" "Blue"
Write-ColorOutput "Racing Dashboard - Deployment Automation" "Blue"
Write-ColorOutput "==========================================" "Blue"
Write-Host ""

# Get project root directory
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Write-Info "Project root: $ProjectRoot"

# Get password if not provided
if (-not $Password) {
    Write-Host ""
    $Password = Read-Host "Enter password for ${Username}@${ServerIP}" -AsSecureString
}

# Convert SecureString to plain text for SSH (required for automation)
$BSTR = [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($Password)
$PlainPassword = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto($BSTR)
[System.Runtime.InteropServices.Marshal]::ZeroFreeBSTR($BSTR)

# Check for required tools
Write-Info "Checking for required tools..."

# Check for SSH
if (-not (Get-Command ssh -ErrorAction SilentlyContinue)) {
    Write-Error "SSH not found. Please install OpenSSH client."
    Write-Info "Install via: Add-WindowsCapability -Online -Name OpenSSH.Client~~~~0.0.1.0"
    exit 1
}
Write-Success "SSH client found"

# Check for SCP
if (-not (Get-Command scp -ErrorAction SilentlyContinue)) {
    Write-Error "SCP not found. Please install OpenSSH client."
    exit 1
}
Write-Success "SCP found"

# Check if pscp (PuTTY SCP) is available as alternative
$usePutty = $false
if (Get-Command pscp -ErrorAction SilentlyContinue) {
    $usePutty = $true
    Write-Info "PuTTY SCP detected - will use as alternative"
}

Write-Host ""

# Function to execute SSH command with password
function Invoke-SSHCommand {
    param(
        [string]$Command,
        [switch]$Sudo = $false
    )

    $fullCommand = if ($Sudo) { "echo '$PlainPassword' | sudo -S $Command" } else { $Command }

    # Create temporary expect script for password automation
    $expectScript = @"
#!/usr/bin/expect -f
set timeout 300
spawn ssh -p $ServerPort ${Username}@${ServerIP} "$fullCommand"
expect {
    "password:" {
        send "$PlainPassword\r"
        exp_continue
    }
    "yes/no" {
        send "yes\r"
        exp_continue
    }
    eof
}
"@

    # For Windows, we'll use plink (PuTTY) or OpenSSH with sshpass equivalent
    # Since Windows doesn't have expect, we'll use echo password approach

    $sshCmd = "ssh -p $ServerPort ${Username}@${ServerIP} `"$fullCommand`""

    # Note: This is not secure for production. For production, use SSH keys.
    Write-Info "Executing: $Command"

    # Using SSH with password requires third-party tools on Windows
    # Best practice: Use SSH key authentication instead
    # For automation, we'll create a batch file

    $batchFile = "$env:TEMP\ssh_command_temp.bat"
    @"
@echo off
echo $PlainPassword | ssh -p $ServerPort ${Username}@${ServerIP} "$fullCommand"
"@ | Out-File -FilePath $batchFile -Encoding ASCII

    & cmd /c $batchFile
    Remove-Item $batchFile -Force
}

# PHASE 1: Server Audit
if ($Phase -eq "audit" -or $Phase -eq "all") {
    Write-ColorOutput "`n=== PHASE 1: SERVER AUDIT ===" "Yellow"
    Write-Info "Connecting to ${Username}@${ServerIP}:${ServerPort}..."

    # Upload audit script
    Write-Info "Uploading audit script..."

    $auditScript = Join-Path $PSScriptRoot "01_audit_server.sh"

    if ($usePutty) {
        & pscp -P $ServerPort -pw $PlainPassword $auditScript ${Username}@${ServerIP}:/tmp/audit_server.sh
    } else {
        # For OpenSSH, we need to use alternative method
        Write-Warning "OpenSSH on Windows doesn't support password authentication easily."
        Write-Info "Please use one of these methods:"
        Write-Info "1. Set up SSH key authentication (recommended)"
        Write-Info "2. Install PuTTY tools (pscp, plink)"
        Write-Info "3. Use WinSCP or similar GUI tool"
        Write-Host ""
        Write-Info "Manual command to upload audit script:"
        Write-ColorOutput "  scp -P $ServerPort $auditScript ${Username}@${ServerIP}:/tmp/audit_server.sh" "White"
        Write-Host ""

        $response = Read-Host "Have you uploaded the audit script manually? (yes/no)"
        if ($response -ne "yes") {
            Write-Error "Audit script not uploaded. Exiting."
            exit 1
        }
    }

    Write-Success "Audit script uploaded"

    # Execute audit script
    Write-Info "Running server audit..."
    Write-Info "You will need to enter the password when prompted..."

    & ssh -p $ServerPort ${Username}@${ServerIP} "chmod +x /tmp/audit_server.sh && /tmp/audit_server.sh"

    if ($LASTEXITCODE -ne 0) {
        Write-Error "Server audit failed. Please resolve issues before continuing."
        exit 1
    }

    Write-Success "Server audit completed successfully"
    Write-Host ""
}

# PHASE 2: Upload Project Files
if ($Phase -eq "upload" -or $Phase -eq "all") {
    Write-ColorOutput "`n=== PHASE 2: UPLOAD PROJECT FILES ===" "Yellow"

    $remoteDir = "/home/${Username}/racing-dashboard"

    Write-Info "Creating remote directory: $remoteDir"
    & ssh -p $ServerPort ${Username}@${ServerIP} "mkdir -p $remoteDir"

    Write-Info "Uploading project files (this may take several minutes)..."
    Write-Warning "Note: This will NOT upload organized_data/ (18.5GB) or venv/ directories"

    # Files and directories to exclude
    $excludePatterns = @(
        "organized_data",
        "venv",
        "myenv",
        "__pycache__",
        "*.pyc",
        "*.pyo",
        ".git",
        "catboost_info",
        "output",
        "reports",
        "csv_testing",
        "node_modules",
        ".env"
    )

    # Create temporary rsync exclude file
    $excludeFile = "$env:TEMP\rsync_exclude.txt"
    $excludePatterns | Out-File -FilePath $excludeFile -Encoding ASCII

    Write-Info "Manual upload command (use this if automated upload fails):"
    Write-ColorOutput "  scp -P $ServerPort -r $ProjectRoot ${Username}@${ServerIP}:$remoteDir" "White"
    Write-Host ""

    # For Windows, we'll provide instructions
    Write-Info "Recommended approach for Windows:"
    Write-Info "1. Install WinSCP (https://winscp.net/)"
    Write-Info "2. Connect to: sftp://${Username}@${ServerIP}:${ServerPort}"
    Write-Info "3. Upload entire project folder to: $remoteDir"
    Write-Info "4. Exclude: organized_data/, venv/, myenv/, __pycache__/"
    Write-Host ""

    $response = Read-Host "Have you uploaded the project files? (yes/no)"
    if ($response -ne "yes") {
        Write-Warning "Project files not uploaded. You can upload manually later."
    } else {
        Write-Success "Project files uploaded"
    }

    Remove-Item $excludeFile -Force -ErrorAction SilentlyContinue
    Write-Host ""
}

# PHASE 3: Deploy
if ($Phase -eq "deploy" -or $Phase -eq "all") {
    Write-ColorOutput "`n=== PHASE 3: DEPLOYMENT ===" "Yellow"

    $remoteDir = "/home/${Username}/racing-dashboard"

    # Upload deployment script
    Write-Info "Uploading deployment script..."
    $deployScript = Join-Path $PSScriptRoot "02_deploy_dashboard.sh"

    Write-Info "Manual command:"
    Write-ColorOutput "  scp -P $ServerPort $deployScript ${Username}@${ServerIP}:$remoteDir/deploy.sh" "White"
    Write-Host ""

    $response = Read-Host "Upload deployment script manually, then press Enter to continue..."

    # Execute deployment
    Write-Info "Starting deployment process..."
    Write-Info "This will install packages and configure services..."
    Write-Warning "You may be prompted for the sudo password multiple times"
    Write-Host ""

    & ssh -p $ServerPort ${Username}@${ServerIP} "cd $remoteDir && chmod +x deploy.sh && ./deploy.sh"

    if ($LASTEXITCODE -eq 0) {
        Write-Success "Deployment completed successfully!"
        Write-Host ""
        Write-ColorOutput "=== DEPLOYMENT SUMMARY ===" "Green"
        Write-ColorOutput "Dashboard URL: http://${ServerIP}" "Cyan"
        Write-ColorOutput "API URL: http://${ServerIP}/api" "Cyan"
        Write-Host ""
        Write-Info "Next steps:"
        Write-Info "1. Open browser and navigate to http://${ServerIP}"
        Write-Info "2. Upload master_racing_data.csv"
        Write-Info "3. Test all dashboard features"
        Write-Host ""
        Write-Info "To check service status:"
        Write-ColorOutput "  ssh -p $ServerPort ${Username}@${ServerIP} 'sudo systemctl status racing-api racing-dashboard'" "White"
        Write-Host ""
        Write-Info "To view logs:"
        Write-ColorOutput "  ssh -p $ServerPort ${Username}@${ServerIP} 'sudo journalctl -u racing-dashboard -f'" "White"
    } else {
        Write-Error "Deployment failed. Check the output above for errors."
        Write-Info "To debug, SSH into the server and check logs:"
        Write-ColorOutput "  ssh -p $ServerPort ${Username}@${ServerIP}" "White"
        exit 1
    }
}

Write-Host ""
Write-ColorOutput "==========================================" "Blue"
Write-ColorOutput "Deployment script completed" "Blue"
Write-ColorOutput "==========================================" "Blue"
