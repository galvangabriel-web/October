@echo off
REM ============================================================================
REM Test Setup Verification Script
REM ============================================================================
REM This script checks if all prerequisites for drag & drop testing are met.
REM Run this BEFORE executing tests to ensure everything is configured.
REM ============================================================================

setlocal enabledelayedexpansion

set "GREEN=[92m"
set "YELLOW=[93m"
set "RED=[91m"
set "BLUE=[94m"
set "RESET=[0m"

echo.
echo %BLUE%================================================================%RESET%
echo %BLUE%   Test Setup Verification - Drag ^& Drop Testing%RESET%
echo %BLUE%================================================================%RESET%
echo.

set PASS_COUNT=0
set FAIL_COUNT=0
set WARN_COUNT=0

REM Check 1: Node.js
echo %BLUE%[1/8]%RESET% Checking Node.js...
where node >nul 2>&1
if %errorlevel%==0 (
    for /f "tokens=*" %%i in ('node --version') do set NODE_VERSION=%%i
    echo %GREEN%   ✓ Node.js found: !NODE_VERSION!%RESET%
    set /a PASS_COUNT+=1
) else (
    echo %RED%   ✗ Node.js NOT found%RESET%
    echo %RED%     Install from: https://nodejs.org/%RESET%
    set /a FAIL_COUNT+=1
)

REM Check 2: npm
echo %BLUE%[2/8]%RESET% Checking npm...
where npm >nul 2>&1
if %errorlevel%==0 (
    for /f "tokens=*" %%i in ('npm --version') do set NPM_VERSION=%%i
    echo %GREEN%   ✓ npm found: !NPM_VERSION!%RESET%
    set /a PASS_COUNT+=1
) else (
    echo %RED%   ✗ npm NOT found%RESET%
    set /a FAIL_COUNT+=1
)

REM Check 3: Puppeteer
echo %BLUE%[3/8]%RESET% Checking Puppeteer...
cd /d "%~dp0.."
npm list puppeteer >nul 2>&1
if %errorlevel%==0 (
    for /f "tokens=2" %%i in ('npm list puppeteer ^| findstr puppeteer') do set PUPPETEER_VERSION=%%i
    echo %GREEN%   ✓ Puppeteer installed%RESET%
    set /a PASS_COUNT+=1
) else (
    echo %YELLOW%   ! Puppeteer NOT installed%RESET%
    echo %YELLOW%     Run: npm install%RESET%
    set /a WARN_COUNT+=1
)

REM Check 4: Test script exists
echo %BLUE%[4/8]%RESET% Checking test script...
if exist "%~dp0test_drag_drop_automated.js" (
    for %%A in ("%~dp0test_drag_drop_automated.js") do set TEST_SIZE=%%~zA
    echo %GREEN%   ✓ test_drag_drop_automated.js found (%%~zA bytes)%RESET%
    set /a PASS_COUNT+=1
) else (
    echo %RED%   ✗ test_drag_drop_automated.js NOT found%RESET%
    set /a FAIL_COUNT+=1
)

REM Check 5: Documentation exists
echo %BLUE%[5/8]%RESET% Checking documentation...
set DOC_COUNT=0
if exist "%~dp0README_DRAG_DROP_TESTING.md" set /a DOC_COUNT+=1
if exist "%~dp0QUICK_START_DRAG_DROP_TESTING.md" set /a DOC_COUNT+=1
if exist "%~dp0TEST_EXECUTION_SUMMARY.md" set /a DOC_COUNT+=1
if !DOC_COUNT! geq 2 (
    echo %GREEN%   ✓ Documentation files found (!DOC_COUNT!/3)%RESET%
    set /a PASS_COUNT+=1
) else (
    echo %YELLOW%   ! Some documentation missing (!DOC_COUNT!/3)%RESET%
    set /a WARN_COUNT+=1
)

REM Check 6: Test file exists
echo %BLUE%[6/8]%RESET% Checking test data file...
if exist "%~dp0..\master_racing_data.csv" (
    for %%A in ("%~dp0..\master_racing_data.csv") do (
        set /a SIZE_KB=%%~zA/1024
        echo %GREEN%   ✓ master_racing_data.csv found (!SIZE_KB! KB)%RESET%
    )
    set /a PASS_COUNT+=1
) else (
    echo %YELLOW%   ! master_racing_data.csv NOT found%RESET%
    echo %YELLOW%     Tests may fail or need configuration%RESET%
    set /a WARN_COUNT+=1
)

REM Check 7: Dashboard availability
echo %BLUE%[7/8]%RESET% Checking dashboard...
curl -s http://localhost:8050 >nul 2>&1
if %errorlevel%==0 (
    echo %GREEN%   ✓ Dashboard is running (http://localhost:8050)%RESET%
    set /a PASS_COUNT+=1
) else (
    echo %YELLOW%   ! Dashboard NOT running%RESET%
    echo %YELLOW%     Start with: python src/dashboard/app.py%RESET%
    set /a WARN_COUNT+=1
)

REM Check 8: Python availability (for dashboard)
echo %BLUE%[8/8]%RESET% Checking Python...
where python >nul 2>&1
if %errorlevel%==0 (
    for /f "tokens=*" %%i in ('python --version') do set PYTHON_VERSION=%%i
    echo %GREEN%   ✓ Python found: !PYTHON_VERSION!%RESET%
    set /a PASS_COUNT+=1
) else (
    echo %YELLOW%   ! Python NOT found (needed for dashboard)%RESET%
    set /a WARN_COUNT+=1
)

REM Summary
echo.
echo %BLUE%================================================================%RESET%
echo %BLUE%   SUMMARY%RESET%
echo %BLUE%================================================================%RESET%
echo.
echo %GREEN%   ✓ Passed: %PASS_COUNT%/8%RESET%
if %WARN_COUNT% gtr 0 (
    echo %YELLOW%   ! Warnings: %WARN_COUNT%%RESET%
)
if %FAIL_COUNT% gtr 0 (
    echo %RED%   ✗ Failed: %FAIL_COUNT%%RESET%
)
echo.

REM Recommendations
if %FAIL_COUNT% gtr 0 (
    echo %RED%CRITICAL ISSUES FOUND - Tests will likely fail!%RESET%
    echo.
    echo %YELLOW%Action Required:%RESET%
    where node >nul 2>&1 || echo   1. Install Node.js from https://nodejs.org/
    if not exist "%~dp0test_drag_drop_automated.js" echo   2. Verify test files are in correct location
    echo.
    exit /b 1
)

if %WARN_COUNT% gtr 0 (
    echo %YELLOW%WARNINGS FOUND - Tests may fail or require setup%RESET%
    echo.
    echo %YELLOW%Recommended Actions:%RESET%
    npm list puppeteer >nul 2>&1 || echo   1. Install Puppeteer: npm install
    if not exist "%~dp0..\master_racing_data.csv" echo   2. Ensure test data file exists or update test config
    curl -s http://localhost:8050 >nul 2>&1 || echo   3. Start dashboard: python src/dashboard/app.py
    echo.
    echo %YELLOW%Continue with tests? Most issues are non-critical.%RESET%
    choice /C YN /M "Proceed"
    if !errorlevel! neq 1 exit /b 0
)

echo %GREEN%✓ All critical checks passed!%RESET%
echo.
echo %BLUE%Ready to run tests!%RESET%
echo.
echo %GREEN%Run tests with:%RESET%
echo   - run_drag_drop_tests.bat (Windows batch script)
echo   - node test_drag_drop_automated.js (Direct)
echo   - npm run test:drag-drop (npm script)
echo.
echo %BLUE%Documentation:%RESET%
echo   - README_DRAG_DROP_TESTING.md (comprehensive guide)
echo   - QUICK_START_DRAG_DROP_TESTING.md (quick reference)
echo   - TEST_EXECUTION_SUMMARY.md (detailed summary)
echo.

pause
exit /b 0
