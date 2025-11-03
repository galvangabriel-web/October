@echo off
REM ============================================================================
REM Drag & Drop Test Runner for Racing Dashboard
REM ============================================================================
REM
REM This script automates the execution of drag & drop tests for the
REM Racing Analytics Dashboard. It handles all prerequisites and provides
REM clear output.
REM
REM Usage:
REM   run_drag_drop_tests.bat              - Run tests in visual mode
REM   run_drag_drop_tests.bat headless     - Run tests in headless mode (CI/CD)
REM   run_drag_drop_tests.bat help         - Show this help
REM
REM ============================================================================

setlocal enabledelayedexpansion

REM Colors for Windows (requires ANSI support - Windows 10+)
set "GREEN=[92m"
set "YELLOW=[93m"
set "RED=[91m"
set "BLUE=[94m"
set "RESET=[0m"

echo.
echo %BLUE%================================================================%RESET%
echo %BLUE%   Drag ^& Drop Automated Testing - Racing Dashboard%RESET%
echo %BLUE%================================================================%RESET%
echo.

REM Check for help flag
if "%1"=="help" (
    echo Usage:
    echo   run_drag_drop_tests.bat              - Run tests in visual mode
    echo   run_drag_drop_tests.bat headless     - Run tests in headless mode
    echo   run_drag_drop_tests.bat help         - Show this help
    echo.
    echo Test Modes:
    echo   Visual Mode     - Opens browser window, tests run slowly for observation
    echo   Headless Mode   - No browser window, tests run at full speed (CI/CD)
    echo.
    goto :eof
)

REM Step 1: Check Node.js installation
echo %BLUE%[1/6]%RESET% Checking Node.js installation...
where node >nul 2>&1
if %errorlevel% neq 0 (
    echo %RED%ERROR: Node.js not found!%RESET%
    echo.
    echo Please install Node.js from: https://nodejs.org/
    echo Recommended version: 18.x or higher
    echo.
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('node --version') do set NODE_VERSION=%%i
echo %GREEN%   Node.js version: %NODE_VERSION%%RESET%

REM Step 2: Check npm installation
echo %BLUE%[2/6]%RESET% Checking npm installation...
where npm >nul 2>&1
if %errorlevel% neq 0 (
    echo %RED%ERROR: npm not found!%RESET%
    echo.
    echo npm should be installed with Node.js
    echo Please reinstall Node.js from: https://nodejs.org/
    echo.
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('npm --version') do set NPM_VERSION=%%i
echo %GREEN%   npm version: %NPM_VERSION%%RESET%

REM Step 3: Install dependencies if needed
echo %BLUE%[3/6]%RESET% Checking dependencies...
cd /d "%~dp0.."
if not exist "node_modules\puppeteer" (
    echo %YELLOW%   Puppeteer not found. Installing dependencies...%RESET%
    call npm install
    if %errorlevel% neq 0 (
        echo %RED%ERROR: Failed to install dependencies!%RESET%
        pause
        exit /b 1
    )
    echo %GREEN%   Dependencies installed successfully%RESET%
) else (
    echo %GREEN%   Puppeteer already installed%RESET%
)

REM Step 4: Check if dashboard is running
echo %BLUE%[4/6]%RESET% Checking if dashboard is running...
curl -s http://localhost:8050 >nul 2>&1
if %errorlevel% neq 0 (
    echo %YELLOW%   WARNING: Dashboard not detected at http://localhost:8050%RESET%
    echo.
    echo %YELLOW%   The dashboard should be running before tests start.%RESET%
    echo %YELLOW%   Please start it in another terminal:%RESET%
    echo %YELLOW%      python src/dashboard/app.py%RESET%
    echo.
    choice /C YN /M "Continue anyway (tests may fail)"
    if !errorlevel! neq 1 (
        echo %RED%Tests aborted by user%RESET%
        exit /b 0
    )
) else (
    echo %GREEN%   Dashboard is running%RESET%
)

REM Step 5: Check test file exists
echo %BLUE%[5/6]%RESET% Checking test file...
set "TEST_FILE=%~dp0..\master_racing_data.csv"
if not exist "%TEST_FILE%" (
    echo %YELLOW%   WARNING: master_racing_data.csv not found%RESET%
    echo %YELLOW%   Expected location: %TEST_FILE%%RESET%
    echo.
    echo %YELLOW%   Tests may use a different file path or fail.%RESET%
    echo.
    choice /C YN /M "Continue anyway"
    if !errorlevel! neq 1 (
        echo %RED%Tests aborted by user%RESET%
        exit /b 0
    )
) else (
    echo %GREEN%   Test file found: master_racing_data.csv%RESET%
)

REM Step 6: Run tests
echo %BLUE%[6/6]%RESET% Running tests...
echo.

cd /d "%~dp0"

REM Check for headless mode
if "%1"=="headless" (
    echo %YELLOW%Running in HEADLESS mode (no browser window)%RESET%
    echo.
    REM Note: Would need to modify test_drag_drop_automated.js to accept CLI args
    REM For now, user must manually edit CONFIG.headless in the test file
    echo %YELLOW%NOTE: To run in true headless mode, edit test_drag_drop_automated.js:%RESET%
    echo %YELLOW%      Set CONFIG.headless = true%RESET%
    echo.
) else (
    echo %GREEN%Running in VISUAL mode (browser window will be visible)%RESET%
    echo.
)

REM Execute tests
node test_drag_drop_automated.js
set TEST_RESULT=%errorlevel%

echo.
echo %BLUE%================================================================%RESET%

if %TEST_RESULT%==0 (
    echo %GREEN%   ALL TESTS PASSED%RESET%
    echo %BLUE%================================================================%RESET%
    echo.
    echo %GREEN%Test reports generated:%RESET%
    echo   - HTML Report: %~dp0test_report_drag_drop.html
    echo   - JSON Results: %~dp0test_results_drag_drop.json
    echo   - Screenshots: %~dp0screenshots\
    echo.
    echo %GREEN%Open HTML report:%RESET%
    echo   file:///%~dp0test_report_drag_drop.html
    echo.

    REM Ask to open report
    choice /C YN /M "Open HTML report in browser"
    if !errorlevel!==1 (
        start "" "%~dp0test_report_drag_drop.html"
    )
) else (
    echo %RED%   TESTS FAILED%RESET%
    echo %BLUE%================================================================%RESET%
    echo.
    echo %RED%Check the reports for details:%RESET%
    echo   - HTML Report: %~dp0test_report_drag_drop.html
    echo   - JSON Results: %~dp0test_results_drag_drop.json
    echo   - Screenshots: %~dp0screenshots\
    echo.

    REM Ask to open report
    choice /C YN /M "Open HTML report to see errors"
    if !errorlevel!==1 (
        start "" "%~dp0test_report_drag_drop.html"
    )
)

echo.
pause
exit /b %TEST_RESULT%
