@echo off
REM ============================================================================
REM GitHub Deployment Script - GR Cup Racing Analytics
REM ============================================================================

echo.
echo ============================================================================
echo   GR CUP RACING ANALYTICS - GitHub Deployment
echo ============================================================================
echo.

echo [1/6] Checking git status...
git status --short
echo.

echo [2/6] Staging source code...
git add src/
echo     Source code staged!

echo [3/6] Staging models and documentation...
git add data/models/
git add tests/
git add deployment/
git add index.html
git add README.md
git add .gitignore
git add GITHUB_DEPLOYMENT.txt
echo     Models and documentation staged!

echo [4/6] Staging core utilities...
git add requirements.txt
git add setup.bat setup.sh
git add data_loader.py
git add analyze_all_data.py
git add inventory_data.py
echo     Core utilities staged!

echo [5/6] Review staged changes:
git status
echo.

echo [6/6] Ready to commit and push!
echo.
echo Next steps:
echo   1. Review staged changes above
echo   2. Run: git commit -m "feat: GitHub deployment - source code, models, HTML docs"
echo   3. Run: git push -u origin main
echo.
echo ============================================================================
echo   Files staged successfully!
echo   See GITHUB_DEPLOYMENT.txt for complete deployment guide
echo ============================================================================
echo.

pause
