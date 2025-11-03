@echo off
REM SSH Helper Script - Automates SSH connections to racing dashboard server
REM Usage: ssh_helper.bat [command]

SET SERVER=tactical@200.58.107.214
SET PORT=5197
SET PASSWORD=1253*1253*Win1

REM Check if command argument provided
IF "%~1"=="" (
    echo Connecting to server...
    echo Password: %PASSWORD%
    echo.
    echo Run: ssh -p %PORT% %SERVER%
    echo Then paste password: %PASSWORD%
    pause
    ssh -p %PORT% %SERVER%
) ELSE (
    echo Running command: %*
    echo.
    ssh -p %PORT% %SERVER% "%*"
)
