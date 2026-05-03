@echo off
title Load Shedding Optimizer - Server
echo ========================================
echo Building Load Shedding Optimizer...
echo ========================================
g++ main.cpp -o main -std=c++14 -O2 -lws2_32 -w
if errorlevel 1 (
    echo.
    echo BUILD FAILED.
    pause
    exit /b 1
)
echo.
echo Build OK. Starting server...
echo Open http://localhost:8080 in your browser.
echo Press Ctrl+C in this window to stop the server.
echo ========================================
.\main --serve
pause