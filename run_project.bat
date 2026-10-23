@echo off
setlocal enabledelayedexpansion
title Fine-Grained Encrypted Traffic Classification using GNN (GraphSAGE)

echo ===============================================================================
echo  FINE-GRAINED ENCRYPTED TRAFFIC CLASSIFICATION (GraphSAGE / GAT)
echo  Windows Automated Research Launcher
echo ===============================================================================
echo.
echo  1. Run End-to-End Experiment: Fine-Grained Application Classification
echo  2. Run End-to-End Experiment: Binary VPN vs Non-VPN
echo  3. Launch Demonstration Interface (GUI)
echo  4. Launch Demonstration Interface (CLI Sample Mode)
echo  5. Run Complete Test Suite
echo  6. View Public Benchmark Datasets Catalog (ISCXVPN2016 / USTC-TFC2016)
echo  7. Exit
echo.

set /p choice="Select an option [1-7]: "

if "%choice%"=="1" (
    echo.
    echo [*] Starting Fine-Grained Application Classification Experiment...
    python -m src.pipeline --experiment fine_grained_app
    pause
    goto end
)

if "%choice%"=="2" (
    echo.
    echo [*] Starting Binary VPN vs Non-VPN Experiment...
    python -m src.pipeline --experiment binary_vpn
    pause
    goto end
)

if "%choice%"=="3" (
    echo.
    echo [*] Launching Native Windows GUI...
    python app.py
    goto end
)

if "%choice%"=="4" (
    echo.
    echo [*] Running CLI Demonstration...
    python app.py --cli
    pause
    goto end
)

if "%choice%"=="5" (
    echo.
    echo [*] Running Unit Test Suite...
    python -m unittest discover tests -v
    pause
    goto end
)

if "%choice%"=="6" (
    echo.
    python -m src.pipeline --show_catalog
    pause
    goto end
)

if "%choice%"=="7" (
    goto end
)

:end
echo Exiting.
