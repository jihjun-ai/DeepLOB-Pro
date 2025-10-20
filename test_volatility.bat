@echo off
chcp 65001 > nul
call conda activate deeplob-pro
python scripts\quick_verify_volatility.py
pause
