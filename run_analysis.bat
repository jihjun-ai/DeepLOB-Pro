@echo off
call conda activate deeplob-pro
python scripts\quick_analyze.py > analysis_output.txt 2>&1
type analysis_output.txt
