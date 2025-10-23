@echo off
REM ============================================================================
REM 測試 Label Viewer 預處理數據查看器 (app_preprocessed.py v4.0)
REM ============================================================================

echo ========================================
echo 測試 Label Viewer 預處理數據查看器
echo ========================================
echo.

REM 激活 Conda 環境
echo [1/3] 激活 Conda 環境...
call conda activate deeplob-pro
if errorlevel 1 (
    echo [錯誤] 無法激活 deeplob-pro 環境
    pause
    exit /b 1
)
echo [OK] Conda 環境激活成功
echo.

REM 檢查依賴
echo [2/3] 檢查依賴套件...
python -c "import dash; import plotly; print('[OK] Dash 和 Plotly 已安裝')"
if errorlevel 1 (
    echo [錯誤] 缺少必要套件，請先安裝:
    echo pip install dash plotly
    pause
    exit /b 1
)
echo.

REM 啟動應用
echo [3/3] 啟動 Label Viewer...
echo.
echo ========================================
echo 應用啟動中...
echo ========================================
echo.
echo 瀏覽器將自動打開以下網址:
echo    http://localhost:8051
echo.
echo 使用說明:
echo    1. 輸入日期目錄路徑（例如: data/preprocessed_v5_1hz/daily/20250901）
echo    2. 點擊「載入目錄」按鈕
echo    3. 選擇股票（個股或全部股票）
echo    4. 勾選要顯示的圖表類型
echo.
echo 支援的圖表:
echo    - 中間價折線圖 (mids)
echo    - LOB 特徵矩陣 (features)
echo    - 標籤陣列圖 (labels)
echo    - 事件數量圖 (bucket_event_count)
echo    - 時間桶遮罩圖 (bucket_mask)
echo    - 標籤預覽分布
echo    - 元數據表格
echo.
echo 按 Ctrl+C 停止應用
echo ========================================
echo.

REM 切換到正確目錄並啟動
cd /d "%~dp0..\label_viewer"
python app_preprocessed.py

pause
