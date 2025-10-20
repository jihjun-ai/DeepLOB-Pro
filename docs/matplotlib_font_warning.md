# Matplotlib 中文字體警告說明

## 警告訊息

```
UserWarning: Glyph 39006 (\N{CJK UNIFIED IDEOGRAPH-985E}) missing from current font.
```

## 這是什麼？

這是 **matplotlib 中文字體警告**，表示：
- matplotlib 預設字體不支援中文字符
- 嘗試渲染中文時找不到對應字形
- **不影響程式執行**，只影響圖表中文顯示

## 是否正常？

✅ **完全正常**，這只是字體問題，不是程式錯誤。

### 影響範圍
- ❌ **不影響**：程式執行、資料處理、報告生成
- ⚠️ **輕微影響**：圖表中的中文可能顯示為方塊 `▯`

## 已採取的解決方案

`check_data_health_v5.py` v1.1 已加入：

1. **自動偵測中文字體**：
   ```python
   mpl.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS']
   ```

2. **圖表使用英文標籤**：
   - 原本：「標籤分布」→ 改為：「Label Distribution」
   - 原本：「權重分布」→ 改為：「Weight Distribution」
   - 避免字體問題，提高兼容性

## 如何完全消除警告？

### 方法一：使用 `--plot` 時自動處理（推薦）
腳本已自動設定中文字體，Windows 系統會優先使用：
1. **Microsoft JhengHei**（微軟正黑體）- Windows 10/11 預設
2. **SimHei**（黑體）- 簡體中文系統
3. **Arial Unicode MS** - 備用字體

### 方法二：手動設定全域字體（進階）

```python
# 在任何繪圖前執行
import matplotlib.pyplot as plt
import matplotlib as mpl

# Windows 系統
mpl.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
mpl.rcParams['axes.unicode_minus'] = False

# macOS 系統
mpl.rcParams['font.sans-serif'] = ['PingFang HK', 'Arial Unicode MS']

# Linux 系統
mpl.rcParams['font.sans-serif'] = ['Noto Sans CJK TC', 'WenQuanYi Micro Hei']
```

### 方法三：安裝中文字體包（Linux）

```bash
# Ubuntu/Debian
sudo apt-get install fonts-noto-cjk

# CentOS/RHEL
sudo yum install google-noto-sans-cjk-tc-fonts

# 重建 matplotlib 字體快取
rm -rf ~/.cache/matplotlib
```

## 驗證字體支援

```python
import matplotlib.font_manager as fm

# 列出所有可用字體
fonts = [f.name for f in fm.fontManager.ttflist]
print("可用中文字體:")
for font in fonts:
    if any(keyword in font for keyword in ['Hei', 'Sung', 'Ming', 'Kai', 'CJK', 'Unicode']):
        print(f"  - {font}")
```

## 當前狀態（v1.1）

✅ **已解決**：
- 圖表標籤改用英文（避免字體問題）
- 自動設定中文字體（Windows 系統）
- 警告訊息不影響功能

⚠️ **已知限制**：
- Linux 系統可能仍需手動安裝中文字體
- 某些極舊的 Windows 版本可能缺少微軟正黑體

## 建議

### 一般用戶
- **忽略警告**：不影響資料檢查功能
- **不使用 `--plot`**：只生成 JSON 報告即可
  ```bash
  python scripts/check_data_health_v5.py --data-dir ./data/processed_v5/npz
  # 不加 --plot 就不會生成圖表，也就沒有字體警告
  ```

### 需要視覺化的用戶
- **使用最新版本**（v1.1+）：已改用英文標籤
- **Windows 用戶**：直接使用，自動處理
- **Linux/macOS 用戶**：安裝中文字體包（可選）

## 範例輸出

### 無警告（v1.1+）
```bash
$ python scripts/check_data_health_v5.py --data-dir ./data/processed_v5/npz --plot

======================================================================
6. 視覺化 (Visualizations)
======================================================================

✅ 視覺化已保存: ./data/processed_v5/npz/health_visualizations.png
```

### 舊版警告（v1.0）
```bash
# 舊版可能出現的警告（已修復）
UserWarning: Glyph 39006 (\N{CJK UNIFIED IDEOGRAPH-985E}) missing from current font.
```

## 總結

| 問題 | 嚴重性 | 是否需要處理 |
|------|--------|--------------|
| 警告訊息 | 🟢 低 | 否（已自動處理） |
| 圖表中文 | 🟡 中 | 可選（改用英文） |
| 程式功能 | 🟢 無影響 | 否 |

**結論**：這是正常的字體相容性警告，v1.1 已改用英文標籤，可安全忽略。

---

**更新日期**：2025-10-20
**版本**：v1.1
**狀態**：✅ 已解決
