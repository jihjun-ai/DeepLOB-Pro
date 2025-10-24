# DeepLOB-Pro 後續階段路線圖

**日期**: 2025-10-24
**當前狀態**: ✅ 階段一（DeepLOB 訓練）已完成
**目標**: 完成台股高頻交易系統（DeepLOB + Stable-Baselines3）

---

## 📍 當前進度總覽

### 已完成 ✅

| 階段 | 任務 | 狀態 | 完成日期 |
|------|------|------|---------|
| **階段一** | **DeepLOB 模型訓練** | ✅ **完成** | 2025-10-24 |
| └─ 數據預處理 | V7 雙階段處理 | ✅ | 2025-10-24 |
| └─ 模型訓練 | 實驗 1-6 調參 | ✅ | 2025-10-24 |
| └─ 性能驗證 | 測試集評估 | ✅ | 2025-10-24 |
| └─ 文檔記錄 | 調參歷史完整 | ✅ | 2025-10-24 |

### 待完成 ⏳

| 階段 | 預計時間 | 優先級 |
|------|---------|--------|
| **階段二** | 1-2 週 | 🔥 高 |
| **階段三** | 1 週 | 🔥 高 |
| **階段四** | 1-2 週 | 中 |
| **階段五** | 持續 | 中 |

---

## 🎯 階段二：SB3 強化學習整合（優先執行）

### 目標
將預訓練 DeepLOB 模型整合到 Stable-Baselines3 PPO 算法，訓練交易策略。

### 核心任務

#### 2.1 環境驗證 ✅（已完成）
- [x] 驗證 `TaiwanLOBTradingEnv` 符合 Gymnasium 標準
- [x] 確認觀測空間（28 維）與動作空間（3 動作）
- [x] 運行 `scripts/verify_env.py` 通過所有檢查

**狀態**: ✅ 已完成（參考 CLAUDE.md 階段三）

---

#### 2.2 基礎 PPO 訓練 ✅（已完成）
- [x] 實現 `scripts/train_sb3.py`
- [x] 使用 `MlpPolicy` 快速測試
- [x] 驗證訓練管線完整性

**狀態**: ✅ 已完成（參考 CLAUDE.md 階段四）

---

#### 2.3 整合 DeepLOB 特徵提取器 ✅（已完成）

**已完成文件**:
- ✅ `src/models/deeplob_feature_extractor.py` - DeepLOB 特徵提取器
- ✅ `scripts/train_sb3_deeplob.py` - 完整訓練腳本

**核心設計**:
```python
# DeepLOB 作為凍結的特徵提取器
class DeepLOBExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=64):
        self.deeplob = load_pretrained_deeplob()  # 載入最佳檢查點
        self.deeplob.eval()  # 凍結權重
        for param in self.deeplob.parameters():
            param.requires_grad = False
```

**狀態**: ✅ 已完成（參考 CLAUDE.md 階段五）

---

#### 2.4 完整訓練與評估 ⏳（待執行）

**訓練指令**:
```bash
# 完整訓練（1M steps，推薦）
conda activate deeplob-pro
python scripts/train_sb3_deeplob.py \
    --deeplob-checkpoint checkpoints/v5/deeplob_v5_best.pth \
    --timesteps 1000000 \
    --save-freq 50000

# 監控訓練
tensorboard --logdir logs/sb3_deeplob/
```

**預期輸出**:
- 檢查點: `checkpoints/sb3/ppo_deeplob/best_model.zip`
- 日誌: `logs/sb3_deeplob/`
- 評估報告: `results/sb3_eval/`

**驗收標準**:
- [ ] 訓練完成 1M steps 無錯誤
- [ ] Sharpe Ratio > 1.5（目標 > 2.0）
- [ ] 平均收益 > 0（正收益）
- [ ] GPU 利用率 > 70%

**預計時間**: 4-8 小時（RTX 5090）

---

#### 2.5 評估與分析 ⏳（待執行）

**評估指令**:
```bash
# 評估最佳模型
python scripts/evaluate_sb3.py \
    --model checkpoints/sb3/ppo_deeplob/best_model \
    --n-episodes 20 \
    --save-report

# 查看報告
cat results/sb3_eval/evaluation_report.json
```

**評估指標**（7 大類）:
1. **收益指標**: 總收益、收益率、Sharpe Ratio
2. **風險指標**: 最大回撤、勝率
3. **交易統計**: 交易次數、成本、持倉時間
4. **性能對比**: vs Buy-and-Hold, vs Random Policy

**驗收標準**:
- [ ] Sharpe Ratio > 1.5
- [ ] Max Drawdown < 15%
- [ ] 勝率 > 45%
- [ ] 交易成本 < 總收益 20%

**預計時間**: 1-2 小時

---

### 階段二總結

| 任務 | 狀態 | 預計時間 |
|------|------|---------|
| 2.1 環境驗證 | ✅ 完成 | - |
| 2.2 基礎訓練 | ✅ 完成 | - |
| 2.3 DeepLOB 整合 | ✅ 完成 | - |
| 2.4 完整訓練 | ⏳ **待執行** | 4-8 小時 |
| 2.5 評估分析 | ⏳ **待執行** | 1-2 小時 |
| **總計** | **60% 完成** | **5-10 小時** |

**關鍵文件**:
- DeepLOB 檢查點: `checkpoints/v5/deeplob_v5_best.pth` ✅
- 訓練腳本: `scripts/train_sb3_deeplob.py` ✅
- 評估腳本: `scripts/evaluate_sb3.py` ✅
- 環境定義: `src/envs/tw_lob_trading_env.py` ✅

---

## 🎯 階段三：超參數優化

### 目標
通過系統性調參提升強化學習策略性能。

### 3.1 手動調參 ⏳

**優化方向**:

| 參數 | 當前值 | 建議範圍 | 目標 |
|------|--------|---------|------|
| **學習率** | 3e-4 | 1e-4 ~ 1e-3 | 找到最優收斂速度 |
| **Gamma** | 0.99 | 0.95 ~ 0.995 | 平衡短期/長期收益 |
| **Entropy 係數** | 0.01 | 0.001 ~ 0.05 | 控制探索程度 |
| **N Steps** | 2048 | 1024 ~ 4096 | 平衡訓練速度/穩定性 |
| **Batch Size** | 64 | 32 ~ 256 | GPU 利用率最大化 |

**實驗策略**:
1. 建立基線（當前配置）
2. 每次調整 1-2 個參數
3. 記錄 Sharpe Ratio 與訓練時間
4. 選擇 Top 3 配置組合測試

**驗收標準**:
- [ ] Sharpe Ratio 提升 > 10%
- [ ] 找到穩定配置（3 次運行標準差 <5%）

**預計時間**: 3-5 天（5-10 次實驗）

---

### 3.2 獎勵函數調整 ⏳

**當前獎勵函數**:
```python
reward = pnl - transaction_cost - inventory_penalty - risk_penalty
```

**優化方向**:

| 組件 | 當前 | 建議調整 |
|------|------|---------|
| **PnL 權重** | 1.0 | 0.8 ~ 1.2 |
| **交易成本** | -0.001 | -0.0005 ~ -0.002 |
| **庫存懲罰** | -0.01 × position² | 調整係數 |
| **風險調整** | -0.05 × volatility | 調整係數 |

**新增組件**（可選）:
- [ ] 連續持倉獎勵（減少過度交易）
- [ ] 趨勢跟隨獎勵（配合 DeepLOB 預測）
- [ ] 風險調整收益（Sharpe Ratio 導向）

**驗收標準**:
- [ ] 交易頻率合理（10-30 次/episode）
- [ ] 持倉利用率 > 50%
- [ ] 風險調整後收益提升 > 15%

**預計時間**: 2-3 天

---

### 3.3 自動化調參（可選）⏳

**工具**: Optuna

**實現步驟**:
1. 定義搜索空間（6-8 個關鍵參數）
2. 設置優化目標（Sharpe Ratio）
3. 運行 50-100 次試驗
4. 選擇 Top 5 配置驗證

**配置文件**: `configs/sb3_optuna_search.yaml`

**預計時間**: 1-2 天（自動運行）

---

### 階段三總結

| 任務 | 優先級 | 預計時間 |
|------|--------|---------|
| 3.1 手動調參 | 🔥 高 | 3-5 天 |
| 3.2 獎勵函數 | 🔥 高 | 2-3 天 |
| 3.3 自動調參 | 中 | 1-2 天（可選）|
| **總計** | - | **5-10 天** |

---

## 🎯 階段四：回測與部署

### 4.1 回測系統開發 ⏳

**目標**: 在歷史數據上驗證策略表現

**核心功能**:
- [ ] 滑點模擬（0.01-0.03%）
- [ ] 市場衝擊成本
- [ ] 多時段回測（早盤/午盤/尾盤）
- [ ] 風險指標計算

**輸出報告**:
```
回測期間: 2024-01-01 ~ 2024-12-31
總收益率: +35.6%
年化收益: +42.3%
Sharpe Ratio: 2.15
Max Drawdown: -8.3%
勝率: 52.7%
平均持倉: 3.2 小時
```

**驗收標準**:
- [ ] 回測系統完整實現
- [ ] 年化收益 > 30%
- [ ] Sharpe Ratio > 2.0
- [ ] Max Drawdown < 15%

**預計時間**: 3-5 天

---

### 4.2 模型壓縮與優化 ⏳

**優化方向**:

| 技術 | 目標 | 預期效果 |
|------|------|---------|
| **TorchScript** | 序列化模型 | 推理加速 30-50% |
| **量化 (FP16)** | 減少顯存 | 顯存減少 50% |
| **批次推理** | 向量化 | 吞吐量提升 2-3x |

**實現步驟**:
1. 轉換 DeepLOB 為 TorchScript
2. 量化權重 FP32 → FP16
3. 測試精度損失 (<1%)
4. 基準測試推理速度

**驗收標準**:
- [ ] 推理速度 > 1000 samples/sec
- [ ] 精度損失 < 1%
- [ ] 顯存使用 < 4GB

**預計時間**: 2-3 天

---

### 4.3 生產部署準備 ⏳

**部署架構**:
```
數據源 (LOB Feed)
    ↓
預處理模組 (標準化)
    ↓
DeepLOB 預測 (GPU)
    ↓
PPO 策略決策 (CPU/GPU)
    ↓
訂單執行引擎
    ↓
風控模組 (停損/停利)
```

**核心模組**:
- [ ] 實時數據接口
- [ ] 推理服務器（FastAPI）
- [ ] 訂單管理系統
- [ ] 監控與告警

**驗收標準**:
- [ ] 端到端延遲 < 50ms
- [ ] 系統穩定性 > 99.9%
- [ ] 錯誤恢復機制完整

**預計時間**: 5-7 天

---

### 階段四總結

| 任務 | 優先級 | 預計時間 |
|------|--------|---------|
| 4.1 回測系統 | 🔥 高 | 3-5 天 |
| 4.2 模型優化 | 中 | 2-3 天 |
| 4.3 部署準備 | 中 | 5-7 天 |
| **總計** | - | **10-15 天** |

---

## 🎯 階段五：持續監控與改進

### 5.1 性能監控 ⏳

**監控指標**:
- 每日收益率
- Sharpe Ratio（滾動 30 天）
- 交易成本佔比
- GPU 利用率
- 推理延遲

**工具**:
- TensorBoard（訓練監控）
- Grafana + Prometheus（生產監控）
- Jupyter Notebook（分析報告）

**預計時間**: 持續

---

### 5.2 模型更新策略 ⏳

**更新觸發條件**:
- 性能下降 > 10%（連續 7 天）
- 市場環境劇變（波動率翻倍）
- 新數據累積 > 3 個月

**更新流程**:
1. 收集新數據
2. 重新訓練 DeepLOB
3. 微調 PPO 策略
4. A/B 測試驗證
5. 灰度發布

**預計時間**: 每季度 1-2 次

---

### 5.3 錯誤分析與改進 ⏳

**分析維度**:
- 錯誤交易案例（虧損 > 5%）
- 漏失機會（應買未買）
- 市場環境分類（牛市/熊市/震盪）

**改進方向**:
- 調整決策閾值
- 增加市場狀態特徵
- 優化風控參數

**預計時間**: 持續

---

## 📅 總體時間規劃

| 階段 | 預計時間 | 依賴關係 | 開始日期 |
|------|---------|---------|---------|
| ✅ 階段一 | - | - | 已完成 |
| ⏳ **階段二** | **5-10 小時** | 階段一 | **立即開始** 🔥 |
| ⏳ 階段三 | 5-10 天 | 階段二 | 2025-10-25 |
| ⏳ 階段四 | 10-15 天 | 階段三 | 2025-11-05 |
| ⏳ 階段五 | 持續 | 階段四 | 2025-11-20 |

**預計完成時間**: 2025-11-20（全功能生產系統）

---

## 🚀 立即行動清單

### 本週重點（2025-10-24 ~ 2025-10-30）

#### Day 1（今天）✅
- [x] 完成 DeepLOB 調參與驗證
- [x] 生成路線圖文檔
- [ ] 準備 SB3 訓練環境

#### Day 2-3 ⏳
- [ ] 執行 SB3 完整訓練（1M steps）
- [ ] 監控訓練過程（TensorBoard）
- [ ] 記錄訓練指標

#### Day 4-5 ⏳
- [ ] 評估訓練結果
- [ ] 分析交易行為
- [ ] 調整基礎參數

#### Day 6-7 ⏳
- [ ] 開始超參數優化
- [ ] 測試獎勵函數變體
- [ ] 記錄實驗結果

---

## 📋 檢查清單

### 階段二開始前確認 ✅

- [x] DeepLOB 最佳檢查點存在: `checkpoints/v5/deeplob_v5_best.pth`
- [x] 測試集評估完成: Test Acc 50.01%, F1 0.4891
- [x] 訓練腳本準備好: `scripts/train_sb3_deeplob.py`
- [x] 評估腳本準備好: `scripts/evaluate_sb3.py`
- [x] 環境驗證通過: `scripts/verify_env.py`
- [x] GPU 可用: RTX 5090 (32GB VRAM)
- [x] 數據準備好: `data/processed_v7/npz/`

### 階段三開始前確認 ⏳

- [ ] SB3 基線訓練完成（Sharpe Ratio 記錄）
- [ ] 評估報告生成（7 大指標）
- [ ] 訓練日誌完整（TensorBoard）
- [ ] 檢查點保存正常

### 階段四開始前確認 ⏳

- [ ] 最優超參數確定
- [ ] 策略性能穩定（3 次運行標準差 <5%）
- [ ] Sharpe Ratio > 2.0

---

## 📚 參考文檔

### 核心文檔
- **專案概述**: [CLAUDE.md](CLAUDE.md) - 完整專案指南
- **調參歷史**: [docs/20251024-deeplob調參歷史.md](docs/20251024-deeplob調參歷史.md) - 實驗記錄
- **SB3 實作**: [docs/SB3_IMPLEMENTATION_REPORT.md](docs/SB3_IMPLEMENTATION_REPORT.md) - 階段 3-7 報告

### 技術文檔
- **數據處理**: [docs/V6_TWO_STAGE_PIPELINE_GUIDE.md](docs/V6_TWO_STAGE_PIPELINE_GUIDE.md)
- **環境定義**: [src/envs/tw_lob_trading_env.py](src/envs/tw_lob_trading_env.py)
- **特徵提取器**: [src/models/deeplob_feature_extractor.py](src/models/deeplob_feature_extractor.py)

### 外部資源
- Stable-Baselines3 文檔: https://stable-baselines3.readthedocs.io/
- PPO 論文: "Proximal Policy Optimization Algorithms" (Schulman et al., 2017)
- DeepLOB 論文: "Deep Convolutional Neural Networks for Limit Order Books" (Zhang et al., 2019)

---

## 💡 關鍵提醒

### 成功因素
1. ✅ **DeepLOB 已達生產級別** - Test Acc 50.01%, 可重現性 <0.1%
2. ✅ **代碼框架完整** - 所有核心腳本已實現
3. ✅ **硬體充足** - RTX 5090 足夠支撐訓練
4. ⚠️ **需要時間驗證** - 強化學習需要充分訓練（1M+ steps）

### 風險因素
1. ⚠️ **強化學習不穩定** - 可能需要多次調參
2. ⚠️ **獎勵函數設計** - 決定策略行為
3. ⚠️ **過擬合風險** - 需要充分驗證泛化性
4. ⚠️ **市場環境變化** - 歷史性能不保證未來收益

### 應對策略
1. 📊 **保守評估** - Sharpe Ratio > 1.5 即可接受
2. 🔍 **充分驗證** - 多時段、多市場環境測試
3. 📈 **持續監控** - 部署後密切關注性能
4. 🛡️ **風控優先** - 嚴格停損/停利機制

---

## 📞 問題與支援

### 常見問題

**Q: 階段二訓練需要多久？**
A: RTX 5090 上約 4-8 小時（1M steps），可先用 10K steps 快速測試（5-10 分鐘）。

**Q: 如何判斷訓練是否成功？**
A: 檢查 TensorBoard 中 `rollout/ep_rew_mean` 是否持續上升，目標正收益。

**Q: Sharpe Ratio 多少算合格？**
A: 1.5+ 即可接受，2.0+ 為優秀，3.0+ 為卓越。

**Q: 如果訓練不穩定怎麼辦？**
A: 降低學習率（3e-4 → 1e-4），增加 batch size，調整獎勵函數權重。

**Q: 需要重新訓練 DeepLOB 嗎？**
A: 不需要，當前檢查點（50.01% Test Acc）已達標，直接用於特徵提取。

---

**最後更新**: 2025-10-24
**版本**: v1.0
**狀態**: 🚀 準備開始階段二

---

# 快速開始

```bash
# 1. 激活環境
conda activate deeplob-pro

# 2. 驗證環境（可選）
python scripts/verify_env.py

# 3. 開始訓練（推薦）
python scripts/train_sb3_deeplob.py \
    --deeplob-checkpoint checkpoints/v5/deeplob_v5_best.pth \
    --timesteps 1000000 \
    --save-freq 50000

# 4. 監控訓練
tensorboard --logdir logs/sb3_deeplob/

# 5. 評估結果
python scripts/evaluate_sb3.py \
    --model checkpoints/sb3/ppo_deeplob/best_model \
    --n-episodes 20 \
    --save-report
```

**Good Luck! 🚀**
