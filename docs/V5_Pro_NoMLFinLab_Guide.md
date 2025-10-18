
# V4 → V5 Pro 重構：**不用 mlfinlab**，以 `triple-barrier` + `arch` + `skfolio` + `vectorbt` + `nautilus-trader` 完成專業化資料管線

> 目的：把 V4 的固定 k 步標籤與簡單波動率，全面升級成「資訊條形 + 波動自適應 + Triple-Barrier（PyPI 版）+ Purged/CPCV + 權重化學習 + 可回測驗證」，並與 **DeepLOB** 訓練無縫銜接。**不使用 mlfinlab**。

---

## 0) 成果與範圍
- **取代模塊**：
  - 標籤：改用 `triple-barrier`（PyPI）事件打標（止盈/止損/到期 + 觸發步數/原因）。
  - 波動率：改用 `arch`（EWMA / Yang–Zhang / GARCH）。
  - 交叉驗證：改用 `skfolio.CombinatorialPurgedCV`（自帶 purge/embargo）。
  - 回測驗證：改用 `vectorbt`（SL/TP/時間退出 + 向量化評分）。
  - 條形聚合：可選用 `nautilus-trader` 的 Imbalance Bars；若暫不用，維持 V4 的 10-tick 聚合亦可。
- **DeepLOB 對接**：輸出 `X (N,W,F)`, `y_tb (N,)`, `w (N,)`, `meta`。保留既有抽窗、標準化與訓練腳本；新增 **樣本權重訓練** 與 **CPCV 切分**。

---

## 1) 環境建議（Python 3.12）
```bash
conda create -n deeplob-pro-sb3 python=3.12 -y
conda activate deeplob-pro-sb3
pip install pandas>=2 numpy>=1.26 scipy>=1.11 scikit-learn>=1.3 numba>=0.58
pip install triple-barrier arch skfolio vectorbt nautilus-trader
```
> 如需 SB3/RL，再安裝：`pip install "stable-baselines3[extra]" gymnasium shimmy tensorboard`，以及對應 CPU/CUDA 的 PyTorch。

---

## 2) 新資料流（不含 mlfinlab）
```
Tick/LOB 原始資料
 └─ A. 條形聚合（Imbalance Bars） ← nautilus-trader（可選）
 └─ B. 波動率估計（EWMA/YZ/GARCH） ← arch + 自定函式
 └─ C. Triple-Barrier 打標（PyPI 版） ← triple-barrier
 └─ D. 樣本權重（報酬 × 時間衰減 × 類別平衡） ← sklearn
 └─ E. DeepLOB 抽窗/正規化 → 輸出 X, y_tb, w, meta
 └─ F. CPCV + Embargo 驗證 ← skfolio
 └─ G. 向量化回測（SL/TP/T） ← vectorbt
```

---

## 3) 波動率估計（`arch`）

```python
import numpy as np
import pandas as pd

def ewma_vol(close: pd.Series, halflife: int = 60) -> pd.Series:
    ret = np.log(close).diff()
    var = ret.ewm(halflife=halflife, adjust=False).var(bias=False)
    return np.sqrt(var).fillna(method="bfill")

def yang_zhang_vol(ohlc: pd.DataFrame, window: int = 60) -> pd.Series:
    o, h, l, c = [ohlc[k].astype(float) for k in ['open','high','low','close']]
    k = 0.34 / (1.34 + (window + 1)/(window - 1))
    log_oc = np.log(o / c.shift(1))
    log_co = np.log(c / o)
    rs = (np.log(h / l))**2
    sigma_o = log_oc.rolling(window).var()
    sigma_c = log_co.rolling(window).var()
    sigma_rs = rs.rolling(window).mean()
    yz = sigma_o + k * sigma_c + (1 - k) * sigma_rs
    return np.sqrt(yz).replace([np.inf, -np.inf], np.nan).fillna(method="bfill")

def garch11_vol(close: pd.Series) -> pd.Series:
    from arch import arch_model
    ret = 100 * np.log(close).diff().dropna()
    am = arch_model(ret, vol='GARCH', p=1, q=1, dist='normal')
    res = am.fit(disp='off')
    fcast = res.forecast(horizon=1, reindex=True).variance
    vol = np.sqrt(fcast.squeeze()) / 100.0
    return vol.reindex(close.index).fillna(method="bfill")
```

---

## 4) Triple-Barrier 打標（`triple-barrier` PyPI 版，非 mlfinlab）

```python
import numpy as np
import pandas as pd
from triple_barrier import triple_barrier as tb

def tb_labels(close: pd.Series,
              vol: pd.Series,
              up_mult: float = 1.5,
              dn_mult: float = 1.5,
              t_vert: int = 200) -> pd.DataFrame:
    """
    回傳：y ∈ {-1,0,1}, ret, tt(歷時), why{'up','down','time'}, up_p, dn_p
    """
    events = tb.get_events(
        close=close.values,
        daily_vol=vol.values,
        tp_multiplier=up_mult,
        sl_multiplier=dn_mult,
        max_holding=t_vert
    )
    bins = tb.get_bins(close.values, events)

    out = pd.DataFrame(index=close.index[:len(bins)])
    out["ret"] = bins["ret"].astype(float)
    out["y"] = np.sign(out["ret"]).astype(int)     # -1,0,1
    out["tt"] = events["t1"]                       # 觸發步數
    m = {"tp":"up","sl":"down","t1":"time"}
    out["why"] = pd.Series(events["label"]).map(m).values
    out["up_p"] = close.iloc[:len(bins)] * (1 + up_mult * vol.iloc[:len(bins)])
    out["dn_p"] = close.iloc[:len(bins)] * (1 - dn_mult * vol.iloc[:len(bins)])
    return out.dropna()
```

> 轉成 DeepLOB 標籤 `{0,1,2}`：`y_tb = tb_df["y"].map({-1:0, 0:1, 1:2})`。

---

## 5) 樣本權重（`scikit-learn`）
```python
from sklearn.utils.class_weight import compute_class_weight

def make_sample_weight(ret: pd.Series, tt: pd.Series, y: pd.Series,
                       tau: float = 100.0, scale: float = 10.0,
                       balance: bool = True) -> pd.Series:
    base = np.abs(ret.values) * scale * np.exp(-tt.values / float(tau))
    base = np.clip(base, 1e-3, None)

    if balance:
        classes = np.array(sorted(y.unique()))
        cls_w = compute_class_weight('balanced', classes=classes, y=y.values)
        w_map = dict(zip(classes, cls_w))
        cw = y.map(w_map).values
        w = base * cw
    else:
        w = base
    w = w / np.mean(w)
    return pd.Series(w, index=y.index)
```

---

## 6) DeepLOB 抽窗對接（維持 V4 API）
```python
import numpy as np

def make_windows(X_df, y_ser, w_ser, window=100, horizon=1):
    feats = X_df.columns.tolist()
    X, y, w = [], [], []
    for i in range(window, len(X_df) - horizon):
        X.append(X_df.iloc[i-window:i][feats].values)
        y.append(y_ser.iloc[i + horizon - 1])
        w.append(w_ser.iloc[i + horizon - 1])
    return np.asarray(X, dtype=np.float32), np.asarray(y), np.asarray(w, dtype=np.float32)
```

---

## 7) CPCV + Embargo（`skfolio`）
```python
from skfolio.model_selection import CombinatorialPurgedCV

def cpcv_splits(n_splits=5, n_test_groups=2, embargo=0.05):
    return CombinatorialPurgedCV(
        n_splits=n_splits,
        n_test_groups=n_test_groups,
        embargo=embargo,
        shuffle=False
    )
# 用法：for tr_idx, va_idx in cpcv_splits().split(df): ...
```

---

## 8) 向量化回測（`vectorbt`）
```python
import vectorbt as vbt
import numpy as np
import pandas as pd

def backtest_tb(close: pd.Series, y_tb: pd.Series,
                sl: float, tp: float, max_holding: int):
    entries = y_tb.eq(2)
    exits   = y_tb.eq(0)

    pf = vbt.Portfolio.from_signals(
        close=close,
        entries=entries,
        exits=exits,
        sl_stop=sl,
        tp_stop=tp,
        freq="1min",
        max_hold=max_holding
    )
    return {
        "return": float(pf.total_return()),
        "sharpe": float(pf.sharpe_ratio()),
        "max_dd": float(pf.max_drawdown())
    }
```

---

## 9) Nautilus Trader 生成 Imbalance Bars（概念流程）
```python
# 概念性示意，實際以官方 API/版本為準
from nautilus_trader.core import BarSpecification
from nautilus_trader.indicators.bars import ValueImbalanceBarBuilder
from nautilus_trader.model.data import QuoteTick

spec = BarSpecification(imposed="VALUE_IMBALANCE", target=0.7)
builder = ValueImbalanceBarBuilder(spec=spec)

# for tick in stream_quote_ticks(...):
#     bar = builder.update(QuoteTick.from_dict(tick))
#     if bar is not None:
#         persist_bar(bar)
```

---

## 10) 與 `train_deeplob_generic.py` 整合（最小變更）
- `Dataset.__getitem__` 回傳 `(X, y, w)`；
- Loss 改用 `reduction='none'`，再乘以 `w` 後平均：
```python
criterion = torch.nn.CrossEntropyLoss(reduction='none')
loss_vec = criterion(logits, y.long())
loss = (loss_vec * w).mean()
```
- 驗證/訓練切分使用 `CombinatorialPurgedCV` 產生的索引。

---

## 11) A/B 門檻（V4 → V5 Pro）
- 分類：Macro-F1 上升、Balanced Acc ≥ 60%、MCC 上升；持平比例 35–55%。
- 回測：同一成本模型下，Sharpe 上升或不降且 MaxDD 下降。
- 事件診斷：觸發原因分布（up/down/time）合理，事件壽命分布健康。

---

## 12) 執行骨架（從原始到 NPZ）
```python
def run_pipeline(lob_df, price_series, ohlc=None, cfg=None):
    if cfg["vol_method"] == "ewma":
        vol = ewma_vol(price_series, cfg.get("halflife", 60))
    elif cfg["vol_method"] == "yz":
        vol = yang_zhang_vol(ohlc, cfg.get("window", 60))
    elif cfg["vol_method"] == "garch":
        vol = garch11_vol(price_series)
    else:
        raise ValueError("Unknown vol method")

    tb = tb_labels(price_series, vol,
                   up_mult=cfg.get("up_mult", 1.5),
                   dn_mult=cfg.get("dn_mult", 1.5),
                   t_vert=cfg.get("t_vert", 200))

    w = make_sample_weight(tb["ret"], tb["tt"], tb["y"],
                           tau=cfg.get("tau", 100.0),
                           scale=cfg.get("scale", 10.0),
                           balance=cfg.get("balance", True))

    X, y, w_arr = make_windows(lob_df, tb["y"].map({-1:0, 0:1, 1:2}), w)

    return X, y, w_arr, tb
```

---

## 13) 配置建議（範例）
```yaml
vol_method: ewma        # ewma / yz / garch
halflife: 60
up_mult: 1.5
dn_mult: 1.5
t_vert: 200
tau: 100.0
scale: 10.0
balance: true
```

---

**結語**：這份 **V5 Pro（無 mlfinlab）** 指南以 `triple-barrier`、`arch`、`skfolio`、`vectorbt`、`nautilus-trader` 為核心，完整覆蓋 V4 的痛點，並與 DeepLOB 訓練腳本最小改動即用。
