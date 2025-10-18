"""
DeepLOB-style 台股數據提取（避免資料洩漏、滾動 5 日 z-score、事件視野 k 標註）

相較於 v2 版的主要改動：
1) **動態標準化（關鍵）**：改為「逐股票、逐交易日」在**處理當日之前**，使用「過去 rolling_days（預設 5）個交易日」的均值/標準差做 z-score，避免把測試/驗證日資訊洩漏到訓練集（符合 DeepLOB 設計）。
2) **處理流程順序**：先正規化 -> 再標註 -> 再切序列。原 v2 於全資料分割後才做全域 StandardScaler，存在洩漏風險。
3) **事件視野 k**：維持事件驅動（k 代表事件步數而非秒），與論文一致；alpha（百分比閾值）同樣沿用。
4) **特徵維度守衛**：預設檢查 features 維度 = 20（台股 5 檔 × 價格/數量 × 雙邊），可透過 `--allow-any-feature-dim` 支援其他維度。
5) **跨日滾動統計**：為每檔股票維護 daily mean/std 的 FIFO 緩存，用於下個交易日的標準化；前幾天不足 5 日時使用「已觀測到的歷史天數」。
6) **標籤健康檢查**：預設啟用標籤分佈檢查，避免「只有一種訊號」問題；單一類別占比預設不超過 75%，可透過 `--max-class-ratio` 調整。
7) **輸出一致**：沿用 v2 的 .npz（X, y, stock_ids）、scaler.pkl（改存 daily stats 字典）、stock_to_id.json；並新增 normalization_meta.json 方便追溯（含標籤健康檢查結果）。

用法：
    # 使用預設參數（推薦）
    python extract_tw_stock_data_v3.py

    # 或自訂參數（例）
    python extract_tw_stock_data_v3.py \
        --temp-dir data/temp \
        --output-dir data/processed \
        --min-ticks 1000 \
        --seq-len 100 \
        --k 50 \
        --alpha 0.005 \
        --rolling-days 5 \
        --split-mode temporal \
        --max-samples-per-stock-per-day 15000

說明：
- **temp-dir**：放逐日 .txt；檔名（不含副檔名）會當作交易日 ID（例如 20240103）。
- **split-mode**：`temporal` = 按日期 70/15/15（不足 3 天時有特例）；`mixed` = 60/20/20 混合。
- **generate_labels**與**create_sequences** 來自 `TWTickDataExtractor`，本檔以「正規化先行」方式供給原函式。
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import json
from collections import defaultdict, deque
import pickle

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from data.tw_tick_extractor import TWTickDataExtractor  # 需提供 features 與 mid_price 欄位

# ----------------------------
# 供「||」分隔之台股委買委賣前五檔資料的解析器（示例）
# ----------------------------

def process_vendor_file(file_path: str, verbose: bool = False):
    """
    解析每行以 '||' 分隔之台股五檔快照，並**強制**套用一組清洗規則（無開關）：
      A. 去重：同一檔 (Symbol, MatchTime) 保留最後一筆。
      B. 去除 LastVolume=0。
      C. 去除試撮 IsTrialMatch='1'。
      D. 價量合理性：所有價格/數量需為非負數；買一價 < 賣一價；spread=ask1-bid1 > 0。
      E. 價格區間：LastPrice、bid/ask 都需位於 [LowerPrice, UpperPrice]。
      F. 累計量單調性：每檔股票內，TotalVolume 應非遞減；若當前 < 前一筆則剔除當前筆。

    欄位對應（與你的 C# 定義一致）：
        0 QType
        1 Symbol
        2 Name
        3 ReferencePrice
        4 UpperPrice
        5 LowerPrice
        6 OpenPrice
        7 HighPrice
        8 LowPrice
        9 LastPrice
        10 LastVolume
        11 TotalVolume
        12-21 Bid1~Bid5 (Price, Volume) 共 10 欄
        22-31 Ask1~Ask5 (Price, Volume) 共 10 欄
        32 MatchTime(HHMMSS)
        33 IsTrialMatch

    輸出：
        { symbol: DataFrame(['features': ndarray[20], 'mid_price': float]) }
    features = [bid價1..5, bid量1..5, ask價1..5, ask量1..5]（20 維）。
    """
    out = {}
    raw_records = []

    def _to_float(x, default=0.0):
        try:
            return float(x)
        except Exception:
            return default

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = [p for p in line.strip().split('||') if p != '']
            if len(parts) < 34:
                continue
            # 先取基礎欄位
            qtype = parts[0]
            symbol = parts[1]
            ref_px = _to_float(parts[3])
            upper_px = _to_float(parts[4])
            lower_px = _to_float(parts[5])
            last_price = _to_float(parts[9])
            last_volume = int(_to_float(parts[10]))
            total_volume = int(_to_float(parts[11]))

            # C) 去掉試撮
            is_trial = str(parts[33]).strip()
            if is_trial == '1':
                continue
            # B) 去掉 LastVolume=0
            if last_volume == 0:
                continue

            # 解析五檔價量
            nums = [_to_float(parts[idx]) for idx in range(12, 32)]  # 12..31
            bid_pairs = [(nums[i], nums[i+1]) for i in range(0, 10, 2)]
            ask_pairs = [(nums[i], nums[i+1]) for i in range(10, 20, 2)]
            bid_prices = [max(0.0, p) for p,_ in bid_pairs]
            bid_vols   = [max(0.0, q) for _,q in bid_pairs]
            ask_prices = [max(0.0, p) for p,_ in ask_pairs]
            ask_vols   = [max(0.0, q) for _,q in ask_pairs]

            # D) 價量合理性與 E) 區間檢查
            best_bid = bid_prices[0]
            best_ask = ask_prices[0]
            if best_bid <= 0 or best_ask <= 0:
                continue
            spread = best_ask - best_bid
            if not np.isfinite(spread) or spread <= 0:
                continue
            # 區間檢查：若參考區間可用，驗證所有相關價格
            def in_band(px):
                if upper_px > 0 and lower_px > 0 and upper_px >= lower_px:
                    return (lower_px <= px <= upper_px)
                return True
            price_list_to_check = [last_price] + bid_prices + ask_prices
            if not all(in_band(px) for px in price_list_to_check):
                continue

            # 組裝 features & mid
            feat = bid_prices + bid_vols + ask_prices + ask_vols
            match_time = parts[32]
            mid = (best_bid + best_ask) / 2.0

            raw_records.append({
                'Symbol': symbol,
                'MatchTime': match_time,
                'LastPrice': last_price,
                'LastVolume': last_volume,
                'TotalVolume': total_volume,
                'features': np.asarray(feat, dtype=np.float32),
                'mid_price': float(mid)
            })

    if not raw_records:
        return out

    df = pd.DataFrame.from_records(raw_records)

    # A) 去重：同 (Symbol, MatchTime) 保留最後一筆
    df = df.drop_duplicates(subset=['Symbol', 'MatchTime'], keep='last').reset_index(drop=True)

    # F) TotalVolume 非遞減：在每個 Symbol 內，剔除 total_volume 回退的行
    def drop_tv_regress(g):
        tv = g['TotalVolume'].values
        keep = np.ones(len(tv), dtype=bool)
        last = -1
        for i, v in enumerate(tv):
            if v < last:  # 回退
                keep[i] = False
            else:
                last = v
        return g[keep]
    df = df.groupby('Symbol', group_keys=False, as_index=False).apply(drop_tv_regress).reset_index(drop=True)

    # 分組輸出
    for sym, g in df.groupby('Symbol'):
        out[sym] = g[['features','mid_price']].reset_index(drop=True)

    if verbose:
        print(f"  [vendor] 解析 {len(df):,} 筆（清洗+去重後），股票數 {len(out)}")
    return out

    df = pd.DataFrame.from_records(records)

    # 清洗 1) 去重：以 (Symbol, MatchTime) 當鍵，保留最後出現的一筆
    # 若你希望保留第一筆，改成 keep='first'
    df = df.drop_duplicates(subset=['Symbol', 'MatchTime'], keep='last').reset_index(drop=True)

    # 分組輸出
    for sym, g in df.groupby('Symbol'):
        out[sym] = g[['features','mid_price']].reset_index(drop=True)

    if verbose:
        print(f"  [vendor] 解析 {len(df):,} 筆（去重後），股票數 {len(out)}")
    return out
    # 按 symbol 分組成 DataFrame，欄位：features, mid_price
    df = pd.DataFrame(rows, columns=['symbol','features','mid_price'])
    for sym, g in df.groupby('symbol'):
        out[sym] = g[['features','mid_price']].reset_index(drop=True)
    if verbose:
        print(f"  [vendor] 解析 {sum(len(v) for v in out.values()):,} 筆，股票數 {len(out)}")
    return out


# ----------------------------
# 參數解析
# ----------------------------

def parse_args():
    p = argparse.ArgumentParser(description='DeepLOB-style 台股數據提取（5 日滾動標準化 + 事件視野標註）')
    p.add_argument('--temp-dir', type=str, default='data/temp', help='輸入原始資料目錄')
    p.add_argument('--output-dir', type=str, default='data/processed', help='輸出處理後資料目錄')
    p.add_argument('--min-ticks', type=int, default=1000, help='最少 tick 數量')
    p.add_argument('--seq-len', type=int, default=100, help='序列長度')
    p.add_argument('--k', type=int, default=50, help='事件視野步數')
    p.add_argument('--alpha', type=float, default=0.005, help='價格變動閾值（百分比）')
    p.add_argument('--rolling-days', type=int, default=5, help='用於 z-score 的歷史交易日數（DeepLOB 用 5 日）')
    p.add_argument('--max-samples-per-stock-per-day', type=int, default=15000, help='每檔股票每日最大樣本數')
    p.add_argument('--split-mode', type=str, default='temporal', choices=['temporal', 'mixed'], help='資料分割模式')
    p.add_argument('--allow-any-feature-dim', action='store_true', help='允許非 20 維特徵（台股標準為 20 維）')
    p.add_argument('--input-format', type=str, default='tw_vendor', choices=['tw_vendor','extractor'], help='tw_vendor=使用本檔的 || 解析器；extractor=呼叫 TWTickDataExtractor')
    p.add_argument('--label-health-check', action='store_true', default=True, help='啟用標籤健康檢查（預設開啟）')
    p.add_argument('--no-label-health-check', dest='label_health_check', action='store_false', help='停用標籤健康檢查')
    p.add_argument('--max-class-ratio', type=float, default=0.75, help='單一類別最大允許占比（0-1，預設 0.75 = 75%%）')
    return p.parse_args()


# ----------------------------
# 工具：逐股票跨日滾動統計
# ----------------------------

class RollingDailyStats:
    """為每檔股票維護最近 N 天的 (mean, std)，供次日標準化使用。"""

    def __init__(self, rolling_days: int = 5):
        self.N = rolling_days
        # stats[symbol] = deque(maxlen=N) 其中元素是 {'mean': ndarray[F], 'std': ndarray[F], 'date': str}
        self.stats = defaultdict(lambda: deque(maxlen=self.N))

    def update_with_day(self, symbol: str, day_features: np.ndarray, date: str):
        """在**當日處理完成後**，更新供未來使用的 (mean, std)。
        day_features: (ticks, feat)
        """
        if day_features.ndim != 2:
            raise ValueError(f"day_features 應為 2D (ticks, feat)，收到 {day_features.shape}")
        mu = day_features.mean(axis=0)
        sigma = day_features.std(axis=0)
        # 避免除以 0
        sigma[sigma == 0] = 1.0
        self.stats[symbol].append({'mean': mu, 'std': sigma, 'date': date})

    def get_norm_params_for_day(self, symbol: str):
        """回傳該股票「已知歷史」的平均/標準差，若無歷史則回傳 None（呼叫端自行決策）。"""
        dq = self.stats.get(symbol, None)
        if dq is None or len(dq) == 0:
            return None
        # 聚合歷史天的均值/方差：對每日 mean 取平均、std 取平均（等權），也可改加權
        means = np.stack([item['mean'] for item in dq])
        stds = np.stack([item['std'] for item in dq])
        mu = means.mean(axis=0)
        sigma = stds.mean(axis=0)
        sigma[sigma == 0] = 1.0
        return mu, sigma


# ----------------------------
# 標籤健康檢查（避免極端偏斜）
# ----------------------------

def check_label_health(labels, dataset_name='', max_single_class_ratio=0.75, verbose=True):
    """
    檢查標籤分佈是否健康，避免「只有一種訊號」的問題。

    Args:
        labels: ndarray，標籤陣列（假設為 0, 1, 2 三分類）
        dataset_name: str，資料集名稱（用於日誌）
        max_single_class_ratio: float，單一類別最大允許占比（預設 0.75 = 75%）
        verbose: bool，是否顯示詳細資訊

    Returns:
        bool: True 表示健康，False 表示標籤分佈極端偏斜
        dict: 統計資訊
    """
    if len(labels) == 0:
        return False, {'error': 'empty labels'}

    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    ratios = counts / total

    # 計算統計
    stats = {
        'total_samples': total,
        'num_classes': len(unique),
        'class_distribution': {int(cls): {'count': int(cnt), 'ratio': float(ratio)}
                               for cls, cnt, ratio in zip(unique, counts, ratios)},
        'max_ratio': float(ratios.max()),
        'min_ratio': float(ratios.min()),
        'is_healthy': ratios.max() <= max_single_class_ratio
    }

    # 詳細輸出
    if verbose:
        prefix = f"[{dataset_name}] " if dataset_name else ""
        print(f"{prefix}標籤健康檢查:")
        print(f"  總樣本數: {total:,}")
        print(f"  類別數: {len(unique)}")
        for cls in sorted(unique):
            idx = np.where(unique == cls)[0][0]
            cnt, ratio = counts[idx], ratios[idx]
            status = "✓" if ratio <= max_single_class_ratio else "✗ 過高"
            print(f"    類別 {cls}: {cnt:,} ({ratio*100:.2f}%) {status}")

        if not stats['is_healthy']:
            print(f"  ⚠️  警告: 類別 {unique[ratios.argmax()]} 占比 {ratios.max()*100:.1f}% 超過閾值 {max_single_class_ratio*100:.0f}%")
            print(f"      建議: 調整 k 或 alpha 參數以平衡標籤分佈")
        else:
            print(f"  ✓ 標籤分佈健康（最大占比 {ratios.max()*100:.1f}% ≤ {max_single_class_ratio*100:.0f}%）")

    return stats['is_healthy'], stats


# ----------------------------
# 分割策略（沿用 v2 的語義，文檔化輸出）
# ----------------------------

def temporal_split(all_data_by_day):
    num_days = len(all_data_by_day)
    if num_days == 1:
        data = all_data_by_day[0]
        n = len(data['X'])
        n_train = int(n * 0.6)
        n_val = int(n * 0.2)
        return (
            {'X': data['X'][:n_train], 'y': data['y'][:n_train], 'stock_ids': data['stock_ids'][:n_train]},
            {'X': data['X'][n_train:n_train+n_val], 'y': data['y'][n_train:n_train+n_val], 'stock_ids': data['stock_ids'][n_train:n_train+n_val]},
            {'X': data['X'][n_train+n_val:], 'y': data['y'][n_train+n_val:], 'stock_ids': data['stock_ids'][n_train+n_val:]},
        )
    elif num_days == 2:
        d1, d2 = all_data_by_day[0], all_data_by_day[1]
        n1 = len(d1['X']); n_train = int(n1 * 0.8)
        return (
            {'X': d1['X'][:n_train], 'y': d1['y'][:n_train], 'stock_ids': d1['stock_ids'][:n_train]},
            {'X': d1['X'][n_train:], 'y': d1['y'][n_train:], 'stock_ids': d1['stock_ids'][n_train:]},
            {'X': d2['X'], 'y': d2['y'], 'stock_ids': d2['stock_ids']},
        )
    else:
        train_days = int(num_days * 0.70)
        val_days = max(1, int(num_days * 0.15))
        test_days = max(1, num_days - train_days - val_days)
        train_days = num_days - val_days - test_days
        train = _merge_days(all_data_by_day[:train_days])
        val = _merge_days(all_data_by_day[train_days:train_days+val_days])
        test = _merge_days(all_data_by_day[train_days+val_days:])
        return train, val, test


def _merge_days(days_list):
    if len(days_list) == 1:
        d = days_list[0]
        return {'X': d['X'], 'y': d['y'], 'stock_ids': d['stock_ids']}
    return {
        'X': np.vstack([d['X'] for d in days_list]),
        'y': np.concatenate([d['y'] for d in days_list]),
        'stock_ids': np.concatenate([d['stock_ids'] for d in days_list])
    }


# ----------------------------
# 主流程
# ----------------------------

def main():
    args = parse_args()
    temp_dir = Path(args.temp_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(temp_dir.glob('*.txt'))
    if not txt_files:
        print(f"[ERROR] 在 {temp_dir} 找不到 .txt 檔")
        return

    print("="*70)
    print("DeepLOB-style 台股數據提取 (滾動 5 日標準化 + 事件視野 k)")
    print("="*70)
    print(f"檔案天數: {len(txt_files)} | 視窗長度 seq_len={args.seq_len} | k={args.k} | alpha={args.alpha}")
    print(f"rolling_days={args.rolling_days} | 分割模式={args.split_mode}")

    extractor = TWTickDataExtractor(filter_trial_match=True)

    # 逐股票的滾動日統計，用於「下一個交易日」的 z-score
    rolling_stats = RollingDailyStats(rolling_days=args.rolling_days)

    all_data_by_day = []
    all_stock_symbols = set()

    # 統計：標籤健康檢查排除的股票數
    skipped_stocks_count = 0
    total_stocks_processed = 0

    for day_idx, txt_file in enumerate(txt_files, 1):
        date_str = txt_file.stem
        print(f"\n[Day {day_idx}/{len(txt_files)}] {date_str}")

        # 解析當日原始 tick
        if args.input_format == 'tw_vendor':
            day_raw = process_vendor_file(str(txt_file), verbose=False)
        else:
            day_raw = extractor.process_file(str(txt_file), symbols=None, verbose=False)
        if not day_raw:
            print("  [WARNING] 無有效數據；跳過")
            continue

        # 準備當日容器
        day_sequences = []
        day_labels = []
        day_stock_ids = []
        stock_to_id = {}

        # 先統計當日每檔股票的原始 features，用於**隔日**更新滾動統計
        per_symbol_features_for_stats = {}

        for symbol, df in day_raw.items():
            total_stocks_processed += 1

            if len(df) < args.min_ticks:
                continue

            feats_series = df['features']
            # features shape check (台股標準：20 維 = 5 檔 × 價/量 × 雙邊)
            sample_feat = np.asarray(feats_series.iloc[0]) if len(feats_series) else None
            if sample_feat is None:
                continue
            feat_dim = sample_feat.shape[-1]

            # 台股資料標準為 20 維，若需支援其他維度可用 --allow-any-feature-dim
            if not args.allow_any_feature_dim and feat_dim != 20:
                print(f"  [SKIP] {symbol} 特徵維度={feat_dim} 非台股標準 20 維；可用 --allow-any-feature-dim 放行")
                continue

            # 股票 ID
            if symbol not in stock_to_id:
                stock_to_id[symbol] = len(stock_to_id)

            # 準備原始矩陣 (ticks, feat)
            features_raw = np.stack(feats_series.values)
            per_symbol_features_for_stats[symbol] = features_raw  # 供收盤後更新滾動統計

            # 取得「歷史 1..5 日」的 z-score 參數，若無歷史則暫用當日前 10% tick 作為冷啟估計
            params = rolling_stats.get_norm_params_for_day(symbol)
            if params is None:
                warm_n = max(10, int(0.1 * len(features_raw)))
                mu = features_raw[:warm_n].mean(axis=0)
                sigma = features_raw[:warm_n].std(axis=0)
                sigma[sigma == 0] = 1.0
            else:
                mu, sigma = params

            features_norm = (features_raw - mu) / sigma

            # 生成標籤（事件視野 k、百分比閾值 alpha；由 extractor 實作，與 v2 相同介面）
            mid_prices = df['mid_price'].values
            labels, _ = extractor.generate_labels(mid_prices, k=args.k, alpha=args.alpha)
            if len(labels) == 0:
                continue

            # 對齊（labels 比 features 少 k 個）
            features_norm = features_norm[:len(labels)]

            # 建立序列
            X_seq, y_seq = extractor.create_sequences(features_norm, labels, seq_len=args.seq_len)
            if len(X_seq) == 0:
                continue

            # 每天每股樣本上限
            if len(X_seq) > args.max_samples_per_stock_per_day:
                idx = np.random.choice(len(X_seq), args.max_samples_per_stock_per_day, replace=False)
                X_seq = X_seq[idx]
                y_seq = y_seq[idx]

            # 標籤健康檢查（逐股票、逐日）
            if args.label_health_check:
                is_healthy, _ = check_label_health(
                    y_seq,
                    dataset_name='',
                    max_single_class_ratio=args.max_class_ratio,
                    verbose=False
                )
                if not is_healthy:
                    # 計算標籤分佈以顯示警告
                    _, counts_labels = np.unique(y_seq, return_counts=True)
                    max_ratio = counts_labels.max() / len(y_seq)
                    print(f"  [SKIP] {symbol} 標籤極端偏斜（最大占比 {max_ratio*100:.1f}% > {args.max_class_ratio*100:.0f}%），排除當日資料")
                    skipped_stocks_count += 1
                    continue

            stock_id = stock_to_id[symbol]
            stock_ids_seq = np.full(len(X_seq), stock_id, dtype=np.int64)

            day_sequences.append(X_seq)
            day_labels.append(y_seq)
            day_stock_ids.append(stock_ids_seq)
            all_stock_symbols.add(symbol)

        # 收集當日樣本
        if not day_sequences:
            print("  [WARNING] 當日無可用序列；跳過")
            continue

        day_X = np.vstack(day_sequences)
        day_y = np.concatenate(day_labels)
        day_sid = np.concatenate(day_stock_ids)

        all_data_by_day.append({
            'X': day_X, 'y': day_y, 'stock_ids': day_sid,
            'stock_to_id': stock_to_id, 'date': date_str
        })

        # 當日標籤分佈統計（簡要）
        unique_day, counts_day = np.unique(day_y, return_counts=True)
        label_dist_str = ", ".join([f"類別{cls}: {cnt:,} ({cnt/len(day_y)*100:.1f}%)"
                                     for cls, cnt in zip(unique_day, counts_day)])
        print(f"  ✅ 當日樣本: {len(day_X):,}；股票數: {len(stock_to_id)}")
        print(f"     標籤分佈: {label_dist_str}")

        # 收盤後：更新滾動統計（供下一日使用）
        for symbol, feats_raw in per_symbol_features_for_stats.items():
            rolling_stats.update_with_day(symbol, feats_raw, date_str)

    # 摘要
    print("\n" + "="*70)
    print("數據處理摘要")
    print("="*70)
    print(f"有效交易日: {len(all_data_by_day)} | 總股票數（去重）: {len(all_stock_symbols)}")
    print(f"處理股票次數: {total_stocks_processed} | 標籤健康檢查排除: {skipped_stocks_count} ({skipped_stocks_count/total_stocks_processed*100:.2f}%)" if total_stocks_processed > 0 else "處理股票次數: 0")
    if len(all_data_by_day) == 0:
        print("[ERROR] 無有效資料，終止")
        return

    # 分割
    print("\n" + "="*70)
    print(f"數據分割（模式: {args.split_mode}）")
    print("="*70)
    if args.split_mode == 'temporal':
        train, val, test = temporal_split(all_data_by_day)
    else:
        # 混合：拼接後 60/20/20
        all_X = np.vstack([d['X'] for d in all_data_by_day])
        all_y = np.concatenate([d['y'] for d in all_data_by_day])
        all_sid = np.concatenate([d['stock_ids'] for d in all_data_by_day])
        n = len(all_X); n_train = int(n*0.6); n_val = int(n*0.2)
        train = {'X': all_X[:n_train], 'y': all_y[:n_train], 'stock_ids': all_sid[:n_train]}
        val   = {'X': all_X[n_train:n_train+n_val], 'y': all_y[n_train:n_train+n_val], 'stock_ids': all_sid[n_train:n_train+n_val]}
        test  = {'X': all_X[n_train+n_val:], 'y': all_y[n_train+n_val:], 'stock_ids': all_sid[n_train+n_val:]}

    print(f"訓練集: {len(train['X']):,} | 驗證集: {len(val['X']):,} | 測試集: {len(test['X']):,}")

    # 形狀檢查
    for split_name, split in [('train', train), ('val', val), ('test', test)]:
        if split['X'].ndim != 3:
            raise ValueError(f"{split_name} X 應為 3D (N, seq_len, feat)，但為 {split['X'].shape}")

    # 標籤分佈統計（最終結果）
    print("\n" + "="*70)
    print("最終標籤分佈統計")
    print("="*70)

    health_results = {}
    for split_name, split in [('訓練集', train), ('驗證集', val), ('測試集', test)]:
        is_healthy, stats = check_label_health(
            split['y'],
            dataset_name=split_name,
            max_single_class_ratio=args.max_class_ratio,
            verbose=True
        )
        health_results[split_name] = stats
        print()

    if args.label_health_check:
        print("註: 已在個股層級排除標籤極端偏斜的股票（閾值 {:.0f}%）".format(args.max_class_ratio*100))
        print()

    # 輸出
    print("\n" + "="*70)
    print("保存數據")
    print("="*70)

    for name, split in [('train', train), ('val', val), ('test', test)]:
        out = output_dir / f'stock_embedding_{name}.npz'
        np.savez_compressed(out, X=split['X'], y=split['y'], stock_ids=split['stock_ids'])
        print(f"  {out.name}: {len(split['X']):,} 樣本")

    # 保存「每日統計」而非單一 scaler（與 v2 不同）
    # 格式：stats[symbol] = [{'date':..., 'mean': list, 'std': list}, ...]（僅為追溯記錄）
    stats_serializable = {sym: [
        {'date': item['date'], 'mean': item['mean'].tolist(), 'std': item['std'].tolist()} for item in dq
    ] for sym, dq in rolling_stats.stats.items()}

    # 準備元數據（包含標籤健康檢查結果）
    meta_data = {
        'rolling_days': args.rolling_days,
        'symbols': sorted(list(all_stock_symbols)),
        'per_symbol_daily_stats': stats_serializable
    }

    # 添加標籤健康檢查結果
    if args.label_health_check and 'health_results' in locals():
        meta_data['label_health_check'] = {
            'enabled': True,
            'max_class_ratio_threshold': args.max_class_ratio,
            'skipped_stocks_count': skipped_stocks_count,
            'total_stocks_processed': total_stocks_processed,
            'skip_rate': skipped_stocks_count / total_stocks_processed if total_stocks_processed > 0 else 0,
            'final_distribution': health_results
        }

    with open(output_dir / 'normalization_meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta_data, f, ensure_ascii=False, indent=2)

    # 仍輸出一個「占位」scaler.pkl 以相容舊程式（內容只是說明）
    with open(output_dir / 'scaler.pkl', 'wb') as f:
        placeholder = {
            'note': 'v3 版不再使用單一 StandardScaler，請改讀 normalization_meta.json 以取得每日滾動標準化參數（DeepLOB-style）'
        }
        pickle.dump(placeholder, f)

    # 合併股票映射（取最後一天的映射；如需穩定映射可另行維護）
    final_stock_to_id = {}
    for d in all_data_by_day:
        final_stock_to_id.update(d['stock_to_id'])
    with open(output_dir / 'stock_to_id.json', 'w', encoding='utf-8') as f:
        json.dump(final_stock_to_id, f, ensure_ascii=False, indent=2)

    print("\n[完成] 輸出位於:", output_dir)


if __name__ == '__main__':
    main()
