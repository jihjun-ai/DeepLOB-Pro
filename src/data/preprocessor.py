"""LOB 序列數據預處理器

此模組負責 LOB 數據的預處理工作，包括標準化、序列創建和數據集分割。
預處理是機器學習流程中的關鍵步驟，直接影響模型的訓練效果和收斂速度。

主要功能:
    1. 數據標準化 (Normalization)
        - Z-Score 標準化
        - Min-Max 標準化
        - Robust 標準化

    2. 缺失值處理
        - 前向填充
        - 後向填充
        - 線性插值
        - 刪除缺失

    3. 時間序列窗口創建
        - 滑動窗口 (Sliding Window)
        - 可調整步長 (Stride)
        - 自動對齊標籤

    4. 數據集分割
        - 時間序列感知分割 (避免未來洩漏)
        - 訓練/驗證/測試集劃分
        - 可自訂比例

    5. 數據增強
        - 高斯噪聲注入
        - 增加訓練樣本多樣性

設計原則:
    - 先標準化再創建序列
    - 保持時間順序 (避免數據洩漏)
    - 只在訓練集上擬合標準化器
    - 驗證集和測試集使用訓練集的統計量

作者: RLlib-DeepLOB 專案團隊
更新: 2025-01-09
"""

import numpy as np
from typing import Tuple, Optional, Dict
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import logging

logger = logging.getLogger(__name__)


class LOBPreprocessor:
    """LOB 數據預處理器

    此類別提供完整的數據預處理管線，從原始 LOB 數據到模型可用的序列數據。

    核心功能:
        1. 標準化: 將特徵縮放到相似範圍，加速訓練收斂
        2. 序列創建: 將時間序列數據轉換為固定長度的窗口
        3. 數據分割: 按時間順序分割訓練/驗證/測試集
        4. 缺失值處理: 處理數據中的異常值和缺失值
        5. 數據增強: 通過添加噪聲增加訓練樣本

    標準化方法:
        Z-Score (標準化):
            公式: (x - mean) / std
            特點: 均值為0，標準差為1
            適用: 數據接近常態分佈，無極端異常值

        Min-Max (最小最大化):
            公式: (x - min) / (max - min)
            特點: 縮放到 [0, 1] 區間
            適用: 需要有界範圍，對異常值敏感

        Robust (穩健標準化):
            公式: (x - median) / IQR
            特點: 使用中位數和四分位距，對異常值穩健
            適用: 數據包含較多異常值

    序列創建:
        使用滑動窗口技術將時間序列轉換為監督學習問題
        輸入: (N,) 時間序列
        輸出: (M, T, F) 序列陣列
            M: 序列數量
            T: 序列長度 (時間步)
            F: 特徵維度
    """

    def __init__(
        self,
        normalization_method: str = 'z-score',
        sequence_length: int = 100,
        prediction_horizon: int = 10
    ):
        """初始化 LOB 預處理器

        此方法設定預處理參數並初始化標準化器。

        參數:
            normalization_method: 標準化方法
                選項: 'z-score', 'min-max', 'robust'
                預設: 'z-score'
                建議: 金融數據通常使用 z-score 或 robust

            sequence_length: 輸入序列長度 (時間步數)
                預設: 100
                含義: 模型一次觀察多少個歷史時間點
                建議: 根據預測任務調整，通常 50-200

            prediction_horizon: 預測時間範圍 (k步向前預測)
                預設: 10
                含義: 預測未來第 k 個時間步的價格變動
                選項: 通常為 10, 20, 30, 50, 100

        初始化後設定:
            self.scaler: sklearn 標準化器實例
            self.is_fitted: 標準化器是否已擬合的標記
        """
        self.normalization_method = normalization_method
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon

        # 根據方法選擇對應的標準化器
        if normalization_method == 'z-score':
            self.scaler = StandardScaler()
        elif normalization_method == 'min-max':
            self.scaler = MinMaxScaler()
        elif normalization_method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"未知的標準化方法: {normalization_method}")

        # 標準化器擬合狀態標記
        self.is_fitted = False

    def normalize(
        self,
        data: np.ndarray,
        fit: bool = True,
        clip_outliers: bool = True,
        clip_std: float = 5.0
    ) -> np.ndarray:
        """標準化 LOB 數據

        此方法對 LOB 特徵進行標準化處理，可選擇是否擬合標準化器。

        參數:
            data: 特徵陣列，形狀 (N, F)
                N 為樣本數量
                F 為特徵維度

            fit: 是否擬合標準化器
                True: 在數據上擬合標準化器（訓練集）
                False: 使用已擬合的標準化器轉換（驗證/測試集）

            clip_outliers: 是否裁剪異常值
                True: 將標準化後的值限制在合理範圍內
                False: 不裁剪，保留所有值

            clip_std: 裁剪閾值（標準差倍數）
                預設: 5.0
                含義: 將值裁剪到 [-5σ, +5σ] 範圍
                建議: 3.0-5.0，根據數據分佈調整

        返回:
            np.ndarray: 標準化後的數據，形狀 (N, F)

        異常:
            RuntimeError: 若 fit=False 但標準化器未擬合

        標準化流程:
            1. 若 fit=True，在數據上擬合標準化器
            2. 使用標準化器轉換數據
            3. 若啟用，裁剪異常值到指定範圍
            4. 返回標準化後的數據

        重要提示:
            訓練集: fit=True（擬合並轉換）
            驗證集: fit=False（僅轉換，使用訓練集統計量）
            測試集: fit=False（僅轉換，使用訓練集統計量）
        """
        # 擬合標準化器（僅訓練集）
        if fit:
            self.scaler.fit(data)
            self.is_fitted = True
            logger.info(f"已在 {data.shape[0]} 個樣本上擬合標準化器")

        # 檢查標準化器狀態
        if not self.is_fitted:
            raise RuntimeError("標準化器未擬合。請先使用 fit=True 進行擬合。")

        # 轉換數據
        normalized = self.scaler.transform(data).astype(np.float32)

        # 裁剪異常值（僅適用於 z-score）
        if clip_outliers and self.normalization_method == 'z-score':
            normalized = np.clip(normalized, -clip_std, clip_std)

        return normalized

    def denormalize(self, data: np.ndarray) -> np.ndarray:
        """反標準化數據

        此方法將標準化後的數據轉換回原始尺度。

        參數:
            data: 標準化後的數據，形狀 (N, F)

        返回:
            np.ndarray: 原始尺度的數據，形狀 (N, F)

        異常:
            RuntimeError: 若標準化器未擬合

        用途:
            1. 將模型預測結果轉換回原始尺度
            2. 可視化和解釋模型輸出
            3. 計算實際的價格變動幅度

        注意:
            必須使用與標準化時相同的標準化器實例
        """
        # 檢查標準化器狀態
        if not self.is_fitted:
            raise RuntimeError("標準化器未擬合")

        # 執行反轉換
        return self.scaler.inverse_transform(data).astype(np.float32)

    def handle_missing_values(
        self,
        data: np.ndarray,
        method: str = 'forward_fill',
        max_consecutive_nan: int = 5
    ) -> np.ndarray:
        """處理 LOB 數據中的缺失值

        此方法檢測並處理數據中的缺失值 (NaN)，確保數據完整性。

        參數:
            data: 特徵陣列，形狀 (N, F)
                N 為樣本數量
                F 為特徵維度

            method: 缺失值處理方法
                'forward_fill': 前向填充（使用前一個有效值）
                'backward_fill': 後向填充（使用後一個有效值）
                'interpolate': 線性插值
                'drop': 刪除包含缺失值的行

            max_consecutive_nan: 最大連續缺失數量
                預設: 5
                用途: 當連續缺失超過此閾值時，可能需要特殊處理

        返回:
            np.ndarray: 處理後的數據，形狀可能改變（若使用 drop）

        異常:
            ValueError: 若提供未知的處理方法

        處理方法說明:
            前向填充 (Forward Fill):
                用前一個有效值填充缺失值
                適用: 時間序列數據，假設值變化緩慢

            後向填充 (Backward Fill):
                用後一個有效值填充缺失值
                適用: 需要使用未來資訊的情況（注意數據洩漏）

            線性插值 (Interpolate):
                在缺失值前後的有效值之間線性插值
                適用: 數據變化平滑，缺失較少

            刪除 (Drop):
                直接刪除包含缺失值的樣本
                適用: 缺失值很少，且不影響數據集大小

        建議:
            高頻金融數據通常使用前向填充
            避免後向填充（可能導致未來洩漏）
        """
        # 檢查是否存在缺失值
        if not np.isnan(data).any():
            return data

        # 記錄缺失值數量
        logger.warning(f"發現 {np.isnan(data).sum()} 個缺失值")

        if method == 'forward_fill':
            # ===== 前向填充 =====
            # 沿時間軸用前一個有效值填充
            mask = np.isnan(data)
            idx = np.where(~mask, np.arange(mask.shape[0])[:, None], 0)
            np.maximum.accumulate(idx, axis=0, out=idx)
            data = data[idx, np.arange(idx.shape[1])]

        elif method == 'backward_fill':
            # ===== 後向填充 =====
            # 翻轉數據，前向填充，再翻轉回來
            data = np.flip(data, axis=0)
            mask = np.isnan(data)
            idx = np.where(~mask, np.arange(mask.shape[0])[:, None], mask.shape[0] - 1)
            np.minimum.accumulate(idx, axis=0, out=idx)
            data = data[idx, np.arange(idx.shape[1])]
            data = np.flip(data, axis=0)

        elif method == 'interpolate':
            # ===== 線性插值 =====
            # 對每一列獨立進行插值
            from scipy.interpolate import interp1d
            for col in range(data.shape[1]):
                mask = ~np.isnan(data[:, col])
                # 至少需要2個有效值才能插值
                if mask.sum() > 1:
                    f = interp1d(
                        np.where(mask)[0],
                        data[mask, col],
                        kind='linear',
                        fill_value='extrapolate'
                    )
                    data[:, col] = f(np.arange(len(data)))

        elif method == 'drop':
            # ===== 刪除缺失值 =====
            # 刪除任何包含 NaN 的行
            data = data[~np.isnan(data).any(axis=1)]

        else:
            raise ValueError(f"未知的處理方法: {method}")

        return data.astype(np.float32)

    def create_sequences(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        stride: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """創建時間序列滑動窗口序列

        此方法使用滑動窗口技術將時間序列數據轉換為監督學習格式。

        參數:
            features: 特徵陣列，形狀 (N, F)
                N 為總時間步數
                F 為特徵維度

            labels: 標籤陣列，形狀 (N,) 或 (N, H)
                N 為總時間步數
                H 為標籤維度（多時間範圍預測）

            stride: 滑動窗口步長
                預設: 1（每次移動1個時間步）
                用途: 控制序列重疊程度
                說明: stride=1 最大重疊，stride=T 無重疊

        返回:
            Tuple[np.ndarray, np.ndarray]:
                - sequences: 序列陣列，形狀 (M, T, F)
                    M 為序列數量
                    T 為序列長度（sequence_length）
                    F 為特徵維度

                - sequence_labels: 序列標籤，形狀 (M,) 或 (M, H)
                    對應每個序列最後一個時間步的標籤

        異常:
            ValueError: 若樣本數量少於序列長度

        滑動窗口示意:
            時間序列: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            T=3, stride=1:
                序列1: [0, 1, 2] → 標籤: label[2]
                序列2: [1, 2, 3] → 標籤: label[3]
                序列3: [2, 3, 4] → 標籤: label[4]
                ...

        重要設計:
            標籤對應序列的最後一個時間步
            這確保模型使用歷史數據預測當前狀態
            避免使用未來資訊（數據洩漏）
        """
        N, F = features.shape
        T = self.sequence_length

        # 驗證樣本數量充足
        if N < T:
            raise ValueError(f"樣本數量不足 ({N})，無法創建長度為 {T} 的序列")

        # 計算可創建的序列數量
        num_sequences = (N - T) // stride + 1

        # 預分配記憶體（提升效能）
        sequences = np.zeros((num_sequences, T, F), dtype=np.float32)
        sequence_labels = np.zeros(
            (num_sequences,) + labels.shape[1:] if labels.ndim > 1 else (num_sequences,),
            dtype=labels.dtype
        )

        # ===== 創建序列 =====
        for i in range(num_sequences):
            start_idx = i * stride
            end_idx = start_idx + T
            # 提取時間窗口的特徵
            sequences[i] = features[start_idx:end_idx]
            # 標籤對應窗口最後一個時間步
            sequence_labels[i] = labels[end_idx - 1]

        logger.info(
            f"已創建 {num_sequences} 個序列，序列長度 {T}，步長 {stride}，來源樣本 {N} 個"
        )

        return sequences, sequence_labels

    def train_val_test_split(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        test_ratio: float = 0.2
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], ...]:
        """時間序列感知的數據集分割

        此方法將序列數據按時間順序分割為訓練、驗證和測試集。
        嚴格保持時間順序，避免未來資訊洩漏到訓練集。

        參數:
            sequences: 序列陣列，形狀 (M, T, F)
                M 為序列數量
                T 為序列長度
                F 為特徵維度

            labels: 標籤陣列，形狀 (M,) 或 (M, H)
                M 為序列數量
                H 為標籤維度

            train_ratio: 訓練集比例
                預設: 0.6 (60%)
                用途: 用於訓練模型

            val_ratio: 驗證集比例
                預設: 0.2 (20%)
                用途: 調整超參數、早停

            test_ratio: 測試集比例
                預設: 0.2 (20%)
                用途: 最終評估模型性能

        返回:
            Tuple[Tuple[np.ndarray, np.ndarray], ...]:
                ((X_train, y_train), (X_val, y_val), (X_test, y_test))

                X_train, X_val, X_test: 特徵序列
                y_train, y_val, y_test: 對應標籤

        異常:
            AssertionError: 若三個比例之和不為 1

        分割策略:
            時間序列: [早期 ─────────────────── 晚期]
                       ├─────訓練─────┤─驗證─┤測試┤

            特點:
                訓練集: 最早期數據
                驗證集: 中期數據
                測試集: 最晚期數據

        重要原則:
            絕對不能隨機打亂時間序列數據
            訓練集只能使用比驗證集更早的數據
            驗證集只能使用比測試集更早的數據
            這模擬真實交易場景（用歷史預測未來）
        """
        # 驗證比例總和為 1
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "訓練、驗證、測試比例之和必須為 1"

        N = len(sequences)

        # ===== 計算分割點 =====
        # 訓練集結束位置
        train_end = int(N * train_ratio)
        # 驗證集結束位置
        val_end = int(N * (train_ratio + val_ratio))

        # ===== 分割數據（保持時間順序）=====
        # 訓練集: 最早期數據
        X_train = sequences[:train_end]
        y_train = labels[:train_end]

        # 驗證集: 中期數據
        X_val = sequences[train_end:val_end]
        y_val = labels[train_end:val_end]

        # 測試集: 最晚期數據
        X_test = sequences[val_end:]
        y_test = labels[val_end:]

        logger.info(
            f"數據集分割完成: 訓練集={len(X_train)}, 驗證集={len(X_val)}, 測試集={len(X_test)}"
        )

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def augment_data(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
        noise_std: float = 0.01,
        augmentation_factor: int = 2
    ) -> Tuple[np.ndarray, np.ndarray]:
        """數據增強 - 通過添加高斯噪聲

        此方法通過向原始數據添加隨機噪聲來擴充訓練集，
        增加模型的泛化能力和對噪聲的魯棒性。

        參數:
            sequences: 序列陣列，形狀 (M, T, F)
                M 為序列數量
                T 為序列長度
                F 為特徵維度

            labels: 標籤陣列，形狀 (M,) 或 (M, H)
                M 為序列數量

            noise_std: 高斯噪聲的標準差
                預設: 0.01
                建議: 0.001-0.05，取決於數據已標準化程度
                過大: 可能改變數據分佈
                過小: 增強效果不明顯

            augmentation_factor: 增強倍數（包含原始數據）
                預設: 2（原始 + 1份噪聲版本）
                選項: 通常 2-5
                說明: 最終數據量 = 原始量 × 增強倍數

        返回:
            Tuple[np.ndarray, np.ndarray]:
                - augmented_sequences: 增強後的序列
                - augmented_labels: 對應的標籤（複製）

        數據增強策略:
            對每個樣本:
                1. 保留原始版本
                2. 創建 (augmentation_factor-1) 個噪聲版本
                3. 噪聲版本 = 原始 + 隨機高斯噪聲
                4. 所有版本使用相同標籤

        注意事項:
            僅在訓練集上進行數據增強
            驗證集和測試集保持原始數據
            過度增強可能導致過擬合增強噪聲模式
        """
        # 若增強倍數 ≤ 1，直接返回原始數據
        if augmentation_factor <= 1:
            return sequences, labels

        logger.info(f"正在進行 {augmentation_factor}x 數據增強，噪聲標準差={noise_std}")

        # 儲存所有版本的數據
        augmented_sequences = [sequences]
        augmented_labels = [labels]

        # ===== 創建噪聲版本 =====
        for _ in range(augmentation_factor - 1):
            # 生成與數據形狀相同的高斯噪聲
            noise = np.random.randn(*sequences.shape) * noise_std
            # 原始數據 + 噪聲
            noisy_sequences = sequences + noise
            augmented_sequences.append(noisy_sequences.astype(np.float32))
            # 標籤保持不變（複製）
            augmented_labels.append(labels)

        # ===== 合併所有版本 =====
        return (
            np.concatenate(augmented_sequences, axis=0),
            np.concatenate(augmented_labels, axis=0)
        )

    def get_statistics(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """計算數據統計量

        此方法計算數據的描述性統計量，用於數據探索和驗證。

        參數:
            data: 特徵陣列，形狀 (N, F)
                N 為樣本數量
                F 為特徵維度

        返回:
            Dict[str, np.ndarray]: 統計量字典
                'mean': 各特徵均值，形狀 (F,)
                'std': 各特徵標準差，形狀 (F,)
                'min': 各特徵最小值，形狀 (F,)
                'max': 各特徵最大值，形狀 (F,)
                'median': 各特徵中位數，形狀 (F,)

        用途:
            1. 數據探索分析 (EDA)
            2. 檢測異常特徵（極值、零方差）
            3. 驗證標準化效果
            4. 比較訓練集和測試集分佈

        檢查項目:
            均值: 應接近 0（若已 z-score 標準化）
            標準差: 應接近 1（若已 z-score 標準化）
            最小/最大值: 檢測異常極值
            中位數: 檢測數據偏態程度
        """
        return {
            'mean': np.mean(data, axis=0),      # 均值
            'std': np.std(data, axis=0),        # 標準差
            'min': np.min(data, axis=0),        # 最小值
            'max': np.max(data, axis=0),        # 最大值
            'median': np.median(data, axis=0)   # 中位數
        }
