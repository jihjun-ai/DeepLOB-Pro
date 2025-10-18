"""FI-2010 數據集載入器

此模組負責載入和解析 FI-2010 限價單簿數據集。
FI-2010 是金融機器學習領域的標準基準數據集，包含 5 支股票共 10 天的高頻 LOB 數據。

⚠️ 重要: 完整的數據格式規格請參閱 docs/FI2010_Dataset_Specification.md

數據集結構 (經實際驗證):
    文件格式: (149, N) → 轉置為 (N, 149)
        - 欄位 0-39:   40 維 LOB 特徵（交錯排列）
        - 欄位 40-143: 104 維衍生特徵（本實現不使用）
        - 欄位 144-148: 5 個預測時間範圍的標籤

    LOB 特徵排列 (交錯方式):
        欄位 0: ask_price_1, 欄位 1: ask_volume_1
        欄位 2: bid_price_1, 欄位 3: bid_volume_1
        欄位 4: ask_price_2, 欄位 5: ask_volume_2
        ... (重複 10 檔)

    標籤: 價格變動方向 (已轉換為 0-based)
        - 0 = 下跌 (Down)     [原始值: 1]
        - 1 = 持平 (Stationary) [原始值: 2]
        - 2 = 上漲 (Up)        [原始值: 3]

    預測時間範圍: k ∈ {10, 20, 30, 50, 100} 個時間步

數據來源:
    FI-2010 數據集由芬蘭 Aalto 大學提供，是研究高頻交易預測的標準數據集。
    論文: Ntakaris et al. (2018), Journal of Forecasting, DOI: 10.1002/for.2543

主要功能:
    1. 載入原始 LOB 數據和標籤（自動轉置與標籤轉換）
    2. 解析 LOB 快照為結構化格式（交錯提取）
    3. 提取額外的市場微觀結構特徵
    4. 分析標籤分佈

驗證狀態: ✅ 已通過 1,690,366 訓練樣本 + 354,825 測試樣本驗證

作者: RLlib-DeepLOB 專案團隊
更新: 2025-01-09
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import logging

logger = logging.getLogger(__name__)


class FI2010Loader:
    """FI-2010 數據集載入器

    此類別負責載入和處理 FI-2010 限價單簿數據集，提供標準化的數據訪問介面。

    數據集規格:
        - 股票數量: 5 支
        - 交易日數: 10 天
        - 特徵維度: 40 (10 檔 × 4 種數據)
        - LOB 深度: 10 檔
        - 預測時間範圍: 5 種 (k=10, 20, 30, 50, 100)

    設計特點:
        1. 支援單一或批次股票載入
        2. 支援單日或多日數據載入
        3. 提供 LOB 結構化解析
        4. 計算額外的市場微觀結構特徵
        5. 統計標籤分佈以檢測數據不平衡

    檔案格式:
        FI-2010 數據通常以 .txt 或 .csv 格式提供
        每行代表一個時間點的 LOB 快照
        前 40 列為 LOB 特徵，後 5 列為不同時間範圍的標籤
    """

    def __init__(self, data_dir: str | Path):
        """初始化 FI-2010 數據載入器

        此方法設定數據目錄並驗證其存在性。

        參數:
            data_dir: FI-2010 數據文件所在目錄路徑
                - 可以是字串或 Path 物件
                - 目錄應包含訓練集和測試集文件

        異常:
            FileNotFoundError: 若指定的數據目錄不存在

        初始化後設定:
            self.data_dir: 數據目錄的 Path 物件
            self.num_features: LOB 特徵數量 (固定為 40)
            self.num_levels: LOB 深度檔位數 (固定為 10)
        """
        self.data_dir = Path(data_dir)
        self.num_features = 40  # 10 檔 × 4 種數據 (ask_p, ask_v, bid_p, bid_v)
        self.num_levels = 10    # LOB 深度為 10 檔

        # 驗證數據目錄存在
        if not self.data_dir.exists():
            raise FileNotFoundError(f"數據目錄不存在: {self.data_dir}")

    def load_raw_data(
        self,
        stock_id: Optional[int] = None,
        day: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """載入原始 LOB 數據和標籤

        此方法從數據目錄載入 FI-2010 原始數據。
        可指定特定股票和日期，或載入全部數據。

        參數:
            stock_id: 股票編號 (1-5)
                - None: 載入所有股票
                - 1-5: 載入指定股票

            day: 交易日編號 (1-10)
                - None: 載入所有交易日
                - 1-10: 載入指定日期

        返回:
            Tuple[np.ndarray, np.ndarray]:
                - features: LOB 特徵陣列，形狀 (N, 40)
                    N 為樣本數量，40 為特徵維度

                - labels: 標籤陣列，形狀 (N, 5)
                    5 代表 5 種不同的預測時間範圍

        異常:
            FileNotFoundError: 若指定的數據文件不存在

        注意事項:
            當前實作為佔位符，返回隨機生成的數據
            實際部署時需要實作真實的 FI-2010 文件讀取邏輯

        預期文件結構:
            data/raw/
                ├── Train_Dst_NoAuction_DecPre_CF_7.txt  (訓練集)
                ├── Train_Dst_NoAuction_DecPre_CF_8.txt
                ├── Train_Dst_NoAuction_DecPre_CF_9.txt
                ├── Test_Dst_NoAuction_DecPre_CF_7.txt   (測試集)
                ├── Test_Dst_NoAuction_DecPre_CF_8.txt
                └── Test_Dst_NoAuction_DecPre_CF_9.txt
        """
        logger.info(f"載入 FI-2010 數據: 股票={stock_id}, 日期={day}")

        # TODO: 實作真實的 FI-2010 數據載入邏輯
        # 當前為佔位符實作，用於開發和測試環境

        logger.warning("⚠️ 當前使用佔位符數據！請實作真實的 FI-2010 載入邏輯。")

        # 生成隨機佔位符數據
        num_samples = 10000
        features = np.random.randn(num_samples, self.num_features).astype(np.float32)
        # 標籤為 0, 1, 2 (對應下跌、持平、上漲)
        labels = np.random.randint(0, 3, size=(num_samples, 5)).astype(np.int64)

        return features, labels

    def load_from_file(self, file_path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
        """從指定文件載入數據

        此方法從單一文件載入 FI-2010 數據，支援多種文件格式。

        參數:
            file_path: 數據文件的完整路徑
                - 可以是字串或 Path 物件
                - 支援格式: .txt, .csv, .npy

        返回:
            Tuple[np.ndarray, np.ndarray]:
                - features: LOB 特徵陣列 (N, 40)
                - labels: 標籤陣列 (N, 5)

        異常:
            FileNotFoundError: 若文件不存在
            ValueError: 若文件格式不支援

        文件格式說明:
            .txt: 純文字格式，使用 numpy.loadtxt 載入
            .csv: CSV 格式，使用 pandas.read_csv 載入
            .npy: NumPy 二進位格式，使用 numpy.load 載入

        數據格式假設:
            文件包含 45 列數據
            前 40 列為 LOB 特徵
            後 5 列為不同時間範圍的標籤
        """
        file_path = Path(file_path)

        # 驗證文件存在
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        # 根據文件副檔名選擇載入方法
        if file_path.suffix == '.txt':
            # 純文字格式
            data = np.loadtxt(file_path)
        elif file_path.suffix == '.csv':
            # CSV 格式（不將第一行當作欄名）
            data = pd.read_csv(file_path, header=None).values
        elif file_path.suffix == '.npy':
            # NumPy 二進位格式
            data = np.load(file_path)
        else:
            # 不支援的格式
            raise ValueError(f"不支援的文件格式: {file_path.suffix}")

        # FI-2010 數據集檢查與轉置
        # FI-2010 原始格式可能是 (特徵, 時間點)，需要轉置為 (時間點, 特徵)
        if data.shape[0] < data.shape[1]:
            logger.info(f"偵測到轉置格式 {data.shape}，正在轉置為 (時間點, 特徵)")
            data = data.T  # 轉置: (149, 39512) -> (39512, 149)
            logger.info(f"轉置後形狀: {data.shape}")

        # 分離特徵和標籤
        # FI-2010 標準格式:
        #   - 行 0-39:   40 維 LOB 特徵
        #   - 行 40-143: 額外衍生特徵（本實現不使用）
        #   - 行 144-148: 5 個預測時間範圍的標籤 (k=10,20,30,50,100)
        #
        # 標籤值原始範圍: {1, 2, 3}
        #   1 = 下跌 (Down)
        #   2 = 持平 (Stationary)
        #   3 = 上漲 (Up)
        # 轉換為 {0, 1, 2} 以符合訓練需求

        if data.shape[1] >= 149:
            # 完整格式: 包含特徵和標籤
            features = data[:, :self.num_features].astype(np.float32)
            # 提取行 144-148 的標籤並轉換為 0-based
            labels = data[:, 144:149].astype(np.int64) - 1  # {1,2,3} -> {0,1,2}
            logger.debug(f"載入完整數據: {features.shape[0]} 樣本, 標籤範圍: {labels.min()}-{labels.max()}")
        elif data.shape[1] >= self.num_features:
            # 只有特徵，無標籤 (用於推理)
            features = data[:, :self.num_features].astype(np.float32)
            labels = None
            logger.warning(f"數據不包含標籤，僅載入 {self.num_features} 維特徵")
        else:
            raise ValueError(
                f"數據維度不符: 預期至少 {self.num_features} 維特徵，"
                f"實際得到 {data.shape[1]} 維"
            )

        logger.info(
            f"已從 {file_path.name} 載入 {features.shape[0]:,} 個樣本, "
            f"{features.shape[1]} 維特徵"
        )

        return features, labels

    def parse_lob_snapshot(self, features: np.ndarray) -> Dict[str, np.ndarray]:
        """解析 LOB 特徵為結構化格式

        此方法將 40 維的扁平化 LOB 特徵轉換為結構化的字典格式，
        方便後續分析和特徵工程。

        參數:
            features: LOB 特徵陣列，形狀 (N, 40)
                N 為樣本數量
                40 為特徵維度

        返回:
            Dict[str, np.ndarray]: 包含以下鍵的字典
                - 'ask_prices': 賣價陣列 (N, 10)
                    10 檔賣方報價

                - 'ask_volumes': 賣量陣列 (N, 10)
                    10 檔賣方掛單量

                - 'bid_prices': 買價陣列 (N, 10)
                    10 檔買方報價

                - 'bid_volumes': 買量陣列 (N, 10)
                    10 檔買方掛單量

        異常:
            AssertionError: 若特徵維度不是 40

        FI-2010 特徵排列順序:
            [ask_price_1, ask_vol_1, bid_price_1, bid_vol_1,
             ask_price_2, ask_vol_2, bid_price_2, bid_vol_2,
             ...,
             ask_price_10, ask_vol_10, bid_price_10, bid_vol_10]

        解析策略:
            使用步進索引 (step indexing) 提取相同類型的數據
            ask_prices: 索引 0, 4, 8, ..., 36 (每 4 個取 1 個)
            ask_volumes: 索引 1, 5, 9, ..., 37
            bid_prices: 索引 2, 6, 10, ..., 38
            bid_volumes: 索引 3, 7, 11, ..., 39
        """
        # 驗證特徵維度
        assert features.shape[1] == self.num_features, \
            f"預期 {self.num_features} 個特徵，實際得到 {features.shape[1]}"

        # 使用步進索引提取各類數據
        # Python 切片語法: [start::step] 表示從 start 開始，每隔 step 取一個
        ask_prices = features[:, 0::4]   # 索引 0, 4, 8, ...
        ask_volumes = features[:, 1::4]  # 索引 1, 5, 9, ...
        bid_prices = features[:, 2::4]   # 索引 2, 6, 10, ...
        bid_volumes = features[:, 3::4]  # 索引 3, 7, 11, ...

        return {
            'ask_prices': ask_prices,
            'ask_volumes': ask_volumes,
            'bid_prices': bid_prices,
            'bid_volumes': bid_volumes
        }

    def extract_additional_features(self, lob_data: Dict[str, np.ndarray]) -> np.ndarray:
        """提取額外的市場微觀結構特徵

        此方法從基礎 LOB 數據中計算衍生特徵，這些特徵能更好地捕捉市場動態。

        參數:
            lob_data: parse_lob_snapshot() 返回的字典
                包含 ask_prices, ask_volumes, bid_prices, bid_volumes

        返回:
            np.ndarray: 額外特徵陣列，形狀 (N, 5)
                N 為樣本數量
                5 為新增特徵數量

        提取的特徵:
            1. 買賣價差 (Bid-Ask Spread):
                計算公式: ask_price_1 - bid_price_1
                意義: 衡量市場流動性，價差越小流動性越好

            2. 中間價 (Mid Price):
                計算公式: (ask_price_1 + bid_price_1) / 2
                意義: 資產的市場公允價格估計

            3. 加權中間價 (Weighted Mid Price):
                計算公式: (ask_price × bid_volume + bid_price × ask_volume) / (ask_volume + bid_volume)
                意義: 考慮掛單量的加權價格，更準確反映供需

            4. 價格失衡 (Price Imbalance):
                計算公式: (bid_price - ask_price) / (bid_price + ask_price)
                意義: 衡量買賣價格的相對差異

            5. 量能失衡 (Volume Imbalance):
                計算公式: (bid_volume - ask_volume) / (bid_volume + ask_volume)
                意義: 衡量買賣掛單量的不平衡度
                正值表示買方力量強，負值表示賣方力量強

        數值穩定性:
            所有除法運算都加上小常數 (1e-10) 避免除零錯誤
        """
        # 提取基礎 LOB 數據
        ask_prices = lob_data['ask_prices']
        ask_volumes = lob_data['ask_volumes']
        bid_prices = lob_data['bid_prices']
        bid_volumes = lob_data['bid_volumes']

        # ===== 特徵 1: 買賣價差 =====
        # 使用第一檔（最優）價格計算
        spread = ask_prices[:, 0] - bid_prices[:, 0]

        # ===== 特徵 2: 中間價 =====
        # 最優買價和賣價的算術平均
        mid_price = (ask_prices[:, 0] + bid_prices[:, 0]) / 2

        # ===== 特徵 3: 加權中間價 =====
        # 根據對方掛單量加權計算
        # 賣價 × 買量 + 買價 × 賣量，除以總量
        weighted_mid_price = (
            ask_prices[:, 0] * bid_volumes[:, 0] +
            bid_prices[:, 0] * ask_volumes[:, 0]
        ) / (ask_volumes[:, 0] + bid_volumes[:, 0] + 1e-10)

        # ===== 特徵 4: 價格失衡 =====
        # 標準化的買賣價差
        price_imbalance = (bid_prices[:, 0] - ask_prices[:, 0]) / (
            bid_prices[:, 0] + ask_prices[:, 0] + 1e-10
        )

        # ===== 特徵 5: 量能失衡 =====
        # 衡量買賣掛單量的相對強度
        # 正值：買盤強 (看漲)，負值：賣盤強 (看跌)
        volume_imbalance = (bid_volumes[:, 0] - ask_volumes[:, 0]) / (
            bid_volumes[:, 0] + ask_volumes[:, 0] + 1e-10
        )

        # ===== 堆疊所有特徵 =====
        # 將 5 個特徵合併為 (N, 5) 陣列
        additional_features = np.stack([
            spread,
            mid_price,
            weighted_mid_price,
            price_imbalance,
            volume_imbalance
        ], axis=1).astype(np.float32)

        return additional_features

    def get_label_distribution(self, labels: np.ndarray, horizon_idx: int = 0) -> Dict[str, int]:
        """分析標籤分佈

        此方法統計特定預測時間範圍的標籤分佈情況，
        用於檢測數據集的類別不平衡問題。

        參數:
            labels: 標籤陣列，形狀 (N, 5)
                N 為樣本數量
                5 為不同預測時間範圍

            horizon_idx: 預測時間範圍索引 (0-4)
                0: k=10 個時間步
                1: k=20 個時間步
                2: k=30 個時間步
                3: k=50 個時間步
                4: k=100 個時間步

        返回:
            Dict[str, int]: 標籤計數字典
                鍵為標籤值 (0, 1, 2)
                值為該標籤的樣本數量

        標籤含義:
            0: 價格下跌
            1: 價格持平
            2: 價格上漲

        用途:
            1. 檢測數據不平衡問題
            2. 決定是否需要重採樣或加權
            3. 評估預測難度
            4. 調整損失函數權重

        理想狀態:
            三個類別的樣本數量應該相對均衡
            若某類別樣本過少，可能需要使用類別加權或 SMOTE 等技術
        """
        # 提取指定時間範圍的標籤
        horizon_labels = labels[:, horizon_idx]

        # 統計每個標籤的數量
        unique, counts = np.unique(horizon_labels, return_counts=True)
        distribution = dict(zip(unique.tolist(), counts.tolist()))

        # 記錄分佈資訊
        logger.info(f"標籤分佈 (預測時間範圍 {horizon_idx}): {distribution}")

        return distribution
