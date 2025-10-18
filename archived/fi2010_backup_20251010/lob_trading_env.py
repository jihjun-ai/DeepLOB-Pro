"""LOB 交易環境 - 強化學習訓練環境

此模組實作了基於限價單簿(LOB)數據的交易環境，符合 Gymnasium 標準介面。
用於訓練強化學習交易策略，整合 DeepLOB 價格預測模型作為輔助信號。

主要功能:
    1. 模擬高頻交易環境，支援 Hold/Buy/Sell 動作
    2. 整合 DeepLOB 模型提供價格變動預測
    3. 多組件獎勵函數（PnL、成本、庫存、風險）
    4. 追蹤交易狀態和歷史記錄
    5. 支援向量和序列兩種觀測模式

環境規格:
    觀測空間:
        - 向量模式: (48,) = LOB特徵(40) + DeepLOB預測(3) + 交易狀態(5)
        - 序列模式: (100, 40) = 100時間步 × 40維LOB特徵

    動作空間:
        - Discrete(3): {0: Hold, 1: Buy, 2: Sell}

    獎勵:
        - 多組件: PnL - 交易成本 - 庫存懲罰 - 風險懲罰

使用範例:
    >>> env = LOBTradingEnv({
    ...     'max_steps': 500,
    ...     'transaction_cost_rate': 0.001,
    ...     'deeplob_checkpoint': 'checkpoints/deeplob/best_model.pt'
    ... })
    >>> obs, info = env.reset()
    >>> action = env.action_space.sample()
    >>> obs, reward, terminated, truncated, info = env.step(action)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
from typing import Dict, Tuple, Optional, Any
import logging

from ..models.deeplob import DeepLOB
from .reward_shaper import RewardShaper
from .env_factory import EnvDataProvider

logger = logging.getLogger(__name__)


class LOBTradingEnv(gym.Env):
    """限價單簿交易環境

    這是一個 Gymnasium 標準的強化學習環境，模擬基於 LOB 數據的高頻交易。
    環境整合了 DeepLOB 模型來提供價格預測信號，幫助 RL 智能體做出更好的交易決策。

    環境特點:
        - 真實的 LOB 數據流模擬
        - 整合深度學習價格預測
        - 考慮交易成本和市場影響
        - 多維度狀態追蹤
        - 靈活的獎勵塑形機制

    狀態變數:
        current_step: 當前時間步（0 到 max_steps）
        position: 當前持倉 {-1: 空倉, 0: 平倉, 1: 多倉}
        entry_price: 進場價格
        balance: 帳戶餘額
        inventory: 未實現盈虧（mark-to-market）
        total_cost: 累計交易成本
        prev_action: 上一個動作
    """

    metadata = {'render_modes': ['human']}

    def __init__(self, config: Dict):
        """初始化 LOB 交易環境

        參數:
            config: 環境配置字典，包含以下鍵值:
                data_dir (str): LOB 數據目錄，預設 'data/processed'
                max_steps (int): 每個 episode 的最大步數，預設 500
                initial_balance (float): 初始資金，預設 10000.0
                transaction_cost_rate (float): 交易成本率，預設 0.001 (0.1%)
                max_position (int): 最大持倉量，預設 1
                obs_mode (str): 觀測模式 {'vector', 'sequence'}，預設 'vector'
                deeplob_checkpoint (str): DeepLOB 模型檢查點路徑（可選）
                reward_config (dict): 獎勵函數配置（傳遞給 RewardShaper）

        環境配置範例:
            config = {
                'max_steps': 500,
                'initial_balance': 10000.0,
                'transaction_cost_rate': 0.001,  # 0.1% 手續費
                'max_position': 1,  # 只允許 -1, 0, 1
                'obs_mode': 'vector',  # 或 'sequence'
                'deeplob_checkpoint': 'checkpoints/deeplob/best_model.pt',
                'reward_config': {
                    'pnl_scale': 1.0,
                    'cost_penalty': 1.0,
                    'inventory_penalty': 0.01,
                    'risk_penalty': 0.005
                }
            }
        """
        super().__init__()

        self.config = config

        # ===== 數據配置 =====
        self.data_dir = config.get('data_dir', 'data/processed')
        self.max_steps = config.get('max_steps', 500)  # Episode 長度
        self.initial_balance = config.get('initial_balance', 10000.0)  # 初始資金

        # ===== 交易參數 =====
        # 交易成本率（包含手續費和滑點）
        self.transaction_cost_rate = config.get('transaction_cost_rate', 0.001)
        # 最大持倉限制（風險控制）
        self.max_position = config.get('max_position', 1)

        # ===== 觀測模式 =====
        # 'vector': 當前時刻的向量觀測 (48維)
        # 'sequence': 完整時間序列 (100, 40)
        self.obs_mode = config.get('obs_mode', 'vector')

        # ===== 定義觀測空間 =====
        if self.obs_mode == 'vector':
            # 向量模式: LOB(40) + DeepLOB預測(3) + 交易狀態(5) = 48維
            obs_dim = 40 + 3 + 5
            self.observation_space = spaces.Box(
                low=-np.inf,   # 允許負值（標準化後的數據）
                high=np.inf,
                shape=(obs_dim,),
                dtype=np.float32
            )
        else:  # sequence
            # 序列模式: 100個時間步 × 40維LOB特徵
            # 用於需要完整時間序列的模型（如LSTM）
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(100, 40),
                dtype=np.float32
            )

        # ===== 定義動作空間 =====
        # Discrete(3): 三個離散動作
        #   0: Hold  - 持有當前倉位（不交易）
        #   1: Buy   - 買入（增加多倉或減少空倉）
        #   2: Sell  - 賣出（增加空倉或減少多倉）
        self.action_space = spaces.Discrete(3)

        # ===== 載入 DeepLOB 模型（可選）=====
        # DeepLOB 提供價格變動預測，作為輔助信號
        self.deeplob_model = None
        deeplob_checkpoint = config.get('deeplob_checkpoint')
        if deeplob_checkpoint:
            self._load_deeplob_model(deeplob_checkpoint)

        # ===== 初始化獎勵塑形器 =====
        # 負責計算多組件獎勵（PnL、成本、庫存、風險）
        self.reward_shaper = RewardShaper(config.get('reward_config', {}))

        # ===== 初始化數據提供者 =====
        # 使用 EnvDataProvider 載入真實 FI-2010 數據
        # 數據模式: 'train', 'val', 'test'
        self.data_mode = config.get('data_mode', 'train')
        self.data_provider = config.get('data_provider', None)  # 外部傳入的 provider

        # ===== 載入交易數據 =====
        self._load_data()

        # ===== 初始化狀態變數 =====
        self.current_step = 0      # 當前步數
        self.position = 0          # 持倉: {-1: 空倉, 0: 平倉, 1: 多倉}
        self.entry_price = 0.0     # 進場價格
        self.balance = self.initial_balance  # 帳戶餘額
        self.inventory = 0.0       # 未實現盈虧（浮動盈虧）
        self.total_cost = 0.0      # 累計交易成本
        self.prev_action = 0       # 上一個動作

        # ===== 歷史記錄 =====
        self.lob_history = []      # LOB 數據歷史（保持100個時間步）
        self.trade_history = []    # 交易記錄列表

        logger.info(f"LOB 交易環境已初始化: 觀測空間={self.observation_space.shape}")

    def _load_deeplob_model(self, checkpoint_path: str):
        """載入預訓練的 DeepLOB 模型

        DeepLOB 模型用於預測價格變動方向，其輸出作為觀測的一部分
        提供給 RL 智能體。模型在推理時會被設為評估模式（eval）。

        參數:
            checkpoint_path: 模型檢查點文件路徑 (.pt 或 .pth)

        異常處理:
            如果載入失敗，會記錄警告並使用隨機預測代替
            這樣即使沒有預訓練模型，環境仍可正常運行
        """
        try:
            # 創建 DeepLOB 模型實例
            self.deeplob_model = DeepLOB()
            # 載入權重（CPU 模式，避免 GPU 記憶體問題）
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            self.deeplob_model.load_state_dict(state_dict)
            # 設為評估模式（關閉 dropout 等）
            self.deeplob_model.eval()
            logger.info(f"成功載入 DeepLOB 模型: {checkpoint_path}")
        except Exception as e:
            logger.warning(f"DeepLOB 模型載入失敗: {e}。將使用隨機預測。")
            self.deeplob_model = None

    def _load_data(self):
        """載入 LOB 交易數據

        從 EnvDataProvider 載入預處理後的真實 FI-2010 數據。
        根據環境配置的 data_mode，載入對應的數據分割：
            - 'train': 訓練集數據（用於訓練 RL 策略）
            - 'val': 驗證集數據（用於訓練過程中的評估）
            - 'test': 測試集數據（用於最終評估）

        數據格式:
            - lob_data: (N, 40) LOB 特徵矩陣
              * N: 時間步數
              * 40: LOB 特徵維度（10 檔買賣價量）
            - prices: (N,) 中間價格序列
              * 用於計算 PnL 和獎勵

        數據來源:
            1. 如果提供了 data_provider，直接使用
            2. 否則創建新的 EnvDataProvider 實例
            3. 根據 data_mode 獲取對應數據分割

        異常處理:
            如果數據載入失敗，會降級到隨機數據模式
            這確保環境在數據缺失時仍可運行（用於測試）

        注意:
            - 訓練環境使用 'train' 模式
            - 評估環境使用 'val' 或 'test' 模式
            - 數據會被快取，避免重複載入
        """
        try:
            # ===== 步驟1: 獲取或創建數據提供者 =====
            if self.data_provider is None:
                # 沒有外部傳入 provider，創建新實例
                logger.info(f"創建新的數據提供者，數據目錄: {self.data_dir}")
                self.data_provider = EnvDataProvider(
                    data_dir=self.data_dir,
                    normalization_method='z-score',
                    sequence_length=100,
                    train_ratio=0.6,
                    val_ratio=0.2,
                    test_ratio=0.2
                )

            # ===== 步驟2: 根據模式載入數據 =====
            if self.data_mode == 'train':
                self.lob_data, self.prices = self.data_provider.get_train_data()
                logger.info("✅ 載入訓練集數據")
            elif self.data_mode == 'val':
                self.lob_data, self.prices = self.data_provider.get_val_data()
                logger.info("✅ 載入驗證集數據")
            elif self.data_mode == 'test':
                self.lob_data, self.prices = self.data_provider.get_test_data()
                logger.info("✅ 載入測試集數據")
            else:
                raise ValueError(f"未知的數據模式: {self.data_mode}，應為 'train'/'val'/'test'")

            # ===== 步驟3: 驗證數據格式 =====
            assert self.lob_data.shape[1] == 40, \
                f"LOB 數據維度錯誤: 期望 40，實際 {self.lob_data.shape[1]}"
            assert len(self.lob_data) == len(self.prices), \
                f"數據長度不匹配: LOB={len(self.lob_data)}, Prices={len(self.prices)}"

            self.data_length = len(self.lob_data)
            logger.info(
                f"✅ 成功載入 {self.data_mode} 數據: "
                f"{self.data_length:,} 個時間步, "
                f"LOB 形狀={self.lob_data.shape}, "
                f"價格範圍=[{self.prices.min():.2f}, {self.prices.max():.2f}]"
            )

        except Exception as e:
            # ===== 降級處理：使用隨機數據 =====
            logger.error(f"❌ 數據載入失敗: {e}")
            logger.warning("⚠️ 降級到隨機數據模式（僅用於測試）")

            # 生成隨機 LOB 數據
            self.lob_data = np.random.randn(1000, 40).astype(np.float32)
            # 生成隨機價格序列（累積隨機漫步）
            self.prices = np.cumsum(np.random.randn(1000) * 0.01).astype(np.float32) + 100.0

            self.data_length = len(self.lob_data)
            logger.warning(f"使用隨機數據: {self.data_length} 個時間步")

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """重置環境到初始狀態

        每個 episode 開始時調用此方法，將所有狀態變數重置，
        並隨機選擇一個起始位置開始新的交易 episode。

        參數:
            seed: 隨機種子，用於可重現的實驗
            options: 額外選項（目前未使用）

        返回:
            obs: 初始觀測，形狀為 (48,) 或 (100, 40)
            info: 資訊字典，包含當前狀態統計

        重置流程:
            1. 重置所有交易狀態變數
            2. 清空歷史記錄
            3. 隨機選擇數據起始位置
            4. 初始化 LOB 歷史（需要100個時間步）
            5. 生成初始觀測

        注意:
            - Episode 起始位置是隨機的，確保訓練多樣性
            - 需要預留100個時間步用於 LOB 歷史
            - 每次 reset 都會選擇不同的市場狀態
        """
        # 調用父類 reset（處理隨機種子）
        super().reset(seed=seed)

        # ===== 重置交易狀態 =====
        self.current_step = 0              # 重置步數計數器
        self.position = 0                  # 平倉狀態
        self.entry_price = 0.0             # 清空進場價
        self.balance = self.initial_balance  # 重置資金
        self.inventory = 0.0               # 清空未實現盈虧
        self.total_cost = 0.0              # 清空累計成本
        self.prev_action = 0               # 重置前一動作

        # ===== 清空歷史記錄 =====
        self.lob_history = []              # 清空 LOB 歷史
        self.trade_history = []            # 清空交易記錄

        # ===== 隨機選擇 Episode 起始位置 =====
        # 確保有足夠的數據（需要100步LOB歷史 + max_steps交易步數）
        max_start = max(1, self.data_length - self.max_steps - 100)
        start_idx = np.random.randint(0, max_start)
        self.episode_start_idx = start_idx

        # 初始化 LOB 歷史（前100個時間步）
        # 這是 DeepLOB 和序列模型所需的輸入
        self.lob_history = self.lob_data[start_idx:start_idx + 100].tolist()

        # ===== 生成初始觀測和資訊 =====
        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """執行一個時間步的交易動作

        這是環境的核心方法，執行智能體選擇的動作並返回結果。

        參數:
            action: 動作選擇
                0 - Hold: 持有當前倉位
                1 - Buy: 買入（做多或平空）
                2 - Sell: 賣出（做空或平多）

        返回:
            obs: 新的觀測
            reward: 即時獎勵（多組件合成）
            terminated: Episode 是否因達成目標而結束
            truncated: Episode 是否因步數限制而截斷
            info: 詳細資訊字典

        執行流程:
            1. 獲取當前價格
            2. 執行交易動作，更新倉位
            3. 計算交易成本
            4. 更新庫存（mark-to-market）
            5. 推進到下一時間步
            6. 計算獎勵
            7. 檢查終止條件
            8. 返回新觀測和獎勵

        動作映射:
            動作 0 (Hold) 對應倉位不變
            動作 1 (Buy) 對應倉位 +1（受 max_position 限制）
            動作 2 (Sell) 對應倉位 -1（受 max_position 限制）

        獎勵組成:
            - PnL: 盈虧（正為盈利，負為虧損）
            - 交易成本: 手續費和滑點（負值）
            - 庫存懲罰: 避免長時間持倉（負值）
            - 風險懲罰: 基於波動率的風險（負值）
        """
        # ===== 步驟1: 獲取當前市場價格 =====
        current_idx = self.episode_start_idx + len(self.lob_history) - 1
        current_price = self.prices[current_idx]

        # ===== 步驟2: 保存前一狀態（用於獎勵計算）=====
        prev_position = self.position
        prev_state = {
            'position': self.position,
            'entry_price': self.entry_price,
            'prev_action': self.prev_action
        }

        # ===== 步驟3: 執行交易動作 =====
        transaction_cost = 0.0
        action_changed = False

        # 檢查動作是否導致倉位變化
        # 動作映射: Hold=0, Buy=1, Sell=2
        # 倉位映射: Short=-1, Flat=0, Long=1
        target_position_change = action - 1  # {-1, 0, 1}

        if action != prev_position + 1:  # 倉位需要改變
            action_changed = True

            # 計算交易成本（基於交易金額）
            position_change = abs(action - (prev_position + 1))
            transaction_cost = position_change * current_price * self.transaction_cost_rate
            self.total_cost += transaction_cost

            # ===== 更新倉位 =====
            if action == 0:  # Sell（做空或減倉）
                new_position = self.position - 1
                self.position = max(-self.max_position, new_position)
            elif action == 2:  # Buy（做多或加倉）
                new_position = self.position + 1
                self.position = min(self.max_position, new_position)
            # action == 1: Hold，倉位不變

            # ===== 記錄交易（如果倉位實際改變）=====
            if self.position != prev_position:
                self.trade_history.append({
                    'step': self.current_step,
                    'action': action,
                    'price': current_price,
                    'position': self.position,
                    'cost': transaction_cost
                })

            # ===== 更新進場價格（開新倉時）=====
            # 從平倉狀態開倉時，記錄進場價
            if prev_position == 0 and self.position != 0:
                self.entry_price = current_price

        # ===== 步驟4: 更新庫存（未實現盈虧）=====
        # 使用 mark-to-market 方式計算浮動盈虧
        if self.position != 0:
            # 庫存 = 倉位 × (當前價 - 進場價)
            self.inventory = self.position * (current_price - self.entry_price)
        else:
            # 平倉狀態無庫存
            self.inventory = 0.0

        # ===== 步驟5: 推進到下一時間步 =====
        self.current_step += 1

        # 添加新的 LOB 數據到歷史
        if self.episode_start_idx + len(self.lob_history) < self.data_length:
            next_lob = self.lob_data[self.episode_start_idx + len(self.lob_history)]
            self.lob_history.append(next_lob.tolist())

            # 保持固定長度的歷史窗口（100個時間步）
            if len(self.lob_history) > 100:
                self.lob_history.pop(0)  # 移除最舊的數據

        # ===== 步驟6: 獲取新價格 =====
        next_idx = min(self.episode_start_idx + len(self.lob_history) - 1,
                       self.data_length - 1)
        next_price = self.prices[next_idx]
        price_change = next_price - current_price

        # ===== 步驟7: 構建新狀態（用於獎勵計算）=====
        new_state = {
            'position': self.position,
            'current_price': next_price,
            'inventory': self.inventory,
            'volatility': 0.01  # TODO: 實作真實的波動率計算
        }

        # ===== 步驟8: 計算獎勵 =====
        # 使用 RewardShaper 計算多組件獎勵
        reward, reward_info = self.reward_shaper.calculate_reward(
            prev_state=prev_state,
            action=action,
            new_state=new_state,
            transaction_cost=transaction_cost
        )

        # ===== 步驟9: 檢查終止條件 =====
        # terminated: 達成某個目標（當前未使用）
        # truncated: 達到最大步數限制
        terminated = False
        truncated = self.current_step >= self.max_steps

        # ===== 步驟10: 生成觀測和資訊 =====
        obs = self._get_observation()
        info = self._get_info()

        # 添加額外資訊
        info.update(reward_info)  # 包含獎勵各組件
        info['action'] = action
        info['transaction_cost'] = transaction_cost
        info['price'] = next_price
        info['price_change'] = price_change

        # 更新前一動作
        self.prev_action = action

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """獲取當前觀測

        根據配置的觀測模式生成觀測向量或序列。

        觀測模式:
            1. 向量模式 (vector):
               - 當前 LOB 特徵 (40維)
               - DeepLOB 預測機率 (3維): [下跌, 穩定, 上漲]
               - 交易狀態 (5維):
                 * 標準化倉位: position / max_position
                 * 標準化庫存: inventory / initial_balance
                 * 標準化成本: total_cost / initial_balance
                 * 時間進度: current_step / max_steps
                 * 標準化前一動作: prev_action / 2.0
               總計: 48維向量

            2. 序列模式 (sequence):
               - 完整 LOB 時間序列: (100, 40)
               - 用於需要完整歷史的模型（如 LSTM）

        返回:
            obs: 觀測數組
                - 向量模式: (48,)
                - 序列模式: (100, 40)

        DeepLOB 預測:
            如果載入了 DeepLOB 模型，會進行前向推理獲取預測
            如果未載入模型，則使用均勻隨機機率代替
            這確保環境在沒有預訓練模型時仍可運行

        狀態標準化:
            所有狀態特徵都經過標準化，範圍大致在 [-1, 1]
            這有助於神經網路訓練的穩定性
        """
        # 獲取當前時刻的 LOB 特徵
        current_lob = np.array(self.lob_history[-1], dtype=np.float32)

        # ===== 序列模式：直接返回完整序列 =====
        if self.obs_mode == 'sequence':
            return np.array(self.lob_history, dtype=np.float32)

        # ===== 向量模式：組合多種特徵 =====

        # --- 1. DeepLOB 價格預測 (3維) ---
        if self.deeplob_model is not None:
            # 使用預訓練模型進行推理
            with torch.no_grad():  # 不需要梯度
                # 轉換為 tensor: (1, 100, 40)
                lob_seq = torch.FloatTensor(self.lob_history).unsqueeze(0)
                # 獲取預測機率: (3,) [下跌, 穩定, 上漲]
                deeplob_probs = self.deeplob_model.predict_proba(lob_seq)[0].numpy()
        else:
            # 沒有模型時使用隨機機率
            deeplob_probs = np.random.rand(3).astype(np.float32)
            deeplob_probs /= deeplob_probs.sum()  # 標準化為機率分佈

        # --- 2. 交易狀態特徵 (5維) ---
        state_features = np.array([
            # 標準化倉位 [-1, 1]
            self.position / self.max_position if self.max_position > 0 else 0.0,

            # 標準化庫存（相對於初始資金）
            self.inventory / self.initial_balance,

            # 標準化累計成本（相對於初始資金）
            self.total_cost / self.initial_balance,

            # 時間進度 [0, 1]
            self.current_step / self.max_steps,

            # 標準化前一動作 [0, 1]
            self.prev_action / 2.0  # 動作範圍 {0, 1, 2}
        ], dtype=np.float32)

        # --- 3. 串接所有特徵 ---
        # 最終觀測: LOB(40) + DeepLOB(3) + State(5) = 48維
        obs = np.concatenate([current_lob, deeplob_probs, state_features])

        return obs

    def _get_info(self) -> Dict[str, Any]:
        """獲取環境資訊字典

        返回當前環境狀態的詳細資訊，用於監控和除錯。

        返回:
            info: 包含以下鍵的字典:
                step: 當前步數
                position: 當前倉位
                balance: 帳戶餘額
                inventory: 未實現盈虧
                total_cost: 累計交易成本
                num_trades: 交易次數

        用途:
            - 訓練過程監控
            - 績效統計
            - 除錯和分析
            - 記錄到 TensorBoard 或日誌
        """
        return {
            'step': self.current_step,           # 當前步數
            'position': self.position,           # 倉位 {-1, 0, 1}
            'balance': self.balance,             # 餘額
            'inventory': self.inventory,         # 浮動盈虧
            'total_cost': self.total_cost,       # 累計成本
            'num_trades': len(self.trade_history)  # 交易次數
        }

    def render(self):
        """渲染環境狀態（文字輸出）

        定期輸出環境狀態資訊，用於監控訓練過程。

        輸出頻率:
            每100步輸出一次

        輸出內容:
            - 當前步數
            - 當前倉位
            - 未實現盈虧
            - 累計交易次數

        注意:
            這是簡化版實作，實際應用中可以：
            - 繪製價格走勢圖
            - 顯示持倉變化
            - 展示盈虧曲線
            - 使用 matplotlib 可視化
        """
        if self.current_step % 100 == 0:
            print(
                f"[Step {self.current_step}] "
                f"倉位: {self.position:+d}, "
                f"庫存: {self.inventory:+.2f}, "
                f"交易次數: {len(self.trade_history)}"
            )

    def close(self):
        """清理環境資源

        關閉環境時調用，用於釋放資源。

        當前實作:
            空方法（無需清理）

        未來可能需要:
            - 關閉數據庫連接
            - 保存交易記錄
            - 釋放 GPU 記憶體
            - 關閉日誌文件
        """
        pass


# ===== 環境註冊輔助函數 =====

def make_lob_trading_env(config: Dict) -> LOBTradingEnv:
    """創建 LOB 交易環境的工廠函數

    這是一個便利函數，用於創建配置好的環境實例。

    參數:
        config: 環境配置字典

    返回:
        配置好的 LOBTradingEnv 實例

    使用範例:
        >>> config = {
        ...     'max_steps': 500,
        ...     'transaction_cost_rate': 0.001,
        ...     'deeplob_checkpoint': 'checkpoints/deeplob/best_model.pt'
        ... }
        >>> env = make_lob_trading_env(config)
        >>> obs, info = env.reset()
    """
    return LOBTradingEnv(config)
