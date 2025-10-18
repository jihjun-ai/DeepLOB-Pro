"""訓練監控回調 - TrainingMonitorCallbacks

此模組實作 RLlib 訓練回調，用於監控強化學習訓練過程中的各種指標。
提供實時性能監控、異常檢測和訓練診斷功能。

主要功能:
    1. 監控訓練指標 (reward, loss, entropy, KL divergence)
    2. 監控環境指標 (交易統計, PnL, 勝率)
    3. 監控模型健康 (梯度範數, 動作分佈)
    4. TensorBoard 日誌記錄
    5. 異常檢測與告警

回調觸發點:
    - on_episode_start: Episode 開始時
    - on_episode_step: 每個 step 後
    - on_episode_end: Episode 結束時
    - on_train_result: 每個訓練迭代後
    - on_algorithm_init: 算法初始化時

使用範例:
    >>> from ray.rllib.algorithms.ppo import PPOConfig
    >>> config = PPOConfig()
    >>> config.callbacks(TrainingMonitorCallbacks)
    >>> algo = config.build()
    >>> algo.train()

作者: RLlib-DeepLOB 專案團隊
更新: 2025-01-10
"""

import numpy as np
from typing import Dict, Optional, Any
import logging
from collections import defaultdict, deque

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import PolicyID

# 兼容舊版 RLlib：Episode 可能不存在，使用 EpisodeV2
try:
    from ray.rllib.evaluation import Episode
except ImportError:
    Episode = EpisodeV2  # 舊版 Ray 使用 EpisodeV2

logger = logging.getLogger(__name__)


class TrainingMonitorCallbacks(DefaultCallbacks):
    """訓練監控回調類別

    此類別擴展 RLlib 的 DefaultCallbacks，添加自定義的監控邏輯。
    在訓練過程中收集和記錄各種性能指標，幫助診斷訓練問題。

    監控類別:
        1. **基礎訓練指標**:
           - episode_reward: Episode 總獎勵
           - episode_length: Episode 長度
           - policy_loss, vf_loss: 策略和價值函數損失
           - entropy: 策略熵（探索度）
           - kl: KL 散度

        2. **交易統計**:
           - num_trades: 交易次數
           - win_rate: 勝率
           - avg_pnl: 平均盈虧
           - total_cost: 交易成本
           - position_distribution: 持倉分佈

        3. **模型健康**:
           - grad_norm: 梯度範數
           - action_distribution: 動作分佈
           - deeplob_prediction_variance: 預測方差

        4. **異常檢測**:
           - reward 是否退化（總是 0 或常數）
           - 動作分佈是否退化（只選一個動作）
           - 梯度爆炸/消失

    屬性:
        episode_rewards (deque): 最近 N 個 episode 的獎勵
        episode_lengths (deque): 最近 N 個 episode 的長度
        action_counts (defaultdict): 動作選擇次數統計
        trade_stats (dict): 交易統計資訊
    """

    def __init__(self):
        """初始化回調

        設置內部狀態變數和統計容器。
        """
        super().__init__()

        # ===== 性能統計容器 =====
        # 保持最近 100 個 episode 的獎勵歷史
        self.episode_rewards = deque(maxlen=100)
        # 保持最近 100 個 episode 的長度歷史
        self.episode_lengths = deque(maxlen=100)

        # ===== 交易統計 =====
        self.action_counts = defaultdict(int)  # 動作計數
        self.trade_stats = {
            'num_trades': 0,
            'total_pnl': 0.0,
            'winning_trades': 0,
            'losing_trades': 0,
        }

        # ===== 診斷標誌 =====
        self.iteration_count = 0  # 訓練迭代計數器
        self.last_log_iter = 0    # 上次記錄日誌的迭代

        # ===== Episode 數據存儲 (新 API 兼容) =====
        # SingleAgentEpisode 不允許動態添加屬性，使用外部字典
        self.episode_data = {}  # {episode_id: {'actions': [], 'positions': [], ...}}

        logger.info("[OK] TrainingMonitorCallbacks 已初始化")

    def _get_episode_id(self, episode):
        """獲取 episode 的唯一 ID"""
        if hasattr(episode, 'episode_id'):
            return episode.episode_id
        elif hasattr(episode, 'id_'):
            return episode.id_
        else:
            return id(episode)  # 使用對象 ID 作為fallback

    def _get_episode_data(self, episode, key: str, default=None):
        """統一獲取 episode 數據（兼容新舊 API）"""
        if hasattr(episode, 'user_data'):
            # 舊 API
            return episode.user_data.get(key, default) if default is not None else episode.user_data[key]
        else:
            # 新 API - 使用外部存儲
            episode_id = self._get_episode_id(episode)
            if episode_id not in self.episode_data:
                return default
            return self.episode_data[episode_id].get(key, default)

    def _set_episode_data(self, episode, key: str, value):
        """統一設置 episode 數據（兼容新舊 API）"""
        if hasattr(episode, 'user_data'):
            # 舊 API
            episode.user_data[key] = value
        else:
            # 新 API - 使用外部存儲
            episode_id = self._get_episode_id(episode)
            if episode_id not in self.episode_data:
                self.episode_data[episode_id] = {}
            self.episode_data[episode_id][key] = value

    def _append_episode_data(self, episode, key: str, value):
        """統一追加 episode 數據（兼容新舊 API）"""
        data = self._get_episode_data(episode, key, [])
        if data is None:
            data = []
        data.append(value)
        self._set_episode_data(episode, key, data)

    def _cleanup_episode_data(self, episode):
        """清理完成的 episode 數據"""
        if not hasattr(episode, 'user_data'):
            # 只在新 API 需要清理
            episode_id = self._get_episode_id(episode)
            if episode_id in self.episode_data:
                del self.episode_data[episode_id]

    def on_episode_start(
        self,
        *,
        episode: Episode | EpisodeV2,
        worker: Optional[RolloutWorker] = None,
        base_env: Optional[BaseEnv] = None,
        policies: Optional[Dict[PolicyID, Policy]] = None,
        env_runner: Optional[Any] = None,
        env: Optional[Any] = None,
        env_index: int = 0,
        **kwargs
    ) -> None:
        """Episode 開始時的回調

        在每個 episode 開始時調用，用於初始化 episode 級別的統計。

        參數:
            worker: RolloutWorker 實例
            base_env: 環境
            policies: 策略字典
            episode: Episode 物件
            **kwargs: 額外參數

        初始化:
            - 重置 episode 統計
            - 初始化自定義統計變數
        """
        # 初始化 episode 自定義統計（兼容新舊 API）
        self._set_episode_data(episode, 'actions', [])
        self._set_episode_data(episode, 'positions', [])
        self._set_episode_data(episode, 'trades', [])
        self._set_episode_data(episode, 'pnl_components', [])

    def on_episode_step(
        self,
        *,
        episode: Episode | EpisodeV2,
        worker: Optional[RolloutWorker] = None,
        base_env: Optional[BaseEnv] = None,
        policies: Optional[Dict[PolicyID, Policy]] = None,
        env_runner: Optional[Any] = None,
        env: Optional[Any] = None,
        env_index: int = 0,
        **kwargs
    ) -> None:
        """Episode 每步執行後的回調

        在每個時間步執行後調用，用於收集細粒度的統計資訊。

        參數:
            worker: RolloutWorker 實例
            base_env: 環境
            policies: 策略字典
            episode: Episode 物件
            **kwargs: 額外參數

        收集信息:
            - 動作選擇
            - 持倉狀態
            - 交易事件
            - 獎勵組件
        """
        # 獲取最後一個 step 的資訊
        # 兼容新舊 API: SingleAgentEpisode 使用 t, EpisodeV2 使用 length
        episode_len = episode.t if hasattr(episode, 't') else episode.length
        if episode_len == 0:
            return  # 跳過第一個 step

        # 從環境獲取 info
        try:
            # 嘗試從 base_env 獲取環境 ID
            sub_envs = base_env.get_sub_environments()
            if isinstance(sub_envs, dict):
                env_id = list(sub_envs.keys())[0]
            elif isinstance(sub_envs, list):
                env_id = 0  # 使用索引 0
            else:
                env_id = 0  # 預設使用 0

            last_info = episode.last_info_for(env_id)
        except Exception:
            # 無法獲取 info，跳過
            return

        # 記錄動作 (兼容 EpisodeV2 API)
        try:
            # 嘗試新 API
            if hasattr(episode, 'last_action_for'):
                last_action = episode.last_action_for()
            else:
                # EpisodeV2 使用 _last_actions 字典
                last_action = episode._last_actions.get(env_id)
        except Exception:
            last_action = None

        if last_action is not None:
            self._append_episode_data(episode, 'actions', int(last_action))
            self.action_counts[int(last_action)] += 1

        # ===== 關鍵修正: 檢查 last_info 是否為 None =====
        if last_info is None:
            return  # 如果 last_info 為 None，直接返回

        # 記錄持倉
        if 'position' in last_info:
            self._append_episode_data(episode, 'positions', last_info['position'])

        # 記錄交易
        if last_info.get('num_trades', 0) > len(self._get_episode_data(episode, 'trades', [])):
            # 兼容新舊 API
            step = episode.t if hasattr(episode, 't') else episode.length
            self._append_episode_data(episode, 'trades', {
                'step': step,
                'action': int(last_action) if last_action is not None else None,
                'cost': last_info.get('transaction_cost', 0.0)
            })

        # 記錄獎勵組件（如果環境提供）
        if 'pnl' in last_info:
            self._append_episode_data(episode, 'pnl_components', {
                'pnl': last_info.get('pnl', 0.0),
                'cost': last_info.get('cost', 0.0),
                'inventory_penalty': last_info.get('inventory_penalty', 0.0),
                'risk_penalty': last_info.get('risk_penalty', 0.0),
            })

    def on_episode_end(
        self,
        *,
        episode: Episode | EpisodeV2,
        worker: Optional[RolloutWorker] = None,
        base_env: Optional[BaseEnv] = None,
        policies: Optional[Dict[PolicyID, Policy]] = None,
        env_runner: Optional[Any] = None,
        env: Optional[Any] = None,
        env_index: int = 0,
        **kwargs
    ) -> None:
        """Episode 結束時的回調

        在每個 episode 結束時調用，用於聚合和記錄 episode 統計。

        參數:
            worker: RolloutWorker 實例
            base_env: 環境
            policies: 策略字典
            episode: Episode 物件
            **kwargs: 額外參數

        記錄指標:
            - episode_reward: 總獎勵
            - episode_length: 長度
            - num_trades: 交易次數
            - action_entropy: 動作熵
            - avg_pnl: 平均 PnL
        """
        # 記錄基礎指標
        # 兼容新舊 API: SingleAgentEpisode 使用 t 和 get_return(), EpisodeV2 使用 length 和 total_reward
        episode_reward = episode.get_return() if hasattr(episode, 'get_return') else episode.total_reward
        episode_length = episode.t if hasattr(episode, 't') else episode.length

        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)

        # ===== 交易統計 =====
        num_trades = len(self._get_episode_data(episode, 'trades', []))

        # 兼容新舊 API: SingleAgentEpisode 沒有 custom_metrics，暫存在自定義字典
        def set_custom_metric(episode, key, value):
            if hasattr(episode, 'custom_metrics'):
                episode.custom_metrics[key] = value
            else:
                # 新 API: 暫存在 episode_data，稍後在 on_train_result 中聚合
                self._set_episode_data(episode, f'metric_{key}', value)

        set_custom_metric(episode, 'num_trades', num_trades)

        # 動作分佈熵（探索度）
        if len(self._get_episode_data(episode, 'actions', [])) > 0:
            actions = np.array(self._get_episode_data(episode, 'actions', []))
            # 計算動作分佈
            action_dist = np.bincount(actions, minlength=3) / len(actions)
            # 計算熵: -sum(p * log(p))
            entropy = -np.sum(action_dist * np.log(action_dist + 1e-10))
            set_custom_metric(episode, 'action_entropy', entropy)

            # 記錄每個動作的比例
            set_custom_metric(episode, 'action_0_ratio', action_dist[0])  # Hold
            set_custom_metric(episode, 'action_1_ratio', action_dist[1])  # Buy
            set_custom_metric(episode, 'action_2_ratio', action_dist[2])  # Sell

        # ===== 持倉統計 =====
        if len(self._get_episode_data(episode, 'positions', [])) > 0:
            positions = np.array(self._get_episode_data(episode, 'positions', []))
            # 持倉時間比例
            long_ratio = np.mean(positions == 1)   # 多倉
            short_ratio = np.mean(positions == -1) # 空倉
            flat_ratio = np.mean(positions == 0)   # 平倉

            set_custom_metric(episode, 'long_position_ratio', long_ratio)
            set_custom_metric(episode, 'short_position_ratio', short_ratio)
            set_custom_metric(episode, 'flat_position_ratio', flat_ratio)

        # ===== PnL 組件統計 =====
        if len(self._get_episode_data(episode, 'pnl_components', [])) > 0:
            pnl_components = self._get_episode_data(episode, 'pnl_components', [])

            # 平均各組件
            avg_pnl = np.mean([c['pnl'] for c in pnl_components])
            avg_cost = np.mean([c['cost'] for c in pnl_components])
            avg_inventory_penalty = np.mean([c['inventory_penalty'] for c in pnl_components])
            avg_risk_penalty = np.mean([c['risk_penalty'] for c in pnl_components])

            set_custom_metric(episode, 'avg_pnl', avg_pnl)
            set_custom_metric(episode, 'avg_cost', avg_cost)
            set_custom_metric(episode, 'avg_inventory_penalty', avg_inventory_penalty)
            set_custom_metric(episode, 'avg_risk_penalty', avg_risk_penalty)

            # 總 PnL
            total_pnl = sum([c['pnl'] for c in pnl_components])
            set_custom_metric(episode, 'total_pnl', total_pnl)

            # 勝率（正 PnL 步數比例）
            if len(pnl_components) > 0:
                win_steps = sum(1 for c in pnl_components if c['pnl'] > 0)
                win_rate = win_steps / len(pnl_components)
                set_custom_metric(episode, 'win_rate', win_rate)

        # 清理 episode 數據
        self._cleanup_episode_data(episode)

    def on_train_result(
        self,
        *,
        algorithm,
        result: Dict,
        **kwargs
    ) -> None:
        """訓練迭代結束時的回調

        在每個訓練迭代（多個 episode）後調用，用於記錄訓練級別的統計。

        參數:
            algorithm: 訓練算法實例
            result: 訓練結果字典
            **kwargs: 額外參數

        記錄內容:
            - 訓練損失（policy_loss, vf_loss）
            - 探索指標（entropy, KL）
            - 性能趨勢（moving average）
            - 異常檢測
        """
        self.iteration_count += 1

        # ===== 基礎訓練指標 =====
        # 這些指標由 RLlib 自動計算
        metrics = {
            'episode_reward_mean': result.get('episode_reward_mean', 0.0),
            'episode_len_mean': result.get('episode_len_mean', 0.0),
            'episodes_this_iter': result.get('episodes_this_iter', 0),
            'timesteps_this_iter': result.get('timesteps_this_iter', 0),
        }

        # ===== 策略訓練指標 =====
        # 從 learner_stats 獲取
        learner_stats = result.get('info', {}).get('learner', {}).get('default_policy', {})
        if learner_stats:
            metrics.update({
                'policy_loss': learner_stats.get('learner_stats', {}).get('policy_loss', 0.0),
                'vf_loss': learner_stats.get('learner_stats', {}).get('vf_loss', 0.0),
                'entropy': learner_stats.get('learner_stats', {}).get('entropy', 0.0),
                'kl': learner_stats.get('learner_stats', {}).get('kl', 0.0),
                'grad_gnorm': learner_stats.get('learner_stats', {}).get('grad_gnorm', 0.0),
            })

        # ===== 移動平均統計 =====
        if len(self.episode_rewards) > 0:
            result['custom_metrics']['reward_moving_avg'] = np.mean(list(self.episode_rewards))
            result['custom_metrics']['reward_std'] = np.std(list(self.episode_rewards))

        if len(self.episode_lengths) > 0:
            result['custom_metrics']['length_moving_avg'] = np.mean(list(self.episode_lengths))

        # ===== 動作分佈統計 =====
        total_actions = sum(self.action_counts.values())
        if total_actions > 0:
            action_dist = {
                f'action_{k}_ratio': v / total_actions
                for k, v in self.action_counts.items()
            }
            result['custom_metrics'].update(action_dist)

        # ===== 異常檢測 =====
        warnings = []

        # 檢測 1: Reward 退化（總是 0）
        if len(self.episode_rewards) >= 10:
            recent_rewards = list(self.episode_rewards)[-10:]
            if all(abs(r) < 1e-6 for r in recent_rewards):
                warnings.append("[WARN] Reward 全為 0！檢查獎勵函數。")

        # 檢測 2: 動作退化（只選一個動作）
        if total_actions > 100:
            max_action_ratio = max(self.action_counts.values()) / total_actions
            if max_action_ratio > 0.95:
                warnings.append(f"[WARN] 動作分佈退化！{max_action_ratio*100:.1f}% 選同一動作。")

        # 檢測 3: 梯度爆炸
        grad_norm = metrics.get('grad_gnorm', 0.0)
        if grad_norm > 10.0:
            warnings.append(f"[WARN] 梯度範數過大: {grad_norm:.2f}。可能需要梯度裁剪。")

        # 輸出警告
        if warnings and self.iteration_count % 10 == 0:
            for warning in warnings:
                logger.warning(warning)

        # ===== 定期日誌輸出 =====
        if self.iteration_count % 10 == 0:
            # 構建動作分佈字串（僅當有動作數據時）
            action_dist_str = ""
            if total_actions > 0:
                action_dist_str = (
                    f"\n"
                    f"🎯 動作分佈:\n"
                    f"  - Hold: {action_dist.get('action_0_ratio', 0)*100:.1f}%\n"
                    f"  - Buy: {action_dist.get('action_1_ratio', 0)*100:.1f}%\n"
                    f"  - Sell: {action_dist.get('action_2_ratio', 0)*100:.1f}%\n"
                )

            logger.info(
                f"\n{'='*80}\n"
                f"訓練迭代 {self.iteration_count}\n"
                f"{'='*80}\n"
                f"📊 性能指標:\n"
                f"  - Episode Reward (平均): {metrics.get('episode_reward_mean', 0):.4f}\n"
                f"  - Episode Length (平均): {metrics.get('episode_len_mean', 0):.1f}\n"
                f"  - Episodes: {metrics.get('episodes_this_iter', 0)}\n"
                f"  - Timesteps: {metrics.get('timesteps_this_iter', 0)}\n"
                f"\n"
                f"🧠 訓練指標:\n"
                f"  - Policy Loss: {metrics.get('policy_loss', 0):.4f}\n"
                f"  - Value Loss: {metrics.get('vf_loss', 0):.4f}\n"
                f"  - Entropy: {metrics.get('entropy', 0):.4f}\n"
                f"  - KL: {metrics.get('kl', 0):.6f}\n"
                f"  - Grad Norm: {metrics.get('grad_gnorm', 0):.4f}\n"
                f"{action_dist_str}"
                f"{'='*80}\n"
            )
