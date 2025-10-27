"""TensorBoard 數據分析腳本

此腳本自動從 TensorBoard 日誌中提取關鍵指標，生成結構化報告，
方便 AI 分析訓練狀況並提供優化建議。

使用方法:
    python scripts/analyze_tensorboard.py --logdir logs/sb3_deeplob/PPO_1
    python scripts/analyze_tensorboard.py --logdir logs/sb3_deeplob/PPO_1 --output results/analysis.json

輸出:
    - JSON 格式的結構化數據
    - Markdown 格式的報告
    - 可視化圖表（可選）
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
from datetime import datetime

# 添加項目根目錄到 Python 路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def parse_args():
    """解析命令行參數"""
    parser = argparse.ArgumentParser(
        description="分析 TensorBoard 訓練日誌",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
    # 分析單個訓練日誌
    python scripts/analyze_tensorboard.py --logdir logs/sb3_deeplob/PPO_1

    # 分析並保存 JSON 報告
    python scripts/analyze_tensorboard.py --logdir logs/sb3_deeplob/PPO_1 --output results/analysis.json

    # 分析並生成 Markdown 報告
    python scripts/analyze_tensorboard.py --logdir logs/sb3_deeplob/PPO_1 --format markdown

    # 比較多個實驗
    python scripts/analyze_tensorboard.py --compare logs/sb3_deeplob/
        """
    )

    parser.add_argument(
        "--logdir",
        type=str,
        required=True,
        help="TensorBoard 日誌目錄路徑"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="輸出文件路徑（不指定則輸出到終端）"
    )

    parser.add_argument(
        "--format",
        type=str,
        choices=["json", "markdown", "both"],
        default="both",
        help="輸出格式: json, markdown, 或 both（預設）"
    )

    parser.add_argument(
        "--compare",
        action="store_true",
        help="比較模式：分析目錄下所有實驗並對比"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="顯示詳細信息"
    )

    parser.add_argument(
        "--sampling",
        type=str,
        choices=["segment", "inflection", "both"],
        default="both",
        help="採樣模式: segment（分段採樣）, inflection（轉折點採樣）, both（兩者都包含，預設）"
    )

    parser.add_argument(
        "--segments",
        type=int,
        default=20,
        help="分段採樣的段數（預設 20）"
    )

    parser.add_argument(
        "--inflection-sensitivity",
        type=float,
        default=0.1,
        help="轉折點敏感度 (0.01-1.0)，越小越敏感，預設 0.1"
    )

    return parser.parse_args()


class TensorBoardAnalyzer:
    """TensorBoard 日誌分析器"""

    def __init__(self, logdir: str, verbose: bool = False,
                 sampling_mode: str = "both", num_segments: int = 20,
                 inflection_sensitivity: float = 0.1):
        """
        初始化分析器

        參數:
            logdir: TensorBoard 日誌目錄
            verbose: 是否顯示詳細信息
            sampling_mode: 採樣模式 ("segment", "inflection", "both")
            num_segments: 分段採樣的段數
            inflection_sensitivity: 轉折點敏感度
        """
        self.logdir = Path(logdir)
        self.verbose = verbose
        self.sampling_mode = sampling_mode
        self.num_segments = num_segments
        self.inflection_sensitivity = inflection_sensitivity
        self.data = {}
        self.summary = {}

    def load_data(self) -> bool:
        """載入 TensorBoard 數據"""
        try:
            from tensorboard.backend.event_processing import event_accumulator

            if self.verbose:
                print(f"[LOAD] 載入日誌: {self.logdir}")

            # 創建事件累加器
            ea = event_accumulator.EventAccumulator(
                str(self.logdir),
                size_guidance={
                    event_accumulator.SCALARS: 0,  # 載入所有標量數據
                }
            )
            ea.Reload()

            # 獲取所有標量標籤
            tags = ea.Tags()['scalars']

            if self.verbose:
                print(f"[OK] 找到 {len(tags)} 個指標")

            # 提取每個指標的數據
            for tag in tags:
                events = ea.Scalars(tag)
                self.data[tag] = {
                    'steps': [e.step for e in events],
                    'values': [e.value for e in events],
                    'wall_times': [e.wall_time for e in events]
                }

            return True

        except ImportError:
            print("[ERROR] 需要安裝 tensorboard")
            print("   請執行: pip install tensorboard")
            return False

        except Exception as e:
            print(f"[ERROR] 載入數據失敗: {e}")
            return False

    def analyze(self) -> Dict[str, Any]:
        """分析數據並生成摘要"""
        if not self.data:
            print("[WARNING] 沒有數據可分析")
            return {}

        self.summary = {
            "metadata": self._analyze_metadata(),
            "training_progress": self._analyze_training_progress(),
            "performance_metrics": self._analyze_performance(),
            "stability_metrics": self._analyze_stability(),
            "diagnostic": self._analyze_diagnostic(),
            "recommendations": self._generate_recommendations()
        }

        return self.summary

    def _analyze_metadata(self) -> Dict[str, Any]:
        """分析元數據"""
        # 找到任意一個指標來獲取基本信息
        first_tag = next(iter(self.data.keys()))
        steps = self.data[first_tag]['steps']
        wall_times = self.data[first_tag]['wall_times']

        total_steps = steps[-1] if steps else 0
        start_time = wall_times[0] if wall_times else 0
        end_time = wall_times[-1] if wall_times else 0
        duration = end_time - start_time

        return {
            "total_steps": int(total_steps),
            "num_metrics": len(self.data),
            "duration_seconds": float(duration),
            "duration_hours": float(duration / 3600),
            "start_time": datetime.fromtimestamp(start_time).isoformat() if start_time else None,
            "end_time": datetime.fromtimestamp(end_time).isoformat() if end_time else None,
            "steps_per_second": float(total_steps / duration) if duration > 0 else 0
        }

    def _analyze_training_progress(self) -> Dict[str, Any]:
        """分析訓練進度"""
        progress = {}

        # 分析平均 episode 獎勵
        if 'rollout/ep_rew_mean' in self.data:
            values = self.data['rollout/ep_rew_mean']['values']
            steps = self.data['rollout/ep_rew_mean']['steps']

            progress['episode_reward'] = {
                'initial': float(values[0]) if values else None,
                'final': float(values[-1]) if values else None,
                'max': float(max(values)) if values else None,
                'min': float(min(values)) if values else None,
                'mean': float(np.mean(values)) if values else None,
                'std': float(np.std(values)) if values else None,
                'improvement': float(values[-1] - values[0]) if len(values) > 1 else None,
                'trend': self._calculate_trend(steps, values),
                'timeseries': self._sample_timeseries(steps, values)  # ⭐ 新增：採樣數據
            }

        # 分析平均 episode 長度
        if 'rollout/ep_len_mean' in self.data:
            values = self.data['rollout/ep_len_mean']['values']

            progress['episode_length'] = {
                'mean': float(np.mean(values)) if values else None,
                'final': float(values[-1]) if values else None
            }

        return progress

    def _analyze_performance(self) -> Dict[str, Any]:
        """分析性能指標"""
        performance = {}

        # 分析損失
        if 'train/loss' in self.data:
            values = self.data['train/loss']['values']
            steps = self.data['train/loss']['steps']

            performance['total_loss'] = {
                'initial': float(values[0]) if values else None,
                'final': float(values[-1]) if values else None,
                'mean': float(np.mean(values)) if values else None,
                'trend': self._calculate_trend(steps, values),
                'converged': self._check_convergence(values),
                'timeseries': self._sample_timeseries(steps, values)  # ⭐ 新增
            }

        # 分析策略損失
        if 'train/policy_loss' in self.data:
            values = self.data['train/policy_loss']['values']

            performance['policy_loss'] = {
                'final': float(values[-1]) if values else None,
                'mean': float(np.mean(values)) if values else None,
                'std': float(np.std(values)) if values else None
            }

        # 分析價值損失
        if 'train/value_loss' in self.data:
            values = self.data['train/value_loss']['values']

            performance['value_loss'] = {
                'final': float(values[-1]) if values else None,
                'mean': float(np.mean(values)) if values else None,
                'std': float(np.std(values)) if values else None
            }

        # 分析解釋方差
        if 'train/explained_variance' in self.data:
            values = self.data['train/explained_variance']['values']

            performance['explained_variance'] = {
                'final': float(values[-1]) if values else None,
                'mean': float(np.mean(values)) if values else None,
                'is_good': float(np.mean(values)) > 0.7 if values else None
            }

        return performance

    def _analyze_stability(self) -> Dict[str, Any]:
        """分析訓練穩定性"""
        stability = {}

        # 分析 KL 散度
        if 'train/approx_kl' in self.data:
            values = self.data['train/approx_kl']['values']

            stability['kl_divergence'] = {
                'mean': float(np.mean(values)) if values else None,
                'max': float(max(values)) if values else None,
                'is_stable': float(np.mean(values)) < 0.02 if values else None
            }

        # 分析 Clip 比例
        if 'train/clip_fraction' in self.data:
            values = self.data['train/clip_fraction']['values']

            stability['clip_fraction'] = {
                'mean': float(np.mean(values)) if values else None,
                'is_good': 0.05 < float(np.mean(values)) < 0.3 if values else None
            }

        # 分析熵損失
        if 'train/entropy_loss' in self.data:
            values = self.data['train/entropy_loss']['values']

            stability['entropy'] = {
                'final': float(values[-1]) if values else None,
                'mean': float(np.mean(values)) if values else None,
                'trend': self._calculate_trend(
                    self.data['train/entropy_loss']['steps'],
                    values
                )
            }

        # 分析獎勵穩定性
        if 'rollout/ep_rew_mean' in self.data:
            values = self.data['rollout/ep_rew_mean']['values']

            # 計算最近 20% 數據的標準差
            recent_values = values[-len(values)//5:] if len(values) > 10 else values
            stability['reward_stability'] = {
                'recent_std': float(np.std(recent_values)) if recent_values else None,
                'recent_mean': float(np.mean(recent_values)) if recent_values else None,
                'coefficient_of_variation': float(
                    np.std(recent_values) / np.mean(recent_values)
                ) if recent_values and np.mean(recent_values) != 0 else None
            }

        return stability

    def _analyze_diagnostic(self) -> Dict[str, Any]:
        """診斷潛在問題"""
        issues = []
        warnings = []
        suggestions = []

        # 檢查獎勵是否上升
        if 'rollout/ep_rew_mean' in self.data:
            values = self.data['rollout/ep_rew_mean']['values']
            trend = self._calculate_trend(
                self.data['rollout/ep_rew_mean']['steps'],
                values
            )

            if trend and trend['slope'] < 0:
                issues.append({
                    "type": "獎勵下降",
                    "severity": "high",
                    "message": "平均獎勵呈下降趨勢，訓練可能出現問題"
                })
            elif trend and abs(trend['slope']) < 0.001:
                warnings.append({
                    "type": "獎勵不變",
                    "severity": "medium",
                    "message": "平均獎勵幾乎不變，可能陷入局部最優"
                })

        # 檢查 KL 散度
        if 'train/approx_kl' in self.data:
            values = self.data['train/approx_kl']['values']
            mean_kl = np.mean(values) if values else 0

            if mean_kl > 0.02:
                issues.append({
                    "type": "KL散度過高",
                    "severity": "high",
                    "message": f"KL散度平均值 {mean_kl:.4f} > 0.02，訓練可能不穩定"
                })
                suggestions.append("降低學習率或減小 clip_range")

        # 檢查解釋方差
        if 'train/explained_variance' in self.data:
            values = self.data['train/explained_variance']['values']
            mean_ev = np.mean(values) if values else 0

            if mean_ev < 0.5:
                warnings.append({
                    "type": "解釋方差低",
                    "severity": "medium",
                    "message": f"解釋方差 {mean_ev:.4f} < 0.5，價值函數擬合不佳"
                })
                suggestions.append("增加網絡容量或調整 vf_coef")

        # 檢查是否有 NaN
        has_nan = False
        for tag, data in self.data.items():
            if any(np.isnan(v) for v in data['values']):
                has_nan = True
                break

        if has_nan:
            issues.append({
                "type": "NaN數值",
                "severity": "critical",
                "message": "訓練中出現 NaN，訓練已失敗"
            })
            suggestions.append("降低學習率、減小 max_grad_norm")

        return {
            "issues": issues,
            "warnings": warnings,
            "suggestions": suggestions,
            "health_score": self._calculate_health_score(issues, warnings)
        }

    def _sample_timeseries(self, steps: List[int], values: List[float]) -> Dict[str, Any]:
        """採樣時間序列數據

        返回分段採樣和/或轉折點採樣的結果
        """
        if not steps or not values:
            return {}

        result = {}

        # 分段採樣
        if self.sampling_mode in ["segment", "both"]:
            result['segment_samples'] = self._segment_sampling(steps, values)

        # 轉折點採樣
        if self.sampling_mode in ["inflection", "both"]:
            result['inflection_samples'] = self._inflection_sampling(steps, values)

        return result

    def _segment_sampling(self, steps: List[int], values: List[float]) -> List[Dict[str, float]]:
        """分段採樣：根據數據量自動調整段數

        自動調整規則：
        - < 10 點：返回所有點
        - 10-50 點：分成 5 段
        - 50-200 點：分成 10 段
        - 200-1000 點：分成 20 段
        - 1000-5000 點：分成 50 段
        - > 5000 點：分成 100 段
        """
        n = len(steps)

        # 根據數據量自動確定段數
        if n < 10:
            # 太少，返回所有點
            actual_segments = n
        elif n < 50:
            actual_segments = min(5, n)
        elif n < 200:
            actual_segments = min(10, n)
        elif n < 1000:
            actual_segments = min(20, n)
        elif n < 5000:
            actual_segments = min(50, n)
        else:
            actual_segments = min(100, n)

        # 如果用戶指定了段數，使用用戶指定的（但不超過數據點數）
        if self.num_segments > 0:
            actual_segments = min(self.num_segments, n)

        if self.verbose:
            print(f"  數據點數: {n}, 採樣段數: {actual_segments}")

        # 如果段數等於數據點數，返回所有點
        if actual_segments >= n:
            return [{
                'step': int(s),
                'value': float(v),
                'min': float(v),
                'max': float(v),
                'std': 0.0
            } for s, v in zip(steps, values)]

        segment_size = n // actual_segments
        samples = []

        for i in range(actual_segments):
            start_idx = i * segment_size
            end_idx = (i + 1) * segment_size if i < actual_segments - 1 else n

            # 取該段的中間 step 和平均 value
            segment_steps = steps[start_idx:end_idx]
            segment_values = values[start_idx:end_idx]

            mid_step = segment_steps[len(segment_steps) // 2]
            avg_value = np.mean(segment_values)

            samples.append({
                'step': int(mid_step),
                'value': float(avg_value),
                'min': float(np.min(segment_values)),
                'max': float(np.max(segment_values)),
                'std': float(np.std(segment_values))
            })

        return samples

    def _inflection_sampling(self, steps: List[int], values: List[float]) -> List[Dict[str, float]]:
        """轉折點採樣：檢測趨勢變化的關鍵點

        使用滑動窗口檢測斜率變化，當斜率變化超過閾值時記錄為轉折點
        """
        if len(values) < 10:
            return [{'step': int(s), 'value': float(v)} for s, v in zip(steps, values)]

        samples = []
        window_size = max(5, len(values) // 50)  # 窗口大小：最小5，最大為數據的2%

        # 始終包含第一個點
        samples.append({
            'step': int(steps[0]),
            'value': float(values[0]),
            'type': 'start'
        })

        # 計算滑動窗口的斜率
        slopes = []
        for i in range(len(values) - window_size):
            window_x = np.array(range(window_size))
            window_y = np.array(values[i:i+window_size])
            slope = np.polyfit(window_x, window_y, 1)[0]
            slopes.append(slope)

        # 檢測斜率變化（轉折點）
        if slopes:
            slope_std = np.std(slopes)
            threshold = slope_std * self.inflection_sensitivity

            prev_slope = slopes[0]
            for i in range(1, len(slopes)):
                curr_slope = slopes[i]
                slope_change = abs(curr_slope - prev_slope)

                # 如果斜率變化超過閾值，記錄為轉折點
                if slope_change > threshold:
                    idx = i + window_size // 2
                    if idx < len(steps):
                        # 判斷轉折類型
                        if prev_slope < 0 and curr_slope > 0:
                            inflection_type = 'valley'  # 谷底（向上轉）
                        elif prev_slope > 0 and curr_slope < 0:
                            inflection_type = 'peak'    # 峰頂（向下轉）
                        else:
                            inflection_type = 'change'  # 一般轉折

                        samples.append({
                            'step': int(steps[idx]),
                            'value': float(values[idx]),
                            'type': inflection_type,
                            'slope_before': float(prev_slope),
                            'slope_after': float(curr_slope)
                        })
                        prev_slope = curr_slope

        # 始終包含最後一個點
        samples.append({
            'step': int(steps[-1]),
            'value': float(values[-1]),
            'type': 'end'
        })

        return samples

    def _calculate_trend(self, steps: List[int], values: List[float]) -> Optional[Dict[str, float]]:
        """計算趨勢（線性回歸）"""
        if len(values) < 2:
            return None

        # 使用 numpy 進行線性回歸
        x = np.array(steps)
        y = np.array(values)

        # 計算斜率和截距
        coeffs = np.polyfit(x, y, 1)
        slope = float(coeffs[0])
        intercept = float(coeffs[1])

        # 計算 R²
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        return {
            'slope': slope,
            'intercept': intercept,
            'r_squared': float(r_squared),
            'direction': 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable'
        }

    def _check_convergence(self, values: List[float], window: int = 50) -> bool:
        """檢查是否收斂"""
        if len(values) < window * 2:
            return False

        # 比較最近兩個窗口的標準差
        recent = values[-window:]
        previous = values[-2*window:-window]

        std_recent = np.std(recent)
        std_previous = np.std(previous)

        # 如果最近的標準差明顯小於之前的，認為已收斂
        return std_recent < std_previous * 0.5

    def _calculate_health_score(self, issues: List[Dict], warnings: List[Dict]) -> float:
        """計算訓練健康度分數 (0-100)"""
        score = 100.0

        # 每個 critical issue 扣 50 分
        critical_count = sum(1 for i in issues if i.get('severity') == 'critical')
        score -= critical_count * 50

        # 每個 high issue 扣 20 分
        high_count = sum(1 for i in issues if i.get('severity') == 'high')
        score -= high_count * 20

        # 每個 medium warning 扣 10 分
        medium_count = sum(1 for w in warnings if w.get('severity') == 'medium')
        score -= medium_count * 10

        return max(0.0, score)

    def _generate_recommendations(self) -> List[str]:
        """生成優化建議"""
        recommendations = []

        # 基於獎勵趨勢
        if 'rollout/ep_rew_mean' in self.data:
            values = self.data['rollout/ep_rew_mean']['values']
            if values:
                final_reward = values[-1]
                improvement = values[-1] - values[0] if len(values) > 1 else 0

                if improvement < 0:
                    recommendations.append({
                        "category": "獎勵優化",
                        "priority": "high",
                        "suggestion": "獎勵下降，建議檢查獎勵函數設計或降低學習率"
                    })
                elif improvement < np.std(values) * 0.5:
                    recommendations.append({
                        "category": "獎勵優化",
                        "priority": "medium",
                        "suggestion": "獎勵提升緩慢，可嘗試調整 pnl_scale 或 cost_penalty"
                    })

        # 基於穩定性
        if 'train/approx_kl' in self.data:
            kl_values = self.data['train/approx_kl']['values']
            if kl_values and np.mean(kl_values) > 0.02:
                recommendations.append({
                    "category": "穩定性",
                    "priority": "high",
                    "suggestion": "KL散度過高，建議降低 learning_rate 或 clip_range"
                })

        # 基於解釋方差
        if 'train/explained_variance' in self.data:
            ev_values = self.data['train/explained_variance']['values']
            if ev_values and np.mean(ev_values) < 0.7:
                recommendations.append({
                    "category": "模型容量",
                    "priority": "medium",
                    "suggestion": "解釋方差偏低，建議增加 lstm_hidden_size 或調整網絡架構"
                })

        # 基於訓練步數
        metadata = self._analyze_metadata()
        if metadata['total_steps'] < 100000:
            recommendations.append({
                "category": "訓練時長",
                "priority": "low",
                "suggestion": f"當前訓練步數較少 ({metadata['total_steps']:,})，建議至少訓練 500K+ steps"
            })

        return recommendations

    def _convert_to_json_serializable(self, obj):
        """遞歸轉換 NumPy 類型為 Python 原生類型"""
        if isinstance(obj, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    def save_json(self, output_path: str):
        """保存 JSON 報告"""
        # 轉換所有 NumPy 類型為 Python 原生類型
        serializable_summary = self._convert_to_json_serializable(self.summary)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_summary, f, indent=2, ensure_ascii=False)

        print(f"[OK] JSON 報告已保存: {output_path}")

    def save_markdown(self, output_path: str):
        """保存 Markdown 報告"""
        md_content = self._generate_markdown()

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(md_content)

        print(f"[OK] Markdown 報告已保存: {output_path}")

    def _generate_markdown(self) -> str:
        """生成 Markdown 格式報告"""
        lines = []

        # 標題
        lines.append("# TensorBoard 訓練分析報告\n")
        lines.append(f"**生成時間**: {datetime.now().isoformat()}\n")
        lines.append(f"**日誌目錄**: `{self.logdir}`\n")
        lines.append("---\n")

        # 元數據
        if 'metadata' in self.summary:
            meta = self.summary['metadata']
            lines.append("## [STATS] 訓練概況\n")
            lines.append(f"- **總訓練步數**: {meta['total_steps']:,}")
            lines.append(f"- **訓練時長**: {meta['duration_hours']:.2f} 小時")
            lines.append(f"- **訓練速度**: {meta['steps_per_second']:.1f} steps/秒")
            lines.append(f"- **指標數量**: {meta['num_metrics']}")
            lines.append("")

        # 訓練進度
        if 'training_progress' in self.summary:
            progress = self.summary['training_progress']
            lines.append("## [PROGRESS] 訓練進度\n")

            if 'episode_reward' in progress:
                rew = progress['episode_reward']
                lines.append("### Episode 獎勵\n")
                lines.append(f"- **初始值**: {rew['initial']:.2f}")
                lines.append(f"- **最終值**: {rew['final']:.2f}")
                lines.append(f"- **最大值**: {rew['max']:.2f}")
                lines.append(f"- **平均值**: {rew['mean']:.2f} ± {rew['std']:.2f}")
                lines.append(f"- **總提升**: {rew['improvement']:.2f}")

                if rew['trend']:
                    trend = rew['trend']
                    lines.append(f"- **趨勢**: {trend['direction']} (R² = {trend['r_squared']:.3f})")

                lines.append("")

        # 性能指標
        if 'performance_metrics' in self.summary:
            perf = self.summary['performance_metrics']
            lines.append("## [METRICS] 性能指標\n")

            if 'total_loss' in perf:
                loss = perf['total_loss']
                lines.append("### 總損失\n")
                lines.append(f"- **初始值**: {loss['initial']:.4f}")
                lines.append(f"- **最終值**: {loss['final']:.4f}")
                lines.append(f"- **平均值**: {loss['mean']:.4f}")
                lines.append(f"- **趨勢**: {loss['trend']['direction'] if loss['trend'] else 'N/A'}")
                lines.append(f"- **已收斂**: {'[OK] 是' if loss['converged'] else '[NO] 否'}")
                lines.append("")

            if 'explained_variance' in perf:
                ev = perf['explained_variance']
                lines.append("### 解釋方差\n")
                lines.append(f"- **最終值**: {ev['final']:.4f}")
                lines.append(f"- **平均值**: {ev['mean']:.4f}")
                lines.append(f"- **評估**: {'[OK] 良好 (>0.7)' if ev['is_good'] else '[WARN] 偏低 (<0.7)'}")
                lines.append("")

        # 穩定性指標
        if 'stability_metrics' in self.summary:
            stab = self.summary['stability_metrics']
            lines.append("## 🔒 穩定性分析\n")

            if 'kl_divergence' in stab:
                kl = stab['kl_divergence']
                lines.append("### KL 散度\n")
                lines.append(f"- **平均值**: {kl['mean']:.6f}")
                lines.append(f"- **最大值**: {kl['max']:.6f}")
                lines.append(f"- **穩定性**: {'[OK] 穩定 (<0.02)' if kl['is_stable'] else '[WARN] 不穩定 (>0.02)'}")
                lines.append("")

            if 'reward_stability' in stab:
                rs = stab['reward_stability']
                lines.append("### 獎勵穩定性\n")
                lines.append(f"- **最近平均**: {rs['recent_mean']:.2f}")
                lines.append(f"- **最近標準差**: {rs['recent_std']:.2f}")
                if rs['coefficient_of_variation']:
                    lines.append(f"- **變異係數**: {rs['coefficient_of_variation']:.3f}")
                lines.append("")

        # 診斷結果
        if 'diagnostic' in self.summary:
            diag = self.summary['diagnostic']
            lines.append("## 🏥 診斷報告\n")

            lines.append(f"### 健康度評分: {diag['health_score']:.0f}/100\n")

            if diag['issues']:
                lines.append("### [ISSUES] 問題\n")
                for issue in diag['issues']:
                    lines.append(f"- **[{issue['severity'].upper()}]** {issue['type']}: {issue['message']}")
                lines.append("")

            if diag['warnings']:
                lines.append("### [WARNINGS] 警告\n")
                for warning in diag['warnings']:
                    lines.append(f"- **[{warning['severity'].upper()}]** {warning['type']}: {warning['message']}")
                lines.append("")

            if diag['suggestions']:
                lines.append("### 💡 建議\n")
                for suggestion in diag['suggestions']:
                    lines.append(f"- {suggestion}")
                lines.append("")

        # 優化建議
        if 'recommendations' in self.summary:
            recs = self.summary['recommendations']
            lines.append("## 🚀 優化建議\n")

            # 按優先級排序
            high_priority = [r for r in recs if r.get('priority') == 'high']
            medium_priority = [r for r in recs if r.get('priority') == 'medium']
            low_priority = [r for r in recs if r.get('priority') == 'low']

            if high_priority:
                lines.append("### 🔴 高優先級\n")
                for rec in high_priority:
                    lines.append(f"- **{rec['category']}**: {rec['suggestion']}")
                lines.append("")

            if medium_priority:
                lines.append("### 🟡 中優先級\n")
                for rec in medium_priority:
                    lines.append(f"- **{rec['category']}**: {rec['suggestion']}")
                lines.append("")

            if low_priority:
                lines.append("### 🟢 低優先級\n")
                for rec in low_priority:
                    lines.append(f"- **{rec['category']}**: {rec['suggestion']}")
                lines.append("")

        # 結論
        lines.append("## [CONCLUSION] 結論\n")

        if 'diagnostic' in self.summary:
            health = self.summary['diagnostic']['health_score']
            if health >= 80:
                lines.append("[OK] **訓練狀況良好**，可以繼續當前配置。")
            elif health >= 60:
                lines.append("[WARN] **訓練狀況尚可**，建議根據上述建議進行優化。")
            else:
                lines.append("[ERROR] **訓練存在問題**，請優先處理高優先級問題。")

        lines.append("\n---\n")
        lines.append(f"*此報告由 analyze_tensorboard.py 自動生成*")

        return "\n".join(lines)

    def print_summary(self):
        """打印摘要到終端"""
        print("\n" + "=" * 70)
        print("TensorBoard 訓練分析報告")
        print("=" * 70)

        # 打印 Markdown 內容
        print(self._generate_markdown())


def compare_experiments(base_dir: str, verbose: bool = False) -> Dict[str, Any]:
    """比較多個實驗"""
    base_path = Path(base_dir)

    # 找到所有 TensorBoard 日誌目錄（包含 events.out.tfevents 文件）
    exp_dirs = []
    for item in base_path.rglob("events.out.tfevents*"):
        exp_dir = item.parent
        if exp_dir not in exp_dirs:
            exp_dirs.append(exp_dir)

    if not exp_dirs:
        print(f"[WARNING] 在 {base_dir} 中未找到 TensorBoard 日誌")
        return {}

    print(f"[INFO] 找到 {len(exp_dirs)} 個實驗")

    # 分析每個實驗
    results = {}
    for exp_dir in exp_dirs:
        exp_name = exp_dir.name
        print(f"\n分析實驗: {exp_name}")

        analyzer = TensorBoardAnalyzer(exp_dir, verbose=verbose)
        if analyzer.load_data():
            analyzer.analyze()
            results[exp_name] = analyzer.summary

    # 生成對比報告
    comparison = {
        "num_experiments": len(results),
        "experiments": results,
        "ranking": _rank_experiments(results)
    }

    return comparison


def _rank_experiments(results: Dict[str, Dict]) -> List[Dict[str, Any]]:
    """對實驗進行排名"""
    rankings = []

    for exp_name, summary in results.items():
        score = 0.0

        # 基於最終獎勵
        if 'training_progress' in summary and 'episode_reward' in summary['training_progress']:
            final_reward = summary['training_progress']['episode_reward'].get('final', 0)
            score += final_reward

        # 基於健康度
        if 'diagnostic' in summary:
            health = summary['diagnostic'].get('health_score', 0)
            score += health / 10  # 歸一化

        rankings.append({
            "name": exp_name,
            "score": score,
            "final_reward": summary['training_progress']['episode_reward'].get('final')
            if 'training_progress' in summary and 'episode_reward' in summary['training_progress']
            else None,
            "health_score": summary['diagnostic'].get('health_score')
            if 'diagnostic' in summary else None
        })

    # 按分數排序
    rankings.sort(key=lambda x: x['score'], reverse=True)

    return rankings


def main():
    """主函數"""
    args = parse_args()

    if args.compare:
        # 比較模式
        print("[COMPARE] 比較模式: 分析多個實驗")
        comparison = compare_experiments(args.logdir, verbose=args.verbose)

        if comparison:
            # 保存對比結果
            if args.output:
                output_path = args.output
            else:
                output_path = "results/experiment_comparison.json"

            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(comparison, f, indent=2, ensure_ascii=False)

            print(f"\n[OK] 對比報告已保存: {output_path}")

            # 打印排名
            print("\n" + "=" * 70)
            print("實驗排名")
            print("=" * 70)
            for i, rank in enumerate(comparison['ranking'], 1):
                print(f"{i}. {rank['name']}")
                print(f"   分數: {rank['score']:.2f}")
                if rank['final_reward'] is not None:
                    print(f"   最終獎勵: {rank['final_reward']:.2f}")
                if rank['health_score'] is not None:
                    print(f"   健康度: {rank['health_score']:.0f}/100")
                print()

    else:
        # 單個實驗分析
        analyzer = TensorBoardAnalyzer(
            args.logdir,
            verbose=args.verbose,
            sampling_mode=args.sampling,
            num_segments=args.segments,
            inflection_sensitivity=args.inflection_sensitivity
        )

        # 載入數據
        if not analyzer.load_data():
            return 1

        # 分析數據
        print("\n[ANALYZE] 分析數據...")
        analyzer.analyze()

        # 輸出結果
        if args.format in ['json', 'both']:
            if args.output:
                json_path = args.output if args.output.endswith('.json') else f"{args.output}.json"
            else:
                json_path = "results/tensorboard_analysis.json"

            os.makedirs(os.path.dirname(json_path), exist_ok=True)
            analyzer.save_json(json_path)

        if args.format in ['markdown', 'both']:
            if args.output:
                md_path = args.output.replace('.json', '.md') if args.output.endswith('.json') else f"{args.output}.md"
            else:
                md_path = "results/tensorboard_analysis.md"

            os.makedirs(os.path.dirname(md_path), exist_ok=True)
            analyzer.save_markdown(md_path)

        # 打印摘要
        if not args.output:
            analyzer.print_summary()

    return 0


if __name__ == "__main__":
    exit(main())
