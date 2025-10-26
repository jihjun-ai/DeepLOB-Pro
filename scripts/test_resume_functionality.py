"""測試 train_sb3_deeplob.py 續訓功能

此腳本用於快速測試續訓功能是否正常工作

使用方法:
    python scripts/test_resume_functionality.py

作者: SB3-DeepLOB 專案團隊
日期: 2025-10-26
"""

import os
import sys
import subprocess
import time
from pathlib import Path

# 添加專案根目錄到路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def run_command(cmd: str, description: str):
    """執行命令並記錄結果"""
    print("\n" + "=" * 60)
    print(f"🚀 {description}")
    print("=" * 60)
    print(f"命令: {cmd}")
    print()

    start_time = time.time()
    result = subprocess.run(cmd, shell=True)
    duration = time.time() - start_time

    if result.returncode == 0:
        print(f"\n✅ {description} 成功 ({duration:.1f}秒)")
        return True
    else:
        print(f"\n❌ {description} 失敗 ({duration:.1f}秒)")
        return False


def main():
    print("=" * 60)
    print("🧪 train_sb3_deeplob.py 續訓功能測試")
    print("=" * 60)

    # 檢查是否有 DeepLOB 檢查點
    deeplob_checkpoint = "checkpoints/v5/deeplob_v5_best.pth"
    if not os.path.exists(deeplob_checkpoint):
        print(f"\n❌ DeepLOB 檢查點不存在: {deeplob_checkpoint}")
        print("請先訓練 DeepLOB 模型或修改檢查點路徑")
        return 1

    print(f"\n✅ DeepLOB 檢查點已找到: {deeplob_checkpoint}")

    # 測試步驟
    steps = [
        {
            "name": "第一階段訓練（5K steps）",
            "cmd": (
                "python scripts/train_sb3_deeplob.py "
                "--timesteps 5000 "
                "--config configs/sb3_deeplob_config.yaml"
            ),
            "checkpoint": "checkpoints/sb3/ppo_deeplob/ppo_deeplob_final.zip"
        },
        {
            "name": "續訓測試（再訓練 3K steps，步數累加）",
            "cmd": (
                "python scripts/train_sb3_deeplob.py "
                "--resume checkpoints/sb3/ppo_deeplob/ppo_deeplob_final.zip "
                "--timesteps 3000 "
                "--config configs/sb3_deeplob_config.yaml"
            ),
            "checkpoint": None  # 不檢查
        },
        {
            "name": "續訓測試（重置步數，訓練 2K steps）",
            "cmd": (
                "python scripts/train_sb3_deeplob.py "
                "--resume checkpoints/sb3/ppo_deeplob/ppo_deeplob_final.zip "
                "--reset-timesteps "
                "--timesteps 2000 "
                "--config configs/sb3_deeplob_config.yaml"
            ),
            "checkpoint": None
        }
    ]

    # 執行測試
    for i, step in enumerate(steps, 1):
        print(f"\n{'#' * 60}")
        print(f"測試步驟 {i}/{len(steps)}: {step['name']}")
        print(f"{'#' * 60}")

        success = run_command(step['cmd'], step['name'])

        if not success:
            print(f"\n❌ 測試失敗於步驟 {i}: {step['name']}")
            return 1

        # 檢查檢查點是否存在
        if step.get('checkpoint'):
            if os.path.exists(step['checkpoint']):
                print(f"✅ 檢查點已生成: {step['checkpoint']}")
            else:
                print(f"❌ 檢查點未找到: {step['checkpoint']}")
                return 1

        # 短暫等待
        time.sleep(2)

    # 測試完成
    print("\n" + "=" * 60)
    print("🎉 所有測試通過！")
    print("=" * 60)
    print("\n續訓功能驗證完成，包括:")
    print("  ✅ 從頭訓練")
    print("  ✅ 續訓（步數累加）")
    print("  ✅ 續訓（重置步數）")
    print("\n建議下一步:")
    print("  1. 檢查 TensorBoard 日誌: tensorboard --logdir logs/sb3_deeplob/")
    print("  2. 評估模型: python scripts/evaluate_sb3.py --model checkpoints/sb3/ppo_deeplob/best_model")
    print("  3. 開始完整訓練: python scripts/train_sb3_deeplob.py --timesteps 1000000")

    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  測試被用戶中斷")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ 測試過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
