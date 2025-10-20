#!/usr/bin/env python
"""
快速驗證修復配置的有效性
- 訓練 5 個 epochs
- 觀察 Class 1 Recall 是否提升
- 比較三個版本（保守/中等/激進）
"""
import os
import sys
import subprocess
from pathlib import Path

# 添加 src 到 Python 路徑
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utils.yaml_manager import YAMLManager

def run_quick_test(config_path: str, epochs: int = 5):
    """運行快速測試（指定 epochs）"""
    print(f"\n{'='*70}")
    print(f"測試配置: {config_path}")
    print(f"快速測試: {epochs} epochs")
    print(f"{'='*70}\n")

    # 使用 YAMLManager 讀取配置（保留註釋和格式）
    yaml_manager = YAMLManager(config_path)

    # 臨時覆蓋 epochs
    original_epochs = yaml_manager.train.epochs
    yaml_manager.train.epochs = epochs

    # 保存臨時配置
    temp_config_path = config_path.replace('.yaml', '_temp.yaml')
    yaml_manager.save_as(temp_config_path)

    # 運行訓練
    cmd = [
        sys.executable,
        'scripts/train_deeplob_v5.py',
        '--config', temp_config_path
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"\n✅ {config_path} 測試完成！")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {config_path} 測試失敗: {e}")
    finally:
        # 清理臨時配置
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)

def main():
    """主函數"""
    print("=" * 70)
    print("DeepLOB V5 修復配置快速驗證")
    print("=" * 70)
    print("\n目標：驗證三個修復配置是否能提升 Class 1 Recall")
    print("策略：每個配置訓練 5 epochs，觀察初步效果\n")

    configs = [
        'configs/train_v5_fix_conservative.yaml',
        'configs/train_v5_fix_moderate.yaml',
        'configs/train_v5_fix_aggressive.yaml',
    ]

    for config_path in configs:
        if not os.path.exists(config_path):
            print(f"⚠️ 配置文件不存在: {config_path}")
            continue

        run_quick_test(config_path, epochs=5)

    print("\n" + "=" * 70)
    print("所有測試完成！")
    print("=" * 70)
    print("\n下一步：")
    print("1. 檢查 logs/ 目錄下的日誌文件")
    print("2. 觀察 Class 1 Recall 是否有提升")
    print("3. 選擇效果最好的配置進行完整訓練（40-50 epochs）")
    print("\n完整訓練命令範例：")
    print("  python scripts/train_deeplob_v5.py \\")
    print("      --config configs/train_v5_fix_moderate.yaml")

if __name__ == '__main__':
    main()
