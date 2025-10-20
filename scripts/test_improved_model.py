#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_improved_model.py - 測試改進版 DeepLOB 模型
=============================================================================
快速驗證改進版模型是否正常工作

使用方式:
    conda activate deeplob-pro
    python scripts/test_improved_model.py
=============================================================================
"""

import sys
from pathlib import Path
import torch

# 添加專案根目錄
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.deeplob_improved import DeepLOBImproved

def test_model_creation():
    """測試模型創建"""
    print("="*60)
    print("測試 1: 模型創建")
    print("="*60)

    try:
        model = DeepLOBImproved(
            num_classes=3,
            conv_filters=32,
            lstm_hidden=64,
            fc_hidden=64,
            dropout=0.6,
            use_layer_norm=True,
            use_attention=True,
            input_shape=(100, 20)
        )
        print("✅ 模型創建成功")

        # 統計參數
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"\n模型統計:")
        print(f"  總參數量: {total_params:,}")
        print(f"  可訓練參數: {trainable_params:,}")

        return model

    except Exception as e:
        print(f"❌ 模型創建失敗: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_forward_pass(model):
    """測試前向傳播"""
    print("\n" + "="*60)
    print("測試 2: 前向傳播")
    print("="*60)

    try:
        batch_size = 16
        seq_len = 100
        features = 20

        # 創建隨機輸入
        x = torch.randn(batch_size, seq_len, features)
        print(f"輸入形狀: {x.shape}")

        # 前向傳播
        logits = model(x)
        print(f"輸出形狀: {logits.shape}")

        # 檢查輸出
        assert logits.shape == (batch_size, 3), f"輸出形狀錯誤: {logits.shape}"
        assert not torch.isnan(logits).any(), "輸出包含 NaN"
        assert not torch.isinf(logits).any(), "輸出包含 Inf"

        print(f"✅ 前向傳播成功")
        print(f"  輸出範圍: [{logits.min():.4f}, {logits.max():.4f}]")

        return True

    except Exception as e:
        print(f"❌ 前向傳播失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_attention_weights(model):
    """測試注意力權重"""
    print("\n" + "="*60)
    print("測試 3: 注意力機制")
    print("="*60)

    if not model.use_attention:
        print("⚠️  模型未啟用注意力機制，跳過測試")
        return True

    try:
        x = torch.randn(8, 100, 20)

        # 獲取注意力權重
        logits, attn_weights = model(x, return_attention=True)

        print(f"注意力權重形狀: {attn_weights.shape}")
        print(f"注意力權重範圍: [{attn_weights.min():.4f}, {attn_weights.max():.4f}]")

        # 檢查權重和為 1
        attn_sum = attn_weights.sum(dim=1)
        print(f"權重和: {attn_sum[0].item():.6f} (應該 ≈ 1.0)")

        assert attn_weights.shape == (8, 100, 1), f"注意力權重形狀錯誤: {attn_weights.shape}"
        assert torch.allclose(attn_sum, torch.ones_like(attn_sum), atol=1e-5), "注意力權重和不為 1"

        print("✅ 注意力機制正常")

        # 顯示權重分佈
        print(f"\n權重統計:")
        print(f"  Mean: {attn_weights.mean():.6f}")
        print(f"  Std:  {attn_weights.std():.6f}")
        print(f"  Max:  {attn_weights.max():.6f}")
        print(f"  Min:  {attn_weights.min():.6f}")

        return True

    except Exception as e:
        print(f"❌ 注意力測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_backward_pass(model):
    """測試反向傳播"""
    print("\n" + "="*60)
    print("測試 4: 反向傳播")
    print("="*60)

    try:
        x = torch.randn(16, 100, 20, requires_grad=True)
        labels = torch.randint(0, 3, (16,))

        # 前向
        logits = model(x)

        # 計算損失
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(logits, labels)

        print(f"損失值: {loss.item():.4f}")

        # 反向
        loss.backward()

        # 檢查梯度
        has_grad = False
        grad_norms = []

        for name, param in model.named_parameters():
            if param.grad is not None:
                has_grad = True
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)

                # 檢查異常梯度
                if torch.isnan(param.grad).any():
                    print(f"  ⚠️  {name} 梯度包含 NaN")
                if torch.isinf(param.grad).any():
                    print(f"  ⚠️  {name} 梯度包含 Inf")

        assert has_grad, "沒有計算任何梯度"

        avg_grad_norm = sum(grad_norms) / len(grad_norms)
        max_grad_norm = max(grad_norms)

        print(f"✅ 反向傳播成功")
        print(f"  平均梯度範數: {avg_grad_norm:.4f}")
        print(f"  最大梯度範數: {max_grad_norm:.4f}")

        if max_grad_norm > 10.0:
            print(f"  ⚠️  警告: 梯度範數較大 ({max_grad_norm:.2f})")

        return True

    except Exception as e:
        print(f"❌ 反向傳播失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gpu_support():
    """測試 GPU 支援"""
    print("\n" + "="*60)
    print("測試 5: GPU 支援")
    print("="*60)

    if not torch.cuda.is_available():
        print("⚠️  CUDA 不可用，跳過 GPU 測試")
        return True

    try:
        device = torch.device("cuda")

        model = DeepLOBImproved(
            num_classes=3,
            lstm_hidden=64,
            dropout=0.6,
            use_layer_norm=True,
            use_attention=True
        ).to(device)

        x = torch.randn(16, 100, 20).to(device)

        # 前向傳播
        logits = model(x)

        assert logits.device.type == "cuda", "輸出未在 GPU 上"

        print(f"✅ GPU 支援正常")
        print(f"  設備: {torch.cuda.get_device_name(0)}")
        print(f"  記憶體使用: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")

        return True

    except Exception as e:
        print(f"❌ GPU 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函數"""
    print("\n" + "="*60)
    print("🧪 改進版 DeepLOB 模型測試")
    print("="*60)

    results = {}

    # 測試 1: 模型創建
    model = test_model_creation()
    results['creation'] = model is not None

    if model is None:
        print("\n❌ 模型創建失敗，後續測試跳過")
        return 1

    # 測試 2: 前向傳播
    results['forward'] = test_forward_pass(model)

    # 測試 3: 注意力機制
    results['attention'] = test_attention_weights(model)

    # 測試 4: 反向傳播
    results['backward'] = test_backward_pass(model)

    # 測試 5: GPU 支援
    results['gpu'] = test_gpu_support()

    # 總結
    print("\n" + "="*60)
    print("測試總結")
    print("="*60)

    for test_name, passed in results.items():
        status = "✅ 通過" if passed else "❌ 失敗"
        print(f"  {test_name.capitalize():12s}: {status}")

    all_passed = all(results.values())

    if all_passed:
        print("\n🎉 所有測試通過！改進版模型可以使用。")
        print("\n下一步:")
        print("  conda activate deeplob-pro")
        print("  python scripts/train_deeplob_v5.py \\")
        print("      --config configs/train_v5_improved.yaml \\")
        print("      --epochs 10")
        return 0
    else:
        print("\n❌ 部分測試失敗，請檢查錯誤訊息")
        return 1


if __name__ == "__main__":
    sys.exit(main())
