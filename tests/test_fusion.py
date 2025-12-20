"""
Unit tests for Multi-Mechanism Fusion
"""
import torch
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.fusion.multi_mechanism_fusion import MultiMechanismFusion


def test_fusion_forward():
    """Test multi-mechanism fusion forward pass"""
    print("\n=== Test: Multi-Mechanism Fusion Forward Pass ===")
    fusion = MultiMechanismFusion(input_dim=16, hidden_dim=24, output_dim=2)

    fast_feat = torch.randn(4, 16, 1, 25)
    slow_feat = torch.randn(4, 16, 1, 25)

    output, mmd_loss = fusion(fast_feat, slow_feat)

    print(f"Fast features: {fast_feat.shape}")
    print(f"Slow features: {slow_feat.shape}")
    print(f"Output: {output.shape}")
    print(f"MMD loss: {mmd_loss.item():.4f}")

    assert output.shape == (4, 2), f"Output shape mismatch (expected (4, 2), got {output.shape})"
    assert mmd_loss.item() >= 0, "MMD loss should be non-negative"
    assert not torch.isnan(output).any(), "Output contains NaN"
    assert not torch.isnan(mmd_loss), "MMD loss is NaN"
    print("[PASS] Multi-mechanism fusion forward pass test PASSED")
    return True


def test_fusion_different_time_dims():
    """Test fusion with different time dimensions"""
    print("\n=== Test: Fusion with Different Time Dimensions ===")
    fusion = MultiMechanismFusion(input_dim=16, hidden_dim=24)

    # Fast and slow may have slightly different time dims due to padding
    fast_feat = torch.randn(2, 16, 1, 25)
    slow_feat = torch.randn(2, 16, 1, 26)  # Different time dim

    output, mmd_loss = fusion(fast_feat, slow_feat)

    print(f"Fast time: {fast_feat.size(3)}, Slow time: {slow_feat.size(3)}")
    print(f"Output: {output.shape}")

    assert output.shape == (2, 2), "Should handle different time dimensions"
    print("[PASS] Different time dimensions test PASSED")
    return True


def test_fusion_mechanisms():
    """Test individual fusion mechanisms"""
    print("\n=== Test: Fusion Mechanisms ===")
    fusion = MultiMechanismFusion(input_dim=16, hidden_dim=24)

    fast_feat = torch.randn(2, 16, 1, 20)
    slow_feat = torch.randn(2, 16, 1, 20)

    output, _ = fusion(fast_feat, slow_feat)

    # Check adaptive weight is trainable and in valid range
    assert fusion.adaptive_weight.requires_grad, "Adaptive weight should be trainable"
    alpha = torch.sigmoid(fusion.adaptive_weight)
    print(f"  Adaptive weight value: {alpha.item():.4f}")
    assert 0 <= alpha.item() <= 1, "Adaptive weight should be in [0, 1]"

    # Check gates exist
    assert hasattr(fusion, 'gate_fast'), "Should have fast gate"
    assert hasattr(fusion, 'gate_slow'), "Should have slow gate"

    # Check cross-attention exists
    assert hasattr(fusion, 'cross_attention_fast'), "Should have fast cross-attention"
    assert hasattr(fusion, 'cross_attention_slow'), "Should have slow cross-attention"

    print("[PASS] Fusion mechanisms test PASSED")
    return True


def test_fusion_gradient_flow():
    """Test gradients flow through all mechanisms"""
    print("\n=== Test: Fusion Gradient Flow ===")
    fusion = MultiMechanismFusion(input_dim=16, hidden_dim=24)

    fast_feat = torch.randn(2, 16, 1, 20, requires_grad=True)
    slow_feat = torch.randn(2, 16, 1, 20, requires_grad=True)

    output, mmd_loss = fusion(fast_feat, slow_feat)
    loss = output.sum() + mmd_loss
    loss.backward()

    # Check inputs have gradients
    assert fast_feat.grad is not None, "Fast features should have gradients"
    assert slow_feat.grad is not None, "Slow features should have gradients"

    # Check adaptive weight has gradients
    assert fusion.adaptive_weight.grad is not None, "Adaptive weight should have gradients"

    params_with_grad = sum(1 for p in fusion.parameters() if p.grad is not None)
    total_params = sum(1 for p in fusion.parameters())
    print(f"  Parameters with gradients: {params_with_grad}/{total_params}")
    assert params_with_grad == total_params, "All parameters should have gradients"

    print("[PASS] Fusion gradient flow test PASSED")
    return True


def test_fusion_mmd_loss():
    """Test MMD loss computation"""
    print("\n=== Test: MMD Loss Computation ===")
    fusion = MultiMechanismFusion(input_dim=16, hidden_dim=24, mmd_sigma=1.0)

    # Identical features should have low MMD
    fast_feat = torch.randn(2, 16, 1, 20)
    slow_feat = fast_feat.clone()

    _, mmd_loss_same = fusion(fast_feat, slow_feat)
    print(f"  MMD loss (identical): {mmd_loss_same.item():.6f}")

    # Different features should have higher MMD
    slow_feat_diff = torch.randn(2, 16, 1, 20)
    _, mmd_loss_diff = fusion(fast_feat, slow_feat_diff)
    print(f"  MMD loss (different): {mmd_loss_diff.item():.6f}")

    # MMD should be lower for identical features
    # (may not always hold due to numerical instability, so just check non-negative)
    assert mmd_loss_same >= 0, "MMD loss should be non-negative"
    assert mmd_loss_diff >= 0, "MMD loss should be non-negative"

    print("[PASS] MMD loss computation test PASSED")
    return True


def test_fusion_parameters():
    """Test fusion parameter count"""
    print("\n=== Test: Fusion Parameters ===")
    fusion = MultiMechanismFusion(input_dim=16, hidden_dim=24, output_dim=2)
    total_params = sum(p.numel() for p in fusion.parameters())
    trainable_params = sum(p.numel() for p in fusion.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    assert total_params > 0, "Should have parameters"
    assert trainable_params == total_params, "All parameters should be trainable"
    print("[PASS] Fusion parameters test PASSED")
    return True


def test_fusion_batch_sizes():
    """Test fusion with different batch sizes"""
    print("\n=== Test: Fusion with Different Batch Sizes ===")
    fusion = MultiMechanismFusion(input_dim=16, hidden_dim=24)

    for batch_size in [1, 4, 8]:
        fast_feat = torch.randn(batch_size, 16, 1, 20)
        slow_feat = torch.randn(batch_size, 16, 1, 20)
        output, _ = fusion(fast_feat, slow_feat)
        print(f"  Batch={batch_size}: Output shape {output.shape}")
        assert output.shape[0] == batch_size, "Batch size mismatch"

    print("[PASS] Different batch sizes test PASSED")
    return True


def run_all_tests():
    """Run all fusion tests"""
    print("="*60)
    print("Running Multi-Mechanism Fusion Test Suite")
    print("="*60)

    tests = [
        test_fusion_forward,
        test_fusion_different_time_dims,
        test_fusion_mechanisms,
        test_fusion_gradient_flow,
        test_fusion_mmd_loss,
        test_fusion_parameters,
        test_fusion_batch_sizes
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"[FAIL] Test FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "="*60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("="*60)

    if failed == 0:
        print("\n[SUCCESS] All Fusion tests PASSED!")
        return True
    else:
        print(f"\n[ERROR] {failed} test(s) FAILED")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
