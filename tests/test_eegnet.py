"""
Unit tests for EEGNet (Fast Pathway)
"""
import torch
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.fast_pathway.eegnet import EEGNet


def test_eegnet_forward():
    """Test EEGNet forward pass with correct dimensions"""
    print("\n=== Test: EEGNet Forward Pass ===")
    model = EEGNet(num_channels=59, num_timepoints=100)
    x = torch.randn(4, 1, 59, 100)
    y = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")

    assert y.shape[0] == 4, "Batch size mismatch"
    assert y.shape[1] == 16, "Feature channels mismatch (expected F2=16)"
    assert y.shape[2] == 1, "Spatial dimension should be 1 after spatial conv"
    assert y.shape[3] > 0, "Temporal dimension should be > 0"
    print("[PASS] EEGNet forward pass test PASSED")
    return True


def test_eegnet_parameters():
    """Test EEGNet parameter count"""
    print("\n=== Test: EEGNet Parameters ===")
    model = EEGNet()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # EEGNet should be lightweight (< 10k params)
    assert total_params < 10000, f"Model too large: {total_params:,} parameters (expected < 10k)"
    assert trainable_params == total_params, "All parameters should be trainable"
    print("[PASS] EEGNet parameter count test PASSED")
    return True


def test_eegnet_different_input_sizes():
    """Test EEGNet with different temporal lengths"""
    print("\n=== Test: EEGNet with Different Input Sizes ===")
    model = EEGNet(num_channels=59, num_timepoints=100)

    for time_length in [100, 150, 200]:
        x = torch.randn(2, 1, 59, time_length)
        y = model(x)
        print(f"  Time={time_length}: Input {x.shape} → Output {y.shape}")
        assert y.shape[0] == 2, "Batch size mismatch"
        assert y.shape[1] == 16, "Feature channels mismatch"

    print("[PASS] EEGNet flexible input test PASSED")
    return True


def test_eegnet_gradient_flow():
    """Test that gradients flow through the model"""
    print("\n=== Test: EEGNet Gradient Flow ===")
    model = EEGNet(num_channels=59, num_timepoints=100)
    x = torch.randn(2, 1, 59, 100, requires_grad=True)

    y = model(x)
    loss = y.sum()
    loss.backward()

    # Check input has gradients
    assert x.grad is not None, "Input gradients should exist"

    # Check all model parameters have gradients
    params_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
    total_params = sum(1 for p in model.parameters())
    print(f"  Parameters with gradients: {params_with_grad}/{total_params}")
    assert params_with_grad == total_params, "All parameters should have gradients"

    print("[PASS] EEGNet gradient flow test PASSED")
    return True


def test_eegnet_output_range():
    """Test that output values are reasonable"""
    print("\n=== Test: EEGNet Output Range ===")
    model = EEGNet()
    model.eval()

    with torch.no_grad():
        x = torch.randn(4, 1, 59, 100)
        y = model(x)

    print(f"  Output mean: {y.mean().item():.4f}")
    print(f"  Output std: {y.std().item():.4f}")
    print(f"  Output min: {y.min().item():.4f}")
    print(f"  Output max: {y.max().item():.4f}")

    # ELU activation allows negative values but should be bounded
    assert not torch.isnan(y).any(), "Output contains NaN"
    assert not torch.isinf(y).any(), "Output contains Inf"
    print("[PASS] EEGNet output range test PASSED")
    return True


def run_all_tests():
    """Run all EEGNet tests"""
    print("="*60)
    print("Running EEGNet Test Suite")
    print("="*60)

    tests = [
        test_eegnet_forward,
        test_eegnet_parameters,
        test_eegnet_different_input_sizes,
        test_eegnet_gradient_flow,
        test_eegnet_output_range
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"[FAIL] Test FAILED: {e}")
            failed += 1

    print("\n" + "="*60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("="*60)

    if failed == 0:
        print("\n[SUCCESS] All EEGNet tests PASSED!")
        return True
    else:
        print(f"\n[ERROR] {failed} test(s) FAILED")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
