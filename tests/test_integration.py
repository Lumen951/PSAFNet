"""
Comprehensive integration test for complete Dual-Pathway Network
"""
import torch
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from DualPathwayNet import DualPathwayNet


def test_end_to_end():
    """Test complete forward pass"""
    print("\n=== Test: End-to-End Forward Pass ===")
    model = DualPathwayNet(
        num_channels=59,
        num_timepoints=200,
        split_point=100
    )

    x = torch.randn(4, 1, 59, 200)
    output, mmd_loss = model(x)

    print(f"Input: {x.shape}")
    print(f"Output: {output.shape}")
    print(f"MMD loss: {mmd_loss.item():.4f}")

    assert output.shape == (4, 2), "Output shape mismatch"
    assert mmd_loss.item() >= 0, "MMD loss should be non-negative"
    print("[PASS] End-to-end test PASSED")
    return True


def test_gradient_flow():
    """Test backpropagation through entire model"""
    print("\n=== Test: Gradient Flow ===")
    model = DualPathwayNet()
    x = torch.randn(2, 1, 59, 200)
    labels = torch.tensor([0, 1])

    output, mmd_loss = model(x)
    loss = torch.nn.functional.cross_entropy(output, labels) + mmd_loss
    loss.backward()

    # Check gradients exist
    has_grad = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert has_grad, "Some parameters don't have gradients"

    params_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
    total_params = sum(1 for p in model.parameters())
    print(f"  Parameters with gradients: {params_with_grad}/{total_params}")
    print("[PASS] Gradient flow test PASSED")
    return True


def test_different_split_points():
    """Test model with different time splits"""
    print("\n=== Test: Different Split Points ===")
    for split_point in [80, 100, 120]:
        model = DualPathwayNet(split_point=split_point)
        x = torch.randn(2, 1, 59, 200)
        output, _ = model(x)
        print(f"  Split={split_point}: Output shape {output.shape}")
        assert output.shape == (2, 2)
    print("[PASS] Different split points test PASSED")
    return True


def test_model_size():
    """Test model parameter count"""
    print("\n=== Test: Model Size ===")
    model = DualPathwayNet()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    # Should be reasonable size (< 500k params)
    assert total_params < 500000, f"Model too large: {total_params:,} parameters"
    assert total_params > 100000, f"Model seems too small: {total_params:,} parameters"
    print("[PASS] Model size test PASSED")
    return True


def test_batch_processing():
    """Test different batch sizes"""
    print("\n=== Test: Batch Processing ===")
    model = DualPathwayNet()
    model.eval()

    for batch_size in [1, 4, 8, 16]:
        with torch.no_grad():
            x = torch.randn(batch_size, 1, 59, 200)
            output, _ = model(x)
            assert output.shape[0] == batch_size
            print(f"  Batch size {batch_size}: OK")

    print("[PASS] Batch processing test PASSED")
    return True


def test_eval_mode():
    """Test model in eval mode"""
    print("\n=== Test: Eval Mode ===")
    model = DualPathwayNet()
    model.eval()

    x = torch.randn(2, 1, 59, 200)

    with torch.no_grad():
        output1, _ = model(x)
        output2, _ = model(x)

    # Should be deterministic in eval mode
    assert torch.allclose(output1, output2, atol=1e-5), "Output should be deterministic in eval mode"
    print("[PASS] Eval mode test PASSED")
    return True


def run_all_tests():
    """Run all integration tests"""
    print("="*60)
    print("Running Dual-Pathway Network Integration Tests")
    print("="*60)

    tests = [
        test_end_to_end,
        test_gradient_flow,
        test_different_split_points,
        test_model_size,
        test_batch_processing,
        test_eval_mode
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
        print("\n[SUCCESS] All integration tests PASSED!")
        return True
    else:
        print(f"\n[ERROR] {failed} test(s) FAILED")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
