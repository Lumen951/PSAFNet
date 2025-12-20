"""
Unit tests for C-RNN (Slow Pathway)
"""
import torch
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.slow_pathway.crnn import CRNN


def test_crnn_forward():
    """Test C-RNN forward pass"""
    print("\n=== Test: C-RNN Forward Pass ===")
    model = CRNN(num_channels=59, num_timepoints=100)
    x = torch.randn(4, 1, 59, 100)
    y = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")

    assert y.shape[0] == 4, "Batch size mismatch"
    assert y.shape[1] == 16, "Feature channels mismatch (expected 16)"
    assert y.shape[2] == 1, "Spatial dimension should be 1"
    # Temporal dimension may increase slightly due to padding in multi-scale convs
    assert y.shape[3] >= 100, f"Temporal dimension should be >= input (got {y.shape[3]})"
    print("[PASS] C-RNN forward pass test PASSED")
    return True


def test_crnn_parameters():
    """Test C-RNN parameter count"""
    print("\n=== Test: C-RNN Parameters ===")
    model = CRNN()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # C-RNN should have more params than EEGNet due to LSTM
    assert total_params > 10000, f"Model seems too small: {total_params:,}"
    assert total_params < 500000, f"Model too large: {total_params:,}"
    assert trainable_params == total_params, "All parameters should be trainable"
    print("[PASS] C-RNN parameter count test PASSED")
    return True


def test_crnn_lstm_bidirectional():
    """Test LSTM bidirectional output"""
    print("\n=== Test: C-RNN LSTM Bidirectional ===")
    model = CRNN(rnn_hidden=32, rnn_layers=2)
    x = torch.randn(2, 1, 59, 100)

    # Check that LSTM is bidirectional
    assert model.lstm.bidirectional == True, "LSTM should be bidirectional"
    print(f"  LSTM hidden size: {model.lstm.hidden_size}")
    print(f"  LSTM num layers: {model.lstm.num_layers}")
    print(f"  LSTM bidirectional: {model.lstm.bidirectional}")

    y = model(x)
    print(f"  Output shape: {y.shape}")
    print("[PASS] C-RNN LSTM bidirectional test PASSED")
    return True


def test_crnn_multiscale_conv():
    """Test multi-scale temporal convolutions"""
    print("\n=== Test: C-RNN Multi-scale Convolutions ===")
    model = CRNN()
    x = torch.randn(2, 1, 59, 100)

    # Extract intermediate features
    t1 = model.temp_conv1(x)
    t2 = model.temp_conv2(x)
    t3 = model.temp_conv3(x)

    print(f"  Conv1 (kernel=32) output: {t1.shape}")
    print(f"  Conv2 (kernel=64) output: {t2.shape}")
    print(f"  Conv3 (kernel=96) output: {t3.shape}")

    # All should have same spatial dimensions after padding
    assert t1.shape[2:] == t2.shape[2:] == t3.shape[2:], \
        "Multi-scale convs should have same output shape"
    print("[PASS] C-RNN multi-scale convolutions test PASSED")
    return True


def test_crnn_gradient_flow():
    """Test gradient flow through LSTM"""
    print("\n=== Test: C-RNN Gradient Flow ===")
    model = CRNN()
    x = torch.randn(2, 1, 59, 100, requires_grad=True)

    y = model(x)
    loss = y.sum()
    loss.backward()

    # Check gradients exist
    assert x.grad is not None, "Input should have gradients"

    params_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
    total_params = sum(1 for p in model.parameters())
    print(f"  Parameters with gradients: {params_with_grad}/{total_params}")
    assert params_with_grad == total_params, "All parameters should have gradients"
    print("[PASS] C-RNN gradient flow test PASSED")
    return True


def test_crnn_different_input_sizes():
    """Test C-RNN with different temporal lengths"""
    print("\n=== Test: C-RNN with Different Input Sizes ===")
    model = CRNN(num_channels=59, num_timepoints=100)

    for time_length in [80, 100, 120]:
        x = torch.randn(2, 1, 59, time_length)
        y = model(x)
        print(f"  Time={time_length}: Input {x.shape} -> Output {y.shape}")
        assert y.shape[0] == 2, "Batch size mismatch"
        assert y.shape[1] == 16, "Feature channels mismatch"
        # Output time may be slightly larger due to padding
        assert abs(y.shape[3] - time_length) <= 2, \
            f"Output time should be close to input (got {y.shape[3]} vs {time_length})"

    print("[PASS] C-RNN flexible input test PASSED")
    return True


def test_crnn_output_range():
    """Test output values are reasonable"""
    print("\n=== Test: C-RNN Output Range ===")
    model = CRNN()
    model.eval()

    with torch.no_grad():
        x = torch.randn(4, 1, 59, 100)
        y = model(x)

    print(f"  Output mean: {y.mean().item():.4f}")
    print(f"  Output std: {y.std().item():.4f}")
    print(f"  Output min: {y.min().item():.4f}")
    print(f"  Output max: {y.max().item():.4f}")

    assert not torch.isnan(y).any(), "Output contains NaN"
    assert not torch.isinf(y).any(), "Output contains Inf"
    print("[PASS] C-RNN output range test PASSED")
    return True


def run_all_tests():
    """Run all C-RNN tests"""
    print("="*60)
    print("Running C-RNN Test Suite")
    print("="*60)

    tests = [
        test_crnn_forward,
        test_crnn_parameters,
        test_crnn_lstm_bidirectional,
        test_crnn_multiscale_conv,
        test_crnn_gradient_flow,
        test_crnn_different_input_sizes,
        test_crnn_output_range
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
        print("\n[SUCCESS] All C-RNN tests PASSED!")
        return True
    else:
        print(f"\n[ERROR] {failed} test(s) FAILED")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
