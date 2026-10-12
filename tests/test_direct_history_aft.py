import numpy as np
import torch

from haze_forecast.direct_history_aft import DirectHistoryAFT, fit_scale, transform


def test_missing_scaling_is_training_only():
    x = np.array([[1., np.nan], [3., np.nan]])
    mean, scale = fit_scale(x)
    np.testing.assert_array_equal(mean, [2, 0])
    np.testing.assert_array_equal(scale, [1, 1])
    z = transform(np.array([[5., np.nan]]), mean, scale)
    np.testing.assert_array_equal(z, [[3., 0., 0., 1.]])


def test_model_batch_independence_and_shape():
    torch.manual_seed(42)
    model = DirectHistoryAFT(24, 208).eval()
    history, snapshot = torch.randn(3, 24, 24), torch.randn(3, 208)
    actual = model(history, snapshot)
    assert actual.shape == (3,)
    torch.testing.assert_close(actual[:1], model(history[:1], snapshot[:1]))


def test_unseen_training_channel_cannot_inject_unscaled_values():
    z = transform(np.array([[5000., np.nan]]), np.zeros(2), np.ones(2), supported=np.array([False, False]))
    np.testing.assert_array_equal(z, [[0., 0., 0., 1.]])


def test_history_reaches_output_and_gradient():
    torch.manual_seed(42)
    model = DirectHistoryAFT(24, 208)
    h = torch.randn(2, 24, 24, requires_grad=True)
    model(h, torch.zeros(2, 208)).sum().backward()
    assert h.grad[:, 0].abs().sum() > 0
    assert h.grad[:, -1].abs().sum() > 0
