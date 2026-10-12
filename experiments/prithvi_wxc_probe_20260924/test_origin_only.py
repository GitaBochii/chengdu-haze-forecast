import numpy as np
import torch

from haze_forecast.regional_recovery_network import RecoveryPM25GNN
from experiments.prithvi_wxc_probe_20260924.origin_only import RecoveryOriginOnly


def test_origin_only_exact_hazard_and_gradients():
    torch.set_num_threads(2)
    torch.manual_seed(24)
    args = (17, np.array([[104.,30.],[104.1,30.1],[104.2,30.2]]), torch.zeros(2), torch.ones(2))
    original = RecoveryPM25GNN(*args)
    fast = RecoveryOriginOnly(*args)
    fast.load_state_dict(original.state_dict(), strict=True)
    x = torch.randn(2,24,3,17)
    active = torch.tensor([[[True,False],[True,True],[False,True]]]*2)
    curves, expected = original(x,active)
    absent, actual = fast(x,active)
    assert curves.shape == (2,72,3,2) and absent is None
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    expected.square().mean().backward()
    actual.square().mean().backward()
    for (name,a),(other,b) in zip(original.named_parameters(),fast.named_parameters()):
        assert name == other
        if a.grad is None:
            assert b.grad is None
        else:
            torch.testing.assert_close(a.grad,b.grad,rtol=0,atol=0)
