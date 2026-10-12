import numpy as np
import pytest
import torch

from haze_forecast.regional_airformer import create_airformer
from scripts.intervene_inactive_spatial import restricted_assignment, spatial_support


def test_source_axis_removed_and_sectors_renormalized():
    original = torch.ones(3, 3, 1) / 3
    assignment, mask = restricted_assignment(original, [True, False, True])
    assert (assignment[:, 1] == 0).all()
    torch.testing.assert_close(assignment.sum(1), torch.ones(3, 1))
    assert not mask.any()
    restored, _ = restricted_assignment(original, [True, True, True])
    assert torch.equal(restored, original)


def test_inactive_input_cannot_reach_active_outputs_and_buffers_restore():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    model = create_airformer(17, np.array([[104., 30.], [104.1, 30.1], [104.2, 30.2]])).eval()
    before = {key: value.clone() for key, value in model.state_dict().items()}
    x = torch.randn(2, 24, 3, 17)
    altered = x.clone()
    altered[:, :, 1] += 20
    with torch.no_grad(), spatial_support(model, [True, False, True]):
        left, right = model(x), model(altered)
    torch.testing.assert_close(left[:, :, [0, 2]], right[:, :, [0, 2]], rtol=0, atol=1e-6)
    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key])


def test_buffers_restore_on_exception_and_empty_support_rejected():
    torch.set_num_threads(1)
    model = create_airformer(17, np.array([[104., 30.], [104.1, 30.1]])).eval()
    before = {key: value.clone() for key, value in model.state_dict().items()}
    with pytest.raises(RuntimeError, match='probe'), spatial_support(model, [True, False]):
        raise RuntimeError('probe')
    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key])
    with pytest.raises(ValueError):
        restricted_assignment(model.assignment, [False, False])
