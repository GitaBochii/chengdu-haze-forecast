import torch

from probe import make_model


def test_encoder_matches_official_forward_and_ignores_future_truth():
    from spatial_encoder import encode_spatial
    torch.manual_seed(24)
    model = make_model(False).eval()
    batch = {"x": torch.randn(1, 2, 160, 16, 16),
             "y": torch.randn(1, 160, 16, 16),
             "static": torch.randn(1, 6, 16, 16),
             "climate": torch.randn(1, 160, 16, 16),
             "input_time": torch.tensor([6.]), "lead_time": torch.tensor([6.])}
    captured = []
    hook = model.encoder.register_forward_hook(lambda m, args, out: captured.append(out.detach()))
    with torch.inference_mode():
        torch.manual_seed(100)
        model(batch)
        hook.remove()
        torch.manual_seed(100)
        actual = encode_spatial(model, batch)
        torch.testing.assert_close(actual, captured[0])
        batch.pop("y")
        torch.manual_seed(100)
        torch.testing.assert_close(encode_spatial(model, batch), actual, rtol=0, atol=0)
