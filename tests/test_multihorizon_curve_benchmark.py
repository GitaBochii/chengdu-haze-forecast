from haze_forecast.multihorizon_curve import select_hybrid


def test_select_hybrid_uses_2024_choice_and_requires_2025_improvement():
    assert select_hybrid({"aft": 16.4, "curve": 15.8}, {"aft": 26.27, "curve": 25.0}) == "curve"
    assert select_hybrid({"aft": 16.4, "curve": 15.8}, {"aft": 26.27, "curve": 27.0}) is None
