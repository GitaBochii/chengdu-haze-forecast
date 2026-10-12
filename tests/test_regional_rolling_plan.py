import pytest

from scripts.train_regional_rolling import fold_years


def test_selection_year_is_before_prediction_year():
    assert fold_years(2022) == (2014, 2020, 2021, 2022)
    assert fold_years(2023) == (2014, 2021, 2022, 2023)
    with pytest.raises(ValueError):
        fold_years(2025)
