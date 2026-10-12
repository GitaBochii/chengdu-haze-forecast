from scripts.extract_future_weather_sequence import select_records


def test_autumn_selection_never_redownloads_winter():
    rows = [{'quarter': 1, 'url': 'winter'}, {'quarter': 4, 'url': 'autumn'}]
    assert select_records(rows, autumn=True) == [rows[1]]
    assert select_records(rows, autumn=False) == rows
