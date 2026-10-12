import pytest
import pandas as pd

from full_climatology_inventory import required_paths, collect_metadata, select_entries, CANONICAL_API


def test_support_clock_and_calendar():
    frame = pd.DataFrame({'support_latest_utc': [
        '2020-12-31T18:00:00Z', '2020-12-30T18:00:00Z',
        '2020-12-30T18:00:00Z']})
    paths = required_paths(frame)
    assert paths == sorted([
        'climatology/climate_surface_doy001_hour00.nc',
        'climatology/climate_vertical_doy001_hour00.nc',
        'climatology/climate_surface_doy365_hour00.nc',
        'climatology/climate_vertical_doy365_hour00.nc'])


@pytest.mark.parametrize('value', ['2020-01-01', None, '2020-01-01T01:00:00Z'])
def test_invalid_support_clock(value):
    with pytest.raises(ValueError):
        required_paths(pd.DataFrame({'support_latest_utc': [value]}))


class Response:
    def __init__(self, items, next_url=None):
        self.items = items
        self.links = {'next': {'url': next_url}} if next_url else {}
    def raise_for_status(self):
        pass
    def json(self):
        return self.items


class Session:
    def __init__(self, responses):
        self.responses = iter(responses)
    def get(self, *args, **kwargs):
        return next(self.responses)


def test_pagination_complete_and_duplicate_guard():
    entries, pages = collect_metadata(Session([
        Response([{'path': 'a'}], CANONICAL_API + '?cursor=x'),
        Response([{'path': 'b'}])]))
    assert len(entries) == 2 and len(pages) == 2
    with pytest.raises(ValueError, match='duplicate'):
        collect_metadata(Session([
            Response([{'path': 'a'}], CANONICAL_API + '?cursor=x'),
            Response([{'path': 'a'}])]))


def test_rejects_untrusted_next_and_truncated_pagination():
    with pytest.raises(ValueError, match='destination'):
        collect_metadata(Session([Response([], 'https://evil.invalid')]))
    with pytest.raises(ValueError, match='incomplete'):
        collect_metadata(Session([Response([], CANONICAL_API + '?cursor=x')]), max_pages=1)


def test_selection_requires_sha_and_consistent_size():
    item = {'path': 'a', 'size': 12, 'lfs': {'oid': 'a' * 64, 'size': 12}}
    selected, missing = select_entries(['a', 'b'], {'a': item})
    assert missing == ['b'] and selected[0]['lfs']['oid'] == 'a' * 64
    item['lfs']['size'] = 13
    with pytest.raises(ValueError):
        select_entries(['a'], {'a': item})
