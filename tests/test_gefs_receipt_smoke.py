import pytest

from scripts.smoke_gefs_receipt import checked_range


class Response:
    status_code = 206

    def __init__(self):
        self.headers = {'Content-Range': 'bytes 10-21/100'}

    def iter_content(self, _size):
        yield b'GRIBxxxx7777'


def test_valid_range():
    assert checked_range(Response(), 10, 21, 12)[0] == b'GRIBxxxx7777'


@pytest.mark.parametrize('change', ['status', 'range', 'size'])
def test_rejects_unbounded_or_mismatched_range(change):
    response = Response()
    limit = 12
    if change == 'status':
        response.status_code = 200
    if change == 'range':
        response.headers = {'Content-Range': 'bytes 0-11/100'}
    if change == 'size':
        limit = 5
    with pytest.raises(ValueError):
        checked_range(response, 10, 21, limit)
