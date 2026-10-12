import importlib.util
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location(
    'climatology_range_probe', Path(__file__).with_name('climatology_range_probe.py'))
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_regional_slab_offsets():
    assert MODULE.slab_range(100, (361, 576), 4, None) == (
        100 + 226 * 576 * 4, 100 + 256 * 576 * 4 - 1)
    assert MODULE.slab_range(100, (14, 361, 576), 4, 13) == (
        100 + (13 * 361 + 226) * 576 * 4,
        100 + (13 * 361 + 256) * 576 * 4 - 1)


@pytest.mark.parametrize('offset,shape,level', [
    (None, (361, 576), None), (0, (360, 576), None),
    (0, (14, 361, 576), 14), (0, (14, 361, 576), None),
    (0, (361, 576), 0),
])
def test_invalid_layout_rejected(offset, shape, level):
    with pytest.raises(ValueError):
        MODULE.slab_range(offset, shape, 4, level)


class Response:
    def __init__(self, status=206, content_range='bytes 10-13/100', body=b'abcd'):
        self.status_code = status
        self.headers = {'Content-Range': content_range}
        self.body = body

    def iter_content(self, chunk_size):
        yield self.body


def test_exact_response():
    assert MODULE.read_range_response(Response(), 10, 13, 100) == b'abcd'


@pytest.mark.parametrize('response', [
    Response(status=200), Response(content_range='bytes 11-14/100'),
    Response(content_range='bytes 10-13/101'), Response(body=b'abc'),
    Response(body=b'abcde'),
])
def test_invalid_responses_rejected(response):
    with pytest.raises(ValueError):
        MODULE.read_range_response(response, 10, 13, 100)
