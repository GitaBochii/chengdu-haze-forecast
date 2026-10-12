import importlib.util
import io
from pathlib import Path
import pickle

import pytest

spec = importlib.util.spec_from_file_location("metadata", Path(__file__).with_name("inspect_pickle_metadata.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_unknown_global_is_rejected_without_execution():
    with pytest.raises(pickle.UnpicklingError):
        module.MetadataOnlyUnpickler(io.BytesIO(pickle.dumps(eval))).load()


def test_storage_descriptor_does_not_allocate_storage():
    parser = module.MetadataOnlyUnpickler(io.BytesIO())
    result = parser.persistent_load(("storage", "float32", "42", "cuda:0", 1000000000))
    assert result["elements"] == 1000000000
    assert isinstance(result, dict)
    with pytest.raises(pickle.UnpicklingError):
        parser.persistent_load(("storage", "float32", "42", "cuda:0", -1))
