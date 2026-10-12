"""Decode tensor descriptors only using a closed whitelist; no storage allocation."""
from collections import OrderedDict
import io
import json
import pickle
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parent


def tensor_descriptor(storage, offset, shape, stride, requires_grad, hooks, *extra):
    return {"storage": storage, "offset": offset, "shape": list(shape), "stride": list(stride)}


class MetadataOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {
            ("collections", "OrderedDict"): OrderedDict,
            ("torch", "FloatStorage"): "float32",
            ("torch", "BoolStorage"): "bool",
            ("torch._utils", "_rebuild_tensor_v2"): tensor_descriptor,
        }
        if (module, name) not in allowed:
            raise pickle.UnpicklingError("Unsupported metadata global")
        return allowed[module, name]

    def persistent_load(self, identifier):
        if not isinstance(identifier, tuple) or len(identifier) != 5 or identifier[0] != "storage":
            raise pickle.UnpicklingError("Unsupported storage descriptor")
        _, dtype, key, location, count = identifier
        if dtype not in ("float32", "bool") or not isinstance(count, int) or count < 0:
            raise pickle.UnpicklingError("Invalid storage descriptor")
        return {"dtype": dtype, "key": key, "location": location, "elements": count}


def main():
    directory = json.loads((ROOT / "zip_directory.json").read_text())
    info = next(x for x in directory["metadata"] if x["name"].endswith("data.pkl"))
    path = ROOT / "weights/prithvi.wxc.2300m.v1.pt.partial"
    if not path.exists():
        path = path.with_suffix("")
    with path.open("rb") as stream:
        stream.seek(info["header_offset"])
        header = stream.read(30)
        if header[:4] != b"PK\x03\x04":
            raise ValueError("Invalid ZIP local header")
        name_size, extra_size = struct.unpack_from("<HH", header, 26)
        stream.seek(name_size+extra_size, 1)
        metadata = stream.read(info["bytes"])
    checkpoint = MetadataOnlyUnpickler(io.BytesIO(metadata)).load()
    state = checkpoint["model_state"]
    storages = {v["storage"]["key"]: v["storage"] for v in state.values()}
    size = sum(s["elements"]*(4 if s["dtype"] == "float32" else 1) for s in storages.values())
    report = {"status": "descriptors_decoded_not_weight_values", "top_level_keys": list(checkpoint),
              "model_state_storages_bytes": size, "model_state_tensor_count": len(state),
              "model_state": state, "pretrained_loaded": False, "whole_file_sha256_verified": False}
    (ROOT / "tensor_descriptors.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k != "model_state"}, indent=2))


if __name__ == "__main__":
    main()
