import copy

import pytest


def inventory():
    return {"revision": "e2f62b3e19890f1a5e225bb123acb4b33df85ef8",
            "candidate_manifest_sha256": "a"*64, "missing": [],
            "files": [{"path": "merra-2/MERRA2_sfc_20200101.nc", "size": 100,
                       "lfs": {"oid": "b"*64}}]}


def test_accepts_pinned_hashed_inventory():
    from collect_pilot_data import validate_inventory
    assert validate_inventory(inventory(), "a"*64) == inventory()["files"]


@pytest.mark.parametrize("fault", ["revision", "candidate", "missing", "path", "hash", "duplicate"])
def test_rejects_untrusted_inventory(fault):
    from collect_pilot_data import validate_inventory
    data = copy.deepcopy(inventory())
    if fault == "revision":
        data["revision"] = "main"
    elif fault == "candidate":
        data["candidate_manifest_sha256"] = "c"*64
    elif fault == "missing":
        data["missing"] = ["unknown.nc"]
    elif fault == "path":
        data["files"][0]["path"] = "../escape.nc"
    elif fault == "hash":
        data["files"][0]["lfs"] = {}
    else:
        data["files"] *= 2
    with pytest.raises(ValueError):
        validate_inventory(data, "a"*64)
