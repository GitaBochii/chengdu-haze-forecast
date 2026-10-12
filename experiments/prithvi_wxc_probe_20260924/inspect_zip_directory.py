"""Read archive directory only, without unpickling or retrieving tensor values."""
import json
from pathlib import Path
import zipfile

from inspect_checkpoint import RangeReader

ROOT = Path(__file__).resolve().parent


def main():
    revision = "e2f62b3e19890f1a5e225bb123acb4b33df85ef8"
    url = f"https://huggingface.co/Prithvi-WxC/prithvi.wxc.2300m.v1/resolve/{revision}/prithvi.wxc.2300m.v1.pt"
    reader = RangeReader(url, 28447290466)
    try:
        with zipfile.ZipFile(reader) as archive:
            files = archive.infolist()
            report = {"status": "zip_directory_read", "records": len(files),
                      "metadata": [{"name": f.filename, "bytes": f.file_size,
                                    "header_offset": f.header_offset} for f in files if "/data/" not in f.filename],
                      "stored_bytes": sum(f.compress_size for f in files),
                      "uncompressed_bytes": sum(f.file_size for f in files),
                      "bytes_read": reader.received, "pretrained_loaded": False}
    except Exception as exc:
        report = {"status": "failed", "error_type": type(exc).__name__, "bytes_read": reader.received}
    (ROOT / "zip_directory.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
