"""Interactive, non-persistent Earthdata login and a bounded MAIAC pilot download."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def login_from_form(client, username, password):
    keys = ("EARTHDATA_USERNAME", "EARTHDATA_PASSWORD", "EARTHDATA_TOKEN")
    previous = {key: os.environ.get(key) for key in keys}
    try:
        os.environ["EARTHDATA_USERNAME"] = username
        os.environ["EARTHDATA_PASSWORD"] = password
        os.environ.pop("EARTHDATA_TOKEN", None)
        return client.login(strategy="environment", persist=False)
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def credential_form(on_ready=None):
    import tkinter as tk
    from tkinter import ttk

    window = tk.Tk()
    window.title("Earthdata 登录")
    window.geometry("460x290")
    window.resizable(False, False)
    frame = ttk.Frame(window, padding=24)
    frame.pack(fill="both", expand=True)
    frame.columnconfigure(0, weight=1)
    ttk.Label(frame, text="Earthdata 用户名").grid(row=0, column=0, sticky="w")
    username = ttk.Entry(frame)
    username.grid(row=1, column=0, sticky="ew", pady=(5, 14))
    ttk.Label(frame, text="密码").grid(row=2, column=0, sticky="w")
    password = ttk.Entry(frame, show="*")
    password.grid(row=3, column=0, sticky="ew", pady=(5, 8))
    visible = tk.BooleanVar(value=False)

    def toggle():
        password.configure(show="" if visible.get() else "*")

    ttk.Checkbutton(frame, text="显示密码", variable=visible, command=toggle).grid(
        row=4, column=0, sticky="w"
    )
    status = ttk.Label(frame, text="凭据仅用于本次连接，不保存到文件。")
    status.grid(row=5, column=0, sticky="w", pady=10)
    result = None

    def submit(event=None):
        nonlocal result
        if not username.get().strip() or not password.get():
            status.configure(text="请填写用户名和密码。")
            return
        result = (username.get().strip(), password.get())
        password.delete(0, "end")
        window.destroy()

    ttk.Button(frame, text="登录并下载", command=submit).grid(row=6, column=0, sticky="e")
    window.bind("<Return>", submit)
    window.protocol("WM_DELETE_WINDOW", window.destroy)
    window.lift()
    window.attributes("-topmost", True)
    window.after(700, lambda: window.attributes("-topmost", False))
    username.focus_force()
    if on_ready is not None:
        window.after(50, lambda: on_ready(window, username, password, visible, toggle, submit))
    window.mainloop()
    return result


def inspect_download(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        if stream.read(4) != b"\x0e\x03\x13\x01":
            raise ValueError("Expected HDF4 file, not a login or error response")
        stream.seek(0)
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "name": path.name,
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
        "validation": "signature_only_not_science_quality",
    }


def inspect_downloads(paths):
    if not paths:
        raise ValueError("No files downloaded")
    return [inspect_download(path) for path in paths]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog-only", action="store_true")
    args = parser.parse_args()
    import earthaccess

    query = dict(
        short_name="MCD19A2", version="061",
        bounding_box=(103.5, 30.2, 104.7, 31.1),
        temporal=("2024-01-01", "2024-01-03"), count=2,
    )
    results = earthaccess.search_data(**query)
    if not results:
        raise ValueError("No matching granules")
    # Metadata is public; no authenticated session, cookies or tokens are serialized.
    metadata = [dict(item) for item in results]
    output = ROOT / "data/raw/maiac_pilot_20260924"
    output.mkdir(parents=True, exist_ok=True)
    (output / "catalog.json").write_text(
        json.dumps({"query": query, "granules": metadata}, indent=2), encoding="utf-8"
    )
    print(f"Found {len(results)} granules. Metadata: {output / 'catalog.json'}")
    if args.catalog_only:
        return
    print("请在 Earthdata 登录框中输入账号和密码。")
    credentials = credential_form()
    if credentials is None:
        print("已取消登录。")
        return
    print("正在连接 Earthdata，请稍候……")
    try:
        auth = login_from_form(earthaccess, *credentials)
    finally:
        del credentials
    if not auth.authenticated:
        raise RuntimeError("Earthdata authentication failed")
    started = datetime.now(timezone.utc).isoformat()
    paths = earthaccess.download(results, str(output), threads=1)
    files = inspect_downloads(paths)
    if len(files) != len(results):
        raise ValueError("Incomplete pilot download")
    manifest = {
        "download_started_at": started,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "product": "MCD19A2.061", "files": files,
        "operational_availability_proven": False,
        "science_quality_checked": False,
    }
    (output / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print("下载完成，文件头和 SHA256 已检查。尚未检查像元质量，尚未训练模型。")
    print(output)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # Authentication exceptions can contain sensitive response details.
        print(f"操作未完成（{type(exc).__name__}）。请检查网络、账号激活和下载权限。")
        sys.exit(1)
