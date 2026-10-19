"""Explicit report exports with collision checks and per-report manifests."""
import hashlib
import json
from pathlib import Path
import re

import numpy as np


def safe_name(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError("Report names must be simple filenames without directory components.")
    return value


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__} as report metadata.")


def save_reports(directory, *, tables=None, documents=None, prefix="selection",
                 overwrite=False, indexed=()):
    """Write named CSV/JSON files; preflight all collisions before writing.

    Names are exact filenames, independent of prefix. Prefix names the manifest.
    Existing unrelated files are never removed. Index is exported only for named
    matrix tables. A manifest lists the files written by this call and hashes.
    """
    directory = Path(directory)
    safe_name(prefix)
    tables, documents = dict(tables or {}), dict(documents or {})
    manifest_name = f"{prefix}_manifest.json"
    filenames = list(tables) + list(documents) + [manifest_name]
    if len(set(filenames)) != len(filenames):
        raise ValueError("Duplicate report filenames.")
    for name in filenames:
        safe_name(name)
        if (directory / name).is_symlink():
            raise ValueError("Report destinations must not be symlinks.")
    if not set(indexed).issubset(tables):
        raise ValueError("Indexed files must refer to supplied tables.")
    collisions = [name for name in filenames if (directory / name).exists()]
    if collisions and not overwrite:
        raise FileExistsError("Report files already exist: " + ", ".join(collisions))
    # Serialize before modifying existing reports (e.g. reject invalid JSON first).
    payloads = {name: table.to_csv(index=name in indexed).encode("utf-8")
                for name, table in tables.items()}
    payloads.update({name: (json.dumps(doc, indent=2, ensure_ascii=False,
                                      allow_nan=False, default=_json_default) + "\n").encode("utf-8")
                     for name, doc in documents.items()})
    manifest = {"format_version": 1, "files": [
        {"name": name, "sha256": hashlib.sha256(data).hexdigest()}
        for name, data in payloads.items()
    ]}
    payloads[manifest_name] = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
    directory.mkdir(parents=True, exist_ok=True)
    for name, data in payloads.items():
        path = directory / name
        path.write_bytes(data)
    return {name: directory / name for name in payloads}
