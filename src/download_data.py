"""Acquire and validate official NHANES files without model evaluation."""
from __future__ import annotations

import hashlib
import io
import json
import logging
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "config/study.json").read_text())
MANIFEST = ROOT / "manifests/downloads.json"
LOGGER = logging.getLogger(__name__)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate(data: bytes, kind: str) -> dict:
    if kind == "xpt":
        if not data.startswith(b"HEADER RECORD*******LIBRARY HEADER RECORD!!!!!!!"):
            raise ValueError("Response is not a SAS XPORT file")
        frame = pd.read_sas(io.BytesIO(data), format="xport", encoding="utf-8")
        if "SEQN" not in frame or frame.SEQN.isna().any() or frame.SEQN.duplicated().any():
            raise ValueError("Missing or duplicated participant identifiers")
        return {"rows": len(frame), "columns": frame.columns.tolist()}
    text = data.decode("utf-8", errors="replace")
    if "Codebook" not in text or "SEQN" not in text:
        raise ValueError("Response is not the expected NHANES codebook")
    return {}


def acquire(task: dict, previous: dict | None) -> dict:
    path = ROOT / task["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        data = path.read_bytes()
        details = validate(data, task["kind"])
        if previous and digest(data) != previous["sha256"]:
            raise ValueError(f"Cached file identity changed: {path}")
        return {**task, **details, "bytes": len(data), "sha256": digest(data),
                "retrieved_utc": previous["retrieved_utc"] if previous else datetime.now(timezone.utc).isoformat(),
                "status": "validated_cached"}
    last_error = None
    for attempt in range(3):
        try:
            request = urllib.request.Request(task["url"], headers={"User-Agent": "AIHEALTH academic data audit/0.1"})
            with urllib.request.urlopen(request, timeout=35) as response:
                data = response.read()
                resolved_url = response.url
            details = validate(data, task["kind"])
            partial = path.with_suffix(path.suffix + ".partial")
            partial.write_bytes(data)
            partial.replace(path)
            return {**task, **details, "resolved_url": resolved_url, "bytes": len(data),
                    "sha256": digest(data), "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                    "status": "downloaded_validated"}
        except Exception as exc:
            last_error = exc
            LOGGER.warning("Attempt %d failed for %s: %s", attempt + 1, task["path"], exc)
            time.sleep(attempt + 1)
    raise RuntimeError(f"Could not retrieve {task['url']}: {last_error}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    previous = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {"files": []}
    cached = {f["path"]: f for f in previous["files"]}
    tasks = []
    for cycle in CONFIG["cycles"]:
        for module in CONFIG["modules"]:
            name = f"{module}_{cycle['suffix']}"
            for kind, extension in [("xpt", "xpt"), ("codebook", "htm")]:
                filename = f"{name}.{extension}"
                tasks.append({"cycle": cycle["name"], "module": module, "kind": kind,
                              "url": f"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{cycle['year']}/DataFiles/{filename}",
                              "path": f"data/raw/{cycle['name']}/{filename}"})
    results, errors = [], []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(acquire, task, cached.get(task["path"])): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                result = future.result()
                results.append(result)
                LOGGER.info("Validated %s", task["path"])
            except Exception as exc:
                errors.append({**task, "error": str(exc)})
                LOGGER.error("FAILED %s: %s", task["path"], exc)
            MANIFEST.write_text(json.dumps({"files": sorted(results, key=lambda f: f["path"]),
                                           "errors": errors, "expected_files": len(tasks),
                                           "checksum_note": "Locally calculated hashes, not publisher-provided checksums"}, indent=2))
    if errors:
        raise SystemExit(f"{len(errors)} downloads failed; audit must not proceed")
    LOGGER.info("All %d official files validated", len(results))


if __name__ == "__main__":
    main()
