"""One-shot, no-redirect source-only acquisition for the 12-galaxy RG expansion."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

from sigma_theory_compiler import (
    open_gravity_refracted_gravity_things_heracles_sparc_3d_expansion_preflight_v1 as preflight,
)


ROOT = Path(__file__).resolve().parents[1]
PRIVATE_ROOT = ROOT / "work/private/open-gravity-rg-12gal-source-only-v1"
INVENTORY_PATH = ROOT / "work/rg-12gal-source-download-inventory-v1.json"


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001, ANN201
        raise urllib.error.HTTPError(req.full_url, code, "redirect refused", headers, fp)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def filename(row: dict[str, object]) -> str:
    suffix = ".fits.gz" if str(row["url"]).endswith(".fits.gz") else ".fits"
    return f"{row['object_id']}__{row['survey']}__{row['role']}{suffix}"


def download_exact(opener, row: dict[str, object], target: Path) -> dict[str, object]:  # noqa: ANN001
    expected = int(row["bytes"])
    url = str(row["url"])
    request = urllib.request.Request(url, method="GET", headers={"Accept-Encoding": "identity"})
    temporary: Path | None = None
    try:
        with opener.open(request, timeout=120) as response:
            if response.status != 200 or response.geturl() != url:
                raise RuntimeError("unexpected response or redirect")
            if int(response.headers["Content-Length"]) != expected:
                raise RuntimeError("Content-Length changed")
            with tempfile.NamedTemporaryFile(dir=PRIVATE_ROOT, delete=False) as handle:
                temporary = Path(handle.name)
                remaining = expected
                while remaining:
                    chunk = response.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise RuntimeError("source body truncated")
                    handle.write(chunk)
                    remaining -= len(chunk)
                handle.flush()
                os.fsync(handle.fileno())
        if temporary.stat().st_size != expected:
            raise RuntimeError("source byte count changed")
        os.link(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {
        "object_id": row["object_id"],
        "survey": row["survey"],
        "role": row["role"],
        "url": url,
        "relative_path": target.relative_to(ROOT).as_posix(),
        "bytes": target.stat().st_size,
        "sha256": sha256(target),
    }


def main() -> None:
    config = preflight.load_config()
    rows = preflight.flatten_source_files(config)
    PRIVATE_ROOT.mkdir(parents=True, exist_ok=True)
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler(),
        NoRedirect(),
        urllib.request.HTTPHandler(),
        urllib.request.HTTPSHandler(),
    )
    inventory = []
    for index, row in enumerate(rows, 1):
        target = PRIVATE_ROOT / filename(row)
        if target.exists():
            if target.stat().st_size != row["bytes"]:
                raise RuntimeError(f"existing source size changed: {target.name}")
            record = {
                "object_id": row["object_id"],
                "survey": row["survey"],
                "role": row["role"],
                "url": row["url"],
                "relative_path": target.relative_to(ROOT).as_posix(),
                "bytes": target.stat().st_size,
                "sha256": sha256(target),
            }
        else:
            record = download_exact(opener, row, target)
        inventory.append(record)
        print(f"{index:02d}/{len(rows)} {target.name} {record['bytes']} {record['sha256']}")
    payload = {
        "schema": "invariant-work-rg-12gal-source-download-inventory-1.0",
        "file_count": len(inventory),
        "bytes": sum(row["bytes"] for row in inventory),
        "records": inventory,
    }
    payload["content_sha256"] = preflight.content_sha256(payload)
    if INVENTORY_PATH.exists():
        if json.loads(INVENTORY_PATH.read_text(encoding="utf-8")) != payload:
            raise RuntimeError("existing inventory changed")
    else:
        INVENTORY_PATH.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps({key: payload[key] for key in ("file_count", "bytes", "content_sha256")}))


if __name__ == "__main__":
    main()
