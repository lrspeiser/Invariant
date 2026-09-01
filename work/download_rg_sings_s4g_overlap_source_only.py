"""Acquire the six SINGS source-only overlap files with no redirects or retries."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRIVATE_ROOT = ROOT / "work/private/open-gravity-rg-sings-s4g-overlap-source-only-v1"
INVENTORY_PATH = ROOT / "work/rg-sings-s4g-overlap-source-download-inventory-v1.json"
ROWS = (
    ("NGC2976", "STELLAR_IRAC1_FLUX", "ngc2976_v7.phot.1.fits", 7_882_560),
    ("NGC2976", "STELLAR_IRAC1_WEIGHT", "ngc2976_v7.phot.1_wt.fits", 7_882_560),
    ("NGC3198", "STELLAR_IRAC1_FLUX", "ngc3198_v7.phot.1.fits", 8_167_680),
    ("NGC3198", "STELLAR_IRAC1_WEIGHT", "ngc3198_v7.phot.1_wt.fits", 8_167_680),
    ("NGC3521", "STELLAR_IRAC1_FLUX", "ngc3521_v7.phot.1.fits", 10_232_640),
    ("NGC3521", "STELLAR_IRAC1_WEIGHT", "ngc3521_v7.phot.1_wt.fits", 10_232_640),
)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    """Reject every redirect before a second request can be dispatched."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001, ANN201
        raise urllib.error.HTTPError(req.full_url, code, "redirect refused", headers, fp)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _content_sha256(payload: dict[str, object]) -> str:
    clean = dict(payload)
    clean.pop("content_sha256", None)
    encoded = json.dumps(clean, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _url(object_id: str, filename: str) -> str:
    return (
        "https://irsa.ipac.caltech.edu/data/SPITZER/SINGS/galaxies/"
        f"{object_id.lower()}/IRAC/{filename}"
    )


def _download_exact(opener, row: tuple[str, str, str, int], target: Path) -> None:  # noqa: ANN001
    object_id, _role, filename, expected = row
    url = _url(object_id, filename)
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


def main() -> None:
    PRIVATE_ROOT.mkdir(parents=True, exist_ok=True)
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler(),
        NoRedirect(),
        urllib.request.HTTPHandler(),
        urllib.request.HTTPSHandler(),
    )
    records: list[dict[str, object]] = []
    for index, row in enumerate(ROWS, 1):
        object_id, role, filename, expected = row
        target = PRIVATE_ROOT / f"{object_id}__SINGS_IRAC1__{role}.fits"
        if not target.exists():
            _download_exact(opener, row, target)
        if target.stat().st_size != expected:
            raise RuntimeError("existing source size changed")
        record = {
            "object_id": object_id,
            "survey": "SINGS_IRAC1",
            "role": role,
            "url": _url(object_id, filename),
            "relative_path": target.relative_to(ROOT).as_posix(),
            "bytes": expected,
            "sha256": _sha256(target),
        }
        records.append(record)
        print(f"{index}/6 {target.name} {expected} {record['sha256']}")
    payload: dict[str, object] = {
        "schema": "invariant-work-rg-sings-s4g-overlap-source-download-inventory-1.0",
        "purpose": "Source-only SINGS/S4G stellar-conversion benchmark; no rotation response.",
        "paper_bindings": {
            "sings": "https://arxiv.org/abs/astro-ph/0305437",
            "s4g_mass_maps": "https://arxiv.org/abs/1410.0009",
            "fixed_ml_3p6": "https://arxiv.org/abs/1402.5210",
        },
        "objects": ["NGC2976", "NGC3198", "NGC3521"],
        "file_count": len(records),
        "bytes": sum(int(record["bytes"]) for record in records),
        "records": records,
        "response_files_opened": 0,
        "response_rows_opened": 0,
        "scores_computed": 0,
    }
    payload["content_sha256"] = _content_sha256(payload)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if INVENTORY_PATH.exists():
        if INVENTORY_PATH.read_text(encoding="utf-8") != encoded:
            raise RuntimeError("existing inventory changed")
    else:
        INVENTORY_PATH.write_text(encoded, encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("file_count", "bytes", "content_sha256")}))


if __name__ == "__main__":
    main()
