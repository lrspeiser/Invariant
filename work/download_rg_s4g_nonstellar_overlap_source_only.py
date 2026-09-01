"""Acquire three S4G P5 nonstellar benchmark maps with no redirects or retries."""

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
INVENTORY_PATH = ROOT / "work/rg-s4g-nonstellar-overlap-source-download-inventory-v1.json"
ROWS = (
    ("NGC2976", 2_494_080),
    ("NGC3198", 2_580_480),
    ("NGC3521", 3_245_760),
)


class NoRedirect(urllib.request.HTTPRedirectHandler):
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


def main() -> None:
    PRIVATE_ROOT.mkdir(parents=True, exist_ok=True)
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler(),
        NoRedirect(),
        urllib.request.HTTPHandler(),
        urllib.request.HTTPSHandler(),
    )
    records: list[dict[str, object]] = []
    for index, (object_id, expected) in enumerate(ROWS, 1):
        url = (
            "https://irsa.ipac.caltech.edu/data/SPITZER/S4G/galaxies/"
            f"{object_id}/P5/{object_id}.nonstellar.fits"
        )
        target = PRIVATE_ROOT / f"{object_id}__S4G_P5__NONSTELLAR_MAP.fits"
        if not target.exists():
            request = urllib.request.Request(
                url, method="GET", headers={"Accept-Encoding": "identity"}
            )
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
        if target.stat().st_size != expected:
            raise RuntimeError("existing source size changed")
        record = {
            "object_id": object_id,
            "survey": "S4G_P5",
            "role": "NONSTELLAR_MAP",
            "url": url,
            "relative_path": target.relative_to(ROOT).as_posix(),
            "bytes": expected,
            "sha256": _sha256(target),
        }
        records.append(record)
        print(f"{index}/3 {target.name} {expected} {record['sha256']}")
    payload: dict[str, object] = {
        "schema": "invariant-work-rg-s4g-nonstellar-overlap-source-download-inventory-1.0",
        "purpose": "Source-only reconstruction benchmark for SINGS-to-stellar conversion.",
        "paper": "https://arxiv.org/abs/1410.0009",
        "file_count": 3,
        "bytes": sum(expected for _object_id, expected in ROWS),
        "records": records,
        "response_files_opened": 0,
        "response_rows_opened": 0,
        "scores_computed": 0,
    }
    payload["content_sha256"] = _content_sha256(payload)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if INVENTORY_PATH.exists() and INVENTORY_PATH.read_text(encoding="utf-8") != encoded:
        raise RuntimeError("existing inventory changed")
    if not INVENTORY_PATH.exists():
        INVENTORY_PATH.write_text(encoded, encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("file_count", "bytes", "content_sha256")}))


if __name__ == "__main__":
    main()
