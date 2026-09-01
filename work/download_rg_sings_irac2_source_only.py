"""Acquire ten SINGS IRAC2 flux/weight pairs with no redirects or retries."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRIVATE_ROOT = ROOT / "work/private/open-gravity-rg-sings-irac2-source-only-v1"
INVENTORY_PATH = ROOT / "work/rg-sings-irac2-source-download-inventory-v1.json"
OBJECTS = (
    ("UGC04305", "hoii", 10_699_200, "PRODUCTION_FALLBACK"),
    ("NGC2841", "ngc2841", 7_718_400, "PRODUCTION_FALLBACK"),
    ("IC2574", "ic2574", 20_577_600, "PRODUCTION_FALLBACK"),
    ("DDO154", "ddo154", 4_000_320, "PRODUCTION_FALLBACK"),
    ("NGC5055", "ngc5055", 21_841_920, "PRODUCTION_FALLBACK"),
    ("NGC6946", "ngc6946", 22_783_680, "PRODUCTION_FALLBACK"),
    ("NGC7331", "ngc7331", 10_981_440, "PRODUCTION_FALLBACK"),
    ("NGC2976", "ngc2976", 7_822_080, "S4G_OVERLAP_BENCHMARK"),
    ("NGC3198", "ngc3198", 8_124_480, "S4G_OVERLAP_BENCHMARK"),
    ("NGC3521", "ngc3521", 10_195_200, "S4G_OVERLAP_BENCHMARK"),
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


def _url(slug: str, role: str) -> str:
    suffix = ".phot.2.fits" if role == "STELLAR_IRAC2_FLUX" else ".phot.2_wt.fits"
    return f"https://irsa.ipac.caltech.edu/data/SPITZER/SINGS/galaxies/{slug}/IRAC/{slug}_v7{suffix}"


def _download(opener, url: str, expected: int, target: Path) -> None:  # noqa: ANN001
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
    for object_id, slug, expected, disposition in OBJECTS:
        for role in ("STELLAR_IRAC2_FLUX", "STELLAR_IRAC2_WEIGHT"):
            url = _url(slug, role)
            target = PRIVATE_ROOT / f"{object_id}__SINGS_IRAC2__{role}.fits"
            if not target.exists():
                _download(opener, url, expected, target)
            if target.stat().st_size != expected:
                raise RuntimeError("existing source size changed")
            record = {
                "object_id": object_id,
                "survey": "SINGS_IRAC2",
                "role": role,
                "disposition": disposition,
                "url": url,
                "relative_path": target.relative_to(ROOT).as_posix(),
                "bytes": expected,
                "sha256": _sha256(target),
            }
            records.append(record)
            print(f"{len(records):02d}/20 {target.name} {expected} {record['sha256']}")
    payload: dict[str, object] = {
        "schema": "invariant-work-rg-sings-irac2-source-download-inventory-1.0",
        "purpose": "Source-only IRAC2 support for published color-dependent stellar M/L and S4G overlap validation.",
        "paper_bindings": {
            "sings_dr5": "https://irsa.ipac.caltech.edu/data/SPITZER/SINGS/doc/sings_fifth_delivery_v2.pdf",
            "s4g_p5": "https://arxiv.org/abs/1410.0009",
            "ml_3p6": "https://arxiv.org/abs/1402.5210",
        },
        "object_count": len(OBJECTS),
        "production_fallback_objects": 7,
        "overlap_benchmark_objects": 3,
        "file_count": len(records),
        "bytes": sum(int(record["bytes"]) for record in records),
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
