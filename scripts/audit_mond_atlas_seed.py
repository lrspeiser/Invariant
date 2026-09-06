"""Audit existing local source assets for the proposed MOND observation atlas.

This verifies file identity, not scientific suitability or 3D reconstruction.
No network access is performed. Run from any directory with the project Python.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RECEIPTS = (
    "work/gravity-first-principles/things-cube-acquisition-001/receipt.json",
    "work/gravity-first-principles/stellar-co-acquisition-001/receipt.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    rows = []
    receipt_records = []
    cached_hashes = {}
    for relative_receipt in RECEIPTS:
        receipt_path = REPO / relative_receipt
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt_records.append({
            "file": relative_receipt,
            "sha256": sha256(receipt_path),
            "declared_status": receipt.get("status"),
        })
        for item in receipt["files"]:
            path = (REPO / item["file"].replace("\\", "/")).resolve()
            if not path.is_relative_to(REPO):
                raise ValueError(f"Receipt path outside repository: {path}")
            exists = path.is_file()
            actual_bytes = path.stat().st_size if exists else None
            if exists and path not in cached_hashes:
                cached_hashes[path] = sha256(path)
            actual_hash = cached_hashes.get(path)
            expected_bytes = item.get("actual_bytes", item.get("bytes"))
            rows.append({
                "galaxy": item["name"],
                "role": item.get("role", "HI_SPECTRAL_CUBE"),
                "receipt": relative_receipt,
                "url": item["url"],
                "file": path.relative_to(REPO).as_posix(),
                "exists": exists,
                "expected_bytes": expected_bytes,
                "actual_bytes": actual_bytes,
                "expected_sha256": item["sha256"],
                "actual_sha256": actual_hash,
                "identity_verified": (
                    exists and actual_bytes == expected_bytes
                    and actual_hash == item["sha256"]
                ),
                "scientifically_validated_by_this_audit": False,
            })
    by_galaxy = {}
    for name in sorted({row["galaxy"] for row in rows}):
        items = [row for row in rows if row["galaxy"] == name]
        by_galaxy[name] = {
            "asset_entries": len(items),
            "verified_entries": sum(row["identity_verified"] for row in items),
            "roles": dict(sorted(Counter(row["role"] for row in items).items())),
            "development_exposure": "previously_analyzed",
            "complete_spatial_3d_baryon_model_demonstrated": False,
        }
    failures = sum(not row["identity_verified"] for row in rows)
    result = {
        "status": "LOCAL_ASSET_IDENTITY_VERIFIED" if failures == 0 else "LOCAL_ASSET_IDENTITY_FAILURES",
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Only the two named acquisition receipts, not an exhaustive machine inventory.",
        "receipts": receipt_records,
        "asset_entries": len(rows),
        "unique_files": len({row["file"] for row in rows}),
        "galaxies": len(by_galaxy),
        "verified_entries": len(rows) - failures,
        "failed_entries": failures,
        "unique_existing_file_bytes": sum(path.stat().st_size for path in cached_hashes),
        "new_observational_downloads": 0,
        "new_mond_predictions": 0,
        "new_spatial_3d_reconstructions": 0,
        "limitations": [
            "Checksums establish file identity, not correct astrometry, masks, conversion factors or scientific completeness.",
            "Existing source footprints do not guarantee co-spatial coverage of stars, HI and molecular gas.",
            "A spectral cube's third coordinate is frequency or velocity, not spatial depth.",
            "All 12 seed galaxies were previously exposed in development.",
        ],
        "by_galaxy": by_galaxy,
        "assets": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in (
        "status", "asset_entries", "unique_files", "galaxies",
        "verified_entries", "failed_entries", "unique_existing_file_bytes",
        "new_observational_downloads", "new_mond_predictions",
    )}, indent=2))
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
