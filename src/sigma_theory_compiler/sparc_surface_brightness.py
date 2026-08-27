"""Build and validate an exploration-only SPARC surface-brightness supplement."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from zipfile import ZipFile

from .sigma_core import canonical_json_bytes, canonical_sha256
from .sparc_full_sample import DATASET_PATH, assemble

SCHEMA = "invariant-sparc-surface-brightness-exploration-1.0"
OUTPUT_PATH = "configs/sparc_surface_brightness_exploration_v1.json"
EXPECTED_ARCHIVE_SHA256 = "5e131528f5906e52f305f85d89ca7ea34b21d45fca0e7167d26ddc0c3e893227"
EXPECTED_ARCHIVE_BYTES = 119_649
DECIMAL = re.compile(r"^-?(?:0|[1-9][0-9]*)\.[0-9]+$")
HEADER = (
    "# Rad\tVobs\terrV\tVgas\tVdisk\tVbul\tSBdisk\tSBbul\t\t",
    "# kpc\tkm/s\tkm/s\tkm/s\tkm/s\tkm/s\tL/pc^2\tL/pc^2",
)


class SparcSurfaceBrightnessError(ValueError):
    """The source archive, derived supplement, or population binding changed."""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows_sha256(rows: Sequence[Sequence[str]]) -> str:
    return hashlib.sha256(canonical_json_bytes(rows)).hexdigest()


def build_asset(root: Path, archive: Path) -> dict[str, Any]:
    """Cross-check source rows and retain only target-blind exploration photometry."""

    root = root.resolve()
    archive = archive.resolve()
    if archive.stat().st_size != EXPECTED_ARCHIVE_BYTES:
        raise SparcSurfaceBrightnessError("SPARC archive byte count changed")
    if _file_sha256(archive) != EXPECTED_ARCHIVE_SHA256:
        raise SparcSurfaceBrightnessError("SPARC archive SHA-256 changed")
    original = json.loads((root / DATASET_PATH).read_text(encoding="utf-8"))
    original_by_name = {row["name"]: row for row in original["galaxies"]}
    population = assemble(root)
    exploration_names = [galaxy.name for galaxy in population.exploration]
    galaxies = []
    with ZipFile(archive) as zipped:
        members = [name for name in zipped.namelist() if name.endswith("_rotmod.dat")]
        if len(members) != 175:
            raise SparcSurfaceBrightnessError("SPARC archive member count changed")
        for name in exploration_names:
            member = f"sparc_database/{name}_rotmod.dat"
            info = zipped.getinfo(member)
            payload = zipped.read(member)
            lines = payload.decode("utf-8").splitlines()
            expected_distance = str(original_by_name[name]["distance_mpc"])
            if lines[:3] != [f"# Distance = {expected_distance} Mpc", *HEADER]:
                raise SparcSurfaceBrightnessError(f"SPARC header changed for {name}")
            fields = [line.split() for line in lines[3:] if line.strip()]
            if any(len(row) != 8 or any(DECIMAL.fullmatch(value) is None for value in row) for row in fields):
                raise SparcSurfaceBrightnessError(f"SPARC row grammar changed for {name}")
            original_rows = original_by_name[name]["rows"]
            if [row[:6] for row in fields] != original_rows:
                raise SparcSurfaceBrightnessError(
                    f"SPARC kinematic cross-check failed for {name}"
                )
            photometry = [row[6:] for row in fields]
            galaxies.append(
                {
                    "galaxy": name,
                    "point_count": len(photometry),
                    "rows": photometry,
                    "rows_sha256": _rows_sha256(photometry),
                    "source_member": member,
                    "source_member_crc32": format(info.CRC, "08x"),
                    "source_member_sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
    body: dict[str, Any] = {
        "schema_version": SCHEMA,
        "claim": "Published SPARC surface-brightness inputs for admitted exploration galaxies only; no velocity target is retained in this supplement.",
        "columns": [
            {"id": "SBdisk", "unit": "L_sun/pc^2"},
            {"id": "SBbul", "unit": "L_sun/pc^2"},
        ],
        "counts": {
            "confirmation_galaxies": 0,
            "confirmation_rows": 0,
            "exploration_galaxies": len(galaxies),
            "exploration_rows": sum(row["point_count"] for row in galaxies),
        },
        "galaxies": sorted(galaxies, key=lambda row: row["galaxy"]),
        "provenance": {
            "archive_basename": archive.name,
            "archive_bytes": EXPECTED_ARCHIVE_BYTES,
            "archive_sha256": EXPECTED_ARCHIVE_SHA256,
            "crosscheck": "All six pre-existing published columns and the distance header equal the authenticated full-sample asset before the two photometry columns are retained.",
            "dataset": "SPARC mass models, VizieR catalogue J/AJ/152/157",
            "original_asset": DATASET_PATH,
            "source": "Lelli, McGaugh and Schombert, AJ 152, 157 (2016)",
            "source_url": "https://cdsarc.cds.unistra.fr/viz-bin/cat/J/AJ/152/157",
        },
    }
    body["content_sha256"] = canonical_sha256(body)
    return body


def validate_asset(asset: Mapping[str, Any], *, root: Path) -> None:
    """Validate content, population, row counts, and target-blind scope."""

    root = root.resolve()
    if asset.get("schema_version") != SCHEMA:
        raise SparcSurfaceBrightnessError("surface-brightness schema changed")
    body = dict(asset)
    supplied = body.pop("content_sha256", None)
    if supplied != canonical_sha256(body):
        raise SparcSurfaceBrightnessError("surface-brightness content seal changed")
    counts = asset.get("counts", {})
    if counts.get("confirmation_galaxies") != 0 or counts.get("confirmation_rows") != 0:
        raise SparcSurfaceBrightnessError("surface-brightness asset contains confirmation data")
    population = assemble(root)
    expected = {galaxy.name: galaxy.count for galaxy in population.exploration}
    rows = asset.get("galaxies", [])
    actual = {row["galaxy"]: row["point_count"] for row in rows}
    if actual != expected:
        raise SparcSurfaceBrightnessError("surface-brightness population changed")
    if counts.get("exploration_galaxies") != len(expected) or counts.get(
        "exploration_rows"
    ) != sum(expected.values()):
        raise SparcSurfaceBrightnessError("surface-brightness counts changed")
    for row in rows:
        values = row.get("rows", [])
        if len(values) != row["point_count"] or _rows_sha256(values) != row.get(
            "rows_sha256"
        ):
            raise SparcSurfaceBrightnessError("surface-brightness galaxy seal changed")
        if any(
            len(pair) != 2 or any(DECIMAL.fullmatch(str(value)) is None for value in pair)
            for pair in values
        ):
            raise SparcSurfaceBrightnessError("surface-brightness row changed")


def load_asset(root: Path) -> Mapping[str, Any]:
    asset = json.loads((root.resolve() / OUTPUT_PATH).read_text(encoding="utf-8"))
    validate_asset(asset, root=root)
    return asset


def _write_immutable(path: Path, value: Mapping[str, Any]) -> None:
    payload = canonical_json_bytes(value)
    if path.exists():
        if path.read_bytes() != payload:
            raise SparcSurfaceBrightnessError(
                f"refusing to overwrite immutable surface-brightness asset: {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--validate-checked", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.validate_checked:
        load_asset(root)
        return 0
    if args.archive is None:
        parser.error("--archive is required when building the asset")
    asset = build_asset(root, args.archive)
    _write_immutable(root / OUTPUT_PATH, asset)
    print(
        json.dumps(
            {
                "content_sha256": asset["content_sha256"],
                "exploration_galaxies": asset["counts"]["exploration_galaxies"],
                "exploration_rows": asset["counts"]["exploration_rows"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SparcSurfaceBrightnessError",
    "build_asset",
    "load_asset",
    "validate_asset",
]
