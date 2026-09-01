"""Acquire exact S4G geometry catalogs for the five direct-map RG objects."""

from __future__ import annotations

import hashlib
import json
import gzip
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / "work/private/open-gravity-rg-s4g-geometry-metadata-source-only-v1"
OUTPUT = ROOT / "work/rg-s4g-geometry-metadata-source-inventory-v1.json"
OBJECTS = ("NGC2903", "NGC2976", "NGC3198", "NGC3521", "NGC4214")
SOURCES = (
    {
        "source_id": "S4G_CATALOG_V2",
        "url": "https://cdsarc.cds.unistra.fr/ftp/J/PASP/122/1397/s4g.dat.gz",
        "paper": "https://arxiv.org/abs/1010.1592",
        "filename": "s4g.dat.gz",
        "gzip": True,
        "record_bytes": 156,
        "expected_records": 2352,
    },
    {
        "source_id": "S4G_PIPELINE4_OUTER_GEOMETRY",
        "url": "https://cdsarc.cds.unistra.fr/ftp/J/ApJS/219/4/table1.dat",
        "paper": "https://ui.adsabs.harvard.edu/abs/2015ApJS..219....4S/abstract",
        "filename": "pipeline4-table1.dat",
        "gzip": False,
        "record_bytes": 91,
        "expected_records": 2352,
    },
)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        raise RuntimeError(f"unexpected redirect {code}")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _content_sha256(value: dict[str, object]) -> str:
    clean = dict(value)
    clean.pop("content_sha256", None)
    return _sha256(json.dumps(clean, sort_keys=True, separators=(",", ":")).encode())


def _fetch(url: str) -> bytes:
    opener = urllib.request.build_opener(NoRedirect())
    request = urllib.request.Request(url, headers={"User-Agent": "Invariant-source-metadata/1"})
    with opener.open(request, timeout=60) as response:
        if response.status != 200 or response.geturl() != url:
            raise RuntimeError("unexpected metadata response")
        return response.read()


def _rows(data: bytes, expected_records: int) -> list[str]:
    text = data.decode("ascii")
    rows = text.splitlines()
    if len(rows) != expected_records:
        raise RuntimeError(f"record count changed: {len(rows)}")
    return rows


def _parse_s4g(row: str) -> dict[str, object]:
    def optional_float(value: str) -> float | None:
        return float(value) if value.strip() else None

    return {
        "object_id": row[0:10].strip(),
        "ra_deg": float(row[13:22]),
        "dec_deg": float(row[23:32]),
        "semi_major_25p5_arcsec": optional_float(row[34:39]),
        "position_angle_deg": optional_float(row[40:46]),
        "ellipticity": optional_float(row[47:52]),
        "distance_mpc": optional_float(row[119:126]),
    }


def _parse_pipeline4(row: str) -> dict[str, object]:
    return {
        "object_id": row[0:10].strip(),
        "center_x_pixel": float(row[11:18]),
        "center_y_pixel": float(row[19:26]),
        "outer_position_angle_deg": float(row[27:32]),
        "outer_position_angle_sd_deg": float(row[33:37]),
        "outer_ellipticity": float(row[38:43]),
        "outer_ellipticity_sd": float(row[44:49]),
        "outer_radius_min_arcsec": int(row[50:54]),
        "outer_radius_max_arcsec": int(row[55:59]),
        "orientation_flag": row[60:62].strip(),
        "sky_mjy_sr": float(row[63:70]),
        "sky_sd_mjy_sr": float(row[71:77]),
        "rms_mjy_sr": float(row[78:84]),
    }


def main() -> None:
    PRIVATE.mkdir(parents=True, exist_ok=True)
    source_records: list[dict[str, object]] = []
    tables: dict[str, list[str]] = {}
    for source in SOURCES:
        data = _fetch(str(source["url"]))
        path = PRIVATE / str(source["filename"])
        if path.exists() and path.read_bytes() != data:
            raise RuntimeError("existing catalog bytes changed")
        if not path.exists():
            path.write_bytes(data)
        decoded = gzip.decompress(data) if bool(source["gzip"]) else data
        rows = _rows(decoded, int(source["expected_records"]))
        if any(len(row) > int(source["record_bytes"]) for row in rows):
            raise RuntimeError("record width changed")
        tables[str(source["source_id"])] = rows
        source_records.append(
            {
                **source,
                "bytes": len(data),
                "sha256": _sha256(data),
                "decoded_bytes": len(decoded),
                "decoded_sha256": _sha256(decoded),
                "relative_path": path.relative_to(ROOT).as_posix(),
            }
        )

    catalog = {_parse_s4g(row)["object_id"]: _parse_s4g(row) for row in tables["S4G_CATALOG_V2"]}
    pipeline = {
        _parse_pipeline4(row)["object_id"]: _parse_pipeline4(row)
        for row in tables["S4G_PIPELINE4_OUTER_GEOMETRY"]
        if len(row) >= 84
    }
    objects = []
    for object_id in OBJECTS:
        if object_id not in catalog or object_id not in pipeline:
            raise RuntimeError(f"missing S4G geometry for {object_id}")
        p4 = pipeline[object_id]
        objects.append(
            {
                "object_id": object_id,
                "catalog_v2": catalog[object_id],
                "pipeline4": p4,
                "builder_orientation_disposition": (
                    "PRIMARY_OUTER_ISOPHOTE_GEOMETRY"
                    if p4["orientation_flag"] == "ok"
                    else "ORIENTATION_UNCERTAIN_REQUIRE_GEOMETRY_SENSITIVITY_CELLS"
                ),
            }
        )

    payload: dict[str, object] = {
        "schema": "invariant-work-rg-s4g-geometry-metadata-source-inventory-1.0",
        "purpose": "Exact public source geometry and distance metadata for the five direct S4G-map objects, acquired before any response-blind 3D lift.",
        "sources": source_records,
        "objects": objects,
        "object_count": len(objects),
        "response_files_opened": 0,
        "response_rows_opened": 0,
        "scores_computed": 0,
    }
    payload["content_sha256"] = _content_sha256(payload)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if OUTPUT.exists() and OUTPUT.read_text(encoding="utf-8") != encoded:
        raise RuntimeError("existing metadata inventory changed")
    if not OUTPUT.exists():
        OUTPUT.write_text(encoded, encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
