"""Build response-free FITS metadata for the acquired 12-galaxy source maps."""

from __future__ import annotations

import gzip
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from astropy.io import fits

from sigma_theory_compiler import (
    open_gravity_refracted_gravity_things_heracles_sparc_3d_expansion_preflight_v1 as preflight,
)


ROOT = Path(__file__).resolve().parents[1]
DOWNLOAD_INVENTORY = ROOT / "work/rg-12gal-source-download-inventory-v1.json"
OUTPUT = ROOT / "work/rg-12gal-source-fits-inventory-v1.json"
BEAM_PATTERN = re.compile(
    r"CLEAN BMAJ=\s*([0-9.+\-Ee]+) BMIN=\s*([0-9.+\-Ee]+) BPA=\s*([0-9.+\-Ee]+)"
)


def sha256_bytes(path: Path) -> str:
    digest = hashlib.sha256()
    opener = gzip.open if path.suffix == ".gz" else Path.open
    if path.suffix == ".gz":
        handle = opener(path, "rb")
    else:
        handle = opener(path, "rb")
    with handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def beam_from_header(header: fits.Header) -> tuple[str | None, list[float] | None]:
    if "BMAJ" in header and "BMIN" in header:
        return "HEADER", [float(header["BMAJ"]), float(header["BMIN"]), float(header.get("BPA", 0.0))]
    history = header.get("HISTORY", [])
    if isinstance(history, str):
        history = [history]
    matches = [match for line in history if (match := BEAM_PATTERN.search(str(line)))]
    if len(matches) == 1:
        return "AIPS_HISTORY", [float(value) for value in matches[0].groups()]
    return None, None


def main() -> None:
    download = json.loads(DOWNLOAD_INVENTORY.read_text(encoding="utf-8"))
    rows = []
    for source in download["records"]:
        path = ROOT / source["relative_path"]
        with fits.open(path, memmap=False, do_not_scale_image_data=False) as hdus:
            if len(hdus) != 1 or hdus[0].data is None:
                raise RuntimeError(f"unexpected HDU inventory: {path.name}")
            array = np.asarray(hdus[0].data)
            header = hdus[0].header
            beam_source, beam = beam_from_header(header)
            row = {
                **source,
                "hdu_count": len(hdus),
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "bunit": str(header.get("BUNIT", "")),
                "ctype1": str(header.get("CTYPE1", "")),
                "ctype2": str(header.get("CTYPE2", "")),
                "rest_hz": header.get("RESTFRQ", header.get("RESTFREQ")),
                "beam_source": beam_source,
                "beam_deg": beam,
                "total_pixels": int(array.size),
                "finite_pixels": int(np.isfinite(array).sum()),
                "decompressed_sha256": sha256_bytes(path),
            }
            row["row_sha256"] = preflight.content_sha256(row)
            rows.append(row)
    payload = {
        "schema": "invariant-work-rg-12gal-source-fits-inventory-1.0",
        "file_count": len(rows),
        "bytes": sum(row["bytes"] for row in rows),
        "total_pixels": sum(row["total_pixels"] for row in rows),
        "finite_pixels": sum(row["finite_pixels"] for row in rows),
        "ordered_file_sha_root": hashlib.sha256(
            "\n".join(row["sha256"] for row in rows).encode("ascii")
        ).hexdigest(),
        "ordered_decompressed_sha_root": hashlib.sha256(
            "\n".join(row["decompressed_sha256"] for row in rows).encode("ascii")
        ).hexdigest(),
        "ordered_record_root": preflight.content_sha256(rows),
        "records": rows,
    }
    payload["content_sha256"] = preflight.content_sha256(payload)
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {key: payload[key] for key in payload if key not in {"records"}}, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
