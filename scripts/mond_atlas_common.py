"""Small, offline primitives shared by the observational MOND atlas."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/mond_atlas_v1.json"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_text_digest(path, expected):
    raw = Path(path).read_bytes()
    raw_hash = hashlib.sha256(raw).hexdigest()
    if raw_hash == expected:
        return "exact_bytes"
    if hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest() == expected:
        return "CRLF_to_LF_equivalent_only"
    raise ValueError("Source hash mismatch: " + str(path))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                          encoding="utf-8", newline="\n")


def write_csv(path, rows, fields=None):
    rows = list(rows)
    fields = fields or list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def number(value):
    try:
        v = float(value)
        return v if math.isfinite(v) and v > -900 else None
    except (ValueError, TypeError):
        return None


def canonical_name(name):
    name = str(name).strip().upper()
    alias = {"HOII": "UGC4305", "HO_II": "UGC4305", "HOLMBERGII": "UGC4305"}
    if name in alias:
        return alias[name]
    match = re.fullmatch(r"(NGC|IC|UGC|DDO)[ _]*0*(\d+)", name)
    return match[1] + str(int(match[2])) if match else name


def fits_primary_header(path):
    """Read plain/gzip FITS primary header for inventory only, never infer WCS.

    Supports standard scalar cards. Fail on a missing END rather than reading data.
    Image arrays, projection transforms and FITS checksums are outside this reader.
    """
    result = {}
    with Path(path).open("rb") as check:
        compressed = check.read(2) == b"\x1f\x8b"
    opener = gzip.open if compressed else open
    with opener(path, "rb") as stream:
        for _ in range(256):
            block = stream.read(2880)
            if len(block) != 2880:
                raise ValueError("truncated FITS header")
            for offset in range(0, 2880, 80):
                card = block[offset:offset + 80].decode("ascii")
                key = card[:8].strip()
                if key == "END":
                    if result.get("SIMPLE") is not True:
                        raise ValueError("not a FITS primary header")
                    return result
                if card[8:10] != "= ":
                    continue
                raw = card[10:].strip()
                if raw.startswith("'"):
                    # FITS doubled quotes are escaped within string values.
                    match = re.match(r"'((?:[^']|'')*)'", raw)
                    if match:
                        result[key] = match[1].replace("''", "'").strip()
                else:
                    raw = raw.split("/")[0].strip()
                    if raw in ("T", "F"):
                        result[key] = raw == "T"
                    else:
                        try:
                            result[key] = float(raw.replace("D", "E"))
                        except ValueError:
                            result[key] = raw
    raise ValueError("FITS header exceeds inventory limit")


def sparc_inputs():
    import io
    import zipfile
    curves_path = ROOT / "configs/sparc_rotation_curves_full_v1.json"
    archive_path = ROOT / "work/private/matched-concentration-001/Rotmod_LTG.zip"
    meta_path = ROOT / "work/gravity-first-principles/map-response-metadata-001/SPARC_Lelli2016c.mrt"
    curves = read_json(curves_path)["galaxies"]
    names = {g["name"] for g in curves}
    metadata = {}
    for line in meta_path.read_text(encoding="utf-8").splitlines():
        f = line.split()
        if f and f[0] in names:
            if len(f) != 19:
                raise ValueError("SPARC metadata schema changed")
            metadata[f[0]] = dict(hubble_type=int(f[1]), distance_mpc=float(f[2]),
                distance_error_mpc=float(f[3]), distance_method=int(f[4]),
                inclination_deg=float(f[5]), inclination_error_deg=float(f[6]),
                luminosity_1e9_lsun=float(f[7]), reff_kpc=float(f[9]),
                effective_sb_lsun_pc2=float(f[10]), rdisk_kpc=float(f[11]),
                central_sb_lsun_pc2=float(f[12]), hi_mass_1e9_msun=float(f[13]),
                hi_radius_kpc=float(f[14]), quality=int(f[17]), motion_references=f[18])
    if len(metadata) != len(curves):
        raise ValueError("missing SPARC metadata")
    photometry = {}
    with zipfile.ZipFile(archive_path) as archive:
        lookup = {Path(n).name: n for n in archive.namelist()}
        for g in curves:
            raw = archive.read(lookup[g["name"] + "_rotmod.dat"])
            if hashlib.sha256(raw).hexdigest() != g["provenance"]["source_file_sha256"]:
                raise ValueError("SPARC archive member hash mismatch: " + g["name"])
            rows = [line.split() for line in io.StringIO(raw.decode("utf-8"))
                    if line.strip() and not line.startswith("#")]
            if [row[:6] for row in rows] != g["rows"]:
                raise ValueError("SPARC archive/table disagreement: " + g["name"])
            photometry[g["name"]] = [[float(v) for v in row[6:8]] for row in rows]
    return curves, metadata, photometry, [curves_path, archive_path, meta_path]
