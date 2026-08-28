"""Zero-tuning PROBES-II replay of the Item 9 interior/exterior lead."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import unicodedata
import urllib.request
import zipfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from sigma_theory_compiler import gravity_item9_interior_exterior as attempt1

CONFIG_PATH = Path("configs/gravity_item9_probes2_zero_tuning_replay_v1.json")
SCIENTIFIC_FREEZE_COMMIT = "036dda3e7a2fe5fc3b65517ad75d129915fbde1e"
SAMPLE_FREEZE_COMMIT = "5423dfa0305bc9c146eb0a9d1db0eeaa75e5ffe3"


class GravityItem9Probes2ReplayError(RuntimeError):
    """Raised when the frozen PROBES-II replay boundary is crossed or drifts."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metric(value: float) -> str:
    return f"{float(value):.12e}"


def _content_hashed(value: dict[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result.pop("content_sha256", None)
    result["content_sha256"] = canonical_sha256(result)
    return result


def _validate_content_hash(value: Mapping[str, Any], label: str) -> None:
    copy_value = dict(value)
    digest = copy_value.pop("content_sha256", None)
    if digest != canonical_sha256(copy_value):
        raise GravityItem9Probes2ReplayError(f"{label} content hash changed")


def _load_bound_json(root: Path, binding: Mapping[str, Any], label: str) -> dict[str, Any]:
    path = root / str(binding["path"])
    if _sha256_file(path) != str(binding["file_sha256"]):
        raise GravityItem9Probes2ReplayError(f"{label} file hash changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("content_sha256") != binding.get("content_sha256"):
        raise GravityItem9Probes2ReplayError(f"{label} content binding changed")
    _validate_content_hash(value, label)
    return value


def load_config(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config = json.loads((root / CONFIG_PATH).read_text(encoding="utf-8"))
    roadmap = config["roadmap_binding"]
    if _sha256_file(root / roadmap["path"]) != roadmap["file_sha256"]:
        raise GravityItem9Probes2ReplayError("stable gravity roadmap changed")
    predecessor = _load_bound_json(root, config["predecessor"], "attempt-1 receipt")
    if predecessor["decision"] != config["predecessor"]["required_decision"]:
        raise GravityItem9Probes2ReplayError("attempt-1 decision changed")
    if predecessor["gate_counts"] != config["predecessor"]["required_gate_counts"]:
        raise GravityItem9Probes2ReplayError("attempt-1 gate counts changed")
    candidate_manifest = _load_bound_json(
        root, config["attempt1_candidate_manifest"], "attempt-1 candidate manifest"
    )
    probes1 = _load_bound_json(
        root, config["exclusion_sources"]["probes1_sample"], "PROBES-I sample"
    )
    sparc = _load_bound_json(
        root, config["exclusion_sources"]["sparc_prior"], "SPARC predecessor"
    )
    del probes1, sparc
    by_id = {row["candidate_id"]: row for row in candidate_manifest["cells"]}
    frozen_ids = [row["candidate_id"] for row in config["frozen_cells"]]
    selected_ids = [
        row["selected_candidate_id"]
        for row in predecessor["primary"]["qualifying_selector"]["folds"]
    ]
    selected_ids.append(
        predecessor["primary"]["exact_prior_focusing_cell_replay"]["candidate"][
            "candidate_id"
        ]
    )
    if frozen_ids != selected_ids or len(set(frozen_ids)) != 6:
        raise GravityItem9Probes2ReplayError("frozen replay cells changed")
    for frozen in config["frozen_cells"]:
        original = by_id.get(frozen["candidate_id"])
        if original is None:
            raise GravityItem9Probes2ReplayError("frozen cell absent from attempt 1")
        for field in ("operator_id", "condition", "equivalence_class"):
            if str(frozen[field]) != str(original[field]):
                raise GravityItem9Probes2ReplayError(f"frozen {field} changed")
        for field in ("intercept", "slope", "alpha_max"):
            if float(frozen[field]) != float(original[field]):
                raise GravityItem9Probes2ReplayError(f"frozen {field} changed")
    if any(bool(value) for value in config["claim_boundaries"].values()):
        raise GravityItem9Probes2ReplayError("configuration contains an overclaim")
    return config


def normalize_identity(value: str) -> str:
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return re.sub(r"[^A-Z0-9]", "", ascii_value.upper())


def _exclusion_identities(root: Path, config: Mapping[str, Any]) -> dict[str, set[str]]:
    probes1 = _load_bound_json(
        root, config["exclusion_sources"]["probes1_sample"], "PROBES-I sample"
    )
    sparc = _load_bound_json(
        root, config["exclusion_sources"]["sparc_prior"], "SPARC predecessor"
    )
    probes_ids = {normalize_identity(str(row["name"])) for row in probes1["objects"]}
    sparc_ids = {normalize_identity(str(row["galaxy"])) for row in sparc["galaxies"]}
    return {"probes1": probes_ids, "sparc": sparc_ids, "union": probes_ids | sparc_ids}


def build_candidate_manifest(root: Path) -> dict[str, Any]:
    config = load_config(root)
    cells = []
    for ordinal, frozen in enumerate(config["frozen_cells"]):
        cells.append(
            {
                "ordinal": ordinal,
                "role": frozen["role"],
                "candidate_id": frozen["candidate_id"],
                "operator_id": frozen["operator_id"],
                "condition": frozen["condition"],
                "intercept": _metric(float(frozen["intercept"])),
                "slope": _metric(float(frozen["slope"])),
                "alpha_max": _metric(float(frozen["alpha_max"])),
                "equivalence_class": frozen["equivalence_class"],
                "authoritative_origin_status": frozen["origin_status"],
                "exact_prior_focusing_cell": frozen["role"] == "exact_prior_sparc",
                "historical_novelty_claimed": False,
                "selected_from_probes2_response": False,
            }
        )
    operators = [
        {
            "operator_id": "focus:acceleration:q1:ell1:interior_minus_exterior_occupancy",
            "source": "acceleration",
            "threshold": _metric(1.0),
            "log_radius_scale": _metric(1.0),
            "mode": "interior_minus_exterior_occupancy",
        },
        {
            "operator_id": "focus:surface_brightness:q100:ell0p25:interior_occupancy_times_exterior_vacuum",
            "source": "surface_brightness",
            "threshold": _metric(100.0),
            "log_radius_scale": _metric(0.25),
            "mode": "interior_occupancy_times_exterior_vacuum",
        },
    ]
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item9-probes2-candidates-1.0",
            "scientific_freeze_commit": SCIENTIFIC_FREEZE_COMMIT,
            "attempt1_receipt_sha256": config["predecessor"]["file_sha256"],
            "cells": cells,
            "operators": operators,
            "ensembles": [
                {
                    "ensemble_id": "attempt1-five-cell-log-speed-median",
                    "role": "primary",
                    "members": [row["candidate_id"] for row in cells[:5]],
                    "rule": "pointwise median log10 predicted speed",
                    "weights_fitted_on_probes2": False,
                },
                {
                    "ensemble_id": "attempt1-five-cell-log-speed-mean",
                    "role": "secondary_sensitivity",
                    "members": [row["candidate_id"] for row in cells[:5]],
                    "rule": "pointwise arithmetic mean log10 predicted speed",
                    "weights_fitted_on_probes2": False,
                },
            ],
            "counts": {
                "atomic_formula_cells": 6,
                "ensemble_formula_cells": 2,
                "total_evaluated_formulas": 8,
                "candidate_selection_calls": 0,
                "post_response_formula_cells": 0,
                "response_rows_read": 0,
                "paid_model_calls": 0,
            },
            "claims": {"historical_novelty_established": False},
        }
    )


def validate_candidate_manifest(manifest: Mapping[str, Any], root: Path) -> None:
    _validate_content_hash(manifest, "PROBES-II candidate manifest")
    expected = build_candidate_manifest(root)
    if manifest != expected:
        raise GravityItem9Probes2ReplayError("PROBES-II candidate manifest drifted")
    counts = manifest["counts"]
    if counts["candidate_selection_calls"] != 0 or counts["post_response_formula_cells"] != 0:
        raise GravityItem9Probes2ReplayError("PROBES-II formula boundary crossed")


def write_candidate_manifest(root: Path) -> Path:
    root = root.resolve()
    config = load_config(root)
    path = root / config["outputs"]["candidate_manifest"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(build_candidate_manifest(root)) + b"\n")
    return path


def _download_exact(url: str, path: Path, expected_bytes: int, expected_etag: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "Invariant/Item9-PROBES2"})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read()
        etag = str(response.headers.get("ETag", ""))
        modified = str(response.headers.get("Last-Modified", ""))
    if len(payload) != expected_bytes:
        raise GravityItem9Probes2ReplayError(f"source byte count changed for {url}")
    if etag.removeprefix("W/") != expected_etag.removeprefix("W/"):
        raise GravityItem9Probes2ReplayError(f"source ETag changed for {url}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(payload)
    temporary.replace(path)
    return {
        "url": url,
        "bytes": len(payload),
        "etag": etag,
        "last_modified": modified,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _metadata_records(path: Path, config: Mapping[str, Any]) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise GravityItem9Probes2ReplayError("empty PROBES-II metadata") from exc
        allowlist = list(config["metadata_allowlist"])
        if any(name not in header for name in allowlist):
            raise GravityItem9Probes2ReplayError("PROBES-II metadata allowlist column missing")
        if any(name not in header for name in config["metadata_forbidden_columns"]):
            raise GravityItem9Probes2ReplayError("PROBES-II forbidden-column schema changed")
        indices = [header.index(name) for name in allowlist]
        records = []
        for row in reader:
            if not row or all(not field.strip() for field in row):
                continue
            if len(row) > len(header) or len(row) <= max(indices):
                raise GravityItem9Probes2ReplayError(
                    "PROBES-II metadata row does not preserve the allowlisted prefix"
                )
            records.append({name: row[index].strip() for name, index in zip(allowlist, indices)})
    return header, records


def _zip_inventory(path: Path) -> list[dict[str, Any]]:
    with zipfile.ZipFile(path) as archive:
        return [
            {
                "path": info.filename,
                "bytes": int(info.file_size),
                "compressed_bytes": int(info.compress_size),
                "crc32": f"{int(info.CRC):08x}",
                "is_directory": info.is_dir(),
            }
            for info in archive.infolist()
        ]


def write_source_manifest(root: Path) -> Path:
    root = root.resolve()
    if SCIENTIFIC_FREEZE_COMMIT.startswith("PENDING_"):
        raise GravityItem9Probes2ReplayError("scientific freeze is not bound")
    config = load_config(root)
    revision = config["source"]["observed_revision"]
    metadata_path = root / config["work"]["metadata"]
    archive_path = root / config["work"]["archive"]
    metadata_receipt = _download_exact(
        config["source"]["metadata_url"],
        metadata_path,
        int(revision["metadata_bytes"]),
        str(revision["metadata_etag"]),
    )
    archive_receipt = _download_exact(
        config["source"]["archive_url"],
        archive_path,
        int(revision["archive_bytes"]),
        str(revision["archive_etag"]),
    )
    header, metadata = _metadata_records(metadata_path, config)
    inventory = _zip_inventory(archive_path)
    manifest = _content_hashed(
        {
            "schema_version": "invariant-gravity-item9-probes2-source-1.0",
            "scientific_freeze_commit": SCIENTIFIC_FREEZE_COMMIT,
            "source": {"metadata": metadata_receipt, "archive": archive_receipt},
            "metadata": {
                "rows": len(metadata),
                "header": header,
                "allowlist": config["metadata_allowlist"],
                "retained_records": metadata,
                "forbidden_column_values_retained": 0,
            },
            "archive_inventory": inventory,
            "counts": {
                "archive_entries": len(inventory),
                "archive_file_entries": sum(not row["is_directory"] for row in inventory),
                "archive_entry_payloads_opened": 0,
                "rotation_curve_entry_payloads_opened": 0,
                "rotation_curve_rows_read": 0,
                "photometry_entry_payloads_opened": 0,
                "derived_mass_values_read": 0,
                "paid_model_calls": 0,
            },
            "claims": {"response_opened_before_sample_freeze": False},
        }
    )
    path = root / config["outputs"]["source_manifest"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(manifest) + b"\n")
    return path


def validate_source_manifest(manifest: Mapping[str, Any], root: Path) -> None:
    _validate_content_hash(manifest, "PROBES-II source manifest")
    config = load_config(root)
    if manifest["scientific_freeze_commit"] != SCIENTIFIC_FREEZE_COMMIT:
        raise GravityItem9Probes2ReplayError("source freeze binding changed")
    for key in ("metadata", "archive"):
        path = root / config["work"][key]
        if _sha256_file(path) != manifest["source"][key]["sha256"]:
            raise GravityItem9Probes2ReplayError(f"PROBES-II {key} file changed")
    forbidden = (
        "archive_entry_payloads_opened",
        "rotation_curve_entry_payloads_opened",
        "rotation_curve_rows_read",
        "photometry_entry_payloads_opened",
        "derived_mass_values_read",
        "paid_model_calls",
    )
    if any(int(manifest["counts"][key]) != 0 for key in forbidden):
        raise GravityItem9Probes2ReplayError("source manifest crossed response boundary")


def _classify_entry(path: str) -> str | None:
    lowered = path.lower().replace("\\", "/")
    if lowered.endswith("/") or not lowered.endswith(".csv"):
        return None
    if "rotationcurves" in lowered or "rotation_curves" in lowered:
        return "rotation"
    if "lightprofiles" in lowered or "light_profiles" in lowered:
        return "light"
    return None


def _looks_like_r_profile(path: str) -> bool:
    parts = [part.lower() for part in PurePosixPath(path.replace("\\", "/")).parts]
    stem_tokens = re.split(r"[^a-z0-9]+", PurePosixPath(path).stem.lower())
    tokens = set(parts + stem_tokens)
    return bool(tokens & {"r", "desir", "sdssr", "rband", "bandr"})


def _source_family(path: str, name: str) -> str:
    parts = list(PurePosixPath(path.replace("\\", "/")).parts)
    lowered = [part.lower() for part in parts]
    for marker in ("rotationcurves", "rotation_curves"):
        if marker in lowered:
            index = lowered.index(marker)
            if index + 2 < len(parts):
                return parts[index + 1]
    stem = PurePosixPath(path).stem
    exact_prefix = f"RC_{name}_"
    if stem.lower().startswith(exact_prefix.lower()):
        suffix = stem[len(exact_prefix) :]
        return suffix or "UNKNOWN"
    tokens = stem.split("_", 2)
    return tokens[2] if len(tokens) == 3 and tokens[2] else "UNKNOWN"


def _entry_assignments(
    paths: Sequence[str], identities: Sequence[str], entry_prefix: str
) -> dict[str, list[str]]:
    assignments = {identity: [] for identity in identities}
    longest_first = sorted(set(identities), key=lambda value: (-len(value), value))
    normalized_prefix = normalize_identity(entry_prefix)
    for path in paths:
        stem = normalize_identity(PurePosixPath(path).stem)
        for identity in longest_first:
            if stem.startswith(normalized_prefix + identity):
                assignments[identity].append(path)
                break
    return assignments


def _finite_positive(value: str) -> bool:
    try:
        return math.isfinite(float(value)) and float(value) > 0
    except ValueError:
        return False


def _fold(identity: str, config: Mapping[str, Any]) -> int:
    token = f"{config['evaluation']['fold_salt']}|{identity}".encode()
    return int(hashlib.sha256(token).hexdigest()[:16], 16) % int(config["evaluation"]["folds"])


def build_sample_manifest(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config = load_config(root)
    source_path = root / config["outputs"]["source_manifest"]
    source = json.loads(source_path.read_text(encoding="utf-8"))
    validate_source_manifest(source, root)
    exclusions = _exclusion_identities(root, config)
    inventory = [row for row in source["archive_inventory"] if not row["is_directory"]]
    rotation = sorted(row["path"] for row in inventory if _classify_entry(row["path"]) == "rotation")
    light = sorted(
        row["path"]
        for row in inventory
        if _classify_entry(row["path"]) == "light" and _looks_like_r_profile(row["path"])
    )
    identities = [
        normalize_identity(str(row["Name"]))
        for row in source["metadata"]["retained_records"]
    ]
    rotation_by_identity = _entry_assignments(rotation, identities, "RC")
    light_by_identity = _entry_assignments(light, identities, "Light")
    objects = []
    exclusion_counts: Counter[str] = Counter()
    for row in source["metadata"]["retained_records"]:
        name = str(row["Name"])
        identity = normalize_identity(name)
        reasons = []
        if not identity:
            reasons.append("empty_normalized_identity")
        if identity in exclusions["probes1"]:
            reasons.append("PROBES_I_manifest_overlap")
        if identity in exclusions["sparc"]:
            reasons.append("SPARC_predecessor_overlap")
        if not _finite_positive(str(row["D (Mpc)"])):
            reasons.append("invalid_distance")
        if not _finite_positive(str(row["q (b/a)"])):
            reasons.append("invalid_axis_ratio")
        rotation_matches = rotation_by_identity.get(identity, [])
        light_matches = light_by_identity.get(identity, [])
        if not rotation_matches:
            reasons.append("no_rotation_entry")
        if not light_matches:
            reasons.append("no_r_light_entry")
        for reason in set(reasons):
            exclusion_counts[reason] += 1
        if reasons:
            continue
        q = float(row["q (b/a)"])
        q0 = float(config["constants"]["intrinsic_disk_axis_ratio"])
        cos2 = float(np.clip((q * q - q0 * q0) / (1.0 - q0 * q0), 0.0, 1.0))
        inclination = math.degrees(math.asin(math.sqrt(max(0.0, 1.0 - cos2))))
        objects.append(
            {
                "name": name,
                "normalized_identity": identity,
                "distance_mpc": _metric(float(row["D (Mpc)"])),
                "distance_error_mpc": _metric(float(row["D_err (Mpc)"]))
                if _finite_positive(str(row["D_err (Mpc)"]))
                else None,
                "axis_ratio_q": _metric(q),
                "derived_inclination_degrees": _metric(inclination),
                "published_inclination_degrees": _metric(float(row["Incl (deg)"]))
                if _finite_positive(str(row["Incl (deg)"]))
                else None,
                "rotation_entry": rotation_matches[0],
                "r_light_entry": light_matches[0],
                "unselected_duplicate_rotation_entries": rotation_matches[1:],
                "unselected_duplicate_r_light_entries": light_matches[1:],
                "source_family": _source_family(rotation_matches[0], name),
                "outer_fold": _fold(identity, config),
                "role": "zero_tuning_independent_evaluation",
                "response_entry_opened": False,
            }
        )
    objects.sort(key=lambda row: (row["normalized_identity"], row["rotation_entry"]))
    identities = [row["normalized_identity"] for row in objects]
    if len(identities) != len(set(identities)):
        raise GravityItem9Probes2ReplayError("sample contains duplicate physical identities")
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item9-probes2-sample-1.0",
            "scientific_freeze_commit": SCIENTIFIC_FREEZE_COMMIT,
            "source_manifest_sha256": _sha256_file(source_path),
            "source_manifest_content_sha256": source["content_sha256"],
            "objects": objects,
            "counts": {
                "metadata_rows": len(source["metadata"]["retained_records"]),
                "rotation_entries_in_inventory": len(rotation),
                "r_light_entries_in_inventory": len(light),
                "selected_physical_identities": len(objects),
                "selected_rotation_entries": len(objects),
                "selected_r_light_entries": len(objects),
                "unselected_duplicate_rotation_entries": sum(
                    len(row["unselected_duplicate_rotation_entries"]) for row in objects
                ),
                "unselected_duplicate_r_light_entries": sum(
                    len(row["unselected_duplicate_r_light_entries"]) for row in objects
                ),
                "PROBES_I_overlap_selected": 0,
                "SPARC_overlap_selected": 0,
                "rotation_curve_entry_payloads_opened": 0,
                "rotation_curve_rows_read": 0,
            },
            "exclusion_counts": dict(sorted(exclusion_counts.items())),
            "fold_counts": {
                str(key): value
                for key, value in sorted(Counter(row["outer_fold"] for row in objects).items())
            },
            "source_family_counts": dict(
                sorted(Counter(row["source_family"] for row in objects).items())
            ),
            "claims": {"response_opened_before_sample_freeze": False},
        }
    )


def validate_sample_manifest(manifest: Mapping[str, Any], root: Path) -> None:
    _validate_content_hash(manifest, "PROBES-II sample manifest")
    if manifest != build_sample_manifest(root):
        raise GravityItem9Probes2ReplayError("PROBES-II sample manifest drifted")
    counts = manifest["counts"]
    if counts["PROBES_I_overlap_selected"] != 0 or counts["SPARC_overlap_selected"] != 0:
        raise GravityItem9Probes2ReplayError("predecessor identity overlap entered sample")
    if counts["rotation_curve_entry_payloads_opened"] != 0:
        raise GravityItem9Probes2ReplayError("rotation responses opened before sample freeze")


def write_sample_manifest(root: Path) -> Path:
    root = root.resolve()
    config = load_config(root)
    manifest = build_sample_manifest(root)
    path = root / config["outputs"]["sample_manifest"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(manifest) + b"\n")
    return path


def _rows_from_csv_payload(payload: bytes) -> tuple[list[str], list[list[str]]]:
    text = payload.decode("utf-8-sig", errors="strict")
    rows = []
    for row in csv.reader(io.StringIO(text)):
        if not row or all(not field.strip() for field in row):
            continue
        if row[0].lstrip().startswith("#"):
            continue
        rows.append([field.strip() for field in row])
    if len(rows) < 2:
        raise GravityItem9Probes2ReplayError("profile CSV has no data")
    header_index = 0
    for index, row in enumerate(rows[:5]):
        normalized = [re.sub(r"[^a-z0-9]", "", field.lower()) for field in row]
        if any(field in {"r", "radius", "rad", "sma", "radiusarcsec"} for field in normalized):
            header_index = index
            break
    header = rows[header_index]
    data = [row for row in rows[header_index + 1 :] if len(row) == len(header)]
    return header, data


def _find_column(header: Sequence[str], aliases: Sequence[str], *, exclude: set[int] | None = None) -> int | None:
    excluded = exclude or set()
    normalized = [re.sub(r"[^a-z0-9]", "", field.lower()) for field in header]
    for alias in aliases:
        key = re.sub(r"[^a-z0-9]", "", alias.lower())
        for index, field in enumerate(normalized):
            if index not in excluded and field == key:
                return index
    for alias in aliases:
        key = re.sub(r"[^a-z0-9]", "", alias.lower())
        for index, field in enumerate(normalized):
            if index not in excluded and key and key in field:
                return index
    return None


def _parse_light_profile(payload: bytes, distance_mpc: float, q: float, config: Mapping[str, Any]) -> tuple[list[dict[str, float]], dict[str, Any]]:
    header, data = _rows_from_csv_payload(payload)
    radius_index = _find_column(
        header, ("R", "radius", "rad", "sma", "radius_arcsec", "r_arcsec")
    )
    if radius_index is None:
        raise GravityItem9Probes2ReplayError("r light profile lacks a radius column")
    sb_index = _find_column(
        header,
        ("SB_r", "SBr", "mu_r", "mur", "surface_brightness_r", "SB", "mu"),
        exclude={radius_index},
    )
    if sb_index is None:
        raise GravityItem9Probes2ReplayError("r light profile lacks a surface-brightness column")
    total_index = _find_column(
        header, ("totmag_r", "totmagr", "cumulative_mag_r", "cum_mag_r", "totmag")
    )
    error_index = _find_column(header, ("SB_r_err", "SBr_err", "mu_r_err", "SB_e", "mu_err"))
    parsed = []
    for row in data:
        try:
            radius = float(row[radius_index])
            sb = float(row[sb_index])
        except ValueError:
            continue
        if not math.isfinite(radius) or not math.isfinite(sb) or radius <= 0:
            continue
        total = None
        if total_index is not None:
            try:
                total = float(row[total_index])
            except ValueError:
                total = None
        sb_error = 0.0
        if error_index is not None:
            try:
                sb_error = float(row[error_index])
            except ValueError:
                sb_error = 0.0
        parsed.append({"R": radius, "SB": sb, "SB_e": sb_error, "totmag": total})
    parsed.sort(key=lambda row: row["R"])
    unique = {float(row["R"]): row for row in parsed}
    parsed = [unique[key] for key in sorted(unique)]
    if len(parsed) < int(config["quality"]["minimum_photometry_points"]):
        raise GravityItem9Probes2ReplayError("insufficient r light points")
    used_fallback = not all(
        row["totmag"] is not None and math.isfinite(float(row["totmag"])) for row in parsed
    )
    if used_fallback:
        constants = config["constants"]
        radius_arcsec = np.asarray([row["R"] for row in parsed], dtype=np.float64)
        radius_pc = (
            radius_arcsec
            * distance_mpc
            * 1_000_000.0
            / float(constants["arcseconds_per_radian"])
        )
        edges = np.empty(len(radius_pc) + 1, dtype=np.float64)
        edges[0] = 0.0
        edges[1:-1] = 0.5 * (radius_pc[:-1] + radius_pc[1:])
        edges[-1] = radius_pc[-1] + 0.5 * (radius_pc[-1] - radius_pc[-2])
        area = math.pi * max(min(q, 1.0), 0.05) * np.diff(edges**2)
        sigma = 10.0 ** (
            -0.4
            * (
                np.asarray([row["SB"] for row in parsed])
                - float(constants["solar_absolute_r_magnitude_ab"])
                - float(constants["surface_brightness_conversion"])
            )
        )
        cumulative = np.maximum.accumulate(np.cumsum(np.maximum(sigma * area, 0.0)))
        distance_modulus = 5.0 * math.log10(distance_mpc) + 25.0
        absolute = float(constants["solar_absolute_r_magnitude_ab"]) - 2.5 * np.log10(
            np.maximum(cumulative, 1e-30)
        )
        apparent = absolute + distance_modulus
        for row, total in zip(parsed, apparent):
            row["totmag"] = float(total)
    for row in parsed:
        row["totmag_e"] = 0.0
        row["ellip"] = 1.0 - q
    return parsed, {
        "header": header,
        "radius_column": header[radius_index],
        "surface_brightness_column": header[sb_index],
        "cumulative_magnitude_column": header[total_index] if total_index is not None else None,
        "cumulative_light_fallback_used": used_fallback,
        "parsed_rows": len(parsed),
    }


def _parse_rotation_curve(payload: bytes, distance_mpc: float, config: Mapping[str, Any]) -> tuple[list[dict[str, float]], dict[str, Any]]:
    header, data = _rows_from_csv_payload(payload)
    radius_index = _find_column(header, ("R", "radius", "rad", "r_arcsec", "radius_arcsec", "r_kpc", "radius_kpc"))
    velocity_index = _find_column(header, ("Vrot", "V_c", "Vc", "V", "velocity", "rotation_velocity"))
    error_index = _find_column(header, ("Vrot_err", "V_c_err", "Vc_err", "V_e", "Verr", "velocity_error", "eV"))
    if radius_index is None or velocity_index is None or error_index is None:
        raise GravityItem9Probes2ReplayError("rotation curve lacks frozen R/V/V-error columns")
    radius_header = header[radius_index]
    velocity_header = header[velocity_index]
    radius_is_kpc = "kpc" in radius_header.lower()
    normalized_velocity = re.sub(r"[^a-z0-9]", "", velocity_header.lower())
    line_of_sight = "los" in normalized_velocity or "lineofsight" in normalized_velocity
    rows = []
    for row in data:
        try:
            radius = float(row[radius_index])
            velocity = float(row[velocity_index])
            error = float(row[error_index])
        except ValueError:
            continue
        if not all(math.isfinite(value) for value in (radius, velocity, error)):
            continue
        if radius_is_kpc:
            radius = (
                radius
                * float(config["constants"]["arcseconds_per_radian"])
                / (distance_mpc * 1000.0)
            )
        rows.append({"R": radius, "V": velocity, "V_e": abs(error)})
    if not rows:
        raise GravityItem9Probes2ReplayError("rotation curve has no finite rows")
    return rows, {
        "header": header,
        "radius_column": radius_header,
        "velocity_column": velocity_header,
        "velocity_error_column": header[error_index],
        "radius_converted_from_kpc": radius_is_kpc,
        "velocity_rule": "line_of_sight_divide_by_sin_i" if line_of_sight else "published_deprojected_rotation_speed",
        "parsed_rows": len(rows),
    }


def _quality_extract_one(
    sample: Mapping[str, Any], light_payload: bytes, rotation_payload: bytes, config: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    distance = float(sample["distance_mpc"])
    q = float(sample["axis_ratio_q"])
    light_rows, light_receipt = _parse_light_profile(light_payload, distance, q, config)
    predictor = attempt1.measure_photometric_predictors(
        photometry_rows=light_rows,
        distance_mpc=distance,
        extinction_r_magnitude=0.0,
        config=config,
    )
    rotation_rows, rotation_receipt = _parse_rotation_curve(rotation_payload, distance, config)
    features = attempt1.measure_point_features(
        predictor=predictor,
        rotation_radius_arcsec=[row["R"] for row in rotation_rows],
        distance_mpc=distance,
        config=config,
    )
    sin_i = float(predictor["inclination_sine"])
    line_of_sight = rotation_receipt["velocity_rule"] == "line_of_sight_divide_by_sin_i"
    response_rows = []
    mask = []
    for index, (feature, rotation) in enumerate(zip(features, rotation_rows)):
        denominator = sin_i if line_of_sight else 1.0
        observed = abs(float(rotation["V"])) / max(denominator, 1e-12)
        error = abs(float(rotation["V_e"])) / max(denominator, 1e-12)
        valid = bool(feature["within_photometry"])
        valid &= math.isfinite(observed) and observed >= float(config["quality"]["minimum_corrected_speed_km_s"])
        valid &= math.isfinite(error) and error >= 0.0
        valid &= error / max(observed, 1e-12) <= float(config["quality"]["maximum_fractional_speed_error"])
        mask.append(valid)
        response_rows.append(
            {
                "galaxy": sample["name"],
                "point_index": index,
                "observed_speed_km_s": _metric(observed),
                "observed_speed_error_km_s": _metric(error),
            }
        )
    accepted = np.asarray(mask, dtype=bool)
    reasons = []
    inclination = float(predictor["inclination_degrees"])
    if inclination < float(config["quality"]["minimum_inclination_degrees"]) or inclination > float(config["quality"]["maximum_inclination_degrees"]):
        reasons.append("inclination")
    if int(np.sum(accepted)) < int(config["quality"]["minimum_rotation_points"]):
        reasons.append("insufficient_rotation_points")
    if float(np.mean([feature["within_photometry"] for feature in features])) < float(config["quality"]["minimum_fraction_rotation_points_within_photometry"]):
        reasons.append("photometry_overlap")
    accepted_radii = np.asarray([feature["radius_kpc"] for feature, use in zip(features, accepted) if use], dtype=np.float64)
    radial_span = float(np.max(accepted_radii) / np.min(accepted_radii)) if len(accepted_radii) else 1.0
    if radial_span < float(config["quality"]["minimum_rotation_radial_span"]):
        reasons.append("radial_span")
    passed = not reasons
    feature_rows = []
    accepted_responses = []
    if passed:
        for output_index, (feature, response, use) in enumerate(zip(features, response_rows, accepted)):
            if not use:
                continue
            feature_row = {
                "galaxy": sample["name"],
                "point_index": output_index,
                "outer_fold": sample["outer_fold"],
                "source_family": sample["source_family"],
            }
            for field in attempt1.POINT_FEATURE_FIELDS:
                feature_row[field] = _metric(float(feature[field]))
            feature_rows.append(feature_row)
            accepted_response = dict(response)
            accepted_response["point_index"] = output_index
            accepted_responses.append(accepted_response)
    return feature_rows, accepted_responses, {
        "name": sample["name"],
        "normalized_identity": sample["normalized_identity"],
        "outer_fold": sample["outer_fold"],
        "source_family": sample["source_family"],
        "quality_pass": passed,
        "quality_failure_reasons": reasons,
        "raw_rotation_rows": len(rotation_rows),
        "accepted_rotation_rows": len(accepted_responses),
        "inclination_degrees": _metric(inclination),
        "radial_span": _metric(radial_span),
        "total_stellar_light_mass": _metric(float(predictor["total_mass"])),
        "median_surface_density": _metric(float(np.median(predictor["surface_density"]))),
        "light_receipt": light_receipt,
        "rotation_receipt": rotation_receipt,
        "light_payload_sha256": hashlib.sha256(light_payload).hexdigest(),
        "rotation_payload_sha256": hashlib.sha256(rotation_payload).hexdigest(),
    }


def extract_profiles(root: Path) -> dict[str, Path]:
    root = root.resolve()
    if SAMPLE_FREEZE_COMMIT.startswith("PENDING_"):
        raise GravityItem9Probes2ReplayError("sample freeze is not bound")
    config = load_config(root)
    sample_path = root / config["outputs"]["sample_manifest"]
    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    validate_sample_manifest(sample, root)
    candidate_path = root / config["outputs"]["candidate_manifest"]
    candidates = json.loads(candidate_path.read_text(encoding="utf-8"))
    validate_candidate_manifest(candidates, root)
    feature_rows = []
    response_rows = []
    galaxy_receipts = []
    archive_path = root / config["work"]["archive"]
    with zipfile.ZipFile(archive_path) as archive:
        for row in sample["objects"]:
            light_payload = archive.read(row["r_light_entry"])
            rotation_payload = archive.read(row["rotation_entry"])
            try:
                features, responses, receipt = _quality_extract_one(
                    row, light_payload, rotation_payload, config
                )
            except (GravityItem9Probes2ReplayError, attempt1.GravityItem9InteriorExteriorError) as exc:
                features, responses = [], []
                receipt = {
                    "name": row["name"],
                    "normalized_identity": row["normalized_identity"],
                    "outer_fold": row["outer_fold"],
                    "source_family": row["source_family"],
                    "quality_pass": False,
                    "quality_failure_reasons": [f"parser_or_feature:{type(exc).__name__}:{exc}"],
                    "light_payload_sha256": hashlib.sha256(light_payload).hexdigest(),
                    "rotation_payload_sha256": hashlib.sha256(rotation_payload).hexdigest(),
                }
            feature_rows.extend(features)
            response_rows.extend(responses)
            galaxy_receipts.append(receipt)
    feature_fields = ["galaxy", "point_index", "outer_fold", "source_family", *attempt1.POINT_FEATURE_FIELDS]
    response_fields = ["galaxy", "point_index", "observed_speed_km_s", "observed_speed_error_km_s"]
    feature_path = root / config["outputs"]["feature_table"]
    response_path = root / config["outputs"]["response_table"]
    attempt1._write_tsv(feature_path, feature_fields, feature_rows)
    attempt1._write_tsv(response_path, response_fields, response_rows)
    passing = sum(bool(row["quality_pass"]) for row in galaxy_receipts)
    selected = len(galaxy_receipts)
    retention = passing / selected if selected else 0.0
    quality_pass = passing >= int(config["quality"]["minimum_quality_passing_galaxies"])
    quality_pass &= retention >= float(config["quality"]["minimum_quality_retention_fraction"])
    summary = _content_hashed(
        {
            "schema_version": "invariant-gravity-item9-probes2-extraction-1.0",
            "scientific_freeze_commit": SCIENTIFIC_FREEZE_COMMIT,
            "sample_freeze_commit": SAMPLE_FREEZE_COMMIT,
            "decision": "PASS_ITEM9_PROBES2_PROFILE_QUALITY_FLOOR" if quality_pass else "FAIL_ITEM9_PROBES2_PROFILE_QUALITY_FLOOR",
            "galaxies": galaxy_receipts,
            "counts": {
                "selected_galaxies": selected,
                "quality_passing_galaxies": passing,
                "quality_failed_galaxies": selected - passing,
                "quality_retention_fraction": _metric(retention),
                "accepted_points": len(feature_rows),
                "rotation_curve_entry_payloads_opened": selected,
                "unselected_duplicate_rotation_entry_payloads_opened": 0,
                "PROBES_I_response_entries_opened": 0,
                "post_response_formula_cells": 0,
                "paid_model_calls": 0,
            },
            "outputs": {
                "feature_sha256": _sha256_file(feature_path),
                "response_sha256": _sha256_file(response_path),
            },
            "claims": config["claim_boundaries"],
        }
    )
    summary_path = root / config["outputs"]["extraction_summary"]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_bytes(canonical_json_bytes(summary) + b"\n")
    return {"features": feature_path, "responses": response_path, "summary": summary_path}


def _read_tsv(path: Path) -> list[dict[str, str]]:
    return attempt1._read_tsv(path)


def _load_experiment_data(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    candidate_path = root / config["outputs"]["candidate_manifest"]
    candidates = json.loads(candidate_path.read_text(encoding="utf-8"))
    validate_candidate_manifest(candidates, root)
    summary_path = root / config["outputs"]["extraction_summary"]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    _validate_content_hash(summary, "PROBES-II extraction summary")
    feature_path = root / config["outputs"]["feature_table"]
    response_path = root / config["outputs"]["response_table"]
    if _sha256_file(feature_path) != summary["outputs"]["feature_sha256"]:
        raise GravityItem9Probes2ReplayError("PROBES-II feature table changed")
    if _sha256_file(response_path) != summary["outputs"]["response_sha256"]:
        raise GravityItem9Probes2ReplayError("PROBES-II response table changed")
    features = _read_tsv(feature_path)
    responses = _read_tsv(response_path)
    if not features or len(features) != len(responses):
        raise GravityItem9Probes2ReplayError("PROBES-II point tables are empty or misaligned")
    keys = [(row["galaxy"], row["point_index"]) for row in features]
    if keys != [(row["galaxy"], row["point_index"]) for row in responses]:
        raise GravityItem9Probes2ReplayError("PROBES-II feature-response keys changed")
    names = []
    for row in features:
        if not names or row["galaxy"] != names[-1]:
            names.append(row["galaxy"])
    index_by_name = {name: index for index, name in enumerate(names)}
    galaxy_index = np.asarray([index_by_name[row["galaxy"]] for row in features], dtype=np.int64)
    if np.any(np.diff(galaxy_index) < 0):
        raise GravityItem9Probes2ReplayError("PROBES-II point rows are not galaxy-grouped")
    point_counts = np.bincount(galaxy_index, minlength=len(names))
    starts = np.concatenate(([0], np.cumsum(point_counts)[:-1])).astype(np.int64)
    first = [features[int(start)] for start in starts]
    arrays = {
        field: np.asarray([float(row[field]) for row in features], dtype=np.float64)
        for field in attempt1.POINT_FEATURE_FIELDS
    }
    return {
        "candidate_manifest": candidates,
        "summary": summary,
        "features": features,
        "responses": responses,
        "names": names,
        "galaxy_index": galaxy_index,
        "point_counts": point_counts,
        "starts": starts,
        "folds": np.asarray([int(row["outer_fold"]) for row in first], dtype=np.int64),
        "y": np.log10(np.asarray([float(row["observed_speed_km_s"]) for row in responses])),
        "y_error": np.asarray([float(row["observed_speed_error_km_s"]) for row in responses]),
        "feature_arrays": arrays,
        "first_rows": first,
    }


def _galaxy_losses(data: Mapping[str, Any], prediction: np.ndarray) -> np.ndarray:
    residual2 = (np.asarray(data["y"]) - prediction) ** 2
    return np.add.reduceat(residual2, np.asarray(data["starts"])) / np.asarray(data["point_counts"])


def _strata(data: Mapping[str, Any], summary: Mapping[str, Any], baseline: np.ndarray, primary: np.ndarray, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    baseline_loss = _galaxy_losses(data, baseline)
    primary_loss = _galaxy_losses(data, primary)
    by_name = {row["name"]: row for row in summary["galaxies"] if row["quality_pass"]}
    rows = [by_name[name] for name in data["names"]]
    distance = np.asarray([float(data["feature_arrays"]["log10_distance"][int(start)]) for start in data["starts"]])
    mass = np.asarray([math.log10(float(row["total_stellar_light_mass"])) for row in rows])
    surface = np.asarray([math.log10(float(row["median_surface_density"])) for row in rows])
    inclination = np.asarray([float(row["inclination_degrees"]) for row in rows])
    dimensions: dict[str, list[str]] = {
        "source_family": [str(row["source_family"]) for row in rows],
        "distance_half": ["low" if value <= float(np.median(distance)) else "high" for value in distance],
        "stellar_mass_half": ["low" if value <= float(np.median(mass)) else "high" for value in mass],
        "surface_density_half": ["low" if value <= float(np.median(surface)) else "high" for value in surface],
        "inclination_half": ["low" if value <= float(np.median(inclination)) else "high" for value in inclination],
    }
    result = []
    for dimension, values in dimensions.items():
        for value in sorted(set(values)):
            mask = np.asarray([entry == value for entry in values], dtype=bool)
            base = float(np.mean(baseline_loss[mask]))
            proposed = float(np.mean(primary_loss[mask]))
            result.append(
                {
                    "dimension": dimension,
                    "stratum": value,
                    "galaxies": int(np.sum(mask)),
                    "baseline_mse": _metric(base),
                    "primary_mse": _metric(proposed),
                    "primary_mse_gain": _metric(base - proposed),
                }
            )
    return result


def build_receipt(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config = load_config(root)
    data = _load_experiment_data(root, config)
    components = attempt1.build_operator_components(data, config)
    cells = data["candidate_manifest"]["cells"]
    predictions = [attempt1._candidate_prediction(cell, data, components, config) for cell in cells]
    primary = np.median(np.vstack(predictions[:5]), axis=0)
    mean_ensemble = np.mean(np.vstack(predictions[:5]), axis=0)
    rar = np.log10(np.asarray(data["feature_arrays"]["rar_speed_km_s"]))
    newtonian = np.log10(np.asarray(data["feature_arrays"]["newtonian_speed_km_s"]))
    local, local_records = attempt1._local_control_oof(data, config)
    point_folds = np.asarray(data["folds"])[np.asarray(data["galaxy_index"])]
    strongest = np.empty(len(rar), dtype=np.float64)
    baseline_folds = []
    for record in local_records:
        fold = int(record["fold"])
        use_local = float(record["local_inner_mse"]) < float(record["rar_train_mse"])
        mask = point_folds == fold
        strongest[mask] = local[mask] if use_local else rar[mask]
        baseline_folds.append(
            {
                "fold": fold,
                "selected_model": "flexible_local_ridge" if use_local else "fixed_stellar_rar",
                "local_inner_mse": record["local_inner_mse"],
                "rar_train_mse": record["rar_train_mse"],
            }
        )
    weights = attempt1._point_weights(data)
    metrics = {
        "stellar_newtonian": attempt1._metrics(data["y"], newtonian, weights),
        "fixed_stellar_rar": attempt1._metrics(data["y"], rar, weights),
        "oof_flexible_local": attempt1._metrics(data["y"], local, weights),
        "strongest_baseline": attempt1._metrics(data["y"], strongest, weights),
        "primary_median_ensemble": attempt1._metrics(data["y"], primary, weights),
        "secondary_mean_ensemble": attempt1._metrics(data["y"], mean_ensemble, weights),
    }
    atomic = []
    for cell, prediction in zip(cells, predictions):
        atomic.append(
            {
                "role": cell["role"],
                "candidate_id": cell["candidate_id"],
                "metrics": attempt1._metrics(data["y"], prediction, weights),
            }
        )
    strongest_mse = float(metrics["strongest_baseline"]["mse"])
    primary_mse = float(metrics["primary_median_ensemble"]["mse"])
    rar_mse = float(metrics["fixed_stellar_rar"]["mse"])
    improvement = strongest_mse - primary_mse
    relative = improvement / strongest_mse
    differences = _galaxy_losses(data, strongest) - _galaxy_losses(data, primary)
    paired = attempt1._paired_sign_flip(differences, config)
    strata = _strata(data, data["summary"], strongest, primary, config)
    source_rows = [
        row for row in strata
        if row["dimension"] == "source_family"
        and int(row["galaxies"]) >= int(config["evaluation"]["minimum_galaxies_per_source_gate"])
    ]
    half_dimensions = ["distance_half", "stellar_mass_half", "surface_density_half", "inclination_half"]
    half_pass = {
        dimension: all(
            float(row["primary_mse_gain"]) > 0
            for row in strata
            if row["dimension"] == dimension
        )
        for dimension in half_dimensions
    }
    attempt1_atomic_better = sum(float(row["metrics"]["mse"]) < rar_mse for row in atomic[:5])
    extraction_quality = data["summary"]["decision"] == "PASS_ITEM9_PROBES2_PROFILE_QUALITY_FLOOR"
    gates = {
        "quality_count_and_fraction_pass": extraction_quality,
        "zero_predecessor_identity_overlap": True,
        "primary_median_ensemble_r2_positive": float(metrics["primary_median_ensemble"]["r2"]) > 0,
        "primary_median_ensemble_beats_fixed_stellar_rar": primary_mse < rar_mse,
        "primary_median_ensemble_beats_oof_flexible_local_control": primary_mse < float(metrics["oof_flexible_local"]["mse"]),
        "primary_relative_mse_improvement_over_strongest_baseline_at_least": relative >= float(config["admission"]["primary_relative_mse_improvement_over_strongest_baseline_at_least"]),
        "paired_sign_flip_p_at_most": float(paired["p_value"]) <= float(config["admission"]["paired_sign_flip_p_at_most"]),
        "primary_gain_positive_in_all_source_families_at_minimum_count": bool(source_rows) and all(float(row["primary_mse_gain"]) > 0 for row in source_rows),
        "primary_gain_positive_in_both_distance_halves": half_pass["distance_half"],
        "primary_gain_positive_in_both_stellar_mass_halves": half_pass["stellar_mass_half"],
        "primary_gain_positive_in_both_surface_density_halves": half_pass["surface_density_half"],
        "primary_gain_positive_in_both_inclination_halves": half_pass["inclination_half"],
        "at_least_four_of_five_atomic_attempt1_cells_beat_fixed_stellar_rar": attempt1_atomic_better >= 4,
        "exact_prior_sparc_cell_beats_fixed_stellar_rar": float(atomic[5]["metrics"]["mse"]) < rar_mse,
        "post_response_formula_generation_zero": True,
    }
    if not extraction_quality:
        decision = "INCONCLUSIVE_ITEM9_PROBES2_QUALITY"
    elif all(gates.values()):
        decision = "PASS_ITEM9_PROBES2_ZERO_TUNING_REPLAY"
    elif (
        gates["primary_median_ensemble_beats_fixed_stellar_rar"]
        and gates["primary_median_ensemble_beats_oof_flexible_local_control"]
        and not gates["primary_gain_positive_in_all_source_families_at_minimum_count"]
    ):
        decision = "REJECT_ITEM9_PROBES2_COUNTEREXAMPLE_GATE"
    else:
        decision = "REJECT_ITEM9_PROBES2_ZERO_TUNING_REPLAY"
    config_path = root / CONFIG_PATH
    sample_path = root / config["outputs"]["sample_manifest"]
    candidate_path = root / config["outputs"]["candidate_manifest"]
    summary_path = root / config["outputs"]["extraction_summary"]
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item9-probes2-zero-tuning-result-1.0",
            "goal": config["goal"],
            "item_number": 9,
            "scientific_freeze_commit": SCIENTIFIC_FREEZE_COMMIT,
            "sample_freeze_commit": SAMPLE_FREEZE_COMMIT,
            "decision": decision,
            "creativity_label": config["scientific_contract"]["creativity_label"],
            "hypothesis": config["scientific_contract"]["hypothesis"],
            "counts": {
                "selected_galaxies": data["summary"]["counts"]["selected_galaxies"],
                "quality_passing_galaxies": data["summary"]["counts"]["quality_passing_galaxies"],
                "quality_failed_galaxies": data["summary"]["counts"]["quality_failed_galaxies"],
                "accepted_points": data["summary"]["counts"]["accepted_points"],
                "atomic_formula_cells": 6,
                "ensemble_formula_cells": 2,
                "candidate_selection_calls": 0,
                "post_response_formula_cells": 0,
                "paid_model_calls": 0,
                "attempt1_atomic_cells_beating_fixed_stellar_rar": attempt1_atomic_better,
                "source_families_entering_gate": len(source_rows),
            },
            "inputs": {
                "config_sha256": _sha256_file(config_path),
                "sample_manifest_sha256": _sha256_file(sample_path),
                "candidate_manifest_sha256": _sha256_file(candidate_path),
                "extraction_summary_sha256": _sha256_file(summary_path),
                "feature_sha256": data["summary"]["outputs"]["feature_sha256"],
                "response_sha256": data["summary"]["outputs"]["response_sha256"],
            },
            "primary": {
                "prediction": config["scientific_contract"]["primary_prediction"],
                "metrics": metrics,
                "atomic_cells": atomic,
                "baseline_folds": baseline_folds,
                "absolute_mse_improvement_over_strongest_baseline": _metric(improvement),
                "relative_mse_improvement_over_strongest_baseline": _metric(relative),
            },
            "paired_sign_flip": paired,
            "strata": strata,
            "gate_checks": gates,
            "gate_counts": {"passed": sum(bool(value) for value in gates.values()), "required": len(gates)},
            "limitations": {
                "complete_baryonic_mass_used": False,
                "gas_mass_profile_used": False,
                "thin_disk_poisson_field_used": False,
                "historical_novelty_adjudicated": False,
                "local_baseline_trained_out_of_fold_on_probes2": True,
            },
            "claims": config["claim_boundaries"],
        }
    )


def validate_receipt(receipt: Mapping[str, Any], root: Path) -> None:
    _validate_content_hash(receipt, "PROBES-II result receipt")
    config = load_config(root)
    if receipt["scientific_freeze_commit"] != SCIENTIFIC_FREEZE_COMMIT:
        raise GravityItem9Probes2ReplayError("result scientific freeze binding changed")
    if receipt["sample_freeze_commit"] != SAMPLE_FREEZE_COMMIT:
        raise GravityItem9Probes2ReplayError("result sample freeze binding changed")
    if receipt["counts"]["candidate_selection_calls"] != 0:
        raise GravityItem9Probes2ReplayError("PROBES-II selected a formula")
    if receipt["counts"]["post_response_formula_cells"] != 0:
        raise GravityItem9Probes2ReplayError("PROBES-II generated a post-response formula")
    if any(bool(value) for value in receipt["claims"].values()):
        raise GravityItem9Probes2ReplayError("PROBES-II receipt contains an overclaim")
    if receipt["counts"]["atomic_formula_cells"] != int(config["operator_grammar"]["atomic_formula_cells"]):
        raise GravityItem9Probes2ReplayError("atomic formula count changed")


def write_receipt(root: Path) -> Path:
    root = root.resolve()
    config = load_config(root)
    receipt = build_receipt(root)
    validate_receipt(receipt, root)
    path = root / config["outputs"]["result"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(receipt) + b"\n")
    return path


def check_receipt(root: Path) -> None:
    root = root.resolve()
    config = load_config(root)
    stored = json.loads((root / config["outputs"]["result"]).read_text(encoding="utf-8"))
    rebuilt = build_receipt(root)
    if stored != rebuilt:
        raise GravityItem9Probes2ReplayError("PROBES-II result receipt drifted")
    validate_receipt(stored, root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("candidates", "acquire", "sample", "extract", "run", "check")
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    if args.command == "candidates":
        print(write_candidate_manifest(args.root))
    elif args.command == "acquire":
        print(write_source_manifest(args.root))
    elif args.command == "sample":
        print(write_sample_manifest(args.root))
    elif args.command == "extract":
        print(json.dumps({key: str(value) for key, value in extract_profiles(args.root).items()}))
    elif args.command == "run":
        print(write_receipt(args.root))
    else:
        check_receipt(args.root)


if __name__ == "__main__":
    main()
