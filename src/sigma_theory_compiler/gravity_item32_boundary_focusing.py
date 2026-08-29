"""Frozen Item 32 vector/tensor baryonic-boundary focusing experiment.

The experiment assigns roles using response-blind MaNGA/GEMA predictors, excludes
every Item 30 and Item 31 role, and downloads MAPS files only for committed
exploration identities. Candidate fields are generated before any kinematic
response is opened. They are evaluated as signed axial directions, so an
unobservable reversal of galaxy spin is never counted as a prediction error.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import hmac
import io
import json
import math
import time
import urllib.request
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from sigma_theory_compiler.gravity_item22_polarization_superposition import (
    _canonical_bytes,
    _content_hashed,
    _git,
    _improvement,
    _read_json,
    _read_tsv,
    _require_ancestor,
    _sha256_bytes,
    _sha256_file,
    _to_numpy,
    _verify_content_hash,
    _write_json,
    _write_tsv,
)
from sigma_theory_compiler.gravity_item29_nonlinear_self_interaction import _backend
from sigma_theory_compiler.gravity_item30_screening_mechanisms import _candidate_digest

CONFIG_PATH = Path("configs/gravity_item32_boundary_focusing_v1.json")
MODULE_PATH = Path("src/sigma_theory_compiler/gravity_item32_boundary_focusing.py")
GOAL_PATH = Path("docs/GRAVITY_HIDDEN_VARIABLE_AND_THEORY_SEARCH_GOALS.md")
_ADMISSIBLE_CACHE: dict[str, tuple[dict[str, np.ndarray], dict[str, Any]]] = {}


class GravityItem32Error(RuntimeError):
    """Raised when an Item 32 freeze, leakage, source, or replay invariant fails."""


def load_config(root: Path) -> dict[str, Any]:
    config = _read_json(root / CONFIG_PATH)
    expected = "invariant-gravity-item32-boundary-focusing-config-1.0"
    if config.get("schema_version") != expected or int(config.get("item", -1)) != 32:
        raise GravityItem32Error("unexpected Item 32 config")
    if _sha256_file(root / GOAL_PATH) != str(config["stable_goal_sha256"]):
        raise GravityItem32Error("stable gravity goal changed")
    if int(config["candidate_generator"]["raw_candidate_cells"]) != 262144:
        raise GravityItem32Error("raw candidate boundary changed")
    if int(config["candidate_generator"]["post_response_cells"]) != 0:
        raise GravityItem32Error("post-response candidates entered Item 32")
    if not bool(config["discovery_policy"]["equal_initial_viability"]):
        raise GravityItem32Error("equal-viability policy changed")
    if bool(config["scope"]["confirmation_opening_authorized"]):
        raise GravityItem32Error("confirmation opening is not authorized")
    if bool(config["scope"]["paid_api_calls_authorized"]):
        raise GravityItem32Error("paid calls are outside Item 32")
    if not bool(config["sources"]["maps"]["confirmation_download_forbidden"]):
        raise GravityItem32Error("confirmation download boundary changed")
    for relative, expected_hash in config["scientific_dependencies"].items():
        if _sha256_file(root / str(relative)) != str(expected_hash):
            raise GravityItem32Error(f"scientific dependency changed: {relative}")
    return config


def _contract_digest(config: Mapping[str, Any]) -> str:
    value = json.loads(json.dumps(config))
    value["scientific_freeze_commit"] = "<BOUND_COMMIT>"
    value["sample_freeze_commit"] = "<BOUND_COMMIT>"
    return _sha256_bytes(_canonical_bytes(value))


def _source_paths(root: Path, config: Mapping[str, Any]) -> dict[str, Path]:
    base = root / str(config["paths"]["source_dir"])
    keys = (
        "predictors",
        "predictor_source_manifest",
        "sample_manifest",
        "candidate_manifest",
        "map_components",
        "map_source_manifest",
        "kinematic_responses",
        "compute_manifest",
    )
    return {key: base / str(config["paths"][key]) for key in keys}


def verify_science_freeze(root: Path, config: Mapping[str, Any]) -> None:
    commit = str(config["scientific_freeze_commit"])
    _require_ancestor(root, commit, "scientific freeze")
    frozen_config = json.loads(str(_git(root, "show", f"{commit}:{CONFIG_PATH.as_posix()}")))
    if _contract_digest(frozen_config) != _contract_digest(config):
        raise GravityItem32Error("scientific contract differs from frozen commit")
    frozen_module = _git(root, "show", f"{commit}:{MODULE_PATH.as_posix()}", text_mode=False)
    if not isinstance(frozen_module, bytes):
        raise GravityItem32Error("could not read frozen Item 32 module")
    if _sha256_bytes(frozen_module) != _sha256_file(root / MODULE_PATH):
        raise GravityItem32Error("Item 32 module differs from scientific freeze")


def verify_sample_freeze(root: Path, config: Mapping[str, Any]) -> None:
    commit = str(config["sample_freeze_commit"])
    _require_ancestor(root, commit, "sample freeze")
    paths = _source_paths(root, config)
    for key in (
        "predictors",
        "predictor_source_manifest",
        "sample_manifest",
        "candidate_manifest",
    ):
        relative = paths[key].relative_to(root).as_posix()
        frozen = _git(root, "show", f"{commit}:{relative}", text_mode=False)
        if not isinstance(frozen, bytes) or _sha256_bytes(frozen) != _sha256_file(paths[key]):
            raise GravityItem32Error(f"{key} differs from sample freeze")


def _hmac_rank(key: str, value: str) -> str:
    return hmac.new(key.encode(), value.encode(), hashlib.sha256).hexdigest()


def generate_raw_candidates(config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    generator = config["candidate_generator"]
    radices = {
        "polarity": len(generator["polarities"]),
        "amplitude": len(generator["amplitudes"]),
        "threshold": len(generator["boundary_thresholds"]),
        "sharpness": len(generator["sharpness"]),
        "radial": len(generator["radial_exponents"]),
        "shape": len(generator["shape_powers"]),
        "smoothing": len(generator["smoothing_sigmas_spaxel"]),
    }
    per_niche = int(generator["raw_candidate_cells"]) // 4
    if int(np.prod(list(radices.values()))) != per_niche:
        raise GravityItem32Error("mixed-radix grammar does not fill each niche exactly")
    pieces: dict[str, list[np.ndarray]] = {"niche": []} | {key: [] for key in radices}
    for niche in range(4):
        working = np.arange(per_niche, dtype=np.int64)
        decoded: dict[str, np.ndarray] = {}
        for key, radix in reversed(list(radices.items())):
            decoded[key] = (working % radix).astype(np.int16)
            working //= radix
        if np.any(working != 0):
            raise GravityItem32Error("candidate decoder overflow")
        pieces["niche"].append(np.full(per_niche, niche, dtype=np.int16))
        for key in radices:
            pieces[key].append(decoded[key])
    arrays = {key: np.concatenate(value) for key, value in pieces.items()}
    random = np.random.Generator(np.random.PCG64(int(generator["seed"])))
    order = random.permutation(len(arrays["niche"]))
    return {key: value[order] for key, value in arrays.items()}


def _candidate_values(
    config: Mapping[str, Any], arrays: Mapping[str, np.ndarray], begin: int, end: int, xp: Any
) -> dict[str, Any]:
    generator = config["candidate_generator"]
    index = {key: arrays[key][begin:end] for key in arrays}
    return {
        "niche": xp.asarray(index["niche"]),
        "polarity": xp.asarray(np.asarray(generator["polarities"])[index["polarity"]]),
        "amplitude": xp.asarray(np.asarray(generator["amplitudes"])[index["amplitude"]]),
        "threshold": xp.asarray(
            np.asarray(generator["boundary_thresholds"])[index["threshold"]]
        ),
        "sharpness": xp.asarray(np.asarray(generator["sharpness"])[index["sharpness"]]),
        "radial": xp.asarray(
            np.asarray(generator["radial_exponents"])[index["radial"]]
        ),
        "shape": xp.asarray(np.asarray(generator["shape_powers"])[index["shape"]]),
        "smoothing": xp.asarray(index["smoothing"]),
    }


def _analytic_direction_patterns() -> np.ndarray:
    return np.asarray(
        [
            [1.0, 0.65, -0.35, -0.9, 0.2, 0.75, -0.55, 0.45],
            [0.1, 0.95, -0.8, 0.3, -0.65, 0.5, 0.75, -0.4],
            [0.85, -0.2, 0.55, -0.7, 0.95, -0.45, 0.15, 0.6],
            [-0.6, 0.4, 0.9, -0.15, 0.7, -0.85, 0.35, 0.1],
        ],
        dtype=np.float64,
    )


def _analytic_candidate_response(
    config: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    strengths: np.ndarray,
    radii: np.ndarray,
    begin: int,
    end: int,
) -> np.ndarray:
    values = _candidate_values(config, arrays, begin, end, np)
    smooth_scale = np.asarray([1.0, 0.86, 0.67, 0.49], dtype=np.float64)
    effective_strength = strengths[None, :] * smooth_scale[values["smoothing"]][:, None]
    activation = np.power(
        effective_strength / (effective_strength + values["threshold"][:, None]),
        values["sharpness"][:, None] * values["shape"][:, None],
    )
    directions = _analytic_direction_patterns()[values["niche"]]
    radial = np.power(np.maximum(radii[None, :], 0.05), values["radial"][:, None])
    return (
        values["polarity"][:, None]
        * values["amplitude"][:, None]
        * activation
        * directions
        * radial
    )


def _wrap_axial_degrees(value: Any, xp: Any = np) -> Any:
    return xp.mod(value + 90.0, 180.0) - 90.0


def _admissible_candidates(
    config: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    cache_key = _sha256_bytes(
        _canonical_bytes(
            {
                "candidate_generator": config["candidate_generator"],
                "admissibility": config["admissibility"],
            }
        )
    )
    if cache_key in _ADMISSIBLE_CACHE:
        return _ADMISSIBLE_CACHE[cache_key]
    raw = generate_raw_candidates(config)
    count = len(raw["niche"])
    gates = config["admissibility"]
    keep = np.zeros(count, dtype=bool)
    local_response = np.full(count, np.nan)
    material_response = np.full(count, np.nan)
    maximum_bend = np.full(count, np.nan)
    batch = int(config["evaluation"]["candidate_batch_size"])
    radii = np.asarray([0.25, 0.45, 0.7, 0.95, 1.15, 1.35, 1.5, 0.6])
    local_strengths = np.full(8, float(gates["local_boundary_strength"]))
    material_strengths = np.full(8, float(gates["material_boundary_strength"]))
    adversarial_strengths = np.asarray([0.03, 0.08, 0.17, 0.31, 0.55, 0.9, 1.4, 2.2])
    for begin in range(0, count, batch):
        end = min(begin + batch, count)
        local = _analytic_candidate_response(config, raw, local_strengths, radii, begin, end)
        material = _analytic_candidate_response(
            config, raw, material_strengths, radii, begin, end
        )
        adversarial = _analytic_candidate_response(
            config, raw, adversarial_strengths, radii, begin, end
        )
        local_response[begin:end] = np.max(np.abs(local), axis=1)
        material_response[begin:end] = np.max(np.abs(material), axis=1)
        angles = np.degrees(np.arctan2(adversarial, 1.0))
        maximum_bend[begin:end] = np.max(np.abs(angles), axis=1)
        keep[begin:end] = (
            np.all(np.isfinite(adversarial), axis=1)
            & (
                local_response[begin:end]
                <= float(gates["maximum_local_fractional_direction_response"])
            )
            & (
                material_response[begin:end]
                >= float(gates["minimum_material_direction_response"])
            )
            & (
                maximum_bend[begin:end]
                <= float(gates["maximum_predicted_bend_degrees"])
            )
        )
    arrays = {key: value[keep] for key, value in raw.items()}
    signature_parts = []
    for begin in range(0, len(arrays["niche"]), batch):
        end = min(begin + batch, len(arrays["niche"]))
        delta = _analytic_candidate_response(
            config, arrays, adversarial_strengths, radii, begin, end
        )
        signature_parts.append(
            np.round(
                np.degrees(np.arctan2(delta, 1.0)),
                int(gates["behavioral_equivalence_precision_decimal_places"]),
            )
        )
    signatures = np.concatenate(signature_parts) if signature_parts else np.empty((0, 8))
    classes = len(np.unique(signatures, axis=0))
    raw_counts = Counter(int(value) for value in raw["niche"])
    admitted_counts = Counter(int(value) for value in arrays["niche"])
    audit = {
        "raw_candidates": count,
        "raw_per_niche": {str(key): raw_counts[key] for key in range(4)},
        "admissible_candidates": len(arrays["niche"]),
        "admissible_per_niche": {str(key): admitted_counts[key] for key in range(4)},
        "raw_candidate_digest": _candidate_digest(raw),
        "admissible_candidate_digest": _candidate_digest(arrays),
        "exact_parameter_signatures": len(
            np.unique(np.column_stack([arrays[key] for key in sorted(arrays)]), axis=0)
        ),
        "behavioral_equivalence_classes_adversarial": classes,
        "behavioral_duplicate_cells_adversarial": len(arrays["niche"]) - classes,
        "maximum_admitted_local_fractional_direction_response": float(
            np.max(local_response[keep])
        ),
        "minimum_admitted_material_direction_response": float(
            np.min(material_response[keep])
        ),
        "maximum_admitted_predicted_bend_degrees": float(np.max(maximum_bend[keep])),
    }
    generator = config["candidate_generator"]
    expected_fields = (
        ("expected_raw_candidate_digest", "raw_candidate_digest"),
        ("expected_admissible_candidate_digest", "admissible_candidate_digest"),
        ("expected_admissible_candidates", "admissible_candidates"),
        (
            "expected_behavioral_equivalence_classes_adversarial",
            "behavioral_equivalence_classes_adversarial",
        ),
    )
    for expected_key, audit_key in expected_fields:
        expected = generator.get(expected_key)
        if expected not in (None, "TO_BE_MEASURED", -1) and audit[audit_key] != expected:
            raise GravityItem32Error(f"candidate invariant changed: {expected_key}")
    expected_niches = generator.get("expected_admissible_per_niche")
    if (
        expected_niches
        and all(int(value) >= 0 for value in expected_niches.values())
        and audit["admissible_per_niche"] != expected_niches
    ):
        raise GravityItem32Error("admissible niche counts changed")
    _ADMISSIBLE_CACHE[cache_key] = arrays, audit
    return arrays, audit


def _candidate_manifest(config: Mapping[str, Any]) -> dict[str, Any]:
    _, audit = _admissible_candidates(config)
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item32-candidate-manifest-1.0",
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "algorithm": config["candidate_generator"]["algorithm"],
            "seed": config["candidate_generator"]["seed"],
            "niches": config["candidate_generator"]["niches"],
            "historical_novelty_claimed": config["candidate_generator"][
                "historical_novelty_claimed"
            ],
            "equivalence_boundaries": config["candidate_generator"][
                "equivalence_boundaries"
            ],
            "post_response_cells": 0,
            "audit": audit,
        }
    )


def _sample_manifest(
    config: Mapping[str, Any], pool: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    sample = config["sample"]
    mass_median = float(np.median([float(row["log_stellar_mass"]) for row in pool]))
    sersic_median = float(np.median([float(row["sersic_index"]) for row in pool]))
    cells: dict[str, list[dict[str, Any]]] = {
        f"m{mass}-n{sersic}": [] for mass in range(2) for sersic in range(2)
    }
    for source in pool:
        row = dict(source)
        mass_bin = int(float(row["log_stellar_mass"]) >= mass_median)
        sersic_bin = int(float(row["sersic_index"]) >= sersic_median)
        cell = f"m{mass_bin}-n{sersic_bin}"
        row.update({"mass_bin": mass_bin, "sersic_bin": sersic_bin, "sample_cell": cell})
        cells[cell].append(row)
    objects: list[dict[str, Any]] = []
    cell_counts: dict[str, dict[str, int]] = {}
    selected_per_cell = int(sample["selected_per_mass_sersic_cell"])
    confirmations_per_cell = int(sample["confirmation_per_cell"])
    for cell, values in sorted(cells.items()):
        ranked = sorted(
            values,
            key=lambda row: _hmac_rank(
                str(sample["role_key"]), f"select|{row['plateifu']}"
            ),
        )
        selected = ranked[:selected_per_cell]
        confirmation_ids = {
            str(row["plateifu"])
            for row in sorted(
                selected,
                key=lambda row: _hmac_rank(
                    str(sample["role_key"]), f"confirmation|{row['plateifu']}"
                ),
            )[:confirmations_per_cell]
        }
        exploration = [row for row in selected if str(row["plateifu"]) not in confirmation_ids]
        exploration = sorted(
            exploration,
            key=lambda row: _hmac_rank(str(sample["fold_key"]), str(row["plateifu"])),
        )
        fold_by_id = {
            str(row["plateifu"]): index % int(sample["outer_folds"])
            for index, row in enumerate(exploration)
        }
        for row in selected:
            identity = str(row["plateifu"])
            is_confirmation = identity in confirmation_ids
            row.update(
                {
                    "role": "reserved_confirmation" if is_confirmation else "exploration",
                    "outer_fold": None if is_confirmation else fold_by_id[identity],
                    "response_read": False,
                    "map_downloaded": False,
                }
            )
            objects.append(row)
        cell_counts[cell] = {
            "eligible": len(values),
            "selected": len(selected),
            "exploration": len(exploration),
            "reserved_confirmation": len(confirmation_ids),
        }
    objects.sort(key=lambda row: str(row["plateifu"]))
    roles = Counter(str(row["role"]) for row in objects)
    folds = Counter(
        int(row["outer_fold"]) for row in objects if row["role"] == "exploration"
    )
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item32-sample-manifest-1.0",
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "mass_median": f"{mass_median:.12e}",
            "sersic_median": f"{sersic_median:.12e}",
            "objects": objects,
            "selected_cell_counts": cell_counts,
            "fold_counts_exploration": {
                str(key): folds[key] for key in range(int(sample["outer_folds"]))
            },
            "counts": {
                "fresh_disk_pool": len(pool),
                "selected": len(objects),
                "exploration": roles["exploration"],
                "reserved_confirmation": roles["reserved_confirmation"],
                "response_rows_read": 0,
                "map_downloads": 0,
            },
            "claims": {
                "response_values_read": 0,
                "confirmation_values_read": 0,
                "confirmation_maps_downloaded": 0,
                "object_identity_used_as_numeric_feature": False,
            },
        }
    )


def _fresh_pool(root: Path, config: Mapping[str, Any]) -> tuple[list[dict[str, str]], set[str]]:
    inherited = _read_tsv(root / str(config["sources"]["inherited_predictors"]))
    if len(inherited) != int(config["sample"]["expected_inherited_predictors"]):
        raise GravityItem32Error("inherited predictor count changed")
    prior_ids: set[str] = set()
    for key in ("item30_sample_manifest", "item31_sample_manifest"):
        predecessor = _read_json(root / str(config["sources"][key]))
        _verify_content_hash(predecessor, key)
        prior_ids.update(str(row["plateifu"]) for row in predecessor["objects"])
    sample = config["sample"]
    pool = [
        row
        for row in inherited
        if str(row["plateifu"]) not in prior_ids
        and float(row["snr_med_g"]) >= float(sample["predictor_minimum_snr_med_g"])
        and float(row["sersic_index"]) <= float(sample["maximum_sersic_index"])
        and float(sample["minimum_axis_ratio"])
        <= float(row["axis_ratio"])
        <= float(sample["maximum_axis_ratio"])
    ]
    if len(pool) != int(sample["expected_fresh_disk_pool"]):
        raise GravityItem32Error("fresh response-blind disk pool changed")
    return pool, prior_ids


def prepare_predictors(root: Path) -> dict[str, Path]:
    root = root.resolve()
    config = load_config(root)
    verify_science_freeze(root, config)
    paths = _source_paths(root, config)
    pool, prior_ids = _fresh_pool(root, config)
    sample_manifest = _sample_manifest(config, pool)
    expected = config["sample"]
    expected_counts = {
        "fresh_disk_pool": int(expected["expected_fresh_disk_pool"]),
        "selected": int(expected["expected_selected"]),
        "exploration": int(expected["expected_exploration"]),
        "reserved_confirmation": int(expected["expected_confirmation"]),
        "response_rows_read": 0,
        "map_downloads": 0,
    }
    if sample_manifest["counts"] != expected_counts:
        raise GravityItem32Error("frozen Item 32 sample counts changed")
    selected = sample_manifest["objects"]
    inherited = _read_tsv(root / str(config["sources"]["inherited_predictors"]))
    inherited_columns = list(inherited[0])
    extra_columns = [
        "mass_bin",
        "sersic_bin",
        "sample_cell",
        "role",
        "outer_fold",
        "response_read",
        "map_downloaded",
    ]
    _write_tsv(paths["predictors"], selected, [*inherited_columns, *extra_columns])
    source_manifest = _content_hashed(
        {
            "schema_version": "invariant-gravity-item32-predictor-source-manifest-1.0",
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "inherited_predictor_path": config["sources"]["inherited_predictors"],
            "inherited_predictor_sha256": _sha256_file(
                root / str(config["sources"]["inherited_predictors"])
            ),
            "item30_sample_sha256": _sha256_file(
                root / str(config["sources"]["item30_sample_manifest"])
            ),
            "item31_sample_sha256": _sha256_file(
                root / str(config["sources"]["item31_sample_manifest"])
            ),
            "counts": {
                "inherited_predictors": int(expected["expected_inherited_predictors"]),
                "predecessor_roles_excluded": len(prior_ids),
                "fresh_disk_pool": len(pool),
                "selected": len(selected),
                "response_columns_read": 0,
            },
            "cuts": {
                "minimum_snr_med_g": expected["predictor_minimum_snr_med_g"],
                "maximum_sersic_index": expected["maximum_sersic_index"],
                "axis_ratio": [expected["minimum_axis_ratio"], expected["maximum_axis_ratio"]],
            },
            "claims": {
                "target_blind": True,
                "confirmation_values_read": 0,
                "post_response_formula_cells": 0,
            },
        }
    )
    _write_json(paths["predictor_source_manifest"], source_manifest)
    _write_json(paths["sample_manifest"], sample_manifest)
    _write_json(paths["candidate_manifest"], _candidate_manifest(config))
    return paths


def _download(url: str) -> tuple[bytes, dict[str, str]]:
    request = urllib.request.Request(url, headers={"User-Agent": "Invariant-Item32/1.0"})
    with urllib.request.urlopen(request, timeout=240) as response:
        payload = response.read()
        headers = {str(key).lower(): str(value) for key, value in response.headers.items()}
    if not payload:
        raise GravityItem32Error(f"empty source response: {url}")
    return payload, headers


def _maps_location(config: Mapping[str, Any], plateifu: str) -> tuple[str, str]:
    parts = str(plateifu).split("-")
    if len(parts) != 2 or any(not part.isdigit() for part in parts):
        raise GravityItem32Error("invalid MaNGA plateifu")
    plate, ifu = parts
    source = config["sources"]["maps"]
    filename = str(source["filename_template"]).format(
        plateifu=plateifu, daptype=source["daptype"]
    )
    url = "/".join(
        (
            str(source["base_url"]).rstrip("/"),
            str(source["daptype"]),
            plate,
            ifu,
            filename,
        )
    )
    return filename, url


def _maps_payload(
    root: Path, config: Mapping[str, Any], plateifu: str
) -> tuple[bytes, str, str, dict[str, str]]:
    filename, url = _maps_location(config, plateifu)
    cache = root / str(config["sources"]["maps"]["raw_cache"])
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / filename
    if path.exists():
        payload = path.read_bytes()
        headers = {"cache": "hit"}
    else:
        attempts = int(config["sources"]["maps"]["download_attempts"])
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                payload, headers = _download(url)
                break
            except (OSError, TimeoutError) as exc:
                last_error = exc
                if attempt + 1 < attempts:
                    time.sleep(float(attempt + 1))
        else:
            raise GravityItem32Error(
                f"failed to download {plateifu} after {attempts} attempts"
            ) from last_error
        path.write_bytes(payload)
    return payload, filename, url, headers


def _channel_index(hdu: Any, label: str) -> int:
    channels = int(hdu.header.get("NAXIS3", 1))
    for ordinal in range(1, channels + 1):
        if str(hdu.header.get(f"C{ordinal}", "")).strip() == label:
            return ordinal - 1
    raise GravityItem32Error(f"required MAPS channel {label!r} missing from {hdu.name}")


def _normalized_gaussian(values: np.ndarray, valid: np.ndarray, sigma: float) -> np.ndarray:
    try:
        from scipy.ndimage import gaussian_filter
    except ImportError as exc:
        raise GravityItem32Error("Item 32 requires scipy for frozen map smoothing") from exc
    numerator = gaussian_filter(np.where(valid, values, 0.0), sigma=sigma, mode="nearest")
    denominator = gaussian_filter(valid.astype(np.float64), sigma=sigma, mode="nearest")
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 1e-8,
    )


def _quadrant_count(azimuth_degrees: np.ndarray) -> int:
    quadrants = np.floor(np.mod(azimuth_degrees, 360.0) / 90.0).astype(int)
    return len({int(value) for value in quadrants})


def _axial_harmonic_fit(
    velocity: np.ndarray,
    radius: np.ndarray,
    azimuth_degrees: np.ndarray,
    inverse_variance: np.ndarray,
    bounds: Sequence[float],
    minimum_count: int,
    minimum_quadrants: int,
    minimum_amplitude: float,
    maximum_condition: float,
    label: str,
) -> dict[str, float | int]:
    lower, upper = (float(value) for value in bounds)
    selected = (
        np.isfinite(velocity)
        & np.isfinite(radius)
        & np.isfinite(azimuth_degrees)
        & np.isfinite(inverse_variance)
        & (inverse_variance > 0)
        & (radius >= lower)
        & (radius < upper)
    )
    count = int(np.count_nonzero(selected))
    if count < minimum_count:
        raise GravityItem32Error(f"insufficient {label} measurements")
    azimuth = azimuth_degrees[selected]
    quadrants = _quadrant_count(azimuth)
    if quadrants < minimum_quadrants:
        raise GravityItem32Error(f"insufficient {label} azimuth coverage")
    radians = np.radians(azimuth)
    design = np.column_stack((np.ones(count), np.cos(radians), np.sin(radians)))
    weights = inverse_variance[selected]
    normal = design.T @ (weights[:, None] * design)
    condition = float(np.linalg.cond(normal))
    if not math.isfinite(condition) or condition > maximum_condition:
        raise GravityItem32Error(f"ill-conditioned {label} harmonic fit")
    coefficient = np.linalg.solve(normal, design.T @ (weights * velocity[selected]))
    amplitude = float(math.hypot(float(coefficient[1]), float(coefficient[2])))
    if amplitude < minimum_amplitude:
        raise GravityItem32Error(f"weak {label} harmonic amplitude")
    angle = float(
        _wrap_axial_degrees(
            math.degrees(math.atan2(float(coefficient[2]), float(coefficient[1])))
        )
    )
    return {
        "measurements": count,
        "azimuth_quadrants": quadrants,
        "condition_number": condition,
        "harmonic_amplitude_km_s": amplitude,
        "axial_angle_degrees": angle,
    }


def _unique_stellar_measurements(
    bin_ids: np.ndarray,
    valid: np.ndarray,
    inverse_variance: np.ndarray,
    velocity: np.ndarray,
    radius: np.ndarray,
    azimuth: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    flat_ids = bin_ids.ravel()
    flat_valid = valid.ravel()
    flat_ivar = inverse_variance.ravel()
    indices = []
    for bin_id in np.unique(flat_ids[flat_valid]):
        members = np.flatnonzero(flat_valid & (flat_ids == bin_id))
        indices.append(int(members[int(np.argmax(flat_ivar[members]))]))
    chosen = np.asarray(indices, dtype=int)
    return (
        velocity.ravel()[chosen],
        radius.ravel()[chosen],
        azimuth.ravel()[chosen],
        inverse_variance.ravel()[chosen],
    )


def _boundary_basis(
    flux: np.ndarray,
    flux_ivar: np.ndarray,
    spaxel_snr: np.ndarray,
    radius: np.ndarray,
    azimuth: np.ndarray,
    config: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, float | int]]:
    source = config["map_source"]
    valid = (
        np.isfinite(flux)
        & np.isfinite(flux_ivar)
        & np.isfinite(spaxel_snr)
        & np.isfinite(radius)
        & np.isfinite(azimuth)
        & (flux > 0)
        & (flux_ivar > 0)
        & (spaxel_snr >= float(source["minimum_spaxel_snr"]))
    )
    domain = valid & (radius >= 0.2) & (radius < 1.5)
    annulus = np.full(radius.shape, -1, dtype=np.int8)
    for index, bounds in enumerate(source["annuli_re"]):
        annulus[(radius >= float(bounds[0])) & (radius < float(bounds[1]))] = index
    counts = [int(np.count_nonzero(domain & (annulus == index))) for index in range(2)]
    if min(counts) < int(source["minimum_source_pixels_per_annulus"]):
        raise GravityItem32Error("insufficient continuum source pixels")
    positive_median = float(np.median(flux[domain]))
    transformed = np.log1p(np.maximum(flux, 0.0) / positive_median)
    radial_y, radial_x = np.gradient(np.where(np.isfinite(radius), radius, 0.0))
    radial_norm = np.hypot(radial_x, radial_y)
    er_x = np.divide(radial_x, radial_norm, out=np.zeros_like(radius), where=radial_norm > 1e-8)
    er_y = np.divide(radial_y, radial_norm, out=np.zeros_like(radius), where=radial_norm > 1e-8)
    et_x, et_y = -er_y, er_x
    smoothing = [float(value) for value in source["smoothing_sigmas_spaxel"]]
    boundary_stack = []
    direction_stack = []
    anisotropy_stack = []
    for sigma in smoothing:
        smooth = _normalized_gaussian(transformed, valid, sigma)
        grad_y, grad_x = np.gradient(smooth)
        magnitude = np.hypot(grad_x, grad_y)
        positive = magnitude[domain & (magnitude > 0)]
        normalization = float(np.median(positive)) if len(positive) else 1.0
        boundary = np.clip(magnitude / max(normalization, 1e-12), 0.0, 10.0)
        nx = np.divide(grad_x, magnitude, out=np.zeros_like(grad_x), where=magnitude > 1e-12)
        ny = np.divide(grad_y, magnitude, out=np.zeros_like(grad_y), where=magnitude > 1e-12)
        nr = nx * er_x + ny * er_y
        nt = nx * et_x + ny * et_y
        grad_x_y, grad_x_x = np.gradient(grad_x)
        grad_y_y, grad_y_x = np.gradient(grad_y)
        hxx = grad_x_x
        hyy = grad_y_y
        hxy = 0.5 * (grad_x_y + grad_y_x)
        orientation = 0.5 * np.arctan2(2.0 * hxy, hxx - hyy)
        px, py = np.cos(orientation), np.sin(orientation)
        trace = hxx + hyy
        split = np.sqrt(np.maximum((hxx - hyy) ** 2 + 4.0 * hxy**2, 0.0))
        lambda_one = 0.5 * (trace + split)
        lambda_two = 0.5 * (trace - split)
        anisotropy = np.divide(
            np.abs(lambda_one - lambda_two),
            np.abs(lambda_one) + np.abs(lambda_two) + 1e-12,
        )
        pr = px * er_x + py * er_y
        pt = px * et_x + py * et_y
        directions = np.stack((nt, 2.0 * nr * nt, nr, 2.0 * anisotropy * pr * pt))
        boundary_stack.append(boundary)
        direction_stack.append(directions)
        anisotropy_stack.append(anisotropy)
    use = domain
    weights = np.sqrt(np.maximum(flux[use], 0.0))
    weights /= max(float(np.mean(weights)), 1e-12)
    default_boundary = boundary_stack[min(2, len(boundary_stack) - 1)][use]
    default_anisotropy = anisotropy_stack[min(2, len(anisotropy_stack) - 1)][use]
    radial_values = radius[use]
    total_weight = float(np.sum(weights))
    moments = [
        float(np.sum(weights * radial_values**order) / total_weight) for order in range(1, 5)
    ]
    rotated_flux = np.rot90(flux, 2)
    rotated_valid = np.rot90(valid, 2)
    common = domain & rotated_valid
    asymmetry = float(
        np.sum(np.abs(flux[common] - rotated_flux[common]))
        / max(2.0 * np.sum(np.abs(flux[common])), 1e-12)
    )
    y_grid, x_grid = np.indices(flux.shape, dtype=np.float64)
    center_y = 0.5 * (flux.shape[0] - 1)
    center_x = 0.5 * (flux.shape[1] - 1)
    centroid_x = float(np.sum(weights * x_grid[use]) / total_weight)
    centroid_y = float(np.sum(weights * y_grid[use]) / total_weight)
    re_per_pixel = float(np.median(radial_norm[use & (radial_norm > 0)]))
    centroid_offset = math.hypot(centroid_x - center_x, centroid_y - center_y) * re_per_pixel
    features = {
        "flux_m1": moments[0],
        "flux_m2": moments[1],
        "flux_m3": moments[2],
        "flux_m4": moments[3],
        "flux_asymmetry": asymmetry,
        "boundary_median": float(np.median(default_boundary)),
        "boundary_p90": float(np.quantile(default_boundary, 0.9)),
        "hessian_anisotropy": float(np.sum(weights * default_anisotropy) / total_weight),
        "centroid_offset_re": centroid_offset,
        "inner_source_pixels": counts[0],
        "outer_source_pixels": counts[1],
    }
    radians = np.radians(azimuth[use])
    basis = {
        "boundary": np.asarray([value[use] for value in boundary_stack], dtype=np.float32),
        "direction": np.asarray(
            [[niche[use] for niche in value] for value in direction_stack],
            dtype=np.float32,
        ).transpose(1, 0, 2),
        "radius": radius[use].astype(np.float32),
        "cosine": np.cos(radians).astype(np.float32),
        "sine": np.sin(radians).astype(np.float32),
        "weight": weights.astype(np.float32),
        "annulus": annulus[use].astype(np.int8),
    }
    return basis, features


def _response_pair(
    values: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    config: Mapping[str, Any],
    channel: str,
) -> tuple[bool, list[dict[str, float | int]], str | None]:
    response = config["response"]
    if channel == "stellar":
        minimum_count = int(response["minimum_stellar_bins_per_annulus"])
        minimum_quadrants = int(response["minimum_stellar_azimuth_quadrants"])
        minimum_amplitude = float(response["minimum_stellar_harmonic_amplitude_km_s"])
        maximum_bend = float(response["maximum_absolute_stellar_bend_degrees"])
    else:
        minimum_count = int(response["minimum_halpha_spaxels_per_annulus"])
        minimum_quadrants = int(response["minimum_halpha_azimuth_quadrants"])
        minimum_amplitude = float(response["minimum_halpha_harmonic_amplitude_km_s"])
        maximum_bend = float(response["maximum_absolute_halpha_bend_degrees"])
    summaries = []
    try:
        for label, bounds in zip(config["map_source"]["annulus_labels"], config["map_source"]["annuli_re"]):
            summary = _axial_harmonic_fit(
                *values,
                bounds,
                minimum_count,
                minimum_quadrants,
                minimum_amplitude,
                float(response["maximum_design_condition_number"]),
                f"{channel} {label}",
            )
            if abs(float(summary["axial_angle_degrees"])) > maximum_bend:
                raise GravityItem32Error(f"extreme {channel} axial bend")
            summaries.append(summary)
    except GravityItem32Error as exc:
        return False, [], str(exc)
    return True, summaries, None


def _derive_map_payload(
    compressed_payload: bytes,
    sample_row: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    try:
        from astropy.io import fits

        payload = gzip.decompress(compressed_payload)
        with fits.open(io.BytesIO(payload), memmap=False) as hdus:
            source = config["sources"]["maps"]
            primary = hdus[0].header
            for key, expected in source["required_primary_headers"].items():
                if str(primary.get(key, "")).strip() != str(expected):
                    raise GravityItem32Error(f"MaNGA MAPS header {key} changed")
            if str(primary.get("PLATEIFU", "")).strip() != str(sample_row["plateifu"]):
                raise GravityItem32Error("MaNGA MAPS plateifu changed")
            if str(primary.get("MANGAID", "")).strip().upper() != str(
                sample_row["mangaid"]
            ).upper():
                raise GravityItem32Error("MaNGA MAPS mangaid changed")
            required = [hdus[str(name)] for name in source["required_extensions"]]
            if bool(source["fits_checksum_required"]):
                for hdu in [hdus[0], *required]:
                    if hdu.verify_checksum() != 1 or hdu.verify_datasum() != 1:
                        raise GravityItem32Error("MaNGA MAPS FITS checksum failed")
            channels = source["channels"]
            spaxel_coordinates = hdus["SPX_ELLCOO"]
            bin_coordinates = hdus["BIN_LWELLCOO"]
            radius = np.asarray(
                spaxel_coordinates.data[
                    _channel_index(spaxel_coordinates, str(channels["radius_re"]))
                ],
                dtype=np.float64,
            )
            azimuth = np.asarray(
                spaxel_coordinates.data[
                    _channel_index(spaxel_coordinates, str(channels["spaxel_azimuth"]))
                ],
                dtype=np.float64,
            )
            flux = np.asarray(hdus["SPX_MFLUX"].data, dtype=np.float64)
            flux_ivar = np.asarray(hdus["SPX_MFLUX_IVAR"].data, dtype=np.float64)
            spaxel_snr = np.asarray(hdus["SPX_SNR"].data, dtype=np.float64)
            basis, features = _boundary_basis(
                flux, flux_ivar, spaxel_snr, radius, azimuth, config
            )

            stellar_radius = np.asarray(
                bin_coordinates.data[
                    _channel_index(bin_coordinates, str(channels["radius_re"]))
                ],
                dtype=np.float64,
            )
            stellar_azimuth = np.asarray(
                bin_coordinates.data[
                    _channel_index(bin_coordinates, str(channels["bin_azimuth"]))
                ],
                dtype=np.float64,
            )
            bin_hdu = hdus["BINID"]
            stellar_bin = np.asarray(
                bin_hdu.data[_channel_index(bin_hdu, str(channels["stellar_bin_id"]))],
                dtype=np.int64,
            )
            stellar_velocity = np.asarray(hdus["STELLAR_VEL"].data, dtype=np.float64)
            stellar_ivar = np.asarray(hdus["STELLAR_VEL_IVAR"].data, dtype=np.float64)
            stellar_mask = np.asarray(hdus["STELLAR_VEL_MASK"].data, dtype=np.int64)
            stellar_snr = np.asarray(hdus["BIN_SNR"].data, dtype=np.float64)
            stellar_fom = hdus["STELLAR_FOM"]
            stellar_rchi2 = np.asarray(
                stellar_fom.data[
                    _channel_index(stellar_fom, str(channels["stellar_rchi2"]))
                ],
                dtype=np.float64,
            )
            response = config["response"]
            stellar_valid = (
                (stellar_bin >= 0)
                & (stellar_mask == 0)
                & (stellar_ivar > 0)
                & (stellar_snr >= float(response["minimum_stellar_bin_snr"]))
                & (stellar_rchi2 >= 0)
                & (stellar_rchi2 <= float(response["maximum_stellar_rchi2"]))
                & np.isfinite(stellar_velocity)
                & np.isfinite(stellar_radius)
                & np.isfinite(stellar_azimuth)
                & np.isfinite(stellar_ivar)
                & np.isfinite(stellar_snr)
                & np.isfinite(stellar_rchi2)
            )
            stellar_values = _unique_stellar_measurements(
                stellar_bin,
                stellar_valid,
                stellar_ivar,
                stellar_velocity,
                stellar_radius,
                stellar_azimuth,
            )
            halpha_channel = _channel_index(hdus["EMLINE_GVEL"], str(channels["halpha"]))
            halpha_velocity = np.asarray(
                hdus["EMLINE_GVEL"].data[halpha_channel], dtype=np.float64
            )
            halpha_ivar = np.asarray(
                hdus["EMLINE_GVEL_IVAR"].data[halpha_channel], dtype=np.float64
            )
            halpha_mask = np.asarray(
                hdus["EMLINE_GVEL_MASK"].data[halpha_channel], dtype=np.int64
            )
            halpha_anr = np.asarray(
                hdus["EMLINE_GANR"].data[halpha_channel], dtype=np.float64
            )
            halpha_ew = np.asarray(
                hdus["EMLINE_GEW"].data[halpha_channel], dtype=np.float64
            )
            halpha_ew_mask = np.asarray(
                hdus["EMLINE_GEW_MASK"].data[halpha_channel], dtype=np.int64
            )
            halpha_rchi2 = np.asarray(
                hdus["EMLINE_LFOM"].data[halpha_channel], dtype=np.float64
            )
            halpha_valid = (
                (halpha_mask == 0)
                & (halpha_ew_mask == 0)
                & (halpha_ivar > 0)
                & (halpha_anr >= float(response["minimum_halpha_anr"]))
                & (halpha_ew >= float(response["minimum_halpha_ew_angstrom"]))
                & (halpha_rchi2 >= 0)
                & (halpha_rchi2 <= float(response["maximum_halpha_rchi2"]))
                & np.isfinite(halpha_velocity)
                & np.isfinite(radius)
                & np.isfinite(azimuth)
                & np.isfinite(halpha_ivar)
                & np.isfinite(halpha_anr)
                & np.isfinite(halpha_ew)
                & np.isfinite(halpha_rchi2)
            )
            halpha_values = (
                halpha_velocity[halpha_valid],
                radius[halpha_valid],
                azimuth[halpha_valid],
                halpha_ivar[halpha_valid],
            )
            drp3qual = int(primary.get("DRP3QUAL", -1))
            dapqual = int(primary.get("DAPQUAL", -1))
    except GravityItem32Error:
        raise
    except (OSError, ValueError, TypeError, IndexError, KeyError, AttributeError) as exc:
        raise GravityItem32Error("invalid MaNGA MAPS FITS") from exc
    stellar_quality, stellar_summaries, stellar_reason = _response_pair(
        stellar_values, config, "stellar"
    )
    halpha_quality, halpha_summaries, halpha_reason = _response_pair(
        halpha_values, config, "halpha"
    )
    record = {
        "plateifu": sample_row["plateifu"],
        "mangaid": sample_row["mangaid"],
        "drp3qual": drp3qual,
        "dapqual": dapqual,
        "source_features": features,
        "stellar_quality_pass": stellar_quality,
        "stellar_quality_reason": stellar_reason,
        "stellar_annuli": stellar_summaries,
        "halpha_quality_pass": halpha_quality,
        "halpha_quality_reason": halpha_reason,
        "halpha_annuli": halpha_summaries,
    }
    return basis, record


def _pad_basis(records: Sequence[tuple[str, dict[str, np.ndarray]]]) -> dict[str, np.ndarray]:
    maximum = max(len(basis["radius"]) for _, basis in records)
    count = len(records)
    output = {
        "plateifu": np.asarray([identity for identity, _ in records]),
        "pixel_count": np.zeros(count, dtype=np.int32),
        "boundary": np.zeros((count, 4, maximum), dtype=np.float32),
        "direction": np.zeros((count, 4, 4, maximum), dtype=np.float32),
        "radius": np.zeros((count, maximum), dtype=np.float32),
        "cosine": np.zeros((count, maximum), dtype=np.float32),
        "sine": np.zeros((count, maximum), dtype=np.float32),
        "weight": np.zeros((count, maximum), dtype=np.float32),
        "annulus": np.full((count, maximum), -1, dtype=np.int8),
    }
    for index, (_, basis) in enumerate(records):
        size = len(basis["radius"])
        output["pixel_count"][index] = size
        for key in ("boundary", "direction", "radius", "cosine", "sine", "weight", "annulus"):
            output[key][index, ..., :size] = basis[key]
    return output


def acquire_maps(root: Path) -> dict[str, Path]:
    root = root.resolve()
    config = load_config(root)
    verify_science_freeze(root, config)
    verify_sample_freeze(root, config)
    paths = _source_paths(root, config)
    sample_manifest = _read_json(paths["sample_manifest"])
    _verify_content_hash(sample_manifest, "Item 32 sample manifest")
    exploration = sorted(
        (row for row in sample_manifest["objects"] if row["role"] == "exploration"),
        key=lambda row: str(row["plateifu"]),
    )
    confirmation = {
        str(row["plateifu"])
        for row in sample_manifest["objects"]
        if row["role"] == "reserved_confirmation"
    }
    if len(exploration) != int(config["sample"]["expected_exploration"]):
        raise GravityItem32Error("exploration role count changed before MAPS access")
    basis_records: list[tuple[str, dict[str, np.ndarray]]] = []
    response_records = []
    files = []
    failures = []
    for sample_row in exploration:
        identity = str(sample_row["plateifu"])
        if identity in confirmation:
            raise GravityItem32Error("confirmation identity entered MAPS download")
        payload, filename, url, headers = _maps_payload(root, config, identity)
        file_record = {
            "plateifu": identity,
            "file_name": filename,
            "url": url,
            "file_bytes": len(payload),
            "file_sha256": hashlib.sha256(payload).hexdigest(),
            "cache": headers.get("cache", "miss"),
        }
        try:
            basis, response_record = _derive_map_payload(payload, sample_row, config)
        except GravityItem32Error as exc:
            failures.append({**file_record, "reason": str(exc)})
            continue
        basis_records.append((identity, basis))
        response_records.append({**response_record, **file_record})
        files.append({**file_record, "fits_checksum_verified": True})
    touched = {str(row["plateifu"]) for row in [*files, *failures]}
    if confirmation & touched:
        raise GravityItem32Error("confirmation MAPS file entered exploration source")
    if not basis_records:
        raise GravityItem32Error("no Item 32 continuum bases were extracted")
    padded = _pad_basis(basis_records)
    feature_names = [str(value) for value in config["map_source"]["source_features"]]
    feature_by_id = {
        str(row["plateifu"]): [float(row["source_features"][key]) for key in feature_names]
        for row in response_records
    }
    padded["feature_names"] = np.asarray(feature_names)
    padded["features"] = np.asarray(
        [feature_by_id[str(identity)] for identity in padded["plateifu"]], dtype=np.float64
    )
    paths["map_components"].parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(paths["map_components"], **padded)
    map_manifest = _content_hashed(
        {
            "schema_version": "invariant-gravity-item32-map-source-manifest-1.0",
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "sample_freeze_commit": config["sample_freeze_commit"],
            "data_model": config["sources"]["maps"]["data_model"],
            "files": files,
            "failures": failures,
            "map_components_sha256": _sha256_file(paths["map_components"]),
            "map_components_bytes": paths["map_components"].stat().st_size,
            "pixel_basis_schema": {
                "boundary": "galaxy,smoothing,pixel",
                "direction": "galaxy,niche,smoothing,pixel",
                "radius_cosine_sine_weight_annulus": "galaxy,pixel",
                "candidate_or_response_values_stored": 0,
            },
            "counts": {
                "exploration_maps_attempted": len(exploration),
                "exploration_maps_parsed": len(files),
                "exploration_map_failures": len(failures),
                "confirmation_maps_downloaded": 0,
                "post_response_formula_cells": 0,
                "paid_model_calls": 0,
            },
        }
    )
    responses = _content_hashed(
        {
            "schema_version": "invariant-gravity-item32-kinematic-responses-1.0",
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "sample_freeze_commit": config["sample_freeze_commit"],
            "records": response_records,
            "counts": {
                "source_complete": len(response_records),
                "stellar_complete": sum(
                    bool(row["stellar_quality_pass"]) for row in response_records
                ),
                "halpha_complete": sum(bool(row["halpha_quality_pass"]) for row in response_records),
                "confirmation_response_rows": 0,
                "post_response_formula_cells": 0,
            },
            "claims": {"confirmation_opened": False, "target_used_in_candidate_generation": False},
        }
    )
    _write_json(paths["map_source_manifest"], map_manifest)
    _write_json(paths["kinematic_responses"], responses)
    return paths


def _load_experiment_data(
    root: Path, config: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], dict[str, Any]]:
    paths = _source_paths(root, config)
    for key in ("predictor_source_manifest", "sample_manifest", "candidate_manifest"):
        manifest = _read_json(paths[key])
        _verify_content_hash(manifest, f"Item 32 {key}")
    map_manifest = _read_json(paths["map_source_manifest"])
    responses = _read_json(paths["kinematic_responses"])
    _verify_content_hash(map_manifest, "Item 32 map source manifest")
    _verify_content_hash(responses, "Item 32 kinematic responses")
    if map_manifest["counts"]["confirmation_maps_downloaded"] != 0:
        raise GravityItem32Error("confirmation map entered Item 32")
    if responses["counts"]["confirmation_response_rows"] != 0:
        raise GravityItem32Error("confirmation response entered Item 32")
    if _sha256_file(paths["map_components"]) != map_manifest["map_components_sha256"]:
        raise GravityItem32Error("map component archive changed")
    with np.load(paths["map_components"], allow_pickle=False) as archive:
        basis = {key: np.asarray(archive[key]) for key in archive.files}
    response_by_id = {str(row["plateifu"]): row for row in responses["records"]}
    predictor_by_id = {str(row["plateifu"]): row for row in _read_tsv(paths["predictors"])}
    basis_index = {str(value): index for index, value in enumerate(basis["plateifu"].tolist())}
    complete_ids = sorted(
        identity
        for identity, row in response_by_id.items()
        if bool(row["stellar_quality_pass"])
        and identity in basis_index
        and identity in predictor_by_id
    )
    rows = []
    indices = []
    for identity in complete_ids:
        predictor = predictor_by_id[identity]
        response = response_by_id[identity]
        if predictor["role"] != "exploration":
            raise GravityItem32Error("nonexploration identity entered complete data")
        rows.append(
            {
                **predictor,
                "source_features": response["source_features"],
                "stellar_angles": [
                    float(value["axial_angle_degrees"])
                    for value in response["stellar_annuli"]
                ],
                "halpha_quality_pass": bool(response["halpha_quality_pass"]),
                "halpha_angles": [
                    float(value["axial_angle_degrees"])
                    for value in response["halpha_annuli"]
                ]
                if bool(response["halpha_quality_pass"])
                else None,
            }
        )
        indices.append(basis_index[identity])
    subset: dict[str, np.ndarray] = {}
    for key, value in basis.items():
        if key in ("feature_names",):
            subset[key] = value
        else:
            subset[key] = value[np.asarray(indices, dtype=int)]
    return rows, subset, responses


def _candidate_prediction_matrix(
    config: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    basis: Mapping[str, np.ndarray],
    xp: Any,
) -> Any:
    candidate_count = len(arrays["niche"])
    galaxy_count = len(basis["plateifu"])
    result = xp.empty((candidate_count, galaxy_count, 2), dtype=xp.float32)
    batch_size = int(config["evaluation"]["candidate_batch_size"])
    for galaxy in range(galaxy_count):
        pixel_count = int(basis["pixel_count"][galaxy])
        boundary = xp.asarray(basis["boundary"][galaxy, :, :pixel_count])
        direction = xp.asarray(basis["direction"][galaxy, :, :, :pixel_count])
        radius = xp.maximum(xp.asarray(basis["radius"][galaxy, :pixel_count]), 0.05)
        cosine = xp.asarray(basis["cosine"][galaxy, :pixel_count])
        sine = xp.asarray(basis["sine"][galaxy, :pixel_count])
        weight = xp.asarray(basis["weight"][galaxy, :pixel_count])
        annulus = xp.asarray(basis["annulus"][galaxy, :pixel_count])
        inverse_normal = []
        for annulus_index in range(2):
            selected = annulus == annulus_index
            design = xp.stack((cosine[selected], sine[selected]), axis=1)
            weighted = weight[selected, None] * design
            inverse_normal.append(xp.linalg.inv(design.T @ weighted))
        for begin in range(0, candidate_count, batch_size):
            end = min(begin + batch_size, candidate_count)
            values = _candidate_values(config, arrays, begin, end, xp)
            candidate_boundary = boundary[values["smoothing"]]
            candidate_direction = direction[values["niche"], values["smoothing"]]
            activation = xp.power(
                candidate_boundary
                / (candidate_boundary + values["threshold"][:, None]),
                values["sharpness"][:, None] * values["shape"][:, None],
            )
            radial = xp.power(radius[None, :], values["radial"][:, None])
            field = (
                values["polarity"][:, None]
                * values["amplitude"][:, None]
                * activation
                * candidate_direction
                * radial
            )
            for annulus_index in range(2):
                selected = annulus == annulus_index
                rhs = xp.stack(
                    (
                        xp.sum(
                            field[:, selected]
                            * weight[selected][None, :]
                            * cosine[selected][None, :],
                            axis=1,
                        ),
                        xp.sum(
                            field[:, selected]
                            * weight[selected][None, :]
                            * sine[selected][None, :],
                            axis=1,
                        ),
                    ),
                    axis=1,
                )
                coefficient = rhs @ inverse_normal[annulus_index].T
                angle = xp.degrees(xp.arctan2(coefficient[:, 1], 1.0 + coefficient[:, 0]))
                result[begin:end, galaxy, annulus_index] = _wrap_axial_degrees(angle, xp)
    return result


def _fixed_structural_features(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> np.ndarray:
    normalization = config["evaluation"]["fixed_structural_normalization"]
    values = np.asarray(
        [
            [
                float(row["log_stellar_mass"]),
                float(row["log_half_light_radius"]),
                float(row["log_surface_density"]),
                float(row["sersic_index"]),
                float(row["axis_ratio"]),
                float(row["g_minus_r_color"]),
                float(row["redshift"]),
                float(row["log_snr"]),
            ]
            for row in rows
        ],
        dtype=np.float64,
    )
    keys = [
        "log_stellar_mass",
        "log_half_light_radius",
        "log_surface_density",
        "sersic_index",
        "axis_ratio",
        "g_minus_r_color",
        "redshift",
        "log_snr",
    ]
    center = np.asarray([float(normalization[key][0]) for key in keys])
    scale = np.asarray([float(normalization[key][1]) for key in keys])
    return (values - center) / scale


def _design_matrices(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    structural = _fixed_structural_features(rows, config)
    map_features = np.asarray(
        [
            [float(row["source_features"][key]) for key in config["map_source"]["source_features"]]
            for row in rows
        ],
        dtype=np.float64,
    )
    map_center = np.median(map_features, axis=0)
    map_scale = np.quantile(map_features, 0.75, axis=0) - np.quantile(
        map_features, 0.25, axis=0
    )
    map_scale = np.where(map_scale > 1e-8, map_scale, 1.0)
    normalized_map = np.clip((map_features - map_center) / map_scale, -8.0, 8.0)
    structural_designs = []
    flexible_designs = []
    for annulus in range(2):
        annulus_column = np.full((len(rows), 1), float(annulus))
        structural_design = np.column_stack(
            (np.ones(len(rows)), structural, annulus_column)
        )
        combined = np.column_stack((structural, normalized_map))
        interactions = np.column_stack(
            (
                structural[:, 0] * normalized_map[:, 5],
                structural[:, 3] * normalized_map[:, 4],
                structural[:, 4] * normalized_map[:, 6],
                structural[:, 2] * normalized_map[:, 7],
                structural[:, 0] * normalized_map[:, 8],
            )
        )
        flexible_design = np.column_stack(
            (np.ones(len(rows)), combined, combined**2, interactions, annulus_column)
        )
        structural_designs.append(structural_design)
        flexible_designs.append(flexible_design)
    return np.stack(structural_designs, axis=1), np.stack(flexible_designs, axis=1)


def _ridge_oof(
    y: np.ndarray, design: np.ndarray, folds: np.ndarray, alpha: float
) -> np.ndarray:
    prediction = np.empty_like(y, dtype=np.float64)
    for fold in sorted({int(value) for value in folds}):
        train = folds != fold
        held = folds == fold
        train_design = design[train].reshape(-1, design.shape[-1])
        train_y = y[train].reshape(-1)
        penalty = np.eye(train_design.shape[1]) * float(alpha)
        penalty[0, 0] = 0.0
        coefficient = np.linalg.solve(
            train_design.T @ train_design + penalty,
            train_design.T @ train_y,
        )
        prediction[held] = (design[held].reshape(-1, design.shape[-1]) @ coefficient).reshape(
            -1, 2
        )
    return np.asarray(_wrap_axial_degrees(prediction), dtype=np.float64)


def _angular_mse(
    y: np.ndarray, prediction: np.ndarray, indices: np.ndarray | None = None
) -> float:
    if indices is not None:
        y, prediction = y[indices], prediction[indices]
    difference = np.asarray(_wrap_axial_degrees(prediction - y), dtype=np.float64)
    return float(np.mean(difference**2))


def _nested_select(
    y: np.ndarray, candidate_predictions: Any, folds: np.ndarray, xp: Any
) -> tuple[np.ndarray, list[int]]:
    prediction = np.empty_like(y, dtype=np.float64)
    selected_cells = []
    y_device = xp.asarray(y)
    folds_device = xp.asarray(folds)
    for fold in sorted({int(value) for value in folds}):
        train = folds_device != fold
        difference = _wrap_axial_degrees(
            candidate_predictions[:, train, :] - y_device[None, train, :], xp
        )
        losses = xp.mean(difference**2, axis=(1, 2))
        selected = int(_to_numpy(xp.argmin(losses), xp))
        selected_cells.append(selected)
        held = folds == fold
        prediction[held] = _to_numpy(candidate_predictions[selected, held, :], xp)
    return prediction, selected_cells


def _candidate_record(
    index: int, config: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    values = _candidate_values(config, arrays, index, index + 1, np)
    niche = int(values["niche"][0])
    generator = config["candidate_generator"]
    return {
        "admissible_candidate_index": index,
        "niche_index": niche,
        "niche": generator["niches"][niche]["id"],
        "creativity_label": generator["niches"][niche]["creativity_label"],
        "conservative_status": generator["niches"][niche]["conservative_status"],
        "polarity": float(values["polarity"][0]),
        "amplitude": float(values["amplitude"][0]),
        "boundary_threshold": float(values["threshold"][0]),
        "sharpness": float(values["sharpness"][0]),
        "radial_exponent": float(values["radial"][0]),
        "shape_power": float(values["shape"][0]),
        "smoothing_sigma_spaxel": float(
            generator["smoothing_sigmas_spaxel"][int(values["smoothing"][0])]
        ),
    }


def _metric_block(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    mse = _angular_mse(y, prediction)
    zero_mse = _angular_mse(y, np.zeros_like(y))
    return {"mse_degrees2": mse, "r2_vs_zero_alignment": 1.0 - mse / zero_mse}


def _slice_metrics(
    rows: Sequence[Mapping[str, Any]],
    y: np.ndarray,
    candidate: np.ndarray,
    zero: np.ndarray,
    structural: np.ndarray,
    flexible: np.ndarray,
) -> dict[str, Any]:
    specifications = {
        "stellar_mass": np.asarray([float(row["log_stellar_mass"]) for row in rows]),
        "sersic_index": np.asarray([float(row["sersic_index"]) for row in rows]),
        "axis_ratio": np.asarray([float(row["axis_ratio"]) for row in rows]),
        "boundary_median": np.asarray(
            [float(row["source_features"]["boundary_median"]) for row in rows]
        ),
    }
    result = {}
    for name, values in specifications.items():
        median = float(np.median(values))
        for side, indices in (
            ("low", np.flatnonzero(values < median)),
            ("high", np.flatnonzero(values >= median)),
        ):
            candidate_mse = _angular_mse(y, candidate, indices)
            result[f"{name}_{side}"] = {
                "objects": len(indices),
                "candidate_mse_degrees2": candidate_mse,
                "improvement_vs_zero": _improvement(
                    _angular_mse(y, zero, indices), candidate_mse
                ),
                "improvement_vs_structural": _improvement(
                    _angular_mse(y, structural, indices), candidate_mse
                ),
                "improvement_vs_flexible": _improvement(
                    _angular_mse(y, flexible, indices), candidate_mse
                ),
            }
    return result


def _injection_controls(
    config: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    predictions: Any,
    folds: np.ndarray,
    xp: Any,
) -> dict[str, Any]:
    result = {}
    predictions_cpu = None
    for niche in range(4):
        indices = np.flatnonzero(arrays["niche"] == niche)
        amplitudes = np.asarray(config["candidate_generator"]["amplitudes"])[
            arrays["amplitude"][indices]
        ]
        target = int(indices[int(np.argmax(amplitudes))])
        if predictions_cpu is None:
            predictions_cpu = _to_numpy(predictions, xp)
        injected = np.asarray(predictions_cpu[target], dtype=np.float64)
        recovered, selected = _nested_select(injected, predictions, folds, xp)
        selected_niches = [int(arrays["niche"][index]) for index in selected]
        result[str(niche)] = {
            "target": _candidate_record(target, config, arrays),
            "selected_niches": selected_niches,
            "same_niche_folds": selected_niches.count(niche),
            "recovered": selected_niches.count(niche) >= 4
            and _angular_mse(injected, recovered) <= 1e-10,
        }
    return result


def _selection_aware_permutation(
    config: Mapping[str, Any],
    y: np.ndarray,
    predictions: Any,
    folds: np.ndarray,
    flexible_design: np.ndarray,
    observed_improvement: float,
    xp: Any,
) -> dict[str, Any]:
    random = np.random.Generator(np.random.PCG64(int(config["evaluation"]["permutation_seed"])))
    statistics = []
    for _ in range(int(config["evaluation"]["permutation_trials"])):
        permuted = y[random.permutation(len(y))]
        candidate, _ = _nested_select(permuted, predictions, folds, xp)
        flexible = _ridge_oof(
            permuted,
            flexible_design,
            folds,
            float(config["evaluation"]["ridge_alpha_flexible"]),
        )
        statistics.append(
            _improvement(_angular_mse(permuted, flexible), _angular_mse(permuted, candidate))
        )
    p_value = (1 + sum(value >= observed_improvement for value in statistics)) / (
        1 + len(statistics)
    )
    return {
        "trials": len(statistics),
        "observed_improvement_vs_flexible": observed_improvement,
        "p_value": p_value,
        "maximum_null_improvement": max(statistics),
        "null_improvements": statistics,
    }


def _evaluate(
    config: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    candidate_audit: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    basis: Mapping[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.perf_counter()
    try:
        xp, backend_name, device_name = _backend()
    except Exception as exc:
        raise GravityItem32Error(f"Item 32 requires the frozen CUDA lane: {exc}") from exc
    predictions = _candidate_prediction_matrix(config, arrays, basis, xp)
    xp.cuda.Stream.null.synchronize()
    prediction_seconds = time.perf_counter() - started
    maximum_actual_bend = float(xp.max(xp.abs(predictions)).get())
    crosscheck_count = min(
        int(config["evaluation"]["cpu_crosscheck_candidates"]), len(arrays["niche"])
    )
    crosscheck_galaxies = min(3, len(rows))
    cross_arrays = {key: value[:crosscheck_count] for key, value in arrays.items()}
    cross_basis = {
        key: value
        if key == "feature_names"
        else value[:crosscheck_galaxies]
        for key, value in basis.items()
    }
    cpu_predictions = _candidate_prediction_matrix(config, cross_arrays, cross_basis, np)
    gpu_crosscheck = _to_numpy(
        predictions[:crosscheck_count, :crosscheck_galaxies, :], xp
    )
    cpu_gpu_maximum_absolute_difference = float(
        np.max(np.abs(cpu_predictions - gpu_crosscheck))
    )

    y = np.asarray([row["stellar_angles"] for row in rows], dtype=np.float64)
    folds = np.asarray([int(row["outer_fold"]) for row in rows], dtype=int)
    structural_design, flexible_design = _design_matrices(rows, config)
    zero = np.zeros_like(y)
    structural = _ridge_oof(
        y,
        structural_design,
        folds,
        float(config["evaluation"]["ridge_alpha_structural"]),
    )
    flexible = _ridge_oof(
        y,
        flexible_design,
        folds,
        float(config["evaluation"]["ridge_alpha_flexible"]),
    )
    candidate, selected_cells = _nested_select(y, predictions, folds, xp)
    selected_records = [
        {"outer_fold": fold, **_candidate_record(index, config, arrays)}
        for fold, index in zip(sorted({int(value) for value in folds}), selected_cells)
    ]
    selected_niches = [int(arrays["niche"][index]) for index in selected_cells]
    niche_counts = Counter(selected_niches)
    stable_niche_folds = max(niche_counts.values())

    losses = {
        "zero_alignment": _angular_mse(y, zero),
        "structural": _angular_mse(y, structural),
        "flexible": _angular_mse(y, flexible),
        "candidate": _angular_mse(y, candidate),
    }
    improvements = {
        "vs_zero_alignment": _improvement(losses["zero_alignment"], losses["candidate"]),
        "vs_structural": _improvement(losses["structural"], losses["candidate"]),
        "vs_flexible": _improvement(losses["flexible"], losses["candidate"]),
    }
    slices = _slice_metrics(rows, y, candidate, zero, structural, flexible)
    permutation = _selection_aware_permutation(
        config,
        y,
        predictions,
        folds,
        flexible_design,
        improvements["vs_flexible"],
        xp,
    )
    injections = _injection_controls(config, arrays, predictions, folds, xp)
    zero_candidate, zero_selected = _nested_select(np.zeros_like(y), predictions, folds, xp)
    zero_control_mse = _angular_mse(np.zeros_like(y), zero_candidate)

    halpha_indices = np.asarray(
        [index for index, row in enumerate(rows) if bool(row["halpha_quality_pass"])],
        dtype=int,
    )
    halpha_transfer: dict[str, Any]
    if len(halpha_indices):
        halpha_y = np.asarray(
            [rows[index]["halpha_angles"] for index in halpha_indices], dtype=np.float64
        )
        halpha_folds = folds[halpha_indices]
        halpha_design = flexible_design[halpha_indices]
        halpha_flexible = _ridge_oof(
            halpha_y,
            halpha_design,
            halpha_folds,
            float(config["evaluation"]["ridge_alpha_flexible"]),
        )
        fold_order = sorted({int(value) for value in folds})
        selected_by_fold = dict(zip(fold_order, selected_cells))
        halpha_candidate = np.empty_like(halpha_y)
        for local_index, global_index in enumerate(halpha_indices):
            selected = selected_by_fold[int(folds[global_index])]
            halpha_candidate[local_index] = _to_numpy(
                predictions[selected, global_index, :], xp
            )
        halpha_zero = np.zeros_like(halpha_y)
        halpha_losses = {
            "zero_alignment": _angular_mse(halpha_y, halpha_zero),
            "flexible": _angular_mse(halpha_y, halpha_flexible),
            "candidate": _angular_mse(halpha_y, halpha_candidate),
        }
        halpha_transfer = {
            "objects": len(halpha_indices),
            "eligible": len(halpha_indices)
            >= int(config["sample"]["minimum_complete_halpha_galaxies_for_transfer"]),
            "candidate_reselected": False,
            "losses": halpha_losses,
            "improvements": {
                "vs_zero_alignment": _improvement(
                    halpha_losses["zero_alignment"], halpha_losses["candidate"]
                ),
                "vs_flexible": _improvement(
                    halpha_losses["flexible"], halpha_losses["candidate"]
                ),
            },
        }
    else:
        halpha_transfer = {
            "objects": 0,
            "eligible": False,
            "candidate_reselected": False,
            "losses": None,
            "improvements": None,
        }

    broad_slice_zero_pass = all(
        float(value["improvement_vs_zero"])
        >= float(config["gates"]["minimum_each_broad_slice_improvement_vs_zero"])
        for value in slices.values()
    )
    partial_slices = [
        name
        for name, value in slices.items()
        if float(value["improvement_vs_flexible"])
        >= float(config["gates"]["partial_minimum_slice_improvement_vs_flexible"])
    ]
    quality_pass = len(rows) >= int(config["sample"]["minimum_complete_stellar_galaxies"])
    quality_pass &= len(rows) / int(config["sample"]["expected_exploration"]) >= float(
        config["sample"]["minimum_stellar_quality_retention_fraction"]
    )
    injection_pass = all(bool(value["recovered"]) for value in injections.values())
    cpu_gpu_pass = cpu_gpu_maximum_absolute_difference <= 1e-4
    local_pass = (
        float(candidate_audit["maximum_admitted_local_fractional_direction_response"])
        <= float(config["admissibility"]["maximum_local_fractional_direction_response"])
    )
    actual_bend_pass = maximum_actual_bend <= float(
        config["admissibility"]["maximum_predicted_bend_degrees"]
    ) + 1e-5
    zero_control_pass = zero_control_mse > 1e-12
    stable_niche_pass = stable_niche_folds >= int(config["gates"]["minimum_same_niche_folds"])
    selected_majority_niche = niche_counts.most_common(1)[0][0]
    conservation_eligible = selected_majority_niche != 2
    halpha_pass = bool(halpha_transfer["eligible"])
    if halpha_transfer["eligible"]:
        halpha_pass &= float(halpha_transfer["improvements"]["vs_zero_alignment"]) >= float(
            config["gates"]["minimum_halpha_improvement_vs_zero"]
        )
        halpha_pass &= float(halpha_transfer["improvements"]["vs_flexible"]) >= float(
            config["gates"]["minimum_halpha_improvement_vs_flexible"]
        )
    universal_pass = all(
        (
            quality_pass,
            improvements["vs_zero_alignment"]
            >= float(config["gates"]["minimum_improvement_vs_zero_alignment"]),
            improvements["vs_structural"]
            >= float(config["gates"]["minimum_improvement_vs_structural"]),
            improvements["vs_flexible"]
            >= float(config["gates"]["minimum_improvement_vs_flexible"]),
            broad_slice_zero_pass,
            permutation["p_value"]
            <= float(config["gates"]["maximum_selection_aware_permutation_p"]),
            stable_niche_pass,
            injection_pass,
            zero_control_pass,
            cpu_gpu_pass,
            local_pass,
            actual_bend_pass,
            conservation_eligible,
            halpha_pass,
        )
    )
    phenomenon_pass = all(
        (
            quality_pass,
            improvements["vs_flexible"]
            >= float(config["gates"]["phenomenon_minimum_improvement_vs_flexible"]),
            permutation["p_value"]
            <= float(config["gates"]["phenomenon_maximum_selection_aware_p"]),
            stable_niche_pass,
            injection_pass,
            cpu_gpu_pass,
        )
    )
    partial_pass = quality_pass and bool(partial_slices)
    if not quality_pass:
        decision = "INCONCLUSIVE_QUALITY"
    elif universal_pass:
        decision = "PASS_EXPLORATION_BOTH_TRACKS"
    elif phenomenon_pass:
        decision = "UNIVERSAL_REJECT_PHENOMENON_LEAD"
    elif partial_pass:
        decision = "BOTH_FORMAL_TRACKS_NOT_PROMOTED_SCOPED_PARTIAL_PATTERN_RETAINED"
    else:
        decision = "SCOPED_ITEM32_REJECT"
    object_candidate_loss = np.mean(
        np.asarray(_wrap_axial_degrees(candidate - y), dtype=np.float64) ** 2, axis=1
    )
    object_flexible_loss = np.mean(
        np.asarray(_wrap_axial_degrees(flexible - y), dtype=np.float64) ** 2, axis=1
    )
    counterexample_ids = [
        str(rows[index]["plateifu"])
        for index in np.flatnonzero(object_candidate_loss >= object_flexible_loss)
    ]
    elapsed = time.perf_counter() - started
    nested_searches = 1 + int(config["evaluation"]["permutation_trials"]) + 4 + 1
    residual_evaluations = (
        len(arrays["niche"])
        * 2
        * len(rows)
        * (int(config["sample"]["outer_folds"]) - 1)
        * nested_searches
    )
    result = {
        "decision": decision,
        "quality": {
            "complete_stellar_galaxies": len(rows),
            "expected_exploration": int(config["sample"]["expected_exploration"]),
            "retention_fraction": len(rows) / int(config["sample"]["expected_exploration"]),
            "pass": quality_pass,
        },
        "stellar": {
            "losses": losses,
            "improvements": improvements,
            "metrics": {
                "zero_alignment": _metric_block(y, zero),
                "structural": _metric_block(y, structural),
                "flexible": _metric_block(y, flexible),
                "candidate": _metric_block(y, candidate),
            },
            "selected_cells": selected_records,
            "selected_niche_counts": {str(key): niche_counts[key] for key in range(4)},
            "same_niche_folds": stable_niche_folds,
            "slices": slices,
            "counterexamples_vs_flexible": {
                "count": len(counterexample_ids),
                "plateifu": counterexample_ids,
            },
        },
        "halpha_transfer": halpha_transfer,
        "selection_aware_permutation": permutation,
        "controls": {
            "synthetic_injections": injections,
            "synthetic_injections_pass": injection_pass,
            "zero_alignment": {
                "selected_cells": zero_selected,
                "candidate_mse_degrees2": zero_control_mse,
                "pass": zero_control_pass,
            },
            "local_limit_pass": local_pass,
            "response_blind_actual_bend_degrees": maximum_actual_bend,
            "actual_bend_pass": actual_bend_pass,
            "cpu_gpu_maximum_absolute_difference_degrees": cpu_gpu_maximum_absolute_difference,
            "cpu_gpu_pass": cpu_gpu_pass,
            "selected_majority_conservation_eligible": conservation_eligible,
        },
        "tracks": {
            "universal_gravity_pass": universal_pass,
            "phenomenon_publication_pass": phenomenon_pass,
            "scoped_partial_pass": partial_pass,
            "scoped_partial_slices": partial_slices,
            "paper_claim_allowed": False,
            "paper_claim_requires_unchanged_fresh_replication": True,
        },
        "gates": {
            "broad_slice_zero_pass": broad_slice_zero_pass,
            "stable_niche_pass": stable_niche_pass,
            "halpha_transfer_pass": halpha_pass,
            "confirmation_values_read": 0,
            "post_response_candidate_cells": 0,
        },
    }
    compute = _content_hashed(
        {
            "schema_version": "invariant-gravity-item32-compute-manifest-1.0",
            "backend": backend_name,
            "device": device_name,
            "candidate_prediction_seconds": prediction_seconds,
            "total_evaluation_seconds": elapsed,
            "candidate_count": len(arrays["niche"]),
            "galaxy_count": len(rows),
            "response_blind_pixel_field_evaluations": int(
                len(arrays["niche"]) * np.sum(basis["pixel_count"])
            ),
            "observed_and_null_residual_evaluations": residual_evaluations,
            "permutation_trials": int(config["evaluation"]["permutation_trials"]),
            "cpu_crosscheck_candidates": crosscheck_count,
            "cpu_crosscheck_galaxies": crosscheck_galaxies,
            "cpu_gpu_maximum_absolute_difference_degrees": cpu_gpu_maximum_absolute_difference,
            "paid_model_calls": 0,
            "paid_api_cost_usd": 0.0,
        }
    )
    return result, compute


def _build_receipt(
    root: Path,
    config: Mapping[str, Any],
    result: Mapping[str, Any],
    compute: Mapping[str, Any],
    candidate_audit: Mapping[str, Any],
    responses: Mapping[str, Any],
) -> dict[str, Any]:
    paths = _source_paths(root, config)
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item32-boundary-focusing-result-1.0",
            "item": 32,
            "title": config["title"],
            "hypothesis": config["hypothesis"],
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "sample_freeze_commit": config["sample_freeze_commit"],
            "stable_goal_path": config["stable_goal_path"],
            "stable_goal_sha256": config["stable_goal_sha256"],
            "decision": result["decision"],
            "candidate_audit": dict(candidate_audit),
            "evaluation": dict(result),
            "counts": {
                "exploration_roles": int(config["sample"]["expected_exploration"]),
                "stellar_complete": responses["counts"]["stellar_complete"],
                "halpha_complete": responses["counts"]["halpha_complete"],
                "raw_candidate_cells": candidate_audit["raw_candidates"],
                "admissible_candidate_cells": candidate_audit["admissible_candidates"],
                "behavioral_equivalence_classes": candidate_audit[
                    "behavioral_equivalence_classes_adversarial"
                ],
                "confirmation_values_read": 0,
                "confirmation_maps_downloaded": 0,
                "post_response_candidate_cells": 0,
                "paid_model_calls": 0,
            },
            "artifacts": {
                key: {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": _sha256_file(path),
                }
                for key, path in paths.items()
                if key != "compute_manifest" and path.exists()
            }
            | {
                "compute_manifest": {
                    "path": paths["compute_manifest"].relative_to(root).as_posix(),
                    "sha256": _sha256_file(paths["compute_manifest"]),
                    "content_sha256": compute["content_sha256"],
                }
            },
            "cost": {"paid_model_calls": 0, "paid_api_cost_usd": 0.0},
            "scope": {
                "claim_ceiling": config["scope"]["claim_ceiling"],
                "confirmation_opened": False,
                "historical_novelty_claimed": False,
                "paper_claim_allowed": False,
            },
        }
    )


def run_experiment(root: Path) -> Path:
    root = root.resolve()
    config = load_config(root)
    verify_science_freeze(root, config)
    verify_sample_freeze(root, config)
    paths = _source_paths(root, config)
    arrays, candidate_audit = _admissible_candidates(config)
    rows, basis, responses = _load_experiment_data(root, config)
    result, compute = _evaluate(config, arrays, candidate_audit, rows, basis)
    _write_json(paths["compute_manifest"], compute)
    receipt = _build_receipt(root, config, result, compute, candidate_audit, responses)
    result_path = root / str(config["paths"]["result"])
    _write_json(result_path, receipt)
    return result_path


def validate_checked(root: Path) -> None:
    root = root.resolve()
    config = load_config(root)
    verify_science_freeze(root, config)
    verify_sample_freeze(root, config)
    paths = _source_paths(root, config)
    for key in (
        "predictor_source_manifest",
        "sample_manifest",
        "candidate_manifest",
        "map_source_manifest",
        "kinematic_responses",
        "compute_manifest",
    ):
        manifest = _read_json(paths[key])
        _verify_content_hash(manifest, f"Item 32 {key}")
    candidate_manifest = _read_json(paths["candidate_manifest"])
    arrays, audit = _admissible_candidates(config)
    if candidate_manifest["audit"] != audit:
        raise GravityItem32Error("candidate audit differs from frozen manifest")
    if len(arrays["niche"]) != int(config["candidate_generator"]["expected_admissible_candidates"]):
        raise GravityItem32Error("admissible candidate count changed")
    map_manifest = _read_json(paths["map_source_manifest"])
    responses = _read_json(paths["kinematic_responses"])
    compute = _read_json(paths["compute_manifest"])
    if map_manifest["counts"]["confirmation_maps_downloaded"] != 0:
        raise GravityItem32Error("confirmation map count is nonzero")
    if responses["counts"]["confirmation_response_rows"] != 0:
        raise GravityItem32Error("confirmation response count is nonzero")
    if compute["paid_model_calls"] != 0 or compute["paid_api_cost_usd"] != 0.0:
        raise GravityItem32Error("paid call entered Item 32")
    if _sha256_file(paths["map_components"]) != map_manifest["map_components_sha256"]:
        raise GravityItem32Error("map component archive changed")
    result_path = root / str(config["paths"]["result"])
    receipt = _read_json(result_path)
    _verify_content_hash(receipt, "Item 32 result")
    if receipt["counts"]["confirmation_values_read"] != 0:
        raise GravityItem32Error("confirmation value entered result")
    if receipt["counts"]["post_response_candidate_cells"] != 0:
        raise GravityItem32Error("post-response candidate entered result")
    if receipt["cost"]["paid_api_cost_usd"] != 0.0:
        raise GravityItem32Error("paid cost entered result")
    for key, binding in receipt["artifacts"].items():
        artifact = root / str(binding["path"])
        if _sha256_file(artifact) != str(binding["sha256"]):
            raise GravityItem32Error(f"result artifact changed: {key}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare-predictors")
    subparsers.add_parser("acquire-maps")
    subparsers.add_parser("run")
    subparsers.add_parser("validate-checked")
    subparsers.add_parser("show-candidates")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.root.resolve()
    if args.command == "prepare-predictors":
        print(prepare_predictors(root)["sample_manifest"])
    elif args.command == "acquire-maps":
        print(acquire_maps(root)["kinematic_responses"])
    elif args.command == "run":
        print(run_experiment(root))
    elif args.command == "validate-checked":
        validate_checked(root)
        print("PASS")
    elif args.command == "show-candidates":
        _, audit = _admissible_candidates(load_config(root))
        print(json.dumps(audit, sort_keys=True, indent=2))
    else:
        raise GravityItem32Error(f"unexpected command: {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
