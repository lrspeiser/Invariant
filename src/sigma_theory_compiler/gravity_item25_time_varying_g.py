"""Frozen Item 25 search for cosmic or local-clock evolution of gravitational strength."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from sigma_theory_compiler.gravity_item16_s4tm_qed_field import _parse_vizier_tsv
from sigma_theory_compiler.gravity_item22_polarization_superposition import (
    _backend,
    _download,
    _read_tsv,
    _to_numpy,
    _write_tsv,
)
from sigma_theory_compiler.gravity_item24_temporal_lapse import _hmac_rank, _query_url

CONFIG_PATH = Path("configs/gravity_item25_time_varying_g_v1.json")
MODULE_PATH = Path("src/sigma_theory_compiler/gravity_item25_time_varying_g.py")
GOAL_PATH = Path("docs/GRAVITY_HIDDEN_VARIABLE_AND_THEORY_SEARCH_GOALS.md")


class GravityItem25Error(RuntimeError):
    """Raised when an Item 25 freeze, leakage, or replay invariant is violated."""


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _content_hashed(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result.pop("content_sha256", None)
    result["content_sha256"] = _sha256_bytes(_canonical_bytes(result))
    return result


def _verify_content_hash(payload: Mapping[str, Any], label: str) -> None:
    expected = payload.get("content_sha256")
    if not isinstance(expected, str):
        raise GravityItem25Error(f"{label} has no content hash")
    body = dict(payload)
    body.pop("content_sha256", None)
    if _sha256_bytes(_canonical_bytes(body)) != expected:
        raise GravityItem25Error(f"{label} content hash changed")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(payload))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise GravityItem25Error(f"expected JSON object: {path}")
    return value


def _git(root: Path, *args: str, text_mode: bool = True) -> str | bytes:
    result = subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=text_mode
    )
    return result.stdout.strip() if text_mode else result.stdout


def _require_ancestor(root: Path, commit: str, label: str) -> None:
    if commit.startswith("TO_BE_BOUND"):
        raise GravityItem25Error(f"{label} has not been bound")
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise GravityItem25Error(f"{label} is not an ancestor of HEAD")


def load_config(root: Path) -> dict[str, Any]:
    config = _read_json(root / CONFIG_PATH)
    if (
        config.get("schema_version")
        != "invariant-gravity-item25-time-varying-g-config-1.0"
        or int(config.get("item", -1)) != 25
    ):
        raise GravityItem25Error("unexpected Item 25 config")
    if _sha256_file(root / GOAL_PATH) != str(config["stable_goal_sha256"]):
        raise GravityItem25Error("stable gravity goal changed")
    generator = config["candidate_generator"]
    if int(generator["raw_candidate_cells"]) != 262144:
        raise GravityItem25Error("raw candidate boundary changed")
    if int(generator["post_response_cells"]) != 0:
        raise GravityItem25Error("post-response candidates entered Item 25")
    if bool(config["scope"]["confirmation_opening_authorized"]):
        raise GravityItem25Error("confirmation opening is not authorized")
    if bool(config["scope"]["paid_api_calls_authorized"]):
        raise GravityItem25Error("paid calls are outside Item 25")
    policy = config["discovery_policy"]
    if not bool(policy["equal_initial_viability"]):
        raise GravityItem25Error("equal-viability policy changed")
    if not bool(policy["age_or_history_is_not_privileged"]):
        raise GravityItem25Error("age or history was privileged")
    for relative, digest in config["dependency_sha256"].items():
        if _sha256_file(root / str(relative)) != str(digest):
            raise GravityItem25Error(f"scientific dependency changed: {relative}")
    return config


def _contract_digest(config: Mapping[str, Any]) -> str:
    value = json.loads(json.dumps(config))
    value["scientific_freeze_commit"] = "<BOUND_COMMIT>"
    value["sample_freeze_commit"] = "<BOUND_COMMIT>"
    return _sha256_bytes(_canonical_bytes(value))


def verify_science_freeze(root: Path, config: Mapping[str, Any]) -> None:
    commit = str(config["scientific_freeze_commit"])
    _require_ancestor(root, commit, "scientific freeze")
    frozen = json.loads(str(_git(root, "show", f"{commit}:{CONFIG_PATH.as_posix()}")))
    if _contract_digest(frozen) != _contract_digest(config):
        raise GravityItem25Error("scientific contract differs from frozen commit")
    module = _git(root, "show", f"{commit}:{MODULE_PATH.as_posix()}", text_mode=False)
    if not isinstance(module, bytes) or _sha256_bytes(module) != _sha256_file(root / MODULE_PATH):
        raise GravityItem25Error("Item 25 module differs from scientific freeze")


def _source_paths(root: Path, config: Mapping[str, Any]) -> dict[str, Path]:
    base = root / str(config["paths"]["source_dir"])
    keys = (
        "predictors",
        "predictor_source_manifest",
        "sample_manifest",
        "candidate_manifest",
        "exploration_responses",
        "response_source_manifest",
        "compute_manifest",
    )
    return {key: base / str(config["paths"][key]) for key in keys}


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
        repo_path = paths[key].relative_to(root).as_posix()
        frozen = _git(root, "show", f"{commit}:{repo_path}", text_mode=False)
        if not isinstance(frozen, bytes) or _sha256_bytes(frozen) != _sha256_file(paths[key]):
            raise GravityItem25Error(f"{key} differs from sample freeze")


def generate_raw_candidates(config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    generator = config["candidate_generator"]
    count = int(generator["raw_candidate_cells"])
    per = int(config["discovery_policy"]["equal_raw_capacity_per_mechanism"])
    if count != 4 * per:
        raise GravityItem25Error("mechanism capacity is not equal")
    random = np.random.Generator(np.random.PCG64(int(generator["seed"])))
    arrays: dict[str, np.ndarray] = {
        "niche": np.repeat(np.arange(4, dtype=np.int8), per),
    }
    for key, values in (
        ("amplitude", generator["amplitudes"]),
        ("polarity", generator["polarities"]),
        ("redshift_scale", generator["redshift_scales"]),
        ("transition_width", generator["transition_widths"]),
        ("power", generator["powers"]),
        ("maturity_threshold", generator["maturity_thresholds"]),
        ("relaxation_multiplier", generator["relaxation_multipliers"]),
        ("log_frequency", generator["log_frequencies"]),
        ("phase", generator["phases_rad"]),
    ):
        arrays[key] = random.integers(0, len(values), count, dtype=np.int16)
    return arrays


def _candidate_values(
    config: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    begin: int,
    end: int,
    xp: Any,
) -> dict[str, Any]:
    generator = config["candidate_generator"]
    result = {"niche": xp.asarray(arrays["niche"][begin:end])}
    for array_key, config_key in (
        ("amplitude", "amplitudes"),
        ("polarity", "polarities"),
        ("redshift_scale", "redshift_scales"),
        ("transition_width", "transition_widths"),
        ("power", "powers"),
        ("maturity_threshold", "maturity_thresholds"),
        ("relaxation_multiplier", "relaxation_multipliers"),
        ("log_frequency", "log_frequencies"),
        ("phase", "phases_rad"),
    ):
        choices = xp.asarray(generator[config_key], dtype=xp.float64)
        result[array_key] = choices[xp.asarray(arrays[array_key][begin:end])]
    return result


def _raw_candidate_digest(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(arrays):
        digest.update(key.encode())
        digest.update(np.ascontiguousarray(arrays[key]).tobytes())
    return digest.hexdigest()


def _sigmoid(value: Any, xp: Any) -> Any:
    clipped = xp.clip(value, -60.0, 60.0)
    return 1.0 / (1.0 + xp.exp(-clipped))


def _log_mu(
    values: Mapping[str, Any], z: Any, stellar_fraction: Any, xp: Any
) -> Any:
    """Return natural log(G_eff/G0), candidates by objects."""
    z = xp.asarray(z, dtype=xp.float64)[None, :]
    fraction = xp.asarray(stellar_fraction, dtype=xp.float64)[None, :]
    niche = values["niche"][:, None]
    signed_amplitude = values["polarity"][:, None] * values["amplitude"][:, None]
    scale = values["redshift_scale"][:, None]
    width = values["transition_width"][:, None]
    power = values["power"][:, None]

    scale_factor = 1.0 / (1.0 + z)
    smooth = signed_amplitude * (1.0 - scale_factor**power)

    transition = signed_amplitude * (
        _sigmoid((z - scale) / width, xp) - _sigmoid(-scale / width, xp)
    )

    age_fraction = (1.0 + z) ** -1.5
    maturity_floor = values["maturity_threshold"][:, None]
    relaxation = values["relaxation_multiplier"][:, None]
    local_denominator = relaxation * (xp.maximum(1.0 - fraction, 0.0) + maturity_floor)
    settled = 1.0 - xp.exp(-age_fraction / xp.maximum(local_denominator, 1e-12))
    mature_reference = 1.0 - xp.exp(
        -1.0 / xp.maximum(relaxation * maturity_floor, 1e-12)
    )
    local = signed_amplitude * (settled - mature_reference)

    envelope = z / (z + scale)
    oscillatory = signed_amplitude * envelope * xp.sin(
        values["log_frequency"][:, None] * xp.log1p(z) + values["phase"][:, None]
    )
    return xp.where(
        niche == 0,
        smooth,
        xp.where(niche == 1, transition, xp.where(niche == 2, local, oscillatory)),
    )


def _admissible_candidates(
    config: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    raw = generate_raw_candidates(config)
    physics = config["physics"]
    batch = int(config["evaluation"]["candidate_batch_size"])
    keep = np.zeros(len(raw["niche"]), dtype=bool)
    current_derivative = np.full(len(raw["niche"]), np.nan)
    bbn_mu = np.full(len(raw["niche"]), np.nan)
    recombination_mu = np.full(len(raw["niche"]), np.nan)
    history_min = np.full(len(raw["niche"]), np.nan)
    history_max = np.full(len(raw["niche"]), np.nan)
    z_small = 1e-6
    history_z = np.concatenate(
        [np.asarray([0.0, z_small]), np.logspace(-4.0, 9.0, 96)]
    )
    maturity_grid = np.resize(np.asarray([0.0, 0.2, 0.5, 1.0]), len(history_z))
    for begin in range(0, len(raw["niche"]), batch):
        end = min(begin + batch, len(raw["niche"]))
        values = _candidate_values(config, raw, begin, end, np)
        log_history = _log_mu(values, history_z, maturity_grid, np)
        mu_history = np.exp(log_history)
        log_present = _log_mu(values, np.asarray([0.0, z_small]), np.ones(2), np)
        derivative = (
            np.abs(log_present[:, 1] - log_present[:, 0])
            * float(physics["H0_per_year"])
            / z_small
        )
        early_z = np.asarray(
            [float(physics["recombination_redshift"]), float(physics["bbn_redshift"])]
        )
        early_mu = np.exp(_log_mu(values, early_z, np.zeros(2), np))
        local_ok = derivative <= float(
            physics["maximum_absolute_present_dotG_over_G_per_year"]
        )
        recombination_ok = (
            np.abs(early_mu[:, 0] - 1.0)
            <= float(physics["maximum_absolute_recombination_fractional_change"])
        )
        bbn_ok = (early_mu[:, 1] >= float(physics["bbn_mu_minimum"])) & (
            early_mu[:, 1] <= float(physics["bbn_mu_maximum"])
        )
        domain_ok = np.all(np.isfinite(mu_history), axis=1) & np.all(
            (mu_history >= float(physics["minimum_mu_on_history_grid"]))
            & (mu_history <= float(physics["maximum_mu_on_history_grid"])),
            axis=1,
        )
        keep[begin:end] = local_ok & recombination_ok & bbn_ok & domain_ok
        current_derivative[begin:end] = derivative
        recombination_mu[begin:end] = early_mu[:, 0]
        bbn_mu[begin:end] = early_mu[:, 1]
        history_min[begin:end] = np.min(mu_history, axis=1)
        history_max[begin:end] = np.max(mu_history, axis=1)
    arrays = {key: value[keep] for key, value in raw.items()}
    counts = Counter(int(value) for value in arrays["niche"])
    return arrays, {
        "raw_cells": len(raw["niche"]),
        "raw_niche_counts": {
            str(index): int(np.count_nonzero(raw["niche"] == index)) for index in range(4)
        },
        "admissible_cells": len(arrays["niche"]),
        "admissible_niche_counts": {
            str(index): counts.get(index, 0) for index in range(4)
        },
        "raw_candidate_digest": _raw_candidate_digest(raw),
        "admissible_candidate_digest": _raw_candidate_digest(arrays),
        "maximum_admitted_absolute_dotG_over_G_per_year": float(
            np.max(current_derivative[keep])
        ),
        "admitted_bbn_mu_range": [float(np.min(bbn_mu[keep])), float(np.max(bbn_mu[keep]))],
        "admitted_recombination_mu_range": [
            float(np.min(recombination_mu[keep])),
            float(np.max(recombination_mu[keep])),
        ],
        "admitted_history_mu_range": [
            float(np.min(history_min[keep])),
            float(np.max(history_max[keep])),
        ],
        "filters_are_response_independent": True,
    }


def _candidate_manifest(config: Mapping[str, Any]) -> dict[str, Any]:
    _, audit = _admissible_candidates(config)
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item25-time-varying-g-candidates-1.0",
            "generator": config["candidate_generator"],
            "physics_gates": config["physics"],
            "audit": audit,
            "responses_open_when_generated": False,
            "post_response_candidate_cells": 0,
        }
    )


def _predictor_rows(raw_rows: Sequence[Mapping[str, str]], config: Mapping[str, Any]) -> list[dict[str, Any]]:
    quality = config["quality"]
    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        try:
            sequence = int(raw["Seq"])
            redshift = float(raw["z"])
            log_stellar = float(raw["logM*"])
            log_baryonic = float(raw["logMb"])
            sigma0 = float(raw["sigma0"])
        except (KeyError, TypeError, ValueError):
            continue
        fraction = 10.0 ** (log_stellar - log_baryonic)
        if not (
            float(quality["minimum_redshift"])
            <= redshift
            <= float(quality["maximum_redshift"])
            and float(quality["minimum_log_baryonic_mass"])
            <= log_baryonic
            <= float(quality["maximum_log_baryonic_mass"])
            and float(quality["minimum_stellar_fraction"])
            <= fraction
            <= float(quality["maximum_stellar_fraction"])
            and sigma0 > 0.0
        ):
            continue
        rows.append(
            {
                "sequence": sequence,
                "redshift": redshift,
                "log_stellar_mass": log_stellar,
                "log_baryonic_mass": log_baryonic,
                "stellar_fraction": fraction,
                "sigma0_km_s": sigma0,
            }
        )
    return sorted(rows, key=lambda row: int(row["sequence"]))


def _build_sample(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    sample = config["sample"]
    strata = int(sample["redshift_strata"])
    ordered = sorted(rows, key=lambda row: (float(row["redshift"]), int(row["sequence"])))
    groups = np.array_split(np.asarray(ordered, dtype=object), strata)
    objects: list[dict[str, Any]] = []
    for stratum, group_values in enumerate(groups):
        group = [dict(value) for value in group_values.tolist()]
        ranked = sorted(
            group,
            key=lambda row: _hmac_rank(str(sample["role_key"]), f"km3d:{row['sequence']}"),
        )
        confirmation_count = int(sample["confirmation_per_stratum"])
        confirmations = {int(row["sequence"]) for row in ranked[:confirmation_count]}
        exploration = [row for row in ranked if int(row["sequence"]) not in confirmations]
        exploration = sorted(
            exploration,
            key=lambda row: _hmac_rank(str(sample["fold_key"]), f"km3d:{row['sequence']}"),
        )
        fold_by_sequence = {
            int(row["sequence"]): int((index + stratum) % int(sample["outer_folds"]))
            for index, row in enumerate(exploration)
        }
        for row in group:
            identity = int(row["sequence"])
            role = "confirmation" if identity in confirmations else "exploration"
            objects.append(
                {
                    "identity": identity,
                    "role": role,
                    "redshift_stratum": stratum,
                    "outer_fold": None if role == "confirmation" else fold_by_sequence[identity],
                    "role_rank_sha256": _hmac_rank(
                        str(sample["role_key"]), f"km3d:{identity}"
                    ),
                }
            )
    role_counts = Counter(str(row["role"]) for row in objects)
    fold_counts = Counter(
        int(row["outer_fold"])
        for row in objects
        if row["role"] == "exploration"
    )
    if len(objects) != int(sample["expected_selected"]):
        raise GravityItem25Error(f"selected {len(objects)} predictor rows")
    if role_counts["exploration"] != int(sample["expected_exploration"]):
        raise GravityItem25Error("unexpected exploration count")
    if role_counts["confirmation"] != int(sample["expected_confirmation"]):
        raise GravityItem25Error("unexpected confirmation count")
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item25-time-varying-g-sample-1.0",
            "selection_rule": sample["rule"],
            "response_columns_read": [],
            "confirmation_response_values_read": 0,
            "objects": sorted(objects, key=lambda row: int(row["identity"])),
            "role_counts": dict(sorted(role_counts.items())),
            "fold_counts": {str(key): value for key, value in sorted(fold_counts.items())},
        }
    )


def prepare_predictors(root: Path) -> dict[str, Path]:
    config = load_config(root)
    verify_science_freeze(root, config)
    paths = _source_paths(root, config)
    paths["predictors"].parent.mkdir(parents=True, exist_ok=True)
    url = _query_url(str(config["sources"]["catalog"]), config["sources"]["predictor_columns"])
    body, headers = _download(url)
    raw_rows = _parse_vizier_tsv(body, config["sources"]["predictor_columns"])
    if len(raw_rows) != int(config["sources"]["expected_catalog_rows"]):
        raise GravityItem25Error(f"catalog row count changed: {len(raw_rows)}")
    rows = _predictor_rows(raw_rows, config)
    if len(rows) != int(config["sample"]["expected_selected"]):
        raise GravityItem25Error(f"predictor-valid count changed: {len(rows)}")
    columns = [
        "sequence",
        "redshift",
        "log_stellar_mass",
        "log_baryonic_mass",
        "stellar_fraction",
        "sigma0_km_s",
    ]
    _write_tsv(paths["predictors"], rows, columns)
    predictor_manifest = _content_hashed(
        {
            "schema_version": "invariant-gravity-item25-time-varying-g-predictors-1.0",
            "source": config["sources"]["catalog"],
            "url": url,
            "source_sha256": _sha256_bytes(body),
            "source_bytes": len(body),
            "etag": headers.get("etag"),
            "last_modified": headers.get("last-modified"),
            "predictor_columns_queried": config["sources"]["predictor_columns"],
            "response_columns_queried": [],
            "raw_rows": len(raw_rows),
            "valid_rows": len(rows),
            "predictor_file": {
                "path": paths["predictors"].relative_to(root).as_posix(),
                "sha256": _sha256_file(paths["predictors"]),
            },
        }
    )
    sample_manifest = _build_sample(rows, config)
    candidate_manifest = _candidate_manifest(config)
    _write_json(paths["predictor_source_manifest"], predictor_manifest)
    _write_json(paths["sample_manifest"], sample_manifest)
    _write_json(paths["candidate_manifest"], candidate_manifest)
    return paths


def acquire_responses(root: Path) -> Path:
    config = load_config(root)
    verify_science_freeze(root, config)
    verify_sample_freeze(root, config)
    paths = _source_paths(root, config)
    sample = _read_json(paths["sample_manifest"])
    _verify_content_hash(sample, "sample manifest")
    identities = [
        int(row["identity"]) for row in sample["objects"] if row["role"] == "exploration"
    ]

    def fetch(identity: int) -> tuple[dict[str, Any], dict[str, Any]]:
        url = _query_url(
            str(config["sources"]["catalog"]),
            config["sources"]["response_columns"],
            Seq=identity,
        )
        body, headers = _download(url)
        raw = _parse_vizier_tsv(body, config["sources"]["response_columns"])
        if len(raw) != 1 or int(raw[0]["Seq"]) != identity:
            raise GravityItem25Error(f"response query for {identity} returned {len(raw)} rows")
        row = {"sequence": identity, "vcirc_km_s": float(raw[0]["Vcirc"])}
        receipt = {
            "identity": identity,
            "url": url,
            "sha256": _sha256_bytes(body),
            "bytes": len(body),
            "etag": headers.get("etag"),
            "last_modified": headers.get("last-modified"),
        }
        return row, receipt

    responses: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=12) as executor:
        for row, receipt in executor.map(fetch, identities):
            responses.append(row)
            receipts.append(receipt)
    responses.sort(key=lambda row: int(row["sequence"]))
    receipts.sort(key=lambda row: int(row["identity"]))
    _write_tsv(paths["exploration_responses"], responses, ["sequence", "vcirc_km_s"])
    manifest = _content_hashed(
        {
            "schema_version": "invariant-gravity-item25-time-varying-g-responses-1.0",
            "response_columns_queried": config["sources"]["response_columns"],
            "query_scope": "one exact VizieR Seq query per frozen exploration identity",
            "exploration_values_read": len(responses),
            "confirmation_values_read": 0,
            "response_file": {
                "path": paths["exploration_responses"].relative_to(root).as_posix(),
                "sha256": _sha256_file(paths["exploration_responses"]),
            },
            "source_receipts": receipts,
        }
    )
    _write_json(paths["response_source_manifest"], manifest)
    return paths["response_source_manifest"]


def _load_rows(
    root: Path, config: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    paths = _source_paths(root, config)
    predictors = {int(row["sequence"]): row for row in _read_tsv(paths["predictors"])}
    responses = {
        int(row["sequence"]): float(row["vcirc_km_s"])
        for row in _read_tsv(paths["exploration_responses"])
    }
    sample = _read_json(paths["sample_manifest"])
    response_manifest = _read_json(paths["response_source_manifest"])
    _verify_content_hash(sample, "sample manifest")
    _verify_content_hash(response_manifest, "response manifest")
    rows: list[dict[str, Any]] = []
    quality = config["quality"]
    for role in sample["objects"]:
        if role["role"] != "exploration":
            continue
        identity = int(role["identity"])
        predictor = predictors[identity]
        velocity = responses[identity]
        sigma = float(predictor["sigma0_km_s"])
        gates = {
            "velocity_range": float(quality["minimum_vcirc_km_s"])
            <= velocity
            <= float(quality["maximum_vcirc_km_s"]),
            "rotation_dominated": velocity / sigma
            >= float(quality["minimum_rotation_to_dispersion"]),
        }
        if all(gates.values()):
            rows.append(
                {
                    "identity": identity,
                    "fold": int(role["outer_fold"]),
                    "redshift_stratum": int(role["redshift_stratum"]),
                    "redshift": float(predictor["redshift"]),
                    "log_stellar_mass": float(predictor["log_stellar_mass"]),
                    "log_baryonic_mass": float(predictor["log_baryonic_mass"]),
                    "stellar_fraction": float(predictor["stellar_fraction"]),
                    "sigma0_km_s": sigma,
                    "vcirc_km_s": velocity,
                    "quality_gates": gates,
                }
            )
    if len(rows) < int(config["sample"]["minimum_valid_exploration"]):
        raise GravityItem25Error(
            f"only {len(rows)} exploration rows pass quality; minimum is "
            f"{config['sample']['minimum_valid_exploration']}"
        )
    return sorted(rows, key=lambda row: int(row["identity"])), response_manifest


def _base_design(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    mass = np.asarray([float(row["log_baryonic_mass"]) for row in rows]) - 10.7
    return np.column_stack([np.ones(len(rows)), mass])


def _flex_design(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    mass = np.asarray([float(row["log_baryonic_mass"]) for row in rows]) - 10.7
    fraction = np.asarray([float(row["stellar_fraction"]) for row in rows])
    sigma = np.log10(np.asarray([float(row["sigma0_km_s"]) for row in rows])) - 1.7
    redshift = np.asarray([float(row["redshift"]) for row in rows]) - 1.5
    return np.column_stack(
        [
            mass,
            mass**2,
            fraction,
            sigma,
            redshift,
            redshift**2,
            mass * redshift,
            fraction * redshift,
            sigma * redshift,
        ]
    )


def _linear_predict(
    design: np.ndarray, target: np.ndarray, train: np.ndarray, test: np.ndarray
) -> np.ndarray:
    coefficient = np.linalg.lstsq(design[train], target[train], rcond=None)[0]
    return design[test] @ coefficient


def _ridge_predict(
    design: np.ndarray,
    target: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    alpha: float,
) -> np.ndarray:
    mean = np.mean(design[train], axis=0)
    scale = np.std(design[train], axis=0)
    scale[scale < 1e-12] = 1.0
    train_x = (design[train] - mean) / scale
    test_x = (design[test] - mean) / scale
    train_x = np.column_stack([np.ones(len(train_x)), train_x])
    test_x = np.column_stack([np.ones(len(test_x)), test_x])
    penalty = np.eye(train_x.shape[1]) * alpha
    penalty[0, 0] = 0.0
    coefficient = np.linalg.solve(train_x.T @ train_x + penalty, train_x.T @ target[train])
    return test_x @ coefficient


def _build_term_matrix(
    config: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    rows: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    z = np.asarray([float(row["redshift"]) for row in rows])
    fraction = np.asarray([float(row["stellar_fraction"]) for row in rows])
    batch = int(config["evaluation"]["candidate_batch_size"])
    pieces: list[np.ndarray] = []
    for begin in range(0, len(arrays["niche"]), batch):
        end = min(begin + batch, len(arrays["niche"]))
        values = _candidate_values(config, arrays, begin, end, np)
        pieces.append(0.5 * _log_mu(values, z, fraction, np) / math.log(10.0))
    return np.concatenate(pieces, axis=0)


def _fit_candidate_predictions(
    xp: Any,
    base: np.ndarray,
    target: np.ndarray,
    terms: Any,
    train: np.ndarray,
    test: np.ndarray,
) -> Any:
    train_x = xp.asarray(base[train])
    test_x = xp.asarray(base[test])
    train_y = xp.asarray(target[train])
    pseudo = xp.linalg.pinv(train_x)
    coefficient = (train_y[None, :] - terms[:, train]) @ pseudo.T
    return coefficient @ test_x.T + terms[:, test]


def _select_candidate(
    xp: Any,
    config: Mapping[str, Any],
    base: np.ndarray,
    target: np.ndarray,
    folds: np.ndarray,
    outer_fold: int,
    term_matrix: np.ndarray,
) -> tuple[int, int]:
    candidate_count = term_matrix.shape[0]
    batch = int(config["evaluation"]["candidate_batch_size"])
    best_score = float("inf")
    best_index = -1
    residual_evaluations = 0
    inner_folds = sorted({int(value) for value in folds if int(value) != outer_fold})
    for begin in range(0, candidate_count, batch):
        end = min(begin + batch, candidate_count)
        terms = xp.asarray(term_matrix[begin:end])
        scores = xp.zeros(end - begin, dtype=xp.float64)
        points = 0
        for inner in inner_folds:
            validation = np.where(folds == inner)[0]
            training = np.where((folds != outer_fold) & (folds != inner))[0]
            prediction = _fit_candidate_predictions(
                xp, base, target, terms, training, validation
            )
            residual = prediction - xp.asarray(target[validation])[None, :]
            scores += xp.sum(residual**2, axis=1)
            points += len(validation)
            residual_evaluations += (end - begin) * len(validation)
        scores /= max(points, 1)
        local = int(_to_numpy(xp.argmin(scores), xp))
        score = float(_to_numpy(scores[local], xp))
        if score < best_score:
            best_score = score
            best_index = begin + local
    if best_index < 0:
        raise GravityItem25Error("candidate selection failed")
    return best_index, residual_evaluations


def _oof_search(
    xp: Any,
    config: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    target: np.ndarray,
    folds: np.ndarray,
    term_matrix: np.ndarray,
) -> dict[str, Any]:
    base = _base_design(rows)
    flexible = _flex_design(rows)
    prediction_candidate = np.full(len(rows), np.nan)
    prediction_base = np.full(len(rows), np.nan)
    prediction_flexible = np.full(len(rows), np.nan)
    selected: list[int] = []
    residual_evaluations = 0
    for outer in sorted({int(value) for value in folds}):
        test = np.where(folds == outer)[0]
        train = np.where(folds != outer)[0]
        selected_index, evaluations = _select_candidate(
            xp, config, base, target, folds, outer, term_matrix
        )
        selected.append(selected_index)
        residual_evaluations += evaluations
        terms = xp.asarray(term_matrix[selected_index : selected_index + 1])
        candidate = _fit_candidate_predictions(xp, base, target, terms, train, test)
        prediction_candidate[test] = _to_numpy(candidate[0], xp)
        residual_evaluations += len(test)
        prediction_base[test] = _linear_predict(base, target, train, test)
        prediction_flexible[test] = _ridge_predict(
            flexible,
            target,
            train,
            test,
            float(config["evaluation"]["ridge_alpha"]),
        )
    return {
        "candidate": prediction_candidate,
        "base": prediction_base,
        "flexible": prediction_flexible,
        "selected": selected,
        "residual_evaluations": residual_evaluations,
    }


def _mse(target: np.ndarray, prediction: np.ndarray, indices: np.ndarray | None = None) -> float:
    if indices is None:
        indices = np.arange(len(target))
    return float(np.mean((target[indices] - prediction[indices]) ** 2))


def _improvement(reference: float, candidate: float) -> float:
    return (reference - candidate) / reference if reference > 0.0 else 0.0


def _candidate_record(
    config: Mapping[str, Any], arrays: Mapping[str, np.ndarray], index: int
) -> dict[str, Any]:
    values = _candidate_values(config, arrays, index, index + 1, np)
    niche = int(values["niche"][0])
    result: dict[str, Any] = {
        "index": index,
        "niche": niche,
        "niche_id": config["candidate_generator"]["niches"][niche]["id"],
    }
    for key in (
        "amplitude",
        "polarity",
        "redshift_scale",
        "transition_width",
        "power",
        "maturity_threshold",
        "relaxation_multiplier",
        "log_frequency",
        "phase",
    ):
        result[key] = float(values[key][0])
    return result


def _evaluate(
    root: Path,
    config: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    xp, backend, device = _backend()
    started = time.perf_counter()
    arrays, candidate_audit = _admissible_candidates(config)
    term_matrix = _build_term_matrix(config, arrays, rows)
    folds = np.asarray([int(row["fold"]) for row in rows])
    target = np.log10(np.asarray([float(row["vcirc_km_s"]) for row in rows]))
    observed = _oof_search(xp, config, rows, target, folds, term_matrix)
    residual_evaluations = int(observed["residual_evaluations"])

    candidate_mse = _mse(target, observed["candidate"])
    base_mse = _mse(target, observed["base"])
    flexible_mse = _mse(target, observed["flexible"])
    observed_improvement = _improvement(base_mse, candidate_mse)

    base_full = _base_design(rows)
    base_coefficient = np.linalg.lstsq(base_full, target, rcond=None)[0]
    base_full_prediction = base_full @ base_coefficient
    residual = target - base_full_prediction
    random = np.random.Generator(
        np.random.PCG64(int(config["evaluation"]["permutation_seed"]))
    )
    strata = np.asarray([int(row["redshift_stratum"]) for row in rows])
    null_improvements: list[float] = []
    for trial in range(int(config["evaluation"]["permutation_trials"])):
        permuted = residual.copy()
        for stratum in sorted({int(value) for value in strata}):
            indices = np.where(strata == stratum)[0]
            permuted[indices] = random.permutation(permuted[indices])
        null_target = base_full_prediction + permuted
        null = _oof_search(xp, config, rows, null_target, folds, term_matrix)
        null_candidate_mse = _mse(null_target, null["candidate"])
        null_base_mse = _mse(null_target, null["base"])
        null_improvements.append(_improvement(null_base_mse, null_candidate_mse))
        residual_evaluations += int(null["residual_evaluations"])
        if (trial + 1) % 10 == 0:
            print(
                f"Item 25 selection-aware nulls {trial + 1}/"
                f"{config['evaluation']['permutation_trials']}",
                flush=True,
            )
    permutation_p = (
        1.0 + sum(value >= observed_improvement for value in null_improvements)
    ) / (len(null_improvements) + 1.0)

    synthetic: list[dict[str, Any]] = []
    for niche in range(4):
        eligible = np.where(arrays["niche"] == niche)[0]
        injection_index = int(eligible[len(eligible) // 2])
        injected_target = base_full_prediction + term_matrix[injection_index]
        replay = _oof_search(xp, config, rows, injected_target, folds, term_matrix)
        selected_niches = [int(arrays["niche"][index]) for index in replay["selected"]]
        niche_count = int(np.count_nonzero(np.asarray(selected_niches) == niche))
        synthetic.append(
            {
                "injected_niche": niche,
                "injected_index": injection_index,
                "selected_niches": selected_niches,
                "selected_niche_folds": niche_count,
                "pass": niche_count >= int(config["gates"]["minimum_same_niche_folds"]),
            }
        )
        residual_evaluations += int(replay["residual_evaluations"])

    constant_target = base_full_prediction
    constant = _oof_search(xp, config, rows, constant_target, folds, term_matrix)
    constant_candidate_mse = _mse(constant_target, constant["candidate"])
    constant_base_mse = _mse(constant_target, constant["base"])
    constant_improvement = _improvement(constant_base_mse, constant_candidate_mse)
    constant_pass = constant_improvement <= float(
        config["gates"]["known_constant_G_control_maximum_material_improvement"]
    )
    residual_evaluations += int(constant["residual_evaluations"])

    redshift = np.asarray([float(row["redshift"]) for row in rows])
    mass = np.asarray([float(row["log_baryonic_mass"]) for row in rows])
    slices: dict[str, np.ndarray] = {
        "low_redshift": np.where(redshift <= np.median(redshift))[0],
        "high_redshift": np.where(redshift > np.median(redshift))[0],
        "low_baryonic_mass": np.where(mass <= np.median(mass))[0],
        "high_baryonic_mass": np.where(mass > np.median(mass))[0],
    }
    slice_metrics: dict[str, Any] = {}
    for label, indices in slices.items():
        candidate_value = _mse(target, observed["candidate"], indices)
        base_value = _mse(target, observed["base"], indices)
        flexible_value = _mse(target, observed["flexible"], indices)
        slice_metrics[label] = {
            "objects": len(indices),
            "candidate_mse": candidate_value,
            "calibrated_baryonic_mse": base_value,
            "flexible_mse": flexible_value,
            "improvement_vs_calibrated_baryonic": _improvement(base_value, candidate_value),
            "improvement_vs_flexible": _improvement(flexible_value, candidate_value),
        }

    selected_records = [
        _candidate_record(config, arrays, int(index)) for index in observed["selected"]
    ]
    selected_niches = [int(record["niche"]) for record in selected_records]
    niche_counts = Counter(selected_niches)
    same_niche_folds = max(niche_counts.values())
    counterexamples = int(
        np.count_nonzero(
            (target - observed["candidate"]) ** 2
            > (target - observed["flexible"]) ** 2
        )
    )
    redshift_halves_pass = all(
        slice_metrics[label]["improvement_vs_calibrated_baryonic"]
        >= float(config["gates"]["minimum_each_redshift_half_improvement_vs_calibrated"])
        for label in ("low_redshift", "high_redshift")
    )
    mass_halves_pass = all(
        slice_metrics[label]["improvement_vs_calibrated_baryonic"]
        >= float(config["gates"]["minimum_each_mass_half_improvement_vs_calibrated"])
        for label in ("low_baryonic_mass", "high_baryonic_mass")
    )
    universal_pass = all(
        [
            observed_improvement
            >= float(config["gates"]["minimum_improvement_vs_calibrated_baryonic"]),
            _improvement(flexible_mse, candidate_mse)
            >= float(config["gates"]["minimum_improvement_vs_flexible_nuisance"]),
            redshift_halves_pass,
            mass_halves_pass,
            permutation_p
            <= float(config["gates"]["maximum_selection_aware_permutation_p"]),
            same_niche_folds >= int(config["gates"]["minimum_same_niche_folds"]),
            all(value["pass"] for value in synthetic),
            constant_pass,
        ]
    )
    phenomenon_pass = all(
        [
            _improvement(flexible_mse, candidate_mse)
            >= float(config["gates"]["phenomenon_minimum_improvement_vs_flexible"]),
            permutation_p
            <= float(config["gates"]["maximum_selection_aware_permutation_p"]),
            same_niche_folds >= int(config["gates"]["minimum_same_niche_folds"]),
        ]
    )
    cpu_terms = term_matrix[np.asarray(observed["selected"])]
    gpu_terms = _to_numpy(xp.asarray(cpu_terms), xp)
    cpu_gpu_max = float(np.max(np.abs(cpu_terms - gpu_terms)))
    elapsed = time.perf_counter() - started
    scientific = {
        "valid_objects": len(rows),
        "quality_pass": len(rows) >= int(config["sample"]["minimum_valid_exploration"]),
        "candidate_audit": candidate_audit,
        "metrics": {
            "candidate_mse": candidate_mse,
            "calibrated_baryonic_mse": base_mse,
            "flexible_nuisance_mse": flexible_mse,
            "improvement_vs_calibrated_baryonic": observed_improvement,
            "improvement_vs_flexible_nuisance": _improvement(flexible_mse, candidate_mse),
            "selection_aware_permutation_p": permutation_p,
            "null_improvement_minimum": float(np.min(null_improvements)),
            "null_improvement_median": float(np.median(null_improvements)),
            "null_improvement_maximum": float(np.max(null_improvements)),
            "individual_counterexamples_vs_flexible": counterexamples,
        },
        "slice_metrics": slice_metrics,
        "selected_folds": selected_records,
        "selected_niche_counts": {str(key): value for key, value in sorted(niche_counts.items())},
        "same_niche_folds": same_niche_folds,
        "controls": {
            "synthetic_niche_recovery": synthetic,
            "synthetic_all_pass": all(value["pass"] for value in synthetic),
            "constant_G_improvement_vs_calibrated": constant_improvement,
            "constant_G_pass": constant_pass,
            "cpu_gpu_max_absolute_difference": cpu_gpu_max,
            "cpu_gpu_pass": cpu_gpu_max <= 1e-12,
        },
        "universal_gravity_track_pass": universal_pass,
        "phenomenon_publication_track_pass": phenomenon_pass,
        "paper_claim_allowed": False,
        "formal_status": (
            "PASS_EXPLORATION_BOTH_TRACKS"
            if universal_pass and phenomenon_pass
            else "PASS_EXPLORATION_UNIVERSAL_ONLY"
            if universal_pass
            else "PASS_EXPLORATION_PHENOMENON_LEAD"
            if phenomenon_pass
            else "SCOPED_REJECT_BOTH_TRACKS"
        ),
    }
    compute = {
        "schema_version": "invariant-gravity-item25-time-varying-g-compute-1.0",
        "backend": backend,
        "device": device,
        "admissible_candidates": len(arrays["niche"]),
        "training_residual_evaluations": residual_evaluations,
        "permutation_trials": len(null_improvements),
        "synthetic_full_searches": 4,
        "constant_G_full_searches": 1,
        "wall_seconds": elapsed,
        "paid_model_calls": 0,
        "paid_api_spend_usd": 0.0,
    }
    return scientific, compute


def _build_receipt(
    root: Path,
    config: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    response_manifest: Mapping[str, Any],
    scientific: Mapping[str, Any],
    compute: Mapping[str, Any],
) -> dict[str, Any]:
    paths = _source_paths(root, config)
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item25-time-varying-g-result-1.0",
            "item": 25,
            "title": config["title"],
            "hypothesis": config["hypothesis"],
            "discovery_policy": config["discovery_policy"],
            "theory_and_equivalence_audit": config["theory"],
            "observable_lineage": config["sources"]["observable_lineage"],
            "frozen_boundary": {
                "scientific_freeze_commit": config["scientific_freeze_commit"],
                "sample_freeze_commit": config["sample_freeze_commit"],
                "stable_goal_sha256": config["stable_goal_sha256"],
                "confirmation_opened": False,
                "confirmation_response_values_read": int(
                    response_manifest["confirmation_values_read"]
                ),
                "post_response_formula_generation": False,
            },
            "sample": {
                "valid_exploration_objects": len(rows),
                "valid_identities": [int(row["identity"]) for row in rows],
                "confirmation_identities_remain_sealed": int(
                    config["sample"]["expected_confirmation"]
                ),
            },
            "baselines": {
                "calibrated_baryonic": config["evaluation"]["baseline_calibrated_baryonic"],
                "flexible_nuisance": config["evaluation"]["baseline_flexible_nuisance"],
            },
            "scientific_result": scientific,
            "compute_and_api_cost": compute,
            "counterexamples_and_limitations": [
                "A Tully-Fisher residual is not a direct measurement of G.",
                "The KMOS3D baryonic masses depend on stellar-population and gas-scaling assumptions.",
                "Pressure support is modeled in the published Vcirc and remains an astrophysical systematic.",
                "The sample contains massive star-forming disks at 0.6<=z<=2.6 and does not represent every galaxy population.",
                "A local maturity relation can be an ordinary gas-fraction or evolution correlation even if statistically positive.",
                "Passing frozen scalar bounds is not a covariant cosmology, CMB likelihood, stellar-evolution calculation, or equivalence-principle proof.",
                "No sealed confirmation response was queried.",
            ],
            "exact_next_action": "Preserve every Item 25 branch under the equal-viability two-track policy, independently replicate any phenomenon lead without retuning, and advance the numbered roadmap to Item 26 retarded gravity.",
            "reproducibility": {
                "config_path": CONFIG_PATH.as_posix(),
                "config_sha256": _sha256_file(root / CONFIG_PATH),
                "module_path": MODULE_PATH.as_posix(),
                "module_sha256": _sha256_file(root / MODULE_PATH),
                "predictor_manifest_path": paths["predictor_source_manifest"]
                .relative_to(root)
                .as_posix(),
                "response_manifest_path": paths["response_source_manifest"]
                .relative_to(root)
                .as_posix(),
                "compute_manifest_path": paths["compute_manifest"].relative_to(root).as_posix(),
            },
        }
    )


def run_experiment(root: Path) -> Path:
    config = load_config(root)
    verify_science_freeze(root, config)
    verify_sample_freeze(root, config)
    rows, response_manifest = _load_rows(root, config)
    scientific, compute_raw = _evaluate(root, config, rows)
    paths = _source_paths(root, config)
    compute = _content_hashed(compute_raw)
    _write_json(paths["compute_manifest"], compute)
    result = _build_receipt(root, config, rows, response_manifest, scientific, compute)
    result_path = root / str(config["paths"]["result"])
    _write_json(result_path, result)
    return result_path


def validate_result(root: Path) -> Path:
    config = load_config(root)
    verify_science_freeze(root, config)
    verify_sample_freeze(root, config)
    paths = _source_paths(root, config)
    for key in (
        "predictor_source_manifest",
        "sample_manifest",
        "candidate_manifest",
        "response_source_manifest",
        "compute_manifest",
    ):
        _verify_content_hash(_read_json(paths[key]), key)
    result_path = root / str(config["paths"]["result"])
    result = _read_json(result_path)
    _verify_content_hash(result, "result")
    if int(result["frozen_boundary"]["confirmation_response_values_read"]) != 0:
        raise GravityItem25Error("result opened confirmation data")
    if bool(result["scientific_result"]["paper_claim_allowed"]):
        raise GravityItem25Error("exploration result made a paper claim")
    return result_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare-predictors")
    sub.add_parser("acquire-responses")
    sub.add_parser("run")
    sub.add_parser("validate")
    sub.add_parser("show-candidates")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = Path.cwd()
    if args.command == "prepare-predictors":
        print(prepare_predictors(root)["sample_manifest"].as_posix())
    elif args.command == "acquire-responses":
        print(acquire_responses(root).as_posix())
    elif args.command == "run":
        print(run_experiment(root).as_posix())
    elif args.command == "validate":
        print(validate_result(root).as_posix())
    else:
        print(json.dumps(_candidate_manifest(load_config(root)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
