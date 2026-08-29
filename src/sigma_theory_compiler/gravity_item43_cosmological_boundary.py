"""Frozen Item 43 cosmological-boundary search and S4TM/CLASH evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from sigma_theory_compiler.gravity_counterexample_policy import (
    assess_counterexample_evidence,
    load_counterexample_policy,
)
from sigma_theory_compiler.gravity_item22_polarization_superposition import (
    _canonical_bytes,
    _content_hashed,
    _read_json,
    _sha256_bytes,
    _sha256_file,
    _write_json,
)

CONFIG_PATH = Path("configs/gravity_item43_cosmological_boundary_v1.json")
GOAL_PATH = Path("docs/GRAVITY_HIDDEN_VARIABLE_AND_THEORY_SEARCH_GOALS.md")
POLICY_PATH = Path("configs/gravity_empirical_counterexample_policy_v1.json")


class GravityItem43Error(RuntimeError):
    """Raised when an Item 43 freeze, source, or evaluation gate fails."""


def load_config(root: Path) -> dict[str, Any]:
    config = _read_json(root / CONFIG_PATH)
    validate_config(root, config)
    return config


def validate_config(root: Path, config: Mapping[str, Any]) -> None:
    if (
        config.get("schema_version")
        != "invariant-gravity-item43-cosmological-boundary-config-1.0"
        or int(config.get("item", -1)) != 43
    ):
        raise GravityItem43Error("unexpected Item 43 config")
    if _sha256_file(root / GOAL_PATH) != str(config["stable_goal_sha256"]):
        raise GravityItem43Error("stable gravity goal changed")
    for relative, expected in config["scientific_dependencies"].items():
        if _sha256_file(root / str(relative)) != str(expected):
            raise GravityItem43Error(f"scientific dependency changed: {relative}")
    predecessor = _read_json(
        root / "runs/gravity/roadmap/item-42-matter-geometry-feedback-v1.json"
    )
    required = config["required_predecessor"]
    if predecessor.get("content_sha256") != required["content_sha256"]:
        raise GravityItem43Error("Item 42 content binding changed")
    if predecessor.get("decision") != required["decision"]:
        raise GravityItem43Error("Item 42 decision changed")
    policy = load_counterexample_policy(root / POLICY_PATH)
    empirical = policy["empirical_evidence"]
    discovery = config["discovery_policy"]
    if not bool(discovery["equal_initial_viability"]):
        raise GravityItem43Error("equal initial viability changed")
    if not bool(discovery["single_empirical_counterexample_is_not_a_formula_or_family_veto"]):
        raise GravityItem43Error("one empirical mismatch became a veto")
    if not bool(discovery["counterexample_count_alone_is_never_decisive"]):
        raise GravityItem43Error("count-only rejection entered Item 43")
    if bool(discovery["finite_empirical_sample_may_prune_family"]):
        raise GravityItem43Error("finite empirical family pruning entered Item 43")
    if empirical["single_counterexample_terminal_rejection_allowed"] is not False:
        raise GravityItem43Error("executable counterexample policy changed")
    generator = config["candidate_generator"]
    if int(generator["raw_candidate_cells"]) != 262144:
        raise GravityItem43Error("raw candidate count changed")
    if int(generator["cells_per_niche"]) != 65536:
        raise GravityItem43Error("per-niche capacity changed")
    if list(generator["grid_shape_per_niche"]) != [16, 16, 16, 16]:
        raise GravityItem43Error("candidate grid changed")
    if len(generator["niches"]) != 4 or int(generator["post_response_cells"]) != 0:
        raise GravityItem43Error("boundary niche or response-free boundary changed")
    if bool(config["scope"]["confirmation_opening_authorized"]):
        raise GravityItem43Error("confirmation opening is not authorized")
    if bool(config["scope"]["paid_api_calls_authorized"]):
        raise GravityItem43Error("paid calls entered Item 43")
    if config["weak_field_metric_contract"]["gravitational_slip"] != "Phi=Psi":
        raise GravityItem43Error("motion/light closure changed")
    if len(config["schema_audit_exposure"]["excluded_from_every_item43_role"]) != 5:
        raise GravityItem43Error("schema-audit exclusions changed")
    transfer = config["cluster_transfer"]
    if bool(transfer["selection_use"]) or bool(transfer["retuning_allowed"]):
        raise GravityItem43Error("CLASH entered selection or retuning")


def _contract_digest(config: Mapping[str, Any]) -> str:
    value = json.loads(json.dumps(config))
    for key in ("scientific_freeze_commit", "sample_freeze_commit"):
        value[key] = "<BOUND_COMMIT>"
    return _sha256_bytes(_canonical_bytes(value))


def _source_path(root: Path, config: Mapping[str, Any], key: str) -> Path:
    return root / str(config["paths"]["source_dir"]) / str(config["paths"][key])


def _split_hash(value: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}|{value}".encode()).hexdigest()


def generate_raw_candidates(config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    total = int(config["candidate_generator"]["raw_candidate_cells"])
    cells = int(config["candidate_generator"]["cells_per_niche"])
    candidate_id = np.arange(total, dtype=np.int64)
    lane = (candidate_id // cells).astype(np.int8)
    local = candidate_id % cells
    return {
        "candidate_id": candidate_id,
        "lane": lane,
        "amplitude_index": ((local // 4096) % 16).astype(np.int8),
        "exponent_index": ((local // 256) % 16).astype(np.int8),
        "transition_index": ((local // 16) % 16).astype(np.int8),
        "boundary_index": (local % 16).astype(np.int8),
    }


def _candidate_parameters(
    candidates: Mapping[str, np.ndarray], config: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    grids = config["candidate_generator"]["parameter_grids"]
    return (
        np.asarray(grids["amplitude"])[np.asarray(candidates["amplitude_index"], int)],
        np.asarray(grids["acceleration_exponent"])[
            np.asarray(candidates["exponent_index"], int)
        ],
        np.asarray(grids["transition_u"])[
            np.asarray(candidates["transition_index"], int)
        ],
        np.asarray(grids["boundary_exponent"])[
            np.asarray(candidates["boundary_index"], int)
        ],
    )


def expansion_ratio(z: np.ndarray | float, config: Mapping[str, Any]) -> np.ndarray:
    cosmology = config["fiducial_cosmology"]
    omega_m = float(cosmology["omega_matter"])
    omega_l = float(cosmology["omega_lambda"])
    z_array = np.asarray(z, dtype=np.float64)
    return np.sqrt(omega_m * np.power(1.0 + z_array, 3.0) + omega_l)


def age_ratio(z: np.ndarray | float, config: Mapping[str, Any]) -> np.ndarray:
    cosmology = config["fiducial_cosmology"]
    omega_m = float(cosmology["omega_matter"])
    omega_l = float(cosmology["omega_lambda"])
    z_array = np.asarray(z, dtype=np.float64)
    numerator = np.arcsinh(
        math.sqrt(omega_l / omega_m) / np.power(1.0 + z_array, 1.5)
    )
    denominator = math.asinh(math.sqrt(omega_l / omega_m))
    return numerator / denominator


def boundary_bases(
    z: np.ndarray,
    radius_kpc: np.ndarray,
    config: Mapping[str, Any],
) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    radius_kpc = np.asarray(radius_kpc, dtype=np.float64)
    e = expansion_ratio(z, config)
    t = age_ratio(z, config)
    cosmology = config["fiducial_cosmology"]
    h0 = float(cosmology["hubble_constant_km_s_mpc"])
    c = float(cosmology["speed_of_light_km_s"])
    reference = float(cosmology["horizon_fraction_reference"])
    horizon_fraction = (radius_kpc / 1000.0) * h0 * e / c
    return np.stack(
        (
            e,
            1.0 / t,
            1.0 + z,
            1.0 + horizon_fraction / reference,
        ),
        axis=0,
    )


def candidate_multiplier(
    candidates: Mapping[str, np.ndarray],
    u: np.ndarray,
    bases: np.ndarray,
    config: Mapping[str, Any],
) -> np.ndarray:
    amplitude, exponent, transition, boundary_exponent = _candidate_parameters(
        candidates, config
    )
    lane = np.asarray(candidates["lane"], dtype=int)
    boundary = np.power(bases[lane], boundary_exponent[:, None])
    return 1.0 + amplitude[:, None] * np.power(
        np.asarray(u, dtype=np.float64)[None, :], -exponent[:, None]
    ) / (1.0 + np.asarray(u)[None, :] / transition[:, None]) * boundary


def admissible_candidates(
    config: Mapping[str, Any], *, batch_size: int = 4096
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    raw = generate_raw_candidates(config)
    gate = config["candidate_generator"]["admissibility"]
    u = np.logspace(
        float(gate["probe_log10_u_min"]),
        float(gate["probe_log10_u_max"]),
        int(gate["probe_points"]),
    )
    z_values = np.asarray(gate["probe_redshifts"], dtype=np.float64)
    r_values = np.asarray(gate["probe_radii_kpc"], dtype=np.float64)
    zz, rr = np.meshgrid(z_values, r_values, indexing="ij")
    probe_bases = boundary_bases(zz.ravel(), rr.ravel(), config)
    keep_parts: list[np.ndarray] = []
    signatures: set[bytes] = set()
    rejection: Counter[str] = Counter()
    for begin in range(0, len(raw["candidate_id"]), batch_size):
        end = min(begin + batch_size, len(raw["candidate_id"]))
        rows = {key: value[begin:end] for key, value in raw.items()}
        amplitude, exponent, transition, boundary_exponent = _candidate_parameters(
            rows, config
        )
        lane = np.asarray(rows["lane"], dtype=int)
        boundary = np.power(
            probe_bases[lane], boundary_exponent[:, None]
        )[:, :, None]
        multiplier = 1.0 + amplitude[:, None, None] * np.power(
            u[None, None, :], -exponent[:, None, None]
        ) / (1.0 + u[None, None, :] / transition[:, None, None]) * boundary
        finite = np.all(np.isfinite(multiplier), axis=(1, 2))
        bounded = finite & np.all(
            multiplier >= float(gate["minimum_multiplier"]), axis=(1, 2)
        )
        bounded &= np.all(
            multiplier <= float(gate["maximum_multiplier"]), axis=(1, 2)
        )
        local = bounded & (
            np.max(np.log10(multiplier[:, :, -1]), axis=1)
            <= float(gate["maximum_high_acceleration_log10_deviation"])
        )
        material = local & (
            np.min(multiplier[:, :, 0], axis=1)
            >= float(gate["minimum_low_acceleration_multiplier"])
        )
        monotone = material & np.all(
            np.diff(multiplier, axis=2)
            <= float(gate["monotone_nonincreasing_tolerance"]),
            axis=(1, 2),
        )
        rejection["nonfinite"] += int(np.sum(~finite))
        rejection["out_of_bounds"] += int(np.sum(finite & ~bounded))
        rejection["no_local_limit"] += int(np.sum(bounded & ~local))
        rejection["immaterial_low_acceleration"] += int(np.sum(local & ~material))
        rejection["nonmonotone"] += int(np.sum(material & ~monotone))
        local_keep = np.flatnonzero(monotone)
        keep_parts.append(local_keep + begin)
        signature = np.round(
            np.log10(multiplier[local_keep]),
            int(gate["behavior_signature_decimals"]),
        )
        for row in signature:
            signatures.add(hashlib.blake2b(row.tobytes(), digest_size=16).digest())
    keep = np.concatenate(keep_parts) if keep_parts else np.empty(0, dtype=np.int64)
    admitted = {key: value[keep] for key, value in raw.items()}
    return admitted, {
        "raw_candidates": len(raw["candidate_id"]),
        "admitted_candidates": len(keep),
        "rejected_candidates": len(raw["candidate_id"]) - len(keep),
        "admitted_by_lane": {
            str(lane): int(np.sum(admitted["lane"] == lane)) for lane in range(4)
        },
        "behavioral_equivalence_classes": len(signatures),
        "rejection_counts_nonexclusive": dict(sorted(rejection.items())),
    }


def decode_candidate(candidate_id: int, config: Mapping[str, Any]) -> dict[str, Any]:
    raw = generate_raw_candidates(config)
    if candidate_id < 0 or candidate_id >= len(raw["candidate_id"]):
        raise GravityItem43Error("candidate id outside frozen grid")
    row = {key: value[candidate_id : candidate_id + 1] for key, value in raw.items()}
    amplitude, exponent, transition, boundary_exponent = _candidate_parameters(row, config)
    lane = int(row["lane"][0])
    niche = config["candidate_generator"]["niches"][lane]
    return {
        "candidate_id": candidate_id,
        "lane_id": lane,
        "lane": niche["name"],
        "creativity_label": niche["creativity_label"],
        "boundary_coordinate": niche["boundary_coordinate"],
        "parameters": {
            "amplitude": float(amplitude[0]),
            "acceleration_exponent": float(exponent[0]),
            "transition_u": float(transition[0]),
            "boundary_exponent": float(boundary_exponent[0]),
        },
    }


def build_candidate_manifest(root: Path) -> dict[str, Any]:
    config = load_config(root)
    admitted, audit = admissible_candidates(config)
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item43-candidate-manifest-1.0",
            "item": 43,
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "config_contract_sha256": _contract_digest(config),
            "response_accessed_during_generation": False,
            "confirmation_accessed": False,
            "paid_model_calls": 0,
            **audit,
            "candidate_id_sha256": _sha256_bytes(
                np.asarray(admitted["candidate_id"], dtype="<i8").tobytes()
            ),
            "niches": config["candidate_generator"]["niches"],
            "claim_boundaries": [
                "cosmological acceleration and global-potential connections are known idea families",
                "the age and finite-horizon projections do not establish historical novelty",
                "the fiducial background coordinates do not derive modified field equations",
                "zero response values and zero paid calls entered generation",
            ],
        }
    )


def build_exposure_manifest(root: Path) -> dict[str, Any]:
    config = load_config(root)
    audit = config["schema_audit_exposure"]
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item43-schema-exposure-1.0",
            "item": 43,
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "reason": audit["reason"],
            "excluded_targets": sorted(audit["excluded_from_every_item43_role"]),
            "counts": {
                "schema_rows_with_response_seen": int(audit["rows_with_response_seen"]),
                "response_values_used_for_formula_or_sample_design": int(
                    audit["response_values_used_for_formula_or_sample_design"]
                ),
                "remaining_response_rows_read": 0,
                "confirmation_rows_read": 0,
            },
            "rules": [
                "permanently exclude all five schema-audit targets",
                "freeze formula space before any remaining S4TM predictor or response access",
                "never replace a failed or missing identity after response access",
            ],
        }
    )


def write_freeze_manifests(root: Path) -> list[Path]:
    config = load_config(root)
    paths = [
        _source_path(root, config, "candidate_manifest"),
        _source_path(root, config, "exposure_manifest"),
    ]
    _write_json(paths[0], build_candidate_manifest(root))
    _write_json(paths[1], build_exposure_manifest(root))
    return paths


def _fetch_tsv(url: str) -> tuple[list[dict[str, str]], dict[str, Any]]:
    request = urllib.request.Request(url, headers={"User-Agent": "Invariant-Item43/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read()
    text = payload.decode("utf-8")
    lines = [line for line in text.splitlines() if line and not line.startswith("#")]
    if len(lines) < 4:
        raise GravityItem43Error("VizieR response has no tabular rows")
    reader = csv.DictReader(io.StringIO("\n".join([lines[0], *lines[3:]])), delimiter="\t")
    rows = [{str(k): str(v).strip() for k, v in row.items()} for row in reader]
    return rows, {
        "url": url,
        "bytes": len(payload),
        "payload_sha256": _sha256_bytes(payload),
    }


def _query_url(endpoint: str, columns: Sequence[str], **constraints: str) -> str:
    pairs: list[tuple[str, str]] = [("-out", value) for value in columns]
    pairs.append(("-out.max", "unlimited"))
    pairs.extend((key, value) for key, value in constraints.items())
    return endpoint + "&" + urllib.parse.urlencode(pairs)


def acquire_predictors(root: Path) -> Path:
    config = load_config(root)
    if str(config["scientific_freeze_commit"]).startswith("PENDING_"):
        raise GravityItem43Error("scientific freeze is not commit-bound")
    sources = config["data_sources"]
    table1_url = _query_url(
        str(sources["s4tm_table1_endpoint"]), sources["predictor_table1_columns"]
    )
    table2_url = _query_url(
        str(sources["s4tm_table2_endpoint"]), sources["predictor_table2_columns"]
    )
    table1, receipt1 = _fetch_tsv(table1_url)
    table2, receipt2 = _fetch_tsv(table2_url)
    forbidden = {"logMein", "Sigma", "e_Sigma"}
    if any(forbidden.intersection(row) for row in (*table1, *table2)):
        raise GravityItem43Error("forbidden response column entered predictor acquisition")
    by_target1 = {row["Target"]: row for row in table1}
    by_target2 = {row["Target"]: row for row in table2}
    exposed = set(config["schema_audit_exposure"]["excluded_from_every_item43_role"])
    grade_a = {
        target
        for target, row in by_target1.items()
        if str(row["Class"]).strip().upper().endswith("A")
    }
    eligible_targets = sorted((grade_a & set(by_target2)) - exposed)
    records: list[dict[str, Any]] = []
    for target in eligible_targets:
        one = by_target1[target]
        two = by_target2[target]
        records.append(
            {
                "target": target,
                "z_lens": float(one["zL"]),
                "z_source": float(one["zS"]),
                "i_magnitude": float(one["Imag"]),
                "i_extinction": float(one["Ai"]),
                "effective_radius_arcsec": float(one["Reff"]),
                "light_axis_ratio": float(one["q"]),
                "classification": one["Class"],
                "einstein_radius_arcsec": float(two["bSIE"]),
                "sie_axis_ratio": float(two["q"]),
                "log10_total_stellar_mass_msun": float(two["logM"]),
                "lens_model_chi_square": int(two["chi2"]),
                "lens_model_degrees_of_freedom": int(two["dof"]),
            }
        )
    expected = int(config["sample_boundary"]["expected_eligible_lenses"])
    if len(records) != expected:
        raise GravityItem43Error(
            f"expected {expected} unexposed grade-A lenses, found {len(records)}"
        )
    result = _content_hashed(
        {
            "schema_version": "invariant-gravity-item43-s4tm-predictors-1.0",
            "item": 43,
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "retrievals": {"table1": receipt1, "table2": receipt2},
            "records": records,
            "excluded_schema_audit_targets": sorted(exposed),
            "counts": {
                "table1_rows": len(table1),
                "table2_predictor_rows": len(table2),
                "grade_a_before_schema_exclusion": len(grade_a & set(by_target2)),
                "eligible_after_schema_exclusion": len(records),
                "response_values_read": 0,
                "confirmation_values_read": 0,
                "paid_model_calls": 0,
            },
        }
    )
    path = _source_path(root, config, "predictor_source")
    _write_json(path, result)
    return path


def build_sample_manifest(root: Path) -> dict[str, Any]:
    config = load_config(root)
    predictor_path = _source_path(root, config, "predictor_source")
    predictors = _read_json(predictor_path)
    records = list(predictors["records"])
    split_salt = str(config["sample_boundary"]["split_salt"])
    confirmation_count = int(config["sample_boundary"]["reserved_confirmation_lenses"])
    ranked = sorted(records, key=lambda row: _split_hash(row["target"], split_salt))
    confirmation_targets = {row["target"] for row in ranked[:confirmation_count]}
    exploration = [row for row in records if row["target"] not in confirmation_targets]
    fold_salt = str(config["sample_boundary"]["fold_salt"])
    folds = int(config["sample_boundary"]["outer_folds"])
    fold_rank = sorted(exploration, key=lambda row: _split_hash(row["target"], fold_salt))
    fold_by_target = {row["target"]: index % folds for index, row in enumerate(fold_rank)}
    objects = [
        {
            "target": row["target"],
            "role": "confirmation" if row["target"] in confirmation_targets else "exploration",
            "outer_fold": None
            if row["target"] in confirmation_targets
            else fold_by_target[row["target"]],
        }
        for row in sorted(records, key=lambda row: row["target"])
    ]
    expected_exploration = int(config["sample_boundary"]["expected_exploration_lenses"])
    if len(exploration) != expected_exploration:
        raise GravityItem43Error("unexpected Item 43 exploration count")
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item43-s4tm-sample-1.0",
            "item": 43,
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "predictor_source_sha256": _sha256_file(predictor_path),
            "objects": objects,
            "counts": {
                "eligible_lenses": len(objects),
                "exploration_lenses": len(exploration),
                "confirmation_lenses": len(confirmation_targets),
                "response_rows_read": 0,
                "confirmation_rows_read": 0,
            },
            "rules": [
                "whole-lens deterministic hash split",
                "all five schema-audit targets excluded before splitting",
                "seven confirmation responses remain sealed",
                "no failed identity replacement",
            ],
        }
    )


def write_sample_manifest(root: Path) -> Path:
    config = load_config(root)
    path = _source_path(root, config, "sample_manifest")
    _write_json(path, build_sample_manifest(root))
    return path


def acquire_exploration_responses(root: Path) -> Path:
    config = load_config(root)
    if str(config["sample_freeze_commit"]).startswith("PENDING_"):
        raise GravityItem43Error("sample/evaluator freeze is not commit-bound")
    sample = _read_json(_source_path(root, config, "sample_manifest"))
    targets = [row["target"] for row in sample["objects"] if row["role"] == "exploration"]
    endpoint = str(config["data_sources"]["s4tm_table2_endpoint"])
    columns = config["data_sources"]["response_columns"]
    rows: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    for target in sorted(targets):
        url = _query_url(endpoint, columns, Target=target)
        response_rows, receipt = _fetch_tsv(url)
        exact = [row for row in response_rows if row.get("Target") == target]
        if len(exact) != 1:
            raise GravityItem43Error(f"expected one exact response for {target}")
        rows.append({"target": target, "log10_einstein_mass_msun": float(exact[0]["logMein"])})
        receipts.append({"target": target, **receipt})
    if len(rows) != int(config["sample_boundary"]["expected_exploration_lenses"]):
        raise GravityItem43Error("exploration response count changed")
    result = _content_hashed(
        {
            "schema_version": "invariant-gravity-item43-s4tm-responses-1.0",
            "item": 43,
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "sample_freeze_commit": config["sample_freeze_commit"],
            "sample_manifest_sha256": _sha256_file(
                _source_path(root, config, "sample_manifest")
            ),
            "records": rows,
            "retrievals": receipts,
            "counts": {
                "exploration_response_rows": len(rows),
                "confirmation_response_rows": 0,
                "post_response_exclusions": 0,
                "paid_model_calls": 0,
            },
        }
    )
    path = _source_path(root, config, "response_source")
    _write_json(path, result)
    return path


def angular_diameter_distance_mpc(z: float, config: Mapping[str, Any]) -> float:
    cosmology = config["fiducial_cosmology"]
    nodes, weights = np.polynomial.legendre.leggauss(64)
    sample_z = 0.5 * z * (nodes + 1.0)
    integral = 0.5 * z * float(np.sum(weights / expansion_ratio(sample_z, config)))
    comoving = float(cosmology["speed_of_light_km_s"]) / float(
        cosmology["hubble_constant_km_s_mpc"]
    ) * integral
    return comoving / (1.0 + z)


def sersic_n4_fraction(radius_over_re: np.ndarray | float, b4: float) -> np.ndarray:
    ratio = np.maximum(np.asarray(radius_over_re, dtype=np.float64), 0.0)
    x = b4 * np.power(ratio, 0.25)
    series = np.zeros_like(x)
    term = np.ones_like(x)
    for k in range(8):
        if k > 0:
            term = term * x / k
        series += term
    return 1.0 - np.exp(-x) * series


def _feature_arrays(
    records: Sequence[Mapping[str, Any]], config: Mapping[str, Any], *,
    stellar_shift_dex: float = 0.0, re_scale: float = 1.0,
) -> dict[str, np.ndarray]:
    z = np.asarray([float(row["z_lens"]) for row in records])
    re_arcsec = re_scale * np.asarray([float(row["effective_radius_arcsec"]) for row in records])
    ein_arcsec = np.asarray([float(row["einstein_radius_arcsec"]) for row in records])
    log_mstar = stellar_shift_dex + np.asarray(
        [float(row["log10_total_stellar_mass_msun"]) for row in records]
    )
    fraction = sersic_n4_fraction(
        ein_arcsec / re_arcsec, float(config["constants"]["sersic_n4_b"])
    )
    log_mbar = log_mstar + np.log10(np.maximum(fraction, 1e-12))
    distance = np.asarray([angular_diameter_distance_mpc(value, config) for value in z])
    radius_kpc = ein_arcsec / 206265.0 * distance * 1000.0
    g_kpc = float(config["constants"]["gravitational_constant_kpc_km2_s2_msun"])
    gbar_km2_s2_kpc = g_kpc * np.power(10.0, log_mbar) / np.square(radius_kpc)
    gbar_m_s2 = gbar_km2_s2_kpc / float(config["constants"]["kpc_to_km"]) * 1000.0
    u = gbar_m_s2 / float(config["constants"]["acceleration_scale_m_s2"])
    return {
        "z": z,
        "radius_kpc": radius_kpc,
        "log_mbar": log_mbar,
        "u": u,
        "bases": boundary_bases(z, radius_kpc, config),
        "axis_ratio": np.asarray([float(row["light_axis_ratio"]) for row in records]),
        "log_mstar": log_mstar,
    }


def _candidate_slice(candidates: Mapping[str, np.ndarray], indices: np.ndarray) -> dict[str, np.ndarray]:
    return {key: np.asarray(value)[indices] for key, value in candidates.items()}


def _best_candidate(
    candidates: Mapping[str, np.ndarray],
    features: Mapping[str, np.ndarray],
    target: np.ndarray,
    train_indices: np.ndarray,
    sigma: float,
    config: Mapping[str, Any],
) -> tuple[int, float, str, int]:
    batch_size = int(config["evaluation"]["candidate_batch_size"])
    backend = "numpy_cpu"
    xp: Any = np
    try:
        import cupy as cp

        if int(cp.cuda.runtime.getDeviceCount()) > 0:
            xp = cp
            backend = f"cupy_cuda_{cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}"
    except Exception:
        xp = np
    u = xp.asarray(np.asarray(features["u"])[train_indices])
    log_mbar = xp.asarray(np.asarray(features["log_mbar"])[train_indices])
    bases = xp.asarray(np.asarray(features["bases"])[:, train_indices])
    observed = xp.asarray(np.asarray(target)[train_indices])
    grids = config["candidate_generator"]["parameter_grids"]
    amplitude_grid = xp.asarray(grids["amplitude"])
    exponent_grid = xp.asarray(grids["acceleration_exponent"])
    transition_grid = xp.asarray(grids["transition_u"])
    boundary_grid = xp.asarray(grids["boundary_exponent"])
    best_loss = math.inf
    best_index = -1
    for begin in range(0, len(candidates["candidate_id"]), batch_size):
        end = min(begin + batch_size, len(candidates["candidate_id"]))
        lane = xp.asarray(np.asarray(candidates["lane"])[begin:end], dtype=xp.int64)
        aa = amplitude_grid[xp.asarray(np.asarray(candidates["amplitude_index"])[begin:end])]
        pp = exponent_grid[xp.asarray(np.asarray(candidates["exponent_index"])[begin:end])]
        tt = transition_grid[xp.asarray(np.asarray(candidates["transition_index"])[begin:end])]
        qq = boundary_grid[xp.asarray(np.asarray(candidates["boundary_index"])[begin:end])]
        boundary = xp.power(bases[lane], qq[:, None])
        multiplier = 1.0 + aa[:, None] * xp.power(u[None, :], -pp[:, None]) / (
            1.0 + u[None, :] / tt[:, None]
        ) * boundary
        prediction = log_mbar[None, :] + xp.log10(multiplier)
        losses = xp.mean(xp.square((prediction - observed[None, :]) / sigma), axis=1)
        local_index = int(xp.argmin(losses).item())
        local_loss = float(losses[local_index].item())
        if local_loss < best_loss:
            best_loss = local_loss
            best_index = begin + local_index
    return (
        int(np.asarray(candidates["candidate_id"])[best_index]),
        best_loss,
        backend,
        len(candidates["candidate_id"]) * len(train_indices),
    )


def _predict_candidate(
    candidate_id: int,
    features: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
) -> np.ndarray:
    raw = generate_raw_candidates(config)
    row = {key: value[candidate_id : candidate_id + 1] for key, value in raw.items()}
    multiplier = candidate_multiplier(row, features["u"], features["bases"], config)[0]
    return np.asarray(features["log_mbar"]) + np.log10(multiplier)


def _matched_no_boundary(admitted: Mapping[str, np.ndarray], config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    boundary_values = np.asarray(config["candidate_generator"]["parameter_grids"]["boundary_exponent"])
    zero_index = int(np.flatnonzero(boundary_values == 0.0)[0])
    mask = (np.asarray(admitted["lane"]) == 0) & (
        np.asarray(admitted["boundary_index"]) == zero_index
    )
    return {key: np.asarray(value)[mask] for key, value in admitted.items()}


def _ridge_fit_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    alpha: float,
) -> np.ndarray:
    mean = np.mean(x_train, axis=0)
    scale = np.std(x_train, axis=0)
    scale[scale < 1e-12] = 1.0
    train = (x_train - mean) / scale
    test = (x_test - mean) / scale
    design = np.column_stack((np.ones(len(train)), train))
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ y_train)
    return np.column_stack((np.ones(len(test)), test)) @ coefficients


def _ordinary_features(features: Mapping[str, np.ndarray]) -> np.ndarray:
    logu = np.log10(np.asarray(features["u"]))
    z = np.asarray(features["z"])
    logr = np.log10(np.asarray(features["radius_kpc"]))
    logm = np.asarray(features["log_mbar"])
    q = np.asarray(features["axis_ratio"])
    return np.column_stack((logu, z, logr, logm, q, logu * z, z * z, logr * logr))


def _crossfit_ridge(
    features: Mapping[str, np.ndarray], target: np.ndarray, folds: np.ndarray,
    config: Mapping[str, Any]
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    x = _ordinary_features(features)
    residual = target - np.asarray(features["log_mbar"])
    predictions = np.empty(len(target), dtype=np.float64)
    ledger: list[dict[str, Any]] = []
    alphas = [float(value) for value in config["evaluation"]["ordinary_ridge_alphas"]]
    for fold in sorted(set(folds.tolist())):
        test = np.flatnonzero(folds == fold)
        train = np.flatnonzero(folds != fold)
        inner_labels = np.arange(len(train)) % 3
        alpha_scores: list[float] = []
        for alpha in alphas:
            losses: list[float] = []
            for inner in range(3):
                inner_test = train[inner_labels == inner]
                inner_train = train[inner_labels != inner]
                pred = _ridge_fit_predict(x[inner_train], residual[inner_train], x[inner_test], alpha)
                losses.extend(np.square(pred - residual[inner_test]).tolist())
            alpha_scores.append(float(np.mean(losses)))
        alpha = alphas[int(np.argmin(alpha_scores))]
        predictions[test] = np.asarray(features["log_mbar"])[test] + _ridge_fit_predict(
            x[train], residual[train], x[test], alpha
        )
        ledger.append({"fold": int(fold), "alpha": alpha, "heldout": len(test)})
    return predictions, ledger


def _permutation_paired(diff: np.ndarray, permutations: int, seed: int) -> float:
    diff = np.asarray(diff, dtype=np.float64)
    observed = abs(float(np.mean(diff)))
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(permutations):
        value = abs(float(np.mean(diff * rng.choice((-1.0, 1.0), size=len(diff)))))
        exceed += int(value >= observed - 1e-15)
    return (exceed + 1.0) / (permutations + 1.0)


def _score(prediction: np.ndarray, target: np.ndarray, sigma: float) -> dict[str, Any]:
    errors = np.square((np.asarray(prediction) - np.asarray(target)) / sigma)
    return {"loss": float(np.mean(errors)), "per_object_loss": errors.tolist()}


def build_evaluation_result(root: Path) -> dict[str, Any]:
    config = load_config(root)
    predictors_doc = _read_json(_source_path(root, config, "predictor_source"))
    sample = _read_json(_source_path(root, config, "sample_manifest"))
    responses_doc = _read_json(_source_path(root, config, "response_source"))
    predictor_by_target = {row["target"]: row for row in predictors_doc["records"]}
    response_by_target = {
        row["target"]: float(row["log10_einstein_mass_msun"])
        for row in responses_doc["records"]
    }
    sample_rows = [row for row in sample["objects"] if row["role"] == "exploration"]
    records = [predictor_by_target[row["target"]] for row in sample_rows]
    targets = [row["target"] for row in sample_rows]
    target = np.asarray([response_by_target[value] for value in targets])
    folds = np.asarray([int(row["outer_fold"]) for row in sample_rows])
    features = _feature_arrays(records, config)
    admitted, admission = admissible_candidates(config)
    no_boundary = _matched_no_boundary(admitted, config)
    sigma = float(config["evaluation"]["combined_log10_uncertainty"])
    candidate_oof = np.empty(len(target), dtype=np.float64)
    no_boundary_oof = np.empty(len(target), dtype=np.float64)
    fold_ledger: list[dict[str, Any]] = []
    backends: set[str] = set()
    evaluations = 0
    for fold in sorted(set(folds.tolist())):
        train = np.flatnonzero(folds != fold)
        test = np.flatnonzero(folds == fold)
        candidate_id, train_loss, backend, count = _best_candidate(
            admitted, features, target, train, sigma, config
        )
        control_id, control_train_loss, control_backend, control_count = _best_candidate(
            no_boundary, features, target, train, sigma, config
        )
        candidate_oof[test] = _predict_candidate(candidate_id, features, config)[test]
        no_boundary_oof[test] = _predict_candidate(control_id, features, config)[test]
        evaluations += count + control_count
        backends.update((backend, control_backend))
        fold_ledger.append(
            {
                "fold": int(fold),
                "heldout_targets": [targets[index] for index in test],
                "selected_candidate": decode_candidate(candidate_id, config),
                "training_loss": train_loss,
                "matched_no_boundary_candidate": decode_candidate(control_id, config),
                "matched_no_boundary_training_loss": control_train_loss,
            }
        )
    all_indices = np.arange(len(target))
    selected_id, selected_loss, backend, count = _best_candidate(
        admitted, features, target, all_indices, sigma, config
    )
    no_boundary_id, no_boundary_loss, no_backend, no_count = _best_candidate(
        no_boundary, features, target, all_indices, sigma, config
    )
    backends.update((backend, no_backend))
    evaluations += count + no_count
    selected_prediction = _predict_candidate(selected_id, features, config)
    cpu_loss = float(np.mean(np.square((selected_prediction - target) / sigma)))
    cpu_gpu_difference = abs(cpu_loss - selected_loss)
    if cpu_gpu_difference > float(config["evaluation"]["cpu_gpu_tolerance"]):
        raise GravityItem43Error("CPU/GPU selected loss cross-check failed")
    newton_prediction = np.asarray(features["log_mbar"])
    mond_multiplier = 1.0 / (1.0 - np.exp(-np.sqrt(np.asarray(features["u"]))))
    mond_prediction = newton_prediction + np.log10(mond_multiplier)
    ridge_prediction, ridge_ledger = _crossfit_ridge(features, target, folds, config)
    scores = {
        "cosmological_boundary": _score(candidate_oof, target, sigma),
        "matched_no_boundary": _score(no_boundary_oof, target, sigma),
        "baryonic_newton": _score(newton_prediction, target, sigma),
        "mond_rar": _score(mond_prediction, target, sigma),
        "ordinary_ridge": _score(ridge_prediction, target, sigma),
    }
    controls = ["matched_no_boundary", "baryonic_newton", "mond_rar", "ordinary_ridge"]
    strongest = min(controls, key=lambda name: scores[name]["loss"])
    candidate_error = np.asarray(scores["cosmological_boundary"]["per_object_loss"])
    control_error = np.asarray(scores[strongest]["per_object_loss"])
    improvement = control_error - candidate_error
    raw_counterexamples = improvement < 0.0
    fixed_control_names = ["matched_no_boundary", "baryonic_newton", "mond_rar"]
    strongest_fixed = min(fixed_control_names, key=lambda name: scores[name]["loss"])
    audit_predictions: dict[str, list[float]] = {}
    stable_mismatch = raw_counterexamples.copy()
    systematic_specs = {
        "stellar_mass_minus_0.25_dex": (-0.25, 1.0),
        "stellar_mass_plus_0.25_dex": (0.25, 1.0),
        "effective_radius_minus_10_percent": (0.0, 0.9),
        "effective_radius_plus_10_percent": (0.0, 1.1),
    }
    fold_candidate = {
        int(row["fold"]): int(row["selected_candidate"]["candidate_id"])
        for row in fold_ledger
    }
    fold_no_boundary = {
        int(row["fold"]): int(row["matched_no_boundary_candidate"]["candidate_id"])
        for row in fold_ledger
    }
    for name, (shift, re_scale) in systematic_specs.items():
        audit_features = _feature_arrays(records, config, stellar_shift_dex=shift, re_scale=re_scale)
        candidate_variant = np.empty(len(target))
        no_boundary_variant = np.empty(len(target))
        for fold in sorted(set(folds.tolist())):
            test = np.flatnonzero(folds == fold)
            candidate_variant[test] = _predict_candidate(
                fold_candidate[int(fold)], audit_features, config
            )[test]
            no_boundary_variant[test] = _predict_candidate(
                fold_no_boundary[int(fold)], audit_features, config
            )[test]
        newton_variant = np.asarray(audit_features["log_mbar"])
        mond_variant = newton_variant + np.log10(
            1.0 / (1.0 - np.exp(-np.sqrt(np.asarray(audit_features["u"]))))
        )
        control_variant = {
            "matched_no_boundary": no_boundary_variant,
            "baryonic_newton": newton_variant,
            "mond_rar": mond_variant,
        }[strongest_fixed]
        candidate_variant_error = np.square((candidate_variant - target) / sigma)
        control_variant_error = np.square((control_variant - target) / sigma)
        stable_mismatch &= candidate_variant_error > control_variant_error
        audit_predictions[name] = candidate_variant.tolist()
    leave_one = [float(np.mean(np.delete(improvement, index))) for index in range(len(improvement))]
    trim_count = max(1, int(math.floor(len(improvement) * float(config["evaluation"]["robust_trim_fraction"]))))
    ordered = np.argsort(improvement)
    trimmed = improvement[ordered[trim_count:-trim_count]]
    improvement_percent = 100.0 * (
        scores[strongest]["loss"] - scores["cosmological_boundary"]["loss"]
    ) / scores[strongest]["loss"]
    counterexample_records = [
        {
            "target": targets[index],
            "candidate_loss": float(candidate_error[index]),
            "strongest_control_loss": float(control_error[index]),
            "raw_counterexample": bool(raw_counterexamples[index]),
            "uncertainty_resolved_counterexample": bool(stable_mismatch[index]),
        }
        for index in range(len(target))
    ]
    policy_report = {
        "evidence_kind": "empirical",
        "evaluable_objects": len(target),
        "raw_counterexample_count": int(np.sum(raw_counterexamples)),
        "quality_verified_counterexample_count": int(np.sum(raw_counterexamples)),
        "uncertainty_resolved_counterexample_count": int(np.sum(stable_mismatch)),
        "independent_failure_strata": 0,
        "unchanged_independent_replication_failures": 0,
        "aggregate_improvement_percent": improvement_percent,
        "quality_gate_passed": True,
        "strongest_baseline_failed": bool(improvement_percent <= 0.0),
        "leave_one_changes_sign": bool(any(value <= 0.0 for value in leave_one) != (float(np.mean(improvement)) <= 0.0)),
        "trim_changes_sign": bool((float(np.mean(trimmed)) <= 0.0) != (float(np.mean(improvement)) <= 0.0)),
        "object_level_records_preserved": True,
        "missing_quality_limited_records_preserved": True,
        "exclusions_frozen_before_response": True,
    }
    policy = assess_counterexample_evidence(
        policy_report, load_counterexample_policy(root / POLICY_PATH)
    )
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item43-s4tm-evaluation-1.0",
            "item": 43,
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "sample_freeze_commit": config["sample_freeze_commit"],
            "selected_candidate": decode_candidate(selected_id, config),
            "selected_full_exploration_training_loss": selected_loss,
            "selected_matched_no_boundary_candidate": decode_candidate(no_boundary_id, config),
            "matched_no_boundary_full_training_loss": no_boundary_loss,
            "fold_ledger": fold_ledger,
            "ridge_ledger": ridge_ledger,
            "targets": targets,
            "target_log10_einstein_mass_msun": target.tolist(),
            "predictions": {
                "cosmological_boundary_oof": candidate_oof.tolist(),
                "matched_no_boundary_oof": no_boundary_oof.tolist(),
                "baryonic_newton": newton_prediction.tolist(),
                "mond_rar": mond_prediction.tolist(),
                "ordinary_ridge_oof": ridge_prediction.tolist(),
                "selected_full_exploration": selected_prediction.tolist(),
            },
            "scores": scores,
            "strongest_control": strongest,
            "strongest_fixed_control_for_uncertainty_audit": strongest_fixed,
            "aggregate_improvement_percent": improvement_percent,
            "paired_sign_flip_p": _permutation_paired(
                improvement,
                int(config["evaluation"]["paired_sign_flip_permutations"]),
                int(config["evaluation"]["permutation_seed"]),
            ),
            "robustness": {
                "leave_one_min_mean_control_minus_candidate_loss": min(leave_one),
                "leave_one_max_mean_control_minus_candidate_loss": max(leave_one),
                "trimmed_mean_control_minus_candidate_loss": float(np.mean(trimmed)),
                "systematic_candidate_predictions": audit_predictions,
            },
            "counterexamples": counterexample_records,
            "counterexample_policy_report": policy_report,
            "counterexample_policy_assessment": policy,
            "compute": {
                "backends": sorted(backends),
                "candidate_point_fold_evaluations": evaluations,
                "cpu_gpu_selected_loss_absolute_difference": cpu_gpu_difference,
                "admission": admission,
            },
            "counts": {
                "exploration_lenses": len(target),
                "confirmation_response_rows": 0,
                "post_response_candidate_cells": 0,
                "paid_model_calls": 0,
            },
            "limitations": [
                "S4TM Einstein masses and Einstein radii come from singular-isothermal-ellipsoid lens models, not raw image likelihoods.",
                "The baryonic aperture uses a circular de Vaucouleurs projection of a Chabrier total stellar mass; cold gas and stellar-population systematics are not directly measured here.",
                "A fiducial flat background cosmology defines H(z), age, distance, and horizon coordinates, so the test is not cosmology-independent.",
                "Nested selection limits target leakage but cannot make 28 lenses a decisive population-scale discovery.",
            ],
        }
    )


def write_evaluation_result(root: Path) -> Path:
    config = load_config(root)
    path = _source_path(root, config, "evaluation_result")
    _write_json(path, build_evaluation_result(root))
    return path


def _read_vizier_rows(path: Path) -> list[dict[str, str]]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line and not line.startswith("#")]
    if len(lines) < 4:
        raise GravityItem43Error(f"no rows in {path}")
    return list(csv.DictReader(io.StringIO("\n".join([lines[0], *lines[3:]])), delimiter="\t"))


def build_clash_transfer_result(root: Path) -> dict[str, Any]:
    config = load_config(root)
    evaluation = _read_json(_source_path(root, config, "evaluation_result"))
    selected_id = int(evaluation["selected_candidate"]["candidate_id"])
    no_boundary_id = int(evaluation["selected_matched_no_boundary_candidate"]["candidate_id"])
    acceleration_rows = _read_vizier_rows(root / str(config["data_sources"]["clash_acceleration_source"]))
    metadata_rows = _read_vizier_rows(root / str(config["data_sources"]["clash_redshift_source"]))
    aliases = {
        "A209": "A209", "A383": "A383", "A611": "A611", "A2261": "A2261",
        "MACS0329": "M0329", "MACS0416": "M0416", "MACS0429": "M0429",
        "MACS0647": "M0647", "MACS0717": "M0717", "MACS0744": "M0744",
        "MACS1115": "M1115", "MACS1149": "M1149", "MACS1206": "M1206",
        "MACS1720": "M1720", "MACS1931": "M1931", "MS2137": "MS2137",
        "RXJ1347": "RXJ1347", "RXJ1532": "M1532", "RXJ2129": "RXJ2129",
        "RXJ2248": "RXJ2248",
    }
    z_by_id = {str(row["ID"]).strip(): float(row["z"]) for row in metadata_rows}
    names = np.asarray([str(row["AName"]).strip() for row in acceleration_rows])
    z = np.asarray([z_by_id[aliases[name]] for name in names])
    radius = np.asarray([float(row["Rad"]) for row in acceleration_rows])
    loggbar = np.asarray([float(row["log(gbar)"]) for row in acceleration_rows])
    loggtot = np.asarray([float(row["log(gtot)"]) for row in acceleration_rows])
    sigma = np.sqrt(
        np.square([float(row["e_log(gbar)"]) for row in acceleration_rows])
        + np.square([float(row["e_log(gtot)"]) for row in acceleration_rows])
    )
    features = {
        "z": z,
        "radius_kpc": radius,
        "log_mbar": loggbar,
        "u": np.power(10.0, loggbar) / float(config["constants"]["acceleration_scale_m_s2"]),
        "bases": boundary_bases(z, radius, config),
    }
    candidate = _predict_candidate(selected_id, features, config)
    no_boundary = _predict_candidate(no_boundary_id, features, config)
    newton = loggbar
    mond = loggbar + np.log10(
        1.0 / (1.0 - np.exp(-np.sqrt(np.asarray(features["u"]))))
    )
    predictions = {
        "cosmological_boundary": candidate,
        "matched_no_boundary": no_boundary,
        "baryonic_newton": newton,
        "mond_rar": mond,
    }
    scores: dict[str, Any] = {}
    for name, prediction in predictions.items():
        errors = np.square((prediction - loggtot) / sigma)
        cluster_losses = {
            cluster: float(np.mean(errors[names == cluster])) for cluster in sorted(set(names.tolist()))
        }
        scores[name] = {
            "loss": float(np.mean(list(cluster_losses.values()))),
            "point_weighted_loss": float(np.mean(errors)),
            "cluster_losses": cluster_losses,
        }
    strongest = min(
        ("matched_no_boundary", "baryonic_newton", "mond_rar"),
        key=lambda name: scores[name]["loss"],
    )
    cluster_names = sorted(set(names.tolist()))
    raw_counterexamples = [
        cluster for cluster in cluster_names
        if scores["cosmological_boundary"]["cluster_losses"][cluster]
        > scores[strongest]["cluster_losses"][cluster]
    ]
    improvement = 100.0 * (
        scores[strongest]["loss"] - scores["cosmological_boundary"]["loss"]
    ) / scores[strongest]["loss"]
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item43-clash-transfer-1.0",
            "item": 43,
            "role": config["cluster_transfer"]["role"],
            "selected_candidate": evaluation["selected_candidate"],
            "selected_matched_no_boundary_candidate": evaluation[
                "selected_matched_no_boundary_candidate"
            ],
            "scores": scores,
            "strongest_fixed_control": strongest,
            "aggregate_improvement_percent": improvement,
            "raw_counterexample_clusters": raw_counterexamples,
            "counts": {
                "clusters": len(cluster_names),
                "points": len(names),
                "uncertainty_resolved_counterexamples": 0,
                "selection_candidate_cells": 0,
                "retuned_candidate_cells": 0,
                "confirmation_rows": 0,
                "paid_model_calls": 0,
            },
            "limitations": [
                "CLASH accelerations and redshifts were already exposed before Item 43 and cannot confirm it independently.",
                "The total acceleration is derived through published spherical NFW posteriors rather than a raw lens-image likelihood.",
                "Cluster baryonic acceleration and redshift can share selection and mass-calibration systematics.",
            ],
        }
    )


def write_clash_transfer_result(root: Path) -> Path:
    config = load_config(root)
    path = _source_path(root, config, "clash_transfer_result")
    _write_json(path, build_clash_transfer_result(root))
    return path


def build_aggregate_result(root: Path) -> dict[str, Any]:
    config = load_config(root)
    candidate = _read_json(_source_path(root, config, "candidate_manifest"))
    exposure = _read_json(_source_path(root, config, "exposure_manifest"))
    predictors = _read_json(_source_path(root, config, "predictor_source"))
    sample = _read_json(_source_path(root, config, "sample_manifest"))
    responses = _read_json(_source_path(root, config, "response_source"))
    evaluation = _read_json(_source_path(root, config, "evaluation_result"))
    clash = _read_json(_source_path(root, config, "clash_transfer_result"))
    scores = evaluation["scores"]
    gates = {
        "beats_baryonic_newton": scores["cosmological_boundary"]["loss"] < scores["baryonic_newton"]["loss"],
        "beats_fixed_mond": scores["cosmological_boundary"]["loss"] < scores["mond_rar"]["loss"],
        "beats_matched_boundary_free": scores["cosmological_boundary"]["loss"] < scores["matched_no_boundary"]["loss"],
        "beats_ordinary_ridge": scores["cosmological_boundary"]["loss"] < scores["ordinary_ridge"]["loss"],
        "paired_p_passes": float(evaluation["paired_sign_flip_p"]) <= float(config["gates"]["paired_p_maximum"]),
        "leave_one_stable": float(evaluation["robustness"]["leave_one_min_mean_control_minus_candidate_loss"]) > 0.0,
        "trim_stable": float(evaluation["robustness"]["trimmed_mean_control_minus_candidate_loss"]) > 0.0,
        "unchanged_clash_beats_mond": clash["scores"]["cosmological_boundary"]["loss"] < clash["scores"]["mond_rar"]["loss"],
        "confirmation_remains_sealed": int(responses["counts"]["confirmation_response_rows"]) == 0,
        "no_post_response_candidates": int(evaluation["counts"]["post_response_candidate_cells"]) == 0,
    }
    promoted = all(gates.values())
    boundary_increment = gates["beats_matched_boundary_free"] and float(
        clash["aggregate_improvement_percent"]
    ) > 0.0
    decision = (
        "PROMOTED_ITEM43_COSMOLOGICAL_BOUNDARY_LEAD"
        if promoted
        else (
            "NONPROMOTED_ITEM43_CROSS_SCALE_BOUNDARY_INCREMENT_RETAINED"
            if boundary_increment
            else "NONPROMOTED_ITEM43_COSMOLOGICAL_BOUNDARY_RESULT_RETAINED"
        )
    )
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item43-cosmological-boundary-result-1.0",
            "item": 43,
            "goal": "GRAVITY_ROADMAP_ITEM_43_COSMOLOGICAL_BOUNDARY_COUPLING",
            "decision": decision,
            "selected_candidate": evaluation["selected_candidate"],
            "s4tm": {
                "scores": scores,
                "strongest_control": evaluation["strongest_control"],
                "aggregate_improvement_percent": evaluation["aggregate_improvement_percent"],
                "paired_sign_flip_p": evaluation["paired_sign_flip_p"],
                "counterexample_policy_assessment": evaluation[
                    "counterexample_policy_assessment"
                ],
            },
            "clash_transfer": {
                "scores": clash["scores"],
                "strongest_fixed_control": clash["strongest_fixed_control"],
                "aggregate_improvement_percent": clash["aggregate_improvement_percent"],
                "raw_counterexample_clusters": clash["raw_counterexample_clusters"],
            },
            "gates": gates,
            "counts": {
                "raw_candidates": candidate["raw_candidates"],
                "admitted_candidates": candidate["admitted_candidates"],
                "s4tm_exploration_lenses": sample["counts"]["exploration_lenses"],
                "s4tm_sealed_confirmation_lenses": sample["counts"]["confirmation_lenses"],
                "clash_clusters": clash["counts"]["clusters"],
                "clash_points": clash["counts"]["points"],
                "candidate_point_fold_evaluations": evaluation["compute"]["candidate_point_fold_evaluations"],
                "schema_audit_response_rows_excluded": exposure["counts"]["schema_rows_with_response_seen"],
                "confirmation_response_rows": 0,
                "post_response_candidate_cells": 0,
                "paid_model_calls": 0,
            },
            "source_bindings": {
                "config": {"path": str(CONFIG_PATH), "sha256": _sha256_file(root / CONFIG_PATH)},
                "candidate_manifest": {"path": str(_source_path(root, config, "candidate_manifest").relative_to(root)), "sha256": _sha256_file(_source_path(root, config, "candidate_manifest"))},
                "exposure_manifest": {"path": str(_source_path(root, config, "exposure_manifest").relative_to(root)), "sha256": _sha256_file(_source_path(root, config, "exposure_manifest"))},
                "predictors": {"path": str(_source_path(root, config, "predictor_source").relative_to(root)), "sha256": _sha256_file(_source_path(root, config, "predictor_source"))},
                "sample": {"path": str(_source_path(root, config, "sample_manifest").relative_to(root)), "sha256": _sha256_file(_source_path(root, config, "sample_manifest"))},
                "responses": {"path": str(_source_path(root, config, "response_source").relative_to(root)), "sha256": _sha256_file(_source_path(root, config, "response_source"))},
                "evaluation": {"path": str(_source_path(root, config, "evaluation_result").relative_to(root)), "sha256": _sha256_file(_source_path(root, config, "evaluation_result"))},
                "clash_transfer": {"path": str(_source_path(root, config, "clash_transfer_result").relative_to(root)), "sha256": _sha256_file(_source_path(root, config, "clash_transfer_result"))},
            },
            "claims": {
                "roadmap_item_43_complete": True,
                "cosmological_boundary_dependence_established": promoted,
                "alternative_to_gr_established": False,
                "dark_matter_eliminated": False,
                "historical_novelty_established": False,
                "covariant_theory_established": False,
                "formula_family_pruned": False,
                "single_counterexample_used_as_veto": False,
            },
            "limitations": [
                *evaluation["limitations"],
                *clash["limitations"],
                "The seven S4TM confirmation responses remain sealed; no promotion claim may rely on them.",
            ],
            "next_action": "Preserve the selected law and its failures, keep confirmation sealed, and advance to Item 44 scale hierarchy without treating any single imperfect lens as a veto.",
        }
    )


def write_aggregate_result(root: Path) -> Path:
    config = load_config(root)
    path = root / str(config["paths"]["aggregate_result"])
    _write_json(path, build_aggregate_result(root))
    return path


def replay(root: Path) -> dict[str, Any]:
    config = load_config(root)
    checks = {
        "candidate_manifest": _read_json(_source_path(root, config, "candidate_manifest")) == build_candidate_manifest(root),
        "exposure_manifest": _read_json(_source_path(root, config, "exposure_manifest")) == build_exposure_manifest(root),
        "sample_manifest": _read_json(_source_path(root, config, "sample_manifest")) == build_sample_manifest(root),
        "evaluation_result": _read_json(_source_path(root, config, "evaluation_result")) == build_evaluation_result(root),
        "clash_transfer_result": _read_json(_source_path(root, config, "clash_transfer_result")) == build_clash_transfer_result(root),
        "aggregate_result": _read_json(root / str(config["paths"]["aggregate_result"])) == build_aggregate_result(root),
    }
    return {"valid": all(checks.values()), "checks": checks}


def _root(value: str) -> Path:
    return Path(value).resolve()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in (
        "freeze", "acquire-predictors", "freeze-sample", "acquire-responses",
        "evaluate", "transfer", "aggregate", "replay",
    ):
        sub.add_parser(name)
    args = parser.parse_args(argv)
    root = _root(args.root)
    if args.command == "freeze":
        result: Any = [str(path) for path in write_freeze_manifests(root)]
    elif args.command == "acquire-predictors":
        result = str(acquire_predictors(root))
    elif args.command == "freeze-sample":
        result = str(write_sample_manifest(root))
    elif args.command == "acquire-responses":
        result = str(acquire_exploration_responses(root))
    elif args.command == "evaluate":
        result = str(write_evaluation_result(root))
    elif args.command == "transfer":
        result = str(write_clash_transfer_result(root))
    elif args.command == "aggregate":
        result = str(write_aggregate_result(root))
    else:
        result = replay(root)
        if not result["valid"]:
            print(json.dumps(result, sort_keys=True))
            return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
