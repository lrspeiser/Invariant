"""Exhaustive G1 interaction repair for the NGC2955 counterexample."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .gravity_g0_experiment import load_config as load_g0_config
from .gravity_g0_experiment import score_predictions
from .gravity_g1_atlas_repair_v2 import (
    _rar_base_v2,
    _score_batch,
)
from .gravity_g1_atlas_repair_v2 import (
    validate_receipt as validate_v2_receipt,
)
from .gravity_g1_pilot import (
    FAILURE_NAMES,
    _baseline_contract,
    _best_rows,
    _binding,
    _file_sha256,
    _load_json,
    _merge_best,
    _metric,
)
from .gravity_g1_pilot_v2 import _fit_two_columns
from .gravity_g1_pilot_v3 import FEATURE_IDS, baryonic_features
from .sigma_core import canonical_json_bytes, canonical_sha256
from .sparc_full_sample import Galaxy, assemble

SCHEMA = "invariant-gravity-g1-atlas-interaction-repair-receipt-3.0"
CONFIG_SCHEMA = "invariant-gravity-g1-atlas-interaction-repair-config-3.0"
CONFIG_PATH = "configs/gravity_g1_atlas_repair_v3.json"
SOURCE_PATH = "src/sigma_theory_compiler/gravity_g1_atlas_repair_v3.py"
TEST_PATH = "tests/test_gravity_g1_atlas_repair_v3.py"
OUTPUT_PATH = "runs/gravity/g1-atlas/repair-v3.json"
COMPONENT_COUNT = 3_488
PAIR_COUNT = COMPONENT_COUNT * (COMPONENT_COUNT - 1) // 2


class GravityG1AtlasInteractionRepairError(ValueError):
    """The interaction repair or its evidence is inconsistent."""


def load_config(root: Path) -> Mapping[str, Any]:
    """Load the sealed interaction grammar and validate its failed predecessor."""

    root = root.resolve()
    config = _load_json(root / CONFIG_PATH)
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise GravityG1AtlasInteractionRepairError("G1 interaction config schema changed")
    predecessor = config.get("predecessor_binding")
    if not isinstance(predecessor, Mapping):
        raise GravityG1AtlasInteractionRepairError("G1 interaction predecessor is missing")
    path = root / str(predecessor["path"])
    if _file_sha256(path) != predecessor.get("file_sha256"):
        raise GravityG1AtlasInteractionRepairError("G1 interaction predecessor file changed")
    receipt = _load_json(path)
    validate_v2_receipt(receipt, root=root)
    if receipt.get("content_sha256") != predecessor.get("content_sha256"):
        raise GravityG1AtlasInteractionRepairError("G1 interaction predecessor seal changed")
    if receipt.get("decision") != predecessor.get("required_decision"):
        raise GravityG1AtlasInteractionRepairError("G1 interaction predecessor decision changed")
    if (
        receipt.get("counts", {}).get("union_covered_galaxies")
        != predecessor.get("required_union_covered_galaxies")
    ):
        raise GravityG1AtlasInteractionRepairError("G1 interaction predecessor coverage changed")
    if receipt.get("repair", {}).get("galaxy") != predecessor.get("required_counterexample"):
        raise GravityG1AtlasInteractionRepairError("G1 interaction counterexample changed")
    grammar = config.get("component_grammar", {})
    if grammar.get("component_count") != COMPONENT_COUNT or grammar.get("candidate_count") != PAIR_COUNT:
        raise GravityG1AtlasInteractionRepairError("G1 interaction grammar size changed")
    shell = config.get("candidate_shell", {})
    if shell.get("maximum_local_constants") != 2 or shell.get("proposal_reads_vobs") is not False:
        raise GravityG1AtlasInteractionRepairError("G1 interaction shell boundary changed")
    disclosure = config.get("diagnostic_disclosure", {})
    if (
        disclosure.get("same_exploration_counterexample_used_to_design_grammar") is not True
        or disclosure.get("a_member_of_this_family_was_observed_to_pass_before_sealing")
        is not True
    ):
        raise GravityG1AtlasInteractionRepairError("G1 interaction disclosure changed")
    if config.get("admission", {}).get("confirmation_evaluator_accesses_allowed") != 0:
        raise GravityG1AtlasInteractionRepairError("G1 interaction permits confirmation access")
    return config


def pair_batches(total: int, batch_size: int) -> Iterator[np.ndarray]:
    """Enumerate the first ``total`` unique component pairs in canonical order."""

    if not 1 <= total <= PAIR_COUNT:
        raise GravityG1AtlasInteractionRepairError("interaction pair count is outside grammar")
    produced = 0
    first = 0
    second = 1
    while produced < total:
        take = min(batch_size, total - produced)
        rows: list[np.ndarray] = []
        remaining = take
        while remaining:
            available = COMPONENT_COUNT - second
            width = min(remaining, available)
            rows.append(
                first * COMPONENT_COUNT
                + np.arange(second, second + width, dtype=np.int64)
            )
            second += width
            remaining -= width
            if second == COMPONENT_COUNT:
                first += 1
                second = first + 1
        batch = np.concatenate(rows)
        produced += batch.size
        yield batch


def _chebyshev_values(xp: Any, z: Any, maximum_degree: int, dtype: Any) -> list[Any]:
    values = [xp.ones_like(z), z]
    for _degree in range(2, maximum_degree + 1):
        values.append(dtype(2) * z * values[-1] - values[-2])
    return values


def _legendre_values(xp: Any, z: Any, maximum_degree: int, dtype: Any) -> list[Any]:
    values = [xp.ones_like(z), z]
    for degree in range(2, maximum_degree + 1):
        values.append(
            (dtype(2 * degree - 1) * z * values[-1] - dtype(degree - 1) * values[-2])
            / dtype(degree)
        )
    return values


def normalized_baryonic_features(
    xp: Any,
    galaxy: Galaxy,
    a0: float,
    dtype: Any,
) -> Any:
    """Map each target-blind baryonic feature to [-1, 1]."""

    features = baryonic_features(galaxy, a0)
    rows = xp.asarray(np.vstack([features[name] for name in FEATURE_IDS]), dtype=dtype)
    low = xp.min(rows, axis=1)
    high = xp.max(rows, axis=1)
    span = high - low
    if bool(xp.any(span <= dtype(0))):
        raise GravityG1AtlasInteractionRepairError("constant baryonic feature in repair galaxy")
    return dtype(2) * (rows - low[:, None]) / span[:, None] - dtype(1)


def interaction_components(
    xp: Any,
    normalized_features: Any,
    dtype: Any,
) -> tuple[Any, list[dict[str, Any]]]:
    """Materialize the exact 3,488-component typed interaction grammar."""

    arrays: list[Any] = []
    metadata: list[dict[str, Any]] = []
    chebyshev: list[list[Any]] = []
    frequencies = np.geomspace(0.25, 16.0, 24)
    centers = np.linspace(-1.0, 1.0, 17)
    powers = (0.5, 1.0, 2.0, 3.0)
    scales = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
    for feature_index, feature_id in enumerate(FEATURE_IDS):
        z = normalized_features[feature_index]
        cheb = _chebyshev_values(xp, z, 16, dtype)
        legendre = _legendre_values(xp, z, 16, dtype)
        chebyshev.append(cheb)
        for degree in range(1, 17):
            arrays.append(cheb[degree])
            metadata.append(
                {"degree": degree, "family": "chebyshev", "feature": feature_id}
            )
        for degree in range(1, 17):
            arrays.append(legendre[degree])
            metadata.append(
                {"degree": degree, "family": "legendre", "feature": feature_id}
            )
        for frequency in frequencies:
            arrays.append(xp.sin(dtype(np.pi * frequency) * z))
            metadata.append(
                {
                    "family": "fourier_sine",
                    "feature": feature_id,
                    "frequency": _metric(float(frequency)),
                }
            )
            arrays.append(xp.cos(dtype(np.pi * frequency) * z))
            metadata.append(
                {
                    "family": "fourier_cosine",
                    "feature": feature_id,
                    "frequency": _metric(float(frequency)),
                }
            )
        for center in centers:
            for power in powers:
                if center < 1.0:
                    arrays.append(xp.maximum(dtype(0), z - dtype(center)) ** dtype(power))
                    metadata.append(
                        {
                            "center": _metric(float(center)),
                            "family": "positive_hinge",
                            "feature": feature_id,
                            "power": _metric(power),
                        }
                    )
                if center > -1.0:
                    arrays.append(xp.maximum(dtype(0), dtype(center) - z) ** dtype(power))
                    metadata.append(
                        {
                            "center": _metric(float(center)),
                            "family": "negative_hinge",
                            "feature": feature_id,
                            "power": _metric(power),
                        }
                    )
        for center in centers:
            for scale in scales:
                arrays.append(xp.tanh(dtype(scale) * (z - dtype(center))))
                metadata.append(
                    {
                        "center": _metric(float(center)),
                        "family": "tanh_transition",
                        "feature": feature_id,
                        "scale": _metric(scale),
                    }
                )
    for first_feature in range(len(FEATURE_IDS)):
        for second_feature in range(first_feature + 1, len(FEATURE_IDS)):
            for first_degree in range(1, 7):
                for second_degree in range(1, 7):
                    arrays.append(
                        chebyshev[first_feature][first_degree]
                        * chebyshev[second_feature][second_degree]
                    )
                    metadata.append(
                        {
                            "family": "chebyshev_feature_product",
                            "first_degree": first_degree,
                            "first_feature": FEATURE_IDS[first_feature],
                            "second_degree": second_degree,
                            "second_feature": FEATURE_IDS[second_feature],
                        }
                    )
    if len(arrays) != COMPONENT_COUNT or len(metadata) != COMPONENT_COUNT:
        raise GravityG1AtlasInteractionRepairError("interaction component count changed")
    return xp.stack(arrays), metadata


def replay_candidate(
    galaxy: Galaxy,
    ordinal: int,
    g0_config: Mapping[str, Any],
    *,
    contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Make the authoritative CPU-FP64 decision for one interaction pair."""

    if contract is None:
        contract = _baseline_contract(galaxy, g0_config)
    arrays = contract["arrays"]
    a0 = float(
        next(item for item in g0_config["baselines"] if item["id"] == "empirical_rar")[
            "g_dagger_km2_s2_kpc"
        ]
    )
    normalized = normalized_baryonic_features(np, galaxy, a0, np.float64)
    components, metadata = interaction_components(np, normalized, np.float64)
    first = ordinal // COMPONENT_COUNT
    second = ordinal % COMPONENT_COUNT
    if not 0 <= first < second < COMPONENT_COUNT:
        return {"admitted": False, "failure": "invalid_pair", "ordinal": ordinal}
    phi1 = components[first]
    phi2 = components[second]
    if np.any(~np.isfinite(phi1)) or np.any(~np.isfinite(phi2)):
        return {"admitted": False, "failure": "invalid_domain", "ordinal": ordinal}
    base_v2 = _rar_base_v2(arrays, a0)
    column1 = arrays["radius"] * phi1
    column2 = arrays["radius"] * phi2
    fit_arrays = dict(arrays)
    fit_arrays["vbar2"] = base_v2
    predictions = np.empty_like(arrays["vobs"])
    folds: list[dict[str, Any]] = []
    failure_set: set[str] = set()
    all_pass = True
    for row in contract["fold_rows"]:
        fold = row["fold"]
        try:
            coefficient1, coefficient2 = _fit_two_columns(
                column1, column2, fit_arrays, fold.training
            )
        except Exception:  # noqa: BLE001 - singular fits are typed rejections
            return {"admitted": False, "failure": "ill_conditioned_fold", "ordinal": ordinal}
        held = np.asarray(fold.holdout, dtype=np.int64)
        prediction2 = (
            base_v2[held]
            + coefficient1 * column1[held]
            + coefficient2 * column2[held]
        )
        if np.any(~np.isfinite(prediction2)) or np.any(prediction2 <= 0):
            return {"admitted": False, "failure": "nonpositive_heldout_v2", "ordinal": ordinal}
        predictions[held] = np.sqrt(prediction2)
        score = score_predictions(predictions[held], arrays["vobs"][held], arrays["sigma"][held])
        thresholds = row["thresholds"]
        coverage_count = round(float(score["coverage_two_sigma"]) * len(fold.holdout))
        checks = {
            "beats_newtonian": float(score["chi_square"]) < thresholds["newtonian"],
            "beats_wrong_law": float(score["chi_square"]) < thresholds["wrong"],
            "meets_empirical_rar": float(score["chi_square"]) <= thresholds["rar"],
            "meets_nfw_ceiling": float(score["chi_square"]) <= thresholds["nfw"],
            "meets_two_sigma_coverage": coverage_count >= thresholds["coverage_count"],
        }
        for name, passed in checks.items():
            if not passed:
                failure_set.add(name)
        all_pass &= all(checks.values())
        folds.append(
            {
                "A_km2_s2_kpc": _metric(coefficient1),
                "B_km2_s2_kpc": _metric(coefficient2),
                "checks": checks,
                "fold_id": fold.fold_id,
                "held_out_indices": list(fold.holdout),
                "score": score,
            }
        )
    aggregate_score = score_predictions(predictions, arrays["vobs"], arrays["sigma"])
    baseline = contract["aggregate"]
    coverage_count = round(float(aggregate_score["coverage_two_sigma"]) * galaxy.count)
    aggregate_checks = {
        "beats_newtonian": float(aggregate_score["chi_square"])
        < float(baseline["newtonian_baryons"]["chi_square"]),
        "beats_wrong_law": float(aggregate_score["chi_square"])
        < float(baseline["wrong_high_acceleration_boost"]["chi_square"]),
        "meets_empirical_rar": float(aggregate_score["chi_square"])
        <= float(baseline["empirical_rar"]["chi_square"]),
        "meets_nfw_ceiling": float(aggregate_score["chi_square"])
        <= float(baseline["nfw_halo_ceiling"]["chi_square"]) + 2.0 * galaxy.count,
        "meets_two_sigma_coverage": coverage_count
        >= math.ceil(
            min(0.9, float(baseline["nfw_halo_ceiling"]["coverage_two_sigma"]))
            * galaxy.count
            - 1e-12
        ),
    }
    for name, passed in aggregate_checks.items():
        if not passed:
            failure_set.add(f"aggregate_{name}")
    all_pass &= all(aggregate_checks.values())
    component_metadata = [metadata[first], metadata[second]]
    ir = {
        "base": "empirical_RAR",
        "components": component_metadata,
        "feature_normalization": "within_galaxy_baryonic_minmax_to_minus1_plus1",
        "local_constants": ["A_km2_s2_kpc", "B_km2_s2_kpc"],
        "ordinal": ordinal,
        "shell": "rar2+r*(A*Phi1(z_baryons)+B*Phi2(z_baryons))",
    }
    formula_bytes = len(canonical_json_bytes(ir))
    grammar_bits = math.ceil(math.log2(PAIR_COUNT))
    return {
        "admitted": all_pass,
        "aggregate_checks": aggregate_checks,
        "aggregate_score": aggregate_score,
        "base_family": "empirical_RAR_MOND_phenomenology",
        "components": component_metadata,
        "description_length": {
            "canonical_formula_bytes": formula_bytes,
            "grammar_address_bits": grammar_bits,
            "local_constant_bits": 128,
            "total_bits": 8 * formula_bytes + 128 + grammar_bits,
        },
        "failure_obligations": sorted(failure_set),
        "formula": "V_pred^2=V_RAR^2+r*(A*Phi_1(z_baryons)+B*Phi_2(z_baryons))",
        "folds": folds,
        "historical_novelty_established": False,
        "origin_assessment": "new_combination_of_known_ideas",
        "ordinal": ordinal,
        "prediction_sha256": canonical_sha256(
            [format(float(value), ".15e") for value in predictions]
        ),
    }


def search(
    galaxy: Galaxy,
    config: Mapping[str, Any],
    g0_config: Mapping[str, Any],
    *,
    candidate_count: int,
    use_gpu: bool,
) -> dict[str, Any]:
    """Exhaustively screen the declared interaction pairs."""

    prefilter = config["gpu_prefilter"]
    batch_size = 65_536
    retain = int(prefilter["retained_candidates_for_cpu_replay"])
    contract = _baseline_contract(galaxy, g0_config)
    arrays = contract["arrays"]
    a0 = float(
        next(item for item in g0_config["baselines"] if item["id"] == "empirical_rar")[
            "g_dagger_km2_s2_kpc"
        ]
    )
    base_v2 = _rar_base_v2(arrays, a0)
    if use_gpu:
        import cupy as xp

        device = xp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    else:
        xp = np
        device = "cpu-numpy"
    normalized = normalized_baryonic_features(xp, galaxy, a0, xp.float32)
    component_matrix, _metadata = interaction_components(xp, normalized, xp.float32)
    failure_counts = np.zeros(len(FAILURE_NAMES), dtype=np.int64)
    best_valid: list[tuple[float, int]] = []
    best_prefilter: list[tuple[float, int]] = []
    evaluated = 0
    started = time.perf_counter()
    for host_ordinals in pair_batches(candidate_count, batch_size):
        ordinals = xp.asarray(host_ordinals)
        first = ordinals // COMPONENT_COUNT
        second = ordinals % COMPONENT_COUNT
        phi1 = component_matrix[first]
        phi2 = component_matrix[second]
        valid = xp.all(xp.isfinite(phi1) & xp.isfinite(phi2), axis=1)
        scores, survivor, reasons = _score_batch(
            xp,
            phi1,
            phi2,
            valid,
            contract,
            base_v2,
            dtype=xp.float32,
            relative_slack=float(prefilter["relative_score_slack"]),
            absolute_slack=float(prefilter["absolute_score_slack"]),
            coverage_slack=int(prefilter["coverage_count_slack"]),
        )
        counts = xp.bincount(reasons, minlength=len(FAILURE_NAMES))
        failure_counts += counts.get() if use_gpu else counts
        best_valid = _merge_best(best_valid, _best_rows(xp, scores, valid, ordinals, 16), 64)
        best_prefilter = _merge_best(
            best_prefilter,
            _best_rows(xp, scores, survivor, ordinals, 64),
            retain,
        )
        evaluated += int(host_ordinals.size)
    if use_gpu:
        xp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - started
    if evaluated != candidate_count or int(np.sum(failure_counts)) != candidate_count:
        raise GravityG1AtlasInteractionRepairError("interaction trial accounting changed")
    replayed = [
        replay_candidate(galaxy, ordinal, g0_config, contract=contract)
        for _, ordinal in best_prefilter
    ]
    admitted = [item for item in replayed if item["admitted"]]
    admitted.sort(
        key=lambda item: (
            float(item["aggregate_score"]["chi_square"]),
            int(item["description_length"]["total_bits"]),
            int(item["ordinal"]),
        )
    )
    diagnostics = []
    by_ordinal = {int(item["ordinal"]): item for item in replayed}
    for score, ordinal in best_valid[:16]:
        item = by_ordinal.get(ordinal)
        if item is None:
            item = replay_candidate(galaxy, ordinal, g0_config, contract=contract)
        diagnostics.append(
            {
                "admitted": item["admitted"],
                "cpu_fp64_chi_square": item.get("aggregate_score", {}).get("chi_square"),
                "gpu_fp32_chi_square": _metric(score),
                "ordinal": ordinal,
            }
        )
    return {
        "admitted_count_among_cpu_replays": len(admitted),
        "candidate_count": candidate_count,
        "candidates_per_second": _metric(candidate_count / elapsed),
        "component_count": COMPONENT_COUNT,
        "confirmation_evaluator_access_count": 0,
        "cpu_fp64_admitted_pareto": admitted[:64],
        "cpu_replay_count": len(replayed),
        "device": device,
        "elapsed_seconds": _metric(elapsed),
        "failure_ledger": {
            name: int(count) for name, count in zip(FAILURE_NAMES, failure_counts, strict=True)
        },
        "space_exhausted": candidate_count == PAIR_COUNT,
        "top_domain_valid_diagnostics": diagnostics,
    }


def build_receipt(
    root: Path,
    *,
    candidate_count_override: int | None = None,
    use_gpu: bool = True,
) -> dict[str, Any]:
    """Run the interaction grammar and issue the G1 union decision."""

    root = root.resolve()
    config = load_config(root)
    g0_config = load_g0_config(root)
    galaxies = {galaxy.name: galaxy for galaxy in assemble(root).exploration}
    galaxy = galaxies["NGC2955"]
    candidate_count = PAIR_COUNT if candidate_count_override is None else candidate_count_override
    trial = search(
        galaxy,
        config,
        g0_config,
        candidate_count=candidate_count,
        use_gpu=use_gpu,
    )
    admitted = trial["cpu_fp64_admitted_pareto"]
    full_run = candidate_count_override is None and use_gpu and candidate_count == PAIR_COUNT
    passed = full_run and bool(admitted)
    body: dict[str, Any] = {
        "schema_version": SCHEMA,
        "goal": "G1",
        "decision": "PASS_G1_ATLAS_UNION_139_OF_139" if passed else "BLOCK_G1_INTERACTION_REPAIR",
        "claims": {
            "alternative_to_gr_discovered": False,
            "confirmation_galaxy_evaluated": False,
            "formula_is_universal": False,
            "g2_equivalence_authorized": passed,
            "historical_novelty_established": False,
            "independent_confirmation_completed": False,
        },
        "config": {"content_sha256": canonical_sha256(config), "path": CONFIG_PATH},
        "counts": {
            "cumulative_candidate_galaxy_trials": 14_000_000_000 + candidate_count,
            "new_interaction_candidate_galaxy_trials": candidate_count,
            "confirmation_evaluator_accesses": 0,
            "union_covered_galaxies": 138 + int(bool(admitted)),
            "union_exploration_galaxies": 139,
        },
        "diagnostic_disclosure": config["diagnostic_disclosure"],
        "lineage_assessment": {
            "base": "known_family_instance",
            "component_families": "known_family_instances",
            "construction": "new_combination_of_known_ideas",
            "authoritative_for_novelty": False,
        },
        "predecessor": {
            "content_sha256": config["predecessor_binding"]["content_sha256"],
            "decision": config["predecessor_binding"]["required_decision"],
            "path": config["predecessor_binding"]["path"],
        },
        "repair": {
            "covered": bool(admitted),
            "galaxy": galaxy.name,
            "point_count": galaxy.count,
            "retained_pareto": admitted[:64],
            "trial": trial,
        },
        "limitations": [
            "The interaction grammar was designed on the exploration-only NGC2955 counterexample, and a passing member was observed before this exhaustive replay.",
            "The PASS, if issued, establishes exploration-atlas completeness only; it is not an independent prediction.",
            "RAR and every component basis are known families; only their searched combination is new in this pipeline.",
            "The formula retains two coefficients fitted separately within NGC2955 training folds.",
            "G2 may analyze equivalence, but confirmation galaxies remain sealed until a universal zero-local-constant law reaches G4.",
        ],
        "source_bindings": {
            "config": _binding(root, CONFIG_PATH),
            "source": _binding(root, SOURCE_PATH),
            "test": _binding(root, TEST_PATH),
        },
    }
    body["content_sha256"] = canonical_sha256(body)
    return body


def validate_receipt(receipt: Mapping[str, Any], *, root: Path) -> None:
    """Validate a checked interaction-repair receipt."""

    root = root.resolve()
    if receipt.get("schema_version") != SCHEMA:
        raise GravityG1AtlasInteractionRepairError("interaction receipt schema changed")
    body = dict(receipt)
    supplied = body.pop("content_sha256", None)
    if supplied != canonical_sha256(body):
        raise GravityG1AtlasInteractionRepairError("interaction receipt content seal changed")
    config = load_config(root)
    if receipt.get("config", {}).get("content_sha256") != canonical_sha256(config):
        raise GravityG1AtlasInteractionRepairError("interaction config binding changed")
    for key, path in (("config", CONFIG_PATH), ("source", SOURCE_PATH), ("test", TEST_PATH)):
        if receipt.get("source_bindings", {}).get(key) != _binding(root, path):
            raise GravityG1AtlasInteractionRepairError(f"interaction {key} binding changed")
    counts = receipt.get("counts", {})
    if counts.get("confirmation_evaluator_accesses") != 0:
        raise GravityG1AtlasInteractionRepairError("interaction repair records confirmation access")
    claims = receipt.get("claims", {})
    if claims.get("historical_novelty_established") is not False:
        raise GravityG1AtlasInteractionRepairError("interaction repair overstates novelty")
    if claims.get("independent_confirmation_completed") is not False:
        raise GravityG1AtlasInteractionRepairError("interaction repair overstates confirmation")
    passed = receipt.get("decision") == "PASS_G1_ATLAS_UNION_139_OF_139"
    if passed and (
        counts.get("new_interaction_candidate_galaxy_trials") != PAIR_COUNT
        or counts.get("union_covered_galaxies") != 139
        or receipt.get("repair", {}).get("covered") is not True
        or receipt.get("repair", {}).get("trial", {}).get("space_exhausted") is not True
    ):
        raise GravityG1AtlasInteractionRepairError("interaction PASS is unsupported")


def _write_immutable(path: Path, value: Mapping[str, Any]) -> None:
    payload = canonical_json_bytes(value)
    if path.exists():
        if path.read_bytes() != payload:
            raise GravityG1AtlasInteractionRepairError(
                f"refusing to overwrite immutable receipt: {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--candidate-count", type=int)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--validate-checked", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.validate_checked:
        validate_receipt(_load_json(root / OUTPUT_PATH), root=root)
        return 0
    receipt = build_receipt(
        root,
        candidate_count_override=args.candidate_count,
        use_gpu=not args.cpu_only,
    )
    if args.candidate_count is None and not args.cpu_only:
        _write_immutable(root / OUTPUT_PATH, receipt)
    print(
        json.dumps(
            {
                "content_sha256": receipt["content_sha256"],
                "covered": receipt["repair"]["covered"],
                "decision": receipt["decision"],
            },
            sort_keys=True,
        )
    )
    return 0 if receipt["decision"] == "PASS_G1_ATLAS_UNION_139_OF_139" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "COMPONENT_COUNT",
    "PAIR_COUNT",
    "GravityG1AtlasInteractionRepairError",
    "build_receipt",
    "interaction_components",
    "load_config",
    "normalized_baryonic_features",
    "pair_batches",
    "replay_candidate",
    "search",
    "validate_receipt",
]
