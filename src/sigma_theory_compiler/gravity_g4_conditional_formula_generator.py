"""Search conditional atlas, baryonic-focusing, and speed-synchronization laws."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from .gravity_g0_experiment import score_predictions
from .gravity_g1_pilot import _binding, _file_sha256, _load_json, _metric
from .gravity_g2_equivalence import validate_receipt as validate_g2_receipt
from .gravity_g3_meta_law import GravityG3MetaLawError, _fold_map, formula_basis
from .gravity_g3_meta_law_v2 import validate_receipt as validate_g3_receipt
from .gravity_g4_nonlocal_profile_law_construction import (
    _kernel_matrix,
    _log_radius_cell_widths,
    prepare_nonlocal_packets,
)
from .gravity_g4_nonlocal_profile_law_construction import (
    validate_receipt as validate_predecessor_receipt,
)
from .gravity_g4_universal_law_construction import _stratum_assignments
from .sigma_core import canonical_json_bytes, canonical_sha256

SCHEMA = "invariant-gravity-g4-conditional-generator-receipt-4.0"
CONFIG_SCHEMA = "invariant-gravity-g4-conditional-generator-config-4.0"
CONFIG_PATH = "configs/gravity_g4_conditional_formula_generator.json"
SOURCE_PATH = "src/sigma_theory_compiler/gravity_g4_conditional_formula_generator.py"
TEST_PATH = "tests/test_gravity_g4_conditional_formula_generator.py"
OUTPUT_PATH = "runs/gravity/g4/conditional-formula-generator-v4.json"

CONDITION_IDS = (
    "baryonic_compactness",
    "surface_density",
    "gas_dominance",
    "bulge_dominance",
    "vacuum_fraction",
    "radial_span",
    "orbital_speed_coherence",
)
SHRINKAGES = (0.25, 0.5, 0.75, 1.0)
FOCUS_SCALES = (0.25, 0.5, 1.0, 2.0)
FOCUS_SB_THRESHOLDS = (10.0, 100.0, 1000.0)
FOCUS_ACCELERATION_THRESHOLDS = (0.01, 0.1, 1.0)
EVOLUTION_TIMES_GYR: tuple[float | None, ...] = (None, 1.0, 3.0, 10.0)
LOGISTIC_INTERCEPTS = (-4.0, -2.0, 0.0, 2.0, 4.0)
LOGISTIC_SLOPES = (-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0)
KM_S_PER_KPC_TO_GYR_INV = 1.0227121650537077


class GravityG4ConditionalGeneratorError(ValueError):
    """The conditional generator contract, computation, or evidence is inconsistent."""


def load_config(root: Path) -> Mapping[str, Any]:
    """Load the frozen generator contract and validate its complete lineage."""

    root = root.resolve()
    config = _load_json(root / CONFIG_PATH)
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise GravityG4ConditionalGeneratorError("conditional generator config changed")
    validators = (
        ("predecessor_binding", validate_predecessor_receipt),
        ("g2_binding", validate_g2_receipt),
        ("g3_binding", validate_g3_receipt),
    )
    for key, validator in validators:
        binding = config.get(key, {})
        path = root / str(binding.get("path"))
        if _file_sha256(path) != binding.get("file_sha256"):
            raise GravityG4ConditionalGeneratorError(f"{key} file changed")
        receipt = _load_json(path)
        validator(receipt, root=root)
        if receipt.get("content_sha256") != binding.get("content_sha256") or receipt.get(
            "decision"
        ) != binding.get("required_decision"):
            raise GravityG4ConditionalGeneratorError(f"{key} content changed")
    g2 = _load_json(root / str(config["g2_binding"]["path"]))
    if (
        len(g2.get("structural_classes", ())) != config["g2_binding"]["required_structural_classes"]
        or len(g2.get("behavioral_classes", ()))
        != config["g2_binding"]["required_behavioral_classes"]
    ):
        raise GravityG4ConditionalGeneratorError("G2 class population changed")
    if tuple(config.get("galaxy_condition_variables", ())) != CONDITION_IDS:
        raise GravityG4ConditionalGeneratorError("condition-variable grammar changed")
    if config.get("atlas_lane", {}).get("classes_exhausted") != 8609:
        raise GravityG4ConditionalGeneratorError("atlas class count changed")
    if config.get("baryonic_focusing_lane", {}).get("operator_count") != 48:
        raise GravityG4ConditionalGeneratorError("focusing operator count changed")
    if config.get("secular_synchronization_lane", {}).get("operator_count") != 32:
        raise GravityG4ConditionalGeneratorError("synchronization operator count changed")
    if config.get("candidate_accounting", {}).get("total_declared_cells") != 261348:
        raise GravityG4ConditionalGeneratorError("candidate accounting changed")
    if config.get("admission", {}).get("confirmation_evaluator_accesses_allowed") != 0:
        raise GravityG4ConditionalGeneratorError("conditional generator permits confirmation")
    if any(
        lane.get("historical_novelty_claimed") is not False
        for lane in (
            config["baryonic_focusing_lane"],
            config["secular_synchronization_lane"],
        )
    ):
        raise GravityG4ConditionalGeneratorError("concept lane overstates novelty")
    return config


def galaxy_conditions(packet: Mapping[str, Any]) -> dict[str, float]:
    """Return seven bounded galaxy scalars from baryons and geometry, never targets."""

    features = packet["features"]
    radius = np.asarray(packet["arrays"]["radius"], dtype=np.float64)
    sb_total = np.expm1(np.asarray(features["log1p_sb_total"], dtype=np.float64))
    rar_speed = np.sqrt(np.asarray(packet["rar2"], dtype=np.float64))
    values = {
        "baryonic_compactness": math.tanh(float(np.median(features["log_y"])) / 4.0),
        "surface_density": math.tanh((float(np.median(features["log1p_sb_total"])) - 4.0) / 4.0),
        "gas_dominance": 2.0 * float(np.mean(features["gas_fraction"])) - 1.0,
        "bulge_dominance": 2.0 * float(np.mean(features["bulge_fraction"])) - 1.0,
        "vacuum_fraction": 2.0 * float(np.mean(100.0 / (100.0 + sb_total))) - 1.0,
        "radial_span": math.tanh((float(np.log(radius[-1] / radius[0])) - 3.0) / 2.0),
        "orbital_speed_coherence": 2.0 * math.exp(-float(np.std(np.log(rar_speed)))) - 1.0,
    }
    if set(values) != set(CONDITION_IDS) or any(
        not np.isfinite(value) or not -1.0000000001 <= value <= 1.0000000001
        for value in values.values()
    ):
        raise GravityG4ConditionalGeneratorError("invalid galaxy condition")
    return values


def _flatten(
    packets: Sequence[Mapping[str, Any]], assignments: Mapping[str, int]
) -> dict[str, Any]:
    slices = {}
    offset = 0
    condition_by_galaxy = {}
    for packet in packets:
        name = packet["galaxy"].name
        count = packet["galaxy"].count
        slices[name] = (offset, offset + count)
        condition_by_galaxy[name] = galaxy_conditions(packet)
        offset += count
    observed = np.concatenate([packet["arrays"]["vobs"] for packet in packets])
    sigma = np.concatenate([packet["arrays"]["sigma"] for packet in packets])
    return {
        "condition_by_galaxy": condition_by_galaxy,
        "conditions": {
            condition: np.concatenate(
                [
                    np.full(
                        packet["galaxy"].count,
                        condition_by_galaxy[packet["galaxy"].name][condition],
                        dtype=np.float64,
                    )
                    for packet in packets
                ]
            )
            for condition in CONDITION_IDS
        },
        "fold": np.concatenate(
            [
                np.full(
                    packet["galaxy"].count,
                    assignments[packet["galaxy"].name],
                    dtype=np.int64,
                )
                for packet in packets
            ]
        ),
        "observed": observed,
        "rar2": np.concatenate([packet["rar2"] for packet in packets]),
        "sigma": sigma,
        "slices": slices,
        "vobs2": observed**2,
        "weight_v2": (1.0 / (2.0 * observed * sigma)) ** 2,
    }


def _atlas_basis(
    formula_ir: Mapping[str, Any], packets: Sequence[Mapping[str, Any]]
) -> tuple[np.ndarray, np.ndarray]:
    bases = []
    components = []
    for packet in packets:
        try:
            base, columns = formula_basis(formula_ir, packet)
        except (GravityG3MetaLawError, FloatingPointError, ValueError) as error:
            raise GravityG4ConditionalGeneratorError(f"{packet['galaxy'].name}: {error}") from error
        if columns.shape != (packet["galaxy"].count, 2):
            raise GravityG4ConditionalGeneratorError(
                f"{packet['galaxy'].name}: atlas formula does not have two components"
            )
        if np.any(~np.isfinite(base)) or np.any(~np.isfinite(columns)):
            raise GravityG4ConditionalGeneratorError(
                f"{packet['galaxy'].name}: atlas basis is non-finite"
            )
        bases.append(base)
        components.append(columns)
    return np.concatenate(bases), np.vstack(components)


def _condition_design(components: np.ndarray, condition: np.ndarray, degree: int) -> np.ndarray:
    powers = np.column_stack([condition**power for power in range(degree + 1)])
    return np.column_stack(
        [components[:, component, None] * powers for component in range(2)]
    ).reshape(len(condition), 2 * (degree + 1))


def _fit_oof_atlas(
    flat: Mapping[str, Any],
    base: np.ndarray,
    components: np.ndarray,
    condition_id: str,
    degree: int,
    folds: int,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    design = _condition_design(components, flat["conditions"][condition_id], degree)
    target = np.asarray(flat["vobs2"]) - base
    weight = np.asarray(flat["weight_v2"])
    fold_ids = np.asarray(flat["fold"])
    correction = np.empty(len(target), dtype=np.float64)
    coefficients = {}
    for fold in range(folds):
        training = fold_ids != fold
        testing = ~training
        sqrt_weight = np.sqrt(weight[training])
        coefficients[fold] = np.linalg.lstsq(
            design[training] * sqrt_weight[:, None],
            target[training] * sqrt_weight,
            rcond=None,
        )[0]
        correction[testing] = design[testing] @ coefficients[fold]
    if np.any(~np.isfinite(correction)):
        raise GravityG4ConditionalGeneratorError("atlas OOF correction is non-finite")
    return correction, coefficients


def _score_prediction2(prediction2: np.ndarray, flat: Mapping[str, Any]) -> tuple[float, int]:
    invalid = int(np.sum(~np.isfinite(prediction2) | (prediction2 <= 0)))
    velocity = np.sqrt(np.maximum(prediction2, np.finfo(np.float64).tiny))
    chi_square = float(
        np.sum(((velocity - np.asarray(flat["observed"])) / np.asarray(flat["sigma"])) ** 2)
    )
    return chi_square, invalid


def _candidate_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        float(row["chi_square"]),
        int(row["universal_constants"]),
        str(row["candidate_id"]),
    )


def _atlas_rows(
    flat: Mapping[str, Any],
    base: np.ndarray,
    components: np.ndarray,
    class_row: Mapping[str, Any],
    condition_id: str,
    degree: int,
    folds: int,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    correction, _coefficients = _fit_oof_atlas(flat, base, components, condition_id, degree, folds)
    rows = []
    map_id = {1: "affine", 2: "quadratic", 3: "cubic"}[degree]
    for shrinkage in SHRINKAGES:
        chi_square, invalid = _score_prediction2(base + shrinkage * correction, flat)
        rows.append(
            {
                "authoritative_origin_status": class_row["authoritative_origin_status"],
                "candidate_id": (
                    f"atlas:{class_row['class_id']}:{condition_id}:{map_id}:"
                    f"s{format(shrinkage, 'g')}"
                ),
                "chi_square": _metric(chi_square),
                "class_id": class_row["class_id"],
                "condition_id": condition_id,
                "degree": degree,
                "invalid_prediction2": invalid,
                "lane": "conditional_atlas",
                "map": map_id,
                "shrinkage": _metric(shrinkage),
                "universal_constants": 2 * (degree + 1),
            }
        )
    return rows, correction


def _scale_id(value: float) -> str:
    return format(value, "g").replace(".", "p")


def concept_operators(
    packets: Sequence[Mapping[str, Any]], *, operator_limit: int | None = None
) -> list[dict[str, Any]]:
    """Materialize 48 focusing and 32 synchronization operators without V_obs."""

    definitions: list[dict[str, Any]] = []
    for source, thresholds in (
        ("surface_brightness", FOCUS_SB_THRESHOLDS),
        ("baryonic_acceleration", FOCUS_ACCELERATION_THRESHOLDS),
    ):
        for threshold in thresholds:
            for scale in FOCUS_SCALES:
                for mode in (
                    "interior_minus_exterior_occupancy",
                    "interior_occupancy_times_exterior_vacuum",
                ):
                    definitions.append(
                        {
                            "alpha_max": 2.0,
                            "authoritative_origin_status": "COMBINATION",
                            "family": "baryonic_focusing",
                            "log_radius_scale": scale,
                            "mode": mode,
                            "operator_id": (
                                f"focus:{source}:q{_scale_id(threshold)}:"
                                f"ell{_scale_id(scale)}:{mode}"
                            ),
                            "proposer_origin_label": "new_combination_of_known_ideas",
                            "source": source,
                            "threshold": threshold,
                        }
                    )
    for mode in ("interior_speed_equalization", "symmetric_speed_equalization"):
        for scale in FOCUS_SCALES:
            for time_gyr in EVOLUTION_TIMES_GYR:
                time_id = "none" if time_gyr is None else _scale_id(time_gyr)
                definitions.append(
                    {
                        "alpha_max": 1.0,
                        "authoritative_origin_status": "UNRESOLVED",
                        "evolution_time_gyr": time_gyr,
                        "family": "secular_speed_synchronization",
                        "log_radius_scale": scale,
                        "mode": mode,
                        "operator_id": (f"sync:{mode}:ell{_scale_id(scale)}:tgyr{time_id}"),
                        "proposer_origin_label": "proposed_new_construction",
                    }
                )
    if len(definitions) != 80:
        raise GravityG4ConditionalGeneratorError("concept operator count changed")
    if operator_limit is not None:
        definitions = definitions[: max(0, min(operator_limit, len(definitions)))]
    parts: dict[str, list[np.ndarray]] = {row["operator_id"]: [] for row in definitions}
    for packet in packets:
        radius = np.asarray(packet["arrays"]["radius"], dtype=np.float64)
        log_radius = np.log(radius)
        widths = _log_radius_cell_widths(log_radius)
        matrices = {}
        for scale in FOCUS_SCALES:
            matrices[("interior", scale)] = _kernel_matrix(
                log_radius, widths, "interior_exponential", scale
            )
            matrices[("exterior", scale)] = _kernel_matrix(
                log_radius, widths, "exterior_exponential", scale
            )
            matrices[("symmetric", scale)] = _kernel_matrix(
                log_radius, widths, "symmetric_exponential", scale
            )
        sb_total = np.expm1(np.asarray(packet["features"]["log1p_sb_total"], dtype=np.float64))
        acceleration_ratio = np.exp(np.asarray(packet["features"]["log_y"], dtype=np.float64))
        for definition in definitions:
            scale = float(definition["log_radius_scale"])
            if definition["family"] == "baryonic_focusing":
                source = (
                    sb_total if definition["source"] == "surface_brightness" else acceleration_ratio
                )
                threshold = float(definition["threshold"])
                occupancy = source / (source + threshold)
                interior = matrices[("interior", scale)] @ occupancy
                exterior = matrices[("exterior", scale)] @ occupancy
                if definition["mode"] == "interior_minus_exterior_occupancy":
                    feature = interior - exterior
                else:
                    feature = interior * (1.0 - exterior)
                component = radius * float(packet["a0"]) * feature
            else:
                rar2 = np.asarray(packet["rar2"], dtype=np.float64)
                matrix_id = (
                    "interior"
                    if definition["mode"] == "interior_speed_equalization"
                    else "symmetric"
                )
                component = matrices[(matrix_id, scale)] @ rar2 - rar2
                time_gyr = definition["evolution_time_gyr"]
                if time_gyr is not None:
                    cycles = (
                        float(time_gyr)
                        * np.sqrt(rar2)
                        / radius
                        * KM_S_PER_KPC_TO_GYR_INV
                        / (2.0 * np.pi)
                    )
                    component = component * (1.0 - np.exp(-cycles))
            if np.any(~np.isfinite(component)):
                raise GravityG4ConditionalGeneratorError(
                    f"non-finite concept operator {definition['operator_id']}"
                )
            parts[definition["operator_id"]].append(component)
    return [
        {
            **definition,
            "component_v2": np.concatenate(parts[definition["operator_id"]]),
        }
        for definition in definitions
    ]


def _logistic(value: np.ndarray) -> np.ndarray:
    return np.where(
        value >= 0,
        1.0 / (1.0 + np.exp(-value)),
        np.exp(value) / (1.0 + np.exp(value)),
    )


def _concept_correction(
    row: Mapping[str, Any],
    flat: Mapping[str, Any],
    operators: Mapping[str, Mapping[str, Any]],
) -> np.ndarray:
    correction = np.zeros(len(flat["observed"]), dtype=np.float64)
    for item in row["concept_terms"]:
        operator = operators[item["operator_id"]]
        x = np.asarray(flat["conditions"][item["condition_id"]])
        alpha = float(operator["alpha_max"]) * _logistic(
            float(item["intercept"]) + float(item["slope"]) * x
        )
        correction += alpha * np.asarray(operator["component_v2"])
    return correction


def _concept_rows(flat: Mapping[str, Any], operator: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    component = np.asarray(operator["component_v2"])
    for condition_id in CONDITION_IDS:
        x = np.asarray(flat["conditions"][condition_id])
        for intercept in LOGISTIC_INTERCEPTS:
            for slope in LOGISTIC_SLOPES:
                alpha = float(operator["alpha_max"]) * _logistic(intercept + slope * x)
                chi_square, invalid = _score_prediction2(
                    np.asarray(flat["rar2"]) + alpha * component, flat
                )
                term = {
                    "condition_id": condition_id,
                    "intercept": _metric(intercept),
                    "operator_id": operator["operator_id"],
                    "slope": _metric(slope),
                }
                rows.append(
                    {
                        "authoritative_origin_status": operator["authoritative_origin_status"],
                        "candidate_id": (
                            f"concept:{operator['operator_id']}:{condition_id}:"
                            f"a{_scale_id(intercept)}:b{_scale_id(slope)}"
                        ),
                        "chi_square": _metric(chi_square),
                        "concept_terms": [term],
                        "invalid_prediction2": invalid,
                        "lane": operator["family"],
                        "proposer_origin_label": operator["proposer_origin_label"],
                        "universal_constants": 2,
                    }
                )
    return rows


def _prediction_for_atlas_row(
    row: Mapping[str, Any],
    flat: Mapping[str, Any],
    packets: Sequence[Mapping[str, Any]],
    classes: Mapping[str, Mapping[str, Any]],
    folds: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    class_row = classes[str(row["class_id"])]
    base, components = _atlas_basis(class_row["canonical_ir"], packets)
    correction, coefficients = _fit_oof_atlas(
        flat,
        base,
        components,
        str(row["condition_id"]),
        int(row["degree"]),
        folds,
    )
    return (
        base + float(row["shrinkage"]) * correction,
        base,
        components,
        coefficients,
    )


def build_receipt(
    root: Path,
    *,
    class_limit: int | None = None,
    operator_limit: int | None = None,
) -> dict[str, Any]:
    """Exhaust the conditional generator grammar on exploration galaxies only."""

    root = root.resolve()
    config = load_config(root)
    packets = sorted(prepare_nonlocal_packets(root), key=lambda packet: packet["galaxy"].name)
    folds = int(config["whole_galaxy_cross_validation"]["folds"])
    assignments = _fold_map(
        [packet["galaxy"].name for packet in packets],
        str(config["whole_galaxy_cross_validation"]["salt"]),
        folds,
    )
    flat = _flatten(packets, assignments)
    g2 = _load_json(root / str(config["g2_binding"]["path"]))
    all_classes = sorted(g2["structural_classes"], key=lambda row: row["class_id"])
    classes_to_run = (
        all_classes
        if class_limit is None
        else all_classes[: max(0, min(class_limit, len(all_classes)))]
    )
    class_by_id = {row["class_id"]: row for row in all_classes}
    stage_a_candidates = []
    atlas_ledger = []
    atlas_stage_a_cells = 0
    domain_rejected = 0
    for class_row in classes_to_run:
        try:
            base, components = _atlas_basis(class_row["canonical_ir"], packets)
        except GravityG4ConditionalGeneratorError as error:
            domain_rejected += 1
            atlas_stage_a_cells += len(CONDITION_IDS) * len(SHRINKAGES)
            atlas_ledger.append(
                {
                    "class_id": class_row["class_id"],
                    "decision": "REJECT_TARGET_BLIND_DOMAIN",
                    "first_counterexample": str(error),
                }
            )
            continue
        class_candidates = []
        for condition_id in CONDITION_IDS:
            rows, _correction = _atlas_rows(
                flat,
                base,
                components,
                class_row,
                condition_id,
                1,
                folds,
            )
            atlas_stage_a_cells += len(rows)
            class_candidates.extend(rows)
            stage_a_candidates.extend(rows)
        best = min(class_candidates, key=_candidate_key)
        atlas_ledger.append(
            {
                "best_affine_generator": best,
                "class_id": class_row["class_id"],
                "decision": "AFFINE_CONDITION_EVALUATED",
            }
        )
    eligible_stage_a = [row for row in stage_a_candidates if row["invalid_prediction2"] == 0]
    retained_stage_b = []
    retained_stage_b_keys: set[tuple[str, str]] = set()
    for row in sorted(eligible_stage_a, key=_candidate_key):
        key = (str(row["class_id"]), str(row["condition_id"]))
        if key in retained_stage_b_keys:
            continue
        retained_stage_b.append(row)
        retained_stage_b_keys.add(key)
        if len(retained_stage_b) == 64:
            break
    stage_b_candidates = []
    basis_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for retained in retained_stage_b:
        class_id = str(retained["class_id"])
        if class_id not in basis_cache:
            basis_cache[class_id] = _atlas_basis(class_by_id[class_id]["canonical_ir"], packets)
        base, components = basis_cache[class_id]
        for degree in (2, 3):
            rows, _correction = _atlas_rows(
                flat,
                base,
                components,
                class_by_id[class_id],
                str(retained["condition_id"]),
                degree,
                folds,
            )
            stage_b_candidates.extend(rows)
    atlas_candidates = stage_a_candidates + stage_b_candidates

    operators = concept_operators(packets, operator_limit=operator_limit)
    operator_by_id = {row["operator_id"]: row for row in operators}
    concept_single_candidates = []
    concept_ledger = []
    for operator in operators:
        rows = _concept_rows(flat, operator)
        concept_single_candidates.extend(rows)
        concept_ledger.append(
            {
                "best_generator": min(rows, key=_candidate_key),
                "family": operator["family"],
                "operator_id": operator["operator_id"],
            }
        )
    top_concepts = sorted(
        (
            row["best_generator"]
            for row in concept_ledger
            if row["best_generator"]["invalid_prediction2"] == 0
        ),
        key=_candidate_key,
    )[:16]
    concept_pair_candidates = []
    for left, right in combinations(top_concepts, 2):
        terms = left["concept_terms"] + right["concept_terms"]
        row = {
            "candidate_id": f"concept-pair:{left['candidate_id']}+{right['candidate_id']}",
            "concept_terms": terms,
        }
        correction = _concept_correction(row, flat, operator_by_id)
        chi_square, invalid = _score_prediction2(np.asarray(flat["rar2"]) + correction, flat)
        concept_pair_candidates.append(
            {
                **row,
                "authoritative_origin_status": "COMBINATION",
                "chi_square": _metric(chi_square),
                "invalid_prediction2": invalid,
                "lane": "concept_pair",
                "proposer_origin_label": "new_combination_of_known_ideas",
                "universal_constants": 4,
            }
        )
    top_atlas_for_hybrid = sorted(
        [row for row in atlas_candidates if row["invalid_prediction2"] == 0],
        key=_candidate_key,
    )[:8]
    top_concepts_for_hybrid = top_concepts[:8]
    hybrid_candidates = []
    for atlas_row in top_atlas_for_hybrid:
        atlas_prediction2, base, _components, _coefficients = _prediction_for_atlas_row(
            atlas_row, flat, packets, class_by_id, folds
        )
        atlas_correction = atlas_prediction2 - base
        for concept_row in top_concepts_for_hybrid:
            concept_correction = _concept_correction(concept_row, flat, operator_by_id)
            chi_square, invalid = _score_prediction2(
                base + atlas_correction + concept_correction, flat
            )
            hybrid_candidates.append(
                {
                    "atlas_term": atlas_row,
                    "authoritative_origin_status": "COMBINATION",
                    "candidate_id": (
                        f"hybrid:{atlas_row['candidate_id']}+{concept_row['candidate_id']}"
                    ),
                    "chi_square": _metric(chi_square),
                    "concept_terms": concept_row["concept_terms"],
                    "invalid_prediction2": invalid,
                    "lane": "atlas_concept_hybrid",
                    "proposer_origin_label": "new_combination_of_known_ideas",
                    "universal_constants": int(atlas_row["universal_constants"]) + 2,
                }
            )
    all_candidates = (
        atlas_candidates + concept_single_candidates + concept_pair_candidates + hybrid_candidates
    )
    eligible = [row for row in all_candidates if row["invalid_prediction2"] == 0]
    selected = min(eligible, key=_candidate_key, default=None)

    selected_prediction2 = np.asarray(flat["rar2"]).copy()
    selected_final_coefficients = None
    selected_base = np.asarray(flat["rar2"])
    selected_components = None
    selected_atlas_row = None
    selected_fold_coefficients: dict[int, np.ndarray] = {}
    if selected is not None:
        if selected["lane"] == "conditional_atlas":
            selected_atlas_row = selected
        elif selected["lane"] == "atlas_concept_hybrid":
            selected_atlas_row = selected["atlas_term"]
        if selected_atlas_row is not None:
            (
                atlas_prediction2,
                selected_base,
                selected_components,
                selected_fold_coefficients,
            ) = _prediction_for_atlas_row(selected_atlas_row, flat, packets, class_by_id, folds)
            selected_prediction2 = atlas_prediction2
        if "concept_terms" in selected:
            selected_prediction2 = selected_prediction2 + _concept_correction(
                selected, flat, operator_by_id
            )
    per_galaxy = []
    generated_coefficient_vectors = []
    final_direct_chi_square = None
    if selected is not None:
        if selected_atlas_row is not None:
            condition_id = str(selected_atlas_row["condition_id"])
            degree = int(selected_atlas_row["degree"])
            design = _condition_design(
                selected_components,
                np.asarray(flat["conditions"][condition_id]),
                degree,
            )
            target = np.asarray(flat["vobs2"]) - selected_base
            sqrt_weight = np.sqrt(np.asarray(flat["weight_v2"]))
            selected_final_coefficients = np.linalg.lstsq(
                design * sqrt_weight[:, None],
                target * sqrt_weight,
                rcond=None,
            )[0] * float(selected_atlas_row["shrinkage"])
            direct2 = selected_base + design @ selected_final_coefficients
            if "concept_terms" in selected:
                direct2 = direct2 + _concept_correction(selected, flat, operator_by_id)
            final_direct_chi_square, _direct_invalid = _score_prediction2(direct2, flat)
        for packet in packets:
            name = packet["galaxy"].name
            start, stop = flat["slices"][name]
            prediction2 = selected_prediction2[start:stop]
            prediction = np.sqrt(np.maximum(prediction2, np.finfo(np.float64).tiny))
            generated: dict[str, Any] = {
                "condition_values": {
                    key: _metric(value) for key, value in flat["condition_by_galaxy"][name].items()
                },
                "lane": selected["lane"],
            }
            coefficient_vector = []
            if selected_atlas_row is not None:
                condition_id = str(selected_atlas_row["condition_id"])
                x = flat["condition_by_galaxy"][name][condition_id]
                degree = int(selected_atlas_row["degree"])
                fold_coefficients = selected_fold_coefficients[assignments[name]] * float(
                    selected_atlas_row["shrinkage"]
                )
                powers = np.asarray([x**power for power in range(degree + 1)])
                coefficient_vector = [
                    float(
                        fold_coefficients[component * (degree + 1) : (component + 1) * (degree + 1)]
                        @ powers
                    )
                    for component in range(2)
                ]
                generated.update(
                    {
                        "atlas_class_id": selected_atlas_row["class_id"],
                        "coefficient_condition": condition_id,
                        "generated_component_coefficients": [
                            _metric(value) for value in coefficient_vector
                        ],
                        "map": selected_atlas_row["map"],
                    }
                )
            if "concept_terms" in selected:
                generated["concept_terms"] = []
                for term in selected["concept_terms"]:
                    operator = operator_by_id[term["operator_id"]]
                    x = flat["condition_by_galaxy"][name][term["condition_id"]]
                    alpha = float(operator["alpha_max"]) * float(
                        _logistic(np.asarray(float(term["intercept"]) + float(term["slope"]) * x))
                    )
                    coefficient_vector.append(alpha)
                    generated["concept_terms"].append(
                        {
                            **term,
                            "generated_alpha": _metric(alpha),
                        }
                    )
            generated_coefficient_vectors.append(
                tuple(round(value, 12) for value in coefficient_vector)
            )
            per_galaxy.append(
                {
                    "candidate_prediction_sha256": canonical_sha256(
                        [format(float(value), ".15e") for value in prediction]
                    ),
                    "candidate_score": score_predictions(
                        prediction,
                        packet["arrays"]["vobs"],
                        packet["arrays"]["sigma"],
                    ),
                    "fold": assignments[name],
                    "galaxy": name,
                    "generated_formula": generated,
                    "invalid_prediction2": int(
                        np.sum(~np.isfinite(prediction2) | (prediction2 <= 0))
                    ),
                    "point_count": packet["galaxy"].count,
                    "rar_score": score_predictions(
                        np.sqrt(packet["rar2"]),
                        packet["arrays"]["vobs"],
                        packet["arrays"]["sigma"],
                    ),
                }
            )
    candidate_chi = sum(float(row["candidate_score"]["chi_square"]) for row in per_galaxy)
    rar_chi = sum(float(row["rar_score"]["chi_square"]) for row in per_galaxy)
    predecessor = _load_json(root / str(config["predecessor_binding"]["path"]))
    nfw_chi = float(predecessor["scores"]["nfw_ceiling_chi_square"])
    newtonian_chi = float(predecessor["scores"]["newtonian_baryons_chi_square"])
    point_count = sum(row["point_count"] for row in per_galaxy)
    by_name = {row["galaxy"]: row for row in per_galaxy}
    strata = []
    stratum_map = _stratum_assignments(
        packets, int(config["predecessor_binding"].get("bins_per_dimension", 4))
    )
    for dimension, bins_by_name in stratum_map.items():
        for bin_id in range(4):
            names = sorted(name for name, value in bins_by_name.items() if value == bin_id)
            stratum_candidate = sum(
                float(by_name[name]["candidate_score"]["chi_square"]) for name in names
            )
            stratum_rar = sum(float(by_name[name]["rar_score"]["chi_square"]) for name in names)
            strata.append(
                {
                    "bin": bin_id,
                    "candidate_chi_square": _metric(stratum_candidate),
                    "dimension": dimension,
                    "fractional_gain_over_rar": _metric(1.0 - stratum_candidate / stratum_rar),
                    "galaxies": len(names),
                    "rar_chi_square": _metric(stratum_rar),
                }
            )
    full_run = (
        class_limit is None
        and operator_limit is None
        and len(classes_to_run) == 8609
        and len(operators) == 80
    )
    admission = config["admission"]
    nfw_limit = nfw_chi + float(admission["nfw_ceiling_slack_chi_square_per_point"]) * point_count
    gate_checks = {
        "all_139_exploration_galaxies_predicted_once": len(per_galaxy) == 139,
        "all_predictions_positive_and_finite": all(
            row["invalid_prediction2"] == 0 for row in per_galaxy
        ),
        "beats_newtonian_baryons": candidate_chi < newtonian_chi,
        "beats_rar_by_minimum_fraction": (
            rar_chi > 0
            and 1.0 - candidate_chi / rar_chi
            >= float(admission["minimum_fractional_chi_square_gain_over_empirical_rar"])
        ),
        "complete_conditional_generator_grammar_searched": full_run,
        "no_stratum_regresses_beyond_limit": all(
            float(row["fractional_gain_over_rar"])
            >= -float(admission["maximum_fractional_chi_square_regression_vs_rar_in_any_stratum"])
            for row in strata
        ),
        "per_galaxy_fitted_gravitational_constants_zero": True,
        "within_nfw_performance_ceiling": candidate_chi <= nfw_limit,
    }
    passed = all(gate_checks.values())
    actual_cells = (
        atlas_stage_a_cells
        + len(stage_b_candidates)
        + len(concept_single_candidates)
        + len(concept_pair_candidates)
        + len(hybrid_candidates)
    )
    scored_cells = len(all_candidates)
    if full_run and actual_cells != int(config["candidate_accounting"]["total_declared_cells"]):
        raise GravityG4ConditionalGeneratorError(
            "full candidate accounting does not match the frozen grammar"
        )
    if full_run and len(retained_stage_b) != int(
        config["candidate_accounting"]["atlas_stage_b_parent_pairs"]
    ):
        raise GravityG4ConditionalGeneratorError(
            "full higher-order atlas parent count does not match the frozen grammar"
        )
    top_candidates = sorted(eligible, key=_candidate_key)[:256]
    selected_public = None if selected is None else dict(selected)
    if selected_public is not None and "atlas_term" in selected_public:
        selected_public["atlas_term"] = dict(selected_public["atlas_term"])
    body: dict[str, Any] = {
        "schema_version": SCHEMA,
        "goal": "G4_CONDITIONAL_FORMULA_GENERATOR",
        "decision": (
            "PASS_G4_CONDITIONAL_GENERATOR_EXPLORATION_FREEZE"
            if passed
            else "BLOCK_G4_CONDITIONAL_GENERATOR"
        ),
        "claims": {
            "alternative_to_gr_discovered": False,
            "confirmation_authorized": passed,
            "confirmation_galaxy_evaluated": False,
            "historical_novelty_established": False,
            "individual_star_velocities_predicted": False,
            "resonance_dynamics_derived": False,
            "universal_conditional_empirical_law_constructed": selected is not None,
            "zero_per_galaxy_fitted_gravitational_constants": True,
        },
        "config": {"content_sha256": canonical_sha256(config), "path": CONFIG_PATH},
        "counts": {
            "atlas_classes_audited": len(classes_to_run),
            "atlas_classes_rejected_by_target_blind_domain": domain_rejected,
            "atlas_stage_a_cells": atlas_stage_a_cells,
            "atlas_stage_b_parent_pairs": len(retained_stage_b),
            "atlas_stage_b_cells": len(stage_b_candidates),
            "candidate_formula_cells_represented": actual_cells,
            "candidate_formula_cells_scored": scored_cells,
            "candidate_galaxy_evaluations_scored": scored_cells * len(packets),
            "candidate_point_evaluations_scored": scored_cells * point_count,
            "concept_hybrid_cells": len(hybrid_candidates),
            "concept_operators": len(operators),
            "concept_pair_cells": len(concept_pair_candidates),
            "concept_single_cells": len(concept_single_candidates),
            "confirmation_evaluator_accesses": 0,
            "exploration_galaxies": len(per_galaxy),
            "exploration_points": point_count,
            "generated_formula_instances": len(per_galaxy),
            "unique_generated_coefficient_vectors": len(set(generated_coefficient_vectors)),
        },
        "diagnostic_disclosure": config["diagnostic_disclosure"],
        "family_result": {
            "atlas_class_ledger": atlas_ledger,
            "concept_operator_ledger": concept_ledger,
            "exhausted_without_full_gate_survivor": not passed and full_run,
            "top_candidates": top_candidates,
        },
        "gate_checks": gate_checks,
        "galaxies": per_galaxy,
        "scores": {
            "candidate_chi_square": _metric(candidate_chi),
            "direct_all_exploration_refit_chi_square": (
                None if final_direct_chi_square is None else _metric(final_direct_chi_square)
            ),
            "empirical_rar_chi_square": _metric(rar_chi),
            "fractional_gain_over_empirical_rar": _metric(1.0 - candidate_chi / rar_chi),
            "newtonian_baryons_chi_square": _metric(newtonian_chi),
            "nfw_ceiling_chi_square": _metric(nfw_chi),
            "nfw_ceiling_limit_with_slack": _metric(nfw_limit),
            "nfw_ceiling_excess": _metric(candidate_chi - nfw_limit),
        },
        "selected_generator": selected_public,
        "selected_generator_final_universal_coefficients": (
            None
            if selected_final_coefficients is None
            else [_metric(float(value)) for value in selected_final_coefficients]
        ),
        "strata": strata,
        "limitations": [
            "This is model development on the repeatedly inspected exploration population, not independent confirmation.",
            "The atlas lane generates coefficients from one scalar, but a curve fit does not make the relationship first-principles.",
            "Baryonic focusing is a phenomenological nonlocal operator, not a derived gravitational field equation.",
            "Speed synchronization is a static relaxation ansatz; it does not prove resonance, energy exchange, or billions-of-years dynamics.",
            "UNRESOLVED and proposer labels preserve ideas for review; neither establishes historical novelty.",
            "A blocked NFW ceiling keeps all confirmation targets and downstream gates locked.",
        ],
        "source_bindings": {
            "config": _binding(root, CONFIG_PATH),
            "g2": _binding(root, str(config["g2_binding"]["path"])),
            "source": _binding(root, SOURCE_PATH),
            "test": _binding(root, TEST_PATH),
        },
    }
    body["content_sha256"] = canonical_sha256(body)
    return body


def validate_receipt(receipt: Mapping[str, Any], *, root: Path) -> None:
    root = root.resolve()
    if receipt.get("schema_version") != SCHEMA:
        raise GravityG4ConditionalGeneratorError("conditional receipt schema changed")
    body = dict(receipt)
    supplied = body.pop("content_sha256", None)
    if supplied != canonical_sha256(body):
        raise GravityG4ConditionalGeneratorError("conditional receipt seal changed")
    config = load_config(root)
    if receipt.get("config", {}).get("content_sha256") != canonical_sha256(config):
        raise GravityG4ConditionalGeneratorError("conditional config binding changed")
    expected = {
        "config": CONFIG_PATH,
        "g2": str(config["g2_binding"]["path"]),
        "source": SOURCE_PATH,
        "test": TEST_PATH,
    }
    for key, path in expected.items():
        if receipt.get("source_bindings", {}).get(key) != _binding(root, path):
            raise GravityG4ConditionalGeneratorError(f"conditional {key} binding changed")
    counts = receipt.get("counts", {})
    claims = receipt.get("claims", {})
    if counts.get("confirmation_evaluator_accesses") != 0:
        raise GravityG4ConditionalGeneratorError("conditional run accessed confirmation")
    if claims.get("historical_novelty_established") is not False:
        raise GravityG4ConditionalGeneratorError("conditional run overstates novelty")
    if claims.get("resonance_dynamics_derived") is not False:
        raise GravityG4ConditionalGeneratorError("conditional run overstates resonance")
    passed = receipt.get("decision") == ("PASS_G4_CONDITIONAL_GENERATOR_EXPLORATION_FREEZE")
    if passed and (
        not all(receipt.get("gate_checks", {}).values())
        or claims.get("confirmation_authorized") is not True
    ):
        raise GravityG4ConditionalGeneratorError("conditional PASS is unsupported")
    if not passed and claims.get("confirmation_authorized") is not False:
        raise GravityG4ConditionalGeneratorError("blocked conditional run authorizes confirmation")


def _write_immutable(path: Path, value: Mapping[str, Any]) -> None:
    payload = canonical_json_bytes(value)
    if path.exists():
        if path.read_bytes() != payload:
            raise GravityG4ConditionalGeneratorError(
                f"refusing to overwrite immutable conditional receipt: {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--class-limit", type=int)
    parser.add_argument("--operator-limit", type=int)
    parser.add_argument("--validate-checked", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.validate_checked:
        validate_receipt(_load_json(root / OUTPUT_PATH), root=root)
        return 0
    receipt = build_receipt(
        root,
        class_limit=args.class_limit,
        operator_limit=args.operator_limit,
    )
    if args.class_limit is None and args.operator_limit is None:
        _write_immutable(root / OUTPUT_PATH, receipt)
    print(
        json.dumps(
            {
                "content_sha256": receipt["content_sha256"],
                "counts": receipt["counts"],
                "decision": receipt["decision"],
                "scores": receipt["scores"],
                "selected_generator": receipt["selected_generator"],
            },
            sort_keys=True,
        )
    )
    return 0 if receipt["decision"].startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CONDITION_IDS",
    "OUTPUT_PATH",
    "GravityG4ConditionalGeneratorError",
    "build_receipt",
    "concept_operators",
    "galaxy_conditions",
    "load_config",
    "validate_receipt",
]
