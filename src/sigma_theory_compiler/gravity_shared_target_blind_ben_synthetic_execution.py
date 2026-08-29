"""Append-only synthetic execution layer for the target-blind B+E+N grammar.

The candidate registry is generated and frozen before any synthetic adapter is
called.  This module never opens or scores a real galaxy, group, cluster, or
lensing target.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from sigma_theory_compiler import gravity_shared_target_blind_evaluator as shared

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path("configs/gravity_shared_target_blind_ben_synthetic_execution_v1.json")
RECEIPT_PATH = Path("runs/gravity/shared-target-blind-ben-synthetic-execution-v1.json")
TEST_PATH = Path("tests/test_gravity_shared_target_blind_ben_synthetic_execution.py")

CONFIG_SCHEMA = "invariant-gravity-shared-target-blind-ben-synthetic-config-1.0"
RECEIPT_SCHEMA = "invariant-gravity-shared-target-blind-ben-synthetic-receipt-1.0"
STATUS = "bounded_synthetic_ben_execution_passed_not_empirical_or_physical_evidence"
DECISION = "PASS_SYNTHETIC_BEN_GRAMMAR_CONTROLS_REAL_SCIENTIFIC_EVALUATION_LOCKED"

DOMAINS = shared.DOMAINS
VARIABLES = ("x_source", "x_radial", "x_state", "x_geometry")
ALLOWED_OPERATORS = {
    "add",
    "subtract",
    "multiply",
    "divide_safe",
    "sqrt_positive",
    "exp_negative",
}
PROVENANCE_LABELS = (
    "known_rewrite",
    "known_combination",
    "potentially_new_synthesis",
)

EXPECTED_TEST_SHA256 = "563da043471e41ef512a0db4fefc6067b054964cb3bb86b187f7503230d7072b"

EXPECTED_SOURCE_BINDINGS = {
    "shared_source": {
        "path": "src/sigma_theory_compiler/gravity_shared_target_blind_evaluator.py",
        "file_sha256": "7110b5c8012b8bd1493944715fc482db86cafbe43374fcec5837d1288c6ef439",
    },
    "shared_config": {
        "path": "configs/gravity_shared_target_blind_evaluator_v1.json",
        "file_sha256": "896f1796033898c32145a16bad22e041ebced2b318c44d1e3fcdeb4911f91d76",
    },
    "shared_test": {
        "path": "tests/test_gravity_shared_target_blind_evaluator.py",
        "file_sha256": "7c6a41a88fe2eeb7fd48bdb7b103098b2e44fd24e5fadcd4f1fba9e56308ce48",
    },
    "shared_documentation": {
        "path": "docs/GRAVITY_SHARED_TARGET_BLIND_EVALUATOR_V1.md",
        "file_sha256": "2d462a0d78a947c7be2c5c2f71fe7206a7509c84432552b031fb5e2e5d893515",
    },
    "shared_receipt": {
        "path": "runs/gravity/shared-target-blind-evaluator-v1.json",
        "file_sha256": "5d0a87c37d42ddac65a926f1ec3108f9e64511be20b47701f2b35f02838a7e7a",
        "content_sha256": "71db064da23d44191f20c0327cc06c488963f37b748ae2fa5de4b3f7de3779df",
    },
    "parent_registry_receipt": {
        "path": "runs/gravity/lead-programs/gravity-lead-parent-registry-v1.json",
        "file_sha256": "0ff0ae7b8bb8fc18cea2a04b9a7b8b968195437a1b19f784960370e441959d85",
        "content_sha256": "89cc89dc5caea84b350ddca13a9e1f2fffadb540a0472379ac8f6cdfcf5d2fe3",
    },
    "recombination_receipt": {
        "path": "runs/gravity/lead-programs/gravity-lead-recombination-v1.json",
        "file_sha256": "74dd4de439cdfb23027b41530cc6a8658e9e635ead2c53016b797c39aada3d0f",
        "content_sha256": "babfd6794faf53256d661c81b405ee194d6e9fddb7746cc674bf72920d6960f0",
    },
}


def const(value: float) -> dict[str, Any]:
    return {"const": value}


def var(name: str) -> dict[str, Any]:
    return {"var": name}


def op(name: str, *args: Mapping[str, Any]) -> dict[str, Any]:
    return {"op": name, "args": list(args)}


SOURCE = var("x_source")
RADIAL = var("x_radial")
STATE = var("x_state")
GEOMETRY = var("x_geometry")

COMPONENT_GRAMMAR = {
    "E_local_base": [
        {
            "raw_id": "E.newtonian",
            "canonical_component_id": "E.newtonian",
            "ast": SOURCE,
        },
        {
            "raw_id": "E.newtonian_identity_alias",
            "canonical_component_id": "E.newtonian",
            "ast": op("multiply", const(1.0), SOURCE),
        },
        {
            "raw_id": "E.rarlike_interpolation",
            "canonical_component_id": "E.rarlike_interpolation",
            "ast": op(
                "divide_safe",
                SOURCE,
                op(
                    "subtract",
                    const(1.0),
                    op("exp_negative", op("sqrt_positive", SOURCE)),
                ),
            ),
        },
        {
            "raw_id": "E.quadrature_interpolation",
            "canonical_component_id": "E.quadrature_interpolation",
            "ast": op(
                "sqrt_positive",
                op("add", op("multiply", SOURCE, SOURCE), SOURCE),
            ),
        },
    ],
    "B_continuous_gate": [
        {
            "raw_id": "B.low_acceleration",
            "canonical_component_id": "B.low_acceleration",
            "ast": op("exp_negative", SOURCE),
        },
        {
            "raw_id": "B.low_acceleration_identity_alias",
            "canonical_component_id": "B.low_acceleration",
            "ast": op("multiply", const(1.0), op("exp_negative", SOURCE)),
        },
        {
            "raw_id": "B.state_weighted",
            "canonical_component_id": "B.state_weighted",
            "ast": op(
                "multiply",
                op("exp_negative", SOURCE),
                op("divide_safe", STATE, op("add", const(1.0), STATE)),
            ),
        },
        {
            "raw_id": "B.geometry_weighted",
            "canonical_component_id": "B.geometry_weighted",
            "ast": op(
                "multiply",
                op("exp_negative", SOURCE),
                op("add", const(0.5), op("multiply", const(0.5), GEOMETRY)),
            ),
        },
    ],
    "N_additive_channel": [
        {
            "raw_id": "N.null_ablation",
            "canonical_component_id": "N.null_ablation",
            "ast": const(0.0),
        },
        {
            "raw_id": "N.radial_tail",
            "canonical_component_id": "N.radial_tail",
            "ast": op("divide_safe", SOURCE, op("add", const(1.0), RADIAL)),
        },
        {
            "raw_id": "N.radial_tail_identity_alias",
            "canonical_component_id": "N.radial_tail",
            "ast": op(
                "multiply",
                const(1.0),
                op("divide_safe", SOURCE, op("add", const(1.0), RADIAL)),
            ),
        },
        {
            "raw_id": "N.sqrt_radial_tail",
            "canonical_component_id": "N.sqrt_radial_tail",
            "ast": op(
                "divide_safe",
                op("sqrt_positive", SOURCE),
                op("add", const(1.0), RADIAL),
            ),
        },
        {
            "raw_id": "N.state_radial_tail",
            "canonical_component_id": "N.state_radial_tail",
            "ast": op(
                "divide_safe",
                op("multiply", SOURCE, op("add", const(1.0), STATE)),
                op("add", const(1.0), RADIAL),
            ),
        },
    ],
    "A_nuisance": [
        {
            "raw_id": "A.off",
            "canonical_component_id": "A.off",
            "ast": const(1.0),
        },
        {
            "raw_id": "A.off_identity_alias",
            "canonical_component_id": "A.off",
            "ast": op("multiply", const(1.0), const(1.0)),
        },
        {
            "raw_id": "A.geometry_calibration",
            "canonical_component_id": "A.geometry_calibration",
            "ast": op(
                "add",
                const(1.0),
                op(
                    "multiply",
                    const(0.05),
                    op("subtract", GEOMETRY, const(0.5)),
                ),
            ),
        },
    ],
}

TARGET_COMPONENTS = {
    "known_rewrite": {
        "E_local_base": "E.newtonian",
        "B_continuous_gate": "B.low_acceleration",
        "N_additive_channel": "N.null_ablation",
        "A_nuisance": "A.off",
    },
    "known_combination": {
        "E_local_base": "E.newtonian",
        "B_continuous_gate": "B.low_acceleration",
        "N_additive_channel": "N.radial_tail",
        "A_nuisance": "A.off",
    },
    "potentially_new_synthesis": {
        "E_local_base": "E.rarlike_interpolation",
        "B_continuous_gate": "B.state_weighted",
        "N_additive_channel": "N.sqrt_radial_tail",
        "A_nuisance": "A.geometry_calibration",
    },
}

DATA_BOUNDARY = {
    "synthetic_rows_per_domain": 64,
    "real_galaxy_rows_read": 0,
    "real_group_rows_read": 0,
    "real_cluster_rows_read": 0,
    "real_lensing_rows_read": 0,
    "real_target_fields_read": [],
    "real_formula_scores_computed": 0,
    "sealed_rows_read": 0,
    "confirmation_rows_read": 0,
    "independent_rows_read": 0,
    "network_calls": 0,
    "model_calls": 0,
    "paid_calls": 0,
    "gpu_calls": 0,
}

CLAIM_BOUNDARY = {
    "synthetic_grammar_mechanics_validated": True,
    "ben_child_empirically_works": False,
    "candidate_physics_supported": False,
    "same_action_derived": False,
    "historical_novelty_established": False,
    "publication_ready": False,
    "gr_replaced": False,
    "synthetic_recovery_is_scientific_evidence": False,
    "real_scientific_evaluation_unlocked": False,
}

SAME_ACTION_BOUNDARY = {
    "action_or_field_equations_frozen": False,
    "independent_photon_multiplier_allowed": False,
    "synthetic_metric_projection": "Phi_syn=prediction/2; Psi_syn=prediction/2",
    "synthetic_projection_is_physical_derivation": False,
    "real_lensing_score_allowed": False,
    "real_lensing_unlock": False,
    "required_next_gate": (
        "derive metric potentials and matter/light coupling from one frozen action or "
        "field-equation system before any real lensing target access"
    ),
}


class BENSyntheticExecutionError(RuntimeError):
    """Raised when a frozen synthetic B+E+N boundary changes."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def confined(path: Path) -> Path:
    target = path.resolve()
    try:
        target.relative_to(ROOT)
    except ValueError as error:
        raise BENSyntheticExecutionError(f"path escaped repository: {path}") from error
    return target


def strict_keys(value: Any, expected: set[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        actual = set(value) if isinstance(value, dict) else set()
        raise BENSyntheticExecutionError(
            f"{label} keys changed; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def artifact_binding(path: Path) -> dict[str, str]:
    target = confined(path)
    return {
        "path": target.relative_to(ROOT).as_posix(),
        "file_sha256": file_sha256(target),
    }


def validate_binding(binding: Mapping[str, Any], label: str) -> dict[str, Any] | None:
    expected_keys = {"path", "file_sha256"}
    if "content_sha256" in binding:
        expected_keys.add("content_sha256")
    strict_keys(binding, expected_keys, label)
    target = confined(ROOT / str(binding["path"]))
    if not target.is_file() or file_sha256(target) != binding["file_sha256"]:
        raise BENSyntheticExecutionError(f"{label} missing, changed, or swapped")
    if target.suffix != ".json":
        return None
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise BENSyntheticExecutionError(f"{label} is not a JSON object")
    if "content_sha256" in binding and payload.get("content_sha256") != binding["content_sha256"]:
        raise BENSyntheticExecutionError(f"{label} content binding changed")
    return payload


def _number_key(value: float) -> str:
    return format(float(value), ".17g")


def _is_const(node: Mapping[str, Any], value: float) -> bool:
    return set(node) == {"const"} and float(node["const"]) == value


def validate_ast(node: Mapping[str, Any]) -> None:
    if set(node) == {"const"}:
        value = node["const"]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise BENSyntheticExecutionError("AST constant is not numeric")
        if not math.isfinite(float(value)):
            raise BENSyntheticExecutionError("AST constant is not finite")
        return
    if set(node) == {"var"}:
        if node["var"] not in VARIABLES:
            raise BENSyntheticExecutionError("AST variable is not typed")
        return
    strict_keys(node, {"op", "args"}, "AST operator")
    name = node["op"]
    args = node["args"]
    if name not in ALLOWED_OPERATORS or not isinstance(args, list):
        raise BENSyntheticExecutionError("AST operator changed")
    arity = 1 if name in {"sqrt_positive", "exp_negative"} else 2
    if len(args) != arity:
        raise BENSyntheticExecutionError("AST arity changed")
    for child in args:
        if not isinstance(child, dict):
            raise BENSyntheticExecutionError("AST child is not an object")
        validate_ast(child)


def normalize_ast(node: Mapping[str, Any]) -> dict[str, Any]:
    validate_ast(node)
    if set(node) == {"const"}:
        return {"const": float(node["const"])}
    if set(node) == {"var"}:
        return {"var": str(node["var"])}
    name = str(node["op"])
    children = [normalize_ast(child) for child in node["args"]]
    if name == "add":
        if _is_const(children[0], 0.0):
            return children[1]
        if _is_const(children[1], 0.0):
            return children[0]
        children.sort(key=canonical_json)
    elif name == "multiply":
        if any(_is_const(child, 0.0) for child in children):
            return {"const": 0.0}
        if _is_const(children[0], 1.0):
            return children[1]
        if _is_const(children[1], 1.0):
            return children[0]
        children.sort(key=canonical_json)
    elif (
        name == "subtract"
        and _is_const(children[1], 0.0)
        or name == "divide_safe"
        and _is_const(children[1], 1.0)
    ):
        return children[0]
    return {"op": name, "args": children}


def ast_dimension(node: Mapping[str, Any]) -> str:
    validate_ast(node)
    if "const" in node or "var" in node:
        return "1"
    child_dimensions = [ast_dimension(child) for child in node["args"]]
    if any(dimension != "1" for dimension in child_dimensions):
        raise BENSyntheticExecutionError("dimensionally invalid intermediate")
    return "1"


def evaluate_ast(node: Mapping[str, Any], predictors: np.ndarray) -> np.ndarray:
    validate_ast(node)
    predictors = np.asarray(predictors, dtype=np.float64)
    if predictors.ndim != 2 or predictors.shape[1] != 4:
        raise BENSyntheticExecutionError("predictors must be finite Nx4")
    if np.any(~np.isfinite(predictors)):
        raise BENSyntheticExecutionError("predictors are not finite")
    if "const" in node:
        return np.full(len(predictors), float(node["const"]), dtype=np.float64)
    if "var" in node:
        return predictors[:, VARIABLES.index(str(node["var"]))]
    values = [evaluate_ast(child, predictors) for child in node["args"]]
    name = node["op"]
    if name == "add":
        result = values[0] + values[1]
    elif name == "subtract":
        result = values[0] - values[1]
    elif name == "multiply":
        result = values[0] * values[1]
    elif name == "divide_safe":
        if np.any(np.abs(values[1]) <= 1.0e-15):
            raise BENSyntheticExecutionError("safe division denominator reached zero")
        result = values[0] / values[1]
    elif name == "sqrt_positive":
        if np.any(values[0] < 0.0):
            raise BENSyntheticExecutionError("positive square root received negative input")
        result = np.sqrt(values[0])
    elif name == "exp_negative":
        result = np.exp(-values[0])
    else:
        raise BENSyntheticExecutionError(f"unknown operator: {name}")
    if np.any(~np.isfinite(result)):
        raise BENSyntheticExecutionError("formula produced a non-finite value")
    return result


def expression(node: Mapping[str, Any]) -> str:
    if "const" in node:
        return _number_key(float(node["const"]))
    if "var" in node:
        return str(node["var"])
    args = [expression(child) for child in node["args"]]
    name = str(node["op"])
    if name == "add":
        return f"({args[0]}+{args[1]})"
    if name == "subtract":
        return f"({args[0]}-{args[1]})"
    if name == "multiply":
        return f"({args[0]}*{args[1]})"
    if name == "divide_safe":
        return f"divide_safe({args[0]},{args[1]})"
    if name == "sqrt_positive":
        return f"sqrt_positive({args[0]})"
    if name == "exp_negative":
        return f"exp_negative({args[0]})"
    raise BENSyntheticExecutionError(f"unknown expression operator: {name}")


def formula_ast(components: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return op(
        "add",
        op(
            "multiply",
            components["A_nuisance"]["ast"],
            components["E_local_base"]["ast"],
        ),
        op(
            "multiply",
            components["B_continuous_gate"]["ast"],
            components["N_additive_channel"]["ast"],
        ),
    )


def _provenance_for_components(components: Mapping[str, Mapping[str, Any]]) -> str:
    n_id = components["N_additive_channel"]["canonical_component_id"]
    b_id = components["B_continuous_gate"]["canonical_component_id"]
    if n_id == "N.null_ablation":
        return "known_rewrite"
    if b_id == "B.low_acceleration":
        return "known_combination"
    return "potentially_new_synthesis"


def build_candidate_registry(config: Mapping[str, Any]) -> dict[str, Any]:
    grammar = COMPONENT_GRAMMAR
    roles = tuple(grammar)
    if roles != (
        "E_local_base",
        "B_continuous_gate",
        "N_additive_channel",
        "A_nuisance",
    ):
        raise BENSyntheticExecutionError("B+E+N grammar role order changed")
    raw_entries: list[dict[str, Any]] = []
    classes: dict[str, dict[str, Any]] = {}
    for rows in itertools.product(*(grammar[role] for role in roles)):
        components = dict(zip(roles, rows, strict=True))
        raw_ast = formula_ast(components)
        canonical = normalize_ast(raw_ast)
        canonical_sha = sha256_value(canonical)
        raw_id = "raw." + ".".join(str(components[role]["raw_id"]) for role in roles)
        provenance = _provenance_for_components(components)
        raw_entries.append(
            {
                "raw_id": raw_id,
                "component_raw_ids": {role: components[role]["raw_id"] for role in roles},
                "canonical_expression_sha256": canonical_sha,
                "provenance_label": provenance,
            }
        )
        record = classes.setdefault(
            canonical_sha,
            {
                "class_id": f"ben.{canonical_sha[:20]}",
                "canonical_expression_sha256": canonical_sha,
                "canonical_expression": expression(canonical),
                "canonical_ast": canonical,
                "provenance_label": provenance,
                "provenance_is_authoritative_novelty_finding": False,
                "raw_member_ids": [],
            },
        )
        if record["provenance_label"] != provenance:
            raise BENSyntheticExecutionError("equivalence class mixed provenance labels")
        record["raw_member_ids"].append(raw_id)

    class_rows = sorted(classes.values(), key=lambda row: row["class_id"])
    for row in class_rows:
        row["raw_member_ids"].sort()
        row["raw_member_count"] = len(row["raw_member_ids"])
    raw_entries.sort(key=lambda row: row["raw_id"])
    counts_by_provenance = {
        label: sum(row["provenance_label"] == label for row in class_rows)
        for label in PROVENANCE_LABELS
    }
    registry = {
        "schema_version": "invariant-gravity-ben-candidate-registry-1.0",
        "architecture_id": "BEN-additive-cross-scale-v1",
        "formula_template": "A_nuisance*E_local_base+B_continuous_gate*N_additive_channel",
        "generation_packet_sha256": config["generation_packet_sha256"],
        "generation_inputs": ["typed_dimensionless_predictors", "frozen_component_grammar"],
        "generation_inputs_predictor_only": True,
        "M_temporal_phase_operator_included": False,
        "A_role": "bounded_source_calibration_nuisance_only",
        "equivalence_rule": (
            "exact recursive AST normalization with commutative operand sorting and "
            "zero/one identity collapse; no numeric probe equivalence"
        ),
        "raw_candidate_count": len(raw_entries),
        "equivalence_class_count": len(class_rows),
        "equivalence_classes_by_provenance": counts_by_provenance,
        "raw_candidates": raw_entries,
        "equivalence_classes": class_rows,
    }
    registry["content_sha256"] = sha256_value(registry)
    return registry


def _find_component(role: str, raw_id: str) -> dict[str, Any]:
    for row in COMPONENT_GRAMMAR[role]:
        if row["raw_id"] == raw_id:
            return row
    raise BENSyntheticExecutionError(f"unknown frozen component: {role}:{raw_id}")


def target_ast(target: Mapping[str, str]) -> dict[str, Any]:
    components = {role: _find_component(role, raw_id) for role, raw_id in target.items()}
    return normalize_ast(formula_ast(components))


def _class_by_ast(registry: Mapping[str, Any], node: Mapping[str, Any]) -> dict[str, Any]:
    target_sha = sha256_value(normalize_ast(node))
    matches = [
        row
        for row in registry["equivalence_classes"]
        if row["canonical_expression_sha256"] == target_sha
    ]
    if len(matches) != 1:
        raise BENSyntheticExecutionError("target class was not uniquely predeclared")
    return matches[0]


def validate_registry(registry: Mapping[str, Any]) -> None:
    if (
        registry["raw_candidate_count"] != 240
        or registry["equivalence_class_count"] != 60
        or registry["equivalence_classes_by_provenance"]
        != {
            "known_rewrite": 6,
            "known_combination": 18,
            "potentially_new_synthesis": 36,
        }
        or sum(int(row["raw_member_count"]) for row in registry["equivalence_classes"]) != 240
    ):
        raise BENSyntheticExecutionError("raw/equivalence accounting changed")
    if registry["content_sha256"] != sha256_value(
        {key: value for key, value in registry.items() if key != "content_sha256"}
    ):
        raise BENSyntheticExecutionError("candidate registry content hash changed")
    encoded = canonical_json(registry).lower()
    forbidden = (
        "object_id",
        "survey",
        "class_label",
        "observed",
        "response",
        "inferred_total_mass",
        "target_coefficient",
        "sealed_row",
    )
    for token in forbidden:
        if token in encoded:
            raise BENSyntheticExecutionError(f"candidate registry leaked token: {token}")


def _mse(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean(np.square(truth - prediction)))


def _formatted(value: float) -> str:
    return f"{value:.12e}"


def _domain_execution(domain: str, registry: Mapping[str, Any], rows: int) -> dict[str, Any]:
    predictors = shared._synthetic_predictors(domain, rows)
    if predictors.shape != (rows, 4):
        raise BENSyntheticExecutionError("domain adapter row count changed")
    predictions = {
        row["class_id"]: evaluate_ast(row["canonical_ast"], predictors)
        for row in registry["equivalence_classes"]
    }
    recovery: dict[str, Any] = {}
    all_wrong_law_pass = True
    for label, target in TARGET_COMPONENTS.items():
        node = target_ast(target)
        target_class = _class_by_ast(registry, node)
        truth = evaluate_ast(node, predictors)
        scores = {class_id: _mse(truth, prediction) for class_id, prediction in predictions.items()}
        minimum = min(scores.values())
        winners = sorted(
            class_id for class_id, score in scores.items() if abs(score - minimum) <= 1.0e-24
        )
        wrong = shared.evaluate_control_vector(domain, "wrong_law", predictors)
        wrong_mse = _mse(truth, wrong)
        wrong_pass = wrong_mse > 1.0e-10
        all_wrong_law_pass &= wrong_pass
        recovery[label] = {
            "injected_class_id": target_class["class_id"],
            "injected_provenance_label": target_class["provenance_label"],
            "winner_class_ids": winners,
            "injected_class_recovered": target_class["class_id"] in winners and minimum == 0.0,
            "unique_exact_recovery": winners == [target_class["class_id"]] and minimum == 0.0,
            "data_induced_tie_count": len(winners),
            "minimum_mse": _formatted(minimum),
            "wrong_law_mse": _formatted(wrong_mse),
            "wrong_law_rejected": wrong_pass,
        }

    target = TARGET_COMPONENTS["potentially_new_synthesis"]
    components = {role: _find_component(role, raw_id) for role, raw_id in target.items()}
    e_value = evaluate_ast(components["E_local_base"]["ast"], predictors)
    b_value = evaluate_ast(components["B_continuous_gate"]["ast"], predictors)
    n_value = evaluate_ast(components["N_additive_channel"]["ast"], predictors)
    a_value = evaluate_ast(components["A_nuisance"]["ast"], predictors)
    full = a_value * e_value + b_value * n_value
    base_only = a_value * e_value
    additive_only = b_value * n_value
    constant_gate = a_value * e_value + 0.5 * n_value
    nuisance_off = e_value + b_value * n_value
    ablations = {
        "additive_reconstruction_max_abs_error": _formatted(
            float(np.max(np.abs(full - (base_only + additive_only))))
        ),
        "N_channel_nonzero": bool(np.max(np.abs(full - base_only)) > 1.0e-10),
        "B_gate_replaced_by_constant_changes_output": bool(
            np.max(np.abs(full - constant_gate)) > 1.0e-10
        ),
        "A_nuisance_off_changes_output": bool(np.max(np.abs(full - nuisance_off)) > 1.0e-10),
        "N_zero_equals_base_only": True,
        "each_role_separately_evaluated": True,
    }
    return {
        "domain": domain,
        "synthetic_rows": rows,
        "adapter_output_sha256": hashlib.sha256(
            np.ascontiguousarray(predictors, dtype="<f8").tobytes()
        ).hexdigest(),
        "real_rows_read": 0,
        "real_target_fields_read": [],
        "real_scores_computed": 0,
        "recovery": recovery,
        "all_recovery_pass": all(row["injected_class_recovered"] for row in recovery.values()),
        "all_wrong_law_controls_pass": all_wrong_law_pass,
        "channel_ablations": ablations,
    }


def dimension_and_limit_controls(registry: Mapping[str, Any]) -> dict[str, Any]:
    probe = np.asarray(
        [
            [1.0e-6, 0.1, 0.2, 0.1],
            [1.0, 1.0, 0.5, 0.5],
            [1.0e6, 2.0, 0.8, 0.9],
        ],
        dtype=np.float64,
    )
    all_dimensionless = True
    all_finite_nonnegative = True
    for row in registry["equivalence_classes"]:
        all_dimensionless &= ast_dimension(row["canonical_ast"]) == "1"
        values = evaluate_ast(row["canonical_ast"], probe)
        all_finite_nonnegative &= bool(np.all(np.isfinite(values)) and np.all(values >= 0.0))

    high = probe[2:3]
    base_high_relative_errors = []
    gate_high_maxima = []
    for e_row in COMPONENT_GRAMMAR["E_local_base"]:
        if e_row["raw_id"].endswith("alias"):
            continue
        value = evaluate_ast(e_row["ast"], high)[0]
        base_high_relative_errors.append(abs(value - high[0, 0]) / high[0, 0])
    for b_row in COMPONENT_GRAMMAR["B_continuous_gate"]:
        if b_row["raw_id"].endswith("alias"):
            continue
        gate_high_maxima.append(abs(evaluate_ast(b_row["ast"], high)[0]))
    nuisance_probe = np.column_stack(
        (
            np.ones(101),
            np.ones(101),
            np.linspace(0.0, 1.0, 101),
            np.linspace(0.0, 1.0, 101),
        )
    )
    a = evaluate_ast(
        _find_component("A_nuisance", "A.geometry_calibration")["ast"],
        nuisance_probe,
    )
    return {
        "all_intermediates_dimensionless": bool(all_dimensionless),
        "all_probe_predictions_finite_nonnegative": bool(all_finite_nonnegative),
        "maximum_high_source_base_relative_error": _formatted(max(base_high_relative_errors)),
        "maximum_high_source_gate_value": _formatted(max(gate_high_maxima)),
        "high_source_local_limit_pass": bool(max(base_high_relative_errors) <= 1.0e-6),
        "high_source_additive_suppression_pass": bool(max(gate_high_maxima) <= 1.0e-12),
        "A_nuisance_minimum": _formatted(float(np.min(a))),
        "A_nuisance_maximum": _formatted(float(np.max(a))),
        "A_nuisance_bounded_calibration_pass": bool(np.min(a) >= 0.975 and np.max(a) <= 1.025),
        "continuous_gates_only": True,
        "object_survey_or_class_switches": 0,
        "M_temporal_phase_operator_occurrences": 0,
    }


def raw_to_canonical_parity(registry: Mapping[str, Any]) -> dict[str, Any]:
    probe = np.asarray(
        [
            [0.03, 0.1, 0.1, 0.2],
            [0.2, 0.5, 0.4, 0.6],
            [2.0, 1.7, 0.8, 0.9],
        ],
        dtype=np.float64,
    )
    class_map = {row["canonical_expression_sha256"]: row for row in registry["equivalence_classes"]}
    maximum = 0.0
    checked = 0
    roles = tuple(COMPONENT_GRAMMAR)
    for rows in itertools.product(*(COMPONENT_GRAMMAR[role] for role in roles)):
        components = dict(zip(roles, rows, strict=True))
        raw = formula_ast(components)
        canonical = normalize_ast(raw)
        class_row = class_map[sha256_value(canonical)]
        difference = float(
            np.max(
                np.abs(evaluate_ast(raw, probe) - evaluate_ast(class_row["canonical_ast"], probe))
            )
        )
        maximum = max(maximum, difference)
        checked += 1
    return {
        "raw_candidates_checked": checked,
        "maximum_raw_to_canonical_abs_difference": _formatted(maximum),
        "all_raw_to_canonical_parity_pass": maximum <= 1.0e-14 and checked == 240,
    }


def validate_source_receipts(config: Mapping[str, Any]) -> None:
    payloads = {
        name: validate_binding(binding, f"source binding {name}")
        for name, binding in config["source_bindings"].items()
    }
    shared_receipt = payloads["shared_receipt"]
    if (
        shared_receipt is None
        or shared_receipt.get("decision")
        != "PASS_BOUNDED_SCAFFOLD_CONTROLS_ONLY_CHILD_NOT_EXECUTED"
        or shared_receipt.get("architecture_binding", {}).get("structural_child_executed")
        is not False
        or shared_receipt.get("retrospective_exposed_adapter_smokes", {}).get(
            "real_formula_scores_computed"
        )
        != 0
        or shared_receipt.get("claims", {}).get("ben_child_empirically_works") is not False
    ):
        raise BENSyntheticExecutionError("shared evaluator claim boundary changed")
    generation_sha = shared_receipt.get("two_stage_separation", {}).get("generation_packet_sha256")
    if generation_sha != config["generation_packet_sha256"]:
        raise BENSyntheticExecutionError("shared generation packet binding changed")
    parent = payloads["parent_registry_receipt"]
    if parent is None or parent.get("decision") != (
        "PASS_ALL_FIVE_PARENTS_REGISTERED_EVIDENCE_INTACT"
    ):
        raise BENSyntheticExecutionError("parent registry decision changed")
    recombination = payloads["recombination_receipt"]
    top = {} if recombination is None else recombination.get("top_architecture", {})
    if (
        recombination is None
        or recombination.get("decision")
        != "PASS_TARGET_BLIND_STRUCTURAL_PREFLIGHT_CHILDREN_NOT_EXECUTED"
        or top.get("architecture_id") != "BEN-additive-cross-scale-v1"
        or top.get("base_lead") != "emergent_gravity_transition"
        or top.get("gate_lead") != "baryonic_transition_variable"
        or top.get("additive_channel_lead") != "nonlocal_boundary_response"
        or top.get("nuisance_lead") != "dynamical_age_spectral_clock"
        or top.get("deferred_lead") != "massive_field_orbital_resonance"
        or top.get("structural_descendants_only") is not True
    ):
        raise BENSyntheticExecutionError("B+E+N architecture binding changed")


def validate_config_contract(config: Mapping[str, Any]) -> None:
    strict_keys(
        config,
        {
            "schema_version",
            "status",
            "purpose",
            "implementation_source",
            "implementation_source_normalized_sha256",
            "verifier_test",
            "source_bindings",
            "generation_packet_sha256",
            "typed_variables",
            "allowed_operators",
            "component_grammar_sha256",
            "component_grammar_counts",
            "target_components",
            "rows_per_domain",
            "same_action_boundary",
            "data_boundary",
            "claim_boundary",
            "receipt_path",
        },
        "synthetic B+E+N config",
    )
    if (
        config["schema_version"] != CONFIG_SCHEMA
        or config["status"] != "frozen_append_only_synthetic_ben_execution"
        or config["verifier_test"]
        != {"path": TEST_PATH.as_posix(), "file_sha256": EXPECTED_TEST_SHA256}
        or config["source_bindings"] != EXPECTED_SOURCE_BINDINGS
        or config["generation_packet_sha256"]
        != "7bfae7b2d7cfe615bb5f80e2f375e01b72f249237a126c592364eb1dfeaf2785"
        or config["typed_variables"]
        != [{"symbol": name, "dimension": "1", "finite": True} for name in VARIABLES]
        or config["allowed_operators"] != sorted(ALLOWED_OPERATORS)
        or config["component_grammar_sha256"] != sha256_value(COMPONENT_GRAMMAR)
        or config["component_grammar_counts"]
        != {role: len(rows) for role, rows in COMPONENT_GRAMMAR.items()}
        or config["target_components"] != TARGET_COMPONENTS
        or config["rows_per_domain"] != 64
        or config["same_action_boundary"] != SAME_ACTION_BOUNDARY
        or config["data_boundary"] != DATA_BOUNDARY
        or config["claim_boundary"] != CLAIM_BOUNDARY
        or config["receipt_path"] != RECEIPT_PATH.as_posix()
    ):
        raise BENSyntheticExecutionError("synthetic B+E+N frozen contract changed")


def load_config(path: Path, expected_sha256: str) -> dict[str, Any]:
    target = confined(path)
    if target != ROOT / CONFIG_PATH:
        raise BENSyntheticExecutionError("synthetic B+E+N config path changed")
    if not target.is_file() or file_sha256(target) != expected_sha256:
        raise BENSyntheticExecutionError("synthetic B+E+N config hash changed")
    config = json.loads(target.read_text(encoding="utf-8"))
    validate_config_contract(config)
    source = confined(ROOT / str(config["implementation_source"]))
    if (
        source != Path(__file__).resolve()
        or normalized_sha256(source) != config["implementation_source_normalized_sha256"]
    ):
        raise BENSyntheticExecutionError("synthetic B+E+N source changed")
    validate_binding(config["verifier_test"], "verifier test")
    validate_source_receipts(config)
    config["_config_sha256"] = expected_sha256
    return config


def execute(config: Mapping[str, Any]) -> dict[str, Any]:
    registry = build_candidate_registry(config)
    validate_registry(registry)
    parity = raw_to_canonical_parity(registry)
    controls = dimension_and_limit_controls(registry)
    domains = {
        domain: _domain_execution(domain, registry, int(config["rows_per_domain"]))
        for domain in DOMAINS
    }
    all_recovery = all(row["all_recovery_pass"] for row in domains.values())
    all_wrong = all(row["all_wrong_law_controls_pass"] for row in domains.values())
    all_ablations = all(
        row["channel_ablations"]["additive_reconstruction_max_abs_error"] == "0.000000000000e+00"
        and row["channel_ablations"]["N_channel_nonzero"]
        and row["channel_ablations"]["B_gate_replaced_by_constant_changes_output"]
        and row["channel_ablations"]["A_nuisance_off_changes_output"]
        for row in domains.values()
    )
    lensing_target = target_ast(TARGET_COMPONENTS["potentially_new_synthesis"])
    lensing_x = shared._synthetic_predictors("lensing_metric", int(config["rows_per_domain"]))
    metric_prediction = evaluate_ast(lensing_target, lensing_x)
    phi = 0.5 * metric_prediction
    psi = 0.5 * metric_prediction
    metric_error = float(np.max(np.abs(phi + psi - metric_prediction)))
    return {
        "candidate_registry": registry,
        "raw_to_canonical_parity": parity,
        "dimension_and_limit_controls": controls,
        "synthetic_domains": domains,
        "same_action_metric_interface": {
            **SAME_ACTION_BOUNDARY,
            "synthetic_projection_max_abs_error": _formatted(metric_error),
            "synthetic_projection_parity_pass": metric_error == 0.0,
            "empirical_score_key_present": False,
            "empirical_rows_read": 0,
        },
        "gates": {
            "raw_equivalence_accounting_pass": (
                registry["raw_candidate_count"] == 240 and registry["equivalence_class_count"] == 60
            ),
            "raw_to_canonical_parity_pass": parity["all_raw_to_canonical_parity_pass"],
            "dimension_and_limit_controls_pass": all(
                (
                    controls["all_intermediates_dimensionless"],
                    controls["all_probe_predictions_finite_nonnegative"],
                    controls["high_source_local_limit_pass"],
                    controls["high_source_additive_suppression_pass"],
                    controls["A_nuisance_bounded_calibration_pass"],
                )
            ),
            "all_four_domain_injected_recovery_pass": all_recovery,
            "all_four_domain_wrong_law_controls_pass": all_wrong,
            "all_four_domain_channel_ablations_pass": all_ablations,
            "same_action_synthetic_projection_pass": metric_error == 0.0,
            "real_scientific_evaluation_locked": True,
        },
    }


def expected_receipt(config: Mapping[str, Any]) -> dict[str, Any]:
    result = execute(config)
    if not all(result["gates"].values()):
        raise BENSyntheticExecutionError("synthetic B+E+N controls failed")
    body = {
        "schema_version": RECEIPT_SCHEMA,
        "status": STATUS,
        "decision": DECISION,
        "source_bindings": {
            "config": artifact_binding(ROOT / CONFIG_PATH),
            "implementation": artifact_binding(ROOT / str(config["implementation_source"])),
            "test": config["verifier_test"],
            "predecessors": config["source_bindings"],
        },
        "execution_order": [
            "hash_bound_predecessor_validation",
            "predictor_only_candidate_registry_freeze",
            "exact_equivalence_collapse",
            "synthetic_adapter_generation",
            "synthetic_injected_recovery_and_controls",
            "receipt_seal",
        ],
        "generation_completed_before_synthetic_responses": True,
        "post_response_generation_calls": 0,
        "candidate_registry": result["candidate_registry"],
        "raw_to_canonical_parity": result["raw_to_canonical_parity"],
        "dimension_and_limit_controls": result["dimension_and_limit_controls"],
        "synthetic_domains": result["synthetic_domains"],
        "same_action_metric_interface": result["same_action_metric_interface"],
        "gates": result["gates"],
        "compute_accounting": {
            "synthetic_domain_packets": 4,
            "synthetic_injections_per_domain": 3,
            "candidate_registry_vector_evaluation_calls": 240,
            "candidate_registry_row_predictions": 15_360,
            "candidate_score_comparisons": 720,
            "candidate_score_row_comparisons": 46_080,
            "injected_truth_vector_evaluation_calls": 12,
            "wrong_law_vector_evaluation_calls": 12,
            "channel_ablation_component_vector_evaluation_calls": 16,
            "dimension_and_limit_vector_evaluation_calls": 67,
            "raw_to_canonical_vector_evaluation_calls": 480,
            "same_action_metric_vector_evaluation_calls": 1,
            "total_formula_vector_evaluation_calls": 828,
            "synthetic_adapter_calls": 5,
            "real_adapter_calls": 0,
            "real_formula_evaluation_calls": 0,
            "network_calls": 0,
            "model_calls": 0,
            "paid_calls": 0,
            "gpu_calls": 0,
        },
        "data_boundary": DATA_BOUNDARY,
        "claim_boundary": CLAIM_BOUNDARY,
        "provenance_policy": {
            "labels": list(PROVENANCE_LABELS),
            "labels_are_authoritative_historical_novelty_findings": False,
            "known_rewrite_meaning": "null-N ablation of a frozen local base and nuisance",
            "known_combination_meaning": (
                "frozen known-style local base, low-acceleration gate, and additive radial "
                "channel combined without a novelty claim"
            ),
            "potentially_new_synthesis_meaning": (
                "a target-blind component combination flagged for later literature review, "
                "not evidence of novelty"
            ),
        },
        "limitations": [
            "Recovery is guaranteed-plumbing evidence because targets are injected from the frozen finite grammar.",
            "The same-action metric projection is a synthetic interface parity check, not a covariant action or lensing law.",
            "No real galaxy, group, cluster, or lensing target was opened or scored.",
            "The A channel is a bounded calibration nuisance only; M is excluded until causal dynamics exist.",
            "Potentially-new-synthesis labels require specialist prior-art review and do not establish novelty.",
            "Nothing in this receipt supports modified gravity, dark-matter elimination, publication readiness, or a replacement for GR.",
        ],
    }
    body["content_sha256"] = sha256_value(body)
    return body


def require_exact_receipt(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    strict_keys(actual, set(expected), "synthetic B+E+N receipt")
    if dict(actual) != dict(expected):
        raise BENSyntheticExecutionError("synthetic B+E+N receipt does not reconstruct exactly")


def _write_json_no_clobber(path: Path, value: Mapping[str, Any]) -> None:
    target = confined(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = (canonical_json(value) + "\n").encode("utf-8")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError as error:
            raise BENSyntheticExecutionError(
                "atomic no-clobber publication refused existing receipt"
            ) from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_receipt(
    config_path: Path, expected_config_sha256: str, output_path: Path
) -> dict[str, Any]:
    output = confined(output_path)
    if output != ROOT / RECEIPT_PATH:
        raise BENSyntheticExecutionError("synthetic B+E+N receipt path changed")
    config = load_config(config_path, expected_config_sha256)
    receipt = expected_receipt(config)
    _write_json_no_clobber(output, receipt)
    return receipt


def check_receipt(
    config_path: Path, expected_config_sha256: str, receipt_path: Path
) -> dict[str, Any]:
    config = load_config(config_path, expected_config_sha256)
    target = confined(receipt_path)
    if target != ROOT / RECEIPT_PATH:
        raise BENSyntheticExecutionError("synthetic B+E+N receipt path changed")
    actual = json.loads(target.read_text(encoding="utf-8"))
    require_exact_receipt(actual, expected_receipt(config))
    return {
        "valid": True,
        "decision": DECISION,
        "raw_candidate_count": 240,
        "equivalence_class_count": 60,
        "all_synthetic_controls_pass": True,
        "real_scientific_evaluation_unlocked": False,
        "same_action_derived": False,
        "scientific_claim_allowed": False,
        "receipt_sha256": file_sha256(target),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    write = commands.add_parser("write-receipt")
    write.add_argument("--config", type=Path, required=True)
    write.add_argument("--expected-config-sha256", required=True)
    write.add_argument("--output", type=Path, required=True)
    check = commands.add_parser("check")
    check.add_argument("--config", type=Path, required=True)
    check.add_argument("--expected-config-sha256", required=True)
    check.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "write-receipt":
        result = write_receipt(args.config, args.expected_config_sha256, args.output)
    else:
        result = check_receipt(args.config, args.expected_config_sha256, args.receipt)
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
