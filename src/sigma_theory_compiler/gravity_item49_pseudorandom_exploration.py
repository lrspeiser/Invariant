"""Item 49 collision-free seeded exploration of a cross-mechanism formula grammar."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
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
from sigma_theory_compiler.gravity_item44_scale_hierarchy import _predict as _item44_predict
from sigma_theory_compiler.gravity_item45_universal_interactions import (
    _item44_oof,
    _object_weights,
    _ordinary_crossfit,
    _paired_p,
    _predict as _item45_predict,
    _score,
    _variant_arrays as _item45_variant_arrays,
    load_config as _load_item45_config,
)
from sigma_theory_compiler.gravity_item46_dimensionless_generator import (
    _physical_log_values as _item46_physical_log_values,
    _predict as _item46_predict,
    load_config as _load_item46_config,
    pi_vectors as _item46_pi_vectors,
)
from sigma_theory_compiler.gravity_item47_operator_generator import (
    _item45_oof,
    _item46_oof,
    _predict as _item47_predict,
    _shape_by_object,
    load_config as _load_item47_config,
    operator_bank_from_arrays as _item47_operator_bank_from_arrays,
)
from sigma_theory_compiler.gravity_item48_action_generator import (
    _evaluation_arrays as _item48_evaluation_arrays,
    _fixed_oof as _item48_fixed_oof,
    _predict as _item48_predict,
    action_bank_from_arrays as _item48_action_bank_from_arrays,
    load_config as _load_item48_config,
)
from sigma_theory_compiler.pseudorandom_ordinal import PseudorandomOrdinalPermutation


CONFIG_PATH = Path("configs/gravity_item49_pseudorandom_exploration_v1.json")
GOAL_PATH = Path("docs/GRAVITY_HIDDEN_VARIABLE_AND_THEORY_SEARCH_GOALS.md")
POLICY_PATH = Path("configs/gravity_empirical_counterexample_policy_v1.json")
ITEM44_FEATURE_PATH = Path(
    "runs/gravity/roadmap/item-44-scale-hierarchy-v1-source/joint-scale-features.json"
)
ITEM45_FEATURE_PATH = Path(
    "runs/gravity/roadmap/item-45-universal-interactions-v1-source/interaction-features.json"
)
ITEM46_FEATURE_PATH = Path(
    "runs/gravity/roadmap/item-46-dimensionless-generator-v1-source/dimensionless-features.json"
)
ITEM47_FEATURE_PATH = Path(
    "runs/gravity/roadmap/item-47-operator-generator-v1-source/operator-features.json"
)
ITEM48_FEATURE_PATH = Path(
    "runs/gravity/roadmap/item-48-action-generator-v1-source/action-features.json"
)
ITEM47_EVALUATION_PATH = Path(
    "runs/gravity/roadmap/item-47-operator-generator-v1-source/joint-evaluation-result.json"
)
ITEM48_EVALUATION_PATH = Path(
    "runs/gravity/roadmap/item-48-action-generator-v1-source/joint-evaluation-result.json"
)


class GravityItem49Error(RuntimeError):
    """Raised when an Item 49 schedule, program, equivalence, or evidence gate fails."""


def load_config(root: Path) -> dict[str, Any]:
    config = _read_json(root / CONFIG_PATH)
    validate_config(root, config)
    return config


def validate_config(root: Path, config: Mapping[str, Any]) -> None:
    if (
        config.get("schema_version")
        != "invariant-gravity-item49-pseudorandom-exploration-config-1.0"
        or int(config.get("item", -1)) != 49
    ):
        raise GravityItem49Error("unexpected Item 49 config")
    if _sha256_file(root / GOAL_PATH) != str(config["stable_goal_sha256"]):
        raise GravityItem49Error("stable gravity goal changed")
    if re.fullmatch(r"[0-9a-f]{40}", str(config["scientific_freeze_commit"])) is None:
        raise GravityItem49Error("Item 49 scientific freeze is not bound to a commit")
    for relative, expected in config["scientific_dependencies"].items():
        if _sha256_file(root / str(relative)) != str(expected):
            raise GravityItem49Error(f"scientific dependency changed: {relative}")
    predecessor = _read_json(root / "runs/gravity/roadmap/item-48-action-generator-v1.json")
    required = config["required_predecessor"]
    if predecessor.get("content_sha256") != required["content_sha256"]:
        raise GravityItem49Error("Item 48 content binding changed")
    if predecessor.get("decision") != required["decision"]:
        raise GravityItem49Error("Item 48 decision binding changed")
    if int(predecessor["selected_candidate"]["candidate_id"]) != int(
        required["selected_candidate_id"]
    ):
        raise GravityItem49Error("Item 48 candidate binding changed")
    grammar = config["program_grammar"]
    expected_space = (
        int(grammar["primitive_count"])
        * len(grammar["unary_transforms"])
        * int(grammar["primitive_count"])
        * len(grammar["unary_transforms"])
        * len(grammar["binary_operators"])
        * len(grammar["mixing_grid"])
        * math.prod(len(values) for values in grammar["outer_parameter_grids"].values())
    )
    if expected_space != int(grammar["full_ordinal_space"]):
        raise GravityItem49Error("full grammar size changed")
    if expected_space > 2**64:
        raise GravityItem49Error("grammar exceeds scheduler range")
    primitive = config["primitive_bank"]
    if sum(int(primitive[key]) for key in (
        "item45_interactions",
        "item46_dimensionless_pi_groups",
        "item47_operators",
        "item48_action_coordinates",
    )) != int(primitive["total_primitives"]):
        raise GravityItem49Error("primitive count changed")
    if int(primitive["total_primitives"]) != int(grammar["primitive_count"]):
        raise GravityItem49Error("primitive and grammar counts disagree")
    schedules = config["schedules"]
    if (
        int(schedules["pseudorandom"]["sample_positions"]) != 1048576
        or int(schedules["sequential_ordinal_control"]["sample_ordinals"]) != 1048576
        or not bool(schedules["equal_raw_budget"])
    ):
        raise GravityItem49Error("schedule budget changed")
    discovery = config["discovery_policy"]
    if not bool(discovery["single_empirical_counterexample_is_not_a_formula_or_family_veto"]):
        raise GravityItem49Error("one empirical mismatch became a veto")
    if not bool(discovery["counterexample_count_alone_is_never_decisive"]):
        raise GravityItem49Error("count-only rejection entered Item 49")
    if bool(discovery["finite_empirical_sample_may_prune_family"]):
        raise GravityItem49Error("finite empirical family pruning entered Item 49")
    policy = load_counterexample_policy(root / POLICY_PATH)
    if policy["empirical_evidence"]["single_counterexample_terminal_rejection_allowed"] is not False:
        raise GravityItem49Error("executable counterexample policy changed")
    if bool(config["scope"]["full_grammar_exhausted"]):
        raise GravityItem49Error("sampled grammar was mislabeled exhausted")
    if bool(config["scope"]["trillion_formula_campaign_executed"]):
        raise GravityItem49Error("addressed grammar was mislabeled executed")
    if bool(config["scope"]["fresh_confirmation_claim_allowed"]):
        raise GravityItem49Error("fresh confirmation entered retrospective Item 49")


def _contract_digest(config: Mapping[str, Any]) -> str:
    value = json.loads(json.dumps(config))
    value["scientific_freeze_commit"] = "<BOUND_COMMIT>"
    return _sha256_bytes(_canonical_bytes(value))


def _source_path(root: Path, config: Mapping[str, Any], key: str) -> Path:
    return root / str(config["paths"]["source_dir"]) / str(config["paths"][key])


def _primitive_sources(root: Path) -> list[dict[str, Any]]:
    return [
        _read_json(root / ITEM45_FEATURE_PATH),
        _read_json(root / ITEM46_FEATURE_PATH),
        _read_json(root / ITEM47_FEATURE_PATH),
        _read_json(root / ITEM48_FEATURE_PATH),
    ]


def _response_blind_u(root: Path, sources: Sequence[Mapping[str, Any]]) -> np.ndarray:
    item44 = _read_json(root / ITEM44_FEATURE_PATH)
    item45_rows = sources[0]["records"]
    rows = item44["records"]
    if len(rows) != len(item45_rows):
        raise GravityItem49Error("response-blind u row count changed")
    for index, (row, aligned) in enumerate(zip(rows, item45_rows, strict=True)):
        if (
            int(aligned["source_row_index"]) != index
            or row["population"] != aligned["population"]
            or row["object"] != aligned["object"]
        ):
            raise GravityItem49Error("response-blind u/source alignment changed")
    u = np.asarray([row["u"] for row in rows], dtype=float)
    if u.shape != (112,) or np.any(u <= 0.0) or not np.all(np.isfinite(u)):
        raise GravityItem49Error("response-blind u values changed")
    return u


def primitive_labels(sources: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    item45, item46, item47, item48 = sources
    labels: list[dict[str, Any]] = []
    for row in item45["recipe_catalog"]:
        labels.append(
            {
                "primitive_id": len(labels),
                "source_item": 45,
                "source_recipe_id": int(row["recipe_id"]),
                "expression": row["coordinate_expression"],
                "mechanism": row["niche"],
                "creativity_label": row["creativity_label"],
            }
        )
    for row in item46["pi_catalog"]:
        labels.append(
            {
                "primitive_id": len(labels),
                "source_item": 46,
                "source_recipe_id": int(row["recipe_id"]),
                "expression": row["expression"],
                "mechanism": "dimensionless_pi_group",
                "creativity_label": row["creativity_label"],
            }
        )
    for row in item47["operator_catalog"]:
        labels.append(
            {
                "primitive_id": len(labels),
                "source_item": 47,
                "source_recipe_id": int(row["recipe_id"]),
                "expression": f"{row['operator_class']}:{row['source']}:{row['scale']}",
                "mechanism": row["operator_class"],
                "creativity_label": row["creativity_label"],
            }
        )
    for row in item48["action_catalog"]:
        labels.append(
            {
                "primitive_id": len(labels),
                "source_item": 48,
                "source_recipe_id": int(row["recipe_id"]),
                "expression": f"{row['action_class']}:{row['source']}:{row['variant']}",
                "mechanism": row["action_class"],
                "creativity_label": row["creativity_label"],
            }
        )
    return labels


def primitive_bank_from_sources(
    sources: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, dict[str, Any]]:
    item45, item46, item47, item48 = sources
    row_sets = [source["records"] for source in sources]
    row_count = len(row_sets[0])
    if any(len(rows) != row_count for rows in row_sets):
        raise GravityItem49Error("primitive source row counts differ")
    for index, rows in enumerate(zip(*row_sets, strict=True)):
        if any(
            int(row["source_row_index"]) != index
            or row["population"] != rows[0]["population"]
            or row["object"] != rows[0]["object"]
            for row in rows
        ):
            raise GravityItem49Error("primitive source row alignment changed")
    bank = np.column_stack(
        (
            np.asarray([row["interaction_coordinates"] for row in item45["records"]]),
            np.asarray([row["pi_coordinates"] for row in item46["records"]]),
            np.asarray([row["operator_coordinates"] for row in item47["records"]]),
            np.asarray([row["action_coordinates"] for row in item48["records"]]),
        )
    ).T
    if bank.shape != (440, 112) or not np.all(np.isfinite(bank)):
        raise GravityItem49Error("primitive bank shape or finiteness changed")
    if np.any(bank <= 0.0) or np.any(bank >= 1.0):
        raise GravityItem49Error("primitive bank is not strictly bounded")
    groups = np.concatenate(
        (
            np.repeat(45, 64),
            np.repeat(46, 184),
            np.repeat(47, 96),
            np.repeat(48, 96),
        )
    )
    return bank, {
        "shape": list(bank.shape),
        "item_counts": {
            str(item): int(np.sum(groups == item)) for item in (45, 46, 47, 48)
        },
        "bank_sha256": hashlib.sha256(
            np.round(bank, 12).astype("<f8").tobytes()
        ).hexdigest(),
        "minimum": float(np.min(bank)),
        "maximum": float(np.max(bank)),
    }


def build_primitive_receipt(root: Path) -> dict[str, Any]:
    config = load_config(root)
    sources = _primitive_sources(root)
    _bank, audit = primitive_bank_from_sources(sources)
    labels = primitive_labels(sources)
    if len(labels) != 440:
        raise GravityItem49Error("primitive label count changed")
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item49-primitive-bank-1.0",
            "item": 49,
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "response_fields_read": [],
            "response_values_used": 0,
            "audit": audit,
            "primitives": labels,
            "source_content_sha256": {
                str(source["item"]): source["content_sha256"] for source in sources
            },
        }
    )


_RADICES = (16, 16, 16, 16, 8, 8, 440, 8, 440)
_FIELDS = (
    "transition_index",
    "exponent_index",
    "amplitude_index",
    "mixing_index",
    "operator_index",
    "right_transform_index",
    "right_primitive_index",
    "left_transform_index",
    "left_primitive_index",
)


def decode_ordinals(ordinals: np.ndarray, config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    values = np.asarray(ordinals, dtype=np.uint64).copy()
    if np.any(values >= np.uint64(config["program_grammar"]["full_ordinal_space"])):
        raise GravityItem49Error("ordinal outside full grammar")
    decoded: dict[str, np.ndarray] = {"ordinal": values.copy()}
    for field, radix in zip(_FIELDS, _RADICES, strict=True):
        values, digit = np.divmod(values, np.uint64(radix))
        decoded[field] = digit.astype(np.int16)
    if np.any(values != 0):
        raise GravityItem49Error("mixed-radix decoder did not exhaust ordinal")
    return decoded


def _lane_ordinals(config: Mapping[str, Any], lane: str) -> np.ndarray:
    if lane == "pseudorandom":
        schedule = config["schedules"]["pseudorandom"]
        count = int(schedule["sample_positions"])
        permutation = PseudorandomOrdinalPermutation(
            int(config["program_grammar"]["full_ordinal_space"]), str(schedule["seed"])
        )
        return np.fromiter(
            (permutation.at(position) for position in range(count)),
            dtype=np.uint64,
            count=count,
        )
    if lane == "sequential_ordinal_control":
        schedule = config["schedules"]["sequential_ordinal_control"]
        start = int(schedule["start_ordinal"])
        count = int(schedule["sample_ordinals"])
        return np.arange(start, start + count, dtype=np.uint64)
    raise GravityItem49Error(f"unknown schedule lane: {lane}")


def _admissible_parameter_table(config: Mapping[str, Any]) -> np.ndarray:
    gate = config["admissibility"]
    grids = config["program_grammar"]["outer_parameter_grids"]
    amplitude, exponent, transition = np.meshgrid(
        np.asarray(grids["amplitude"], dtype=float),
        np.asarray(grids["acceleration_exponent"], dtype=float),
        np.asarray(grids["transition_u"], dtype=float),
        indexing="ij",
    )
    aa = amplitude.reshape(-1)
    pp = exponent.reshape(-1)
    tt = transition.reshape(-1)
    u = np.logspace(
        float(gate["probe_log10_u_min"]),
        float(gate["probe_log10_u_max"]),
        int(gate["probe_points"]),
    )
    h = np.asarray(gate["probe_program_coordinates"], dtype=float)
    multiplier = 1.0 + aa[:, None, None] * np.power(
        u[None, None, :], -pp[:, None, None]
    ) / (1.0 + u[None, None, :] / tt[:, None, None]) * (
        0.05 + 0.95 * h[None, :, None]
    )
    finite = np.all(np.isfinite(multiplier), axis=(1, 2))
    positive = finite & np.all(multiplier >= float(gate["minimum_multiplier"]), axis=(1, 2))
    bounded = positive & np.all(multiplier <= float(gate["maximum_multiplier"]), axis=(1, 2))
    logs = np.log10(np.maximum(multiplier, np.finfo(float).tiny))
    local = bounded & (
        np.max(np.abs(logs[:, :, -1]), axis=1)
        <= float(gate["maximum_high_acceleration_log10_deviation"])
    )
    return local & (
        np.max(np.abs(logs[:, :, 0]), axis=1)
        >= float(gate["minimum_low_acceleration_absolute_log10_deviation"])
    )


def _physical_mask(decoded: Mapping[str, np.ndarray], config: Mapping[str, Any]) -> np.ndarray:
    table = _admissible_parameter_table(config)
    index = (
        np.asarray(decoded["amplitude_index"], int) * 256
        + np.asarray(decoded["exponent_index"], int) * 16
        + np.asarray(decoded["transition_index"], int)
    )
    return table[index]


def _select_rows(decoded: Mapping[str, np.ndarray], indices: np.ndarray) -> dict[str, np.ndarray]:
    return {key: np.asarray(value)[indices] for key, value in decoded.items()}


def _symbolic_representatives(
    decoded: Mapping[str, np.ndarray], config: Mapping[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    left = (
        np.asarray(decoded["left_primitive_index"], np.uint16) * 8
        + np.asarray(decoded["left_transform_index"], np.uint16)
    )
    right = (
        np.asarray(decoded["right_primitive_index"], np.uint16) * 8
        + np.asarray(decoded["right_transform_index"], np.uint16)
    )
    operator = np.asarray(decoded["operator_index"], np.uint8).copy()
    mixing = np.asarray(decoded["mixing_index"], np.uint8).copy()
    mixing_values = np.asarray(config["program_grammar"]["mixing_grid"], dtype=float)[mixing]
    kind = np.full(len(left), 2, dtype=np.uint8)
    commutative = np.isin(operator, (2, 7)) | (
        np.isin(operator, (0, 5, 6)) & np.isclose(mixing_values, 1.0)
    )
    swap = commutative & (right < left)
    left_copy = left.copy()
    left[swap] = right[swap]
    right[swap] = left_copy[swap]
    same_unit = (left == right) & np.isclose(mixing_values, 1.0)
    zero = same_unit & np.isin(operator, (1, 4))
    unary = same_unit & np.isin(operator, (5, 6))
    kind[zero] = 0
    operator[zero] = 0
    mixing[zero] = 0
    left[zero] = 0
    right[zero] = 0
    kind[unary] = 1
    operator[unary] = 0
    mixing[unary] = 0
    right[unary] = 0
    signature = np.empty(
        len(left),
        dtype=np.dtype(
            [
                ("kind", "u1"),
                ("op", "u1"),
                ("mix", "u1"),
                ("left", "<u2"),
                ("right", "<u2"),
                ("a", "u1"),
                ("p", "u1"),
                ("ut", "u1"),
            ]
        ),
    )
    signature["kind"] = kind
    signature["op"] = operator
    signature["mix"] = mixing
    signature["left"] = left
    signature["right"] = right
    signature["a"] = decoded["amplitude_index"]
    signature["p"] = decoded["exponent_index"]
    signature["ut"] = decoded["transition_index"]
    _unique, first = np.unique(signature, return_index=True)
    representatives = np.sort(first)
    return representatives, {
        "input_candidates": len(left),
        "symbolic_equivalence_classes": len(representatives),
        "symbolic_duplicates_removed": len(left) - len(representatives),
        "zero_program_inputs": int(np.sum(zero)),
        "unary_collapses": int(np.sum(unary)),
        "commutative_swaps": int(np.sum(swap)),
    }


def _unary(values: np.ndarray, index: np.ndarray) -> np.ndarray:
    result = np.empty_like(values)
    for transform in range(8):
        mask = index == transform
        if not np.any(mask):
            continue
        x = values[mask]
        if transform == 0:
            result[mask] = x
        elif transform == 1:
            result[mask] = -x
        elif transform == 2:
            result[mask] = np.abs(x)
        elif transform == 3:
            result[mask] = np.sign(x) * np.square(x)
        elif transform == 4:
            result[mask] = np.sign(x) * np.sqrt(np.abs(x))
        elif transform == 5:
            result[mask] = np.tanh(2.0 * x)
        elif transform == 6:
            result[mask] = x / (0.25 + np.abs(x))
        else:
            result[mask] = np.sin(np.pi * x)
    return result


def _binary(
    left: np.ndarray, right: np.ndarray, operator: np.ndarray, mixing: np.ndarray
) -> np.ndarray:
    result = np.empty_like(left)
    weighted = mixing[:, None] * right
    for op in range(8):
        mask = operator == op
        if not np.any(mask):
            continue
        a = left[mask]
        b = weighted[mask]
        if op == 0:
            result[mask] = a + b
        elif op == 1:
            result[mask] = a - b
        elif op == 2:
            result[mask] = a * b
        elif op == 3:
            result[mask] = a * np.tanh(b)
        elif op == 4:
            result[mask] = (a - b) / (1.0 + np.abs(a) + np.abs(b))
        elif op == 5:
            result[mask] = np.maximum(a, b)
        elif op == 6:
            result[mask] = np.minimum(a, b)
        else:
            product = a * b
            result[mask] = np.sign(product) * np.sqrt(np.abs(product))
    return result


def program_log_multiplier(
    decoded: Mapping[str, np.ndarray],
    primitive_bank: np.ndarray,
    u: np.ndarray,
    config: Mapping[str, Any],
) -> np.ndarray:
    left_raw = 2.0 * primitive_bank[np.asarray(decoded["left_primitive_index"], int)] - 1.0
    right_raw = 2.0 * primitive_bank[np.asarray(decoded["right_primitive_index"], int)] - 1.0
    left = _unary(left_raw, np.asarray(decoded["left_transform_index"], int))
    right = _unary(right_raw, np.asarray(decoded["right_transform_index"], int))
    mixing = np.asarray(config["program_grammar"]["mixing_grid"], dtype=float)[
        np.asarray(decoded["mixing_index"], int)
    ]
    z = _binary(left, right, np.asarray(decoded["operator_index"], int), mixing)
    coordinate = 0.5 + 0.5 * z / (1.0 + np.abs(z))
    grids = config["program_grammar"]["outer_parameter_grids"]
    amplitude = np.asarray(grids["amplitude"], dtype=float)[
        np.asarray(decoded["amplitude_index"], int)
    ]
    exponent = np.asarray(grids["acceleration_exponent"], dtype=float)[
        np.asarray(decoded["exponent_index"], int)
    ]
    transition = np.asarray(grids["transition_u"], dtype=float)[
        np.asarray(decoded["transition_index"], int)
    ]
    multiplier = 1.0 + amplitude[:, None] * np.power(
        np.asarray(u, dtype=float)[None, :], -exponent[:, None]
    ) / (1.0 + np.asarray(u, dtype=float)[None, :] / transition[:, None]) * (
        0.05 + 0.95 * coordinate
    )
    if np.any(multiplier <= 0.0) or not np.all(np.isfinite(multiplier)):
        raise GravityItem49Error("admitted program produced invalid multiplier")
    return np.log10(multiplier)


def _behavioral_representatives(
    decoded: Mapping[str, np.ndarray],
    primitive_bank: np.ndarray,
    u: np.ndarray,
    config: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    batch_size = int(config["evaluation"]["behavior_batch_size"])
    behavior = np.empty((len(decoded["ordinal"]), len(u)), dtype=np.float64)
    for begin in range(0, len(behavior), batch_size):
        end = min(begin + batch_size, len(behavior))
        rows = _select_rows(decoded, np.arange(begin, end))
        behavior[begin:end] = program_log_multiplier(rows, primitive_bank, u, config)
    rounded = np.ascontiguousarray(
        np.round(behavior, int(config["admissibility"]["behavior_signature_decimals"]))
    )
    byte_rows = rounded.view(np.dtype((np.void, rounded.dtype.itemsize * rounded.shape[1]))).ravel()
    _unique, first = np.unique(byte_rows, return_index=True)
    representatives = np.sort(first)
    selected = _select_rows(decoded, representatives)
    selected_behavior = behavior[representatives]
    return selected, selected_behavior, {
        "input_symbolic_classes": len(behavior),
        "behavioral_equivalence_classes": len(representatives),
        "behavioral_duplicates_removed": len(behavior) - len(representatives),
        "behavior_matrix_sha256": hashlib.sha256(
            np.round(selected_behavior, 10).astype("<f8").tobytes()
        ).hexdigest(),
    }


def build_lane_programs(
    root: Path, config: Mapping[str, Any], lane: str
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    ordinals = _lane_ordinals(config, lane)
    if len(np.unique(ordinals)) != len(ordinals):
        raise GravityItem49Error("schedule contains a collision")
    decoded = decode_ordinals(ordinals, config)
    physical = _physical_mask(decoded, config)
    admitted = _select_rows(decoded, np.flatnonzero(physical))
    symbolic_indices, symbolic_audit = _symbolic_representatives(admitted, config)
    symbolic = _select_rows(admitted, symbolic_indices)
    sources = _primitive_sources(root)
    bank, _audit = primitive_bank_from_sources(sources)
    u = _response_blind_u(root, sources)
    behavior_programs, behavior, behavior_audit = _behavioral_representatives(
        symbolic, bank, u, config
    )
    audit = {
        "lane": lane,
        "raw_schedule_positions": len(ordinals),
        "sampled_ordinals_unique": len(np.unique(ordinals)) == len(ordinals),
        "sampled_ordinal_sha256": hashlib.sha256(ordinals.astype("<u8").tobytes()).hexdigest(),
        "first_sampled_ordinals": [int(value) for value in ordinals[:16]],
        "minimum_sampled_ordinal": int(np.min(ordinals)),
        "maximum_sampled_ordinal": int(np.max(ordinals)),
        "physically_admitted_cells": int(np.sum(physical)),
        "physically_rejected_cells": int(np.sum(~physical)),
        **symbolic_audit,
        **behavior_audit,
        "programs_eligible_for_response_scoring": len(behavior_programs["ordinal"]),
        "behavior_representative_ordinal_sha256": hashlib.sha256(
            np.asarray(behavior_programs["ordinal"], dtype="<u8").tobytes()
        ).hexdigest(),
    }
    return behavior_programs, behavior, audit


def build_candidate_manifest(root: Path) -> dict[str, Any]:
    config = load_config(root)
    lane_audits: dict[str, Any] = {}
    for lane in ("pseudorandom", "sequential_ordinal_control"):
        _programs, _behavior, lane_audits[lane] = build_lane_programs(
            root, config, lane
        )
    random_ordinals = _lane_ordinals(config, "pseudorandom")
    sequential_ordinals = _lane_ordinals(config, "sequential_ordinal_control")
    overlap = np.intersect1d(
        random_ordinals, sequential_ordinals, assume_unique=True
    )
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item49-candidate-manifest-1.0",
            "item": 49,
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "config_contract_sha256": _contract_digest(config),
            "response_fields_read_during_program_generation": [],
            "response_values_used_during_program_generation": 0,
            "confirmation_accessed": False,
            "paid_model_calls": 0,
            "full_ordinal_space": int(config["program_grammar"]["full_ordinal_space"]),
            "full_ordinal_space_exhausted": False,
            "trillion_formula_campaign_executed": False,
            "lane_audits": lane_audits,
            "cross_lane_raw_ordinal_overlap": {
                "count": len(overlap),
                "ordinals": [int(value) for value in overlap],
            },
            "total_raw_schedule_positions": sum(
                int(audit["raw_schedule_positions"])
                for audit in lane_audits.values()
            ),
            "total_programs_eligible_for_response_scoring": sum(
                int(audit["programs_eligible_for_response_scoring"])
                for audit in lane_audits.values()
            ),
            "claim_boundaries": [
                "The scheduler can address 6,496,138,035,200 program ordinals; this run samples 1,048,576 ordinals per lane without replacement.",
                "The sequential lane receives the same raw ordinal budget as the pseudorandom lane.",
                "Symbolic and predictor-behavior equivalence are computed without response fields.",
                "A predictor-behavior equivalence class is specific to these 112 development predictor rows and is not a global algebraic identity.",
                "No sampled combination is labeled historically novel without a separate prior-art review.",
                "Neither a single empirical mismatch nor a counterexample count prunes a formula family.",
            ],
        }
    )


def build_exposure_manifest(root: Path) -> dict[str, Any]:
    config = load_config(root)
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item49-exposure-manifest-1.0",
            "item": 49,
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "datasets": [
                {
                    "id": "S4TM_ITEM43_EXPLORATION",
                    "objects": 28,
                    "response_status": "already exposed",
                    "role": "retrospective pseudorandom program development",
                },
                {
                    "id": "CLASH_ACCELERATION",
                    "objects": 20,
                    "points": 84,
                    "response_status": "already exposed",
                    "role": "retrospective pseudorandom program development",
                },
            ],
            "sealed_data": {
                "item43_s4tm_confirmation_lenses": 7,
                "access_authorized": False,
                "response_rows_read": 0,
            },
            "rules": [
                "freeze both equal-budget schedules and all equivalence rules before outcome scoring",
                "build the primitive bank and program behavior signatures without a response field",
                "do not describe retrospective grouped validation as fresh confirmation",
                "preserve all mismatches and apply the global data-quality-aware counterexample policy",
            ],
        }
    )


def write_freeze_manifests(root: Path) -> list[Path]:
    config = load_config(root)
    paths = [
        _source_path(root, config, "candidate_manifest"),
        _source_path(root, config, "primitive_receipt"),
        _source_path(root, config, "exposure_manifest"),
    ]
    _write_json(paths[0], build_candidate_manifest(root))
    _write_json(paths[1], build_primitive_receipt(root))
    _write_json(paths[2], build_exposure_manifest(root))
    return paths


def _best_behavior(
    behavior: np.ndarray,
    arrays: Mapping[str, Any],
    train: np.ndarray,
    config: Mapping[str, Any],
) -> tuple[int, float, str, int]:
    weights_np = _object_weights(arrays, train)
    indices = np.flatnonzero(train)
    backend = "numpy_cpu"
    xp: Any = np
    try:
        import cupy as cp

        if int(cp.cuda.runtime.getDeviceCount()) > 0:
            xp = cp
            name = cp.cuda.runtime.getDeviceProperties(0)["name"]
            backend = "cupy_cuda_" + (
                name.decode() if isinstance(name, bytes) else str(name)
            )
    except Exception:
        xp = np
    residual = xp.asarray(arrays["target"][indices] - arrays["base"][indices])
    sigma = xp.asarray(arrays["sigma"][indices])
    weights = xp.asarray(weights_np[indices])
    best_loss = math.inf
    best_row = -1
    batch_size = int(config["evaluation"]["candidate_batch_size"])
    for begin in range(0, len(behavior), batch_size):
        end = min(begin + batch_size, len(behavior))
        candidate = xp.asarray(behavior[begin:end, indices])
        errors = xp.square((candidate - residual[None, :]) / sigma[None, :])
        losses = xp.sum(errors * weights[None, :], axis=1)
        local = int(xp.argmin(losses).item())
        loss = float(losses[local].item())
        if loss < best_loss:
            best_loss = loss
            best_row = begin + local
    if best_row < 0:
        raise GravityItem49Error("no behavior program was evaluated")
    return best_row, best_loss, backend, len(behavior) * len(indices)


def _program_description(
    programs: Mapping[str, np.ndarray],
    row: int,
    labels: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    grammar = config["program_grammar"]
    left_id = int(programs["left_primitive_index"][row])
    right_id = int(programs["right_primitive_index"][row])
    left = labels[left_id]
    right = labels[right_id]
    left_transform = grammar["unary_transforms"][
        int(programs["left_transform_index"][row])
    ]
    right_transform = grammar["unary_transforms"][
        int(programs["right_transform_index"][row])
    ]
    operator = grammar["binary_operators"][int(programs["operator_index"][row])]
    mixing = float(grammar["mixing_grid"][int(programs["mixing_index"][row])])
    grids = grammar["outer_parameter_grids"]
    source_items = sorted({int(left["source_item"]), int(right["source_item"])})
    creativity = (
        "potentially_new_cross_mechanism_combination"
        if len(source_items) > 1
        else "rewrite_or_recombination_within_a_known_generated_mechanism"
    )
    expression = (
        f"{operator}({left_transform}({left['expression']}), "
        f"{mixing:g}*{right_transform}({right['expression']}))"
    )
    return {
        "behavior_class_index": row,
        "ordinal": int(programs["ordinal"][row]),
        "left_primitive": left,
        "right_primitive": right,
        "left_transform": left_transform,
        "right_transform": right_transform,
        "binary_operator": operator,
        "mixing": mixing,
        "amplitude": float(
            grids["amplitude"][int(programs["amplitude_index"][row])]
        ),
        "acceleration_exponent": float(
            grids["acceleration_exponent"][
                int(programs["exponent_index"][row])
            ]
        ),
        "transition_u": float(
            grids["transition_u"][int(programs["transition_index"][row])]
        ),
        "coordinate_expression": expression,
        "source_items": source_items,
        "creativity_label": creativity,
        "historical_novelty_claimed": False,
    }


def _fixed_behavior_oof(
    fold_rows: Mapping[int, int],
    behavior: np.ndarray,
    arrays: Mapping[str, Any],
) -> np.ndarray:
    prediction = np.empty(len(arrays["target"]), dtype=float)
    for fold, row in fold_rows.items():
        test = arrays["fold"] == fold
        prediction[test] = arrays["base"][test] + behavior[row, test]
    return prediction


def _item47_oof(
    root: Path, arrays: Mapping[str, Any]
) -> tuple[np.ndarray, dict[int, int]]:
    config47 = _load_item47_config(root)
    evaluation = _read_json(root / ITEM47_EVALUATION_PATH)
    fold_ids = {
        int(row["fold"]): int(row["selected_operator"]["candidate_id"])
        for row in evaluation["fold_ledger"]
    }
    prediction = np.empty(len(arrays["target"]), dtype=float)
    for fold, candidate_id in fold_ids.items():
        test = arrays["fold"] == fold
        prediction[test] = _item47_predict(
            candidate_id, arrays, config47, bank_key="operator_bank"
        )[test]
    return prediction, fold_ids


def _item48_oof(
    root: Path, arrays: Mapping[str, Any]
) -> tuple[np.ndarray, dict[int, int]]:
    config48 = _load_item48_config(root)
    evaluation = _read_json(root / ITEM48_EVALUATION_PATH)
    fold_ids = {
        int(row["fold"]): int(row["selected_action"]["candidate_id"])
        for row in evaluation["fold_ledger"]
    }
    return (
        _item48_fixed_oof(fold_ids, arrays, config48, "action_bank"),
        fold_ids,
    )


def _program_behavior(
    programs: Mapping[str, np.ndarray],
    primitive_bank: np.ndarray,
    u: np.ndarray,
    config: Mapping[str, Any],
) -> np.ndarray:
    batch_size = int(config["evaluation"]["behavior_batch_size"])
    behavior = np.empty((len(programs["ordinal"]), len(u)), dtype=float)
    for begin in range(0, len(behavior), batch_size):
        end = min(begin + batch_size, len(behavior))
        behavior[begin:end] = program_log_multiplier(
            _select_rows(programs, np.arange(begin, end)),
            primitive_bank,
            u,
            config,
        )
    return behavior


def _primitive_bank_from_arrays(arrays: Mapping[str, Any]) -> np.ndarray:
    bank = np.vstack(
        (
            np.asarray(arrays["interaction_bank"], dtype=float),
            np.asarray(arrays["pi_bank"], dtype=float),
            np.asarray(arrays["operator_bank"], dtype=float),
            np.asarray(arrays["action_bank"], dtype=float),
        )
    )
    if bank.shape != (440, len(arrays["target"])):
        raise GravityItem49Error("evaluation primitive bank shape changed")
    return bank


def build_evaluation_result(root: Path) -> dict[str, Any]:
    config = load_config(root)
    config48 = _load_item48_config(root)
    arrays = _item48_evaluation_arrays(root, config48)
    labels = primitive_labels(_primitive_sources(root))
    lanes = ("pseudorandom", "sequential_ordinal_control")
    programs: dict[str, dict[str, np.ndarray]] = {}
    behaviors: dict[str, np.ndarray] = {}
    lane_audits: dict[str, Any] = {}
    for lane in lanes:
        programs[lane], behaviors[lane], lane_audits[lane] = build_lane_programs(
            root, config, lane
        )

    fold_rows: dict[str, dict[int, int]] = {lane: {} for lane in lanes}
    oof: dict[str, np.ndarray] = {
        lane: np.empty(len(arrays["target"]), dtype=float) for lane in lanes
    }
    ledger: list[dict[str, Any]] = []
    backends: set[str] = set()
    evaluations_by_lane = {lane: 0 for lane in lanes}
    for fold in range(int(config["evaluation"]["outer_folds"])):
        train = arrays["fold"] != fold
        test = ~train
        selected: dict[str, Any] = {}
        for lane in lanes:
            row, loss, backend, count = _best_behavior(
                behaviors[lane], arrays, train, config
            )
            fold_rows[lane][fold] = row
            oof[lane][test] = arrays["base"][test] + behaviors[lane][row, test]
            backends.add(backend)
            evaluations_by_lane[lane] += count
            selected[lane] = {
                "program": _program_description(
                    programs[lane], row, labels, config
                ),
                "training_balanced_loss": loss,
            }
        ledger.append(
            {
                "fold": fold,
                "selected_pseudorandom_program": selected["pseudorandom"]["program"],
                "pseudorandom_training_balanced_loss": selected["pseudorandom"][
                    "training_balanced_loss"
                ],
                "selected_sequential_program": selected[
                    "sequential_ordinal_control"
                ]["program"],
                "sequential_training_balanced_loss": selected[
                    "sequential_ordinal_control"
                ]["training_balanced_loss"],
                "heldout_s4tm_objects": sorted(
                    set(
                        arrays["object"][
                            test & (arrays["population"] == "S4TM")
                        ].tolist()
                    )
                ),
                "heldout_clash_objects": sorted(
                    set(
                        arrays["object"][
                            test & (arrays["population"] == "CLASH")
                        ].tolist()
                    )
                ),
            }
        )

    all_rows = np.ones(len(arrays["target"]), dtype=bool)
    selected_rows: dict[str, int] = {}
    selected_losses: dict[str, float] = {}
    cpu_gpu_differences: dict[str, float] = {}
    for lane in lanes:
        row, loss, backend, count = _best_behavior(
            behaviors[lane], arrays, all_rows, config
        )
        selected_rows[lane] = row
        selected_losses[lane] = loss
        evaluations_by_lane[lane] += count
        backends.add(backend)
        cpu_loss = _score(
            arrays, arrays["base"] + behaviors[lane][row]
        )["balanced_loss"]
        cpu_gpu_differences[lane] = abs(float(cpu_loss) - loss)
        if cpu_gpu_differences[lane] > float(
            config["evaluation"]["cpu_gpu_tolerance"]
        ):
            raise GravityItem49Error(f"CPU/GPU loss cross-check failed for {lane}")

    item44_oof, fold_item44 = _item44_oof(root, arrays)
    item45_oof, fold_item45 = _item45_oof(root, arrays)
    item46_oof, fold_item46 = _item46_oof(root, arrays)
    item47_oof, fold_item47 = _item47_oof(root, arrays)
    item48_oof, fold_item48 = _item48_oof(root, arrays)
    scores = {
        "pseudorandom_program_search": _score(arrays, oof["pseudorandom"]),
        "sequential_ordinal_control": _score(
            arrays, oof["sequential_ordinal_control"]
        ),
        "item48_action_generator": _score(arrays, item48_oof),
        "item47_operator_generator": _score(arrays, item47_oof),
        "item46_dimensionless_generator": _score(arrays, item46_oof),
        "item45_universal_interaction": _score(arrays, item45_oof),
        "item44_scale_hierarchy": _score(arrays, item44_oof),
        "baryonic_newton": _score(arrays, arrays["base"]),
        "mond_rar": _score(
            arrays,
            arrays["base"]
            + np.log10(1.0 / (1.0 - np.exp(-np.sqrt(arrays["u"])))),
        ),
        "ordinary_ridge": _score(arrays, _ordinary_crossfit(arrays, config)),
    }
    controls = tuple(name for name in scores if name != "pseudorandom_program_search")
    strongest = min(controls, key=lambda name: scores[name]["balanced_loss"])
    candidate_objects = scores["pseudorandom_program_search"]["object_losses"]
    control_objects = scores[strongest]["object_losses"]
    object_keys = sorted(candidate_objects)
    diff = np.asarray(
        [control_objects[key] - candidate_objects[key] for key in object_keys]
    )
    raw_counterexample = diff < 0.0
    stable_counterexample = raw_counterexample.copy()

    config44 = _read_json(root / "configs/gravity_item44_scale_hierarchy_v1.json")
    config45 = _load_item45_config(root)
    config46 = _load_item46_config(root)
    config47 = _load_item47_config(root)
    shapes = _shape_by_object(root, arrays)
    systematic_scores: dict[str, Any] = {}
    for variant_name, population, shift in config["evaluation"]["mass_scale_variants"]:
        varied = _item45_variant_arrays(
            arrays, str(population), float(shift), config45
        )
        varied["pi_bank"] = (
            1.0
            / (
                1.0
                + np.abs(
                    _item46_physical_log_values(varied, config46)
                    @ np.asarray(_item46_pi_vectors(config46), dtype=float).T
                )
            )
        ).T
        varied["operator_bank"] = _item47_operator_bank_from_arrays(
            varied, shapes, config47
        )[1].T
        varied["action_bank"] = _item48_action_bank_from_arrays(
            varied, config48
        )[1].T
        varied_bank = _primitive_bank_from_arrays(varied)
        varied_behavior = {
            lane: _program_behavior(
                programs[lane], varied_bank, np.asarray(varied["u"]), config
            )
            for lane in lanes
        }
        variant_predictions = {
            "pseudorandom_program_search": _fixed_behavior_oof(
                fold_rows["pseudorandom"], varied_behavior["pseudorandom"], varied
            ),
            "sequential_ordinal_control": _fixed_behavior_oof(
                fold_rows["sequential_ordinal_control"],
                varied_behavior["sequential_ordinal_control"],
                varied,
            ),
            "item48_action_generator": _item48_fixed_oof(
                fold_item48, varied, config48, "action_bank"
            ),
        }
        item44_variant = np.empty(len(varied["target"]), dtype=float)
        item45_variant = np.empty(len(varied["target"]), dtype=float)
        item46_variant = np.empty(len(varied["target"]), dtype=float)
        item47_variant = np.empty(len(varied["target"]), dtype=float)
        for fold in range(int(config["evaluation"]["outer_folds"])):
            test = varied["fold"] == fold
            item44_variant[test] = _item44_predict(
                fold_item44[fold], varied, config44
            )[test]
            item45_variant[test] = _item45_predict(
                fold_item45[fold], varied, config45, bank_key="interaction_bank"
            )[test]
            item46_variant[test] = _item46_predict(
                fold_item46[fold], varied, config46, bank_key="pi_bank"
            )[test]
            item47_variant[test] = _item47_predict(
                fold_item47[fold], varied, config47, bank_key="operator_bank"
            )[test]
        variant_predictions.update(
            {
                "item47_operator_generator": item47_variant,
                "item46_dimensionless_generator": item46_variant,
                "item45_universal_interaction": item45_variant,
                "item44_scale_hierarchy": item44_variant,
                "baryonic_newton": varied["base"],
                "mond_rar": varied["base"]
                + np.log10(
                    1.0 / (1.0 - np.exp(-np.sqrt(varied["u"])))
                ),
                "ordinary_ridge": _ordinary_crossfit(varied, config),
            }
        )
        variants = {
            name: _score(varied, prediction)
            for name, prediction in variant_predictions.items()
        }
        systematic_scores[str(variant_name)] = {
            "pseudorandom_program_search": variants[
                "pseudorandom_program_search"
            ],
            "sequential_ordinal_control": variants["sequential_ordinal_control"],
            "item45_primary_control": variants["item45_universal_interaction"],
            "strongest_control_name": strongest,
            "strongest_control": variants[strongest],
        }
        for index, key in enumerate(object_keys):
            stable_counterexample[index] &= (
                variants["pseudorandom_program_search"]["object_losses"][key]
                > variants[strongest]["object_losses"][key]
            )

    leave_one = [
        float(np.mean(np.delete(diff, index))) for index in range(len(diff))
    ]
    trim_count = max(
        1,
        int(len(diff) * float(config["evaluation"]["robust_trim_fraction"])),
    )
    trimmed = np.sort(diff)[trim_count:-trim_count]
    candidate_score = scores["pseudorandom_program_search"]["balanced_loss"]
    improvement = 100.0 * (
        scores[strongest]["balanced_loss"] - candidate_score
    ) / scores[strongest]["balanced_loss"]
    improvement_item45 = 100.0 * (
        scores["item45_universal_interaction"]["balanced_loss"] - candidate_score
    ) / scores["item45_universal_interaction"]["balanced_loss"]
    improvement_sequential = 100.0 * (
        scores["sequential_ordinal_control"]["balanced_loss"] - candidate_score
    ) / scores["sequential_ordinal_control"]["balanced_loss"]
    policy_report = {
        "evidence_kind": "empirical",
        "evaluable_objects": len(object_keys),
        "raw_counterexample_count": int(np.sum(raw_counterexample)),
        "quality_verified_counterexample_count": int(np.sum(raw_counterexample)),
        "uncertainty_resolved_counterexample_count": int(
            np.sum(stable_counterexample)
        ),
        "independent_failure_strata": 0,
        "unchanged_independent_replication_failures": 0,
        "aggregate_improvement_percent": improvement,
        "quality_gate_passed": False,
        "strongest_baseline_failed": bool(improvement <= 0.0),
        "leave_one_changes_sign": bool(
            (min(leave_one) <= 0.0) != (float(np.mean(diff)) <= 0.0)
        ),
        "trim_changes_sign": bool(
            (float(np.mean(trimmed)) <= 0.0) != (float(np.mean(diff)) <= 0.0)
        ),
        "object_level_records_preserved": True,
        "missing_quality_limited_records_preserved": True,
        "exclusions_frozen_before_response": True,
    }
    policy = assess_counterexample_evidence(
        policy_report, load_counterexample_policy(root / POLICY_PATH)
    )
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item49-joint-evaluation-1.0",
            "item": 49,
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "selected_pseudorandom_program": _program_description(
                programs["pseudorandom"],
                selected_rows["pseudorandom"],
                labels,
                config,
            ),
            "selected_pseudorandom_full_data_balanced_training_loss": selected_losses[
                "pseudorandom"
            ],
            "selected_sequential_program": _program_description(
                programs["sequential_ordinal_control"],
                selected_rows["sequential_ordinal_control"],
                labels,
                config,
            ),
            "selected_sequential_full_data_balanced_training_loss": selected_losses[
                "sequential_ordinal_control"
            ],
            "fold_ledger": ledger,
            "scores": scores,
            "strongest_control": strongest,
            "aggregate_improvement_percent": improvement,
            "improvement_over_item45_percent": improvement_item45,
            "improvement_over_equal_budget_sequential_percent": improvement_sequential,
            "paired_sign_flip_p": _paired_p(diff, config),
            "robustness": {
                "leave_one_min_mean_control_minus_candidate_loss": min(leave_one),
                "leave_one_max_mean_control_minus_candidate_loss": max(leave_one),
                "trimmed_mean_control_minus_candidate_loss": float(np.mean(trimmed)),
            },
            "counterexamples": [
                {
                    "object": key,
                    "raw_counterexample": bool(raw_counterexample[index]),
                    "mass_variant_stable_counterexample": bool(
                        stable_counterexample[index]
                    ),
                }
                for index, key in enumerate(object_keys)
            ],
            "systematic_scores": systematic_scores,
            "counterexample_policy_report": policy_report,
            "counterexample_policy_assessment": policy,
            "compute": {
                "backends": sorted(backends),
                "program_point_fold_evaluations_by_lane": evaluations_by_lane,
                "program_point_fold_evaluations": sum(
                    evaluations_by_lane.values()
                ),
                "cpu_gpu_selected_loss_absolute_difference": cpu_gpu_differences,
                "lane_audits": lane_audits,
            },
            "counts": {
                "raw_schedule_positions": sum(
                    audit["raw_schedule_positions"] for audit in lane_audits.values()
                ),
                "physically_admitted_programs": sum(
                    audit["physically_admitted_cells"] for audit in lane_audits.values()
                ),
                "outcome_scored_behavior_classes": sum(
                    audit["programs_eligible_for_response_scoring"]
                    for audit in lane_audits.values()
                ),
                "s4tm_lenses": 28,
                "clash_clusters": 20,
                "clash_points": 84,
                "sealed_confirmation_rows": 0,
                "post_evaluation_candidate_cells": 0,
                "paid_model_calls": 0,
            },
            "limitations": [
                "All response rows were exposed before Item 49; grouped cross-validation is retrospective development, not fresh confirmation.",
                "Only 2,097,152 raw ordinals were sampled from a 6,496,138,035,200-ordinal grammar; the grammar was not exhausted and no trillion-program evaluation was run.",
                "Behavioral equivalence is defined on 112 development predictor rows and may merge programs that differ elsewhere.",
                "The grammar recombines known and earlier generated ingredients; historical novelty is not established by a new sampled expression.",
                "S4TM uses an analytic projected stellar profile without measured gas; CLASH uses model-dependent published acceleration profiles.",
                "Four global baryonic-mass shifts do not exhaust measurement, geometry, selection, or lens-model uncertainty.",
                "Empirical mismatches are retained as evidence; neither one counterexample nor their count prunes a formula family.",
            ],
        }
    )


def write_evaluation_result(root: Path) -> Path:
    config = load_config(root)
    path = _source_path(root, config, "evaluation_result")
    _write_json(path, build_evaluation_result(root))
    return path
