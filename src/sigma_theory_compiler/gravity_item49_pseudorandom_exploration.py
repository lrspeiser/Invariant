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
