"""Item 51 measured RTX GPU screening over a frozen formula stream."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
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
from sigma_theory_compiler.gravity_item45_universal_interactions import (
    _object_weights,
    _paired_p,
    _score,
    _variant_arrays as _item45_variant_arrays,
    load_config as _load_item45_config,
)
from sigma_theory_compiler.gravity_item46_dimensionless_generator import (
    _physical_log_values as _item46_physical_log_values,
    load_config as _load_item46_config,
    pi_vectors as _item46_pi_vectors,
)
from sigma_theory_compiler.gravity_item47_operator_generator import (
    _shape_by_object,
    load_config as _load_item47_config,
    operator_bank_from_arrays as _item47_operator_bank_from_arrays,
)
from sigma_theory_compiler.gravity_item48_action_generator import (
    _evaluation_arrays as _item48_evaluation_arrays,
    action_bank_from_arrays as _item48_action_bank_from_arrays,
    load_config as _load_item48_config,
)
from sigma_theory_compiler.gravity_item49_pseudorandom_exploration import (
    _admissible_parameter_table,
    _item44_oof,
    _item45_oof,
    _item46_oof,
    _item47_oof,
    _primitive_bank_from_arrays,
    _primitive_sources,
    _program_description,
    decode_ordinals,
    load_config as _load_item49_config,
    primitive_labels,
    program_log_multiplier,
)


CONFIG_PATH = Path("configs/gravity_item51_gpu_screening_v1.json")
GOAL_PATH = Path("docs/GRAVITY_HIDDEN_VARIABLE_AND_THEORY_SEARCH_GOALS.md")
POLICY_PATH = Path("configs/gravity_empirical_counterexample_policy_v1.json")
ITEM49_EVALUATION_PATH = Path(
    "runs/gravity/roadmap/item-49-pseudorandom-exploration-v1-source/joint-evaluation-result.json"
)


class GravityItem51Error(RuntimeError):
    """Raised when the Item 51 frozen schedule or GPU evidence contract fails."""


def load_config(root: Path, *, require_bound: bool = True) -> dict[str, Any]:
    config = _read_json(root / CONFIG_PATH)
    validate_config(root, config, require_bound=require_bound)
    return config


def validate_config(
    root: Path, config: Mapping[str, Any], *, require_bound: bool = True
) -> None:
    if (
        config.get("schema_version")
        != "invariant-gravity-item51-gpu-screening-config-1.0"
        or int(config.get("item", -1)) != 51
    ):
        raise GravityItem51Error("unexpected Item 51 config")
    if _sha256_file(root / GOAL_PATH) != str(config["stable_goal_sha256"]):
        raise GravityItem51Error("stable gravity goal changed")
    freeze = str(config["scientific_freeze_commit"])
    if require_bound and re.fullmatch(r"[0-9a-f]{40}", freeze) is None:
        raise GravityItem51Error("Item 51 scientific freeze is not commit-bound")
    if not require_bound and not (
        freeze == "PENDING_FREEZE_COMMIT" or re.fullmatch(r"[0-9a-f]{40}", freeze)
    ):
        raise GravityItem51Error("malformed Item 51 freeze binding")
    for relative, expected in config["scientific_dependencies"].items():
        if _sha256_file(root / str(relative)) != str(expected):
            raise GravityItem51Error(f"scientific dependency changed: {relative}")
    for item in (49, 50):
        filename = (
            "item-49-pseudorandom-exploration-v1.json"
            if item == 49
            else "item-50-llm-creativity-v1.json"
        )
        receipt = _read_json(root / "runs/gravity/roadmap" / filename)
        required = config["required_predecessors"]
        if receipt["decision"] != required[f"item{item}_decision"]:
            raise GravityItem51Error(f"Item {item} decision binding changed")
        if receipt["content_sha256"] != required[f"item{item}_content_sha256"]:
            raise GravityItem51Error(f"Item {item} content binding changed")
    schedule = config["schedule"]
    space = int(config["program_grammar_binding"]["full_ordinal_space"])
    count = int(schedule["sample_positions"])
    stride = int(schedule["coprime_stride"])
    if space != 6_496_138_035_200 or count != 67_108_864:
        raise GravityItem51Error("frozen Item 51 search size changed")
    if math.gcd(stride, space) != 1 or count > space:
        raise GravityItem51Error("affine schedule is not collision-free")
    if int(schedule["start_offset"]) + stride * (count - 1) >= 2**64:
        raise GravityItem51Error("affine schedule would overflow uint64")
    policy = config["discovery_policy"]
    if not policy["schedule_frozen_before_outcome_screen"]:
        raise GravityItem51Error("outcome schedule is not frozen")
    if not policy["single_empirical_counterexample_is_not_a_formula_or_family_veto"]:
        raise GravityItem51Error("one mismatch became a veto")
    if not policy["counterexample_count_alone_is_never_decisive"]:
        raise GravityItem51Error("counterexample counts became a veto")
    if policy["finite_empirical_sample_may_prune_family"]:
        raise GravityItem51Error("finite empirical family pruning entered Item 51")
    if config["scope"]["trillion_formula_campaign_executed"]:
        raise GravityItem51Error("Item 51 was mislabeled a trillion-formula campaign")
    if config["scope"]["fresh_confirmation_claim_allowed"]:
        raise GravityItem51Error("fresh confirmation entered retrospective Item 51")


def _contract_digest(config: Mapping[str, Any]) -> str:
    value = json.loads(json.dumps(config))
    value["scientific_freeze_commit"] = "<BOUND_COMMIT>"
    return _sha256_bytes(_canonical_bytes(value))


def _source_path(root: Path, config: Mapping[str, Any], key: str) -> Path:
    return root / str(config["paths"]["source_dir"]) / str(config["paths"][key])


def _schedule_ordinals(config: Mapping[str, Any], start: int, count: int) -> np.ndarray:
    schedule = config["schedule"]
    positions = np.arange(start, start + count, dtype=np.uint64)
    return np.asarray(
        (
            np.uint64(int(schedule["start_offset"]))
            + np.uint64(int(schedule["coprime_stride"])) * positions
        )
        % np.uint64(config["program_grammar_binding"]["full_ordinal_space"]),
        dtype=np.uint64,
    )


def _select_rows(decoded: Mapping[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    return {key: np.asarray(value)[mask] for key, value in decoded.items()}


def _physical_mask(decoded: Mapping[str, np.ndarray], config49: Mapping[str, Any]) -> np.ndarray:
    table = _admissible_parameter_table(config49)
    index = (
        np.asarray(decoded["amplitude_index"], int) * 256
        + np.asarray(decoded["exponent_index"], int) * 16
        + np.asarray(decoded["transition_index"], int)
    )
    return table[index]


def _canonical_symbolic_keys(decoded: Mapping[str, np.ndarray], config49: Mapping[str, Any]) -> np.ndarray:
    left = (
        np.asarray(decoded["left_primitive_index"], np.uint64) * 8
        + np.asarray(decoded["left_transform_index"], np.uint64)
    )
    right = (
        np.asarray(decoded["right_primitive_index"], np.uint64) * 8
        + np.asarray(decoded["right_transform_index"], np.uint64)
    )
    operator = np.asarray(decoded["operator_index"], np.uint64).copy()
    mixing = np.asarray(decoded["mixing_index"], np.uint64).copy()
    mixing_values = np.asarray(config49["program_grammar"]["mixing_grid"], float)[mixing]
    kind = np.full(len(left), 2, dtype=np.uint64)
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
    key = kind
    key |= operator << np.uint64(2)
    key |= mixing << np.uint64(5)
    key |= left << np.uint64(9)
    key |= right << np.uint64(21)
    key |= np.asarray(decoded["amplitude_index"], np.uint64) << np.uint64(33)
    key |= np.asarray(decoded["exponent_index"], np.uint64) << np.uint64(37)
    key |= np.asarray(decoded["transition_index"], np.uint64) << np.uint64(41)
    return key


def build_preflight_manifest(root: Path, *, live: bool = True) -> dict[str, Any]:
    config = load_config(root, require_bound=live)
    device: dict[str, Any] = {"available": False}
    if live:
        try:
            import cupy as cp

            count = int(cp.cuda.runtime.getDeviceCount())
            if count < 1:
                raise GravityItem51Error("no CUDA device available")
            props = cp.cuda.runtime.getDeviceProperties(0)
            name = props["name"]
            free, total = cp.cuda.runtime.memGetInfo()
            device = {
                "available": True,
                "device_count": count,
                "device_index": 0,
                "name": name.decode() if isinstance(name, bytes) else str(name),
                "compute_capability": f"{props['major']}.{props['minor']}",
                "total_memory_bytes": int(total),
                "free_memory_bytes_at_preflight": int(free),
                "cupy_version": cp.__version__,
            }
        except GravityItem51Error:
            raise
        except Exception as error:
            raise GravityItem51Error(f"CUDA preflight failed: {error}") from error
    schedule = config["schedule"]
    space = int(config["program_grammar_binding"]["full_ordinal_space"])
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item51-preflight-1.0",
            "item": 51,
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "config_contract_sha256": _contract_digest(config),
            "device": device,
            "schedule": {
                "algorithm": schedule["algorithm"],
                "sample_positions": int(schedule["sample_positions"]),
                "full_ordinal_space": space,
                "fraction_of_space_scheduled": int(schedule["sample_positions"]) / space,
                "stride_space_gcd": math.gcd(int(schedule["coprime_stride"]), space),
                "collision_free_by_affine_permutation_proof": True,
            },
            "response_fields_used_to_construct_schedule": [],
            "response_values_used_to_construct_schedule": 0,
            "sealed_confirmation_rows": 0,
            "paid_model_calls": 0,
            "claims": {
                "gpu_throughput_measured": False,
                "candidate_screen_executed": False,
                "full_grammar_exhausted": False,
                "trillion_formula_campaign_executed": False,
                "historical_novelty_established": False,
            },
        }
    )


def write_preflight_manifest(root: Path) -> Path:
    config = load_config(root)
    path = _source_path(root, config, "preflight_manifest")
    _write_json(path, build_preflight_manifest(root))
    return path


def build_candidate_manifest(root: Path) -> dict[str, Any]:
    config = load_config(root)
    config49 = _load_item49_config(root)
    schedule = config["schedule"]
    total = int(schedule["sample_positions"])
    batch_size = int(schedule["raw_batch_size"])
    raw_sha = hashlib.sha256()
    admitted_sha = hashlib.sha256()
    key_chunks: list[np.ndarray] = []
    admitted_count = 0
    first: list[int] = []
    minimum = None
    maximum = None
    for begin in range(0, total, batch_size):
        size = min(batch_size, total - begin)
        ordinals = _schedule_ordinals(config, begin, size)
        raw_sha.update(ordinals.astype("<u8", copy=False).tobytes())
        if not first:
            first = [int(value) for value in ordinals[:16]]
        local_min = int(np.min(ordinals))
        local_max = int(np.max(ordinals))
        minimum = local_min if minimum is None else min(minimum, local_min)
        maximum = local_max if maximum is None else max(maximum, local_max)
        decoded = decode_ordinals(ordinals, config49)
        mask = _physical_mask(decoded, config49)
        admitted = _select_rows(decoded, mask)
        admitted_ordinals = np.asarray(admitted["ordinal"], dtype="<u8")
        admitted_sha.update(admitted_ordinals.tobytes())
        admitted_count += len(admitted_ordinals)
        key_chunks.append(_canonical_symbolic_keys(admitted, config49))
    keys = np.concatenate(key_chunks)
    if len(keys) != admitted_count:
        raise GravityItem51Error("candidate manifest accounting changed")
    symbolic_classes = len(np.unique(keys))
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item51-candidate-manifest-1.0",
            "item": 51,
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "config_contract_sha256": _contract_digest(config),
            "schedule": {
                "algorithm": schedule["algorithm"],
                "raw_schedule_positions": total,
                "raw_schedule_sha256": raw_sha.hexdigest(),
                "first_sampled_ordinals": first,
                "minimum_sampled_ordinal": minimum,
                "maximum_sampled_ordinal": maximum,
                "collision_free_by_affine_permutation_proof": True,
            },
            "physically_admitted_candidates": admitted_count,
            "physically_rejected_candidates": total - admitted_count,
            "admitted_ordinal_sha256": admitted_sha.hexdigest(),
            "exact_symbolic_equivalence_classes": symbolic_classes,
            "symbolic_duplicates_present_in_scored_stream": admitted_count
            - symbolic_classes,
            "outcome_scored_candidates": admitted_count,
            "behavioral_equivalence_not_computed": True,
            "response_fields_used_during_generation_or_symbolic_equivalence": [],
            "response_values_used_during_generation_or_symbolic_equivalence": 0,
            "post_evaluation_candidate_cells": 0,
            "sealed_confirmation_rows": 0,
            "claims": {
                "full_grammar_exhausted": False,
                "trillion_formula_campaign_executed": False,
                "historical_novelty_established": False,
                "symbolic_class_count_exact_for_scheduled_candidates": True,
            },
        }
    )


def write_candidate_manifest(root: Path) -> Path:
    config = load_config(root)
    path = _source_path(root, config, "candidate_manifest")
    _write_json(path, build_candidate_manifest(root))
    return path


def _decode_gpu(ordinals: Any, cp: Any) -> dict[str, Any]:
    radices = (16, 16, 16, 16, 8, 8, 440, 8, 440)
    fields = (
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
    values = ordinals.copy()
    decoded: dict[str, Any] = {"ordinal": ordinals.copy()}
    for field, radix in zip(fields, radices, strict=True):
        digit = values % cp.uint64(radix)
        values = values // cp.uint64(radix)
        decoded[field] = digit.astype(cp.int16)
    return decoded


def _unary_gpu(values: Any, index: Any, cp: Any) -> Any:
    result = cp.empty_like(values)
    for transform in range(8):
        mask = index == transform
        x = values[mask]
        if transform == 0:
            result[mask] = x
        elif transform == 1:
            result[mask] = -x
        elif transform == 2:
            result[mask] = cp.abs(x)
        elif transform == 3:
            result[mask] = cp.sign(x) * cp.square(x)
        elif transform == 4:
            result[mask] = cp.sign(x) * cp.sqrt(cp.abs(x))
        elif transform == 5:
            result[mask] = cp.tanh(2.0 * x)
        elif transform == 6:
            result[mask] = x / (0.25 + cp.abs(x))
        else:
            result[mask] = cp.sin(cp.pi * x)
    return result


def _binary_gpu(left: Any, right: Any, operator: Any, mixing: Any, cp: Any) -> Any:
    result = cp.empty_like(left)
    weighted = mixing[:, None] * right
    for op in range(8):
        mask = operator == op
        a = left[mask]
        b = weighted[mask]
        if op == 0:
            result[mask] = a + b
        elif op == 1:
            result[mask] = a - b
        elif op == 2:
            result[mask] = a * b
        elif op == 3:
            result[mask] = a * cp.tanh(b)
        elif op == 4:
            result[mask] = (a - b) / (1.0 + cp.abs(a) + cp.abs(b))
        elif op == 5:
            result[mask] = cp.maximum(a, b)
        elif op == 6:
            result[mask] = cp.minimum(a, b)
        else:
            product = a * b
            result[mask] = cp.sign(product) * cp.sqrt(cp.abs(product))
    return result


def _gpu_context(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    try:
        import cupy as cp
    except Exception as error:
        raise GravityItem51Error(f"CuPy is required for Item 51: {error}") from error
    if int(cp.cuda.runtime.getDeviceCount()) < 1:
        raise GravityItem51Error("no CUDA device available")
    config48 = _load_item48_config(root)
    arrays = _item48_evaluation_arrays(root, config48)
    bank = _primitive_bank_from_arrays(arrays)
    config49 = _load_item49_config(root)
    selection_names = [f"fold_{fold}_train" for fold in range(5)] + ["full"]
    weight_rows = []
    for fold in range(5):
        weight_rows.append(_object_weights(arrays, arrays["fold"] != fold))
    weight_rows.append(_object_weights(arrays, np.ones(len(arrays["target"]), dtype=bool)))
    props = cp.cuda.runtime.getDeviceProperties(0)
    name = props["name"]
    return {
        "cp": cp,
        "arrays": arrays,
        "config49": config49,
        "primitive_bank": cp.asarray(bank),
        "u": cp.asarray(arrays["u"], dtype=cp.float64),
        "residual": cp.asarray(arrays["target"] - arrays["base"], dtype=cp.float64),
        "sigma": cp.asarray(arrays["sigma"], dtype=cp.float64),
        "weights": cp.asarray(np.vstack(weight_rows).T, dtype=cp.float64),
        "selection_names": selection_names,
        "physical_table": cp.asarray(_admissible_parameter_table(config49)),
        "mixing_grid": cp.asarray(config49["program_grammar"]["mixing_grid"], dtype=cp.float64),
        "amplitude_grid": cp.asarray(
            config49["program_grammar"]["outer_parameter_grids"]["amplitude"],
            dtype=cp.float64,
        ),
        "exponent_grid": cp.asarray(
            config49["program_grammar"]["outer_parameter_grids"]["acceleration_exponent"],
            dtype=cp.float64,
        ),
        "transition_grid": cp.asarray(
            config49["program_grammar"]["outer_parameter_grids"]["transition_u"],
            dtype=cp.float64,
        ),
        "backend": "cupy_cuda_" + (
            name.decode() if isinstance(name, bytes) else str(name)
        ),
    }


def _gpu_score_batch(
    config: Mapping[str, Any], context: Mapping[str, Any], start: int, count: int
) -> dict[str, Any]:
    cp = context["cp"]
    ordinal_np = _schedule_ordinals(config, start, count)
    decoded = _decode_gpu(cp.asarray(ordinal_np), cp)
    physical_index = (
        decoded["amplitude_index"].astype(cp.int32) * 256
        + decoded["exponent_index"].astype(cp.int32) * 16
        + decoded["transition_index"].astype(cp.int32)
    )
    mask = context["physical_table"][physical_index]
    decoded = {key: value[mask] for key, value in decoded.items()}
    admitted = int(decoded["ordinal"].size)
    if admitted == 0:
        return {"admitted": 0, "best": {}}
    bank = context["primitive_bank"]
    left_raw = 2.0 * bank[decoded["left_primitive_index"].astype(cp.int32)] - 1.0
    right_raw = 2.0 * bank[decoded["right_primitive_index"].astype(cp.int32)] - 1.0
    left = _unary_gpu(left_raw, decoded["left_transform_index"], cp)
    right = _unary_gpu(right_raw, decoded["right_transform_index"], cp)
    mixing = context["mixing_grid"][decoded["mixing_index"].astype(cp.int32)]
    coordinate_raw = _binary_gpu(
        left, right, decoded["operator_index"], mixing, cp
    )
    coordinate = 0.5 + 0.5 * coordinate_raw / (1.0 + cp.abs(coordinate_raw))
    amplitude = context["amplitude_grid"][decoded["amplitude_index"].astype(cp.int32)]
    exponent = context["exponent_grid"][decoded["exponent_index"].astype(cp.int32)]
    transition = context["transition_grid"][decoded["transition_index"].astype(cp.int32)]
    u = context["u"]
    multiplier = 1.0 + amplitude[:, None] * cp.power(
        u[None, :], -exponent[:, None]
    ) / (1.0 + u[None, :] / transition[:, None]) * (
        0.05 + 0.95 * coordinate
    )
    behavior = cp.log10(multiplier)
    errors = cp.square(
        (behavior - context["residual"][None, :]) / context["sigma"][None, :]
    )
    losses = errors @ context["weights"]
    best: dict[str, Any] = {}
    for column, name in enumerate(context["selection_names"]):
        row = int(cp.argmin(losses[:, column]).item())
        best[name] = {
            "ordinal": int(decoded["ordinal"][row].item()),
            "loss": float(losses[row, column].item()),
        }
    return {"admitted": admitted, "best": best}


def _device_synchronize(context: Mapping[str, Any]) -> None:
    context["cp"].cuda.Stream.null.synchronize()


def run_throughput_benchmark(root: Path) -> dict[str, Any]:
    config = load_config(root)
    context = _gpu_context(root, config)
    contract = config["throughput_benchmark"]
    rows = []
    start = 0
    for batch_size in contract["raw_batch_sizes"]:
        batch_size = int(batch_size)
        for _ in range(int(contract["warmup_repetitions"])):
            _gpu_score_batch(config, context, start, batch_size)
            _device_synchronize(context)
        durations = []
        admitted_counts = []
        for _ in range(int(contract["measured_repetitions"])):
            before = time.perf_counter()
            result = _gpu_score_batch(config, context, start, batch_size)
            _device_synchronize(context)
            durations.append(time.perf_counter() - before)
            admitted_counts.append(int(result["admitted"]))
        if len(set(admitted_counts)) != 1:
            raise GravityItem51Error("benchmark admission count changed across repeats")
        admitted = admitted_counts[0]
        seconds = float(np.median(durations))
        point_folds = admitted * len(context["arrays"]["target"]) * 5
        rows.append(
            {
                "raw_batch_size": batch_size,
                "physically_admitted_candidates": admitted,
                "measured_seconds": durations,
                "median_seconds": seconds,
                "raw_candidates_per_second": batch_size / seconds,
                "admitted_candidates_per_second": admitted / seconds,
                "candidate_point_fold_evaluations_per_second": point_folds / seconds,
            }
        )
        start += batch_size
    fastest = max(rows, key=lambda row: row["candidate_point_fold_evaluations_per_second"])
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item51-throughput-benchmark-1.0",
            "item": 51,
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "backend": context["backend"],
            "warmup_repetitions": int(contract["warmup_repetitions"]),
            "measured_repetitions": int(contract["measured_repetitions"]),
            "synchronized_each_repetition": True,
            "measurements": rows,
            "fastest_measurement": fastest,
            "scope": "End-to-end batch generation, physical filtering, formula evaluation, and six selection-loss columns; excludes candidate-manifest equivalence work and result serialization.",
            "claims": {
                "actual_local_gpu_throughput_measured": True,
                "continuous_sustained_rate_claimed": False,
                "trillion_formula_campaign_executed": False,
            },
        }
    )


def write_throughput_benchmark(root: Path) -> Path:
    config = load_config(root)
    path = _source_path(root, config, "benchmark_result")
    _write_json(path, run_throughput_benchmark(root))
    return path


def run_gpu_screen(root: Path) -> dict[str, Any]:
    config = load_config(root)
    context = _gpu_context(root, config)
    schedule = config["schedule"]
    total = int(schedule["sample_positions"])
    batch_size = int(schedule["raw_batch_size"])
    best = {
        name: {"ordinal": None, "loss": math.inf}
        for name in context["selection_names"]
    }
    admitted = 0
    batches = 0
    before = time.perf_counter()
    for begin in range(0, total, batch_size):
        result = _gpu_score_batch(config, context, begin, min(batch_size, total - begin))
        admitted += int(result["admitted"])
        batches += 1
        for name, row in result["best"].items():
            if row["loss"] < best[name]["loss"]:
                best[name] = row
    _device_synchronize(context)
    seconds = time.perf_counter() - before
    point_folds = admitted * len(context["arrays"]["target"]) * 5
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item51-gpu-screen-1.0",
            "item": 51,
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "backend": context["backend"],
            "raw_schedule_positions": total,
            "raw_batches": batches,
            "physically_admitted_candidates": admitted,
            "selection_masks": len(context["selection_names"]),
            "candidate_point_fold_evaluations": point_folds,
            "wall_seconds": seconds,
            "raw_candidates_per_second": total / seconds,
            "admitted_candidates_per_second": admitted / seconds,
            "candidate_point_fold_evaluations_per_second": point_folds / seconds,
            "best_by_selection_mask": best,
            "post_evaluation_candidate_cells": 0,
            "sealed_confirmation_rows": 0,
            "paid_model_calls": 0,
            "claims": {
                "entire_frozen_schedule_executed": True,
                "full_grammar_exhausted": False,
                "trillion_formula_campaign_executed": False,
                "formula_family_pruned": False,
            },
        }
    )


def write_gpu_screen(root: Path) -> Path:
    config = load_config(root)
    path = _source_path(root, config, "screen_result")
    _write_json(path, run_gpu_screen(root))
    return path


def _behavior_for_ordinals(
    root: Path, arrays: Mapping[str, Any], ordinals: Sequence[int]
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    config49 = _load_item49_config(root)
    decoded = decode_ordinals(np.asarray(ordinals, dtype=np.uint64), config49)
    if not np.all(_physical_mask(decoded, config49)):
        raise GravityItem51Error("selected GPU ordinal is not physically admitted")
    bank = _primitive_bank_from_arrays(arrays)
    behavior = program_log_multiplier(
        decoded, bank, np.asarray(arrays["u"], dtype=float), config49
    )
    return decoded, behavior


def _selection_loss(
    arrays: Mapping[str, Any], behavior: np.ndarray, selected: np.ndarray
) -> float:
    weights = _object_weights(arrays, selected)
    residual = arrays["target"] - arrays["base"]
    return float(np.sum(np.square((behavior - residual) / arrays["sigma"]) * weights))


def _fixed_oof(
    arrays: Mapping[str, Any], fold_ordinals: Mapping[int, int], behavior: np.ndarray
) -> np.ndarray:
    prediction = np.empty(len(arrays["target"]), dtype=float)
    ordered_folds = sorted(fold_ordinals)
    for row, fold in enumerate(ordered_folds):
        test = arrays["fold"] == fold
        prediction[test] = arrays["base"][test] + behavior[row, test]
    return prediction


def build_evaluation_result(root: Path) -> dict[str, Any]:
    config = load_config(root)
    config48 = _load_item48_config(root)
    arrays = _item48_evaluation_arrays(root, config48)
    screen = _read_json(_source_path(root, config, "screen_result"))
    candidate = _read_json(_source_path(root, config, "candidate_manifest"))
    benchmark = _read_json(_source_path(root, config, "benchmark_result"))
    if screen["physically_admitted_candidates"] != candidate["physically_admitted_candidates"]:
        raise GravityItem51Error("GPU and response-blind admission counts disagree")
    fold_ordinals = {
        fold: int(screen["best_by_selection_mask"][f"fold_{fold}_train"]["ordinal"])
        for fold in range(int(config["evaluation"]["outer_folds"]))
    }
    full_ordinal = int(screen["best_by_selection_mask"]["full"]["ordinal"])
    ordered_ordinals = [fold_ordinals[fold] for fold in sorted(fold_ordinals)]
    programs, behavior = _behavior_for_ordinals(root, arrays, ordered_ordinals)
    full_programs, full_behavior = _behavior_for_ordinals(root, arrays, [full_ordinal])
    cpu_gpu_differences: dict[str, float] = {}
    ledger = []
    labels = primitive_labels(_primitive_sources(root))
    config49 = _load_item49_config(root)
    for row, fold in enumerate(sorted(fold_ordinals)):
        train = arrays["fold"] != fold
        test = ~train
        cpu_loss = _selection_loss(arrays, behavior[row], train)
        gpu_loss = float(
            screen["best_by_selection_mask"][f"fold_{fold}_train"]["loss"]
        )
        difference = abs(cpu_loss - gpu_loss)
        cpu_gpu_differences[f"fold_{fold}_train"] = difference
        if difference > float(config["evaluation"]["cpu_gpu_tolerance"]):
            raise GravityItem51Error(f"CPU/GPU loss cross-check failed for fold {fold}")
        ledger.append(
            {
                "fold": fold,
                "selected_gpu_program": _program_description(
                    programs, row, labels, config49
                ),
                "gpu_training_balanced_loss": gpu_loss,
                "cpu_training_balanced_loss": cpu_loss,
                "heldout_s4tm_objects": sorted(
                    set(arrays["object"][test & (arrays["population"] == "S4TM")].tolist())
                ),
                "heldout_clash_objects": sorted(
                    set(arrays["object"][test & (arrays["population"] == "CLASH")].tolist())
                ),
            }
        )
    full_cpu_loss = _selection_loss(
        arrays, full_behavior[0], np.ones(len(arrays["target"]), dtype=bool)
    )
    full_gpu_loss = float(screen["best_by_selection_mask"]["full"]["loss"])
    cpu_gpu_differences["full"] = abs(full_cpu_loss - full_gpu_loss)
    if cpu_gpu_differences["full"] > float(config["evaluation"]["cpu_gpu_tolerance"]):
        raise GravityItem51Error("CPU/GPU full-data loss cross-check failed")

    prediction = _fixed_oof(arrays, fold_ordinals, behavior)
    item49_evaluation = _read_json(root / ITEM49_EVALUATION_PATH)
    scores = dict(item49_evaluation["scores"])
    scores["gpu_stream_search"] = _score(arrays, prediction)
    controls = tuple(name for name in scores if name != "gpu_stream_search")
    strongest = min(controls, key=lambda name: scores[name]["balanced_loss"])
    candidate_objects = scores["gpu_stream_search"]["object_losses"]
    control_objects = scores[strongest]["object_losses"]
    object_keys = sorted(candidate_objects)
    diff = np.asarray(
        [control_objects[key] - candidate_objects[key] for key in object_keys]
    )
    raw_counterexample = diff < 0.0
    stable_counterexample = raw_counterexample.copy()

    config45 = _load_item45_config(root)
    config46 = _load_item46_config(root)
    config47 = _load_item47_config(root)
    shapes = _shape_by_object(root, arrays)
    systematic_scores: dict[str, Any] = {}
    for variant_name, population, shift in config["evaluation"]["mass_scale_variants"]:
        varied = _item45_variant_arrays(arrays, str(population), float(shift), config45)
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
        varied["action_bank"] = _item48_action_bank_from_arrays(varied, config48)[1].T
        _varied_programs, varied_behavior = _behavior_for_ordinals(
            root, varied, ordered_ordinals
        )
        varied_prediction = _fixed_oof(varied, fold_ordinals, varied_behavior)
        candidate_variant = _score(varied, varied_prediction)
        control_variant = item49_evaluation["systematic_scores"][str(variant_name)][
            "item45_primary_control"
        ]
        systematic_scores[str(variant_name)] = {
            "gpu_stream_search": candidate_variant,
            "item45_primary_control": control_variant,
        }
        for index, key in enumerate(object_keys):
            stable_counterexample[index] &= (
                candidate_variant["object_losses"][key]
                > control_variant["object_losses"][key]
            )

    leave_one = [float(np.mean(np.delete(diff, index))) for index in range(len(diff))]
    trim_count = max(
        1, int(len(diff) * float(config["evaluation"]["robust_trim_fraction"]))
    )
    trimmed = np.sort(diff)[trim_count:-trim_count]
    candidate_score = scores["gpu_stream_search"]["balanced_loss"]
    improvement = 100.0 * (
        scores[strongest]["balanced_loss"] - candidate_score
    ) / scores[strongest]["balanced_loss"]
    improvement_item45 = 100.0 * (
        scores["item45_universal_interaction"]["balanced_loss"] - candidate_score
    ) / scores["item45_universal_interaction"]["balanced_loss"]
    improvement_item49 = 100.0 * (
        scores["pseudorandom_program_search"]["balanced_loss"] - candidate_score
    ) / scores["pseudorandom_program_search"]["balanced_loss"]
    policy_report = {
        "evidence_kind": "empirical",
        "evaluable_objects": len(object_keys),
        "raw_counterexample_count": int(np.sum(raw_counterexample)),
        "quality_verified_counterexample_count": int(np.sum(raw_counterexample)),
        "uncertainty_resolved_counterexample_count": int(np.sum(stable_counterexample)),
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
            "schema_version": "invariant-gravity-item51-joint-evaluation-1.0",
            "item": 51,
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "selected_gpu_program": _program_description(
                full_programs, 0, labels, config49
            ),
            "selected_gpu_full_data_balanced_training_loss": full_cpu_loss,
            "fold_ledger": ledger,
            "scores": scores,
            "strongest_control": strongest,
            "aggregate_improvement_percent": improvement,
            "improvement_over_item45_percent": improvement_item45,
            "improvement_over_item49_percent": improvement_item49,
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
                    "mass_variant_stable_counterexample": bool(stable_counterexample[index]),
                }
                for index, key in enumerate(object_keys)
            ],
            "systematic_scores": systematic_scores,
            "counterexample_policy_report": policy_report,
            "counterexample_policy_assessment": policy,
            "compute": {
                "backend": screen["backend"],
                "raw_schedule_positions": screen["raw_schedule_positions"],
                "physically_admitted_candidates": screen[
                    "physically_admitted_candidates"
                ],
                "exact_symbolic_equivalence_classes": candidate[
                    "exact_symbolic_equivalence_classes"
                ],
                "candidate_point_fold_evaluations": screen[
                    "candidate_point_fold_evaluations"
                ],
                "campaign_wall_seconds": screen["wall_seconds"],
                "campaign_raw_candidates_per_second": screen[
                    "raw_candidates_per_second"
                ],
                "campaign_candidate_point_fold_evaluations_per_second": screen[
                    "candidate_point_fold_evaluations_per_second"
                ],
                "benchmark_fastest_measurement": benchmark["fastest_measurement"],
                "cpu_gpu_selected_loss_absolute_difference": cpu_gpu_differences,
            },
            "counts": {
                "s4tm_lenses": 28,
                "clash_clusters": 20,
                "clash_points": 84,
                "sealed_confirmation_rows": 0,
                "post_evaluation_candidate_cells": 0,
                "paid_model_calls": 0,
            },
            "limitations": [
                "All empirical response rows were exposed before Item 51; grouped cross-validation is retrospective development, not fresh confirmation.",
                "The run evaluates 67,108,864 scheduled ordinals from a 6,496,138,035,200-ordinal grammar; it neither exhausts the grammar nor evaluates a trillion formulas.",
                "All physically admitted scheduled candidates were scored even when two ordinals share a canonical symbolic class; the exact class count is reported separately.",
                "Behavioral equivalence was not computed for the large stream, so the symbolic class count is not a claim of distinct predictions.",
                "The grammar recombines known and previously generated primitives; a sampled expression is not historically novel without prior-art adjudication.",
                "S4TM uses an analytic projected stellar profile without measured gas; CLASH uses model-dependent published acceleration profiles.",
                "Four global baryonic-mass shifts do not exhaust measurement, geometry, selection, or lens-model uncertainty.",
                "Neither one empirical mismatch nor the number of mismatches prunes a formula family.",
            ],
        }
    )


def write_evaluation_result(root: Path) -> Path:
    config = load_config(root)
    path = _source_path(root, config, "evaluation_result")
    _write_json(path, build_evaluation_result(root))
    return path


def build_aggregate_result(root: Path) -> dict[str, Any]:
    config = load_config(root)
    preflight = _read_json(_source_path(root, config, "preflight_manifest"))
    candidate = _read_json(_source_path(root, config, "candidate_manifest"))
    benchmark = _read_json(_source_path(root, config, "benchmark_result"))
    screen = _read_json(_source_path(root, config, "screen_result"))
    evaluation = _read_json(_source_path(root, config, "evaluation_result"))
    scores = evaluation["scores"]
    gpu = scores["gpu_stream_search"]
    item45 = scores["item45_universal_interaction"]
    promotion = config["evaluation"]["promotion_gates"]
    fold_operators = [
        row["selected_gpu_program"]["binary_operator"]
        for row in evaluation["fold_ledger"]
    ]
    operator_stability = max(
        fold_operators.count(operator) for operator in set(fold_operators)
    )
    gates = {
        "rtx_5090_backend_verified": "RTX 5090" in str(screen["backend"]),
        "throughput_benchmark_completed": bool(
            benchmark["claims"]["actual_local_gpu_throughput_measured"]
        ),
        "entire_67108864_ordinal_schedule_executed": int(
            screen["raw_schedule_positions"]
        )
        == 67_108_864,
        "response_blind_schedule_and_equivalence": bool(
            candidate[
                "response_values_used_during_generation_or_symbolic_equivalence"
            ]
            == 0
        ),
        "gpu_cpu_selected_losses_match": max(
            evaluation["compute"]["cpu_gpu_selected_loss_absolute_difference"].values()
        )
        <= float(config["evaluation"]["cpu_gpu_tolerance"]),
        "balanced_improvement_over_item45_at_least": float(
            evaluation["improvement_over_item45_percent"]
        )
        >= 100.0 * float(promotion["balanced_improvement_over_item45_at_least"]),
        "improves_both_populations_over_item45": all(
            gpu["populations"][population]["loss"]
            < item45["populations"][population]["loss"]
            for population in ("S4TM", "CLASH")
        ),
        "paired_p_at_most": float(evaluation["paired_sign_flip_p"])
        <= float(promotion["paired_p_at_most"]),
        "same_top_level_operator_in_at_least_folds": operator_stability
        >= int(promotion["same_top_level_operator_in_at_least_folds"]),
        "leave_one_and_trim_stable": bool(
            evaluation["robustness"][
                "leave_one_min_mean_control_minus_candidate_loss"
            ]
            > 0.0
            and evaluation["robustness"][
                "trimmed_mean_control_minus_candidate_loss"
            ]
            > 0.0
        ),
        "all_mass_scale_variants_positive": all(
            value["gpu_stream_search"]["balanced_loss"]
            < value["item45_primary_control"]["balanced_loss"]
            for value in evaluation["systematic_scores"].values()
        ),
        "post_evaluation_candidate_cells": int(
            evaluation["counts"]["post_evaluation_candidate_cells"]
        )
        == 0,
        "sealed_confirmation_rows": int(
            evaluation["counts"]["sealed_confirmation_rows"]
        )
        == 0,
        "fresh_confirmation_available": False,
    }
    operational_names = (
        "rtx_5090_backend_verified",
        "throughput_benchmark_completed",
        "entire_67108864_ordinal_schedule_executed",
        "response_blind_schedule_and_equivalence",
        "gpu_cpu_selected_losses_match",
        "post_evaluation_candidate_cells",
        "sealed_confirmation_rows",
    )
    scientific_names = (
        "balanced_improvement_over_item45_at_least",
        "improves_both_populations_over_item45",
        "paired_p_at_most",
        "same_top_level_operator_in_at_least_folds",
        "leave_one_and_trim_stable",
        "all_mass_scale_variants_positive",
    )
    operational_complete = all(gates[name] for name in operational_names)
    scientific_lead = operational_complete and all(
        gates[name] for name in scientific_names
    )
    decision = (
        "RETROSPECTIVE_ITEM51_GPU_LEAD_REQUIRES_FRESH_TEST"
        if scientific_lead
        else (
            "OPERATIONAL_ITEM51_GPU_SCALE_COMPLETE_SCIENTIFIC_LEAD_NOT_DEMONSTRATED"
            if operational_complete
            else "INCOMPLETE_ITEM51_GPU_SCREEN_RETAINED"
        )
    )
    bindings = {}
    for name, key in (
        ("preflight", "preflight_manifest"),
        ("candidate_manifest", "candidate_manifest"),
        ("benchmark", "benchmark_result"),
        ("screen", "screen_result"),
        ("evaluation", "evaluation_result"),
    ):
        path = _source_path(root, config, key)
        bindings[name] = {
            "path": str(path.relative_to(root)),
            "sha256": _sha256_file(path),
        }
    bindings["config"] = {"path": str(CONFIG_PATH), "sha256": _sha256_file(root / CONFIG_PATH)}
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item51-gpu-screening-result-1.0",
            "item": 51,
            "goal": "GRAVITY_ROADMAP_ITEM_51_GPU_SCREENING",
            "decision": decision,
            "selected_gpu_program": evaluation["selected_gpu_program"],
            "scores": scores,
            "strongest_control": evaluation["strongest_control"],
            "aggregate_improvement_percent": evaluation[
                "aggregate_improvement_percent"
            ],
            "improvement_over_item45_percent": evaluation[
                "improvement_over_item45_percent"
            ],
            "improvement_over_item49_percent": evaluation[
                "improvement_over_item49_percent"
            ],
            "paired_sign_flip_p": evaluation["paired_sign_flip_p"],
            "gates": gates,
            "top_level_operator_fold_stability": {
                "operators": fold_operators,
                "maximum_same_operator_folds": operator_stability,
            },
            "counterexample_policy_assessment": evaluation[
                "counterexample_policy_assessment"
            ],
            "counts": {
                "full_addressable_ordinal_space": config[
                    "program_grammar_binding"
                ]["full_ordinal_space"],
                "raw_schedule_positions": candidate["schedule"][
                    "raw_schedule_positions"
                ],
                "physically_admitted_candidates": candidate[
                    "physically_admitted_candidates"
                ],
                "exact_symbolic_equivalence_classes": candidate[
                    "exact_symbolic_equivalence_classes"
                ],
                "outcome_scored_candidates": candidate["outcome_scored_candidates"],
                "candidate_point_fold_evaluations": screen[
                    "candidate_point_fold_evaluations"
                ],
                "s4tm_lenses": 28,
                "clash_clusters": 20,
                "clash_points": 84,
                "sealed_confirmation_rows": 0,
                "post_evaluation_candidate_cells": 0,
                "paid_model_calls": 0,
            },
            "throughput": {
                "backend": screen["backend"],
                "benchmark_fastest": benchmark["fastest_measurement"],
                "campaign_wall_seconds": screen["wall_seconds"],
                "campaign_raw_candidates_per_second": screen[
                    "raw_candidates_per_second"
                ],
                "campaign_candidate_point_fold_evaluations_per_second": screen[
                    "candidate_point_fold_evaluations_per_second"
                ],
            },
            "source_bindings": bindings,
            "claims": {
                "roadmap_item_51_complete": operational_complete,
                "measured_gpu_scaling_demonstrated": operational_complete,
                "gpu_search_beats_item45": bool(
                    evaluation["improvement_over_item45_percent"] > 0.0
                ),
                "fresh_confirmation_completed": False,
                "full_grammar_exhausted": False,
                "trillion_formula_campaign_executed": False,
                "historical_novelty_established": False,
                "alternative_to_gr_established": False,
                "dark_matter_eliminated": False,
                "formula_family_pruned": False,
                "single_counterexample_used_as_veto": False,
            },
            "limitations": evaluation["limitations"],
            "next_action": "Advance to Item 52 and store the exact Item 51 mismatch regions and causes in the failure-space database without globally pruning any formula family.",
            "preflight": preflight,
        }
    )


def write_aggregate_result(root: Path) -> Path:
    config = load_config(root)
    path = root / str(config["paths"]["aggregate_result"])
    _write_json(path, build_aggregate_result(root))
    return path


def _content_hash_valid(value: Mapping[str, Any]) -> bool:
    content = dict(value)
    expected = content.pop("content_sha256", None)
    return expected == _sha256_bytes(_canonical_bytes(content))


def replay(root: Path) -> dict[str, Any]:
    config = load_config(root)
    checks = {
        "preflight_content_hash": _content_hash_valid(
            _read_json(_source_path(root, config, "preflight_manifest"))
        ),
        "candidate_manifest": _read_json(
            _source_path(root, config, "candidate_manifest")
        )
        == build_candidate_manifest(root),
        "benchmark_content_hash": _content_hash_valid(
            _read_json(_source_path(root, config, "benchmark_result"))
        ),
        "screen_content_hash": _content_hash_valid(
            _read_json(_source_path(root, config, "screen_result"))
        ),
        "evaluation_result": _read_json(
            _source_path(root, config, "evaluation_result")
        )
        == build_evaluation_result(root),
        "aggregate_result": _read_json(root / str(config["paths"]["aggregate_result"]))
        == build_aggregate_result(root),
    }
    return {"ok": all(checks.values()), "checks": checks}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "preflight",
            "manifest",
            "benchmark",
            "screen",
            "evaluate",
            "aggregate",
            "replay",
        ),
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.command == "preflight":
        result: Any = str(write_preflight_manifest(root))
    elif args.command == "manifest":
        result = str(write_candidate_manifest(root))
    elif args.command == "benchmark":
        result = str(write_throughput_benchmark(root))
    elif args.command == "screen":
        result = str(write_gpu_screen(root))
    elif args.command == "evaluate":
        result = str(write_evaluation_result(root))
    elif args.command == "aggregate":
        result = str(write_aggregate_result(root))
    else:
        result = replay(root)
        if not result["ok"]:
            print(json.dumps(result, sort_keys=True))
            return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
