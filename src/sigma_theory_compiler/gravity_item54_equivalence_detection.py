"""Item 54 layered formula equivalence with lineage-preserving aliases."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from sigma_theory_compiler.gravity_item22_polarization_superposition import (
    _canonical_bytes,
    _content_hashed,
    _read_json,
    _sha256_bytes,
    _sha256_file,
    _write_json,
)
from sigma_theory_compiler.gravity_item45_universal_interactions import (
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
    _primitive_bank_from_arrays,
    decode_ordinals,
    load_config as _load_item49_config,
    program_log_multiplier,
)
from sigma_theory_compiler.gravity_item51_gpu_screening import (
    _canonical_symbolic_keys,
)
from sigma_theory_compiler.gravity_item53_diversity_preservation import _pool


CONFIG_PATH = Path("configs/gravity_item54_equivalence_detection_v1.json")
GOAL_PATH = Path("docs/GRAVITY_HIDDEN_VARIABLE_AND_THEORY_SEARCH_GOALS.md")
ITEM53_RESULT_PATH = Path("runs/gravity/roadmap/item-53-diversity-preservation-v1.json")
ITEM53_ARCHIVE_PATH = Path(
    "runs/gravity/roadmap/item-53-diversity-preservation-v1-source/archive-manifest.json"
)


class GravityItem54Error(RuntimeError):
    """Raised when equivalence or alias preservation changes."""


def load_config(root: Path, *, require_bound: bool = True) -> dict[str, Any]:
    config = _read_json(root / CONFIG_PATH)
    validate_config(root, config, require_bound=require_bound)
    return config


def validate_config(
    root: Path, config: Mapping[str, Any], *, require_bound: bool = True
) -> None:
    if (
        config.get("schema_version")
        != "invariant-gravity-item54-equivalence-detection-config-1.0"
        or int(config.get("item", -1)) != 54
    ):
        raise GravityItem54Error("unexpected Item 54 config")
    if _sha256_file(root / GOAL_PATH) != config["stable_goal_sha256"]:
        raise GravityItem54Error("stable gravity goal changed")
    freeze = str(config["scientific_freeze_commit"])
    if require_bound and re.fullmatch(r"[0-9a-f]{40}", freeze) is None:
        raise GravityItem54Error("Item 54 scientific freeze is not commit-bound")
    if not require_bound and not (
        freeze == "PENDING_FREEZE_COMMIT" or re.fullmatch(r"[0-9a-f]{40}", freeze)
    ):
        raise GravityItem54Error("malformed Item 54 freeze binding")
    for relative, expected in config["scientific_dependencies"].items():
        if _sha256_file(root / str(relative)) != expected:
            raise GravityItem54Error(f"scientific dependency changed: {relative}")
    predecessor = _read_json(root / ITEM53_RESULT_PATH)
    required = config["required_predecessor"]
    if predecessor["decision"] != required["decision"]:
        raise GravityItem54Error("Item 53 decision binding changed")
    if predecessor["content_sha256"] != required["content_sha256"]:
        raise GravityItem54Error("Item 53 content binding changed")
    policy = config["preservation_policy"]
    for key in (
        "original_ordinals_deleted",
        "lineage_records_deleted",
        "protected_archive_references_deleted",
    ):
        if policy[key] != 0:
            raise GravityItem54Error(f"destructive equivalence policy entered: {key}")
    if policy["formula_family_pruned"]:
        raise GravityItem54Error("equivalence became formula-family pruning")


def _contract_digest(config: Mapping[str, Any]) -> str:
    value = json.loads(json.dumps(config))
    value["scientific_freeze_commit"] = "<BOUND_COMMIT>"
    return _sha256_bytes(_canonical_bytes(value))


def _source_path(root: Path, config: Mapping[str, Any], key: str) -> Path:
    return root / str(config["paths"]["source_dir"]) / str(config["paths"][key])


def build_preflight_manifest(root: Path) -> dict[str, Any]:
    config = load_config(root)
    pool = _pool(root)
    archive = _read_json(root / ITEM53_ARCHIVE_PATH)
    references = sum(len(archive["archives"][name]) for name in config["candidate_pool"]["protected_archives"])
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item54-preflight-1.0",
            "item": 54,
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "config_contract_sha256": _contract_digest(config),
            "candidate_pool_unique_ordinals": len(pool),
            "protected_archive_references": references,
            "equivalence_layers": config["equivalence_layers"],
            "control_suite": config["control_suite"],
            "response_fields_used_for_behavioral_equivalence": [],
            "response_values_used_for_behavioral_equivalence": 0,
            "post_outcome_equivalence_rules": 0,
            "sealed_confirmation_rows": 0,
            "paid_model_calls": 0,
            "preservation_policy": config["preservation_policy"],
        }
    )


def write_preflight_manifest(root: Path) -> Path:
    config = load_config(root)
    path = _source_path(root, config, "preflight_manifest")
    _write_json(path, build_preflight_manifest(root))
    return path


def _environment_arrays(root: Path) -> list[tuple[str, dict[str, Any]]]:
    config45 = _load_item45_config(root)
    config46 = _load_item46_config(root)
    config47 = _load_item47_config(root)
    config48 = _load_item48_config(root)
    primary = _item48_evaluation_arrays(root, config48)
    shapes = _shape_by_object(root, primary)
    environments = [("primary_112_predictor_rows", primary)]
    variants = (
        ("s4tm_stellar_mass_minus_0.25_dex", "S4TM", -0.25),
        ("s4tm_stellar_mass_plus_0.25_dex", "S4TM", 0.25),
        ("clash_baryonic_scale_minus_0.10_dex", "CLASH", -0.10),
        ("clash_baryonic_scale_plus_0.10_dex", "CLASH", 0.10),
    )
    for name, population, shift in variants:
        varied = _item45_variant_arrays(primary, population, shift, config45)
        varied["pi_bank"] = (
            1.0
            / (
                1.0
                + np.abs(
                    _item46_physical_log_values(varied, config46)
                    @ np.asarray(_item46_pi_vectors(config46), float).T
                )
            )
        ).T
        varied["operator_bank"] = _item47_operator_bank_from_arrays(
            varied, shapes, config47
        )[1].T
        varied["action_bank"] = _item48_action_bank_from_arrays(varied, config48)[1].T
        environments.append((name, varied))
    return environments


def _behavior_matrix(
    root: Path, programs: Mapping[str, np.ndarray], environments: Sequence[tuple[str, Mapping[str, Any]]]
) -> np.ndarray:
    config49 = _load_item49_config(root)
    blocks = []
    for _name, arrays in environments:
        blocks.append(
            program_log_multiplier(
                programs,
                _primitive_bank_from_arrays(arrays),
                np.asarray(arrays["u"], float),
                config49,
            )
        )
    return np.concatenate(blocks, axis=1)


def _classes_from_keys(
    keys: np.ndarray, ordinals: np.ndarray, pool_by_ordinal: Mapping[int, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    groups: dict[Any, list[int]] = {}
    for key, ordinal in zip(keys.tolist(), ordinals.tolist(), strict=True):
        groups.setdefault(key, []).append(int(ordinal))
    records = []
    for class_id, members in enumerate(sorted(groups.values(), key=lambda rows: min(rows))):
        members = sorted(members)
        records.append(
            {
                "class_id": class_id,
                "evaluation_representative_ordinal": members[0],
                "member_ordinals": members,
                "member_count": len(members),
                "lineage_memberships": [
                    {
                        "ordinal": ordinal,
                        "item52_region_memberships": pool_by_ordinal[ordinal][
                            "item52_region_memberships"
                        ],
                    }
                    for ordinal in members
                ],
                "member_ordinals_deleted": 0,
                "lineage_records_deleted": 0,
            }
        )
    return records


def build_equivalence_manifest(root: Path) -> dict[str, Any]:
    config = load_config(root)
    pool = _pool(root)
    pool_by_ordinal = {int(row["ordinal"]): row for row in pool}
    ordinals = np.asarray(sorted(pool_by_ordinal), dtype=np.uint64)
    config49 = _load_item49_config(root)
    programs = decode_ordinals(ordinals, config49)
    symbolic_keys = _canonical_symbolic_keys(programs, config49)
    environments = _environment_arrays(root)
    behavior = _behavior_matrix(root, programs, environments)
    rounded = np.ascontiguousarray(
        np.round(behavior, int(config["equivalence_layers"]["behavioral"]["signature_decimals"]))
    )
    byte_rows = rounded.view(
        np.dtype((np.void, rounded.dtype.itemsize * rounded.shape[1]))
    ).ravel()
    _unique, inverse = np.unique(byte_rows, return_inverse=True)
    symbolic_classes = _classes_from_keys(symbolic_keys, ordinals, pool_by_ordinal)
    behavioral_classes = _classes_from_keys(inverse, ordinals, pool_by_ordinal)
    archive = _read_json(root / ITEM53_ARCHIVE_PATH)
    protected_references = [
        {
            "archive": name,
            "slot": slot,
            "ordinal": int(row["ordinal"]),
            "preserved": True,
        }
        for name in config["candidate_pool"]["protected_archives"]
        for slot, row in enumerate(archive["archives"][name])
    ]
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item54-equivalence-manifest-1.0",
            "item": 54,
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "input_unique_ordinals": len(ordinals),
            "identity_classes": len(ordinals),
            "symbolic_classes": symbolic_classes,
            "behavioral_classes": behavioral_classes,
            "counts": {
                "symbolic_equivalence_classes": len(symbolic_classes),
                "symbolic_aliases": len(ordinals) - len(symbolic_classes),
                "multi_environment_behavioral_equivalence_classes": len(behavioral_classes),
                "behavioral_aliases": len(ordinals) - len(behavioral_classes),
                "behavior_signature_environments": len(environments),
                "behavior_signature_predictor_cells_per_candidate": behavior.shape[1],
                "original_ordinals_deleted": 0,
                "lineage_records_deleted": 0,
                "protected_archive_references": len(protected_references),
                "protected_archive_references_deleted": 0,
            },
            "environment_names": [name for name, _arrays in environments],
            "behavior_matrix_sha256": hashlib.sha256(
                rounded.astype("<f8", copy=False).tobytes()
            ).hexdigest(),
            "protected_archive_references": protected_references,
            "claims": {
                "behavioral_equivalence_is_global_algebraic_identity": False,
                "behavioral_equivalence_is_exact_on_rounded_declared_environments": True,
                "every_alias_and_lineage_preserved": True,
                "protected_archive_membership_preserved": True,
                "formula_family_pruned": False,
                "historical_novelty_established": False,
            },
        }
    )


def write_equivalence_manifest(root: Path) -> Path:
    config = load_config(root)
    path = _source_path(root, config, "equivalence_manifest")
    _write_json(path, build_equivalence_manifest(root))
    return path


def _control_programs() -> dict[str, np.ndarray]:
    # Rows: duplicate pair, commutative swap, zero difference/contrast,
    # equal-operand max/min unary collapse, and near-but-distinct transition cells.
    fields = {
        "ordinal": [1, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "transition_index": [12, 12, 12, 12, 12, 12, 12, 12, 14, 15],
        "exponent_index": [0] * 10,
        "amplitude_index": [15] * 10,
        "mixing_index": [11, 11, 5, 5, 11, 11, 11, 11, 11, 11],
        "operator_index": [0, 0, 2, 2, 1, 4, 5, 6, 0, 0],
        "right_transform_index": [0, 0, 3, 1, 0, 0, 2, 2, 0, 0],
        "right_primitive_index": [1, 1, 8, 7, 12, 12, 14, 14, 1, 1],
        "left_transform_index": [0, 0, 1, 3, 0, 0, 2, 2, 0, 0],
        "left_primitive_index": [0, 0, 7, 8, 12, 12, 14, 14, 0, 0],
    }
    return {
        key: np.asarray(values, dtype=np.uint64 if key == "ordinal" else np.int16)
        for key, values in fields.items()
    }


def build_control_test(root: Path) -> dict[str, Any]:
    config = load_config(root)
    config49 = _load_item49_config(root)
    programs = _control_programs()
    keys = _canonical_symbolic_keys(programs, config49)
    primary = _environment_arrays(root)[0][1]
    behavior = _behavior_matrix(root, programs, [("primary", primary)])
    checks = {
        "exact_duplicate_detected": bool(
            np.array_equal(behavior[0], behavior[1]) and keys[0] == keys[1]
        ),
        "commutative_rewrite_detected": bool(
            np.array_equal(behavior[2], behavior[3]) and keys[2] == keys[3]
        ),
        "zero_collapse_detected": bool(
            np.array_equal(behavior[4], behavior[5]) and keys[4] == keys[5]
        ),
        "unary_collapse_detected": bool(
            np.array_equal(behavior[6], behavior[7]) and keys[6] == keys[7]
        ),
        "near_but_unequal_behavior_not_merged": bool(
            not np.array_equal(np.round(behavior[8], 10), np.round(behavior[9], 10))
            and keys[8] != keys[9]
        ),
    }
    expected = config["control_suite"]
    if any(checks[key] is not bool(expected[key]) for key in expected):
        raise GravityItem54Error("equivalence control suite failed")
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item54-control-test-1.0",
            "item": 54,
            "checks": checks,
            "all_controls_passed": all(checks.values()),
            "control_rows": len(programs["ordinal"]),
            "response_values_used": 0,
            "claims": {
                "known_rewrites_detected": True,
                "near_unequal_control_preserved": True,
                "control_suite_proves_global_equivalence_completeness": False,
            },
        }
    )


def write_control_test(root: Path) -> Path:
    config = load_config(root)
    path = _source_path(root, config, "control_test")
    _write_json(path, build_control_test(root))
    return path


def build_aggregate_result(root: Path) -> dict[str, Any]:
    config = load_config(root)
    preflight = _read_json(_source_path(root, config, "preflight_manifest"))
    equivalence = _read_json(_source_path(root, config, "equivalence_manifest"))
    control = _read_json(_source_path(root, config, "control_test"))
    counts = equivalence["counts"]
    gates = {
        "all_control_equivalences_detected": control["all_controls_passed"],
        "all_878_input_ordinals_accounted": equivalence["input_unique_ordinals"]
        == 878,
        "multi_environment_behavior_signatures_completed": counts[
            "behavior_signature_environments"
        ]
        == 5,
        "all_original_ordinals_preserved": counts["original_ordinals_deleted"] == 0,
        "all_lineage_records_preserved": counts["lineage_records_deleted"] == 0,
        "all_128_archive_references_preserved": bool(
            counts["protected_archive_references"] == 128
            and counts["protected_archive_references_deleted"] == 0
        ),
        "post_outcome_equivalence_rules": preflight[
            "post_outcome_equivalence_rules"
        ]
        == 0,
        "sealed_confirmation_rows": preflight["sealed_confirmation_rows"] == 0,
    }
    complete = all(gates.values())
    bindings = {}
    for name, key in (
        ("preflight", "preflight_manifest"),
        ("equivalence_manifest", "equivalence_manifest"),
        ("control_test", "control_test"),
    ):
        path = _source_path(root, config, key)
        bindings[name] = {
            "path": str(path.relative_to(root)),
            "sha256": _sha256_file(path),
        }
    bindings["config"] = {"path": str(CONFIG_PATH), "sha256": _sha256_file(root / CONFIG_PATH)}
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item54-equivalence-detection-result-1.0",
            "item": 54,
            "goal": "GRAVITY_ROADMAP_ITEM_54_EQUIVALENCE_DETECTION",
            "decision": (
                "ITEM54_LAYERED_EQUIVALENCE_OPERATIONAL_WITH_LINEAGE_PRESERVED"
                if complete
                else "INCOMPLETE_ITEM54_EQUIVALENCE_RESULT_RETAINED"
            ),
            "gates": gates,
            "counts": counts,
            "control_test": control,
            "source_bindings": bindings,
            "claims": {
                "roadmap_item_54_complete": complete,
                **equivalence["claims"],
                "original_ordinals_deleted": False,
                "lineage_records_deleted": False,
                "protected_archive_references_deleted": False,
                "global_equivalence_completeness_proved": False,
                "fresh_confirmation_completed": False,
                "alternative_to_gr_established": False,
                "dark_matter_eliminated": False,
            },
            "limitations": [
                "Symbolic equivalence recognizes only the frozen algebraic rewrite rules, not arbitrary computer-algebra identities.",
                "Behavioral equivalence is rounded equality on five declared predictor environments and is not a proof of global functional identity.",
                "Distinct mechanisms can coincide on the probes; all aliases, lineages, and protected archive references therefore remain preserved.",
                "No response target is used in equivalence detection, but the predictor environments originate in already exposed development pipelines.",
            ],
            "next_action": "Advance to Item 55 causal-variable tests using equivalence representatives for compute efficiency while keeping all aliases and lineages addressable.",
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
        "preflight": _read_json(_source_path(root, config, "preflight_manifest"))
        == build_preflight_manifest(root),
        "equivalence_manifest": _read_json(
            _source_path(root, config, "equivalence_manifest")
        )
        == build_equivalence_manifest(root),
        "control_test": _read_json(_source_path(root, config, "control_test"))
        == build_control_test(root),
        "aggregate_result": _read_json(root / str(config["paths"]["aggregate_result"]))
        == build_aggregate_result(root),
    }
    return {"ok": all(checks.values()), "checks": checks}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("preflight", "detect", "controls", "aggregate", "replay")
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.command == "preflight":
        result: Any = str(write_preflight_manifest(root))
    elif args.command == "detect":
        result = str(write_equivalence_manifest(root))
    elif args.command == "controls":
        result = str(write_control_test(root))
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
