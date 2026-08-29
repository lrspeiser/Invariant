"""Item 60: direct-CLASH lensing theory-readiness gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from sigma_theory_compiler.gravity_item22_polarization_superposition import (
    _content_hashed,
    _read_json,
    _sha256_file,
    _write_json,
)

CONFIG_PATH = Path("configs/gravity_item60_direct_clash_lensing_gate_v1.json")
ITEM59_PATH = Path("runs/gravity/roadmap/item-59-xcop-forward-observable-gate-v1.json")


class GravityItem60Error(RuntimeError):
    """Raised when the frozen Item 60 theory or source boundary changes."""


def load_config(root: Path, *, require_bound: bool = True) -> dict[str, Any]:
    config = _read_json(root / CONFIG_PATH)
    validate_config(root, config, require_bound=require_bound)
    return config


def validate_config(
    root: Path, config: Mapping[str, Any], *, require_bound: bool = True
) -> None:
    if (
        config.get("schema_version")
        != "invariant-gravity-item60-direct-clash-lensing-config-1.0"
        or config.get("item") != 60
        or config.get("status")
        != "scientific_freeze_before_direct_lensing_response_access"
    ):
        raise GravityItem60Error("unsupported Item 60 config")
    freeze = str(config.get("scientific_freeze_commit", ""))
    if require_bound and re.fullmatch(r"[0-9a-f]{40}", freeze) is None:
        raise GravityItem60Error("Item 60 scientific freeze is not bound")
    if not require_bound and not (
        freeze == "PENDING_FREEZE_COMMIT" or re.fullmatch(r"[0-9a-f]{40}", freeze)
    ):
        raise GravityItem60Error("invalid Item 60 freeze marker")
    for relative, expected in config["scientific_dependencies"].items():
        path = root / str(relative)
        if not path.is_file() or _sha256_file(path) != str(expected):
            raise GravityItem60Error(f"scientific dependency changed: {relative}")
    item59 = _read_json(root / ITEM59_PATH)
    selected = item59["selection"]["selected_qualifying"]["variant"]
    candidate = config["candidate_contract"]
    if (
        item59.get("decision")
        != "ITEM59_XCOP_FORWARD_OBSERVABLE_GATE_PASSED_DEVELOPMENT_EVIDENCE"
        or selected.get("variant_id") != candidate["variant_id"]
        or selected.get("family_id") != candidate["family_id"]
        or selected.get("parameters") != candidate["parameters"]
        or item59.get("claims", {}).get("direct_lensing_completed") is not False
    ):
        raise GravityItem60Error("Item 59 candidate boundary changed")
    if candidate["formula_or_nuisance_refit_allowed"] is not False:
        raise GravityItem60Error("Item 60 cannot refit the candidate")
    if candidate["gr_lensing_conversion_may_be_assumed"] is not False:
        raise GravityItem60Error("GR lensing conversion cannot be silently assumed")
    channels = [str(row["id"]) for row in config["required_channels"]]
    if channels != [
        "image_positions",
        "parities",
        "shapes",
        "weak_shear",
        "magnification",
        "time_delays",
    ]:
        raise GravityItem60Error("direct-lensing channel set changed")
    policy = config["counterexample_policy"]
    if (
        policy["single_empirical_counterexample_terminal"] is not False
        or policy["counterexample_count_alone_terminal"] is not False
        or policy["finite_sample_may_prune_formula_family"] is not False
        or policy["theory_underdetermination_is_an_empirical_counterexample"] is not False
        or policy["global_family_pruning_allowed"] is not False
    ):
        raise GravityItem60Error("empirical over-pruning is forbidden")
    if any(bool(row.get("target_rows")) for row in config["source_metadata"]):
        raise GravityItem60Error("pre-response source metadata cannot contain target rows")


def _contract_sha(config: Mapping[str, Any]) -> str:
    body = dict(config)
    body.pop("scientific_freeze_commit", None)
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def build_source_receipt(root: Path) -> dict[str, Any]:
    config = load_config(root)
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item60-source-metadata-receipt-1.0",
            "item": 60,
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "config_contract_sha256": _contract_sha(config),
            "sources": config["source_metadata"],
            "source_metadata_records": len(config["source_metadata"]),
            "declared_strong_lensing_position_records": 312,
            "direct_target_rows_opened": 0,
            "direct_target_values_read": 0,
            "gr_or_nfw_mass_rows_read": 0,
            "paid_model_calls": 0,
        }
    )


def _source_path(root: Path, config: Mapping[str, Any], name: str) -> Path:
    return root / str(config["paths"]["source_dir"]) / str(config["paths"][name])


def write_source_receipt(root: Path) -> Path:
    config = load_config(root)
    path = _source_path(root, config, "source_receipt")
    _write_json(path, build_source_receipt(root))
    return path


def evaluate(root: Path) -> dict[str, Any]:
    config = load_config(root)
    receipt_path = _source_path(root, config, "source_receipt")
    if not receipt_path.is_file() or _read_json(receipt_path) != build_source_receipt(root):
        raise GravityItem60Error("source metadata receipt missing or changed")
    supplied = set(map(str, config["candidate_contract"]["available_theory_primitives"]))
    required = list(map(str, config["pre_response_gate"]["required_theory_primitives"]))
    missing = [name for name in required if name not in supplied]
    channel_rows = []
    for row in config["required_channels"]:
        channel_rows.append(
            {
                "channel": row["id"],
                "evaluable": False,
                "reason": "candidate does not define a lensing potential and its derivatives",
                "target_rows_opened": 0,
            }
        )
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item60-theory-readiness-evaluation-1.0",
            "item": 60,
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "candidate": config["candidate_contract"],
            "required_theory_primitives": required,
            "supplied_theory_primitives": sorted(supplied),
            "missing_theory_primitives": missing,
            "channels": channel_rows,
            "evaluable_channels": 0,
            "required_channels": len(channel_rows),
            "pre_response_gate_passed": False,
            "direct_target_rows_opened": 0,
            "empirical_counterexamples": [],
            "theory_counterexamples": [
                {
                    "id": "nonunique_lensing_completion",
                    "scope": "unchanged Item 59 acceleration representation",
                    "finding": (
                        "A radial acceleration for massive tracers does not uniquely determine "
                        "the photon-coupled metric, gravitational slip, lensing Jacobian, or "
                        "Fermat potential."
                    ),
                }
            ],
            "decision": "BLOCKED_THEORY_UNDERDEFINED_RETAIN_CANDIDATE_AND_DO_NOT_OPEN_TARGET_ROWS",
            "next_action": (
                "Generate action-level or explicit two-potential completions whose massive and "
                "null couplings are derived, then freeze one before opening direct lensing rows."
            ),
        }
    )


def build_aggregate(root: Path) -> dict[str, Any]:
    config = load_config(root)
    receipt = build_source_receipt(root)
    result = evaluate(root)
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item60-direct-clash-lensing-gate-1.0",
            "goal": "GRAVITY_ROADMAP_ITEM_60_DIRECT_CLASH_LENSING_GATE",
            "item": 60,
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "hypothesis": config["hypothesis"],
            "source_receipt": receipt,
            "evaluation": result,
            "decision": result["decision"],
            "gate_passed": False,
            "result_class": "BLOCKED",
            "counts": {
                "source_metadata_records": receipt["source_metadata_records"],
                "declared_strong_lensing_position_records": receipt[
                    "declared_strong_lensing_position_records"
                ],
                "required_channels": result["required_channels"],
                "evaluable_channels": result["evaluable_channels"],
                "direct_target_rows_opened": result["direct_target_rows_opened"],
                "empirical_counterexamples": len(result["empirical_counterexamples"]),
                "theory_counterexamples": len(result["theory_counterexamples"]),
                "paid_model_calls": 0,
            },
            "claims": {
                "roadmap_item_60_attempt_complete": True,
                "direct_clash_lensing_gate_passed": False,
                "direct_lensing_fit_performed": False,
                "item59_acceleration_candidate_empirically_rejected": False,
                "formula_or_feature_family_pruned": False,
                "single_empirical_counterexample_used_as_veto": False,
                "alternative_to_gr_established": False,
                "dark_matter_eliminated": False,
                "historical_novelty_established": False,
            },
            "compute": {"backend": "deterministic_schema_audit", "gpu_used": False, "paid_api_cost_usd": 0.0},
            "limitations": [
                "The gate stopped before target values were opened because the candidate cannot produce a unique lensing prediction.",
                "Public source metadata establishes that the requested direct observables exist; it is not an empirical fit.",
                "A future completion must preserve the Item 59 massive-tracer law and derive rather than separately fit lensing slip.",
            ],
            "next_action": result["next_action"],
        }
    )


def write_evaluation(root: Path) -> tuple[Path, Path]:
    config = load_config(root)
    evaluation_path = _source_path(root, config, "evaluation")
    aggregate_path = root / str(config["paths"]["aggregate"])
    _write_json(evaluation_path, evaluate(root))
    _write_json(aggregate_path, build_aggregate(root))
    return evaluation_path, aggregate_path


def replay(root: Path) -> dict[str, Any]:
    config = load_config(root)
    receipt_path = _source_path(root, config, "source_receipt")
    evaluation_path = _source_path(root, config, "evaluation")
    aggregate_path = root / str(config["paths"]["aggregate"])
    checks = {
        "source_receipt": receipt_path.is_file()
        and _read_json(receipt_path) == build_source_receipt(root),
        "evaluation": evaluation_path.is_file()
        and _read_json(evaluation_path) == evaluate(root),
        "aggregate": aggregate_path.is_file()
        and _read_json(aggregate_path) == build_aggregate(root),
    }
    return {"ok": all(checks.values()), "checks": checks}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("source-receipt", "evaluate", "replay"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.command == "source-receipt":
        print(write_source_receipt(root))
        return 0
    if args.command == "evaluate":
        print(json.dumps(build_aggregate(root), sort_keys=True))
        write_evaluation(root)
        return 0
    output = replay(root)
    print(json.dumps(output, sort_keys=True))
    return 0 if output["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
