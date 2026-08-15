from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

CONFIG_SCHEMA = (
    "sigma-quartic-candidate-complete-global-h7-lifespan-system8-successor-gate-config-1.0"
)
RECEIPT_SCHEMA = "sigma-quartic-candidate-complete-global-h7-lifespan-system8-successor-gate-1.0"
CAMPAIGN_ID = "quartic-candidate-complete-global-h7-lifespan-system8-successor-gate-001"
EXPECTED_DIRECTIONS = (
    "subset_0",
    "subset_01",
    "subset_012",
    "subset_0123",
    "subset_013",
    "subset_02",
    "subset_023",
    "subset_03",
    "subset_1",
    "subset_12",
    "subset_123",
    "subset_13",
    "subset_2",
    "subset_23",
    "subset_3",
)
EXPECTED_PREDECESSORS = {
    "system9_candidate_gate": (
        "sigma-quartic-candidate-complete-global-h7-lifespan-gate-1.0",
        "block_all_12_candidate_global_H7_completion_or_completion_grade_obstruction",
    ),
    "system8_independent_g2": (
        "sigma-quartic-tc2-d4-independent-g2-alternative-k55-recurrence-1.0",
        "block_first_exact_broader_symmetrizer_obstruction",
    ),
}
EXPECTED_COUNTS = {
    "selected_candidates": 12,
    "candidate_blocks": 12,
    "candidate_passes": 0,
    "candidate_rejections": 0,
    "prior_finite_unmodified_Sobolev_obstructions": 12,
    "completion_grade_obstructions_before": 0,
    "completion_grade_obstructions_after": 0,
    "candidates_upgraded_to_completion_grade": 0,
    "required_independent_G2_transport_evaluations": 15,
    "proved_independent_G2_transport_evaluations": 15,
    "exact_55_state_lifts_attempted": 3,
    "exact_55_state_lifts_passed": 2,
    "positive_symmetrizer_tubes_proved": 0,
}
EXPECTED_CLAIMS = {
    "all_15_independent_G2_transports_proved": True,
    "all_15_exact_55_state_lifts_proved": False,
    "positive_symmetrizer_tube_proved": False,
    "global_H7_proved": False,
    "bootstrap_closed": False,
    "lifespan_proved": False,
    "completion_grade_obstruction_proved": False,
    "candidate_rejected": False,
    "promotion_authorized": False,
}


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _content_sha(value: Mapping[str, Any]) -> str:
    return _sha({key: item for key, item in value.items() if key != "content_sha256"})


def _normalized_text_sha(path: Path) -> str:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(text.encode()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"System9 successor source is not readable JSON: {path}") from error
    if not isinstance(value, dict):
        raise TypeError("System9 successor source must be one JSON object")
    return value


def _inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("System9 successor path escaped project root") from error
    return path


def _validate_config(config: Mapping[str, Any]) -> None:
    if set(config) != {
        "schema_version",
        "campaign_id",
        "output_path",
        "expected_candidate_count",
        "required_polarization_evaluations",
        "predecessors",
        "completion_contract",
        "claims_policy",
    }:
        raise ValueError("System9 successor config boundary changed")
    if (
        config["schema_version"] != CONFIG_SCHEMA
        or config["campaign_id"] != CAMPAIGN_ID
        or config["expected_candidate_count"] != 12
        or tuple(config["required_polarization_evaluations"]) != EXPECTED_DIRECTIONS
        or config["claims_policy"] != EXPECTED_CLAIMS
    ):
        raise ValueError("System9 successor config boundary changed")
    if config["completion_contract"] != {
        "success": (
            "candidate-bound closed global H7 energy inequality, bootstrap, and explicit "
            "positive lifespan"
        ),
        "obstruction": (
            "candidate-bound mathematically explicit obstruction covering every admitted "
            "H7 closure strategy with full-direction replay"
        ),
        "action_level_lift_failure_is_candidate_completion_grade": False,
        "candidate_rejection_authorized": False,
    }:
        raise ValueError("System9 successor completion definition changed")
    predecessors = config["predecessors"]
    if not isinstance(predecessors, dict) or set(predecessors) != set(EXPECTED_PREDECESSORS):
        raise ValueError("System9 successor predecessor boundary changed")
    for binding in predecessors.values():
        if (
            not isinstance(binding, dict)
            or set(binding) != {"path", "content_sha256"}
            or Path(binding["path"]).is_absolute()
            or len(binding["content_sha256"]) != 64
        ):
            raise ValueError("System9 successor predecessor binding changed")


def _validate_seal(label: str, value: Mapping[str, Any], expected: str) -> None:
    if value.get("content_sha256") != expected or _content_sha(value) != expected:
        raise ValueError(f"System9 successor {label} content seal changed")


def _validate_sources(
    config: Mapping[str, Any], documents: Mapping[str, Mapping[str, Any]]
) -> None:
    if set(documents) != set(EXPECTED_PREDECESSORS):
        raise ValueError("System9 successor source set changed")
    for label, (schema, status) in EXPECTED_PREDECESSORS.items():
        value = documents[label]
        _validate_seal(label, value, config["predecessors"][label]["content_sha256"])
        if value.get("schema_version") != schema or value.get("status") != status:
            raise ValueError(f"System9 successor {label} authority changed")

    prior = documents["system9_candidate_gate"]
    if (
        prior.get("decision") != "BLOCK_SYSTEM9"
        or prior.get("counts", {}).get("selected_candidates") != 12
        or prior.get("counts", {}).get("candidate_blocks") != 12
        or prior.get("counts", {}).get("completion_grade_obstructions") != 0
        or prior.get("counts", {}).get("accepted_full_direction_recurrence_evaluations") != 0
        or prior.get("claims", {}).get("global_H7_proved")
        or prior.get("claims", {}).get("completion_grade_obstruction_proved")
    ):
        raise ValueError("System9 successor prior candidate boundary changed")
    records = prior.get("candidate_records")
    if not isinstance(records, list) or len(records) != 12:
        raise ValueError("System9 successor prior candidate set changed")
    if len({record.get("candidate_id") for record in records}) != 12:
        raise ValueError("System9 successor prior candidate identity changed")
    if any(
        record.get("decision") != "BLOCK_SYSTEM9"
        or record.get("completion_grade")
        or record.get("candidate_rejection_authorized")
        for record in records
    ):
        raise ValueError("System9 successor prior candidate classification changed")

    broader = documents["system8_independent_g2"]
    if (
        broader.get("decision") != "BLOCK_SERIALIZATION"
        or not broader.get("all_15_broader_transport_systems_pass")
        or broader.get("all_15_exact_55_state_lifts_pass")
        or broader.get("positive_tube_proved")
        or broader.get("counts")
        != {
            "55_state_lifts_passed": 2,
            "manifest_registered_after": 154,
            "manifest_registered_before": 154,
            "positive_tubes_proved": 0,
            "registered_packets": 0,
            "remaining_packets": 150,
            "required_evaluations": 15,
            "transport_evaluations_audited": 15,
            "transport_evaluations_solved": 15,
        }
        or broader.get("claims")
        != {
            "all_15_broader_transports_proved": True,
            "all_15_exact_55_state_lifts_proved": False,
            "global_H7_claim": False,
            "higher_K55_registered": False,
            "manifest_advanced": False,
            "missing_packets_inferred_as_zero": False,
            "positive_symmetrizer_tube_proved": False,
        }
    ):
        raise ValueError("System9 successor independent-G2 boundary changed")
    evaluations = broader.get("evaluation_records")
    if not isinstance(evaluations, list) or len(evaluations) != 15:
        raise ValueError("System9 successor independent-G2 evaluation set changed")
    if {record.get("evaluation_id") for record in evaluations} != set(EXPECTED_DIRECTIONS):
        raise ValueError("System9 successor independent-G2 directions changed")
    for record in evaluations:
        if record.get("canonical_route_coefficient_rank") != record.get(
            "canonical_route_augmented_rank"
        ) or record.get("companion_Taylor_remainder_entries") != [0, 0, 0]:
            raise ValueError("System9 successor independent-G2 transport changed")
    lifts = broader.get("lift_records")
    if not isinstance(lifts, list) or [item.get("evaluation_id") for item in lifts] != [
        "subset_0",
        "subset_1",
        "subset_2",
    ]:
        raise ValueError("System9 successor 55-state lift order changed")
    if [item.get("pass") for item in lifts] != [True, True, False]:
        raise ValueError("System9 successor 55-state lift result changed")
    failure = broader.get("first_exact_55_state_lift_failure")
    if failure != {
        "K55_symmetrizer_remainder_entries": [0, 0, 72],
        "K55_symmetry_remainder_entries": [0, 0, 0],
        "evaluation_id": "subset_2",
        "first_missing_primitive": (
            "exact_55_state_transverse_cross_lift_of_independent_G2_metric"
        ),
    }:
        raise ValueError("System9 successor exact lift obstruction changed")


def _load_documents(root: Path, config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        label: _load(_inside(root, binding["path"]))
        for label, binding in config["predecessors"].items()
    }


def _negative_controls(
    config: Mapping[str, Any], documents: Mapping[str, Mapping[str, Any]]
) -> dict[str, dict[str, bool]]:
    def rejects(mutate: Callable[[dict[str, Any], dict[str, Any]], None]) -> bool:
        changed_config = copy.deepcopy(dict(config))
        changed_documents = copy.deepcopy(dict(documents))
        mutate(changed_config, changed_documents)
        for label, value in changed_documents.items():
            value["content_sha256"] = _content_sha(value)
            changed_config["predecessors"][label]["content_sha256"] = value["content_sha256"]
        try:
            _validate_config(changed_config)
            _validate_sources(changed_config, changed_documents)
        except (KeyError, TypeError, ValueError):
            return True
        return False

    def omit_direction(_: dict[str, Any], documents_: dict[str, Any]) -> None:
        documents_["system8_independent_g2"]["evaluation_records"].pop()

    def erase_72(_: dict[str, Any], documents_: dict[str, Any]) -> None:
        documents_["system8_independent_g2"]["first_exact_55_state_lift_failure"][
            "K55_symmetrizer_remainder_entries"
        ] = [0, 0, 0]

    def promote_lift(_: dict[str, Any], documents_: dict[str, Any]) -> None:
        documents_["system8_independent_g2"]["claims"]["all_15_exact_55_state_lifts_proved"] = True

    def upgrade_prior(_: dict[str, Any], documents_: dict[str, Any]) -> None:
        documents_["system9_candidate_gate"]["candidate_records"][0]["completion_grade"] = True

    return {
        "omit_one_transport_evaluation": {"rejected": rejects(omit_direction)},
        "erase_exact_72_entry_lift_obstruction": {"rejected": rejects(erase_72)},
        "promote_partial_lift_to_all_15": {"rejected": rejects(promote_lift)},
        "upgrade_prior_candidate_without_theorem": {"rejected": rejects(upgrade_prior)},
    }


def build_gate(
    root: Path,
    config_path: Path | None = None,
    *,
    documents: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    config_path = config_path or (
        root / "configs/backgrounds/"
        "quartic_candidate_complete_global_h7_lifespan_system8_successor_gate.json"
    )
    config_path = config_path.resolve()
    config = _load(config_path)
    _validate_config(config)
    source_documents = dict(documents) if documents is not None else _load_documents(root, config)
    _validate_sources(config, source_documents)
    prior = source_documents["system9_candidate_gate"]
    broader = source_documents["system8_independent_g2"]

    candidates = []
    for old in prior["candidate_records"]:
        candidates.append(
            {
                "candidate_id": old["candidate_id"],
                "decision": "BLOCK_SYSTEM9",
                "completion_grade_before": False,
                "completion_grade_after": False,
                "new_transport_evidence": {
                    "independent_G2_transport_evaluations_proved": 15,
                    "required_transport_evaluations": 15,
                    "transport_primitive_closed": True,
                    "exact_55_state_lifts_passed_before_failure": 2,
                    "first_lift_failure_evaluation": "subset_2",
                    "first_lift_failure_Taylor_order": 3,
                    "first_lift_failure_symmetrizer_nonzero_entries": 72,
                },
                "why_not_completion_grade": {
                    "candidate_bound": False,
                    "positive_symmetrizer_tube_proved": False,
                    "full_tensor_cancellation_excluded": False,
                    "modified_energy_excluded": False,
                    "Nash_Moser_or_derivative_loss_evolution_excluded": False,
                    "analytic_or_Gevrey_closure_excluded": False,
                },
                "unchanged_success_primitive": old["global_H7_proof_branch"]["missing_hypothesis"],
                "candidate_rejection_authorized": False,
            }
        )

    bindings = {
        label: {
            "path": config["predecessors"][label]["path"],
            "content_sha256": value["content_sha256"],
            "semantic_sha256": _content_sha(value),
        }
        for label, value in source_documents.items()
    }
    local_paths = {
        "config": config_path.relative_to(root).as_posix(),
        "source": (
            "src/sigma_theory_compiler/"
            "quartic_candidate_complete_global_h7_lifespan_system8_successor_gate.py"
        ),
        "test": (
            "tests/test_quartic_candidate_complete_global_h7_lifespan_system8_successor_gate.py"
        ),
    }
    local_bindings = {
        label: {"path": path, "normalized_text_sha256": _normalized_text_sha(root / path)}
        for label, path in local_paths.items()
    }
    result: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "status": "block_all_12_after_exact_system8_transport_success_and_lift_obstruction",
        "decision": "BLOCK_SYSTEM9",
        "counts": EXPECTED_COUNTS,
        "claims": EXPECTED_CLAIMS,
        "completion_contract": config["completion_contract"],
        "candidate_records": candidates,
        "system8_successor_evidence": {
            "transport_audit": {
                "required_evaluations": list(EXPECTED_DIRECTIONS),
                "proved_evaluations": sorted(
                    item["evaluation_id"] for item in broader["evaluation_records"]
                ),
                "all_15_independent_G2_transports_proved": True,
            },
            "exact_55_state_lift_frontier": {
                "passed_evaluations": ["subset_0", "subset_1"],
                "first_failure_evaluation": "subset_2",
                "first_failure_Taylor_order": 3,
                "symmetry_remainder_nonzero_entries_by_order": [0, 0, 0],
                "symmetrizer_remainder_nonzero_entries_by_order": [0, 0, 72],
                "positive_tube_attempted": False,
            },
            "obstruction_scope": (
                "action-level failure of this independent-G2 55-state lift; it is not a "
                "candidate-bound theorem and excludes none of the remaining H7 closure "
                "strategies"
            ),
        },
        "measured_upgrade": {
            "transport_primitive_closed": True,
            "candidate_completion_grade_obstructions_upgraded": [],
            "candidate_completion_grade_upgrade_count": 0,
            "reason": (
                "15/15 equal-eigenspace transport solvability does not supply a 55-state "
                "lift or positive tube; the subset_2/order-3 72-entry lift obstruction is "
                "not candidate-bound and does not cover all admitted H7 closure strategies"
            ),
        },
        "exact_remaining_contract": {
            "first_System8_missing_primitive": (
                "exact_55_state_transverse_cross_lift_of_independent_G2_metric"
            ),
            "first_System8_failure_location": {
                "evaluation_id": "subset_2",
                "Taylor_order": 3,
                "symmetrizer_remainder_nonzero_entries": 72,
            },
            "System9_success_path": (
                "prove the candidate-bound B7 source-good-unknown bound, close the H7 "
                "bootstrap, and derive an explicit positive lifespan for each candidate"
            ),
            "System9_obstruction_path": (
                "prove a candidate-bound theorem covering full-tensor/modified-energy, "
                "Nash-Moser/derivative-loss, and analytic/Gevrey closure"
            ),
            "first_System9_completion_primitive": (
                "candidate_bound_full_tensor_source_good_unknown_B7_bound_or_"
                "all_closure_strategy_completion_grade_obstruction"
            ),
            "partial_candidate_upgrade_forbidden": True,
        },
        "negative_controls": _negative_controls(config, source_documents),
        "source_bindings": bindings,
        "local_bindings": local_bindings,
        "scope": (
            "Successor audit only. It records the exact all-15 independent-G2 transport "
            "success and the first exact 55-state lift obstruction, without promoting an "
            "action-level construction failure to a candidate-global H7 obstruction. No "
            "lifespan, rejection, promotion, observation, or manifest advancement is claimed."
        ),
    }
    result["content_sha256"] = _content_sha(result)
    _validate_result(result)
    return result


def _validate_result(result: Mapping[str, Any]) -> None:
    if set(result) != {
        "schema_version",
        "campaign_id",
        "status",
        "decision",
        "counts",
        "claims",
        "completion_contract",
        "candidate_records",
        "system8_successor_evidence",
        "measured_upgrade",
        "exact_remaining_contract",
        "negative_controls",
        "source_bindings",
        "local_bindings",
        "scope",
        "content_sha256",
    }:
        raise ValueError("System9 successor receipt boundary changed")
    if result.get("content_sha256") != _content_sha(result):
        raise ValueError("System9 successor receipt content seal changed")
    if (
        result.get("schema_version") != RECEIPT_SCHEMA
        or result.get("campaign_id") != CAMPAIGN_ID
        or result.get("status")
        != "block_all_12_after_exact_system8_transport_success_and_lift_obstruction"
        or result.get("decision") != "BLOCK_SYSTEM9"
        or result.get("counts") != EXPECTED_COUNTS
        or result.get("claims") != EXPECTED_CLAIMS
    ):
        raise ValueError("System9 successor receipt boundary changed")
    candidates = result.get("candidate_records")
    if not isinstance(candidates, list) or len(candidates) != 12:
        raise ValueError("System9 successor receipt candidate set changed")
    if len({item.get("candidate_id") for item in candidates}) != 12:
        raise ValueError("System9 successor receipt candidate identity changed")
    for item in candidates:
        scope = item.get("why_not_completion_grade", {})
        if (
            item.get("decision") != "BLOCK_SYSTEM9"
            or item.get("completion_grade_before")
            or item.get("completion_grade_after")
            or item.get("candidate_rejection_authorized")
            or any(scope.values())
        ):
            raise ValueError("System9 successor receipt candidate classification changed")
    evidence = result.get("system8_successor_evidence", {})
    if (
        evidence.get("transport_audit", {}).get("proved_evaluations") != sorted(EXPECTED_DIRECTIONS)
        or not evidence.get("transport_audit", {}).get("all_15_independent_G2_transports_proved")
        or evidence.get("exact_55_state_lift_frontier", {}).get(
            "symmetrizer_remainder_nonzero_entries_by_order"
        )
        != [0, 0, 72]
        or evidence.get("exact_55_state_lift_frontier", {}).get("positive_tube_attempted")
    ):
        raise ValueError("System9 successor receipt evidence changed")
    if result.get("measured_upgrade", {}).get("candidate_completion_grade_upgrade_count") != 0:
        raise ValueError("System9 successor receipt upgrade changed")
    controls = result.get("negative_controls", {})
    if set(controls) != {
        "omit_one_transport_evaluation",
        "erase_exact_72_entry_lift_obstruction",
        "promote_partial_lift_to_all_15",
        "upgrade_prior_candidate_without_theorem",
    } or not all(control.get("rejected") is True for control in controls.values()):
        raise ValueError("System9 successor negative controls changed")
    for forbidden in (
        "global_H7_proved",
        "bootstrap_closed",
        "lifespan_proved",
        "completion_grade_obstruction_proved",
        "candidate_rejected",
        "promotion_authorized",
    ):
        if result["claims"][forbidden]:
            raise ValueError("System9 successor forbidden claim opened")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sigma-system9-system8-successor-gate")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    result = build_gate(arguments.root, arguments.config)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        print(rendered, end="")
    else:
        output = arguments.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
