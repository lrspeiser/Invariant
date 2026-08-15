from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

CONFIG_SCHEMA = "sigma-quartic-candidate-complete-global-h7-lifespan-gate-config-1.0"
RECEIPT_SCHEMA = "sigma-quartic-candidate-complete-global-h7-lifespan-gate-1.0"
EXPECTED_CAMPAIGN_ID = "quartic-candidate-complete-global-h7-lifespan-gate-001"
EXPECTED_CANDIDATES = 12
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
    "global_h7": (
        "sigma-quartic-global-h7-energy-campaign-1.0",
        "audit_all_12_global_H7_energies_single_source_remainder_lifespans_fail_closed",
    ),
    "finite_sobolev_no_go": (
        "sigma-quartic-finite-sobolev-hierarchy-no-go-campaign-1.0",
        "finite_unmodified_Sobolev_hierarchy_refuted_candidates_blocked",
    ),
    "higher_p55": (
        "sigma-quartic-tc2-d4-higher-p55-materializer-result-1.0",
        "pass_exact_45_higher_P55_packets_registered",
    ),
    "higher_h_star": (
        "sigma-quartic-tc2-d4-higher-h-star-materializer-result-1.0",
        "pass_exact_45_physical_H_star_plus_higher_Taylor_packets_materialized",
    ),
    "higher_k55": (
        "sigma-quartic-tc2-d4-higher-k55-registration-1.0",
        "block_higher_K55_at_subset_2_order_3_auxiliary_Riesz_metric_correction",
    ),
    "k55_sylvester_obstruction": (
        "sigma-quartic-tc2-d4-higher-k55-sylvester-obstruction-1.0",
        "block_exact_equal_physical_eigenspace_K55_order_three_obstruction",
    ),
    "physical_metric_transport_no_go": (
        "sigma-quartic-tc2-d4-physical-metric-transport-no-go-1.0",
        "no_go_source_bound_symmetric_physical_metric_transport_cannot_cancel_K55_obstruction",
    ),
    "alternative_symmetrizer": (
        "sigma-quartic-tc2-d4-alternative-symmetrizer-recurrence-audit-1.0",
        "pass_exact_witness_local_companion_alternative_symmetrizer_recurrence_block_55_state_global_registration",
    ),
}
EXPECTED_COUNTS = {
    "selected_candidates": 12,
    "closed_global_H7_proofs": 0,
    "closed_bootstraps": 0,
    "positive_lifespans": 0,
    "finite_unmodified_Sobolev_obstructions": 12,
    "completion_grade_obstructions": 0,
    "candidate_blocks": 12,
    "candidate_passes": 0,
    "candidate_rejections": 0,
    "required_full_direction_evaluations": 15,
    "accepted_full_direction_recurrence_evaluations": 0,
    "witness_local_alternative_evaluations": 1,
    "missing_full_direction_alternative_evaluations": 14,
}
EXPECTED_CLAIMS = {
    "global_H7_proved": False,
    "bootstrap_closed": False,
    "lifespan_proved": False,
    "completion_grade_obstruction_proved": False,
    "full_direction_recurrence_proved": False,
    "candidate_rejected": False,
    "promotion_authorized": False,
    "observation_opened": False,
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
        raise ValueError(f"System9 source is not readable JSON: {path}") from error
    if not isinstance(value, dict):
        raise TypeError("System9 source must be one JSON object")
    return value


def _inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("System9 path escaped project root") from error
    return path


def _validate_config(config: Mapping[str, Any]) -> None:
    expected_keys = {
        "schema_version",
        "campaign_id",
        "output_path",
        "expected_candidate_count",
        "state_sobolev_order",
        "required_polarization_evaluations",
        "completion_contract",
        "predecessors",
        "claims_policy",
    }
    if set(config) != expected_keys:
        raise ValueError("System9 config boundary changed")
    if (
        config["schema_version"] != CONFIG_SCHEMA
        or config["campaign_id"] != EXPECTED_CAMPAIGN_ID
        or config["expected_candidate_count"] != EXPECTED_CANDIDATES
        or config["state_sobolev_order"] != 7
        or tuple(config["required_polarization_evaluations"]) != EXPECTED_DIRECTIONS
        or config["claims_policy"] != EXPECTED_CLAIMS
    ):
        raise ValueError("System9 config boundary changed")
    contract = config["completion_contract"]
    if not isinstance(contract, dict) or contract != {
        "success": (
            "candidate-bound closed global H7 energy inequality, bootstrap, and "
            "explicit positive lifespan"
        ),
        "obstruction": (
            "candidate-bound mathematically explicit obstruction covering every "
            "admitted H7 closure strategy with full-direction replay"
        ),
        "scoped_no_go_is_completion_grade": False,
        "witness_local_recurrence_is_full_direction": False,
        "candidate_rejection_authorized": False,
    }:
        raise ValueError("System9 completion definition changed")
    predecessors = config["predecessors"]
    if not isinstance(predecessors, dict) or set(predecessors) != set(EXPECTED_PREDECESSORS):
        raise ValueError("System9 predecessor set changed")
    for binding in predecessors.values():
        if not isinstance(binding, dict) or set(binding) != {"path", "content_sha256"}:
            raise ValueError("System9 predecessor binding changed")
        if Path(binding["path"]).is_absolute() or len(binding["content_sha256"]) != 64:
            raise ValueError("System9 predecessor binding changed")


def _validate_content_seal(label: str, value: Mapping[str, Any], expected: str) -> None:
    if value.get("content_sha256") != expected or _content_sha(value) != expected:
        raise ValueError(f"System9 {label} content seal changed")


def _validate_sources(
    config: Mapping[str, Any], documents: Mapping[str, Mapping[str, Any]]
) -> None:
    if set(documents) != set(EXPECTED_PREDECESSORS):
        raise ValueError("System9 source document set changed")
    for label, (schema, status) in EXPECTED_PREDECESSORS.items():
        value = documents[label]
        binding = config["predecessors"][label]
        _validate_content_seal(label, value, binding["content_sha256"])
        if value.get("schema_version") != schema:
            raise ValueError(f"System9 {label} schema changed")
        observed_status = value.get("status", value.get("decision"))
        if observed_status != status:
            raise ValueError(f"System9 {label} status changed")

    global_h7 = documents["global_h7"]
    if global_h7.get("counts") != {
        "selected": 12,
        "global_energy_equivalences_certified": 12,
        "global_nonremainder_summations_certified": 12,
        "leading_good_unknown_bindings_verified": 12,
        "closed_global_H7_inequalities": 0,
        "global_H7_sums_applied": 0,
        "lifespans_proved": 0,
        "rejected": 0,
    }:
        raise ValueError("System9 global H7 counts changed")
    certificates = global_h7.get("certificates")
    if not isinstance(certificates, list) or len(certificates) != EXPECTED_CANDIDATES:
        raise ValueError("System9 global H7 candidate set changed")
    for certificate in certificates:
        if (
            not certificate.get("global_H7_energy_equivalence_certified")
            or not certificate.get("global_nonremainder_dyadic_summation_certified")
            or not certificate["strongest_global_differential_inequality"].get(
                "proved_with_explicit_remainder"
            )
            or certificate.get("global_H7_differential_inequality_closed")
            or certificate.get("global_H7_dyadic_sum_applied")
            or certificate.get("nonlinear_lifespan_proved")
        ):
            raise ValueError("System9 global H7 candidate classification changed")

    finite = documents["finite_sobolev_no_go"]
    if (
        finite.get("decision_counts") != {"pass": 0, "reject": 0, "blocked": 12}
        or finite.get("gate_counts", {}).get("finite_order_direct_hierarchy_no_go_certificates")
        != 12
        or finite.get("gate_counts", {}).get("full_tensor_cancellations_proved") != 0
        or finite.get("gate_counts", {}).get("global_H7_closures") != 0
        or finite.get("gate_counts", {}).get("lifespans_proved") != 0
    ):
        raise ValueError("System9 finite-Sobolev counts changed")
    scope_limit = finite.get("theorem", {}).get("scope_limit", "")
    for required in (
        "does not rule out a full tensor cancellation",
        "modified energy",
        "Nash-Moser/derivative-loss evolution",
        "analytic/Gevrey closure",
    ):
        if required not in scope_limit:
            raise ValueError("System9 finite-Sobolev scope limit changed")
    records = finite.get("candidate_records")
    if not isinstance(records, list) or len(records) != EXPECTED_CANDIDATES:
        raise ValueError("System9 finite-Sobolev candidate set changed")
    for record in records:
        if (
            record.get("decision") != "blocked"
            or record.get("direct_finite_Sobolev_hierarchy_closure")
            or record.get("full_tensor_cancellation_proved")
            or record.get("candidate_rejection_authorized")
            or record.get("representative_D2_value") not in {"-2", "-1", "1", "2"}
        ):
            raise ValueError("System9 finite-Sobolev candidate classification changed")

    global_ids = [item["candidate_id"] for item in certificates]
    finite_ids = [item["candidate_id"] for item in records]
    if global_ids != finite_ids or len(set(global_ids)) != EXPECTED_CANDIDATES:
        raise ValueError("System9 candidate identity alignment changed")

    higher_p55 = documents["higher_p55"]
    p55_packets = higher_p55.get("registered_P55_Taylor_orders_two_through_four_packets")
    if (
        higher_p55.get("counts", {}).get("P55_higher_packets_registered") != 45
        or not isinstance(p55_packets, list)
        or sorted({packet["evaluation_id"] for packet in p55_packets})
        != sorted(EXPECTED_DIRECTIONS)
    ):
        raise ValueError("System9 full-direction P55 manifest changed")
    h_star = documents["higher_h_star"]
    if h_star.get("counts", {}).get("H_star_plus_higher_packets") != 45 or sorted(
        {packet["evaluation_id"] for packet in h_star.get("packets", [])}
    ) != sorted(EXPECTED_DIRECTIONS):
        raise ValueError("System9 full-direction H-star manifest changed")

    higher_k55 = documents["higher_k55"]
    completed = higher_k55.get("completed_unregistered_evaluation_checkpoints")
    failure = higher_k55.get("failure_checkpoint")
    if (
        completed != ["subset_0", "subset_1"]
        or not isinstance(failure, dict)
        or failure.get("evaluation_id") != "subset_2"
        or failure.get("Taylor_order") != 3
        or higher_k55.get("counts", {}).get("higher_K55_packets_registered") != 0
        or higher_k55.get("claims", {}).get("all_45_higher_K55_packets_registered")
    ):
        raise ValueError("System9 K55 frontier changed")
    sylvester = documents["k55_sylvester_obstruction"]
    if (
        sylvester.get("counts", {}).get("input_residual_nonzero_polynomial_entries") != 120
        or sylvester.get("counts", {}).get(
            "canonical_Sylvester_correction_nonzero_polynomial_entries"
        )
        != 0
        or "all seven flat eigenspaces" not in sylvester.get("scope", "")
    ):
        raise ValueError("System9 Sylvester obstruction scope changed")
    physical = documents["physical_metric_transport_no_go"]
    if (
        physical.get("counts", {}).get("physical_signs_checked") != 2
        or physical.get("counts", {}).get("transport_map_rank_each") != 0
        or "rational-unit-direction" not in physical.get("scope", "")
        or "does not exclude" not in physical.get("scope", "")
    ):
        raise ValueError("System9 physical metric no-go scope changed")
    alternative = documents["alternative_symmetrizer"]
    witness = alternative.get("exact_witness_local_recurrence")
    if (
        not isinstance(witness, dict)
        or witness.get("evaluation_id") != "subset_2"
        or witness.get("direction") != ["3/5", "4/5", "0"]
        or alternative.get("claims", {}).get("global_coordinate_free_recurrence_proved")
        or alternative.get("claims", {}).get("positive_symmetrizer_tube_proved")
        or alternative.get("counts", {}).get("global_coordinate_free_alternatives") != 0
    ):
        raise ValueError("System9 alternative recurrence scope changed")


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
        try:
            _validate_config(changed_config)
            _validate_sources(changed_config, changed_documents)
        except (KeyError, TypeError, ValueError):
            return True
        return False

    def reseal(value: dict[str, Any]) -> None:
        value["content_sha256"] = _content_sha(value)

    def omit_candidate(_: dict[str, Any], documents_: dict[str, Any]) -> None:
        documents_["global_h7"]["certificates"].pop()
        reseal(documents_["global_h7"])
        documents_["global_h7"]["content_sha256"] = config["predecessors"]["global_h7"][
            "content_sha256"
        ]

    def erase_slice(_: dict[str, Any], documents_: dict[str, Any]) -> None:
        documents_["finite_sobolev_no_go"]["candidate_records"][0]["representative_D2_value"] = "0"
        reseal(documents_["finite_sobolev_no_go"])

    def promote_scoped(config_: dict[str, Any], _: dict[str, Any]) -> None:
        config_["completion_contract"]["scoped_no_go_is_completion_grade"] = True

    def corrupt_direction(config_: dict[str, Any], _: dict[str, Any]) -> None:
        config_["required_polarization_evaluations"][-1] = "subset_corrupt"

    def promote_witness(_: dict[str, Any], documents_: dict[str, Any]) -> None:
        documents_["alternative_symmetrizer"]["claims"][
            "global_coordinate_free_recurrence_proved"
        ] = True
        reseal(documents_["alternative_symmetrizer"])

    return {
        "omit_one_candidate": {"rejected": rejects(omit_candidate)},
        "erase_nonzero_candidate_D2_slice": {"rejected": rejects(erase_slice)},
        "promote_scoped_no_go_to_completion": {"rejected": rejects(promote_scoped)},
        "corrupt_full_direction_manifest": {"rejected": rejects(corrupt_direction)},
        "promote_witness_local_recurrence_to_full_direction": {
            "rejected": rejects(promote_witness)
        },
    }


def build_gate(
    root: Path,
    config_path: Path | None = None,
    *,
    documents: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    config_path = config_path or (
        root / "configs/backgrounds/quartic_candidate_complete_global_h7_lifespan_gate.json"
    )
    config = _load(config_path)
    _validate_config(config)
    source_documents = dict(documents) if documents is not None else _load_documents(root, config)
    _validate_sources(config, source_documents)

    global_by_id = {
        item["candidate_id"]: item for item in source_documents["global_h7"]["certificates"]
    }
    finite_by_id = {
        item["candidate_id"]: item
        for item in source_documents["finite_sobolev_no_go"]["candidate_records"]
    }
    missing_directions = [item for item in EXPECTED_DIRECTIONS if item != "subset_2"]
    candidates = []
    for candidate_id in sorted(global_by_id):
        global_item = global_by_id[candidate_id]
        finite_item = finite_by_id[candidate_id]
        candidates.append(
            {
                "candidate_id": candidate_id,
                "decision": "BLOCK_SYSTEM9",
                "completion_grade": False,
                "global_H7_proof_branch": {
                    "energy_equivalence_certified": True,
                    "known_nonremainder_terms_summed": True,
                    "strongest_inequality_has_explicit_B7_remainder": True,
                    "missing_hypothesis": global_item["bootstrap_and_conditional_lifespan"][
                        "missing_hypothesis"
                    ],
                    "closed_global_H7_inequality": False,
                    "bootstrap_closed": False,
                    "positive_lifespan_proved": False,
                },
                "obstruction_branch": {
                    "finite_unmodified_Sobolev_no_go_proved": True,
                    "theorem_domain": (
                        "R3 compact-frequency Schwartz packets; every finite integer s>=4"
                    ),
                    "representative_D2_value": finite_item["representative_D2_value"],
                    "absolute_growth_multiplier": finite_item["absolute_growth_multiplier"],
                    "full_tensor_cancellation_excluded": False,
                    "modified_energy_excluded": False,
                    "Nash_Moser_or_derivative_loss_evolution_excluded": False,
                    "analytic_or_Gevrey_closure_excluded": False,
                    "full_direction_completion_obstruction_proved": False,
                },
                "shared_System8_scope": {
                    "P55_and_H_star_full_direction_inputs_registered": True,
                    "canonical_K55_failure_evaluation": "subset_2",
                    "canonical_K55_failure_order": 3,
                    "symmetric_physical_metric_transport_no_go_at_witness": True,
                    "witness_local_alternative_recurrence_constructed": True,
                    "accepted_global_coordinate_free_recurrence": False,
                    "accepted_positive_symmetrizer_tube": False,
                    "accepted_full_direction_replays": 0,
                    "required_full_direction_replays": 15,
                    "missing_alternative_recurrence_evaluations": missing_directions,
                    "scope_limit": (
                        "the subset_2 witness and its seven flat eigenspace projections do "
                        "not establish an all-direction or candidate-global H7 obstruction"
                    ),
                },
                "first_missing_primitive": (
                    "candidate_bound_full_tensor_source_good_unknown_B7_bound_or_"
                    "completion_grade_full_direction_obstruction"
                ),
                "candidate_rejection_authorized": False,
            }
        )

    bindings = {}
    for label, binding in config["predecessors"].items():
        bindings[label] = {
            "path": binding["path"],
            "content_sha256": binding["content_sha256"],
            "semantic_sha256": _content_sha(source_documents[label]),
        }
    local_paths = {
        "config": config_path,
        "source": root
        / "src/sigma_theory_compiler/quartic_candidate_complete_global_h7_lifespan_gate.py",
        "test": root / "tests/test_quartic_candidate_complete_global_h7_lifespan_gate.py",
    }
    local_bindings = {
        label: {
            "path": path.resolve().relative_to(root).as_posix(),
            "normalized_text_sha256": _normalized_text_sha(path),
        }
        for label, path in local_paths.items()
    }
    result: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "campaign_id": EXPECTED_CAMPAIGN_ID,
        "status": "block_all_12_candidate_global_H7_completion_or_completion_grade_obstruction",
        "decision": "BLOCK_SYSTEM9",
        "counts": EXPECTED_COUNTS,
        "claims": EXPECTED_CLAIMS,
        "completion_contract": config["completion_contract"],
        "candidate_records": candidates,
        "full_direction_replay_audit": {
            "required_evaluations": list(EXPECTED_DIRECTIONS),
            "P55_input_evaluations_replayed": list(EXPECTED_DIRECTIONS),
            "H_star_input_evaluations_replayed": list(EXPECTED_DIRECTIONS),
            "accepted_alternative_recurrence_evaluations": [],
            "witness_local_only_evaluations": ["subset_2"],
            "missing_accepted_alternative_recurrence_evaluations": missing_directions,
            "full_direction_completion_replay_closed": False,
        },
        "exact_remaining_contract": {
            "success_path": (
                "bind and prove B7(t)<=C_L(R)*sqrt(Q7(t))+C_B(R)*Q7(t) for each "
                "candidate, close the H7 bootstrap, and derive an explicit positive lifespan"
            ),
            "obstruction_path": (
                "prove a candidate-bound theorem covering full-tensor/modified-energy, "
                "Nash-Moser/derivative-loss, and analytic/Gevrey closure on every declared "
                "polarization evaluation"
            ),
            "first_missing_primitive": (
                "candidate_bound_full_tensor_source_good_unknown_B7_bound_or_"
                "completion_grade_full_direction_obstruction"
            ),
            "partial_manifest_advancement_forbidden": True,
        },
        "negative_controls": _negative_controls(config, source_documents),
        "source_bindings": bindings,
        "local_bindings": local_bindings,
        "scope": (
            "Candidate-complete audit of the fixed System9 completion definition. It binds "
            "the strongest current global H7 inequality, the exact unmodified finite-Sobolev "
            "slice no-go, and the witness-local System8 recurrence boundary without promoting "
            "any of them beyond their proved scope. No global H7 proof, completion-grade "
            "obstruction, lifespan, rejection, promotion, or observation is claimed."
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
        "full_direction_replay_audit",
        "exact_remaining_contract",
        "negative_controls",
        "source_bindings",
        "local_bindings",
        "scope",
        "content_sha256",
    }:
        raise ValueError("System9 receipt boundary changed")
    if result.get("content_sha256") != _content_sha(result):
        raise ValueError("System9 receipt content seal changed")
    if (
        result.get("schema_version") != RECEIPT_SCHEMA
        or result.get("campaign_id") != EXPECTED_CAMPAIGN_ID
        or result.get("status")
        != "block_all_12_candidate_global_H7_completion_or_completion_grade_obstruction"
        or result.get("decision") != "BLOCK_SYSTEM9"
        or result.get("counts") != EXPECTED_COUNTS
        or result.get("claims") != EXPECTED_CLAIMS
        or result.get("completion_contract", {}).get("scoped_no_go_is_completion_grade")
        or result.get("completion_contract", {}).get("witness_local_recurrence_is_full_direction")
    ):
        raise ValueError("System9 receipt boundary changed")
    records = result.get("candidate_records")
    if not isinstance(records, list) or len(records) != EXPECTED_CANDIDATES:
        raise ValueError("System9 receipt candidate set changed")
    if len({item.get("candidate_id") for item in records}) != EXPECTED_CANDIDATES:
        raise ValueError("System9 receipt candidate identity changed")
    for item in records:
        obstruction = item.get("obstruction_branch", {})
        replay = item.get("shared_System8_scope", {})
        if (
            item.get("decision") != "BLOCK_SYSTEM9"
            or item.get("completion_grade")
            or item.get("candidate_rejection_authorized")
            or not obstruction.get("finite_unmodified_Sobolev_no_go_proved")
            or any(
                obstruction.get(key)
                for key in (
                    "full_tensor_cancellation_excluded",
                    "modified_energy_excluded",
                    "Nash_Moser_or_derivative_loss_evolution_excluded",
                    "analytic_or_Gevrey_closure_excluded",
                    "full_direction_completion_obstruction_proved",
                )
            )
            or replay.get("accepted_full_direction_replays") != 0
            or replay.get("required_full_direction_replays") != 15
            or replay.get("accepted_global_coordinate_free_recurrence")
        ):
            raise ValueError("System9 receipt candidate classification changed")
    direction = result.get("full_direction_replay_audit", {})
    if (
        direction.get("required_evaluations") != list(EXPECTED_DIRECTIONS)
        or direction.get("accepted_alternative_recurrence_evaluations") != []
        or direction.get("witness_local_only_evaluations") != ["subset_2"]
        or direction.get("full_direction_completion_replay_closed")
    ):
        raise ValueError("System9 receipt direction boundary changed")
    controls = result.get("negative_controls", {})
    if set(controls) != {
        "omit_one_candidate",
        "erase_nonzero_candidate_D2_slice",
        "promote_scoped_no_go_to_completion",
        "corrupt_full_direction_manifest",
        "promote_witness_local_recurrence_to_full_direction",
    } or not all(control.get("rejected") is True for control in controls.values()):
        raise ValueError("System9 negative controls changed")
    if any(result.get("claims", {}).values()):
        raise ValueError("System9 forbidden claim opened")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sigma-system9-global-h7-gate")
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
