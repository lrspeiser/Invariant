"""Operational transaction-event observation contract and exact identifiability limits."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

CONFIG_SCHEMA = "sigma-kastner-schlatter-transaction-event-observable-exposure-config-1.0"
RESULT_SCHEMA = "sigma-kastner-schlatter-transaction-event-observable-exposure-1.0"
FIRST_BLOCKER = (
    "no_registered_detector_level_transaction_event_schema_exposure_response_background_"
    "or_calibration_manifest"
)
SECOND_BLOCKER = (
    "latent_transaction_rate_is_nonidentifiable_from_observed_intensity_without_known_"
    "acceptance_injective_response_and_background"
)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _content_sha(value: Mapping[str, Any]) -> str:
    return _sha({key: item for key, item in value.items() if key != "content_sha256"})


def _inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("transaction-event observable path escapes repository") from error
    return path


def _bound_artifact(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    if set(binding) != {"path", "file_sha256", "content_sha256"}:
        raise ValueError("transaction-event predecessor binding shape changed")
    path = _inside(root, str(binding["path"]))
    if not path.is_file() or _file_sha(path) != binding["file_sha256"]:
        raise ValueError("transaction-event predecessor file hash mismatch")
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("content_sha256") != binding["content_sha256"]
        or _content_sha(value) != binding["content_sha256"]
    ):
        raise ValueError("transaction-event predecessor content hash mismatch")
    return value


def _validate_config(config: Mapping[str, Any]) -> None:
    if (
        set(config)
        != {
            "schema_version",
            "campaign_id",
            "output_path",
            "predecessors",
            "operator_domain",
            "admission_policy",
            "seals",
        }
        or config.get("schema_version") != CONFIG_SCHEMA
    ):
        raise ValueError("transaction-event observable config shape changed")
    if set(config.get("predecessors", {})) != {
        "actualization_history_map",
        "observational_readiness",
    }:
        raise ValueError("transaction-event predecessor set changed")
    if config.get("operator_domain") != {
        "latent_input": "locally finite absorption-endpoint counting measure N on (M,B(M))",
        "observed_output": "locally finite detector-record counting measure Y on (Y,B(Y))",
        "window": "declared relatively compact W subset M with frame and four-volume convention",
        "acceptance": "known measurable a:W->[0,1]",
        "response": "known Markov kernel R(dy|x) from W to observed record space Y",
        "background": "known locally finite measure b(dy)",
    }:
        raise ValueError("transaction-event operator domain changed")
    if config.get("admission_policy") != {
        "compiler_authored_operational_contract_allowed": True,
        "detector_event_equated_to_transaction_without_source_contract": False,
        "latent_rate_identified_without_calibration": False,
        "real_observations_opened": False,
        "candidate_action_rejection_allowed": False,
    }:
        raise ValueError("transaction-event admission policy changed")
    if any(config.get("seals", {}).values()):
        raise ValueError("transaction-event observable seal opened")


def _validate_predecessors(
    history: Mapping[str, Any], readiness: Mapping[str, Any]
) -> list[Mapping[str, Any]]:
    if (
        history.get("decision_counts") != {"pass": 0, "reject": 0, "blocked": 2}
        or history.get("first_blocker")
        != "no_paper_registered_locally_finite_measurable_actualization_history_space_or_set_indexed_counting_map"
        or history.get("compiler_conditional_count_map", {}).get("selects_a_probability_law")
        is not False
        or any(history.get("claim_seals", {}).values())
        or any(history.get("data_seals", {}).values())
    ):
        raise ValueError("actualization-history predecessor boundary changed")
    records = history.get("candidate_records", [])
    if [record.get("branch_id") for record in records] != [
        "eq35_middle_h",
        "eq35_printed_planck",
    ]:
        raise ValueError("actualization-history candidate ordering changed")

    transaction_fields = {
        item.get("field_id"): item
        for item in readiness.get("field_registry", [])
        if item.get("lane_id") == "transaction_poisson"
    }
    required_missing = {
        "transaction.actualized_event_definition",
        "transaction.detector_protocol",
        "transaction.event_spacetime_localization",
        "transaction.exposure_construction",
        "transaction.acceptance_selection",
        "transaction.efficiency_deadtime_duplicates",
        "transaction.background_false_positive",
    }
    if (
        readiness.get("decision") != "blocked_registration_incomplete_observations_sealed"
        or readiness.get("lane_decisions", {}).get("transaction_poisson")
        != "blocked_no_operational_transaction_event_or_exposure_definition"
        or not required_missing.issubset(transaction_fields)
        or any(
            transaction_fields[field].get("status") != "missing_required"
            for field in required_missing
        )
        or readiness.get("observational_access_count") != 0
        or readiness.get("real_data_bundle_count") != 0
        or readiness.get("data_seals", {}).get("observations_opened") is not False
        or readiness.get("data_seals", {}).get("transaction_event_observations_opened") is not False
    ):
        raise ValueError("observational-readiness transaction boundary changed")
    return records


def _minimal_contract() -> dict[str, Any]:
    return {
        "attribution": "compiler_authored_minimal_operational_contract_not_supplied_by_the_paper",
        "latent_event_contract": {
            "event_kind": "actualized absorption endpoint represented once",
            "required_source_bridge": (
                "registered rule distinguishing an RTI actualization from an ordinary detector click"
            ),
            "status": "missing",
        },
        "observation_window_and_exposure": {
            "window": "relatively compact W subset M",
            "frame_and_volume": "declared coordinates/frame and invariant four-volume convention",
            "exposure": "E_A=Integral_W a(x)*1_A(x)*dVol_g(x)",
            "acceptance_range": "0<=a(x)<=1",
            "required_metadata": ["live time", "dead time", "window mask", "units", "version"],
            "status": "missing",
        },
        "observed_record_schema": {
            "record": [
                "immutable event id",
                "timestamp and frame",
                "spatial locator or localization posterior",
                "detector/channel id",
                "selection flags",
                "quality flags",
            ],
            "counting_rules": [
                "one absorption representative",
                "deduplication",
                "pileup",
                "dead time",
            ],
            "status": "missing",
        },
        "response_and_background": {
            "response": "normalized Markov kernel R(dy|x)",
            "background": "locally finite false-positive intensity b(dy)",
            "calibration": "immutable calibration manifest and uncertainty model",
            "status": "missing",
        },
        "observation_operator": {
            "construction": (
                "independently accept each latent x with probability a(x), mark its observed "
                "record by R(dy|x), then superpose an independent background PRM(b)"
            ),
            "conditional_Laplace_functional": (
                "E[exp(-Y(f))|N]=exp(-Integral(1-exp(-f(y)))*b(dy))*"
                "product_{x in N cap W}[1-a(x)+a(x)*Integral exp(-f(y))*R(dy|x)]"
            ),
            "observed_mean_measure": ("nu(dy)=b(dy)+Integral_W a(x)*R(dy|x)*mu(dx)"),
            "Poisson_mapping_theorem": (
                "if N~PRM(mu) and all operational premises hold, Y~PRM(nu)"
            ),
            "converse_from_mean_only": False,
        },
        "analysis_manifest": {
            "required": [
                "calibration split",
                "held-out split",
                "independent exposure-block ids",
                "frozen likelihood and diagnostics",
                "bundle and manifest hashes",
            ],
            "status": "missing",
        },
    }


def _identifiability_theorem() -> dict[str, Any]:
    return {
        "forward_operator": "T_{a,R,b}(mu)=b+R_*(a*mu)",
        "identified_from_first_moment": "only nu=T_{a,R,b}(mu)",
        "necessary_conditions_for_latent_mean_identification": [
            "a and b are independently calibrated and fixed",
            "a is bounded away from zero on the declared latent domain",
            "the response pushforward R_* is injective on the declared model class",
            "window, frame, four-volume, and units are fixed",
        ],
        "conditional_sufficiency": (
            "under those premises, mu is unique on the declared model class whenever "
            "R_*(a*mu)=nu-b has a solution"
        ),
        "law_identification_warning": (
            "identifying mu from the mean does not identify the latent point-process law"
        ),
        "current_contract_satisfies_conditions": False,
        "theory_or_ontology_consequence": False,
    }


def _exact_controls() -> dict[str, Any]:
    return {
        "forward_two_cell_positive_control": {
            "latent_mu": ["2", "3"],
            "acceptance": ["1/2", "1/3"],
            "identity_response": True,
            "background": ["1/4", "1/2"],
            "observed_nu": ["5/4", "3/2"],
            "pass": True,
        },
        "rate_efficiency_scaling_no_go": {
            "model_A": {"latent_mu": "2", "acceptance": "1/2"},
            "model_B": {"latent_mu": "4", "acceptance": "1/4"},
            "observed_signal_mean_both": "1",
            "latent_rates_different": True,
            "identifiable_without_acceptance_calibration": False,
        },
        "signal_background_no_go": {
            "model_A": {"latent_mu": "1", "acceptance": "1", "background": "0"},
            "model_B": {"latent_mu": "1/2", "acceptance": "1", "background": "1/2"},
            "observed_mean_both": "1",
            "identifiable_without_background_calibration": False,
        },
        "rank_deficient_response_no_go": {
            "response_matrix": [["1/2", "1/2"], ["1/2", "1/2"]],
            "latent_A": ["2", "0"],
            "latent_B": ["0", "2"],
            "observed_mean_both": ["1", "1"],
            "response_rank": 1,
            "latent_spatial_measure_identifiable": False,
        },
        "same_mean_different_law_after_thinning": {
            "latent_mean": "2",
            "acceptance": "1/2",
            "observed_mean_both": "1",
            "Poisson_observed_second_factorial_moment": "1",
            "Cox_Z_half_threehalves_observed_second_factorial_moment": "5/4",
            "mean_alone_selects_Poisson": False,
        },
        "detector_click_equivalence_negative_control": {
            "mutation": "declare every photon detector click to be an RTI actualized transaction",
            "rejected": True,
            "reason": "the paper-to-detector ontology bridge is not registered",
        },
        "zero_acceptance_negative_control": {
            "mutation": "infer latent mu on a region where a=0",
            "rejected": True,
            "reason": "the forward operator has a null space on zero-acceptance regions",
        },
    }


def _obligation_ledger() -> list[dict[str, Any]]:
    return [
        {"obligation": "detector_level_transaction_event_definition", "status": "missing"},
        {"obligation": "observation_window_frame_and_four_volume", "status": "missing"},
        {"obligation": "timestamp_and_spacetime_localization_schema", "status": "missing"},
        {"obligation": "acceptance_and_selection_function", "status": "missing"},
        {"obligation": "dead_time_pileup_and_deduplication", "status": "missing"},
        {"obligation": "response_kernel_and_localization_uncertainty", "status": "missing"},
        {"obligation": "background_and_false_positive_measure", "status": "missing"},
        {"obligation": "calibration_manifest_and_uncertainties", "status": "missing"},
        {"obligation": "split_manifest_and_independent_exposure_blocks", "status": "missing"},
        {"obligation": "frozen_likelihood_diagnostics_and_thresholds", "status": "missing"},
        {"obligation": "observation_authorization_and_bundle_hashes", "status": "sealed"},
    ]


def _candidate_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "branch_id": record["branch_id"],
        "beta": record["beta"],
        "operational_contract_is_branch_independent": True,
        "latent_history_map_available_conditionally": True,
        "operational_event_exposure_contract_registered_from_data": False,
        "latent_rate_identifiable": False,
        "observations_opened": False,
        "candidate_action_rejection_authorized": False,
        "candidate_decision": "blocked",
        "first_blocker": FIRST_BLOCKER,
    }


def _validate_result(result: Mapping[str, Any]) -> None:
    if result.get("schema_version") != RESULT_SCHEMA:
        raise ValueError("transaction-event result schema changed")
    if result.get("decision_counts") != {"pass": 0, "reject": 0, "blocked": 2}:
        raise ValueError("transaction-event candidate partition changed")
    if result.get("gate_counts") != {
        "candidate_actions": 2,
        "operational_obligations": 11,
        "operational_obligations_registered": 0,
        "operational_obligations_missing": 10,
        "operational_obligations_sealed": 1,
        "compiler_observation_operator_contracts": 1,
        "exact_nonidentifiability_witnesses": 4,
        "real_observation_bundles": 0,
        "latent_rate_identification_pass": 0,
        "candidate_action_reject": 0,
        "theory_ontology_observational_pass": 0,
    }:
        raise ValueError("transaction-event gate counts changed")
    operator = result.get("minimal_operational_contract", {}).get("observation_operator", {})
    if operator.get("converse_from_mean_only") is not False or "b(dy)+Integral_W" not in str(
        operator.get("observed_mean_measure")
    ):
        raise ValueError("transaction-event observation operator changed")
    theorem = result.get("identifiability_theorem", {})
    if theorem.get("current_contract_satisfies_conditions") is not False:
        raise ValueError("transaction-event identifiability overclaim")
    records = result.get("candidate_records", [])
    if len(records) != 2 or any(
        record.get("operational_event_exposure_contract_registered_from_data") for record in records
    ):
        raise ValueError("transaction-event data registration overclaim")
    if any(record.get("latent_rate_identifiable") for record in records):
        raise ValueError("transaction-event latent-rate overclaim")
    if any(record.get("candidate_action_rejection_authorized") for record in records):
        raise ValueError("transaction-event gate overreached to action rejection")
    if result.get("first_blocker") != FIRST_BLOCKER:
        raise ValueError("transaction-event first blocker changed")
    if any(result.get("claim_seals", {}).values()) or any(result.get("data_seals", {}).values()):
        raise ValueError("transaction-event seal opened")
    if (
        result.get("content_sha256") is not None
        and _content_sha(result) != result["content_sha256"]
    ):
        raise ValueError("transaction-event content hash mismatch")


def build_gate(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    root = config_path.parents[1]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    predecessors = {
        label: _bound_artifact(root, binding) for label, binding in config["predecessors"].items()
    }
    records = _validate_predecessors(
        predecessors["actualization_history_map"], predecessors["observational_readiness"]
    )
    source_path = Path(__file__).resolve()
    test_path = root / "tests/test_kastner_schlatter_transaction_event_observable_exposure_gate.py"
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": config["campaign_id"],
        "scope": (
            "synthetic-only operational observation operator and exact identifiability audit; "
            "no detector, transaction-event, astrophysical, or cosmological observations opened"
        ),
        "synthetic_only": True,
        "source_bindings": {
            **config["predecessors"],
            "config": {
                "path": config_path.relative_to(root).as_posix(),
                "file_sha256": _file_sha(config_path),
            },
            "source": {
                "path": source_path.relative_to(root).as_posix(),
                "file_sha256": _file_sha(source_path),
            },
            "test": {
                "path": test_path.relative_to(root).as_posix(),
                "file_sha256": _file_sha(test_path),
            },
        },
        "operator_domain": config["operator_domain"],
        "minimal_operational_contract": _minimal_contract(),
        "operational_obligation_ledger": _obligation_ledger(),
        "identifiability_theorem": _identifiability_theorem(),
        "exact_controls": _exact_controls(),
        "candidate_records": [_candidate_record(record) for record in records],
        "gate_counts": {
            "candidate_actions": 2,
            "operational_obligations": 11,
            "operational_obligations_registered": 0,
            "operational_obligations_missing": 10,
            "operational_obligations_sealed": 1,
            "compiler_observation_operator_contracts": 1,
            "exact_nonidentifiability_witnesses": 4,
            "real_observation_bundles": 0,
            "latent_rate_identification_pass": 0,
            "candidate_action_reject": 0,
            "theory_ontology_observational_pass": 0,
        },
        "decision_counts": {"pass": 0, "reject": 0, "blocked": 2},
        "decision": "operational_operator_specified_but_data_bridge_and_latent_identification_blocked",
        "first_blocker": FIRST_BLOCKER,
        "secondary_blockers": [
            SECOND_BLOCKER,
            "mean_measure_alone_does_not_identify_the_latent_point_process_law",
            "no_registered_independent_exposure_blocks_or_frozen_Poisson_diagnostics",
            "no_observation_authorization_or_hash_bound_transaction_event_bundle",
        ],
        "claim_seals": {
            "detector_event_equals_transaction_proven": False,
            "latent_transaction_rate_identified": False,
            "paper_or_QED_Poisson_kernel_derived": False,
            "transaction_ontology_validated": False,
            "candidate_action_rejected": False,
            "observational_pass": False,
            "scientific_test_pass": False,
            "theory_validity_claimed": False,
            "dark_sector_elimination_proven": False,
        },
        "data_seals": dict(config["seals"]),
    }
    result["content_sha256"] = _content_sha(result)
    _validate_result(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--output")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    result = build_gate(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = (
        Path(args.output).resolve()
        if args.output
        else config_path.parents[1] / str(config["output_path"])
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
