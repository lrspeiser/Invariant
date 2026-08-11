"""Deterministic-feature Poisson-selector no-go for KS action hypotheses."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path
from typing import Any

CONFIG_SCHEMA = "sigma-kastner-schlatter-deterministic-feature-selector-no-go-config-1.0"
RESULT_SCHEMA = "sigma-kastner-schlatter-deterministic-feature-selector-no-go-1.0"
PDF_SHA256 = "c2f671293d07b21397e745da00a3ce1a2193c00da647a2ebf4147612b76c1780"
FIRST_BLOCKER = (
    "no_registered_paper_QED_or_action_stochastic_feature_outside_the_deterministic_"
    "action_intensity_fiber_to_select_Poisson_over_same_feature_Cox"
)
BRANCHES = [("eq35_middle_h", "1/2"), ("eq35_printed_planck", "1/4")]

EXPECTED_COUNTS = {
    "candidate_actions": 2,
    "candidate_blocked": 2,
    "candidate_pass": 0,
    "candidate_reject": 0,
    "deterministic_feature_fibers": 2,
    "same_feature_Poisson_Cox_pairs": 2,
    "deterministic_factor_selector_no_go_theorems": 2,
    "exact_one_cell_law_separation_witnesses": 2,
    "registered_stochastic_features_outside_fiber": 0,
    "paper_or_QED_selector_derivations": 0,
    "universal_Poisson_characterizations_closed_by_source": 0,
    "theory_ontology_observational_pass": 0,
}

EXPECTED_CLAIM_SEALS = {
    "paper_QED_stochastic_feature_registered": False,
    "candidate_action_stochastic_law_derived": False,
    "candidate_action_selects_Poisson": False,
    "candidate_action_rejected": False,
    "single_cell_witness_claimed_to_characterize_Poisson": False,
    "detector_event_equals_transaction_proven": False,
    "transaction_ontology_validated": False,
    "observational_pass": False,
    "scientific_test_pass": False,
    "theory_validity_claimed": False,
    "dark_sector_elimination_proven": False,
}


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
        raise ValueError("deterministic-feature selector path escapes repository") from error
    return path


def _bound_artifact(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    if set(binding) != {"path", "file_sha256", "content_sha256"}:
        raise ValueError("deterministic-feature predecessor binding shape changed")
    path = _inside(root, str(binding["path"]))
    if not path.is_file() or _file_sha(path) != binding["file_sha256"]:
        raise ValueError("deterministic-feature predecessor file hash mismatch")
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("content_sha256") != binding["content_sha256"]
        or _content_sha(value) != binding["content_sha256"]
    ):
        raise ValueError("deterministic-feature predecessor content hash mismatch")
    return value


def _validate_config(config: Mapping[str, Any]) -> None:
    if (
        set(config)
        != {
            "schema_version",
            "campaign_id",
            "output_path",
            "predecessors",
            "deterministic_feature_domain",
            "admission_policy",
            "seals",
        }
        or config.get("schema_version") != CONFIG_SCHEMA
    ):
        raise ValueError("deterministic-feature config shape changed")
    if set(config["predecessors"]) != {
        "canonical_probability_space",
        "deterministic_compensator",
        "candidate_action_completion",
        "equation_graph",
        "qed_actualization_audit",
        "actualization_history_map",
    }:
        raise ValueError("deterministic-feature predecessor set changed")
    if config.get("deterministic_feature_domain") != {
        "candidate_variables": ["g_mn", "phi"],
        "candidate_intensity": "mu_g_phi(dx)=q0*exp(phi(x))*dVol_g(x)",
        "registered_features": [
            "compiler_action_and_branch_normalization",
            "metric_and_scalar_Euler_Lagrange_equations",
            "positive_intensity_measure_mu_g_phi",
            "conditional_mean_measure_E_N_equals_mu_g_phi",
        ],
        "excluded_stochastic_features": [
            "probability_law_on_counting_measures",
            "second_and_higher_factorial_cumulants",
            "Mecke_or_Papangelou_kernel",
            "independent_increment_or_compensator_identity",
        ],
    }:
        raise ValueError("deterministic-feature domain changed")
    if config.get("admission_policy") != {
        "deterministic_feature_factorization_is_a_valid_no_go_scope": True,
        "single_cell_second_factorial_witness_characterizes_all_Poisson_laws": False,
        "compiler_probability_space_counts_as_physical_selection": False,
        "candidate_action_rejection_allowed": False,
    }:
        raise ValueError("deterministic-feature admission policy changed")
    if not config.get("seals") or any(config["seals"].values()):
        raise ValueError("deterministic-feature seal opened")


def _validate_predecessors(predecessors: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    canonical = predecessors["canonical_probability_space"]
    records = canonical.get("candidate_records", [])
    if (
        canonical.get("decision_counts") != {"blocked": 2, "pass": 0, "reject": 0}
        or [(row.get("branch_id"), row.get("beta")) for row in records] != BRANCHES
        or any(
            row.get("candidate_action_selects_Poisson_completion") is not False
            or row.get("same_action_Cox_completion_witness") is not True
            or row.get("candidate_action_rejection_authorized") is not False
            for row in records
        )
    ):
        raise ValueError("canonical probability-space boundary changed")

    compensator = predecessors["deterministic_compensator"]
    witness = compensator.get("exact_controls", {}).get(
        "same_action_Poisson_Cox_nonidentifiability", {}
    )
    if (
        witness.get("same_deterministic_candidate_action") is not True
        or witness.get("same_unconditional_mean_measure") is not True
        or witness.get("action_selects_between_completions") is not False
    ):
        raise ValueError("deterministic-compensator nonidentifiability changed")

    actions = predecessors["candidate_action_completion"].get("completion_hypotheses", [])
    if [(row.get("branch_id"), row.get("beta")) for row in actions] != BRANCHES or any(
        row.get("paper_authorship_or_derivation") is not False
        or row.get("candidate_action", {}).get("stochastic_law_derived_by_action") is not False
        or row.get("conditional_stochastic_completion", {}).get("derived_from_local_action")
        is not False
        for row in actions
    ):
        raise ValueError("candidate-action stochastic attribution changed")

    qed_records = predecessors["qed_actualization_audit"].get("candidate_records", [])
    history_records = predecessors["actualization_history_map"].get("candidate_records", [])
    if (
        [(row.get("branch_id"), row.get("beta")) for row in qed_records] != BRANCHES
        or any(
            row.get("paper_or_QED_channel_kernel_registered") is not False for row in qed_records
        )
        or [(row.get("branch_id"), row.get("beta")) for row in history_records] != BRANCHES
        or any(row.get("paper_or_QED_kernel_selected") is not False for row in history_records)
    ):
        raise ValueError("paper/QED stochastic selector evidence changed")
    graph_seals = predecessors["equation_graph"].get("claim_seals", {})
    if graph_seals.get("variational_derivation_registered") is not False:
        raise ValueError("equation graph variational boundary changed")
    return actions


def _exact_witness() -> dict[str, Any]:
    cell_mean = Fraction(2)
    z_values = [Fraction(1, 2), Fraction(3, 2)]
    ez = sum(z_values, Fraction()) / len(z_values)
    ez2 = sum((z * z for z in z_values), Fraction()) / len(z_values)
    poisson_variance = cell_mean
    cox_variance = cell_mean * ez + cell_mean**2 * (ez2 - ez**2)
    poisson_factorial_second = cell_mean**2
    cox_factorial_second = cell_mean**2 * ez2
    return {
        "declared_cell": "B with mu_g_phi(B)=2",
        "Poisson_completion": "N_P|g,phi ~ PRM(mu_g_phi)",
        "Cox_completion": ("Z in {1/2,3/2} equiprobable and N_C|g,phi,Z ~ PRM(Z*mu_g_phi)"),
        "mixing_moments": {"E_Z": str(ez), "E_Z_squared": str(ez2)},
        "shared_registered_deterministic_features": {
            "candidate_action": True,
            "Euler_Lagrange_equations": True,
            "intensity_field_mu_g_phi": True,
            "conditional_mean_measure_given_g_phi": str(cell_mean * ez),
        },
        "law_separation": {
            "Poisson_variance": str(poisson_variance),
            "Cox_variance": str(cox_variance),
            "Poisson_second_factorial_moment": str(poisson_factorial_second),
            "Cox_second_factorial_moment": str(cox_factorial_second),
            "Poisson_second_factorial_cumulant": "0",
            "Cox_second_factorial_cumulant": str(cox_factorial_second - (cell_mean * ez) ** 2),
            "Poisson_void_probability": "exp(-2)",
            "Cox_void_probability": "exp(-2)*cosh(1)",
            "laws_are_distinct": cox_factorial_second != poisson_factorial_second,
        },
    }


def _factorization_theorem() -> dict[str, Any]:
    return {
        "theorem_name": "registered_deterministic_feature_factorization_no_go",
        "feature_map": (
            "D(P)=(candidate action and beta, metric/scalar EL equations, mu_g_phi, "
            "E_P[N(.)|g,phi])"
        ),
        "selector_class": "every selector S with S(P)=s(D(P)) for some deterministic s",
        "premise": "D(P_Poisson)=D(P_Cox) for the exact registered witness pair",
        "proof": (
            "Substitution gives S(P_Poisson)=s(D(P_Poisson))=s(D(P_Cox))="
            "S(P_Cox); therefore S cannot uniquely select P_Poisson on this fiber."
        ),
        "conclusion": (
            "no selector using only registered deterministic action, EL, intensity, and mean-"
            "measure data can select Poisson over the same-feature Cox completion"
        ),
        "scope_limit": (
            "the theorem excludes only selectors factoring through D; it neither proves Cox "
            "physical nor rules out a new QED stochastic kernel"
        ),
    }


def _escape_contract() -> dict[str, Any]:
    return {
        "first_new_premise": (
            "a source-bound candidate-specific stochastic feature not measurable through D"
        ),
        "witness_separating_but_not_universally_characterizing_option": (
            "a derived second factorial cumulant on B; zero versus one separates the exact "
            "Poisson/Cox pair but one cell alone does not characterize a Poisson random measure"
        ),
        "universally_sufficient_options": [
            "a paper/QED-derived Mecke identity for all nonnegative measurable test functionals",
            "a paper/QED-derived deterministic Papangelou kernel lambda(x,eta)=q0*exp(phi(x))",
            "a paper/QED-derived set-indexed independent-increment family with Poisson marginals",
            "a paper/QED-derived deterministic-compensator martingale identity on the physical filtration",
        ],
        "registered_options_closed": 0,
        "compiler_canonical_probability_space_is_not_source_selection": True,
    }


def _candidate_record(action: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "branch_id": action["branch_id"],
        "beta": action["beta"],
        "compiler_authored_action": True,
        "deterministic_feature_fiber_no_go": "pass",
        "exact_same_feature_Poisson_Cox_pair": True,
        "registered_stochastic_feature_outside_fiber": False,
        "paper_or_QED_selector_derived": False,
        "candidate_action_selects_Poisson": False,
        "candidate_action_rejection_authorized": False,
        "candidate_decision": "blocked",
        "first_blocker": FIRST_BLOCKER,
    }


def _validate_result(result: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "campaign_id",
        "scope",
        "source_bindings",
        "deterministic_feature_domain",
        "factorization_no_go",
        "exact_controls",
        "minimal_escape_contract",
        "candidate_records",
        "gate_counts",
        "decision_counts",
        "decision",
        "first_blocker",
        "secondary_blockers",
        "claim_seals",
        "data_seals",
        "content_sha256",
    }
    if set(result) != required or result.get("schema_version") != RESULT_SCHEMA:
        raise ValueError("deterministic-feature result shape changed")
    if result.get("decision_counts") != {"pass": 0, "reject": 0, "blocked": 2}:
        raise ValueError("deterministic-feature candidate partition changed")
    if result.get("gate_counts") != EXPECTED_COUNTS:
        raise ValueError("deterministic-feature gate counts changed")
    if result.get("factorization_no_go") != _factorization_theorem():
        raise ValueError("deterministic-feature theorem changed")
    if (
        result.get("exact_controls", {}).get("same_feature_distinct_law_witness")
        != _exact_witness()
    ):
        raise ValueError("deterministic-feature exact witness changed")
    if result.get("minimal_escape_contract") != _escape_contract():
        raise ValueError("deterministic-feature escape contract changed")
    records = result.get("candidate_records", [])
    if [(row.get("branch_id"), row.get("beta")) for row in records] != BRANCHES or any(
        row.get("deterministic_feature_fiber_no_go") != "pass"
        or row.get("registered_stochastic_feature_outside_fiber") is not False
        or row.get("candidate_action_selects_Poisson") is not False
        or row.get("candidate_action_rejection_authorized") is not False
        or row.get("candidate_decision") != "blocked"
        or row.get("first_blocker") != FIRST_BLOCKER
        for row in records
    ):
        raise ValueError("deterministic-feature candidate boundary changed")
    if result.get("first_blocker") != FIRST_BLOCKER:
        raise ValueError("deterministic-feature blocker changed")
    if result.get("claim_seals") != EXPECTED_CLAIM_SEALS or any(result["data_seals"].values()):
        raise ValueError("deterministic-feature seal changed")
    if (
        result.get("content_sha256") is not None
        and _content_sha(result) != result["content_sha256"]
    ):
        raise ValueError("deterministic-feature content hash mismatch")


def build_gate(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    root = config_path.parents[1]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    predecessors = {
        label: _bound_artifact(root, binding) for label, binding in config["predecessors"].items()
    }
    actions = _validate_predecessors(predecessors)
    source_path = Path(__file__).resolve()
    test_path = root / "tests/test_kastner_schlatter_deterministic_feature_selector_no_go.py"
    witness = _exact_witness()
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": config["campaign_id"],
        "scope": (
            "candidate-bound no-go for Poisson selection by any rule factoring only through "
            "the registered deterministic action, EL equations, intensity, and mean measure; "
            "no paper/QED stochastic derivation, action rejection, ontology, or observation inferred"
        ),
        "source_bindings": {
            **config["predecessors"],
            "primary_pdf_sha256": PDF_SHA256,
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
        "deterministic_feature_domain": config["deterministic_feature_domain"],
        "factorization_no_go": _factorization_theorem(),
        "exact_controls": {
            "same_feature_distinct_law_witness": witness,
            "mean_only_negative_control": {
                "mutation": "infer a unique point-process law from E[N(.)|g,phi]=mu_g_phi",
                "rejected": True,
                "counterexample": "the exact same-feature Poisson/Cox pair",
            },
            "single_cell_overreach_negative_control": {
                "mutation": "treat one zero second-factorial-cumulant cell as a universal Poisson characterization",
                "rejected": True,
                "reason": "a universal random-measure law requires a set-indexed joint characterization",
            },
        },
        "minimal_escape_contract": _escape_contract(),
        "candidate_records": [_candidate_record(action) for action in actions],
        "gate_counts": dict(EXPECTED_COUNTS),
        "decision_counts": {"pass": 0, "reject": 0, "blocked": 2},
        "decision": "deterministic_feature_selector_class_ruled_out_stochastic_selection_blocked",
        "first_blocker": FIRST_BLOCKER,
        "secondary_blockers": [
            "no_source_bound_second_or_higher_factorial_cumulant_for_actualizations",
            "no_paper_QED_Mecke_Papangelou_or_physical_compensator_identity",
            "no_measurable_QED_history_pushforward_probability_law",
            "no_operational_transaction_event_filtration_or_exposure_bundle",
        ],
        "claim_seals": dict(EXPECTED_CLAIM_SEALS),
        "data_seals": dict(config["seals"]),
        "content_sha256": None,
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
        else config_path.parents[1] / config["output_path"]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
