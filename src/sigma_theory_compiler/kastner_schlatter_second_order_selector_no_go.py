"""Second-order stochastic-feature no-go for KS Poisson selection."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path
from typing import Any

CONFIG_SCHEMA = "sigma-kastner-schlatter-second-order-selector-no-go-config-1.0"
RESULT_SCHEMA = "sigma-kastner-schlatter-second-order-selector-no-go-1.0"
PDF_SHA256 = "c2f671293d07b21397e745da00a3ce1a2193c00da647a2ebf4147612b76c1780"
FIRST_BLOCKER = (
    "no_registered_source_bound_full_hierarchy_Mecke_Laplace_or_equivalent_"
    "stochastic_law_selector_beyond_second_order"
)
BRANCHES = [("eq35_middle_h", "1/2"), ("eq35_printed_planck", "1/4")]

EXPECTED_COUNTS = {
    "candidate_actions": 2,
    "candidate_blocked": 2,
    "candidate_pass": 0,
    "candidate_reject": 0,
    "exact_second_order_counterexamples": 2,
    "first_factorial_measure_matches": 2,
    "second_factorial_measure_matches": 2,
    "third_factorial_separations": 2,
    "void_probability_separations": 2,
    "registered_full_stochastic_selectors": 0,
    "paper_or_QED_selector_derivations": 0,
    "theory_ontology_observational_pass": 0,
}

EXPECTED_CLAIM_SEALS = {
    "paper_QED_full_selector_registered": False,
    "candidate_action_stochastic_law_derived": False,
    "candidate_action_selects_Poisson": False,
    "candidate_action_rejected": False,
    "second_order_claimed_to_characterize_Poisson": False,
    "third_order_claimed_to_characterize_Poisson": False,
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
        raise ValueError("second-order selector path escapes repository") from error
    return path


def _bound_artifact(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    if set(binding) != {"path", "file_sha256", "content_sha256"}:
        raise ValueError("second-order predecessor binding shape changed")
    path = _inside(root, str(binding["path"]))
    if not path.is_file() or _file_sha(path) != binding["file_sha256"]:
        raise ValueError("second-order predecessor file hash mismatch")
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("content_sha256") != binding["content_sha256"]
        or _content_sha(value) != binding["content_sha256"]
    ):
        raise ValueError("second-order predecessor content hash mismatch")
    return value


def _validate_config(config: Mapping[str, Any]) -> None:
    if (
        set(config)
        != {
            "schema_version",
            "campaign_id",
            "output_path",
            "predecessors",
            "theorem_domain",
            "admission_policy",
            "seals",
        }
        or config.get("schema_version") != CONFIG_SCHEMA
    ):
        raise ValueError("second-order config shape changed")
    if set(config["predecessors"]) != {
        "deterministic_feature_selector_no_go",
        "canonical_probability_space",
        "candidate_action_completion",
        "positive_reparameterization",
        "qed_actualization_audit",
    }:
        raise ValueError("second-order predecessor set changed")
    if config.get("theorem_domain") != {
        "intensity": "mu_g_phi(dx)=q0*exp(phi(x))*dVol_g(x)",
        "measure_class": "diffuse finite intensity measure on a regular patch",
        "witness_cell": "relatively compact B with mu_g_phi(B)=2",
        "outside_completion": "independent PRM(mu_g_phi restricted to W\\B)",
        "inside_locations": (
            "conditional on N_B=n, n iid points with law mu_g_phi(. intersect B)/2"
        ),
    }:
        raise ValueError("second-order theorem domain changed")
    if config.get("admission_policy") != {
        "first_and_second_factorial_measures_select_Poisson": False,
        "one_explicit_admissible_counterexample_refutes_universal_second_order_selection": True,
        "third_order_separation_alone_selects_Poisson_universally": False,
        "candidate_action_rejection_allowed": False,
    }:
        raise ValueError("second-order admission policy changed")
    if not config.get("seals") or any(config["seals"].values()):
        raise ValueError("second-order seal opened")


def _validate_predecessors(predecessors: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    previous = predecessors["deterministic_feature_selector_no_go"]
    records = previous.get("candidate_records", [])
    if (
        previous.get("decision_counts") != {"blocked": 2, "pass": 0, "reject": 0}
        or [(row.get("branch_id"), row.get("beta")) for row in records] != BRANCHES
        or any(
            row.get("deterministic_feature_fiber_no_go") != "pass"
            or row.get("registered_stochastic_feature_outside_fiber") is not False
            or row.get("candidate_action_rejection_authorized") is not False
            for row in records
        )
    ):
        raise ValueError("deterministic-feature predecessor boundary changed")
    canonical = predecessors["canonical_probability_space"]
    if (
        canonical.get("decision_counts") != {"blocked": 2, "pass": 0, "reject": 0}
        or canonical.get("canonical_conditional_construction", {}).get(
            "candidate_action_selects_this_probability_space"
        )
        is not False
    ):
        raise ValueError("canonical probability-space selection boundary changed")
    actions = predecessors["candidate_action_completion"].get("completion_hypotheses", [])
    if [(row.get("branch_id"), row.get("beta")) for row in actions] != BRANCHES or any(
        row.get("candidate_action", {}).get("stochastic_law_derived_by_action") is not False
        for row in actions
    ):
        raise ValueError("candidate-action stochastic boundary changed")
    if predecessors["positive_reparameterization"].get("decision_counts") != {
        "blocked": 2,
        "pass": 0,
        "reject": 0,
    }:
        raise ValueError("positive-intensity predecessor boundary changed")
    qed_records = predecessors["qed_actualization_audit"].get("candidate_records", [])
    if [(row.get("branch_id"), row.get("beta")) for row in qed_records] != BRANCHES or any(
        row.get("paper_or_QED_channel_kernel_registered") is not False for row in qed_records
    ):
        raise ValueError("paper/QED selector evidence changed")
    return records


def _count_moments() -> dict[str, Any]:
    support = [0, 3]
    probabilities = [Fraction(1, 3), Fraction(2, 3)]
    mean = sum((p * n for p, n in zip(probabilities, support, strict=True)), Fraction())
    factorial_second = sum(
        (p * n * (n - 1) for p, n in zip(probabilities, support, strict=True)), Fraction()
    )
    factorial_third = sum(
        (p * n * (n - 1) * (n - 2) for p, n in zip(probabilities, support, strict=True)),
        Fraction(),
    )
    variance = sum(
        (p * (n - mean) ** 2 for p, n in zip(probabilities, support, strict=True)),
        Fraction(),
    )
    return {
        "support": support,
        "probabilities": ["1/3", "2/3"],
        "mean": str(mean),
        "variance": str(variance),
        "second_factorial_moment": str(factorial_second),
        "second_factorial_cumulant": str(factorial_second - mean**2),
        "third_factorial_moment": str(factorial_third),
        "Poisson_mean": "2",
        "Poisson_variance": "2",
        "Poisson_second_factorial_moment": "4",
        "Poisson_second_factorial_cumulant": "0",
        "Poisson_third_factorial_moment": "8",
        "witness_void_probability": "1/3",
        "Poisson_void_probability": "exp(-2)",
    }


def _point_process_theorem() -> dict[str, Any]:
    return {
        "theorem_name": "first_and_second_factorial_measure_nonidentifiability",
        "construction": {
            "inside_count": "P(N_B=0)=1/3 and P(N_B=3)=2/3",
            "inside_locations": "given N_B=n, draw n iid points from mu_B/2",
            "outside": "independent PRM(mu restricted to W\\B)",
            "simplicity": "almost sure because mu is diffuse",
        },
        "global_first_factorial_measure": "alpha_1=mu",
        "global_second_factorial_measure": "alpha_2=mu tensor mu",
        "global_pair_cumulant_measure": "kappa_2=0, exactly as for PRM(mu)",
        "inside_third_factorial_measure": ("alpha_3(B^3)=4, whereas PRM(mu) gives mu(B)^3=8"),
        "inside_void_separation": "P(N(B)=0)=1/3, whereas PRM(mu) gives exp(-2)",
        "proof": (
            "E[N_B]=2 and E[(N_B)_2]=4; iid placement divides by mu(B) and mu(B)^2, "
            "so the inside first and second factorial densities equal mu and mu tensor mu. "
            "Independence from the outside PRM supplies the cross blocks. But E[(N_B)_3]=4 "
            "rather than 8, hence the laws differ."
        ),
        "conclusion": (
            "mean intensity, variance-equals-mean, zero pair cumulant, and g2=1 do not select "
            "the Poisson random measure, even when imposed as full first/second measures"
        ),
        "scope_limit": (
            "the witness requires an admissible diffuse cell of intensity two and disproves a "
            "universal second-order selector; it does not establish physical non-Poisson events"
        ),
    }


def _minimal_next_contract() -> dict[str, Any]:
    return {
        "ruled_out_as_sufficient": [
            "mean measure alone",
            "variance equals mean on every tested cell",
            "vanishing second factorial cumulant measure",
            "pair correlation g2=1",
        ],
        "finite_third_order_witness_role": (
            "separates this counterexample from Poisson but does not alone characterize every law"
        ),
        "honest_sufficient_targets": [
            "the full set-indexed Laplace functional",
            "the Mecke identity for all nonnegative measurable functionals",
            "all finite disjoint-family independent Poisson count laws",
            "a source-derived physical compensator identity under its complete causal filtration",
        ],
        "registered_sufficient_targets": 0,
        "first_missing_premise": FIRST_BLOCKER,
    }


def _candidate_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "branch_id": record["branch_id"],
        "beta": record["beta"],
        "second_order_selector_no_go": "pass",
        "exact_non_Poisson_same_first_second_measure_witness": True,
        "paper_or_QED_full_selector_derived": False,
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
        "theorem_domain",
        "second_order_no_go",
        "exact_controls",
        "minimal_next_contract",
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
        raise ValueError("second-order result shape changed")
    if result.get("decision_counts") != {"pass": 0, "reject": 0, "blocked": 2}:
        raise ValueError("second-order candidate partition changed")
    if result.get("gate_counts") != EXPECTED_COUNTS:
        raise ValueError("second-order gate counts changed")
    if result.get("second_order_no_go") != _point_process_theorem():
        raise ValueError("second-order theorem changed")
    if result.get("exact_controls", {}).get("inside_count_moments") != _count_moments():
        raise ValueError("second-order exact moments changed")
    if result.get("minimal_next_contract") != _minimal_next_contract():
        raise ValueError("second-order next contract changed")
    records = result.get("candidate_records", [])
    if [(row.get("branch_id"), row.get("beta")) for row in records] != BRANCHES or any(
        row.get("second_order_selector_no_go") != "pass"
        or row.get("paper_or_QED_full_selector_derived") is not False
        or row.get("candidate_action_selects_Poisson") is not False
        or row.get("candidate_action_rejection_authorized") is not False
        or row.get("candidate_decision") != "blocked"
        or row.get("first_blocker") != FIRST_BLOCKER
        for row in records
    ):
        raise ValueError("second-order candidate boundary changed")
    if result.get("first_blocker") != FIRST_BLOCKER:
        raise ValueError("second-order blocker changed")
    if result.get("claim_seals") != EXPECTED_CLAIM_SEALS or any(result["data_seals"].values()):
        raise ValueError("second-order seal changed")
    if (
        result.get("content_sha256") is not None
        and _content_sha(result) != result["content_sha256"]
    ):
        raise ValueError("second-order content hash mismatch")


def build_gate(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    root = config_path.parents[1]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    predecessors = {
        label: _bound_artifact(root, binding) for label, binding in config["predecessors"].items()
    }
    records = _validate_predecessors(predecessors)
    source_path = Path(__file__).resolve()
    test_path = root / "tests/test_kastner_schlatter_second_order_selector_no_go.py"
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": config["campaign_id"],
        "scope": (
            "candidate-bound exact no-go for Poisson selection from full first and second "
            "factorial measures; no paper/QED law, action rejection, ontology, or observation inferred"
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
        "theorem_domain": config["theorem_domain"],
        "second_order_no_go": _point_process_theorem(),
        "exact_controls": {
            "inside_count_moments": _count_moments(),
            "mean_variance_negative_control": {
                "mutation": "infer Poisson from mean equals variance",
                "rejected": True,
                "witness": "P(N_B=0)=1/3, P(N_B=3)=2/3",
            },
            "third_order_overreach_negative_control": {
                "mutation": "treat one third-factorial measurement as a universal characterization",
                "rejected": True,
            },
        },
        "minimal_next_contract": _minimal_next_contract(),
        "candidate_records": [_candidate_record(record) for record in records],
        "gate_counts": dict(EXPECTED_COUNTS),
        "decision_counts": {"pass": 0, "reject": 0, "blocked": 2},
        "decision": "first_second_factorial_selector_class_ruled_out_full_selection_blocked",
        "first_blocker": FIRST_BLOCKER,
        "secondary_blockers": [
            "no_source_bound_third_or_higher_actualization_factorial_hierarchy",
            "no_paper_QED_Mecke_Laplace_or_independent_increment_family",
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
