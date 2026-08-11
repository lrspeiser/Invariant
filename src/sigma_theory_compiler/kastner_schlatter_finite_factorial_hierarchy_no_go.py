"""Finite factorial-hierarchy no-go for KS Poisson selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path
from typing import Any

CONFIG_SCHEMA = "sigma-kastner-schlatter-finite-factorial-hierarchy-no-go-config-1.0"
RESULT_SCHEMA = "sigma-kastner-schlatter-finite-factorial-hierarchy-no-go-1.0"
PDF_SHA256 = "c2f671293d07b21397e745da00a3ce1a2193c00da647a2ebf4147612b76c1780"
FIRST_BLOCKER = (
    "no_registered_source_bound_nonfinite_set_indexed_Laplace_Mecke_or_"
    "independent_increment_selector"
)
BRANCHES = [("eq35_middle_h", "1/2"), ("eq35_printed_planck", "1/4")]

EXPECTED_COUNTS = {
    "candidate_actions": 2,
    "candidate_blocked": 2,
    "candidate_pass": 0,
    "candidate_reject": 0,
    "arbitrary_finite_order_no_go_theorems": 1,
    "candidate_bound_hierarchy_counterexamples": 2,
    "exact_control_orders_checked": 6,
    "exact_control_moment_identities_checked": 27,
    "registered_nonfinite_stochastic_selectors": 0,
    "paper_or_QED_selector_derivations": 0,
    "theory_ontology_observational_pass": 0,
}

EXPECTED_CLAIM_SEALS = {
    "paper_QED_nonfinite_selector_registered": False,
    "candidate_action_stochastic_law_derived": False,
    "candidate_action_selects_Poisson": False,
    "candidate_action_rejected": False,
    "finite_factorial_hierarchy_claimed_sufficient": False,
    "all_moments_claimed_sufficient_without_determinacy": False,
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
        raise ValueError("finite-hierarchy path escapes repository") from error
    return path


def _bound_artifact(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    if set(binding) != {"path", "file_sha256", "content_sha256"}:
        raise ValueError("finite-hierarchy predecessor binding shape changed")
    path = _inside(root, str(binding["path"]))
    if not path.is_file() or _file_sha(path) != binding["file_sha256"]:
        raise ValueError("finite-hierarchy predecessor file hash mismatch")
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("content_sha256") != binding["content_sha256"]
        or _content_sha(value) != binding["content_sha256"]
    ):
        raise ValueError("finite-hierarchy predecessor content hash mismatch")
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
            "exact_control_orders",
            "seals",
        }
        or config.get("schema_version") != CONFIG_SCHEMA
    ):
        raise ValueError("finite-hierarchy config shape changed")
    if set(config["predecessors"]) != {
        "second_order_selector_no_go",
        "canonical_probability_space",
        "candidate_action_completion",
        "positive_reparameterization",
        "qed_actualization_audit",
    }:
        raise ValueError("finite-hierarchy predecessor set changed")
    if config.get("theorem_domain") != {
        "finite_order": "arbitrary integer k>=1",
        "intensity": "mu_g_phi(dx)=q0*exp(phi(x))*dVol_g(x)",
        "measure_class": "diffuse finite intensity measure on a regular patch",
        "witness_cell": "relatively compact B with mu_g_phi(B)=1",
        "inside_locations": ("conditional on N_B=n, n iid points with law mu_g_phi(. intersect B)"),
        "outside_completion": "independent PRM(mu_g_phi restricted to W\\B)",
    }:
        raise ValueError("finite-hierarchy theorem domain changed")
    if config.get("admission_policy") != {
        "any_fixed_finite_factorial_hierarchy_selects_Poisson": False,
        "full_Laplace_Mecke_or_joint_family_is_still_sufficient": True,
        "compiler_theorem_counts_as_paper_or_QED_derivation": False,
        "candidate_action_rejection_allowed": False,
    }:
        raise ValueError("finite-hierarchy admission policy changed")
    if config.get("exact_control_orders") != [1, 2, 3, 4, 5, 6]:
        raise ValueError("finite-hierarchy exact control orders changed")
    if not config.get("seals") or any(config["seals"].values()):
        raise ValueError("finite-hierarchy seal opened")


def _validate_predecessors(predecessors: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    second = predecessors["second_order_selector_no_go"]
    records = second.get("candidate_records", [])
    if (
        second.get("decision_counts") != {"blocked": 2, "pass": 0, "reject": 0}
        or [(row.get("branch_id"), row.get("beta")) for row in records] != BRANCHES
        or any(
            row.get("second_order_selector_no_go") != "pass"
            or row.get("paper_or_QED_full_selector_derived") is not False
            or row.get("candidate_action_rejection_authorized") is not False
            for row in records
        )
    ):
        raise ValueError("second-order predecessor boundary changed")
    canonical = predecessors["canonical_probability_space"]
    if (
        canonical.get("canonical_conditional_construction", {}).get(
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


def _falling(n: int, order: int) -> int:
    if order > n:
        return 0
    return math.prod(range(n - order + 1, n + 1))


def _perturbation_residual(k: int, order: int) -> Fraction:
    denominator = 2 * math.factorial(k + 1)
    return sum(
        (
            Fraction(((-1) ** n) * math.comb(k + 1, n) * _falling(n, order), denominator)
            for n in range(k + 2)
        ),
        Fraction(),
    )


def _control(k: int) -> dict[str, Any]:
    denominator = 2 * math.factorial(k + 1)
    scaled_probabilities = [
        Fraction(1, math.factorial(n)) + Fraction(((-1) ** n) * math.comb(k + 1, n), denominator)
        for n in range(k + 2)
    ]
    residuals = [_perturbation_residual(k, order) for order in range(k + 2)]
    return {
        "k": k,
        "modified_support": [0, k + 1],
        "formula": (
            "q_n=exp(-1)*[1/n!+(-1)^n*C(k+1,n)/(2*(k+1)!)] for 0<=n<=k+1; q_n=exp(-1)/n! otherwise"
        ),
        "scaled_modified_probabilities": [str(value) for value in scaled_probabilities],
        "all_modified_probabilities_positive": all(value > 0 for value in scaled_probabilities),
        "factorial_moment_perturbation_residuals_0_through_k": [
            str(value) for value in residuals[: k + 1]
        ],
        "first_unmatched_factorial_order": k + 1,
        "first_unmatched_residual_coefficient_of_exp_minus_1": str(residuals[k + 1]),
        "void_probability_excess": f"exp(-1)/(2*{math.factorial(k + 1)})",
        "normalization_preserved": residuals[0] == 0,
        "orders_1_through_k_match_Poisson": all(value == 0 for value in residuals[1 : k + 1]),
        "order_k_plus_1_differs": residuals[k + 1] != 0,
    }


def _hierarchy_theorem() -> dict[str, Any]:
    return {
        "theorem_name": "arbitrary_finite_factorial_hierarchy_nonidentifiability",
        "count_law": (
            "for each finite k>=1 perturb Poisson(1) on n=0,...,k+1 by "
            "exp(-1)*(-1)^n*C(k+1,n)/(2*(k+1)!)"
        ),
        "normalization_and_positivity": (
            "the signed perturbation sums to zero; its magnitude is at most half the "
            "Poisson mass at every modified n, so all probabilities stay positive"
        ),
        "finite_difference_identity": ("sum_{n=0}^{k+1}(-1)^n*C(k+1,n)*(n)_j=0 for 0<=j<=k"),
        "first_difference": (
            "at j=k+1 the factorial-moment perturbation is (-1)^(k+1)*exp(-1)/2, which is nonzero"
        ),
        "point_process_lift": (
            "on diffuse B with mu(B)=1, place the perturbed count iid with law mu_B and use "
            "an independent PRM outside; factorial moment measures alpha_j equal mu^tensor_j "
            "for every 1<=j<=k but differ on B^(k+1)"
        ),
        "conclusion": (
            "no selector inspecting only any preassigned finite factorial-moment hierarchy can "
            "uniquely select the candidate Poisson random measure"
        ),
        "scope_limit": (
            "the theorem varies the counterexample with k and does not assert that one law "
            "matches every order, nor that an infinite moment hierarchy is determinate without "
            "an analytic or exponential-integrability premise"
        ),
    }


def _next_contract() -> dict[str, Any]:
    return {
        "finite_order_status": "ruled_out_for_every_fixed_k",
        "smallest_honest_remaining_selector_classes": [
            "full set-indexed Laplace functional on all nonnegative compactly supported f",
            "Mecke identity for all nonnegative measurable test functionals",
            "all finite disjoint-family independent Poisson count laws",
            "physical deterministic-compensator martingale identity on the complete causal filtration",
            "infinite factorial hierarchy plus an explicit moment-determinacy theorem",
        ],
        "registered_remaining_selectors": 0,
        "paper_or_QED_attribution": False,
        "first_missing_premise": FIRST_BLOCKER,
    }


def _candidate_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "branch_id": record["branch_id"],
        "beta": record["beta"],
        "arbitrary_finite_factorial_hierarchy_no_go": "pass",
        "candidate_bound_counterexample_for_every_fixed_k": True,
        "registered_nonfinite_selector": False,
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
        "theorem_domain",
        "finite_hierarchy_no_go",
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
        raise ValueError("finite-hierarchy result shape changed")
    if result.get("decision_counts") != {"pass": 0, "reject": 0, "blocked": 2}:
        raise ValueError("finite-hierarchy candidate partition changed")
    if result.get("gate_counts") != EXPECTED_COUNTS:
        raise ValueError("finite-hierarchy gate counts changed")
    if result.get("finite_hierarchy_no_go") != _hierarchy_theorem():
        raise ValueError("finite-hierarchy theorem changed")
    controls = result.get("exact_controls", {}).get("orders_1_through_6", [])
    if controls != [_control(k) for k in range(1, 7)]:
        raise ValueError("finite-hierarchy controls changed")
    if result.get("minimal_next_contract") != _next_contract():
        raise ValueError("finite-hierarchy next contract changed")
    records = result.get("candidate_records", [])
    if [(row.get("branch_id"), row.get("beta")) for row in records] != BRANCHES or any(
        row.get("arbitrary_finite_factorial_hierarchy_no_go") != "pass"
        or row.get("registered_nonfinite_selector") is not False
        or row.get("candidate_action_selects_Poisson") is not False
        or row.get("candidate_action_rejection_authorized") is not False
        or row.get("candidate_decision") != "blocked"
        or row.get("first_blocker") != FIRST_BLOCKER
        for row in records
    ):
        raise ValueError("finite-hierarchy candidate boundary changed")
    if result.get("first_blocker") != FIRST_BLOCKER:
        raise ValueError("finite-hierarchy blocker changed")
    if result.get("claim_seals") != EXPECTED_CLAIM_SEALS or any(result["data_seals"].values()):
        raise ValueError("finite-hierarchy seal changed")
    if (
        result.get("content_sha256") is not None
        and _content_sha(result) != result["content_sha256"]
    ):
        raise ValueError("finite-hierarchy content hash mismatch")


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
    test_path = root / "tests/test_kastner_schlatter_finite_factorial_hierarchy_no_go.py"
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": config["campaign_id"],
        "scope": (
            "candidate-bound theorem excluding every fixed finite factorial-moment hierarchy as "
            "a unique Poisson selector; no paper/QED derivation, rejection, ontology, or observation inferred"
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
        "finite_hierarchy_no_go": _hierarchy_theorem(),
        "exact_controls": {
            "orders_1_through_6": [_control(k) for k in config["exact_control_orders"]],
            "identity_count": sum(k + 1 for k in config["exact_control_orders"]),
        },
        "minimal_next_contract": _next_contract(),
        "candidate_records": [_candidate_record(record) for record in records],
        "gate_counts": dict(EXPECTED_COUNTS),
        "decision_counts": {"pass": 0, "reject": 0, "blocked": 2},
        "decision": "every_fixed_finite_factorial_selector_ruled_out_nonfinite_selection_blocked",
        "first_blocker": FIRST_BLOCKER,
        "secondary_blockers": [
            "no_source_bound_infinite_factorial_hierarchy_or_moment_determinacy_theorem",
            "no_paper_QED_Mecke_Laplace_or_all_partition_joint_family",
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
