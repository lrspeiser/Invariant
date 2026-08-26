"""Compile the MathOverflow Task 2 failures into a sound exclusion ledger."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .exclusion_ledger import build_ledger, make_certificate, validate_ledger
from .frankl_counterexample_verifier import verify_family
from .mathoverflow_task2_evaluator import PROMISING_GENERATOR_IDS, REPAIR_VALID_ID
from .sigma_core import canonical_sha256

DOMAIN = "frankl_existential_residual_delta_counterexample"
LEDGER_ID = "mathoverflow-task2-exclusion-ledger-v1"
FEATURE_CANONICALIZER_ID = "frankl_finite_family_features_v1"


class FranklFailureSpaceError(ValueError):
    """The Task 2 evidence cannot soundly produce the expected failure space."""


def _validate_sealed_artifact(value: Mapping[str, Any], label: str) -> None:
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("content_sha256") != canonical_sha256(body):
        raise FranklFailureSpaceError(f"{label} seal changed")


def _proof_certificate(
    *,
    certificate_id: str,
    failure_mode: str,
    scope_predicate: list[dict[str, Any]],
    proof: Mapping[str, Any],
    explanation: str,
) -> dict[str, Any]:
    proof_sha256 = canonical_sha256(proof)
    return make_certificate(
        certificate_id=certificate_id,
        domain=DOMAIN,
        failure_mode=failure_mode,
        scope_predicate=scope_predicate,
        witness={"proof": dict(proof), "proof_sha256": proof_sha256},
        verifier={
            "artifact_sha256": proof_sha256,
            "decision": "REJECT",
            "kind": "deductive_integer_counting_proof",
        },
        generalization_level="proved_parametric_family",
        explanation=explanation,
    )


def _parametric_certificates(evidence_sha256: str) -> list[dict[str, Any]]:
    candidate_kind = {
        "field": "candidate_kind",
        "op": "eq",
        "value": "finite_union_closed_family",
    }
    canonical_features = {
        "field": "feature_canonicalizer_id",
        "op": "eq",
        "value": FEATURE_CANONICALIZER_ID,
    }
    universal = _proof_certificate(
        certificate_id="frankl.family.universal-element",
        failure_mode="universal_element_supplies_witness",
        scope_predicate=[
            candidate_kind,
            canonical_features,
            {"field": "has_universal_element", "op": "eq", "value": True},
        ],
        proof={
            "assumptions": [
                "F is a finite nonempty union-closed family",
                "u belongs to every member of F",
            ],
            "derivation": [
                "The subfamily of members not containing u is empty.",
                "Its residual maximum degree is 0.",
                "Therefore 0 <= Delta(F)/2, so u satisfies the conjectured inequality.",
            ],
            "conclusion": "F cannot be a counterexample to the existential claim.",
        },
        explanation=(
            "A canonical family with a verified universal element is excluded from the entire "
            "counterexample search, not merely from one textual proposal."
        ),
    )
    powerset = _proof_certificate(
        certificate_id="frankl.family.complete-nonempty-powerset",
        failure_mode="residual_degree_equals_half_delta",
        scope_predicate=[
            candidate_kind,
            canonical_features,
            {
                "field": "construction_class",
                "op": "eq",
                "value": "complete_nonempty_powerset",
            },
            {"field": "universe_size", "op": "ge", "value": 2},
        ],
        proof={
            "parameter": "n = universe size >= 2",
            "counts": {
                "delta": "2^(n-1)",
                "residual_delta_after_omitting_any_x": "2^(n-2)",
            },
            "conclusion": "Every x gives equality with Delta(F)/2, not strict excess.",
        },
        explanation="The full nonempty powerset is an equality family and cannot be a counterexample.",
    )
    complements = _proof_certificate(
        certificate_id="frankl.family.all-n-minus-one-plus-full",
        failure_mode="single_residual_set_has_degree_one",
        scope_predicate=[
            candidate_kind,
            canonical_features,
            {
                "field": "construction_class",
                "op": "eq",
                "value": "all_n_minus_one_subsets_plus_full",
            },
            {"field": "universe_size", "op": "ge", "value": 2},
        ],
        proof={
            "parameter": "n = universe size >= 2",
            "counts": {
                "delta": "n",
                "residual_delta_after_omitting_x": "1",
            },
            "inequality": "1 <= n/2",
            "conclusion": "Every x satisfies the conjectured inequality.",
        },
        explanation=(
            "The family consisting of every (n-1)-subset and the full set is union-closed but "
            "cannot be a strict residual counterexample."
        ),
    )
    heuristic = make_certificate(
        certificate_id="frankl.heuristic.high-symmetry-equality-trap",
        domain=DOMAIN,
        failure_mode="high_symmetry_often_lands_on_equality",
        scope_predicate=[
            candidate_kind,
            canonical_features,
            {"field": "degree_profile", "op": "eq", "value": "highly_symmetric"},
        ],
        witness={
            "observation": (
                "Several symmetric constructions land at or below residual Delta/2; this is a "
                "search warning, not a theorem about all symmetric families."
            )
        },
        verifier={
            "artifact_sha256": evidence_sha256,
            "decision": "UNVERIFIED",
            "kind": "trial_observation",
        },
        generalization_level="finite_enumerated_region",
        status="HEURISTIC_FAILURE",
        explanation="Retain and test these candidates; never prune them from this heuristic alone.",
    )
    return [universal, powerset, complements, heuristic]


def classify_frankl_family(
    raw_family: Sequence[Sequence[int]], *, candidate_id: str
) -> dict[str, Any]:
    """Return verifier-derived features suitable for sound family-certificate matching."""

    receipt = verify_family(raw_family)
    members = {frozenset(row) for row in receipt["family"]}
    universe = frozenset(receipt["universe"])
    universe_size = len(universe)
    has_universal_element = any(
        all(element in member for member in members) for element in universe
    )
    complete_nonempty_powerset = len(members) == (1 << universe_size) - 1
    n_minus_one_plus_full = members == {
        universe,
        *(universe - {element} for element in universe),
    }
    if complete_nonempty_powerset:
        construction_class = "complete_nonempty_powerset"
    elif n_minus_one_plus_full:
        construction_class = "all_n_minus_one_subsets_plus_full"
    else:
        construction_class = "unclassified"
    return {
        "candidate_id": candidate_id,
        "canonical_sha256": receipt["canonical_family_sha256"],
        "features": {
            "candidate_kind": (
                "finite_union_closed_family"
                if receipt["union_closed"]
                else "finite_set_family"
            ),
            "feature_canonicalizer_id": FEATURE_CANONICALIZER_ID,
            "construction_class": construction_class,
            "has_universal_element": has_universal_element,
            "universe_size": universe_size,
            "family_size": receipt["family_size"],
            "delta": receipt["delta"],
            "degree_profile": (
                "highly_symmetric"
                if len(set(receipt["degrees"].values())) == 1
                else "asymmetric"
            ),
        },
        "verifier_receipt": receipt,
    }


def _historical_candidates(public: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": submission["submission_id"],
            "canonical_sha256": canonical_sha256(submission["hypothesis"]),
            "features": {
                "candidate_kind": "llm_submission_text",
                "source_kind": "sealed_task2_submission",
                "target_problem_id": public["problem"]["problem_id"],
            },
        }
        for submission in public["submissions"]
    ]


def _exact_certificates(
    public: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> list[dict[str, Any]]:
    submissions = {row["submission_id"]: row for row in public["submissions"]}
    exact_checks = evidence["promising_generator_exact_checks"]
    certificates = []
    for result in evaluation["evaluations"]:
        submission_id = result["submission_id"]
        candidate_sha256 = canonical_sha256(submissions[submission_id]["hypothesis"])
        exact_check = exact_checks.get(submission_id)
        if exact_check is not None:
            failure_mode = "exact_generator_closure_has_inequality_witness"
            artifact_sha256 = exact_check["verifier_receipt_content_sha256"]
            witness = {
                "submission_id": submission_id,
                "family_size": exact_check["family_size"],
                "delta": exact_check["delta"],
                "residual_delta": exact_check["residual_delta"],
                "exact_counterexample_valid": exact_check["exact_counterexample_valid"],
            }
            explanation = (
                "The supplied generators were closed under union and counted exactly; at least "
                "one element fails the required strict residual inequality."
            )
        elif submission_id == REPAIR_VALID_ID:
            failure_mode = "valid_repair_but_no_counterexample"
            artifact_sha256 = evaluation["content_sha256"]
            witness = {
                "submission_id": submission_id,
                "repair_graph_sha256": result["canonical_repair_graph_sha256"],
                "exact_counterexample_valid": False,
            }
            explanation = (
                "This exact response contains a valid universal-element repair, but it does not "
                "supply the requested counterexample."
            )
        else:
            failure_mode = "no_decisive_counterexample_supplied"
            artifact_sha256 = evaluation["content_sha256"]
            witness = {
                "submission_id": submission_id,
                "counterexample_or_rejection": result["counterexample_or_rejection"],
                "exact_counterexample_valid": False,
            }
            explanation = (
                "This exact sealed response cannot pass the task because it supplies no complete "
                "counterexample; this certificate does not reject its component ideas."
            )
        certificates.append(
            make_certificate(
                certificate_id=f"frankl.exact.{submission_id.removeprefix('submission.')}",
                domain=DOMAIN,
                failure_mode=failure_mode,
                scope_predicate=[
                    {"field": "canonical_sha256", "op": "eq", "value": candidate_sha256}
                ],
                witness=witness,
                verifier={
                    "artifact_sha256": artifact_sha256,
                    "decision": "REJECT",
                    "kind": (
                        "exact_executable_verifier"
                        if exact_check is not None
                        else "sealed_independent_task_evaluation"
                    ),
                },
                generalization_level="exact_instance",
                explanation=explanation,
            )
        )
    return certificates


def build_frankl_failure_ledger(
    public: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    evidence: Mapping[str, Any],
    adjudication: Mapping[str, Any],
) -> dict[str, Any]:
    for value, label in (
        (public, "public submissions"),
        (evaluation, "independent evaluation"),
        (evidence, "blind scoring evidence"),
        (adjudication, "adjudication"),
    ):
        _validate_sealed_artifact(value, label)
    submission_ids = {row["submission_id"] for row in public["submissions"]}
    evaluation_ids = {row["submission_id"] for row in evaluation["evaluations"]}
    if (
        len(submission_ids) != 36
        or submission_ids != evaluation_ids
        or set(evidence["promising_generator_exact_checks"]) != PROMISING_GENERATOR_IDS
        or adjudication.get("decision") != "REJECT"
        or adjudication.get("status") != "PERFORMANCE_OR_CORRECTNESS_GATE_FAILED"
    ):
        raise FranklFailureSpaceError("Task 2 rejection evidence changed")
    candidates = _historical_candidates(public)
    certificates = _exact_certificates(public, evaluation, evidence)
    certificates.extend(_parametric_certificates(evidence["content_sha256"]))
    ledger = build_ledger(
        ledger_id=LEDGER_ID,
        domain=DOMAIN,
        candidates=candidates,
        certificates=certificates,
        frontier_constraints=[
            {
                "kind": "hard_target_condition",
                "constraint": "candidate must be a concrete finite union-closed family",
                "reason": "Textual plausibility is not an independently checkable counterexample.",
            },
            {
                "kind": "proved_exclusion",
                "constraint": "no universal element",
                "reason": "A universal element leaves an empty residual family.",
            },
            {
                "kind": "proved_exclusion",
                "constraint": "avoid certified equality construction classes",
                "reason": "Equality at Delta/2 does not meet strict counterexample arithmetic.",
            },
            {
                "kind": "hard_target_condition",
                "constraint": "for every x, 2 * residual_delta[x] > delta",
                "reason": "One witness at or below half rejects the entire candidate family.",
            },
            {
                "kind": "search_priority_not_pruning_rule",
                "constraint": "test near-regular odd-delta families near the strict boundary",
                "reason": (
                    "The accepted 30-set reference has degree 19 and residual degree 10 for "
                    "every element; this guides exploration without asserting uniqueness."
                ),
            },
            {
                "kind": "epistemic_boundary",
                "constraint": "unmatched means unexplored under current certificates, not novel",
                "reason": "Novelty requires a separate prior-art and equivalence audit.",
            },
        ],
        source_bindings={
            "public_submissions_content_sha256": public["content_sha256"],
            "independent_evaluation_content_sha256": evaluation["content_sha256"],
            "blind_scoring_evidence_content_sha256": evidence["content_sha256"],
            "adjudication_content_sha256": adjudication["content_sha256"],
        },
    )
    if (
        ledger["counts"]["candidates"] != 36
        or ledger["counts"]["excluded_candidates"] != 36
        or ledger["counts"]["surviving_candidates"] != 0
    ):
        raise FranklFailureSpaceError("historical Task 2 candidate classification changed")
    return ledger


def validate_frankl_failure_ledger(
    ledger: Mapping[str, Any],
    public: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    evidence: Mapping[str, Any],
    adjudication: Mapping[str, Any],
) -> None:
    candidates = _historical_candidates(public)
    validate_ledger(ledger, candidates)
    expected = build_frankl_failure_ledger(public, evaluation, evidence, adjudication)
    if ledger != expected:
        raise FranklFailureSpaceError("Frankl exclusion ledger replay changed")


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise FranklFailureSpaceError(f"JSON root is not an object: {path}")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--adjudication", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    sources = [
        _read_json(args.public),
        _read_json(args.evaluation),
        _read_json(args.evidence),
        _read_json(args.adjudication),
    ]
    ledger = build_frankl_failure_ledger(*sources)
    validate_frankl_failure_ledger(ledger, *sources)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "content_sha256": ledger["content_sha256"],
                "excluded_candidates": ledger["counts"]["excluded_candidates"],
                "surviving_candidates": ledger["counts"]["surviving_candidates"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
