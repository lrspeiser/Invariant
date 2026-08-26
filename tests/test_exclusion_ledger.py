from __future__ import annotations

import copy

import pytest

from sigma_theory_compiler.exclusion_ledger import (
    ExclusionLedgerError,
    build_ledger,
    evaluate_candidate,
    make_certificate,
    validate_certificate,
    validate_ledger,
)
from sigma_theory_compiler.sigma_core import canonical_sha256


def _candidate(candidate_id: str, payload: object, **features: object) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "canonical_sha256": canonical_sha256(payload),
        "features": features,
    }


def _certificate(
    certificate_id: str,
    predicate: list[dict[str, object]],
    *,
    status: str = "VERIFIED_EXCLUSION",
    level: str = "proved_parametric_family",
) -> dict[str, object]:
    return make_certificate(
        certificate_id=certificate_id,
        domain="test",
        failure_mode="test_failure",
        scope_predicate=predicate,
        witness={"counterexample": "w"},
        verifier={
            "artifact_sha256": canonical_sha256({"proof": certificate_id}),
            "decision": "REJECT" if status == "VERIFIED_EXCLUSION" else "UNVERIFIED",
        },
        generalization_level=level,
        status=status,
        explanation="Test certificate.",
    )


def test_exact_verified_certificate_prunes_only_the_identical_candidate():
    target = _candidate("target", {"formula": "x+x"})
    other = _candidate("other", {"formula": "2*x"})
    certificate = _certificate(
        "exact.target",
        [{"field": "canonical_sha256", "op": "eq", "value": target["canonical_sha256"]}],
        level="exact_instance",
    )

    assert evaluate_candidate(target, [certificate])["decision"] == "EXCLUDE"
    assert evaluate_candidate(other, [certificate])["decision"] == "SURVIVE"


def test_heuristic_match_warns_but_never_prunes():
    candidate = _candidate("symmetric", {"formula": "x"}, symmetric=True)
    heuristic = _certificate(
        "heuristic.symmetry",
        [{"field": "symmetric", "op": "eq", "value": True}],
        status="HEURISTIC_FAILURE",
    )

    outcome = evaluate_candidate(candidate, [heuristic])

    assert outcome["decision"] == "SURVIVE"
    assert outcome["heuristic_warning_certificate_ids"] == ["heuristic.symmetry"]


def test_proved_parametric_certificate_prunes_only_feature_matches():
    certificate = _certificate(
        "family.universal",
        [{"field": "has_universal_element", "op": "eq", "value": True}],
    )
    match = _candidate("match", [1], has_universal_element=True)
    mismatch = _candidate("mismatch", [2], has_universal_element=False)

    assert evaluate_candidate(match, [certificate])["decision"] == "EXCLUDE"
    assert evaluate_candidate(mismatch, [certificate])["decision"] == "SURVIVE"


def test_missing_feature_does_not_match_even_a_not_equal_predicate():
    certificate = _certificate(
        "family.not-linear",
        [{"field": "shape", "op": "ne", "value": "linear"}],
    )
    candidate = _candidate("unknown", [1])

    assert evaluate_candidate(candidate, [certificate])["decision"] == "SURVIVE"


def test_candidate_features_cannot_override_canonical_identity():
    target = _candidate("target", [1])
    other = _candidate("other", [2], canonical_sha256=target["canonical_sha256"])
    certificate = _certificate(
        "exact.target",
        [{"field": "canonical_sha256", "op": "eq", "value": target["canonical_sha256"]}],
        level="exact_instance",
    )

    assert evaluate_candidate(other, [certificate])["decision"] == "SURVIVE"


def test_ledger_replays_counts_and_frontier():
    excluded = _candidate("excluded", [1], forbidden=True)
    survivor = _candidate("survivor", [2], forbidden=False)
    certificate = _certificate(
        "family.forbidden",
        [{"field": "forbidden", "op": "eq", "value": True}],
    )
    ledger = build_ledger(
        ledger_id="test-ledger",
        domain="test",
        candidates=[survivor, excluded],
        certificates=[certificate],
        frontier_constraints=[{"constraint": "forbidden == false"}],
        source_bindings={"test": canonical_sha256("source")},
    )

    validate_ledger(ledger, [excluded, survivor])
    assert ledger["counts"]["excluded_candidates"] == 1
    assert ledger["survivor_frontier"]["candidate_ids"] == ["survivor"]


def test_tampering_with_certificate_or_ledger_fails_closed():
    candidate = _candidate("candidate", [1], forbidden=True)
    certificate = _certificate(
        "family.forbidden",
        [{"field": "forbidden", "op": "eq", "value": True}],
    )
    tampered_certificate = copy.deepcopy(certificate)
    tampered_certificate["failure_mode"] = "changed"
    with pytest.raises(ExclusionLedgerError, match="seal"):
        validate_certificate(tampered_certificate)

    ledger = build_ledger(
        ledger_id="test-ledger",
        domain="test",
        candidates=[candidate],
        certificates=[certificate],
        frontier_constraints=[],
        source_bindings={},
    )
    tampered_ledger = copy.deepcopy(ledger)
    tampered_ledger["counts"]["excluded_candidates"] = 0
    with pytest.raises(ExclusionLedgerError, match="seal"):
        validate_ledger(tampered_ledger, [candidate])
