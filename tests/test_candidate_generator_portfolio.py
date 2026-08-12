from __future__ import annotations

import copy

import pytest

from sigma_theory_compiler.candidate_generator_portfolio import (
    GeneratorCapability,
    GeneratorPortfolioError,
    build_generator_portfolio,
    validate_generator_portfolio,
)
from sigma_theory_compiler.sigma_core import canonical_sha256


def test_portfolio_records_exact_current_capability_boundary() -> None:
    portfolio = build_generator_portfolio()
    validate_generator_portfolio(portfolio)
    assert portfolio["counts"] == {
        "registered": 7,
        "implemented": 3,
        "partial": 2,
        "disabled": 1,
        "missing": 1,
    }
    capabilities = {item["strategy_id"]: item for item in portfolio["capabilities"]}
    assert set(capabilities) == {
        "bayesian",
        "cross_domain",
        "egraph",
        "evolutionary",
        "grammar",
        "llm",
        "symbolic",
    }
    for strategy in ("bayesian", "egraph", "evolutionary"):
        assert capabilities[strategy]["implementation_status"] == "implemented"
        assert capabilities[strategy]["candidate_artifact_native"] is True
        assert capabilities[strategy]["deterministic_replay"] is True
        assert capabilities[strategy]["canonical_deduplication"] is True
        assert capabilities[strategy]["lineage_bound"] is True
        assert capabilities[strategy]["bounded_execution"] is True
        assert capabilities[strategy]["external_effects_disabled"] is True


def test_registration_never_claims_truth_promotion_or_completion() -> None:
    portfolio = build_generator_portfolio()
    assert portfolio["claims"] == {
        "generator_registration_establishes_scientific_truth": False,
        "generator_registration_authorizes_promotion": False,
        "all_requested_generator_strategies_complete": False,
        "implemented_strategies_use_sigma_core_candidate_artifacts": True,
    }
    assert portfolio["first_remaining_blocker"].startswith("make_grammar_symbolic_llm")


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("claims", "generator_registration_authorizes_promotion"), True),
        (("claims", "all_requested_generator_strategies_complete"), True),
        (("capabilities", 1, "implementation_status"), "implemented"),
        (("counts", "implemented"), 7),
    ],
)
def test_resealed_semantic_tampering_fails_closed(
    path: tuple[object, ...], replacement: object
) -> None:
    value = copy.deepcopy(build_generator_portfolio())
    target = value
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    value["content_sha256"] = canonical_sha256(body)
    with pytest.raises(GeneratorPortfolioError, match="boundary changed"):
        validate_generator_portfolio(value)


def test_nonimplemented_capability_cannot_claim_complete_seals() -> None:
    with pytest.raises(GeneratorPortfolioError, match="complete capability seals"):
        GeneratorCapability(
            "future",
            None,
            "missing",
            True,
            False,
            False,
            False,
            False,
            True,
            "not implemented",
        )
