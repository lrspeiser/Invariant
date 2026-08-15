"""Closed capability registry for Sigma candidate-generation strategies.

This module records implementation boundaries only.  A registered generator can propose or
canonicalize candidate artifacts; registration never grants scientific validity or promotion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .sigma_core import canonical_sha256

PORTFOLIO_SCHEMA = "sigma-candidate-generator-portfolio-1.0"


class GeneratorPortfolioError(ValueError):
    """Raised when the closed generator capability registry is malformed."""


@dataclass(frozen=True, slots=True)
class GeneratorCapability:
    strategy_id: str
    implementation_module: str | None
    implementation_status: str
    candidate_artifact_native: bool
    deterministic_replay: bool
    canonical_deduplication: bool
    lineage_bound: bool
    bounded_execution: bool
    external_effects_disabled: bool
    scope: str

    def __post_init__(self) -> None:
        if self.implementation_status not in {"implemented", "partial", "disabled", "missing"}:
            raise GeneratorPortfolioError("generator implementation status is not registered")
        if not self.strategy_id or not self.scope:
            raise GeneratorPortfolioError("generator strategy and scope must be nonempty")
        if self.implementation_status == "implemented" and self.implementation_module is None:
            raise GeneratorPortfolioError("implemented generator requires a module")
        if self.implementation_status == "missing" and self.implementation_module is not None:
            raise GeneratorPortfolioError("missing generator cannot claim an implementation module")
        if self.implementation_status != "implemented" and any(
            (
                self.candidate_artifact_native,
                self.deterministic_replay,
                self.canonical_deduplication,
                self.lineage_bound,
                self.bounded_execution,
            )
        ):
            raise GeneratorPortfolioError(
                "non-implemented generator cannot claim complete capability seals"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "implementation_module": self.implementation_module,
            "implementation_status": self.implementation_status,
            "candidate_artifact_native": self.candidate_artifact_native,
            "deterministic_replay": self.deterministic_replay,
            "canonical_deduplication": self.canonical_deduplication,
            "lineage_bound": self.lineage_bound,
            "bounded_execution": self.bounded_execution,
            "external_effects_disabled": self.external_effects_disabled,
            "scope": self.scope,
        }


def _capabilities() -> tuple[GeneratorCapability, ...]:
    return (
        GeneratorCapability(
            "bayesian",
            "sigma_theory_compiler.bayesian_candidate_generator",
            "implemented",
            True,
            True,
            True,
            True,
            True,
            True,
            "exact-posterior prioritization of an existing bounded candidate set",
        ),
        GeneratorCapability(
            "cross_domain",
            "sigma_theory_compiler.cross_domain_candidate_generator",
            "implemented",
            True,
            True,
            True,
            True,
            True,
            True,
            "bounded structural transfer through three closed templates across distinct packs",
        ),
        GeneratorCapability(
            "egraph",
            "sigma_theory_compiler.egraph_candidate_generator",
            "implemented",
            True,
            True,
            True,
            True,
            True,
            True,
            "equality saturation under fourteen registered exact rational-algebra rewrites",
        ),
        GeneratorCapability(
            "evolutionary",
            "sigma_theory_compiler.evolutionary_candidate_generator",
            "implemented",
            True,
            True,
            True,
            True,
            True,
            True,
            "bounded callback-driven heuristic population search",
        ),
        GeneratorCapability(
            "grammar",
            "sigma_theory_compiler.grammar_candidate_generator",
            "implemented",
            True,
            True,
            True,
            True,
            True,
            True,
            "bounded exact registered grammar enumeration with typed manifests",
        ),
        GeneratorCapability(
            "llm",
            "sigma_theory_compiler.llm_candidate_generator",
            "implemented",
            True,
            True,
            True,
            True,
            True,
            True,
            "provider-neutral, cost-capped, secret-safe proposals quarantined behind downstream gates",
        ),
        GeneratorCapability(
            "symbolic",
            "sigma_theory_compiler.symbolic_candidate_generator",
            "implemented",
            True,
            True,
            True,
            True,
            True,
            True,
            "bounded exact rational template and coefficient enumeration",
        ),
    )


def build_generator_portfolio() -> dict[str, Any]:
    """Return the sealed, closed generator capability inventory."""

    capabilities = _capabilities()
    if tuple(item.strategy_id for item in capabilities) != tuple(
        sorted(item.strategy_id for item in capabilities)
    ):
        raise GeneratorPortfolioError("generator strategies must be sorted")
    counts = {
        status: sum(item.implementation_status == status for item in capabilities)
        for status in ("implemented", "partial", "disabled", "missing")
    }
    body = {
        "schema_version": PORTFOLIO_SCHEMA,
        "capabilities": [item.to_dict() for item in capabilities],
        "counts": {"registered": len(capabilities), **counts},
        "claims": {
            "generator_registration_establishes_scientific_truth": False,
            "generator_registration_authorizes_promotion": False,
            "all_requested_generator_strategies_complete": True,
            "implemented_strategies_use_sigma_core_candidate_artifacts": True,
        },
        "first_remaining_blocker": (
            "exercise_all_generator_strategies_through_preregistered_domain_gates_and_"
            "held_out_benchmarks_without_granting_generator_self_promotion_authority"
        ),
    }
    return {**body, "content_sha256": canonical_sha256(body)}


def validate_generator_portfolio(value: dict[str, Any]) -> None:
    """Reject unknown, omitted, resealed, or overclaimed portfolio state."""

    if value != build_generator_portfolio():
        raise GeneratorPortfolioError("candidate generator portfolio boundary changed")


__all__ = [
    "PORTFOLIO_SCHEMA",
    "GeneratorCapability",
    "GeneratorPortfolioError",
    "build_generator_portfolio",
    "validate_generator_portfolio",
]
