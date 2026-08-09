from __future__ import annotations

import json
from pathlib import Path

from sigma_theory_compiler.equation_universe import (
    SCHEMA_VERSION,
    EquationUniverse,
    build_equation_universe,
)

ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / "configs" / "equation_universe" / "gravity_seed_v1.json"


def _force_record(expression: str, names: tuple[str, str, str]) -> dict[str, object]:
    force, mass, acceleration = names
    return {
        "equation_id": "QUERY",
        "name": "query",
        "representation": "scalar_sympy",
        "expression": expression,
        "variables": [
            {
                "symbol": force,
                "canonical_name": force,
                "meaning": "force-like variable",
                "dimension": {"M": 1, "L": 1, "T": -2},
            },
            {
                "symbol": mass,
                "canonical_name": mass,
                "meaning": "mass-like variable",
                "dimension": {"M": 1},
            },
            {
                "symbol": acceleration,
                "canonical_name": acceleration,
                "meaning": "acceleration-like variable",
                "dimension": {"L": 1, "T": -2},
            },
        ],
    }


def test_equation_universe_builds_with_verified_derivations_and_provenance(tmp_path) -> None:
    database = tmp_path / "equations.sqlite"
    report = tmp_path / "equations.json"
    result = build_equation_universe(SEED, database, report)
    assert result["import"]["rejected"] == []
    assert result["audit"]["passed"]
    assert result["audit"]["integrity_check"] == "ok"
    assert result["audit"]["counts"]["equations"] == 18
    assert result["audit"]["counts"]["derivations"] == 3
    assert result["audit"]["derivation_proofs"] == {"verified": 3}
    assert result["audit"]["counts"]["equivalence_edges"] >= 1
    assert result["audit"]["source_ingestion_modes"]["metadata_only"] >= 1
    assert report.is_file()


def test_equation_universe_detects_rearrangement_and_typed_alpha_equivalence(tmp_path) -> None:
    database = tmp_path / "equations.sqlite"
    build_equation_universe(SEED, database, tmp_path / "report.json")
    universe = EquationUniverse(database)

    rearranged = universe.classify(_force_record("F - m*a = 0", ("F", "m", "a")))
    assert rearranged["classification"] == "known_semantic_equivalent"
    assert {item["equation_id"] for item in rearranged["semantic_matches"]} >= {
        "EQ-NEWTON-SECOND-LAW",
        "EQ-NEWTON-SECOND-LAW-REARRANGED",
    }

    renamed = universe.classify(_force_record("Q = p*x", ("Q", "p", "x")))
    assert renamed["classification"] == "known_structural_analogue"
    assert not renamed["novelty_claim_allowed"]
    assert renamed["structural_matches"]


def test_equation_universe_rejects_bad_dimensions_and_metadata_copying(tmp_path) -> None:
    database = tmp_path / "equations.sqlite"
    universe = EquationUniverse(database)
    universe.initialize()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "sources": [
            {
                "source_id": "SRC-RESTRICTED",
                "title": "Restricted reference",
                "url": "https://example.invalid/restricted",
                "authors": [],
                "source_kind": "test",
                "ingestion_mode": "metadata_only",
                "policy_reason": "test policy",
            }
        ],
        "equations": [
            {
                "equation_id": "EQ-COPY",
                "name": "Unmarked extraction",
                "representation": "scalar_sympy",
                "expression": "x = y",
                "variables": [
                    {"symbol": "x", "dimension": {"L": 1}},
                    {"symbol": "y", "dimension": {"L": 1}},
                ],
                "source_id": "SRC-RESTRICTED",
                "independently_encoded": False,
            },
            {
                "equation_id": "EQ-BAD-DIMENSION",
                "name": "Bad dimension",
                "representation": "scalar_sympy",
                "expression": "x = y",
                "variables": [
                    {"symbol": "x", "dimension": {"L": 1}},
                    {"symbol": "y", "dimension": {"T": 1}},
                ],
                "source_id": "SRC-RESTRICTED",
                "independently_encoded": True,
            },
        ],
        "derivations": [],
    }
    input_path = tmp_path / "bad.json"
    input_path.write_text(json.dumps(payload), encoding="utf-8")
    imported = universe.import_file(input_path)
    assert imported["equations"] == 0
    assert len(imported["rejected"]) == 2
    reasons = " ".join(item["reason"] for item in imported["rejected"])
    assert "metadata-only" in reasons
    assert "dimension audit failed" in reasons


def test_unmatched_equation_is_not_mislabeled_novel(tmp_path) -> None:
    database = tmp_path / "equations.sqlite"
    build_equation_universe(SEED, database, tmp_path / "report.json")
    result = EquationUniverse(database).classify(
        {
            "equation_id": "QUERY",
            "name": "query",
            "representation": "scalar_sympy",
            "expression": "y = exp(x)",
            "variables": [
                {"symbol": "x", "dimension": {}},
                {"symbol": "y", "dimension": {}},
            ],
        }
    )
    assert result["classification"] == "not_found_in_corpus"
    assert not result["novelty_claim_allowed"]
    assert "does not establish" in result["novelty_warning"]


def test_projected_aether_q_is_registered_as_known_prior_art_screened_definition(
    tmp_path,
) -> None:
    database = tmp_path / "equations.sqlite"
    build_equation_universe(SEED, database, tmp_path / "report.json")
    result = EquationUniverse(database).classify(
        {
            "equation_id": "QUERY-Q",
            "name": "query q",
            "representation": "tensor_dsl",
            "expression": (
                "Q_a_u = (L_sigma^2/a_sigma^2) P^{mu rho} P^{nu sigma} "
                "nabla_mu(a_nu) nabla_rho(a_sigma)"
            ),
            "variables": [],
        }
    )
    assert result["classification"] == "known_semantic_equivalent"
    assert {item["equation_id"] for item in result["semantic_matches"]} == {
        "EQ-SIGMA-PROJECTED-AETHER-Q"
    }
    assert not result["novelty_claim_allowed"]
