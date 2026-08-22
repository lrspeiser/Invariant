from __future__ import annotations

import json
from pathlib import Path

from sigma_theory_compiler import declarative_discovery_operational_campaign as C

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / C.OUTPUT_PATH


def test_operational_campaign_closes_every_runtime_control_without_claiming_open_math() -> None:
    value = C.run_operational_campaign(ROOT)
    assert len(value["extension_admission"]["admitted_declarations"]) == 4
    assert value["extension_admission"]["no_bespoke_module_per_extension"] is True
    assert len(value["creativity"]["families_executed"]) == 10
    assert value["independent_verification"] == {
        "backends": ["exact_arithmetic", "cas", "interval", "lean"],
        "bundles_verified": 12,
        "mathematical_backend_execution_established_by_protocol_quorum": False,
        "minimum_independent_principals": 4,
        "proposer_self_approval_allowed": False,
        "same_control_function_used_for_protocol_bundles": True,
    }
    assert all(
        value["cross_backend_identity_control"][backend]["executed"]
        for backend in ("cas", "exact_arithmetic", "interval", "lean")
    )
    assert value["behavioral_archive"]["occupied_niches"] == 12
    assert value["reachability_qualified_negative"]["status"] == (
        "REACHABILITY_QUALIFIED_NEGATIVE"
    )
    assert value["proof_plan"]["closed"] is True
    assert value["dataset_pipeline"]["heldout_opened_at_final_stage"] is True
    assert value["blind_capability"]["highest_passed"] == 5
    blind_evidence = value["blind_capability"]["evidence"]
    assert [row["passed"] for row in blind_evidence] == [True] * 5 + [False]
    assert all(row["target_commitment"] for row in blind_evidence[2:5])
    assert blind_evidence[1]["benchmark_id"] == (
        "anonymous-natural-sum-blind-rediscovery-001"
    )
    assert blind_evidence[3]["benchmark_id"] == "blind-planetary-laws.kepler-harmonic"
    assert blind_evidence[4]["benchmark_id"] == "prospective.modular_affine"
    assert value["serious_claim_chain"]["release_authorized"] is False
    assert value["claims"]["novel_mathematics_established"] is False
    assert value["claims"]["open_problem_solved"] is False
    audit = value["objective_completion_audit"]
    assert len(audit) == 12
    assert all(row["status"] == "PASS" for row in audit)
    assert [row["requirement_id"] for row in audit] == [
        "R01_DATA_DRIVEN_LANGUAGE_EXTENSION",
        "R02_INDEPENDENT_EXTENSION_ADMISSION",
        "R03_MULTIPLE_CREATIVITY_FAMILIES",
        "R04_GENUINELY_DISTINCT_MATH_VERIFIERS",
        "R05_BEHAVIORAL_MAP_ARCHIVE",
        "R06_COUNTEREXAMPLE_REPAIR_GRADIENT",
        "R07_REACHABILITY_QUALIFIED_NEGATIVES",
        "R08_PROOF_PLAN_SEARCH_AND_BLOCKERS",
        "R09_ORDERED_SEALED_DATASET_PIPELINE",
        "R10_EVIDENCE_BACKED_BLIND_LADDER",
        "R11_SERIOUS_CLAIM_RELEASE_CHAIN",
        "R12_CREATIVE_YIELD_ACCOUNTING",
    ]


def test_operational_campaign_reports_all_requested_creative_yield_axes() -> None:
    metrics = C.run_operational_campaign(ROOT)["creative_yield"]
    assert set(metrics["counts"]) >= {
        "behavioral_niches",
        "counterexample_survivors",
        "proof_plans_closed",
        "unique_proof_mechanisms",
    }
    assert set(metrics["rates"]) == {
        "counterexample_survival",
        "holdout_improvement",
        "proof_completion",
        "verification_yield",
    }
    assert set(metrics["compute"]) == {
        "construction_gpu_hours",
        "construction_to_refutation",
        "gpu_hours_per_positive",
        "refutation_gpu_hours",
    }


def test_operational_receipt_replays_byte_for_byte() -> None:
    value = json.loads(RECEIPT.read_text(encoding="utf-8"))
    C.validate_receipt(value, ROOT)
    assert value == C.run_operational_campaign(ROOT)
