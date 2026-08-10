import json
from pathlib import Path

from sigma_theory_compiler.promotion_orchestrator import (
    ELIGIBILITY,
    PromotionOrchestrator,
    evaluator_binding,
)
from sigma_theory_compiler.static_lift_promotion_evaluator import (
    static_covariant_lift_evaluator,
)

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "configs" / "promotion_pipeline_fail_closed.json"
DESCRIPTOR = ROOT / "configs" / "promotion_static_lift_evaluator.json"
FORMAL_DESCRIPTOR = ROOT / "configs" / "promotion_adm_dirac_principal_evaluator.json"
SOLAR_DESCRIPTOR = ROOT / "configs" / "promotion_solar_known_answer_evaluator.json"
GALAXY_DESCRIPTOR = (
    ROOT / "configs" / "promotion_galaxy_direct_observable_evaluator.json"
)
GENERATOR = ROOT / "configs" / "generator_v2_billion.json"
MANIFEST = ROOT / "runs" / "knowledge-base" / "survivor-export-smoke.json"
SURVIVORS = ROOT / "runs" / "knowledge-base" / "survivors-smoke"
ARTIFACT = ROOT / "runs" / "engine" / "promotion-orchestrator-status.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _descriptor(path: Path = DESCRIPTOR) -> dict:
    descriptor = _load(path)
    descriptor["artifact_path"] = str((ROOT / descriptor["artifact_path"]).resolve())
    return descriptor


def _candidate(identifier: str, expression: str) -> dict:
    return {
        "candidate_id": identifier,
        "correction_expression": expression,
        "data_eligibility": dict(ELIGIBILITY),
    }


def _context() -> dict:
    return {
        "stage_name": "covariant_symbolic_health",
        "category": "symbolic",
        "attempt": 1,
        "input_lineage_sha256": "a" * 64,
        "data_eligibility": dict(ELIGIBILITY),
    }


def test_static_lift_evaluator_passes_rejects_and_blocks_distinctly() -> None:
    passed = static_covariant_lift_evaluator(_candidate("x", "x"), _context())
    assert passed["decision"] == "pass"
    assert passed["classification"]["decision"] == "supported_linear_aether_x_lift"

    blocked = static_covariant_lift_evaluator(_candidate("q", "q+q**2"), _context())
    assert blocked["decision"] == "blocked"
    assert blocked["blocker"] == "unresolved_missing_covariant_q_atom"

    rejected = static_covariant_lift_evaluator(_candidate("z", "q-z"), _context())
    assert rejected["decision"] == "reject"
    assert rejected["classification"]["decision"] == (
        "reject_forbidden_baryonic_action_atom"
    )
    assert all(result["data_eligibility"] == ELIGIBILITY for result in (passed, blocked, rejected))


def test_real_static_lift_promotion_is_reproducible_and_fail_closed(
    tmp_path: Path,
) -> None:
    pipeline = _load(PIPELINE)
    descriptor = _descriptor()
    assert evaluator_binding(descriptor) == pipeline["stages"][1][
        "required_evaluator_binding_sha256"
    ]
    orchestrator = PromotionOrchestrator(tmp_path / "promotion.sqlite", pipeline)
    orchestrator.register_evaluator(descriptor)
    orchestrator.register_evaluator(_descriptor(FORMAL_DESCRIPTOR))
    orchestrator.register_evaluator(_descriptor(SOLAR_DESCRIPTOR))
    orchestrator.register_evaluator(_descriptor(GALAXY_DESCRIPTOR))
    assert orchestrator.import_rust_survivors(
        MANIFEST, GENERATOR, SURVIVORS, maximum_candidates=3
    ) == {"accepted": 3, "duplicates": 0, "limit": 3}
    assert orchestrator.run_ready(maximum_tasks=3) == {
        "evaluated": 3,
        "passed": 0,
        "rejected": 1,
        "blocked": 2,
    }
    status = orchestrator.status()
    assert status["stages"]["covariant_symbolic_health"]["counts"] == {
        "blocked": 2,
        "rejected": 1,
    }
    assert status["stages"]["adm_dirac_principal_health"]["counts"] == {"blocked": 3}
    assert status["unimplemented_gates_fail_closed"] == []
    assert _load(ARTIFACT) == status
