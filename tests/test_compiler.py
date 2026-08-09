import json
from pathlib import Path

from sigma_theory_compiler.compiler import TheoryCompiler
from sigma_theory_compiler.registry import write_registry
from sigma_theory_compiler.validation import run_validation


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "v18_flux_mvp.json"


def test_compiler_accounts_for_every_candidate() -> None:
    registry = TheoryCompiler.from_path(CONFIG).run(CONFIG)
    counts = registry["counts"]
    assert counts["total"] > 0
    assert counts["total"] == (
        counts["rejected_pre_covariant"] + counts["requires_covariant_lift"]
    )
    assert counts["fully_validated_theories"] == 0
    assert registry["enumeration"]["signed_candidates"] == counts["total"]


def test_no_candidate_is_mislabeled_as_covariantly_healthy() -> None:
    registry = TheoryCompiler.from_path(CONFIG).run(CONFIG)
    assert {row["status"] for row in registry["candidates"]} <= {
        "rejected_pre_covariant",
        "requires_covariant_lift",
    }
    for candidate in registry["candidates"]:
        deferred = [gate for gate in candidate["gates"] if gate["status"] == "deferred"]
        assert len(deferred) == 6
        assert "Derivative" in candidate["radial_constitutive_equation"] or "dW_dr" in candidate["radial_constitutive_equation"]


def test_registry_round_trip(tmp_path: Path) -> None:
    registry = TheoryCompiler.from_path(CONFIG).run(CONFIG)
    json_path, markdown_path = write_registry(registry, tmp_path)
    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded["counts"] == registry["counts"]
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "Fully validated theories: 0" in markdown
    assert "Gate semantics" in markdown


def test_known_answer_validation_suite_passes() -> None:
    report = run_validation()
    assert report["counts"] == {"total": 8, "passed": 8, "failed": 0}
