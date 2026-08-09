from pathlib import Path

from sigma_theory_compiler.cli import _parser
from sigma_theory_compiler.grammar_exhaustion import audit_static_grammar_exhaustion

ROOT = Path(__file__).resolve().parents[1]


def test_current_dense_static_grammar_is_exhausted_by_bound_hard_rejections() -> None:
    report = audit_static_grammar_exhaustion(
        ROOT / "runs" / "knowledge-base" / "generated-priority-dense.json",
        ROOT / "runs" / "formal-controls-v1" / "formal-controls.json",
        ROOT
        / "runs"
        / "generated-candidates"
        / "GF-5df8715b319f54cb-static-null-v1"
        / "formal-health"
        / "q-operator-ir.json",
    )
    assert report["status"] == "exhausted_no_admissible_family"
    assert report["queue_count"] == report["rejected_count"] == 124
    assert report["unresolved_count"] == 0
    assert report["decision_counts"] == {
        "reject_forbidden_baryonic_z": 104,
        "reject_registered_projected_q_completion": 20,
    }
    assert report["q_rejection_verified"]
    assert not report["observational_data_opened"]


def test_grammar_exhaustion_cli_is_explicit() -> None:
    args = _parser().parse_args(
        [
            "grammar-exhaustion",
            "--priority",
            "priority.json",
            "--formal-controls",
            "formal.json",
            "--q-operator",
            "q.json",
            "--output",
            "exhaustion.json",
        ]
    )
    assert args.command == "grammar-exhaustion"
