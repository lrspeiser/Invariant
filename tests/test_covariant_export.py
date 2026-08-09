import hashlib
import json
from pathlib import Path

from sigma_theory_compiler.action_ir import load_action_grammar
from sigma_theory_compiler.cli import _parser
from sigma_theory_compiler.covariant_export import (
    export_covariant_candidate_files,
    export_representable_covariant_candidates,
)
from sigma_theory_compiler.formal_backend import load_field_contract

ROOT = Path(__file__).resolve().parents[1]
PRIORITY = ROOT / "runs" / "knowledge-base" / "generated-priority-dense.json"
GRAMMAR = load_action_grammar(ROOT / "configs" / "covariant_action_grammar.json")
CONTRACT = load_field_contract(ROOT / "configs" / "covariant_field_contract.json")


def test_dense_queue_exports_six_exact_specs_and_only_one_formal_backend_lead() -> None:
    raw = PRIORITY.read_bytes()
    report, specs = export_representable_covariant_candidates(
        json.loads(raw),
        GRAMMAR,
        CONTRACT,
        source_priority_sha256=hashlib.sha256(raw).hexdigest(),
    )
    assert report["representable_count"] == 6
    assert len(specs) == 6
    assert report["decision_counts"] == {"reject_higher_jet_regularity": 6}
    assert report["formal_backend_queue"] == []
    assert all(item["exact_static_shape_match"] is True for item in report["records"])
    assert not report["observational_data_opened"]


def test_covariant_export_writes_origin_bound_specs_and_report(tmp_path: Path) -> None:
    path = export_covariant_candidate_files(PRIORITY, GRAMMAR, CONTRACT, tmp_path)
    report = json.loads(path.read_text(encoding="utf-8"))
    spec_files = sorted((tmp_path / "specs").glob("*.json"))
    assert len(spec_files) == report["representable_count"] == 6
    lead = json.loads(
        (tmp_path / "specs" / "GF-5df8715b319f54cb.json").read_text(encoding="utf-8")
    )
    assert lead["generator_origin"]["correction_expression"] == (
        "+(q)+(sqrt(1+(x))-1)"
    )
    assert lead["generator_origin"]["source_priority_sha256"] == report[
        "input_priority_sha256"
    ]


def test_covariant_export_cli_is_explicit() -> None:
    args = _parser().parse_args(
        [
            "covariant-export",
            "--priority",
            "priority.json",
            "--output",
            "export",
        ]
    )
    assert args.command == "covariant-export"
