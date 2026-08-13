from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCUMENT = ROOT / "docs/GOALS_AND_MEASURED_OUTCOMES.md"
MATTER_RECEIPT = ROOT / "runs/math/universal-matter-coupled-pde-control-pack/receipt.json"
P55_RECEIPT = ROOT / (
    "runs/physics-language/"
    "quartic-tc2-d4-flat-reference-p55-spatial-pencil-registration-gate/campaign.json"
)
BATCH_RECEIPT = ROOT / (
    "runs/engine/continuous-scientific-pipeline-epoch-003-formal-receipt-batch-0003/result.json"
)
NATIVE_RECEIPT = ROOT / "runs/math/native-newton-blind-polynomial-tournament/receipt.json"


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_registry_contains_each_goal_once_and_retains_incomplete_boundary() -> None:
    text = DOCUMENT.read_text(encoding="utf-8")
    goal_numbers = [int(value) for value in re.findall(r"^\| (\d+) \|", text, re.MULTILINE)]
    assert goal_numbers == list(range(1, 42))
    assert "strong alpha; not yet scientifically or operationally complete" in text
    assert "It is not yet a comprehensive theorem-discovery\nsystem" in text
    assert "39/39 successful jobs" in text
    assert "A terminal 39/39 successful run remains required" in text


def test_matter_and_p55_counts_are_projected_from_checked_receipts() -> None:
    text = DOCUMENT.read_text(encoding="utf-8")
    matter = _load(MATTER_RECEIPT)
    assert matter["counts"] == {
        "exact_symbolic_replays": 1,
        "floating_point_operations": 0,
        "formal_controls_bound": 6,
        "gate_blocks": 5,
        "gate_passes": 7,
        "gate_rejects": 0,
        "gates": 12,
        "sector_blocks": 2,
        "sector_passes": 1,
        "sector_rejects": 0,
        "sectors": 3,
    }
    assert "7 PASS, 5 BLOCK, 0 REJECT" in text

    p55 = _load(P55_RECEIPT)
    assert p55["counts"]["required_matrix_packets"] == 3
    assert p55["counts"]["missing_matrix_packets"] == 3
    assert p55["counts"]["required_dense_entries"] == 9_075
    assert p55["counts"]["registered_sparse_entries"] == 0
    assert p55["counts"]["minimal_polynomial_entries_reduced"] == 0
    assert "three absent 55x55 sparse pencil matrices" in text
    assert "0/3 packets, 0/144 sparse entries" in text
    assert "0/3,025 minimal-polynomial reductions" in text
    assert "30–45 minute `_symbol_data()`" in text


def test_continuous_cursor_counts_and_product_milestones_are_current() -> None:
    text = DOCUMENT.read_text(encoding="utf-8")
    batch = _load(BATCH_RECEIPT)
    counts = batch["counts"]
    assert counts["cumulative_formally_checked_candidates"] == 226
    assert counts["cumulative_newly_processed_candidates"] == 224
    assert counts["cumulative_reconciled_preserved_candidates"] == 2
    assert counts["cumulative_candidate_rejects"] == 226
    assert counts["remaining_pending_formal_receipts"] == 11_023
    assert "226 checked, 224 new, two reconciled" in text
    assert "11,023 pending" in text

    assert (ROOT / "src/sigma_theory_compiler/formula_discovery_job.py").is_file()
    assert (ROOT / "src/sigma_theory_compiler/formula_discovery_cli.py").is_file()
    assert (ROOT / "docs/formula-discovery-cli-walkthrough.md").is_file()
    assert "Product milestone achieved" in text


def test_native_independent_construction_result_is_measured_and_narrow() -> None:
    text = DOCUMENT.read_text(encoding="utf-8")
    native = _load(NATIVE_RECEIPT)
    assert native["counts"] == {
        "candidate_blocks": 0,
        "candidate_passes": 1,
        "candidate_rejects": 2,
        "candidates": 3,
        "exact_counterexamples": 2,
        "exact_identity_certificates": 1,
        "floating_point_operations": 0,
        "generator_families": 3,
        "generic_exact_solver_invocations": 0,
        "native_formula_constructions": 1,
        "worlds": 1,
    }
    winner = native["candidate_results"][0]
    assert winner["candidate"]["family"] == "symbolic_newton"
    assert winner["candidate"]["coefficients_constant_first"] == [11, -3, 0, 2, -1]
    assert winner["status"] == "PASS"
    assert "1 PASS, 2 REJECT" in text
    assert "Multi-world, non-polynomial" in text
