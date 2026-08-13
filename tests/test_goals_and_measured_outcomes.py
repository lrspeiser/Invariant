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
P55_RESULT = ROOT / (
    "runs/physics-language/quartic-tc2-d4-p55-checkpointable-materializer/result.json"
)
P55_RECURRENCE_RECEIPT = ROOT / (
    "runs/physics-language/"
    "quartic-tc2-d4-coordinate-free-symbolic-recurrence-emitter-p55-registration/"
    "campaign.json"
)
BATCH_RECEIPT = ROOT / (
    "runs/engine/continuous-scientific-pipeline-epoch-003-formal-receipt-batch-0003/result.json"
)
NATIVE_RECEIPT = ROOT / "runs/math/native-newton-blind-polynomial-tournament/receipt.json"
MAXWELL_RECEIPT = ROOT / "runs/math/maxwell-hilbert-noether-interface-gate/receipt.json"
FLUID_RECEIPT = ROOT / "runs/math/barotropic-irrotational-action-gate/receipt.json"
FLUID_STRESS_RECEIPT = ROOT / (
    "runs/math/barotropic-irrotational-stress-conservation-gate/receipt.json"
)


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
    assert "prior BLOCK receipt remains historical evidence" in text

    p55_result = _load(P55_RESULT)
    assert p55_result["counts"]["matrix_packets"] == 3
    assert p55_result["counts"]["sparse_entries"] == 144
    assert p55_result["counts"]["linearity_entries_certified"] == 3_025
    assert p55_result["counts"]["minimal_polynomial_entries_reduced"] == 3_025
    assert p55_result["counts"]["minimal_polynomial_nonzero_remainders"] == 0
    assert p55_result["claims"]["full_direction_sphere_D4_compatibility_proved"] is False
    assert "3/3 exact 55x55 P55 axis matrices" in text

    recurrence = _load(P55_RECURRENCE_RECEIPT)
    assert recurrence["status"] == (
        "block_coordinate_free_D4_recurrence_emitter_missing_298_symbolic_packets"
    )
    assert recurrence["counts"]["required_symbolic_input_packets"] == 304
    assert recurrence["counts"]["registered_symbolic_input_packets"] == 6
    assert recurrence["counts"]["missing_symbolic_input_packets"] == 298
    assert recurrence["counts"]["required_output_rows"] == 117_180
    assert recurrence["counts"]["emitted_output_rows"] == 0
    assert recurrence["counts"]["phase_two_solve_attempts"] == 0
    assert recurrence["claims"]["full_direction_sphere_D4_compatibility_proved"] is False
    assert "3/304 to 6/304 registered" in text
    assert "0/117,180 coefficient rows" in text


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


def test_maxwell_followup_records_profile_success_and_arbitrary_background_block() -> None:
    text = DOCUMENT.read_text(encoding="utf-8")
    maxwell = _load(MAXWELL_RECEIPT)
    assert maxwell["decision"] == "BOUNDED_PASS_WITH_TYPED_BLOCK"
    assert maxwell["counts"]["exact_noether_residuals"] == 12
    assert maxwell["counts"]["exact_structural_residuals"] == 1
    assert maxwell["counts"]["blocks"] == 1
    assert maxwell["claims"]["dedicated_maxwell_registered_profile_interface_closed"] is True
    assert maxwell["claims"]["dedicated_maxwell_arbitrary_background_interface_closed"] is False
    assert "12 exact Noether residuals" in text
    assert "arbitrary curved metric/profile closure remains one typed BLOCK" in text


def test_fluid_followups_close_action_and_stress_only() -> None:
    text = DOCUMENT.read_text(encoding="utf-8")
    fluid = _load(FLUID_RECEIPT)
    assert fluid["decision"] == "PASS_EARLIEST_GATE_ONLY"
    assert [item["outcome"] for item in fluid["gate_results"]] == [
        "PASS",
        "NOT_EVALUATED",
        "NOT_EVALUATED",
        "NOT_EVALUATED",
    ]
    assert fluid["exact_replay"]["equation_of_state"] == "p=rho/3"
    assert fluid["claims"]["vortical_flows_covered"] is False
    assert fluid["claims"]["universal_matter_closure_established"] is False
    assert "vortical flow is excluded" in text

    stress = _load(FLUID_STRESS_RECEIPT)
    assert stress["decision"] == "PASS_SECOND_GATE_ONLY"
    assert stress["counts"] == {
        "blocks": 0,
        "gates_not_evaluated": 2,
        "negative_controls": 2,
        "new_gates_passed": 1,
        "predecessor_gates": 1,
        "registered_exact_residuals": 4,
        "rejects": 0,
        "sectors": 1,
        "specialized_exact_residual_coefficients": 3,
    }
    assert [item["outcome"] for item in stress["gate_results"]] == [
        "PREDECESSOR_PASS",
        "PASS",
        "NOT_EVALUATED",
        "NOT_EVALUATED",
    ]
    assert stress["claims"]["stress_conservation_gate_closed"] is True
    assert stress["claims"]["hyperbolicity_gate_closed"] is False
    assert stress["claims"]["constraint_propagation_gate_closed"] is False
    assert "second receipt now passes the on-shell stress-energy conservation gate" in text
    assert "Hyperbolicity and constraint propagation remain NOT_EVALUATED" in text
