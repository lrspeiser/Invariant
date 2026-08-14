from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import pytest
import sympy as sp

from sigma_theory_compiler.system10_cylindrical_r_positive_gravity_scalar_aw_materializer import (
    _canonical_lf_sha,
    _canonical_sha,
)
from sigma_theory_compiler.system10_cylindrical_r_positive_twelve_candidate_aw_tube_solve import (
    DECISION,
    System10TwelveCandidateAWTubeSolveError,
    _tube_matrix,
    _verify_solution_packet,
    build_census_receipt,
    build_solution_packet,
    run_solutions,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/system10_cylindrical_r_positive_twelve_candidate_aw_tube_solve.json"
SOLUTIONS = ROOT / "runs/math/system10-cylindrical-r-positive-twelve-candidate-aw-tube-solve"
RECEIPT = SOLUTIONS / "receipt.json"
AW_PACKETS = ROOT / "runs/math/system10-cylindrical-r-positive-twelve-candidate-aw"


@pytest.fixture(scope="module")
def config() -> dict[str, object]:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def receipt() -> dict[str, object]:
    return build_census_receipt(CONFIG, SOLUTIONS)


def test_committed_census_replays_exactly(receipt: dict[str, object]) -> None:
    assert receipt == json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert receipt["decision"] == DECISION
    assert receipt["counts"] == {
        "candidate_solution_packets": 12,
        "accelerations_solved": 132,
        "zero_residuals_replayed": 132,
        "candidate_passes": 12,
        "candidate_blocks": 0,
    }


def test_all_atomic_solution_packets_and_entry_seals_replay(
    config: dict[str, object],
) -> None:
    for index in range(12):
        packet = json.loads((SOLUTIONS / f"solution-{index:02d}.json").read_text())
        _verify_solution_packet(packet, index, config)
        assert len(packet["accelerations"]) == 11
        assert len(packet["residual_replay"]["entries"]) == 11
        for row, entry in enumerate(packet["accelerations"]):
            assert entry["entry_sha256"] == _canonical_sha(
                {
                    "row": row,
                    "expression": entry["expression"],
                    "denominator": entry["denominator"],
                }
            )


def test_two_candidate_solutions_rebuild_deterministically() -> None:
    assert build_solution_packet(CONFIG, 0) == json.loads(
        (SOLUTIONS / "solution-00.json").read_text()
    )
    assert build_solution_packet(CONFIG, 6) == json.loads(
        (SOLUTIONS / "solution-06.json").read_text()
    )


def test_all_132_residuals_replay_independently_against_source_A_packets() -> None:
    w_values = sp.Matrix(sp.symbols("W_0:11"))
    for index in range(12):
        aw_packet = json.loads((AW_PACKETS / f"candidate-{index:02d}.json").read_text())
        solution = json.loads((SOLUTIONS / f"solution-{index:02d}.json").read_text())
        matrix, _ = _tube_matrix(aw_packet)
        accelerations = sp.Matrix(
            [sp.sympify(entry["expression"]) for entry in solution["accelerations"]]
        )
        assert [sp.factor(value) for value in matrix * accelerations + w_values] == [0] * 11


def test_W_symbols_bind_explicit_candidate_W_entries() -> None:
    for index in range(12):
        aw_packet = json.loads((AW_PACKETS / f"candidate-{index:02d}.json").read_text())
        solution = json.loads((SOLUTIONS / f"solution-{index:02d}.json").read_text())
        for row, binding in enumerate(solution["sealed_W_inputs"]):
            assert binding["symbol"] == f"W_{row}"
            assert (
                binding["source_W_entry_sha256"]
                == aw_packet["rows"][row]["W_entry"]["entry_sha256"]
            )


def test_every_solution_uses_a_strictly_positive_certified_determinant_margin(
    receipt: dict[str, object],
) -> None:
    assert all(
        Fraction(result["exact_absolute_determinant_lower_bound"]) > 0
        for result in receipt["candidate_results"]
    )
    assert min(
        Fraction(result["exact_absolute_determinant_lower_bound"])
        for result in receipt["candidate_results"]
    ) == Fraction(971540578125, 68719476736)


def test_candidate_solution_formulas_are_not_reused_as_one_answer() -> None:
    first = json.loads((SOLUTIONS / "solution-00.json").read_text())
    second = json.loads((SOLUTIONS / "solution-01.json").read_text())
    assert first["coefficients"] != second["coefficients"]
    assert first["accelerations"] != second["accelerations"]
    assert first["content_sha256"] != second["content_sha256"]


def test_resume_reuses_verified_solution_and_tamper_fails(tmp_path: Path) -> None:
    first = run_solutions(CONFIG, tmp_path, [0], root=ROOT)
    second = run_solutions(CONFIG, tmp_path, [0], root=ROOT)
    assert first == second
    target = tmp_path / "solution-00.json"
    target.write_text("{}\n", encoding="utf-8")
    with pytest.raises(System10TwelveCandidateAWTubeSolveError, match="solution packet seal"):
        run_solutions(CONFIG, tmp_path, [0], root=ROOT)
    assert target.read_text(encoding="utf-8") == "{}\n"


def test_tube_and_predecessor_tamper_fail_closed(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["caps"]["real_v_10_interval"] = ["-1/2", "1/2"]
    tampered = tmp_path / "wide-tube.json"
    tampered.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(System10TwelveCandidateAWTubeSolveError, match="caps changed"):
        build_solution_packet(tampered, 0, root=ROOT)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["predecessor"]["ordered_packet_set_sha256"] = "0" * 64
    tampered = tmp_path / "packet-set.json"
    tampered.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(System10TwelveCandidateAWTubeSolveError, match="packet-set mismatch"):
        build_solution_packet(tampered, 0, root=ROOT)


def test_source_receipt_and_scope_are_closed(
    config: dict[str, object], receipt: dict[str, object]
) -> None:
    for binding in config["source_evidence"].values():
        assert _canonical_lf_sha(ROOT / binding["path"]) == binding["canonical_lf_sha256"]
    body = {key: value for key, value in receipt.items() if key != "content_sha256"}
    assert receipt["content_sha256"] == _canonical_sha(body)
    claims = receipt["claims"]
    assert claims["all_twelve_common_tube_accelerations_solved"] is True
    assert claims["all_twelve_common_tube_residuals_replayed"] is True
    assert claims["all_twelve_global_domains_solved"] is False
    assert claims["full_rhs"] is False
    assert claims["propagation"] is False
    assert claims["hyperbolicity"] is False
