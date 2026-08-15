from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.system10_cylindrical_open_r_divq_c_factorization import (
    DECISION,
    _sealed,
    build_receipt,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/system10_cylindrical_open_r_divq_c_factorization.json"
OUTPUT = ROOT / "runs/math/system10-cylindrical-open-r-divq-c-factorization"


@pytest.fixture(scope="module")
def receipt() -> dict:
    return build_receipt(CONFIG, root=ROOT)


def test_all_four_factorizations_replay_exact_divq_rows(receipt: dict) -> None:
    assert receipt["decision"] == DECISION
    assert receipt["counts"]["divQ_to_C_factorization_rows_closed"] == 4
    assert receipt["counts"]["expanded_operator_terms"] == 191
    rows = receipt["materialization"]["factorization_rows"]
    assert [row["expanded_term_count"] for row in rows] == [43, 55, 50, 43]
    assert all(row["exact_termwise_replay"] for row in rows)
    assert all(row["expanded_row_sha256"] == row["registered_divQ_row_sha256"] for row in rows)


def test_factorization_is_bound_to_all_registered_C_rows(receipt: dict) -> None:
    rows = receipt["materialization"]["factorization_rows"]
    assert all(row["input_rows"] == [f"modified_harmonic_C[{i}]" for i in range(4)] for row in rows)
    assert all(len(set(row["input_row_sha256"])) == 4 for row in rows)
    inventory = receipt["materialization"]["operator_program"]["operator_derivative_inventory"]
    assert inventory["registered_C_operator_supports"] == [8, 8, 8, 8]
    assert inventory["nonzero_first_covariant_derivative_operator_nodes"] > 0
    assert inventory["nonzero_second_covariant_derivative_operator_nodes"] > 0
    assert "-r" in inventory["nonzero_cylindrical_connection_values"]


def test_factorization_closes_for_all_candidates_but_propagation_does_not(receipt: dict) -> None:
    assert len(receipt["candidate_results"]) == 12
    assert all(
        "PASS_COMMON_DIVQ_C_FACTORIZATION" in item["outcome"]
        for item in receipt["candidate_results"]
    )
    blocker = receipt["materialization"]["propagation_audit"]["first_missing_primitive"]
    assert (
        blocker["status"] == "BLOCK_INITIAL_DATA_MAP_UNREGISTERED_AFTER_EXACT_DIVQ_C_FACTORIZATION"
    )
    assert blocker["required_candidate_maps"] == 12
    assert blocker["registered_candidate_maps"] == 0
    assert receipt["claims"]["constraint_propagation_closed"] is False


def test_corrupted_sign_connection_term_and_C_authority_reject(receipt: dict) -> None:
    controls = receipt["materialization"]["negative_controls"]
    assert set(controls) == {
        "flip_Q_sign",
        "drop_cylindrical_connection",
        "drop_first_expanded_term",
        "replace_registered_C_row",
    }
    assert all(control["rejected"] is True for control in controls.values())


def test_checked_receipt_replays_and_tamper_fails(receipt: dict) -> None:
    assert receipt == json.loads((OUTPUT / "receipt.json").read_text(encoding="utf-8"))
    assert _sealed(receipt)
    tampered = copy.deepcopy(receipt)
    tampered["materialization"]["factorization_rows"][0]["difference_nonzero_terms"] = 1
    assert not _sealed(tampered)
