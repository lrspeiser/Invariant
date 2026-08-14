from __future__ import annotations

import json
from pathlib import Path

import pytest

from sigma_theory_compiler.system10_cylindrical_sourced_constraint_row_materializer import (
    System10CylindricalSourcedConstraintRowError,
    build_receipt,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/system10_cylindrical_sourced_constraint_row_materializer.json"
RECEIPT = ROOT / "runs/math/system10-cylindrical-sourced-constraint-row-materializer/receipt.json"


@pytest.fixture(scope="module")
def receipt() -> dict[str, object]:
    return build_receipt(CONFIG)


def test_committed_receipt_replays_exactly(receipt: dict[str, object]) -> None:
    assert receipt == json.loads(RECEIPT.read_text(encoding="utf-8"))


def test_accelerations_cancel_before_integrability_substitution(
    receipt: dict[str, object],
) -> None:
    proof = receipt["materialization"]["acceleration_and_integrability_proof"]
    assert proof["raw_rows_checked"] == 4
    assert proof["partial_0_v_atoms_checked"] == 17
    assert proof["partial_0_v_nonzero_coefficients"] == 0
    assert proof["partial_0_w_atoms_registered"] == 51
    assert proof["integrability_substitutions"] == 51
    assert proof["identity"] == "partial_0 w_iA=partial_i v_A"
    assert proof["forbidden_time_differential_atoms_after_replacement"] == 0


def test_all_twelve_candidates_advance_atomically_by_four_rows(
    receipt: dict[str, object],
) -> None:
    results = receipt["materialization"]["candidate_results"]
    packets = receipt["materialization"]["row_polynomials"]
    assert len(results) == 12
    assert len(packets) == 48
    assert all(item["rows_closed_atomically"] == 4 for item in results)
    assert all(item["term_count"] > 0 for item in packets)
    assert len({item["polynomial_sha256"] for item in packets}) == 48
    assert receipt["counts"]["hamiltonian_momentum_rows_closed"] == 48
    assert receipt["counts"]["specialized_physical_gravity_rows_closed"] == 96


def test_serialized_rows_use_only_spatial_differential_alphabet(
    receipt: dict[str, object],
) -> None:
    packets = receipt["materialization"]["row_polynomials"]
    forbidden = ("partial0_v_", "partial0_w_")
    for packet in packets:
        atoms = {factor["atom"] for term in packet["terms"] for factor in term["factors"]}
        assert not any(atom.startswith(forbidden) for atom in atoms)
        assert packet["coefficient_field"] == "Q(sqrt(2),kappa)"


def test_no_general_or_propagation_claim_follows(receipt: dict[str, object]) -> None:
    assert receipt["decision"].endswith("NO_PROPAGATION_CLAIM")
    assert receipt["claims"]["general_domain_closed"] is False
    assert receipt["claims"]["sourced_constraint_propagation_closed"] is False
    assert receipt["claims"]["general_hyperbolicity_closed"] is False
    assert receipt["counts"]["constraint_propagation_proofs"] == 0


def test_tamper_and_claim_broadening_fail_closed(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["bindings"]["projection_receipt"]["file_sha256"] = "0" * 64
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(System10CylindricalSourcedConstraintRowError, match="hash mismatch"):
        build_receipt(tampered, root=ROOT)

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["claims_policy"]["sourced_constraint_propagation"] = True
    broadened = tmp_path / "broadened.json"
    broadened.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(System10CylindricalSourcedConstraintRowError, match="claims policy"):
        build_receipt(broadened, root=ROOT)
