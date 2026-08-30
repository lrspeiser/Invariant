from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path

import pytest

from sigma_theory_compiler import (
    gravity_matter_lensing_scalar_hamiltonian_necessary_conditions as hamiltonian,
)

ROOT = Path(__file__).resolve().parents[1]


def test_config_and_all_predecessor_commits_are_exactly_bound() -> None:
    config = hamiltonian.load_config(ROOT)
    assert [item["binding_id"] for item in config["predecessor_bindings"]] == [
        "split_gate_action",
        "flrw_necessary_conditions",
        "covariant_field_equations",
        "adm_constraint_propagation",
    ]
    assert [item["git_commit"] for item in config["predecessor_bindings"]] == [
        "03a652acaded1be4cca9af48782b8d54138e54c3",
        "9e85c1d7ae169881e6b2952e779027f837bf48f0",
        "e33fa4e408ea188a3b282662a9fc51ad6252ca90",
        "f6a9bb658795f2a099c08283ec3bb85f9a87776c",
    ]


def test_symbolic_suite_derives_all_frozen_identities() -> None:
    suite = hamiltonian.symbolic_suite()
    assert suite["all_passed"] is True
    assert tuple(item["check_id"] for item in suite["checks"]) == hamiltonian.SYMBOLIC_CHECK_IDS
    assert len(suite["checks"]) == 24
    assert suite["derived_expressions"]["effective_metric_determinant"] == "C^3 K_inv"
    assert suite["derived_expressions"]["illustrative_gate_energy_factor"] == "(1+u)(1-7u)"


def test_numeric_suite_preserves_energy_and_slice_counterexamples() -> None:
    suite = hamiltonian.numeric_suite(hamiltonian.load_config(ROOT))
    assert suite["all_passed"] is True
    assert suite["designed_failures_preserved"] == 2
    by_id = {item["case_id"]: item for item in suite["cases"]}
    energy = by_id["POSITIVE_PRINCIPAL_NEGATIVE_ENERGY"]
    assert energy["legendre_positive"] is True
    assert energy["slice_principal_positive"] is True
    assert energy["energy_positive"] is False
    slicing = by_id["LARGE_SPATIAL_GRADIENT_INVALID_ADM_NORMAL"]
    assert slicing["K_inv"] > 0
    assert slicing["K_ADM"] < 0
    assert slicing["slice_principal_positive"] is False
    assert suite["max_schur_identity_absolute_error"] <= 1e-12


def test_receipt_is_deterministic_and_does_not_complete_cp11_4() -> None:
    config = hamiltonian.load_config(ROOT)
    receipt = hamiltonian.build_receipt(ROOT)
    hamiltonian.validate_receipt(receipt, config)
    assert receipt["adjudication"]["scalar_canonical_hamiltonian_derived"] is True
    assert receipt["adjudication"]["CP11_4_complete"] is False
    assert receipt["adjudication"]["physical_hamiltonian_positive"] is False
    assert receipt["claim_boundary"]["full_no_ghost_result_established"] is False
    assert receipt["claim_boundary"]["motion_and_lensing_jointly_predicted"] is False
    assert receipt["counts"]["observational_rows_opened"] == 0


def test_stored_receipt_rebuilds_exactly() -> None:
    assert hamiltonian.check_receipt(ROOT) == hamiltonian.build_receipt(ROOT)


@pytest.mark.parametrize(
    "section,mutation",
    [
        (
            "adm_legendre_contract",
            lambda value: value.__setitem__("convexity", "always positive"),
        ),
        (
            "homogeneous_energy_contract",
            lambda value: value.__setitem__("sign_transition", "always nonnegative"),
        ),
        (
            "adjudication",
            lambda value: value.__setitem__("CP11_4_complete", True),
        ),
        (
            "claim_boundary",
            lambda value: value.__setitem__("physical_hamiltonian_positivity_established", True),
        ),
    ],
)
def test_coherent_config_section_mutations_fail_closed(
    monkeypatch: pytest.MonkeyPatch, section: str, mutation: object
) -> None:
    original = json.loads((ROOT / hamiltonian.CONFIG_PATH).read_text(encoding="utf-8"))
    changed = deepcopy(original)
    mutation(changed[section])  # type: ignore[operator]
    monkeypatch.setattr(hamiltonian, "_read_json", lambda _path: changed)
    monkeypatch.setattr(hamiltonian, "EXPECTED_CONFIG_CONTENT_SHA256", hamiltonian._sha(changed))
    with pytest.raises(hamiltonian.ScalarHamiltonianError, match="config section changed"):
        hamiltonian.load_config(ROOT)


def test_receipt_overclaim_is_rejected_even_when_rehashed() -> None:
    config = hamiltonian.load_config(ROOT)
    receipt = hamiltonian.build_receipt(ROOT)
    changed = deepcopy(receipt)
    changed["adjudication"]["physical_hamiltonian_positive"] = True
    body = dict(changed)
    body.pop("content_sha256")
    changed["content_sha256"] = hamiltonian._sha(body)
    with pytest.raises(hamiltonian.ScalarHamiltonianError, match="adjudication changed"):
        hamiltonian.validate_receipt(changed, config)


def test_zero_access_contract_is_literal_zero() -> None:
    config = hamiltonian.load_config(ROOT)
    assert set(config["zero_access_and_compute"]) == hamiltonian.ZERO_KEYS
    assert all(value == 0 for value in config["zero_access_and_compute"].values())


def test_atomic_no_replace_is_idempotent_and_refuses_changed_bytes(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    assert hamiltonian._atomic_no_replace(path, b"same") == "CREATED"
    assert hamiltonian._atomic_no_replace(path, b"same") == "EXISTING_IDENTICAL"
    with pytest.raises(hamiltonian.ScalarHamiltonianError, match="refusing to overwrite"):
        hamiltonian._atomic_no_replace(path, b"different")
    assert path.read_bytes() == b"same"


def test_atomic_no_replace_race_preserves_exactly_one_payload(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    payloads = (b"first", b"second")

    def publish(payload: bytes) -> str:
        try:
            return hamiltonian._atomic_no_replace(path, payload)
        except hamiltonian.ScalarHamiltonianError:
            return "REFUSED"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(publish, payloads))
    assert outcomes.count("CREATED") == 1
    assert outcomes.count("REFUSED") == 1
    assert path.read_bytes() in payloads
