from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path

import pytest

from sigma_theory_compiler import gravity_matter_lensing_adm_constraint_propagation as adm

ROOT = Path(__file__).resolve().parents[1]


def test_config_and_predecessor_are_exactly_bound() -> None:
    config = adm.load_config(ROOT)
    binding = config["predecessor_binding"]
    assert binding["git_commit"] == "e33fa4e408ea188a3b282662a9fc51ad6252ca90"
    assert binding["receipt_content_sha256"] == (
        "fa4a566651a5b195f7f5a2cffb87e6228761feabc6e2669fdd86fcc12edea8d5"
    )


def test_symbolic_suite_derives_all_required_identities() -> None:
    suite = adm.symbolic_suite()
    assert suite["all_passed"] is True
    assert tuple(item["check_id"] for item in suite["checks"]) == adm.SYMBOLIC_CHECK_IDS
    assert len(suite["checks"]) == 18


def test_coordinate_projection_uses_independent_four_dimensional_route() -> None:
    normal, spatial = adm._coordinate_projection_residuals()
    assert normal == 0
    assert spatial == 0


def test_constraint_principal_symbol_is_symmetric_hyperbolic() -> None:
    suite = adm.symbolic_suite()
    derived = suite["derived_expressions"]
    assert derived["characteristic_polynomial"] == "lambda**4 - lambda**2"
    assert derived["characteristic_speeds"] == [-1, 0, 0, 1]


def test_numeric_lapse_forms_match_normal_forms() -> None:
    config = adm.load_config(ROOT)
    suite = adm.numeric_suite(config)
    assert suite["all_passed"] is True
    assert len(suite["cases"]) == 3
    assert suite["max_absolute_error"] <= 1e-12


def test_receipt_is_deterministic_and_restrained() -> None:
    receipt = adm.build_receipt(ROOT)
    adm.validate_receipt(receipt, adm.load_config(ROOT))
    assert receipt["adjudication"]["CP11_3_complete"] is True
    assert receipt["adjudication"]["full_H2"] is False
    assert receipt["claim_boundary"]["motion_and_lensing_jointly_predicted"] is False
    assert receipt["counts"]["observational_rows_opened"] == 0


def test_stored_receipt_rebuilds_exactly() -> None:
    assert adm.check_receipt(ROOT) == adm.build_receipt(ROOT)


def test_config_content_mutation_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    original = json.loads((ROOT / adm.CONFIG_PATH).read_text(encoding="utf-8"))
    changed = deepcopy(original)
    changed["constraint_propagation_contract"]["characteristic_speeds_relative_to_normal"] = [
        -2,
        0,
        0,
        2,
    ]
    monkeypatch.setattr(adm, "_read_json", lambda _path: changed)
    monkeypatch.setattr(adm, "EXPECTED_CONFIG_CONTENT_SHA256", adm._sha(changed))
    with pytest.raises(adm.AdmConstraintPropagationError, match="config section changed"):
        adm.load_config(ROOT)


def test_receipt_claim_mutation_is_rejected() -> None:
    config = adm.load_config(ROOT)
    receipt = adm.build_receipt(ROOT)
    changed = deepcopy(receipt)
    changed["adjudication"]["full_H2"] = True
    content = dict(changed)
    content.pop("content_sha256")
    changed["content_sha256"] = adm._sha(content)
    with pytest.raises(adm.AdmConstraintPropagationError):
        adm.validate_receipt(changed, config)


def test_zero_access_contract_is_literal_zero() -> None:
    config = adm.load_config(ROOT)
    assert set(config["zero_access_and_compute"]) == adm.ZERO_KEYS
    assert all(value == 0 for value in config["zero_access_and_compute"].values())


def test_atomic_no_replace_preserves_different_existing_bytes(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    path.write_bytes(b"existing")
    with pytest.raises(adm.AdmConstraintPropagationError, match="refusing to overwrite"):
        adm._atomic_no_replace(path, b"different")
    assert path.read_bytes() == b"existing"


def test_atomic_no_replace_is_idempotent_for_identical_bytes(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    assert adm._atomic_no_replace(path, b"same") == "CREATED"
    assert adm._atomic_no_replace(path, b"same") == "EXISTING_IDENTICAL"
    assert path.read_bytes() == b"same"


def test_atomic_no_replace_race_preserves_exactly_one_payload(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    payloads = (b"first", b"second")

    def publish(payload: bytes) -> str:
        try:
            return adm._atomic_no_replace(path, payload)
        except adm.AdmConstraintPropagationError:
            return "REFUSED"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(publish, payloads))
    assert outcomes.count("CREATED") == 1
    assert outcomes.count("REFUSED") == 1
    assert path.read_bytes() in payloads
