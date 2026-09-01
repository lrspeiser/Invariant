from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from sigma_theory_compiler import gravity_matter_lensing_kinetic_gate_common_time as common

ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict[str, object]:
    return json.loads((ROOT / common.CONFIG_PATH).read_text(encoding="utf-8"))


def test_symbolic_constant_coefficient_reduction() -> None:
    checks = common.symbolic_checks()
    assert all(checks.values())
    assert checks["decoupled_characteristic_polynomial"]
    assert checks["positive_energy_matrix"]
    assert checks["symmetric_first_order_symbol"]
    assert checks["complete_real_first_order_spectrum"]
    assert checks["dt_scalar_form_negative"]


def test_numeric_cases_have_positive_real_speeds_and_invariant_reduction() -> None:
    records = common.numeric_suite(_config())
    assert len(records) == 2
    for record in records:
        assert min(float(value) for value in record["K_eigenvalues"]) > 0.0
        assert min(float(value) for value in record["G_eigenvalues"]) > 0.0
        assert min(float(value) for value in record["generalized_speed_squared"]) > 0.0
        assert float(record["canonical_K_max_error"]) < 2.0e-12
        assert float(record["canonical_G_max_error"]) < 2.0e-12
        assert record["field_redefinition_invariant"] is True
        assert record["positive_principal_energy_probe"] is True


def test_common_dt_is_not_a_subluminality_claim() -> None:
    config = common.load_config(ROOT)
    claims = config["claim_boundary"]
    assert claims["local_common_time_covector_with_metric"] is True
    assert claims["cone_straddling_retained"] is True
    assert claims["local_causal_paradox_established"] is False
    assert claims["global_time_function_established"] is False


def test_first_numeric_case_reproduces_cone_straddling_witness() -> None:
    record = common.numeric_suite(_config())[0]
    values = [float(value) for value in record["generalized_speed_squared"]]
    assert values[0] == pytest.approx(0.6528475788600139, rel=2.0e-15)
    assert values[1] == pytest.approx(1.5979982339788417, rel=2.0e-15)


def test_general_linear_field_redefinition_preserves_speeds() -> None:
    case = _config()["numeric_cases"][1]
    k_matrix = np.array(case["K"], dtype=float)
    g_matrix = np.array(case["G"], dtype=float)
    original = common._generalized_speeds(k_matrix, g_matrix)
    transform = np.array([[0.8, -0.5], [0.3, 1.4]])
    changed = common._generalized_speeds(
        transform.T @ k_matrix @ transform,
        transform.T @ g_matrix @ transform,
    )
    assert np.allclose(original, changed, rtol=2.0e-12, atol=2.0e-12)


def test_predecessor_commit_and_worktree_bindings() -> None:
    config = common.load_config(ROOT)
    receipts = common._validate_bindings(ROOT, config)
    assert set(receipts) == {"EXTERNAL_METRIC_PRINCIPAL_SYMBOL", "CONE_STRADDLING_THEOREM"}
    committed = config["bindings"][0]
    for role in ("config", "module", "test", "receipt"):
        relative = committed[f"{role}_path"]
        assert (
            common._sha256_bytes(common._git_show(ROOT, committed["commit"], relative))
            == (committed[f"{role}_sha256"])
        )


def test_receipt_is_deterministic_and_retains_limits() -> None:
    first = common.build_receipt(ROOT)
    second = common.build_receipt(ROOT)
    assert first == second
    assert first["checks_passed"] == first["checks_total"] == 14
    assert all(first["checks"].values())
    assert first["claim_boundary"]["constant_coefficient_scalar_block_strongly_hyperbolic"]
    assert first["claim_boundary"]["variable_coefficient_strong_hyperbolicity"] is False
    assert first["claim_boundary"]["full_metric_scalar_matter_system_healthy"] is False
    assert all(value == 0 for value in first["access_ledger"].values())
    assert first["content_sha256"] == common._self_hash(first)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("claim_boundary", "local_causal_paradox_established", True),
        ("claim_boundary", "global_time_function_established", True),
        ("claim_boundary", "full_metric_scalar_matter_system_healthy", True),
        ("claim_boundary", "publication_ready", True),
        ("access_ledger", "observational_rows_read", 1),
        ("admission_policy", "observational_scoring_authorized", True),
    ],
)
def test_scope_and_access_mutations_fail_closed(section: str, key: str, value: object) -> None:
    config = copy.deepcopy(_config())
    config[section][key] = value
    with pytest.raises(common.KineticGateCommonTimeError):
        common.validate_config(config)


def test_rehashed_receipt_causal_overclaim_fails_closed() -> None:
    config = common.load_config(ROOT)
    receipt = common.build_receipt(ROOT)
    receipt["claim_boundary"]["global_time_function_established"] = True
    receipt["content_sha256"] = common._self_hash(receipt)
    with pytest.raises(common.KineticGateCommonTimeError, match="claims changed"):
        common.validate_receipt(receipt, config)


def test_atomic_write_is_idempotent_and_no_clobber(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    value = {"content_sha256": "one"}
    assert common._atomic_no_clobber(path, value) == "CREATED"
    assert common._atomic_no_clobber(path, value) == "EXISTING_IDENTICAL"
    with pytest.raises(common.KineticGateCommonTimeError, match="refusing to replace"):
        common._atomic_no_clobber(path, {"content_sha256": "two"})
