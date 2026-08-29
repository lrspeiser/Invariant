from __future__ import annotations

import copy
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from sigma_theory_compiler import gravity_matter_lensing_external_metric_principal_symbol as symbol


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _stored_receipt() -> dict:
    return json.loads((_root() / symbol.OUTPUT_PATH).read_text(encoding="utf-8"))


def test_deterministic_receipt_rebuild_and_restricted_claims() -> None:
    receipt = symbol.check_receipt(_root())
    assert receipt == _stored_receipt()
    assert receipt["counts"]["symbolic_checks"] == 28
    assert receipt["counts"]["symbolic_checks_passed"] == 28
    assert receipt["counts"]["numeric_probes"] == 2
    assert receipt["counts"]["numeric_probes_passed"] == 2
    assert receipt["counts"]["designed_failures_preserved"] == 1
    assert receipt["adjudication"]["H3_scalar_external_metric"].startswith("PARTIAL_")
    assert receipt["adjudication"]["H4_constant_coefficient"].startswith("PARTIAL_")
    assert all(value is False for value in receipt["claim_boundary"].values())
    assert all(value == 0 for value in receipt["zero_access_and_compute"].values())


def test_general_symbol_independent_routes_and_exact_determinant_are_bound() -> None:
    receipt = _stored_receipt()
    suite = receipt["symbolic_suite"]
    assert suite["derivation_routes"] == [
        "negative exact quadratic gradient Hessian",
        "independent linearization of Euler-Lagrange fluxes",
    ]
    by_id = {item["check_id"]: item for item in suite["checks"]}
    for check_id in (
        "S01_QUADRATIC_HESSIAN_PHI_PHI",
        "S03_QUADRATIC_HESSIAN_CROSS",
        "S04_LINEARIZED_EOM_PHI_PHI",
        "S06_LINEARIZED_EOM_CROSS",
        "S07_INDEPENDENT_SYMBOL_MATCH",
        "S08_GENERAL_DETERMINANT",
    ):
        assert by_id[check_id]["passed"] is True
    expressions = suite["expressions"]["general_local_jet"]
    assert expressions["matrix"] == [
        ["C*k2-C_X*vk^2", "-Z_X*vk*wk"],
        ["-Z_X*vk*wk", "Z*k2"],
    ]
    assert expressions["determinant"] == "C*Z*k2^2-C_X*Z*k2*vk^2-Z_X^2*vk^2*wk^2"


def test_u_one_third_obstruction_and_both_numeric_sides_are_preserved() -> None:
    receipt = _stored_receipt()
    obstruction = receipt["designed_obstruction"]
    assert obstruction["no_go_claim"] is False
    assert "(1-3*u)" in obstruction["exact_identity"]
    probes = {item["probe_id"]: item for item in receipt["numeric_suite"]["probes"]}
    below = probes["below_u_one_third"]
    above = probes["above_u_one_third_designed_failure"]
    assert below["gate_contribution_sign"] == "positive"
    assert below["kinetic_determinant_sign"] == "positive"
    assert above["C"] != "0"
    assert above["Z"] != "0"
    assert above["K_phi_phi"] != "0"
    assert above["gate_contribution_sign"] == "negative"
    assert above["kinetic_determinant_sign"] == "negative"


def test_timelike_spacelike_transition_and_regression_checks_are_exact() -> None:
    receipt = _stored_receipt()
    by_id = {item["check_id"]: item for item in receipt["symbolic_suite"]["checks"]}
    for check_id in (
        "S09_TIMELIKE_KINETIC_MATRIX",
        "S11_TIMELIKE_DETERMINANT",
        "S13_SPACELIKE_TIME_BLOCK",
        "S15_SPACELIKE_LONGITUDINAL_BLOCK",
        "S16_SPACELIKE_LONGITUDINAL_DETERMINANT",
        "S21_XPHI_ZERO_REGULAR_FORM",
        "S22_CONSTANT_Z_YUKAWA_SYMBOL",
        "S23_CONSTANT_Z_YUKAWA_RANGE",
        "S24_CHI_GRADIENT_ZERO_REGRESSION",
        "S25_DEEP_AQUAL_LONGITUDINAL_RATIO",
        "S26_DEEP_AQUAL_TRANSITION_DEGENERACY",
        "S27_TIMELIKE_COMMON_COVECTOR_PRECHECK",
        "S28_SPACELIKE_COMMON_COVECTOR_PRECHECK",
    ):
        assert by_id[check_id]["passed"] is True
    deep = receipt["symbolic_suite"]["expressions"]["gate_transition_and_regressions"]["deep_AQUAL"]
    assert deep["longitudinal_speed_squared"] == "2"
    assert "vanish" in deep["transition"]
    common_cone = receipt["symbolic_suite"]["expressions"]["aligned_blocks"][
        "physical_metric_common_cone_precheck"
    ]
    assert common_cone["condition"] == "A>0 and Delta>0"
    assert "algebraic local-cone precheck only" in common_cone["scope"]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["predecessor_bindings"][1].__setitem__("receipt_file_sha256", "0" * 64),
        lambda value: value["principal_symbol_contract"]["matrix"][0].__setitem__(0, "C*k2"),
        lambda value: value["designed_obstruction"].__setitem__("no_go_claim", True),
        lambda value: value["machine_check_contract"]["frozen_numeric_probes"][1].__setitem__(
            "expected_determinant_sign", "positive"
        ),
        lambda value: value["adjudication"].__setitem__("full_H3", True),
        lambda value: value["claim_boundary"].__setitem__("healthy_action_established", True),
        lambda value: value["zero_access_and_compute"].__setitem__("network_calls", 1),
    ],
)
def test_nested_config_mutations_fail_closed(mutation) -> None:
    config = symbol.load_config(_root())
    changed = copy.deepcopy(config)
    mutation(changed)
    with pytest.raises(symbol.GravityMatterLensingPrincipalSymbolError):
        symbol.validate_config(changed)


def test_predecessor_receipt_tampering_fails_before_derivation(tmp_path: Path) -> None:
    config = symbol.load_config(_root())
    for binding in config["predecessor_bindings"]:
        for key in ("config_path", "module_path", "test_path", "receipt_path"):
            relative = Path(binding[key])
            destination = tmp_path / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(_root() / relative, destination)
    target = tmp_path / config["predecessor_bindings"][1]["receipt_path"]
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(
        symbol.GravityMatterLensingPrincipalSymbolError, match="predecessor changed"
    ):
        symbol._validate_predecessors(tmp_path, config)


def test_receipt_nested_mutation_fails_closed() -> None:
    config = symbol.load_config(_root())
    changed = copy.deepcopy(_stored_receipt())
    changed["numeric_suite"]["designed_failure_preserved"] = False
    with pytest.raises(
        symbol.GravityMatterLensingPrincipalSymbolError, match="content hash changed"
    ):
        symbol.validate_receipt(changed, config)


def test_atomic_no_replace_and_race(tmp_path: Path) -> None:
    target = tmp_path / "receipt.json"
    payload = b'{"frozen":true}\n'

    def publish(_: int) -> str:
        try:
            return symbol._atomic_no_replace(target, payload)
        except symbol.GravityMatterLensingPrincipalSymbolError:
            return "RACE_REJECTED"

    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(publish, range(16)))
    assert outcomes.count("CREATED") == 1
    assert set(outcomes) <= {"CREATED", "EXISTING_IDENTICAL", "RACE_REJECTED"}
    assert target.read_bytes() == payload
    with pytest.raises(
        symbol.GravityMatterLensingPrincipalSymbolError,
        match="refusing to overwrite different receipt",
    ):
        symbol._atomic_no_replace(target, b'{"frozen":false}\n')
