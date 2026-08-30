from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import pytest

from sigma_theory_compiler import (
    gravity_matter_lensing_solar_gw_necessary_conditions as necessary,
)

ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict[str, object]:
    return json.loads((ROOT / necessary.CONFIG_PATH).read_text(encoding="utf-8"))


def test_symbolic_suite_passes_exact_inventory() -> None:
    suite = necessary.run_symbolic_suite()
    assert tuple(item["check_id"] for item in suite["checks"]) == necessary.SYMBOLIC_CHECK_IDS
    assert suite["all_passed"] is True
    assert len(suite["checks"]) == 16
    assert all(item["residual"] == "0" for item in suite["checks"])


def test_conformal_ppn_relation_is_derived_with_exact_threshold_budget() -> None:
    config = _config()
    solar = config["conformal_solar_contract"]
    assert "Phi=-(1+epsilon)U" in solar["linear_physical_potentials"]
    assert "Psi=-(1-epsilon)U" in solar["linear_physical_potentials"]
    assert "gamma_PPN-1=-2*epsilon/(1+epsilon)" in solar["ppn_derivation"]
    assert "=-(partial_a ln A)*T_E" in solar["matter_variation_derivation"]
    assert (
        "-delta_S/(2+delta_S)<=epsilon<=delta_S/(2-delta_S)" in solar["signed_necessary_interval"]
    )
    assert "23/1999977" in solar["attractive_necessary_ceiling"]
    assert "23/3999954" in solar["canonical_coupling_subcase"]


def test_ppn_scope_does_not_apply_one_gamma_to_yukawa_profiles() -> None:
    scope = _config()["frozen_background_scope"]
    assert "Only when a=-epsilon*U with constant epsilon" in scope["ppn_subcase"]
    assert "Yukawa or nonlinear screened profile" in scope["ppn_subcase"]
    assert "not an on-shell solution" in scope["solar"]


def test_restricted_yukawa_force_ratio_and_high_acceleration_limit_are_bounded() -> None:
    solar = _config()["conformal_solar_contract"]
    assert "Q_chi=alpha_chi*rho/M_Pl" in solar["split_chi_yukawa_subcase"]
    assert "epsilon_chi(r)=2*alpha_chi^2" in solar["split_chi_yukawa_subcase"]
    assert "epsilon_chi(r)->0" in solar["high_acceleration_implication"]
    assert "does not screen the phi conformal force" in solar["high_acceleration_implication"]
    numeric = necessary.run_numeric_suite(_config())
    factors = [
        float(item["force_suppression_relative_to_massless"])
        for item in numeric["yukawa_force_probes"]
    ]
    assert factors == sorted(factors, reverse=True)
    assert factors[0] == 1.0
    assert all(item["bounded_between_zero_and_one"] for item in numeric["yukawa_force_probes"])


def test_conformal_and_disformal_cone_implications_remain_distinct() -> None:
    cones = _config()["cone_contract"]
    assert "c_GW/c_gamma=1" in cones["conformal_photon_cone"]
    assert "c_gamma=sqrt(Delta)" in cones["homogeneous_disformal_cones"]
    assert "c_GW/c_gamma=Delta^(-1/2)" in cones["homogeneous_disformal_cones"]
    assert "c_GW,r/c_gamma,r=sqrt(1+s)" in cones["static_radial_disformal_cones"]
    assert "Tangential photon speed is unchanged" in cones["static_radial_disformal_cones"]
    assert cones["conformal_gate_status"].startswith("BLOCKED_")
    assert cones["disformal_gate_status"].startswith("BLOCKED_")


def test_thresholds_are_inherited_not_revalidated_measurements() -> None:
    thresholds = _config()["threshold_provenance"]
    assert thresholds["solar_gamma_absolute_ceiling"] == "23/1000000"
    assert thresholds["gw_fractional_speed_ceiling"] == "1/1000000000000000"
    assert "Inherited internal theory-gate thresholds only" in thresholds["scope"]
    assert thresholds["measurement_claim"] is False


def test_both_physical_gates_and_all_observational_claims_remain_blocked() -> None:
    config = _config()
    adjudication = config["gate_adjudication"]
    assert adjudication["solar_gate_passed"] is False
    assert adjudication["gw_gate_passed"] is False
    assert adjudication["physical_matter_source_derived"] is False
    assert adjudication["solar_on_shell_background_solved"] is False
    assert adjudication["late_time_on_shell_background_solved"] is False
    claims = config["claim_boundary"]
    assert claims["restricted_necessary_conditions_established"] is True
    assert all(
        value is False
        for key, value in claims.items()
        if key != "restricted_necessary_conditions_established"
    )
    assert all(value == 0 for value in config["zero_access_and_compute"].values())


def test_receipt_is_deterministic_strict_and_zero_data() -> None:
    first = necessary.build_receipt(ROOT)
    second = necessary.build_receipt(ROOT)
    assert first == second
    assert first["counts"]["symbolic_checks_passed"] == 16
    assert first["counts"]["numeric_yukawa_probes_passed"] == 3
    assert first["counts"]["designed_blocked_gates"] == 2
    assert first["counts"]["observational_rows_opened"] == 0
    assert len(first["remaining_blockers"]) == 7
    necessary.validate_receipt(first, necessary.load_config(ROOT))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda c: c["threshold_provenance"].update({"solar_gamma_absolute_ceiling": "1"}),
            "threshold",
        ),
        (
            lambda c: c["conformal_solar_contract"].update({"ppn_derivation": "assumed"}),
            "PPN relation",
        ),
        (
            lambda c: c["cone_contract"].update({"conformal_gate_status": "PASS"}),
            "cone gate unblocked",
        ),
        (
            lambda c: c["gate_adjudication"].update({"solar_gate_passed": True}),
            "blocked gate overclaimed",
        ),
        (
            lambda c: c["claim_boundary"].update({"gw_speed_observational_pass_established": True}),
            "claim boundary overstated",
        ),
        (
            lambda c: c["zero_access_and_compute"].update({"observational_rows_opened": 1}),
            "access state changed",
        ),
    ],
)
def test_semantic_mutations_fail_closed(
    monkeypatch: pytest.MonkeyPatch, mutation: object, message: str
) -> None:
    config = copy.deepcopy(_config())
    mutation(config)
    monkeypatch.setattr(necessary, "EXPECTED_CONFIG_CONTENT_SHA256", necessary._sha(config))
    with pytest.raises(necessary.SolarGwNecessaryConditionsError, match=message):
        necessary.validate_config(config)


def test_unknown_config_key_fails_strict_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    config = copy.deepcopy(_config())
    config["extra"] = "not allowed"
    monkeypatch.setattr(necessary, "EXPECTED_CONFIG_CONTENT_SHA256", necessary._sha(config))
    with pytest.raises(necessary.SolarGwNecessaryConditionsError, match="config schema"):
        necessary.validate_config(config)


def test_every_predecessor_file_and_receipt_is_bound() -> None:
    config = necessary.load_config(ROOT)
    necessary._validate_predecessors(ROOT, config)
    assert [item["git_commit"][:8] for item in config["predecessor_bindings"]] == [
        "27d8cae5",
        "d1e2491b",
        "03a652ac",
        "7216ff73",
    ]


def test_predecessor_mutation_fails_closed(tmp_path: Path) -> None:
    config = _config()
    needed = [necessary.CONFIG_PATH, necessary.SOURCE_PATH, necessary.TEST_PATH]
    for binding in config["predecessor_bindings"]:
        needed.extend(
            Path(binding[key])
            for key in ("config_path", "module_path", "test_path", "receipt_path")
        )
    for relative in needed:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    changed = tmp_path / config["predecessor_bindings"][0]["receipt_path"]
    changed.write_bytes(changed.read_bytes() + b"\n")
    with pytest.raises(necessary.SolarGwNecessaryConditionsError, match="predecessor changed"):
        necessary.build_receipt(tmp_path)


def test_receipt_claim_and_count_mutations_fail_closed() -> None:
    config = necessary.load_config(ROOT)
    receipt = necessary.build_receipt(ROOT)
    receipt["claim_boundary"]["solar_viability_established"] = True
    body = dict(receipt)
    body.pop("content_sha256")
    receipt["content_sha256"] = necessary._sha(body)
    with pytest.raises(necessary.SolarGwNecessaryConditionsError, match="claims changed"):
        necessary.validate_receipt(receipt, config)

    receipt = necessary.build_receipt(ROOT)
    receipt["counts"]["observational_rows_opened"] = 1
    body = dict(receipt)
    body.pop("content_sha256")
    receipt["content_sha256"] = necessary._sha(body)
    with pytest.raises(necessary.SolarGwNecessaryConditionsError, match="receipt access"):
        necessary.validate_receipt(receipt, config)


def test_atomic_publication_is_idempotent_and_no_clobber(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    payload = b"sealed\n"
    assert necessary._atomic_no_replace(path, payload) == "CREATED"
    assert necessary._atomic_no_replace(path, payload) == "EXISTING_IDENTICAL"
    with pytest.raises(necessary.SolarGwNecessaryConditionsError, match="refusing to overwrite"):
        necessary._atomic_no_replace(path, b"different\n")
    assert path.read_bytes() == payload
