from __future__ import annotations

import copy
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from sigma_theory_compiler import (
    gravity_matter_lensing_flrw_background_necessary_conditions as flrw,
)

ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict[str, object]:
    return json.loads((ROOT / flrw.CONFIG_PATH).read_text(encoding="utf-8"))


def _reseal(receipt: dict[str, object]) -> None:
    body = dict(receipt)
    body.pop("content_sha256")
    receipt["content_sha256"] = flrw._sha(body)


def test_symbolic_suite_passes_exact_inventory_and_two_routes() -> None:
    suite = flrw.run_symbolic_suite()
    assert tuple(item["check_id"] for item in suite["checks"]) == flrw.SYMBOLIC_CHECK_IDS
    assert suite["all_passed"] is True
    assert len(suite["checks"]) == 25
    assert all(item["residual"] == "0" for item in suite["checks"])
    assert suite["checks"][0]["check_id"] == "S01_LAPSE_ENERGY_ROUTE"
    assert suite["checks"][1]["check_id"] == "S02_COVARIANT_ENERGY_ROUTE"


def test_lapse_signature_and_energy_pressure_are_exact() -> None:
    config = _config()
    conventions = config["conventions"]
    assert conventions["signature"] == "(-,+,+,+)"
    assert "N=1 only after lapse variation" in conventions["cosmic_time_gauge"]
    assert "X=dot(phi)^2/(2*N^2)" in conventions["kinetic_invariants"]
    stress = config["stress_energy_contract"]
    assert "Q*(Z-2*X*Z_X)" in stress["scalar_energy_density"]
    assert stress["scalar_pressure"] == "p_s=L_s=P-V+Y0*X_chi-Q*Z"
    assert "2*X*C+2*Y0*X_chi" in stress["enthalpy"]


def test_friedmann_raychaudhuri_and_acceleration_are_necessary_only() -> None:
    config = _config()
    equations = config["background_equations"]
    assert equations["friedmann"] == "3*M_Pl^2*H^2=rho_E+rho_s"
    assert equations["raychaudhuri"].startswith("-2*M_Pl^2*dot(H)=")
    assert "acceleration requires" in equations["acceleration_identity"]
    assert "is not established" in equations["acceleration_identity"]
    assert config["adjudication"]["accelerating_solution_exists"] is False


def test_scalar_equations_retain_gate_mixing_and_source_signs() -> None:
    config = _config()
    equations = config["background_equations"]
    assert "C*dot(phi)" in equations["phi_eom"]
    assert "alpha_phi*T_E" in equations["phi_eom"]
    assert "m_chi^2*Z*chi=alpha_chi*T_E" in equations["chi_eom"]
    assert "m_chi^2*chi*dot(chi)*Z_X" in equations["phi_gate_content"]
    assert "Z_XX*dot(X)" in equations["phi_gate_content"]
    assert "box(f)=-(ddot(f)+3*H*dot(f))" in equations["covariant_specialization"]


def test_physical_and_einstein_continuity_and_matter_limits() -> None:
    exchange = _config()["matter_exchange_contract"]
    assert "d(tilde_t)=A*dt" in exchange["physical_frame"]
    assert "=-T_E*dot(ln A)" in exchange["einstein_frame"]
    assert "=T_E*dot(ln A)" in exchange["scalar_continuity"]
    assert exchange["total_continuity"].endswith("=0")
    assert "a^-4" in exchange["radiation_probe"]
    assert "A*a^-3" in exchange["dust_probe"]
    assert "A^4" in exchange["vacuum_probe"]
    assert "not a demonstrated accelerating solution" in exchange["vacuum_probe"]


def test_gate_energy_sign_change_and_high_u_obstruction_are_preserved() -> None:
    limits = _config()["gate_limits_and_necessary_conditions"]
    assert "(1+u)*(1-7*u)" in limits["exact_gate_energy_factor"]
    assert "changes sign at u=1/7" in limits["exact_gate_energy_factor"]
    assert "rho_gate~-7*Q*u^2" in limits["high_u"]
    assert "not automatic" in limits["high_u"]
    assert "C>0" in limits["timelike_local_health"]
    assert "K=C+2*X*C_X" in limits["timelike_local_health"]
    numeric = flrw.run_numeric_suite(_config())
    assert [item["sign"] for item in numeric["gate_u_probes"]] == [
        "positive",
        "zero",
        "negative",
        "negative",
    ]


def test_conformal_cone_and_disformal_alternative_scope_are_strict() -> None:
    cones = _config()["tensor_photon_and_disformal_contract"]
    assert "c_GW/c_gamma=1" in cones["conformal_branch"]
    assert "not an observational" in cones["conformal_branch"]
    assert "not part of the committed" in cones["disformal_scope"]
    assert "delta_G=1/1000000000000000" in cones["inherited_speed_threshold"]
    assert cones["status"].startswith("BLOCKED_")
    numeric = flrw.run_numeric_suite(_config())
    assert numeric["designed_failures_preserved"] == 2
    assert [item["within_frozen_interval"] for item in numeric["disformal_q_probes"]] == [
        True,
        True,
        False,
        False,
    ]


def test_receipt_is_deterministic_restricted_and_zero_access() -> None:
    first = flrw.build_receipt(ROOT)
    second = flrw.build_receipt(ROOT)
    assert first == second
    assert first["counts"]["symbolic_checks_passed"] == 25
    assert first["counts"]["gate_u_probes_passed"] == 4
    assert first["counts"]["disformal_q_probes_passed"] == 4
    assert first["counts"]["designed_failures_preserved"] == 2
    assert first["adjudication"]["healthy_late_time_history_exists"] is False
    assert first["adjudication"]["observational_fit_performed"] is False
    assert all(value == 0 for value in first["zero_access_and_compute"].values())
    assert len(first["remaining_blockers"]) == 7


def test_all_claims_beyond_restricted_equations_are_false() -> None:
    claims = _config()["claim_boundary"]
    assert claims["restricted_flat_flrw_equations_established"] is True
    assert all(
        value is False
        for key, value in claims.items()
        if key != "restricted_flat_flrw_equations_established"
    )


def test_exact_predecessor_commits_and_artifacts_are_bound() -> None:
    config = flrw.load_config(ROOT)
    flrw._validate_predecessors(ROOT, config)
    assert [item["git_commit"] for item in config["predecessor_bindings"]] == [
        "03a652acaded1be4cca9af48782b8d54138e54c3",
        "7216ff7319e0f38c7639926b31e7f4e881f9a64a",
        "98589e269c362846154764e6b3e400fa300c2a94",
        "ad11eba2ebfc7c12f107cfbea8b969dfd05de101",
    ]
    for binding in config["predecessor_bindings"]:
        commit = binding["git_commit"]
        subprocess.run(
            ["git", "cat-file", "-e", commit],
            cwd=ROOT,
            check=True,
            capture_output=True,
        )
        object_type = subprocess.check_output(["git", "cat-file", "-t", commit], cwd=ROOT).strip()
        assert object_type == b"commit"
        for path_key in ("config_path", "module_path", "test_path", "receipt_path"):
            relative = binding[path_key]
            committed_bytes = subprocess.check_output(
                ["git", "show", f"{commit}:{relative}"], cwd=ROOT
            )
            assert committed_bytes == (ROOT / relative).read_bytes()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda c: c["conventions"].update({"signature": "(+---)"}), "signature"),
        (
            lambda c: c["conventions"].update({"cosmic_time_gauge": "N=1 first"}),
            "lapse convention",
        ),
        (
            lambda c: c["conventions"].update({"source_sign": "opposite"}),
            "source sign",
        ),
        (
            lambda c: c["stress_energy_contract"].update({"scalar_energy_density": "rho=P"}),
            "energy density",
        ),
        (
            lambda c: c["background_equations"].update({"friedmann": "changed"}),
            "Friedmann",
        ),
        (
            lambda c: c["background_equations"].update({"phi_gate_content": "omitted"}),
            "gate mixing",
        ),
        (
            lambda c: c["matter_exchange_contract"].update({"einstein_frame": "conserved"}),
            "matter exchange",
        ),
        (
            lambda c: c["gate_limits_and_necessary_conditions"].update({"high_u": "positive"}),
            "high-u obstruction",
        ),
        (
            lambda c: c["tensor_photon_and_disformal_contract"].update({"status": "PASS"}),
            "disformal gate unblocked",
        ),
        (
            lambda c: c["adjudication"].update({"healthy_late_time_history_exists": True}),
            "blocked result overclaimed",
        ),
        (
            lambda c: c["claim_boundary"].update({"cosmological_solution_established": True}),
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
    monkeypatch.setattr(flrw, "EXPECTED_CONFIG_CONTENT_SHA256", flrw._sha(config))
    with pytest.raises(flrw.FlrwNecessaryConditionsError, match=message):
        flrw.validate_config(config)


def test_predecessor_commit_semantic_mutation_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = copy.deepcopy(_config())
    config["predecessor_bindings"][0]["git_commit"] = "0" * 40
    monkeypatch.setattr(flrw, "EXPECTED_CONFIG_CONTENT_SHA256", flrw._sha(config))
    with pytest.raises(flrw.FlrwNecessaryConditionsError, match="predecessor commits"):
        flrw.validate_config(config)


def test_nested_extra_key_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    config = copy.deepcopy(_config())
    config["background_equations"]["post_result_choice"] = "forbidden"
    monkeypatch.setattr(flrw, "EXPECTED_CONFIG_CONTENT_SHA256", flrw._sha(config))
    with pytest.raises(flrw.FlrwNecessaryConditionsError, match="background equations keys"):
        flrw.validate_config(config)


def test_predecessor_file_mutation_fails_closed(tmp_path: Path) -> None:
    config = _config()
    needed = [flrw.CONFIG_PATH, flrw.SOURCE_PATH, flrw.TEST_PATH]
    for binding in config["predecessor_bindings"]:
        needed.extend(
            Path(binding[key])
            for key in ("config_path", "module_path", "test_path", "receipt_path")
        )
    for relative in needed:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    changed = tmp_path / config["predecessor_bindings"][2]["module_path"]
    changed.write_bytes(changed.read_bytes() + b"\n")
    with pytest.raises(flrw.FlrwNecessaryConditionsError, match="predecessor changed"):
        flrw.build_receipt(tmp_path)


def test_receipt_claim_symbolic_and_access_mutations_fail_closed() -> None:
    config = flrw.load_config(ROOT)
    receipt = flrw.build_receipt(ROOT)
    receipt["claim_boundary"]["healthy_history_established"] = True
    _reseal(receipt)
    with pytest.raises(flrw.FlrwNecessaryConditionsError, match="claims changed"):
        flrw.validate_receipt(receipt, config)

    receipt = flrw.build_receipt(ROOT)
    receipt["symbolic_suite"]["checks"][0]["passed"] = False
    _reseal(receipt)
    with pytest.raises(flrw.FlrwNecessaryConditionsError, match="failed symbolic"):
        flrw.validate_receipt(receipt, config)

    receipt = flrw.build_receipt(ROOT)
    receipt["counts"]["observational_rows_opened"] = 1
    _reseal(receipt)
    with pytest.raises(flrw.FlrwNecessaryConditionsError, match="receipt access"):
        flrw.validate_receipt(receipt, config)

    receipt = flrw.build_receipt(ROOT)
    receipt["numeric_suite"]["gate_u_probes"][0]["post_result_choice"] = True
    _reseal(receipt)
    with pytest.raises(flrw.FlrwNecessaryConditionsError, match="gate result keys"):
        flrw.validate_receipt(receipt, config)


def test_atomic_publication_is_idempotent_and_no_clobber(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    payload = b"sealed\n"
    assert flrw._atomic_no_replace(path, payload) == "CREATED"
    assert flrw._atomic_no_replace(path, payload) == "EXISTING_IDENTICAL"
    with pytest.raises(flrw.FlrwNecessaryConditionsError, match="refusing to overwrite"):
        flrw._atomic_no_replace(path, b"different\n")
    assert path.read_bytes() == payload
