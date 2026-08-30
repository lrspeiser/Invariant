from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import pytest

from sigma_theory_compiler import gravity_matter_lensing_split_gate_action as split_gate

ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict[str, object]:
    return json.loads((ROOT / split_gate.CONFIG_PATH).read_text(encoding="utf-8"))


def test_symbolic_suite_passes_both_derivation_routes() -> None:
    symbolic = split_gate.run_symbolic_suite()
    assert tuple(item["check_id"] for item in symbolic["checks"]) == split_gate.SYMBOLIC_CHECK_IDS
    assert symbolic["all_passed"] is True
    assert len(symbolic["checks"]) == 26
    assert symbolic["derivation_routes"] == [
        "negative exact quadratic gradient Hessian",
        "independent Euler-Lagrange flux linearization",
    ]


def test_numeric_suite_preserves_passes_and_designed_failures() -> None:
    numeric = split_gate.run_numeric_suite(_config())
    assert numeric["all_passed"] is True
    assert numeric["designed_failures_preserved"] == 3
    by_id = {item["case_id"]: item for item in numeric["cases"]}
    assert by_id["TIMELIKE_SMALL_CHI_PASS"]["C_sign"] == "positive"
    assert by_id["TIMELIKE_SMALL_CHI_PASS"]["K_sign"] == "positive"
    assert by_id["TIMELIKE_K_ONLY_FAILURE"]["C_sign"] == "positive"
    assert by_id["TIMELIKE_K_ONLY_FAILURE"]["K_sign"] == "negative"
    assert by_id["TIMELIKE_BOTH_FAILURE"]["C_sign"] == "negative"
    assert by_id["SPACELIKE_LARGE_CHI_LOCAL_PASS"]["K_sign"] == "positive"
    assert by_id["HIGH_U_FIXED_CHI_K_FAILURE"]["K_sign"] == "negative"
    assert by_id["X_ZERO_ANY_FINITE_CHI_GATE_CORRECTION_ZERO"]["C"] == "1"
    assert by_id["TIMELIKE_SMALL_CHI_PASS"]["first_derivative_mixing_class"] == "nonzero"
    assert by_id["TIMELIKE_BOTH_FAILURE"]["coupled_determinant_sign"] == "negative"
    assert (
        by_id["X_ZERO_ANY_FINITE_CHI_GATE_CORRECTION_ZERO"]["first_derivative_mixing_class"]
        == "zero"
    )
    assert len(numeric["high_u_scaling_regression"]) == 3
    assert all(item["approaches_frozen_limits"] for item in numeric["high_u_scaling_regression"])


def test_receipt_is_deterministic_and_claim_limited() -> None:
    first = split_gate.build_receipt(ROOT)
    second = split_gate.build_receipt(ROOT)
    assert first == second
    assert first["counts"]["symbolic_checks_passed"] == 26
    assert first["counts"]["numeric_cases_passed"] == 6
    assert first["counts"]["designed_failures_preserved"] == 3
    assert first["adjudication"]["specific_kinetic_gate_no_go_avoided"] is True
    assert first["adjudication"]["timelike_chi_amplitude_risk_found"] is True
    assert first["adjudication"]["full_health_established"] is False
    assert first["claim_boundary"]["GR_limit_established"] is False
    assert first["claim_boundary"]["lensing_success_established"] is False
    assert all(value == 0 for value in first["zero_access_and_compute"].values())


def test_action_eom_principal_and_range_are_exactly_frozen() -> None:
    config = _config()
    action = config["action_contract"]
    assert action["illustrative_gate"] == "Z(u)=(1+u)^2"
    assert action["novelty_label"] == (
        "KNOWN_FORM_REUSE_AND_STRUCTURAL_RECOMBINATION_NOT_A_NOVELTY_CLAIM"
    )
    eom = config["eom_and_principal_contract"]
    assert eom["C"] == "P_X-Q*Z_X"
    assert eom["C_X"] == "P_XX-Q*Z_XX"
    assert eom["cross_principal"].startswith("The second-order blocks P_phichi=P_chiphi=0 exactly")
    assert "d=m_chi^2*chi*Z_X" in eom["first_derivative_mixing"]
    assert "D_phi*D_chi-d^2*(v.k)^2" in eom["coupled_local_dispersion"]
    assert "only for frozen delta_phi" in config["range_and_limits"]["bare_dispersion"]
    assert "only for frozen delta_phi" in config["range_and_limits"]["bare_range"]


def test_health_bounds_and_replacement_risk_are_explicit() -> None:
    config = _config()
    health = config["local_health_contract"]
    assert health["necessary_external_scalar_conditions"] == [
        "Y0>0",
        "Z>0 and finite bare/frozen-delta-phi mass parameter m_chi^2*Z/Y0",
        "C>0",
        "K>0",
    ]
    assert "chi^2<min" in health["timelike_amplitude_bounds"]
    assert "chi_max^2 proportional to X^-3" in health["high_u_amplitude_scaling"]
    assert "|chi|_max proportional to X^-3/2" in health["high_u_amplitude_scaling"]
    assert "imposes no upper chi-amplitude bound" in health["quasistatic_spacelike_bounds"]
    no_go = config["no_go_adjudication"]
    assert "M_Y=0 identically" in no_go["constant_Y_result"]
    assert "does not establish a healthy action" in no_go["replacement_risk"]


def test_same_action_stress_and_lensing_boundary_are_preserved() -> None:
    config = _config()
    stress = config["stress_conservation_and_lensing"]
    assert stress["scalar_stress_tensor"].startswith("T_munu=C*d_mu(phi)*d_nu(phi)")
    assert "only the total same-action identity is conserved" in stress["with_matter"]
    assert (
        "Massive matter and photons see the same tilde_g" in stress["same_action_lensing_boundary"]
    )
    assert "no independent lensing enhancement" in stress["same_action_lensing_boundary"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda c: c["action_contract"].update({"scalar_action": "changed"}), "action changed"),
        (lambda c: c["action_contract"].update({"illustrative_gate": "Z=exp(u)"}), "gate changed"),
        (lambda c: c["action_contract"].update({"novelty_label": "NOVEL"}), "novelty label"),
        (
            lambda c: c["eom_and_principal_contract"].update({"cross_principal": "nonzero"}),
            "cross-principal",
        ),
        (
            lambda c: c["local_health_contract"].update({"timelike_amplitude_bounds": "none"}),
            "timelike bound",
        ),
        (
            lambda c: c["no_go_adjudication"].update({"constant_Y_result": "healthy"}),
            "no-go result",
        ),
        (lambda c: c["adjudication"].update({"full_health_established": True}), "full gate"),
        (
            lambda c: c["claim_boundary"].update({"Solar_viability_established": True}),
            "claim boundary",
        ),
        (lambda c: c["zero_access_and_compute"].update({"GPU_calls": 1}), "access state"),
    ],
)
def test_semantic_mutations_fail_closed(
    monkeypatch: pytest.MonkeyPatch, mutation: object, message: str
) -> None:
    config = copy.deepcopy(_config())
    mutation(config)
    monkeypatch.setattr(split_gate, "EXPECTED_CONFIG_CONTENT_SHA256", split_gate._sha(config))
    with pytest.raises(split_gate.SplitGateActionError, match=message):
        split_gate.validate_config(config)


def test_predecessor_receipt_mutation_fails_closed(tmp_path: Path) -> None:
    config = _config()
    needed = [split_gate.CONFIG_PATH, split_gate.SOURCE_PATH, split_gate.TEST_PATH]
    needed.extend(Path(item["receipt_path"]) for item in config["predecessor_receipt_bindings"])
    for relative in needed:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    changed = tmp_path / config["predecessor_receipt_bindings"][1]["receipt_path"]
    changed.write_bytes(changed.read_bytes() + b"\n")
    with pytest.raises(split_gate.SplitGateActionError, match="predecessor changed"):
        split_gate.build_receipt(tmp_path)


def test_receipt_claim_mutation_fails_closed() -> None:
    config = split_gate.load_config(ROOT)
    receipt = split_gate.build_receipt(ROOT)
    receipt["claim_boundary"]["lensing_success_established"] = True
    body = dict(receipt)
    body.pop("content_sha256")
    receipt["content_sha256"] = split_gate._sha(body)
    with pytest.raises(split_gate.SplitGateActionError, match="claims changed"):
        split_gate.validate_receipt(receipt, config)


def test_atomic_publication_is_idempotent_and_no_clobber(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    payload = b"sealed\n"
    assert split_gate._atomic_no_replace(path, payload) == "CREATED"
    assert split_gate._atomic_no_replace(path, payload) == "EXISTING_IDENTICAL"
    with pytest.raises(split_gate.SplitGateActionError, match="refusing to overwrite"):
        split_gate._atomic_no_replace(path, b"different\n")
    assert path.read_bytes() == payload
