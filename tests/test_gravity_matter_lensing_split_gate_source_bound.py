from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import pytest

from sigma_theory_compiler import gravity_matter_lensing_split_gate_source_bound as source_bound

ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict[str, object]:
    return json.loads((ROOT / source_bound.CONFIG_PATH).read_text(encoding="utf-8"))


def test_symbolic_suite_passes_exact_inventory() -> None:
    symbolic = source_bound.run_symbolic_suite()
    assert tuple(item["check_id"] for item in symbolic["checks"]) == source_bound.SYMBOLIC_CHECK_IDS
    assert symbolic["all_passed"] is True
    assert len(symbolic["checks"]) == 17


def test_numeric_source_scalings_and_failures_are_preserved() -> None:
    numeric = source_bound.run_numeric_suite(_config())
    assert numeric["all_passed"] is True
    by_id = {item["case_id"]: item for item in numeric["source_scaling_cases"]}
    assert by_id["R2_SUBCRITICAL"]["statuses"] == ["pass", "pass", "pass"]
    assert by_id["R2P5_LOW_COEFFICIENT"]["statuses"] == ["pass", "pass", "pass"]
    assert by_id["R2P5_HIGH_COEFFICIENT"]["statuses"] == ["fail", "fail", "fail"]
    assert by_id["R3_THRESHOLD_CROSSING"]["statuses"] == ["pass", "pass", "fail"]
    assert numeric["designed_failure_cases_preserved"] == 2
    assert numeric["failed_source_points_preserved"] == 4


def test_finite_k_response_is_suppressed_and_ceiling_relaxes() -> None:
    records = source_bound.run_numeric_suite(_config())["finite_k_regression"]
    suppressions = [float(item["suppression"]) for item in records]
    relaxations = [float(item["source_ceiling_relaxation"]) for item in records]
    assert suppressions == sorted(suppressions, reverse=True)
    assert relaxations == sorted(relaxations)
    assert all(float(item["chi_response_for_positive_unit_Q"]) < 0.0 for item in records)
    assert all(
        abs(left * right - 1.0) <= 1e-12
        for left, right in zip(suppressions, relaxations, strict=True)
    )


def test_receipt_is_deterministic_and_partial_only() -> None:
    first = source_bound.build_receipt(ROOT)
    second = source_bound.build_receipt(ROOT)
    assert first == second
    assert first["counts"]["symbolic_checks_passed"] == 17
    assert first["counts"]["source_scaling_cases_passed"] == 4
    assert first["counts"]["finite_k_probes_passed"] == 4
    assert first["adjudication"]["sufficient_source_ceiling_derived"] is True
    assert first["adjudication"]["physical_Q_chi_derived"] is False
    assert first["adjudication"]["physical_on_shell_background"] is False
    assert all(value == 0 for value in first["zero_access_and_compute"].values())


def test_exact_response_green_and_conservatism_are_frozen() -> None:
    config = _config()
    problem = config["frozen_source_problem"]
    assert "total variation gives E_chi=Q_chi" in problem["sign_convention"]
    assert "Y0*(nabla^2-m_eff^2)*chi=Q_chi" in problem["sign_convention"]
    assert problem["fourier_response"].startswith("chi_k=-Q_chi,k/")
    assert "chi(x)=-integral" in problem["green_response"]
    assert "approximately -Q_chi,k" in problem["long_wavelength_response"]
    finite = config["finite_wavelength_contract"]
    assert "conservative mode by mode" in finite["conservative_statement"]
    assert "exact sufficient L-infinity" in finite["sup_norm_statement"]
    assert "different boundaries" in finite["not_claimed"]


def test_source_ceiling_and_high_u_scaling_are_exactly_limited() -> None:
    ceiling = _config()["amplitude_and_source_ceiling"]
    assert "Q_chi,max(X)=m_chi^2*Z(X)*chi_max(X)" in ceiling["exact_sufficient_source_ceiling"]
    assert "m_chi*beta*sqrt(A0/2)" in ceiling["high_u_branch_limits"]
    assert "m_chi*beta*sqrt(B0/14)" in ceiling["high_u_branch_limits"]
    assert "X^(5/2)" in ceiling["high_u_overall_scaling"]
    assert "X^-3/2" in ceiling["high_u_overall_scaling"]


def test_no_baryonic_relation_or_lensing_shortcut_is_claimed() -> None:
    config = _config()
    assert (
        "No relation Q_chi(rho_baryon,X)"
        in config["source_scaling_contract"]["forbidden_inference"]
    )
    obligations = config["source_and_lensing_obligations"]
    assert "same universal matter action" in obligations["source_definition"]
    assert "no independent photon multiplier" in obligations["lensing_boundary"]
    claims = config["claim_boundary"]
    assert claims["baryonic_relation_established"] is False
    assert claims["lensing_success_established"] is False


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda c: c["frozen_source_problem"].update({"sign_convention": "changed"}),
            "source sign",
        ),
        (
            lambda c: c["frozen_source_problem"].update({"bare_response_scope": "decoupled"}),
            "coupled-response",
        ),
        (
            lambda c: c["finite_wavelength_contract"].update(
                {"conservative_statement": "not conservative"}
            ),
            "finite-k",
        ),
        (
            lambda c: c["amplitude_and_source_ceiling"].update(
                {"exact_sufficient_source_ceiling": "none"}
            ),
            "source ceiling",
        ),
        (
            lambda c: c["source_scaling_contract"].update({"forbidden_inference": "rho_baryon=X"}),
            "source-scaling",
        ),
        (
            lambda c: c["source_and_lensing_obligations"].update(
                {"lensing_boundary": "photon multiplier"}
            ),
            "source/lensing",
        ),
        (
            lambda c: c["adjudication"].update({"physical_Q_chi_derived": True}),
            "partial adjudication",
        ),
        (
            lambda c: c["claim_boundary"].update({"physical_on_shell_solution_established": True}),
            "claim boundary",
        ),
        (lambda c: c["zero_access_and_compute"].update({"network_calls": 1}), "access state"),
    ],
)
def test_semantic_mutations_fail_closed(
    monkeypatch: pytest.MonkeyPatch, mutation: object, message: str
) -> None:
    config = copy.deepcopy(_config())
    mutation(config)
    monkeypatch.setattr(source_bound, "EXPECTED_CONFIG_CONTENT_SHA256", source_bound._sha(config))
    with pytest.raises(source_bound.SplitGateSourceBoundError, match=message):
        source_bound.validate_config(config)


def test_predecessor_mutation_fails_closed(tmp_path: Path) -> None:
    config = _config()
    binding = config["predecessor_binding"]
    needed = [source_bound.CONFIG_PATH, source_bound.SOURCE_PATH, source_bound.TEST_PATH]
    needed.extend(
        Path(binding[key]) for key in ("config_path", "module_path", "test_path", "receipt_path")
    )
    for relative in needed:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    changed = tmp_path / binding["module_path"]
    changed.write_bytes(changed.read_bytes() + b"\n")
    with pytest.raises(source_bound.SplitGateSourceBoundError, match="predecessor changed"):
        source_bound.build_receipt(tmp_path)


def test_receipt_claim_mutation_fails_closed() -> None:
    config = source_bound.load_config(ROOT)
    receipt = source_bound.build_receipt(ROOT)
    receipt["claim_boundary"]["physical_on_shell_solution_established"] = True
    body = dict(receipt)
    body.pop("content_sha256")
    receipt["content_sha256"] = source_bound._sha(body)
    with pytest.raises(source_bound.SplitGateSourceBoundError, match="claims changed"):
        source_bound.validate_receipt(receipt, config)


def test_atomic_publication_is_idempotent_and_no_clobber(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    payload = b"sealed\n"
    assert source_bound._atomic_no_replace(path, payload) == "CREATED"
    assert source_bound._atomic_no_replace(path, payload) == "EXISTING_IDENTICAL"
    with pytest.raises(source_bound.SplitGateSourceBoundError, match="refusing to overwrite"):
        source_bound._atomic_no_replace(path, b"different\n")
    assert path.read_bytes() == payload
