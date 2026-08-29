from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import pytest

from sigma_theory_compiler import gravity_matter_lensing_kinetic_gate_conditional_no_go as no_go

ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict[str, object]:
    return json.loads((ROOT / no_go.CONFIG_PATH).read_text(encoding="utf-8"))


def _validate_mutation(monkeypatch: pytest.MonkeyPatch, config: dict[str, object]) -> None:
    monkeypatch.setattr(no_go, "EXPECTED_CONFIG_CONTENT_SHA256", no_go._sha(config))
    no_go.validate_config(config)


def test_symbolic_and_numeric_suites_pass_exact_inventory() -> None:
    symbolic = no_go.run_symbolic_suite()
    numeric = no_go.run_numeric_suite(_config())
    assert tuple(item["check_id"] for item in symbolic["checks"]) == no_go.SYMBOLIC_CHECK_IDS
    assert symbolic["all_passed"] is True
    assert numeric["all_passed"] is True
    assert [item["sign"] for item in numeric["cases"]] == [
        "positive",
        "negative",
        "positive",
        "negative",
        "zero",
    ]


def test_receipt_is_deterministic_and_scope_restricted() -> None:
    first = no_go.build_receipt(ROOT)
    second = no_go.build_receipt(ROOT)
    assert first == second
    assert first["counts"]["symbolic_checks_passed"] == 15
    assert first["counts"]["numeric_cases_passed"] == 5
    assert first["counts"]["bounded_domain_counterexamples"] == 3
    assert first["counts"]["remedies_preregistered"] == 5
    claims = first["claim_boundary"]
    assert claims["conditional_external_metric_timelike_mixing_theorem_established"] is True
    assert claims["unconditional_action_no_go_established"] is False
    assert claims["full_determinant_instability_established"] is False
    assert claims["healthy_remedy_established"] is False
    assert all(value == 0 for value in first["zero_access_and_compute"].values())


def test_theorem_family_and_caveats_are_frozen() -> None:
    config = _config()
    analytic = config["analytic_contract"]
    assert analytic["timelike_sign_criterion"] == ("For X>0 and Z>0, M>=0 iff 4*dq/dt>=q+4*q^2.")
    assert analytic["comparison_blowup_time"] == "T-t0=4*ln(1+1/(4*q0))"
    family = config["family_and_counterexample_contract"]
    assert family["shifted_power_threshold"] == "M>=0 iff 0<u<=3/(1+4*p) on X>0"
    assert family["p_equals_2_regression"].startswith("For p=2 the threshold is u=1/3")
    caveat = config["full_determinant_caveat"]
    assert caveat["stabilization_if_M_negative"].endswith("X_chi<=Z*A_phi/(-M).")


def test_remedies_are_preregistered_but_not_claimed_healthy() -> None:
    remedies = _config()["remedy_preregistration"]
    assert tuple(item["remedy_id"] for item in remedies) == no_go.REMEDY_IDS
    assert all(item["healthy_or_working_claim"] is False for item in remedies)
    assert all(item["required_future_checks"] for item in remedies)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("scope_and_definitions", "physical_branch"), "spacelike branch"),
        (("analytic_contract", "timelike_sign_criterion"), "M>=0 always"),
        (("analytic_contract", "comparison_blowup_time"), "infinite"),
        (("family_and_counterexample_contract", "shifted_power_threshold"), "all u"),
        (("adjudication", "full_H3"), True),
        (("claim_boundary", "unconditional_action_no_go_established"), True),
        (("zero_access_and_compute", "network_calls"), 1),
    ],
)
def test_semantic_mutations_fail_closed(
    monkeypatch: pytest.MonkeyPatch, path: tuple[str, str], value: object
) -> None:
    config = copy.deepcopy(_config())
    config[path[0]][path[1]] = value
    monkeypatch.setattr(no_go, "EXPECTED_CONFIG_CONTENT_SHA256", no_go._sha(config))
    with pytest.raises(no_go.KineticGateConditionalNoGoError):
        no_go.validate_config(config)


def test_predecessor_hash_mutation_fails_closed(tmp_path: Path) -> None:
    config = _config()
    needed = [no_go.CONFIG_PATH, no_go.SOURCE_PATH, no_go.TEST_PATH]
    for binding in config["predecessor_bindings"]:
        needed.extend(
            Path(binding[key])
            for key in ("config_path", "module_path", "test_path", "receipt_path")
        )
    for relative in needed:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    changed = tmp_path / config["predecessor_bindings"][0]["module_path"]
    changed.write_bytes(changed.read_bytes() + b"\n# tampered\n")
    with pytest.raises(no_go.KineticGateConditionalNoGoError, match="predecessor changed"):
        no_go.build_receipt(tmp_path)


def test_receipt_claim_mutation_fails_closed() -> None:
    config = no_go.load_config(ROOT)
    receipt = no_go.build_receipt(ROOT)
    receipt["claim_boundary"]["healthy_action_established"] = True
    body = dict(receipt)
    body.pop("content_sha256")
    receipt["content_sha256"] = no_go._sha(body)
    with pytest.raises(no_go.KineticGateConditionalNoGoError, match="claim boundary changed"):
        no_go.validate_receipt(receipt, config)


def test_atomic_publication_is_idempotent_and_no_clobber(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    payload = b"sealed\n"
    assert no_go._atomic_no_replace(path, payload) == "CREATED"
    assert no_go._atomic_no_replace(path, payload) == "EXISTING_IDENTICAL"
    with pytest.raises(no_go.KineticGateConditionalNoGoError, match="refusing to overwrite"):
        no_go._atomic_no_replace(path, b"different\n")
    assert path.read_bytes() == payload
