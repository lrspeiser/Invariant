from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import pytest

from sigma_theory_compiler import gravity_matter_lensing_universal_conformal_source as source

ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict[str, object]:
    return json.loads((ROOT / source.CONFIG_PATH).read_text(encoding="utf-8"))


def _reseal(receipt: dict[str, object]) -> None:
    body = dict(receipt)
    body.pop("content_sha256")
    receipt["content_sha256"] = source._sha(body)


def test_symbolic_suite_passes_exact_inventory() -> None:
    symbolic = source.run_symbolic_suite()
    assert tuple(item["check_id"] for item in symbolic["checks"]) == source.SYMBOLIC_CHECK_IDS
    assert symbolic["all_passed"] is True
    assert len(symbolic["checks"]) == 18


def test_numeric_trace_sources_have_frozen_signs_and_frame_equality() -> None:
    numeric = source.run_numeric_suite(_config())
    assert numeric["all_passed"] is True
    by_id = {item["case_id"]: item for item in numeric["cases"]}
    assert by_id["DUST_POSITIVE_ALPHA"]["source_sign"] == "positive"
    assert by_id["RADIATION_TRACE_ZERO"]["source_sign"] == "zero"
    assert by_id["STIFF_EOS_NEGATIVE_SOURCE"]["source_sign"] == "negative"
    assert by_id["NEGATIVE_ALPHA_DUST"]["source_sign"] == "negative"
    assert all(float(item["frame_scaled_error"]) == 0.0 for item in numeric["cases"])


def test_receipt_is_deterministic_and_restricted() -> None:
    first = source.build_receipt(ROOT)
    second = source.build_receipt(ROOT)
    assert first == second
    assert first["counts"]["symbolic_checks_passed"] == 18
    assert first["counts"]["numeric_cases_passed"] == 4
    assert first["adjudication"]["same_action_conformal_Q_identity_derived"] is True
    assert first["adjudication"]["physical_source_profile_established"] is False
    assert first["adjudication"]["metric_backreaction"] is False
    assert first["adjudication"]["lensing_prediction"] is False
    assert all(value == 0 for value in first["zero_access_and_compute"].values())


def test_exact_source_frame_factors_and_known_form_label_are_frozen() -> None:
    config = _config()
    frame = config["frame_and_variation_conventions"]
    assert frame["exact_source_identity"] == (
        "Q_chi=-A^4*(d ln A/d chi)*Ttilde=-(d ln A/d chi)*T_E"
    )
    assert "T_E^munu=A^6*Ttilde^munu" in frame["stress_frame_factors"]
    assert "same tilde_g_munu" in frame["universal_matter_action"]
    assert "No independent photon metric" in frame["forbidden"]
    conformal = config["minimal_conformal_factor"]
    assert conformal["choice"] == "A(phi,chi)=A_phi(phi)*exp(alpha*chi/M_Pl)"
    assert conformal["derivative"] == "d ln A/d chi=alpha/M_Pl"
    assert conformal["novelty_label"] == "KNOWN_CONFORMAL_FORM_REUSE_NOT_NOVELTY"


def test_dust_radiation_eos_and_predecessor_sign_are_frozen() -> None:
    matter = _config()["matter_source_contract"]
    assert "Ttilde=-rho" in matter["dust"]
    assert "Q_chi=+(alpha/M_Pl)*A^4*rho" in matter["dust"]
    assert "Ttilde=0" in matter["classical_radiation"]
    assert "rho*(1-3*w)" in matter["general_eos"]
    assert "E_chi=Q_chi" in matter["sign_regression"]
    assert "chi_k=-Q_chi,k" in matter["sign_regression"]


def test_conservation_exchange_and_lensing_ceiling_are_restricted() -> None:
    config = _config()
    exchange = config["conservation_and_exchange"]
    assert "tilde_nabla_mu*Ttilde^mu_nu=0" in exchange["physical_frame"]
    assert "=-Q_phi*d_nu(phi)-Q_chi*d_nu(chi)" in exchange["einstein_frame"]
    assert "+Q_chi*d_nu(chi)" in exchange["total_identity"]
    ceiling = config["stability_ceiling_comparison"]
    assert "|alpha|*|T_E|/M_Pl<Q_chi,max(X)" in ceiling["general_trace_condition"]
    assert "No rho(X)" in ceiling["no_rho_X_inference"]
    lensing = config["lensing_boundary"]
    assert "A^4*A^-2*A^-2=1" in lensing["maxwell_cancellation"]
    assert "no lensing prediction" in lensing["claim_ceiling"]
    assert config["claim_boundary"]["lensing_success_established"] is False


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda c: c["frame_and_variation_conventions"].update(
                {"exact_source_identity": "Q_chi=Ttilde"}
            ),
            "source identity",
        ),
        (
            lambda c: c["frame_and_variation_conventions"].update(
                {"forbidden": "independent photon metric allowed"}
            ),
            "photon shortcut",
        ),
        (
            lambda c: c["minimal_conformal_factor"].update({"novelty_label": "NOVEL"}),
            "minimal conformal factor",
        ),
        (lambda c: c["matter_source_contract"].update({"dust": "changed"}), "matter trace"),
        (
            lambda c: c["matter_source_contract"].update({"sign_regression": "changed"}),
            "source sign regression",
        ),
        (
            lambda c: c["conservation_and_exchange"].update({"einstein_frame": "changed"}),
            "exchange identity",
        ),
        (
            lambda c: c["stability_ceiling_comparison"].update({"no_rho_X_inference": "rho=X"}),
            "stability comparison",
        ),
        (
            lambda c: c["lensing_boundary"].update({"claim_ceiling": "lensing solved"}),
            "lensing boundary",
        ),
        (
            lambda c: c["machine_check_contract"]["numeric_cases"][0].update(
                {"expected_sign": "negative"}
            ),
            "numeric",
        ),
        (
            lambda c: c["adjudication"].update({"lensing_prediction": True}),
            "partial adjudication",
        ),
        (
            lambda c: c["claim_boundary"].update({"observational_support": True}),
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
    monkeypatch.setattr(source, "EXPECTED_CONFIG_CONTENT_SHA256", source._sha(config))
    with pytest.raises(source.UniversalConformalSourceError, match=message):
        source.validate_config(config)


def test_nested_extra_key_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    config = copy.deepcopy(_config())
    config["matter_source_contract"]["post_target_choice"] = "forbidden"
    monkeypatch.setattr(source, "EXPECTED_CONFIG_CONTENT_SHA256", source._sha(config))
    with pytest.raises(source.UniversalConformalSourceError, match="matter source keys changed"):
        source.validate_config(config)


def test_predecessor_mutation_fails_closed(tmp_path: Path) -> None:
    config = _config()
    needed = [source.CONFIG_PATH, source.SOURCE_PATH, source.TEST_PATH]
    needed.extend(Path(binding["receipt_path"]) for binding in config["predecessor_bindings"])
    for relative in needed:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    changed = tmp_path / config["predecessor_bindings"][1]["receipt_path"]
    changed.write_bytes(changed.read_bytes() + b"\n")
    with pytest.raises(source.UniversalConformalSourceError, match="predecessor changed"):
        source.build_receipt(tmp_path)


def test_receipt_source_contract_and_claim_mutations_fail_closed() -> None:
    config = source.load_config(ROOT)
    receipt = source.build_receipt(ROOT)
    receipt["matter_source_contract"]["dust"] = "changed"
    _reseal(receipt)
    with pytest.raises(source.UniversalConformalSourceError, match="source contract changed"):
        source.validate_receipt(receipt, config)

    receipt = source.build_receipt(ROOT)
    receipt["claim_boundary"]["lensing_success_established"] = True
    _reseal(receipt)
    with pytest.raises(source.UniversalConformalSourceError, match="claims changed"):
        source.validate_receipt(receipt, config)


def test_receipt_derived_result_mutation_fails_closed() -> None:
    config = source.load_config(ROOT)
    receipt = source.build_receipt(ROOT)
    receipt["symbolic_suite"]["checks"][0]["passed"] = False
    _reseal(receipt)
    with pytest.raises(source.UniversalConformalSourceError, match="symbolic receipt changed"):
        source.validate_receipt(receipt, config)


def test_atomic_publication_is_idempotent_and_no_clobber(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    payload = b"sealed\n"
    assert source._atomic_no_replace(path, payload) == "CREATED"
    assert source._atomic_no_replace(path, payload) == "EXISTING_IDENTICAL"
    with pytest.raises(source.UniversalConformalSourceError, match="refusing to overwrite"):
        source._atomic_no_replace(path, b"different\n")
    assert path.read_bytes() == payload
