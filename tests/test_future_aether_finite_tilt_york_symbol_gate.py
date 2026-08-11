from __future__ import annotations

import copy
import hashlib
import json
from fractions import Fraction
from pathlib import Path

import pytest
import sympy as sp

from sigma_theory_compiler.future_aether_finite_tilt_york_symbol_gate import (
    FREDHOLM_BLOCKER,
    YORK_SHELL_BLOCKER,
    _derive_york_symbol,
    build_future_aether_finite_tilt_york_symbol_gate,
)
from sigma_theory_compiler.future_aether_nonlinear_lift_characteristic_gate import (
    CHARACTERISTIC_BLOCKER,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/future_aether_finite_tilt_york_symbol_gate.json"
ARTIFACT_PATH = ROOT / "runs/engine/future-aether-finite-tilt-york-symbol-gate.json"
SOURCE_PATH = ROOT / "runs/engine/future-aether-fixed-free-data-principal-gate.json"
INVERSE_PATH = ROOT / "runs/engine/future-aether-regular-adm-inverse-margin-gate.json"


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def rebuilt() -> dict:
    return build_future_aether_finite_tilt_york_symbol_gate(_config(), ROOT)


def test_exact_partition_and_portable_artifact(rebuilt: dict) -> None:
    assert rebuilt == json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    body = {key: value for key, value in rebuilt.items() if key != "content_sha256"}
    assert rebuilt["content_sha256"] == _sha(body)
    assert rebuilt["candidate_count"] == 14
    assert rebuilt["decision_counts"] == {"blocked": 14}
    assert rebuilt["first_blocker_counts"] == {
        YORK_SHELL_BLOCKER: 2,
        FREDHOLM_BLOCKER: 1,
        CHARACTERISTIC_BLOCKER: 11,
    }
    assert rebuilt["formal_pass_count"] == 0
    assert rebuilt["candidate_rejection_authorized_count"] == 0


def test_candidate_action_and_both_predecessor_bindings_are_exact(rebuilt: dict) -> None:
    source = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    inverse = json.loads(INVERSE_PATH.read_text(encoding="utf-8"))
    expected = {item["candidate_id"]: item for item in source["candidate_records"]}
    inverse_expected = {item["candidate_id"]: item for item in inverse["candidate_records"]}
    for item in rebuilt["candidate_records"]:
        predecessor = expected[item["candidate_id"]]
        assert item["typed_action_ir_sha256"] == predecessor["typed_action_ir_sha256"]
        assert item["source_fixed_free_data_record_sha256"] == predecessor["content_sha256"]
        certificate = item["finite_tilt_York_symbol_certificate"]
        if certificate is not None:
            inverse_record = inverse_expected[item["candidate_id"]]
            assert (
                certificate["source_inverse_margin_record_sha256"]
                == inverse_record["content_sha256"]
            )


def test_all_three_regular_candidate_symbols_are_exactly_derived(rebuilt: dict) -> None:
    regular = [
        item
        for item in rebuilt["candidate_records"]
        if item["finite_tilt_York_symbol_certificate"] is not None
    ]
    assert len(regular) == 3
    assert rebuilt["finite_tilt_metric_York_symbol_derived_count"] == 3
    for item in regular:
        certificate = item["finite_tilt_York_symbol_certificate"]
        symbol = certificate["finite_tilt_York_symbol"]
        assert symbol["symbol_shape"] == [3, 3]
        assert symbol["candidate_bound_distributed_Legendre_principal_symbol_derived"] is True
        assert "H_KK-H_KV" in symbol["fixed_vector_momentum_Schur_complement"]
        body = {key: value for key, value in symbol.items() if key != "content_sha256"}
        assert symbol["content_sha256"] == _sha(body)


def test_uniform_positive_candidate_is_elliptic_on_the_full_registered_seed(rebuilt: dict) -> None:
    item = next(
        item for item in rebuilt["candidate_records"] if item["candidate_id"].startswith("G3A-5e9f")
    )
    certificate = item["finite_tilt_York_symbol_certificate"]
    proof = certificate["uniform_principal_ellipticity_certificate"]
    assert item["first_blocker"] == FREDHOLM_BLOCKER
    assert certificate["registered_characteristic_free_tilt_upper"] == "145475033/5963776"
    assert Fraction(certificate["registered_characteristic_free_tilt_upper"]) < 31
    assert certificate["uniform_fixed_free_data_principal_ellipticity_proven"] is True
    assert certificate["exact_nonelliptic_York_shell"] is None
    assert proof["all_y_and_z_principal_determinants_nonzero"] is True
    assert certificate["finite_tilt_York_symbol"]["perpendicular_covector_determinant"] == (
        "-(y - 31)**2*(61*y + 124)/(6144*(y + 2))"
    )


@pytest.mark.parametrize(
    ("candidate_prefix", "root", "polynomial", "amplitude"),
    [
        ("G3A-213faf", "(-5+sqrt(521))/2", "y^2+5*y-124", "40788467/1490944"),
        ("G3A-e6453", "8+sqrt(97)", "y^2-16*y-33", "100"),
    ],
)
def test_two_exact_shells_reject_only_the_global_K_equals_LX_ansatz(
    rebuilt: dict, candidate_prefix: str, root: str, polynomial: str, amplitude: str
) -> None:
    item = next(
        item
        for item in rebuilt["candidate_records"]
        if item["candidate_id"].startswith(candidate_prefix)
    )
    certificate = item["finite_tilt_York_symbol_certificate"]
    shell = certificate["exact_nonelliptic_York_shell"]
    assert item["first_blocker"] == YORK_SHELL_BLOCKER
    assert certificate["registered_characteristic_free_tilt_upper"] == amplitude
    assert certificate["uniform_fixed_free_data_principal_ellipticity_proven"] is False
    assert shell["direction"] == "z=0_perpendicular_to_A"
    assert shell["tilt_squared"] == root
    assert shell["root_polynomial"] == polynomial
    assert shell["strictly_inside_registered_tilt_range"] is True
    assert shell["ansatz_status"] == "nonelliptic_reject_for_K_equals_LX_completion_only"
    assert shell["candidate_rejection_authorized"] is False


def test_symbol_derivation_replays_expected_perpendicular_negative_control() -> None:
    result = _derive_york_symbol({"c1": "1/32", "c2": "1/32", "c3": "0", "c4": "1/32"})
    y = sp.symbols("y")
    determinant = sp.sympify(result["perpendicular_covector_determinant"], locals={"y": y})
    alpha = (-5 + sp.sqrt(521)) / 2
    assert sp.simplify(determinant.subs(y, alpha)) == 0


def test_eleven_characteristic_candidates_are_preserved_and_not_reached(rebuilt: dict) -> None:
    forced = [
        item
        for item in rebuilt["candidate_records"]
        if item["first_blocker"] == CHARACTERISTIC_BLOCKER
    ]
    assert len(forced) == 11
    for item in forced:
        assert item["finite_tilt_York_symbol_certificate"] is None
        assert item["gate_ledger"]["finite_tilt_metric_momentum_to_York_symbol"]["status"] == (
            "not_reached"
        )


def test_no_Fredholm_nonlinear_boundary_or_candidate_rejection_overclaim(rebuilt: dict) -> None:
    assert rebuilt["uniform_fixed_free_data_principal_ellipticity_pass_count"] == 1
    assert rebuilt["exact_nonelliptic_York_shell_count"] == 2
    assert rebuilt["York_ansatz_reject_count"] == 2
    assert rebuilt["weighted_Fredholm_isomorphism_pass_count"] == 0
    assert rebuilt["lower_order_coefficient_bound_pass_count"] == 0
    assert rebuilt["computable_full_inverse_norm_count"] == 0
    assert rebuilt["nonlinear_remainder_bound_pass_count"] == 0
    assert rebuilt["completed_boundary_sign_persistence_count"] == 0
    for item in rebuilt["candidate_records"]:
        assert item["decision"] == "blocked"
        assert item["candidate_rejection_authorized"] is False
        assert item["constraint_satisfying_negative_total_energy_datum_proven"] is False


def test_hash_provenance_and_all_seals(rebuilt: dict) -> None:
    assert rebuilt["bounded_finite_tilt_York_symbol_gate_completed"] is True
    assert rebuilt["automatic_downstream_enqueue_performed"] is False
    assert rebuilt["observational_data_opened"] is False
    assert rebuilt["dark_matter_or_halo_inputs"] is False
    assert rebuilt["redshift_distance_inputs"] is False
    assert rebuilt["paid_llm_spend_usd"] == 0.0
    provenance = rebuilt["provenance"]
    provenance_body = {key: value for key, value in provenance.items() if key != "binding_sha256"}
    assert provenance["binding_sha256"] == _sha(provenance_body)
    for item in rebuilt["candidate_records"]:
        body = {key: value for key, value in item.items() if key != "content_sha256"}
        assert item["content_sha256"] == _sha(body)
        assert item["data_eligibility"] == rebuilt["data_eligibility"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda config: config.update(
                data_eligibility={
                    **config["data_eligibility"],
                    "observational_data_opened": True,
                }
            ),
            "eligibility is open",
        ),
        (lambda config: config.update(observational_authorization=True), "opened observations"),
        (lambda config: config.update(external_paid_llm_calls=True), "enabled paid LLM calls"),
        (
            lambda config: config["budget"].update(maximum_symbol_columns=4),
            "budget is not exact",
        ),
        (
            lambda config: config["source_fixed_free_data_artifact"].update(
                content_sha256="0" * 64
            ),
            "content hash mismatch",
        ),
        (
            lambda config: config["campaign_implementation"].update(file_sha256="0" * 64),
            "file hash mismatch",
        ),
    ],
)
def test_open_seals_budget_and_hash_tampering_fail_closed(mutation, message: str) -> None:
    config = copy.deepcopy(_config())
    mutation(config)
    with pytest.raises(ValueError, match=message):
        build_future_aether_finite_tilt_york_symbol_gate(config, ROOT)


def test_bound_paths_cannot_escape_repository() -> None:
    config = copy.deepcopy(_config())
    config["source_fixed_free_data_artifact"]["path"] = "../outside.json"
    with pytest.raises(ValueError, match="path escapes repository"):
        build_future_aether_finite_tilt_york_symbol_gate(config, ROOT)
