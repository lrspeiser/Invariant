from __future__ import annotations

import copy
import hashlib
import json
from fractions import Fraction
from pathlib import Path

import pytest

from sigma_theory_compiler.future_aether_finite_tilt_york_symbol_gate import (
    YORK_SHELL_BLOCKER,
)
from sigma_theory_compiler.future_aether_nonlinear_lift_characteristic_gate import (
    CHARACTERISTIC_BLOCKER,
)
from sigma_theory_compiler.future_aether_principal_inverse_fredholm_gate import (
    BLOCKER,
    build_future_aether_principal_inverse_fredholm_gate,
    exact_characteristic_boundary_negative_control,
    exact_principal_inverse_control,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/future_aether_principal_inverse_fredholm_gate.json"
ARTIFACT_PATH = ROOT / "runs/engine/future-aether-principal-inverse-fredholm-gate.json"
SOURCE_PATH = ROOT / "runs/engine/future-aether-finite-tilt-york-symbol-gate.json"
SEED_PATH = ROOT / "runs/engine/future-aether-finite-amplitude-negative-seed-gate.json"


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def rebuilt() -> dict:
    return build_future_aether_principal_inverse_fredholm_gate(_config(), ROOT)


def test_exact_partition_and_portable_artifact(rebuilt: dict) -> None:
    assert rebuilt == json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    body = {key: value for key, value in rebuilt.items() if key != "content_sha256"}
    assert rebuilt["content_sha256"] == _sha(body)
    assert rebuilt["candidate_count"] == 14
    assert rebuilt["decision_counts"] == {"blocked": 14}
    assert rebuilt["first_blocker_counts"] == {
        YORK_SHELL_BLOCKER: 2,
        BLOCKER: 1,
        CHARACTERISTIC_BLOCKER: 11,
    }
    assert rebuilt["formal_pass_count"] == 0
    assert rebuilt["candidate_rejection_authorized_count"] == 0


def test_candidate_action_and_both_predecessor_bindings_are_exact(rebuilt: dict) -> None:
    source = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    seed = json.loads(SEED_PATH.read_text(encoding="utf-8"))
    expected = {item["candidate_id"]: item for item in source["candidate_records"]}
    seed_expected = {item["candidate_id"]: item for item in seed["candidate_records"]}
    for item in rebuilt["candidate_records"]:
        predecessor = expected[item["candidate_id"]]
        assert item["typed_action_ir_sha256"] == predecessor["typed_action_ir_sha256"]
        assert item["source_York_symbol_record_sha256"] == predecessor["content_sha256"]
        certificate = item["principal_inverse_fredholm_certificate"]
        if certificate is not None:
            assert (
                certificate["source_compact_seed_record_sha256"]
                == seed_expected[item["candidate_id"]]["content_sha256"]
            )


def test_exact_positive_principal_inverse_control() -> None:
    amplitude = Fraction(145475033, 5963776)
    result = exact_principal_inverse_control(amplitude)
    assert result["York_determinant_absolute_lower_bound"] == (
        "2242384524249817246208/11699218066257491755654875"
    )
    assert result["York_symbol_entry_absolute_upper_bound"] == ("70027702926922225/853598980276224")
    assert result["York_symbol_inverse_2_norm_upper_bound"] == (
        "19899948617721479136777824663801308960694862672913515625/"
        "94454351350403008416711745320902257277818372096"
    )
    assert result["principal_inverse_bound_proven"] is True
    body = {key: value for key, value in result.items() if key != "content_sha256"}
    assert result["content_sha256"] == _sha(body)


def test_exact_characteristic_boundary_negative_control() -> None:
    control = exact_characteristic_boundary_negative_control()
    assert control["excluded_tilt"] == "31"
    assert control["determinant_gap_numerator"] == "0"
    assert control["determinant_gap_positive"] is False
    assert control["principal_inverse_bound_available"] is False
    with pytest.raises(ValueError, match="0<=amplitude<31"):
        exact_principal_inverse_control(Fraction(31))


def test_sole_candidate_has_principal_bound_and_homotopy_only(rebuilt: dict) -> None:
    regular = [
        item
        for item in rebuilt["candidate_records"]
        if item["principal_inverse_fredholm_certificate"] is not None
    ]
    assert len(regular) == 1
    item = regular[0]
    certificate = item["principal_inverse_fredholm_certificate"]
    assert item["candidate_id"].startswith("G3A-5e9f")
    assert item["first_blocker"] == BLOCKER
    assert certificate["uniform_principal_symbol_inverse_bound_proven"] is True
    assert certificate["principal_elliptic_homotopy_to_reference_proven"] is True
    assert (
        certificate["principal_coefficient_AE_contract"][
            "equals_Euclidean_reference_outside_unit_ball"
        ]
        is True
    )
    assert certificate["elliptic_symbol_homotopy"]["no_principal_symbol_crossing"] is True
    assert certificate["missing_distributed_lower_order_registry"] == {
        "H_core_linearization_order_0_and_1_coefficients": "not_registered",
        "momentum_constraint_order_0_and_1_coefficients": "not_registered",
        "weighted_relative_bound_against_principal_part": "not_registered",
        "weighted_kernel_or_coercivity_estimate": "not_registered",
    }


def test_prior_characteristic_and_York_shell_blockers_are_preserved(rebuilt: dict) -> None:
    assert (
        sum(
            item["first_blocker"] == CHARACTERISTIC_BLOCKER for item in rebuilt["candidate_records"]
        )
        == 11
    )
    assert (
        sum(item["first_blocker"] == YORK_SHELL_BLOCKER for item in rebuilt["candidate_records"])
        == 2
    )
    for item in rebuilt["candidate_records"]:
        if item["first_blocker"] != BLOCKER:
            assert item["principal_inverse_fredholm_certificate"] is None
            assert item["gate_ledger"]["uniform_principal_symbol_inverse_bound"]["status"] == (
                "not_reached"
            )


def test_full_weighted_nonlinear_boundary_and_rejection_claims_remain_closed(rebuilt: dict) -> None:
    assert rebuilt["uniform_principal_symbol_inverse_bound_pass_count"] == 1
    assert rebuilt["principal_elliptic_homotopy_to_reference_pass_count"] == 1
    assert rebuilt["distributed_lower_order_coefficient_registry_complete_count"] == 0
    assert rebuilt["weighted_Fredholm_isomorphism_pass_count"] == 0
    assert rebuilt["full_operator_inverse_norm_pass_count"] == 0
    assert rebuilt["nonlinear_remainder_bound_pass_count"] == 0
    assert rebuilt["completed_boundary_sign_persistence_count"] == 0
    for item in rebuilt["candidate_records"]:
        assert item["decision"] == "blocked"
        assert item["candidate_rejection_authorized"] is False
        assert item["constraint_satisfying_negative_total_energy_datum_proven"] is False


def test_hash_provenance_and_all_seals(rebuilt: dict) -> None:
    assert rebuilt["bounded_principal_inverse_fredholm_gate_completed"] is True
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
            lambda config: config["budget"].update(maximum_exact_rational_bounds=4),
            "budget is not exact",
        ),
        (
            lambda config: config["source_york_symbol_artifact"].update(content_sha256="0" * 64),
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
        build_future_aether_principal_inverse_fredholm_gate(config, ROOT)


def test_bound_paths_cannot_escape_repository() -> None:
    config = copy.deepcopy(_config())
    config["source_york_symbol_artifact"]["path"] = "../outside.json"
    with pytest.raises(ValueError, match="path escapes repository"):
        build_future_aether_principal_inverse_fredholm_gate(config, ROOT)
