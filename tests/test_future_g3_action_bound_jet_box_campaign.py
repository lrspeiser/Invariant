from __future__ import annotations

import copy
import hashlib
import json
from fractions import Fraction
from pathlib import Path

import pytest

from sigma_theory_compiler.future_g3_action_bound_jet_box_campaign import (
    DOMAIN_FIELDS,
    FIRST_BLOCKER,
    _sha,
    build_future_g3_action_bound_jet_box_campaign,
)
from sigma_theory_compiler.promotion_orchestrator import ELIGIBILITY

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "future_g3_action_bound_jet_box_campaign.json"
ARTIFACT = ROOT / "runs" / "engine" / "future-g3-action-bound-jet-box-campaign.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def rebuilt() -> dict:
    return build_future_g3_action_bound_jet_box_campaign(_load(CONFIG), ROOT)


def test_committed_artifact_is_exact_rebuild(rebuilt: dict) -> None:
    committed = _load(ARTIFACT)
    assert committed == rebuilt
    body = {key: item for key, item in committed.items() if key != "content_sha256"}
    assert committed["content_sha256"] == _sha(body)
    assert committed["content_sha256"] == (
        "af0617991c4ea0f56dc6866a682350fe94e51ff35ffcf30d293bdea461669e9a"
    )
    assert hashlib.sha256(ARTIFACT.read_bytes()).hexdigest() == (
        "5140970b0ed0c5c88f771e48a37ea8d8e1e4dd59d6f9c8d7b9ff95d156bd9fa7"
    )


def test_all_twelve_action_bound_domain_fields_are_filled(rebuilt: dict) -> None:
    assert rebuilt["domain_registration_filled_field_count"] == 36
    assert rebuilt["domain_registration_missing_field_count"] == 0
    seen_domains = set()
    for record in rebuilt["candidate_records"]:
        domain = record["domain_registration"]
        assert set(domain["registered_values"]) == set(DOMAIN_FIELDS)
        assert all(value is not None for value in domain["registered_values"].values())
        assert domain["filled_field_count"] == 12
        assert domain["unfilled_field_count"] == 0
        assert domain["nonempty_rational_box_registered"] is True
        assert domain["candidate_id"] == record["candidate_id"]
        assert domain["action_sha256"] == record["action_sha256"]
        assert domain["contract_review"].endswith("not_family_transfer")
        seen_domains.add(domain["content_sha256"])
    assert len(seen_domains) == 3


def test_direct_interval_runs_close_every_direction_with_recorded_margins(
    rebuilt: dict,
) -> None:
    expected = {
        "33/4000": (
            "0x1.ff9a1190a03c6p-1",
            "0x1.ff66481576417p-1",
            "0x1.ff00784074302p-1",
            "0x1.fe34b4b8e7fdcp-1",
        ),
        "17/2000": (
            "0x1.ff972c672ca27p-1",
            "0x1.ff618cf9feecdp-1",
            "0x1.fef8d9d2047a0p-1",
            "0x1.fe273c1beef5bp-1",
        ),
        "9/1000": (
            "0x1.ff916ad40df33p-1",
            "0x1.ff581379a7107p-1",
            "0x1.fee9a292729f9p-1",
            "0x1.fe0c5f11f60a2p-1",
        ),
    }
    keys = (
        "common_time_covector_margin",
        "spatial_block_eigenvalue_lower",
        "characteristic_discriminant_lower",
        "slicing_cone_separation",
    )
    for record in rebuilt["candidate_records"]:
        interval = record["uniform_interval_certificate"]
        assert interval["certify_cubic_bssn_domain_invoked"] is True
        assert interval["direct_candidate_interval_run"] is True
        assert interval["prior_interval_certificate_reused"] is False
        assert interval["adapter_certificate"]["status"] == "pass_uniform_local_jet_box"
        assert "no direction sampling" in interval["all_direction_method"]
        margins = interval["certified_margins"]
        assert tuple(margins[key]["exact_binary64_hex"] for key in keys) == expected[record["beta"]]
        assert all(margins[key]["decimal"] > 0 for key in keys)


def test_exact_candidate_lapse_coercivity_and_periodic_dirac_pass(rebuilt: dict) -> None:
    expected = {
        "33/4000": (
            "23509187142818997683/25000000000000000000",
            "26559922036644539217/25000000000000000000",
            "25000000000000000000/23509187142818997683",
        ),
        "17/2000": (
            "5877157922384160083/6250000000000000000",
            "6640228560077696817/6250000000000000000",
            "6250000000000000000/5877157922384160083",
        ),
        "9/1000": (
            "1469220811936702307/1562500000000000000",
            "1660182175064717793/1562500000000000000",
            "1562500000000000000/1469220811936702307",
        ),
    }
    for record in rebuilt["candidate_records"]:
        lapse = record["lapse_and_periodic_dirac"]
        lower, upper, inverse = expected[record["beta"]]
        assert lapse["exact_Delta_N_lower_bound"] == lower
        assert lapse["exact_Delta_N_upper_bound"] == upper
        assert lapse["exact_inverse_norm_upper"] == inverse
        assert Fraction(lower) > 0
        assert Fraction(upper) >= Fraction(lower)
        assert lapse["direct_candidate_coercivity_recompute"] is True
        assert lapse["prior_coercivity_certificate_reused"] is False
        assert lapse["coercivity_certificate"]["function_space_result"]["kernel"] == "{0}"
        assert lapse["periodic_distributed_Dirac"] == "pass"


def test_af_global_energy_and_full_formal_stay_blocked(rebuilt: dict) -> None:
    assert rebuilt["decision_counts"] == {"blocked": 3}
    assert rebuilt["first_blocker_counts"] == {FIRST_BLOCKER: 3}
    assert rebuilt["nonzero_componentwise_box_pass_count"] == 3
    assert rebuilt["uniform_principal_common_cone_pass_count"] == 3
    assert rebuilt["full_Delta_N_derivation_pass_count"] == 3
    assert rebuilt["uniform_Delta_N_coercivity_pass_count"] == 3
    assert rebuilt["periodic_distributed_Dirac_pass_count"] == 3
    assert rebuilt["asymptotically_flat_Dirac_pass_count"] == 0
    assert rebuilt["global_energy_pass_count"] == 0
    assert rebuilt["full_formal_pass_count"] == 0
    for record in rebuilt["candidate_records"]:
        assert record["decision"] == "blocked"
        assert record["first_blocker"] == FIRST_BLOCKER
        assert record["full_formal_pass"] is False
        assert record["lapse_and_periodic_dirac"]["asymptotically_flat_Dirac"] == "blocked"
        assert record["lapse_and_periodic_dirac"]["global_energy"] == "blocked"
        assert record["gate_ledger"]["smaller_beta_or_family_label_transfer"] == (
            "rejected_as_inference"
        )


def test_data_seals_are_closed_and_no_real_data_pass_is_claimed(rebuilt: dict) -> None:
    assert rebuilt["observational_data_opened"] is False
    assert rebuilt["dark_matter_or_halo_inputs"] is False
    assert rebuilt["redshift_distance_inputs"] is False
    assert rebuilt["paid_llm_spend_usd"] == 0.0
    assert rebuilt["synthetic_fixture_role"] == "none_used"
    assert rebuilt["data_eligibility"] == ELIGIBILITY
    for record in rebuilt["candidate_records"]:
        assert record["observational_data_opened"] is False
        assert record["data_eligibility"] == ELIGIBILITY
        assert record["gate_ledger"]["observational_data_seal"] == "pass"


def test_action_domain_operator_and_source_tampering_fail_closed() -> None:
    config = _load(CONFIG)

    action = copy.deepcopy(config)
    action["targets"][0]["action_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="target binding changed"):
        build_future_g3_action_bound_jet_box_campaign(action, ROOT)

    empty = copy.deepcopy(config)
    empty["targets"][0]["componentwise_domain"]["spatial_gradient_component_abs"] = "0"
    with pytest.raises(ValueError, match="empty or malformed"):
        build_future_g3_action_bound_jet_box_campaign(empty, ROOT)

    operator = copy.deepcopy(config)
    operator["operator_domain"]["spatial_boundary_terms"] = "unspecified"
    with pytest.raises(ValueError, match="periodic operator domain changed"):
        build_future_g3_action_bound_jet_box_campaign(operator, ROOT)

    source = copy.deepcopy(config)
    source["adapter_source"]["file_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="campaign source hash mismatch"):
        build_future_g3_action_bound_jet_box_campaign(source, ROOT)

    predecessor = copy.deepcopy(config)
    predecessor["bindings"]["predecessor"]["content_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="predecessor content hash mismatch"):
        build_future_g3_action_bound_jet_box_campaign(predecessor, ROOT)
