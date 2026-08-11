from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.future_g3_componentwise_domain_contract_campaign import (
    FIRST_BLOCKER,
    REQUIRED_DOMAIN_FIELDS,
    _sha,
    build_future_g3_componentwise_domain_contract_campaign,
)
from sigma_theory_compiler.promotion_orchestrator import ELIGIBILITY

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "future_g3_componentwise_domain_contract_campaign.json"
ARTIFACT = ROOT / "runs" / "engine" / "future-g3-componentwise-domain-contract-campaign.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def rebuilt() -> dict:
    return build_future_g3_componentwise_domain_contract_campaign(_load(CONFIG), ROOT)


def test_committed_artifact_is_exact_rebuild(rebuilt: dict) -> None:
    committed = _load(ARTIFACT)
    assert committed == rebuilt
    body = {key: item for key, item in committed.items() if key != "content_sha256"}
    assert committed["content_sha256"] == _sha(body)
    assert hashlib.sha256(ARTIFACT.read_bytes()).hexdigest() == (
        "e453fae2edac28dc9c23020670540326e4acc3380d9d573bd0348f5f58d025b0"
    )
    assert committed["content_sha256"] == (
        "f3bd43bbc9bd26a28f99404930d6e68f4c3d52daf9ec5e0d2500adbeeb162db7"
    )


def test_exact_future_candidate_action_and_beta_bindings(rebuilt: dict) -> None:
    expected = {
        "G3A-8555e529226d13e2e9dacad5": (
            "4f31eb8efc25f3b28fd56d7d6dc6518461b1624f69f3b51ad8f05e6e7374e8eb",
            "33/4000",
        ),
        "G3A-8ec243e6dd285fd92e7b8e0c": (
            "08f435f45ff8f2333451d5cdad37bf201dfe58d254548ba6ccf5814564b98df0",
            "17/2000",
        ),
        "G3A-8d3cce39bcb13ba5061eb78b": (
            "181f6837c6934ff9ffbdba5b32383271704e67781a46b01d86fa31571442da98",
            "9/1000",
        ),
    }
    assert {
        item["candidate_id"]: (item["action_sha256"], item["beta"])
        for item in rebuilt["candidate_records"]
    } == expected
    for record in rebuilt["candidate_records"]:
        assert record["provenance"]["action_sha256"] == record["action_sha256"]
        assert record["data_eligibility"] == ELIGIBILITY


def test_all_direction_certificates_are_exact_but_single_center_only(
    rebuilt: dict,
) -> None:
    expected = {
        "33/4000": (
            "-64003267/64000000",
            "63998911/64000000",
            "63998911/64003267",
            "64007623/64003267",
        ),
        "17/2000": (
            "-16000867/16000000",
            "15999711/16000000",
            "15999711/16000867",
            "16002023/16000867",
        ),
        "9/1000": (
            "-4000243/4000000",
            "3999919/4000000",
            "3999919/4000243",
            "4000567/4000243",
        ),
    }
    for record in rebuilt["candidate_records"]:
        center = record["all_direction_center_certificate"]
        p00, spatial, speed, gap = expected[record["beta"]]
        assert center["effective_metric"]["P00"] == p00
        assert center["effective_metric"]["isotropic_spatial_eigenvalue"] == spatial
        assert (
            center["effective_metric"]["all_unit_spatial_direction_quadratic_form_lower"] == spatial
        )
        assert center["scalar_speed_squared"] == speed
        assert center["slicing_minus_scalar_speed_squared_gap"] == gap
        assert center["status"] == "pass_all_directions_at_single_center_only"
        assert "zero-width center" in center["scope"]
        assert (
            record["gate_ledger"]["all_direction_single_center_principal_and_cone"]
            == "pass_at_center_only"
        )


def test_qualitative_labels_do_not_fabricate_componentwise_boxes(rebuilt: dict) -> None:
    for record in rebuilt["candidate_records"]:
        domain = record["reviewed_domain_contract"]
        assert domain["unfilled_fields"] == REQUIRED_DOMAIN_FIELDS
        assert domain["unfilled_field_count"] == 12
        assert domain["registered_values"] == {field: None for field in REQUIRED_DOMAIN_FIELDS}
        assert domain["nonempty_box_registered"] is False
        assert domain["interval_adapter_invocation_authorized"] is False
        assert domain["qualitative_label_classification"] == (
            "insufficient_no_coordinate_map_norm_definition_frame_normalization_or_anchor"
        )
        interval = record["uniform_interval_attempt"]
        assert interval["certify_cubic_bssn_domain_invoked"] is False
        assert interval["uniform_effective_metric_interval"] is None
        assert interval["all_direction_uniform_certificate"] is None
        assert interval["status"] == "blocked_before_interval_adapter_invocation"


def test_full_lapse_operator_is_derived_without_overclaiming_dirac(rebuilt: dict) -> None:
    expected_center_delta = {
        "33/4000": "32003267/32000000",
        "17/2000": "8000867/8000000",
        "9/1000": "2000243/2000000",
    }
    for record in rebuilt["candidate_records"]:
        lapse = record["lapse_and_dirac_scope"]
        derivation = lapse["full_Delta_N_derivation"]
        assert derivation["full_Delta_N"]["beta_specialization"] == record["beta"]
        assert derivation["full_Delta_N"]["differential_order"] == 0
        assert derivation["full_Delta_N"]["operator_type"] == ("real_multiplication_operator")
        assert set(derivation["exact_residuals"].values()) == {"0"}
        assert lapse["single_center"]["Delta_N"] == expected_center_delta[record["beta"]]
        assert lapse["single_center"]["strictly_positive"] is True
        assert lapse["uniform_coercivity_lower_bound"] is None
        assert lapse["inverse_norm_upper"] is None
        assert lapse["periodic_distributed_Dirac"] == "blocked"
        assert lapse["asymptotically_flat_Dirac"] == "blocked"
        assert lapse["global_energy"] == "blocked"


def test_smaller_beta_and_family_label_are_explicit_nontransfer_controls(
    rebuilt: dict,
) -> None:
    for record in rebuilt["candidate_records"]:
        control = record["nontransfer_control"]
        assert control["candidate_beta_is_smaller"] is True
        assert control["action_identity_match"] is False
        assert control["componentwise_domain_identity_match"] is False
        assert control["monotonicity_transfer_theorem_registered"] is False
        assert control["prior_interval_or_coercivity_bound_reused"] is False
        assert control["decision"] == "pass_negative_control"
        assert record["gate_ledger"]["smaller_beta_or_family_label_transfer"] == (
            "rejected_as_inference"
        )


def test_counts_blockers_and_observation_seals_are_exact(rebuilt: dict) -> None:
    assert rebuilt["candidate_count"] == 3
    assert rebuilt["decision_counts"] == {"blocked": 3}
    assert rebuilt["all_direction_single_center_pass_count"] == 3
    assert rebuilt["nonzero_componentwise_box_pass_count"] == 0
    assert rebuilt["uniform_principal_common_cone_pass_count"] == 0
    assert rebuilt["full_Delta_N_derivation_pass_count"] == 3
    assert rebuilt["uniform_Delta_N_coercivity_pass_count"] == 0
    assert rebuilt["periodic_distributed_Dirac_pass_count"] == 0
    assert rebuilt["asymptotically_flat_Dirac_pass_count"] == 0
    assert rebuilt["full_formal_pass_count"] == 0
    assert rebuilt["first_blocker_counts"] == {FIRST_BLOCKER: 3}
    assert all(record["decision"] == "blocked" for record in rebuilt["candidate_records"])
    assert rebuilt["observational_data_opened"] is False
    assert rebuilt["dark_matter_or_halo_inputs"] is False
    assert rebuilt["redshift_distance_inputs"] is False
    assert rebuilt["paid_llm_spend_usd"] == 0.0
    assert rebuilt["data_eligibility"] == ELIGIBILITY


def test_target_domain_and_binding_tampering_fail_closed(tmp_path: Path) -> None:
    config = _load(CONFIG)
    action = copy.deepcopy(config)
    action["targets"][0]["action_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="target binding changed"):
        build_future_g3_componentwise_domain_contract_campaign(action, ROOT)

    invented = copy.deepcopy(config)
    invented["required_domain_contract"]["registered_values"]["spatial_gradient_component_abs"] = (
        "33/4000"
    )
    with pytest.raises(ValueError, match="required domain contract changed"):
        build_future_g3_componentwise_domain_contract_campaign(invented, ROOT)

    source = copy.deepcopy(config)
    source["adapter_source"]["file_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="campaign source hash mismatch"):
        build_future_g3_componentwise_domain_contract_campaign(source, ROOT)

    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    escaped = copy.deepcopy(config)
    escaped["bindings"]["preflight"] = {
        "path": str(outside),
        "file_sha256": hashlib.sha256(outside.read_bytes()).hexdigest(),
        "content_sha256": "0" * 64,
    }
    with pytest.raises(ValueError, match="path escapes repository"):
        build_future_g3_componentwise_domain_contract_campaign(escaped, ROOT)
