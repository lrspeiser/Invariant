import json
from copy import deepcopy
from pathlib import Path

from sigma_theory_compiler.observation_eligibility import (
    audit_galaxy_observable_protocol,
    audit_solar_observable_protocol,
    audit_solar_source_registration,
)

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "configs" / "observational_evidence_policy.json"
GALAXY_PROTOCOL = ROOT / "configs" / "galaxy_observable_protocol.json"
SOLAR_PROTOCOL = ROOT / "configs" / "solar_observable_protocol.json"
CASSINI_SCE1_SOURCE = (
    ROOT / "configs" / "observations" / "cassini_sce1_source_registration.json"
)
GALAXY_AUDIT = ROOT / "runs" / "observation-protocol" / "galaxy-observable-audit.json"
SOLAR_AUDIT = ROOT / "runs" / "observation-protocol" / "solar-observable-audit.json"
CASSINI_AUDIT = (
    ROOT
    / "runs"
    / "observation-protocol"
    / "cassini-sce1-source-registration-audit.json"
)


def test_unobserved_components_cannot_be_truth_or_rescue() -> None:
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    invisible = policy["unobserved_components"]
    assert invisible["default_status"] == "prohibited_as_truth_or_rescue"
    assert "target labels" in invisible["prohibited_uses"]
    assert "post-hoc rescue of a failed baryons-only law" in invisible["prohibited_uses"]


def test_redshift_is_measurement_not_automatic_distance() -> None:
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    redshift = policy["redshift"]
    assert "wavelength ratio" in redshift["allowed"]
    assert "treating redshift as a distance" in redshift["not_allowed_by_default"]
    assert policy["supernovae"]["default_status"] == "excluded"


def test_policy_prefers_distance_free_observables() -> None:
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    examples = policy["allowed_derived_quantities"]["preferred_distance_free_examples"]
    assert "velocity ratios within one source" in examples
    assert "angular-radius ratios within one source" in examples


def test_galaxy_discovery_protocol_is_sealed_and_observable_to_observable() -> None:
    audit = audit_galaxy_observable_protocol(GALAXY_PROTOCOL, POLICY)
    assert audit["status"] == "pass", audit
    assert audit["split_unit"] == "whole galaxy"
    assert audit["object_specific_gravity_parameters"] == 0
    assert audit["lensing_formula_selection_use"] == "prohibited"
    assert not audit["redshift_distance_allowed_by_default"]
    assert audit["supernova_default_status"] == "excluded"
    assert not audit["observational_dataset_opened"]
    assert not audit["formula_search_authorized"]


def test_galaxy_protocol_rejects_target_leakage_and_early_data_opening(
    tmp_path: Path,
) -> None:
    protocol = json.loads(GALAXY_PROTOCOL.read_text(encoding="utf-8"))
    corrupted = deepcopy(protocol)
    corrupted["data_opened"] = True
    corrupted["discovery_channel"]["inputs"].append("NFW halo mass")
    corrupted["split_contract"]["unit"] = "rotation-curve row"
    corrupted["scoring_contract"]["object_specific_gravity_parameters"] = 1
    corrupted["independent_lensing_falsification"]["formula_selection_use"] = "allowed"
    path = tmp_path / "corrupted-galaxy-protocol.json"
    path.write_text(json.dumps(corrupted), encoding="utf-8")
    audit = audit_galaxy_observable_protocol(path, POLICY)
    assert audit["status"] == "fail"
    assert any("opened data" in error for error in audit["errors"])
    assert any("nfw" in error.casefold() for error in audit["errors"])
    assert any("whole galaxy" in error for error in audit["errors"])
    assert any("object-specific" in error for error in audit["errors"])
    assert any("lensing" in error for error in audit["errors"])


def test_solar_protocol_is_sealed_and_direct_observable() -> None:
    audit = audit_solar_observable_protocol(SOLAR_PROTOCOL, POLICY)
    assert audit["status"] == "pass", audit
    assert audit["quantity_classes"] == [
        "calibrated",
        "derived",
        "latent",
        "model_dependent",
        "raw",
    ]
    assert audit["split_unit"] == "tracking pass or observing session"
    assert audit["object_specific_gravity_parameters"] == 0
    assert not audit["redshift_distance_allowed_by_default"]
    assert audit["supernova_default_status"] == "excluded"
    assert not audit["observational_dataset_opened"]
    assert not audit["formula_search_authorized"]


def test_solar_protocol_rejects_model_labels_leakage_and_early_opening(
    tmp_path: Path,
) -> None:
    protocol = json.loads(SOLAR_PROTOCOL.read_text(encoding="utf-8"))
    corrupted = deepcopy(protocol)
    corrupted["data_opened"] = True
    corrupted["measurement_channel"]["allowed_formula_inputs"].append("fitted PPN gamma")
    corrupted["split_contract"]["unit"] = "individual Doppler row"
    corrupted["scoring_contract"]["object_specific_gravity_parameters"] = 1
    corrupted["quantity_classes"]["model_dependent"]["allowed_as_prediction_truth"] = True
    corrupted["split_contract"]["sealed_test_rule"] = "mutable after target inspection"
    path = tmp_path / "corrupted-solar-protocol.json"
    path.write_text(json.dumps(corrupted), encoding="utf-8")
    audit = audit_solar_observable_protocol(path, POLICY)
    assert audit["status"] == "fail"
    assert any("opened data" in error for error in audit["errors"])
    assert any("ppn gamma" in error.casefold() for error in audit["errors"])
    assert any("tracking pass" in error for error in audit["errors"])
    assert any("object-specific" in error for error in audit["errors"])
    assert any("model-dependent" in error for error in audit["errors"])
    assert any("sealed-test" in error for error in audit["errors"])


def test_cassini_sce1_metadata_registration_is_authoritative_but_not_data_ready() -> None:
    audit = audit_solar_source_registration(CASSINI_SCE1_SOURCE, SOLAR_PROTOCOL, POLICY)
    assert audit["status"] == "pass_metadata_registration", audit
    assert audit["dataset_id"] == "CO-SS-RSS-1-SCE1-V1.0"
    assert audit["registered_catalog_files"] == 16
    assert not audit["dataset_ready"]
    assert not audit["candidate_use_authorized"]
    assert not audit["observational_dataset_opened"]
    assert len(audit["next_required_work"]) >= 6


def test_cassini_registration_rejects_tamper_leakage_and_false_readiness(
    tmp_path: Path,
) -> None:
    manifest = json.loads(CASSINI_SCE1_SOURCE.read_text(encoding="utf-8"))
    corrupted = deepcopy(manifest)
    corrupted["data_opened"] = True
    corrupted["source"]["archive_url"] = "https://example.invalid/untrusted"
    corrupted["remote_catalog_fingerprints"][0]["sha256"] = "not-a-hash"
    corrupted["record_classification"]["model_dependent_records"][
        "allowed_as_prediction_truth"
    ] = True
    corrupted["future_prediction_targets"].append("fitted PPN gamma")
    corrupted["future_split_contract"]["unit"] = "individual record"
    corrupted["readiness"]["dataset_ready"] = True
    path = tmp_path / "corrupted-cassini-registration.json"
    path.write_text(json.dumps(corrupted), encoding="utf-8")
    audit = audit_solar_source_registration(path, SOLAR_PROTOCOL, POLICY)
    assert audit["status"] == "fail"
    assert any("opened primary data" in error for error in audit["errors"])
    assert any("allowlisted" in error for error in audit["errors"])
    assert any("fingerprint" in error for error in audit["errors"])
    assert any("model-dependent" in error for error in audit["errors"])
    assert any("ppn gamma" in error.casefold() for error in audit["errors"])
    assert any("split unit" in error for error in audit["errors"])
    assert any("readiness" in error for error in audit["errors"])


def test_cassini_registration_rejects_duplicate_and_deceptive_catalog_urls(
    tmp_path: Path,
) -> None:
    manifest = json.loads(CASSINI_SCE1_SOURCE.read_text(encoding="utf-8"))
    corrupted = deepcopy(manifest)
    corrupted["source"]["profile_url"] = "https://pds.nasa.gov.example.invalid/profile"
    duplicate = deepcopy(corrupted["remote_catalog_fingerprints"][0])
    duplicate["url"] = "https://atmos.nmsu.edu.example.invalid/duplicate"
    corrupted["remote_catalog_fingerprints"].append(duplicate)
    path = tmp_path / "duplicate-cassini-registration.json"
    path.write_text(json.dumps(corrupted), encoding="utf-8")
    audit = audit_solar_source_registration(path, SOLAR_PROTOCOL, POLICY)
    assert audit["status"] == "fail"
    assert any("profile_url" in error for error in audit["errors"])
    assert any("all eight Cassini" in error for error in audit["errors"])
    assert any("not allowlisted" in error for error in audit["errors"])


def test_checked_in_observation_audits_reproduce_exactly() -> None:
    expected = {
        GALAXY_AUDIT: audit_galaxy_observable_protocol(GALAXY_PROTOCOL, POLICY),
        SOLAR_AUDIT: audit_solar_observable_protocol(SOLAR_PROTOCOL, POLICY),
        CASSINI_AUDIT: audit_solar_source_registration(
            CASSINI_SCE1_SOURCE, SOLAR_PROTOCOL, POLICY
        ),
    }
    for path, result in expected.items():
        assert json.loads(path.read_text(encoding="utf-8")) == result
