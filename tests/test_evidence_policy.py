import json
from copy import deepcopy
from pathlib import Path

from sigma_theory_compiler.observation_eligibility import (
    audit_galaxy_observable_protocol,
)

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "configs" / "observational_evidence_policy.json"
GALAXY_PROTOCOL = ROOT / "configs" / "galaxy_observable_protocol.json"


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
