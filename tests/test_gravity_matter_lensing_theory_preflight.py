from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler import gravity_matter_lensing_theory_preflight as theory


def test_frozen_config_and_role_boundary() -> None:
    config = theory.load_config()
    assert config["schema_version"] == theory.CONFIG_SCHEMA
    assert config["role_mapping"]["A_nuisance"] == "ABSENT_FROM_THEORY_ACTION"
    assert config["role_mapping"]["M_temporal_phase"] == "ABSENT_FROM_THEORY_ACTION"
    metric = config["action_family"]["universal_physical_metric"]
    assert metric["matter_and_photons_use_same_metric"] is True
    assert metric["separate_photon_coefficient_forbidden"] is True


def test_term_provenance_is_complete_without_novelty_claim() -> None:
    config = theory.load_config()
    terms = config["term_provenance_ledger"]
    assert tuple(item["term_id"] for item in terms) == theory.TERM_IDS
    assert [item["provenance_label"] for item in terms].count("known_rewrite") == 1
    assert [item["provenance_label"] for item in terms].count("known_combination") == 4
    assert [item["provenance_label"] for item in terms].count("potentially_new_synthesis") == 2
    assert not any(item["historical_novelty_claim"] for item in terms)


def test_dimensions_equations_and_conditional_dof_are_explicit() -> None:
    config = theory.load_config()
    dimensions = config["conventions_and_dimensions"]["mass_dimensions"]
    assert dimensions["X_phi"] == dimensions["X_chi"] == 4
    assert dimensions["lagrangian_density_without_sqrt_minus_g"] == 4
    equations = config["field_equations_and_symbolic_contract"]
    assert equations["machine_verified_symbolic_derivation"] is False
    assert "[X_chi-m_chi^2*chi^2/2]" in equations["field_stress_definition"]
    assert "(X_chi-m_chi^2*chi^2/2)*Z_chi,u" in equations["phi_equation"]
    assert "-Z_chi(u)*m_chi^2*chi" in equations["chi_equation"]
    assert equations["gradient_and_background_mixing_status"].startswith("BLOCKED_")
    assert len(equations["symbolic_derivation_requirements"]) == 7
    dof = config["degrees_of_freedom"]
    assert (dof["tensor_polarizations"], dof["scalar_fields"]) == (2, 2)
    assert dof["status"] == "CONDITIONAL_NOT_HAMILTONIAN_VERIFIED"


def test_lensing_cancellation_and_disformal_obligation_remain_blocked() -> None:
    config = theory.load_config()
    weak = config["weak_field_matter_and_lensing"]
    assert "cancels" in weak["static_conformal_limit"]["lensing_sum"]
    assert "not an independent photon multiplier" in weak["disformal_effect"]
    assert weak["status"].startswith("BLOCKED_")
    gw = config["health_and_consistency_gates"][5]
    assert gw["gate_id"] == "H6_GW_AND_PHOTON_CONES"
    assert "1e-15" in gw["requirement"]
    assert gw["status"].startswith("BLOCKED_")


def test_only_template_level_health_gate_passes() -> None:
    config = theory.load_config()
    gates = config["health_and_consistency_gates"]
    assert len(gates) == 10
    assert gates[0]["status"] == "PASS_TEMPLATE_LEVEL"
    assert all(item["status"].startswith("BLOCKED_") for item in gates[1:])


def test_gr_and_channel_limits_are_frozen_but_parameters_are_not() -> None:
    config = theory.load_config()
    limits = config["exact_parameter_and_limit_gates"]
    action = config["action_family"]
    assert "Z_chi(u)*(X_chi - m_chi^2*chi^2/2)" in action["action"]
    assert action["gate_and_range"]["ell_chi"] == "1/m_chi"
    assert (
        "B(u)^2*exp(-m_chi*r)/r" in action["finite_range_channel"]["slow_background_green_function"]
    )
    assert action["gate_and_range"]["nonconstant_background_status"].startswith("BLOCKED_")
    assert "Einstein gravity" in limits["gr_limit"]
    assert "B->0" in limits["high_acceleration_limit"]
    assert "fixed range 1/m_chi" in limits["high_acceleration_limit"]
    assert "alpha_chi->0" in limits["zero_finite_channel_limit"]
    assert "d_phi=0" in limits["no_disformal_limit"]
    assert limits["parameter_values_frozen"] is False
    assert limits["ell_chi_mpc_frozen"] is None
    assert limits["eft_cutoff_frozen"] is None


def test_feasibility_is_fail_closed_and_no_data_was_accessed() -> None:
    config = theory.load_config()
    assert config["feasibility_adjudication"]["decision"] == theory.DECISION
    assert config["feasibility_adjudication"]["theory_feasible_for_observational_scoring"] is False
    assert set(config["claim_boundary"].values()) == {False}
    assert set(config["zero_access_and_compute"].values()) == {0}


@pytest.mark.parametrize(
    ("section", "mutator"),
    [
        ("role_mapping", lambda value: value.__setitem__("A_nuisance", "PRESENT")),
        (
            "conventions_and_dimensions",
            lambda value: value["mass_dimensions"].__setitem__("X_phi", 3),
        ),
        ("action_family", lambda value: value.__setitem__("action", "tampered")),
        (
            "term_provenance_ledger",
            lambda value: value[0].__setitem__("historical_novelty_claim", True),
        ),
        (
            "field_equations_and_symbolic_contract",
            lambda value: value.__setitem__("machine_verified_symbolic_derivation", True),
        ),
        ("degrees_of_freedom", lambda value: value.__setitem__("scalar_fields", 1)),
        ("conservation_identity", lambda value: value.__setitem__("status", "VERIFIED")),
        (
            "weak_field_matter_and_lensing",
            lambda value: value.__setitem__("status", "PASSED"),
        ),
        (
            "health_and_consistency_gates",
            lambda value: value[1].__setitem__("status", "PASS"),
        ),
        (
            "exact_parameter_and_limit_gates",
            lambda value: value.__setitem__("parameter_values_frozen", True),
        ),
        (
            "feasibility_adjudication",
            lambda value: value.__setitem__("theory_feasible_for_observational_scoring", True),
        ),
        ("claim_boundary", lambda value: value.__setitem__("scientific_claim_allowed", True)),
        (
            "zero_access_and_compute",
            lambda value: value.__setitem__("observational_files_opened", 1),
        ),
    ],
)
def test_every_nested_contract_class_is_hash_frozen(section: str, mutator: object) -> None:
    config = copy.deepcopy(theory.load_config())
    mutator(config[section])  # type: ignore[operator]
    with pytest.raises(theory.GravityMatterLensingTheoryPreflightError, match="content changed"):
        theory.validate_config_contract(config)


def test_predecessor_source_file_tampering_fails_closed(tmp_path: Path) -> None:
    config = theory.load_config()
    binding = config["source_bindings"][0]
    source = Path(binding["path"])
    destination = tmp_path / source
    destination.parent.mkdir(parents=True)
    destination.write_bytes(source.read_bytes() + b" ")
    with pytest.raises(theory.GravityMatterLensingTheoryPreflightError, match="file hash changed"):
        theory._load_source(binding, tmp_path)


@pytest.mark.parametrize(
    ("source_id", "mutation"),
    [
        (
            "gravity_lead_parent_registry",
            lambda value: value["claim_boundary"].__setitem__(
                "registry_pass_establishes_physical_mechanism", True
            ),
        ),
        (
            "gravity_lead_recombination",
            lambda value: value["safety"].__setitem__("children_executed", 1),
        ),
        (
            "shared_ben_synthetic_execution",
            lambda value: value["claim_boundary"].__setitem__("same_action_derived", True),
        ),
    ],
)
def test_predecessor_semantic_overclaim_fails_closed(source_id: str, mutation: object) -> None:
    config = theory.load_config()
    binding = next(item for item in config["source_bindings"] if item["source_id"] == source_id)
    source = json.loads(Path(binding["path"]).read_text(encoding="utf-8"))
    mutation(source)  # type: ignore[operator]
    with pytest.raises(theory.GravityMatterLensingTheoryPreflightError):
        theory._validate_source_semantics(source_id, source)


def test_build_receipt_binds_sources_implementation_and_blocked_counts() -> None:
    receipt = theory.build_receipt()
    assert receipt["decision"] == theory.DECISION
    assert receipt["counts"] == {
        "source_receipts_validated": 3,
        "terms_total": 7,
        "known_rewrite_terms": 1,
        "known_combination_terms": 4,
        "potentially_new_synthesis_terms": 2,
        "health_gates_total": 10,
        "template_level_gates_passed": 1,
        "health_gates_blocked": 9,
        "conditional_tensor_dof": 2,
        "conditional_scalar_dof": 2,
    }
    assert receipt["implementation_binding"]["source_sha256"] == theory._file_sha(
        theory.SOURCE_PATH
    )
    assert receipt["implementation_binding"]["test_sha256"] == theory._file_sha(theory.TEST_PATH)


def test_atomic_no_replace_preserves_concurrent_winner(tmp_path: Path) -> None:
    target = tmp_path / "receipt.json"
    target.write_bytes(b"winner")
    with pytest.raises(theory.GravityMatterLensingTheoryPreflightError, match="refusing"):
        theory._atomic_no_replace(target, b"candidate")
    assert target.read_bytes() == b"winner"
    assert theory._atomic_no_replace(target, b"winner") == "EXISTING_IDENTICAL"


def test_stored_receipt_is_exact_deterministic_rebuild() -> None:
    receipt = theory.check_receipt()
    assert receipt == theory.build_receipt()
    assert receipt["claim_boundary"]["scientific_claim_allowed"] is False
    assert receipt["zero_access_and_compute"]["observational_files_opened"] == 0


def test_receipt_tampering_fails_closed() -> None:
    config = theory.load_config()
    receipt = theory.build_receipt()
    receipt["counts"]["health_gates_blocked"] = 8
    with pytest.raises(
        theory.GravityMatterLensingTheoryPreflightError, match="content hash invalid"
    ):
        theory.validate_receipt(receipt, config)
