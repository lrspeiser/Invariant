from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.sigma_core import canonical_sha256
from sigma_theory_compiler.universal_matter_coupled_pde_control_pack import (
    CLAIMS,
    CONFIG_PATH,
    GATE_IDS,
    OUTPUT_PATH,
    SECTOR_IDS,
    UniversalMatterControlError,
    build_universal_matter_control_pack,
    validate_checked_artifact,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / CONFIG_PATH).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def rebuilt() -> dict:
    return build_universal_matter_control_pack(CONFIG, ROOT)


@pytest.fixture(scope="module")
def checked(rebuilt: dict) -> dict:
    value = json.loads((ROOT / OUTPUT_PATH).read_text(encoding="utf-8"))
    validate_checked_artifact(value, CONFIG, ROOT)
    assert value == rebuilt
    return value


def _sector(value: dict, sector_id: str) -> dict:
    return next(row for row in value["sector_results"] if row["sector_id"] == sector_id)


def _gate(value: dict, sector_id: str, gate_id: str) -> dict:
    sector = _sector(value, sector_id)
    return next(row for row in sector["gates"] if row["gate_id"] == gate_id)


def _reseal(value: dict) -> None:
    value["content_sha256"] = canonical_sha256(
        {key: item for key, item in value.items() if key != "content_sha256"}
    )


def test_measured_pack_counts_and_terminal_outcomes_are_exact(checked: dict) -> None:
    assert checked["decision"] == "bounded_pass_one_sector_two_blocked_no_rejections"
    assert checked["counts"] == {
        "exact_symbolic_replays": 1,
        "floating_point_operations": 0,
        "formal_controls_bound": 6,
        "gate_blocks": 5,
        "gate_passes": 7,
        "gate_rejects": 0,
        "gates": 12,
        "sector_blocks": 2,
        "sector_passes": 1,
        "sector_rejects": 0,
        "sectors": 3,
    }
    assert [row["sector_id"] for row in checked["sector_results"]] == list(SECTOR_IDS)
    assert [row["status"] for row in checked["sector_results"]] == [
        "PASS",
        "BLOCK",
        "BLOCK",
    ]
    assert checked["first_blocker"] == {
        "sector_id": "maxwell_lorenz_gauge",
        "gate_id": "stress_energy_conservation_interface",
        "outcome": "BLOCK",
        "reason_codes": ["dedicated_maxwell_hilbert_stress_noether_identity"],
    }


def test_scalar_has_action_stress_principal_and_vacuous_constraint_passes(
    checked: dict,
) -> None:
    scalar = _sector(checked, "minimally_coupled_scalar")
    assert scalar["status"] == "PASS"
    assert [gate["gate_id"] for gate in scalar["gates"]] == list(GATE_IDS)
    assert [gate["outcome"] for gate in scalar["gates"]] == ["PASS"] * 4
    action = scalar["gates"][0]["evidence"]
    assert action["physical_metric"] == "g_mu_nu"
    assert action["candidate_gravitational_field_dependencies"] == []
    assert action["diagnostic_dependencies"] == []
    assert action["metric_matter_cross_second_derivative_principal_terms"] == 0
    stress = scalar["gates"][1]["evidence"]
    assert stress["identity_residuals"] == ["0", "0", "0", "0"]
    assert stress["off_shell_identity"] == "nabla^mu T_mu_nu = E_phi nabla_nu(phi)"
    principal = scalar["gates"][2]["evidence"]
    assert principal["scalar_speed_squared"] == "1"
    assert principal["exact_diagonal_physical_eigenbasis"] is True
    assert principal["gravity_h7_used"] is False
    constraint = scalar["gates"][3]["evidence"]
    assert constraint["independent_internal_constraints"] == 0
    assert constraint["gravity_constraint_propagation_claimed"] is False


def test_maxwell_passes_bounded_action_principal_and_constraint_but_blocks_stress(
    checked: dict,
) -> None:
    maxwell = _sector(checked, "maxwell_lorenz_gauge")
    assert maxwell["status"] == "BLOCK"
    assert [gate["outcome"] for gate in maxwell["gates"]] == [
        "PASS",
        "BLOCK",
        "PASS",
        "PASS",
    ]
    action = maxwell["gates"][0]["evidence"]
    assert action["physical_metric"] == "g_mu_nu"
    assert action["mass_term_removed_before_maxwell_use"] is True
    assert action["proca_physical_mode_count_not_reused_for_maxwell"] is True
    stress = maxwell["gates"][1]
    assert stress["reason_codes"] == ["dedicated_maxwell_hilbert_stress_noether_identity"]
    assert stress["evidence"]["dedicated_massless_off_shell_noether_identity_available"] is False
    assert stress["evidence"]["on_shell_conservation_promoted"] is False
    principal = maxwell["gates"][2]["evidence"]
    assert principal["principal_scalar"] == "k**2 - omega**2"
    assert principal["principal_determinant"] == "(k - omega)**4*(k + omega)**4"
    assert principal["characteristic_roots_for_unit_spatial_covector"] == ["-1", "1"]
    assert principal["strong_hyperbolicity"] is True
    assert principal["physical_maxwell_dirac_reduction_proved_here"] is False
    constraint = maxwell["gates"][3]["evidence"]
    assert constraint["constraint_wave_principal"] == "k**2 - omega**2"
    assert constraint["divergence_commutation_residual"] == "0"
    assert constraint["nonlinear_curved_boundary_propagation_claimed"] is False


def test_barotropic_fluid_is_explicitly_blocked_at_every_required_gate(
    checked: dict,
) -> None:
    fluid = _sector(checked, "barotropic_perfect_fluid")
    assert fluid["status"] == "BLOCK"
    assert [gate["outcome"] for gate in fluid["gates"]] == ["BLOCK"] * 4
    reasons = {reason for gate in fluid["gates"] for reason in gate["reason_codes"]}
    assert reasons == {
        "missing_admitted_variational_matter_action",
        "admitted_barotropic_fluid_action",
        "variational_perfect_fluid_stress_tensor",
        "positive_enthalpy_domain",
        "exact_sound_speed_interval",
        "fluid_constraint_propagation_certificate",
    }
    assert fluid["promotion_allowed"] is False


def test_contract_and_formal_evidence_are_hash_bound_without_host_paths(checked: dict) -> None:
    bindings = checked["source_bindings"]["registered_evidence"]
    assert set(bindings) == {
        "canonical_scalar_action_ir",
        "canonical_scalar_principal_ir",
        "covariant_field_contract",
        "formal_controls",
        "proca_action_ir",
        "proca_principal_ir",
    }
    assert all(len(binding["file_sha256"]) == 64 for binding in bindings.values())
    encoded = json.dumps(checked, sort_keys=True)
    assert "C:\\Users\\" not in encoded
    assert "/home/" not in encoded
    assert checked["audit"]["runtime_or_observational_data_accesses"] == 0
    assert checked["audit"]["dark_matter_or_halo_targets"] is False
    assert checked["audit"]["redshift_distance_or_supernova_inputs"] is False


def test_claims_never_promote_bounded_controls_to_universal_or_h7_closure(
    checked: dict,
) -> None:
    assert checked["claims"] == CLAIMS
    assert CLAIMS["scalar_control_pack_complete_within_local_scope"] is True
    assert CLAIMS["maxwell_full_stress_conservation_interface_complete"] is False
    assert CLAIMS["barotropic_fluid_control_pack_complete"] is False
    assert CLAIMS["universal_all_matter_closure_established"] is False
    assert CLAIMS["gravity_h7_theorem_established"] is False
    assert CLAIMS["nonlinear_coupled_gravity_matter_pde_closure_established"] is False
    assert CLAIMS["promotion_authorized"] is False


def test_nonuniversal_metric_and_forbidden_candidate_field_reject_action_gate() -> None:
    wrong_metric = copy.deepcopy(CONFIG)
    wrong_metric["sectors"][0]["action"]["metric"] = "object_specific_metric"
    rejected = build_universal_matter_control_pack(wrong_metric, ROOT)
    gate = _gate(rejected, "minimally_coupled_scalar", "action_level_universal_metric_coupling")
    assert gate["outcome"] == "REJECT"
    assert gate["reason_codes"] == ["nonuniversal_or_missing_physical_metric"]
    assert _sector(rejected, "minimally_coupled_scalar")["status"] == "REJECT"

    forbidden = copy.deepcopy(CONFIG)
    forbidden["sectors"][1]["action"]["dependencies"].append("z_b")
    rejected = build_universal_matter_control_pack(forbidden, ROOT)
    gate = _gate(rejected, "maxwell_lorenz_gauge", "action_level_universal_metric_coupling")
    assert gate["outcome"] == "REJECT"
    assert gate["evidence"]["forbidden_dependencies"] == ["z_b"]


@pytest.mark.parametrize(
    "mutator",
    [
        lambda value: value["counts"].__setitem__("sector_passes", 3),
        lambda value: value["claims"].__setitem__("universal_all_matter_closure_established", True),
        lambda value: value["sector_results"][1]["gates"][1].__setitem__("outcome", "PASS"),
        lambda value: value["sector_results"][2]["gates"][2]["evidence"].__setitem__(
            "conditional_hyperbolicity_not_promoted", False
        ),
        lambda value: value["source_bindings"]["registered_evidence"][
            "formal_controls"
        ].__setitem__("file_sha256", "0" * 64),
        lambda value: value.__setitem__("unknown_top_level_key", True),
    ],
)
def test_resealed_artifact_tampers_fail_closed(checked: dict, mutator) -> None:
    tampered = copy.deepcopy(checked)
    mutator(tampered)
    _reseal(tampered)
    with pytest.raises(UniversalMatterControlError):
        validate_checked_artifact(tampered, CONFIG, ROOT)


def test_malformed_config_and_missing_bound_evidence_fail_closed(tmp_path: Path) -> None:
    malformed = copy.deepcopy(CONFIG)
    malformed["policies"]["missing_evidence_outcome"] = "PASS"
    with pytest.raises(UniversalMatterControlError, match="policy semantics"):
        build_universal_matter_control_pack(malformed, ROOT)

    missing = copy.deepcopy(CONFIG)
    missing["evidence_bindings"]["formal_controls"]["path"] = "runs/formal-controls-v1/missing.json"
    with pytest.raises(UniversalMatterControlError, match="cannot load registered evidence"):
        build_universal_matter_control_pack(missing, ROOT)
