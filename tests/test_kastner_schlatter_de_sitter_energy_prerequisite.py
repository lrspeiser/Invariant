from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.kastner_schlatter_de_sitter_energy_prerequisite import (
    FIRST_BLOCKER,
    _validate_result,
    build_prerequisite,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/kastner_schlatter_de_sitter_energy_prerequisite.json"
ARTIFACT = ROOT / "runs/engine/kastner-schlatter-de-sitter-energy-prerequisite.json"


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def rebuilt() -> dict[str, object]:
    return build_prerequisite(CONFIG)


def test_exact_rebuild_partition_and_counts(rebuilt: dict[str, object]) -> None:
    artifact = _load(ARTIFACT)
    assert rebuilt == artifact
    _validate_result(artifact)
    assert artifact["decision_counts"] == {"pass": 0, "reject": 0, "blocked": 2}
    assert artifact["prerequisite_counts"] == {
        "candidate_actions": 2,
        "exact_de_Sitter_background_radius_pass": 2,
        "covariant_charge_interface_pass": 2,
        "closed_slice_empty_boundary_control_pass": 2,
        "fixed_background_scalar_positive_energy_pass": 2,
        "nontrivial_integrable_coupled_charge_pass": 0,
        "nonlinear_coupled_positive_energy_pass": 0,
        "paper_or_QED_derived_actions": 0,
        "full_formal_admission_pass": 0,
    }
    assert artifact["first_blocker"] == FIRST_BLOCKER


def test_branch_de_sitter_radii_are_exact(rebuilt: dict[str, object]) -> None:
    middle, printed = rebuilt["candidate_records"]
    assert middle["de_Sitter_background"]["exact_radius"] == (
        "L^2=3*c^3/(4*pi*G*h_planck*q0)"
    )
    assert printed["de_Sitter_background"]["exact_radius"] == (
        "L^2=3*c^3/(2*pi*G*h_planck*q0)"
    )
    assert middle["de_Sitter_background"]["radius_squared_relative_to_middle_branch"] == "1"
    assert printed["de_Sitter_background"]["radius_squared_relative_to_middle_branch"] == "2"


def test_covariant_charge_interface_is_exact_but_not_overclaimed(rebuilt: dict[str, object]) -> None:
    for record in rebuilt["candidate_records"]:
        charge = record["covariant_charge_interface"]
        assert charge["status"] == "pass_exact_variational_interface_only"
        assert "delta(Q_xi)-i_xi(theta_g+theta_q)" in charge["surface_variation"]
        assert charge["stationary_scalar_linear_surface_term"] == "theta_q[bar_q=q0]=0"
        assert charge["nontrivial_integrable_charge_value_registered"] is False
        assert record["closed_global_slice_control"]["surface_charge_variation"] == "0"
        assert record["closed_global_slice_control"]["positive_energy_theorem"] is False


def test_fixed_background_scalar_energy_positive_only_with_zero_flux(
    rebuilt: dict[str, object],
) -> None:
    for record in rebuilt["candidate_records"]:
        energy = record["static_patch_scalar_energy"]
        assert energy["nonnegative"] is True
        assert energy["strict_except_zero_field"] is True
        assert energy["conserved_if"] == "Flux_phi[boundary_Sigma]=0"
        assert energy["metric_backreaction_included"] is False
        assert energy["gravitational_charge_positivity_inferred"] is False
        obstruction = record["global_Killing_obstruction"]
        assert obstruction["de_Sitter_has_everywhere_globally_timelike_Killing_field"] is False


def test_negative_controls_reject_scope_errors(rebuilt: dict[str, object]) -> None:
    controls = rebuilt["negative_controls"]
    assert "not asymptotically flat" in controls["ADM_substitution"]["rejected_because"]
    assert "identically trivial" in controls["empty_boundary_positivity"]["rejected_because"]
    assert "metric backreaction" in controls["scalar_to_coupled_inference"]["rejected_because"]
    assert controls["nonzero_flux_conservation"]["admitted"] is False
    assert all(value["admitted"] is False for value in controls.values() if isinstance(value, dict))


def test_attribution_claims_and_data_remain_sealed(rebuilt: dict[str, object]) -> None:
    assert all(record["paper_or_QED_derived"] is False for record in rebuilt["candidate_records"])
    assert all(record["candidate_rejection_authorized"] is False for record in rebuilt["candidate_records"])
    assert all(value is False for value in rebuilt["claim_seals"].values())
    assert all(value is False for value in rebuilt["data_seals"].values())


def test_lineage_config_and_result_tampering_fail_closed(
    tmp_path: Path, rebuilt: dict[str, object]
) -> None:
    config = _load(CONFIG)
    tampered = copy.deepcopy(config)
    tampered["predecessors"]["formal_admission_artifact"]["content_sha256"] = "0" * 64
    config_path = tmp_path / "configs" / CONFIG.name
    config_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="file hash mismatch|content hash mismatch"):
        build_prerequisite(config_path)

    opened = copy.deepcopy(config)
    opened["seals"]["observations_opened"] = True
    config_path.write_text(json.dumps(opened), encoding="utf-8")
    with pytest.raises(ValueError, match="seals changed"):
        build_prerequisite(config_path)

    attributed = copy.deepcopy(rebuilt)
    attributed["candidate_records"][0]["paper_or_QED_derived"] = True
    attributed.pop("content_sha256")
    with pytest.raises(ValueError, match="attribution changed"):
        _validate_result(attributed)

    overclaim = copy.deepcopy(rebuilt)
    overclaim["candidate_records"][0]["full_formal_admission_pass"] = True
    overclaim.pop("content_sha256")
    with pytest.raises(ValueError, match="overreached to full admission"):
        _validate_result(overclaim)


def test_source_bindings_are_exact_and_portable(rebuilt: dict[str, object]) -> None:
    bindings = rebuilt["source_bindings"]
    for label in ("config", "source", "test"):
        binding = bindings[label]
        assert _file_sha(ROOT / binding["path"]) == binding["file_sha256"]
        assert not Path(binding["path"]).is_absolute()
    assert all("C:/" not in str(value) and "C:\\" not in str(value) for value in bindings.values())
