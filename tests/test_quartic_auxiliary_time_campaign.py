from __future__ import annotations

import copy
import json
from pathlib import Path

from sigma_theory_compiler.quartic_auxiliary_time_campaign import (
    generic_auxiliary_time_reconstruction_control,
    run_quartic_auxiliary_time_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "runs/physics-language"
DIRAC_PATH = BASE / "quartic-dirac-hamiltonian-campaign/campaign.json"
ENERGY_PATH = BASE / "quartic-linearized-energy-campaign/campaign.json"
RECONSTRUCTION_PATH = BASE / "quartic-constraint-reconstruction-campaign/campaign.json"
CONFIG_PATH = ROOT / "configs/backgrounds/quartic_auxiliary_time_campaign.json"
ARTIFACT_PATH = BASE / "quartic-auxiliary-time-campaign/campaign.json"


def _inputs() -> tuple[dict, dict, dict, dict]:
    return (
        json.loads(DIRAC_PATH.read_text()),
        json.loads(ENERGY_PATH.read_text()),
        json.loads(RECONSTRUCTION_PATH.read_text()),
        json.loads(CONFIG_PATH.read_text()),
    )


def test_generic_auxiliary_time_identities_are_exact() -> None:
    passed, evidence = generic_auxiliary_time_reconstruction_control()
    assert passed, evidence
    assert evidence["identity_residuals"] == {
        "scalar_equation_on_acceleration_solution": "0",
        "lapse_time": "0",
        "shift_time": "0",
    }
    assert evidence["negative_control"]["rejected"]


def test_all_quartic_candidates_have_auxiliary_time_bounds() -> None:
    dirac, energy, reconstruction, config = _inputs()
    result = run_quartic_auxiliary_time_campaign(
        dirac, energy, reconstruction, config
    )
    artifact = json.loads(ARTIFACT_PATH.read_text())
    assert result["status"] == (
        "pass_all_12_linear_auxiliary_time_reconstructions"
    )
    assert result["counts"] == {
        "selected": 12,
        "linear_auxiliary_time_reconstruction_passed": 12,
        "rejected": 0,
    }
    assert result["content_sha256"] == artifact["content_sha256"]
    assert all(
        item["time_reconstruction_operator"]["combined_upper_numeric"] > 0
        and item["chained_energy_tube"]["final_initial_E_s_strict_upper_numeric"]
        > 0
        and "does not bound nonlinear constraint products" in item["scope"]
        for item in result["certificates"]
    )


def test_auxiliary_time_campaign_rejects_hash_and_sobolev_failures() -> None:
    dirac, energy, reconstruction, config = _inputs()
    corrupted = copy.deepcopy(reconstruction)
    corrupted["energy_campaign_sha256"] = "corrupted"
    result = run_quartic_auxiliary_time_campaign(
        dirac, energy, corrupted, config
    )
    assert result["status"] == "reject"
    assert "hash mismatch" in " ".join(result["errors"])

    bad_energy = copy.deepcopy(energy)
    for item in bad_energy["certificates"]:
        item["quadratic_energy"]["sobolev_order"] = 3
    bad_reconstruction = copy.deepcopy(reconstruction)
    bad_reconstruction["energy_campaign_sha256"] = bad_energy["content_sha256"]
    result = run_quartic_auxiliary_time_campaign(
        dirac, bad_energy, bad_reconstruction, config
    )
    assert result["status"] == "reject"
    assert "Sobolev order" in " ".join(result["errors"])
