from __future__ import annotations

import copy
import json
from pathlib import Path

from sigma_theory_compiler.quartic_constraint_reconstruction_campaign import (
    generic_scalar_constraint_reconstruction_control,
    run_quartic_constraint_reconstruction_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
DIRAC_PATH = (
    ROOT / "runs/physics-language/quartic-dirac-hamiltonian-campaign/campaign.json"
)
ENERGY_PATH = (
    ROOT / "runs/physics-language/quartic-linearized-energy-campaign/campaign.json"
)
CONFIG_PATH = (
    ROOT / "configs/backgrounds/quartic_constraint_reconstruction_campaign.json"
)
ARTIFACT_PATH = (
    ROOT
    / "runs/physics-language/quartic-constraint-reconstruction-campaign/campaign.json"
)


def _inputs() -> tuple[dict, dict, dict]:
    return (
        json.loads(DIRAC_PATH.read_text()),
        json.loads(ENERGY_PATH.read_text()),
        json.loads(CONFIG_PATH.read_text()),
    )


def test_generic_scalar_constraint_reconstruction_is_exact() -> None:
    passed, evidence = generic_scalar_constraint_reconstruction_control()
    assert passed, evidence
    assert evidence["identity_residuals"] == {
        "lapse": "0",
        "shift": "0",
        "closed_shift": "0",
    }
    assert evidence["negative_control"]["rejected"]
    assert "k=0" in evidence["kernel_contract"]


def test_all_quartic_candidates_reconstruct_linear_auxiliaries() -> None:
    dirac, energy, config = _inputs()
    result = run_quartic_constraint_reconstruction_campaign(dirac, energy, config)
    artifact = json.loads(ARTIFACT_PATH.read_text())
    assert result["status"] == "pass_all_12_linear_constraint_reconstructions"
    assert result["counts"] == {
        "selected": 12,
        "linear_constraint_reconstruction_passed": 12,
        "rejected": 0,
    }
    assert result["content_sha256"] == artifact["content_sha256"]
    assert all(
        item["operator_norm_bounds"]["combined_reconstruction_upper_numeric"] > 0
        and item["chained_energy_tube"]["final_initial_E_s_strict_upper_numeric"]
        > 0
        and "does not control their time derivatives" in item["scope"]
        for item in result["certificates"]
    )


def test_reconstruction_rejects_infrared_and_hash_failures() -> None:
    dirac, energy, config = _inputs()
    infrared = copy.deepcopy(config)
    infrared["minimum_nonzero_wave_number"] = "0"
    result = run_quartic_constraint_reconstruction_campaign(dirac, energy, infrared)
    assert result["status"] == "reject"
    assert "strictly positive" in " ".join(result["errors"])

    corrupted = copy.deepcopy(energy)
    corrupted["dirac_campaign_sha256"] = "corrupted"
    result = run_quartic_constraint_reconstruction_campaign(dirac, corrupted, config)
    assert result["status"] == "reject"
    assert "hash mismatch" in " ".join(result["errors"])
