from __future__ import annotations

import copy
import json
from pathlib import Path

from sigma_theory_compiler.quartic_linearized_energy_campaign import (
    certify_quartic_linearized_energy_candidate,
    run_quartic_linearized_energy_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
DIRAC_PATH = (
    ROOT / "runs/physics-language/quartic-dirac-hamiltonian-campaign/campaign.json"
)
CONFIG_PATH = ROOT / "configs/backgrounds/quartic_linearized_energy_campaign.json"
ARTIFACT_PATH = (
    ROOT / "runs/physics-language/quartic-linearized-energy-campaign/campaign.json"
)


def _inputs() -> tuple[dict, dict]:
    return json.loads(DIRAC_PATH.read_text()), json.loads(CONFIG_PATH.read_text())


def test_all_quartic_candidates_have_finite_horizon_linearized_energy() -> None:
    campaign, config = _inputs()
    result = run_quartic_linearized_energy_campaign(campaign, config)
    artifact = json.loads(ARTIFACT_PATH.read_text())
    assert result["status"] == (
        "pass_all_12_finite_horizon_linearized_inhomogeneous_energies"
    )
    assert result["counts"] == {
        "selected": 12,
        "finite_horizon_linearized_energy_passed": 12,
        "rejected": 0,
    }
    assert result["content_sha256"] == artifact["content_sha256"]
    assert all(
        certificate["quadratic_energy"]["all_spatial_wavenumbers"]
        and certificate["quadratic_energy"]["energy_amplification_upper_numeric"]
        >= 1
        and certificate["physical_derivative_tube"][
            "initial_E_s_strict_upper_numeric"
        ]
        > 0
        for certificate in result["certificates"]
    )


def test_linearized_energy_campaign_negative_controls_reject() -> None:
    campaign, config = _inputs()
    result = run_quartic_linearized_energy_campaign(campaign, config)
    assert all(
        control["rejected"] for control in result["negative_controls"].values()
    )


def test_linearized_energy_rejects_invalid_fraction_and_sobolev_order() -> None:
    campaign, config = _inputs()
    candidate = campaign["certificates"][0]
    bad_fraction = copy.deepcopy(config)
    bad_fraction["terminal_amplitude_squared_fraction"] = "0"
    result = run_quartic_linearized_energy_campaign(campaign, bad_fraction)
    assert result["status"] == "reject"
    assert "strictly between" in " ".join(result["errors"])

    bad_sobolev = copy.deepcopy(config)
    bad_sobolev["sobolev_order"] = 2
    result = run_quartic_linearized_energy_campaign(campaign, bad_sobolev)
    assert result["status"] == "reject"
    assert "Sobolev order" in " ".join(result["errors"])

    direct = certify_quartic_linearized_energy_candidate(candidate, config)
    assert "not a nonlinear PDE trapping theorem" in direct["scope"]
