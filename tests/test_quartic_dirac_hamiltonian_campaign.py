from __future__ import annotations

import copy
import json
from pathlib import Path

from sigma_theory_compiler.quartic_dirac_hamiltonian_campaign import (
    run_quartic_dirac_hamiltonian_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
IR = ROOT / "runs/physics-language/horndeski-l2-l4-polynomial-ir.json"
BINDINGS = (
    ROOT / "runs/physics-language/quartic-linear-x-symbol-campaign/campaign.json"
)
SYMMETRIZERS = (
    ROOT
    / "runs/physics-language/quartic-symmetrizer-uniform-domain-campaign/campaign.json"
)
CONFIG = ROOT / "configs/backgrounds/quartic_dirac_hamiltonian_campaign.json"
ARTIFACT = (
    ROOT
    / "runs/physics-language/quartic-dirac-hamiltonian-campaign/campaign.json"
)


def _inputs() -> tuple[dict, dict, dict, dict]:
    return tuple(
        json.loads(path.read_text(encoding="utf-8"))
        for path in (IR, BINDINGS, SYMMETRIZERS, CONFIG)
    )


def test_all_twelve_have_on_shell_dirac_and_positive_quadratic_energy() -> None:
    ir, bindings, symmetrizers, config = _inputs()
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    result = run_quartic_dirac_hamiltonian_campaign(
        ir, bindings, symmetrizers, config
    )
    assert result["status"] == (
        "pass_all_12_local_on_shell_adm_dirac_and_quadratic_hamiltonian"
    ), result
    assert result["counts"] == {
        "selected": 12,
        "local_on_shell_adm_dirac_hamiltonian_passed": 12,
        "rejected": 0,
    }
    assert result["content_sha256"] == artifact["content_sha256"]
    assert len(result["certificates"]) == 12
    for certificate in result["certificates"]:
        assert certificate["status"] == (
            "pass_local_on_shell_adm_dirac_and_quadratic_hamiltonian"
        )
        witness = certificate["on_shell_local_flrw_witness"]
        assert set(witness["equation_residuals"].values()) == {"0"}
        assert witness["lapse_constraint_time_derivative_residual"] == "0"
        assert witness["regular_local_solution"]
        assert certificate["certified_local_jet_embedding"]["all_inside"]
        adm = certificate["adm_hessian_and_primary_constraint"]
        assert (adm["rank"], adm["nullity"]) == (6, 1)
        assert adm["primary_constraint"] == "p_V_star=0"
        chain = certificate["dirac_chain"]
        assert chain["pairing_is_strictly_positive"]
        assert chain["constraint_count"]["physical_configuration_dof"] == 3
        assert not chain["higher_constraints"]
        hamiltonian = certificate["on_shell_quadratic_physical_hamiltonian"]
        assert hamiltonian["strictly_positive"]
        assert all(hamiltonian["source_reduction_controls_passed"].values())
        invariant = certificate["forward_homogeneous_invariant_domain"]
        assert invariant["passed"]
        assert invariant["sign_proof"]["A_star_squared_strictly_decreases"]
        assert invariant["uniform_absolute_bounds"][
            "all_local_jets_inside_symmetrizer_box"
        ]
        assert all(
            value is True
            for value in invariant[
                "health_signs_for_every_finite_future_time"
            ].values()
            if isinstance(value, bool)
        )
    assert all(item["rejected"] for item in result["negative_controls"].values())


def test_symmetrizer_hash_mismatch_rejects_before_dirac_promotion() -> None:
    ir, bindings, symmetrizers, config = _inputs()
    corrupted = copy.deepcopy(symmetrizers)
    corrupted["binding_campaign_sha256"] = "corrupted"
    result = run_quartic_dirac_hamiltonian_campaign(
        ir, bindings, corrupted, config
    )
    assert result["status"] == "reject"
    assert "hash mismatch" in " ".join(result["errors"])
