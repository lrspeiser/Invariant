from __future__ import annotations

import copy
import json
from pathlib import Path

from sigma_theory_compiler.flrw_campaign import run_flrw_background_campaign

ROOT = Path(__file__).resolve().parents[1]
IR_PATH = ROOT / "runs/physics-language/horndeski-l2-l4-polynomial-ir.json"
CONFIG_PATH = (
    ROOT / "configs/backgrounds/horndeski_l2_l4_interval_campaign.json"
)


def _inputs() -> tuple[dict, dict]:
    return json.loads(IR_PATH.read_text()), json.loads(CONFIG_PATH.read_text())


def _reduced_inputs() -> tuple[dict, dict]:
    ir, config = _inputs()
    ir = copy.deepcopy(ir)
    for axis in ir["mutation_space"]["axes"]:
        if axis["coefficient"] in {"c20", "d10"}:
            axis["values"] = ["-1", "0", "1"]
        else:
            axis["values"] = ["0"]
    template = config["background_template"]
    template["tau_end"] = 0.005
    template["step"] = 0.0025
    return ir, config


def test_campaign_classifies_and_certifies_every_eligible_assignment() -> None:
    ir, config = _reduced_inputs()
    result = run_flrw_background_campaign(ir, config)
    manifest = result["manifest"]
    assert manifest["status"] == (
        "pass_all_generalized_harmonic_candidates_interval_certified"
    )
    assert manifest["counts"] == {
        "total": 9,
        "generalized_harmonic_eligible": 3,
        "interval_certified": 3,
        "cubic_G3_only_flrw_screened": 5,
        "cubic_G3_only_flrw_rejected": 1,
        "modified_harmonic_unresolved": 6,
        "modified_harmonic_not_background_screened": 0,
        "rejected": 0,
        "obstruction_classes": {
            "generalized_harmonic_kessence": 3,
            "modified_harmonic_G3_only": 6,
        },
    }
    assert len(result["certificates"]) == 8
    for certificate in result["certificates"].values():
        assert certificate["status"] == "pass_interval_certified"
    assert sum(
        certificate["formulation_certificate"]["nonlinear_scalar_energy_certificate"]
        == "pass_positive_homogeneous_energy_and_legendre_margin"
        for certificate in result["certificates"].values()
    ) == 3
    cubic_certificates = [
        certificate
        for certificate in result["certificates"].values()
        if certificate["formulation_certificate"]["proof_route"]
        == "cubic_horndeski_bssn_weak_field"
    ]
    assert len(cubic_certificates) == 5
    for certificate in cubic_certificates:
        diagnostic = certificate["formulation_certificate"][
            "cubic_bssn_homogeneous_diagnostic"
        ]
        assert len(diagnostic["derivative_ratio_upper_bounds"]) == 15
        assert diagnostic["scalar_slicing_cone_gap_min_abs"] > 0
    ranking = manifest["cubic_G3_only_diagnostic_ranking"]
    assert len(ranking) == 5
    assert [item["maximum_derivative_ratio"] for item in ranking] == sorted(
        item["maximum_derivative_ratio"] for item in ranking
    )


def test_campaign_cardinality_limit_rejects_before_enumeration() -> None:
    ir, config = _reduced_inputs()
    config["maximum_assignments"] = 8
    result = run_flrw_background_campaign(ir, config)
    manifest = result["manifest"]
    assert manifest["status"] == "reject"
    assert manifest["counts"]["total"] == 0
    assert "exceeds campaign maximum" in " ".join(manifest["errors"])
