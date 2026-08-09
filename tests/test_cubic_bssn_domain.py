from __future__ import annotations

import copy
import json
from pathlib import Path

from sigma_theory_compiler.cubic_bssn_domain import (
    certify_cubic_bssn_domain,
    generic_cubic_scalar_effective_metric_control,
    run_cubic_bssn_domain_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
IR_PATH = ROOT / "runs/physics-language/horndeski-l2-l4-polynomial-ir.json"
FLRW_CAMPAIGN_PATH = (
    ROOT / "runs/physics-language/horndeski-l2-l4-interval-campaign/campaign.json"
)
CONFIG_PATH = ROOT / "configs/backgrounds/cubic_bssn_uniform_domain_campaign.json"
ARTIFACT_PATH = (
    ROOT / "runs/physics-language/cubic-bssn-uniform-domain-campaign/campaign.json"
)


def _inputs() -> tuple[dict, dict, dict, dict[str, dict]]:
    ir = json.loads(IR_PATH.read_text())
    campaign = json.loads(FLRW_CAMPAIGN_PATH.read_text())
    config = json.loads(CONFIG_PATH.read_text())
    certificates = {
        record["candidate_id"]: json.loads(
            (FLRW_CAMPAIGN_PATH.parent / record["certificate"]).read_text()
        )
        for record in campaign["candidates"]
        if record.get("status")
        == "pass_flrw_interval_cubic_weak_field_bounds_unresolved"
    }
    return ir, campaign, config, certificates


def test_generic_cubic_trace_reversed_effective_metric_is_exact() -> None:
    passed, result = generic_cubic_scalar_effective_metric_control()
    assert passed, result
    assert result["source_contraction_residual"] == "0"
    assert result["closed_form_residual"] == "0"
    assert all(residual == "0" for residual in result["symmetry_residuals"])
    assert result["negative_control"] == {
        "corruption": "omit trace reversal in metric-equation substitution",
        "exact_witness_residual": "27/4",
        "rejected": True,
    }


def test_all_screened_cubic_candidates_have_uniform_local_jet_boxes() -> None:
    ir, campaign, config, certificates = _inputs()
    result = run_cubic_bssn_domain_campaign(ir, campaign, certificates, config)
    manifest = result["manifest"]
    artifact = json.loads(ARTIFACT_PATH.read_text())
    assert manifest["status"] == (
        "pass_all_screened_cubic_candidates_have_uniform_local_jet_boxes"
    )
    assert manifest["counts"] == {
        "screened_cubic_candidates": 5,
        "uniform_domain_certified": 5,
        "rejected": 0,
    }
    assert manifest["content_sha256"] == artifact["content_sha256"]
    assert len(result["certificates"]) == 5
    assert all(
        record["certified_hessian_component_radius_lower"] > 0
        and record["spatial_block_eigenvalue_lower"] > 0
        and record["slicing_cone_polynomial_upper"] < 0
        for record in manifest["ranking"]
    )


def test_cubic_domain_rejects_bad_gauge_and_non_timelike_gradient_box() -> None:
    ir, _campaign, config, certificates = _inputs()
    trajectory = next(iter(certificates.values()))
    bad_sigma = copy.deepcopy(config["domain_template"])
    bad_sigma["slicing_parameter_sigma"] = 0.5
    sigma_result = certify_cubic_bssn_domain(ir, trajectory, bad_sigma)
    assert sigma_result["status"] == "reject"
    assert "sigma>1/2" in " ".join(sigma_result["errors"])

    non_timelike = copy.deepcopy(config["domain_template"])
    non_timelike["domain_extension"]["spatial_gradient_abs"] = 1.0
    gradient_result = certify_cubic_bssn_domain(ir, trajectory, non_timelike)
    assert gradient_result["status"] == "reject"
    assert "non-timelike" in " ".join(gradient_result["errors"])


def test_first_failing_hessian_radius_is_rejected() -> None:
    ir, _campaign, config, certificates = _inputs()
    artifact = json.loads(ARTIFACT_PATH.read_text())
    candidate = artifact["ranking"][0]
    trajectory = certificates[candidate["candidate_id"]]
    failing = copy.deepcopy(config["domain_template"])
    failing["domain_extension"]["hessian_component_abs"] = candidate[
        "first_failing_hessian_component_radius_upper"
    ]
    result = certify_cubic_bssn_domain(ir, trajectory, failing)
    assert result["status"] == "reject"
    assert "cone" in " ".join(result["errors"])
