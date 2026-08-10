from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import sympy as sp

from sigma_theory_compiler.cubic_bssn_domain import cubic_scalar_effective_metric
from sigma_theory_compiler.g3_seed_weak_cell_formal_audit import (
    _sha,
    _validate_target,
    build_g3_seed_weak_cell_formal_audit,
)
from sigma_theory_compiler.promotion_orchestrator import ELIGIBILITY

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "g3_seed_weak_cell_formal_audit.json"
ARTIFACT_PATH = ROOT / "runs" / "engine" / "g3-seed-weak-cell-formal-audit.json"


@pytest.fixture(scope="module")
def rebuilt() -> dict:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return build_g3_seed_weak_cell_formal_audit(config, ROOT)


def test_artifact_is_exact_rebuild(rebuilt: dict) -> None:
    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    assert artifact == rebuilt
    body = {key: value for key, value in artifact.items() if key != "content_sha256"}
    assert artifact["content_sha256"] == _sha(body)
    assert hashlib.sha256(ARTIFACT_PATH.read_bytes()).hexdigest() == (
        "7822ee3013b903669005746d51d6c198f4774244f5f8444603e645dca67a1145"
    )


def test_exact_center_matches_source_effective_metric(rebuilt: dict) -> None:
    zero = sp.Integer(0)
    matrix = cubic_scalar_effective_metric(
        x=sp.Rational(1, 2),
        gradient_covariant=[sp.Integer(1), zero, zero, zero],
        hessian_covariant=[[zero for _ in range(4)] for _ in range(4)],
        g2_x=zero,
        g2_xx=zero,
        g3_phi=zero,
        g3_x_phi=zero,
        g3_x=-sp.Rational(1, 100),
        g3_xx=zero,
    )
    assert matrix == sp.diag(
        -sp.Rational(40003, 40000),
        sp.Rational(39999, 40000),
        sp.Rational(39999, 40000),
        sp.Rational(39999, 40000),
    ).tolist()
    center = rebuilt["candidate_records"][0]["candidate_certificate"]["center_principal_calibration"]
    assert center["effective_P00"] == "-40003/40000"
    assert center["effective_spatial_eigenvalue"] == "39999/40000"
    assert center["scalar_speed_squared"] == "39999/40003"
    assert center["scalar_slicing_cone_gap_squared"] == "40007/40003"
    assert center["status"] == "pass_at_center_only"


def test_uniform_weak_cell_and_dirac_remain_exactly_blocked(rebuilt: dict) -> None:
    record = rebuilt["candidate_records"][0]
    certificate = record["candidate_certificate"]
    weak = certificate["declared_weak_cell_audit"]
    assert weak["status"] == "blocked"
    assert weak["componentwise_gradient_bounds"] is None
    assert weak["componentwise_hessian_bounds"] is None
    assert weak["curvature_bounds"] is None
    assert weak["source_numeric_threshold_for_much_less_than_one"] is None
    assert certificate["adm_primary"]["status"] == "pass"
    assert certificate["distributed_dirac"]["status"] == "blocked"
    assert record["decision"] == "blocked"
    assert record["necessary_condition_rejection_found"] is False
    assert record["first_missing_adm_dirac_premise"] == (
        "candidate_specific_Delta_N_boundary_operator"
    )
    assert record["first_missing_uniform_principal_premise"] == (
        "componentwise_normalized_local_jet_box"
    )


def test_adapter_replays_and_eligibility_are_sealed(rebuilt: dict) -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert rebuilt["invoked_adapter_entrypoints"] == [
        item["entrypoint"] for item in config["formal_adapters"]
    ]
    assert rebuilt["decision_counts"] == {"blocked": 1}
    assert rebuilt["formal_pass_count"] == 0
    assert rebuilt["solar_bundle_count"] == 0
    assert rebuilt["observational_data_opened"] is False
    assert rebuilt["paid_llm_spend_usd"] == 0.0
    assert rebuilt["data_eligibility"] == ELIGIBILITY


def test_tampered_action_binding_is_rejected() -> None:
    predecessor = json.loads(
        (ROOT / "runs" / "engine" / "covariant-grammar-v3-seed-compilation-campaign.json").read_text(
            encoding="utf-8"
        )
    )
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    target = copy.deepcopy(config["target_seed"])
    record = next(item for item in predecessor["candidate_records"] if item["seed_id"] == target["seed_id"])
    target["action_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="action hash mismatch"):
        _validate_target(record, target)
