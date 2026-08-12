import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_annular_k55_c6_campaign import (
    generic_annular_k55_c6_control,
    run_quartic_annular_k55_c6_campaign,
    validate_quartic_annular_k55_c6_artifact,
)

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "physics-language"
PATHS = {
    "symmetrizer": RUNS / "quartic-symmetrizer-uniform-domain-campaign" / "campaign.json",
    "moser": RUNS / "quartic-quasilinear-moser-campaign" / "campaign.json",
    "pde": RUNS / "quartic-nonquasilinear-pde-campaign" / "campaign.json",
    "tube": RUNS / "quartic-coordinate-jet-tube-campaign" / "campaign.json",
    "solved": RUNS / "quartic-solved-source-moser-campaign" / "campaign.json",
    "full": RUNS / "quartic-full-symmetrizer-moser-campaign" / "campaign.json",
    "symbol_c4": RUNS / "quartic-symmetrizer-symbol-moser-campaign" / "campaign.json",
    "r3": RUNS / "quartic-r3-sobolev-calculus-campaign" / "campaign.json",
    "anti_wick": RUNS / "quartic-anti-wick-composition-campaign" / "campaign.json",
}
CONFIG = ROOT / "configs" / "backgrounds" / "quartic_annular_k55_c6_campaign.json"
ARTIFACT = RUNS / "quartic-annular-k55-c6-campaign" / "campaign.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _inputs() -> list[dict]:
    return [_load(path) for path in PATHS.values()]


def test_c6_inverse_normalization_cutoff_and_spatial_controls_are_exact() -> None:
    passed, control = generic_annular_k55_c6_control()
    assert passed
    assert set(control["bivariate_inverse_residuals"].values()) == {"0"}
    assert set(control["homogeneous_Bell_residuals"].values()) == {"0"}
    assert control["chi6_derivative_residual"] == "0"
    assert set(control["chi6_endpoint_residuals"]) == {"0"}
    assert control["coordinate_second_chain_residual"] == "0"
    assert control["spatial_second_chain_residual"] == "0"
    assert control["negative_control"]["rejected"]


def test_all_candidates_receive_targeted_c6_bounds_and_principal_constants() -> None:
    result = run_quartic_annular_k55_c6_campaign(*_inputs(), _load(CONFIG))
    assert result["status"] == (
        "pass_all_12_targeted_annular_K55_C6_principal_composition_constants"
    )
    assert result["counts"] == {
        "selected": 12,
        "targeted_C6_bounds_passed": 12,
        "principal_composition_constants_instantiated": 12,
        "full_dyadic_energies_closed": 0,
        "rejected": 0,
    }
    assert all(
        set(item["C4_reproduction_residuals"].values()) == {"0"}
        and item["anti_wick_principal_composition_remainder_instantiated"]
        and not item["full_dyadic_energy_closed"]
        and item["principal_anti_wick_composition_constant"]["numeric"] > 0
        and set(item["required_spatial_frequency_K55_bounds"])
        == {
            "2,4",
            "0,6",
            "0,5",
            "1,4",
        }
        for item in result["certificates"]
    )
    assert result == _load(ARTIFACT)


def test_wrong_order_and_self_consistently_corrupt_provenance_reject() -> None:
    inputs = _inputs()
    config = _load(CONFIG)
    wrong = dict(config)
    wrong["maximum_total_derivative_order"] = 4
    result = run_quartic_annular_k55_c6_campaign(*inputs, wrong)
    assert result["status"] == "reject"
    assert "targeted C6 contract" in result["errors"][0]

    corrupt = json.loads(json.dumps(inputs[5]))
    corrupt["upstream_sha256"]["solved_source"] = "0" * 64
    body = {key: value for key, value in corrupt.items() if key != "content_sha256"}
    corrupt["content_sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()
    inputs[5] = corrupt
    result = run_quartic_annular_k55_c6_campaign(*inputs, config)
    assert result["status"] == "reject"
    assert "not the registered artifact" in result["errors"][0]


def test_public_validator_rejects_resealed_false_dyadic_closure() -> None:
    artifact, config = _load(ARTIFACT), _load(CONFIG)
    validate_quartic_annular_k55_c6_artifact(artifact, ROOT, config)
    promoted = json.loads(json.dumps(artifact))
    promoted["counts"]["full_dyadic_energies_closed"] = 12
    body = {key: value for key, value in promoted.items() if key != "content_sha256"}
    promoted["content_sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()
    with pytest.raises(ValueError, match="deterministic reconstruction"):
        validate_quartic_annular_k55_c6_artifact(promoted, ROOT, config)
