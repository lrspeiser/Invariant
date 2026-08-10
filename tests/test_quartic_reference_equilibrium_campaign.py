import hashlib
import json
from pathlib import Path

from sigma_theory_compiler.quartic_reference_equilibrium_campaign import (
    generic_reference_equilibrium_control,
    run_quartic_reference_equilibrium_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "physics-language"
PATHS = (
    RUNS / "quartic-finite-low-operator-campaign" / "campaign.json",
    RUNS / "quartic-nonlinear-evolution-campaign" / "campaign.json",
    RUNS / "quartic-solved-source-moser-campaign" / "campaign.json",
    RUNS / "quartic-quasilinear-moser-campaign" / "campaign.json",
    RUNS / "quartic-nonquasilinear-pde-campaign" / "campaign.json",
    RUNS / "quartic-coordinate-jet-tube-campaign" / "campaign.json",
    RUNS / "quartic-euler-remainder-majorant-campaign" / "campaign.json",
    RUNS / "horndeski-l2-l4-interval-campaign" / "campaign.json",
)
CONFIG = ROOT / "configs" / "backgrounds" / "quartic_reference_equilibrium_campaign.json"
ARTIFACT = RUNS / "quartic-reference-equilibrium-campaign" / "campaign.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _inputs() -> list[dict]:
    return [_load(path) for path in PATHS]


def _rehash(campaign: dict) -> None:
    body = {key: value for key, value in campaign.items() if key != "content_sha256"}
    campaign["content_sha256"] = hashlib.sha256(
        json.dumps(
            body, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode()
    ).hexdigest()


def test_exact_reference_residuals_and_negative_controls() -> None:
    passed, control = generic_reference_equilibrium_control()
    assert passed
    exact = control["exact_Euler_evaluation"]
    assert exact["acceleration_independent_residuals"] == ["0"] * 11
    assert exact["affine_residual_zero"]
    assert exact["time_block_determinant"] == "6561*M2**10/4096"
    assert exact["time_block_rank"] == 11
    assert all(item["rejected"] for item in control["negative_controls"].values())


def test_all_candidates_have_equilibrium_and_background_subtracted_L2_source() -> None:
    result = run_quartic_reference_equilibrium_campaign(*_inputs(), _load(CONFIG))
    assert result["status"] == (
        "pass_all_12_exact_reference_equilibria_and_L2_source_conventions"
    )
    assert result["counts"] == {
        "selected": 12,
        "exact_reference_equilibria_passed": 12,
        "localized_L2_source_conventions_passed": 12,
        "matched_FLRW_candidates_unresolved": 12,
        "global_H7_sums_applied": 0,
        "lifespans_proved": 0,
        "rejected": 0,
    }
    assert all(
        item["reference_equilibrium"]["F_reference_equals_zero"]
        and item["localized_whole_space_source"]["certified"]
        and item["localized_whole_space_source"][
            "background_subtracted_source_is_L2"
        ]
        and item["FLRW_audit"]["status"]
        == "unresolved_modified_harmonic_uniform_bound_required"
        and not item["FLRW_audit"]["used_as_reference_background"]
        and not item["global_H7_dyadic_sum_applied"]
        and not item["nonlinear_lifespan_proved"]
        for item in result["certificates"]
    )
    assert result == _load(ARTIFACT)


def test_omitted_corrupt_provenance_and_false_convention_reject() -> None:
    inputs = _inputs()
    config = _load(CONFIG)

    omitted = list(inputs)
    omitted[0] = {}
    result = run_quartic_reference_equilibrium_campaign(*omitted, config)
    assert result["status"] == "reject"
    assert "finite_low prerequisite status mismatch" in result["errors"][0]

    false_convention = dict(config)
    false_convention["whole_space_source_convention"] = "raw_constant_source"
    result = run_quartic_reference_equilibrium_campaign(*inputs, false_convention)
    assert result["status"] == "reject"

    corrupt = json.loads(json.dumps(inputs[2]))
    corrupt["upstream_sha256"]["euler_remainder"] = "0" * 64
    _rehash(corrupt)
    inputs[2] = corrupt
    result = run_quartic_reference_equilibrium_campaign(*inputs, config)
    assert result["status"] == "reject"
    assert "finite-low-to-solved provenance mismatch" in result["errors"][0]

    finite_low = json.loads(json.dumps(inputs[0]))
    finite_low["upstream_sha256"]["solved_source"] = corrupt["content_sha256"]
    _rehash(finite_low)
    inputs[0] = finite_low
    result = run_quartic_reference_equilibrium_campaign(*inputs, config)
    assert result["status"] == "reject"
    assert "solved-to-euler provenance mismatch" in result["errors"][0]
