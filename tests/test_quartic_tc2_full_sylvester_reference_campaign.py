import hashlib
import json
from pathlib import Path

from sigma_theory_compiler.quartic_tc2_full_sylvester_reference_campaign import (
    generic_tc2_full_sylvester_reference_control,
    run_quartic_tc2_full_sylvester_reference_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "physics-language"
PATHS = (
    RUNS / "quartic-tc2-minimal-coupled-remedy-no-go-campaign" / "campaign.json",
    RUNS / "quartic-tc2-symmetrizer-no-go-campaign" / "campaign.json",
    RUNS / "quartic-two-channel-induced-operator-campaign" / "campaign.json",
    RUNS / "quartic-full-symmetrizer-moser-campaign" / "campaign.json",
)
CONFIG = (
    ROOT
    / "configs"
    / "backgrounds"
    / "quartic_tc2_full_sylvester_reference_campaign.json"
)
ARTIFACT = (
    RUNS / "quartic-tc2-full-sylvester-reference-campaign" / "campaign.json"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _inputs() -> list[dict]:
    return [_load(path) for path in PATHS]


def _rehash(campaign: dict) -> None:
    body = {key: value for key, value in campaign.items() if key != "content_sha256"}
    campaign["content_sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def test_eigenbasis_formula_gap_and_negatives_are_exact() -> None:
    passed, control = generic_tc2_full_sylvester_reference_control()
    assert passed
    assert control["eigenbasis_formula"]["generic_off_diagonal_residual"] == "0"
    assert control["spectral_gap"] == {"minimum": "1/6", "inverse_upper": "6"}
    assert all(item["rejected"] for item in control["negative_controls"].values())


def test_reference_solution_closes_but_variable_tc2_stays_open() -> None:
    result = run_quartic_tc2_full_sylvester_reference_campaign(
        *_inputs(), _load(CONFIG)
    )
    assert result["status"] == (
        "pass_all_12_full_reference_TC2_Sylvester_solutions_"
        "variable_extension_global_H7_fail_closed"
    )
    packet = result["common_full_reference_Sylvester_packet"]
    assert packet["all_eigenspace_diagonal_solvability_blocks_zero"]
    assert packet["deltaK_column10_Frobenius_square"] == "1253060/9"
    assert packet["TC2_columns"][2]["deltaK_nonzero_entries"] == 24
    assert result["counts"] == {
        "selected": 12,
        "full_reference_Sylvester_solutions": 12,
        "Hermitian_deltaK_solutions": 12,
        "reference_positivity_radii": 12,
        "reference_CK3_low_factor_bounds": 12,
        "variable_coefficient_solvability_proofs": 0,
        "CK1_closures": 0,
        "TC2_closures": 0,
        "B7_branches_closed": 0,
        "global_H7_closures": 0,
        "lifespans_proved": 0,
        "rejected": 0,
    }
    first = result["certificates"][0]
    assert first["full_reference_Sylvester_solution"][
        "exact_Sylvester_residual_zero"
    ]
    assert first["positivity_smallness"]["closed_at_reference"]
    assert first["CK3_spatial_derivative_cost"][
        "reference_low_factor_bound_closed"
    ]
    assert not first["first_variable_coefficient_obstruction"]["closed"]
    assert not first["connection_to_TC2_B7_global_H7"]["TC2_closed"]
    assert result == _load(ARTIFACT)


def test_hash_tamper_variable_promotion_and_wrong_class_reject() -> None:
    inputs = _inputs()
    config = _load(CONFIG)

    corrupt = json.loads(json.dumps(inputs[0]))
    corrupt["upstream_sha256"]["TC2_symmetrizer_no_go"] = "0" * 64
    _rehash(corrupt)
    result = run_quartic_tc2_full_sylvester_reference_campaign(
        corrupt, *inputs[1:], config
    )
    assert result["status"] == "reject"
    assert "upstream provenance mismatch" in result["errors"][0]

    wrong_class = dict(config)
    wrong_class["deltaK_class"] = "spectral_commuting"
    result = run_quartic_tc2_full_sylvester_reference_campaign(
        *inputs, wrong_class
    )
    assert result["status"] == "reject"
    assert "unsupported full Sylvester contract" in result["errors"][0]

    for policy in (
        "variable_extension_policy",
        "global_H7_policy",
        "lifespan_policy",
    ):
        false_promotion = dict(config)
        false_promotion[policy] = "pass"
        result = run_quartic_tc2_full_sylvester_reference_campaign(
            *inputs, false_promotion
        )
        assert result["status"] == "reject"
