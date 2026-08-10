import hashlib
import json
from pathlib import Path

from sigma_theory_compiler.quartic_tc2_symmetrizer_no_go_campaign import (
    generic_tc2_symmetrizer_no_go_control,
    run_quartic_tc2_symmetrizer_no_go_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "physics-language"
PATHS = (
    RUNS / "quartic-two-channel-induced-operator-campaign" / "campaign.json",
    RUNS / "quartic-first-order-reduction-campaign" / "campaign.json",
    RUNS / "quartic-nonquasilinear-pde-campaign" / "campaign.json",
    RUNS / "quartic-full-symmetrizer-moser-campaign" / "campaign.json",
)
CONFIG = (
    ROOT
    / "configs"
    / "backgrounds"
    / "quartic_tc2_symmetrizer_no_go_campaign.json"
)
ARTIFACT = RUNS / "quartic-tc2-symmetrizer-no-go-campaign" / "campaign.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _inputs() -> list[dict]:
    return [_load(path) for path in PATHS]


def _rehash(campaign: dict) -> None:
    body = {key: value for key, value in campaign.items() if key != "content_sha256"}
    campaign["content_sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def test_rank_obstruction_adjoint_completion_and_negatives_are_exact() -> None:
    passed, control = generic_tc2_symmetrizer_no_go_control()
    assert passed
    rank = control["rank_obstruction"]
    assert rank["actual_direction_1_rank"] == 2
    assert rank["actual_direction_1_minor_rows_22_32_columns_10_7"] == (
        "4*alpha**2"
    )
    assert control["exact_adjoint_completion"]["control_residual_zero"]
    assert all(item["rejected"] for item in control["negative_controls"].values())


def test_all_candidates_have_bounded_no_go_and_tc2_stays_open() -> None:
    result = run_quartic_tc2_symmetrizer_no_go_campaign(
        *_inputs(), _load(CONFIG)
    )
    assert result["status"] == (
        "pass_all_12_exact_TC2_unchanged_K55_no_gos_"
        "reciprocal_blocks_missing_global_H7_fail_closed"
    )
    assert result["counts"] == {
        "selected": 12,
        "exact_direction_1_rank_obstructions": 12,
        "unchanged_K55_TC2_absorption_no_gos": 12,
        "canonical_reciprocal_blocks_identified": 12,
        "canonical_reciprocal_blocks_derived_from_state": 0,
        "TC2_closures": 0,
        "B7_branches_closed": 0,
        "global_H7_closures": 0,
        "lifespans_proved": 0,
        "rejected": 0,
    }
    first = result["certificates"][0]
    assert first["actual_direction_1_packet"]["rank"] == 2
    assert first["bounded_no_go"][
        "unchanged_K55_separate_TC2_absorption_refuted"
    ]
    assert first["canonical_missing_completion"][
        "makes_K55_times_completed_block_Hermitian"
    ]
    assert not first["canonical_missing_completion"][
        "present_in_current_modified_state"
    ]
    assert not first["connection_to_TC2_B7_global_H7"]["TC2_closed"]
    assert result == _load(ARTIFACT)


def test_packet_tamper_ansatz_change_and_false_promotions_reject() -> None:
    inputs = _inputs()
    config = _load(CONFIG)

    corrupt = json.loads(json.dumps(inputs[0]))
    corrupt["upstream_sha256"]["first_order_P55"] = "0" * 64
    _rehash(corrupt)
    result = run_quartic_tc2_symmetrizer_no_go_campaign(
        corrupt, *inputs[1:], config
    )
    assert result["status"] == "reject"
    assert "upstream provenance mismatch" in result["errors"][0]

    wrong_ansatz = dict(config)
    wrong_ansatz["ansatz_class"] = "coupled_K55_plus_deltaK"
    result = run_quartic_tc2_symmetrizer_no_go_campaign(*inputs, wrong_ansatz)
    assert result["status"] == "reject"
    assert "unsupported TC2 no-go contract" in result["errors"][0]

    for policy in (
        "reciprocal_completion_policy",
        "global_H7_policy",
        "lifespan_policy",
    ):
        false_promotion = dict(config)
        false_promotion[policy] = "pass"
        result = run_quartic_tc2_symmetrizer_no_go_campaign(
            *inputs, false_promotion
        )
        assert result["status"] == "reject"
