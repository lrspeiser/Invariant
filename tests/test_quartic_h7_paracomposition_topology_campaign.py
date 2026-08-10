import hashlib
import json
from pathlib import Path

from sigma_theory_compiler.quartic_h7_paracomposition_topology_campaign import (
    generic_h7_paracomposition_topology_control,
    run_quartic_h7_paracomposition_topology_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "physics-language"
PATHS = (
    RUNS / "quartic-solved-source-c9-extension-campaign" / "campaign.json",
    RUNS / "quartic-full-source-jacobian-arithmetic-campaign" / "campaign.json",
    RUNS / "quartic-dyadic-localization-campaign" / "campaign.json",
    RUNS / "quartic-global-h7-energy-campaign" / "campaign.json",
)
CONFIG = (
    ROOT
    / "configs"
    / "backgrounds"
    / "quartic_h7_paracomposition_topology_campaign.json"
)
ARTIFACT = (
    RUNS / "quartic-h7-paracomposition-topology-campaign" / "campaign.json"
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


def test_atom_topology_partitions_bony_branches_remote_sum_and_negatives() -> None:
    passed, control = generic_h7_paracomposition_topology_control()
    assert passed
    assert control["coordinate_atom_topology"]["count_residual"] == 0
    assert control["coordinate_atom_topology"][
        "combined_coordinate_L2_injection_upper"
    ] == "2"
    ledger = control["seventh_derivative_tame_partition_ledger"]
    assert ledger["all_integer_partitions"] == 15
    assert ledger["nonprincipal_partition_count"] == 14
    assert ledger["Faa_di_Bruno_multiplicity_sum"] == 876
    assert ledger["all_nonprincipal_topologies_compatible"]
    assert control["Bony_branch_partition"]["branch_total_residual"] == 0
    assert control["remote_resonant_shell_summation"][
        "shell_index_summation_constant_instantiated"
    ]
    assert all(item["rejected"] for item in control["negative_controls"].values())


def test_all_candidates_get_tame_ledgers_but_high_low_branch_stays_open() -> None:
    result = run_quartic_h7_paracomposition_topology_campaign(
        *_inputs(), _load(CONFIG)
    )
    assert result["status"] == (
        "pass_all_12_H7_atom_topologies_and_recombined_tame_ledgers_"
        "high_low_paraproduct_fail_closed"
    )
    assert result["counts"] == {
        "selected": 12,
        "coordinate_topologies_instantiated": 12,
        "C9_vector_recombined_tame_ledgers_instantiated": 12,
        "principal_low_high_good_unknown_branches_closed": 12,
        "remote_shell_index_summations_instantiated": 12,
        "coefficient_high_state_low_branches_closed": 0,
        "resonant_Fourier_operator_constants_instantiated": 0,
        "complete_paracomposition_remainders_closed": 0,
        "global_H7_closures": 0,
        "lifespans_proved": 0,
        "rejected": 0,
    }
    first = result["certificates"][0]
    assert first["Bony_branches"]["coefficient_low_state_high"][
        "entry_residuals_zero"
    ] == 3025
    assert first["Bony_branches"]["coefficient_high_state_low"][
        "status"
    ] == "fail_closed"
    assert first["Bony_branches"]["balanced_resonant"]["status"] == (
        "shell_index_sum_only_operator_constant_fail_closed"
    )
    assert first["C9_vector_recombined_tame_constant_instantiated"]
    assert not first["complete_paracomposition_remainder_closed"]
    assert not first["global_H7_differential_inequality_closed"]
    assert result == _load(ARTIFACT)


def test_provenance_topology_and_false_branch_promotions_reject() -> None:
    inputs = _inputs()
    config = _load(CONFIG)

    corrupt = json.loads(json.dumps(inputs[3]))
    corrupt["upstream_sha256"]["dyadic"] = "0" * 64
    _rehash(corrupt)
    result = run_quartic_h7_paracomposition_topology_campaign(
        *inputs[:3], corrupt, config
    )
    assert result["status"] == "reject"
    assert "provenance mismatch" in result["errors"][0]

    wrong_topology = dict(config)
    wrong_topology["second_atom_sobolev_order"] = 7
    result = run_quartic_h7_paracomposition_topology_campaign(*inputs, wrong_topology)
    assert result["status"] == "reject"
    assert "unsupported H7 paracomposition topology contract" in result["errors"][0]

    for policy in (
        "high_low_branch_policy",
        "resonant_operator_policy",
        "global_H7_policy",
        "lifespan_policy",
    ):
        false_promotion = dict(config)
        false_promotion[policy] = "pass"
        result = run_quartic_h7_paracomposition_topology_campaign(
            *inputs, false_promotion
        )
        assert result["status"] == "reject"
