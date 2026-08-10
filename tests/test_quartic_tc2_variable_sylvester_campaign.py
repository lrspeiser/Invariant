import hashlib
import json
from pathlib import Path

from sigma_theory_compiler.quartic_tc2_variable_sylvester_campaign import (
    _coordinate_atom_to_jet_packet,
    _linearized_einstein_upper,
    _reference_and_first_jet_packet,
    generic_tc2_variable_sylvester_control,
    run_quartic_tc2_variable_sylvester_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "physics-language"
PATHS = (
    RUNS / "quartic-tc2-full-sylvester-reference-campaign" / "campaign.json",
    RUNS / "quartic-two-channel-induced-operator-campaign" / "campaign.json",
    RUNS / "quartic-full-symmetrizer-moser-campaign" / "campaign.json",
    RUNS / "quartic-component-jacobian-contract-campaign" / "campaign.json",
)
CONFIG = (
    ROOT
    / "configs"
    / "backgrounds"
    / "quartic_tc2_variable_sylvester_campaign.json"
)
ARTIFACT = (
    RUNS / "quartic-tc2-variable-sylvester-campaign" / "campaign.json"
)


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


def test_exact_first_derivative_identity_and_negative_controls() -> None:
    passed, control = generic_tc2_variable_sylvester_control()
    assert passed
    assert control["first_derivative_residual"] == "0"
    assert all(item["rejected"] for item in control["negative_controls"].values())

    reference = _reference_and_first_jet_packet()["packet"]
    assert len(reference["jet_derivative_records"]) == 24
    assert all(
        item["diagonal_compressions_zero"]
        and item["deltaK_prime_Hermitian"]
        and item["first_order_Sylvester_residual_zero"]
        for item in reference["jet_derivative_records"]
    )
    assert any(
        item["omit_deltaK0_Pprime_full_residual_nonzero_entries"] > 0
        for item in reference["jet_derivative_records"]
    )


def test_coordinate_153_to_24_map_has_exact_flat_einstein_witnesses() -> None:
    coordinate = _coordinate_atom_to_jet_packet()["packet"]
    assert len(coordinate["coordinate_atom_basis"]) == 153
    assert coordinate["active_atom_count"] == 73
    assert coordinate["zero_atom_count"] == 80
    assert coordinate["nonzero_scalar_count"] == 88
    assert {
        key: str(value)
        for key, value in _linearized_einstein_upper((1, 1), 0).items()
    } == {"G_22": "-1/2", "G_33": "-1/2"}
    assert {
        key: str(value)
        for key, value in _linearized_einstein_upper((1, 2), 5).items()
    } == {"G_00": "sqrt(2)/2", "G_33": "-sqrt(2)/2"}


def test_all_153_first_order_extensions_are_constructed_but_h7_stays_open() -> None:
    result = run_quartic_tc2_variable_sylvester_campaign(
        *_inputs(), _load(CONFIG)
    )
    assert result["status"] == (
        "pass_all_12_first_order_variable_deltaK_extensions_"
        "higher_orders_global_H7_fail_closed"
    )
    assert result["counts"] == {
        "selected": 12,
        "coordinate_atoms_audited": 153,
        "solvable_first_derivative_atoms": 153,
        "obstructed_first_derivative_atoms": 0,
        "first_order_variable_extensions": 12,
        "first_order_no_gos": 0,
        "TC2_closures": 0,
        "global_H7_closures": 0,
        "lifespans_proved": 0,
    }
    packet = result["common_variable_solvability_packet"]
    assert packet["counts"]["constructed_deltaK_first_derivatives"] == 153
    assert all(
        item["all_equal_eigenspace_compressions_zero"]
        and item["deltaK_prime_Hermitian"]
        and item["first_order_Sylvester_residual_zero"]
        for item in packet["coordinate_atom_records"]
    )
    assert all(
        item["rejected"] for item in packet["exact_negative_controls"].values()
    )
    first = result["certificates"][0]
    assert first["first_order_variable_extension"]["closed"]
    assert first["affine_deltaK_positivity"][
        "closed_for_affine_first_order_extension"
    ]
    assert not first["connection_to_TC2_B7_global_H7"]["TC2_closed"]
    assert result == _load(ARTIFACT)


def test_hash_tamper_atom_omission_and_false_h7_promotion_reject() -> None:
    inputs = _inputs()
    config = _load(CONFIG)

    corrupt = json.loads(json.dumps(inputs[0]))
    corrupt["upstream_sha256"]["induced_operator"] = "0" * 64
    _rehash(corrupt)
    result = run_quartic_tc2_variable_sylvester_campaign(
        corrupt, *inputs[1:], config
    )
    assert result["status"] == "reject"
    assert "upstream provenance mismatch" in result["errors"][0]

    omitted_atom = dict(config)
    omitted_atom["coordinate_atom_dimension"] = 152
    result = run_quartic_tc2_variable_sylvester_campaign(
        *inputs, omitted_atom
    )
    assert result["status"] == "reject"
    assert "unsupported variable Sylvester contract" in result["errors"][0]

    for policy in ("global_H7_policy", "lifespan_policy"):
        promoted = dict(config)
        promoted[policy] = "pass"
        result = run_quartic_tc2_variable_sylvester_campaign(
            *inputs, promoted
        )
        assert result["status"] == "reject"
