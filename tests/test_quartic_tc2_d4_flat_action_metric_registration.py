from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_tc2_d4_flat_action_metric_registration import (
    FlatActionMetricRegistrationError,
    _content_hash,
    build_campaign,
    construct_flat_action_metric,
    validate_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/backgrounds/quartic_tc2_d4_flat_action_metric_registration.json"
ARTIFACT = ROOT / (
    "runs/physics-language/quartic-tc2-d4-flat-action-metric-registration/campaign.json"
)


@pytest.fixture(scope="module")
def artifact() -> dict:
    document = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    validate_campaign(document, ROOT)
    return document


def test_exact_action_metric_is_constructed_without_full_symbol_build(artifact: dict) -> None:
    exact = artifact["exact_construction"]
    assert exact["A_0"]["shape"] == [11, 11]
    assert exact["B_0"]["shape"] == [11, 11]
    assert exact["h_plus_0"]["shape"] == [22, 22]
    assert exact["symmetry_residual_zero"] is True
    assert artifact["counts"]["full_symbol_build_calls"] == 0
    assert artifact["claims"]["cold_full_symbol_build_used"] is False


def test_sparse_packets_replay_deterministically(artifact: dict) -> None:
    exact = construct_flat_action_metric()
    for name in ("A_0", "B_0", "h_plus_0"):
        assert exact[name] == artifact["exact_construction"][name]
        assert exact[name]["content_sha256"] == _content_hash(exact[name])
        assert exact[name]["nonzero_count"] > 0
    assert len(exact["evaluation_packets"]) == 3


def test_h_plus_block_identity_and_symmetry(artifact: dict) -> None:
    exact = artifact["exact_construction"]
    a_entries = {(row["row"], row["column"]): row["value"] for row in exact["A_0"]["entries"]}
    b_entries = {(row["row"], row["column"]): row["value"] for row in exact["B_0"]["entries"]}
    h_entries = {(row["row"], row["column"]): row["value"] for row in exact["h_plus_0"]["entries"]}
    assert all(h_entries[(row, column + 11)] == value for (row, column), value in a_entries.items())
    assert all(h_entries[(row + 11, column)] == value for (row, column), value in a_entries.items())
    assert all(h_entries[(row, column)] == value for (row, column), value in b_entries.items())
    assert all(h_entries[(column, row)] == value for (row, column), value in h_entries.items())


def test_bounded_constructor_matches_committed_sympy_formula(artifact: dict) -> None:
    import sympy as sp

    from sigma_theory_compiler.horndeski_principal import (
        _first_order_generalized_pencil,
        _metric_action_block,
        _symmetric_basis,
    )

    xi = sp.Matrix(sp.symbols("xi_0:4", real=True))
    inverse = sp.diag(-1, 1, 1, 1)
    baseline, correction, _ = _metric_action_block(
        inverse_metric=inverse,
        xi_lower=xi,
        gradient_lower=sp.zeros(4, 1),
        alpha=sp.Integer(0),
        m2=sp.Integer(1),
        basis=_symmetric_basis(),
    )
    assert correction.is_zero_matrix
    action = sp.zeros(11)
    action[:10, :10] = baseline
    action[10, 10] = -(xi.T * inverse * xi)[0]
    pencil = _first_order_generalized_pencil(action, xi[0])
    substitutions = {xi[1]: 1, xi[2]: 0, xi[3]: 0}
    expected_a = pencil["A"].subs(substitutions)
    expected_b = pencil["B"].subs(substitutions)

    def from_packet(packet: dict) -> sp.Matrix:
        matrix = sp.zeros(*packet["shape"])
        for entry in packet["entries"]:
            matrix[entry["row"], entry["column"]] = sp.sympify(entry["value"])
        return matrix

    assert from_packet(artifact["exact_construction"]["A_0"]) == expected_a
    assert from_packet(artifact["exact_construction"]["B_0"]) == expected_b
    expected_h = expected_b.row_join(expected_a).col_join(expected_a.row_join(sp.zeros(11)))
    assert from_packet(artifact["exact_construction"]["h_plus_0"]) == expected_h


def test_manifest_stays_at_34_until_K0_is_constructed(artifact: dict) -> None:
    boundary = artifact["manifest_boundary"]
    assert boundary["registered_symbolic_input_packets"] == 34
    assert boundary["missing_symbolic_input_packets"] == 270
    assert boundary["manifest_advanced"] is False
    assert artifact["claims"]["K55_Taylor_order_zero_packets_registered"] is False
    assert artifact["next_exact_gate"]["cold_symbol_build_required"] is False


def test_replay_and_semantic_tamper_fail_closed(artifact: dict) -> None:
    assert build_campaign(ROOT, CONFIG) == artifact
    assert artifact["content_sha256"] == _content_hash(artifact)
    assert all(value == {"rejected": True} for value in artifact["negative_controls"].values())
    tampered = copy.deepcopy(artifact)
    tampered["counts"]["full_symbol_build_calls"] = 1
    tampered["content_sha256"] = _content_hash(tampered)
    with pytest.raises(FlatActionMetricRegistrationError, match="campaign replay mismatch"):
        validate_campaign(tampered, ROOT)
