from __future__ import annotations

import json
from pathlib import Path

import pytest
import sympy as sp

from sigma_theory_compiler.system10_cylindrical_r_positive_gravity_scalar_aw_invertibility import (
    DECISION,
    System10GravityScalarAWInvertibilityError,
    _canonical_lf_sha,
    _canonical_sha,
    build_receipt,
    write_receipt,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/system10_cylindrical_r_positive_gravity_scalar_aw_invertibility.json"
RECEIPT = (
    ROOT / "runs/math/system10-cylindrical-r-positive-gravity-scalar-aw-invertibility/receipt.json"
)


@pytest.fixture(scope="module")
def replayed() -> dict[str, object]:
    return build_receipt(CONFIG)


def test_committed_receipt_replays_exactly(replayed: dict[str, object]) -> None:
    assert replayed == json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert replayed["decision"] == DECISION


def test_exact_slice_determinant_and_admissible_singular_witness(
    replayed: dict[str, object],
) -> None:
    specialized = replayed["specialized_slice"]
    witness = replayed["exact_singular_witness"]
    v_10 = sp.Symbol("v_10")
    determinant = sp.sympify(specialized["determinant"])
    assert sp.factor(determinant) == (
        -sp.Rational(6561, 16384) * (v_10**2 + 2) ** 6 * (3 * v_10**2 - 1)
    )
    assert sp.simplify(determinant.subs(v_10, sp.sqrt(3) / 3)) == 0
    assert witness["rank"] == 10
    assert witness["domain_certificate"] == "r=1>0"


def test_null_vectors_independently_replay_against_sealed_matrix(
    replayed: dict[str, object],
) -> None:
    matrix = sp.Matrix(
        [
            [sp.sympify(expression) for expression in row]
            for row in replayed["specialized_slice"]["matrix"]
        ]
    ).subs(sp.Symbol("v_10"), sp.sqrt(3) / 3)
    null = sp.Matrix([0] * 10 + [1])
    assert matrix * null == sp.zeros(11, 1)
    assert matrix.T * null == sp.zeros(11, 1)
    assert matrix.rank() == 10


def test_block_is_representative_only_and_does_not_overclaim(
    replayed: dict[str, object],
) -> None:
    assert replayed["candidate"]["scope"] == "representative_candidate_only"
    conclusion = replayed["conclusion"]
    assert conclusion["global_invertibility_over_fixed_r_positive_state_domain"] is False
    assert conclusion["unique_global_acceleration_solve"] is False
    assert conclusion["all_11_accelerations_solved"] is False
    assert conclusion["residual_replay_after_solve"] is False
    assert "other eleven candidates" in " ".join(replayed["nonclaims"])


def test_receipt_and_packet_seals_are_closed(replayed: dict[str, object]) -> None:
    body = {key: value for key, value in replayed.items() if key != "content_sha256"}
    assert replayed["content_sha256"] == _canonical_sha(body)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert (
        replayed["source_bindings"]["ordered_aw_row_set_sha256"]
        == config["aw_packet"]["ordered_row_set_sha256"]
    )
    assert len(config["aw_packet"]["rows"]) == 11


def test_source_evidence_uses_canonical_lf_hashes() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    for binding in config["source_evidence"].values():
        assert _canonical_lf_sha(ROOT / binding["path"]) == binding["canonical_lf_sha256"]


def test_cap_and_domain_tamper_fail_closed(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["caps"]["slice_r"] = "0"
    tampered = tmp_path / "r-zero.json"
    tampered.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(System10GravityScalarAWInvertibilityError, match="caps changed"):
        build_receipt(tampered, root=ROOT)


def test_row_byte_and_content_tamper_fail_closed(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["aw_packet"]["rows"][0]["canonical_lf_sha256"] = "0" * 64
    tampered = tmp_path / "row-byte.json"
    tampered.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(System10GravityScalarAWInvertibilityError, match="row hash mismatch"):
        build_receipt(tampered, root=ROOT)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["aw_packet"]["rows"][0]["content_sha256"] = "0" * 64
    tampered = tmp_path / "row-content.json"
    tampered.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(System10GravityScalarAWInvertibilityError, match="row seal mismatch"):
        build_receipt(tampered, root=ROOT)


def test_write_is_atomic_and_never_overwrites(tmp_path: Path) -> None:
    output = tmp_path / "receipt.json"
    first = write_receipt(CONFIG, output)
    assert json.loads(output.read_text(encoding="utf-8")) == first
    with pytest.raises(System10GravityScalarAWInvertibilityError, match="refusing to overwrite"):
        write_receipt(CONFIG, output)


def test_receipt_tamper_changes_content_seal(replayed: dict[str, object]) -> None:
    tampered = json.loads(json.dumps(replayed))
    tampered["exact_singular_witness"]["rank"] = 11
    body = {key: value for key, value in tampered.items() if key != "content_sha256"}
    assert tampered["content_sha256"] != _canonical_sha(body)
