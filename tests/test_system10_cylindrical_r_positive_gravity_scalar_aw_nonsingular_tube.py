from __future__ import annotations

import json
from pathlib import Path

import pytest
import sympy as sp

from sigma_theory_compiler.system10_cylindrical_r_positive_gravity_scalar_aw_invertibility import (
    _load_matrix,
)
from sigma_theory_compiler.system10_cylindrical_r_positive_gravity_scalar_aw_materializer import (
    _canonical_lf_sha,
    _canonical_sha,
)
from sigma_theory_compiler.system10_cylindrical_r_positive_gravity_scalar_aw_nonsingular_tube import (
    DECISION,
    System10GravityScalarAWNonsingularTubeError,
    build_receipt,
    write_receipt,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/system10_cylindrical_r_positive_gravity_scalar_aw_nonsingular_tube.json"
RECEIPT = (
    ROOT
    / "runs/math/system10-cylindrical-r-positive-gravity-scalar-aw-nonsingular-tube/receipt.json"
)


@pytest.fixture(scope="module")
def replayed() -> dict[str, object]:
    return build_receipt(CONFIG)


def test_committed_tube_solve_replays_exactly(replayed: dict[str, object]) -> None:
    assert replayed == json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert replayed["decision"] == DECISION


def test_preregistered_tube_has_exact_positive_determinant_margin(
    replayed: dict[str, object],
) -> None:
    certificate = replayed["invertibility_certificate"]
    x = sp.Symbol("x", real=True)
    absolute_determinant = sp.sympify(certificate["absolute_determinant_as_x"], locals={"x": x})
    assert sp.factor(sp.diff(absolute_determinant, x)) == (
        -sp.Rational(137781, 16384) * x * (x + 2) ** 5
    )
    assert absolute_determinant.subs(x, sp.Rational(1, 4)) == sp.Rational(3486784401, 268435456)
    assert certificate["exact_absolute_lower_bound"] == "3486784401/268435456"
    assert replayed["preregistered_tube"]["real_v_10_interval"] == ["-1/2", "1/2"]


def test_all_11_accelerations_and_residuals_replay_independently(
    replayed: dict[str, object],
) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    prior_config = json.loads((ROOT / config["predecessor"]["config_path"]).read_text())
    matrix = _load_matrix(prior_config, ROOT)
    r = sp.Symbol("r")
    zero_symbols = [sp.Symbol(name) for name in replayed["preregistered_tube"]["zeroed_symbols"]]
    matrix = matrix.xreplace({symbol: 0 for symbol in zero_symbols}).subs(r, 1)
    accelerations = sp.Matrix(
        [sp.sympify(entry["expression"]) for entry in replayed["accelerations"]]
    )
    w_values = sp.Matrix(sp.symbols("W_0:11"))
    assert len(accelerations) == 11
    assert [sp.factor(value) for value in matrix * accelerations + w_values] == [0] * 11
    assert replayed["residual_replay"]["all_zero"] is True
    assert replayed["residual_replay"]["count"] == 11


def test_acceleration_and_residual_entries_have_individual_seals(
    replayed: dict[str, object],
) -> None:
    for row, entry in enumerate(replayed["accelerations"]):
        assert entry["row"] == row
        assert entry["entry_sha256"] == _canonical_sha(
            {"row": row, "expression": entry["expression"]}
        )
    for row, entry in enumerate(replayed["residual_replay"]["entries"]):
        assert entry["entry_sha256"] == _canonical_sha({"row": row, "expression": "0"})


def test_W_placeholders_are_bound_to_explicit_acceleration_free_packet_entries(
    replayed: dict[str, object],
) -> None:
    bindings = replayed["sealed_W_inputs"]
    assert [binding["symbol"] for binding in bindings] == [f"W_{row}" for row in range(11)]
    assert all(len(binding["source_W_entry_sha256"]) == 64 for binding in bindings)
    assert all("on the preregistered tube slice" in binding["meaning"] for binding in bindings)


def test_claims_are_narrow_and_preserve_global_block(replayed: dict[str, object]) -> None:
    claims = replayed["claims"]
    assert claims["representative_tube_invertible"] is True
    assert claims["representative_tube_all_11_accelerations_solved"] is True
    assert claims["representative_tube_all_11_residuals_replayed"] is True
    assert claims["global_representative_domain_invertible"] is False
    assert claims["other_candidates_solved"] is False
    assert claims["full_rhs"] is False
    assert claims["propagation"] is False
    assert claims["hyperbolicity"] is False


def test_receipt_and_source_seals_are_closed(replayed: dict[str, object]) -> None:
    body = {key: value for key, value in replayed.items() if key != "content_sha256"}
    assert replayed["content_sha256"] == _canonical_sha(body)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    for binding in config["source_evidence"].values():
        assert _canonical_lf_sha(ROOT / binding["path"]) == binding["canonical_lf_sha256"]


def test_tube_formula_and_predecessor_tamper_fail_closed(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["caps"]["v_10_interval"] = ["-2/3", "2/3"]
    tampered = tmp_path / "wide-tube.json"
    tampered.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(System10GravityScalarAWNonsingularTubeError, match="caps changed"):
        build_receipt(tampered, root=ROOT)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["expected_accelerations"][10] = "0"
    tampered = tmp_path / "bad-solve.json"
    tampered.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(System10GravityScalarAWNonsingularTubeError, match="solve changed"):
        build_receipt(tampered, root=ROOT)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["predecessor"]["receipt_content_sha256"] = "0" * 64
    tampered = tmp_path / "bad-predecessor.json"
    tampered.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(System10GravityScalarAWNonsingularTubeError, match="receipt mismatch"):
        build_receipt(tampered, root=ROOT)


def test_W_binding_tamper_fails_closed(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["ordered_w_entry_set_sha256"] = "0" * 64
    tampered = tmp_path / "bad-W.json"
    tampered.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(System10GravityScalarAWNonsingularTubeError, match="ordered W cap changed"):
        build_receipt(tampered, root=ROOT)


def test_write_is_atomic_and_never_overwrites(tmp_path: Path) -> None:
    output = tmp_path / "receipt.json"
    first = write_receipt(CONFIG, output)
    assert json.loads(output.read_text(encoding="utf-8")) == first
    with pytest.raises(System10GravityScalarAWNonsingularTubeError, match="refusing to overwrite"):
        write_receipt(CONFIG, output)
