from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_tc2_d4_p55_checkpointable_materializer import (
    C0_SCHEMA,
    P55MaterializerError,
    _atomic_write_immutable,
    _content_hash,
    _sphere_groebner,
    _with_hash,
    build_c0,
    build_plan,
    next_units,
    validate_final_result,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/backgrounds/quartic_tc2_d4_p55_checkpointable_materializer.json"
PLAN = ROOT / "runs/physics-language/quartic-tc2-d4-p55-checkpointable-materializer/plan.json"
RESULT = ROOT / "runs/physics-language/quartic-tc2-d4-p55-checkpointable-materializer/result.json"


def test_sealed_plan_matches_pristine_checkpoint_state(tmp_path: Path) -> None:
    plan = build_plan(ROOT, CONFIG, tmp_path / "checkpoints")
    sealed = json.loads(PLAN.read_text(encoding="utf-8"))
    normalized = copy.deepcopy(plan)
    normalized["checkpoint_directory"] = sealed["checkpoint_directory"]
    normalized["content_sha256"] = _content_hash(normalized)
    assert normalized == sealed
    assert sealed["next_units"] == ["C0"]
    assert sealed["authoritative_BLOCK_receipt_mutated"] is False


def test_c0_verifies_all_live_and_receipt_seals() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    c0 = build_c0(ROOT, config)
    assert c0["schema_version"] == C0_SCHEMA
    assert c0["authoritative_BLOCK_unchanged"] is True
    assert c0["source"]["required_functions"] == [
        "_symbol_data",
        "_extract_spatial_blocks",
        "_full_first_order_pencil",
    ]


def test_resume_order_is_c0_then_indivisible_c1(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    assert next_units(checkpoint_dir) == ["C0"]
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    c0 = build_c0(ROOT, config)
    _atomic_write_immutable(checkpoint_dir / "c0-seals.json", c0, 1_000_000)
    assert next_units(checkpoint_dir) == ["C1"]


def test_immutable_checkpoint_accepts_identical_replay_and_rejects_change(
    tmp_path: Path,
) -> None:
    path = tmp_path / "checkpoint.json"
    document = _with_hash({"schema_version": "test", "value": 1})
    _atomic_write_immutable(path, document, 1_000_000)
    _atomic_write_immutable(path, document, 1_000_000)
    changed = _with_hash({"schema_version": "test", "value": 2})
    with pytest.raises(P55MaterializerError, match="immutable checkpoint conflict"):
        _atomic_write_immutable(path, changed, 1_000_000)


def test_checkpoint_byte_cap_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(P55MaterializerError, match="exceeds byte cap"):
        _atomic_write_immutable(tmp_path / "large.json", {"payload": "x" * 100}, 10)


def test_sphere_reducer_accepts_exact_rational_sqrt2_coefficients() -> None:
    import sympy as sp

    n1, n2, n3 = sp.symbols("n1 n2 n3")
    relation = n1**2 + n2**2 + n3**2 - 1
    reducer = _sphere_groebner(n1, n2, n3)
    assert reducer.reduce(-sp.Rational(49, 576) * relation)[1] == 0
    assert reducer.reduce(sp.sqrt(2) * sp.Rational(1, 9) * relation)[1] == 0


def test_config_source_tamper_fails_closed(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["live_source"]["file_sha256"] = "0" * 64
    config["content_sha256"] = _content_hash(config)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(P55MaterializerError, match="source file seal mismatch"):
        build_plan(ROOT, path, tmp_path / "checkpoints")


def test_config_cap_tamper_fails_closed(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["caps"]["C1"]["wall_seconds"] = 0
    config["content_sha256"] = _content_hash(config)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(P55MaterializerError, match="invalid materializer config"):
        build_plan(ROOT, path, tmp_path / "checkpoints")


def test_plan_preserves_all_broad_false_claims(tmp_path: Path) -> None:
    claims = build_plan(ROOT, CONFIG, tmp_path / "checkpoints")["claims"]
    assert claims
    assert all(value is False for value in claims.values())


def test_portable_final_result_replays_exact_matrices_and_polynomial() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    validate_final_result(result, ROOT, CONFIG)
    assert result["counts"]["matrix_packets"] == 3
    assert result["counts"]["sparse_entries"] == 144
    assert result["counts"]["minimal_polynomial_entries_reduced"] == 3025
    assert result["counts"]["minimal_polynomial_nonzero_remainders"] == 0
    assert result["claims"]["flat_reference_P55_sphere_minimal_polynomial_certified"] is True
    assert result["claims"]["full_direction_sphere_D4_compatibility_proved"] is False


def test_resealed_final_overclaim_is_rejected() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    result["claims"]["global_H7_closed"] = True
    result["content_sha256"] = _content_hash(result)
    with pytest.raises(P55MaterializerError, match="claim boundary"):
        validate_final_result(result, ROOT, CONFIG)
