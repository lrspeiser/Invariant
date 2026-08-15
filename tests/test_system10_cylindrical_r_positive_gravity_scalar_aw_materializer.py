from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from sigma_theory_compiler.system10_cylindrical_r_positive_gravity_scalar_aw_materializer import (
    System10GravityScalarAWMaterializerError,
    _canonical_lf_sha,
    _canonical_sha,
    _verify_checkpoint,
    build_row_checkpoint,
    run_rows,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/system10_cylindrical_r_positive_gravity_scalar_aw_materializer.json"
ROW10 = (
    ROOT / "runs/math/system10-cylindrical-r-positive-gravity-scalar-aw-materializer/row-10.json"
)
ROWS_DIR = ROOT / "runs/math/system10-cylindrical-r-positive-gravity-scalar-aw-materializer"


@pytest.fixture(scope="module")
def replayed_rows() -> list[dict[str, Any]]:
    return [build_row_checkpoint(CONFIG, row) for row in range(11)]


@pytest.fixture(scope="module")
def row10(replayed_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return replayed_rows[10]


def test_committed_scalar_checkpoint_replays_exactly(row10: dict[str, Any]) -> None:
    assert row10 == json.loads(ROW10.read_text(encoding="utf-8"))
    assert row10["row"] == 10
    assert row10["field_pair"] == "gravity_scalar"


def test_committed_representative_census_materializes_all_121_A_and_11_W_entries(
    replayed_rows: list[dict[str, Any]],
) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config_sha = _canonical_sha(config)
    checkpoints = [
        json.loads((ROWS_DIR / f"row-{row:02d}.json").read_text(encoding="utf-8"))
        for row in range(11)
    ]
    assert replayed_rows == checkpoints
    assert [item["row"] for item in checkpoints] == list(range(11))
    assert sum(len(item["A_entries"]) for item in checkpoints) == 121
    assert [
        sum(entry["expression"] != "0" for entry in item["A_entries"]) for item in checkpoints
    ] == [4, 1, 1, 1, 6, 5, 5, 6, 5, 6, 7]
    for row, checkpoint in enumerate(checkpoints):
        _verify_checkpoint(checkpoint, row, config_sha)
        assert checkpoint["certificates"]["affine_residual"] == "0"
        assert checkpoint["certificates"]["domain_excludes_all_poles"] is True
        assert checkpoint["claims"]["solved_acceleration_row"] is False


def test_scalar_checkpoint_materializes_all_A_entries_and_W_without_global_factor(
    row10: dict[str, Any],
) -> None:
    assert len(row10["A_entries"]) == 11
    assert [item["column"] for item in row10["A_entries"]] == list(range(11))
    assert len({item["entry_sha256"] for item in row10["A_entries"]}) == 11
    certificate = row10["certificates"]
    assert certificate["affine_residual"] == "0"
    assert certificate["acceleration_free_A_entries"] == 11
    assert certificate["acceleration_free_W_entries"] == 1
    assert certificate["global_factorization_used"] is False
    assert certificate["row_checkpoint_before_other_rows"] is True
    assert certificate["coordinate_poles"] == ["r=0"]


def test_scalar_expressions_are_coordinate_arithmetic_not_component_placeholders(
    row10: dict[str, Any],
) -> None:
    expressions = [item["expression"] for item in row10["A_entries"]]
    expressions.append(row10["W_entry"]["expression"])
    assert all("exact_component_input" not in expression for expression in expressions)
    assert all("partial0_v_" not in expression for expression in expressions)
    assert any("r" in expression for expression in expressions)
    assert any("partial_1_w_1_10" in expression for expression in expressions)


def test_checkpoint_seal_and_source_contract_are_closed(row10: dict[str, Any]) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    _verify_checkpoint(row10, 10, _canonical_sha(config))
    body = {key: value for key, value in row10.items() if key != "content_sha256"}
    assert row10["content_sha256"] == _canonical_sha(body)
    assert (
        row10["source_bindings"]["nonlinear_source_sha256"]
        == (config["source_evidence"]["nonlinear_source"]["canonical_lf_sha256"])
    )


def test_resumable_worker_writes_once_and_reuses_verified_checkpoint(
    tmp_path: Path,
) -> None:
    first = run_rows(CONFIG, tmp_path, [10], root=ROOT)
    second = run_rows(CONFIG, tmp_path, [10], root=ROOT)
    assert first == second
    assert first[0] == json.loads((tmp_path / "row-10.json").read_text(encoding="utf-8"))
    assert list(tmp_path.glob(".row-*.tmp")) == []


def test_tampered_resume_checkpoint_fails_without_overwrite(tmp_path: Path) -> None:
    target = tmp_path / "row-10.json"
    target.write_text("{}\n", encoding="utf-8")
    with pytest.raises(System10GravityScalarAWMaterializerError, match="seal mismatch"):
        run_rows(CONFIG, tmp_path, [10], root=ROOT)
    assert target.read_text(encoding="utf-8") == "{}\n"


def test_row_and_cap_tamper_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(System10GravityScalarAWMaterializerError, match="outside frozen cap"):
        build_row_checkpoint(CONFIG, 11)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["caps"]["wall_seconds_per_row"] = 121
    tampered = tmp_path / "config.json"
    tampered.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(System10GravityScalarAWMaterializerError, match="caps changed"):
        build_row_checkpoint(tampered, 10, root=ROOT)


def test_binding_and_source_tamper_fail_closed(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["bindings"]["aw_readiness"]["canonical_lf_sha256"] = "0" * 64
    tampered_binding = tmp_path / "binding.json"
    tampered_binding.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(System10GravityScalarAWMaterializerError, match="hash mismatch"):
        build_row_checkpoint(tampered_binding, 10, root=ROOT)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["source_evidence"]["nonlinear_source"]["canonical_lf_sha256"] = "0" * 64
    tampered_source = tmp_path / "source.json"
    tampered_source.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(System10GravityScalarAWMaterializerError, match="source hash mismatch"):
        build_row_checkpoint(tampered_source, 10, root=ROOT)


def test_self_evidence_uses_canonical_lf_hashes() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    for binding in config["source_evidence"].values():
        assert _canonical_lf_sha(ROOT / binding["path"]) == binding["canonical_lf_sha256"]
