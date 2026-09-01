from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from sigma_theory_compiler import open_gravity_twell_400_source_shaped_rebind_replay_v1 as subject

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def config() -> dict[str, object]:
    return subject.load_config(ROOT)


@pytest.fixture(scope="session")
def cards(config: dict[str, object]) -> list[dict[str, object]]:
    return subject.load_cards(ROOT, config)


@pytest.fixture(scope="session")
def source_material(
    config: dict[str, object],
) -> tuple[list[dict[str, object]], dict[str, dict[str, object]], list[dict[str, object]]]:
    scenarios, refs = subject.load_xcop_scenario_metadata(ROOT, config)
    projections, rows, _ = subject.load_xcop_source_projections(ROOT, config, refs)
    return scenarios, projections, rows


def test_frozen_config_and_predecessor_hashes(config: dict[str, object]) -> None:
    assert config["schema"] == subject.CONFIG_SCHEMA
    assert len(config["frozen_predecessors"]) == 14
    assert len(config["source_releases"]) == 5
    assert all(value is False for value in config["exclusions"].values())
    bindings = {row["id"]: row for row in config["frozen_predecessors"]}
    assert bindings["TWELL_400_FINAL_V3_CARDS"]["raw_sha256"] == (
        "4bd70177bf6f8aabbbf2d106f002575f9c1850cfffb45ca9103165da29b9b44f"
    )
    assert bindings["STATIC_RADIAL_ADAPTER_V1_MODULE"]["raw_sha256"] == (
        "7db918df47f612df3c42792c25d407bf5c485db6f7830f8bd690b57b95b5968f"
    )
    assert bindings["GENERIC_RUNNER_V2_INDEPENDENT_AUDIT"]["raw_sha256"] == (
        "cda03daed64b2fdd530ba95c5dacf336c35be1c0f8afaafef45af67ca0cca8e7"
    )


def test_config_mutation_is_rejected(config: dict[str, object]) -> None:
    mutated = copy.deepcopy(config)
    mutated["status"] = "MUTATED"
    with pytest.raises(subject.TwellSourceRebindError, match="status changed"):
        subject.validate_config(ROOT, mutated)


def test_exact_card_and_parameter_cell_identity(cards: list[dict[str, object]]) -> None:
    assert len(cards) == 400
    assert Counter(row["entry_kind"] for row in cards) == {"ATOMIC": 380, "COMPOUND": 20}
    assert sum(row["parameter_cell_count"] for row in cards) == 1184
    assert len({row["card_sha256"] for row in cards}) == 400
    assert len({row["card"]["hashes"]["formula_sha256"] for row in cards}) == 400
    assert len(
        {
            cell["cell_id"]
            for row in cards
            for cell in row["card"]["parameter_cells"]
        }
    ) == 1184


def test_card_mutation_is_rejected(
    config: dict[str, object], cards: list[dict[str, object]], tmp_path: Path
) -> None:
    mutated = copy.deepcopy(cards[0])
    mutated["card"]["boundaries"] = ["mutated"]
    assert subject.content_sha256(mutated["card"]) != mutated["card_sha256"]
    path = tmp_path / "cards.jsonl"
    path.write_text(json.dumps(mutated), encoding="utf-8")


def test_exact_gap_classification(cards: list[dict[str, object]]) -> None:
    card_counts, cell_counts = subject._class_counts(cards)
    assert card_counts == {
        "PROVISIONAL_STATIC": 126,
        "TEMPORAL_ARCHITECTURE": 84,
        "MISSING_DRIVER_OR_COMPOUND_ADAPTER": 190,
    }
    assert cell_counts == {
        "PROVISIONAL_STATIC": 370,
        "TEMPORAL_ARCHITECTURE": 253,
        "MISSING_DRIVER_OR_COMPOUND_ADAPTER": 561,
    }


def test_compatibility_ledger_exact_counts(
    config: dict[str, object], cards: list[dict[str, object]]
) -> None:
    rows = subject.build_compatibility_ledger(config, cards)
    assert len(rows) == 2000
    assert Counter(row["status"] for row in rows) == {
        "EXECUTABLE": 110,
        "SOURCE_BLOCKED": 290,
        "INCOMPATIBLE_FEATURE_SET": 1600,
    }
    assert len({(row["formula_id"], row["domain"], row["source_release"]) for row in rows}) == 2000
    assert all(
        row["compatibility_sha256"]
        == subject.content_sha256({key: value for key, value in row.items() if key != "compatibility_sha256"})
        for row in rows
    )


def test_exact_parameter_cell_disposition(cards: list[dict[str, object]]) -> None:
    rows = subject.build_disposition_ledger(cards)
    assert len(rows) == 1184
    assert len({row["cell_id"] for row in rows}) == 1184
    assert Counter(row["status"] for row in rows) == {
        "COMPLETED_WITH_SCENARIO_LEVEL_NUMERICAL_VALIDITY_RETAINED": 324,
        "SOURCE_BLOCKED": 860,
    }
    assert sum(row["reason"].startswith("TEMPORAL_") for row in rows) == 253
    assert sum(row["reason"].startswith("MISSING_") for row in rows) == 561
    assert sum(row["reason"].startswith("D13_") for row in rows) == 46


def test_execution_bindings_are_exact_and_d13_stays_blocked(
    config: dict[str, object], cards: list[dict[str, object]]
) -> None:
    rows = subject.build_execution_bindings(config, cards)
    assert len(rows) == 126
    assert Counter(row["status"] for row in rows) == {"EXECUTABLE": 110, "SOURCE_BLOCKED": 16}
    assert sum(len(row["parameter_cell_ids"]) for row in rows if row["status"] == "EXECUTABLE") == 324
    assert sum(len(row["parameter_cell_ids"]) for row in rows if row["status"] == "SOURCE_BLOCKED") == 46
    assert all(row["callable"] is None for row in rows if row["status"] == "SOURCE_BLOCKED")
    assert all("D13_GASF" in row["driver_ids"] for row in rows if row["status"] == "SOURCE_BLOCKED")


def test_xcop_source_metadata_is_typed_and_response_values_are_not_loaded(
    config: dict[str, object], source_material: tuple[object, object, object]
) -> None:
    scenarios, projections, rows = source_material
    assert len(scenarios) == 192
    assert len(projections) == len(rows) == 8
    assert Counter(row["object_id"] for row in scenarios) == {
        key: 24 for key in projections
    }
    assert all("D13_GASF" not in projection["physical"] for projection in projections.values())
    assert all(projection["xi"].shape == (257,) for projection in projections.values())
    assert all(np.all(np.diff(projection["radius_m"]) > 0.0) for projection in projections.values())
    assert all(row["D13_GASF_available"] is False for row in rows)
    xcop = next(row for row in config["source_releases"] if row["domain"] == "cluster")
    with np.load(ROOT / xcop["values_path"], allow_pickle=False) as archive:
        assert any(key.startswith("response__") for key in archive.files)
        requested = [
            subject._source_key(object_id, feature["id"])
            for object_id in projections
            for feature in config["feature_contracts"][xcop["feature_contract"]]
        ]
        assert len(requested) == 72
        assert all(key.startswith("source__") for key in requested)


def test_common_abi_is_byte_deterministic_and_rejects_temporal_or_mutated_cell(
    cards: list[dict[str, object]], source_material: tuple[object, object, object]
) -> None:
    _, projections, _ = source_material
    card = next(
        row
        for row in cards
        if row["concept_id"] == "TW2-A06-D01"
    )
    cell = card["card"]["parameter_cells"][-1]
    source = projections[min(projections)]
    first = subject.execute_twell_static_common_abi(card, source, cell)
    second = subject.execute_twell_static_common_abi(card, source, cell)
    assert first["prediction_sha256"] == second["prediction_sha256"]
    assert np.array_equal(first["factor"], second["factor"])
    assert np.array_equal(first["g_eff_m_s2"], second["g_eff_m_s2"])
    mutated_cell = copy.deepcopy(cell)
    mutated_cell["value"]["lambda"] = 0.123
    with pytest.raises(subject.TwellSourceRebindError, match="parameter cell not in card"):
        subject.execute_twell_static_common_abi(card, source, mutated_cell)
    temporal = next(row for row in cards if row["architecture_id"] == "A15_RETARDED")
    with pytest.raises(subject.TwellSourceRebindError, match="not statically adapted"):
        subject.execute_twell_static_common_abi(
            temporal, source, temporal["card"]["parameter_cells"][0]
        )


def test_source_projection_invariance_gates(
    source_material: tuple[object, object, object]
) -> None:
    _, projections, _ = source_material
    report = subject._invariance_gates(projections)
    assert report["rotation_axis_permutation"]["status"] == "PASS"
    assert report["translation"]["status"] == "PASS"
    assert report["unit_roundtrip_m_to_kpc_to_m"]["status"] == "PASS"
    assert report["time_static_only"]["temporal_architecture_count_executed"] == 0
    assert report["time_static_only"]["static_replication_of_temporal_law_count"] == 0


def test_frozen_package_rebuild_and_access_accounting() -> None:
    receipt = subject.validate_package(ROOT)
    assert receipt["compatibility_row_count"] == 2000
    assert receipt["parameter_cell_count"] == 1184
    assert receipt["unique_execution_counts"] == {"COMPLETED": 2554, "NUMERICAL_INVALID": 38}
    assert receipt["replay_counts"] == {"COMPLETED": 61296, "NUMERICAL_INVALID": 912}
    assert receipt["access_accounting"]["source_npz_members_opened"] == 72
    assert receipt["access_accounting"]["response_npz_members_opened"] == 0
    assert receipt["access_accounting"]["response_values_opened"] == 0
    assert receipt["access_accounting"]["scientific_scores_computed"] == 0
    assert receipt["independent_audit_completed"] is False
    assert receipt["distinct_independent_audit_required"] is True


def test_frozen_artifact_line_counts() -> None:
    output = ROOT / subject.OUTPUT_DIR
    expected = {
        subject.COMPATIBILITY_PATH.name: 2000,
        subject.DISPOSITION_PATH.name: 1184,
        subject.BINDINGS_PATH.name: 126,
        subject.SOURCE_ROWS_PATH.name: 8,
        subject.UNIQUE_EXECUTIONS_PATH.name: 2592,
        subject.REPLAY_PATH.name: 62208,
    }
    for name, count in expected.items():
        assert sum(1 for _ in (output / name).open("rb")) == count


def test_every_artifact_hash_is_bound() -> None:
    receipt = json.loads((ROOT / subject.RECEIPT_PATH).read_text(encoding="utf-8"))
    for name, expected in receipt["artifact_sha256"].items():
        assert subject.file_sha256(ROOT / subject.OUTPUT_DIR / name) == expected
    assert receipt["content_sha256"] == subject._receipt_content_sha256(receipt)
