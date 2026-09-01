from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler import (
    open_gravity_solar_planetary_source_shaped_synthetic_injection_matrix_v1 as v1,
)
from sigma_theory_compiler import (
    open_gravity_solar_planetary_source_shaped_synthetic_injection_matrix_v2 as v2,
)
from sigma_theory_compiler.sigma_core import SchemaViolation

ROOT = Path(__file__).resolve().parents[1]


def test_v2_binds_every_v1_subject_and_artifact_byte_exact() -> None:
    config = v2.load_config()
    v2.validate_config(config)
    assert len(config["predecessor_bindings"]) == 9
    for row in config["predecessor_bindings"]:
        path = ROOT / row["path"]
        assert path.stat().st_size == row["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("status",), "CHANGED"),
        (("lint_corrections", 0, "after"), "# removed"),
        (("predecessor_bindings", 0, "sha256"), "0" * 64),
        (("access_contract", "observational_response_rows_opened"), 1),
    ],
)
def test_material_v2_mutations_fail_closed(path: tuple[object, ...], value: object) -> None:
    config = copy.deepcopy(v2.load_config())
    cursor: object = config
    for key in path[:-1]:
        cursor = cursor[key]  # type: ignore[index]
    cursor[path[-1]] = value  # type: ignore[index]
    with pytest.raises(SchemaViolation):
        v2.validate_config(
            config,
            verify_predecessor=path == ("predecessor_bindings", 0, "sha256"),
        )


def test_corrected_projection_is_exactly_the_two_allowed_source_edits() -> None:
    config = v2.load_config()
    source = (ROOT / v2._binding(config, "V1_MODULE")["path"]).read_text(encoding="utf-8")
    expected = source
    for correction in config["lint_corrections"]:
        assert expected.count(correction["before"]) == 1
        expected = expected.replace(correction["before"], correction["after"], 1)
    projection = v2.corrected_module_projection(config)
    assert projection == expected.encode("utf-8")
    projected_text = projection.decode("utf-8")
    assert "    DiscoveryStatus,\n" not in projected_text
    assert projected_text.count("except Exception as error:  # noqa: BLE001") == 1
    assert len(projection) - len(source.encode("utf-8")) == -5


def test_v1_replay_and_all_scientific_results_remain_identical() -> None:
    replayed = v1.check()
    receipt, _projection = v2.build_receipt()
    assert receipt["predecessor_replay_passed"] is True
    assert receipt["scientific_payload_reused_byte_exact"] is True
    assert receipt["new_scientific_scenarios_generated"] == 0
    assert receipt["new_candidate_comparisons_computed"] == 0
    assert receipt["preserved_v1_results"] == v2.load_config()["preserved_v1_results"]
    assert receipt["preserved_v1_artifact_sha256"] == replayed["artifact_sha256"]
    assert receipt["predecessor_receipt_content_sha256"] == replayed["content_sha256"]


def test_v2_receipt_is_lint_only_response_blind_and_awaits_distinct_audit() -> None:
    receipt = json.loads((ROOT / v2.RECEIPT_PATH).read_text(encoding="utf-8"))
    assert receipt["status"] == "FROZEN_LINT_ONLY_SUCCESSOR_COMPLETE_AWAITING_DISTINCT_AUDIT"
    assert receipt["scientific_claim"] == "NONE_LINT_ONLY_SUCCESSOR"
    assert receipt["independent_audit_completed"] is False
    assert receipt["distinct_independent_audit_required"] is True
    assert receipt["v1_subject_bytes_modified"] is False
    assert receipt["lint_correction_count"] == 2
    assert all(value == 0 for value in receipt["access_accounting"].values())


def test_v2_output_contains_only_projection_and_receipt() -> None:
    output = ROOT / v2.OUTPUT_DIR
    assert sorted(path.name for path in output.iterdir()) == [
        "corrected-module-projection.py",
        "receipt.json",
    ]
    receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    projection = (output / "corrected-module-projection.py").read_bytes()
    assert hashlib.sha256(projection).hexdigest() == receipt["corrected_projection_sha256"]


def test_v2_frozen_replay_is_byte_identical() -> None:
    receipt = v2.check()
    written = json.loads((ROOT / v2.RECEIPT_PATH).read_text(encoding="utf-8"))
    assert receipt == written
