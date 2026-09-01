from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path

import pytest

from sigma_theory_compiler import open_gravity_rg_s4g_geometry_source_v1 as geometry


def _config() -> dict[str, object]:
    return geometry.load_config(verify_package=False)


def test_public_sources_and_exact_five_object_ledger() -> None:
    config = _config()
    receipt = geometry.build_receipt(config)

    assert [row["object_id"] for row in receipt["objects"]] == [
        "NGC2903",
        "NGC2976",
        "NGC3198",
        "NGC3521",
        "NGC4214",
    ]
    assert receipt["object_count"] == 5
    assert receipt["reliable_orientation_count"] == 3
    assert receipt["uncertain_orientation_count"] == 2
    assert receipt["source_evidence"] == {
        "S4G_CATALOG_V2": {
            "raw_bytes": 133104,
            "raw_sha256": "ffbbcdcd91f947e0599232264bd40d46cbefa6051fdfb8bdbfc69d2d52fc0d12",
            "decoded_bytes": 368614,
            "decoded_sha256": "bc872c485e47bef0976a8e47c5c7b7cef0fa34c5bca2ba7687ee332c550fe0c9",
            "record_count": 2352,
        },
        "S4G_PIPELINE4_OUTER_GEOMETRY": {
            "raw_bytes": 214082,
            "raw_sha256": "c1d90e8824afbe4dd261ceaa663b6dc484f1cd90751f89dc0db140a5418638ea",
            "decoded_bytes": 214082,
            "decoded_sha256": "c1d90e8824afbe4dd261ceaa663b6dc484f1cd90751f89dc0db140a5418638ea",
            "record_count": 2352,
        },
    }


def test_uncertain_orientations_are_not_promoted_to_single_truth() -> None:
    receipt = geometry.build_receipt(_config())
    by_name = {row["object_id"]: row for row in receipt["objects"]}

    assert by_name["NGC3521"]["orientation_flag"] == "u"
    assert by_name["NGC4214"]["orientation_flag"] == "u"
    assert "SENSITIVITY_CELLS" in by_name["NGC3521"]["disposition"]
    assert "SENSITIVITY_CELLS" in by_name["NGC4214"]["disposition"]
    assert receipt["future_builder_contract"]["fit_geometry_to_response"] is False
    assert receipt["future_builder_contract"]["retain_all_geometry_failures"] is True


def test_intrinsic_thickness_cells_are_ordered_and_model_labeled() -> None:
    receipt = geometry.build_receipt(_config())
    for row in receipt["objects"]:
        cells = row["inclination_model_cells"]
        assert [cell["intrinsic_axis_ratio_q0"] for cell in cells] == [0.0, 0.13, 0.2]
        assert all(cell["role"] == "MODEL_GEOMETRY_SENSITIVITY" for cell in cells)
        inclinations = [cell["inclination_deg"] for cell in cells]
        assert inclinations == sorted(inclinations)
        assert all(0.0 <= value <= 90.0 for value in inclinations)


def test_geometry_claim_ceiling_and_zero_response_access() -> None:
    receipt = geometry.build_receipt(_config())
    assert receipt["claims"] == {
        "public_geometry_source_validated": True,
        "five_object_geometry_records_bound": True,
        "all_orientations_reliable": False,
        "three_dimensional_sources_built": False,
        "response_scored": False,
        "refracted_gravity_supported": False,
        "publication_or_discovery_claim": False,
    }
    access = receipt["access_state"]
    assert access["scientific_image_files_opened"] == 0
    assert access["response_files_opened"] == 0
    assert access["response_rows_opened"] == 0
    assert access["scores_computed"] == 0
    assert access["network_calls_by_validator"] == 0


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("status",), "CONFIRMED"),
        (("claims", "three_dimensional_sources_built"), True),
        (("claims", "response_scored"), True),
        (("future_builder_contract", "fit_geometry_to_response"), True),
        (("future_builder_contract", "retain_all_geometry_failures"), False),
        (("objects", 4, "orientation_flag"), "ok"),
        (("output_path",), "runs/gravity/forged.json"),
    ],
)
def test_semantic_mutations_fail_closed(path: tuple[object, ...], value: object) -> None:
    mutated = copy.deepcopy(_config())
    cursor: object = mutated
    for key in path[:-1]:
        cursor = cursor[key]  # type: ignore[index]
    cursor[path[-1]] = value  # type: ignore[index]

    with pytest.raises(geometry.GeometrySourceError):
        geometry.validate_config(mutated)


def test_fixed_cli_has_no_caller_selected_source_or_output_paths() -> None:
    assert tuple(inspect.signature(geometry.write_receipt).parameters) == ()
    assert tuple(inspect.signature(geometry.check_receipt).parameters) == ()
    choices = geometry._parser()._actions[1].choices
    assert choices == ("write", "check", "status")


def test_atomic_no_clobber_preserves_existing_bytes(tmp_path: Path) -> None:
    target = tmp_path / "receipt.json"
    payload = b'{"a":1}'
    assert geometry._atomic_no_clobber(target, payload) == "CREATED"
    assert geometry._atomic_no_clobber(target, payload) == "EXISTING_IDENTICAL"
    with pytest.raises(geometry.GeometrySourceError):
        geometry._atomic_no_clobber(target, b'{"a":2}')
    assert target.read_bytes() == payload


def test_receipt_is_canonical_and_self_hashes() -> None:
    receipt = geometry.build_receipt(_config())
    payload = geometry.canonical_bytes(receipt)
    assert json.loads(payload) == receipt
    assert receipt["content_sha256"] == geometry.content_sha256(receipt)
