from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler import open_gravity_source_availability_contract_v1 as source

ROOT = Path(__file__).resolve().parents[1]


def _rows_by_key(
    config: dict, wanted: set[tuple[str, str, str]]
) -> dict[tuple[str, str, str], dict]:
    found: dict[tuple[str, str, str], dict] = {}
    for row in source.iter_matrix_rows(config):
        key = (row["mechanism_id"], row["object_id"], row["observable_id"])
        if key in wanted:
            found[key] = row
    return found


def test_config_covers_exact_frozen_catalog_and_objects() -> None:
    config = source.load_config(ROOT)
    assert len(source.twell_concept_ids()) == 400
    assert source.content_sha256(source.twell_concept_ids()) == source.EXPECTED_TWELL_IDS_SHA256
    assert len(source.mechanism_catalog(config)) == 420
    assert len(config["objects"]["SPARC"]) == 139
    assert len(config["objects"]["XCOP"]) == 8
    assert set(config["mechanism_registry"]["discovery_lanes"]) == {
        "CORE",
        "ADJACENT",
        "ORTHOGONAL",
        "RIVALS_CONTROLS",
        "WILDCARD",
    }


@pytest.mark.parametrize(
    "mutation",
    [
        lambda config: config.__setitem__("purpose", "SCIENTIFIC CONFIRMATION"),
        lambda config: config.__setitem__("status", "CONFIRMED"),
        lambda config: config["incident_ledger"][0].__setitem__("remedy", "no restriction"),
        lambda config: config["objects"]["SPARC"].__setitem__(0, "FAKE_OBJECT"),
        lambda config: config["GP01_source_availability"]["GP01-L"].__setitem__(
            "SPARC", "UNKNOWN_SOURCE_BLOCKED"
        ),
        lambda config: config["comparator_inventory"][0].__setitem__(
            "mode", "EXECUTABLE_FORWARD_SOLVER"
        ),
        lambda config: config["legacy_multiplicity_lower_bound"].__setitem__(
            "status", "COMPLETE_RESET_ALLOWED"
        ),
        lambda config: config["out_of_scope"].__setitem__(0, "all science proven"),
    ],
)
def test_config_semantics_are_exactly_frozen(mutation) -> None:
    config = json.loads((ROOT / source.CONFIG_PATH).read_text(encoding="utf-8"))
    mutation(config)
    with pytest.raises(source.SourceAvailabilityError, match="config semantics changed"):
        source.validate_config(copy.deepcopy(config))


def test_matrix_has_exact_cartesian_size_and_is_deterministic() -> None:
    config = source.load_config(ROOT)
    first = source.matrix_summary(config)
    second = source.matrix_summary(config)
    assert first == second
    assert first["expanded_tuple_count"] == 65_100
    assert first["domain_tuple_counts"] == {"SPARC": 58_380, "XCOP": 6_720}
    assert first["rows_materialized_in_receipt"] == 0


def test_source_statuses_are_object_specific_without_scoring() -> None:
    config = source.load_config(ROOT)
    wanted = {
        ("TW2-A01-D01", "CamB", "ROTATION_CURVE"),
        ("X01", "A85", "PRESSURE_PROFILE"),
        ("X01", "A3266", "TEMPERATURE_PROFILE"),
        ("GP01-AQUAL", "CamB", "ROTATION_CURVE"),
        ("GP01-AQUAL", "A85", "PRESSURE_PROFILE"),
        ("GP01-TELEGRAPH", "A85", "TEMPERATURE_PROFILE"),
        ("QG02", "A85", "PRESSURE_PROFILE"),
    }
    rows = _rows_by_key(config, wanted)
    assert set(rows) == wanted

    atomic = rows[("TW2-A01-D01", "CamB", "ROTATION_CURVE")]
    assert atomic["driver_source_status"] == "SOURCE_AVAILABLE"
    assert atomic["mechanism_source_status"] == "UNKNOWN_SOURCE_BLOCKED"

    x01_direct = rows[("X01", "A85", "PRESSURE_PROFILE")]
    assert x01_direct["stellar_source_status"] == "SOURCE_AVAILABLE"
    assert x01_direct["mechanism_source_status"] == "SOURCE_AVAILABLE_SPHERICAL_RADIAL_ONLY"

    x01_nuisance = rows[("X01", "A3266", "TEMPERATURE_PROFILE")]
    assert x01_nuisance["stellar_source_status"] == "SOURCE_MISSING"
    assert (
        x01_nuisance["mechanism_source_status"]
        == "SOURCE_AVAILABLE_WITH_SHARED_GLOBAL_STELLAR_NUISANCE"
    )

    assert (
        rows[("GP01-AQUAL", "CamB", "ROTATION_CURVE")]["mechanism_source_status"]
        == "SOURCE_MISSING"
    )
    assert (
        rows[("GP01-AQUAL", "A85", "PRESSURE_PROFILE")]["mechanism_source_status"]
        == "SOURCE_AVAILABLE_SPHERICAL_RADIAL_ONLY"
    )
    assert (
        rows[("GP01-TELEGRAPH", "A85", "TEMPERATURE_PROFILE")]["mechanism_source_status"]
        == "SOURCE_MISSING"
    )
    assert (
        rows[("QG02", "A85", "PRESSURE_PROFILE")]["mechanism_source_status"]
        == "UNKNOWN_SOURCE_BLOCKED"
    )
    assert all(row["source_only_no_scoring_authority"] for row in rows.values())


def test_comparator_inventory_does_not_promote_analogues() -> None:
    config = source.load_config(ROOT)
    comparators = {row["id"]: row for row in config["comparator_inventory"]}
    for comparator in (
        "PENNER_2026",
        "REFRACTED_GRAVITY",
        "EMOND",
        "MOG_STVG",
        "PUBLISHED_NONLOCAL_GRAVITY",
    ):
        assert "NO_" in comparators[comparator]["mode"]
        assert comparators[comparator]["SPARC"] == "UNKNOWN_SOURCE_BLOCKED"
        assert comparators[comparator]["XCOP"] == "UNKNOWN_SOURCE_BLOCKED"
    assert comparators["AQUAL_EFE"]["SPARC"] == "SOURCE_MISSING"
    assert "EQUIVALENCE_ONLY" in comparators["AQUAL_EFE"]["mode"]


def test_receipt_has_zero_access_and_campaign_remains_unfrozen() -> None:
    receipt = source.build_receipt(ROOT)
    assert set(receipt["zero_access"].values()) == {0}
    assert receipt["campaign_manifest"] == {
        "status": "UNFROZEN",
        "reason": "registry_and_GP01_audits_and_independent_candidate_fixing_are_not_complete",
        "response_execution_authorized_by_this_receipt": False,
    }
    assert len(receipt["source_auditor_incidents"]) == 2
    assert receipt["legacy_multiplicity_lower_bound"]["status"].startswith(
        "LEGACY_LEDGER_INCOMPLETE"
    )


def test_append_only_writer_refuses_nonidentical_content(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    assert source._atomic_no_clobber(path, b"one\n") == "CREATED"
    assert source._atomic_no_clobber(path, b"one\n") == "EXISTING_IDENTICAL"
    with pytest.raises(source.SourceAvailabilityError, match="refusing to overwrite"):
        source._atomic_no_clobber(path, b"two\n")


def test_serialized_receipt_contains_no_matrix_rows_or_response_values() -> None:
    receipt = source.build_receipt(ROOT)
    encoded = json.dumps(receipt, sort_keys=True)
    assert '"rows"' not in encoded
    assert '"scores"' not in encoded
    assert receipt["matrix"]["rows_materialized_in_receipt"] == 0
    assert receipt["zero_access"]["scientific_response_payloads_read_by_generator"] == 0
