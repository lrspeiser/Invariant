from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path

import pytest

from sigma_theory_compiler import (
    open_gravity_refracted_gravity_things_heracles_sparc_3d_expansion_source_acquisition_v1 as acquisition,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def config() -> dict[str, object]:
    return acquisition.load_config()


@pytest.fixture(scope="module")
def rows(config: dict[str, object]) -> list[dict[str, object]]:
    return acquisition.build_inventory(config)


def _raw_config() -> dict[str, object]:
    return json.loads((ROOT / acquisition.CONFIG_PATH).read_text(encoding="utf-8"))


def test_exact_inventory_counts_and_roots(
    config: dict[str, object], rows: list[dict[str, object]]
) -> None:
    inventory = config["inventory_contract"]
    assert len(rows) == 77
    assert sum(row["bytes"] for row in rows) == 351_461_418
    assert sum(row["total_pixels"] for row in rows) == 87_444_699
    assert sum(row["finite_pixels"] for row in rows) == 60_869_171
    assert acquisition.content_sha256(rows) == inventory["ordered_record_root_sha256"]
    assert Counter(row["survey"] for row in rows) == {
        "S4G_P5": 15,
        "SINGS_IRAC1": 14,
        "THINGS": 24,
        "HERACLES": 24,
    }


def test_every_file_has_unique_raw_and_decompressed_seals(rows: list[dict[str, object]]) -> None:
    assert len({row["relative_path"] for row in rows}) == 77
    assert all(len(row["sha256"]) == 64 for row in rows)
    assert all(len(row["decompressed_sha256"]) == 64 for row in rows)
    assert all(row["hdu_count"] == 1 for row in rows)


def test_survey_fits_contracts(rows: list[dict[str, object]]) -> None:
    for row in rows:
        if row["survey"] in {"S4G_P5", "SINGS_IRAC1"}:
            assert len(row["shape"]) == 2
            assert row["bunit"] == "MJy/sr"
            assert row["ctype1"].startswith("RA---TAN")
            assert row["ctype2"].startswith("DEC--TAN")
        elif row["survey"] == "THINGS":
            assert row["shape"] == [1, 1, 1024, 1024]
            assert row["bunit"] == "JY/B*M/S"
            assert row["beam_source"] == "AIPS_HISTORY"
            assert row["rest_hz"] in {1420405750.0, 1420405752.0}
        else:
            assert row["survey"] == "HERACLES"
            assert len(row["shape"]) == 2
            assert row["bunit"] == "K KM/S"
            assert row["ctype1"].startswith("RA---TAN")
            assert row["ctype2"].startswith("DEC--TAN")


def test_all_twelve_objects_have_required_source_roles(rows: list[dict[str, object]]) -> None:
    for object_id in acquisition.preflight._OBJECTS:
        object_rows = [row for row in rows if row["object_id"] == object_id]
        roles = {row["role"] for row in object_rows}
        assert {"HI_MOM0_NATURAL", "HI_MOM0_ROBUST", "CO21_MOM0", "CO21_EMOM0"} <= roles
        assert "STELLAR_MASS_MAP" in roles or "STELLAR_IRAC1_FLUX" in roles


def test_transport_and_response_boundary(config: dict[str, object]) -> None:
    transport = config["transport_accounting"]
    boundary = config["scientific_boundary"]
    assert transport == {
        "successful_source_gets": 77,
        "failed_gets": 0,
        "redirects_followed": 0,
        "retries": 0,
        "network_body_bytes": 351_461_418,
        "response_or_velocity_gets": 0,
        "paid_cost_usd": 0.0,
    }
    assert boundary["source_images_opened"] is True
    assert boundary["source_pixels_inspected"] is True
    assert boundary["velocity_or_rotation_response_opened"] is False
    assert boundary["response_rows_opened"] == 0
    assert boundary["scores_computed"] == 0
    assert boundary["models_fit"] == 0


def test_builder_admission_is_not_overclaimed(config: dict[str, object]) -> None:
    admission = config["builder_admission"]
    assert admission["s4g_objects_builder_ready"] == 5
    assert admission["sings_objects_conversion_builder_required"] == 7
    assert admission["sings_s4g_overlap_objects_for_source_only_validation"] == [
        "NGC2976",
        "NGC3198",
        "NGC3521",
    ]
    assert admission["source_download_is_not_builder_validation"] is True
    assert admission["response_scoring_allowed_by_this_package"] is False


def test_receipt_exactly_rebuilds(config: dict[str, object]) -> None:
    receipt = acquisition.build_receipt(config)
    acquisition.validate_receipt(receipt, config)
    assert receipt["decision"] == (
        "PASS_EXACT_SOURCE_BYTES_AND_FITS_SCHEMAS_READY_FOR_RESPONSE_BLIND_BUILDERS"
    )
    assert receipt["claims"]["exact_source_bytes_and_fits_schemas_validated"] is True
    assert receipt["claims"]["sings_flux_to_mass_builder_validated"] is False
    assert receipt["claims"]["three_dimensional_sources_built"] is False
    assert receipt["claims"]["rotation_curves_scored"] is False


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("status",), "PUBLICATION_READY"),
        (("inventory_contract", "file_count"), 76),
        (("inventory_contract", "compressed_source_bytes"), 1),
        (("transport_accounting", "successful_source_gets"), 76),
        (("transport_accounting", "redirects_followed"), 1),
        (("scientific_boundary", "response_rows_opened"), 1),
        (("claims", "sings_flux_to_mass_builder_validated"), True),
        (("claims", "publication_or_discovery_claim"), True),
    ],
)
def test_semantic_mutations_fail_closed(path: tuple[str, ...], value: object) -> None:
    mutated = copy.deepcopy(_raw_config())
    target = mutated
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(acquisition.ExpansionSourceError):
        acquisition.validate_config(mutated)


def test_receipt_mutation_fails_closed(config: dict[str, object]) -> None:
    receipt = acquisition.build_receipt(config)
    receipt["checks"]["no_velocity_map_cube_or_rotation_response"] = False
    receipt["content_sha256"] = acquisition.content_sha256(
        {key: value for key, value in receipt.items() if key != "content_sha256"}
    )
    with pytest.raises(acquisition.ExpansionSourceError):
        acquisition.validate_receipt(receipt, config)


def test_cli_has_no_caller_selected_input_or_output_path() -> None:
    parser = acquisition._parser()
    assert vars(parser.parse_args(["status"])) == {"command": "status"}
    with pytest.raises(SystemExit):
        parser.parse_args(["check", "--output", "C:/forged.json"])
