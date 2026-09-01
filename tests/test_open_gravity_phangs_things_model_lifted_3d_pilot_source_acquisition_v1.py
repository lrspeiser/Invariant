from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler import (
    open_gravity_phangs_things_model_lifted_3d_pilot_source_acquisition_v1 as acquisition,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / acquisition.CONFIG_PATH).read_text(encoding="utf-8"))


def test_config_is_exact_and_development_only() -> None:
    acquisition.validate_config(CONFIG)
    assert CONFIG["scientific_boundary"]["development_only"] is True
    assert CONFIG["scientific_boundary"]["fresh_confirmation"] is False


def test_exact_source_inventory_and_roots() -> None:
    rows = acquisition.build_inventory(CONFIG)
    assert len(rows) == 21
    assert sum(row["bytes"] for row in rows) == 74_030_400
    assert sum(row["total_pixels"] for row in rows) == 18_026_980
    assert sum(row["finite_pixels"] for row in rows) == 16_452_117
    assert (
        acquisition.content_sha256(rows)
        == CONFIG["inventory_contract"]["ordered_inventory_root_sha256"]
    )


def test_every_source_role_is_present_for_each_object() -> None:
    rows = acquisition.build_inventory(CONFIG)
    for object_id in ("NGC2903", "NGC3351", "NGC3627"):
        subset = [row for row in rows if row["id"].startswith(f"{object_id}:")]
        assert len(subset) == 7


def test_things_beams_are_parsed_from_aips_history() -> None:
    rows = acquisition.build_inventory(CONFIG)
    things = [row for row in rows if ":THINGS:" in row["id"]]
    assert len(things) == 6
    assert all(row["beam_source"] == "AIPS_HISTORY" for row in things)
    assert all(row["beam_deg"][0] > 0 and row["beam_deg"][1] > 0 for row in things)


def test_phangs_beams_are_standard_header_values() -> None:
    rows = acquisition.build_inventory(CONFIG)
    phangs = [row for row in rows if ":PHANGS_ALMA:" in row["id"]]
    assert len(phangs) == 6
    assert all(row["beam_source"] == "HEADER" for row in phangs)
    assert all(row["rest_hz"] == 230_538_000_000.0 for row in phangs)


def test_no_response_product_is_in_inventory() -> None:
    rows = acquisition.build_inventory(CONFIG)
    forbidden = ("MOM1", "MOM2", "CUBE", "VROT", "VELOCITY")
    assert not any(token in row["id"].upper() for row in rows for token in forbidden)


def test_transport_overrun_is_disclosed_exactly() -> None:
    transport = CONFIG["transport_accounting"]
    assert transport["successful_source_gets"] == 21
    assert transport["failed_redirect_gets"] == 1
    assert transport["total_get_attempts"] == 22
    assert transport["total_network_body_bytes"] == 74_030_652
    assert transport["redirects_followed"] == 0
    assert transport["retries"] == 0


def test_receipt_is_deterministic_and_has_no_scoring() -> None:
    first = acquisition.build_receipt(CONFIG)
    second = acquisition.build_receipt(copy.deepcopy(CONFIG))
    assert first == second
    assert first["access_state"]["response_rows_read"] == 0
    assert first["access_state"]["scores_computed"] == 0
    assert first["access_state"]["models_fit"] == 0


def test_receipt_forgery_rejects_even_when_rehashed() -> None:
    forged = acquisition.build_receipt(CONFIG)
    forged["decision"] = "PUBLICATION_READY"
    forged.pop("content_sha256")
    forged["content_sha256"] = acquisition.content_sha256(forged)
    with pytest.raises(acquisition.SourceAcquisitionError, match="exactly rebuild"):
        acquisition.validate_receipt(forged, CONFIG)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("status",), "CONFIRMED"),
        (("inventory_contract", "file_count"), 20),
        (("transport_accounting", "failed_redirect_gets"), 0),
        (("transport_accounting", "total_get_attempts"), 21),
        (("scientific_boundary", "response_rows_opened"), 1),
        (("scientific_boundary", "development_only"), False),
        (("claims", "publication_ready"), True),
    ],
)
def test_material_config_mutations_fail_closed(path: tuple[str, ...], value: object) -> None:
    forged = copy.deepcopy(CONFIG)
    target = forged
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(acquisition.SourceAcquisitionError):
        acquisition.validate_config(forged)


def test_cli_has_no_arbitrary_path_arguments() -> None:
    with pytest.raises(SystemExit):
        acquisition._parser().parse_args(["check", "--output", "attacker.json"])
