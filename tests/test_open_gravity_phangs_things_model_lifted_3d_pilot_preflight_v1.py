from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

from sigma_theory_compiler import (
    open_gravity_phangs_things_model_lifted_3d_pilot_preflight_v1 as pilot,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / pilot.CONFIG_PATH).read_text(encoding="utf-8"))


def test_config_and_exact_source_intersection() -> None:
    pilot.validate_config(CONFIG)
    selection = CONFIG["selection_contract"]
    assert selection["eligible_objects"] == ["NGC2903", "NGC3351", "NGC3627"]
    assert selection["eligible_count"] == 3
    assert selection["reserved_confirmation_objects_opened"] == 0
    assert selection["selection_used_rotation_values"] is False


def test_exact_file_roles_counts_and_bytes() -> None:
    rows = CONFIG["source_files"]
    assert len(rows) == 21
    assert len({row["url"] for row in rows}) == 21
    assert sum(row["bytes"] for row in rows) == 74_030_400
    for object_id in ("NGC2903", "NGC3351", "NGC3627"):
        subset = [row for row in rows if row["object_id"] == object_id]
        assert len(subset) == 7
        assert {row["role"] for row in subset} == pilot._EXPECTED_ROLES


def test_things_urls_are_direct_single_get_endpoints() -> None:
    things = [row for row in CONFIG["source_files"] if row["survey"] == "THINGS"]
    assert len(things) == 6
    assert all(row["url"].startswith("https://things.www3.mpia.de/Data_files/") for row in things)
    assert all("www2.mpia-hd.mpg.de" not in row["url"] for row in things)


@pytest.mark.parametrize("token", ["MOM1", "MOM2", "CUBE", "VROT", "VELOCITY"])
def test_response_bearing_source_url_rejects(token: str) -> None:
    forged = copy.deepcopy(CONFIG)
    forged["source_files"][0]["url"] += token
    with pytest.raises(pilot.PilotPreflightError, match="config semantics changed"):
        pilot.validate_config(forged)


def test_photometric_inclinations_are_independently_recomputed() -> None:
    expected = {
        "NGC2903": 62.69129057081534,
        "NGC3351": 43.44024716027847,
        "NGC3627": 54.583570652458924,
    }
    for row in CONFIG["object_metadata"]:
        actual = pilot.photometric_inclination_deg(row["s4g_outer_ellipticity"], 0.2)
        assert math.isclose(actual, expected[row["object_id"]], abs_tol=1e-12)


def test_inclination_rejects_invalid_geometry() -> None:
    with pytest.raises(pilot.PilotPreflightError):
        pilot.photometric_inclination_deg(-0.1, 0.2)
    with pytest.raises(pilot.PilotPreflightError):
        pilot.photometric_inclination_deg(0.2, 1.0)


def test_same_acceleration_fixture_separates_missing_variables() -> None:
    fixture = pilot._same_acceleration_fixture()
    rows = fixture["rows"]
    assert len({row["local_acceleration_m_s2"] for row in rows}) == 1
    potentials = [row["potential_proxy_abs_phi_over_c2"] for row in rows]
    curvatures = [row["curvature_proxy_s2"] for row in rows]
    assert potentials == sorted(potentials)
    assert curvatures == sorted(curvatures, reverse=True)
    assert potentials[-1] / potentials[0] > 1e7
    assert curvatures[0] / curvatures[-1] > 1e7


def test_receipt_is_deterministic_and_zero_access() -> None:
    first = pilot.build_receipt(CONFIG)
    second = pilot.build_receipt(copy.deepcopy(CONFIG))
    assert first == second
    pilot.validate_receipt(first, CONFIG)
    assert first["sources"]["file_count"] == 21
    assert first["sources"]["network_byte_ceiling"] == 74_030_400
    assert all(value == 0 for value in first["access_state"].values())


def test_receipt_forgery_rejects_even_when_rehashed() -> None:
    forged = pilot.build_receipt(CONFIG)
    forged["decision"] = "PUBLICATION_READY"
    forged.pop("content_sha256")
    forged["content_sha256"] = pilot.content_sha256(forged)
    with pytest.raises(pilot.PilotPreflightError, match="exactly rebuild"):
        pilot.validate_receipt(forged, CONFIG)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("status",), "CONFIRMED"),
        (("selection_contract", "eligible_count"), 4),
        (("selection_contract", "reserved_confirmation_objects_opened"), 1),
        (("future_source_acquisition", "exact_get_ceiling"), 22),
        (("source_transform", "geometry_is_observed_3d"), True),
        (("response_boundary", "fresh_confirmation_claim_allowed"), True),
        (("construction_incident", "occurred"), False),
        (("claims", "publication_ready"), True),
        (("access_state", "scores_computed"), 1),
    ],
)
def test_material_config_mutations_fail_closed(path: tuple[str, ...], value: object) -> None:
    forged = copy.deepcopy(CONFIG)
    target = forged
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(pilot.PilotPreflightError):
        pilot.validate_config(forged)


def test_development_only_response_boundary_is_explicit() -> None:
    boundary = CONFIG["response_boundary"]
    incident = CONFIG["construction_incident"]
    assert boundary["response_was_opened_in_prior_development_work"] is True
    assert boundary["fresh_confirmation_claim_allowed"] is False
    assert boundary["current_preflight_response_rows_opened"] == 0
    assert incident["occurred"] is True
    assert "development-only" in incident["consequence"]


def test_source_transform_has_no_single_response_selected_thickness() -> None:
    transform = CONFIG["source_transform"]
    assert transform["geometry_label"] == "MODEL_LIFTED_2P5D_TO_3D"
    assert len(transform["stellar_height_over_radial_scale_cells"]) == 3
    assert len(transform["gas_height_pc_cells"]) == 3
    assert "score every Cartesian" in transform["cell_policy"]


def test_load_config_and_canonical_receipt_after_seal() -> None:
    loaded = pilot.load_config()
    assert loaded == CONFIG
    assert pilot.check_receipt() == "VALID"


def test_cli_does_not_offer_arbitrary_read_or_write_paths() -> None:
    parser = pilot._parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["check", "--output", "attacker.json"])
