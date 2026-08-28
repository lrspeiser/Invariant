from __future__ import annotations

import copy
import inspect
import json
from collections import Counter
from pathlib import Path

import pytest

import sigma_theory_compiler.gravity_item2_axes_group_geometry as groups
from sigma_theory_compiler.sigma_core import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / groups.SAMPLE_MANIFEST_PATH


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _reseal(value: dict[str, object]) -> dict[str, object]:
    value.pop("content_sha256", None)
    value["content_sha256"] = canonical_sha256(value)
    return value


def test_attempt_five_is_frozen_before_member_redshifts() -> None:
    config = groups.load_config(ROOT)
    assert config["status"] == "frozen_before_selected_member_redshift_access"
    assert config["authorization"]["selected_exploration_member_rows_allowed"] is True
    assert config["authorization"]["reserved_confirmation_member_rows_allowed"] is False
    assert config["authorization"]["published_group_velocity_columns_allowed"] is False
    assert config["target_blind_sample"]["reserved_confirmation_target_accesses_allowed"] == 0
    assert config["authorization"]["paid_model_calls_allowed"] is False


def test_metadata_query_contains_no_published_dynamics_or_xray_target() -> None:
    config = groups.load_config(ROOT)
    source = config["catalog_sources"]
    assert source["metadata_allowed_columns"] == ["Group", "Nmemb", "zsp", "LR195", "D10"]
    assert set(source["metadata_forbidden_columns"]) == {
        "sigmaGAP",
        "e_sigmaGAP",
        "sigmaMAD",
        "R200c",
    }
    assert "tablec2" not in source["metadata_query_url"]
    assert "tablec3" not in source["metadata_query_url"]
    text = (ROOT / source["metadata_path"]).read_text(encoding="utf-8")
    for forbidden in source["metadata_forbidden_columns"]:
        assert f"#Column\t{forbidden}\t" not in text


def test_sample_is_deterministic_balanced_and_target_blind() -> None:
    config = groups.load_config(ROOT)
    manifest = _load(SAMPLE)
    groups.validate_sample_manifest(manifest, config=config)
    assert groups.build_sample_manifest(ROOT) == manifest
    assert manifest["selection_boundary"] == {
        "metadata_endpoint_queries": 1,
        "published_group_velocity_columns_read": 0,
        "selected_member_rows_opened": 0,
        "selected_member_redshifts_read": 0,
        "reserved_confirmation_target_accesses": 0,
        "xray_target_columns_read": 0,
    }
    objects = manifest["objects"]
    assert len(objects) == 270
    assert len({row["group"] for row in objects}) == 270
    assert Counter(row["role"] for row in objects) == {
        "exploration": 180,
        "reserved_confirmation": 90,
    }
    assert Counter((row["richness_bin"], row["role"]) for row in objects) == {
        (richness_bin, role): count
        for richness_bin in range(3)
        for role, count in (("exploration", 60), ("reserved_confirmation", 30))
    }


def test_selection_code_has_no_member_or_velocity_input() -> None:
    signature = inspect.signature(groups.eligible_metadata_rows)
    assert tuple(signature.parameters) == ("rows", "config")
    source = inspect.getsource(groups.build_sample_manifest)
    assert "member_query" not in source
    assert "sigmaGAP" not in source
    assert "sigmaMAD" not in source
    assert "R200c" not in source


@pytest.mark.parametrize(
    "claim",
    [
        "alternative_to_gr_established",
        "confirmation_opened",
        "group_finder_independence_established",
        "member_response_seen_during_selection",
        "roadmap_item_2_complete",
    ],
)
def test_resealed_sample_overclaim_is_rejected(claim: str) -> None:
    config = groups.load_config(ROOT)
    manifest = copy.deepcopy(_load(SAMPLE))
    manifest["claims"][claim] = True
    with pytest.raises(groups.GravityItem2AxesGroupError):
        groups.validate_sample_manifest(_reseal(manifest), config=config)


def test_config_admits_membership_provenance_limitation() -> None:
    config = groups.load_config(ROOT)
    limitations = " ".join(config["provenance_limitations"])
    assert "FoF" in limitations
    assert "Clean algorithm" in limitations
    assert "redshifts and mass-model assumptions" in limitations
    assert config["claim_boundaries"]["group_finder_independence_established"] is False
