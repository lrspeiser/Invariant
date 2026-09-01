from __future__ import annotations

import copy
import inspect
import json

import pytest

from sigma_theory_compiler import (
    open_gravity_dissipative_capture_camels_source_preflight_v1 as subject,
)


def _raw_config() -> dict[str, object]:
    return json.loads(subject.CONFIG_PATH.read_text(encoding="utf-8"))


def test_exact_package_pins() -> None:
    assert subject.file_sha256(subject.CONFIG_PATH) == subject._CONFIG_RAW_SHA256
    assert subject.content_sha256(_raw_config()) == subject._CONFIG_CONTENT_SHA256
    assert subject.module_semantic_sha256() == subject._MODULE_SEMANTIC_SHA256
    assert subject.file_sha256(subject.TEST_PATH) == subject._TEST_RAW_SHA256


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("status",), "PASS"),
        (("eligibility_gates", "direct_nbody_groups"), True),
        (("eligibility_gates", "documented_cross_tree_object_matching"), True),
        (("eligibility_gates", "first_pericenter_timing_equivalent_to_tng100"), True),
        (("access_contract", "scientific_tree_or_group_rows_opened"), 1),
        (("access_contract", "new_real_data_scores"), 1),
        (("outputs", "receipt"), "elsewhere.json"),
    ],
)
def test_blocked_claim_mutations_reject(path: tuple[str, ...], value: object) -> None:
    config = copy.deepcopy(_raw_config())
    target = config
    for key in path[:-1]:
        target = target[key]  # type: ignore[index,assignment]
    target[path[-1]] = value  # type: ignore[index]
    with pytest.raises(subject.CamelsPreflightError):
        subject.validate_config(config)


def test_exact_candidate_grid_and_url_expansion() -> None:
    config = subject.load_config()
    snaps = [row["snap"] for row in config["candidate"]["snapshots"]]
    assert snaps == list(range(62, 91, 2))
    assert subject.snapshot_url(config, "hydro", 62).endswith("snapshot_062.hdf5")
    assert subject.snapshot_url(config, "nbody", 90).endswith("snapshot_090.hdf5")
    assert subject.group_url(config, "hydro", 72).endswith("groups_072.hdf5")
    assert "IllustrisTNG_DM" in subject.group_url(config, "nbody", 88)


def test_manifest_retains_every_direct_and_blocked_file() -> None:
    manifest = subject.source_manifest(subject.load_config())
    assert manifest["accessible_file_count"] == 50
    assert manifest["blocked_file_count"] == 15
    assert manifest["official_cryptographic_checksum_count"] == 0
    assert manifest["local_sha256_acquisition_count"] == 1
    assert all(row["http_status"] == 403 for row in manifest["blocked_files"])
    assert all(row["exact_bytes"] is None for row in manifest["blocked_files"])
    assert manifest["scientific_rows_opened"] == 0
    assert manifest["scientific_scores_computed"] == 0


def test_bounded_matching_acquisition_is_hash_bound_without_decode() -> None:
    config = subject.load_config()
    subject._validate_acquisition(config)
    acquisition = config["http_metadata"]["local_acquisitions"][0]
    assert acquisition["local_path"] == subject.SOURCE_PATH.as_posix()
    assert acquisition["bytes"] == 160096
    assert subject.file_sha256(subject.SOURCE_PATH) == acquisition["sha256"]
    assert config["access_contract"]["hdf5_structures_opened"] == 0


def test_required_fields_and_matching_boundary_are_explicit() -> None:
    config = subject.load_config()
    fields = config["field_mapping"]
    assert "PartType0/EnergyDissipation" in fields["gas_receiver_and_controls"]
    assert "PartType0/GFM_CoolingRate" in fields["gas_receiver_and_controls"]
    assert "PartType0/Machnumber" in fields["gas_receiver_and_controls"]
    assert "/IDs" in fields["particle_membership"]
    assert "not object-level matching" in fields["cross_run_matching"]


def test_cli_is_hard_bound_and_check_has_no_write_surface() -> None:
    assert not inspect.signature(subject.build).parameters
    assert not inspect.signature(subject.check).parameters
    with pytest.raises(SystemExit):
        subject.main(["check", "--output", "elsewhere.json"])
