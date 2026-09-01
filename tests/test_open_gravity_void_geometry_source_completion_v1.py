from __future__ import annotations

import copy

import pytest

from sigma_theory_compiler import open_gravity_void_geometry_source_completion_v1 as source


def test_mask_hashes_match_published_release() -> None:
    config = source.load_config()
    mask = config["mask"]
    path = source.Path(mask["local_path"])
    assert path.stat().st_size == 65031
    assert source.sha256_file(path) == mask["sha256"]
    assert source.md5_file(path) == "26795848b686186cf2aa09708097bfad"


def test_pickle_is_inspected_without_executing_unlisted_globals() -> None:
    config = source.load_config()
    structure = source.safe_pickle_structure(source.Path(config["mask"]["local_path"]))
    assert structure["mask_shape"] == [360, 180]
    assert structure["mask_dtype"] == "bool"
    assert structure["true_pixels"] == 9133
    assert structure["mask_resolution"] == 1
    assert structure["dist_limits_h_inverse_mpc"] == [0.0, 332.3856506347656]


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("status", "CONFIRMED"),
        ("output_path", "attacker.json"),
    ],
)
def test_config_mutations_reject(key: str, value: object) -> None:
    config = copy.deepcopy(source.load_config())
    config[key] = value
    with pytest.raises(source.VoidGeometrySourceError):
        source.validate_config(config)


def test_receipt_is_deterministic_and_zero_response() -> None:
    first = source.build_receipt()
    second = source.build_receipt()
    assert first == second
    assert first["content_sha256"] == source._self_hash(first)
    assert first["access_accounting"]["scientific_rows_decoded"] == 0
    assert first["access_accounting"]["response_values_inspected"] == 0
    assert first["claim_boundary"]["real_data_fit"] is False


def test_written_receipt_replays() -> None:
    assert source.check_receipt() == source.build_receipt()
