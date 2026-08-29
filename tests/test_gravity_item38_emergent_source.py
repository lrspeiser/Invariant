import io
import tarfile
from pathlib import Path

import pytest

from sigma_theory_compiler.gravity_item38_emergent_gravity import GravityItem38Error, load_config
from sigma_theory_compiler.gravity_item38_emergent_source import (
    _role_for_member,
    register_source_headers,
    verify_scientific_freeze,
)

ROOT = Path(__file__).resolve().parents[1]


def _write_test_tar(path: Path, names: list[str]) -> None:
    with tarfile.open(path, mode="w:") as bundle:
        for name in names:
            payload = b"99 88 77\n"
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            bundle.addfile(info, io.BytesIO(payload))


def _required_names() -> list[str]:
    return [
        "Fig-9_RAR-KiDS-isolated_Massbin-1.txt",
        "Fig-9_RAR-KiDS-isolated_Massbin-2.txt",
        "Fig-9_RAR-KiDS-isolated_Massbin-3.txt",
        "Fig-9_RAR-KiDS-isolated_Massbin-4.txt",
        "Fig-8_RAR-KiDS-isolated_Colorbin_1.txt",
        "Fig-8_RAR-KiDS-isolated_Colorbin_2.txt",
        "Fig-9_RAR-KiDS-isolated_Massbins_covmatrix.txt",
        "Fig-8_RAR-KiDS-isolated_Colorbins_covmatrix.txt",
        "README.txt",
        "unused.txt",
        "._Fig-9_RAR-KiDS-isolated_Massbin-1.txt",
    ]


def test_item38_scientific_freeze_is_bound_and_replays() -> None:
    config = load_config(ROOT)
    assert len(config["scientific_freeze_commit"]) == 40
    verify_scientific_freeze(ROOT, config)


def test_item38_role_rules_keep_confirmation_sealed_and_transfers_separate() -> None:
    assert _role_for_member("Fig-9_RAR-KiDS-isolated_Massbin-1.txt") == "exploration"
    assert (
        _role_for_member("Fig-9_RAR-KiDS-isolated_Massbin-4.txt")
        == "sealed_confirmation"
    )
    assert (
        _role_for_member("Fig-8_RAR-KiDS-isolated_Colorbin_1.txt")
        == "unchanged_color_transfer"
    )
    assert _role_for_member("._Fig-9_RAR-KiDS-isolated_Massbin-1.txt") is None


def test_item38_header_registration_records_no_payload_access(tmp_path: Path) -> None:
    archive = tmp_path / "source.tar"
    _write_test_tar(archive, _required_names())
    source_path = tmp_path / "source.json"
    sample_path = tmp_path / "sample.json"
    source, sample = register_source_headers(
        ROOT, archive, source_path, sample_path
    )
    assert source["header_registration_only"] is True
    assert source["semantic_member_payload_bytes_read"] == 0
    assert all(not row["member_payload_opened"] for row in source["members"])
    assert len(sample["exploration"]) == 3
    assert len(sample["sealed_confirmation"]) == 1
    assert sample["confirmation_access_budget"] == 0
    assert sample["appledouble_ignored_count"] == 1


def test_item38_header_registration_fails_closed_if_a_role_is_missing(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "source.tar"
    names = _required_names()
    names.remove("Fig-9_RAR-KiDS-isolated_Massbin-4.txt")
    _write_test_tar(archive, names)
    with pytest.raises(GravityItem38Error, match="sealed_confirmation"):
        register_source_headers(
            ROOT,
            archive,
            tmp_path / "source.json",
            tmp_path / "sample.json",
        )
