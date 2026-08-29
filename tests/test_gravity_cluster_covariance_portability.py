from __future__ import annotations

import copy
import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from sigma_theory_compiler import gravity_cluster_covariance_portability as portability

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def receipt() -> dict[str, object]:
    return portability.build_portable_receipt(ROOT)


def test_portable_check_requires_no_external_archive_or_work_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = portability._file_sha
    observed: list[Path] = []

    def reject_work(path: Path) -> str:
        resolved = path.resolve()
        relative = resolved.relative_to(ROOT)
        assert relative.parts[0] != "work"
        assert not resolved.name.endswith(".tar.gz")
        observed.append(relative)
        return original(path)

    monkeypatch.setattr(portability, "_file_sha", reject_work)
    receipt = portability.build_portable_receipt(ROOT)
    assert receipt["decision"] == (
        "PASS_PORTABLE_INTEGRITY_EXTERNAL_ARCHIVE_NOT_REQUIRED_OR_INCLUDED"
    )
    assert receipt["counts"]["external_archive_files_read"] == 0
    assert observed


def test_portable_receipt_binds_all_frozen_evidence(receipt: dict[str, object]) -> None:
    assert receipt["counts"] == {
        "frozen_v1_files_verified": 6,
        "receipt_content_hashes_verified": 2,
        "tracked_standalone_pressure_files_verified": 8,
        "external_covariance_members_manifested": 8,
        "external_archive_files_read": 0,
        "scientific_payload_rows_read": 0,
        "scientific_scores_computed": 0,
        "network_calls": 0,
    }
    assert receipt["lineage"] == {
        "reconstruction_decision": (
            "DEVELOPMENT_PRESSURE_COVARIANCE_PILOT_RECONSTRUCTIBLE_CP5_REMAINS_PARTIAL"
        ),
        "scoring_decision": "FAIL_FROZEN_PRESSURE_RANKING_ROBUSTNESS",
        "CP5_1_status": (
            "DEVELOPMENT_PRESSURE_COVARIANCE_SCORED_NOT_COMPONENT_COMPLETE"
        ),
        "reconstructed_matrices": 8,
        "scored_pressure_rows": 54,
    }
    assert not receipt["claims"]["archive_license_verified"]
    assert not receipt["claims"]["archive_redistribution_allowed"]
    assert not receipt["claims"]["portable_package_is_full_replay_complete"]
    assert not receipt["claims"]["scientific_rescoring_performed"]


def test_six_v1_artifacts_remain_byte_exact() -> None:
    config = portability.load_config(ROOT)
    for row in config["frozen_v1_bindings"]:
        path = ROOT / row["path"]
        assert portability._file_sha(path) == row["file_sha256"]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["covariance_members"][0].__setitem__("sha256", "0" * 64),
        lambda value: value["standalone_pressure_files"][0].__setitem__(
            "sha256", "0" * 64
        ),
        lambda value: value["frozen_v1_bindings"][2].__setitem__(
            "file_sha256", "0" * 64
        ),
        lambda value: value["external_archive_contract"].__setitem__(
            "included_in_portable_package", True
        ),
        lambda value: value["claim_boundary"].__setitem__(
            "archive_license_verified", True
        ),
    ],
)
def test_member_hash_result_and_license_mutations_fail_closed(mutation: object) -> None:
    config = copy.deepcopy(portability.load_config(ROOT))
    mutation(config)  # type: ignore[operator]
    with pytest.raises(portability.GravityClusterCovariancePortabilityError):
        portability.validate_config(config)


def test_rehashed_result_boundary_tampering_is_rejected() -> None:
    config = portability.load_config(ROOT)
    paths, receipts = portability._load_bound_files(ROOT, config)
    tampered = copy.deepcopy(receipts)
    tampered["SCORING_RECEIPT"]["decision"] = "PASS"
    with pytest.raises(
        portability.GravityClusterCovariancePortabilityError,
        match="result boundary",
    ):
        portability._validate_receipt_lineage(config, paths, tampered)


def test_optional_archive_preflight_checks_archive_and_member_hashes(
    tmp_path: Path,
) -> None:
    payloads = {"A/a.fits": b"alpha", "B/b.fits": b"beta"}
    archive_path = tmp_path / "source.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        for name, payload in payloads.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    members = [
        {
            "cluster": name.split("/", 1)[0],
            "member": name,
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        for name, payload in payloads.items()
    ]
    result = portability.inspect_external_archive(
        archive_path,
        portability._file_sha(archive_path),
        archive_path.stat().st_size,
        members,
    )
    assert result["decision"] == (
        "PASS_EXTERNAL_ARCHIVE_PREFLIGHT_FULL_REPLAY_NOT_EXECUTED"
    )
    assert result["scientific_rows_parsed"] == 0
    assert not result["full_replay_executed"]
    assert not result["redistribution_rights_verified"]

    changed = copy.deepcopy(members)
    changed[0]["sha256"] = "0" * 64
    with pytest.raises(
        portability.GravityClusterCovariancePortabilityError,
        match="member changed",
    ):
        portability.inspect_external_archive(
            archive_path,
            portability._file_sha(archive_path),
            archive_path.stat().st_size,
            changed,
        )


def test_stored_receipt_rebuilds_exactly() -> None:
    stored = json.loads((ROOT / portability.OUTPUT_PATH).read_text(encoding="utf-8"))
    portability.validate_receipt(stored, ROOT)
    assert stored == portability.build_portable_receipt(ROOT)
