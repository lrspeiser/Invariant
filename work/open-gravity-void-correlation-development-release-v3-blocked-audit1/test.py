from __future__ import annotations

import gzip
import hashlib
import inspect
import io
import json
from pathlib import Path

import pytest

from sigma_theory_compiler import open_gravity_void_correlation_development_release_v3 as release


def _source_paths() -> set[Path]:
    return {
        (release.REPO_ROOT / source["path"]).resolve()
        for source in json.loads(release.CONFIG_PATH.read_text(encoding="utf-8"))["sources"].values()
    }


def test_blocked_v1_and_v2_packets_are_byte_exact() -> None:
    config = release.load_config()
    release.validate_blocked_v2(config)
    for section in config["blocked_v2"].values():
        assert release.v2.canonical_file(section["path"]).read_bytes() == release.v2.canonical_file(
            section["preserved_path"]
        ).read_bytes()
    release.v2.validate_blocked_v1(release.v2.load_config())


def test_build_check_resolve_or_open_no_scientific_source(monkeypatch: pytest.MonkeyPatch) -> None:
    sources = _source_paths()
    original_open = Path.open
    original_canonical = release.v2.canonical_file
    resolutions: list[Path] = []

    def guarded_open(path: Path, *args, **kwargs):
        if path.resolve() in sources:
            raise AssertionError(f"scientific source opened: {path}")
        return original_open(path, *args, **kwargs)

    def guarded_canonical(relative: str) -> Path:
        candidate = (release.REPO_ROOT / relative).resolve()
        if candidate in sources:
            resolutions.append(candidate)
            raise AssertionError(f"scientific source resolved: {candidate}")
        return original_canonical(relative)

    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(release.v2, "canonical_file", guarded_canonical)
    receipt = release.build_receipt()
    assert all(value == 0 for value in receipt["access_accounting"].values())
    assert resolutions == []


def test_sole_production_runner_accepts_no_inputs_or_reported_artifacts() -> None:
    assert list(inspect.signature(release.run_development_once).parameters) == []
    assert list(inspect.signature(release._OwnedDevelopmentRun.execute).parameters) == ["self"]
    with pytest.raises(TypeError):
        release.run_development_once({"caller_reported": "artifact"})
    with pytest.raises(release.DevelopmentReleaseV3Error, match="constructor is not authorized"):
        release._OwnedDevelopmentRun(None, b"", {}, {}, {})


def test_missing_v3_audit_stops_before_source_even_though_v2_gate_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = _source_paths()
    original = release.v2.canonical_file
    observed: list[str] = []

    def guarded(relative: str) -> Path:
        observed.append(relative)
        assert (release.REPO_ROOT / relative).resolve() not in sources
        return original(relative)

    monkeypatch.setattr(release.v2, "canonical_file", guarded)
    with pytest.raises(release.v2.DevelopmentReleaseV2Error, match="missing"):
        release.run_development_once()
    assert release.load_config()["authorization_contract"]["future_path"] in observed
    assert not any((release.REPO_ROOT / relative).resolve() in sources for relative in observed)
    runner_names = set(release.run_development_once.__code__.co_names) | set(
        release._OwnedDevelopmentRun.execute.__code__.co_names
    )
    assert not {"OneShotDevelopmentGate", "source_open", "cf4_row"} & runner_names


def test_runner_source_ownership_and_private_counter_surface_is_structural() -> None:
    source = inspect.getsource(release._OwnedDevelopmentRun)
    assert "path.open(\"rb\")" in source
    assert "self._counts" in source
    assert "OneShotDevelopmentGate" not in source
    assert "scientific_decoded" not in source
    assert "artifacts:" not in inspect.getsource(release.run_development_once)


def test_hashing_reader_hashes_exact_compressed_bytes() -> None:
    compressed = gzip.compress(b"one\ntwo\n", mtime=0)
    raw = io.BytesIO(compressed)
    reader = release._HashingReader(raw)
    with gzip.GzipFile(fileobj=reader, mode="rb") as stream:
        assert list(stream) == [b"one\n", b"two\n"]
    assert reader.bytes_read == len(compressed)
    assert reader.digest.hexdigest() == hashlib.sha256(compressed).hexdigest()


def test_small_exact_pcg64_regeneration_matches_frozen_reference() -> None:
    rows = release._synthetic_permutation_rows()
    regenerated = release.regenerate_permutations_from_rows(rows, 4)
    reference = release.v1._permutation_reference(rows, 4)
    assert regenerated["observed"] == reference["observed"]
    assert regenerated["permutation_statistics"] == reference["permutation_statistics"]
    assert regenerated["tail_count"] == reference["tail_count"]
    assert regenerated["p_value"] == reference["p_value"]
    assert len(regenerated["order_hashes"]) == 4
    assert regenerated["order_root_sha256"] == release._order_root(regenerated["order_hashes"])


def test_coherent_all_ones_permutation_forgery_is_rejected() -> None:
    rows = release._synthetic_permutation_rows()
    exact = release.regenerate_permutations_from_rows(rows, 4)
    forged_statistics = [1.0] * 4
    tail = sum(value >= exact["observed"] for value in forged_statistics)
    forged = {
        "observed": exact["observed"],
        "permutation_statistics": forged_statistics,
        "tail_count": tail,
        "p_value": (1 + tail) / 5,
    }
    with pytest.raises(release.DevelopmentReleaseV3Error, match="statistic mismatch"):
        release._exact_validate_regenerated(
            rows,
            forged,
            exact["order_hashes"],
            permutations=4,
        )


def test_rehashed_order_forgery_is_rejected_even_with_exact_statistics() -> None:
    rows = release._synthetic_permutation_rows()
    exact = release.regenerate_permutations_from_rows(rows, 3)
    claimed = {
        key: exact[key]
        for key in ("observed", "permutation_statistics", "tail_count", "p_value")
    }
    order_hashes = list(exact["order_hashes"])
    order_hashes[0] = "0" * 64
    with pytest.raises(release.DevelopmentReleaseV3Error, match="order-hash mismatch"):
        release._exact_validate_regenerated(rows, claimed, order_hashes, permutations=3)


def test_final_validator_hardcodes_all_10000_regenerations() -> None:
    source = inspect.getsource(release.validate_package_payloads_v3)
    assert "permutations=_PERMUTATIONS" in source
    assert release._PERMUTATIONS == 10000
    assert release._SEED == 902104729
    assert release.np.__version__ == "2.2.6"


def test_config_and_receipt_mutations_fail_closed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    original = release.file_sha256
    monkeypatch.setattr(
        release,
        "file_sha256",
        lambda path: "0" * 64 if path == release.CONFIG_PATH else original(path),
    )
    with pytest.raises(release.DevelopmentReleaseV3Error, match="config raw drift"):
        release.load_config()
    monkeypatch.undo()
    receipt = release.check_receipt()
    receipt["decision"] = "REHASHED_MUTATION"
    receipt["content_sha256"] = release._self_hash(receipt)
    path = tmp_path / "receipt.json"
    path.write_bytes(release._pretty(receipt))
    monkeypatch.setattr(release, "OUTPUT_PATH", path)
    with pytest.raises(release.DevelopmentReleaseV3Error, match="receipt drift"):
        release.check_receipt()


def test_owned_runtime_monkeypatch_fails_before_authorization(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(release, "regenerate_permutations_from_rows", lambda *args, **kwargs: {})
    with pytest.raises(release.DevelopmentReleaseV3Error, match="owned runtime identity drift"):
        release.validate_code_pins()


def test_cli_is_source_free_only() -> None:
    with pytest.raises(SystemExit):
        release.main(["run-development"])
