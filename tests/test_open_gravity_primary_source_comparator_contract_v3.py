from __future__ import annotations

import copy
from pathlib import Path

import pytest

from sigma_theory_compiler import open_gravity_primary_source_comparator_contract_v3 as prior
from sigma_theory_compiler.open_gravity_primary_source_comparator_contract_v3 import (
    CONFIG_PATH,
    EXPECTED_CONFIG_CANONICAL_SHA256,
    EXPECTED_OUTPUT_PATH,
    EXPECTED_SECTION_SEALS,
    EXPECTED_UNSEALED_ROOT_SHA256,
    MODULE_PATH,
    OUTPUT_PATH,
    TEST_PATH,
    PriorArtResealError,
    _canonical_receipt_path,
    _sha256_bytes,
    _write_no_clobber,
    build_receipt,
    check_receipt,
    content_sha256,
    load_config,
    main,
    receipt_content_sha256,
    validate_config,
)

ROOT = Path(__file__).resolve().parents[1]


def test_config_is_exactly_sealed() -> None:
    config = load_config(ROOT)
    assert content_sha256(config) == EXPECTED_CONFIG_CANONICAL_SHA256
    assert config["section_seals"] == {
        **EXPECTED_SECTION_SEALS,
        "unsealed_root_sha256": EXPECTED_UNSEALED_ROOT_SHA256,
    }


@pytest.mark.parametrize("section", tuple(EXPECTED_SECTION_SEALS))
def test_every_section_mutation_fails_closed(section: str) -> None:
    config = copy.deepcopy(load_config(ROOT))
    value = config[section]
    if isinstance(value, list):
        value[0]["sha256"] = "0" * 64
    else:
        first = next(iter(value))
        if isinstance(value[first], bool):
            value[first] = not value[first]
        elif isinstance(value[first], str):
            value[first] += "_MUTATED"
        else:
            raise TypeError(section)
    with pytest.raises(PriorArtResealError, match="sealed section changed"):
        validate_config(config)


def test_coordinated_mutation_still_fails_hardcoded_seal() -> None:
    config = copy.deepcopy(load_config(ROOT))
    config["payload_contract"]["primary_source_count"] += 1
    config["section_seals"]["payload_contract"] = content_sha256(config["payload_contract"])
    unsealed = {key: value for key, value in config.items() if key != "section_seals"}
    config["section_seals"]["unsealed_root_sha256"] = content_sha256(unsealed)
    with pytest.raises(PriorArtResealError, match="sealed section changed"):
        validate_config(config)


def test_receipt_rebinds_final_registry_schema_and_gp01() -> None:
    receipt = build_receipt(ROOT)
    assert receipt["payload"] == {
        "path": "configs/open_gravity_primary_source_comparator_contract_v1.json",
        "file_sha256": "8cefb8bf6cf737759b646015ecd5f81857ffc8914049a28252557dad47b490a8",
        "semantic_content_sha256": receipt["payload"]["semantic_content_sha256"],
        "source_content_mutated": False,
        "primary_source_count": 31,
        "dynamical_comparator_count": 11,
        "ontology_prior_art_count": 13,
        "light_gravity_analogy_count": 13,
    }
    assert receipt["registry_rebind"]["commit"] == "74cf6412"
    assert receipt["registry_rebind"]["config_sha256"] == (
        "4b65e4dc919d51462ca78e47c9b1314aa2ac3cf5b8c158b2cc19d758a4214e0d"
    )
    assert receipt["registry_rebind"]["mechanism_schema_sha256"] == (
        "5c14dc4b4b5e5e457e80410f8e19cf92b575f527ef8b07a00958825c60605396"
    )
    assert receipt["gp01_rebind"]["commit"] == "35f70938"
    assert receipt["claim_boundary"]["campaign_execution_authority"] is False
    assert receipt["claim_boundary"]["eligible_as_final_prior_art_manifest_input"] is True
    assert receipt["canonical_path_enforcement"] == {
        "receipt_path": EXPECTED_OUTPUT_PATH,
        "check_path_override_accepted": False,
        "cli_output_override_accepted": False,
        "path_verified_before_receipt_read": True,
    }
    assert receipt["predecessor"]["status"] == (
        "BLOCKED_CANONICAL_PATH_AUDIT_COUNTEREVIDENCE_NO_AUTHORITY"
    )


def test_receipt_is_deterministic_and_self_hashed() -> None:
    first = build_receipt(ROOT)
    second = build_receipt(ROOT)
    assert first == second
    assert first["content_sha256"] == receipt_content_sha256(first)


def test_build_reads_only_exact_metadata_allowlist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(ROOT)
    allowed = {
        (ROOT / CONFIG_PATH).resolve(),
        (ROOT / MODULE_PATH).resolve(),
        (ROOT / TEST_PATH).resolve(),
    }
    for row in config["hard_bindings"]:
        path = Path(row["path"])
        allowed.add((path if path.is_absolute() else ROOT / path).resolve())
    original = Path.read_bytes
    opened: list[Path] = []

    def traced(path: Path) -> bytes:
        resolved = path.resolve()
        opened.append(resolved)
        assert resolved in allowed
        assert "response" not in resolved.name.lower()
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", traced)
    receipt = build_receipt(ROOT)
    assert set(opened) == allowed
    assert len(receipt["access_audit"]["allowlisted_metadata_files_opened"]) == len(allowed)
    assert all(
        value == 0
        for key, value in receipt["access_audit"].items()
        if key != "allowlisted_metadata_files_opened"
    )


def test_atomic_writer_refuses_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    _write_no_clobber(path, b"first\n")
    with pytest.raises(PriorArtResealError, match="refusing to overwrite"):
        _write_no_clobber(path, b"second\n")
    assert path.read_bytes() == b"first\n"


def test_bound_predecessor_and_final_files_have_frozen_hashes() -> None:
    config = load_config(ROOT)
    for row in config["hard_bindings"]:
        path = Path(row["path"])
        target = path if path.is_absolute() else ROOT / path
        assert _sha256_bytes(target.read_bytes()) == row["sha256"]


def test_check_is_hard_bound_to_canonical_path_before_any_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    trap = tmp_path / "attacker-selected-receipt.json"
    trap.write_text("{}", encoding="utf-8")
    with pytest.raises(TypeError):
        check_receipt(ROOT, trap)  # type: ignore[call-arg]

    monkeypatch.setattr(prior, "OUTPUT_PATH", Path("arbitrary/override.json"))

    def forbidden_read(*_args: object, **_kwargs: object) -> bytes:
        raise AssertionError("path mismatch must fail before any file read")

    monkeypatch.setattr(Path, "read_bytes", forbidden_read)
    monkeypatch.setattr(Path, "read_text", forbidden_read)
    with pytest.raises(PriorArtResealError, match="path constant changed"):
        check_receipt(ROOT)


@pytest.mark.parametrize("command", ["build", "check"])
def test_cli_rejects_arbitrary_output_path_without_reading_it(
    command: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    trap = tmp_path / "attacker-selected-receipt.json"
    trap.write_text("{}", encoding="utf-8")
    original_read_text = Path.read_text

    def guarded_read_text(path: Path, *args: object, **kwargs: object) -> str:
        assert path.resolve() != trap.resolve()
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    with pytest.raises(SystemExit) as error:
        main([command, "--output", str(trap)])
    assert error.value.code == 2


def test_canonical_receipt_path_is_exact() -> None:
    assert _canonical_receipt_path(ROOT) == ROOT / OUTPUT_PATH


def test_stored_receipt_matches_rebuild() -> None:
    assert check_receipt(ROOT) == build_receipt(ROOT)
