from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import pytest

from sigma_theory_compiler import open_gravity_void_correlation_development_release_v4 as release


def _source_paths() -> set[Path]:
    config = json.loads(release.CONFIG_PATH.read_text(encoding="utf-8"))
    return {(release.REPO_ROOT / source["path"]).resolve() for source in config["sources"].values()}


def test_blocked_v1_through_v3_are_preserved_byte_exact() -> None:
    config = release.load_config()
    release.validate_blocked_v3(config)
    for section in config["blocked_v3"].values():
        assert release.v2.canonical_file(section["path"]).read_bytes() == release.v2.canonical_file(
            section["preserved_path"]
        ).read_bytes()
    release.v3.validate_blocked_v2(release.v3.load_config())


def test_build_and_check_never_resolve_or_open_scientific_source(monkeypatch: pytest.MonkeyPatch) -> None:
    sources = _source_paths()
    original_open = Path.open
    original_canonical = release.v2.canonical_file
    resolved: list[Path] = []

    def guarded_open(path: Path, *args, **kwargs):
        if path.resolve() in sources:
            raise AssertionError(f"scientific source opened: {path}")
        return original_open(path, *args, **kwargs)

    def guarded_canonical(relative: str) -> Path:
        candidate = (release.REPO_ROOT / relative).resolve()
        if candidate in sources:
            resolved.append(candidate)
            raise AssertionError(f"scientific source resolved: {candidate}")
        return original_canonical(relative)

    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(release.v2, "canonical_file", guarded_canonical)
    receipt = release.build_receipt()
    assert all(value == 0 for value in receipt["access_accounting"].values())
    assert resolved == []


def test_no_exported_secret_constructor_gate_or_promotion_surface() -> None:
    forbidden = {
        "_OWNED_RUN_SECRET",
        "_OwnedDevelopmentRun",
        "_promote_fixed_payloads",
        "promote_fixed_package",
        "final_write",
    }
    assert not forbidden & set(vars(release))
    assert release.run_development_once.__closure__ is None
    assert list(inspect.signature(release.run_development_once).parameters) == []
    with pytest.raises(TypeError):
        release.run_development_once({"artifacts": {}})
    with pytest.raises(TypeError):
        release.run_development_once(object())


def test_private_final_write_is_nested_no_argument_single_call_after_consumption() -> None:
    structure = release._run_structure()
    assert structure == {
        "one_nested_final_write": True,
        "final_write_no_arguments": True,
        "local_unforgeable_capability_after_consumption": True,
        "single_final_write_call": True,
        "blocked_v3_surfaces_absent": True,
    }
    source = inspect.getsource(release.run_development_once)
    assert source.index("_consume_fixed_authorization") < source.index("final_write_capability = object()")
    assert source.index("final_write_capability = object()") < source.index("def final_write()")
    assert "unspent_final_write_capability is final_write_capability" in source
    assert "return final_write()" in source
    assert "v3._promote_fixed_payloads" not in source
    assert "v3._OWNED_RUN_SECRET" not in source
    assert "v3._OwnedDevelopmentRun" not in source


def test_zero_open_promotion_bypass_is_impossible_and_missing_audit_stops_first(
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
    assert not release.FINAL_DIRECTORY.exists()
    assert not release.CONSUMPTION_DIRECTORY.exists()
    assert not release.FAILURE_DIRECTORY.exists()


def test_write_boundary_rechecks_every_frozen_identity_and_path() -> None:
    source = inspect.getsource(release.run_development_once)
    required = (
        "frozen_mapping_identity == id(generated_payloads)",
        "frozen_payload_identities ==",
        "frozen_payload_hashes ==",
        "check_receipt() == contract_receipt",
        "validate_authorization_bytes(authorization_payload, contract_receipt) == authorization",
        "audit_path.read_bytes() == authorization_payload",
        "marker == _consumption_marker",
        "marker_path == CONSUMPTION_DIRECTORY",
        "marker_path.is_file()",
        "marker_path.read_bytes() == _pretty(marker)",
        "validate_code_pins()",
        "validate_blocked_v3(load_config())",
        "validate_package_payloads_v4",
        "_validate_fixed_directory",
    )
    assert all(token in source for token in required)


def test_no_caller_payload_or_path_writer_can_target_final_package() -> None:
    assert "_atomic_no_clobber" not in vars(release)
    assert list(inspect.signature(release._write_contract_receipt).parameters) == []
    source = inspect.getsource(release._write_contract_receipt)
    assert "OUTPUT_PATH" in source
    assert "FINAL_DIRECTORY" not in source
    assert "PROMOTED_COMPLETE" not in source


def test_final_promotion_primitive_exists_only_in_owned_runner_ast() -> None:
    tree = ast.parse(Path(release.MODULE_PATH).read_text(encoding="utf-8"))
    owners: set[str] = set()
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        has_final_primitive = any(
            (isinstance(descendant, ast.Attribute) and descendant.attr == "MoveFileExW")
            or (
                isinstance(descendant, ast.Call)
                and isinstance(descendant.func, ast.Attribute)
                and isinstance(descendant.func.value, ast.Name)
                and descendant.func.value.id == "os"
                and descendant.func.attr == "rename"
            )
            or (isinstance(descendant, ast.Constant) and descendant.value == "PROMOTED_COMPLETE")
            for descendant in ast.walk(node)
        )
        if has_final_primitive:
            owners.add(node.name)
    assert owners == {"run_development_once"}


def test_small_pcg64_regeneration_and_all_ones_rejection_are_retained() -> None:
    rows = release.v3._synthetic_permutation_rows()
    exact = release.regenerate_permutations_from_rows(rows, 4)
    reference = release.v1._permutation_reference(rows, 4)
    assert exact["permutation_statistics"] == reference["permutation_statistics"]
    forged_statistics = [1.0] * 4
    tail = sum(value >= exact["observed"] for value in forged_statistics)
    forged = {
        "observed": exact["observed"],
        "permutation_statistics": forged_statistics,
        "tail_count": tail,
        "p_value": (1 + tail) / 5,
    }
    with pytest.raises(release.v3.DevelopmentReleaseV3Error, match="statistic mismatch"):
        release.v3._exact_validate_regenerated(
            rows,
            forged,
            exact["order_hashes"],
            permutations=4,
        )


def test_final_validator_still_requires_exact_10000_regeneration() -> None:
    source = inspect.getsource(release.validate_package_payloads_v4)
    assert "permutations=_PERMUTATIONS" in source
    assert release._PERMUTATIONS == 10000
    assert release._SEED == 902104729
    assert release.v3.np.__version__ == "2.2.6"


def test_runtime_monkeypatch_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(release, "run_development_once", lambda: "FORGED_SUCCESS")
    with pytest.raises(release.DevelopmentReleaseV4Error, match="runtime identity drift"):
        release.validate_code_pins()


def test_config_and_rehashed_receipt_mutations_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    original = release.file_sha256
    monkeypatch.setattr(
        release,
        "file_sha256",
        lambda path: "0" * 64 if path == release.CONFIG_PATH else original(path),
    )
    with pytest.raises(release.DevelopmentReleaseV4Error, match="config raw drift"):
        release.load_config()
    monkeypatch.undo()
    receipt = release.check_receipt()
    receipt["decision"] = "REHASHED_FORGERY"
    receipt["content_sha256"] = release._self_hash(receipt)
    path = tmp_path / "receipt.json"
    path.write_bytes(release._pretty(receipt))
    monkeypatch.setattr(release, "OUTPUT_PATH", path)
    with pytest.raises(release.DevelopmentReleaseV4Error, match="receipt drift"):
        release.check_receipt()


def test_cli_exposes_source_free_commands_only() -> None:
    with pytest.raises(SystemExit):
        release.main(["run-development"])
