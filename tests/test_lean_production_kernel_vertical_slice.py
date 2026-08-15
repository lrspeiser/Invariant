from __future__ import annotations

import copy
import json
import os
import shutil
from pathlib import Path

import pytest

from sigma_theory_compiler.lean_production_kernel_vertical_slice import (
    CONFIG_PATH,
    OUTPUT_PATH,
    THEOREM_PATH,
    _content_sha,
    _platform_key,
    _resolve_executable,
    build_live_receipt,
    validate_checked_receipt,
    validate_live_receipt,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def checked_receipt():
    value = json.loads((ROOT / OUTPUT_PATH).read_text(encoding="utf-8"))
    validate_checked_receipt(value, root=ROOT)
    return value


def _reseal(value):
    value["content_sha256"] = _content_sha(value)
    return value


def _live_environment() -> dict[str, str] | None:
    environment = dict(os.environ)
    if environment.get("INVARIANT_LEAN_EXECUTABLE") or shutil.which("lean"):
        return environment
    candidate = (
        Path.home()
        / ".cache"
        / "invariant"
        / "lean"
        / "v4.33.0"
        / "lean-4.33.0-windows"
        / "bin"
        / "lean.exe"
    )
    if candidate.is_file():
        environment["INVARIANT_LEAN_EXECUTABLE"] = str(candidate)
        return environment
    return None


def test_checked_windows_receipt_is_portable_historical_evidence(checked_receipt):
    assert checked_receipt["receipt_role"] == "checked_windows_historical"
    assert checked_receipt["decision"] == "pass_real_lean_kernel_vertical_slice"
    assert checked_receipt["toolchain_receipt"]["platform"] == "windows-x86_64"
    encoded = json.dumps(checked_receipt, sort_keys=True)
    assert "C:\\Users\\" not in encoded
    assert checked_receipt["toolchain_receipt"]["executable_path_persisted"] is False


def test_real_theorem_and_exact_dependency_closure_are_bound(checked_receipt):
    source = (ROOT / THEOREM_PATH).read_text(encoding="utf-8")
    assert "theorem kernelSmoke (n : Nat) : n = n := Eq.refl n" in source
    adapter = checked_receipt["adapter_receipt"]
    assert adapter["decision"] == "pass_lean_checked_closed_premise"
    assert adapter["execution"]["exit_code"] == 0
    assert adapter["dependency_audit"] == {
        "protocol_version": "INVARIANT_LEAN_DEPENDENCY_AUDIT_V1",
        "reported_target": "Invariant.kernelSmoke",
        "dependencies": ["Eq.refl"],
        "closure_valid": True,
    }


def test_registered_release_and_windows_identity_are_exact(checked_receipt):
    toolchain = checked_receipt["toolchain_receipt"]
    assert toolchain["official_release"] == "leanprover/lean4:v4.33.0"
    assert toolchain["commit"] == "d8b18978322de05a8f3dba51ef03cf5461676c17"
    assert toolchain["platform_asset"]["archive_sha256"] == (
        "60d045a2ef45fca55a620b7d55be682e8439ec8d1fc9a8bcd2615da7dffba26a"
    )
    assert toolchain["executable_sha256"] == (
        "dd86e9b24990b1da425ea4af910f016e4db8f9a25c9ddad27bc6bee3690e677f"
    )
    assert toolchain["registered_executable_sha256_matched"] is True


def test_environment_resolution_precedes_path(tmp_path: Path):
    environment_lean = tmp_path / "environment-lean"
    path_lean = tmp_path / "path-lean"
    environment_lean.write_bytes(b"environment")
    path_lean.write_bytes(b"path")
    config = json.loads((ROOT / CONFIG_PATH).read_text(encoding="utf-8"))
    resolved, source = _resolve_executable(
        config,
        environment={"INVARIANT_LEAN_EXECUTABLE": str(environment_lean)},
        which=lambda _: str(path_lean),
    )
    assert resolved == environment_lean.resolve()
    assert source == "environment"


def test_path_resolution_is_used_without_environment(tmp_path: Path):
    path_lean = tmp_path / "path-lean"
    path_lean.write_bytes(b"path")
    config = json.loads((ROOT / CONFIG_PATH).read_text(encoding="utf-8"))
    resolved, source = _resolve_executable(config, environment={}, which=lambda _: str(path_lean))
    assert resolved == path_lean.resolve()
    assert source == "PATH"


def test_platform_registry_is_closed():
    assert _platform_key(system="Windows", machine="AMD64") == "windows-x86_64"
    assert _platform_key(system="Linux", machine="x86_64") == "linux-x86_64"
    with pytest.raises(ValueError, match="unregistered"):
        _platform_key(system="Darwin", machine="arm64")


def test_optional_real_live_replay_passes_when_registered_lean_is_available():
    environment = _live_environment()
    if environment is None:
        pytest.skip("registered Lean is not installed; checked receipt remains independently valid")
    live = build_live_receipt(ROOT / CONFIG_PATH, environment=environment)
    validate_live_receipt(live, ROOT / CONFIG_PATH, environment=environment)
    assert live["receipt_role"] == "live_replay"
    assert live["decision"] == "pass_real_lean_kernel_vertical_slice"
    assert live["adapter_receipt"]["dependency_audit"]["dependencies"] == ["Eq.refl"]
    assert "C:\\Users\\" not in json.dumps(live, sort_keys=True)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda value: value["toolchain_receipt"].__setitem__("commit", "0" * 40),
        lambda value: value["toolchain_receipt"].__setitem__("executable_sha256", "0" * 64),
        lambda value: value["toolchain_receipt"].__setitem__(
            "version_output", "Lean (version 4.34.0, forged, commit " + "0" * 40 + ", Release)"
        ),
        lambda value: value["adapter_receipt"]["dependency_audit"].__setitem__(
            "dependencies", ["False.elim"]
        ),
        lambda value: value["claim_seals"].__setitem__("scientific_truth_inferred", True),
    ],
)
def test_resealed_version_identity_closure_and_claim_tampers_fail_closed(checked_receipt, mutator):
    tampered = copy.deepcopy(checked_receipt)
    mutator(tampered)
    if "adapter_receipt" in tampered:
        tampered["adapter_receipt"]["content_sha256"] = _content_sha(tampered["adapter_receipt"])
    _reseal(tampered)
    with pytest.raises(ValueError):
        validate_checked_receipt(tampered, root=ROOT)


def test_unknown_key_and_host_path_fail_closed(checked_receipt):
    unknown = copy.deepcopy(checked_receipt)
    unknown["unknown"] = True
    _reseal(unknown)
    with pytest.raises(ValueError, match="keys or seal"):
        validate_checked_receipt(unknown, root=ROOT)

    leaked = copy.deepcopy(checked_receipt)
    leaked["scope"] = r"host executable C:\Users\someone\lean.exe"
    _reseal(leaked)
    with pytest.raises(ValueError, match="host path"):
        validate_checked_receipt(leaked, root=ROOT)
