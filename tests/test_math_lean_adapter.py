from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from sigma_theory_compiler.math_lean_adapter import (
    AUDIT_PROTOCOL,
    ChildProcessResult,
    DependencyProtocolError,
    LeanAdapterConfig,
    LeanAdapterError,
    LeanResultValidationError,
    build_allowed_premise_manifest,
    parse_dependency_output,
    run_lean_adapter,
    validate_allowed_premise_manifest,
    validate_lean_adapter_result,
)


def _audit(target: str, *dependencies: str) -> bytes:
    lines = [
        f"{AUDIT_PROTOCOL}_BEGIN",
        f"target={target}",
        *(f"dependency={dependency}" for dependency in dependencies),
        "result=checked",
        f"{AUDIT_PROTOCOL}_END",
    ]
    return ("\n".join(lines) + "\n").encode()


def _fixture(tmp_path: Path) -> tuple[LeanAdapterConfig, Path, Path]:
    executable = tmp_path / "lean-fake"
    executable.write_bytes(b"fake-lean-v1")
    source = tmp_path / "Proof.lean"
    source.write_text("theorem Invariant.demo : True := by trivial\n", encoding="utf-8")
    config = LeanAdapterConfig(
        target="Invariant.demo",
        allowed_premises=("Eq.refl", "True.intro"),
        equivalent_targets=("Invariant.demoAlias",),
        forbidden_premises=("Classical.choice",),
        forbidden_prefixes=("Unsafe",),
        executable=executable,
        timeout_seconds=2,
    )
    return config, executable, source


def _reseal(value: dict) -> None:
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    value["content_sha256"] = hashlib.sha256(encoded.encode()).hexdigest()


def test_manifest_is_closed_normalized_and_hash_bound(tmp_path: Path) -> None:
    config, _, _ = _fixture(tmp_path)
    manifest = build_allowed_premise_manifest(config)
    validate_allowed_premise_manifest(manifest, config)

    assert manifest["target"] == "Invariant.demo"
    assert manifest["allowed_premises"] == ["Eq.refl", "True.intro"]
    assert manifest["forbidden_prefixes"] == ["Unsafe."]

    changed = deepcopy(manifest)
    changed["allowed_premises"].append("False.elim")
    _reseal(changed)
    with pytest.raises(LeanResultValidationError):
        validate_allowed_premise_manifest(changed, config)

    with pytest.raises(LeanAdapterError, match="target"):
        build_allowed_premise_manifest(
            LeanAdapterConfig(target="T", allowed_premises=("_root_.T",))
        )


def test_absent_lean_blocks_without_running_or_passing(tmp_path: Path) -> None:
    source = tmp_path / "Proof.lean"
    source.write_text("theorem T : True := by trivial\n", encoding="utf-8")
    config = LeanAdapterConfig(target="T", executable=tmp_path / "missing-lean")
    called = False

    def runner(command: tuple[str, ...], cwd: Path, timeout: float) -> ChildProcessResult:
        nonlocal called
        called = True
        raise AssertionError((command, cwd, timeout))

    result = run_lean_adapter(config, source, runner=runner, environment={})
    validate_lean_adapter_result(result, config, source)

    assert called is False
    assert result["status"] == "block"
    assert result["decision"] == "block_lean_unavailable"
    assert result["claims"] == {
        "lean_available": False,
        "formal_target_checked": False,
        "scientific_truth_inferred": False,
    }

    forged = deepcopy(result)
    forged["status"] = "pass"
    forged["decision"] = "pass_lean_checked_closed_premise"
    forged["claims"]["formal_target_checked"] = True
    _reseal(forged)
    with pytest.raises(LeanResultValidationError):
        validate_lean_adapter_result(forged, config, source)


def test_exact_allowed_dependency_output_passes(tmp_path: Path) -> None:
    config, executable, source = _fixture(tmp_path)
    observed: list[tuple[tuple[str, ...], Path, float]] = []

    def runner(command: tuple[str, ...], cwd: Path, timeout: float) -> ChildProcessResult:
        observed.append((command, cwd, timeout))
        return ChildProcessResult(0, _audit("Invariant.demo", "True.intro", "Eq.refl"))

    result = run_lean_adapter(config, source, runner=runner)
    validate_lean_adapter_result(result, config, source)

    assert result["status"] == "pass"
    assert result["decision"] == "pass_lean_checked_closed_premise"
    assert result["dependency_audit"]["dependencies"] == ["Eq.refl", "True.intro"]
    assert result["claims"]["formal_target_checked"] is True
    assert result["claims"]["scientific_truth_inferred"] is False
    assert observed == [((str(executable.resolve()), str(source.resolve())), tmp_path.resolve(), 2)]


@pytest.mark.parametrize(
    ("dependency", "reason"),
    [
        ("Invariant.demo", "target or equivalent"),
        ("_root_.Invariant.demoAlias", "target or equivalent"),
        ("Classical.choice", "forbidden premise"),
        ("Unsafe.escape", "forbidden premise"),
        ("False.elim", "outside the allowed-premise closure"),
    ],
)
def test_target_equivalent_forbidden_and_out_of_closure_dependencies_reject(
    tmp_path: Path, dependency: str, reason: str
) -> None:
    config, _, source = _fixture(tmp_path)

    def runner(command: tuple[str, ...], cwd: Path, timeout: float) -> ChildProcessResult:
        del command, cwd, timeout
        return ChildProcessResult(0, _audit("Invariant.demo", dependency))

    result = run_lean_adapter(config, source, runner=runner)
    validate_lean_adapter_result(result, config, source)

    assert result["status"] == "reject"
    assert result["decision"] == "reject_lean_dependency_policy"
    assert reason in result["reason"]
    assert result["claims"]["formal_target_checked"] is False


def test_malformed_duplicate_protocol_rejects(tmp_path: Path) -> None:
    config, _, source = _fixture(tmp_path)
    duplicated = _audit("Invariant.demo", "Eq.refl") + _audit("Invariant.demo", "Eq.refl")

    with pytest.raises(DependencyProtocolError, match="exactly one"):
        parse_dependency_output(duplicated)

    result = run_lean_adapter(
        config,
        source,
        runner=lambda command, cwd, timeout: ChildProcessResult(0, duplicated),
    )
    validate_lean_adapter_result(result, config, source)
    assert result["decision"] == "reject_lean_dependency_protocol"


def test_timeout_and_nonzero_exit_block(tmp_path: Path) -> None:
    config, _, source = _fixture(tmp_path)
    timed_out = run_lean_adapter(
        config,
        source,
        runner=lambda command, cwd, timeout: ChildProcessResult(None, timed_out=True),
    )
    failed = run_lean_adapter(
        config,
        source,
        runner=lambda command, cwd, timeout: ChildProcessResult(2, stderr=b"failure"),
    )

    validate_lean_adapter_result(timed_out, config, source)
    validate_lean_adapter_result(failed, config, source)
    assert timed_out["decision"] == "block_lean_timeout"
    assert timed_out["execution"]["timed_out"] is True
    assert failed["decision"] == "block_lean_process_failure"
    assert failed["execution"]["exit_code"] == 2
    assert not timed_out["claims"]["formal_target_checked"]
    assert not failed["claims"]["formal_target_checked"]


def test_resealed_result_tamper_and_source_tamper_fail_validation(tmp_path: Path) -> None:
    config, _, source = _fixture(tmp_path)
    result = run_lean_adapter(
        config,
        source,
        runner=lambda command, cwd, timeout: ChildProcessResult(
            0, _audit("Invariant.demo", "Eq.refl")
        ),
    )

    changed = deepcopy(result)
    changed["claims"]["scientific_truth_inferred"] = True
    _reseal(changed)
    with pytest.raises(LeanResultValidationError):
        validate_lean_adapter_result(changed, config, source)

    source.write_text("theorem Invariant.demo : False := by sorry\n", encoding="utf-8")
    with pytest.raises(LeanResultValidationError, match="source"):
        validate_lean_adapter_result(result, config, source)
