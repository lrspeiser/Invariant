"""Fail-closed Lean proof adapter boundary for Math Pack v1."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MANIFEST_SCHEMA = "invariant-lean-allowed-premise-manifest-1.0"
RESULT_SCHEMA = "invariant-lean-adapter-result-1.0"
AUDIT_PROTOCOL = "INVARIANT_LEAN_DEPENDENCY_AUDIT_V1"
MAX_AUDIT_OUTPUT_BYTES = 1024 * 1024
MAX_SOURCE_BYTES = 4 * 1024 * 1024
_BEGIN = f"{AUDIT_PROTOCOL}_BEGIN"
_END = f"{AUDIT_PROTOCOL}_END"
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_']*(?:\.[A-Za-z_][A-Za-z0-9_']*)*\Z")

_MANIFEST_KEYS = {
    "allowed_premises",
    "content_sha256",
    "equivalent_targets",
    "forbidden_prefixes",
    "forbidden_premises",
    "schema_version",
    "target",
}
_RESULT_KEYS = {
    "claims",
    "content_sha256",
    "decision",
    "dependency_audit",
    "executable",
    "execution",
    "manifest_sha256",
    "reason",
    "schema_version",
    "source_sha256",
    "status",
    "target",
}
_EXECUTABLE_KEYS = {"configured", "discovered", "discovery_source", "identity_sha256"}
_EXECUTION_KEYS = {"attempted", "exit_code", "stderr_sha256", "stdout_sha256", "timed_out"}
_AUDIT_KEYS = {"closure_valid", "dependencies", "protocol_version", "reported_target"}
_CLAIM_KEYS = {"formal_target_checked", "lean_available", "scientific_truth_inferred"}


class LeanAdapterError(ValueError):
    """Base class for malformed adapter inputs."""


class DependencyProtocolError(LeanAdapterError):
    """Raised when Lean output does not contain one closed dependency audit."""


class DependencyPolicyError(LeanAdapterError):
    """Raised when a dependency escapes the configured premise closure."""


class LeanResultValidationError(LeanAdapterError):
    """Raised when an adapter result has been altered or mismatched."""


@dataclass(frozen=True, slots=True)
class LeanAdapterConfig:
    target: str
    allowed_premises: tuple[str, ...] = ()
    equivalent_targets: tuple[str, ...] = ()
    forbidden_premises: tuple[str, ...] = ()
    forbidden_prefixes: tuple[str, ...] = ()
    executable: str | Path | None = None
    arguments: tuple[str, ...] = ()
    timeout_seconds: float = 10.0
    executable_environment_variable: str = "INVARIANT_LEAN_EXECUTABLE"

    def __post_init__(self) -> None:
        if not 0 < self.timeout_seconds <= 300:
            raise LeanAdapterError("Lean timeout must be in (0, 300] seconds")
        if not self.executable_environment_variable.isidentifier():
            raise LeanAdapterError("Lean executable environment variable is invalid")
        if any("\x00" in argument for argument in self.arguments):
            raise LeanAdapterError("Lean arguments cannot contain NUL bytes")


@dataclass(frozen=True, slots=True)
class ChildProcessResult:
    exit_code: int | None
    stdout: bytes = b""
    stderr: bytes = b""
    timed_out: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.stdout, bytes) or not isinstance(self.stderr, bytes):
            raise TypeError("child output must be bytes")
        if self.timed_out and self.exit_code is not None:
            raise ValueError("timed-out child cannot have an exit code")


ChildRunner = Callable[[tuple[str, ...], Path, float], ChildProcessResult]


@dataclass(frozen=True, slots=True)
class LeanExecutableDiscovery:
    configured: bool
    path: Path | None
    source: str | None


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha_data(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seal(value: dict[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["content_sha256"] = _sha_data(result)
    return result


def _verify_seal(value: Mapping[str, Any]) -> None:
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("content_sha256") != _sha_data(body):
        raise LeanResultValidationError("Lean adapter result content hash changed")


def normalize_lean_name(value: str) -> str:
    """Normalize the one supported root-qualified Lean identifier spelling."""

    if not isinstance(value, str) or value != value.strip() or not value:
        raise LeanAdapterError("Lean names must be nonempty and whitespace-free")
    normalized = value.removeprefix("_root_.")
    if not _NAME.fullmatch(normalized):
        raise LeanAdapterError(f"unsupported Lean identifier: {value!r}")
    return normalized


def _normalize_unique(values: tuple[str, ...], *, label: str) -> tuple[str, ...]:
    normalized = tuple(normalize_lean_name(value) for value in values)
    if len(normalized) != len(set(normalized)):
        raise LeanAdapterError(f"{label} contains duplicate or equivalent Lean names")
    return tuple(sorted(normalized))


def build_allowed_premise_manifest(config: LeanAdapterConfig) -> dict[str, Any]:
    """Build a sealed, closed-world premise manifest."""

    target = normalize_lean_name(config.target)
    allowed = _normalize_unique(config.allowed_premises, label="allowed premises")
    equivalents = _normalize_unique(config.equivalent_targets, label="equivalent targets")
    forbidden = _normalize_unique(config.forbidden_premises, label="forbidden premises")
    prefixes = tuple(
        f"{normalize_lean_name(value.removesuffix('.'))}." for value in config.forbidden_prefixes
    )
    if len(prefixes) != len(set(prefixes)):
        raise LeanAdapterError("forbidden prefixes contain duplicates")
    prefixes = tuple(sorted(prefixes))
    if target in equivalents:
        raise LeanAdapterError("equivalent targets cannot repeat the target")
    target_family = {target, *equivalents}
    if target_family & set(allowed):
        raise LeanAdapterError("target or equivalent target cannot be an allowed premise")
    overlap = set(allowed) & set(forbidden)
    if overlap or any(
        dependency.startswith(prefix) or dependency == prefix.removesuffix(".")
        for dependency in allowed
        for prefix in prefixes
    ):
        raise LeanAdapterError("allowed premise is also forbidden")
    return _seal(
        {
            "schema_version": MANIFEST_SCHEMA,
            "target": target,
            "allowed_premises": list(allowed),
            "equivalent_targets": list(equivalents),
            "forbidden_premises": list(forbidden),
            "forbidden_prefixes": list(prefixes),
        }
    )


def validate_allowed_premise_manifest(
    manifest: Mapping[str, Any], config: LeanAdapterConfig
) -> None:
    if set(manifest) != _MANIFEST_KEYS:
        raise LeanResultValidationError("Lean premise manifest schema changed")
    _verify_seal(manifest)
    if dict(manifest) != build_allowed_premise_manifest(config):
        raise LeanResultValidationError("Lean premise manifest does not match configuration")


def discover_lean_executable(
    config: LeanAdapterConfig,
    *,
    environment: Mapping[str, str] | None = None,
) -> LeanExecutableDiscovery:
    """Resolve only an explicit or explicitly environment-configured Lean executable."""

    environment = os.environ if environment is None else environment
    raw: str | None
    source: str | None
    if config.executable is not None:
        raw = os.fspath(config.executable)
        source = "explicit"
    else:
        raw = environment.get(config.executable_environment_variable)
        source = "environment" if raw else None
    if not raw:
        return LeanExecutableDiscovery(False, None, None)
    if "\x00" in raw:
        raise LeanAdapterError("configured Lean executable contains a NUL byte")
    candidate = Path(raw).expanduser()
    if candidate.is_absolute() or candidate.parent != Path("."):
        resolved = candidate.resolve()
        return LeanExecutableDiscovery(True, resolved if resolved.is_file() else None, source)
    found = shutil.which(raw)
    resolved = Path(found).resolve() if found else None
    return LeanExecutableDiscovery(
        True, resolved if resolved and resolved.is_file() else None, source
    )


def parse_dependency_output(output: bytes) -> tuple[str, tuple[str, ...]]:
    """Parse exactly one tagged dependency audit block from Lean stdout."""

    if len(output) > MAX_AUDIT_OUTPUT_BYTES:
        raise DependencyProtocolError("Lean dependency output exceeds the byte limit")
    try:
        lines = output.decode("utf-8", errors="strict").splitlines()
    except UnicodeDecodeError as error:
        raise DependencyProtocolError("Lean dependency output is not UTF-8") from error
    begins = [index for index, line in enumerate(lines) if line == _BEGIN]
    ends = [index for index, line in enumerate(lines) if line == _END]
    if len(begins) != 1 or len(ends) != 1 or ends[0] <= begins[0]:
        raise DependencyProtocolError("Lean output must contain exactly one complete audit block")
    body = lines[begins[0] + 1 : ends[0]]
    targets: list[str] = []
    results: list[str] = []
    dependencies: list[str] = []
    for line in body:
        if line.startswith("target="):
            targets.append(normalize_lean_name(line.removeprefix("target=")))
        elif line.startswith("dependency="):
            dependencies.append(normalize_lean_name(line.removeprefix("dependency=")))
        elif line.startswith("result="):
            results.append(line.removeprefix("result="))
        else:
            raise DependencyProtocolError("Lean audit block contains an unknown record")
    if len(targets) != 1 or results != ["checked"]:
        raise DependencyProtocolError("Lean audit target or checked result is missing")
    if len(dependencies) != len(set(dependencies)):
        raise DependencyProtocolError("Lean audit contains duplicate or equivalent dependencies")
    return targets[0], tuple(sorted(dependencies))


def check_dependency_closure(
    reported_target: str,
    dependencies: tuple[str, ...],
    manifest: Mapping[str, Any],
) -> None:
    """Reject targets, equivalents, forbidden names, and every out-of-closure premise."""

    target = normalize_lean_name(str(manifest["target"]))
    if normalize_lean_name(reported_target) != target:
        raise DependencyPolicyError("Lean audit target does not match the manifest target")
    target_family = {target, *manifest["equivalent_targets"]}
    allowed = set(manifest["allowed_premises"])
    forbidden = set(manifest["forbidden_premises"])
    prefixes = tuple(manifest["forbidden_prefixes"])
    for dependency in dependencies:
        normalized = normalize_lean_name(dependency)
        if normalized in target_family:
            raise DependencyPolicyError("target or equivalent target appears as a dependency")
        if normalized in forbidden or any(
            normalized == prefix.removesuffix(".") or normalized.startswith(prefix)
            for prefix in prefixes
        ):
            raise DependencyPolicyError("forbidden premise appears as a dependency")
        if normalized not in allowed:
            raise DependencyPolicyError("dependency is outside the allowed-premise closure")


def _run_owned_child(command: tuple[str, ...], cwd: Path, timeout: float) -> ChildProcessResult:
    creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            check=False,
            timeout=timeout,
            creationflags=creation_flags,
        )
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout if isinstance(error.stdout, bytes) else b""
        stderr = error.stderr if isinstance(error.stderr, bytes) else b""
        return ChildProcessResult(None, stdout, stderr, timed_out=True)
    except OSError as error:
        return ChildProcessResult(-1, stderr=str(error).encode("utf-8", errors="replace"))
    return ChildProcessResult(result.returncode, result.stdout, result.stderr)


def _result(
    *,
    manifest: Mapping[str, Any],
    status: str,
    decision: str,
    reason: str,
    discovery: LeanExecutableDiscovery,
    source_sha256: str | None,
    process: ChildProcessResult | None,
    reported_target: str | None = None,
    dependencies: tuple[str, ...] = (),
    closure_valid: bool = False,
) -> dict[str, Any]:
    available = discovery.path is not None
    formal_checked = status == "pass"
    empty_sha = _sha_bytes(b"")
    return _seal(
        {
            "schema_version": RESULT_SCHEMA,
            "status": status,
            "decision": decision,
            "reason": reason,
            "target": manifest["target"],
            "manifest_sha256": manifest["content_sha256"],
            "source_sha256": source_sha256,
            "executable": {
                "configured": discovery.configured,
                "discovered": available,
                "discovery_source": discovery.source,
                "identity_sha256": _sha_file(discovery.path) if discovery.path else None,
            },
            "execution": {
                "attempted": process is not None,
                "timed_out": process.timed_out if process else False,
                "exit_code": process.exit_code if process else None,
                "stdout_sha256": _sha_bytes(process.stdout) if process else empty_sha,
                "stderr_sha256": _sha_bytes(process.stderr) if process else empty_sha,
            },
            "dependency_audit": {
                "protocol_version": AUDIT_PROTOCOL if reported_target is not None else None,
                "reported_target": reported_target,
                "dependencies": list(dependencies),
                "closure_valid": closure_valid,
            },
            "claims": {
                "lean_available": available,
                "formal_target_checked": formal_checked,
                "scientific_truth_inferred": False,
            },
        }
    )


def run_lean_adapter(
    config: LeanAdapterConfig,
    source_path: str | Path,
    *,
    runner: ChildRunner = _run_owned_child,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Run one configured Lean child and return a sealed pass/block/reject result."""

    manifest = build_allowed_premise_manifest(config)
    discovery = discover_lean_executable(config, environment=environment)
    if discovery.path is None:
        return _result(
            manifest=manifest,
            status="block",
            decision="block_lean_unavailable",
            reason="configured Lean executable was not discovered",
            discovery=discovery,
            source_sha256=None,
            process=None,
        )
    source = Path(source_path).resolve()
    if source.suffix != ".lean" or not source.is_file() or source.stat().st_size > MAX_SOURCE_BYTES:
        return _result(
            manifest=manifest,
            status="block",
            decision="block_lean_source_unavailable",
            reason="Lean source is missing, not .lean, or exceeds the byte limit",
            discovery=discovery,
            source_sha256=None,
            process=None,
        )
    source_sha = _sha_file(source)
    command = (str(discovery.path), *config.arguments, str(source))
    process = runner(command, source.parent, config.timeout_seconds)
    if not isinstance(process, ChildProcessResult):
        raise TypeError("Lean child runner returned an unsupported result")
    if process.timed_out:
        return _result(
            manifest=manifest,
            status="block",
            decision="block_lean_timeout",
            reason="Lean child exceeded its configured timeout",
            discovery=discovery,
            source_sha256=source_sha,
            process=process,
        )
    if process.exit_code != 0:
        return _result(
            manifest=manifest,
            status="block",
            decision="block_lean_process_failure",
            reason="Lean child returned a nonzero exit code",
            discovery=discovery,
            source_sha256=source_sha,
            process=process,
        )
    try:
        reported_target, dependencies = parse_dependency_output(process.stdout)
    except (DependencyProtocolError, LeanAdapterError) as error:
        return _result(
            manifest=manifest,
            status="reject",
            decision="reject_lean_dependency_protocol",
            reason=str(error),
            discovery=discovery,
            source_sha256=source_sha,
            process=process,
        )
    try:
        check_dependency_closure(reported_target, dependencies, manifest)
    except DependencyPolicyError as error:
        return _result(
            manifest=manifest,
            status="reject",
            decision="reject_lean_dependency_policy",
            reason=str(error),
            discovery=discovery,
            source_sha256=source_sha,
            process=process,
            reported_target=reported_target,
            dependencies=dependencies,
        )
    return _result(
        manifest=manifest,
        status="pass",
        decision="pass_lean_checked_closed_premise",
        reason="Lean checked the target and reported only allowed premises",
        discovery=discovery,
        source_sha256=source_sha,
        process=process,
        reported_target=reported_target,
        dependencies=dependencies,
        closure_valid=True,
    )


def validate_lean_adapter_result(
    result: Mapping[str, Any],
    config: LeanAdapterConfig,
    source_path: str | Path,
    *,
    environment: Mapping[str, str] | None = None,
) -> None:
    """Validate closed schemas, seals, bindings, and fail-closed decision semantics."""

    if set(result) != _RESULT_KEYS:
        raise LeanResultValidationError("Lean adapter result schema changed")
    for key, expected in (
        ("executable", _EXECUTABLE_KEYS),
        ("execution", _EXECUTION_KEYS),
        ("dependency_audit", _AUDIT_KEYS),
        ("claims", _CLAIM_KEYS),
    ):
        nested = result.get(key)
        if not isinstance(nested, Mapping) or set(nested) != expected:
            raise LeanResultValidationError(f"Lean adapter {key} schema changed")
    _verify_seal(result)
    manifest = build_allowed_premise_manifest(config)
    if (
        result.get("schema_version") != RESULT_SCHEMA
        or result.get("target") != manifest["target"]
        or result.get("manifest_sha256") != manifest["content_sha256"]
        or result["claims"].get("scientific_truth_inferred") is not False
    ):
        raise LeanResultValidationError("Lean adapter result binding or broad claim changed")

    source = Path(source_path).resolve()
    expected_source = _sha_file(source) if source.is_file() and source.suffix == ".lean" else None
    if result.get("source_sha256") not in {None, expected_source}:
        raise LeanResultValidationError("Lean adapter source binding changed")

    discovery = discover_lean_executable(config, environment=environment)
    expected_executable = {
        "configured": discovery.configured,
        "discovered": discovery.path is not None,
        "discovery_source": discovery.source,
        "identity_sha256": _sha_file(discovery.path) if discovery.path else None,
    }
    if result["executable"] != expected_executable:
        raise LeanResultValidationError("Lean executable discovery binding changed")

    status = result.get("status")
    decision = result.get("decision")
    claims = result["claims"]
    execution = result["execution"]
    audit = result["dependency_audit"]
    valid_decisions = {
        "pass": {"pass_lean_checked_closed_premise"},
        "block": {
            "block_lean_unavailable",
            "block_lean_source_unavailable",
            "block_lean_timeout",
            "block_lean_process_failure",
        },
        "reject": {"reject_lean_dependency_protocol", "reject_lean_dependency_policy"},
    }
    reasons = {
        "pass_lean_checked_closed_premise": (
            "Lean checked the target and reported only allowed premises"
        ),
        "block_lean_unavailable": "configured Lean executable was not discovered",
        "block_lean_source_unavailable": (
            "Lean source is missing, not .lean, or exceeds the byte limit"
        ),
        "block_lean_timeout": "Lean child exceeded its configured timeout",
        "block_lean_process_failure": "Lean child returned a nonzero exit code",
    }
    if status not in valid_decisions or decision not in valid_decisions[status]:
        raise LeanResultValidationError("Lean adapter status and decision disagree")
    if decision in reasons and result.get("reason") != reasons[decision]:
        raise LeanResultValidationError("Lean adapter decision reason changed")
    if claims.get("lean_available") is not result["executable"]["discovered"]:
        raise LeanResultValidationError("Lean availability claim changed")
    hashes = (
        execution.get("stdout_sha256"),
        execution.get("stderr_sha256"),
        result["executable"].get("identity_sha256"),
    )
    if any(value is not None and not re.fullmatch(r"[0-9a-f]{64}", value) for value in hashes):
        raise LeanResultValidationError("Lean adapter byte identity is malformed")
    if status == "pass":
        dependencies = tuple(audit["dependencies"])
        try:
            check_dependency_closure(str(audit["reported_target"]), dependencies, manifest)
        except (LeanAdapterError, KeyError, TypeError) as error:
            raise LeanResultValidationError("Lean pass dependency closure changed") from error
        if (
            claims
            != {
                "lean_available": True,
                "formal_target_checked": True,
                "scientific_truth_inferred": False,
            }
            or execution["attempted"] is not True
            or execution["timed_out"] is not False
            or execution["exit_code"] != 0
            or audit["protocol_version"] != AUDIT_PROTOCOL
            or audit["closure_valid"] is not True
            or expected_source is None
            or result["source_sha256"] != expected_source
        ):
            raise LeanResultValidationError("Lean pass semantics changed")
    elif claims.get("formal_target_checked") is not False or audit["closure_valid"] is not False:
        raise LeanResultValidationError("non-pass Lean result asserts a formal check")
    empty_audit = {
        "protocol_version": None,
        "reported_target": None,
        "dependencies": [],
        "closure_valid": False,
    }
    if decision == "block_lean_unavailable" and (
        claims.get("lean_available") is not False
        or result["source_sha256"] is not None
        or execution["attempted"] is not False
        or audit != empty_audit
    ):
        raise LeanResultValidationError("unavailable Lean result semantics changed")
    if decision == "block_lean_source_unavailable" and (
        claims.get("lean_available") is not True
        or result["source_sha256"] is not None
        or execution["attempted"] is not False
        or audit != empty_audit
    ):
        raise LeanResultValidationError("unavailable Lean source result semantics changed")
    if decision == "block_lean_timeout" and (
        result["source_sha256"] != expected_source
        or execution["attempted"] is not True
        or execution["timed_out"] is not True
        or execution["exit_code"] is not None
        or audit != empty_audit
    ):
        raise LeanResultValidationError("Lean timeout result semantics changed")
    if decision == "block_lean_process_failure" and (
        result["source_sha256"] != expected_source
        or execution["attempted"] is not True
        or execution["timed_out"] is not False
        or not isinstance(execution["exit_code"], int)
        or execution["exit_code"] == 0
        or audit != empty_audit
    ):
        raise LeanResultValidationError("Lean process failure semantics changed")
    if decision == "reject_lean_dependency_protocol" and (
        result["source_sha256"] != expected_source
        or execution["attempted"] is not True
        or execution["timed_out"] is not False
        or execution["exit_code"] != 0
        or audit != empty_audit
    ):
        raise LeanResultValidationError("Lean dependency protocol rejection semantics changed")
    if decision == "reject_lean_dependency_policy" and (
        result["source_sha256"] != expected_source
        or execution["attempted"] is not True
        or execution["timed_out"] is not False
        or execution["exit_code"] != 0
        or audit["protocol_version"] != AUDIT_PROTOCOL
        or audit["reported_target"] is None
    ):
        raise LeanResultValidationError("Lean dependency policy rejection semantics changed")
