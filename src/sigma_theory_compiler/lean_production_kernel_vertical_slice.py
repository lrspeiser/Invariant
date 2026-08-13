"""Portable real-Lean kernel slice with separate checked and live receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .math_lean_adapter import (
    AUDIT_PROTOCOL,
    LeanAdapterConfig,
    build_allowed_premise_manifest,
    run_lean_adapter,
)
from .math_lean_adapter import (
    RESULT_SCHEMA as LEAN_RESULT_SCHEMA,
)

CONFIG_SCHEMA = "invariant-lean-production-kernel-vertical-slice-config-2.0"
RESULT_SCHEMA = "invariant-lean-production-kernel-vertical-slice-result-2.0"
CAMPAIGN_ID = "lean-production-kernel-vertical-slice-001"
CONFIG_PATH = "configs/math/lean_production_kernel_vertical_slice.json"
SOURCE_PATH = "src/sigma_theory_compiler/lean_production_kernel_vertical_slice.py"
TEST_PATH = "tests/test_lean_production_kernel_vertical_slice.py"
THEOREM_PATH = "formal/lean/InvariantKernelSmoke.lean"
OUTPUT_PATH = "runs/math/lean-production-kernel-vertical-slice/receipt.json"
RELEASE = "leanprover/lean4:v4.33.0"
VERSION = "4.33.0"
COMMIT = "d8b18978322de05a8f3dba51ef03cf5461676c17"
ADAPTER_BINDING = {
    "source": {
        "path": "src/sigma_theory_compiler/math_lean_adapter.py",
        "file_sha256": "18380105b3e5549593450c04f79deb1f2384635d43822988af702f638be356d5",
    },
    "test": {
        "path": "tests/test_math_lean_adapter.py",
        "file_sha256": "03ffc05ec5e439d5743bd64dacc76c8b7e53365f209226cf3495bbfcd53f2878",
    },
}
PREMISE_POLICY = {
    "allowed_premises": ["Eq.refl"],
    "equivalent_targets": [],
    "forbidden_premises": ["Classical.choice", "False.elim"],
    "forbidden_prefixes": ["KnownAnswer", "Unsafe"],
}
EXECUTABLE_RESOLUTION = {
    "environment_variable": "INVARIANT_LEAN_EXECUTABLE",
    "path_fallback": "lean",
}
PLATFORM_ASSETS = {
    "linux-x86_64": {
        "asset_url": (
            "https://github.com/leanprover/lean4/releases/download/v4.33.0/"
            "lean-4.33.0-linux.tar.zst"
        ),
        "archive_bytes": 574882764,
        "archive_sha256": "4b3fb03c29a1e0a253fb1d11f9bae3725f19a0dc6fc09b3ea16d2c9df3349e2c",
        "executable_sha256": None,
    },
    "windows-x86_64": {
        "asset_url": (
            "https://github.com/leanprover/lean4/releases/download/v4.33.0/"
            "lean-4.33.0-windows.tar.zst"
        ),
        "archive_bytes": 583425362,
        "archive_sha256": "60d045a2ef45fca55a620b7d55be682e8439ec8d1fc9a8bcd2615da7dffba26a",
        "executable_sha256": "dd86e9b24990b1da425ea4af910f016e4db8f9a25c9ddad27bc6bee3690e677f",
    },
}
TOOLCHAIN = {
    "official_release": RELEASE,
    "version": VERSION,
    "commit": COMMIT,
    "platform_assets": PLATFORM_ASSETS,
}
_RESULT_KEYS = {
    "adapter_receipt",
    "campaign_id",
    "claim_seals",
    "content_sha256",
    "decision",
    "dependency_closure",
    "first_blocker",
    "readiness_counts",
    "receipt_role",
    "schema_version",
    "scope",
    "source_bindings",
    "theorem_contract",
    "toolchain_receipt",
}
_ADAPTER_KEYS = {
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
_VERSION_PATTERN = re.compile(
    rf"Lean \(version {re.escape(VERSION)}, [^,]+, commit {COMMIT}, Release\)\Z"
)
_AUDIT_STDOUT_SHA256 = hashlib.sha256(
    (
        f"{AUDIT_PROTOCOL}_BEGIN\n"
        "target=Invariant.kernelSmoke\n"
        "dependency=Eq.refl\n"
        "result=checked\n"
        f"{AUDIT_PROTOCOL}_END\n"
    ).encode()
).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _content_sha(value: Mapping[str, Any]) -> str:
    return _sha({key: item for key, item in value.items() if key != "content_sha256"})


def _inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise ValueError("Lean vertical-slice path escapes project root")
    return path


def _validate_config(config: Mapping[str, Any], *, root: Path) -> None:
    if config != {
        "schema_version": CONFIG_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "output_path": OUTPUT_PATH,
        "theorem_source_path": THEOREM_PATH,
        "target": "Invariant.kernelSmoke",
        "timeout_seconds": 30,
        "premise_policy": PREMISE_POLICY,
        "adapter_binding": ADAPTER_BINDING,
        "executable_resolution": EXECUTABLE_RESOLUTION,
        "toolchain": TOOLCHAIN,
    }:
        raise ValueError("Lean vertical-slice config boundary changed")
    for binding in ADAPTER_BINDING.values():
        if _file_sha(_inside(root, binding["path"])) != binding["file_sha256"]:
            raise ValueError("Lean vertical-slice adapter dependency changed")


def _platform_key(*, system: str | None = None, machine: str | None = None) -> str:
    system = platform.system() if system is None else system
    machine = platform.machine() if machine is None else machine
    normalized_machine = machine.lower()
    if normalized_machine in {"amd64", "x86_64"}:
        normalized_machine = "x86_64"
    key = f"{system.lower()}-{normalized_machine}"
    if key not in PLATFORM_ASSETS:
        raise ValueError(f"Lean vertical-slice platform is unregistered: {key}")
    return key


def _resolve_executable(
    config: Mapping[str, Any],
    *,
    environment: Mapping[str, str] | None = None,
    which: Callable[[str], str | None] = shutil.which,
) -> tuple[Path | None, str | None]:
    environment = os.environ if environment is None else environment
    resolution = config["executable_resolution"]
    variable = str(resolution["environment_variable"])
    raw = environment.get(variable)
    source = "environment" if raw else "PATH"
    if not raw:
        raw = which(str(resolution["path_fallback"]))
    if not raw or "\x00" in raw:
        return None, None
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute() and candidate.parent == Path("."):
        found = which(raw)
        candidate = Path(found) if found else candidate
    resolved = candidate.resolve()
    return (resolved, source) if resolved.is_file() else (None, None)


def _probe_toolchain(executable: Path, platform_key: str, source: str) -> dict[str, Any]:
    executable_sha = _file_sha(executable)
    registered_sha = PLATFORM_ASSETS[platform_key]["executable_sha256"]
    if registered_sha is not None and executable_sha != registered_sha:
        raise ValueError("Lean vertical-slice registered executable identity changed")
    creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    try:
        process = subprocess.run(
            (str(executable), "--version"),
            capture_output=True,
            check=False,
            timeout=30,
            creationflags=creation_flags,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ValueError("Lean vertical-slice version probe failed") from error
    try:
        version_output = process.stdout.decode("utf-8", errors="strict").strip()
    except UnicodeDecodeError as error:
        raise ValueError("Lean vertical-slice version output is not UTF-8") from error
    if process.returncode != 0 or process.stderr or not _VERSION_PATTERN.fullmatch(version_output):
        raise ValueError("Lean vertical-slice version or commit changed")
    return {
        "platform": platform_key,
        "official_release": RELEASE,
        "version": VERSION,
        "commit": COMMIT,
        "platform_asset": PLATFORM_ASSETS[platform_key],
        "executable_sha256": executable_sha,
        "registered_executable_sha256_matched": (True if registered_sha is not None else None),
        "version_output": version_output,
        "version_probe_exit_code": process.returncode,
        "resolution_source": source,
        "executable_path_persisted": False,
        "archive_and_binary_outside_git_history": True,
    }


def _adapter_config(config: Mapping[str, Any], executable: Path) -> LeanAdapterConfig:
    policy = config["premise_policy"]
    return LeanAdapterConfig(
        target=str(config["target"]),
        allowed_premises=tuple(policy["allowed_premises"]),
        equivalent_targets=tuple(policy["equivalent_targets"]),
        forbidden_premises=tuple(policy["forbidden_premises"]),
        forbidden_prefixes=tuple(policy["forbidden_prefixes"]),
        executable=executable,
        timeout_seconds=float(config["timeout_seconds"]),
    )


def build_live_receipt(
    config_path: Path,
    *,
    environment: Mapping[str, str] | None = None,
    receipt_role: str = "live_replay",
) -> dict[str, Any]:
    if receipt_role not in {"live_replay", "checked_windows_historical"}:
        raise ValueError("Lean vertical-slice receipt role is invalid")
    root = config_path.resolve().parents[2]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config, root=root)
    executable, resolution_source = _resolve_executable(config, environment=environment)
    if executable is None or resolution_source is None:
        raise ValueError("Lean vertical-slice executable was not resolved from environment or PATH")
    platform_key = _platform_key()
    toolchain_receipt = _probe_toolchain(executable, platform_key, resolution_source)
    theorem_path = _inside(root, THEOREM_PATH)
    adapter_config = _adapter_config(config, executable)
    manifest = build_allowed_premise_manifest(adapter_config)
    adapter_receipt = run_lean_adapter(adapter_config, theorem_path, environment={})
    if adapter_receipt["decision"] != "pass_lean_checked_closed_premise":
        raise ValueError(f"Lean vertical-slice adapter did not pass: {adapter_receipt['decision']}")
    body = {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "receipt_role": receipt_role,
        "decision": "pass_real_lean_kernel_vertical_slice",
        "first_blocker": "none_for_bounded_kernel_smoke_theorem",
        "theorem_contract": {
            "target": "Invariant.kernelSmoke",
            "statement": "for every natural number n, n = n",
            "proof_term": "Eq.refl n",
            "known_answer_role": "bounded_kernel_and_dependency_protocol_smoke_theorem_only",
            "source_sha256": _file_sha(theorem_path),
            "source_bytes": theorem_path.stat().st_size,
        },
        "dependency_closure": {
            "allowed_premise_manifest": manifest,
            "declared_dependencies": ["Eq.refl"],
            "closure_checked_by_adapter": True,
            "target_or_equivalent_allowed_as_dependency": False,
            "out_of_manifest_dependency_allowed": False,
        },
        "adapter_receipt": adapter_receipt,
        "toolchain_receipt": toolchain_receipt,
        "readiness_counts": {
            "theorems_presented": 1,
            "kernel_executions_attempted": 1,
            "kernel_checked_theorems": 1,
            "blocked_theorems": 0,
            "rejected_theorems": 0,
        },
        "claim_seals": {
            "lean_executable_discovered": True,
            "bounded_theorem_kernel_checked": True,
            "dependency_closure_validated": True,
            "scientific_truth_inferred": False,
            "general_formal_completion_claimed": False,
            "physics_claimed": False,
        },
        "source_bindings": {
            "source": {"path": SOURCE_PATH, "file_sha256": _file_sha(_inside(root, SOURCE_PATH))},
            "config": {"path": CONFIG_PATH, "file_sha256": _file_sha(config_path)},
            "test": {"path": TEST_PATH, "file_sha256": _file_sha(_inside(root, TEST_PATH))},
            "theorem": {"path": THEOREM_PATH, "file_sha256": _file_sha(theorem_path)},
            "adapter": ADAPTER_BINDING,
        },
        "scope": (
            "one known-answer Nat reflexivity theorem and its declared Eq.refl premise closure; "
            "no scientific truth, general theorem-proving, production availability, or physics claim"
        ),
    }
    return {**body, "content_sha256": _sha(body)}


def _validate_adapter_semantics(
    receipt: Mapping[str, Any], theorem_sha: str, toolchain: Mapping[str, Any]
) -> None:
    if set(receipt) != _ADAPTER_KEYS or receipt.get("content_sha256") != _content_sha(receipt):
        raise ValueError("Lean vertical-slice adapter receipt schema or seal changed")
    if (
        set(receipt.get("claims", {}))
        != {"formal_target_checked", "lean_available", "scientific_truth_inferred"}
        or set(receipt.get("executable", {}))
        != {"configured", "discovered", "discovery_source", "identity_sha256"}
        or set(receipt.get("execution", {}))
        != {"attempted", "exit_code", "stderr_sha256", "stdout_sha256", "timed_out"}
        or set(receipt.get("dependency_audit", {}))
        != {"closure_valid", "dependencies", "protocol_version", "reported_target"}
    ):
        raise ValueError("Lean vertical-slice adapter nested schema changed")
    expected_manifest = build_allowed_premise_manifest(
        LeanAdapterConfig(
            target="Invariant.kernelSmoke",
            allowed_premises=("Eq.refl",),
            forbidden_premises=("Classical.choice", "False.elim"),
            forbidden_prefixes=("KnownAnswer", "Unsafe"),
        )
    )
    if (
        receipt.get("schema_version") != LEAN_RESULT_SCHEMA
        or receipt.get("status") != "pass"
        or receipt.get("decision") != "pass_lean_checked_closed_premise"
        or receipt.get("reason") != "Lean checked the target and reported only allowed premises"
        or receipt.get("target") != "Invariant.kernelSmoke"
        or receipt.get("manifest_sha256") != expected_manifest["content_sha256"]
        or receipt.get("source_sha256") != theorem_sha
        or receipt.get("executable")
        != {
            "configured": True,
            "discovered": True,
            "discovery_source": "explicit",
            "identity_sha256": toolchain["executable_sha256"],
        }
        or receipt.get("execution", {}).get("attempted") is not True
        or receipt.get("execution", {}).get("timed_out") is not False
        or receipt.get("execution", {}).get("exit_code") != 0
        or receipt.get("execution", {}).get("stderr_sha256") != hashlib.sha256(b"").hexdigest()
        or receipt.get("execution", {}).get("stdout_sha256") != _AUDIT_STDOUT_SHA256
        or receipt.get("dependency_audit")
        != {
            "protocol_version": AUDIT_PROTOCOL,
            "reported_target": "Invariant.kernelSmoke",
            "dependencies": ["Eq.refl"],
            "closure_valid": True,
        }
        or receipt.get("claims")
        != {
            "lean_available": True,
            "formal_target_checked": True,
            "scientific_truth_inferred": False,
        }
    ):
        raise ValueError("Lean vertical-slice adapter pass semantics changed")


def validate_checked_receipt(value: Mapping[str, Any], *, root: Path) -> None:
    if set(value) != _RESULT_KEYS or value.get("content_sha256") != _content_sha(value):
        raise ValueError("Lean vertical-slice result keys or seal changed")
    encoded = json.dumps(value, sort_keys=True)
    if re.search(r"[A-Za-z]:\\|/(?:home|Users)/", encoded):
        raise ValueError("Lean vertical-slice checked receipt persisted a host path")
    config_path = _inside(root, CONFIG_PATH)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config, root=root)
    theorem_path = _inside(root, THEOREM_PATH)
    theorem_sha = _file_sha(theorem_path)
    toolchain = value.get("toolchain_receipt")
    if not isinstance(toolchain, Mapping):
        raise TypeError("Lean vertical-slice toolchain receipt missing")
    platform_key = toolchain.get("platform")
    asset = PLATFORM_ASSETS.get(str(platform_key))
    executable_sha = toolchain.get("executable_sha256")
    toolchain_keys = {
        "archive_and_binary_outside_git_history",
        "commit",
        "executable_path_persisted",
        "executable_sha256",
        "official_release",
        "platform",
        "platform_asset",
        "registered_executable_sha256_matched",
        "resolution_source",
        "version",
        "version_output",
        "version_probe_exit_code",
    }
    if (
        value.get("schema_version") != RESULT_SCHEMA
        or value.get("campaign_id") != CAMPAIGN_ID
        or value.get("receipt_role") not in {"checked_windows_historical", "live_replay"}
        or value.get("decision") != "pass_real_lean_kernel_vertical_slice"
        or value.get("first_blocker") != "none_for_bounded_kernel_smoke_theorem"
        or set(toolchain) != toolchain_keys
        or asset is None
        or toolchain.get("official_release") != RELEASE
        or toolchain.get("version") != VERSION
        or toolchain.get("commit") != COMMIT
        or toolchain.get("platform_asset") != asset
        or not isinstance(executable_sha, str)
        or not re.fullmatch(r"[0-9a-f]{64}", executable_sha)
        or toolchain.get("registered_executable_sha256_matched")
        is not (True if asset["executable_sha256"] is not None else None)
        or (asset["executable_sha256"] is not None and executable_sha != asset["executable_sha256"])
        or not _VERSION_PATTERN.fullmatch(str(toolchain.get("version_output")))
        or toolchain.get("version_probe_exit_code") != 0
        or toolchain.get("resolution_source") not in {"environment", "PATH"}
        or toolchain.get("executable_path_persisted") is not False
        or toolchain.get("archive_and_binary_outside_git_history") is not True
        or (
            value.get("receipt_role") == "checked_windows_historical"
            and platform_key != "windows-x86_64"
        )
    ):
        raise ValueError("Lean vertical-slice registered toolchain semantics changed")
    _validate_adapter_semantics(value["adapter_receipt"], theorem_sha, toolchain)
    expected_manifest = build_allowed_premise_manifest(
        LeanAdapterConfig(
            target="Invariant.kernelSmoke",
            allowed_premises=("Eq.refl",),
            forbidden_premises=("Classical.choice", "False.elim"),
            forbidden_prefixes=("KnownAnswer", "Unsafe"),
        )
    )
    expected_scope = (
        "one known-answer Nat reflexivity theorem and its declared Eq.refl premise closure; "
        "no scientific truth, general theorem-proving, production availability, or physics claim"
    )
    if (
        value.get("theorem_contract")
        != {
            "target": "Invariant.kernelSmoke",
            "statement": "for every natural number n, n = n",
            "proof_term": "Eq.refl n",
            "known_answer_role": "bounded_kernel_and_dependency_protocol_smoke_theorem_only",
            "source_sha256": theorem_sha,
            "source_bytes": theorem_path.stat().st_size,
        }
        or value.get("dependency_closure")
        != {
            "allowed_premise_manifest": expected_manifest,
            "declared_dependencies": ["Eq.refl"],
            "closure_checked_by_adapter": True,
            "target_or_equivalent_allowed_as_dependency": False,
            "out_of_manifest_dependency_allowed": False,
        }
        or value.get("readiness_counts")
        != {
            "theorems_presented": 1,
            "kernel_executions_attempted": 1,
            "kernel_checked_theorems": 1,
            "blocked_theorems": 0,
            "rejected_theorems": 0,
        }
        or value.get("claim_seals")
        != {
            "lean_executable_discovered": True,
            "bounded_theorem_kernel_checked": True,
            "dependency_closure_validated": True,
            "scientific_truth_inferred": False,
            "general_formal_completion_claimed": False,
            "physics_claimed": False,
        }
        or set(value.get("source_bindings", {}))
        != {"source", "config", "test", "theorem", "adapter"}
        or any(
            binding
            != {
                "path": path,
                "file_sha256": _file_sha(_inside(root, path)),
            }
            for binding, path in (
                (value["source_bindings"]["source"], SOURCE_PATH),
                (value["source_bindings"]["config"], CONFIG_PATH),
                (value["source_bindings"]["test"], TEST_PATH),
                (value["source_bindings"]["theorem"], THEOREM_PATH),
            )
        )
        or value["source_bindings"].get("adapter") != ADAPTER_BINDING
        or value.get("scope") != expected_scope
    ):
        raise ValueError("Lean vertical-slice checked evidence bindings changed")


def validate_live_receipt(
    value: Mapping[str, Any],
    config_path: Path,
    *,
    environment: Mapping[str, str] | None = None,
) -> None:
    root = config_path.resolve().parents[2]
    validate_checked_receipt(value, root=root)
    if value.get("receipt_role") != "live_replay":
        raise ValueError("Lean live receipt role changed")
    expected = build_live_receipt(config_path, environment=environment)
    if value != expected:
        raise ValueError("Lean live receipt differs from current kernel replay")


def write_live_receipt(
    config_path: Path,
    output_path: Path,
    *,
    environment: Mapping[str, str] | None = None,
    receipt_role: str = "live_replay",
) -> Path:
    receipt = build_live_receipt(config_path, environment=environment, receipt_role=receipt_role)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(CONFIG_PATH))
    parser.add_argument(
        "--mode", choices=("validate-checked", "emit-live"), default="validate-checked"
    )
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    root = arguments.config.resolve().parents[2]
    if arguments.mode == "validate-checked":
        checked = json.loads(_inside(root, OUTPUT_PATH).read_text(encoding="utf-8"))
        validate_checked_receipt(checked, root=root)
        print(_inside(root, OUTPUT_PATH))
        return 0
    if arguments.output is None:
        parser.error("--output is required for emit-live")
    output = arguments.output.resolve()
    write_live_receipt(arguments.config, output)
    live = json.loads(output.read_text(encoding="utf-8"))
    validate_live_receipt(live, arguments.config)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
