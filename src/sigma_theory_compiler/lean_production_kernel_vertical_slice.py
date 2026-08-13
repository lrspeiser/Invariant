"""Run one real theorem through the production Lean adapter or seal its readiness blocker."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .math_lean_adapter import (
    LeanAdapterConfig,
    build_allowed_premise_manifest,
    run_lean_adapter,
    validate_lean_adapter_result,
)

CONFIG_SCHEMA = "invariant-lean-production-kernel-vertical-slice-config-1.0"
RESULT_SCHEMA = "invariant-lean-production-kernel-vertical-slice-result-1.0"
CAMPAIGN_ID = "lean-production-kernel-vertical-slice-001"
CONFIG_PATH = "configs/math/lean_production_kernel_vertical_slice.json"
SOURCE_PATH = "src/sigma_theory_compiler/lean_production_kernel_vertical_slice.py"
TEST_PATH = "tests/test_lean_production_kernel_vertical_slice.py"
THEOREM_PATH = "formal/lean/InvariantKernelSmoke.lean"
OUTPUT_PATH = "runs/math/lean-production-kernel-vertical-slice/receipt.json"
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
PINNED_EXECUTABLE = (
    "C:\\Users\\henry\\.cache\\invariant\\lean\\v4.33.0\\lean-4.33.0-windows\\bin\\lean.exe"
)
TOOLCHAIN = {
    "official_release": "leanprover/lean4:v4.33.0",
    "asset_url": (
        "https://github.com/leanprover/lean4/releases/download/v4.33.0/lean-4.33.0-windows.tar.zst"
    ),
    "archive_bytes": 583425362,
    "archive_sha256": "60d045a2ef45fca55a620b7d55be682e8439ec8d1fc9a8bcd2615da7dffba26a",
    "executable_sha256": "dd86e9b24990b1da425ea4af910f016e4db8f9a25c9ddad27bc6bee3690e677f",
    "version_output": (
        "Lean (version 4.33.0, x86_64-w64-windows-gnu, commit "
        "d8b18978322de05a8f3dba51ef03cf5461676c17, Release)"
    ),
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
    "schema_version",
    "scope",
    "source_bindings",
    "theorem_contract",
    "toolchain_receipt",
}


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


def _adapter_config(config: Mapping[str, Any]) -> LeanAdapterConfig:
    policy = config["premise_policy"]
    return LeanAdapterConfig(
        target=str(config["target"]),
        allowed_premises=tuple(policy["allowed_premises"]),
        equivalent_targets=tuple(policy["equivalent_targets"]),
        forbidden_premises=tuple(policy["forbidden_premises"]),
        forbidden_prefixes=tuple(policy["forbidden_prefixes"]),
        executable=str(config["executable"]),
        timeout_seconds=float(config["timeout_seconds"]),
    )


def _validate_config(config: Mapping[str, Any], *, root: Path) -> None:
    if config != {
        "schema_version": CONFIG_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "output_path": OUTPUT_PATH,
        "theorem_source_path": THEOREM_PATH,
        "target": "Invariant.kernelSmoke",
        "executable": PINNED_EXECUTABLE,
        "timeout_seconds": 30,
        "premise_policy": PREMISE_POLICY,
        "adapter_binding": ADAPTER_BINDING,
        "toolchain": TOOLCHAIN,
    }:
        raise ValueError("Lean vertical-slice config boundary changed")
    for binding in ADAPTER_BINDING.values():
        if _file_sha(_inside(root, binding["path"])) != binding["file_sha256"]:
            raise ValueError("Lean vertical-slice adapter dependency changed")


def _validate_pinned_toolchain(config: Mapping[str, Any]) -> dict[str, Any]:
    executable = Path(str(config["executable"])).resolve()
    toolchain = config["toolchain"]
    if not executable.is_file():
        raise ValueError("pinned Lean executable is missing")
    executable_sha = _file_sha(executable)
    if executable_sha != toolchain["executable_sha256"]:
        raise ValueError("pinned Lean executable identity changed")
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
        raise ValueError("pinned Lean version probe failed") from error
    stdout = process.stdout.decode("utf-8", errors="strict").strip()
    if process.returncode != 0 or process.stderr or stdout != toolchain["version_output"]:
        raise ValueError("pinned Lean version identity changed")
    return {
        **toolchain,
        "executable_path": str(executable),
        "live_executable_sha256": executable_sha,
        "live_version_output": stdout,
        "version_probe_exit_code": process.returncode,
        "archive_and_binary_outside_git_history": True,
    }


def build_receipt(config_path: Path) -> dict[str, Any]:
    root = config_path.resolve().parents[2]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config, root=root)
    toolchain_receipt = _validate_pinned_toolchain(config)
    theorem_path = _inside(root, THEOREM_PATH)
    adapter_config = _adapter_config(config)
    manifest = build_allowed_premise_manifest(adapter_config)
    adapter_receipt = run_lean_adapter(adapter_config, theorem_path)
    validate_lean_adapter_result(adapter_receipt, adapter_config, theorem_path)
    passed = adapter_receipt["decision"] == "pass_lean_checked_closed_premise"
    blocked_unavailable = adapter_receipt["decision"] == "block_lean_unavailable"
    if not (passed or blocked_unavailable):
        decision = "block_external_proof_kernel_execution_failed"
        first_blocker = str(adapter_receipt["decision"])
    elif passed:
        decision = "pass_real_lean_kernel_vertical_slice"
        first_blocker = "none_for_bounded_kernel_smoke_theorem"
    else:
        decision = "block_lean_executable_not_discovered"
        first_blocker = "install_or_explicitly_configure_a_pinned_Lean_executable"
    body = {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "decision": decision,
        "first_blocker": first_blocker,
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
            "closure_checked_by_adapter": adapter_receipt["dependency_audit"]["closure_valid"],
            "target_or_equivalent_allowed_as_dependency": False,
            "out_of_manifest_dependency_allowed": False,
        },
        "adapter_receipt": adapter_receipt,
        "toolchain_receipt": toolchain_receipt,
        "readiness_counts": {
            "theorems_presented": 1,
            "kernel_executions_attempted": int(adapter_receipt["execution"]["attempted"]),
            "kernel_checked_theorems": int(passed),
            "blocked_theorems": int(not passed),
            "rejected_theorems": int(adapter_receipt["status"] == "reject"),
        },
        "claim_seals": {
            "lean_executable_discovered": adapter_receipt["claims"]["lean_available"],
            "bounded_theorem_kernel_checked": adapter_receipt["claims"]["formal_target_checked"],
            "dependency_closure_validated": adapter_receipt["dependency_audit"]["closure_valid"],
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


def validate_receipt(value: Mapping[str, Any], *, root: Path) -> None:
    if set(value) != _RESULT_KEYS:
        raise ValueError("Lean vertical-slice result keys changed")
    if value.get("content_sha256") != _content_sha(value):
        raise ValueError("Lean vertical-slice content seal changed")
    expected = build_receipt(_inside(root, CONFIG_PATH))
    if value != expected:
        raise ValueError("Lean vertical-slice receipt differs from live adapter result")


def write_receipt(config_path: Path) -> Path:
    receipt = build_receipt(config_path)
    output = _inside(config_path.resolve().parents[2], OUTPUT_PATH)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(CONFIG_PATH))
    print(write_receipt(parser.parse_args().config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
