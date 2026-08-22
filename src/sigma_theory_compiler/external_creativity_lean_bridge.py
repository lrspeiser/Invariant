"""Kernel-check the two externally authored known-formula controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .math_lean_adapter import LeanAdapterConfig, run_lean_adapter
from .sigma_core import canonical_sha256

SCHEMA_VERSION = "invariant-external-creativity-lean-bridge-1.0"
SOURCE_PATH = "formal/lean/ExternalKnownFormulaControls.lean"
OUTPUT_PATH = "runs/math/external-creativity-known-controls-lean/receipt.json"
TARGET = "Invariant.externalKnownFormulaControls"
ALLOWED_PREMISES = (
    "Invariant.recoveredKineticNormalForm",
    "Invariant.recoveredSumSquaresNormalForm",
    "Lean.Parser.Tactic.ac_rfl",
    "Nat.add_mul",
    "Nat.mul_add",
    "Nat.mul_one",
)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def adapter_config(executable: str | Path | None = None) -> LeanAdapterConfig:
    return LeanAdapterConfig(
        target=TARGET,
        allowed_premises=ALLOWED_PREMISES,
        forbidden_premises=("Classical.choice", "False.elim"),
        forbidden_prefixes=("KnownAnswer", "Unsafe"),
        executable=executable,
        timeout_seconds=60,
    )


def run_bridge(
    root: Path,
    *,
    executable: str | Path | None = None,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    source = (root / SOURCE_PATH).resolve()
    if root not in source.parents or not source.is_file():
        raise ValueError("external creativity Lean source escaped the project root")
    if executable is None:
        executable = shutil.which("lean")
    adapter = run_lean_adapter(
        adapter_config(executable),
        source,
        environment={} if environment is None else environment,
    )
    passed = adapter["decision"] == "pass_lean_checked_closed_premise"
    body = {
        "schema_version": SCHEMA_VERSION,
        "adapter_receipt": adapter,
        "claims": {
            "known_formula_normal_forms_kernel_checked": passed,
            "novel_formula_established": False,
            "physical_law_proved": False,
        },
        "source_path": SOURCE_PATH,
        "source_sha256": _file_sha256(source),
        "status": "PASS" if passed else "BLOCKED_LEAN_UNAVAILABLE_OR_REJECTED",
        "target": TARGET,
    }
    body["content_sha256"] = canonical_sha256(body)
    return body


def validate_receipt(receipt: Mapping[str, Any]) -> None:
    body = {key: value for key, value in receipt.items() if key != "content_sha256"}
    if receipt.get("content_sha256") != canonical_sha256(body):
        raise ValueError("external creativity Lean receipt seal changed")
    if receipt.get("schema_version") != SCHEMA_VERSION or receipt.get("target") != TARGET:
        raise ValueError("external creativity Lean receipt identity changed")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    receipt = run_bridge(args.root)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
