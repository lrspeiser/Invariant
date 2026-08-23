"""Kernel-check the known-formula controls and reject candidate-specific mutations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .math_lean_adapter import ChildRunner, LeanAdapterConfig, run_lean_adapter
from .sigma_core import canonical_sha256

SCHEMA_VERSION = "invariant-external-creativity-lean-bridge-1.1"
SOURCE_PATH = "formal/lean/ExternalKnownFormulaControls.lean"
TOOLCHAIN_PATH = "lean-toolchain"
OUTPUT_PATH = "runs/math/external-creativity-known-controls-lean/receipt.json"
TARGET = "Invariant.externalKnownFormulaControls"
CI_REPOSITORY = "lrspeiser/Invariant"
CI_WORKFLOW = "External creativity validation"
CI_JOB = "lean-kernel"
ARTIFACT_NAME = "external-creativity-lean"
ALLOWED_PREMISES = (
    "Invariant.recoveredKineticNormalForm",
    "Invariant.recoveredSumSquaresNormalForm",
    "Nat.add_mul",
    "Nat.mul_add",
    "Nat.mul_one",
    "Rat.mul_assoc",
    "Rat.mul_comm",
)
NEGATIVE_CONTROLS: tuple[dict[str, Any], ...] = (
    {
        "control_id": "lean-unit-offset-openstax-0702",
        "benchmark_id": "external.authority-openstax-0702",
        "candidate_id": "candidate.0dd1fa8201b4bc8c2e6877d9",
        "candidate_expression": "(1/2)*x0*x1**2",
        "mutation_operator": "add_exact_unit",
        "target": "Invariant.externalOpenStaxKineticUnitOffsetFalse",
        "source_path": "formal/lean/ExternalKnownFormulaKineticUnitOffsetFalse.lean",
        "witness_inputs": {"x0": "0/1", "x1": "0/1"},
        "expected_residual": "1/1",
    },
    {
        "control_id": "lean-unit-offset-nist-0244",
        "benchmark_id": "external.authority-nist-0244",
        "candidate_id": "candidate.0082bc01d3dc5192fc25ea54",
        "candidate_expression": "x0*(x0 + 1)*(2*x0 + 1)/6",
        "mutation_operator": "add_exact_unit",
        "target": "Invariant.externalNistSumSquaresUnitOffsetFalse",
        "source_path": "formal/lean/ExternalKnownFormulaSumSquaresUnitOffsetFalse.lean",
        "witness_inputs": {"x0": "0/1"},
        "expected_residual": "1/1",
    },
)
_HEX_40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX_64 = re.compile(r"[0-9a-f]{64}\Z")


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _under(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("external creativity Lean source escaped the project root") from error
    if not path.is_file():
        raise ValueError("external creativity Lean source is missing")
    return path


def adapter_config(executable: str | Path | None = None) -> LeanAdapterConfig:
    return LeanAdapterConfig(
        target=TARGET,
        allowed_premises=ALLOWED_PREMISES,
        forbidden_premises=("Classical.choice", "False.elim"),
        forbidden_prefixes=("KnownAnswer", "Unsafe"),
        executable=executable,
        timeout_seconds=60,
    )


def negative_adapter_config(
    control: Mapping[str, Any], executable: str | Path | None = None
) -> LeanAdapterConfig:
    return LeanAdapterConfig(
        target=str(control["target"]),
        forbidden_premises=("Classical.choice", "False.elim"),
        forbidden_prefixes=("KnownAnswer", "Unsafe"),
        executable=executable,
        timeout_seconds=60,
    )


def _run_adapter(
    config: LeanAdapterConfig,
    source: Path,
    *,
    runner: ChildRunner | None,
    environment: Mapping[str, str] | None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"environment": {} if environment is None else environment}
    if runner is not None:
        kwargs["runner"] = runner
    return run_lean_adapter(config, source, **kwargs)


def _ci_provenance(environment: Mapping[str, str] | None) -> dict[str, Any]:
    environment = os.environ if environment is None else environment
    repository = environment.get("GITHUB_REPOSITORY")
    workflow = environment.get("GITHUB_WORKFLOW")
    job = environment.get("INVARIANT_EVIDENCE_JOB")
    run_id_raw = environment.get("GITHUB_RUN_ID", "")
    attempt_raw = environment.get("GITHUB_RUN_ATTEMPT", "")
    head_sha = environment.get("INVARIANT_EVIDENCE_HEAD_SHA") or environment.get("GITHUB_SHA")
    event_name = environment.get("GITHUB_EVENT_NAME")
    runner_os = environment.get("RUNNER_OS")
    runner_arch = environment.get("RUNNER_ARCH")
    complete = (
        environment.get("GITHUB_ACTIONS") == "true"
        and repository == CI_REPOSITORY
        and workflow == CI_WORKFLOW
        and job == CI_JOB
        and run_id_raw.isdigit()
        and attempt_raw.isdigit()
        and isinstance(head_sha, str)
        and _HEX_40.fullmatch(head_sha) is not None
        and event_name in {"pull_request", "push", "workflow_dispatch"}
        and runner_os == "Linux"
        and isinstance(runner_arch, str)
        and bool(runner_arch)
    )
    return {
        "provider": "github_actions" if complete else "local_or_incomplete",
        "complete": complete,
        "repository": repository,
        "workflow": workflow,
        "job": job,
        "run_id": int(run_id_raw) if run_id_raw.isdigit() else None,
        "run_attempt": int(attempt_raw) if attempt_raw.isdigit() else None,
        "head_sha": head_sha,
        "event_name": event_name,
        "runner_os": runner_os,
        "runner_arch": runner_arch,
        "artifact_name": ARTIFACT_NAME,
    }


def _rejection_receipt(
    adapter: Mapping[str, Any], control: Mapping[str, Any], source_sha256: str
) -> dict[str, Any]:
    execution = adapter.get("execution", {})
    executable = adapter.get("executable", {})
    if (
        adapter.get("status") != "block"
        or adapter.get("decision") != "block_lean_process_failure"
        or execution.get("attempted") is not True
        or execution.get("timed_out") is not False
        or not isinstance(execution.get("exit_code"), int)
        or execution["exit_code"] == 0
        or not _HEX_64.fullmatch(str(executable.get("identity_sha256", "")))
    ):
        raise ValueError(f"Lean did not reject wrong-formula control {control['control_id']}")
    body = {
        "schema_version": "invariant-external-creativity-lean-rejection-1.0",
        "target": control["target"],
        "source_sha256": source_sha256,
        "executable_identity_sha256": executable["identity_sha256"],
        "status": "block",
        "decision": "block_lean_process_failure",
        "execution": {
            "attempted": True,
            "timed_out": False,
            "nonzero_exit_code": True,
            "stdout_sha256": execution["stdout_sha256"],
            "stderr_sha256": execution["stderr_sha256"],
        },
        "diagnostic_bytes_persisted": False,
    }
    return {**body, "content_sha256": canonical_sha256(body)}


def _negative_result(
    root: Path,
    control: Mapping[str, Any],
    executable: str | Path | None,
    *,
    runner: ChildRunner | None,
    environment: Mapping[str, str] | None,
    positive_passed: bool,
) -> dict[str, Any]:
    source = _under(root, str(control["source_path"]))
    source_sha256 = _file_sha256(source)
    rejection = None
    outcome = "NOT_RUN_POSITIVE_CONTROL_BLOCKED"
    if positive_passed:
        adapter = _run_adapter(
            negative_adapter_config(control, executable),
            source,
            runner=runner,
            environment=environment,
        )
        rejection = _rejection_receipt(adapter, control, source_sha256)
        outcome = "REJECTED_BY_LEAN_KERNEL"
    return {
        **dict(control),
        "candidate_expression_sha256": canonical_sha256(
            {"expression": control["candidate_expression"]}
        ),
        "source_sha256": source_sha256,
        "outcome": outcome,
        "rejection_receipt": rejection,
    }


def run_bridge(
    root: Path,
    *,
    executable: str | Path | None = None,
    environment: Mapping[str, str] | None = None,
    ci_environment: Mapping[str, str] | None = None,
    runner: ChildRunner | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    source = _under(root, SOURCE_PATH)
    toolchain = _under(root, TOOLCHAIN_PATH)
    if executable is None:
        executable = shutil.which("lean")
    adapter = _run_adapter(
        adapter_config(executable), source, runner=runner, environment=environment
    )
    positive_passed = adapter["decision"] == "pass_lean_checked_closed_premise"
    controls = [
        _negative_result(
            root,
            control,
            executable,
            runner=runner,
            environment=environment,
            positive_passed=positive_passed,
        )
        for control in NEGATIVE_CONTROLS
    ]
    rejected = sum(item["outcome"] == "REJECTED_BY_LEAN_KERNEL" for item in controls)
    passed = positive_passed and rejected == len(NEGATIVE_CONTROLS)
    attempts = int(adapter["execution"]["attempted"]) + sum(
        item["rejection_receipt"] is not None for item in controls
    )
    body = {
        "schema_version": SCHEMA_VERSION,
        "adapter_receipt": adapter,
        "ci_provenance": _ci_provenance(ci_environment),
        "claims": {
            "known_formula_normal_forms_kernel_checked": positive_passed,
            "candidate_specific_unit_offset_mutations_kernel_rejected": passed,
            "novel_formula_established": False,
            "physical_law_proved": False,
        },
        "counts": {
            "kernel_executions_attempted": attempts,
            "kernel_positive_passes": int(positive_passed),
            "wrong_formula_controls_required": len(NEGATIVE_CONTROLS),
            "wrong_formula_controls_rejected": rejected,
        },
        "source_path": SOURCE_PATH,
        "source_sha256": _file_sha256(source),
        "status": "PASS" if passed else "BLOCKED_LEAN_UNAVAILABLE_OR_REJECTED",
        "target": TARGET,
        "toolchain": {
            "path": TOOLCHAIN_PATH,
            "source_sha256": _file_sha256(toolchain),
            "declaration": toolchain.read_text(encoding="utf-8").strip(),
        },
        "wrong_formula_kernel_controls": controls,
    }
    body["content_sha256"] = canonical_sha256(body)
    validate_receipt(body, root)
    return body


def _validate_rejection(
    value: Mapping[str, Any], control: Mapping[str, Any], executable_sha256: str
) -> None:
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("content_sha256") != canonical_sha256(body):
        raise ValueError("external creativity Lean rejection seal changed")
    execution = value.get("execution", {})
    if (
        value.get("schema_version") != "invariant-external-creativity-lean-rejection-1.0"
        or value.get("target") != control["target"]
        or value.get("source_sha256") != control["source_sha256"]
        or value.get("executable_identity_sha256") != executable_sha256
        or value.get("status") != "block"
        or value.get("decision") != "block_lean_process_failure"
        or value.get("diagnostic_bytes_persisted") is not False
        or execution.get("attempted") is not True
        or execution.get("timed_out") is not False
        or execution.get("nonzero_exit_code") is not True
        or not _HEX_64.fullmatch(str(execution.get("stdout_sha256", "")))
        or not _HEX_64.fullmatch(str(execution.get("stderr_sha256", "")))
    ):
        raise ValueError("external creativity Lean rejection semantics changed")


def validate_receipt(
    receipt: Mapping[str, Any],
    root: Path | None = None,
    *,
    require_ci_provenance: bool = False,
) -> None:
    expected_keys = {
        "adapter_receipt",
        "ci_provenance",
        "claims",
        "content_sha256",
        "counts",
        "schema_version",
        "source_path",
        "source_sha256",
        "status",
        "target",
        "toolchain",
        "wrong_formula_kernel_controls",
    }
    if set(receipt) != expected_keys:
        raise ValueError("external creativity Lean receipt schema changed")
    body = {key: value for key, value in receipt.items() if key != "content_sha256"}
    if receipt.get("content_sha256") != canonical_sha256(body):
        raise ValueError("external creativity Lean receipt seal changed")
    if receipt.get("schema_version") != SCHEMA_VERSION or receipt.get("target") != TARGET:
        raise ValueError("external creativity Lean receipt identity changed")
    adapter = receipt.get("adapter_receipt", {})
    adapter_body = {key: value for key, value in adapter.items() if key != "content_sha256"}
    if adapter.get("content_sha256") != canonical_sha256(adapter_body):
        raise ValueError("external creativity positive Lean adapter seal changed")
    positive = adapter.get("decision") == "pass_lean_checked_closed_premise"
    executable_sha256 = str(adapter.get("executable", {}).get("identity_sha256", ""))
    controls = receipt.get("wrong_formula_kernel_controls")
    if not isinstance(controls, list) or len(controls) != len(NEGATIVE_CONTROLS):
        raise ValueError("external creativity Lean mutation coverage changed")
    rejected = 0
    for item, expected in zip(controls, NEGATIVE_CONTROLS, strict=True):
        source_sha256 = item.get("source_sha256")
        expected_item = {
            **expected,
            "candidate_expression_sha256": canonical_sha256(
                {"expression": expected["candidate_expression"]}
            ),
            "source_sha256": source_sha256,
            "outcome": item.get("outcome"),
            "rejection_receipt": item.get("rejection_receipt"),
        }
        if dict(item) != expected_item or not _HEX_64.fullmatch(str(source_sha256)):
            raise ValueError("external creativity Lean mutation binding changed")
        if item["outcome"] == "REJECTED_BY_LEAN_KERNEL":
            rejection = item.get("rejection_receipt")
            if not isinstance(rejection, Mapping):
                raise ValueError("external creativity Lean rejection receipt is missing")
            _validate_rejection(rejection, item, executable_sha256)
            rejected += 1
        elif (
            item["outcome"] != "NOT_RUN_POSITIVE_CONTROL_BLOCKED"
            or item.get("rejection_receipt") is not None
        ):
            raise ValueError("external creativity Lean mutation outcome changed")
    passed = positive and rejected == len(NEGATIVE_CONTROLS)
    expected_counts = {
        "kernel_executions_attempted": int(adapter.get("execution", {}).get("attempted") is True)
        + rejected,
        "kernel_positive_passes": int(positive),
        "wrong_formula_controls_required": len(NEGATIVE_CONTROLS),
        "wrong_formula_controls_rejected": rejected,
    }
    if (
        receipt.get("counts") != expected_counts
        or receipt.get("status") != ("PASS" if passed else "BLOCKED_LEAN_UNAVAILABLE_OR_REJECTED")
        or receipt.get("claims")
        != {
            "known_formula_normal_forms_kernel_checked": positive,
            "candidate_specific_unit_offset_mutations_kernel_rejected": passed,
            "novel_formula_established": False,
            "physical_law_proved": False,
        }
    ):
        raise ValueError("external creativity Lean receipt decision changed")
    provenance = receipt.get("ci_provenance", {})
    if require_ci_provenance and provenance.get("complete") is not True:
        raise ValueError("external creativity Lean CI provenance is incomplete")
    if provenance.get("complete") is True and (
        provenance.get("provider") != "github_actions"
        or provenance.get("repository") != CI_REPOSITORY
        or provenance.get("workflow") != CI_WORKFLOW
        or provenance.get("job") != CI_JOB
        or not isinstance(provenance.get("run_id"), int)
        or not isinstance(provenance.get("run_attempt"), int)
        or not _HEX_40.fullmatch(str(provenance.get("head_sha", "")))
        or provenance.get("event_name") not in {"pull_request", "push", "workflow_dispatch"}
        or provenance.get("runner_os") != "Linux"
        or provenance.get("artifact_name") != ARTIFACT_NAME
    ):
        raise ValueError("external creativity Lean CI provenance changed")
    if root is not None:
        root = root.resolve()
        source = _under(root, SOURCE_PATH)
        toolchain = _under(root, TOOLCHAIN_PATH)
        if (
            receipt.get("source_path") != SOURCE_PATH
            or receipt.get("source_sha256") != _file_sha256(source)
            or receipt.get("toolchain")
            != {
                "path": TOOLCHAIN_PATH,
                "source_sha256": _file_sha256(toolchain),
                "declaration": toolchain.read_text(encoding="utf-8").strip(),
            }
        ):
            raise ValueError("external creativity Lean positive source binding changed")
        for item in controls:
            if item["source_sha256"] != _file_sha256(_under(root, item["source_path"])):
                raise ValueError("external creativity Lean negative source binding changed")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    receipt = run_bridge(args.root)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
