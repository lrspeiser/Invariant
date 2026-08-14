"""Bounded counterexample-guided repair for exact Formula Discovery jobs.

The repair boundary is deliberately small and auditable: round one must produce an
exact held-out counterexample, that witness becomes a synthesis constraint in round
two, and a declared basis extension is the only permitted search change.  The loop
has a hard two-round cap.  A surviving failure is therefore a typed BLOCK rather
than an unbounded retry.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from .formula_discovery_job import (
    run_formula_discovery_job,
    validate_formula_discovery_result,
)
from .formula_discovery_lean_translation import (
    translate_formula_discovery_pass,
    validate_formula_discovery_lean_translation,
    write_lean_source,
)
from .math_lean_adapter import LeanAdapterConfig, run_lean_adapter
from .sigma_core import canonical_sha256

CONFIG_SCHEMA = "system5-counterexample-guided-repair-config-1.0"
TRACE_SCHEMA = "system5-counterexample-guided-repair-trace-1.0"
RECEIPT_SCHEMA = "system5-counterexample-guided-repair-receipt-1.0"
LEAN_SUMMARY_SCHEMA = "system5-portable-lean-check-1.0"

_CONFIG_KEYS = {
    "initial_problem",
    "max_rounds",
    "repair_basis_append",
    "repair_id",
    "schema_version",
}
_TRACE_KEYS = {
    "claims",
    "content_sha256",
    "counterexample_constraint",
    "counts",
    "decision",
    "final_candidate",
    "final_problem",
    "final_translation",
    "reason_codes",
    "repair_config_sha256",
    "repair_id",
    "rounds",
    "schema_version",
    "scope",
}
_RECEIPT_KEYS = {
    "claims",
    "content_sha256",
    "decision",
    "kernel_check",
    "repair_config_sha256",
    "repair_trace",
    "schema_version",
    "scope",
    "toolchain",
}
_KERNEL_KEYS = {
    "claims",
    "content_sha256",
    "decision",
    "dependency_audit",
    "execution",
    "manifest_sha256",
    "schema_version",
    "source_sha256",
    "target",
}


class CounterexampleGuidedRepairError(ValueError):
    """Raised when a repair trace, receipt, or configuration fails closed."""


def _seal(body: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(body)
    value["content_sha256"] = canonical_sha256(value)
    return value


def _validate_config(value: Mapping[str, Any]) -> dict[str, Any]:
    if set(value) != _CONFIG_KEYS or value.get("schema_version") != CONFIG_SCHEMA:
        raise CounterexampleGuidedRepairError("repair config schema changed")
    repair_id = value.get("repair_id")
    basis_append = value.get("repair_basis_append")
    if (
        not isinstance(repair_id, str)
        or not repair_id
        or len(repair_id) > 96
        or value.get("max_rounds") != 2
        or not isinstance(basis_append, list)
        or not basis_append
        or len(basis_append) > 8
        or any(not isinstance(item, str) or not item for item in basis_append)
        or not isinstance(value.get("initial_problem"), Mapping)
    ):
        raise CounterexampleGuidedRepairError("repair config is malformed or unbounded")
    return deepcopy(dict(value))


def _round_record(
    index: int,
    problem: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    consumed_witness_sha256: str | None,
) -> dict[str, Any]:
    candidate = result.get("candidate")
    return {
        "round": index,
        "problem_sha256": canonical_sha256(problem),
        "result_content_sha256": result["content_sha256"],
        "decision": result["decision"],
        "reason_codes": list(result["reason_codes"]),
        "candidate_content_sha256": (
            candidate.get("content_sha256") if isinstance(candidate, Mapping) else None
        ),
        "synthesis_rows": result["counts"]["synthesis_rows"],
        "validation_rows_checked": result["counts"]["validation_rows_checked"],
        "counterexamples_found": result["counts"]["counterexamples_found"],
        "consumed_witness_sha256": consumed_witness_sha256,
    }


def _derive_round_two(
    config: Mapping[str, Any],
    first_result: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if first_result.get("decision") != "REJECT" or first_result.get("reason_codes") != [
        "heldout_counterexample"
    ]:
        raise CounterexampleGuidedRepairError(
            "round one did not produce the required exact counterexample"
        )
    witness = first_result.get("validation", {}).get("counterexample")
    if not isinstance(witness, Mapping) or set(witness) != {
        "expected",
        "observed",
        "point",
        "reason",
        "residual",
        "row_index",
    }:
        raise CounterexampleGuidedRepairError("round-one counterexample schema changed")

    problem = deepcopy(config["initial_problem"])
    constraints = problem.get("constraints")
    validation = problem.get("validation")
    solver = problem.get("solver")
    if (
        not isinstance(constraints, dict)
        or constraints.get("kind") != "evaluations"
        or not isinstance(constraints.get("rows"), list)
        or not isinstance(validation, dict)
        or validation.get("kind") != "evaluations"
        or not isinstance(validation.get("rows"), list)
        or not isinstance(solver, dict)
        or solver.get("kind") != "exact_linear_basis_v1"
        or not isinstance(solver.get("basis"), list)
    ):
        raise CounterexampleGuidedRepairError("repair supports evaluation jobs only")
    row_index = witness["row_index"]
    rows = validation["rows"]
    if (
        isinstance(row_index, bool)
        or not isinstance(row_index, int)
        or not 0 <= row_index < len(rows)
        or rows[row_index].get("point") != witness["point"]
        or rows[row_index].get("value") != witness["expected"]
    ):
        raise CounterexampleGuidedRepairError("counterexample is not bound to validation input")
    if any(row.get("point") == witness["point"] for row in constraints["rows"]):
        raise CounterexampleGuidedRepairError("counterexample point was already used for synthesis")

    counterexample_constraint = {
        "point": deepcopy(witness["point"]),
        "value": deepcopy(witness["expected"]),
        "source_round": 1,
        "witness_sha256": canonical_sha256(witness),
    }
    constraints["rows"].append(
        {"point": deepcopy(witness["point"]), "value": deepcopy(witness["expected"])}
    )
    del rows[row_index]
    if not rows:
        raise CounterexampleGuidedRepairError("repair consumed the entire validation set")
    old_basis = list(solver["basis"])
    additions = list(config["repair_basis_append"])
    if any(item in old_basis for item in additions):
        raise CounterexampleGuidedRepairError("repair basis extension is not new")
    solver["basis"] = old_basis + additions
    problem["job_id"] = f"{config['repair_id']}.round2"
    return problem, counterexample_constraint


def build_repair_trace(config: Mapping[str, Any]) -> dict[str, Any]:
    """Run the bounded two-round repair and return a deterministic sealed trace."""

    parsed = _validate_config(config)
    first_problem = parsed["initial_problem"]
    first = run_formula_discovery_job(first_problem)
    validate_formula_discovery_result(first, first_problem)
    second_problem, witness_constraint = _derive_round_two(parsed, first)
    witness_sha = witness_constraint["witness_sha256"]
    second = run_formula_discovery_job(second_problem)
    validate_formula_discovery_result(second, second_problem)

    translation = None
    if second["decision"] == "PASS":
        translation = translate_formula_discovery_pass(second, second_problem)
        validate_formula_discovery_lean_translation(translation, second, second_problem)
        decision = "PASS"
        reason_codes = ["counterexample_guided_repair_pass"]
    else:
        decision = "BLOCK"
        reason_codes = [
            "repair_budget_exhausted" if second["decision"] == "REJECT" else "repair_round_blocked"
        ]

    body = {
        "schema_version": TRACE_SCHEMA,
        "repair_id": parsed["repair_id"],
        "repair_config_sha256": canonical_sha256(parsed),
        "decision": decision,
        "reason_codes": reason_codes,
        "rounds": [
            _round_record(1, first_problem, first, consumed_witness_sha256=None),
            _round_record(
                2,
                second_problem,
                second,
                consumed_witness_sha256=witness_sha,
            ),
        ],
        "counterexample_constraint": witness_constraint,
        "final_problem": second_problem,
        "final_candidate": second["candidate"] if decision == "PASS" else None,
        "final_translation": translation,
        "counts": {
            "rounds_executed": 2,
            "counterexamples_consumed": 1,
            "basis_terms_added": len(parsed["repair_basis_append"]),
            "final_candidates_passed": int(decision == "PASS"),
            "kernel_checks_executed": 0,
        },
        "claims": {
            "first_candidate_exactly_rejected": True,
            "counterexample_constrained_round_two": True,
            "bounded_loop_terminated": True,
            "final_candidate_independently_validated": decision == "PASS",
            "lean_kernel_executed": False,
            "general_repair_completeness": False,
            "novelty_established": False,
        },
        "scope": (
            "two-round exact evaluation repair with one declared basis extension; a PASS is "
            "only a validated candidate inside that bounded representation"
        ),
    }
    return _seal(body)


def validate_repair_trace(value: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    """Reject any trace change, including a resealed chronology or witness mutation."""

    if set(value) != _TRACE_KEYS or value.get("schema_version") != TRACE_SCHEMA:
        raise CounterexampleGuidedRepairError("repair trace schema changed")
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("content_sha256") != canonical_sha256(body):
        raise CounterexampleGuidedRepairError("repair trace seal changed")
    if dict(value) != build_repair_trace(config):
        raise CounterexampleGuidedRepairError("repair trace exact replay changed")


def _resolve_lean_executable() -> Path | None:
    configured = os.environ.get("INVARIANT_LEAN_EXECUTABLE")
    if configured and Path(configured).is_file():
        return Path(configured)
    discovered = shutil.which("lean")
    if discovered:
        return Path(discovered)
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
    return candidate if candidate.is_file() else None


def _portable_kernel_summary(adapter: Mapping[str, Any]) -> dict[str, Any]:
    return _seal(
        {
            "schema_version": LEAN_SUMMARY_SCHEMA,
            "decision": adapter["decision"],
            "target": adapter["target"],
            "manifest_sha256": adapter["manifest_sha256"],
            "source_sha256": adapter["source_sha256"],
            "execution": dict(adapter["execution"]),
            "dependency_audit": dict(adapter["dependency_audit"]),
            "claims": dict(adapter["claims"]),
        }
    )


def build_checked_receipt(
    config: Mapping[str, Any], *, lean_executable: str | Path | None = None
) -> dict[str, Any]:
    """Execute the generated repaired theorem and return a path-free receipt."""

    parsed = _validate_config(config)
    trace = build_repair_trace(parsed)
    kernel_check = None
    if trace["decision"] == "PASS":
        translation = trace["final_translation"]
        manifest = translation["premise_manifest"]
        executable = Path(lean_executable) if lean_executable else _resolve_lean_executable()
        with tempfile.TemporaryDirectory(prefix="invariant-system5-") as temporary:
            directory = Path(temporary)
            source_path = write_lean_source(translation, directory / "RepairedFormula.lean")
            (directory / "lean-toolchain").write_text(
                "leanprover/lean4:v4.33.0\n", encoding="utf-8", newline="\n"
            )
            adapter = run_lean_adapter(
                LeanAdapterConfig(
                    target=translation["target"],
                    allowed_premises=tuple(manifest["allowed_premises"]),
                    forbidden_premises=tuple(manifest["forbidden_premises"]),
                    forbidden_prefixes=tuple(
                        item.removesuffix(".") for item in manifest["forbidden_prefixes"]
                    ),
                    executable=executable,
                    timeout_seconds=30,
                ),
                source_path,
                environment={},
            )
        kernel_check = _portable_kernel_summary(adapter)
        if kernel_check["decision"] != "pass_lean_checked_closed_premise":
            raise CounterexampleGuidedRepairError("repaired theorem did not pass the Lean kernel")

    toolchain = Path("lean-toolchain")
    toolchain_text = toolchain.read_text(encoding="utf-8") if toolchain.is_file() else ""
    body = {
        "schema_version": RECEIPT_SCHEMA,
        "decision": trace["decision"],
        "repair_config_sha256": canonical_sha256(parsed),
        "repair_trace": trace,
        "kernel_check": kernel_check,
        "toolchain": {
            "declaration": toolchain_text.strip(),
            "file_sha256": __import__("hashlib").sha256(toolchain_text.encode()).hexdigest(),
        },
        "claims": {
            "counterexample_repair_replayed": True,
            "lean_kernel_executed": kernel_check is not None,
            "closed_premise_kernel_pass": kernel_check is not None,
            "general_repair_completeness": False,
            "scientific_truth_inferred": False,
        },
        "scope": (
            "path-free checked evidence for one bounded two-round repair; no completeness, "
            "novelty, or scientific promotion follows"
        ),
    }
    return _seal(body)


def validate_checked_receipt(value: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    """Validate a historical checked receipt without requiring a host executable path."""

    if set(value) != _RECEIPT_KEYS or value.get("schema_version") != RECEIPT_SCHEMA:
        raise CounterexampleGuidedRepairError("checked receipt schema changed")
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("content_sha256") != canonical_sha256(body):
        raise CounterexampleGuidedRepairError("checked receipt seal changed")
    parsed = _validate_config(config)
    if value.get("repair_config_sha256") != canonical_sha256(parsed):
        raise CounterexampleGuidedRepairError("checked receipt config binding changed")
    expected_claims = {
        "counterexample_repair_replayed": True,
        "lean_kernel_executed": value.get("decision") == "PASS",
        "closed_premise_kernel_pass": value.get("decision") == "PASS",
        "general_repair_completeness": False,
        "scientific_truth_inferred": False,
    }
    toolchain = value.get("toolchain")
    if (
        value.get("claims") != expected_claims
        or not isinstance(toolchain, Mapping)
        or set(toolchain) != {"declaration", "file_sha256"}
        or toolchain.get("declaration") != "leanprover/lean4:v4.33.0"
        or toolchain.get("file_sha256")
        != __import__("hashlib").sha256(b"leanprover/lean4:v4.33.0\n").hexdigest()
    ):
        raise CounterexampleGuidedRepairError("checked receipt claims or toolchain changed")
    trace = value.get("repair_trace")
    if not isinstance(trace, Mapping):
        raise CounterexampleGuidedRepairError("checked receipt trace is missing")
    validate_repair_trace(trace, parsed)
    if value.get("decision") != trace.get("decision"):
        raise CounterexampleGuidedRepairError("checked receipt decision changed")
    kernel = value.get("kernel_check")
    if trace["decision"] == "PASS":
        translation = trace["final_translation"]
        if not isinstance(kernel, Mapping) or set(kernel) != _KERNEL_KEYS:
            raise CounterexampleGuidedRepairError("checked receipt kernel evidence is missing")
        kernel_body = {key: item for key, item in kernel.items() if key != "content_sha256"}
        if (
            kernel.get("schema_version") != LEAN_SUMMARY_SCHEMA
            or kernel.get("content_sha256") != canonical_sha256(kernel_body)
            or kernel.get("decision") != "pass_lean_checked_closed_premise"
            or kernel.get("target") != translation["target"]
            or kernel.get("source_sha256") != translation["source_sha256"]
            or kernel.get("manifest_sha256") != translation["premise_manifest"]["content_sha256"]
            or kernel.get("dependency_audit", {}).get("closure_valid") is not True
            or kernel.get("claims", {}).get("formal_target_checked") is not True
        ):
            raise CounterexampleGuidedRepairError("checked receipt kernel binding changed")
    elif kernel is not None:
        raise CounterexampleGuidedRepairError("BLOCK receipt contains a kernel pass")


def load_config(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise CounterexampleGuidedRepairError("repair config must be a JSON object")
    return _validate_config(value)


def write_checked_receipt(
    config_path: str | Path,
    output_path: str | Path,
    *,
    lean_executable: str | Path | None = None,
) -> Path:
    """Materialize one canonical checked receipt without embedding host paths."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    value = build_checked_receipt(load_config(config_path), lean_executable=lean_executable)
    output.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return output


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--lean-executable")
    arguments = parser.parse_args()
    write_checked_receipt(
        arguments.config,
        arguments.output,
        lean_executable=arguments.lean_executable,
    )
    return 0


__all__ = [
    "CONFIG_SCHEMA",
    "LEAN_SUMMARY_SCHEMA",
    "RECEIPT_SCHEMA",
    "TRACE_SCHEMA",
    "CounterexampleGuidedRepairError",
    "build_checked_receipt",
    "build_repair_trace",
    "load_config",
    "validate_checked_receipt",
    "validate_repair_trace",
    "write_checked_receipt",
]


if __name__ == "__main__":
    raise SystemExit(_main())
