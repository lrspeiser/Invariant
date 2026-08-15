"""Bounded file API and CLI for Formula Discovery Job v1/v2 orchestration.

Run with ``python -m sigma_theory_compiler.formula_discovery_cli``. Formula-job commands perform no
network, subprocess, database, or secret access. Proof-v2 commands launch only the registered,
owned Lean 4.33 child process. Outputs are exclusively published at caller-selected new paths and
existing files are never overwritten.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .formula_discovery_job import SYSTEM_CAPS, run_formula_discovery_job
from .formula_discovery_job_v2 import PROBLEM_SCHEMA as PROBLEM_SCHEMA_V2
from .formula_discovery_job_v2 import SYSTEM_CAPS as SYSTEM_CAPS_V2
from .formula_discovery_job_v2 import run_formula_discovery_job_v2
from .formula_discovery_orchestration import (
    build_formula_discovery_orchestration,
    validate_formula_discovery_orchestration,
)
from .formula_discovery_proof_v2 import (
    FormulaDiscoveryProofV2Error,
    build_proof_v2_receipt,
    load_proof_v2_receipt,
    validate_live_proof_v2_receipt,
    write_proof_v2_receipt,
)
from .sigma_core import SchemaViolation, canonical_sha256

PUBLIC_RESULT_SCHEMA = "sigma-formula-discovery-public-result-1.0"
PUBLIC_RESULT_SCHEMA_V2 = "sigma-formula-discovery-public-result-2.0"
MAXIMUM_PROBLEM_FILE_BYTES = max(
    SYSTEM_CAPS["max_problem_bytes"], SYSTEM_CAPS_V2["max_problem_bytes"]
)
MAXIMUM_RESULT_FILE_BYTES = 16 * 1024 * 1024

EXIT_PASS = 0
EXIT_REJECT = 10
EXIT_BLOCK = 20
EXIT_SCHEMA_ERROR = 30
EXIT_OPERATIONAL_ERROR = 40

CLAIMS = {
    "candidate_is_scientific_law": False,
    "existing_output_overwritten": False,
    "network_access_performed": False,
    "novelty_established": False,
    "promotion_authorized": False,
}

_RESULT_KEYS = {
    "claims",
    "content_sha256",
    "decision",
    "discovery_job",
    "orchestration",
    "problem_sha256",
    "reason_codes",
    "schema_version",
    "scope",
}
_EXIT_CODES = {
    "PASS": EXIT_PASS,
    "REJECT": EXIT_REJECT,
    "BLOCK": EXIT_BLOCK,
    "SCHEMA_ERROR": EXIT_SCHEMA_ERROR,
}


class PublicFormulaDiscoveryError(ValueError):
    """Base error for bounded file/API operations."""


class ProblemFileError(PublicFormulaDiscoveryError):
    """The input is not one bounded strict JSON object."""


class ImmutableOutputError(PublicFormulaDiscoveryError):
    """An output path is unsafe, already exists, or could not be published."""


class PublicReplayError(PublicFormulaDiscoveryError):
    """A stored public result or report does not replay exactly."""


def _reject_float(_: str) -> None:
    raise ProblemFileError("floating-point JSON numbers are forbidden")


def _reject_constant(_: str) -> None:
    raise ProblemFileError("non-finite JSON constants are forbidden")


def _closed_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProblemFileError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _load_json_object(path: Path, *, maximum_bytes: int, label: str) -> dict[str, Any]:
    try:
        if not path.is_file():
            raise ProblemFileError(f"{label} path is not a regular file")
        payload = path.read_bytes()
    except OSError as error:
        raise ProblemFileError(f"could not read {label}") from error
    if not payload or len(payload) > maximum_bytes:
        raise ProblemFileError(f"{label} byte budget violated")
    try:
        text = payload.decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=_closed_object,
            parse_float=_reject_float,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProblemFileError(f"{label} is not strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ProblemFileError(f"{label} must be one JSON object")
    return value


def run_public_formula_discovery(problem: Mapping[str, Any]) -> dict[str, Any]:
    """Run the job and, when it emits a candidate, the hard-gate/Pareto adapter."""

    is_v2 = problem.get("schema_version") == PROBLEM_SCHEMA_V2
    discovery = (
        run_formula_discovery_job_v2(problem) if is_v2 else run_formula_discovery_job(problem)
    )
    if not discovery["problem_schema_valid"]:
        decision = "SCHEMA_ERROR"
    else:
        decision = discovery["decision"]
    orchestration = None
    if discovery["candidate"] is not None:
        batch_id = f"public.{canonical_sha256(problem)[:24]}"
        orchestration = build_formula_discovery_orchestration((problem,), batch_id=batch_id)
        validate_formula_discovery_orchestration(orchestration, (problem,))
    body = {
        "schema_version": PUBLIC_RESULT_SCHEMA_V2 if is_v2 else PUBLIC_RESULT_SCHEMA,
        "problem_sha256": discovery["problem_sha256"],
        "decision": decision,
        "reason_codes": list(discovery["reason_codes"]),
        "discovery_job": discovery,
        "orchestration": orchestration,
        "claims": dict(CLAIMS),
        "scope": (
            f"bounded exact Formula Discovery Job {'v2' if is_v2 else 'v1'} plus hard-gate "
            "evaluation and Pareto when a "
            "candidate exists; this receipt does not establish truth, novelty, or promotion"
        ),
    }
    return {**body, "content_sha256": canonical_sha256(body)}


def validate_public_formula_discovery(
    result: Mapping[str, Any], problem: Mapping[str, Any]
) -> None:
    """Validate the closed envelope and deterministically replay every nested layer."""

    expected_schema = (
        PUBLIC_RESULT_SCHEMA_V2
        if problem.get("schema_version") == PROBLEM_SCHEMA_V2
        else PUBLIC_RESULT_SCHEMA
    )
    if set(result) != _RESULT_KEYS or result.get("schema_version") != expected_schema:
        raise PublicReplayError("public result schema changed")
    body = {key: value for key, value in result.items() if key != "content_sha256"}
    try:
        expected_seal = canonical_sha256(body)
    except (SchemaViolation, TypeError, ValueError) as error:
        raise PublicReplayError("public result is not canonical JSON") from error
    if result.get("content_sha256") != expected_seal:
        raise PublicReplayError("public result seal changed")
    expected = run_public_formula_discovery(problem)
    if dict(result) != expected:
        raise PublicReplayError("public result exact replay changed")


def _markdown_rational(value: Mapping[str, Any] | None) -> str:
    if value is None:
        return "undefined"
    numerator, denominator = value["numerator"], value["denominator"]
    return str(numerator) if denominator == 1 else f"{numerator}/{denominator}"


def _markdown_point(value: Mapping[str, Any]) -> str:
    if set(value) == {"numerator", "denominator"}:
        return _markdown_rational(value)
    return ", ".join(
        f"{name}={_markdown_rational(coordinate)}" for name, coordinate in sorted(value.items())
    )


def render_formula_discovery_markdown(result: Mapping[str, Any]) -> str:
    """Render a deterministic human-readable report from one validated result."""

    job = result["discovery_job"]
    lines = [
        "# Formula Discovery Report",
        "",
        f"- Decision: **{result['decision']}**",
        f"- Job: `{job['job_id']}`",
        f"- Discovery class: `{job.get('class_id') or 'linear_basis_v1'}`",
        f"- Problem SHA-256: `{result['problem_sha256'] or 'unavailable'}`",
        f"- Result SHA-256: `{result['content_sha256']}`",
        "",
        "## Discovery",
        "",
        f"- Job decision: `{job['decision']}`",
        f"- Reason codes: `{', '.join(job['reason_codes']) or 'none'}`",
        f"- Candidates emitted: {job['counts']['candidates_emitted']}",
        f"- Synthesis rows: {job['counts']['synthesis_rows']}",
        f"- Validation rows checked: {job['counts']['validation_rows_checked']}",
    ]
    synthesis = job["synthesis"]
    if synthesis is not None:
        lines.extend(
            (
                f"- Synthesis classification: `{synthesis['outcome']}`",
                (
                    "- Exact rank: not applicable (finite exact enumeration)"
                    if synthesis["rank"] is None
                    else f"- Exact rank: {synthesis['rank']} / {synthesis['column_count']}"
                ),
            )
        )
        if synthesis["expression"] is not None:
            lines.append(f"- Recovered expression: `{synthesis['expression']}`")
    validation = job["validation"]
    if validation is not None and validation["counterexample"] is not None:
        witness = validation["counterexample"]
        lines.extend(
            (
                "",
                "## Exact counterexample",
                "",
                f"- Validation row: {witness['row_index']}",
                f"- Point: `{_markdown_point(witness['point'])}`",
                f"- Expected: `{_markdown_rational(witness['expected'])}`",
                f"- Observed: `{_markdown_rational(witness['observed'])}`",
                f"- Residual: `{_markdown_rational(witness['residual'])}`",
            )
        )
    orchestration = result["orchestration"]
    lines.extend(("", "## Hard-gate evaluation", ""))
    if orchestration is None:
        lines.append("No candidate was emitted, so evaluation and Pareto were not opened.")
    else:
        evaluation = orchestration["evaluations"][0]
        for gate in evaluation["gate_outcomes"]:
            reasons = ", ".join(gate["reason_codes"]) or "none"
            lines.append(f"- `{gate['gate_id']}`: **{gate['status'].upper()}** ({reasons})")
        lines.extend(
            (
                "",
                "## Exact Pareto",
                "",
                (
                    "- Hard-gate eligible candidates: "
                    f"{orchestration['pareto']['counts']['hard_gate_eligible']}"
                ),
                f"- Pareto fronts: {orchestration['pareto']['counts']['pareto_fronts']}",
            )
        )
    lines.extend(
        (
            "",
            "## Claim boundary",
            "",
            "This report does not establish scientific truth, novelty, or authorization to promote.",
            "",
        )
    )
    return "\n".join(lines)


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def _normalized_path(path: Path) -> str:
    return os.path.normcase(os.path.abspath(path))


def _prepare_output_paths(problem_path: Path, result_path: Path, report_path: Path) -> None:
    paths = (problem_path, result_path, report_path)
    if len({_normalized_path(path) for path in paths}) != len(paths):
        raise ImmutableOutputError("problem and output paths must be distinct")
    for path in (result_path, report_path):
        if path.exists():
            raise ImmutableOutputError(f"output already exists: {path}")
        if not path.parent.is_dir():
            raise ImmutableOutputError(f"output parent is not a directory: {path.parent}")


def _staged_file(target: Path, payload: bytes) -> Path:
    try:
        descriptor, raw_path = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
        )
        temporary = Path(raw_path)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        return temporary
    except OSError as error:
        raise ImmutableOutputError(f"could not stage output for {target}") from error


def _publish_pair(
    result_path: Path, result_payload: bytes, report_path: Path, report_payload: bytes
) -> None:
    """Publish two staged files with exclusive hard links and rollback partial publication."""

    result_temp = _staged_file(result_path, result_payload)
    report_temp: Path | None = None
    result_created = False
    report_created = False
    try:
        report_temp = _staged_file(report_path, report_payload)
        os.link(result_temp, result_path)
        result_created = True
        os.link(report_temp, report_path)
        report_created = True
    except OSError as error:
        if report_created and report_path.read_bytes() == report_payload:
            report_path.unlink()
        if result_created and result_path.read_bytes() == result_payload:
            result_path.unlink()
        raise ImmutableOutputError("atomic immutable output publication failed") from error
    finally:
        for temporary in (result_temp, report_temp):
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass


def run_formula_discovery_files(
    problem_path: Path, result_path: Path, report_path: Path
) -> dict[str, Any]:
    """Read one problem and immutably publish its sealed JSON result and Markdown report."""

    _prepare_output_paths(problem_path, result_path, report_path)
    problem = _load_json_object(
        problem_path, maximum_bytes=MAXIMUM_PROBLEM_FILE_BYTES, label="problem"
    )
    result = run_public_formula_discovery(problem)
    validate_public_formula_discovery(result, problem)
    report = render_formula_discovery_markdown(result)
    _publish_pair(result_path, _json_bytes(result), report_path, report.encode("utf-8"))
    return result


def validate_formula_discovery_files(
    problem_path: Path, result_path: Path, report_path: Path
) -> dict[str, Any]:
    """Read-only validation of the problem, sealed result, Markdown, and exact replay."""

    problem = _load_json_object(
        problem_path, maximum_bytes=MAXIMUM_PROBLEM_FILE_BYTES, label="problem"
    )
    result = _load_json_object(result_path, maximum_bytes=MAXIMUM_RESULT_FILE_BYTES, label="result")
    validate_public_formula_discovery(result, problem)
    try:
        report = report_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise PublicReplayError("could not read UTF-8 Markdown report") from error
    if report != render_formula_discovery_markdown(result):
        raise PublicReplayError("Markdown report exact replay changed")
    return result


def exit_code_for_result(result: Mapping[str, Any]) -> int:
    try:
        return _EXIT_CODES[result["decision"]]
    except (KeyError, TypeError) as error:
        raise PublicReplayError("public result decision changed") from error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sigma-formula-discovery")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "validate"):
        child = subparsers.add_parser(command)
        child.add_argument("--problem", type=Path, required=True)
        child.add_argument("--result", type=Path, required=True)
        child.add_argument("--report", type=Path, required=True)
    for command in ("proof-v2-run", "proof-v2-validate"):
        child = subparsers.add_parser(command)
        child.add_argument("--result", type=Path, required=True)
        child.add_argument("--lean", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "proof-v2-run":
            result = build_proof_v2_receipt(executable=arguments.lean)
            write_proof_v2_receipt(result, arguments.result)
        elif arguments.command == "proof-v2-validate":
            result = load_proof_v2_receipt(arguments.result)
            validate_live_proof_v2_receipt(result, executable=arguments.lean)
        elif arguments.command == "run":
            result = run_formula_discovery_files(
                arguments.problem, arguments.result, arguments.report
            )
        else:
            result = validate_formula_discovery_files(
                arguments.problem, arguments.result, arguments.report
            )
    except ProblemFileError as error:
        print(json.dumps({"decision": "SCHEMA_ERROR", "error": str(error)}))
        return EXIT_SCHEMA_ERROR
    except (PublicFormulaDiscoveryError, FormulaDiscoveryProofV2Error) as error:
        print(json.dumps({"decision": "OPERATIONAL_ERROR", "error": str(error)}))
        return EXIT_OPERATIONAL_ERROR
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "content_sha256": result["content_sha256"],
            },
            sort_keys=True,
        )
    )
    return exit_code_for_result(result)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EXIT_BLOCK",
    "EXIT_OPERATIONAL_ERROR",
    "EXIT_PASS",
    "EXIT_REJECT",
    "EXIT_SCHEMA_ERROR",
    "PUBLIC_RESULT_SCHEMA",
    "PUBLIC_RESULT_SCHEMA_V2",
    "ImmutableOutputError",
    "ProblemFileError",
    "PublicFormulaDiscoveryError",
    "PublicReplayError",
    "exit_code_for_result",
    "main",
    "render_formula_discovery_markdown",
    "run_formula_discovery_files",
    "run_public_formula_discovery",
    "validate_formula_discovery_files",
    "validate_public_formula_discovery",
]
