"""Immutable file API and CLI for exact active formula identifiability."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .active_formula_identifiability import (
    SYSTEM_CAPS,
    ActiveIdentifiabilityError,
    resume_active_identifiability,
    run_active_identifiability,
    validate_initial_result,
    validate_resumed_result,
)

EXIT_PASS = 0
EXIT_REJECT = 10
EXIT_BLOCK = 20
EXIT_SCHEMA_ERROR = 30
EXIT_OPERATIONAL_ERROR = 40
MAXIMUM_INPUT_BYTES = SYSTEM_CAPS["max_problem_bytes"]
MAXIMUM_RESULT_BYTES = 16 * 1024 * 1024


class ActiveIdentifiabilityFileError(ValueError):
    """A strict input or immutable output file crossed the public boundary."""


def _reject_float(_: str) -> None:
    raise ActiveIdentifiabilityFileError("floating-point JSON numbers are forbidden")


def _reject_constant(_: str) -> None:
    raise ActiveIdentifiabilityFileError("non-finite JSON constants are forbidden")


def _closed_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ActiveIdentifiabilityFileError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _load_object(path: Path, *, maximum: int, label: str) -> dict[str, Any]:
    try:
        if not path.is_file():
            raise ActiveIdentifiabilityFileError(f"{label} path is not a regular file")
        payload = path.read_bytes()
    except OSError as error:
        raise ActiveIdentifiabilityFileError(f"could not read {label}") from error
    if not payload or len(payload) > maximum:
        raise ActiveIdentifiabilityFileError(f"{label} byte budget violated")
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_closed_object,
            parse_float=_reject_float,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ActiveIdentifiabilityFileError(f"{label} is not strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ActiveIdentifiabilityFileError(f"{label} must be one JSON object")
    return value


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def _rational_text(value: Mapping[str, Any]) -> str:
    numerator, denominator = value["numerator"], value["denominator"]
    return str(numerator) if denominator == 1 else f"{numerator}/{denominator}"


def render_initial_markdown(result: Mapping[str, Any]) -> str:
    """Render a deterministic ambiguity/query report."""

    lines = [
        "# Active Formula Identifiability",
        "",
        f"- Decision: **{result['decision']}**",
        f"- Session: `{result['session_id']}`",
        f"- Reason: `{', '.join(result['reason_codes'])}`",
        f"- Problem SHA-256: `{result['problem_sha256'] or 'unavailable'}`",
        f"- Initial result SHA-256: `{result['content_sha256']}`",
        f"- Declared hypotheses: {result['counts']['declared_hypotheses']}",
        f"- Surviving hypotheses: {result['counts']['surviving_hypotheses']}",
    ]
    ambiguity = result["ambiguity_witness"]
    if ambiguity is not None:
        lines.extend(
            (
                "",
                "## Exact ambiguity witness",
                "",
                f"- Equivalence class: `{', '.join(ambiguity['equivalence_class'])}`",
                (
                    "- Indistinguishable pair: "
                    f"`{ambiguity['indistinguishable_pair'][0]}` and "
                    f"`{ambiguity['indistinguishable_pair'][1]}`"
                ),
                "- Every survivor matches every public observation exactly.",
            )
        )
    query = result["query"]
    lines.extend(("", "## Active query", ""))
    if query is None:
        lines.append("No legal informative query was emitted within the declared budget.")
    else:
        lines.extend(
            (
                f"- Query point: `{_rational_text(query['point'])}`",
                f"- Exact answer partitions: {query['partition_count']}",
                f"- Worst-case remaining hypotheses: {query['worst_case_remaining']}",
                f"- Query receipt SHA-256: `{query['content_sha256']}`",
            )
        )
        for group in query["prediction_partition"]:
            lines.append(
                f"- Answer `{_rational_text(group['value'])}` -> "
                f"`{', '.join(group['hypothesis_ids'])}`"
            )
    lines.extend(
        (
            "",
            "## Claim boundary",
            "",
            "Ambiguous public data are never treated as a unique formula. Any eventual PASS is ",
            "unique only inside the caller-declared finite hypothesis family.",
            "",
        )
    )
    return "\n".join(lines)


def render_resumed_markdown(result: Mapping[str, Any]) -> str:
    """Render a deterministic answer/resumption report."""

    lines = [
        "# Active Formula Identification Result",
        "",
        f"- Decision: **{result['decision']}**",
        f"- Session: `{result['session_id']}`",
        f"- Reason: `{', '.join(result['reason_codes'])}`",
        f"- Initial result SHA-256: `{result['initial_result_sha256']}`",
        f"- Answer SHA-256: `{result['answer_content_sha256']}`",
        f"- Survivors after answer: {result['counts']['survivors_after_answer']}",
    ]
    if result["candidate"] is not None:
        representation = result["candidate"]["representation"]
        lines.extend(
            (
                f"- Identified hypothesis: `{representation['hypothesis_id']}`",
                f"- Identified expression: `{representation['expression']}`",
                f"- Candidate SHA-256: `{result['candidate']['content_sha256']}`",
                f"- Proof SHA-256: `{result['proof_certificate']['content_sha256']}`",
            )
        )
    lines.extend(
        (
            "",
            "## Claim boundary",
            "",
            "The proof establishes exact uniqueness only within the declared hypothesis family; ",
            "it does not establish novelty or uniqueness over an unbounded formula space.",
            "",
        )
    )
    return "\n".join(lines)


def _normalized(path: Path) -> str:
    return os.path.normcase(os.path.abspath(path))


def _prepare_outputs(inputs: Sequence[Path], result_path: Path, report_path: Path) -> None:
    all_paths = (*inputs, result_path, report_path)
    if len({_normalized(path) for path in all_paths}) != len(all_paths):
        raise ActiveIdentifiabilityFileError("input and output paths must be distinct")
    for path in (result_path, report_path):
        if path.exists():
            raise ActiveIdentifiabilityFileError(f"output already exists: {path}")
        if not path.parent.is_dir():
            raise ActiveIdentifiabilityFileError(f"output parent is not a directory: {path.parent}")


def _stage(target: Path, payload: bytes) -> Path:
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
        raise ActiveIdentifiabilityFileError(f"could not stage output for {target}") from error


def _publish_pair(result_path: Path, result: bytes, report_path: Path, report: bytes) -> None:
    result_temp = _stage(result_path, result)
    report_temp: Path | None = None
    result_created = False
    report_created = False
    try:
        report_temp = _stage(report_path, report)
        os.link(result_temp, result_path)
        result_created = True
        os.link(report_temp, report_path)
        report_created = True
    except OSError as error:
        if report_created and report_path.read_bytes() == report:
            report_path.unlink()
        if result_created and result_path.read_bytes() == result:
            result_path.unlink()
        raise ActiveIdentifiabilityFileError("atomic immutable publication failed") from error
    finally:
        for temporary in (result_temp, report_temp):
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass


def start_files(problem_path: Path, result_path: Path, report_path: Path) -> dict[str, Any]:
    _prepare_outputs((problem_path,), result_path, report_path)
    problem = _load_object(problem_path, maximum=MAXIMUM_INPUT_BYTES, label="problem")
    result = run_active_identifiability(problem)
    validate_initial_result(result, problem)
    report = render_initial_markdown(result)
    _publish_pair(result_path, _json_bytes(result), report_path, report.encode("utf-8"))
    return result


def validate_start_files(
    problem_path: Path, result_path: Path, report_path: Path
) -> dict[str, Any]:
    problem = _load_object(problem_path, maximum=MAXIMUM_INPUT_BYTES, label="problem")
    result = _load_object(result_path, maximum=MAXIMUM_RESULT_BYTES, label="initial result")
    validate_initial_result(result, problem)
    try:
        report = report_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ActiveIdentifiabilityFileError("could not read initial report") from error
    if report != render_initial_markdown(result):
        raise ActiveIdentifiabilityFileError("initial Markdown replay changed")
    return result


def resume_files(
    problem_path: Path,
    initial_path: Path,
    answer_path: Path,
    result_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    _prepare_outputs((problem_path, initial_path, answer_path), result_path, report_path)
    problem = _load_object(problem_path, maximum=MAXIMUM_INPUT_BYTES, label="problem")
    initial = _load_object(initial_path, maximum=MAXIMUM_RESULT_BYTES, label="initial result")
    answer = _load_object(answer_path, maximum=MAXIMUM_INPUT_BYTES, label="answer")
    result = resume_active_identifiability(problem, initial, answer)
    validate_resumed_result(result, problem, initial, answer)
    report = render_resumed_markdown(result)
    _publish_pair(result_path, _json_bytes(result), report_path, report.encode("utf-8"))
    return result


def validate_resume_files(
    problem_path: Path,
    initial_path: Path,
    answer_path: Path,
    result_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    problem = _load_object(problem_path, maximum=MAXIMUM_INPUT_BYTES, label="problem")
    initial = _load_object(initial_path, maximum=MAXIMUM_RESULT_BYTES, label="initial result")
    answer = _load_object(answer_path, maximum=MAXIMUM_INPUT_BYTES, label="answer")
    result = _load_object(result_path, maximum=MAXIMUM_RESULT_BYTES, label="resumed result")
    validate_resumed_result(result, problem, initial, answer)
    try:
        report = report_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ActiveIdentifiabilityFileError("could not read resumed report") from error
    if report != render_resumed_markdown(result):
        raise ActiveIdentifiabilityFileError("resumed Markdown replay changed")
    return result


def _exit_code(result: Mapping[str, Any]) -> int:
    if result.get("problem_schema_valid") is False:
        return EXIT_SCHEMA_ERROR
    return {"PASS": EXIT_PASS, "REJECT": EXIT_REJECT, "BLOCK": EXIT_BLOCK}[result["decision"]]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sigma-active-identify")
    children = parser.add_subparsers(dest="command", required=True)
    for command in ("start", "validate-start"):
        child = children.add_parser(command)
        child.add_argument("--problem", type=Path, required=True)
        child.add_argument("--result", type=Path, required=True)
        child.add_argument("--report", type=Path, required=True)
    for command in ("resume", "validate-resume"):
        child = children.add_parser(command)
        child.add_argument("--problem", type=Path, required=True)
        child.add_argument("--initial", type=Path, required=True)
        child.add_argument("--answer", type=Path, required=True)
        child.add_argument("--result", type=Path, required=True)
        child.add_argument("--report", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "start":
            result = start_files(arguments.problem, arguments.result, arguments.report)
        elif arguments.command == "validate-start":
            result = validate_start_files(arguments.problem, arguments.result, arguments.report)
        elif arguments.command == "resume":
            result = resume_files(
                arguments.problem,
                arguments.initial,
                arguments.answer,
                arguments.result,
                arguments.report,
            )
        else:
            result = validate_resume_files(
                arguments.problem,
                arguments.initial,
                arguments.answer,
                arguments.result,
                arguments.report,
            )
    except ActiveIdentifiabilityFileError as error:
        print(json.dumps({"decision": "SCHEMA_ERROR", "error": str(error)}))
        return EXIT_SCHEMA_ERROR
    except ActiveIdentifiabilityError as error:
        print(json.dumps({"decision": "OPERATIONAL_ERROR", "error": str(error)}))
        return EXIT_OPERATIONAL_ERROR
    print(
        json.dumps(
            {"decision": result["decision"], "content_sha256": result["content_sha256"]},
            sort_keys=True,
        )
    )
    return _exit_code(result)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EXIT_BLOCK",
    "EXIT_OPERATIONAL_ERROR",
    "EXIT_PASS",
    "EXIT_REJECT",
    "EXIT_SCHEMA_ERROR",
    "ActiveIdentifiabilityFileError",
    "main",
    "render_initial_markdown",
    "render_resumed_markdown",
    "resume_files",
    "start_files",
    "validate_resume_files",
    "validate_start_files",
]
