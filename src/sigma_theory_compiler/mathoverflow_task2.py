"""Seal and run an available-now Task 2 trial on a real MathOverflow question.

Only metadata may be inspected before authorization.  The selected question body may be opened
after authorization, while its accepted answer stays closed until all blinded submissions have
been frozen.  The public source is disclosed as weaker than a privately held future target.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import html
import json
import os
import re
import subprocess
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from . import broken_arxiv_task2 as broken
from .claude_creativity_api import (
    ClaudeCallStatus,
    ClaudeCreativityClient,
    ClaudeRole,
    Transport,
    urllib_transport,
)
from .core_credential import CredentialActivationError, activated_credential
from .sigma_core import canonical_sha256

CONFIG_SCHEMA = "invariant-mathoverflow-task2-config-1.0"
AUTHORIZATION_SCHEMA = "invariant-mathoverflow-task2-authorization-1.0"
SOURCE_SCHEMA = "invariant-mathoverflow-task2-source-check-1.0"
STAGED_SCHEMA = "invariant-mathoverflow-task2-staged-problem-1.0"
REFERENCE_SCHEMA = "invariant-mathoverflow-task2-reference-1.0"
DEFAULT_CONFIG = Path("configs/mathoverflow_task2_available_now.json")
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")

JsonFetcher = Callable[[str], Mapping[str, Any]]


class MathOverflowTask2Error(ValueError):
    """The available-now source, chronology, or trial binding failed closed."""


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag in {"br", "p", "div", "li", "pre", "blockquote", "h1", "h2", "h3"}:
            self.parts.append("\n")

    def text(self) -> str:
        return "\n".join(line.strip() for line in "".join(self.parts).splitlines() if line.strip())


def _strict_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise MathOverflowTask2Error(f"{label} keys changed")


def _sealed(body: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(body)
    result["content_sha256"] = canonical_sha256(body)
    return result


def _validate_seal(value: Mapping[str, Any], schema: str, label: str) -> None:
    if value.get("schema_version") != schema:
        raise MathOverflowTask2Error(f"{label} schema changed")
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("content_sha256") != canonical_sha256(body):
        raise MathOverflowTask2Error(f"{label} seal changed")


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MathOverflowTask2Error(f"could not read JSON: {path}") from error
    if not isinstance(value, Mapping):
        raise MathOverflowTask2Error(f"JSON root is not an object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _git_commit(root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    commit = completed.stdout.strip()
    if _COMMIT.fullmatch(commit) is None:
        raise MathOverflowTask2Error("current Git commit is invalid")
    return commit


def _fetch_json(url: str) -> Mapping[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "Invariant-task2/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        value = json.load(response)
    if not isinstance(value, Mapping):
        raise MathOverflowTask2Error("source response is not an object")
    return value


def _plain_text(value: str) -> str:
    parser = _TextExtractor()
    parser.feed(html.unescape(value))
    text = parser.text().strip()
    if not text:
        raise MathOverflowTask2Error("selected question text is empty")
    return text


def load_config(root: Path, path: Path = DEFAULT_CONFIG) -> Mapping[str, Any]:
    config_path = path if path.is_absolute() else root / path
    config = _read_json(config_path)
    _strict_keys(
        config,
        {
            "adjudication",
            "base_generation_config",
            "blindness_disclosure",
            "dependency_paths",
            "implementation_paths",
            "pass_gate",
            "schema_version",
            "selection",
            "source",
            "task_id",
        },
        "MathOverflow Task 2 config",
    )
    source = config["source"]
    disclosure = config["blindness_disclosure"]
    if (
        config["schema_version"] != CONFIG_SCHEMA
        or source.get("site") != "mathoverflow"
        or source.get("tagged") != "counterexamples"
        or source.get("pool_closes_at_authorization") is not True
        or source.get("require_accepted_answer") is not True
        or source.get("question_body_may_be_read_during_source_check") is not False
        or source.get("accepted_answer_may_be_read_before_submissions_freeze") is not False
        or config["selection"].get("algorithm") != "sha256_min_rank_v1"
        or config["selection"].get("select_count") != 1
        or config["selection"].get("manual_substitution_allowed") is not False
        or disclosure.get("preauthorization_title_fields_materialized_by_default_api_probe")
        != 100
        or disclosure.get("preauthorization_titles_displayed_to_operator") != 0
        or disclosure.get("preauthorization_titles_sent_to_candidate_generator") != 0
        or disclosure.get("preauthorization_question_bodies_seen") != 0
        or disclosure.get("preauthorization_answer_bodies_seen") != 0
        or disclosure.get("public_source_not_private") is not True
        or disclosure.get("model_training_exclusion_proven") is not False
        or disclosure.get("historical_novelty_claim_allowed") is not False
    ):
        raise MathOverflowTask2Error("MathOverflow Task 2 blindness contract changed")
    paths = config["implementation_paths"]
    dependencies = config["dependency_paths"]
    if (
        not isinstance(paths, list)
        or not isinstance(dependencies, list)
        or len(paths) != len(set(paths))
        or len(dependencies) != len(set(dependencies))
        or not paths
        or not dependencies
    ):
        raise MathOverflowTask2Error("MathOverflow Task 2 path binding changed")
    return config


def effective_generation_config(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    base = broken.load_config(root, Path(str(config["base_generation_config"])))
    effective = copy.deepcopy(dict(base))
    effective["task_id"] = config["task_id"]
    effective["selection"]["seed"] = config["selection"]["seed"]
    effective["adjudication"]["allowed_verifier_kinds"] = list(
        config["adjudication"]["allowed_verifier_kinds"]
    )
    return effective


def build_authorization(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    now = datetime.now(UTC)
    paths = list(config["implementation_paths"]) + list(config["dependency_paths"])
    file_hashes = {path: _sha256_file(root / path) for path in sorted(paths)}
    body = {
        "schema_version": AUTHORIZATION_SCHEMA,
        "task_id": config["task_id"],
        "authorization_timestamp": now.isoformat(),
        "authorization_epoch": int(now.timestamp()),
        "frozen_git_commit": _git_commit(root),
        "config_sha256": canonical_sha256(config),
        "file_sha256": file_hashes,
        "selector_commitment": canonical_sha256(config["selection"]),
        "pool_contract_sha256": canonical_sha256(config["source"]),
        "preauthorization_disclosure_sha256": canonical_sha256(
            config["blindness_disclosure"]
        ),
        "question_titles_read": 0,
        "question_bodies_read": 0,
        "answer_bodies_read": 0,
        "status": "AUTHORIZED_BODY_AND_ANSWER_BLIND",
    }
    return _sealed(body)


def validate_authorization(
    authorization: Mapping[str, Any], root: Path, config: Mapping[str, Any]
) -> None:
    _validate_seal(authorization, AUTHORIZATION_SCHEMA, "MathOverflow authorization")
    paths = list(config["implementation_paths"]) + list(config["dependency_paths"])
    expected_hashes = {path: _sha256_file(root / path) for path in sorted(paths)}
    if (
        authorization.get("task_id") != config["task_id"]
        or authorization.get("config_sha256") != canonical_sha256(config)
        or authorization.get("file_sha256") != expected_hashes
        or authorization.get("selector_commitment") != canonical_sha256(config["selection"])
        or authorization.get("pool_contract_sha256") != canonical_sha256(config["source"])
        or authorization.get("preauthorization_disclosure_sha256")
        != canonical_sha256(config["blindness_disclosure"])
        or authorization.get("question_titles_read") != 0
        or authorization.get("question_bodies_read") != 0
        or authorization.get("answer_bodies_read") != 0
        or authorization.get("status") != "AUTHORIZED_BODY_AND_ANSWER_BLIND"
    ):
        raise MathOverflowTask2Error("MathOverflow authorization binding changed")


def _metadata_url(config: Mapping[str, Any]) -> str:
    source = config["source"]
    query = urllib.parse.urlencode(
        {
            "site": source["site"],
            "tagged": source["tagged"],
            "pagesize": 100,
            "order": "desc",
            "sort": "creation",
            "filter": source["metadata_filter"],
        }
    )
    return f"{source['api_base']}/questions?{query}"


def check_source(
    authorization: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    fetch_json: JsonFetcher = _fetch_json,
) -> dict[str, Any]:
    response = fetch_json(_metadata_url(config))
    items = response.get("items", [])
    if not isinstance(items, list):
        raise MathOverflowTask2Error("MathOverflow metadata items are invalid")
    source = config["source"]
    cutoff = authorization["authorization_epoch"]
    eligible: list[dict[str, Any]] = []
    allowed_domains = set(source["eligible_domain_tags_any"])
    for raw in items:
        if not isinstance(raw, Mapping):
            continue
        tags = raw.get("tags", [])
        if (
            isinstance(raw.get("question_id"), int)
            and isinstance(raw.get("creation_date"), int)
            and source["minimum_creation_epoch"] <= raw["creation_date"] <= cutoff
            and raw.get("is_answered") is True
            and isinstance(raw.get("accepted_answer_id"), int)
            and isinstance(raw.get("answer_count"), int)
            and raw["answer_count"] >= 1
            and "closed_date" not in raw
            and isinstance(tags, list)
            and source["required_tag"] in tags
            and bool(allowed_domains.intersection(tags))
        ):
            eligible.append(
                {
                    "accepted_answer_id": raw["accepted_answer_id"],
                    "answer_count": raw["answer_count"],
                    "creation_date": raw["creation_date"],
                    "is_answered": True,
                    "last_activity_date": raw.get("last_activity_date"),
                    "question_id": raw["question_id"],
                    "tags": sorted(tags),
                }
            )
    seed = config["selection"]["seed"]
    eligible.sort(
        key=lambda row: hashlib.sha256(
            f"{seed}\0{row['question_id']}".encode()
        ).hexdigest()
    )
    selected = eligible[0] if eligible else None
    body = {
        "schema_version": SOURCE_SCHEMA,
        "task_id": config["task_id"],
        "authorization_content_sha256": authorization["content_sha256"],
        "metadata_query": {
            "filter": source["metadata_filter"],
            "question_titles_read": 0,
            "question_bodies_read": 0,
            "answer_bodies_read": 0,
            "rows_returned": len(items),
        },
        "eligible_metadata": eligible,
        "selected_question": selected,
        "status": "READY_SELECTED_BODY_UNREAD" if selected else "BLOCKED_NO_ELIGIBLE_QUESTION",
    }
    return _sealed(body)


def validate_source(source_check: Mapping[str, Any], authorization: Mapping[str, Any]) -> None:
    _validate_seal(source_check, SOURCE_SCHEMA, "MathOverflow source check")
    query = source_check.get("metadata_query", {})
    selected = source_check.get("selected_question")
    if (
        source_check.get("authorization_content_sha256") != authorization["content_sha256"]
        or query.get("question_titles_read") != 0
        or query.get("question_bodies_read") != 0
        or query.get("answer_bodies_read") != 0
        or (selected is None) is not (source_check.get("status") == "BLOCKED_NO_ELIGIBLE_QUESTION")
    ):
        raise MathOverflowTask2Error("MathOverflow source blindness or status changed")


def _question_url(config: Mapping[str, Any], question_id: int) -> str:
    source = config["source"]
    query = urllib.parse.urlencode(
        {"site": source["site"], "filter": source["question_filter"]}
    )
    return f"{source['api_base']}/questions/{question_id}?{query}"


def stage_question(
    source_check: Mapping[str, Any],
    authorization: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    fetch_json: JsonFetcher = _fetch_json,
) -> dict[str, Any]:
    validate_source(source_check, authorization)
    selected = source_check.get("selected_question")
    if not isinstance(selected, Mapping):
        raise MathOverflowTask2Error("no eligible MathOverflow question is selected")
    question_id = selected["question_id"]
    response = fetch_json(_question_url(config, question_id))
    items = response.get("items", [])
    if not isinstance(items, list) or len(items) != 1 or not isinstance(items[0], Mapping):
        raise MathOverflowTask2Error("selected MathOverflow question response is invalid")
    row = items[0]
    if (
        row.get("question_id") != question_id
        or row.get("accepted_answer_id") != selected["accepted_answer_id"]
        or not isinstance(row.get("title"), str)
        or not isinstance(row.get("body"), str)
    ):
        raise MathOverflowTask2Error("selected MathOverflow question changed after selection")
    statement = f"{html.unescape(row['title']).strip()}\n\n{_plain_text(row['body'])}"
    statement_sha = hashlib.sha256(statement.encode("utf-8")).hexdigest()
    return _sealed(
        {
            "schema_version": STAGED_SCHEMA,
            "task_id": config["task_id"],
            "authorization_content_sha256": authorization["content_sha256"],
            "source_check_content_sha256": source_check["content_sha256"],
            "release_binding": {
                "dataset_id": f"mathoverflow/questions/{question_id}",
                "revision": str(row.get("last_activity_date", selected["last_activity_date"])),
            },
            "selection": {
                "accepted_answer_id": selected["accepted_answer_id"],
                "problem_id": f"mathoverflow:{question_id}",
                "question_id": question_id,
                "statement": statement,
                "statement_sha256": statement_sha,
            },
            "blindness": {
                "accepted_answer_body_read": 0,
                "question_body_read_after_authorization": True,
                "reference_material_opened": False,
            },
            "claims": {
                "historical_novelty_established": False,
                "model_training_exclusion_established": False,
                "public_real_problem": True,
            },
            "status": "STAGED_ANSWER_BLIND_READY_FOR_GENERATION",
        }
    )


def run_generation(
    staged: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    root: Path,
    unblinding_key: bytes,
    credential_file: Path | None = None,
    transport: Transport = urllib_transport,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    effective = effective_generation_config(root, config)
    specs = broken.build_arm_specs(staged, effective)
    clients = {
        arm: ClaudeCreativityClient(broken._client_config(effective), transport)
        for arm in effective["trial"]["arms"]
    }
    public_payload = {
        "benchmark_kind": "fresh_mathoverflow_counterexample_question",
        "problem_id": staged["selection"]["problem_id"],
        "source_dataset": staged["release_binding"]["dataset_id"],
        "statement": staged["selection"]["statement"],
    }
    benchmark_id = "task2.mo." + staged["selection"]["statement_sha256"][:20]
    ordered_specs = sorted(
        [spec for rows in specs.values() for spec in rows],
        key=lambda spec: (
            spec["slot_index"],
            hashlib.sha256(
                f"{staged['content_sha256']}:{spec['slot_index']}:{spec['arm']}".encode()
            ).hexdigest(),
        ),
    )
    environment = None
    if credential_file is not None:
        environment = dict(os.environ)
        environment[effective["trial"]["claude"]["credential_env_var"]] = ""
        environment["INVARIANT_ENV_FILE"] = str(credential_file.resolve())
    candidates = []
    try:
        with activated_credential(
            project_root=root.resolve(),
            env_var=effective["trial"]["claude"]["credential_env_var"],
            environment=environment,
        ) as activation:
            for spec in ordered_specs:
                result = clients[spec["arm"]].run(
                    ClaudeRole(spec["role"]),
                    benchmark_id,
                    public_payload,
                    instruction_override=spec["instruction"],
                    system_override=spec["system"],
                    hypothesis_slots=1,
                )
                if (
                    result.status is not ClaudeCallStatus.COMPLETED
                    or result.output is None
                    or len(result.output.hypotheses) != 1
                ):
                    raise MathOverflowTask2Error("MathOverflow candidate slot did not complete")
                candidates.append(
                    {
                        "arm": spec["arm"],
                        "call": result.to_dict(),
                        "falsifier_family": spec["falsifier_family"],
                        "hypothesis": result.output.hypotheses[0].to_dict(),
                        "role": spec["role"],
                        "slot_index": spec["slot_index"],
                    }
                )
            activation_evidence = activation.to_evidence()
    except CredentialActivationError as error:
        raise MathOverflowTask2Error(str(error)) from error
    return broken.compile_generation(
        staged,
        effective,
        candidates,
        unblinding_key=unblinding_key,
        credential_activation=activation_evidence,
    )


def _answer_url(config: Mapping[str, Any], answer_id: int) -> str:
    source = config["source"]
    query = urllib.parse.urlencode(
        {"site": source["site"], "filter": source["answer_filter"]}
    )
    return f"{source['api_base']}/answers/{answer_id}?{query}"


def open_reference(
    public: Mapping[str, Any],
    staged: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    fetch_json: JsonFetcher = _fetch_json,
) -> dict[str, Any]:
    if (
        public.get("staged_problem_content_sha256") != staged["content_sha256"]
        or public.get("blindness", {}).get("submissions_frozen") is not True
        or len(public.get("submissions", [])) != 36
    ):
        raise MathOverflowTask2Error("accepted answer cannot open before 36 submissions freeze")
    answer_id = staged["selection"]["accepted_answer_id"]
    response = fetch_json(_answer_url(config, answer_id))
    items = response.get("items", [])
    if not isinstance(items, list) or len(items) != 1 or not isinstance(items[0], Mapping):
        raise MathOverflowTask2Error("accepted answer response is invalid")
    answer = items[0]
    if (
        answer.get("answer_id") != answer_id
        or answer.get("question_id") != staged["selection"]["question_id"]
        or answer.get("is_accepted") is not True
        or not isinstance(answer.get("body"), str)
    ):
        raise MathOverflowTask2Error("accepted answer binding changed")
    return _sealed(
        {
            "schema_version": REFERENCE_SCHEMA,
            "task_id": config["task_id"],
            "public_submissions_content_sha256": public["content_sha256"],
            "staged_problem_content_sha256": staged["content_sha256"],
            "answer_id": answer_id,
            "question_id": staged["selection"]["question_id"],
            "accepted_answer": _plain_text(answer["body"]),
            "source_link": answer.get("link"),
            "reference_opened_after_submissions_frozen": True,
            "claims": {
                "candidate_correctness_automatically_established": False,
                "historical_novelty_established": False,
            },
            "status": "REFERENCE_OPEN_FOR_INDEPENDENT_SCORING",
        }
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("authorize", "validate-authorization", "check-source", "stage"):
        command = commands.add_parser(name)
        command.add_argument("--root", type=Path, default=Path.cwd())
        command.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        if name != "authorize":
            command.add_argument("--authorization", type=Path, required=True)
        if name == "check-source":
            command.add_argument("--output", type=Path, required=True)
        if name == "stage":
            command.add_argument("--source-check", type=Path, required=True)
            command.add_argument("--output", type=Path, required=True)
        if name == "authorize":
            command.add_argument("--output", type=Path, required=True)
    generate = commands.add_parser("run-generation")
    generate.add_argument("--root", type=Path, default=Path.cwd())
    generate.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    generate.add_argument("--staged", type=Path, required=True)
    generate.add_argument("--credential-file", type=Path)
    generate.add_argument("--public-output", type=Path, required=True)
    generate.add_argument("--receipt-output", type=Path, required=True)
    generate.add_argument("--coordinator-output", type=Path, required=True)
    reference = commands.add_parser("open-reference")
    reference.add_argument("--root", type=Path, default=Path.cwd())
    reference.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    reference.add_argument("--staged", type=Path, required=True)
    reference.add_argument("--public-submissions", type=Path, required=True)
    reference.add_argument("--output", type=Path, required=True)

    args = parser.parse_args(argv)
    root = args.root.resolve()
    config = load_config(root, args.config)
    if args.command == "authorize":
        result = build_authorization(root, config)
        _write_json(args.output, result)
    elif args.command == "validate-authorization":
        authorization = _read_json(args.authorization)
        validate_authorization(authorization, root, config)
        result = authorization
    elif args.command == "check-source":
        authorization = _read_json(args.authorization)
        validate_authorization(authorization, root, config)
        result = check_source(authorization, config)
        _write_json(args.output, result)
    elif args.command == "stage":
        authorization = _read_json(args.authorization)
        validate_authorization(authorization, root, config)
        result = stage_question(_read_json(args.source_check), authorization, config)
        _write_json(args.output, result)
    elif args.command == "run-generation":
        staged = _read_json(args.staged)
        public, receipt, coordinator = run_generation(
            staged,
            config,
            root=root,
            unblinding_key=os.urandom(32),
            credential_file=args.credential_file,
        )
        _write_json(args.public_output, public)
        _write_json(args.receipt_output, receipt)
        _write_json(args.coordinator_output, coordinator)
        result = receipt
    else:
        result = open_reference(
            _read_json(args.public_submissions), _read_json(args.staged), config
        )
        _write_json(args.output, result)
    print(json.dumps({"content_sha256": result["content_sha256"], "status": result["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
