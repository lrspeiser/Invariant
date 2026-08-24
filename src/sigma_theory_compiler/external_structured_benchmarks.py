"""Build blind tensor, variational, and transform tasks from upstream executable tests.

The public generation packet contains normalized setup code, a query, and a commitment.  It omits
the upstream repository identity, test name, expected expression, and exact source URI.  A private
target packet opens those commitments and a coordinator receipt binds the exact HTTPS response
bytes.  The current upstream responses are not detached-signed, so the pack is useful for future
creativity rotations but is not eligible for a level-5 count.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import textwrap
import urllib.parse
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .claim_specific_prior_art import Transport, urllib_transport
from .sigma_core import canonical_sha256

CONFIG_PATH = "configs/external_structured_benchmark_pack.json"
SOURCE_PATH = "src/sigma_theory_compiler/external_structured_benchmarks.py"
TEST_PATH = "tests/test_external_structured_benchmarks.py"
GENERATION_PATH = "runs/math/external-structured-benchmarks/2026-08-23-002-generation.json"
TARGET_PATH = "runs/math/external-structured-benchmarks/2026-08-23-002-targets.json"
RECEIPT_PATH = "runs/math/external-structured-benchmarks/2026-08-23-002-receipt.json"
CONFIG_SCHEMA = "invariant-external-structured-benchmark-config-1.0"
GENERATION_SCHEMA = "invariant-external-structured-generation-packet-1.0"
TARGET_SCHEMA = "invariant-external-structured-target-packet-1.0"
RECEIPT_SCHEMA = "invariant-external-structured-coordinator-receipt-1.0"
_HEX = frozenset("0123456789abcdef")
_FAMILIES = {"tensor_identity", "transform_relation", "variational_functional"}
_SOURCE_IDENTITIES = {
    "sympy-euler-tests": (
        "variational_functional",
        "/sympy/sympy/master/sympy/calculus/tests/test_euler.py",
        "derive_euler_lagrange_equations",
    ),
    "sympy-tensor-tests": (
        "tensor_identity",
        "/sympy/sympy/master/sympy/tensor/tests/test_tensor.py",
        "canonicalize_tensor_expression",
    ),
    "sympy-finite-difference-tests": (
        "transform_relation",
        "/sympy/sympy/master/sympy/calculus/tests/test_finite_diff.py",
        "derive_finite_difference_transform",
    ),
}


class StructuredBenchmarkError(ValueError):
    """An upstream source, AST selector, blind packet, or claim boundary failed closed."""


def _strict(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise StructuredBenchmarkError(f"{label} keys changed")


def _sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise StructuredBenchmarkError(f"{label} is not a lowercase SHA-256 digest")
    return value


def _normalized_file_sha256(path: Path) -> str:
    raw = path.read_bytes()
    try:
        raw = raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n").encode()
    except UnicodeDecodeError:
        pass
    return hashlib.sha256(raw).hexdigest()


def _utc(value: str | None) -> str:
    value = value or datetime.now(UTC).isoformat().replace("+00:00", "Z")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise StructuredBenchmarkError("retrieval time is not ISO-8601") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise StructuredBenchmarkError("retrieval time lacks a UTC offset")
    return parsed.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _bound_path(root: Path, value: str | Path, label: str) -> tuple[str, Path]:
    root = root.resolve()
    candidate = (root / value).resolve()
    try:
        relative = candidate.relative_to(root).as_posix()
    except ValueError as error:
        raise StructuredBenchmarkError(f"{label} escaped the repository root") from error
    return relative, candidate


def _sealed(value: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(value)
    body["content_sha256"] = canonical_sha256(body)
    return body


def load_config(root: Path, config_path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    _, resolved = _bound_path(root, config_path, "structured benchmark config")
    value = json.loads(resolved.read_text(encoding="utf-8"))
    _strict(
        value,
        {
            "generator_principal_id",
            "pack_id",
            "rotation_epoch",
            "schema_version",
            "source_policy",
            "sources",
            "task_policy",
        },
        "structured benchmark config",
    )
    if (
        value["schema_version"] != CONFIG_SCHEMA
        or value["pack_id"] != "structured-symbolic-live-rotation"
        or value["generator_principal_id"] != "invariant.discovery-engine"
        or re.fullmatch(r"20[0-9]{2}-[0-9]{2}-[0-9]{2}-[0-9]{3}", value["rotation_epoch"]) is None
    ):
        raise StructuredBenchmarkError("structured benchmark identity changed")
    task_policy = value["task_policy"]
    _strict(
        task_policy,
        {"minimum_tasks", "minimum_tasks_per_family"},
        "structured task policy",
    )
    if task_policy != {"minimum_tasks": 12, "minimum_tasks_per_family": 4}:
        raise StructuredBenchmarkError("structured task coverage weakened")
    policy = value["source_policy"]
    _strict(
        policy,
        {
            "allowed_host",
            "allowed_repository",
            "cryptographic_signature_required_for_level5",
            "maximum_response_bytes",
            "request_timeout_seconds",
            "user_agent",
        },
        "structured source policy",
    )
    if (
        policy["allowed_host"] != "raw.githubusercontent.com"
        or policy["allowed_repository"] != "sympy/sympy"
        or policy["cryptographic_signature_required_for_level5"] is not True
        or not 100_000 <= policy["maximum_response_bytes"] <= 1_000_000
        or not 1 <= policy["request_timeout_seconds"] <= 60
        or not isinstance(policy["user_agent"], str)
        or "Invariant" not in policy["user_agent"]
    ):
        raise StructuredBenchmarkError("structured source policy weakened")
    sources = value["sources"]
    if not isinstance(sources, list) or len(sources) != 3:
        raise StructuredBenchmarkError("structured source registry changed")
    seen_sources: set[str] = set()
    seen_functions: set[tuple[str, str]] = set()
    family_counts: Counter[str] = Counter()
    for source in sources:
        _strict(
            source,
            {
                "external_principal_id",
                "representation_family",
                "selectors",
                "source_id",
                "source_uri",
            },
            "structured source",
        )
        source_id = source["source_id"]
        expected = _SOURCE_IDENTITIES.get(source_id)
        parsed = urllib.parse.urlparse(source["source_uri"])
        if (
            expected is None
            or source_id in seen_sources
            or source["external_principal_id"] != "external.sympy-project"
            or source["external_principal_id"] == value["generator_principal_id"]
            or source["representation_family"] != expected[0]
            or parsed.scheme != "https"
            or parsed.hostname != policy["allowed_host"]
            or parsed.path != expected[1]
            or parsed.query
            or parsed.fragment
        ):
            raise StructuredBenchmarkError("structured external source identity changed")
        selectors = source["selectors"]
        if not isinstance(selectors, list) or len(selectors) != 4:
            raise StructuredBenchmarkError("structured selector coverage changed")
        for selector in selectors:
            _strict(
                selector,
                {"assertion_ordinal", "task_kind", "test_function"},
                "structured selector",
            )
            identity = (source_id, selector["test_function"])
            if (
                not isinstance(selector["test_function"], str)
                or re.fullmatch(r"test_[a-z0-9_]+", selector["test_function"]) is None
                or selector["assertion_ordinal"] != 0
                or selector["task_kind"] != expected[2]
                or identity in seen_functions
            ):
                raise StructuredBenchmarkError("structured AST selector changed")
            seen_functions.add(identity)
            family_counts[source["representation_family"]] += 1
        seen_sources.add(source_id)
    if seen_sources != set(_SOURCE_IDENTITIES) or family_counts != Counter(
        {family: 4 for family in _FAMILIES}
    ):
        raise StructuredBenchmarkError("structured benchmark balance changed")
    return value


def _decode_source(body: bytes) -> str:
    if not body or body.startswith(b"\xef\xbb\xbf"):
        raise StructuredBenchmarkError("upstream Python source is empty or BOM-prefixed")
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError as error:
        raise StructuredBenchmarkError("upstream Python source is not UTF-8") from error
    try:
        ast.parse(text)
    except SyntaxError as error:
        raise StructuredBenchmarkError("upstream Python source does not parse") from error
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _complete_response(status: int, headers: Mapping[str, str], body: bytes) -> bool:
    if status == 200:
        return True
    content_range = headers.get("content-range", "")
    match = re.fullmatch(r"bytes 0-([0-9]+)/([0-9]+)", content_range)
    if status != 206 or match is None:
        return False
    final_index, total_bytes = (int(item) for item in match.groups())
    return final_index + 1 == total_bytes == len(body)


def _extract_assertion(
    source_text: str, test_function: str, assertion_ordinal: int
) -> dict[str, str]:
    tree = ast.parse(source_text)
    functions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == test_function
    ]
    if len(functions) != 1:
        raise StructuredBenchmarkError("upstream test function is missing or duplicated")
    function = functions[0]
    assertions = [node for node in function.body if isinstance(node, ast.Assert)]
    if assertion_ordinal < 0 or assertion_ordinal >= len(assertions):
        raise StructuredBenchmarkError("upstream assertion ordinal is unavailable")
    selected = assertions[assertion_ordinal]
    if (
        not isinstance(selected.test, ast.Compare)
        or len(selected.test.ops) != 1
        or not isinstance(selected.test.ops[0], ast.Eq)
        or len(selected.test.comparators) != 1
        or selected.msg is not None
    ):
        raise StructuredBenchmarkError("upstream assertion is not one exact equality")
    selected_index = function.body.index(selected)
    context_nodes = function.body[:selected_index]
    if not context_nodes or any(isinstance(node, ast.Assert) for node in context_nodes):
        raise StructuredBenchmarkError("selected task context contains an earlier target")
    context_segments = [ast.get_source_segment(source_text, node) for node in context_nodes]
    query_segment = ast.get_source_segment(source_text, selected.test.left)
    target_segment = ast.get_source_segment(source_text, selected.test.comparators[0])
    if any(segment is None for segment in context_segments):
        raise StructuredBenchmarkError("upstream setup source segment is unavailable")
    context_source = "\n".join(
        textwrap.dedent(segment).strip() for segment in context_segments if segment is not None
    )
    query_expression = textwrap.dedent(query_segment or "").strip()
    target_expression = textwrap.dedent(target_segment or "").strip()
    function_source = ast.get_source_segment(source_text, function)
    if (
        not context_source
        or not query_expression
        or not target_expression
        or function_source is None
    ):
        raise StructuredBenchmarkError("upstream task extraction is incomplete")
    return {
        "context_source": context_source,
        "query_expression": query_expression,
        "target_expression": target_expression,
        "upstream_function_source": function_source,
    }


def _instruction(family: str) -> str:
    if family == "tensor_identity":
        return (
            "Derive the exact canonical tensor expression requested by query_expression, "
            "respecting variance, contractions, commutation, and declared index symmetries."
        )
    if family == "variational_functional":
        return (
            "Derive the exact Euler-Lagrange equation list requested by query_expression from "
            "the declared functional; preserve derivative order and coupled fields."
        )
    if family == "transform_relation":
        return (
            "Derive the exact finite-difference transform requested by query_expression from "
            "the declared samples or symbolic expression; preserve shifts and exact weights."
        )
    raise StructuredBenchmarkError("unknown structured representation family")


def _source_binding(root: Path, path: str, label: str) -> dict[str, str]:
    relative, resolved = _bound_path(root, path, label)
    return {"path": relative, "sha256": _normalized_file_sha256(resolved)}


def build_pack(
    root: Path,
    *,
    transport: Transport = urllib_transport,
    retrieved_utc: str | None = None,
    config_path: str | Path = CONFIG_PATH,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    root = root.resolve()
    relative_config, _ = _bound_path(root, config_path, "structured benchmark config")
    config = load_config(root, relative_config)
    retrieved = _utc(retrieved_utc)
    policy = config["source_policy"]
    headers = {
        "Accept": "text/plain",
        "Range": f"bytes=0-{policy['maximum_response_bytes'] - 1}",
        "User-Agent": policy["user_agent"],
    }
    tasks: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    for source in config["sources"]:
        response = transport(
            source["source_uri"],
            headers,
            policy["request_timeout_seconds"],
            policy["maximum_response_bytes"],
        )
        if not _complete_response(response.status, response.headers, response.body):
            raise StructuredBenchmarkError(
                f"external structured source unavailable: {source['source_id']}"
            )
        if len(response.body) > policy["maximum_response_bytes"]:
            raise StructuredBenchmarkError("external structured response exceeded byte ceiling")
        source_text = _decode_source(response.body)
        response_sha = hashlib.sha256(response.body).hexdigest()
        evidence.append(
            {
                "content_type": response.headers.get("content-type", "")[:200],
                "external_principal_id": source["external_principal_id"],
                "http_status": response.status,
                "retrieved_utc": retrieved,
                "source_id": source["source_id"],
                "source_content_range": response.headers.get("content-range"),
                "source_response_bytes": len(response.body),
                "source_response_sha256": response_sha,
                "source_uri": source["source_uri"],
                "transport": "https",
            }
        )
        for selector in source["selectors"]:
            extracted = _extract_assertion(
                source_text,
                selector["test_function"],
                selector["assertion_ordinal"],
            )
            target_body = {
                "assertion_ordinal": selector["assertion_ordinal"],
                "context_sha256": hashlib.sha256(extracted["context_source"].encode()).hexdigest(),
                "external_principal_id": source["external_principal_id"],
                "query_expression": extracted["query_expression"],
                "source_id": source["source_id"],
                "source_response_sha256": response_sha,
                "source_uri": source["source_uri"],
                "target_expression": extracted["target_expression"],
                "test_function": selector["test_function"],
                "upstream_function_source": extracted["upstream_function_source"],
            }
            commitment = canonical_sha256(target_body)
            task_id = (
                "blind."
                + canonical_sha256({"commitment": commitment, "epoch": config["rotation_epoch"]})[
                    :24
                ]
            )
            tasks.append(
                {
                    "context_source": extracted["context_source"],
                    "external_source_disclosed_to_generator": False,
                    "instruction": _instruction(source["representation_family"]),
                    "query_expression": extracted["query_expression"],
                    "representation_family": source["representation_family"],
                    "rotation_epoch": config["rotation_epoch"],
                    "target_commitment": commitment,
                    "task_id": task_id,
                    "task_kind": selector["task_kind"],
                }
            )
            targets.append({"task_id": task_id, **target_body})
    generation = _sealed(
        {
            "claims": {
                "external_source_identity_visible_to_generator": False,
                "target_expression_visible_to_generator": False,
            },
            "pack_id": config["pack_id"],
            "rotation_epoch": config["rotation_epoch"],
            "schema_version": GENERATION_SCHEMA,
            "tasks": tasks,
        }
    )
    target_packet = _sealed(
        {
            "pack_id": config["pack_id"],
            "rotation_epoch": config["rotation_epoch"],
            "schema_version": TARGET_SCHEMA,
            "targets": targets,
        }
    )
    receipt = _sealed(
        {
            "blind_protocol": {
                "generation_packet_excludes_source_identities": True,
                "generation_packet_excludes_target_expressions": True,
                "target_commitments_verified": True,
            },
            "claims": {
                "https_origin_is_cryptographic_source_signature": False,
                "pack_is_level5_success": False,
                "targets_are_literature_novel": False,
                "upstream_test_assertions_are_independent_proofs": False,
            },
            "coverage": {
                "external_principals": len({item["external_principal_id"] for item in evidence}),
                "representation_counts": dict(
                    sorted(Counter(task["representation_family"] for task in tasks).items())
                ),
                "tasks": len(tasks),
                "unique_external_response_hashes": len(
                    {item["source_response_sha256"] for item in evidence}
                ),
            },
            "pack_id": config["pack_id"],
            "release_gate": {
                "level5_eligible": False,
                "status": "CREATIVITY_BENCHMARK_READY_LEVEL5_BLOCKED_UNSIGNED_SOURCE",
            },
            "retrieved_utc": retrieved,
            "rotation_epoch": config["rotation_epoch"],
            "schema_version": RECEIPT_SCHEMA,
            "source_bindings": {
                "config": _source_binding(root, relative_config, "structured benchmark config"),
                "generation_packet_content_sha256": generation["content_sha256"],
                "module": _source_binding(root, SOURCE_PATH, "structured benchmark module"),
                "target_packet_content_sha256": target_packet["content_sha256"],
                "tests": _source_binding(root, TEST_PATH, "structured benchmark tests"),
            },
            "source_evidence": evidence,
            "source_signature": {
                "cryptographic_signature_verified": False,
                "external_https_origin_hash_bound": True,
                "status": "PENDING_DISTINCT_PRINCIPAL_SIGNATURE",
            },
        }
    )
    validate_pack(generation, target_packet, receipt, root)
    return generation, target_packet, receipt


def _validate_seal(value: Mapping[str, Any], schema: str, label: str) -> None:
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("schema_version") != schema or value.get("content_sha256") != canonical_sha256(
        body
    ):
        raise StructuredBenchmarkError(f"{label} identity or content seal changed")


def validate_pack(
    generation: Mapping[str, Any],
    targets: Mapping[str, Any],
    receipt: Mapping[str, Any],
    root: Path | None = None,
) -> None:
    for value, schema, label in (
        (generation, GENERATION_SCHEMA, "structured generation packet"),
        (targets, TARGET_SCHEMA, "structured target packet"),
        (receipt, RECEIPT_SCHEMA, "structured coordinator receipt"),
    ):
        _validate_seal(value, schema, label)
    if (
        generation.get("pack_id") != "structured-symbolic-live-rotation"
        or targets.get("pack_id") != generation["pack_id"]
        or receipt.get("pack_id") != generation["pack_id"]
        or targets.get("rotation_epoch") != generation.get("rotation_epoch")
        or receipt.get("rotation_epoch") != generation.get("rotation_epoch")
    ):
        raise StructuredBenchmarkError("structured packet identity changed")
    tasks = generation.get("tasks", [])
    target_rows = targets.get("targets", [])
    if len(tasks) != 12 or len(target_rows) != 12:
        raise StructuredBenchmarkError("structured task count changed")
    if len({item.get("task_id") for item in tasks}) != len(tasks):
        raise StructuredBenchmarkError("structured generation tasks are not unique")
    target_by_id = {item.get("task_id"): item for item in target_rows}
    if len(target_by_id) != len(target_rows) or None in target_by_id:
        raise StructuredBenchmarkError("structured target tasks are not unique")
    expected_task_keys = {
        "context_source",
        "external_source_disclosed_to_generator",
        "instruction",
        "query_expression",
        "representation_family",
        "rotation_epoch",
        "target_commitment",
        "task_id",
        "task_kind",
    }
    expected_target_keys = {
        "assertion_ordinal",
        "context_sha256",
        "external_principal_id",
        "query_expression",
        "source_id",
        "source_response_sha256",
        "source_uri",
        "target_expression",
        "task_id",
        "test_function",
        "upstream_function_source",
    }
    for task in tasks:
        _strict(task, expected_task_keys, "structured generation task")
        if (
            task["external_source_disclosed_to_generator"] is not False
            or task["representation_family"] not in _FAMILIES
            or task["task_id"] not in target_by_id
            or "source_id" in task
            or "source_uri" in task
            or "target_expression" in task
            or "test_function" in task
        ):
            raise StructuredBenchmarkError("structured generation packet leaked target identity")
        target = target_by_id[task["task_id"]]
        _strict(target, expected_target_keys, "structured target task")
        target_body = {key: value for key, value in target.items() if key != "task_id"}
        if task["target_commitment"] != canonical_sha256(target_body):
            raise StructuredBenchmarkError("structured target commitment changed")
        expected_id = (
            "blind."
            + canonical_sha256(
                {
                    "commitment": task["target_commitment"],
                    "epoch": generation["rotation_epoch"],
                }
            )[:24]
        )
        if task["task_id"] != expected_id:
            raise StructuredBenchmarkError("structured opaque task identity changed")
        extracted = _extract_assertion(
            target["upstream_function_source"],
            target["test_function"],
            target["assertion_ordinal"],
        )
        if (
            target["context_sha256"]
            != hashlib.sha256(extracted["context_source"].encode()).hexdigest()
            or target["query_expression"] != extracted["query_expression"]
            or target["target_expression"] != extracted["target_expression"]
            or task["context_source"] != extracted["context_source"]
            or task["query_expression"] != extracted["query_expression"]
            or task["instruction"] != _instruction(task["representation_family"])
        ):
            raise StructuredBenchmarkError("structured task no longer opens upstream assertion")
        _sha(target["source_response_sha256"], "structured source response hash")
    evidence = receipt.get("source_evidence", [])
    evidence_by_id = {item.get("source_id"): item for item in evidence}
    if len(evidence) != 3 or len(evidence_by_id) != 3:
        raise StructuredBenchmarkError("structured source evidence is missing or duplicated")
    for target in target_rows:
        source = evidence_by_id.get(target["source_id"])
        if source is None or any(
            target[key] != source[key]
            for key in (
                "external_principal_id",
                "source_response_sha256",
                "source_uri",
            )
        ):
            raise StructuredBenchmarkError("structured target lost its external source binding")
    for source_id, source in evidence_by_id.items():
        expected = _SOURCE_IDENTITIES.get(source_id)
        parsed = urllib.parse.urlparse(source.get("source_uri", ""))
        if (
            expected is None
            or source.get("external_principal_id") != "external.sympy-project"
            or source.get("http_status") not in {200, 206}
            or source.get("transport") != "https"
            or not isinstance(source.get("source_response_bytes"), int)
            or source["source_response_bytes"] <= 0
            or parsed.scheme != "https"
            or parsed.hostname != "raw.githubusercontent.com"
            or parsed.path != expected[1]
            or not _complete_response(
                source["http_status"],
                {"content-range": source.get("source_content_range") or ""},
                b"0" * source["source_response_bytes"],
            )
        ):
            raise StructuredBenchmarkError("structured external source evidence changed")
        _sha(source.get("source_response_sha256"), "structured source response hash")
    coverage = receipt.get("coverage", {})
    bindings = receipt.get("source_bindings", {})
    if (
        generation.get("claims")
        != {
            "external_source_identity_visible_to_generator": False,
            "target_expression_visible_to_generator": False,
        }
        or receipt.get("blind_protocol")
        != {
            "generation_packet_excludes_source_identities": True,
            "generation_packet_excludes_target_expressions": True,
            "target_commitments_verified": True,
        }
        or coverage
        != {
            "external_principals": 1,
            "representation_counts": {family: 4 for family in sorted(_FAMILIES)},
            "tasks": 12,
            "unique_external_response_hashes": len(
                {item["source_response_sha256"] for item in evidence}
            ),
        }
        or bindings.get("generation_packet_content_sha256") != generation["content_sha256"]
        or bindings.get("target_packet_content_sha256") != targets["content_sha256"]
        or receipt.get("source_signature")
        != {
            "cryptographic_signature_verified": False,
            "external_https_origin_hash_bound": True,
            "status": "PENDING_DISTINCT_PRINCIPAL_SIGNATURE",
        }
        or receipt.get("release_gate")
        != {
            "level5_eligible": False,
            "status": "CREATIVITY_BENCHMARK_READY_LEVEL5_BLOCKED_UNSIGNED_SOURCE",
        }
        or set(receipt.get("claims", {}).values()) != {False}
        or set(receipt.get("claims", {}))
        != {
            "https_origin_is_cryptographic_source_signature",
            "pack_is_level5_success",
            "targets_are_literature_novel",
            "upstream_test_assertions_are_independent_proofs",
        }
    ):
        raise StructuredBenchmarkError("structured benchmark release boundary changed")
    if root is not None:
        for binding_name, expected_path in (
            ("config", CONFIG_PATH),
            ("module", SOURCE_PATH),
            ("tests", TEST_PATH),
        ):
            binding = bindings.get(binding_name, {})
            relative, resolved = _bound_path(
                root, binding.get("path", ""), f"structured {binding_name} binding"
            )
            if (
                relative != expected_path
                or binding.get("path") != relative
                or binding.get("sha256") != _normalized_file_sha256(resolved)
            ):
                raise StructuredBenchmarkError(f"structured {binding_name} source binding changed")


def reproduce_pack(
    root: Path,
    generation: Mapping[str, Any],
    targets: Mapping[str, Any],
    receipt: Mapping[str, Any],
    *,
    transport: Transport = urllib_transport,
    config_path: str | Path = CONFIG_PATH,
) -> None:
    validate_pack(generation, targets, receipt, root)
    rebuilt = build_pack(
        root,
        transport=transport,
        retrieved_utc=receipt["retrieved_utc"],
        config_path=config_path,
    )
    if rebuilt != (dict(generation), dict(targets), dict(receipt)):
        raise StructuredBenchmarkError(
            "live structured source refetch did not reproduce the sealed pack"
        )


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_stored_pack(
    root: Path, receipt_path: str | Path = RECEIPT_PATH
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    relative_receipt, resolved_receipt = _bound_path(
        root, receipt_path, "stored structured receipt"
    )
    if relative_receipt != RECEIPT_PATH:
        raise StructuredBenchmarkError("stored structured receipt identity changed")
    _, generation_path = _bound_path(root, GENERATION_PATH, "stored generation packet")
    _, target_path = _bound_path(root, TARGET_PATH, "stored target packet")
    generation = _load(generation_path)
    targets = _load(target_path)
    receipt = _load(resolved_receipt)
    validate_pack(generation, targets, receipt, root)
    return generation, targets, receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--root", type=Path, default=Path.cwd())
    build.add_argument("--config", type=Path, default=Path(CONFIG_PATH))
    build.add_argument("--generation-output", type=Path, required=True)
    build.add_argument("--target-output", type=Path, required=True)
    build.add_argument("--receipt-output", type=Path, required=True)
    for command in ("validate", "reproduce"):
        child = subparsers.add_parser(command)
        child.add_argument("--root", type=Path, default=Path.cwd())
        child.add_argument("--config", type=Path, default=Path(CONFIG_PATH))
        child.add_argument("--generation", type=Path, required=True)
        child.add_argument("--targets", type=Path, required=True)
        child.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "build":
        generation, targets, receipt = build_pack(args.root, config_path=args.config)
        for path, value in (
            (args.generation_output, generation),
            (args.target_output, targets),
            (args.receipt_output, receipt),
        ):
            _write(path, value)
    else:
        generation = _load(args.generation)
        targets = _load(args.targets)
        receipt = _load(args.receipt)
        if args.command == "validate":
            validate_pack(generation, targets, receipt, args.root)
        else:
            reproduce_pack(
                args.root,
                generation,
                targets,
                receipt,
                config_path=args.config,
            )
    print(
        json.dumps(
            {
                "content_sha256": receipt["content_sha256"],
                "level5_eligible": receipt["release_gate"]["level5_eligible"],
                "status": receipt["release_gate"]["status"],
                "tasks": receipt["coverage"]["tasks"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
