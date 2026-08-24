"""Promote a live Claude campaign into a credential-free, claim-neutral evidence receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .sigma_core import canonical_sha256

SCHEMA_VERSION = "invariant-external-creativity-live-api-evidence-1.0"
OUTPUT_PATH = "runs/math/external-creativity-validation/live-api-evidence.json"
CORE_PATH = "runs/math/core-creative-discovery/live-runtime.json"


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_evidence_from_receipt(
    source: Mapping[str, Any], *, source_file_sha256: str
) -> dict[str, Any]:
    """Remove prompts, outputs, and credentials from one completed live receipt.

    ``source_file_sha256`` binds the exact transient serialization inspected by the caller.  The
    raw live receipt never needs to be written to the repository.
    """

    if not isinstance(source_file_sha256, str) or len(source_file_sha256) != 64:
        raise ValueError("live campaign source file hash is invalid")
    source_body = {key: value for key, value in source.items() if key != "content_sha256"}
    if source.get("content_sha256") != canonical_sha256(source_body):
        raise ValueError("live campaign source receipt seal changed")
    claude = source.get("claude")
    if not isinstance(claude, Mapping):
        raise TypeError("live campaign has no Claude evidence")
    calls = claude.get("calls")
    if (
        claude.get("status") != "PASS"
        or claude.get("completed_calls") != claude.get("required_calls")
        or not isinstance(calls, list)
        or len(calls) != claude.get("completed_calls")
    ):
        raise ValueError("live Claude campaign did not complete its frozen call budget")

    safe_calls = []
    for call in calls:
        evidence = call.get("evidence")
        if not isinstance(evidence, Mapping) or evidence.get("credential_persisted") is not False:
            raise ValueError("live Claude call did not prove credential exclusion")
        safe_calls.append(
            {
                "api_response_id": evidence.get("api_response_id"),
                "benchmark_id": call.get("benchmark_id"),
                "capabilities_sha256": evidence.get("model_evidence", {}).get(
                    "capabilities_sha256"
                ),
                "credential_persisted": False,
                "creative_context_injected": evidence.get(
                    "creative_context_injected", False
                ),
                "creative_context_sha256": evidence.get("creative_context_sha256"),
                "model": evidence.get("model"),
                "output_sha256": evidence.get("output_sha256"),
                "prompt_sha256": evidence.get(
                    "provider_prompt_sha256", evidence.get("prompt_sha256")
                ),
                "raw_output_sha256": evidence.get(
                    "provider_raw_output_sha256", evidence.get("raw_output_sha256")
                ),
                "request_schema_sha256": evidence.get(
                    "provider_request_schema_sha256",
                    evidence.get("request_schema_sha256"),
                ),
                "role": call.get("role"),
                "usage": dict(evidence.get("usage", {})),
                "wire_contract_adapter_used": evidence.get(
                    "wire_contract_adapter_used", False
                ),
            }
        )

    benchmark_summaries = []
    for benchmark in source.get("benchmarks", []):
        ranked = benchmark.get("ranked_candidates", [])
        if not ranked:
            raise ValueError("live campaign benchmark has no ranked candidate")
        best = ranked[0]
        benchmark_summaries.append(
            {
                "benchmark_id": benchmark.get("benchmark_id"),
                "best_candidate": {
                    "candidate_id": best.get("candidate_id"),
                    "expression": best.get("expression"),
                    "holdout_loss": best.get("holdout_loss"),
                    "proposer": best.get("proposer"),
                },
                "capability_level": benchmark.get("capability_level"),
                "claude_executable_candidate_count": sum(
                    item.get("proposer") == "claude_api" for item in ranked
                ),
                "known_formula_rediscovered": benchmark.get("claims", {}).get(
                    "known_formula_rediscovered"
                ),
                "target_kind": benchmark.get("target_kind"),
                "unique_behaviors": benchmark.get("unique_behaviors"),
                "unique_proof_mechanisms": benchmark.get("unique_proof_mechanisms"),
            }
        )

    body = {
        "schema_version": SCHEMA_VERSION,
        "benchmarks": benchmark_summaries,
        "calls": safe_calls,
        "claims": {
            "credential_material_included": False,
            "live_claude_api_campaign_completed": True,
            "model_output_is_verifier_authority": False,
            "novel_formula_established": False,
            "open_problem_solved": False,
        },
        "source_receipt": {
            "content_sha256": source["content_sha256"],
            "file_sha256": source_file_sha256,
            "raw_prompts_or_outputs_copied": False,
        },
        "usage": dict(claude.get("budget", {})),
    }
    body["content_sha256"] = canonical_sha256(body)
    return body


def build_evidence(source_path: Path) -> dict[str, Any]:
    source_path = source_path.resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    return build_evidence_from_receipt(source, source_file_sha256=_file_sha256(source_path))


def build_evidence_from_core_receipt(
    source: Mapping[str, Any], *, root: Path
) -> dict[str, Any]:
    """Promote the already-sanitized live evidence embedded in a valid core receipt.

    This is a deterministic no-call export.  The lazy import avoids a module import cycle because
    the core validator uses :func:`validate_evidence` for the same embedded object.
    """

    from .core_creative_discovery import validate_receipt as validate_core_receipt

    validate_core_receipt(source, root.resolve())
    evidence = source.get("claude_runtime", {}).get("evidence")
    if not isinstance(evidence, Mapping):
        raise TypeError("core receipt has no sanitized live API evidence")
    promoted = dict(evidence)
    validate_evidence(promoted)
    return promoted


def validate_evidence(evidence: Mapping[str, Any]) -> None:
    body = {key: value for key, value in evidence.items() if key != "content_sha256"}
    if evidence.get("content_sha256") != canonical_sha256(body):
        raise ValueError("live API evidence seal changed")
    if evidence.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("live API evidence schema changed")
    claims = evidence.get("claims", {})
    if (
        claims.get("live_claude_api_campaign_completed") is not True
        or claims.get("credential_material_included") is not False
        or claims.get("model_output_is_verifier_authority") is not False
        or claims.get("novel_formula_established") is not False
        or claims.get("open_problem_solved") is not False
    ):
        raise ValueError("live API evidence claim boundary changed")
    calls = evidence.get("calls", [])
    usage = evidence.get("usage", {})
    if (
        not isinstance(calls, list)
        or not calls
        or usage.get("calls") != len(calls)
        or usage.get("total_tokens", 0) <= 0
        or len({call.get("api_response_id") for call in calls}) != len(calls)
        or {call.get("role") for call in calls} != {"proposer", "critic"}
        or any(call.get("credential_persisted") is not False for call in calls)
    ):
        raise ValueError("live API evidence credential boundary changed")
    for call in calls:
        if any(
            not isinstance(call.get(key), str) or len(call[key]) != 64
            for key in (
                "capabilities_sha256",
                "output_sha256",
                "prompt_sha256",
                "raw_output_sha256",
                "request_schema_sha256",
            )
        ):
            raise ValueError("live API call evidence binding changed")
        injected = call.get("creative_context_injected", False)
        context_sha = call.get("creative_context_sha256")
        if injected is True and (
            not isinstance(context_sha, str)
            or len(context_sha) != 64
            or any(character not in "0123456789abcdef" for character in context_sha)
        ):
            raise ValueError("live API creative context binding changed")
        if injected is not True and context_sha is not None:
            raise ValueError("un-injected live API call claims a creative context")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path)
    source.add_argument("--core-receipt", type=Path)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path(OUTPUT_PATH))
    args = parser.parse_args(argv)
    if args.core_receipt is not None:
        core = json.loads(args.core_receipt.read_text(encoding="utf-8"))
        evidence = build_evidence_from_core_receipt(core, root=args.root)
    else:
        evidence = build_evidence(args.input)
    validate_evidence(evidence)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
