"""Replay sealed live-LLM core evidence on a credential-free CI host."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .core_creative_discovery import validate_receipt as validate_core_receipt
from .retained_piecewise_descendant_campaign import (
    validate_receipt as validate_descendant_receipt,
)
from .sigma_core import canonical_sha256

SCHEMA_VERSION = "invariant-core-creative-ci-reproduction-1.0"
PROJECTION_SCHEMA = "invariant-core-live-llm-evidence-projection-1.0"
CORE_PATH = "runs/math/core-creative-discovery/live-runtime.json"
DESCENDANT_PATH = "runs/math/retained-piecewise-descendant-campaign/live-runtime.json"
CI_REPOSITORY = "lrspeiser/Invariant"
CI_WORKFLOW = "External creativity validation"
CI_JOB = "deterministic-reproduction"
MODEL = "claude-opus-4-6"
_HEX_40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX_64 = re.compile(r"[0-9a-f]{64}\Z")
_ARTIFACT_NAME = re.compile(
    r"external-creativity-(?:ubuntu|windows)-latest-3\.(?:11|12)\Z"
)


class CoreCreativeCIReproductionError(ValueError):
    """The credential-free CI reproduction probe failed closed."""


def _under(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise CoreCreativeCIReproductionError("core reproduction path escaped project root") from error
    if not path.is_file():
        raise CoreCreativeCIReproductionError(f"core reproduction source is missing: {relative}")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CoreCreativeCIReproductionError("core reproduction source is not an object")
    return value


def _strict(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise CoreCreativeCIReproductionError(f"{label} keys changed")


def llm_evidence_projection(
    core: Mapping[str, Any], descendant: Mapping[str, Any]
) -> dict[str, Any]:
    """Return the live evidence slice that is stable across deterministic rebinds."""

    core_runtime = core.get("claude_runtime", {})
    core_evidence = core_runtime.get("evidence", {})
    descendant_runtime = core.get("retained_piecewise_descendant_claude_runtime", {})
    descendant_summary = descendant.get("summary", {})
    descendant_novelty = descendant.get("novelty_axes", {})
    credential = core.get("credential_activation", {})
    prompt_context = core.get("llm_prompt_context", {})
    lineage = core.get("idea_lineage_archive", {})
    projection = {
        "schema_version": PROJECTION_SCHEMA,
        "app_id": core.get("app_id"),
        "core_lane": {
            "authenticated_messages_api_working": core_runtime.get(
                "authenticated_messages_api_working"
            ),
            "completed_calls": core_runtime.get("completed_calls"),
            "evidence_content_sha256": core_evidence.get("content_sha256"),
            "model": core_runtime.get("model"),
            "roles_completed": core_runtime.get("roles_completed"),
            "status": core_runtime.get("status"),
            "total_tokens": core_evidence.get("usage", {}).get("total_tokens"),
        },
        "descendant_lane": {
            "admitted_executable_descendants": descendant_summary.get(
                "admitted_executable_descendants"
            ),
            "authenticated_messages_api_working": descendant_runtime.get(
                "authenticated_messages_api_working"
            ),
            "completed_calls": descendant_runtime.get("completed_calls"),
            "descendant_ideas_retained": descendant_summary.get(
                "descendant_ideas_retained"
            ),
            "distinct_behavior_signatures": descendant_novelty.get(
                "distinct_descendant_behavior_signatures"
            ),
            "distinct_proof_mechanism_signatures": descendant_novelty.get(
                "distinct_descendant_proof_mechanism_signatures"
            ),
            "model": descendant_runtime.get("model"),
            "receipt_content_sha256": descendant.get("content_sha256"),
            "roles_completed": descendant_runtime.get("roles_completed"),
            "status": descendant_runtime.get("status"),
            "total_tokens": descendant_runtime.get("usage", {}).get("total_tokens"),
        },
        "credential_boundary": {
            "credential_env_var": credential.get("credential_env_var"),
            "credential_persisted": credential.get("credential_persisted"),
            "credential_value_recorded": credential.get("credential_value_recorded"),
            "injected_into_original_process": credential.get("injected_into_process"),
            "source_kind": credential.get("source_kind"),
            "source_locator_sha256": credential.get("source_locator_sha256"),
        },
        "prompt_and_lineage": {
            "idea_lineage_archive_sha256": lineage.get("content_sha256"),
            "prompt_context_sha256": prompt_context.get("content_sha256"),
            "prompt_context_status": prompt_context.get("status"),
        },
        "claim_boundary": {
            "llm_is_verifier_authority": core.get("claims", {}).get(
                "claude_is_verifier_authority"
            ),
            "literature_novelty_established": core.get("claims", {}).get(
                "novel_formula_established"
            ),
            "open_problem_solved": core.get("claims", {}).get("open_problem_solved"),
        },
    }
    validate_projection(projection)
    return projection


def validate_projection(value: Mapping[str, Any]) -> None:
    _strict(
        value,
        {
            "app_id",
            "claim_boundary",
            "core_lane",
            "credential_boundary",
            "descendant_lane",
            "prompt_and_lineage",
            "schema_version",
        },
        "core LLM evidence projection",
    )
    _strict(
        value["core_lane"],
        {
            "authenticated_messages_api_working",
            "completed_calls",
            "evidence_content_sha256",
            "model",
            "roles_completed",
            "status",
            "total_tokens",
        },
        "core LLM lane",
    )
    _strict(
        value["descendant_lane"],
        {
            "admitted_executable_descendants",
            "authenticated_messages_api_working",
            "completed_calls",
            "descendant_ideas_retained",
            "distinct_behavior_signatures",
            "distinct_proof_mechanism_signatures",
            "model",
            "receipt_content_sha256",
            "roles_completed",
            "status",
            "total_tokens",
        },
        "descendant LLM lane",
    )
    _strict(
        value["credential_boundary"],
        {
            "credential_env_var",
            "credential_persisted",
            "credential_value_recorded",
            "injected_into_original_process",
            "source_kind",
            "source_locator_sha256",
        },
        "credential boundary",
    )
    core = value["core_lane"]
    descendant = value["descendant_lane"]
    credential = value["credential_boundary"]
    boundary = value["claim_boundary"]
    prompt = value["prompt_and_lineage"]
    if (
        value.get("schema_version") != PROJECTION_SCHEMA
        or value.get("app_id") != "invariant.core-creative-discovery"
        or core.get("status") != "PASS_REQUIRED_CORE_PARTICIPATION"
        or core.get("authenticated_messages_api_working") is not True
        or core.get("completed_calls", 0) < 8
        or core.get("model") != MODEL
        or set(core.get("roles_completed", [])) != {"proposer", "critic"}
        or core.get("total_tokens", 0) <= 0
        or _HEX_64.fullmatch(str(core.get("evidence_content_sha256", ""))) is None
        or descendant.get("status") != "PASS_LIVE_CORE_DESCENDANT_PARTICIPATION"
        or descendant.get("authenticated_messages_api_working") is not True
        or descendant.get("completed_calls", 0) < 6
        or descendant.get("model") != MODEL
        or set(descendant.get("roles_completed", []))
        != {"recombiner", "representation_inventor"}
        or descendant.get("total_tokens", 0) <= 0
        or descendant.get("descendant_ideas_retained", 0) < 1
        or descendant.get("admitted_executable_descendants", 0) < 1
        or _HEX_64.fullmatch(str(descendant.get("receipt_content_sha256", ""))) is None
        or credential.get("credential_env_var") != "ANTHROPIC_API_KEY"
        or credential.get("source_kind") != "user_invariant_env_file"
        or credential.get("injected_into_original_process") is not True
        or credential.get("credential_persisted") is not False
        or credential.get("credential_value_recorded") is not False
        or _HEX_64.fullmatch(str(credential.get("source_locator_sha256", ""))) is None
        or boundary
        != {
            "llm_is_verifier_authority": False,
            "literature_novelty_established": False,
            "open_problem_solved": False,
        }
        or prompt.get("prompt_context_status")
        != "PASS_CONTEXT_BOUND_TO_AUTHENTICATED_CALLS"
        or _HEX_64.fullmatch(str(prompt.get("prompt_context_sha256", ""))) is None
        or _HEX_64.fullmatch(str(prompt.get("idea_lineage_archive_sha256", ""))) is None
    ):
        raise CoreCreativeCIReproductionError("core LLM evidence projection policy failed")


def _ci_provenance(environment: Mapping[str, str] | None) -> dict[str, Any]:
    environment = os.environ if environment is None else environment
    run_id_raw = environment.get("GITHUB_RUN_ID", "")
    attempt_raw = environment.get("GITHUB_RUN_ATTEMPT", "")
    head_sha = environment.get("INVARIANT_EVIDENCE_HEAD_SHA") or environment.get("GITHUB_SHA")
    artifact_name = environment.get("INVARIANT_EVIDENCE_ARTIFACT_NAME")
    operating_system = environment.get("INVARIANT_EVIDENCE_OPERATING_SYSTEM")
    python_version = environment.get("INVARIANT_EVIDENCE_PYTHON_VERSION")
    runner_os = environment.get("RUNNER_OS")
    complete = (
        environment.get("GITHUB_ACTIONS") == "true"
        and environment.get("GITHUB_REPOSITORY") == CI_REPOSITORY
        and environment.get("GITHUB_WORKFLOW") == CI_WORKFLOW
        and environment.get("INVARIANT_EVIDENCE_JOB") == CI_JOB
        and run_id_raw.isdigit()
        and attempt_raw.isdigit()
        and isinstance(head_sha, str)
        and _HEX_40.fullmatch(head_sha) is not None
        and environment.get("GITHUB_EVENT_NAME")
        in {"pull_request", "push", "workflow_dispatch"}
        and artifact_name is not None
        and _ARTIFACT_NAME.fullmatch(artifact_name) is not None
        and operating_system in {"ubuntu-latest", "windows-latest"}
        and python_version in {"3.11", "3.12"}
        and runner_os == ("Linux" if operating_system == "ubuntu-latest" else "Windows")
        and bool(environment.get("RUNNER_ARCH"))
        and bool(environment.get("RUNNER_NAME"))
    )
    return {
        "artifact_name": artifact_name,
        "complete": complete,
        "event_name": environment.get("GITHUB_EVENT_NAME"),
        "head_sha": head_sha,
        "job": environment.get("INVARIANT_EVIDENCE_JOB"),
        "operating_system": operating_system,
        "provider": "github_actions" if complete else "local_or_incomplete",
        "python_version": python_version,
        "repository": environment.get("GITHUB_REPOSITORY"),
        "run_attempt": int(attempt_raw) if attempt_raw.isdigit() else None,
        "run_id": int(run_id_raw) if run_id_raw.isdigit() else None,
        "runner_arch": environment.get("RUNNER_ARCH"),
        "runner_name": environment.get("RUNNER_NAME"),
        "runner_os": runner_os,
        "workflow": environment.get("GITHUB_WORKFLOW"),
    }


def build_receipt(
    root: Path, environment: Mapping[str, str] | None = None
) -> dict[str, Any]:
    root = root.resolve()
    core = _read_json(_under(root, CORE_PATH))
    descendant = _read_json(_under(root, DESCENDANT_PATH))
    validate_core_receipt(core, root)
    validate_descendant_receipt(descendant, root)
    projection = llm_evidence_projection(core, descendant)
    provenance = _ci_provenance(environment)
    body = {
        "schema_version": SCHEMA_VERSION,
        "ci_provenance": provenance,
        "source_bindings": {
            "core_runtime": {
                "content_sha256": core["content_sha256"],
                "path": CORE_PATH,
            },
            "descendant_runtime": {
                "content_sha256": descendant["content_sha256"],
                "path": DESCENDANT_PATH,
            },
        },
        "llm_evidence_projection": projection,
        "llm_evidence_projection_sha256": canonical_sha256(projection),
        "verification": {
            "core_runtime_validated": True,
            "descendant_runtime_validated": True,
            "new_provider_calls": 0,
            "provider_credential_available_on_reproduction_host": False,
            "status": (
                "PASS_CORE_LLM_EVIDENCE_REPRODUCTION"
                if provenance["complete"]
                else "BLOCKED_INCOMPLETE_CI_PROVENANCE"
            ),
        },
        "claim_boundary": {
            "authenticated_live_evidence_replayed": True,
            "credential_value_or_path_persisted": False,
            "literature_novelty_established": False,
            "physical_bare_metal_identity_claimed": False,
        },
    }
    receipt = {**body, "content_sha256": canonical_sha256(body)}
    validate_receipt(receipt, root)
    return receipt


def validate_receipt(
    value: Mapping[str, Any],
    root: Path | None = None,
    *,
    require_ci_provenance: bool = False,
) -> None:
    _strict(
        value,
        {
            "ci_provenance",
            "claim_boundary",
            "content_sha256",
            "llm_evidence_projection",
            "llm_evidence_projection_sha256",
            "schema_version",
            "source_bindings",
            "verification",
        },
        "core CI reproduction receipt",
    )
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if (
        value.get("content_sha256") != canonical_sha256(body)
        or value.get("schema_version") != SCHEMA_VERSION
    ):
        raise CoreCreativeCIReproductionError("core CI reproduction receipt seal changed")
    projection = value["llm_evidence_projection"]
    validate_projection(projection)
    if value.get("llm_evidence_projection_sha256") != canonical_sha256(projection):
        raise CoreCreativeCIReproductionError("core LLM evidence projection seal changed")
    provenance = value["ci_provenance"]
    complete = provenance.get("complete") is True
    if require_ci_provenance and not complete:
        raise CoreCreativeCIReproductionError("core CI provenance is incomplete")
    if complete and (
        provenance.get("provider") != "github_actions"
        or provenance.get("repository") != CI_REPOSITORY
        or provenance.get("workflow") != CI_WORKFLOW
        or provenance.get("job") != CI_JOB
        or not isinstance(provenance.get("run_id"), int)
        or not isinstance(provenance.get("run_attempt"), int)
        or _HEX_40.fullmatch(str(provenance.get("head_sha", ""))) is None
        or provenance.get("event_name")
        not in {"pull_request", "push", "workflow_dispatch"}
        or _ARTIFACT_NAME.fullmatch(str(provenance.get("artifact_name", ""))) is None
        or provenance.get("operating_system") not in {"ubuntu-latest", "windows-latest"}
        or provenance.get("python_version") not in {"3.11", "3.12"}
        or not provenance.get("runner_name")
    ):
        raise CoreCreativeCIReproductionError("core CI provenance changed")
    expected_status = (
        "PASS_CORE_LLM_EVIDENCE_REPRODUCTION"
        if complete
        else "BLOCKED_INCOMPLETE_CI_PROVENANCE"
    )
    if value.get("verification") != {
        "core_runtime_validated": True,
        "descendant_runtime_validated": True,
        "new_provider_calls": 0,
        "provider_credential_available_on_reproduction_host": False,
        "status": expected_status,
    }:
        raise CoreCreativeCIReproductionError("core CI verification boundary changed")
    if value.get("claim_boundary") != {
        "authenticated_live_evidence_replayed": True,
        "credential_value_or_path_persisted": False,
        "literature_novelty_established": False,
        "physical_bare_metal_identity_claimed": False,
    }:
        raise CoreCreativeCIReproductionError("core CI claim boundary changed")
    if root is not None:
        root = root.resolve()
        core = _read_json(_under(root, CORE_PATH))
        descendant = _read_json(_under(root, DESCENDANT_PATH))
        validate_core_receipt(core, root)
        validate_descendant_receipt(descendant, root)
        expected_bindings = {
            "core_runtime": {"content_sha256": core["content_sha256"], "path": CORE_PATH},
            "descendant_runtime": {
                "content_sha256": descendant["content_sha256"],
                "path": DESCENDANT_PATH,
            },
        }
        if (
            value.get("source_bindings") != expected_bindings
            or projection != llm_evidence_projection(core, descendant)
        ):
            raise CoreCreativeCIReproductionError("core CI source binding changed")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-ci-provenance", action="store_true")
    args = parser.parse_args(argv)
    receipt = build_receipt(args.root)
    validate_receipt(receipt, args.root, require_ci_provenance=args.require_ci_provenance)
    output = args.output if args.output.is_absolute() else args.root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "content_sha256": receipt["content_sha256"],
                "llm_evidence_projection_sha256": receipt[
                    "llm_evidence_projection_sha256"
                ],
                "status": receipt["verification"]["status"],
            },
            sort_keys=True,
        )
    )
    return 0 if receipt["verification"]["status"].startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
