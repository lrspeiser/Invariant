"""Live, fail-closed core application for creative formula and proof discovery.

Unlike the deterministic public Formula Discovery Job, this application requires authenticated
Claude participation.  It performs the full blind proposer/critic campaign in memory, persists only
credential-free evidence, and binds deterministic discovery plus multi-host verification receipts.
Claude is available across creative roles, never as a verifier or novelty authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .claude_creativity_api import ClaudeRole
from .core_credential import CredentialActivationError, activated_credential
from .creative_expansion import build_creative_expansion, validate_creative_expansion
from .dataset_challenge_suite import validate_dataset_challenges
from .declarative_discovery_operational_campaign import (
    validate_receipt as validate_operational_receipt,
)
from .expanded_math_grammar_controls import (
    validate_receipt as validate_expanded_grammar_receipt,
)
from .external_creativity_live_evidence import (
    build_evidence_from_receipt,
    validate_evidence,
)
from .external_creativity_multi_host import validate_receipt as validate_multi_host_receipt
from .external_creativity_validation import run_campaign
from .idea_lineage import build_idea_archive, validate_idea_archive
from .independent_proof_plan_search import validate_proof_plan_search
from .serious_claim_verification_ladder import (
    REQUIRED_STAGES as SERIOUS_CLAIM_STAGES,
)
from .serious_claim_verification_ladder import (
    validate_receipt as validate_serious_claim_ladder,
)
from .sigma_core import canonical_sha256

CONFIG_PATH = "configs/core_creative_discovery.json"
OUTPUT_PATH = "runs/math/core-creative-discovery/live-runtime.json"
SCHEMA_VERSION = "invariant-core-creative-discovery-runtime-1.2"
CONFIG_SCHEMA = "invariant-core-creative-discovery-config-1.2"


class CoreCreativeDiscoveryError(ValueError):
    """The live core application or one of its required gates failed closed."""


CampaignRunner = Callable[[Path], Mapping[str, Any]]


def _normalized_file_sha256(path: Path) -> str:
    raw = path.read_bytes()
    try:
        raw = raw.decode("utf-8").replace("\r\n", "\n").encode("utf-8")
    except UnicodeDecodeError:
        pass
    return hashlib.sha256(raw).hexdigest()


def _serialized_sha256(value: Mapping[str, Any]) -> str:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_config(root: Path) -> dict[str, Any]:
    path = root / CONFIG_PATH
    value = json.loads(path.read_text(encoding="utf-8"))
    if set(value) != {"app_id", "claude", "components", "release_policy", "schema_version"}:
        raise CoreCreativeDiscoveryError("core creative discovery config keys changed")
    if value["schema_version"] != CONFIG_SCHEMA or value["app_id"] != (
        "invariant.core-creative-discovery"
    ):
        raise CoreCreativeDiscoveryError("core creative discovery config identity changed")
    claude = value["claude"]
    if set(claude) != {
        "credential_env_var",
        "available_creative_roles",
        "required_completed_calls",
        "required_model",
        "required_roles",
    }:
        raise CoreCreativeDiscoveryError("core Claude policy keys changed")
    if (
        claude["credential_env_var"] != "ANTHROPIC_API_KEY"
        or claude["required_completed_calls"] < 8
        or set(claude["required_roles"]) != {"proposer", "critic"}
        or set(claude["available_creative_roles"]) != {role.value for role in ClaudeRole}
    ):
        raise CoreCreativeDiscoveryError("core Claude participation policy is too weak")
    policy = value["release_policy"]
    if (
        policy.get("required_serious_claim_backends")
        != ["exact_arithmetic", "cas", "smt", "interval", "lean"]
        or policy.get("human_prior_art_review_required") is not True
        or policy.get("minimum_independent_level5_passes_before_open_problem", 0) < 3
    ):
        raise CoreCreativeDiscoveryError("core release policy is too weak")
    if set(value["components"]) != {
        "dataset_challenge_receipt",
        "declarative_operational_receipt",
        "expanded_typed_grammar_receipt",
        "live_evidence_output",
        "multi_host_reproduction_receipt",
        "proof_plan_search_receipt",
        "serious_claim_verification_ladder_receipt",
    }:
        raise CoreCreativeDiscoveryError("core component bindings changed")
    return value


def _load_bound_receipts(
    root: Path, config: Mapping[str, Any]
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    components = config["components"]
    dataset_path = root / components["dataset_challenge_receipt"]
    operational_path = root / components["declarative_operational_receipt"]
    multi_host_path = root / components["multi_host_reproduction_receipt"]
    expanded_grammar_path = root / components["expanded_typed_grammar_receipt"]
    proof_plan_path = root / components["proof_plan_search_receipt"]
    serious_claim_ladder_path = root / components[
        "serious_claim_verification_ladder_receipt"
    ]
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    operational = json.loads(operational_path.read_text(encoding="utf-8"))
    multi_host = json.loads(multi_host_path.read_text(encoding="utf-8"))
    expanded_grammar = json.loads(expanded_grammar_path.read_text(encoding="utf-8"))
    proof_plan_search = json.loads(proof_plan_path.read_text(encoding="utf-8"))
    serious_claim_ladder = json.loads(
        serious_claim_ladder_path.read_text(encoding="utf-8")
    )
    validate_dataset_challenges(dataset, root)
    validate_operational_receipt(operational, root)
    validate_multi_host_receipt(multi_host)
    validate_expanded_grammar_receipt(expanded_grammar, root)
    validate_proof_plan_search(proof_plan_search, root)
    validate_serious_claim_ladder(serious_claim_ladder, root)
    return (
        operational,
        multi_host,
        expanded_grammar,
        dataset,
        proof_plan_search,
        serious_claim_ladder,
    )


def _validate_live_campaign(campaign: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    claude = campaign.get("claude", {})
    calls = claude.get("calls", [])
    policy = config["claude"]
    roles = {call.get("role") for call in calls if call.get("status") == "completed"}
    models = {
        call.get("evidence", {}).get("model")
        for call in calls
        if call.get("status") == "completed"
    }
    structured = {
        call.get("evidence", {}).get("model_evidence", {}).get(
            "structured_outputs_supported"
        )
        for call in calls
        if call.get("status") == "completed"
    }
    if (
        campaign.get("claims", {}).get("claude_used_throughout") is not True
        or claude.get("status") != "PASS"
        or claude.get("completed_calls") != policy["required_completed_calls"]
        or claude.get("proposer_hypotheses", 0) < 1
        or len(calls) != policy["required_completed_calls"]
        or roles != set(policy["required_roles"])
        or models != {policy["required_model"]}
        or structured != {True}
    ):
        raise CoreCreativeDiscoveryError("authenticated Claude core participation failed")


def run_core(
    root: Path,
    *,
    credential_file: Path | None = None,
    campaign_runner: CampaignRunner | None = None,
) -> dict[str, Any]:
    """Execute a live core run and return only sanitized, sealed evidence."""

    root = root.resolve()
    config = _load_config(root)
    (
        operational,
        multi_host,
        expanded_grammar,
        dataset_challenges,
        proof_plan_search,
        serious_claim_ladder,
    ) = _load_bound_receipts(root, config)
    environment = None
    if credential_file is not None:
        environment = dict(os.environ)
        environment["INVARIANT_ENV_FILE"] = str(credential_file.resolve())

    def live_runner(project_root: Path) -> Mapping[str, Any]:
        return run_campaign(project_root, live_claude=True)

    runner = campaign_runner or live_runner
    try:
        with activated_credential(
            project_root=root,
            env_var=config["claude"]["credential_env_var"],
            environment=environment,
        ) as activation:
            campaign = dict(runner(root))
            _validate_live_campaign(campaign, config)
            idea_archive = build_idea_archive(campaign)
            creative_expansion = build_creative_expansion(idea_archive, proof_plan_search)
            live_evidence = build_evidence_from_receipt(
                campaign, source_file_sha256=_serialized_sha256(campaign)
            )
    except CredentialActivationError as error:
        raise CoreCreativeDiscoveryError(str(error)) from error
    validate_evidence(live_evidence)
    level5 = campaign["open_problem_gate"]["level5_process_passes"]
    prior_art_reviews = [
        benchmark.get("prior_art", {}).get("human_review", {}).get("status")
        for benchmark in campaign.get("benchmarks", [])
    ]
    serious_claims = sum(
        bool(benchmark.get("claims", {}).get("serious_claim_released"))
        for benchmark in campaign.get("benchmarks", [])
    )
    body = {
        "schema_version": SCHEMA_VERSION,
        "app_id": config["app_id"],
        "source_bindings": {
            "config": {
                "path": CONFIG_PATH,
                "sha256": _normalized_file_sha256(root / CONFIG_PATH),
            },
            "dataset_challenge_receipt": {
                "content_sha256": dataset_challenges["content_sha256"],
                "path": config["components"]["dataset_challenge_receipt"],
            },
            "declarative_operational_receipt": {
                "content_sha256": operational["content_sha256"],
                "path": config["components"]["declarative_operational_receipt"],
            },
            "multi_host_reproduction_receipt": {
                "content_sha256": multi_host["content_sha256"],
                "path": config["components"]["multi_host_reproduction_receipt"],
            },
            "expanded_typed_grammar_receipt": {
                "content_sha256": expanded_grammar["content_sha256"],
                "path": config["components"]["expanded_typed_grammar_receipt"],
            },
            "proof_plan_search_receipt": {
                "content_sha256": proof_plan_search["content_sha256"],
                "path": config["components"]["proof_plan_search_receipt"],
            },
            "serious_claim_verification_ladder_receipt": {
                "content_sha256": serious_claim_ladder["content_sha256"],
                "path": config["components"][
                    "serious_claim_verification_ladder_receipt"
                ],
            },
        },
        "credential_activation": activation.to_evidence(),
        "claude_runtime": {
            "authenticated_messages_api_working": True,
            "available_creative_roles": sorted(config["claude"]["available_creative_roles"]),
            "completed_calls": live_evidence["usage"]["calls"],
            "evidence": live_evidence,
            "model": config["claude"]["required_model"],
            "roles_completed": sorted(config["claude"]["required_roles"]),
            "status": "PASS_REQUIRED_CORE_PARTICIPATION",
        },
        "idea_lineage_archive": idea_archive,
        "creative_expansion": creative_expansion,
        "dataset_challenges": dataset_challenges,
        "proof_plan_search": proof_plan_search,
        "discovery_runtime": {
            "declarative_extensions_admitted": len(
                operational["extension_admission"]["admitted_declarations"]
            ),
            "distinct_behavior_niches": operational["behavioral_archive"]["occupied_niches"],
            "proof_plan_closed": operational["proof_plan"]["closed"],
            "proof_plan_mechanisms": operational["proof_plan"]["mechanisms"],
            "typed_formula_kinds": expanded_grammar["summary"]["admitted_formula_kinds"],
            "typed_grammar_controls_passed": expanded_grammar["summary"]["controls_passed"],
            "typed_grammar_status": expanded_grammar["summary"]["status"],
            "dataset_challenge_kinds": dataset_challenges["summary"]["challenge_kinds"],
            "dataset_mutation_controls_rejected": dataset_challenges["summary"][
                "mutation_controls_rejected"
            ],
            "dataset_positive_controls_passed": dataset_challenges["summary"][
                "positive_controls_passed"
            ],
            "dataset_challenge_status": dataset_challenges["summary"]["status"],
            "independent_proof_plan_mechanisms": proof_plan_search["summary"]["mechanisms"],
            "independent_proof_plan_mutations_rejected": proof_plan_search["summary"][
                "mutation_controls_rejected"
            ],
            "independent_proof_plan_routes_closed": proof_plan_search["summary"][
                "positive_routes_closed"
            ],
            "independent_proof_plan_status": proof_plan_search["summary"]["status"],
        },
        "verification": {
            "backends_required_for_serious_claim": config["release_policy"][
                "required_serious_claim_backends"
            ],
            "multi_host_status": multi_host["reproduction"]["status"],
            "received_machines": multi_host["reproduction"]["received_machines"],
            "lean_kernel_checked": multi_host["lean"]["kernel_checked"],
            "serious_claim_ladder_status": serious_claim_ladder["summary"]["status"],
            "serious_claim_required_stage_order": serious_claim_ladder["summary"][
                "required_stage_order"
            ],
            "serious_claims_released_by_ladder": serious_claim_ladder["release_gate"][
                "serious_claims_released"
            ],
        },
        "release_gate": {
            "human_prior_art_reviews_complete": all(
                status == "COMPLETED" for status in prior_art_reviews
            ),
            "level5_process_passes": level5,
            "minimum_level5_process_passes": config["release_policy"][
                "minimum_independent_level5_passes_before_open_problem"
            ],
            "open_problem_authorized": campaign["open_problem_gate"]["authorized"],
            "serious_claims_released": serious_claims,
            "status": "BLOCKED_CALIBRATION_OR_HUMAN_REVIEW_INCOMPLETE",
        },
        "claims": {
            "claude_is_verifier_authority": False,
            "credential_material_persisted": False,
            "llm_origin_assessment_is_novelty_authority": False,
            "novel_formula_established": False,
            "open_problem_solved": False,
        },
    }
    if (
        body["release_gate"]["open_problem_authorized"] is True
        or serious_claims != 0
        or body["claims"]["credential_material_persisted"] is not False
    ):
        raise CoreCreativeDiscoveryError("core release boundary opened unexpectedly")
    body["content_sha256"] = canonical_sha256(body)
    validate_receipt(body, root)
    return body


def validate_receipt(value: Mapping[str, Any], root: Path | None = None) -> None:
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("content_sha256") != canonical_sha256(body):
        raise CoreCreativeDiscoveryError("core runtime receipt content seal changed")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise CoreCreativeDiscoveryError("core runtime receipt schema changed")
    runtime = value.get("claude_runtime", {})
    credential = value.get("credential_activation", {})
    if (
        runtime.get("status") != "PASS_REQUIRED_CORE_PARTICIPATION"
        or runtime.get("authenticated_messages_api_working") is not True
        or runtime.get("completed_calls", 0) < 8
        or set(runtime.get("available_creative_roles", [])) != {role.value for role in ClaudeRole}
        or credential.get("credential_persisted") is not False
        or credential.get("credential_value_recorded") is not False
    ):
        raise CoreCreativeDiscoveryError("core Claude health evidence changed")
    validate_evidence(runtime["evidence"])
    live_calls = runtime["evidence"].get("calls", [])
    if any(
        call.get("wire_contract_adapter_used") is not (call.get("role") == "critic")
        for call in live_calls
    ):
        raise CoreCreativeDiscoveryError("core provider wire-contract evidence changed")
    validate_idea_archive(value.get("idea_lineage_archive", {}))
    proof_plan_search = value.get("proof_plan_search", {})
    validate_proof_plan_search(proof_plan_search, root)
    validate_creative_expansion(value.get("creative_expansion", {}), proof_plan_search)
    validate_dataset_challenges(value.get("dataset_challenges", {}), root)
    discovery = value.get("discovery_runtime", {})
    if (
        discovery.get("dataset_challenge_kinds")
        != ["intervention", "noisy", "shifted", "unidentifiable"]
        or discovery.get("dataset_positive_controls_passed") != 4
        or discovery.get("dataset_mutation_controls_rejected") != 4
        or discovery.get("dataset_challenge_status")
        != "PASS_EXECUTABLE_DATASET_CHALLENGES"
        or discovery.get("independent_proof_plan_mechanisms")
        != [
            "induction",
            "invariant_preservation",
            "bijection_or_involution",
            "minimal_counterexample_descent",
            "transform_and_extract",
            "contradiction",
        ]
        or discovery.get("independent_proof_plan_routes_closed") != 6
        or discovery.get("independent_proof_plan_mutations_rejected") != 6
        or discovery.get("independent_proof_plan_status")
        != "PASS_INDEPENDENT_PROOF_PLAN_SEARCH"
        or discovery.get("typed_formula_kinds")
        != [
            "finite_product",
            "finite_sum",
            "generating_function",
            "modular_relation",
            "recurrence",
            "tensor_identity",
            "variational_functional",
        ]
        or discovery.get("typed_grammar_controls_passed") != 7
        or discovery.get("typed_grammar_status")
        != "PASS_EXPANDED_TYPED_GRAMMAR_CONTROLS"
    ):
        raise CoreCreativeDiscoveryError("core expanded typed grammar evidence changed")
    verification = value.get("verification", {})
    if (
        verification.get("multi_host_status") != "PASS_MULTI_HOST_REPRODUCTION"
        or verification.get("received_machines", 0) < 2
        or verification.get("lean_kernel_checked") is not True
        or verification.get("serious_claim_ladder_status")
        != "PASS_CANDIDATE_BOUND_LADDER_CALIBRATION"
        or tuple(verification.get("serious_claim_required_stage_order", ()))
        != SERIOUS_CLAIM_STAGES
        or verification.get("serious_claims_released_by_ladder") != 0
    ):
        raise CoreCreativeDiscoveryError("core verification evidence changed")
    claims = value.get("claims", {})
    if any(claims.get(key) is not False for key in claims):
        raise CoreCreativeDiscoveryError("core claim boundary changed")
    if root is not None:
        root = root.resolve()
        bindings = value.get("source_bindings", {})
        config_binding = bindings.get("config", {})
        if (
            config_binding.get("path") != CONFIG_PATH
            or config_binding.get("sha256") != _normalized_file_sha256(root / CONFIG_PATH)
        ):
            raise CoreCreativeDiscoveryError("core config source binding changed")
        for key in (
            "dataset_challenge_receipt",
            "declarative_operational_receipt",
            "expanded_typed_grammar_receipt",
            "multi_host_reproduction_receipt",
            "proof_plan_search_receipt",
            "serious_claim_verification_ladder_receipt",
        ):
            binding = bindings.get(key, {})
            path = (root / str(binding.get("path", ""))).resolve()
            try:
                path.relative_to(root)
            except ValueError as error:
                raise CoreCreativeDiscoveryError("core receipt binding escapes root") from error
            bound = json.loads(path.read_text(encoding="utf-8"))
            if binding.get("content_sha256") != bound.get("content_sha256"):
                raise CoreCreativeDiscoveryError("core receipt source binding changed")
            if key == "expanded_typed_grammar_receipt":
                validate_expanded_grammar_receipt(bound, root)
            if key == "dataset_challenge_receipt":
                validate_dataset_challenges(bound, root)
            if key == "proof_plan_search_receipt":
                validate_proof_plan_search(bound, root)
            if key == "serious_claim_verification_ladder_receipt":
                validate_serious_claim_ladder(bound, root)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run live Claude-backed core discovery")
    run.add_argument("--root", type=Path, default=Path.cwd())
    run.add_argument("--credential-file", type=Path)
    run.add_argument("--output", type=Path, default=Path(OUTPUT_PATH))
    validate = subparsers.add_parser("validate", help="validate a sanitized core receipt")
    validate.add_argument("--receipt", type=Path, default=Path(OUTPUT_PATH))
    validate.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    if args.command == "validate":
        receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
        validate_receipt(receipt, args.root)
    else:
        receipt = run_core(args.root, credential_file=args.credential_file)
        output = args.output if args.output.is_absolute() else args.root / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "claude_status": receipt["claude_runtime"]["status"],
                "completed_calls": receipt["claude_runtime"]["completed_calls"],
                "content_sha256": receipt["content_sha256"],
                "release_status": receipt["release_gate"]["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
