"""Generate quarantined G4 theory-family proposals with one budget-capped LLM call."""

from __future__ import annotations

import argparse
import json
import os
import re
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .gravity_g1_pilot import _binding, _file_sha256, _load_json, _metric
from .gravity_g4_photometric_law_construction import (
    validate_receipt as validate_g4_failure_receipt,
)
from .sigma_core import canonical_json_bytes, canonical_sha256

SCHEMA = "invariant-gravity-g4-llm-creativity-receipt-1.0"
FAILURE_SCHEMA = "invariant-gravity-g4-llm-creativity-failure-receipt-1.0"
CONFIG_SCHEMA = "invariant-gravity-g4-llm-creativity-config-1.0"
CONFIG_PATH = "configs/gravity_g4_llm_creativity_campaign.json"
SOURCE_PATH = "src/sigma_theory_compiler/gravity_g4_llm_creativity_campaign.py"
TEST_PATH = "tests/test_gravity_g4_llm_creativity_campaign.py"
OUTPUT_PATH = "runs/gravity/g4/llm-creativity-proposals-v1.json"
FAILURE_OUTPUT_PATH = "runs/gravity/g4/llm-creativity-campaign-failure-v1.json"
RAW_RESPONSE_PATH = "runs/gravity/g4/llm-creativity-provider-response-v2.json"
REQUEST_ID = "gravity-g4-creativity-20260827-004"

PROPOSAL_FIELDS = {
    "proposal_id",
    "title",
    "origin_self_assessment",
    "known_analogue",
    "mechanism",
    "equation_template",
    "variables",
    "universal_parameters",
    "why_not_merely_a_rewrite",
    "expected_observational_signature",
    "cheapest_falsifier",
    "likely_failure_mode",
    "creativity_score",
    "physical_plausibility_score",
    "testability_score",
}


class GravityG4LlmCreativityError(ValueError):
    """The LLM creativity contract, provider result, or evidence is inconsistent."""


def load_config(root: Path) -> Mapping[str, Any]:
    """Load the live-call contract and validate the blocked G4 predecessor."""

    root = root.resolve()
    config = _load_json(root / CONFIG_PATH)
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise GravityG4LlmCreativityError("LLM creativity config schema changed")
    binding = config.get("failure_context_binding", {})
    path = root / str(binding.get("path"))
    if _file_sha256(path) != binding.get("file_sha256"):
        raise GravityG4LlmCreativityError("LLM failure-context file changed")
    failure = _load_json(path)
    validate_g4_failure_receipt(failure, root=root)
    if (
        failure.get("content_sha256") != binding.get("content_sha256")
        or failure.get("decision") != binding.get("required_decision")
    ):
        raise GravityG4LlmCreativityError("LLM failure context changed")
    provider = config.get("provider", {})
    if (
        provider.get("model") != "claude-opus-5"
        or provider.get("maximum_calls") != 4
        or provider.get("maximum_call_usd") != "5.000000"
        or provider.get("maximum_campaign_usd") != "20.000000"
        or provider.get("api_key_env_var") != "ANTHROPIC_API_KEY"
    ):
        raise GravityG4LlmCreativityError("LLM provider or budget changed")
    contract = config.get("proposal_contract", {})
    if contract.get("proposal_count") != 12:
        raise GravityG4LlmCreativityError("LLM proposal count changed")
    if set(contract.get("required_fields", ())) != PROPOSAL_FIELDS:
        raise GravityG4LlmCreativityError("LLM proposal fields changed")
    if contract.get("historical_novelty_may_be_claimed") is not False:
        raise GravityG4LlmCreativityError("LLM contract permits novelty claims")
    if config.get("data_boundary", {}).get("confirmation_evaluator_accesses_allowed") != 0:
        raise GravityG4LlmCreativityError("LLM contract permits confirmation access")
    retained = config.get("attempt_audit_before_final_retry", {}).get(
        "retained_failed_response", {}
    )
    if _file_sha256(root / str(retained.get("path"))) != retained.get("file_sha256"):
        raise GravityG4LlmCreativityError("retained failed provider response changed")
    final = config.get("final_failed_response", {})
    if (
        config.get("campaign_closed") is not True
        or _file_sha256(root / str(final.get("path"))) != final.get("file_sha256")
        or final.get("usable_proposals") != 0
    ):
        raise GravityG4LlmCreativityError("final failed provider response changed")
    return config


def build_prompt(config: Mapping[str, Any]) -> str:
    """Construct a target-free creative brief from aggregate failure geometry."""

    labels = ", ".join(config["proposal_contract"]["origin_labels"])
    return f"""You are proposing new mathematical grammar families for a falsifiable galaxy-dynamics search.

Empirical situation (exploration/model-development data only):
- 139 galaxies and 2,720 rotation-curve points; an entirely sealed partition remains unobserved.
- Newtonian baryons: chi-square 1,697,326.398.
- A frozen empirical radial-acceleration relation (RAR): 130,714.689.
- A target-blind Extra Trees residual learner: 122,436.722, useful but not a compact law.
- Best compact acceleration-only correction: 125,143.249, classified KNOWN_FAMILY.
- A 1,710-cell photometric grammar added disk/bulge surface-brightness profiles; best chi-square 124,807.946, classified COMBINATION, and two population strata became over 20% worse than RAR.
- The unchanged flexible per-galaxy NFW-shaped performance ceiling plus slack is 33,458.807. The remaining gap is 91,349.139, so cosmetic RAR recalibration is inadequate.
- Per-galaxy formula atlases fit very well with two local coefficients, but their classes and coefficients do not transfer accurately to unseen galaxies. A viable family must explain that variation without hidden object-specific gravitational parameters.

Allowed target-blind inputs:
r in kpc; published V_gas, V_disk, V_bulge in km/s; fixed stellar mass-to-light multipliers; published disk and bulge surface brightness in L_sun/pc^2; dimensionally valid local derivatives, integrals, convolution kernels, and universal constants. Environmental fields may be proposed only if directly measurable and universally coupled.

Forbidden:
observed target velocities in the formula; object names/IDs; observer distance as a causal input; per-object gravity constants; lookup tables; unseen-matter profiles or fitted halo quantities; claims of historical novelty; treating your own score as evidence.

Task:
Return exactly 12 mechanistically distinct formula/operator families. Favor creative first-principles mechanisms, nonlocal or geometric constructions, modified-inertia possibilities, conservation-derived operators, and cross-domain analogies when dimensionally coherent. Do not merely rename an acceleration interpolation. Each must be implementable as a finite typed grammar and cheap enough for numerical screening. Include the cheapest decisive falsifier and the most likely way it fails.

For origin_self_assessment use exactly one of: {labels}. This label is non-authoritative and will never prune the idea. If a proposal resembles MOND/AQUAL, modified inertia, conformal gravity, emergent/entropic gravity, superfluid analogies, nonlocal gravity, disk-potential corrections, or another known family, say so plainly in known_analogue and choose a conservative label. Use proposed_new_construction only for a specific recombination or operator you cannot recognize as a standard instance; never equate it with historical novelty.

Scores are integers 1..5 and are self-assessments only. Universal parameters must number at most {config['proposal_contract']['maximum_universal_parameters_per_proposal']}. Equation templates must state dimensions or normalizations sufficiently clearly for a typed implementation.

Return one JSON object with schema_version "g4-creative-proposals-1.0" and a proposals array containing exactly 12 proposal objects. Return JSON only, without a Markdown fence or surrounding commentary. Count the array entries before returning it.
"""


def output_schema(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the provider-enforced structured-output schema."""

    # The rich proposal objects are JSON-encoded inside twelve required string
    # slots to stay below the provider's compiled-grammar complexity limit.
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["schema_version", "proposals"],
        "properties": {
            "schema_version": {
                "type": "string",
                "enum": ["g4-creative-proposals-1.0"],
            },
            "proposals": {
                "type": "object",
                "additionalProperties": False,
                "required": [f"proposal_{index:02d}" for index in range(1, 13)],
                "properties": {
                    f"proposal_{index:02d}": {
                        "type": "string",
                        "description": (
                            "A JSON-serialized proposal object matching the proposal contract"
                        ),
                    }
                    for index in range(1, 13)
                },
            },
        },
    }


def build_provider_request(config: Mapping[str, Any], prompt: str) -> dict[str, Any]:
    provider = config["provider"]
    return {
        "model": provider["model"],
        "max_tokens": int(provider["maximum_output_tokens"]),
        "messages": [{"role": "user", "content": prompt}],
        "output_config": {"effort": provider["effort"]},
        "system": "Generate diverse scientific hypotheses, disclose known analogues, and never present a proposal or self-assessment as empirical validation or proof of novelty.",
    }


def validate_proposals(value: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate provider output again locally even when structured output was requested."""

    if set(value) != {"schema_version", "proposals"}:
        raise GravityG4LlmCreativityError("proposal envelope shape changed")
    if value.get("schema_version") != "g4-creative-proposals-1.0":
        raise GravityG4LlmCreativityError("proposal schema changed")
    proposals = value.get("proposals")
    if not isinstance(proposals, list) or len(proposals) != int(
        config["proposal_contract"]["proposal_count"]
    ):
        raise GravityG4LlmCreativityError("proposal count changed")
    allowed_labels = set(config["proposal_contract"]["origin_labels"])
    seen = set()
    normalized = []
    for proposal in proposals:
        if not isinstance(proposal, Mapping) or set(proposal) != PROPOSAL_FIELDS:
            raise GravityG4LlmCreativityError("proposal fields changed")
        proposal_id = str(proposal["proposal_id"])
        if not re.fullmatch(r"g4-[a-z0-9-]{3,64}", proposal_id) or proposal_id in seen:
            raise GravityG4LlmCreativityError("proposal ID is invalid or duplicated")
        if proposal["origin_self_assessment"] not in allowed_labels:
            raise GravityG4LlmCreativityError("proposal origin label changed")
        for field in PROPOSAL_FIELDS - {
            "variables",
            "universal_parameters",
            "creativity_score",
            "physical_plausibility_score",
            "testability_score",
        }:
            if not isinstance(proposal[field], str) or not proposal[field].strip():
                raise GravityG4LlmCreativityError("proposal text field is empty")
        parameters = proposal["universal_parameters"]
        variables = proposal["variables"]
        if not isinstance(parameters, list) or not isinstance(variables, list):
            raise GravityG4LlmCreativityError("proposal variable lists changed")
        if any(not isinstance(item, str) or not item.strip() for item in parameters + variables):
            raise GravityG4LlmCreativityError("proposal variable list contains empty text")
        if len(parameters) > int(
            config["proposal_contract"]["maximum_universal_parameters_per_proposal"]
        ):
            raise GravityG4LlmCreativityError("proposal has too many parameters")
        for score in (
            "creativity_score",
            "physical_plausibility_score",
            "testability_score",
        ):
            if isinstance(proposal[score], bool) or not 1 <= int(proposal[score]) <= 5:
                raise GravityG4LlmCreativityError("proposal score is outside range")
        normalized.append({key: proposal[key] for key in sorted(PROPOSAL_FIELDS)})
        seen.add(proposal_id)
    return {"proposals": normalized, "schema_version": value["schema_version"]}


def canonicalize_proposal_ids(value: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Replace provider-written display IDs with stable positional artifact IDs."""

    proposals_value = value.get("proposals")
    if isinstance(proposals_value, Mapping):
        expected = {f"proposal_{index:02d}" for index in range(1, 13)}
        if set(proposals_value) != expected:
            raise GravityG4LlmCreativityError("provider proposal slots changed")
        proposals = []
        for key in sorted(expected):
            proposal = proposals_value[key]
            if isinstance(proposal, str):
                try:
                    proposal = json.loads(proposal)
                except json.JSONDecodeError as error:
                    raise GravityG4LlmCreativityError(
                        f"provider proposal slot {key} is not JSON"
                    ) from error
            proposals.append(proposal)
    elif isinstance(proposals_value, list):
        proposals = proposals_value
    else:
        raise GravityG4LlmCreativityError("provider proposal collection is missing")
    normalized = []
    provider_ids = []
    for index, proposal in enumerate(proposals, start=1):
        if not isinstance(proposal, Mapping):
            raise GravityG4LlmCreativityError("provider proposal is not an object")
        row = dict(proposal)
        provider_ids.append(str(row.get("proposal_id", "")))
        row["proposal_id"] = f"g4-proposal-{index:02d}"
        normalized.append(row)
    return {
        "schema_version": value.get("schema_version"),
        "proposals": normalized,
    }, provider_ids


Provider = Callable[[Mapping[str, Any], str, Mapping[str, Any]], Mapping[str, Any]]


def anthropic_provider(
    payload: Mapping[str, Any], secret: str, config: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Call Anthropic without ever serializing the API key into an artifact."""

    provider = config["provider"]
    request = urllib.request.Request(
        str(provider["endpoint"]),
        data=canonical_json_bytes(payload),
        headers={
            "anthropic-version": str(provider["anthropic_version"]),
            "content-type": "application/json",
            "x-api-key": secret,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")
        raise GravityG4LlmCreativityError(
            f"Anthropic request failed with HTTP {error.code}: {detail}"
        ) from error


def _response_text(response: Mapping[str, Any]) -> str:
    blocks = response.get("content")
    if not isinstance(blocks, list):
        raise GravityG4LlmCreativityError("provider content is missing")
    text = "".join(
        str(block.get("text", ""))
        for block in blocks
        if isinstance(block, Mapping) and block.get("type") == "text"
    )
    if not text:
        raise GravityG4LlmCreativityError("provider returned no text output")
    return text


def _decode_response_json(response: Mapping[str, Any]) -> Mapping[str, Any]:
    """Decode plain JSON, tolerating only a single surrounding Markdown fence."""

    text = _response_text(response).strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise GravityG4LlmCreativityError("provider output is not JSON") from error
    if not isinstance(value, Mapping):
        raise GravityG4LlmCreativityError("provider JSON envelope is not an object")
    return value


def build_receipt(
    root: Path,
    *,
    provider: Provider,
    secret: str,
    raw_response_path: Path | None = None,
) -> dict[str, Any]:
    """Make one live or mocked provider call and quarantine the proposals."""

    root = root.resolve()
    config = load_config(root)
    prompt = build_prompt(config)
    payload = build_provider_request(config, prompt)
    prompt_bytes = len(prompt.encode("utf-8"))
    maximum_output = int(config["provider"]["maximum_output_tokens"])
    maximum_cost = (
        prompt_bytes
        * float(config["provider"]["conservative_pricing_ceiling_usd_per_million_input_tokens"])
        + maximum_output
        * float(config["provider"]["conservative_pricing_ceiling_usd_per_million_output_tokens"])
    ) / 1_000_000
    if maximum_cost > float(config["provider"]["maximum_call_usd"]):
        raise GravityG4LlmCreativityError("conservative request cost exceeds call cap")
    response = provider(payload, secret, config)
    if raw_response_path is not None:
        _write_immutable(raw_response_path, response)
    if response.get("model") != config["provider"]["model"]:
        raise GravityG4LlmCreativityError("provider model changed")
    decoded = _decode_response_json(response)
    canonicalized, provider_ids = canonicalize_proposal_ids(decoded)
    proposals = validate_proposals(canonicalized, config)
    usage = response.get("usage", {})
    input_tokens = int(usage.get("input_tokens", 0))
    output_tokens = int(usage.get("output_tokens", 0))
    if input_tokens <= 0 or output_tokens <= 0 or output_tokens > maximum_output:
        raise GravityG4LlmCreativityError("provider usage changed")
    conservative_usage_cost = (
        input_tokens
        * float(config["provider"]["conservative_pricing_ceiling_usd_per_million_input_tokens"])
        + output_tokens
        * float(config["provider"]["conservative_pricing_ceiling_usd_per_million_output_tokens"])
    ) / 1_000_000
    if conservative_usage_cost > float(config["provider"]["maximum_call_usd"]):
        raise GravityG4LlmCreativityError("provider usage exceeds call cap")
    label_counts = {
        label: sum(
            row["origin_self_assessment"] == label for row in proposals["proposals"]
        )
        for label in config["proposal_contract"]["origin_labels"]
    }
    body: dict[str, Any] = {
        "schema_version": SCHEMA,
        "goal": "G4_CREATIVE_GRAMMAR_GENERATION",
        "decision": "QUARANTINE_G4_LLM_PROPOSALS_FOR_TYPED_TESTING",
        "claims": {
            "alternative_to_gr_discovered": False,
            "confirmation_galaxy_evaluated": False,
            "empirical_validation_completed": False,
            "historical_novelty_established": False,
            "llm_self_assessment_is_authoritative": False,
        },
        "config": {"content_sha256": canonical_sha256(config), "path": CONFIG_PATH},
        "counts": {
            "confirmation_evaluator_accesses": 0,
            "completed_inference_calls": 4,
            "provider_http_requests": 10,
            "proposals": len(proposals["proposals"]),
        },
        "diagnostic_disclosure": config["diagnostic_disclosure"],
        "lineage": {
            "failure_context_content_sha256": config["failure_context_binding"][
                "content_sha256"
            ],
            "output_schema_sha256": canonical_sha256(output_schema(config)),
            "prompt_sha256": canonical_sha256(prompt),
            "provider_request_sha256": canonical_sha256(payload),
            "request_id": REQUEST_ID,
            "raw_provider_response": (
                _binding(root, RAW_RESPONSE_PATH)
                if raw_response_path is not None
                else None
            ),
        },
        "origin_label_counts": label_counts,
        "proposals": proposals["proposals"],
        "provider_proposal_ids_before_canonicalization": provider_ids,
        "provider": {
            "conservative_campaign_cost_ceiling_usd": config["provider"][
                "maximum_campaign_usd"
            ],
            "conservative_maximum_request_cost_usd": _metric(maximum_cost),
            "conservative_usage_cost_ceiling_usd": _metric(conservative_usage_cost),
            "input_tokens": input_tokens,
            "maximum_call_usd": config["provider"]["maximum_call_usd"],
            "model": response["model"],
            "output_tokens": output_tokens,
            "provider_request_id": str(response.get("id", "unavailable")),
            "prior_completed_inference_usage_available": False,
            "stop_reason": str(response.get("stop_reason", "unknown")),
        },
        "limitations": [
            "These are untested proposals generated after the model saw aggregate exploration failures.",
            "Origin labels are proposer self-assessments and cannot establish novelty or prune a branch.",
            "Every proposal requires typed dimensional implementation, finite search expansion, numerical screening, and CPU replay before it can become a candidate.",
            "No per-galaxy target, residual, identity, or confirmation datum was sent to the provider.",
            "Two earlier completed inferences were lost before persistence after local validation failures; their usage and content are unavailable.",
            "A third persisted response contained twelve empty proposal slots and no usable proposal.",
            "Four earlier HTTP requests were rejected at schema validation before inference.",
        ],
        "source_bindings": {
            "config": _binding(root, CONFIG_PATH),
            "source": _binding(root, SOURCE_PATH),
            "test": _binding(root, TEST_PATH),
        },
    }
    body["content_sha256"] = canonical_sha256(body)
    return body


def validate_receipt(receipt: Mapping[str, Any], *, root: Path) -> None:
    root = root.resolve()
    if receipt.get("schema_version") != SCHEMA:
        raise GravityG4LlmCreativityError("LLM creativity receipt schema changed")
    body = dict(receipt)
    supplied = body.pop("content_sha256", None)
    if supplied != canonical_sha256(body):
        raise GravityG4LlmCreativityError("LLM creativity receipt seal changed")
    config = load_config(root)
    if receipt.get("config", {}).get("content_sha256") != canonical_sha256(config):
        raise GravityG4LlmCreativityError("LLM creativity config binding changed")
    for key, path in (("config", CONFIG_PATH), ("source", SOURCE_PATH), ("test", TEST_PATH)):
        if receipt.get("source_bindings", {}).get(key) != _binding(root, path):
            raise GravityG4LlmCreativityError(f"LLM creativity {key} binding changed")
    claims = receipt.get("claims", {})
    counts = receipt.get("counts", {})
    if (
        counts.get("completed_inference_calls") != 4
        or counts.get("provider_http_requests") != 10
        or counts.get("confirmation_evaluator_accesses") != 0
    ):
        raise GravityG4LlmCreativityError("LLM creativity access counts changed")
    if claims.get("historical_novelty_established") is not False:
        raise GravityG4LlmCreativityError("LLM creativity overstates novelty")
    if claims.get("empirical_validation_completed") is not False:
        raise GravityG4LlmCreativityError("LLM creativity overstates validation")
    validate_proposals(
        {
            "proposals": receipt.get("proposals"),
            "schema_version": "g4-creative-proposals-1.0",
        },
        config,
    )


def build_failure_receipt(root: Path) -> dict[str, Any]:
    """Seal the exhausted campaign without manufacturing proposals from failed calls."""

    root = root.resolve()
    config = load_config(root)
    first_path = root / str(
        config["attempt_audit_before_final_retry"]["retained_failed_response"]["path"]
    )
    final_path = root / str(config["final_failed_response"]["path"])
    first = _load_json(first_path)
    final = _load_json(final_path)
    first_usage = first.get("usage", {})
    final_usage = final.get("usage", {})
    if (
        first.get("model") != config["provider"]["model"]
        or final.get("model") != config["provider"]["model"]
        or int(first_usage.get("output_tokens", 0)) != 15_090
        or int(final_usage.get("output_tokens", 0)) != 20_000
        or final.get("stop_reason") != "max_tokens"
    ):
        raise GravityG4LlmCreativityError("retained provider failure evidence changed")
    rates = config["provider"]
    retained_input = int(first_usage["input_tokens"]) + int(final_usage["input_tokens"])
    retained_output = int(first_usage["output_tokens"]) + int(
        final_usage["output_tokens"]
    )
    retained_cost_ceiling = (
        retained_input
        * float(rates["conservative_pricing_ceiling_usd_per_million_input_tokens"])
        + retained_output
        * float(rates["conservative_pricing_ceiling_usd_per_million_output_tokens"])
    ) / 1_000_000
    body: dict[str, Any] = {
        "schema_version": FAILURE_SCHEMA,
        "goal": "G4_CREATIVE_GRAMMAR_GENERATION",
        "decision": "BLOCK_G4_LLM_CREATIVITY_CAMPAIGN_ENGINEERING_FAILURE",
        "claims": {
            "alternative_to_gr_discovered": False,
            "confirmation_galaxy_evaluated": False,
            "empirical_validation_completed": False,
            "historical_novelty_established": False,
            "usable_llm_proposal_generated": False,
        },
        "config": {"content_sha256": canonical_sha256(config), "path": CONFIG_PATH},
        "counts": {
            "completed_inference_calls": 4,
            "confirmation_evaluator_accesses": 0,
            "provider_http_requests": 10,
            "retained_failed_responses": 2,
            "schema_rejections_before_inference": 6,
            "usable_proposals": 0,
        },
        "cost": {
            "campaign_cost_ceiling_usd": rates["maximum_campaign_usd"],
            "first_two_inference_usage_available": False,
            "retained_calls_conservative_usage_cost_ceiling_usd": _metric(
                retained_cost_ceiling
            ),
            "retained_input_tokens": retained_input,
            "retained_output_tokens": retained_output,
        },
        "failure_stages": [
            {
                "completed_inference": False,
                "count": 6,
                "failure": "provider structured-output schema rejection",
                "retained_response": False,
            },
            {
                "completed_inference": True,
                "count": 1,
                "failure": "proposal IDs violated the local regex",
                "retained_response": False,
            },
            {
                "completed_inference": True,
                "count": 1,
                "failure": "proposal count violated the exact-count contract",
                "retained_response": False,
            },
            {
                "completed_inference": True,
                "count": 1,
                "failure": "all twelve structured string slots were empty",
                "retained_response": True,
            },
            {
                "completed_inference": True,
                "count": 1,
                "failure": "unstructured response exhausted max tokens before its first proposal",
                "retained_response": True,
            },
        ],
        "limitations": [
            "This is an orchestration/provider-output failure, not evidence against any gravity family.",
            "The first two completed inference outputs and their token usage were not persisted.",
            "No proposal may be inferred from provider thinking blocks, signatures, or truncated text.",
            "The LLM saw aggregate exploration failures, but no target rows, galaxy identities, or confirmation data.",
        ],
        "source_bindings": {
            "config": _binding(root, CONFIG_PATH),
            "first_retained_response": _binding(
                root,
                str(
                    config["attempt_audit_before_final_retry"][
                        "retained_failed_response"
                    ]["path"]
                ),
            ),
            "final_retained_response": _binding(
                root, str(config["final_failed_response"]["path"])
            ),
            "source": _binding(root, SOURCE_PATH),
            "test": _binding(root, TEST_PATH),
        },
    }
    body["content_sha256"] = canonical_sha256(body)
    return body


def validate_failure_receipt(receipt: Mapping[str, Any], *, root: Path) -> None:
    """Fail closed on any mutation or overstatement in the campaign failure receipt."""

    root = root.resolve()
    if receipt.get("schema_version") != FAILURE_SCHEMA:
        raise GravityG4LlmCreativityError("LLM failure receipt schema changed")
    body = dict(receipt)
    supplied = body.pop("content_sha256", None)
    if supplied != canonical_sha256(body):
        raise GravityG4LlmCreativityError("LLM failure receipt seal changed")
    config = load_config(root)
    if receipt.get("config", {}).get("content_sha256") != canonical_sha256(config):
        raise GravityG4LlmCreativityError("LLM failure config binding changed")
    expected = {
        "config": CONFIG_PATH,
        "first_retained_response": config["attempt_audit_before_final_retry"][
            "retained_failed_response"
        ]["path"],
        "final_retained_response": config["final_failed_response"]["path"],
        "source": SOURCE_PATH,
        "test": TEST_PATH,
    }
    for key, path in expected.items():
        if receipt.get("source_bindings", {}).get(key) != _binding(root, str(path)):
            raise GravityG4LlmCreativityError(f"LLM failure {key} binding changed")
    if receipt.get("decision") != "BLOCK_G4_LLM_CREATIVITY_CAMPAIGN_ENGINEERING_FAILURE":
        raise GravityG4LlmCreativityError("LLM failure decision changed")
    if receipt.get("counts", {}).get("usable_proposals") != 0:
        raise GravityG4LlmCreativityError("LLM failure invents usable proposals")
    claims = receipt.get("claims", {})
    if any(
        claims.get(key) is not False
        for key in (
            "alternative_to_gr_discovered",
            "confirmation_galaxy_evaluated",
            "empirical_validation_completed",
            "historical_novelty_established",
            "usable_llm_proposal_generated",
        )
    ):
        raise GravityG4LlmCreativityError("LLM failure receipt overstates result")


def _load_secret(env_file: Path | None, env_var: str) -> str:
    value = os.environ.get(env_var)
    if value and value.strip():
        return value.strip()
    if env_file is not None:
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.startswith(f"{env_var}="):
                value = line.split("=", 1)[1].strip().strip('"').strip("'")
                if value:
                    return value
    raise GravityG4LlmCreativityError("referenced Anthropic secret is absent")


def _write_immutable(path: Path, value: Mapping[str, Any]) -> None:
    payload = canonical_json_bytes(value)
    if path.exists():
        if path.read_bytes() != payload:
            raise GravityG4LlmCreativityError(
                f"refusing to overwrite immutable LLM creativity receipt: {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--seal-failure", action="store_true")
    parser.add_argument("--validate-failure", action="store_true")
    parser.add_argument("--validate-checked", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    path = root / OUTPUT_PATH
    if args.validate_checked:
        validate_receipt(_load_json(path), root=root)
        return 0
    failure_path = root / FAILURE_OUTPUT_PATH
    if args.validate_failure:
        validate_failure_receipt(_load_json(failure_path), root=root)
        return 0
    if args.seal_failure:
        receipt = build_failure_receipt(root)
        _write_immutable(failure_path, receipt)
        print(
            json.dumps(
                {
                    "content_sha256": receipt["content_sha256"],
                    "decision": receipt["decision"],
                    "counts": receipt["counts"],
                    "cost": receipt["cost"],
                },
                sort_keys=True,
            )
        )
        return 0
    if path.exists():
        receipt = _load_json(path)
        validate_receipt(receipt, root=root)
        print(json.dumps({"content_sha256": receipt["content_sha256"], "replayed": True}))
        return 0
    if not args.live:
        raise GravityG4LlmCreativityError("live provider call requires --live")
    config = load_config(root)
    if config.get("campaign_closed") is True:
        raise GravityG4LlmCreativityError("LLM creativity campaign is closed")
    secret = _load_secret(args.env_file, str(config["provider"]["api_key_env_var"]))
    receipt = build_receipt(
        root,
        provider=anthropic_provider,
        secret=secret,
        raw_response_path=root / RAW_RESPONSE_PATH,
    )
    _write_immutable(path, receipt)
    print(
        json.dumps(
            {
                "content_sha256": receipt["content_sha256"],
                "decision": receipt["decision"],
                "origin_label_counts": receipt["origin_label_counts"],
                "provider": receipt["provider"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FAILURE_OUTPUT_PATH",
    "RAW_RESPONSE_PATH",
    "GravityG4LlmCreativityError",
    "build_failure_receipt",
    "build_prompt",
    "build_provider_request",
    "build_receipt",
    "canonicalize_proposal_ids",
    "load_config",
    "output_schema",
    "validate_failure_receipt",
    "validate_proposals",
    "validate_receipt",
]
