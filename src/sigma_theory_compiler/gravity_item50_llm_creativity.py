"""Item 50 blind multi-model creativity campaign and executable gravity test."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np

from sigma_theory_compiler.claude_creativity_api import (
    ANTHROPIC_VERSION,
    MESSAGES_ENDPOINT,
    MODELS_ENDPOINT,
    urllib_transport,
)
from sigma_theory_compiler.core_credential import activated_credential
from sigma_theory_compiler.gravity_item22_polarization_superposition import (
    _canonical_bytes,
    _content_hashed,
    _read_json,
    _sha256_bytes,
    _sha256_file,
    _write_json,
)
from sigma_theory_compiler.gravity_item49_pseudorandom_exploration import (
    _admissible_parameter_table,
    _behavioral_representatives,
    _contract_digest as _item49_contract_digest,
    _primitive_sources,
    _response_blind_u,
    _select_rows,
    load_config as _load_item49_config,
    primitive_bank_from_sources,
    primitive_labels,
)
from sigma_theory_compiler.pseudorandom_ordinal import PseudorandomOrdinalPermutation


CONFIG_PATH = Path("configs/gravity_item50_llm_creativity_v1.json")
GOAL_PATH = Path("docs/GRAVITY_HIDDEN_VARIABLE_AND_THEORY_SEARCH_GOALS.md")
ITEM49_RESULT_PATH = Path("runs/gravity/roadmap/item-49-pseudorandom-exploration-v1.json")
ITEM49_PRIMITIVE_PATH = Path(
    "runs/gravity/roadmap/item-49-pseudorandom-exploration-v1-source/primitive-bank-receipt.json"
)
JOURNAL_NAME = "provider-attempt-journal.json"
CRITIC_JOURNAL_NAME = "critic-attempt-journal.json"


class GravityItem50Error(RuntimeError):
    """Raised when an Item 50 freeze, provider, compiler, or evidence gate fails."""


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def load_config(root: Path) -> dict[str, Any]:
    config = _read_json(root / CONFIG_PATH)
    validate_config(root, config)
    return config


def validate_config(root: Path, config: Mapping[str, Any]) -> None:
    if (
        config.get("schema_version")
        != "invariant-gravity-item50-llm-creativity-config-1.0"
        or int(config.get("item", -1)) != 50
    ):
        raise GravityItem50Error("unexpected Item 50 config")
    if _sha256_file(root / GOAL_PATH) != str(config["stable_goal_sha256"]):
        raise GravityItem50Error("stable gravity goal changed")
    if re.fullmatch(r"[0-9a-f]{40}", str(config["scientific_freeze_commit"])) is None:
        raise GravityItem50Error("Item 50 scientific freeze is not bound")
    for relative, expected in config["scientific_dependencies"].items():
        if _sha256_file(root / str(relative)) != str(expected):
            raise GravityItem50Error(f"scientific dependency changed: {relative}")
    predecessor = _read_json(root / ITEM49_RESULT_PATH)
    required = config["required_predecessor"]
    if (
        predecessor.get("content_sha256") != required["content_sha256"]
        or predecessor.get("decision") != required["decision"]
        or int(predecessor["selected_pseudorandom_program"]["ordinal"])
        != int(required["selected_ordinal"])
    ):
        raise GravityItem50Error("Item 49 predecessor binding changed")
    provider = config["provider"]
    generation = config["ensemble"]["generation_calls"]
    critics = config["ensemble"]["critic_calls"]
    if len(generation) != 6 or len(critics) != 3:
        raise GravityItem50Error("Item 50 call schedule changed")
    if len(generation) + len(critics) != int(provider["maximum_calls"]):
        raise GravityItem50Error("provider call cap and schedule disagree")
    if int(provider["maximum_total_proposals"]) != len(generation) * int(
        provider["proposals_per_generation_call"]
    ):
        raise GravityItem50Error("proposal budget changed")
    models = {row["model"] for row in generation + critics}
    if models != {"claude-opus-5", "claude-sonnet-5", "claude-opus-4-8"}:
        raise GravityItem50Error("declared model ensemble changed")
    for critic in critics:
        if critic["model"] in set(critic["reviews_models"]):
            raise GravityItem50Error("a critic reviews its own model")
    maximum_input_tokens = (
        int(provider["maximum_prompt_bytes_per_call"])
        * int(provider["maximum_calls"])
    )
    maximum_output_tokens = (
        int(provider["maximum_output_tokens_per_call"])
        * int(provider["maximum_calls"])
    )
    conservative = (
        Decimal(maximum_input_tokens)
        * Decimal(provider["conservative_input_usd_per_million_tokens"])
        + Decimal(maximum_output_tokens)
        * Decimal(provider["conservative_output_usd_per_million_tokens"])
    ) / Decimal(1_000_000)
    if conservative > Decimal(provider["conservative_maximum_campaign_usd"]):
        raise GravityItem50Error("conservative provider budget is not closed")
    if Decimal(provider["conservative_maximum_campaign_usd"]) > Decimal(
        provider["user_authorized_maximum_usd"]
    ):
        raise GravityItem50Error("provider budget exceeds user authorization")
    proposal = config["proposal_contract"]
    if proposal["origin_labels"] != [
        "known_formula",
        "algebraic_rewrite",
        "known_family_combination",
        "potentially_new_synthesis",
        "uncertain",
    ]:
        raise GravityItem50Error("origin labels changed")
    policy = config["discovery_policy"]
    if (
        not policy["single_empirical_counterexample_is_not_a_formula_or_family_veto"]
        or not policy["counterexample_count_alone_is_never_decisive"]
        or policy["finite_empirical_sample_may_prune_family"]
        or not policy["all_provider_proposals_retained_with_lineage_even_if_nonexecutable"]
    ):
        raise GravityItem50Error("non-pruning discovery policy changed")
    if config["scope"]["proposal_prompt_reads_empirical_response_values"]:
        raise GravityItem50Error("provider proposal prompt opened response values")
    if config["scope"]["critic_prompt_reads_empirical_response_values"]:
        raise GravityItem50Error("provider critic prompt opened response values")
    if config["scope"]["historical_novelty_may_be_claimed"]:
        raise GravityItem50Error("provider campaign opened a novelty gate")
    if config["data_boundary"]["sealed_rows_may_be_read"]:
        raise GravityItem50Error("provider campaign opened sealed rows")
    if int(config["candidate_expansion"]["structural_space"]) != 1_585_561_600:
        raise GravityItem50Error("matched structural space changed")
    config49 = _load_item49_config(root)
    if _item49_contract_digest(config49) != str(
        predecessor["source_bindings"]["config"].get("contract_sha256", "")
    ):
        # Older Item 49 aggregate binds the file rather than exposing its contract digest.
        if _sha256_file(root / "configs/gravity_item49_pseudorandom_exploration_v1.json") != str(
            predecessor["source_bindings"]["config"]["sha256"]
        ):
            raise GravityItem50Error("Item 49 grammar contract changed")


def _contract_digest(config: Mapping[str, Any]) -> str:
    value = json.loads(json.dumps(config))
    value["scientific_freeze_commit"] = "<BOUND_COMMIT>"
    value["status"] = "<BOUND_STATUS>"
    return _sha256_bytes(_canonical_bytes(value))


def _source_path(root: Path, config: Mapping[str, Any], key: str) -> Path:
    return root / str(config["paths"]["source_dir"]) / str(config["paths"][key])


def _journal_path(root: Path, config: Mapping[str, Any], name: str) -> Path:
    return root / str(config["paths"]["source_dir"]) / name


def _catalog(root: Path) -> list[dict[str, Any]]:
    receipt = _read_json(root / ITEM49_PRIMITIVE_PATH)
    rows = receipt["primitives"]
    if len(rows) != 440 or [int(row["primitive_id"]) for row in rows] != list(range(440)):
        raise GravityItem50Error("Item 50 primitive catalog changed")
    return rows


def _catalog_for_prompt(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": int(row["primitive_id"]),
            "item": int(row["source_item"]),
            "mechanism": str(row["mechanism"]),
            "expression": str(row["expression"]),
        }
        for row in rows
    ]


def _proposal_schema(config: Mapping[str, Any]) -> dict[str, Any]:
    contract = config["proposal_contract"]
    proposal = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "proposal_id": {"type": "string"},
            "title": {"type": "string"},
            "origin_self_assessment": {
                "type": "string",
                "enum": contract["origin_labels"],
            },
            "known_analogues": {
                "type": "array",
                "items": {"type": "string"}
            },
            "source_domains": {
                "type": "array",
                "items": {"type": "string"}
            },
            "mechanism": {"type": "string"},
            "left_primitive_id": {"type": "integer"},
            "left_transform": {"type": "string", "enum": contract["unary_transforms"]},
            "right_primitive_id": {"type": "integer"},
            "right_transform": {"type": "string", "enum": contract["unary_transforms"]},
            "binary_operator": {"type": "string", "enum": contract["binary_operators"]},
            "mixing": {"type": "number", "enum": contract["mixing_grid"]},
            "suggested_amplitude": {
                "type": "number",
                "enum": [0.3, 0.6, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
            },
            "suggested_acceleration_exponent": {
                "type": "number",
                "enum": [0.2, 0.25, 0.3, 0.35, 0.4],
            },
            "suggested_transition_u": {
                "type": "number",
                "enum": contract["transition_u_grid"],
            },
            "why_not_merely_a_rewrite": {"type": "string"},
            "expected_observational_signature": {"type": "string"},
            "cheapest_falsifier": {"type": "string"},
            "likely_failure_mode": {"type": "string"},
        },
        "required": contract["required_fields"],
    }
    slots = int(config["provider"]["proposals_per_generation_call"])
    names = [f"idea_{index:02d}" for index in range(1, slots + 1)]
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "proposals": {
                "type": "object",
                "additionalProperties": False,
                "properties": {name: proposal for name in names},
                "required": names,
            }
        },
        "required": ["proposals"],
    }


def _critic_schema(
    proposal_ids: Sequence[str], config: Mapping[str, Any]
) -> dict[str, Any]:
    assessment = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "lineage_reclassification": {
                "type": "string",
                "enum": config["proposal_contract"]["origin_labels"],
            },
            "nearest_known_analogue": {"type": "string"},
            "dimensional_consistency": {
                "type": "string",
                "enum": ["consistent", "repairable", "inconsistent", "uncertain"],
            },
            "independent_physical_concern": {"type": "string"},
            "suggested_repair": {"type": "string"},
            "retain_for_empirical_test": {"type": "boolean"},
            "confidence": {"type": "integer"},
        },
        "required": [
            "lineage_reclassification",
            "nearest_known_analogue",
            "dimensional_consistency",
            "independent_physical_concern",
            "suggested_repair",
            "retain_for_empirical_test",
            "confidence",
        ],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "assessments": {
                "type": "object",
                "additionalProperties": False,
                "properties": {proposal_id: assessment for proposal_id in proposal_ids},
                "required": list(proposal_ids),
            }
        },
        "required": ["assessments"],
    }


def _generation_prompt(
    root: Path, config: Mapping[str, Any], call: Mapping[str, Any]
) -> str:
    role = call["role"]
    role_instruction = (
        "Construct physically motivated mechanisms from conservation, geometry, fields, or causal dynamics."
        if role == "mechanism_synthesizer"
        else "Search distant analogies in optics, condensed matter, control theory, networks, memory, resonance, and effective field theory, then make them dimensionally testable."
    )
    payload = {
        "objective": (
            "Propose one universal baryons-plus-geometry gravity response structure that could be tested on both galaxy dynamics and cluster lensing. This is blind formula generation: no observed outcomes, residuals, scores, winners, object identities, or confirmation data are supplied."
        ),
        "role_instruction": role_instruction,
        "formula_semantics": (
            "Each primitive is a response-blind coordinate H in (0,1). The compiler maps x=2H-1, applies each selected unary transform, combines them with the selected binary operator and mixing coefficient, bounds z as H_program=0.5+0.5*z/(1+abs(z)), and expands the structure over every pre-admitted outer law nu=1+A*u^(-p)/(1+u/u_t)*(0.05+0.95*H_program)."
        ),
        "lineage_labels": {
            "known_formula": "a recognizable published formula or direct named instance",
            "algebraic_rewrite": "the same known relationship under algebraic or variable rewriting",
            "known_family_combination": "a straightforward combination of known mechanisms",
            "potentially_new_synthesis": "a specific cross-domain synthesis you cannot recognize as a standard instance; this is not a historical novelty claim",
            "uncertain": "lineage is unclear; preserve the uncertainty",
        },
        "requirements": [
            "Return exactly eight structurally distinct executable recipes.",
            "Name known analogues conservatively and explain whether the recipe is more than a rewrite.",
            "Use only primitive IDs and grammar values in the supplied catalog.",
            "Choose a suggested outer triplet satisfying the frozen outer_parameter_rule.",
            "Prefer distinct mechanisms over cosmetic parameter changes.",
            "State a falsifier and likely failure mode; neither is grounds to delete the idea.",
            "Do not claim proof, empirical support, or historical novelty.",
        ],
        "outer_parameter_rule": config["proposal_contract"]["outer_parameter_rule"],
        "primitive_catalog": _catalog_for_prompt(_catalog(root)),
    }
    prompt = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    if len(prompt.encode("utf-8")) > int(
        config["provider"]["maximum_prompt_bytes_per_call"]
    ):
        raise GravityItem50Error("generation prompt exceeds frozen byte cap")
    forbidden = ("log10_observed_quantity", "object_losses", "selected_ordinal")
    if any(token in prompt for token in forbidden):
        raise GravityItem50Error("generation prompt contains forbidden response material")
    return prompt


def _critic_prompt(
    proposals: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> str:
    summaries = [
        {
            key: proposal[key]
            for key in (
                "proposal_id",
                "title",
                "origin_self_assessment",
                "known_analogues",
                "source_domains",
                "mechanism",
                "left_primitive_id",
                "left_transform",
                "right_primitive_id",
                "right_transform",
                "binary_operator",
                "mixing",
                "suggested_amplitude",
                "suggested_acceleration_exponent",
                "suggested_transition_u",
                "why_not_merely_a_rewrite",
                "expected_observational_signature",
                "cheapest_falsifier",
                "likely_failure_mode",
            )
        }
        for proposal in proposals
    ]
    payload = {
        "objective": (
            "Independently audit these blind gravity structures for lineage, dimensional coherence, and physical weaknesses. You have no observed outcomes, losses, object identities, or confirmation data. Your advice is archived but never vetoes an executable proposal."
        ),
        "origin_labels": config["proposal_contract"]["origin_labels"],
        "proposals": summaries,
    }
    prompt = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    if len(prompt.encode("utf-8")) > int(
        config["provider"]["maximum_prompt_bytes_per_call"]
    ):
        raise GravityItem50Error("critic prompt exceeds frozen byte cap")
    return prompt


def _model_evidence(model: str, credential: str, timeout: float) -> dict[str, Any]:
    import urllib.parse

    status, payload = urllib_transport(
        "GET",
        f"{MODELS_ENDPOINT}/{urllib.parse.quote(model, safe='')}",
        {"anthropic-version": ANTHROPIC_VERSION, "x-api-key": credential},
        None,
        timeout,
    )
    capabilities = payload.get("capabilities", {})
    structured = (
        capabilities.get("structured_outputs", {})
        if isinstance(capabilities, Mapping)
        else {}
    )
    if (
        status != 200
        or payload.get("id") != model
        or not isinstance(structured, Mapping)
        or structured.get("supported") is not True
    ):
        raise GravityItem50Error(f"model lacks structured-output evidence: {model}")
    return {
        "model": model,
        "structured_outputs_supported": True,
        "capabilities_sha256": _sha256_bytes(_canonical_bytes(capabilities)),
    }


def _provider_call(
    *,
    credential: str,
    model: str,
    prompt: str,
    schema: Mapping[str, Any],
    config: Mapping[str, Any],
    model_evidence: Mapping[str, Any],
    system: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    body = {
        "max_tokens": int(config["provider"]["maximum_output_tokens_per_call"]),
        "messages": [{"content": prompt, "role": "user"}],
        "model": model,
        "output_config": {
            "effort": config["provider"]["effort"],
            "format": {"schema": schema, "type": "json_schema"},
        },
        "system": system,
    }
    request_bytes = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    status, response = urllib_transport(
        "POST",
        MESSAGES_ENDPOINT,
        {
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
            "x-api-key": credential,
        },
        request_bytes,
        float(config["provider"]["timeout_seconds"]),
    )
    if status != 200 or response.get("type") != "message":
        raise GravityItem50Error("provider returned an invalid message envelope")
    if response.get("stop_reason") != "end_turn" or response.get("role") != "assistant":
        raise GravityItem50Error("provider response did not complete normally")
    content = response.get("content")
    if (
        not isinstance(content, list)
        or len(content) != 1
        or not isinstance(content[0], Mapping)
        or content[0].get("type") != "text"
        or not isinstance(content[0].get("text"), str)
    ):
        raise GravityItem50Error("provider structured content changed")
    try:
        output = json.loads(content[0]["text"])
    except json.JSONDecodeError as error:
        raise GravityItem50Error("provider structured text is not JSON") from error
    usage = response.get("usage", {})
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    if (
        isinstance(input_tokens, bool)
        or isinstance(output_tokens, bool)
        or not isinstance(input_tokens, int)
        or not isinstance(output_tokens, int)
    ):
        raise GravityItem50Error("provider usage evidence changed")
    prices = config["provider"]["standard_prices_usd_per_million_tokens"][model]
    estimated_cost = (
        Decimal(input_tokens) * Decimal(prices["input"])
        + Decimal(output_tokens) * Decimal(prices["output"])
    ) / Decimal(1_000_000)
    evidence = {
        "api_response_id": str(response.get("id")),
        "model": model,
        "model_evidence": dict(model_evidence),
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "prompt_bytes": len(prompt.encode()),
        "request_schema_sha256": _sha256_bytes(_canonical_bytes(schema)),
        "raw_output_sha256": hashlib.sha256(content[0]["text"].encode()).hexdigest(),
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        "estimated_standard_cost_usd": f"{estimated_cost:.6f}",
        "credential_persisted": False,
        "header_names": ["anthropic-version", "content-type", "x-api-key"],
        "network_calls": 1,
    }
    return output, evidence


def _text(value: Any, label: str, maximum: int = 4096) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GravityItem50Error(f"{label} is empty")
    result = value.strip()
    if len(result.encode()) > maximum:
        result = result.encode()[:maximum].decode(errors="ignore").rstrip()
    return result


def _text_list(value: Any, label: str, maximum: int = 4) -> list[str]:
    if not isinstance(value, list):
        raise GravityItem50Error(f"{label} is not an array")
    result: list[str] = []
    for item in value:
        text = _text(item, label, 512)
        if text not in result:
            result.append(text)
    return result[:maximum]


def _outer_indices(proposal: Mapping[str, Any], config49: Mapping[str, Any]) -> tuple[int, int, int]:
    grids = config49["program_grammar"]["outer_parameter_grids"]
    try:
        a = list(grids["amplitude"]).index(float(proposal["suggested_amplitude"]))
        p = list(grids["acceleration_exponent"]).index(
            float(proposal["suggested_acceleration_exponent"])
        )
        ut = list(grids["transition_u"]).index(float(proposal["suggested_transition_u"]))
    except ValueError as error:
        raise GravityItem50Error("proposal outer value is outside the grammar") from error
    return a, p, ut


def _normalize_proposal(
    raw: Mapping[str, Any],
    *,
    call: Mapping[str, Any],
    slot: int,
    config: Mapping[str, Any],
    config49: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or set(raw) != set(
        config["proposal_contract"]["required_fields"]
    ):
        raise GravityItem50Error("provider proposal fields changed")
    origin = raw["origin_self_assessment"]
    if origin not in config["proposal_contract"]["origin_labels"]:
        raise GravityItem50Error("provider origin label changed")
    transforms = config["proposal_contract"]["unary_transforms"]
    operators = config["proposal_contract"]["binary_operators"]
    left = int(raw["left_primitive_id"])
    right = int(raw["right_primitive_id"])
    if not 0 <= left < 440 or not 0 <= right < 440:
        raise GravityItem50Error("provider primitive ID is invalid")
    if raw["left_transform"] not in transforms or raw["right_transform"] not in transforms:
        raise GravityItem50Error("provider transform is invalid")
    if raw["binary_operator"] not in operators:
        raise GravityItem50Error("provider operator is invalid")
    if float(raw["mixing"]) not in config["proposal_contract"]["mixing_grid"]:
        raise GravityItem50Error("provider mixing value is invalid")
    a, p, ut = _outer_indices(raw, config49)
    parameter_index = a * 256 + p * 16 + ut
    suggested_outer_admitted = bool(_admissible_parameter_table(config49)[parameter_index])
    provider_id = _text(raw["proposal_id"], "provider proposal ID", 256)
    return {
        "proposal_id": f"item50-{call['call_id']}-{slot:02d}",
        "provider_proposal_id": provider_id,
        "provider_call_id": call["call_id"],
        "provider_model": call["model"],
        "provider_role": call["role"],
        "title": _text(raw["title"], "proposal title"),
        "origin_self_assessment": origin,
        "known_analogues": _text_list(raw["known_analogues"], "known analogues"),
        "source_domains": _text_list(raw["source_domains"], "source domains"),
        "mechanism": _text(raw["mechanism"], "proposal mechanism"),
        "left_primitive_id": left,
        "left_transform": raw["left_transform"],
        "right_primitive_id": right,
        "right_transform": raw["right_transform"],
        "binary_operator": raw["binary_operator"],
        "mixing": float(raw["mixing"]),
        "suggested_amplitude": float(raw["suggested_amplitude"]),
        "suggested_acceleration_exponent": float(
            raw["suggested_acceleration_exponent"]
        ),
        "suggested_transition_u": float(raw["suggested_transition_u"]),
        "suggested_outer_cell_physically_admitted": suggested_outer_admitted,
        "why_not_merely_a_rewrite": _text(
            raw["why_not_merely_a_rewrite"], "rewrite explanation"
        ),
        "expected_observational_signature": _text(
            raw["expected_observational_signature"], "observational signature"
        ),
        "cheapest_falsifier": _text(raw["cheapest_falsifier"], "falsifier"),
        "likely_failure_mode": _text(raw["likely_failure_mode"], "failure mode"),
        "historical_novelty_claimed": False,
        "retained_regardless_of_origin_or_critic_label": True,
    }


def _normalize_generation_output(
    output: Mapping[str, Any],
    call: Mapping[str, Any],
    config: Mapping[str, Any],
    config49: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if not isinstance(output, Mapping) or set(output) != {"proposals"}:
        raise GravityItem50Error("generation output envelope changed")
    proposals = output["proposals"]
    slots = int(config["provider"]["proposals_per_generation_call"])
    expected = {f"idea_{index:02d}" for index in range(1, slots + 1)}
    if not isinstance(proposals, Mapping) or set(proposals) != expected:
        raise GravityItem50Error("generation output slots changed")
    return [
        _normalize_proposal(
            proposals[name],
            call=call,
            slot=index,
            config=config,
            config49=config49,
        )
        for index, name in enumerate(sorted(expected), 1)
    ]


def build_preflight_manifest(root: Path, *, live: bool = False) -> dict[str, Any]:
    config = load_config(root)
    prompts = {
        call["call_id"]: {
            "model": call["model"],
            "role": call["role"],
            "prompt_bytes": len(_generation_prompt(root, config, call).encode()),
            "prompt_sha256": hashlib.sha256(
                _generation_prompt(root, config, call).encode()
            ).hexdigest(),
        }
        for call in config["ensemble"]["generation_calls"]
    }
    model_evidence: dict[str, Any] = {}
    activation_evidence: dict[str, Any] | None = None
    network_calls = 0
    if live:
        with activated_credential(
            project_root=root, environment=dict(os.environ)
        ) as activation:
            credential = os.environ[config["provider"]["credential_env_var"]]
            activation_evidence = activation.to_evidence()
            for model in sorted(
                {row["model"] for row in config["ensemble"]["generation_calls"]}
            ):
                model_evidence[model] = _model_evidence(
                    model,
                    credential,
                    float(config["provider"]["timeout_seconds"]),
                )
                network_calls += 1
    conservative = Decimal(config["provider"]["conservative_maximum_campaign_usd"])
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item50-preflight-1.0",
            "item": 50,
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "config_contract_sha256": _contract_digest(config),
            "prompts": prompts,
            "proposal_schema_sha256": _sha256_bytes(
                _canonical_bytes(_proposal_schema(config))
            ),
            "model_evidence": model_evidence,
            "credential_activation": activation_evidence,
            "credential_value_recorded": False,
            "network_calls": network_calls,
            "paid_inference_calls": 0,
            "conservative_maximum_campaign_usd": f"{conservative:.6f}",
            "user_authorized_maximum_usd": config["provider"][
                "user_authorized_maximum_usd"
            ],
            "response_fields_in_provider_prompts": [],
            "sealed_rows_read": 0,
        }
    )


def write_preflight_manifest(root: Path, *, live: bool = True) -> Path:
    config = load_config(root)
    path = _source_path(root, config, "preflight_manifest")
    _write_json(path, build_preflight_manifest(root, live=live))
    return path


def _empty_journal(schema: str, config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": schema,
        "item": 50,
        "scientific_freeze_commit": config["scientific_freeze_commit"],
        "calls": [],
        "credential_material_persisted": False,
    }


def _load_journal(path: Path, schema: str, config: Mapping[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return _empty_journal(schema, config)
    journal = _read_json(path)
    if (
        journal.get("schema_version") != schema
        or journal.get("scientific_freeze_commit") != config["scientific_freeze_commit"]
        or journal.get("credential_material_persisted") is not False
        or not isinstance(journal.get("calls"), list)
    ):
        raise GravityItem50Error("provider attempt journal changed")
    return journal


def _preflight_model_evidence(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    receipt = _read_json(_source_path(root, config, "preflight_manifest"))
    evidence = receipt.get("model_evidence", {})
    required = {row["model"] for row in config["ensemble"]["generation_calls"]}
    if set(evidence) != required or not all(
        row.get("structured_outputs_supported") is True for row in evidence.values()
    ):
        raise GravityItem50Error("live model preflight is missing")
    return evidence


def _call_cost_total(calls: Sequence[Mapping[str, Any]]) -> Decimal:
    return sum(
        (Decimal(row["evidence"]["estimated_standard_cost_usd"]) for row in calls),
        Decimal("0"),
    )


def run_provider_proposals(root: Path) -> dict[str, Any]:
    config = load_config(root)
    config49 = _load_item49_config(root)
    model_evidence = _preflight_model_evidence(root, config)
    journal_path = _journal_path(root, config, JOURNAL_NAME)
    journal = _load_journal(
        journal_path, "invariant-gravity-item50-provider-journal-1.0", config
    )
    expected_calls = config["ensemble"]["generation_calls"]
    completed = {row["call_id"]: row for row in journal["calls"]}
    if not set(completed).issubset({row["call_id"] for row in expected_calls}):
        raise GravityItem50Error("proposal journal contains an undeclared call")
    with activated_credential(
        project_root=root, environment=dict(os.environ)
    ) as activation:
        credential = os.environ[config["provider"]["credential_env_var"]]
        for call in expected_calls:
            if call["call_id"] in completed:
                continue
            prompt = _generation_prompt(root, config, call)
            output, evidence = _provider_call(
                credential=credential,
                model=call["model"],
                prompt=prompt,
                schema=_proposal_schema(config),
                config=config,
                model_evidence=model_evidence[call["model"]],
                system=(
                    "You are one fallible idea generator in a blind scientific search. Return only the required structured object. Generate diverse, executable formula structures; disclose known analogues; distinguish known formulas, rewrites, combinations, potentially new syntheses, and uncertainty. These labels are lineage metadata, never proof of novelty or correctness."
                ),
            )
            proposals = _normalize_generation_output(
                output, call, config, config49
            )
            row = {
                "call_id": call["call_id"],
                "model": call["model"],
                "role": call["role"],
                "completed_utc": _now(),
                "evidence": evidence,
                "proposals": proposals,
                "credential_activation": activation.to_evidence(),
            }
            journal["calls"].append(row)
            completed[call["call_id"]] = row
            journal["last_updated_utc"] = _now()
            _write_json(journal_path, journal)
            if _call_cost_total(journal["calls"]) > Decimal(
                config["provider"]["conservative_maximum_campaign_usd"]
            ):
                raise GravityItem50Error("proposal calls exceeded campaign cost cap")
    ordered = [completed[row["call_id"]] for row in expected_calls]
    proposals = [proposal for call in ordered for proposal in call["proposals"]]
    if len(proposals) != int(config["provider"]["maximum_total_proposals"]):
        raise GravityItem50Error("provider proposal count changed")
    response_ids = [row["evidence"]["api_response_id"] for row in ordered]
    if len(set(response_ids)) != len(response_ids):
        raise GravityItem50Error("provider response IDs are not distinct")
    receipt = _content_hashed(
        {
            "schema_version": "invariant-gravity-item50-provider-proposals-1.0",
            "item": 50,
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "calls": ordered,
            "proposals": proposals,
            "counts": {
                "provider_calls": len(ordered),
                "proposals": len(proposals),
                "models": len(set(row["model"] for row in ordered)),
                "origin_labels": dict(
                    sorted(Counter(row["origin_self_assessment"] for row in proposals).items())
                ),
                "suggested_outer_cells_not_admitted": sum(
                    not row["suggested_outer_cell_physically_admitted"] for row in proposals
                ),
            },
            "usage": {
                "input_tokens": sum(
                    row["evidence"]["usage"]["input_tokens"] for row in ordered
                ),
                "output_tokens": sum(
                    row["evidence"]["usage"]["output_tokens"] for row in ordered
                ),
                "estimated_standard_cost_usd": f"{_call_cost_total(ordered):.6f}",
            },
            "claims": {
                "credential_material_persisted": False,
                "provider_labels_are_authoritative": False,
                "historical_novelty_established": False,
                "formula_correctness_established": False,
                "empirical_response_values_in_prompt": False,
                "sealed_rows_read": 0,
            },
        }
    )
    _write_json(_source_path(root, config, "proposal_receipt"), receipt)
    return receipt


def _normalize_critic_output(
    output: Mapping[str, Any], proposal_ids: Sequence[str], config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    if not isinstance(output, Mapping) or set(output) != {"assessments"}:
        raise GravityItem50Error("critic output envelope changed")
    assessments = output["assessments"]
    if not isinstance(assessments, Mapping) or set(assessments) != set(proposal_ids):
        raise GravityItem50Error("critic proposal coverage changed")
    result = []
    for proposal_id in proposal_ids:
        row = assessments[proposal_id]
        expected = {
            "lineage_reclassification",
            "nearest_known_analogue",
            "dimensional_consistency",
            "independent_physical_concern",
            "suggested_repair",
            "retain_for_empirical_test",
            "confidence",
        }
        if not isinstance(row, Mapping) or set(row) != expected:
            raise GravityItem50Error("critic assessment fields changed")
        if row["lineage_reclassification"] not in config["proposal_contract"][
            "origin_labels"
        ]:
            raise GravityItem50Error("critic lineage label changed")
        if row["dimensional_consistency"] not in {
            "consistent",
            "repairable",
            "inconsistent",
            "uncertain",
        }:
            raise GravityItem50Error("critic dimensional verdict changed")
        confidence = int(row["confidence"])
        if not 1 <= confidence <= 5:
            raise GravityItem50Error("critic confidence changed")
        result.append(
            {
                "proposal_id": proposal_id,
                "lineage_reclassification": row["lineage_reclassification"],
                "nearest_known_analogue": _text(
                    row["nearest_known_analogue"], "critic analogue"
                ),
                "dimensional_consistency": row["dimensional_consistency"],
                "independent_physical_concern": _text(
                    row["independent_physical_concern"], "critic concern"
                ),
                "suggested_repair": _text(row["suggested_repair"], "critic repair"),
                "retain_for_empirical_test": bool(row["retain_for_empirical_test"]),
                "confidence": confidence,
                "advisory_only": True,
                "proposal_pruned": False,
            }
        )
    return result


def run_provider_critiques(root: Path) -> dict[str, Any]:
    config = load_config(root)
    model_evidence = _preflight_model_evidence(root, config)
    proposal_receipt = _read_json(_source_path(root, config, "proposal_receipt"))
    proposals = proposal_receipt["proposals"]
    journal_path = _journal_path(root, config, CRITIC_JOURNAL_NAME)
    journal = _load_journal(
        journal_path, "invariant-gravity-item50-critic-journal-1.0", config
    )
    expected_calls = config["ensemble"]["critic_calls"]
    completed = {row["call_id"]: row for row in journal["calls"]}
    if not set(completed).issubset({row["call_id"] for row in expected_calls}):
        raise GravityItem50Error("critic journal contains an undeclared call")
    proposal_cost = Decimal(proposal_receipt["usage"]["estimated_standard_cost_usd"])
    with activated_credential(
        project_root=root, environment=dict(os.environ)
    ) as activation:
        credential = os.environ[config["provider"]["credential_env_var"]]
        for call in expected_calls:
            if call["call_id"] in completed:
                continue
            reviewed = [
                row for row in proposals if row["provider_model"] in call["reviews_models"]
            ]
            if len(reviewed) != 16 or any(
                row["provider_model"] == call["model"] for row in reviewed
            ):
                raise GravityItem50Error("critic independence allocation changed")
            proposal_ids = [row["proposal_id"] for row in reviewed]
            prompt = _critic_prompt(reviewed, config)
            output, evidence = _provider_call(
                credential=credential,
                model=call["model"],
                prompt=prompt,
                schema=_critic_schema(proposal_ids, config),
                config=config,
                model_evidence=model_evidence[call["model"]],
                system=(
                    "You are an independent, fallible scientific critic. Audit lineage and physics without observed outcomes. Your critique is advisory, cannot verify a formula, cannot establish novelty, and cannot delete an executable idea. Return only the required structured object."
                ),
            )
            assessments = _normalize_critic_output(
                output, proposal_ids, config
            )
            row = {
                "call_id": call["call_id"],
                "model": call["model"],
                "reviews_models": call["reviews_models"],
                "completed_utc": _now(),
                "evidence": evidence,
                "assessments": assessments,
                "credential_activation": activation.to_evidence(),
            }
            journal["calls"].append(row)
            completed[call["call_id"]] = row
            journal["last_updated_utc"] = _now()
            _write_json(journal_path, journal)
            if proposal_cost + _call_cost_total(journal["calls"]) > Decimal(
                config["provider"]["conservative_maximum_campaign_usd"]
            ):
                raise GravityItem50Error("critic calls exceeded campaign cost cap")
    ordered = [completed[row["call_id"]] for row in expected_calls]
    assessments = [item for call in ordered for item in call["assessments"]]
    if len(assessments) != len(proposals) or len(
        {row["proposal_id"] for row in assessments}
    ) != len(proposals):
        raise GravityItem50Error("critic did not cover every proposal exactly once")
    receipt = _content_hashed(
        {
            "schema_version": "invariant-gravity-item50-provider-critiques-1.0",
            "item": 50,
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "calls": ordered,
            "assessments": assessments,
            "counts": {
                "critic_calls": len(ordered),
                "assessments": len(assessments),
                "critic_retain_false_advisories": sum(
                    not row["retain_for_empirical_test"] for row in assessments
                ),
                "dimensional_consistency": dict(
                    sorted(Counter(row["dimensional_consistency"] for row in assessments).items())
                ),
                "lineage_reclassifications": dict(
                    sorted(
                        Counter(
                            row["lineage_reclassification"] for row in assessments
                        ).items()
                    )
                ),
            },
            "usage": {
                "input_tokens": sum(
                    row["evidence"]["usage"]["input_tokens"] for row in ordered
                ),
                "output_tokens": sum(
                    row["evidence"]["usage"]["output_tokens"] for row in ordered
                ),
                "estimated_standard_cost_usd": f"{_call_cost_total(ordered):.6f}",
                "campaign_including_proposals_estimated_standard_cost_usd": f"{proposal_cost + _call_cost_total(ordered):.6f}",
            },
            "claims": {
                "same_model_reviewed_own_proposal": False,
                "critic_is_verification_authority": False,
                "critic_advice_pruned_proposals": False,
                "historical_novelty_established": False,
                "empirical_response_values_in_prompt": False,
                "sealed_rows_read": 0,
            },
        }
    )
    _write_json(_source_path(root, config, "critic_receipt"), receipt)
    return receipt


def _structure_from_proposal(
    proposal: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, int]:
    contract = config["proposal_contract"]
    return {
        "left_primitive_index": int(proposal["left_primitive_id"]),
        "left_transform_index": contract["unary_transforms"].index(
            proposal["left_transform"]
        ),
        "right_primitive_index": int(proposal["right_primitive_id"]),
        "right_transform_index": contract["unary_transforms"].index(
            proposal["right_transform"]
        ),
        "operator_index": contract["binary_operators"].index(
            proposal["binary_operator"]
        ),
        "mixing_index": contract["mixing_grid"].index(float(proposal["mixing"])),
    }


def _control_structures(config: Mapping[str, Any]) -> list[dict[str, int]]:
    count = int(config["candidate_expansion"]["matched_control_raw_structures"])
    permutation = PseudorandomOrdinalPermutation(
        int(config["candidate_expansion"]["structural_space"]),
        str(config["candidate_expansion"]["matched_control_seed"]),
    )
    result = []
    radices = (16, 8, 8, 440, 8, 440)
    fields = (
        "mixing_index",
        "operator_index",
        "right_transform_index",
        "right_primitive_index",
        "left_transform_index",
        "left_primitive_index",
    )
    for position in range(count):
        value = int(permutation.at(position))
        row: dict[str, int] = {}
        for field, radix in zip(fields, radices, strict=True):
            value, digit = divmod(value, radix)
            row[field] = digit
        if value:
            raise GravityItem50Error("control structural decoder failed")
        result.append(row)
    return result


def _canonical_structure(
    structure: Mapping[str, int], config: Mapping[str, Any]
) -> tuple[Any, ...]:
    left = int(structure["left_primitive_index"]) * 8 + int(
        structure["left_transform_index"]
    )
    right = int(structure["right_primitive_index"]) * 8 + int(
        structure["right_transform_index"]
    )
    operator = int(structure["operator_index"])
    mixing_index = int(structure["mixing_index"])
    mixing = float(config["proposal_contract"]["mixing_grid"][mixing_index])
    if operator in (2, 7) or (operator in (0, 5, 6) and math.isclose(mixing, 1.0)):
        left, right = sorted((left, right))
    if left == right and math.isclose(mixing, 1.0) and operator in (1, 4):
        return ("zero",)
    if left == right and math.isclose(mixing, 1.0) and operator in (5, 6):
        return ("unary", left)
    return ("binary", operator, mixing_index, left, right)


def _unique_structures(
    structures: Sequence[Mapping[str, int]], config: Mapping[str, Any]
) -> tuple[list[dict[str, int]], list[int], dict[str, Any]]:
    seen: dict[tuple[Any, ...], int] = {}
    unique: list[dict[str, int]] = []
    source_to_unique: list[int] = []
    for row in structures:
        signature = _canonical_structure(row, config)
        if signature not in seen:
            seen[signature] = len(unique)
            unique.append(dict(row))
        source_to_unique.append(seen[signature])
    return unique, source_to_unique, {
        "raw_structures": len(structures),
        "symbolic_structure_classes": len(unique),
        "symbolic_structure_duplicates": len(structures) - len(unique),
        "structure_signature_sha256": _sha256_bytes(_canonical_bytes([list(key) for key in seen])),
    }


def _expand_structures(
    structures: Sequence[Mapping[str, int]], config49: Mapping[str, Any]
) -> dict[str, np.ndarray]:
    admitted = np.flatnonzero(_admissible_parameter_table(config49))
    if len(admitted) != 336:
        raise GravityItem50Error("admitted Item 49 outer grid changed")
    count = len(structures) * len(admitted)
    result: dict[str, np.ndarray] = {
        "candidate_id": np.arange(count, dtype=np.int64),
        "structure_index": np.repeat(np.arange(len(structures), dtype=np.int32), len(admitted)),
        "amplitude_index": np.tile((admitted // 256).astype(np.int16), len(structures)),
        "exponent_index": np.tile(((admitted // 16) % 16).astype(np.int16), len(structures)),
        "transition_index": np.tile((admitted % 16).astype(np.int16), len(structures)),
    }
    for field in (
        "left_primitive_index",
        "left_transform_index",
        "right_primitive_index",
        "right_transform_index",
        "operator_index",
        "mixing_index",
    ):
        result[field] = np.repeat(
            np.asarray([row[field] for row in structures], dtype=np.int16), len(admitted)
        )
    structural = (
        result["mixing_index"].astype(np.int64)
        + 16
        * (
            result["operator_index"].astype(np.int64)
            + 8
            * (
                result["right_transform_index"].astype(np.int64)
                + 8
                * (
                    result["right_primitive_index"].astype(np.int64)
                    + 440
                    * (
                        result["left_transform_index"].astype(np.int64)
                        + 8 * result["left_primitive_index"].astype(np.int64)
                    )
                )
            )
        )
    )
    result["ordinal"] = (
        result["transition_index"].astype(np.int64)
        + 16
        * (
            result["exponent_index"].astype(np.int64)
            + 16
            * (
                result["amplitude_index"].astype(np.int64)
                + 16 * structural
            )
        )
    ).astype(np.uint64)
    return result


def build_lane_candidates(
    root: Path, config: Mapping[str, Any], lane: str
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any], list[dict[str, int]], list[int]]:
    config49 = _load_item49_config(root)
    if lane == "llm_ensemble":
        receipt = _read_json(_source_path(root, config, "proposal_receipt"))
        raw_structures = [
            _structure_from_proposal(row, config) for row in receipt["proposals"]
        ]
    elif lane == "matched_seeded_random":
        raw_structures = _control_structures(config)
    else:
        raise GravityItem50Error(f"unknown Item 50 lane: {lane}")
    unique, source_to_unique, structure_audit = _unique_structures(
        raw_structures, config
    )
    expanded = _expand_structures(unique, config49)
    sources = _primitive_sources(root)
    bank, _bank_audit = primitive_bank_from_sources(sources)
    u = _response_blind_u(root, sources)
    programs, behavior, behavior_audit = _behavioral_representatives(
        expanded, bank, u, config49
    )
    audit = {
        "lane": lane,
        **structure_audit,
        "outer_cells_per_symbolic_structure": 336,
        "expanded_candidate_cells": len(expanded["candidate_id"]),
        **behavior_audit,
        "outcome_scoring_classes": len(programs["candidate_id"]),
        "behavior_candidate_ordinal_sha256": hashlib.sha256(
            np.asarray(programs["ordinal"], dtype="<u8").tobytes()
        ).hexdigest(),
    }
    return programs, behavior, audit, unique, source_to_unique


def build_candidate_manifest(root: Path) -> dict[str, Any]:
    config = load_config(root)
    proposal_receipt = _read_json(_source_path(root, config, "proposal_receipt"))
    critic_receipt = _read_json(_source_path(root, config, "critic_receipt"))
    lane_audits: dict[str, Any] = {}
    lineage: dict[str, Any] = {}
    for lane in ("llm_ensemble", "matched_seeded_random"):
        _programs, _behavior, audit, structures, source_to_unique = build_lane_candidates(
            root, config, lane
        )
        lane_audits[lane] = audit
        lineage[lane] = {
            "unique_structures": structures,
            "source_to_unique_structure": source_to_unique,
        }
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item50-candidate-manifest-1.0",
            "item": 50,
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "config_contract_sha256": _contract_digest(config),
            "provider_proposal_receipt_content_sha256": proposal_receipt["content_sha256"],
            "provider_critic_receipt_content_sha256": critic_receipt["content_sha256"],
            "response_fields_read_during_generation_compilation_or_equivalence": [],
            "response_values_used_during_generation_compilation_or_equivalence": 0,
            "critic_advice_pruned_proposals": False,
            "all_provider_proposals_retained": True,
            "lane_audits": lane_audits,
            "lineage": lineage,
            "counts": {
                "provider_proposals": len(proposal_receipt["proposals"]),
                "critic_assessments": len(critic_receipt["assessments"]),
                "raw_structures_per_lane": 48,
                "paid_model_calls": proposal_receipt["counts"]["provider_calls"]
                + critic_receipt["counts"]["critic_calls"],
                "sealed_rows_read": 0,
            },
            "claims": {
                "historical_novelty_established": False,
                "provider_is_verifier": False,
                "critic_is_verifier": False,
                "matched_control_is_equal_raw_structure_count": True,
                "fresh_confirmation_completed": False,
            },
        }
    )


def write_candidate_manifest(root: Path) -> Path:
    config = load_config(root)
    path = _source_path(root, config, "candidate_manifest")
    _write_json(path, build_candidate_manifest(root))
    return path
