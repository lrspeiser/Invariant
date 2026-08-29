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
from sigma_theory_compiler.gravity_counterexample_policy import (
    assess_counterexample_evidence,
    load_counterexample_policy,
)
from sigma_theory_compiler.gravity_item44_scale_hierarchy import _predict as _item44_predict
from sigma_theory_compiler.gravity_item45_universal_interactions import (
    _item44_oof,
    _ordinary_crossfit,
    _paired_p,
    _predict as _item45_predict,
    _score,
    _variant_arrays as _item45_variant_arrays,
    load_config as _load_item45_config,
)
from sigma_theory_compiler.gravity_item46_dimensionless_generator import (
    _physical_log_values as _item46_physical_log_values,
    _predict as _item46_predict,
    load_config as _load_item46_config,
    pi_vectors as _item46_pi_vectors,
)
from sigma_theory_compiler.gravity_item47_operator_generator import (
    _item45_oof,
    _item46_oof,
    _predict as _item47_predict,
    _shape_by_object,
    load_config as _load_item47_config,
    operator_bank_from_arrays as _item47_operator_bank_from_arrays,
)
from sigma_theory_compiler.gravity_item48_action_generator import (
    _evaluation_arrays as _item48_evaluation_arrays,
    action_bank_from_arrays as _item48_action_bank_from_arrays,
    load_config as _load_item48_config,
)
from sigma_theory_compiler.gravity_item49_pseudorandom_exploration import (
    _admissible_parameter_table,
    _best_behavior,
    _behavioral_representatives,
    _contract_digest as _item49_contract_digest,
    _fixed_behavior_oof,
    _item47_oof,
    _program_behavior,
    _primitive_bank_from_arrays,
    _primitive_sources,
    _response_blind_u,
    _select_rows,
    build_lane_programs as _build_item49_lane_programs,
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
ITEM49_EVALUATION_PATH = Path(
    "runs/gravity/roadmap/item-49-pseudorandom-exploration-v1-source/joint-evaluation-result.json"
)
POLICY_PATH = Path("configs/gravity_empirical_counterexample_policy_v1.json")
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
    if len(generation) + len(critics) != int(provider["maximum_successful_calls"]):
        raise GravityItem50Error("provider successful-call cap and schedule disagree")
    if int(provider["maximum_provider_attempts"]) != int(
        provider["maximum_successful_calls"]
    ) + int(provider["pre_durable_journal_attempt_audit"]["provider_attempts"]):
        raise GravityItem50Error("provider attempt audit and cap disagree")
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
        * int(provider["maximum_provider_attempts"])
    )
    maximum_output_tokens = (
        int(provider["maximum_output_tokens_per_call"])
        * int(provider["maximum_provider_attempts"])
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
    slots = int(config["provider"]["proposals_per_generation_call"])
    names = [f"idea_{index:02d}" for index in range(1, slots + 1)]
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "proposals": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    name: {
                        "type": "string",
                        "description": (
                            "One JSON-serialized proposal object containing exactly the frozen required fields."
                        ),
                    }
                    for name in names
                },
                "required": names,
            }
        },
        "required": ["proposals"],
    }


def _critic_schema(
    proposal_ids: Sequence[str], config: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "assessments": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    proposal_id: {
                        "type": "string",
                        "description": (
                            "One JSON-serialized independent assessment with exactly the frozen critic fields."
                        ),
                    }
                    for proposal_id in proposal_ids
                },
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
        "required_proposal_fields": config["proposal_contract"]["required_fields"],
        "proposal_field_types": {
            "proposal_id": "nonempty string",
            "title": "nonempty string",
            "origin_self_assessment": "one allowed lineage-label string",
            "known_analogues": "JSON array of one to four nonempty strings",
            "source_domains": "JSON array of one to four nonempty strings",
            "mechanism": "nonempty string",
            "left_primitive_id": "integer 0..439",
            "left_transform": "one allowed unary-transform string",
            "right_primitive_id": "integer 0..439",
            "right_transform": "one allowed unary-transform string",
            "binary_operator": "one allowed binary-operator string",
            "mixing": "one allowed numeric mixing-grid value",
            "suggested_amplitude": "one allowed numeric amplitude",
            "suggested_acceleration_exponent": "one allowed numeric exponent",
            "suggested_transition_u": "one allowed numeric transition value",
            "why_not_merely_a_rewrite": "nonempty string",
            "expected_observational_signature": "nonempty string",
            "cheapest_falsifier": "nonempty string",
            "likely_failure_mode": "nonempty string"
        },
        "allowed_unary_transforms": config["proposal_contract"]["unary_transforms"],
        "allowed_binary_operators": config["proposal_contract"]["binary_operators"],
        "allowed_mixing_grid": config["proposal_contract"]["mixing_grid"],
        "wire_format": (
            "Each idea_NN value must be a JSON-serialized object string, not prose, with exactly required_proposal_fields."
        ),
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
        "required_assessment_fields": [
            "lineage_reclassification",
            "nearest_known_analogue",
            "dimensional_consistency",
            "independent_physical_concern",
            "suggested_repair",
            "retain_for_empirical_test",
            "confidence"
        ],
        "assessment_field_types": {
            "lineage_reclassification": "one allowed lineage-label string",
            "nearest_known_analogue": "nonempty string",
            "dimensional_consistency": "one of consistent, repairable, inconsistent, uncertain",
            "independent_physical_concern": "nonempty string",
            "suggested_repair": "nonempty string; say no repair needed when appropriate",
            "retain_for_empirical_test": "JSON boolean",
            "confidence": "integer 1..5"
        },
        "wire_format": (
            "Each proposal-ID value must be a JSON-serialized assessment object string with exactly required_assessment_fields."
        ),
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
    if not isinstance(content, list) or any(
        not isinstance(block, Mapping) for block in content
    ):
        raise GravityItem50Error("provider structured content changed")
    text_blocks = [
        block
        for block in content
        if block.get("type") == "text" and isinstance(block.get("text"), str)
    ]
    if len(text_blocks) != 1:
        raise GravityItem50Error("provider did not return exactly one structured text block")
    text_block = text_blocks[0]
    try:
        output = json.loads(text_block["text"])
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
        "raw_output_sha256": hashlib.sha256(text_block["text"].encode()).hexdigest(),
        "provider_content_block_types": [str(block.get("type")) for block in content],
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
    local_compilation_issues: list[str] = []
    try:
        a, p, ut = _outer_indices(raw, config49)
        parameter_index = a * 256 + p * 16 + ut
        suggested_outer_admitted = bool(
            _admissible_parameter_table(config49)[parameter_index]
        )
        if not suggested_outer_admitted:
            local_compilation_issues.append(
                "suggested_outer_triplet_fails_frozen_physical_admission"
            )
    except GravityItem50Error:
        suggested_outer_admitted = False
        local_compilation_issues.append("suggested_outer_value_outside_frozen_grid")
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
        "structure_executable_for_frozen_outer_expansion": True,
        "local_compilation_issues": local_compilation_issues,
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
    def quarantine(slot: int, raw_value: Any, issue: str) -> dict[str, Any]:
        return {
            "proposal_id": f"item50-{call['call_id']}-{slot:02d}",
            "provider_proposal_id": f"unparsed-slot-{slot:02d}",
            "provider_call_id": call["call_id"],
            "provider_model": call["model"],
            "provider_role": call["role"],
            "title": "Quarantined non-executable provider slot",
            "origin_self_assessment": "uncertain",
            "known_analogues": [],
            "source_domains": [],
            "mechanism": "Provider slot retained without a compilable structured mechanism.",
            "left_primitive_id": None,
            "left_transform": None,
            "right_primitive_id": None,
            "right_transform": None,
            "binary_operator": None,
            "mixing": None,
            "suggested_amplitude": None,
            "suggested_acceleration_exponent": None,
            "suggested_transition_u": None,
            "suggested_outer_cell_physically_admitted": False,
            "structure_executable_for_frozen_outer_expansion": False,
            "local_compilation_issues": [issue],
            "why_not_merely_a_rewrite": "Unresolved because the provider slot did not compile.",
            "expected_observational_signature": "Unresolved pending an independently generated repair.",
            "cheapest_falsifier": "No empirical test is possible until the slot has a typed structure.",
            "likely_failure_mode": "Malformed provider serialization.",
            "historical_novelty_claimed": False,
            "retained_regardless_of_origin_or_critic_label": True,
            "raw_provider_slot": raw_value,
        }

    result: list[dict[str, Any]] = []
    for slot, name in enumerate(sorted(expected), 1):
        value = proposals[name]
        if not isinstance(value, str):
            result.append(quarantine(slot, value, "provider_slot_not_a_string"))
            continue
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            result.append(quarantine(slot, value, "provider_slot_not_json"))
            continue
        if not isinstance(parsed, Mapping):
            result.append(quarantine(slot, parsed, "provider_slot_not_an_object"))
            continue
        try:
            result.append(
                _normalize_proposal(
                    parsed,
                    call=call,
                    slot=slot,
                    config=config,
                    config49=config49,
                )
            )
        except GravityItem50Error as error:
            result.append(
                quarantine(slot, parsed, f"local_compilation_failed:{str(error)}")
            )
    return result


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
            row = completed.get(call["call_id"])
            if row is None:
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
                row = {
                    "call_id": call["call_id"],
                    "model": call["model"],
                    "role": call["role"],
                    "completed_utc": _now(),
                    "evidence": evidence,
                    "raw_provider_output": output,
                    "normalization_status": "provider_completed_pending_local_normalization",
                    "proposals": None,
                    "credential_activation": activation.to_evidence(),
                }
                journal["calls"].append(row)
                completed[call["call_id"]] = row
                journal["last_updated_utc"] = _now()
                _write_json(journal_path, journal)
            if row.get("normalization_status") != "normalized":
                row["proposals"] = _normalize_generation_output(
                    row["raw_provider_output"], call, config, config49
                )
                row["normalization_status"] = "normalized"
                journal["last_updated_utc"] = _now()
                _write_json(journal_path, journal)
            if _call_cost_total(journal["calls"]) > Decimal(
                config["provider"]["conservative_maximum_campaign_usd"]
            ):
                raise GravityItem50Error("proposal calls exceeded campaign cost cap")
    ordered = [completed[row["call_id"]] for row in expected_calls]
    if any(row.get("normalization_status") != "normalized" for row in ordered):
        raise GravityItem50Error("a provider call was not normalized")
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
                "provider_attempts_including_unretained_pre_journal_call": len(ordered)
                + int(
                    config["provider"]["pre_durable_journal_attempt_audit"][
                        "provider_attempts"
                    ]
                ),
                "proposals": len(proposals),
                "executable_structures": sum(
                    row["structure_executable_for_frozen_outer_expansion"]
                    for row in proposals
                ),
                "quarantined_nonexecutable_slots": sum(
                    not row["structure_executable_for_frozen_outer_expansion"]
                    for row in proposals
                ),
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
            "pre_durable_journal_attempt_audit": config["provider"][
                "pre_durable_journal_attempt_audit"
            ],
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
        raw_row = row
        parse_issue: str | None = None
        if not isinstance(row, str):
            parse_issue = "critic_slot_not_a_string"
        else:
            try:
                row = json.loads(row)
            except json.JSONDecodeError:
                parse_issue = "critic_slot_not_json"
        expected = {
            "lineage_reclassification",
            "nearest_known_analogue",
            "dimensional_consistency",
            "independent_physical_concern",
            "suggested_repair",
            "retain_for_empirical_test",
            "confidence",
        }
        if parse_issue is None and (
            not isinstance(row, Mapping) or set(row) != expected
        ):
            parse_issue = "critic_assessment_fields_changed"
        if parse_issue is None and row["lineage_reclassification"] not in config[
            "proposal_contract"
        ]["origin_labels"]:
            parse_issue = "critic_lineage_label_changed"
        if parse_issue is None and row["dimensional_consistency"] not in {
            "consistent",
            "repairable",
            "inconsistent",
            "uncertain",
        }:
            parse_issue = "critic_dimensional_verdict_changed"
        if parse_issue is None:
            try:
                confidence = int(row["confidence"])
            except (TypeError, ValueError):
                parse_issue = "critic_confidence_not_an_integer"
            else:
                if not 1 <= confidence <= 5:
                    parse_issue = "critic_confidence_outside_range"
        if parse_issue is not None:
            result.append(
                {
                    "proposal_id": proposal_id,
                    "lineage_reclassification": "uncertain",
                    "nearest_known_analogue": "No valid independent assessment returned.",
                    "dimensional_consistency": "uncertain",
                    "independent_physical_concern": "Critic slot did not compile; no adverse inference is authorized.",
                    "suggested_repair": "Retain the proposal and request a future independent critique without changing the empirical formula.",
                    "retain_for_empirical_test": True,
                    "confidence": 1,
                    "advisory_only": True,
                    "proposal_pruned": False,
                    "local_critic_issue": parse_issue,
                    "raw_provider_slot": raw_row,
                }
            )
            continue
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
                "local_critic_issue": None,
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
            reviewed = [
                row for row in proposals if row["provider_model"] in call["reviews_models"]
            ]
            if len(reviewed) != 16 or any(
                row["provider_model"] == call["model"] for row in reviewed
            ):
                raise GravityItem50Error("critic independence allocation changed")
            proposal_ids = [row["proposal_id"] for row in reviewed]
            row = completed.get(call["call_id"])
            if row is None:
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
                row = {
                    "call_id": call["call_id"],
                    "model": call["model"],
                    "reviews_models": call["reviews_models"],
                    "completed_utc": _now(),
                    "evidence": evidence,
                    "raw_provider_output": output,
                    "normalization_status": "provider_completed_pending_local_normalization",
                    "assessments": None,
                    "credential_activation": activation.to_evidence(),
                }
                journal["calls"].append(row)
                completed[call["call_id"]] = row
                journal["last_updated_utc"] = _now()
                _write_json(journal_path, journal)
            if row.get("normalization_status") != "normalized":
                row["assessments"] = _normalize_critic_output(
                    row["raw_provider_output"], proposal_ids, config
                )
                row["normalization_status"] = "normalized"
                journal["last_updated_utc"] = _now()
                _write_json(journal_path, journal)
            if proposal_cost + _call_cost_total(journal["calls"]) > Decimal(
                config["provider"]["conservative_maximum_campaign_usd"]
            ):
                raise GravityItem50Error("critic calls exceeded campaign cost cap")
    ordered = [completed[row["call_id"]] for row in expected_calls]
    if any(row.get("normalization_status") != "normalized" for row in ordered):
        raise GravityItem50Error("a critic call was not normalized")
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
            _structure_from_proposal(row, config)
            for row in receipt["proposals"]
            if row["structure_executable_for_frozen_outer_expansion"]
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
                "executable_provider_structures": sum(
                    row["structure_executable_for_frozen_outer_expansion"]
                    for row in proposal_receipt["proposals"]
                ),
                "critic_assessments": len(critic_receipt["assessments"]),
                "provider_generation_slots": 48,
                "matched_control_raw_structures": 48,
                "successful_paid_model_calls": proposal_receipt["counts"]["provider_calls"]
                + critic_receipt["counts"]["critic_calls"],
                "provider_attempts_including_unretained_call": proposal_receipt["counts"][
                    "provider_attempts_including_unretained_pre_journal_call"
                ]
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


def _describe_candidate(
    root: Path,
    config: Mapping[str, Any],
    lane: str,
    programs: Mapping[str, np.ndarray],
    row: int,
) -> dict[str, Any]:
    config49 = _load_item49_config(root)
    contract = config["proposal_contract"]
    labels = _catalog(root)
    grids = config49["program_grammar"]["outer_parameter_grids"]
    structure_index = int(programs["structure_index"][row])
    left_id = int(programs["left_primitive_index"][row])
    right_id = int(programs["right_primitive_index"][row])
    result: dict[str, Any] = {
        "lane": lane,
        "behavior_class_index": row,
        "candidate_id": int(programs["candidate_id"][row]),
        "ordinal": int(programs["ordinal"][row]),
        "structure_index": structure_index,
        "left_primitive": labels[left_id],
        "left_transform": contract["unary_transforms"][
            int(programs["left_transform_index"][row])
        ],
        "right_primitive": labels[right_id],
        "right_transform": contract["unary_transforms"][
            int(programs["right_transform_index"][row])
        ],
        "binary_operator": contract["binary_operators"][
            int(programs["operator_index"][row])
        ],
        "mixing": float(
            contract["mixing_grid"][int(programs["mixing_index"][row])]
        ),
        "amplitude": float(
            grids["amplitude"][int(programs["amplitude_index"][row])]
        ),
        "acceleration_exponent": float(
            grids["acceleration_exponent"][int(programs["exponent_index"][row])]
        ),
        "transition_u": float(
            grids["transition_u"][int(programs["transition_index"][row])]
        ),
        "historical_novelty_claimed": False,
    }
    if lane == "llm_ensemble":
        proposal_receipt = _read_json(_source_path(root, config, "proposal_receipt"))
        critic_receipt = _read_json(_source_path(root, config, "critic_receipt"))
        executable = [
            proposal
            for proposal in proposal_receipt["proposals"]
            if proposal["structure_executable_for_frozen_outer_expansion"]
        ]
        _unique, source_to_unique, _audit = _unique_structures(
            [_structure_from_proposal(proposal, config) for proposal in executable], config
        )
        proposal_ids = [
            executable[index]["proposal_id"]
            for index, unique_index in enumerate(source_to_unique)
            if unique_index == structure_index
        ]
        proposals = [
            proposal for proposal in executable if proposal["proposal_id"] in proposal_ids
        ]
        criticism = {
            row["proposal_id"]: row
            for row in critic_receipt["assessments"]
            if row["proposal_id"] in proposal_ids
        }
        result["provider_lineage"] = [
            {
                "proposal_id": proposal["proposal_id"],
                "provider_model": proposal["provider_model"],
                "provider_role": proposal["provider_role"],
                "title": proposal["title"],
                "origin_self_assessment": proposal["origin_self_assessment"],
                "known_analogues": proposal["known_analogues"],
                "source_domains": proposal["source_domains"],
                "mechanism": proposal["mechanism"],
                "why_not_merely_a_rewrite": proposal["why_not_merely_a_rewrite"],
                "expected_observational_signature": proposal[
                    "expected_observational_signature"
                ],
                "cheapest_falsifier": proposal["cheapest_falsifier"],
                "likely_failure_mode": proposal["likely_failure_mode"],
                "independent_critique": criticism[proposal["proposal_id"]],
            }
            for proposal in proposals
        ]
    return result


def _item49_oof(
    root: Path, arrays: Mapping[str, Any]
) -> tuple[np.ndarray, dict[int, int], dict[str, np.ndarray], np.ndarray]:
    config49 = _load_item49_config(root)
    programs, behavior, _audit = _build_item49_lane_programs(
        root, config49, "pseudorandom"
    )
    evaluation = _read_json(root / ITEM49_EVALUATION_PATH)
    fold_rows = {
        int(row["fold"]): int(
            row["selected_pseudorandom_program"]["behavior_class_index"]
        )
        for row in evaluation["fold_ledger"]
    }
    return (
        _fixed_behavior_oof(fold_rows, behavior, arrays),
        fold_rows,
        programs,
        behavior,
    )


def _structure_diversity(
    structures: Sequence[Mapping[str, int]], root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    labels = _catalog(root)
    item_pairs = [
        tuple(
            sorted(
                (
                    int(labels[row["left_primitive_index"]]["source_item"]),
                    int(labels[row["right_primitive_index"]]["source_item"]),
                )
            )
        )
        for row in structures
    ]
    operators = [
        config["proposal_contract"]["binary_operators"][row["operator_index"]]
        for row in structures
    ]
    primitives = {
        int(row[field])
        for row in structures
        for field in ("left_primitive_index", "right_primitive_index")
    }
    return {
        "structures": len(structures),
        "distinct_primitives": len(primitives),
        "distinct_source_item_pairs": len(set(item_pairs)),
        "cross_item_structures": sum(left != right for left, right in item_pairs),
        "distinct_operators": len(set(operators)),
        "operator_counts": dict(sorted(Counter(operators).items())),
        "source_item_pair_counts": {
            f"{left}-{right}": count
            for (left, right), count in sorted(Counter(item_pairs).items())
        },
    }


def build_evaluation_result(root: Path) -> dict[str, Any]:
    config = load_config(root)
    config49 = _load_item49_config(root)
    config48 = _load_item48_config(root)
    arrays = _item48_evaluation_arrays(root, config48)
    lanes = ("llm_ensemble", "matched_seeded_random")
    programs: dict[str, dict[str, np.ndarray]] = {}
    behaviors: dict[str, np.ndarray] = {}
    lane_audits: dict[str, Any] = {}
    structures: dict[str, list[dict[str, int]]] = {}
    for lane in lanes:
        (
            programs[lane],
            behaviors[lane],
            lane_audits[lane],
            structures[lane],
            _mapping,
        ) = build_lane_candidates(root, config, lane)

    fold_rows: dict[str, dict[int, int]] = {lane: {} for lane in lanes}
    oof = {lane: np.empty(len(arrays["target"]), dtype=float) for lane in lanes}
    ledger: list[dict[str, Any]] = []
    evaluations_by_lane = {lane: 0 for lane in lanes}
    backends: set[str] = set()
    for fold in range(int(config["evaluation"]["outer_folds"])):
        train = arrays["fold"] != fold
        test = ~train
        selected: dict[str, Any] = {}
        for lane in lanes:
            row, loss, backend, count = _best_behavior(
                behaviors[lane], arrays, train, config
            )
            fold_rows[lane][fold] = row
            oof[lane][test] = arrays["base"][test] + behaviors[lane][row, test]
            evaluations_by_lane[lane] += count
            backends.add(backend)
            selected[lane] = {
                "program": _describe_candidate(root, config, lane, programs[lane], row),
                "training_balanced_loss": loss,
            }
        ledger.append(
            {
                "fold": fold,
                "selected_llm_program": selected["llm_ensemble"]["program"],
                "llm_training_balanced_loss": selected["llm_ensemble"][
                    "training_balanced_loss"
                ],
                "selected_matched_random_program": selected[
                    "matched_seeded_random"
                ]["program"],
                "matched_random_training_balanced_loss": selected[
                    "matched_seeded_random"
                ]["training_balanced_loss"],
                "heldout_s4tm_objects": sorted(
                    set(
                        arrays["object"][
                            test & (arrays["population"] == "S4TM")
                        ].tolist()
                    )
                ),
                "heldout_clash_objects": sorted(
                    set(
                        arrays["object"][
                            test & (arrays["population"] == "CLASH")
                        ].tolist()
                    )
                ),
            }
        )

    selected_rows: dict[str, int] = {}
    selected_losses: dict[str, float] = {}
    cpu_gpu_differences: dict[str, float] = {}
    all_rows = np.ones(len(arrays["target"]), dtype=bool)
    for lane in lanes:
        row, loss, backend, count = _best_behavior(
            behaviors[lane], arrays, all_rows, config
        )
        selected_rows[lane] = row
        selected_losses[lane] = loss
        evaluations_by_lane[lane] += count
        backends.add(backend)
        cpu_loss = _score(arrays, arrays["base"] + behaviors[lane][row])[
            "balanced_loss"
        ]
        cpu_gpu_differences[lane] = abs(float(cpu_loss) - loss)
        if cpu_gpu_differences[lane] > float(
            config["evaluation"]["cpu_gpu_tolerance"]
        ):
            raise GravityItem50Error(f"CPU/GPU loss cross-check failed for {lane}")

    item44_oof, fold_item44 = _item44_oof(root, arrays)
    item45_oof, fold_item45 = _item45_oof(root, arrays)
    item46_oof, fold_item46 = _item46_oof(root, arrays)
    item47_oof, fold_item47 = _item47_oof(root, arrays)
    item49_oof, fold_item49, item49_programs, _item49_behavior = _item49_oof(
        root, arrays
    )
    scores = {
        "llm_ensemble_search": _score(arrays, oof["llm_ensemble"]),
        "matched_seeded_random_search": _score(
            arrays, oof["matched_seeded_random"]
        ),
        "item49_pseudorandom_program": _score(arrays, item49_oof),
        "item47_operator_generator": _score(arrays, item47_oof),
        "item46_dimensionless_generator": _score(arrays, item46_oof),
        "item45_universal_interaction": _score(arrays, item45_oof),
        "item44_scale_hierarchy": _score(arrays, item44_oof),
        "baryonic_newton": _score(arrays, arrays["base"]),
        "ordinary_ridge": _score(arrays, _ordinary_crossfit(arrays, config)),
    }
    controls = tuple(name for name in scores if name != "llm_ensemble_search")
    strongest = min(controls, key=lambda name: scores[name]["balanced_loss"])
    candidate_objects = scores["llm_ensemble_search"]["object_losses"]
    control_objects = scores[strongest]["object_losses"]
    object_keys = sorted(candidate_objects)
    diff = np.asarray(
        [control_objects[key] - candidate_objects[key] for key in object_keys]
    )
    raw_counterexample = diff < 0.0
    stable_counterexample = raw_counterexample.copy()

    config44 = _read_json(root / "configs/gravity_item44_scale_hierarchy_v1.json")
    config45 = _load_item45_config(root)
    config46 = _load_item46_config(root)
    config47 = _load_item47_config(root)
    shapes = _shape_by_object(root, arrays)
    systematic_scores: dict[str, Any] = {}
    for variant_name, population, shift in config["evaluation"]["mass_scale_variants"]:
        varied = _item45_variant_arrays(
            arrays, str(population), float(shift), config45
        )
        varied["pi_bank"] = (
            1.0
            / (
                1.0
                + np.abs(
                    _item46_physical_log_values(varied, config46)
                    @ np.asarray(_item46_pi_vectors(config46), dtype=float).T
                )
            )
        ).T
        varied["operator_bank"] = _item47_operator_bank_from_arrays(
            varied, shapes, config47
        )[1].T
        varied["action_bank"] = _item48_action_bank_from_arrays(
            varied, config48
        )[1].T
        varied_bank = _primitive_bank_from_arrays(varied)
        varied_behavior = {
            lane: _program_behavior(
                programs[lane], varied_bank, np.asarray(varied["u"]), config49
            )
            for lane in lanes
        }
        varied_item49_behavior = _program_behavior(
            item49_programs, varied_bank, np.asarray(varied["u"]), config49
        )
        predictions = {
            "llm_ensemble_search": _fixed_behavior_oof(
                fold_rows["llm_ensemble"], varied_behavior["llm_ensemble"], varied
            ),
            "matched_seeded_random_search": _fixed_behavior_oof(
                fold_rows["matched_seeded_random"],
                varied_behavior["matched_seeded_random"],
                varied,
            ),
            "item49_pseudorandom_program": _fixed_behavior_oof(
                fold_item49, varied_item49_behavior, varied
            ),
        }
        item44_variant = np.empty(len(varied["target"]), dtype=float)
        item45_variant = np.empty(len(varied["target"]), dtype=float)
        item46_variant = np.empty(len(varied["target"]), dtype=float)
        item47_variant = np.empty(len(varied["target"]), dtype=float)
        for fold in range(int(config["evaluation"]["outer_folds"])):
            test = varied["fold"] == fold
            item44_variant[test] = _item44_predict(
                fold_item44[fold], varied, config44
            )[test]
            item45_variant[test] = _item45_predict(
                fold_item45[fold], varied, config45, bank_key="interaction_bank"
            )[test]
            item46_variant[test] = _item46_predict(
                fold_item46[fold], varied, config46, bank_key="pi_bank"
            )[test]
            item47_variant[test] = _item47_predict(
                fold_item47[fold], varied, config47, bank_key="operator_bank"
            )[test]
        predictions.update(
            {
                "item47_operator_generator": item47_variant,
                "item46_dimensionless_generator": item46_variant,
                "item45_universal_interaction": item45_variant,
                "item44_scale_hierarchy": item44_variant,
                "baryonic_newton": varied["base"],
                "ordinary_ridge": _ordinary_crossfit(varied, config),
            }
        )
        variants = {
            name: _score(varied, prediction)
            for name, prediction in predictions.items()
        }
        systematic_scores[str(variant_name)] = {
            "llm_ensemble_search": variants["llm_ensemble_search"],
            "matched_seeded_random_search": variants[
                "matched_seeded_random_search"
            ],
            "item45_primary_control": variants["item45_universal_interaction"],
            "strongest_control_name": strongest,
            "strongest_control": variants[strongest],
        }
        for index, key in enumerate(object_keys):
            stable_counterexample[index] &= (
                variants["llm_ensemble_search"]["object_losses"][key]
                > variants[strongest]["object_losses"][key]
            )

    leave_one = [
        float(np.mean(np.delete(diff, index))) for index in range(len(diff))
    ]
    trim_count = max(
        1, int(len(diff) * float(config["evaluation"]["robust_trim_fraction"]))
    )
    trimmed = np.sort(diff)[trim_count:-trim_count]
    llm_loss = scores["llm_ensemble_search"]["balanced_loss"]
    improvement = 100.0 * (
        scores[strongest]["balanced_loss"] - llm_loss
    ) / scores[strongest]["balanced_loss"]
    improvement_item45 = 100.0 * (
        scores["item45_universal_interaction"]["balanced_loss"] - llm_loss
    ) / scores["item45_universal_interaction"]["balanced_loss"]
    improvement_random = 100.0 * (
        scores["matched_seeded_random_search"]["balanced_loss"] - llm_loss
    ) / scores["matched_seeded_random_search"]["balanced_loss"]
    policy_report = {
        "evidence_kind": "empirical",
        "evaluable_objects": len(object_keys),
        "raw_counterexample_count": int(np.sum(raw_counterexample)),
        "quality_verified_counterexample_count": int(np.sum(raw_counterexample)),
        "uncertainty_resolved_counterexample_count": int(
            np.sum(stable_counterexample)
        ),
        "independent_failure_strata": 0,
        "unchanged_independent_replication_failures": 0,
        "aggregate_improvement_percent": improvement,
        "quality_gate_passed": False,
        "strongest_baseline_failed": bool(improvement <= 0.0),
        "leave_one_changes_sign": bool(
            (min(leave_one) <= 0.0) != (float(np.mean(diff)) <= 0.0)
        ),
        "trim_changes_sign": bool(
            (float(np.mean(trimmed)) <= 0.0) != (float(np.mean(diff)) <= 0.0)
        ),
        "object_level_records_preserved": True,
        "missing_quality_limited_records_preserved": True,
        "exclusions_frozen_before_response": True,
    }
    policy = assess_counterexample_evidence(
        policy_report, load_counterexample_policy(root / POLICY_PATH)
    )
    proposal_receipt = _read_json(_source_path(root, config, "proposal_receipt"))
    critic_receipt = _read_json(_source_path(root, config, "critic_receipt"))
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item50-joint-evaluation-1.0",
            "item": 50,
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "selected_llm_program": _describe_candidate(
                root, config, "llm_ensemble", programs["llm_ensemble"], selected_rows["llm_ensemble"]
            ),
            "selected_llm_full_data_balanced_training_loss": selected_losses[
                "llm_ensemble"
            ],
            "selected_matched_random_program": _describe_candidate(
                root,
                config,
                "matched_seeded_random",
                programs["matched_seeded_random"],
                selected_rows["matched_seeded_random"],
            ),
            "selected_matched_random_full_data_balanced_training_loss": selected_losses[
                "matched_seeded_random"
            ],
            "fold_ledger": ledger,
            "scores": scores,
            "strongest_control": strongest,
            "aggregate_improvement_percent": improvement,
            "improvement_over_item45_percent": improvement_item45,
            "improvement_over_matched_random_percent": improvement_random,
            "paired_sign_flip_p": _paired_p(diff, config),
            "robustness": {
                "leave_one_min_mean_control_minus_candidate_loss": min(leave_one),
                "leave_one_max_mean_control_minus_candidate_loss": max(leave_one),
                "trimmed_mean_control_minus_candidate_loss": float(np.mean(trimmed)),
            },
            "diversity": {
                lane: _structure_diversity(structures[lane], root, config)
                for lane in lanes
            },
            "counterexamples": [
                {
                    "object": key,
                    "raw_counterexample": bool(raw_counterexample[index]),
                    "mass_variant_stable_counterexample": bool(
                        stable_counterexample[index]
                    ),
                }
                for index, key in enumerate(object_keys)
            ],
            "systematic_scores": systematic_scores,
            "counterexample_policy_report": policy_report,
            "counterexample_policy_assessment": policy,
            "compute": {
                "backends": sorted(backends),
                "candidate_point_fold_evaluations_by_lane": evaluations_by_lane,
                "candidate_point_fold_evaluations": sum(evaluations_by_lane.values()),
                "cpu_gpu_selected_loss_absolute_difference": cpu_gpu_differences,
                "lane_audits": lane_audits,
            },
            "provider": {
                "successful_calls": 9,
                "provider_attempts": 11,
                "recoverable_input_tokens": proposal_receipt["usage"]["input_tokens"]
                + critic_receipt["usage"]["input_tokens"],
                "recoverable_output_tokens": proposal_receipt["usage"]["output_tokens"]
                + critic_receipt["usage"]["output_tokens"],
                "recoverable_estimated_standard_cost_usd": critic_receipt["usage"][
                    "campaign_including_proposals_estimated_standard_cost_usd"
                ],
                "unrecoverable_completed_attempts": 2,
                "credential_material_persisted": False,
            },
            "counts": {
                "provider_proposal_slots": 48,
                "executable_llm_structures": len(structures["llm_ensemble"]),
                "matched_random_structures": len(structures["matched_seeded_random"]),
                "llm_outcome_scoring_classes": len(programs["llm_ensemble"]["candidate_id"]),
                "matched_random_outcome_scoring_classes": len(
                    programs["matched_seeded_random"]["candidate_id"]
                ),
                "s4tm_lenses": 28,
                "clash_clusters": 20,
                "clash_points": 84,
                "sealed_confirmation_rows": 0,
                "post_evaluation_candidate_cells": 0,
            },
            "limitations": [
                "All empirical response rows were exposed before Item 50; this is retrospective development, not fresh confirmation.",
                "The language models selected structures only from a hand-declared grammar and primitive catalog; they did not invent arbitrary field equations.",
                "Provider and critic labels are fallible lineage metadata and do not establish historical novelty.",
                "Two completed early provider attempts were not durably captured; their token usage and cost are unrecoverable and are disclosed rather than estimated as actuals.",
                "One proposal and one critique slot were placeholders; both remain archived and neither became an empirical formula or veto.",
                "The matched random control has 48 executable structures versus 47 for the LLM lane because the non-executable LLM slot was not replaced.",
                "S4TM uses an analytic projected stellar profile without measured gas; CLASH uses model-dependent published acceleration profiles.",
                "Four global mass shifts do not exhaust measurement, geometry, selection, or lens-model uncertainty.",
                "Neither one empirical mismatch nor the number of mismatches prunes a formula family.",
            ],
        }
    )


def write_evaluation_result(root: Path) -> Path:
    config = load_config(root)
    path = _source_path(root, config, "evaluation_result")
    _write_json(path, build_evaluation_result(root))
    return path


def build_aggregate_result(root: Path) -> dict[str, Any]:
    config = load_config(root)
    preflight = _read_json(_source_path(root, config, "preflight_manifest"))
    proposals = _read_json(_source_path(root, config, "proposal_receipt"))
    critics = _read_json(_source_path(root, config, "critic_receipt"))
    candidate = _read_json(_source_path(root, config, "candidate_manifest"))
    evaluation = _read_json(_source_path(root, config, "evaluation_result"))
    scores = evaluation["scores"]
    llm = scores["llm_ensemble_search"]
    item45 = scores["item45_universal_interaction"]
    promotion = config["evaluation"]["promotion_gates"]
    gates = {
        "three_model_ensemble_completed": proposals["counts"]["models"] == 3,
        "six_generation_calls_completed": proposals["counts"]["provider_calls"] == 6,
        "three_independent_critic_calls_completed": critics["counts"]["critic_calls"] == 3,
        "all_proposal_slots_retained": proposals["counts"]["proposals"] == 48,
        "all_proposal_slots_independently_assessed": critics["counts"]["assessments"] == 48,
        "critic_advice_pruned_no_proposals": critics["claims"][
            "critic_advice_pruned_proposals"
        ]
        is False,
        "response_blind_generation_compilation_and_equivalence": candidate[
            "response_values_used_during_generation_compilation_or_equivalence"
        ]
        == 0,
        "balanced_improvement_over_item45_at_least": float(
            evaluation["improvement_over_item45_percent"]
        )
        >= 100.0 * float(promotion["balanced_improvement_over_item45_at_least"]),
        "improves_both_populations_over_item45": all(
            llm["populations"][population]["loss"]
            < item45["populations"][population]["loss"]
            for population in ("S4TM", "CLASH")
        ),
        "balanced_improvement_over_matched_random_at_least": float(
            evaluation["improvement_over_matched_random_percent"]
        )
        >= 100.0
        * float(promotion["balanced_improvement_over_matched_random_at_least"]),
        "paired_p_at_most": float(evaluation["paired_sign_flip_p"])
        <= float(promotion["paired_p_at_most"]),
        "leave_one_and_trim_stable": bool(
            float(
                evaluation["robustness"][
                    "leave_one_min_mean_control_minus_candidate_loss"
                ]
            )
            > 0.0
            and float(
                evaluation["robustness"][
                    "trimmed_mean_control_minus_candidate_loss"
                ]
            )
            > 0.0
        ),
        "all_mass_scale_variants_positive": all(
            value["llm_ensemble_search"]["balanced_loss"]
            < value["item45_primary_control"]["balanced_loss"]
            for value in evaluation["systematic_scores"].values()
        ),
        "post_evaluation_candidate_cells": int(
            evaluation["counts"]["post_evaluation_candidate_cells"]
        )
        == 0,
        "sealed_confirmation_rows": int(
            evaluation["counts"]["sealed_confirmation_rows"]
        )
        == 0,
        "fresh_confirmation_available": False,
    }
    operational_names = (
        "three_model_ensemble_completed",
        "six_generation_calls_completed",
        "three_independent_critic_calls_completed",
        "all_proposal_slots_retained",
        "all_proposal_slots_independently_assessed",
        "critic_advice_pruned_no_proposals",
        "response_blind_generation_compilation_and_equivalence",
        "post_evaluation_candidate_cells",
        "sealed_confirmation_rows",
    )
    scientific_names = (
        "balanced_improvement_over_item45_at_least",
        "improves_both_populations_over_item45",
        "balanced_improvement_over_matched_random_at_least",
        "paired_p_at_most",
        "leave_one_and_trim_stable",
        "all_mass_scale_variants_positive",
    )
    operational_complete = all(gates[name] for name in operational_names)
    scientific_lead = operational_complete and all(
        gates[name] for name in scientific_names
    )
    decision = (
        "RETROSPECTIVE_ITEM50_LLM_LEAD_REQUIRES_FRESH_TEST"
        if scientific_lead
        else (
            "OPERATIONAL_ITEM50_COMPLETE_COMPARATIVE_VALUE_NOT_DEMONSTRATED"
            if operational_complete
            else "INCOMPLETE_ITEM50_LLM_PIPELINE_RETAINED"
        )
    )
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item50-llm-creativity-result-1.0",
            "item": 50,
            "goal": "GRAVITY_ROADMAP_ITEM_50_LLM_CREATIVITY",
            "decision": decision,
            "selected_llm_program": evaluation["selected_llm_program"],
            "selected_matched_random_program": evaluation[
                "selected_matched_random_program"
            ],
            "scores": scores,
            "strongest_control": evaluation["strongest_control"],
            "aggregate_improvement_percent": evaluation[
                "aggregate_improvement_percent"
            ],
            "improvement_over_item45_percent": evaluation[
                "improvement_over_item45_percent"
            ],
            "improvement_over_matched_random_percent": evaluation[
                "improvement_over_matched_random_percent"
            ],
            "paired_sign_flip_p": evaluation["paired_sign_flip_p"],
            "gates": gates,
            "diversity": evaluation["diversity"],
            "counterexample_policy_assessment": evaluation[
                "counterexample_policy_assessment"
            ],
            "provider": evaluation["provider"],
            "counts": {
                "provider_models": proposals["counts"]["models"],
                "provider_proposal_slots": proposals["counts"]["proposals"],
                "executable_llm_structures": evaluation["counts"][
                    "executable_llm_structures"
                ],
                "matched_random_structures": evaluation["counts"][
                    "matched_random_structures"
                ],
                "llm_outcome_scoring_classes": evaluation["counts"][
                    "llm_outcome_scoring_classes"
                ],
                "matched_random_outcome_scoring_classes": evaluation["counts"][
                    "matched_random_outcome_scoring_classes"
                ],
                "candidate_point_fold_evaluations": evaluation["compute"][
                    "candidate_point_fold_evaluations"
                ],
                "successful_paid_model_calls": 9,
                "provider_attempts": 11,
                "s4tm_lenses": 28,
                "clash_clusters": 20,
                "clash_points": 84,
                "sealed_confirmation_rows": 0,
                "post_evaluation_candidate_cells": 0,
            },
            "source_bindings": {
                "config": {"path": str(CONFIG_PATH), "sha256": _sha256_file(root / CONFIG_PATH)},
                "preflight": {
                    "path": str(_source_path(root, config, "preflight_manifest").relative_to(root)),
                    "sha256": _sha256_file(_source_path(root, config, "preflight_manifest")),
                },
                "proposals": {
                    "path": str(_source_path(root, config, "proposal_receipt").relative_to(root)),
                    "sha256": _sha256_file(_source_path(root, config, "proposal_receipt")),
                },
                "critics": {
                    "path": str(_source_path(root, config, "critic_receipt").relative_to(root)),
                    "sha256": _sha256_file(_source_path(root, config, "critic_receipt")),
                },
                "candidate_manifest": {
                    "path": str(_source_path(root, config, "candidate_manifest").relative_to(root)),
                    "sha256": _sha256_file(_source_path(root, config, "candidate_manifest")),
                },
                "evaluation": {
                    "path": str(_source_path(root, config, "evaluation_result").relative_to(root)),
                    "sha256": _sha256_file(_source_path(root, config, "evaluation_result")),
                },
            },
            "claims": {
                "roadmap_item_50_complete": operational_complete,
                "llm_pipeline_operational": operational_complete,
                "llm_comparative_creativity_value_demonstrated": False,
                "llm_predictive_value_over_matched_random_demonstrated": bool(
                    evaluation["improvement_over_matched_random_percent"] > 0.0
                ),
                "llm_predictive_value_over_item45_demonstrated": bool(
                    evaluation["improvement_over_item45_percent"] > 0.0
                ),
                "provider_or_critic_is_verifier": False,
                "fresh_confirmation_completed": False,
                "historical_novelty_established": False,
                "alternative_to_gr_established": False,
                "dark_matter_eliminated": False,
                "formula_family_pruned": False,
                "single_counterexample_used_as_veto": False,
            },
            "limitations": evaluation["limitations"],
            "next_action": "Advance to Item 51 GPU screening. Preserve every Item 50 proposal and critique, but treat this ensemble/prompt as a failed comparative configuration unless a preregistered new campaign changes the mechanism rather than merely retrying stochastic outputs.",
            "preflight": preflight,
        }
    )


def write_aggregate_result(root: Path) -> Path:
    config = load_config(root)
    path = root / str(config["paths"]["aggregate_result"])
    _write_json(path, build_aggregate_result(root))
    return path


def _content_hash_valid(value: Mapping[str, Any]) -> bool:
    content = dict(value)
    expected = content.pop("content_sha256", None)
    return expected == _sha256_bytes(_canonical_bytes(content))


def replay(root: Path) -> dict[str, Any]:
    config = load_config(root)
    proposals = _read_json(_source_path(root, config, "proposal_receipt"))
    critics = _read_json(_source_path(root, config, "critic_receipt"))
    checks = {
        "provider_proposal_receipt_content_hash": _content_hash_valid(proposals),
        "provider_critic_receipt_content_hash": _content_hash_valid(critics),
        "candidate_manifest": _read_json(_source_path(root, config, "candidate_manifest"))
        == build_candidate_manifest(root),
        "evaluation_result": _read_json(_source_path(root, config, "evaluation_result"))
        == build_evaluation_result(root),
        "aggregate_result": _read_json(root / str(config["paths"]["aggregate_result"]))
        == build_aggregate_result(root),
    }
    return {"ok": all(checks.values()), "checks": checks}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "preflight",
            "propose",
            "critique",
            "compile",
            "evaluate",
            "aggregate",
            "replay",
        ),
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.command == "preflight":
        result: Any = str(write_preflight_manifest(root, live=True))
    elif args.command == "propose":
        result = run_provider_proposals(root)
    elif args.command == "critique":
        result = run_provider_critiques(root)
    elif args.command == "compile":
        result = str(write_candidate_manifest(root))
    elif args.command == "evaluate":
        result = str(write_evaluation_result(root))
    elif args.command == "aggregate":
        result = str(write_aggregate_result(root))
    else:
        result = replay(root)
        if not result["ok"]:
            print(json.dumps(result, sort_keys=True))
            return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
