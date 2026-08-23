"""Provider-compatible wire adapter for the external/core Claude campaign.

The sealed confirmatory experiment binds the legacy Claude client byte-for-byte, so provider
compatibility changes belong outside that historical source.  This adapter adds an explicit client
identity and compacts only the critic response grammar on the wire.  It converts the provider's
array response back into the legacy exact-coverage object before the frozen client validates it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from .claude_creativity_api import (
    ClaudeCreativityError,
    Transport,
    urllib_transport,
)
from .sigma_core import canonical_sha256

CLIENT_USER_AGENT = "invariant-core/0.1"
_COVERAGE_INSTRUCTION = (
    " Return exactly one steering action for every supplied candidate ID, with no missing, "
    "extra, or duplicate IDs."
)


def _critic_candidate_ids(schema: Mapping[str, Any]) -> tuple[str, ...]:
    properties = schema.get("properties")
    if not isinstance(properties, Mapping):
        return ()
    role = properties.get("role")
    actions = properties.get("steering_actions")
    if (
        not isinstance(role, Mapping)
        or role.get("const") != "critic"
        or not isinstance(actions, Mapping)
        or actions.get("type") != "object"
    ):
        return ()
    candidate_ids = actions.get("required")
    action_properties = actions.get("properties")
    if (
        not isinstance(candidate_ids, list)
        or not candidate_ids
        or len(candidate_ids) > 64
        or not isinstance(action_properties, Mapping)
        or set(candidate_ids) != set(action_properties)
        or len(candidate_ids) != len(set(candidate_ids))
        or any(not isinstance(item, str) for item in candidate_ids)
    ):
        raise ClaudeCreativityError("external critic schema candidate coverage is invalid")
    templates = [action_properties[item] for item in candidate_ids]
    if any(template != templates[0] for template in templates[1:]):
        raise ClaudeCreativityError("external critic action schemas are not identical")
    return tuple(candidate_ids)


def _compact_critic_schema(
    schema: Mapping[str, Any], candidate_ids: Sequence[str]
) -> dict[str, Any]:
    compact = json.loads(json.dumps(schema))
    actions = compact["properties"]["steering_actions"]
    template = json.loads(json.dumps(actions["properties"][candidate_ids[0]]))
    template["properties"] = {
        "blocker_kind": template["properties"]["blocker_kind"],
        "candidate_id": {"type": "string", "enum": list(candidate_ids)},
        "distance_denominator": template["properties"]["distance_denominator"],
        "distance_numerator": template["properties"]["distance_numerator"],
        "repair": template["properties"]["repair"],
        "verdict": template["properties"]["verdict"],
    }
    template["required"] = [
        "blocker_kind",
        "candidate_id",
        "distance_denominator",
        "distance_numerator",
        "repair",
        "verdict",
    ]
    compact["properties"]["steering_actions"] = {
        "type": "array",
        "items": template,
    }
    return compact


def _structured_text(response: Mapping[str, Any]) -> dict[str, Any]:
    content = response.get("content")
    if (
        not isinstance(content, list)
        or len(content) != 1
        or not isinstance(content[0], Mapping)
        or content[0].get("type") != "text"
        or not isinstance(content[0].get("text"), str)
    ):
        raise ClaudeCreativityError("external Claude response content changed")
    try:
        output = json.loads(content[0]["text"])
    except json.JSONDecodeError as error:
        raise ClaudeCreativityError("external Claude structured text is not JSON") from error
    if not isinstance(output, dict):
        raise ClaudeCreativityError("external Claude structured output is not an object")
    return output


def _legacy_critic_output(
    output: Mapping[str, Any], candidate_ids: Sequence[str]
) -> dict[str, Any]:
    actions = output.get("steering_actions")
    if not isinstance(actions, list):
        raise ClaudeCreativityError("external critic response actions are not an array")
    mapped: dict[str, dict[str, Any]] = {}
    for action in actions:
        if not isinstance(action, Mapping):
            raise ClaudeCreativityError("external critic response action is not an object")
        candidate_id = action.get("candidate_id")
        if not isinstance(candidate_id, str) or candidate_id in mapped:
            raise ClaudeCreativityError("external critic response has duplicate or invalid IDs")
        mapped[candidate_id] = {
            key: value for key, value in action.items() if key != "candidate_id"
        }
    if set(mapped) != set(candidate_ids):
        raise ClaudeCreativityError("external critic response changed candidate coverage")
    adapted = dict(output)
    adapted["steering_actions"] = {
        candidate_id: mapped[candidate_id] for candidate_id in candidate_ids
    }
    return adapted


class ProviderCompatibleClaudeTransport:
    """Adapt the live wire contract while retaining hashes of the actual provider payloads."""

    def __init__(self, transport: Transport = urllib_transport) -> None:
        self.transport = transport
        self._evidence: dict[str, dict[str, Any]] = {}

    def evidence_for(self, response_id: str) -> Mapping[str, Any]:
        return dict(self._evidence.get(response_id, {}))

    def __call__(
        self,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout: float,
    ) -> tuple[int, Mapping[str, Any]]:
        provider_headers = dict(headers)
        provider_headers["user-agent"] = CLIENT_USER_AGENT
        if method != "POST" or body is None:
            return self.transport(method, url, provider_headers, body, timeout)

        try:
            request = json.loads(body)
            schema = request["output_config"]["format"]["schema"]
            prompt = json.loads(request["messages"][0]["content"])
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            raise ClaudeCreativityError("external Claude request envelope changed") from error
        if not isinstance(schema, Mapping) or not isinstance(prompt, dict):
            raise ClaudeCreativityError("external Claude request schema or prompt changed")

        candidate_ids = _critic_candidate_ids(schema)
        wire_contract_adapted = bool(candidate_ids)
        if candidate_ids:
            request["output_config"]["format"]["schema"] = _compact_critic_schema(
                schema, candidate_ids
            )
            instruction = prompt.get("instruction")
            if not isinstance(instruction, str):
                raise ClaudeCreativityError("external critic instruction changed")
            prompt["instruction"] = instruction + _COVERAGE_INSTRUCTION
            request["messages"][0]["content"] = json.dumps(
                prompt, sort_keys=True, separators=(",", ":")
            )
        provider_body = json.dumps(request, sort_keys=True, separators=(",", ":")).encode()
        status, response = self.transport(
            method, url, provider_headers, provider_body, timeout
        )
        if status != 200 or response.get("type") != "message":
            return status, response

        provider_output = _structured_text(response)
        adapted_output = (
            _legacy_critic_output(provider_output, candidate_ids)
            if candidate_ids
            else provider_output
        )
        adapted_response = dict(response)
        adapted_response["content"] = [
            {
                "type": "text",
                "text": json.dumps(adapted_output, sort_keys=True, separators=(",", ":")),
            }
        ]
        response_id = response.get("id")
        if isinstance(response_id, str):
            provider_schema = request["output_config"]["format"]["schema"]
            provider_prompt = request["messages"][0]["content"]
            self._evidence[response_id] = {
                "provider_header_names": sorted(name.lower() for name in provider_headers),
                "provider_prompt_sha256": hashlib.sha256(provider_prompt.encode()).hexdigest(),
                "provider_raw_output_sha256": canonical_sha256(provider_output),
                "provider_request_schema_sha256": canonical_sha256(provider_schema),
                "wire_contract_adapter_used": wire_contract_adapted,
            }
        return status, adapted_response


__all__ = ["CLIENT_USER_AGENT", "ProviderCompatibleClaudeTransport"]
