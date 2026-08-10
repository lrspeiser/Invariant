from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .promotion_orchestrator import ELIGIBILITY
from .static_dictionary import classify_generator_expression

EVALUATOR_ID = "static-covariant-lift-v1"
EVALUATOR_VERSION = "1.0.0"
DICTIONARY_FILE_SHA256 = "54a50e6f20d8e8d59d7d34d4186a615637353175b44cca636c41fa80b873f7bf"
DICTIONARY_CONTENT_SHA256 = "0179ce22a456fe5845414563d3e97a0e9869f612caa8c61792da9b722020ff73"


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _dictionary() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    path = root / "runs" / "static-lift" / "einstein-aether-static-dictionary-ir.json"
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != DICTIONARY_FILE_SHA256:
        raise ValueError("static dictionary file hash mismatch")
    value = json.loads(raw)
    declared = value.get("content_sha256")
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if (
        declared != DICTIONARY_CONTENT_SHA256
        or hashlib.sha256(_canonical(body).encode()).hexdigest() != declared
    ):
        raise ValueError("static dictionary content hash mismatch")
    if value.get("schema_version") != "sigma-static-dictionary-ir-1.0":
        raise ValueError("unsupported static dictionary schema")
    return value


def static_covariant_lift_evaluator(
    candidate: dict[str, Any], context: dict[str, Any]
) -> dict[str, Any]:
    """Classify a sampled-static formula without pretending an unresolved lift failed physics."""

    if context.get("data_eligibility") != ELIGIBILITY:
        raise ValueError("promotion context eligibility is not fail-closed")
    expression = str(candidate.get("correction_expression", ""))
    if not expression:
        raise ValueError("candidate lacks its exact correction expression")
    dictionary = _dictionary()
    legacy = dictionary["legacy_generator_dictionary"]
    classification = classify_generator_expression(
        expression,
        aether_x_available=legacy["x"]["status"] == "derived",
        exact_q_action_match=legacy["q"]["status"] == "derived_exact_action_match",
    )
    lift_decision = str(classification["decision"])
    if lift_decision == "reject_forbidden_baryonic_action_atom":
        decision = "reject"
        blocker = None
    elif lift_decision in {
        "supported_linear_aether_x_lift",
        "supported_exact_projected_aether_q_lift",
    }:
        decision = "pass"
        blocker = None
    else:
        decision = "blocked"
        blocker = lift_decision
    result = {
        "decision": decision,
        "candidate_id": str(candidate["candidate_id"]),
        "input_lineage_sha256": str(context["input_lineage_sha256"]),
        "evaluator_id": EVALUATOR_ID,
        "evaluator_version": EVALUATOR_VERSION,
        "dictionary_file_sha256": DICTIONARY_FILE_SHA256,
        "dictionary_content_sha256": DICTIONARY_CONTENT_SHA256,
        "expression_sha256": hashlib.sha256(expression.encode()).hexdigest(),
        "classification": classification,
        "scope": (
            "necessary covariant-lift classification only; pass is not formal health, "
            "observational support, novelty, or a gravity-theory promotion"
        ),
        "data_eligibility": dict(ELIGIBILITY),
    }
    if blocker is not None:
        result["blocker"] = blocker
    return result
