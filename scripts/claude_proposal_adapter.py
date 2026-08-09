from __future__ import annotations

import argparse
import json
import subprocess
import sys
from typing import Any

REQUIRED = [
    "proposal_type",
    "action",
    "fields",
    "symmetries",
    "universal_constants",
    "derivative_order",
    "degeneracy_conditions",
    "matter_metric",
    "claimed_static_limit",
    "expected_dof",
    "evasion_rationale",
    "falsification_tests",
    "literature_overlap",
    "bounded_grammar",
]

SCHEMA = {
    "type": "object",
    "additionalProperties": True,
    "required": REQUIRED,
    "properties": {
        "proposal_type": {"type": "string"},
        "action": {"type": "string"},
        "fields": {"type": "array", "items": {"type": "string"}},
        "symmetries": {"type": "array", "items": {"type": "string"}},
        "universal_constants": {"type": "array", "items": {"type": "string"}},
        "derivative_order": {"type": "integer", "minimum": 0},
        "degeneracy_conditions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "expression", "equals", "variables", "status"],
                "properties": {
                    "id": {"type": "string"},
                    "expression": {"type": "string"},
                    "equals": {"type": ["number", "string"]},
                    "variables": {"type": "array", "items": {"type": "string"}},
                    "status": {"type": "string", "enum": ["declared_unverified"]},
                },
            },
        },
        "matter_metric": {"type": "string"},
        "claimed_static_limit": {"type": "string"},
        "expected_dof": {"type": "string"},
        "evasion_rationale": {"type": "array", "items": {"type": "string"}},
        "falsification_tests": {"type": "array", "items": {"type": "string"}},
        "literature_overlap": {"type": "array", "items": {"type": "string"}},
        "mechanism_tags": {"type": "array", "items": {"type": "string"}},
        "family_id": {"type": "string"},
        "bounded_grammar": {
            "type": "object",
            "additionalProperties": True,
            "required": ["basis", "max_terms", "coefficient_alphabet"],
            "properties": {
                "basis": {"type": "array", "items": {"type": "string"}},
                "max_terms": {"type": "integer", "minimum": 1, "maximum": 12},
                "coefficient_alphabet": {"type": "array", "items": {"type": "number"}},
            },
        },
    },
}


def extract_proposal(wrapper: dict[str, Any]) -> dict[str, Any]:
    if all(field in wrapper for field in REQUIRED):
        return wrapper
    structured = wrapper.get("structured_output")
    if isinstance(structured, dict):
        return structured
    result = wrapper.get("result")
    if isinstance(result, str):
        parsed = json.loads(result)
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("Claude output did not contain a structured proposal object")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="sonnet")
    parser.add_argument("--effort", default="high")
    parser.add_argument("--max-budget-usd", type=float, default=1.0)
    parser.add_argument("--claude", default="claude")
    args = parser.parse_args()
    packet = json.load(sys.stdin)
    prompt = (
        "You are a constrained mathematical research proposer, not a truth judge. Analyze the "
        "failure packet below and return exactly one bounded, falsifiable covariant gravity-action "
        "proposal matching the supplied JSON schema. Never use prohibited evidence as ground truth; "
        "never claim the proposal passed a test; state unknowns explicitly. "
        "For derivative_order greater than one, degeneracy_conditions must contain explicit "
        "machine-readable algebraic expressions equal to a declared right-hand side; prose-only "
        "claims are invalid. For lower derivative order, return an empty array. "
        "Do not reproduce any "
        "literal string from scientific_contract.prohibited_evidence_patterns anywhere in the "
        "output, including in negations, disclaimers, or proposed falsification tests.\n\n"
        + json.dumps(packet, sort_keys=True)
    )
    command = [
        args.claude,
        "-p",
        "--safe-mode",
        "--tools",
        "",
        "--permission-mode",
        "dontAsk",
        "--no-session-persistence",
        "--model",
        args.model,
        "--effort",
        args.effort,
        "--max-budget-usd",
        str(args.max_budget_usd),
        "--output-format",
        "json",
        "--json-schema",
        json.dumps(SCHEMA, separators=(",", ":")),
        prompt,
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        shell=False,
    )
    if completed.returncode != 0:
        print(completed.stderr[-4000:], file=sys.stderr)
        return completed.returncode or 1
    wrapper = json.loads(completed.stdout)
    proposal = extract_proposal(wrapper)
    result = {
        "schema_version": "sigma-llm-adapter-result-1.0",
        "provider": "claude-code",
        "proposal": proposal,
        "total_cost_usd": wrapper.get("total_cost_usd"),
        "usage": wrapper.get("usage", {}),
        "model_usage": wrapper.get("modelUsage", {}),
        "terminal_reason": wrapper.get("terminal_reason"),
    }
    json.dump(result, sys.stdout, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
