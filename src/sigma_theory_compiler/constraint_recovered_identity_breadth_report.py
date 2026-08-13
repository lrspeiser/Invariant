"""Build deterministic human-readable twins for the recovered identity breadth receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .constraint_recovered_identity_breadth_lean_bridge import (
    OUTPUT_PATH as RECEIPT_PATH,
)
from .constraint_recovered_identity_breadth_lean_bridge import (
    validate_checked_receipt,
)

REPORT_SCHEMA = "invariant-constraint-recovered-identity-breadth-report-1.0"
SOURCE_PATH = "src/sigma_theory_compiler/constraint_recovered_identity_breadth_report.py"
TEST_PATH = "tests/test_constraint_recovered_identity_breadth_report.py"
MARKDOWN_PATH = "docs/notebooks/generated/constraint-recovered-identity-breadth.md"
IPYNB_PATH = "docs/notebooks/generated/constraint-recovered-identity-breadth.ipynb"
CLAIMS = {
    "checked_receipt_native_validated": True,
    "quartic_integer_coefficient_replay_documented": True,
    "partial_fraction_integer_coefficient_replay_documented": True,
    "quartic_real_lean_check_documented": True,
    "false_quartic_control_rejection_documented": True,
    "general_formula_discovery_established": False,
    "novelty_established": False,
    "promotion_authorized": False,
    "scientific_or_physics_truth_inferred": False,
}
_REPORT_KEYS = {
    "bindings",
    "claims",
    "content_sha256",
    "counts",
    "decision",
    "false_control",
    "lean_check",
    "partial_fraction_derivation",
    "quartic_derivation",
    "schema_version",
    "scope",
}


def _resolve(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("identity breadth report path is not portable")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("identity breadth report path escapes project root") from error
    return resolved


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _content_sha(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        _canonical({key: item for key, item in value.items() if key != "content_sha256"})
    ).hexdigest()


def _text_file_sha(path: Path) -> str:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(text.encode()).hexdigest()


def _load_receipt(root: Path) -> dict[str, Any]:
    path = _resolve(root, RECEIPT_PATH)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("identity breadth receipt must be an object")
    validate_checked_receipt(value, root=root)
    return value


def build_report(root: Path) -> dict[str, Any]:
    """Validate the checked receipt and derive one closed explanatory object."""

    root = root.resolve()
    receipt = _load_receipt(root)
    replays = {row["world_id"]: row for row in receipt["integer_polynomial_replays"]}
    quartic = replays["constraint.hidden_quartic"]
    partial = replays["constraint.hidden_partial_fraction"]
    adapter = receipt["adapter_receipt"]
    false = receipt["false_control"]
    if (
        quartic["computed_coefficients_constant_first"] != [-30, -1, 0, 2, 1]
        or partial["computed_numerator_coefficients_constant_first"] != [127, 53, 6]
        or partial["computed_denominator_coefficients_constant_first"] != [70, 59, 14, 1]
        or adapter["decision"] != "pass_lean_checked_closed_premise"
        or false["adapter_decision"] != "block_lean_process_failure"
    ):
        raise ValueError("identity breadth report source semantics changed")

    body = {
        "schema_version": REPORT_SCHEMA,
        "bindings": {
            "receipt": {
                "path": RECEIPT_PATH,
                "file_sha256": _text_file_sha(_resolve(root, RECEIPT_PATH)),
                "content_sha256": receipt["content_sha256"],
                "receipt_role": receipt["receipt_role"],
            },
            "builder": {
                "path": SOURCE_PATH,
                "canonical_lf_sha256": _text_file_sha(_resolve(root, SOURCE_PATH)),
            },
            "test": {
                "path": TEST_PATH,
                "canonical_lf_sha256": _text_file_sha(_resolve(root, TEST_PATH)),
            },
        },
        "decision": receipt["decision"],
        "counts": dict(receipt["counts"]),
        "quartic_derivation": {
            "recovered_expression": "x^4 + 2x^3 - x - 30",
            "factorization": "(x - 2)(x + 3)(x^2 + x + 5)",
            "first_product": {
                "calculation": "(x - 2)(x + 3) = x^2 + x - 6",
                "coefficients_constant_first": [-6, 1, 1],
            },
            "coefficient_calculations": [
                "constant: (-6)(5) = -30",
                "x: (-6)(1) + (1)(5) = -1",
                "x^2: (-6)(1) + (1)(1) + (1)(5) = 0",
                "x^3: (1)(1) + (1)(1) = 2",
                "x^4: (1)(1) = 1",
            ],
            "factor_coefficients_constant_first": quartic["factor_coefficients_constant_first"],
            "computed_coefficients_constant_first": quartic["computed_coefficients_constant_first"],
            "recovered_coefficients_constant_first": quartic[
                "recovered_coefficients_constant_first"
            ],
            "exact_equality": quartic["exact_equality"],
            "floating_point_operations": quartic["floating_point_operations"],
            "replay_content_sha256": quartic["content_sha256"],
        },
        "partial_fraction_derivation": {
            "target_expression": "3/(x + 2) - 2/(x + 5) + 5/(x + 7)",
            "common_denominator": "(x + 2)(x + 5)(x + 7)",
            "numerator_terms": [
                "3(x + 5)(x + 7) = 3x^2 + 36x + 105",
                "-2(x + 2)(x + 7) = -2x^2 - 18x - 28",
                "5(x + 2)(x + 5) = 5x^2 + 35x + 50",
            ],
            "numerator_sum": "6x^2 + 53x + 127",
            "denominator_expansion": "x^3 + 14x^2 + 59x + 70",
            "recovered_expression": ("(6x^2 + 53x + 127)/(x^3 + 14x^2 + 59x + 70)"),
            "computed_numerator_coefficients_constant_first": partial[
                "computed_numerator_coefficients_constant_first"
            ],
            "computed_denominator_coefficients_constant_first": partial[
                "computed_denominator_coefficients_constant_first"
            ],
            "regular_domain_exclusions": partial["regular_domain_exclusions"],
            "exact_equality": partial["exact_equality"],
            "floating_point_operations": partial["floating_point_operations"],
            "replay_content_sha256": partial["content_sha256"],
        },
        "lean_check": {
            "target": adapter["target"],
            "decision": adapter["decision"],
            "exit_code": adapter["execution"]["exit_code"],
            "dependencies": adapter["dependency_audit"]["dependencies"],
            "dependency_closure_valid": adapter["dependency_audit"]["closure_valid"],
            "theorem_statement": receipt["theorem_contract"]["statement"],
            "proof_method": receipt["theorem_contract"]["proof_method"],
            "sorry_or_axiom_used": receipt["theorem_contract"]["sorry_or_axiom_used"],
            "toolchain_version": receipt["toolchain_receipt"]["version"],
            "toolchain_commit": receipt["toolchain_receipt"]["commit"],
        },
        "false_control": {
            "target": false["target"],
            "alteration": false["alteration"],
            "decision": false["adapter_decision"],
            "nonzero_exit_code": false["nonzero_exit_code"],
            "rejected_before_receipt_promotion": false["rejected_before_receipt_promotion"],
        },
        "claims": dict(CLAIMS),
        "scope": (
            "human-readable projection of one validated checked receipt covering two synthetic "
            "constraint-recovered identities; it changes no candidate, proof, gate, rank, or "
            "promotion decision"
        ),
    }
    return {**body, "content_sha256": hashlib.sha256(_canonical(body)).hexdigest()}


def validate_report(value: Mapping[str, Any]) -> None:
    if (
        set(value) != _REPORT_KEYS
        or value.get("schema_version") != REPORT_SCHEMA
        or value.get("content_sha256") != _content_sha(value)
    ):
        raise ValueError("identity breadth report schema or seal changed")
    if value.get("claims") != CLAIMS:
        raise ValueError("identity breadth report claim boundary changed")
    counts = value.get("counts", {})
    quartic = value.get("quartic_derivation", {})
    partial = value.get("partial_fraction_derivation", {})
    lean = value.get("lean_check", {})
    false = value.get("false_control", {})
    if (
        counts.get("recovered_worlds_bound") != 2
        or counts.get("recovered_candidates_bound") != 14
        or counts.get("integer_polynomial_replays") != 2
        or counts.get("kernel_checked_theorems") != 1
        or counts.get("false_controls_rejected") != 1
        or quartic.get("computed_coefficients_constant_first") != [-30, -1, 0, 2, 1]
        or quartic.get("exact_equality") is not True
        or quartic.get("floating_point_operations") != 0
        or partial.get("computed_numerator_coefficients_constant_first") != [127, 53, 6]
        or partial.get("computed_denominator_coefficients_constant_first") != [70, 59, 14, 1]
        or partial.get("regular_domain_exclusions") != [-7, -5, -2]
        or partial.get("exact_equality") is not True
        or partial.get("floating_point_operations") != 0
        or lean.get("decision") != "pass_lean_checked_closed_premise"
        or lean.get("exit_code") != 0
        or lean.get("dependency_closure_valid") is not True
        or lean.get("sorry_or_axiom_used") is not False
        or false.get("decision") != "block_lean_process_failure"
        or false.get("nonzero_exit_code") is not True
        or false.get("rejected_before_receipt_promotion") is not True
    ):
        raise ValueError("identity breadth report semantic boundary changed")


def render_markdown(report: Mapping[str, Any]) -> str:
    validate_report(report)
    binding = report["bindings"]["receipt"]
    quartic = report["quartic_derivation"]
    partial = report["partial_fraction_derivation"]
    lean = report["lean_check"]
    false = report["false_control"]
    lines = [
        "# Two recovered identities: arithmetic replay and an independent Lean check",
        "",
        "> This notebook is generated from a fail-closed checked receipt. It documents two",
        "> bounded synthetic recoveries; it does not establish general discovery or novelty.",
        "",
        "## Evidence binding",
        "",
        f"The source receipt is `{binding['path']}` with content SHA-256 `{binding['content_sha256']}` and canonical file SHA-256 `{binding['file_sha256']}`. Its terminal decision is `{report['decision']}`.",
        "",
        "The receipt binds two recovered worlds, 14 lineage candidates, two exact symbolic certificates, two integer replays, one successful kernel theorem, and one rejected false control.",
        "",
        "## 1. Quartic identity",
        "",
        f"The recovered expanded expression is **{quartic['recovered_expression']}**. The independent factorized form is **{quartic['factorization']}**.",
        "",
        f"First, {quartic['first_product']['calculation']}. Multiplying its coefficient vector `[-6, 1, 1]` by `[5, 1, 1]` gives:",
        "",
    ]
    lines.extend(f"- {calculation}" for calculation in quartic["coefficient_calculations"])
    lines.extend(
        [
            "",
            "Thus the constant-first vector is `[-30, -1, 0, 2, 1]`, exactly the recovered coefficient vector. This replay used integer additions and multiplications only: zero floating-point operations.",
            "",
            "## 2. Partial-fraction identity",
            "",
            f"Start from **{partial['target_expression']}** over the common denominator **{partial['common_denominator']}**. Its three numerator contributions are:",
            "",
        ]
    )
    lines.extend(f"- {term}" for term in partial["numerator_terms"])
    lines.extend(
        [
            "",
            f"Adding them yields **{partial['numerator_sum']}**. Expanding the common denominator yields **{partial['denominator_expansion']}**. Therefore the result is **{partial['recovered_expression']}**.",
            "",
            "The exact integer replay produced numerator coefficients `[127, 53, 6]` and denominator coefficients `[70, 59, 14, 1]`. Equality is asserted only on the regular domain; the excluded points are `x = -7, -5, -2`.",
            "",
            "## 3. Independent Lean kernel check",
            "",
            f"Lean {lean['toolchain_version']} checked `{lean['target']}` with exit code {lean['exit_code']}. The theorem executes the same constant-first `List Int` convolution and proves the result with `decide`. The dependency audit closed over `{', '.join(lean['dependencies'])}`; no `sorry` or user axiom was admitted.",
            "",
            "This Lean theorem checks the quartic coefficient identity independently of the recovery campaign's SymPy certificate. The partial-fraction identity is independently replayed by the bridge's closed integer arithmetic, but is not claimed here as a second Lean theorem.",
            "",
            "## 4. Deliberate failure",
            "",
            f"The negative control changed the constant coefficient from `-30` to `-29`. Lean returned `{false['decision']}` with a nonzero exit code, and the result was rejected before receipt promotion. This demonstrates that the bridge records a failed proof rather than silently accepting or rewriting it.",
            "",
            "## Boundary of the result",
            "",
            "These are two synthetic identities recovered inside a preregistered exact grammar. The evidence establishes exact replay for both and a real Lean check for the quartic. It does not establish general formula discovery, mathematical novelty, scientific truth, physics truth, or promotion eligibility.",
            "",
            f"Report content SHA-256: `{report['content_sha256']}`.",
            "",
        ]
    )
    return "\n".join(lines)


def render_ipynb(report: Mapping[str, Any]) -> dict[str, Any]:
    markdown = render_markdown(report)
    return {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {"report_content_sha256": report["content_sha256"]},
                "source": markdown.splitlines(keepends=True),
            }
        ],
        "metadata": {
            "language_info": {"name": "markdown"},
            "report_content_sha256": report["content_sha256"],
            "report_schema": REPORT_SCHEMA,
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def build_outputs(root: Path) -> tuple[str, str]:
    report = build_report(root)
    validate_report(report)
    markdown = render_markdown(report)
    notebook = json.dumps(render_ipynb(report), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    return markdown, notebook


def _write_immutable(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"immutable identity breadth report differs: {path}")
        return
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(content)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    arguments = parser.parse_args()
    root = Path(arguments.project_root).resolve()
    markdown, notebook = build_outputs(root)
    _write_immutable(_resolve(root, MARKDOWN_PATH), markdown)
    _write_immutable(_resolve(root, IPYNB_PATH), notebook)
    print(
        json.dumps(
            {
                "markdown_sha256": hashlib.sha256(markdown.encode()).hexdigest(),
                "ipynb_sha256": hashlib.sha256(notebook.encode()).hexdigest(),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
