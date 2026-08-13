"""Receipt-derived human report contrasting blind guessing with exact recovery."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .constraint_conditioned_semantic_recovery_tournament import (
    validate_campaign as validate_recovery_campaign,
)
from .semantic_formula_proof_holdout_tournament import (
    validate_campaign as validate_blind_campaign,
)
from .sigma_core import canonical_sha256

REPORT_SCHEMA = "sigma-semantic-discovery-contrast-report-1.0"
BLIND_PATH = "runs/math/semantic-formula-proof-holdout-tournament/campaign.json"
RECOVERY_PATH = "runs/math/constraint-conditioned-semantic-recovery-tournament/campaign.json"
SOURCE_PATH = "src/sigma_theory_compiler/semantic_discovery_contrast_report.py"
TEST_PATH = "tests/test_semantic_discovery_contrast_report.py"
MARKDOWN_PATH = "runs/math/semantic-discovery-contrast-report/report.md"
IPYNB_PATH = "runs/math/semantic-discovery-contrast-report/report.ipynb"
CLAIMS = {
    "both_source_receipts_native_validated": True,
    "blind_fixed_budget_result_is_zero_of_twenty_one": True,
    "constraint_conditioned_result_is_twenty_one_of_twenty_one": True,
    "recovery_uses_one_generic_exact_solver": True,
    "cohorts_share_hidden_targets": False,
    "contrast_is_a_matched_target_performance_comparison": False,
    "native_generators_independently_discovered_recovered_formulas": False,
    "constraint_recovery_establishes_general_discovery": False,
    "scientific_truth_established": False,
    "novelty_established": False,
    "promotion_authorized": False,
}


def _resolve(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("semantic contrast path is not portable")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("semantic contrast path escapes project root") from error
    return resolved


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("semantic contrast receipt must be an object")
    return value


def _chronology(receipt: Mapping[str, Any], *, recovery: bool) -> dict[str, int]:
    phase = receipt["phase_ledger"]
    generation_key = (
        "generation_and_solve_events_before_unseal"
        if recovery
        else "generation_events_before_unseal"
    )
    return {
        "generation_events_before_unseal": phase[generation_key],
        "pre_unseal_target_access_count": phase["pre_unseal_target_access_count"],
        "atomic_unseal_batches": phase["atomic_unseal_batches"],
        "target_records_unsealed": phase["target_records_unsealed"],
        "post_unseal_generation_count": phase["post_unseal_generation_count"],
        "post_unseal_tuning_events": phase["post_unseal_tuning_events"],
    }


def _certificate_counts(receipt: Mapping[str, Any], *, candidate: bool) -> dict[str, int]:
    schemas: Counter[str] = Counter()
    for world in receipt["world_results"]:
        if candidate:
            certificates = [row["proof_certificate"] for row in world["assessments"].values()]
        else:
            certificates = [world["reference_proof_certificate"]]
        schemas.update(certificate["schema_version"] for certificate in certificates)
    return dict(sorted(schemas.items()))


def _validate_semantics(blind: Mapping[str, Any], recovery: Mapping[str, Any]) -> None:
    blind_counts = blind["counts"]
    recovery_counts = recovery["counts"]
    if blind_counts != {
        "candidate_blocks": 0,
        "candidate_passes": 0,
        "candidate_rejects": 21,
        "exact_counterexamples": 21,
        "generator_families": 7,
        "pareto_eligible_candidates": 0,
        "reference_proof_certificates": 3,
        "structured_candidates": 21,
        "world_blocks": 0,
        "world_passes": 0,
        "world_rejects": 3,
        "worlds": 3,
    }:
        raise ValueError("blind semantic receipt counts changed")
    expected_recovery = {
        "candidate_blocks": 0,
        "candidate_passes": 21,
        "candidate_rejects": 0,
        "control_blocks": 2,
        "control_rejects": 1,
        "generator_families": 7,
        "generic_synthesis_invocations": 21,
        "metric_receipts": 42,
        "native_generator_invocations": 21,
        "pareto_eligible_candidates": 21,
        "proof_certificates": 21,
        "world_passes": 3,
        "worlds": 3,
    }
    if recovery_counts != expected_recovery:
        raise ValueError("constraint-conditioned recovery receipt counts changed")
    if _chronology(blind, recovery=False) != _chronology(recovery, recovery=True):
        raise ValueError("paired chronology changed")
    if _chronology(blind, recovery=False) != {
        "generation_events_before_unseal": 21,
        "pre_unseal_target_access_count": 0,
        "atomic_unseal_batches": 1,
        "target_records_unsealed": 3,
        "post_unseal_generation_count": 0,
        "post_unseal_tuning_events": 0,
    }:
        raise ValueError("prospective chronology boundary changed")
    if any(
        world["terminal_status_counts"] != {"reject": 7}
        or world["metric_receipts"]
        or any(
            assessment["status"] != "reject"
            or assessment["proof_certificate"] is not None
            or assessment["counterexample"] is None
            for assessment in world["assessments"].values()
        )
        for world in blind["world_results"]
    ):
        raise ValueError("blind rejection evidence changed")
    if any(
        world["terminal_status_counts"] != {"pass": 7}
        or len(world["metric_receipts"]) != 14
        or any(
            assessment["status"] != "pass" or assessment["proof_certificate"] is None
            for assessment in world["assessments"].values()
        )
        for world in recovery["world_results"]
    ):
        raise ValueError("recovery certificate evidence changed")


def build_report(root: Path) -> dict[str, Any]:
    """Native-validate both receipts and derive one closed contrast object."""

    root = root.resolve()
    blind_path = _resolve(root, BLIND_PATH)
    recovery_path = _resolve(root, RECOVERY_PATH)
    blind = _load_json(blind_path)
    recovery = _load_json(recovery_path)
    validate_blind_campaign(blind, root)
    validate_recovery_campaign(recovery, root)
    _validate_semantics(blind, recovery)

    public_worlds = []
    for world in recovery["preregistration"]["worlds"]:
        constraints = world["constraints"]
        if constraints["kind"] == "evaluations":
            evidence_kind = "exact_input_output_examples"
            evidence_count = len(constraints["rows"])
        else:
            evidence_kind = "exact_base_and_successor_recurrence_axioms"
            evidence_count = 2
        public_worlds.append(
            {
                "world_id": world["world_id"],
                "basis_term_count": len(world["basis"]),
                "evidence_kind": evidence_kind,
                "evidence_count": evidence_count,
                "constraints_sha256": canonical_sha256(constraints),
                "target_commitment_sha256": world["target_commitment_sha256"],
                "closed_form_was_public": False,
            }
        )

    controls = [
        {
            "control_id": row["control_id"],
            "expected_outcome": row["expected_outcome"],
            "observed_outcome": row["observed_outcome"],
            "reason": row["solver_receipt"]["reason"],
            "rank": row["solver_receipt"]["rank"],
            "augmented_rank": row["solver_receipt"]["augmented_rank"],
        }
        for row in recovery["control_results"]
    ]
    unique_expression_counts = []
    for world in recovery["world_results"]:
        expressions = {
            candidate["representation"]["expression_sha256"] for candidate in world["candidates"]
        }
        unique_expression_counts.append(
            {
                "world_id": world["world_id"],
                "lineage_candidates": len(world["candidates"]),
                "unique_recovered_expressions": len(expressions),
            }
        )

    body = {
        "schema_version": REPORT_SCHEMA,
        "bindings": {
            "blind_receipt": {
                "path": BLIND_PATH,
                "file_sha256": _file_sha(blind_path),
                "content_sha256": blind["content_sha256"],
                "source_bindings": blind["source_bindings"],
            },
            "recovery_receipt": {
                "path": RECOVERY_PATH,
                "file_sha256": _file_sha(recovery_path),
                "content_sha256": recovery["content_sha256"],
                "source_bindings": recovery["source_bindings"],
            },
            "builder": {
                "path": SOURCE_PATH,
                "file_sha256": _file_sha(_resolve(root, SOURCE_PATH)),
            },
            "test": {
                "path": TEST_PATH,
                "file_sha256": _file_sha(_resolve(root, TEST_PATH)),
            },
        },
        "contrast": {
            "blind": {
                "worlds": 3,
                "native_generator_families": 7,
                "structured_candidates": 21,
                "passes": 0,
                "rejects": 21,
                "blocks": 0,
                "exact_counterexamples": 21,
                "pareto_eligible_candidates": 0,
            },
            "constraint_conditioned": {
                "worlds": 3,
                "native_generator_families": 7,
                "recovered_candidates": 21,
                "passes": 21,
                "rejects": 0,
                "blocks": 0,
                "proof_certificates": 21,
                "pareto_eligible_candidates": 21,
            },
            "paired_cohorts_use_distinct_hidden_worlds": True,
        },
        "chronology": {
            "blind": _chronology(blind, recovery=False),
            "constraint_conditioned": _chronology(recovery, recovery=True),
        },
        "public_evidence": {
            "worlds": public_worlds,
            "interpretation": (
                "Public target-derived exact examples or recurrence axioms constrain declared "
                "bases without publishing a closed-form target or its coefficient vector."
            ),
        },
        "generic_solver": {
            "grammar": recovery["preregistration"]["generic_grammar"],
            "synthesis_invocations": recovery["counts"]["generic_synthesis_invocations"],
            "lineage_role": recovery["preregistration"]["policies"]["generator_role"],
            "unique_expression_counts": unique_expression_counts,
            "interpretation": (
                "Every family supplies native provenance and deterministic row-order material; "
                "the same exact rank classifier and rational linear solve reconstructs the formula."
            ),
        },
        "certificates": {
            "blind_reference_certificate_schemas": _certificate_counts(blind, candidate=False),
            "blind_candidate_exact_counterexamples": 21,
            "recovery_candidate_certificate_schemas": _certificate_counts(recovery, candidate=True),
            "recovery_reference_certificate_schemas": _certificate_counts(
                recovery, candidate=False
            ),
            "certificate_interpretation": (
                "Certificates check equality or induction after unseal inside the registered "
                "grammar; they do not prove general discovery or scientific significance."
            ),
        },
        "negative_controls": controls,
        "claims": dict(CLAIMS),
        "scope": (
            "receipt-derived methodological contrast between distinct three-world cohorts; it "
            "changes no candidate, target, proof, counterexample, gate, metric, rank, or promotion"
        ),
        "next_gate": (
            "matched independently authored worlds outside a declared linear basis, with an "
            "external proof kernel and the same prospective chronology"
        ),
    }
    return {**body, "content_sha256": canonical_sha256(body)}


def validate_report(value: Mapping[str, Any]) -> None:
    expected = {
        "schema_version",
        "bindings",
        "contrast",
        "chronology",
        "public_evidence",
        "generic_solver",
        "certificates",
        "negative_controls",
        "claims",
        "scope",
        "next_gate",
        "content_sha256",
    }
    if set(value) != expected or value.get("schema_version") != REPORT_SCHEMA:
        raise ValueError("semantic contrast report schema changed")
    body = {key: child for key, child in value.items() if key != "content_sha256"}
    if value.get("content_sha256") != canonical_sha256(body):
        raise ValueError("semantic contrast report self-seal changed")
    if value.get("claims") != CLAIMS:
        raise ValueError("semantic contrast report claim boundary changed")
    if (
        value["contrast"]["blind"]["passes"] != 0
        or value["contrast"]["blind"]["rejects"] != 21
        or value["contrast"]["constraint_conditioned"]["passes"] != 21
        or not value["contrast"]["paired_cohorts_use_distinct_hidden_worlds"]
        or value["chronology"]["blind"] != value["chronology"]["constraint_conditioned"]
        or value["chronology"]["blind"]["generation_events_before_unseal"] != 21
        or value["generic_solver"]["synthesis_invocations"] != 21
        or value["negative_controls"]
        != [
            {
                "control_id": "malformed_unknown_symbol",
                "expected_outcome": "BLOCK",
                "observed_outcome": "BLOCK",
                "reason": "malformed_constraints:ValueError",
                "rank": 0,
                "augmented_rank": 0,
            },
            {
                "control_id": "underdetermined_rank_deficient",
                "expected_outcome": "BLOCK",
                "observed_outcome": "BLOCK",
                "reason": "underdetermined_exact_constraints",
                "rank": 1,
                "augmented_rank": 1,
            },
            {
                "control_id": "noisy_inconsistent_duplicate",
                "expected_outcome": "REJECT",
                "observed_outcome": "REJECT",
                "reason": "inconsistent_exact_constraints",
                "rank": 2,
                "augmented_rank": 3,
            },
        ]
        or any(row["closed_form_was_public"] for row in value["public_evidence"]["worlds"])
        or any(
            row["lineage_candidates"] != 7 or row["unique_recovered_expressions"] != 1
            for row in value["generic_solver"]["unique_expression_counts"]
        )
    ):
        raise ValueError("semantic contrast report semantic boundary changed")


def render_markdown(report: Mapping[str, Any]) -> str:
    validate_report(report)
    bindings = report["bindings"]
    blind = report["contrast"]["blind"]
    recovery = report["contrast"]["constraint_conditioned"]
    chronology = report["chronology"]["blind"]
    public_rows = report["public_evidence"]["worlds"]
    solver = report["generic_solver"]
    certificates = report["certificates"]
    controls = report["negative_controls"]
    lines = [
        "# Semantic discovery contrast: blind guessing and exact constrained recovery",
        "",
        "> Both campaigns are prospectively sealed and receipt-validated. Their hidden worlds",
        "> are distinct, so this is a methodological contrast—not a matched-target scorecard.",
        "",
        "## Receipt bindings",
        "",
        "| Receipt | Path | File SHA-256 | Content SHA-256 |",
        "| --- | --- | --- | --- |",
        f"| Blind structured guessing | `{bindings['blind_receipt']['path']}` | `{bindings['blind_receipt']['file_sha256']}` | `{bindings['blind_receipt']['content_sha256']}` |",
        f"| Constraint-conditioned recovery | `{bindings['recovery_receipt']['path']}` | `{bindings['recovery_receipt']['file_sha256']}` | `{bindings['recovery_receipt']['content_sha256']}` |",
        "",
        "Both JSON receipts were reopened and replayed through their native campaign validators before this projection was built.",
        "",
        "## The exact contrast",
        "",
        "| Campaign | Worlds | Candidates | PASS | REJECT | BLOCK | Evidence after unseal | Pareto eligible |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
        f"| Blind structured guessing | {blind['worlds']} | {blind['structured_candidates']} | {blind['passes']} | {blind['rejects']} | {blind['blocks']} | {blind['exact_counterexamples']} exact counterexamples | {blind['pareto_eligible_candidates']} |",
        f"| Constraint-conditioned recovery | {recovery['worlds']} | {recovery['recovered_candidates']} | {recovery['passes']} | {recovery['rejects']} | {recovery['blocks']} | {recovery['proof_certificates']} exact certificates | {recovery['pareto_eligible_candidates']} |",
        "",
        "The blind campaign produced structured, target-free formulas, but every one of its 21 candidates failed exact equality after unseal. The recovery campaign gave the same seven native families public semantic constraints and then applied one generic exact operator; all 21 lineage-bound candidates passed.",
        "",
        "## Chronology",
        "",
        f"Each campaign completed **{chronology['generation_events_before_unseal']} generation events** across three worlds and seven native families before one atomic unseal. Pre-unseal target access was **{chronology['pre_unseal_target_access_count']}**; the batch disclosed **{chronology['target_records_unsealed']} targets**. Post-unseal generation and tuning were both zero.",
        "",
        "The recovery constraints were committed publicly before generation. The hidden target records remained separately committed until the atomic unseal.",
        "",
        "## What public evidence was added",
        "",
        "| Recovery world | Declared basis terms | Public evidence | Evidence count | Constraint SHA-256 | Closed form public? |",
        "| --- | ---: | --- | ---: | --- | --- |",
    ]
    lines.extend(
        f"| `{row['world_id']}` | {row['basis_term_count']} | `{row['evidence_kind']}` | {row['evidence_count']} | `{row['constraints_sha256']}` | no |"
        for row in public_rows
    )
    lines.extend(
        [
            "",
            report["public_evidence"]["interpretation"],
            "",
            "## The generic recovery operator",
            "",
            f"The declared grammar is `{solver['grammar']['expression_form']}` over exact `{solver['grammar']['coefficient_domain']}` coefficients. The operator converts evaluation examples or recurrence coefficient identities into one rational linear system. It classifies rank exactly: a unique solution emits a candidate, an underdetermined system BLOCKs, an inconsistent system REJECTs, and malformed input BLOCKs.",
            "",
            f"The operator ran **{solver['synthesis_invocations']} times**. Every world has seven native lineage candidates but exactly one recovered expression across those lineages. Native proposals determine provenance and row order; they do not supply the recovered coefficients. The semantic recovery therefore belongs to the generic solver plus the public constraints, not to seven independent discoveries.",
            "",
            "## Exact certificates after unseal",
            "",
            f"The blind receipt contains three validated reference certificates—{certificates['blind_reference_certificate_schemas']}—and 21 exact candidate counterexamples. The recovery receipt contains 21 checked candidate certificates—{certificates['recovery_candidate_certificate_schemas']}—plus three reference certificates—{certificates['recovery_reference_certificate_schemas']}. Polynomial and rational worlds use exact rational-identity certificates; the recurrence world checks its base case and symbolic successor identity.",
            "",
            certificates["certificate_interpretation"],
            "",
            "## Fail-closed negative controls",
            "",
            "| Control | Expected | Observed | Exact reason | rank(A) | rank([A|b]) |",
            "| --- | --- | --- | --- | ---: | ---: |",
        ]
    )
    lines.extend(
        f"| `{row['control_id']}` | {row['expected_outcome']} | {row['observed_outcome']} | `{row['reason']}` | {row['rank']} | {row['augmented_rank']} |"
        for row in controls
    )
    lines.extend(
        [
            "",
            "## What the 21/21 result does—and does not—mean",
            "",
            "The success shows that exact, sufficient semantic conditions can turn a registered linear grammar into a determinate synthesis problem. It does not show that unconstrained native generators independently found the formulas. It does not cover out-of-basis targets, nonlinear coefficient searches, noisy-data inference, external proof kernels, scientific significance, novelty, or promotion.",
            "",
            "Because the blind and recovery cohorts use different hidden targets, subtracting 0/21 from 21/21 is not a matched-world effect estimate. The next sharp gate is a matched, independently authored, out-of-basis campaign with an external proof kernel and the same one-unseal chronology.",
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
            "report_schema": REPORT_SCHEMA,
            "report_content_sha256": report["content_sha256"],
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def build_outputs(root: Path) -> tuple[str, str]:
    report = build_report(root)
    validate_report(report)
    markdown = render_markdown(report)
    notebook = json.dumps(render_ipynb(report), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    validate_output_pair(report, markdown, notebook)
    return markdown, notebook


def validate_output_pair(report: Mapping[str, Any], markdown: str, notebook_text: str) -> None:
    validate_report(report)
    expected_markdown = render_markdown(report)
    expected_notebook = (
        json.dumps(render_ipynb(report), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )
    if markdown != expected_markdown or notebook_text != expected_notebook:
        raise ValueError("semantic contrast output pair changed")
    notebook = json.loads(notebook_text)
    if (
        "".join(notebook["cells"][0]["source"]) != markdown
        or notebook["metadata"]["report_content_sha256"] != report["content_sha256"]
    ):
        raise ValueError("semantic contrast outputs are not semantic twins")


def _write_immutable(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"immutable semantic contrast output differs: {path}")
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
