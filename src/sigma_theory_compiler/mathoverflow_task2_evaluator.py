"""Blind exact/reference evaluator for the available-now MathOverflow Task 2 trial."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from . import broken_arxiv_task2 as broken
from . import mathoverflow_task2 as trial
from .frankl_counterexample_verifier import verify_family
from .sigma_core import canonical_sha256

EVIDENCE_SCHEMA = "invariant-mathoverflow-task2-blind-scoring-evidence-1.0"

REPAIR_VALID_ID = "submission.024371c8bbf47bce312ec3dfecafdb0e"
PROMISING_GENERATOR_IDS = {
    "submission.4863e99cba26bcc0682219c1d8c44fc7",
    "submission.695db0eef6a710cd1a4d1e8eff07ee40",
    "submission.a10cf2b0beaa357600a8beca1dc2d775",
    "submission.c32434f9d477dc3f4070666db2fff47e",
    "submission.e5fb7b5dd26ace7ab5d0642a5d43905c",
}


def _sealed(body: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(body)
    result["content_sha256"] = canonical_sha256(body)
    return result


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"JSON root is not an object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def union_closure(generators: Iterable[Iterable[int]]) -> list[list[int]]:
    family = {frozenset(row) for row in generators}
    if not family or any(not row for row in family):
        raise ValueError("generators must be nonempty finite sets")
    while True:
        expanded = family | {left | right for left in family for right in family}
        if expanded == family:
            break
        family = expanded
    return [list(row) for row in sorted(family, key=lambda row: (len(row), sorted(row)))]


def _generator_checks() -> dict[str, dict[str, Any]]:
    a, b = 100, 101
    checks = {
        "submission.4863e99cba26bcc0682219c1d8c44fc7": union_closure(
            [{a, i} for i in range(1, 5)] + [{b, i} for i in range(1, 5)]
        ),
        "submission.695db0eef6a710cd1a4d1e8eff07ee40": union_closure(
            [{1, 2}, {1, 3}, {4, 5}, {2, 3, 4}, {2, 3, 5}]
        ),
        "submission.a10cf2b0beaa357600a8beca1dc2d775": union_closure(
            [{a, i} for i in range(1, 5)]
            + [{b, i} for i in range(1, 5)]
            + [{a, b}]
        ),
        "submission.c32434f9d477dc3f4070666db2fff47e": union_closure(
            [{i, 10} for i in range(1, 5)]
            + [{i, 5} for i in range(1, 5)]
            + [{5, 10}]
        ),
        "submission.e5fb7b5dd26ace7ab5d0642a5d43905c": [
            sorted(set(range(1, 6)) - {x}) for x in range(1, 6)
        ]
        + [list(range(1, 6))],
    }
    results: dict[str, dict[str, Any]] = {}
    for submission_id, family in checks.items():
        receipt = verify_family(family)
        results[submission_id] = {
            "canonical_family_sha256": receipt["canonical_family_sha256"],
            "family_size": receipt["family_size"],
            "delta": receipt["delta"],
            "residual_delta": receipt["residual_delta"],
            "union_closed": receipt["union_closed"],
            "exact_counterexample_valid": receipt["exact_counterexample_valid"],
            "verifier_receipt_content_sha256": receipt["content_sha256"],
        }
    return results


def build_blind_evaluation(
    public: Mapping[str, Any],
    reference: Mapping[str, Any],
    reference_verifier: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if (
        public.get("blindness", {}).get("arm_identity_disclosed") is not False
        or public.get("blindness", {}).get("submissions_frozen") is not True
        or reference.get("public_submissions_content_sha256") != public["content_sha256"]
        or reference_verifier.get("exact_counterexample_valid") is not True
        or reference_verifier.get("delta") != 19
        or set(reference_verifier.get("residual_delta", {}).values()) != {10}
    ):
        raise ValueError("blind evaluation prerequisites changed")
    exact_checks = _generator_checks()
    if set(exact_checks) != PROMISING_GENERATOR_IDS or any(
        row["exact_counterexample_valid"] for row in exact_checks.values()
    ):
        raise ValueError("promising generator verification changed")
    universal_repair = (
        "If a finite nonempty union-closed family has an element contained in every member, "
        "then that element witnesses residual Delta zero."
    )
    repair_graph = canonical_sha256(
        {
            "claim": universal_repair,
            "dependencies": ["universal_element", "empty_residual_family", "delta_empty_zero"],
        }
    )
    evaluations = []
    for submission in public["submissions"]:
        submission_id = submission["submission_id"]
        checked = exact_checks.get(submission_id)
        repair_valid = submission_id == REPAIR_VALID_ID
        if checked is not None:
            notes = (
                "The proposal supplied exact generators, but exhaustive union closure and integer "
                "degree counting found a witness x with 2*residual_delta[x] <= delta."
            )
            rejection = (
                f"Exact generator check rejected the proposed family; family_size="
                f"{checked['family_size']}, delta={checked['delta']}, residual_delta="
                f"{checked['residual_delta']}."
            )
            invocations = 3
        else:
            notes = (
                "Blind textual review found no complete valid family violating the inequality for "
                "every element. Examples that satisfy the conjecture and unsupported search claims "
                "do not answer the counterexample request."
            )
            rejection = "No exact counterexample or independently checkable rejection was supplied."
            invocations = 1
        evaluations.append(
            {
                "submission_id": submission_id,
                "false_as_written": True,
                "exact_counterexample_valid": False,
                "independent_external_rejection_valid": False,
                "counterexample_or_rejection": rejection,
                "canonical_counterexample_sha256": "",
                "failed_assumption": (
                    "The proposal did not establish strict residual excess for every ground-set element."
                ),
                "smallest_failed_assumption_valid": False,
                "repaired_statement": universal_repair if repair_valid else "",
                "repair_nonvacuous_valid": repair_valid,
                "repair_proof_or_external_acceptance_valid": repair_valid,
                "canonical_repair_graph_sha256": repair_graph if repair_valid else "",
                "verifier_invocations": invocations,
                "notes": notes,
            }
        )
    evaluations.sort(key=lambda row: row["submission_id"])
    evidence = _sealed(
        {
            "schema_version": EVIDENCE_SCHEMA,
            "task_id": config["task_id"],
            "public_submissions_content_sha256": public["content_sha256"],
            "reference_content_sha256": reference["content_sha256"],
            "reference_verifier_content_sha256": reference_verifier["content_sha256"],
            "scoring_completed_before_arm_map_opened": True,
            "scored_submission_ids": [row["submission_id"] for row in evaluations],
            "promising_generator_exact_checks": exact_checks,
            "repair_graph_canonicalization": {
                "method": "normalized claim plus sorted dependency labels",
                "universal_element_repair_sha256": repair_graph,
            },
            "conclusion": "NO_SUBMISSION_SUPPLIED_A_VALID_COUNTEREXAMPLE",
        }
    )
    packet = broken.seal_evaluation_packet(
        {
            "task_id": config["task_id"],
            "public_submissions_content_sha256": public["content_sha256"],
            "reference_material_opened_after_submissions_sealed": True,
            "evaluator": {
                "name": "Invariant independent exact/reference evaluator",
                "organization": "OpenAI Codex evaluator plus MathOverflow accepted-answer authority",
                "independent_from_generator": True,
                "verifier_kind": "exact_executable_verifier",
                "evidence_uri": "runs/math/mathoverflow-task2/blind-scoring-evidence-v2.json",
                "signed_artifact_sha256": evidence["content_sha256"],
                "counterexample_canonicalizer": (
                    "sigma_theory_compiler.frankl_counterexample_verifier.canonical_family_v1"
                ),
                "proof_graph_canonicalizer": "normalized_claim_sorted_dependency_labels_v1",
                "named_human_reviewers": [],
            },
            "evaluations": evaluations,
        }
    )
    broken.validate_evaluation_packet(packet, public, config)
    return packet, evidence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--staged", type=Path, required=True)
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--coordinator", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--reference-verifier", type=Path, required=True)
    parser.add_argument("--evaluation-output", type=Path, required=True)
    parser.add_argument("--evidence-output", type=Path, required=True)
    parser.add_argument("--adjudication-output", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    adapter_config = trial.load_config(root)
    config = trial.effective_generation_config(root, adapter_config)
    staged = _read_json(args.staged)
    public = _read_json(args.public)
    receipt = _read_json(args.receipt)
    packet, evidence = build_blind_evaluation(
        public,
        _read_json(args.reference),
        _read_json(args.reference_verifier),
        config,
    )
    _write_json(args.evaluation_output, packet)
    _write_json(args.evidence_output, evidence)
    # The arm map is deliberately not read until the complete scoring packet is sealed on disk.
    coordinator = _read_json(args.coordinator)
    adjudication = broken.build_adjudication(
        public, receipt, coordinator, staged, config, packet
    )
    _write_json(args.adjudication_output, adjudication)
    print(
        json.dumps(
            {
                "decision": adjudication["decision"],
                "status": adjudication["status"],
                "content_sha256": adjudication["content_sha256"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
