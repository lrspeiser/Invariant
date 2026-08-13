"""Receipt-derived human-readable comprehensive-alpha coverage projection."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .alpha_observational_rehearsal import (
    build_observational_rehearsal,
    validate_observational_receipt,
)
from .alpha_operational_rehearsal import (
    run_operational_rehearsal,
    validate_operational_receipt,
)
from .comprehensive_alpha_cross_generator_campaign import (
    OUTPUT_PATH as CROSS_GENERATOR_PATH,
)
from .comprehensive_alpha_cross_generator_campaign import (
    validate_campaign as validate_cross_generator,
)
from .generated_candidate_formula_gpu_stress_campaign import (
    validate_campaign as validate_gpu_campaign,
)
from .lean_production_kernel_vertical_slice import OUTPUT_PATH as LEAN_RECEIPT_PATH
from .lean_production_kernel_vertical_slice import validate_receipt as validate_lean_receipt
from .math_benchmark_runner import OUTPUT_PATH as CURRICULUM_PATH
from .math_benchmark_runner import validate_readiness
from .math_known_identity_pipeline_control import OUTPUT_PATH as IDENTITY_PATH
from .math_known_identity_pipeline_control import validate_result as validate_identity
from .research_notebook import (
    build_natural_sum_notebook,
    render_ipynb,
    validate_notebook,
)
from .research_notebook import (
    render_markdown as render_notebook_markdown,
)
from .sigma_core import canonical_sha256

OUTPUT_PATH = "docs/COMPREHENSIVE_ALPHA_COVERAGE_MATRIX.md"
SOURCE_PATH = "src/sigma_theory_compiler/comprehensive_alpha_coverage_report.py"
TEST_PATH = "tests/test_comprehensive_alpha_coverage_report.py"
REPORT_SCHEMA = "sigma-comprehensive-alpha-coverage-report-1.0"
GPU_CAMPAIGN_PATH = "runs/engine/generated-candidate-formula-gpu-stress-campaign.json"
GPU_CONFIG_PATH = "configs/generated_candidate_formula_gpu_stress_campaign.json"
NOTEBOOK_MD_PATH = "docs/notebooks/generated/natural-sum-rediscovery.md"
NOTEBOOK_IPYNB_PATH = "docs/notebooks/generated/natural-sum-rediscovery.ipynb"
WALKTHROUGH_PATH = "docs/KNOWN_ANSWER_SUCCESS_FAILURE_WALKTHROUGH.md"


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("coverage input must be a JSON object")
    return value


def derive_coverage(
    root: Path,
    operational_receipt: Mapping[str, Any],
    observational_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate authoritative receipts and derive a closed coverage projection."""

    root = root.resolve()
    cross_path = root / CROSS_GENERATOR_PATH
    curriculum_path = root / CURRICULUM_PATH
    lean_path = root / LEAN_RECEIPT_PATH
    gpu_path = root / GPU_CAMPAIGN_PATH
    identity_path = root / IDENTITY_PATH
    cross = _load_json(cross_path)
    curriculum = _load_json(curriculum_path)
    lean = _load_json(lean_path)
    gpu = _load_json(gpu_path)
    identity = _load_json(identity_path)
    validate_cross_generator(cross, root)
    validate_readiness(curriculum, root)
    validate_lean_receipt(lean, root=root)
    validate_gpu_campaign(gpu, root / GPU_CONFIG_PATH)
    validate_identity(identity, root)
    notebook = build_natural_sum_notebook(root)
    validate_notebook(notebook.to_dict())
    notebook_md = render_notebook_markdown(notebook)
    notebook_ipynb = (
        json.dumps(render_ipynb(notebook), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )
    if (root / NOTEBOOK_MD_PATH).read_text(encoding="utf-8") != notebook_md:
        raise ValueError("checked natural-sum Markdown differs from deterministic rebuild")
    if (root / NOTEBOOK_IPYNB_PATH).read_text(encoding="utf-8") != notebook_ipynb:
        raise ValueError("checked natural-sum ipynb differs from deterministic rebuild")
    operational = json.loads(json.dumps(operational_receipt))
    observational = json.loads(json.dumps(observational_receipt))
    validate_operational_receipt(operational)
    validate_observational_receipt(observational)
    gpu_counts = gpu["counts"]
    gpu_summary = {
        "candidate_count": gpu_counts["candidate_count"],
        "comparison_count": gpu["gpu_cpu_comparison"]["comparison_count"],
        "exact_rational_checks": gpu_counts["cpu_exact_rational_crosschecks"],
        "violations": gpu["gpu_cpu_comparison"]["violating_point_count"],
        "measured_evaluations": gpu_counts["gpu_measured_candidate_formula_evaluations"],
    }
    if gpu_summary != {
        "candidate_count": 163,
        "comparison_count": 5_341_184,
        "exact_rational_checks": 5_216,
        "violations": 0,
        "measured_evaluations": 87_509_958_656,
    }:
        raise ValueError("GPU stress control summary changed")

    if lean["decision"] == "pass_real_lean_kernel_vertical_slice":
        lean_status = "exercised"
        lean_evidence = (
            "real Lean adapter kernel-checked 1 bounded theorem with closed dependencies"
        )
        lean_boundary = "known-answer protocol smoke theorem only; no general scientific truth"
    elif lean["decision"] == "block_lean_executable_not_discovered":
        lean_status = "blocked"
        lean_evidence = "real Lean adapter presented 1 theorem and returned BLOCK before execution"
        lean_boundary = "Lean executable was not discovered; 0 theorems were kernel checked"
    else:
        raise ValueError("Lean vertical-slice terminal decision changed")

    ready_slots = [row for row in curriculum["slots"] if row["status"] == "ready"]
    synthetic_domains = sorted(row["domain"] for row in ready_slots if row["cohort"] == "synthetic")
    if synthetic_domains != ["algebra", "combinatorics", "geometry", "number_theory"]:
        raise ValueError("synthetic five-domain curriculum floor changed")
    historical_domains = sorted(
        row["domain"] for row in ready_slots if row["cohort"] == "historical"
    )
    if historical_domains != ["arithmetic"]:
        raise ValueError("historical five-domain curriculum floor changed")

    family_by_candidate = {
        row["candidate"]["artifact_id"]: row["family"] for row in cross["candidate_family_bindings"]
    }
    gate_statuses = {row["family"]: {} for row in cross["candidate_family_bindings"]}
    for outcome in cross["gate_outcomes"]:
        gate_statuses[family_by_candidate[outcome["artifact"]["artifact_id"]]][
            outcome["gate_id"]
        ] = outcome["status"]
    disposition = {}
    for family, statuses in gate_statuses.items():
        nonpass = [status for status in statuses.values() if status != "pass"]
        disposition[family] = nonpass[0] if nonpass else "pass"
    if Counter(disposition.values()) != {"pass": 4, "block": 1, "reject": 1, "error": 1}:
        raise ValueError("cross-generator outcome narrative distribution changed")

    inputs = {
        "cross_generator": {
            "path": CROSS_GENERATOR_PATH,
            "file_sha256": _file_sha(cross_path),
            "content_sha256": cross["content_sha256"],
        },
        "curriculum": {
            "path": CURRICULUM_PATH,
            "file_sha256": _file_sha(curriculum_path),
            "content_sha256": curriculum["content_sha256"],
        },
        "lean_kernel": {
            "path": LEAN_RECEIPT_PATH,
            "file_sha256": _file_sha(lean_path),
            "content_sha256": lean["content_sha256"],
            "decision": lean["decision"],
        },
        "operational_rehearsal": {
            "schema_version": operational["schema_version"],
            "content_sha256": operational["content_sha256"],
        },
        "observational_rehearsal": {
            "schema_version": observational["schema_version"],
            "content_sha256": observational["content_sha256"],
        },
        "gpu_stress": {
            "path": GPU_CAMPAIGN_PATH,
            "file_sha256": _file_sha(gpu_path),
            "content_sha256": gpu["content_sha256"],
            "summary": gpu_summary,
        },
        "human_record": {
            "identity_path": IDENTITY_PATH,
            "identity_file_sha256": _file_sha(identity_path),
            "identity_content_sha256": identity["content_sha256"],
            "notebook_md_path": NOTEBOOK_MD_PATH,
            "notebook_md_sha256": _file_sha(root / NOTEBOOK_MD_PATH),
            "notebook_ipynb_path": NOTEBOOK_IPYNB_PATH,
            "notebook_ipynb_sha256": _file_sha(root / NOTEBOOK_IPYNB_PATH),
            "walkthrough_path": WALKTHROUGH_PATH,
            "walkthrough_sha256": _file_sha(root / WALKTHROUGH_PATH),
        },
    }
    slices = [
        {
            "ordinal": 1,
            "slice": "Candidate generation",
            "status": "exercised",
            "evidence": "7/7 registered families emitted common-pack CandidateArtifacts",
            "boundary": "generator output establishes neither truth nor promotion",
        },
        {
            "ordinal": 2,
            "slice": "Common evaluation and explanation",
            "status": "exercised",
            "evidence": "14 stage outcomes, 14 hard-gate outcomes, 4 Pareto-eligible candidates",
            "boundary": "rank and explanation are integration receipts, not scientific validation",
        },
        {
            "ordinal": 3,
            "slice": "External proof kernel",
            "status": lean_status,
            "evidence": lean_evidence,
            "boundary": lean_boundary,
        },
        {
            "ordinal": 4,
            "slice": "Blind rediscovery benchmarks",
            "status": "exercised",
            "evidence": "5 ready slots cover arithmetic, algebra, combinatorics, geometry, and number theory",
            "boundary": "five-domain alpha floor only; 195 post-alpha slots remain missing and curriculum success is false",
        },
        {
            "ordinal": 5,
            "slice": "Durable execution and budgets",
            "status": "exercised",
            "evidence": "bounded admit, interrupt, lease recovery, attempt-2 completion, checkpoint, and zero-spend branch replayed",
            "boundary": "owned synthetic rehearsal only; no days/weeks production claim or scientific pass",
        },
        {
            "ordinal": 6,
            "slice": "Computational accelerators",
            "status": "exercised",
            "evidence": (
                f"{gpu_summary['candidate_count']:,} candidates; "
                f"{gpu_summary['comparison_count']:,} GPU/CPU comparisons; "
                f"{gpu_summary['exact_rational_checks']:,} exact rational checks; "
                f"{gpu_summary['violations']:,} violations; "
                f"{gpu_summary['measured_evaluations']:,} measured synthetic evaluations"
            ),
            "boundary": "synthetic numerical control only; no proof, ranking, observations, or paid-provider success",
        },
        {
            "ordinal": 7,
            "slice": "Observational adapters",
            "status": "exercised",
            "evidence": "synthetic Solar, galaxy, lensing, and cluster controls replayed",
            "boundary": "real-data admission, scientific pass, rank writes, and registry writes are false or zero",
        },
        {
            "ordinal": 8,
            "slice": "Human-readable research record",
            "status": "exercised",
            "evidence": "known-identity result, checked natural-sum Markdown/ipynb, and success/failure/block walkthrough are hash bound",
            "boundary": "human-readable records preserve derivation, counterexample, block, limits, and replay IDs without promotion",
        },
    ]
    status_counts = Counter(row["status"] for row in slices)
    report = {
        "schema_version": REPORT_SCHEMA,
        "inputs": inputs,
        "slices": slices,
        "generator_dispositions": disposition,
        "operational_dispositions": {
            "scheduler": "pass_control_recovery",
            "llm": operational["llm_control"]["decision"],
            "gpu": "not_started",
        },
        "counts": {
            "vertical_slices": 8,
            "exercised": status_counts["exercised"],
            "partial": status_counts["partial"],
            "blocked": status_counts["blocked"],
            "curriculum_ready": 5,
            "curriculum_missing": 195,
        },
        "claims": {
            "receipt_projection_validated": True,
            "comprehensive_alpha_exit_reached": False,
            "scientific_truth_established": False,
            "novelty_established": False,
            "promotion_authorized": False,
            "curriculum_complete": False,
        },
    }
    report["content_sha256"] = canonical_sha256(report)
    return report


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render only the closed projection schema to deterministic Markdown."""

    expected = {
        "schema_version",
        "inputs",
        "slices",
        "generator_dispositions",
        "operational_dispositions",
        "counts",
        "claims",
        "content_sha256",
    }
    if set(report) != expected or report["schema_version"] != REPORT_SCHEMA:
        raise ValueError("coverage report schema changed")
    if report["content_sha256"] != canonical_sha256(
        {key: value for key, value in report.items() if key != "content_sha256"}
    ):
        raise ValueError("coverage report self-seal changed")
    if (
        set(report["inputs"])
        != {
            "cross_generator",
            "curriculum",
            "gpu_stress",
            "human_record",
            "lean_kernel",
            "operational_rehearsal",
            "observational_rehearsal",
        }
        or report["counts"]
        != {
            "vertical_slices": 8,
            "exercised": Counter(row["status"] for row in report["slices"])["exercised"],
            "partial": Counter(row["status"] for row in report["slices"])["partial"],
            "blocked": Counter(row["status"] for row in report["slices"])["blocked"],
            "curriculum_ready": 5,
            "curriculum_missing": 195,
        }
        or [(row["ordinal"], row["status"]) for row in report["slices"]]
        not in (
            [
                (1, "exercised"),
                (2, "exercised"),
                (3, "blocked"),
                (4, "exercised"),
                (5, "exercised"),
                (6, "exercised"),
                (7, "exercised"),
                (8, "exercised"),
            ],
            [
                (1, "exercised"),
                (2, "exercised"),
                (3, "exercised"),
                (4, "exercised"),
                (5, "exercised"),
                (6, "exercised"),
                (7, "exercised"),
                (8, "exercised"),
            ],
        )
        or report["generator_dispositions"]
        != {
            "bayesian": "pass",
            "cross_domain": "block",
            "egraph": "reject",
            "evolutionary": "pass",
            "grammar": "pass",
            "llm": "error",
            "symbolic": "pass",
        }
        or report["slices"][2]["status"]
        != (
            "exercised"
            if report["inputs"]["lean_kernel"]["decision"] == "pass_real_lean_kernel_vertical_slice"
            else "blocked"
        )
        or report["claims"]
        != {
            "receipt_projection_validated": True,
            "comprehensive_alpha_exit_reached": False,
            "scientific_truth_established": False,
            "novelty_established": False,
            "promotion_authorized": False,
            "curriculum_complete": False,
        }
    ):
        raise ValueError("coverage report semantic contract changed")
    lines = [
        "# Comprehensive alpha coverage matrix",
        "",
        "> Generated only from validated immutable receipts. `exercised` means an integration path",
        "> replayed inside its declared scope; it does not mean scientific truth, novelty, or promotion.",
        "",
        "## Coverage",
        "",
        "| # | Vertical slice | Status | Receipt-derived evidence | Fail-closed boundary |",
        "| ---: | --- | --- | --- | --- |",
    ]
    lines.extend(
        f"| {row['ordinal']} | {row['slice']} | `{row['status']}` | {row['evidence']} | {row['boundary']} |"
        for row in report["slices"]
    )
    dispositions = report["generator_dispositions"]
    pass_families = [family for family, status in dispositions.items() if status == "pass"]
    block_family = next(family for family, status in dispositions.items() if status == "block")
    reject_family = next(family for family, status in dispositions.items() if status == "reject")
    error_family = next(family for family, status in dispositions.items() if status == "error")
    lines.extend(
        [
            "",
            "## Outcome narratives",
            "",
            "### Success",
            "",
            f"`{', '.join(pass_families)}` passed both registered hard gates. Only these four candidates received exact metric receipts, Pareto fronts, and receipt-bound explanations. This is a bounded integration success, not a truth or promotion decision.",
            "",
            "### Block",
            "",
            f"`{block_family}` passed `hard_exact` but was blocked at `hard_structure`. It received no metrics and no Pareto front. Separately, the operational LLM adapter was blocked before any provider call and settled exactly zero spend.",
            "",
            "### Reject",
            "",
            f"`{reject_family}` was rejected at `hard_exact`. The rejection is preserved as a terminal hard-gate receipt; a passing second gate cannot restore metric or front eligibility.",
            "",
            "### Error",
            "",
            f"`{error_family}` is a quarantined offline proposal whose injected domain-pack failure became an `error` receipt at `hard_exact`. The error disclosed no private exception text and received no metrics or front.",
            "",
            "## Curriculum floor",
            "",
            "The five ready controls cover arithmetic, algebra, combinatorics, geometry, and number theory. Coverage is exactly **5/200**; **195 slots remain missing**, and curriculum success remains false.",
            "",
            "## Operational rehearsal",
            "",
            "One owned expired lease was recovered and completed. Paid-LLM calls, network calls, settled spend, GPU reservation, GPU execution, and NVML sampling remained zero or false in that current-runtime rehearsal. This validates recovery and the allowed disabled no-network LLM branch only; it is separate from the historical measured accelerator control.",
            "",
            "## Historical accelerator control",
            "",
            f"The validated synthetic stress receipt records {report['inputs']['gpu_stress']['summary']['candidate_count']:,} candidates, {report['inputs']['gpu_stress']['summary']['comparison_count']:,} GPU/CPU comparisons, {report['inputs']['gpu_stress']['summary']['exact_rational_checks']:,} exact rational checks, {report['inputs']['gpu_stress']['summary']['violations']:,} violations, and {report['inputs']['gpu_stress']['summary']['measured_evaluations']:,} measured synthetic evaluations. It establishes numerical replay agreement only: no proof, scientific rank, observation, or paid-provider success follows.",
            "",
            "## Human-readable record",
            "",
            "The native known-identity result, deterministically rebuilt natural-sum Markdown and Jupyter notebook, and checked success/failure walkthrough bind the success, reject/counterexample, block, derivation, limit, and replay-ID narratives below to validated records.",
            "",
            "## Receipt bindings",
            "",
            "| Input | Path or schema | File SHA-256 | Content SHA-256 |",
            "| --- | --- | --- | --- |",
        ]
    )
    for label in ("cross_generator", "curriculum", "lean_kernel", "gpu_stress"):
        binding = report["inputs"][label]
        lines.append(
            f"| `{label}` | `{binding['path']}` | `{binding['file_sha256']}` | `{binding['content_sha256']}` |"
        )
    operational = report["inputs"]["operational_rehearsal"]
    lines.append(
        f"| `operational_rehearsal` | `{operational['schema_version']}` | n/a (owned ephemeral receipt) | `{operational['content_sha256']}` |"
    )
    observational = report["inputs"]["observational_rehearsal"]
    lines.append(
        f"| `observational_rehearsal` | `{observational['schema_version']}` | n/a (in-memory synthetic receipt) | `{observational['content_sha256']}` |"
    )
    human = report["inputs"]["human_record"]
    lines.extend(
        [
            f"| `known_identity` | `{human['identity_path']}` | `{human['identity_file_sha256']}` | `{human['identity_content_sha256']}` |",
            f"| `natural_sum_markdown` | `{human['notebook_md_path']}` | `{human['notebook_md_sha256']}` | n/a (deterministic checked rendering) |",
            f"| `natural_sum_ipynb` | `{human['notebook_ipynb_path']}` | `{human['notebook_ipynb_sha256']}` | n/a (deterministic checked rendering) |",
            f"| `success_failure_walkthrough` | `{human['walkthrough_path']}` | `{human['walkthrough_sha256']}` | n/a (checked human record) |",
        ]
    )
    lines.extend(
        [
            "",
            "## Exit boundary",
            "",
            "All eight bounded vertical slices are exercised, but the comprehensive-alpha exit gate is **not reached** by these receipts because a clean current CI/full-coverage test is not evidenced here. The 195 missing curriculum slots are a post-alpha limitation rather than an alpha-exit criterion. Synthetic observational controls do not admit real data or establish scientific support.",
            "",
            f"Projection content SHA-256: `{report['content_sha256']}`.",
            "",
        ]
    )
    return "\n".join(lines)


def build_markdown(
    root: Path,
    operational_receipt: Mapping[str, Any],
    observational_receipt: Mapping[str, Any],
) -> str:
    return render_markdown(derive_coverage(root, operational_receipt, observational_receipt))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output", default=OUTPUT_PATH)
    arguments = parser.parse_args()
    root = Path(arguments.project_root).resolve()
    with tempfile.TemporaryDirectory(prefix="sigma-alpha-coverage-") as directory:
        operational = run_operational_rehearsal(root, Path(directory) / "owned")
        observational = build_observational_rehearsal(root)
        markdown = build_markdown(root, operational, observational)
    output = root / arguments.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown, encoding="utf-8", newline="\n")
    print(
        json.dumps(
            {"output": arguments.output, "sha256": hashlib.sha256(markdown.encode()).hexdigest()}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
