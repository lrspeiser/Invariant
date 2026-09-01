"""Fail-closed verifier for the Open-Gravity nine-lane final deliverable.

This verifier checks report structure and evidence-path closure.  It does not
recompute scientific receipts and therefore cannot promote a lane result; it
only prevents an incomplete prose handoff from being labelled final.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_REPORT = (
    REPO / "work" / "open-gravity-nine-lane-comprehensive-final-report-draft-2026-08-31.md"
)
DEFAULT_EVIDENCE = (
    REPO / "work" / "open-gravity-nine-lane-final-evidence-tables-draft-2026-08-31.md"
)
DEFAULT_REGISTRY = REPO / "work" / "open-gravity-theory-formula-status-registry-draft.md"
DEFAULT_COMPLETION = REPO / "work" / "open-gravity-session-completion-audit.md"

REQUIRED_REPORT_HEADINGS = (
    "## Ranked nine-lane result",
    "## Per-object and per-dataset results",
    "## Strongest result in each category",
    "### Strongest observation:",
    "### Strongest theorem:",
    "### Strongest method:",
    "## Exact conventional countermodels",
    "## Complete blocked and invalid ledger",
    "## Three fastest independent falsifiers",
    "## Publish-now judgment",
    "## Evidence paths and integrity anchors",
)

REQUIRED_JUDGMENTS = (
    "Empirical fit",
    "Structural identifiability",
    "Physical consistency",
    "Novelty",
    "Publication value",
)

REQUIRED_LANE_DATA_MARKERS = (
    "### Lane 1 —",
    "### Lane 2 —",
    "### Lane 3 —",
    "### Lane 5 —",
    "### Lane 6 —",
    "### Lanes 4, 7, 8, and 9 —",
    "Lane 4's independently audited synthetic v2 matrix",
)


def require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def inline_evidence_paths(text: str) -> set[str]:
    paths: set[str] = set()
    for token in re.findall(r"`([^`]+)`", text):
        normalized = token.replace("\\", "/")
        if normalized.startswith(("runs/", "work/", "configs/", "src/", "tests/")):
            paths.add(normalized.rstrip(".,;:"))
    return paths


def verify(
    report: Path,
    evidence: Path,
    registry: Path,
    completion: Path,
    allow_pending: bool,
) -> list[str]:
    failures: list[str] = []
    for path, label in (
        (report, "report"),
        (evidence, "evidence tables"),
        (registry, "registry"),
        (completion, "completion audit"),
    ):
        require(path.is_file(), f"missing {label}: {path}", failures)
    if failures:
        return failures

    report_text = report.read_text(encoding="utf-8")
    evidence_text = evidence.read_text(encoding="utf-8")
    registry_text = registry.read_text(encoding="utf-8")
    completion_text = completion.read_text(encoding="utf-8")

    for heading in REQUIRED_REPORT_HEADINGS:
        require(heading in report_text, f"missing report heading: {heading}", failures)
    for judgment in REQUIRED_JUDGMENTS:
        require(judgment in report_text, f"missing separated judgment: {judgment}", failures)
    for marker in REQUIRED_LANE_DATA_MARKERS:
        require(marker in report_text, f"missing per-object/data marker: {marker}", failures)

    rank_matches = re.findall(r"^\|\s*([1-9])\s*\|\s*([1-9])\s+—", report_text, flags=re.MULTILINE)
    ranks = [int(rank) for rank, _lane in rank_matches]
    lanes = [int(lane) for _rank, lane in rank_matches]
    require(ranks == list(range(1, 10)), f"rank rows are not exactly 1..9: {ranks}", failures)
    require(
        sorted(lanes) == list(range(1, 10)),
        f"lane rows are not a permutation of 1..9: {lanes}",
        failures,
    )
    states = [
        int(value)
        for value in re.findall(
            r"^\|\s*[1-9]\s*\|\s*[1-9]\s+—[^|]*,\s*R([0-8])\s*\|", report_text, flags=re.MULTILINE
        )
    ]
    require(
        states == list(range(9)),
        f"ranking states are not exactly R0..R8 in rank order: {states}",
        failures,
    )
    evidence_rank_matches = re.findall(
        r"^\|\s*([1-9])\s*\|\s*([1-9])\s+—[^|]*\|\s*R([0-8])\s*\|",
        evidence_text,
        flags=re.MULTILINE,
    )
    evidence_ranking = [
        (int(rank), int(lane), int(state)) for rank, lane, state in evidence_rank_matches
    ]
    expected_ranking = list(zip(range(1, 10), lanes, states, strict=True))
    require(
        evidence_ranking == expected_ranking,
        f"report/evidence ranking mismatch: report={expected_ranking}, evidence={evidence_ranking}",
        failures,
    )

    falsifier_section = report_text.partition("## Three fastest independent falsifiers")[
        2
    ].partition("## Publish-now judgment")[0]
    falsifier_numbers = re.findall(r"(?m)^([1-3])\.\s+\*\*", falsifier_section)
    require(
        falsifier_numbers == ["1", "2", "3"],
        f"expected exactly three numbered falsifiers, got {falsifier_numbers}",
        failures,
    )

    if not allow_pending:
        require(
            "[PENDING" not in report_text,
            "final report still contains a [PENDING ...] gate",
            failures,
        )
        require(
            "Status: **DRAFT" not in report_text, "final report still labels itself DRAFT", failures
        )
        require(
            not re.search(
                r"await(?:s|ing)? (?:a )?distinct audit|independent audit pending",
                report_text,
                flags=re.IGNORECASE,
            ),
            "final report still contains an unresolved distinct-audit statement",
            failures,
        )
        require(
            "ADMISSION_AUDIT_PENDING" not in registry_text,
            "registry still marks TWELL admission audit pending",
            failures,
        )

    evidence_section = report_text.partition("## Evidence paths and integrity anchors")[2]
    referenced = inline_evidence_paths(evidence_section)
    missing = sorted(path for path in referenced if not (REPO / path).exists())
    require(not missing, f"missing referenced evidence paths: {missing}", failures)

    require(
        "## 1. Auditable ranking rule and five independent judgments" in evidence_text,
        "evidence tables lack ranking/judgment section",
        failures,
    )
    require(
        "## 2. Per-object/data appendix" in evidence_text,
        "evidence tables lack per-object appendix",
        failures,
    )
    require("## Reading rules" in registry_text, "registry lacks reading rules", failures)
    require(
        "## A. Nine active lanes and named formula branches" in registry_text,
        "registry lacks nine-lane branch table",
        failures,
    )
    require(
        "## Global requirements" in completion_text,
        "completion audit lacks global requirements",
        failures,
    )
    require(
        "## Final deliverable checklist" in completion_text,
        "completion audit lacks final checklist",
        failures,
    )
    completion_global = completion_text.partition("## Global requirements")[2].partition(
        "## Lane evidence"
    )[0]
    require(
        "| PARTIAL |" not in completion_global,
        "completion audit has a PARTIAL global requirement",
        failures,
    )
    require(
        "| BLOCK |" not in completion_global,
        "completion audit has a BLOCK global requirement",
        failures,
    )
    require(
        "- [ ]" not in completion_text,
        "completion audit contains an unchecked deliverable",
        failures,
    )

    twell_rows = set(
        re.findall(r"^\|\s*\d+\s*\|\s*(TW2-A\d{2}-D\d{2})\s*\|", registry_text, flags=re.MULTILINE)
    )
    require(
        len(twell_rows) == 380,
        f"registry has {len(twell_rows)} unique atomic TWELL rows, expected 380",
        failures,
    )
    x_rows = set(re.findall(r"^\|\s*\d+\s*\|\s*(X\d{2})\s*\|", registry_text, flags=re.MULTILINE))
    require(
        len(x_rows) == 20, f"registry has {len(x_rows)} TWELL cross rows, expected 20", failures
    )

    registry_expected_counts = {
        "GP01 branches": (r"^\|\s*GP01-", 7),
        "QG ontology rows": (r"^\|\s*QG\d{2}\s", 13),
        "Q00-Q15 programs": (r"^\|\s*Q(?:0\d|1[0-5])\s", 16),
        "DC00-DC07 formulas": (r"^\|\s*DC0[0-7]\s", 8),
        "CM01 control": (r"^\|\s*CM01\s", 1),
        "K00-K06 kernels": (r"^\|\s*K0[0-6]\s", 7),
        "Lane-6 mechanisms": (r"^\|\s*L6\s", 8),
        "VQ00-VQ10 laws": (r"^\|\s*VQ(?:0\d|10)\s", 11),
        "DPEL01": (r"^\|\s*DPEL01\s", 1),
        "DGKT01": (r"^\|\s*DGKT01\s", 1),
    }
    for label, (pattern, expected_count) in registry_expected_counts.items():
        observed = len(re.findall(pattern, registry_text, flags=re.MULTILINE))
        require(
            observed == expected_count,
            f"registry has {observed} {label}, expected {expected_count}",
            failures,
        )
    historical_items = set(re.findall(r"^\|\s*Item\s+(\d+):", registry_text, flags=re.MULTILINE))
    require(
        len(historical_items) == 72,
        f"registry has {len(historical_items)} historical items, expected 72",
        failures,
    )
    require("\\n" not in registry_text, "registry contains a literal escaped newline", failures)
    require(
        not re.search(r"(?m)^\+", registry_text),
        "registry contains a leading patch-marker plus sign",
        failures,
    )

    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--completion", type=Path, default=DEFAULT_COMPLETION)
    parser.add_argument("--allow-pending", action="store_true")
    args = parser.parse_args()

    failures = verify(
        args.report.resolve(),
        args.evidence.resolve(),
        args.registry.resolve(),
        args.completion.resolve(),
        args.allow_pending,
    )
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    print("PASS: Open-Gravity final deliverable structure and evidence paths are closed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
