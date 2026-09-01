"""Mechanically update the 400 TWELL registry rows from the frozen replay ledger."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
LEDGER = REPO / "runs" / "gravity" / "open-gravity-twell-400-source-shaped-rebind-replay-v1" / "compatibility-ledger.jsonl"

OLD_APPENDIX_TEXT = (
    "All rows share: empirical status `no response/zero real scores`; structural status "
    "`exact target-free operator probes passed, source matrix pending`; health `phenomenology "
    "only unless the card itself supplies a fuller closure`; novelty `formula-basis registration "
    "only`; next falsifier `admit through a frozen source matrix, equivalence audit, conventional "
    "comparator and response-blind development test`. `ord` is the zero-based canonical stream "
    "order. `card` and `formula` are exact SHA-256 values, so each row resolves unambiguously to "
    "the authoritative JSONL stream and root above."
)


def load_statuses() -> dict[str, dict[str, str]]:
    by_formula: dict[str, dict[str, str]] = defaultdict(dict)
    with LEDGER.open(encoding="utf-8") as handle:
        for raw in handle:
            row = json.loads(raw)
            by_formula[row["formula_id"]][row["domain"]] = row["status"]
    if len(by_formula) != 400:
        raise RuntimeError(f"expected 400 formulas, found {len(by_formula)}")
    expected_domains = {"galaxy", "cluster", "solar-system", "strong-lens", "void-cosmology"}
    for formula_id, statuses in by_formula.items():
        if set(statuses) != expected_domains:
            raise RuntimeError(f"domain mismatch for {formula_id}: {sorted(statuses)}")
    return dict(by_formula)


def update_registry(path: Path, admission_label: str) -> tuple[int, int]:
    statuses = load_statuses()
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    changed = 0
    executable = 0
    for index, line in enumerate(lines):
        if not line.startswith("|"):
            continue
        fields = line.split("|")
        if len(fields) < 11:
            continue
        formula_id = fields[2].strip()
        if formula_id not in statuses:
            continue
        cluster = statuses[formula_id]["cluster"]
        if cluster == "EXECUTABLE":
            executable += 1
        elif cluster != "SOURCE_BLOCKED":
            raise RuntimeError(f"unexpected cluster status for {formula_id}: {cluster}")
        other = {domain: status for domain, status in statuses[formula_id].items() if domain != "cluster"}
        if set(other.values()) != {"INCOMPATIBLE_FEATURE_SET"}:
            raise RuntimeError(f"unexpected noncluster status for {formula_id}: {other}")
        fields[7] = (
            f" SOURCE_SHAPED_REPLAY_V1: cluster {cluster}; galaxy/solar-system/strong-lens/"
            f"void-cosmology INCOMPATIBLE_FEATURE_SET; SYNTHETIC_ONLY; 0 real scores; {admission_label} "
        )
        lines[index] = "|".join(fields)
        changed += 1
    if changed != 400 or executable != 110:
        raise RuntimeError(f"unexpected update counts changed={changed}, executable={executable}")

    new_text = "\n".join(lines) + ("\n" if text.endswith("\n") else "")
    new_appendix = (
        "All 400 rows are now bound to the frozen source-shaped replay: 110 formulas are executable "
        "on the audited X-COP spherical-static source ABI and 290 are source-blocked there; all 400 "
        "are incompatible with the current PHANGS, Solar, lens and void feature ABIs. The replay is "
        "synthetic-only with zero real scores. Across 2,592 unique X-COP object/formula/cell executions, "
        "2,554 completed and 38 derivative-screen executions remained numerical-invalid; the 62,208-row "
        "fan-out contains 61,296 complete and 912 invalid rows. `ord` is the zero-based canonical stream "
        "order. `card` and `formula` are exact SHA-256 values, so each row resolves unambiguously to the "
        "authoritative JSONL stream and root above."
    )
    if OLD_APPENDIX_TEXT in new_text:
        new_text = new_text.replace(OLD_APPENDIX_TEXT, new_appendix, 1)
    elif new_appendix not in new_text:
        raise RuntimeError("could not locate or verify appendix status paragraph")
    path.write_text(new_text, encoding="utf-8", newline="\n")
    return changed, executable


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--admission-label", required=True)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    for raw_path in args.paths:
        path = raw_path.resolve()
        changed, executable = update_registry(path, args.admission_label)
        print(f"UPDATED {path}: {changed} rows, {executable} executable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
