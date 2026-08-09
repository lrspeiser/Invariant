from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_registry(registry: dict[str, Any], output_directory: str | Path) -> tuple[Path, Path]:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "registry.json"
    markdown_path = output / "summary.md"
    json_path.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    counts = registry["counts"]
    survivors = [
        candidate
        for candidate in registry["candidates"]
        if candidate["status"] == "requires_covariant_lift"
    ]
    lines = [
        "# Sigma Theory Compiler run",
        "",
        registry["scope_claim"],
        "",
        "## Outcome",
        "",
        f"- Enumerated signed candidates: {counts['total']}",
        f"- Rejected before covariant work: {counts['rejected_pre_covariant']}",
        f"- Static-sector survivors requiring a covariant lift: {counts['requires_covariant_lift']}",
        "- Fully validated theories: 0",
        "",
        f"> {registry['scientific_warning']}",
        "",
        "## Static-sector survivors",
        "",
        "| ID | Complexity | Coupling | Correction |",
        "|---|---:|---:|---|",
    ]
    for candidate in survivors:
        lines.append(
            f"| `{candidate['candidate_id']}` | {candidate['complexity']} | "
            f"{candidate['coupling']:+g} | `{candidate['canonical_expression']}` |"
        )
    if not survivors:
        lines.append("| — | — | — | No candidate survived the declared static gates. |")
    lines.extend(
        [
            "",
            "## Gate semantics",
            "",
            "- `pass`: this implementation produced positive evidence for the named bounded check.",
            "- `reject`: the candidate is dead in this grammar; later gates stay closed.",
            "- `deferred`: the compiler has not performed the check and makes no health claim.",
            "",
            "The JSON registry contains the derived radial Euler–Lagrange equation and evidence for every gate of every candidate.",
        ]
    )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, markdown_path

