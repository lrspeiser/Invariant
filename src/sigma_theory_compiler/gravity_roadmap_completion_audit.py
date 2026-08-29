"""Reporting-only audit that every numbered gravity-roadmap item has a receipt."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from sigma_theory_compiler.gravity_item22_polarization_superposition import (
    _content_hashed,
    _read_json,
    _sha256_file,
    _write_json,
)
from sigma_theory_compiler.gravity_items71_72_final_gates import replay as replay_final

GOAL_PATH = Path("docs/GRAVITY_HIDDEN_VARIABLE_AND_THEORY_SEARCH_GOALS.md")
ROADMAP_DIR = Path("runs/gravity/roadmap")
OUTPUT_PATH = ROADMAP_DIR / "roadmap-72-item-completion-audit-v1.json"
EXPECTED_GOAL_SHA256 = "b9db7e1aa76deaf28ce2e840cd3d7db4cf7a086b1c7ca473800780c264a0e772"


class GravityRoadmapCompletionError(RuntimeError):
    """Raised when the immutable goal or receipt inventory is incomplete."""


def build_audit(root: Path) -> dict[str, Any]:
    goal_path = root / GOAL_PATH
    if _sha256_file(goal_path) != EXPECTED_GOAL_SHA256:
        raise GravityRoadmapCompletionError("stable gravity goal changed")
    inventory: dict[str, list[dict[str, Any]]] = {}
    missing = []
    invalid = []
    total = 0
    for item in range(1, 73):
        files = sorted((root / ROADMAP_DIR).glob(f"item-{item:02d}-*.json"))
        if not files:
            missing.append(item)
            inventory[str(item)] = []
            continue
        rows = []
        for path in files:
            try:
                value = _read_json(path)
            except (OSError, json.JSONDecodeError):
                invalid.append(str(path.relative_to(root)).replace("\\", "/"))
                continue
            nested_result = value.get("result")
            nested_decision = (
                nested_result.get("decision") if isinstance(nested_result, dict) else None
            )
            rows.append(
                {
                    "path": str(path.relative_to(root)).replace("\\", "/"),
                    "sha256": _sha256_file(path),
                    "schema_version": value.get("schema_version"),
                    "decision": value.get("decision", nested_decision),
                    "content_sha256_present": len(str(value.get("content_sha256", ""))) == 64,
                }
            )
            total += 1
        inventory[str(item)] = rows
    final_replay = replay_final(root)
    complete = not missing and not invalid and all(inventory.values()) and final_replay["ok"]
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-roadmap-completion-audit-1.0",
            "goal": "GRAVITY_HIDDEN_VARIABLE_AND_THEORY_SEARCH_72_ITEM_EXECUTION_AUDIT",
            "stable_goal_path": str(GOAL_PATH).replace("\\", "/"),
            "stable_goal_sha256": EXPECTED_GOAL_SHA256,
            "numbered_items": 72,
            "items_with_top_level_receipts": sum(bool(rows) for rows in inventory.values()),
            "top_level_receipt_files": total,
            "missing_items": missing,
            "invalid_json_receipts": invalid,
            "inventory": inventory,
            "final_gate_replay": final_replay,
            "execution_audit_complete": bool(complete),
            "claims": {
                "every_numbered_item_has_a_top_level_receipt": not missing,
                "every_numbered_item_passed": False,
                "alternative_to_gr_established": False,
                "dark_matter_eliminated": False,
                "external_independent_confirmation_established": False,
                "historical_novelty_established": False,
                "single_empirical_counterexample_used_as_universal_veto": False,
            },
            "interpretation": "Execution completion means every roadmap item was attempted and recorded; it does not mean every scientific gate passed or the central target was achieved.",
            "next_program": "Generate screened action-level transition-law descendants, then freeze direct group, lensing, and external X-ray/SZ confirmation tests.",
        }
    )


def write_audit(root: Path) -> Path:
    path = root / OUTPUT_PATH
    _write_json(path, build_audit(root))
    return path


def replay(root: Path) -> dict[str, Any]:
    path = root / OUTPUT_PATH
    expected = build_audit(root)
    return {"ok": path.is_file() and _read_json(path) == expected, "path": str(path)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit", "replay"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.command == "audit":
        path = write_audit(root)
        print(path)
        return 0
    result = replay(root)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
