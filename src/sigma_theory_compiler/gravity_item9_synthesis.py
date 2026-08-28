"""Deterministic synthesis for gravity roadmap Item 9."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .sigma_core import canonical_json_bytes, canonical_sha256

ROADMAP_PATH = "docs/GRAVITY_HIDDEN_VARIABLE_AND_THEORY_SEARCH_GOALS.md"
ROADMAP_SHA256 = "b9db7e1aa76deaf28ce2e840cd3d7db4cf7a086b1c7ca473800780c264a0e772"
ATTEMPT1_PATH = "runs/gravity/roadmap/item-09-interior-exterior-v1.json"
ATTEMPT1_FILE_SHA256 = "e86833a9a97244d01856f5005b81849d14f59974a12be3994e2c95c829512ef3"
ATTEMPT1_CONTENT_SHA256 = "8aba65a3743ec95b7de1601461a7e710e6a0fb03ea4ca8a1b06fb8edca55c6b7"
ATTEMPT1_DECISION = "REJECT_ITEM9_INTERIOR_EXTERIOR_EXPLORATION"
ATTEMPT2_PATH = "runs/gravity/roadmap/item-09-probes2-zero-tuning-v1.json"
ATTEMPT2_FILE_SHA256 = "09e2896cf78bd9782835ebbc31d833261dd8293ff3ebef967937add41b37de4f"
ATTEMPT2_CONTENT_SHA256 = "5250b829f1704b6a0cc1fc7a6e5497d0feb3ca47ac6816762de40e139c9deda7"
ATTEMPT2_DECISION = "INCONCLUSIVE_ITEM9_PROBES2_QUALITY"
OUTPUT_PATH = "runs/gravity/roadmap/item-09-synthesis-v1.json"


class GravityItem9SynthesisError(RuntimeError):
    """Raised when an Item 9 input or synthesis invariant drifts."""


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_result(
    root: Path, path: str, file_sha256: str, content_sha256: str, decision: str, label: str
) -> dict[str, Any]:
    source = root / path
    if _sha256_file(source) != file_sha256:
        raise GravityItem9SynthesisError(f"{label} result file changed")
    result = json.loads(source.read_text(encoding="utf-8"))
    if result.get("content_sha256") != content_sha256:
        raise GravityItem9SynthesisError(f"{label} content binding changed")
    content = dict(result)
    content.pop("content_sha256", None)
    if canonical_sha256(content) != content_sha256:
        raise GravityItem9SynthesisError(f"{label} content hash changed")
    if result.get("decision") != decision:
        raise GravityItem9SynthesisError(f"{label} decision changed")
    if any(bool(value) for value in result["claims"].values()):
        raise GravityItem9SynthesisError(f"{label} contains an overclaim")
    return result


def build_receipt(root: Path) -> dict[str, Any]:
    root = root.resolve()
    if _sha256_file(root / ROADMAP_PATH) != ROADMAP_SHA256:
        raise GravityItem9SynthesisError("stable roadmap changed")
    attempt1 = _load_result(
        root,
        ATTEMPT1_PATH,
        ATTEMPT1_FILE_SHA256,
        ATTEMPT1_CONTENT_SHA256,
        ATTEMPT1_DECISION,
        "Item 9 attempt 1",
    )
    attempt2 = _load_result(
        root,
        ATTEMPT2_PATH,
        ATTEMPT2_FILE_SHA256,
        ATTEMPT2_CONTENT_SHA256,
        ATTEMPT2_DECISION,
        "Item 9 attempt 2",
    )
    if attempt1["gate_counts"] != {"passed": 11, "required": 12}:
        raise GravityItem9SynthesisError("attempt-1 gate count changed")
    if attempt2["gate_counts"] != {"passed": 6, "required": 15}:
        raise GravityItem9SynthesisError("attempt-2 gate count changed")
    if int(attempt1["counts"]["reserved_confirmation_rotation_entries_opened"]) != 0:
        raise GravityItem9SynthesisError("PROBES-I confirmation boundary opened")
    if int(attempt2["counts"]["candidate_selection_calls"]) != 0:
        raise GravityItem9SynthesisError("PROBES-II selected a replay formula")
    if int(attempt2["counts"]["post_response_formula_cells"]) != 0:
        raise GravityItem9SynthesisError("PROBES-II generated a response-derived formula")
    attempt1_gain = float(
        attempt1["primary"]["qualifying_selector"][
            "relative_mse_improvement_over_strongest_baseline"
        ]
    )
    attempt2_gain = float(
        attempt2["primary"]["relative_mse_improvement_over_strongest_baseline"]
    )
    if attempt1_gain <= 0.13 or attempt2_gain >= -0.66:
        raise GravityItem9SynthesisError("cross-dataset result pattern changed")
    source_gate = attempt2["gate_checks"][
        "primary_gain_positive_in_all_source_families_at_minimum_count"
    ]
    if source_gate:
        raise GravityItem9SynthesisError("attempt-2 source counterexample disappeared")

    receipt: dict[str, Any] = {
        "schema_version": "invariant-gravity-item9-synthesis-1.0",
        "goal": "GRAVITY_ROADMAP_ITEM_09_INTERIOR_EXTERIOR_BALANCE_SYNTHESIS",
        "item_number": 9,
        "decision": "REJECT_ITEM9_TESTED_STELLAR_LIGHT_OCCUPANCY_ADVANCE_ITEM10",
        "evidence": {
            "attempt1": {
                "path": ATTEMPT1_PATH,
                "file_sha256": ATTEMPT1_FILE_SHA256,
                "content_sha256": ATTEMPT1_CONTENT_SHA256,
                "decision": ATTEMPT1_DECISION,
                "quality_passing_galaxies": attempt1["counts"][
                    "quality_passing_exploration_galaxies"
                ],
                "points": attempt1["counts"]["exploration_points"],
                "relative_mse_improvement_over_strongest_baseline": attempt1[
                    "primary"
                ]["qualifying_selector"][
                    "relative_mse_improvement_over_strongest_baseline"
                ],
                "r2": attempt1["primary"]["qualifying_selector"]["metrics"]["r2"],
                "paired_p_value": attempt1["paired_sign_flip"]["p_value"],
                "failed_source_family": "SHIVir",
                "gates": attempt1["gate_counts"],
            },
            "attempt2": {
                "path": ATTEMPT2_PATH,
                "file_sha256": ATTEMPT2_FILE_SHA256,
                "content_sha256": ATTEMPT2_CONTENT_SHA256,
                "decision": ATTEMPT2_DECISION,
                "selected_galaxies": attempt2["counts"]["selected_galaxies"],
                "quality_passing_galaxies": attempt2["counts"][
                    "quality_passing_galaxies"
                ],
                "points": attempt2["counts"]["accepted_points"],
                "primary_mse": attempt2["primary"]["metrics"][
                    "primary_median_ensemble"
                ]["mse"],
                "fixed_stellar_rar_mse": attempt2["primary"]["metrics"][
                    "fixed_stellar_rar"
                ]["mse"],
                "strongest_baseline_mse": attempt2["primary"]["metrics"][
                    "strongest_baseline"
                ]["mse"],
                "relative_mse_improvement_over_strongest_baseline": attempt2[
                    "primary"
                ]["relative_mse_improvement_over_strongest_baseline"],
                "paired_p_value": attempt2["paired_sign_flip"]["p_value"],
                "attempt1_cells_beating_fixed_rar": attempt2["counts"][
                    "attempt1_atomic_cells_beating_fixed_stellar_rar"
                ],
                "gates": attempt2["gate_counts"],
            },
        },
        "scope_rejected": [
            "promotion of the exact stellar-light acceleration-occupancy I_in-I_out operator with the five inherited logistic amplitude maps as one universal galaxy law",
            "promotion of the exact earlier surface-brightness focusing cell as one universal galaxy law",
            "retuning either exact family on the 823 opened PROBES-I galaxies or 136 valid opened PROBES-II galaxies",
            "using dataset-specific amplitude maps, object identities, or opened alternate curves to rescue the tested formulas",
        ],
        "positive_structure_retained": [
            "all five PROBES-I outer folds selected the same acceleration-occupancy interior-minus-exterior operator",
            "the PROBES-I selector improved over its strongest local baseline by 13.30 percent",
            "all five inherited cells and the exact earlier SPARC cell improved over fixed stellar RAR in the untouched PROBES-II valid subset",
            "the nonlocal perturbation is a reproducible RAR correction but is not competitive with a flexible local profile law across both datasets",
        ],
        "failure_space": {
            "label": "FAILED_UNIVERSAL_STELLAR_LIGHT_OCCUPANCY_REGION",
            "counterexamples": [
                "the 28-galaxy SHIVir subset regresses in attempt 1",
                "the attempt-2 primary is 66.59 percent worse than the OOF flexible local control",
                "the attempt-2 primary loses in both halves of distance, stellar mass, surface density, and inclination",
                "three of four attempt-2 source families meeting the frozen count gate regress",
            ],
            "reuse_rule": (
                "Use these exact cells and equivalent rescalings for failure-space rejection. "
                "Reconsider interior/exterior physics only with materially new baryonic fields, "
                "operator structure, field equations, response type, or a genuinely independent "
                "frozen dataset; never by retuning an opened response."
            ),
        },
        "not_rejected": [
            "gas-inclusive and complete-baryon interior/exterior fields",
            "vector or tensor redirection of gravity toward baryon-rich regions",
            "action-derived nonlocal kernels, causal history kernels, or covariant boundary terms",
            "baryonic edges, shells, interfaces, and finite-domain terms assigned to roadmap Item 10",
            "dark matter, general relativity, or every modified-gravity theory",
        ],
        "why_item_complete": (
            "Attempt 1 provides a large quality-passing positive lead but a frozen survey "
            "counterexample. Attempt 2 then performs the required zero-tuning transfer with zero "
            "predecessor identity overlap. Its formal quality floor is inconclusive at 136 versus "
            "150 galaxies, so it cannot alone reject the family; however its valid 4,604-point "
            "diagnostic preserves the small RAR improvement while losing to the frozen flexible "
            "local control by 66.59 percent in every broad stratum. Retuning would reuse opened "
            "responses. The tested stellar-light occupancy region is therefore closed as a "
            "promoted universal law, while materially different nonlocal mechanisms remain open."
        ),
        "counts": {
            "attempts": 2,
            "attempt1_quality_galaxies": attempt1["counts"][
                "quality_passing_exploration_galaxies"
            ],
            "attempt2_quality_galaxies": attempt2["counts"]["quality_passing_galaxies"],
            "attempt1_points": attempt1["counts"]["exploration_points"],
            "attempt2_points": attempt2["counts"]["accepted_points"],
            "attempt1_formula_cells_screened": attempt1["counts"][
                "candidate_formula_cells"
            ],
            "attempt2_atomic_formula_cells": attempt2["counts"]["atomic_formula_cells"],
            "attempt2_ensemble_formula_cells": attempt2["counts"][
                "ensemble_formula_cells"
            ],
            "probes1_confirmation_entries_opened": attempt1["counts"][
                "reserved_confirmation_rotation_entries_opened"
            ],
            "probes2_alternate_rotation_entries_opened": 0,
            "post_response_formula_generation": 0,
            "paid_model_calls": 0,
        },
        "claim_boundaries": {
            "all_interior_exterior_gravity_theories_rejected": False,
            "tested_stellar_light_occupancy_family_promoted": False,
            "probes1_confirmation_opened": False,
            "probes2_alternate_curves_opened": False,
            "roadmap_item_9_complete": True,
            "roadmap_item_10_authorized_next": True,
            "alternative_to_gr_established": False,
            "historical_novelty_established": False,
        },
        "next_action": (
            "Advance to Item 10 baryonic boundaries. Before accessing a new response, freeze "
            "creative edge, shell, interface, finite-domain, and field-focusing terms that are "
            "not algebraically equivalent to the failed Item 9 occupancy cells. Do not open "
            "PROBES-I confirmations or PROBES-II alternate curves."
        ),
        "content_sha256": None,
    }
    content = dict(receipt)
    content.pop("content_sha256")
    receipt["content_sha256"] = canonical_sha256(content)
    return receipt


def write_receipt(root: Path) -> Path:
    root = root.resolve()
    path = root / OUTPUT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(build_receipt(root)) + b"\n")
    return path


def check_receipt(root: Path) -> None:
    root = root.resolve()
    stored = json.loads((root / OUTPUT_PATH).read_text(encoding="utf-8"))
    if stored != build_receipt(root):
        raise GravityItem9SynthesisError("Item 9 synthesis receipt drifted")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        check_receipt(args.root)
    else:
        print(write_receipt(args.root))


if __name__ == "__main__":
    main()
