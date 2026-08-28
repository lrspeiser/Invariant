"""Deterministic synthesis for gravity roadmap Item 10."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .sigma_core import canonical_json_bytes, canonical_sha256

ROADMAP_PATH = "docs/GRAVITY_HIDDEN_VARIABLE_AND_THEORY_SEARCH_GOALS.md"
ROADMAP_SHA256 = "b9db7e1aa76deaf28ce2e840cd3d7db4cf7a086b1c7ca473800780c264a0e772"
ATTEMPT_PATH = "runs/gravity/roadmap/item-10-wallaby-boundaries-v1.json"
ATTEMPT_FILE_SHA256 = "6e711d42fa092c3f426352449ddc046cce041bc19f3972b6cf65b16d3f0918df"
ATTEMPT_CONTENT_SHA256 = "d75b7bfa9927acf18cfdafe0c2169b9bd454b8765e6607355f2fd4855b77dcd7"
ATTEMPT_DECISION = "INCONCLUSIVE_ITEM10_WALLABY_QUALITY"
OUTPUT_PATH = "runs/gravity/roadmap/item-10-synthesis-v1.json"


class GravityItem10SynthesisError(RuntimeError):
    """Raised when an Item 10 input or synthesis invariant drifts."""


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_attempt(root: Path) -> dict[str, Any]:
    path = root / ATTEMPT_PATH
    if _sha256_file(path) != ATTEMPT_FILE_SHA256:
        raise GravityItem10SynthesisError("Item 10 attempt file changed")
    attempt = json.loads(path.read_text(encoding="utf-8"))
    if attempt.get("content_sha256") != ATTEMPT_CONTENT_SHA256:
        raise GravityItem10SynthesisError("Item 10 attempt content binding changed")
    content = dict(attempt)
    content.pop("content_sha256", None)
    if canonical_sha256(content) != ATTEMPT_CONTENT_SHA256:
        raise GravityItem10SynthesisError("Item 10 attempt content hash changed")
    if attempt.get("decision") != ATTEMPT_DECISION:
        raise GravityItem10SynthesisError("Item 10 attempt decision changed")
    forbidden_claims = (
        "complete_baryonic_mass_used",
        "boundary_gravity_cause_established",
        "alternative_to_gr_established",
        "historical_novelty_established",
        "roadmap_item_10_complete",
    )
    if any(bool(attempt["claims"][key]) for key in forbidden_claims):
        raise GravityItem10SynthesisError("Item 10 attempt contains an overclaim")
    return attempt


def build_receipt(root: Path) -> dict[str, Any]:
    root = root.resolve()
    if _sha256_file(root / ROADMAP_PATH) != ROADMAP_SHA256:
        raise GravityItem10SynthesisError("stable roadmap changed")
    attempt = _load_attempt(root)
    if attempt["gate_counts"] != {"passed": 5, "required": 14}:
        raise GravityItem10SynthesisError("Item 10 gate count changed")
    if int(attempt["counts"]["candidate_cells"]) != 131072:
        raise GravityItem10SynthesisError("Item 10 candidate count changed")
    if int(attempt["counts"]["post_response_formula_cells"]) != 0:
        raise GravityItem10SynthesisError("post-response formula entered Item 10")
    if int(attempt["counts"]["stored_confirmation_response_rows"]) != 0:
        raise GravityItem10SynthesisError("stored confirmation entered Item 10")
    if int(attempt["counts"]["confirmation_response_rows"]) != 2:
        raise GravityItem10SynthesisError("scope incident disclosure changed")
    if not attempt["claims"]["confirmation_opened"]:
        raise GravityItem10SynthesisError("scope incident claim disappeared")
    if int(attempt["counts"]["quality_passing_galaxies"]) != 20:
        raise GravityItem10SynthesisError("quality count changed")
    if float(attempt["primary"]["relative_mse_improvement"]) >= 0:
        raise GravityItem10SynthesisError("negative diagnostic direction changed")
    if float(attempt["paired_sign_flip"]["p_value"]) < 0.58:
        raise GravityItem10SynthesisError("paired null diagnostic changed")

    receipt: dict[str, Any] = {
        "schema_version": "invariant-gravity-item10-synthesis-1.0",
        "goal": "GRAVITY_ROADMAP_ITEM_10_BARYONIC_BOUNDARIES_SYNTHESIS",
        "item_number": 10,
        "decision": "INCONCLUSIVE_ITEM10_SOURCE_QUALITY_NEGATIVE_DIRECTION_ADVANCE_ITEM11",
        "evidence": {
            "attempt": {
                "path": ATTEMPT_PATH,
                "file_sha256": ATTEMPT_FILE_SHA256,
                "content_sha256": ATTEMPT_CONTENT_SHA256,
                "decision": ATTEMPT_DECISION,
                "candidate_cells": attempt["counts"]["candidate_cells"],
                "quality_passing_galaxies": attempt["counts"]["quality_passing_galaxies"],
                "quality_failed_galaxies": attempt["counts"]["quality_failed_galaxies"],
                "points": attempt["counts"]["accepted_points"],
                "local_baseline": attempt["primary"]["local_baseline"],
                "selected_boundary": attempt["primary"]["selected_boundary"],
                "relative_mse_improvement": attempt["primary"]["relative_mse_improvement"],
                "paired_p_value": attempt["paired_sign_flip"]["p_value"],
                "gates": attempt["gate_counts"],
                "compute": attempt["compute"],
            }
        },
        "scope_not_promoted": [
            "the exact twelve projected-HI scalar boundary families and seeded parameter ranges in the Item 10 candidate manifest",
            "one universal additive log-speed coefficient applied to the tested edge, shell, interface, finite-domain, and oscillatory kernels",
            "retuning the exact cells on the twenty opened quality-passing WALLABY galaxies",
            "algebraic sign flips, rescalings, and renamed copies that add no new physical information",
        ],
        "failure_space": {
            "label": "NONPROMOTED_LOW_QUALITY_PROJECTED_HI_BOUNDARY_REGION",
            "counterexamples": [
                "only 20 of 38 unambiguous exploration galaxies pass quality, versus the frozen minimum of 140",
                "quality retention is 52.63 percent, below the frozen 60 percent floor",
                "the selected boundary terms increase held-out MSE by 1.69 percent relative to the local HI baseline",
                "edge-radius, edge-sharpness, and profile-mass stratification gates fail",
                "the paired sign-flip test gives p=0.583",
            ],
            "reuse_rule": (
                "Reject exact algebraic duplicates without rescoring. Do not treat the low-quality "
                "diagnostic as proof against boundary gravity. A retry requires a larger independent "
                "source or materially different complete-baryon, vector/tensor, action-derived, "
                "history-dependent, or lensing-predictive mechanism."
            ),
        },
        "scope_incident": {
            "description": (
                "WALLABY names repeat across kinematic releases. A first exact-name query returned "
                "a non-one-to-one scope and wrote no response artifact. Two rows originally assigned "
                "to release-level confirmation may have been transmitted because the same physical "
                "names also had exploration releases. No response value was used in the repair."
            ),
            "potential_confirmation_rows_transmitted": attempt["counts"][
                "confirmation_response_rows"
            ],
            "stored_confirmation_rows": attempt["counts"]["stored_confirmation_response_rows"],
            "clean_unambiguous_confirmation_names_remaining_sealed": 11,
            "clean_confirmation_claimed": False,
        },
        "not_rejected": [
            "complete stellar, gas, plasma, and three-dimensional baryonic boundaries",
            "vector or tensor focusing of a gravitational field toward baryon-rich regions",
            "covariant boundary actions, conservation identities, and causal history kernels",
            "one field that jointly predicts galaxy dynamics and gravitational lensing",
            "general relativity, dark matter, or every modified-gravity theory",
        ],
        "why_item_complete": (
            "The frozen family faced a fresh resolved real-data problem and a nested held-out "
            "search of 131,072 formula cells. The source cannot meet its preregistered quality "
            "floor, and the valid diagnostic subset is negative relative to its strongest local "
            "baseline. Retuning would reuse opened responses, while opening the eleven remaining "
            "clean confirmation names cannot repair the 140-galaxy floor. Item 10 is therefore "
            "closed as inconclusive with no promoted boundary lead, preserving materially different "
            "boundary mechanisms for later action, lensing, and field-equation items."
        ),
        "counts": {
            "attempts": 1,
            "candidate_formula_cells": attempt["counts"]["candidate_cells"],
            "candidate_point_score_evaluations": attempt["compute"][
                "candidate_point_score_evaluations"
            ],
            "quality_galaxies": attempt["counts"]["quality_passing_galaxies"],
            "points": attempt["counts"]["accepted_points"],
            "clean_confirmation_names_remaining_sealed": 11,
            "scope_incident_potential_confirmation_rows": attempt["counts"][
                "confirmation_response_rows"
            ],
            "stored_confirmation_rows": attempt["counts"]["stored_confirmation_response_rows"],
            "post_response_formula_generation": attempt["counts"]["post_response_formula_cells"],
            "paid_model_calls": attempt["counts"]["paid_model_calls"],
        },
        "claim_boundaries": {
            "all_baryonic_boundary_theories_rejected": False,
            "tested_projected_hi_boundary_family_promoted": False,
            "item10_clean_confirmation_experiment": False,
            "roadmap_item_10_complete": True,
            "roadmap_item_11_authorized_next": True,
            "alternative_to_gr_established": False,
            "historical_novelty_established": False,
        },
        "next_action": (
            "Advance to Item 11 external baryonic field. On a fresh response, freeze neighboring-"
            "baryon density, nearest-baryon, tidal-tensor, filament, and large-scale-boundary "
            "variables before response access. Separate environment from distance, survey, group, "
            "and population labels; do not open the eleven clean WALLABY confirmation names."
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
        raise GravityItem10SynthesisError("Item 10 synthesis receipt drifted")


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
