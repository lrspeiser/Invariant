"""Deterministic synthesis for gravity roadmap Item 11."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .sigma_core import canonical_json_bytes, canonical_sha256

ROADMAP_PATH = "docs/GRAVITY_HIDDEN_VARIABLE_AND_THEORY_SEARCH_GOALS.md"
ROADMAP_SHA256 = "b9db7e1aa76deaf28ce2e840cd3d7db4cf7a086b1c7ca473800780c264a0e772"
ATTEMPT_PATH = "runs/gravity/roadmap/item-11-neargalcat-external-field-v1.json"
ATTEMPT_FILE_SHA256 = "4042cffcbd5dccd231997cacdede83154e1ad65d004ef48b74cf98a9ffd588a4"
ATTEMPT_CONTENT_SHA256 = "3f532a0d6c94de8a3c3d43f38047a00298bda644ba2f8fb47d94ed0a2dc1a0e0"
ATTEMPT_DECISION = "INCONCLUSIVE_ITEM11_NEARGALCAT_QUALITY"
OUTPUT_PATH = "runs/gravity/roadmap/item-11-synthesis-v1.json"


class GravityItem11SynthesisError(RuntimeError):
    """Raised when an Item 11 input or synthesis invariant drifts."""


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_attempt(root: Path) -> dict[str, Any]:
    path = root / ATTEMPT_PATH
    if _sha256_file(path) != ATTEMPT_FILE_SHA256:
        raise GravityItem11SynthesisError("Item 11 attempt file changed")
    attempt = json.loads(path.read_text(encoding="utf-8"))
    content = dict(attempt)
    content_sha256 = content.pop("content_sha256", None)
    if content_sha256 != ATTEMPT_CONTENT_SHA256:
        raise GravityItem11SynthesisError("Item 11 content binding changed")
    if canonical_sha256(content) != ATTEMPT_CONTENT_SHA256:
        raise GravityItem11SynthesisError("Item 11 content hash changed")
    if attempt.get("decision") != ATTEMPT_DECISION:
        raise GravityItem11SynthesisError("Item 11 decision changed")
    if any(bool(value) for value in attempt["claims"].values()):
        raise GravityItem11SynthesisError("Item 11 attempt contains an overclaim")
    return attempt


def build_receipt(root: Path) -> dict[str, Any]:
    root = root.resolve()
    if _sha256_file(root / ROADMAP_PATH) != ROADMAP_SHA256:
        raise GravityItem11SynthesisError("stable roadmap changed")
    attempt = _load_attempt(root)
    if attempt["gate_counts"] != {"passed": 6, "required": 14}:
        raise GravityItem11SynthesisError("Item 11 gate count changed")
    if int(attempt["counts"]["candidate_cells"]) != 262144:
        raise GravityItem11SynthesisError("Item 11 candidate count changed")
    if int(attempt["counts"]["quality_passing_galaxies"]) != 119:
        raise GravityItem11SynthesisError("Item 11 quality count changed")
    if int(attempt["counts"]["confirmation_response_rows"]) != 0:
        raise GravityItem11SynthesisError("Item 11 confirmation boundary opened")
    if int(attempt["counts"]["post_response_formula_cells"]) != 0:
        raise GravityItem11SynthesisError("post-response formula entered Item 11")
    if float(attempt["primary"]["relative_mse_improvement"]) >= -0.13:
        raise GravityItem11SynthesisError("Item 11 negative diagnostic changed")
    if float(attempt["paired_sign_flip"]["p_value"]) != 0.99:
        raise GravityItem11SynthesisError("Item 11 paired diagnostic changed")

    receipt: dict[str, Any] = {
        "schema_version": "invariant-gravity-item11-synthesis-1.0",
        "goal": "GRAVITY_ROADMAP_ITEM_11_EXTERNAL_BARYONIC_FIELD_SYNTHESIS",
        "item_number": 11,
        "decision": "INCONCLUSIVE_ITEM11_QUALITY_NEGATIVE_DIRECTION_ADVANCE_ITEM12",
        "evidence": {
            "attempt": {
                "path": ATTEMPT_PATH,
                "file_sha256": ATTEMPT_FILE_SHA256,
                "content_sha256": ATTEMPT_CONTENT_SHA256,
                "decision": ATTEMPT_DECISION,
                "candidate_cells": attempt["counts"]["candidate_cells"],
                "quality_passing_galaxies": attempt["counts"]["quality_passing_galaxies"],
                "quality_failed_galaxies": attempt["counts"]["quality_failed_galaxies"],
                "internal_baryon_baseline": attempt["primary"]["internal_baryon_baseline"],
                "selected_external_field": attempt["primary"]["selected_external_field"],
                "relative_mse_improvement": attempt["primary"]["relative_mse_improvement"],
                "paired_p_value": attempt["paired_sign_flip"]["p_value"],
                "gates": attempt["gate_counts"],
                "compute": attempt["compute"],
            }
        },
        "scope_not_promoted": [
            "the exact seeded scalar transforms of published Theta1, Theta5, and one-megaparsec K-band luminosity density",
            "the tested dominance, isolation, suppression, resonance, coherence, saddle, and log-periodic combinations",
            "one universal additive log-rotation coefficient after the fixed internal-baryon baseline",
            "retuning exact or algebraically equivalent cells on the 119 opened valid galaxies",
        ],
        "failure_space": {
            "label": "NONPROMOTED_PUBLISHED_SCALAR_ENVIRONMENT_SUMMARY_REGION",
            "counterexamples": [
                "the selected terms are 13.64 percent worse than the internal-baryon baseline",
                "paired sign-flip p is 0.990",
                "both group and isolated strata regress",
                "both halves of baryonic mass, gas fraction, and distance regress",
                "fold-selected mechanisms vary and fitted coefficients change sign",
            ],
            "reuse_rule": (
                "Reject exact algebraic duplicates without rescoring. Reconsider environment only "
                "with materially new neighbor vectors and masses, tidal tensors, filament geometry, "
                "causal history, field equations, or joint dynamics and lensing responses."
            ),
        },
        "not_rejected": [
            "three-dimensional neighbor mass geometry and vector tidal fields",
            "large-scale filaments, void boundaries, retarded external fields, and environment history",
            "action-derived external-field effects with conservation and lensing predictions",
            "general relativity, dark matter, or every modified-gravity theory",
        ],
        "why_item_complete": (
            "The frozen family faced 119 wholly independent quality-passing galaxies and 262,144 "
            "nested-selected formulas. Although the source misses the frozen 150-galaxy and 60-percent "
            "quality floors, every broad valid-set diagnostic is negative. Retuning would reuse opened "
            "responses, while 82 confirmations remain sealed. Item 11 is closed without a promoted "
            "lead, preserving materially different reconstructed or field-equation environments."
        ),
        "counts": {
            "attempts": 1,
            "candidate_formula_cells": attempt["counts"]["candidate_cells"],
            "candidate_galaxy_score_evaluations": attempt["compute"][
                "candidate_galaxy_score_evaluations"
            ],
            "quality_galaxies": attempt["counts"]["quality_passing_galaxies"],
            "confirmation_rows_opened": attempt["counts"]["confirmation_response_rows"],
            "post_response_formula_generation": attempt["counts"]["post_response_formula_cells"],
            "paid_model_calls": attempt["counts"]["paid_model_calls"],
        },
        "claim_boundaries": {
            "all_external_field_theories_rejected": False,
            "tested_scalar_environment_family_promoted": False,
            "confirmation_opened": False,
            "roadmap_item_11_complete": True,
            "roadmap_item_12_authorized_next": True,
            "alternative_to_gr_established": False,
            "historical_novelty_established": False,
        },
        "next_action": (
            "Advance to Item 12 dynamical age on a fresh response. Freeze stellar-population age, "
            "specific-star-formation, gas-depletion, orbital-settling, and relaxation proxies before "
            "response access; keep mass, morphology, distance, and survey variables as controls."
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
        raise GravityItem11SynthesisError("Item 11 synthesis receipt drifted")


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
