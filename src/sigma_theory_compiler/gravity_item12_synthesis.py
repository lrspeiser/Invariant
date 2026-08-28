"""Deterministic synthesis for gravity roadmap Item 12."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .sigma_core import canonical_json_bytes, canonical_sha256

ROADMAP_PATH = "docs/GRAVITY_HIDDEN_VARIABLE_AND_THEORY_SEARCH_GOALS.md"
ROADMAP_SHA256 = "b9db7e1aa76deaf28ce2e840cd3d7db4cf7a086b1c7ca473800780c264a0e772"
ATTEMPT_PATH = "runs/gravity/roadmap/item-12-manga-dynamical-age-v1.json"
ATTEMPT_FILE_SHA256 = "d134a8c9f5cd3e87d25bdb0cf38b390ca45ac85a9387abae412d95b5f109d2bc"
ATTEMPT_CONTENT_SHA256 = "d2638ae1c05f4fa96124b91c79aec6f73a73a225eab1b86b772fd91cdd34f62c"
ATTEMPT_DECISION = "PASS_ITEM12_MANGA_DYNAMICAL_AGE_EXPLORATION"
OUTPUT_PATH = "runs/gravity/roadmap/item-12-synthesis-v1.json"


class GravityItem12SynthesisError(RuntimeError):
    """Raised when an Item 12 input or synthesis invariant drifts."""


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_attempt(root: Path) -> dict[str, Any]:
    path = root / ATTEMPT_PATH
    if _sha256_file(path) != ATTEMPT_FILE_SHA256:
        raise GravityItem12SynthesisError("Item 12 attempt file changed")
    attempt = json.loads(path.read_text(encoding="utf-8"))
    content = dict(attempt)
    content_sha256 = content.pop("content_sha256", None)
    if content_sha256 != ATTEMPT_CONTENT_SHA256:
        raise GravityItem12SynthesisError("Item 12 content binding changed")
    if canonical_sha256(content) != ATTEMPT_CONTENT_SHA256:
        raise GravityItem12SynthesisError("Item 12 content hash changed")
    if attempt.get("decision") != ATTEMPT_DECISION:
        raise GravityItem12SynthesisError("Item 12 decision changed")
    if any(bool(value) for value in attempt["claims"].values()):
        raise GravityItem12SynthesisError("Item 12 attempt contains an overclaim")
    return attempt


def build_receipt(root: Path) -> dict[str, Any]:
    root = root.resolve()
    if _sha256_file(root / ROADMAP_PATH) != ROADMAP_SHA256:
        raise GravityItem12SynthesisError("stable roadmap changed")
    attempt = _load_attempt(root)
    if attempt["gate_counts"] != {"passed": 13, "required": 13}:
        raise GravityItem12SynthesisError("Item 12 gate count changed")
    if int(attempt["counts"]["candidate_cells"]) != 262144:
        raise GravityItem12SynthesisError("Item 12 candidate count changed")
    if int(attempt["counts"]["quality_passing_galaxies"]) != 585:
        raise GravityItem12SynthesisError("Item 12 quality count changed")
    if int(attempt["counts"]["confirmation_response_rows"]) != 0:
        raise GravityItem12SynthesisError("Item 12 confirmation boundary opened")
    if int(attempt["counts"]["post_response_formula_cells"]) != 0:
        raise GravityItem12SynthesisError("post-response formula entered Item 12")
    if float(attempt["primary"]["relative_mse_improvement"]) <= 0.18:
        raise GravityItem12SynthesisError("Item 12 positive diagnostic changed")
    if float(attempt["paired_sign_flip"]["p_value"]) != 0.001:
        raise GravityItem12SynthesisError("Item 12 paired diagnostic changed")
    selections = attempt["primary"]["outer_fold_selections"]
    if {row["selected_family"] for row in selections} != {"spectral_clock_consensus"}:
        raise GravityItem12SynthesisError("Item 12 fold-family consensus changed")
    if {row["modulation"] for row in selections} != {"stellar_surface_density"}:
        raise GravityItem12SynthesisError("Item 12 fold modulation consensus changed")

    receipt: dict[str, Any] = {
        "schema_version": "invariant-gravity-item12-synthesis-1.0",
        "goal": "GRAVITY_ROADMAP_ITEM_12_DYNAMICAL_AGE_SYNTHESIS",
        "item_number": 12,
        "decision": "PASS_ITEM12_EXPLORATION_LEAD_RETAINED_ADVANCE_ITEM13",
        "evidence": {
            "attempt": {
                "path": ATTEMPT_PATH,
                "file_sha256": ATTEMPT_FILE_SHA256,
                "content_sha256": ATTEMPT_CONTENT_SHA256,
                "decision": ATTEMPT_DECISION,
                "candidate_cells": attempt["counts"]["candidate_cells"],
                "quality_passing_galaxies": attempt["counts"]["quality_passing_galaxies"],
                "quality_failed_galaxies": attempt["counts"]["quality_failed_galaxies"],
                "structural_baseline": attempt["primary"]["structural_baseline"],
                "selected_dynamical_clock": attempt["primary"]["selected_dynamical_clock"],
                "relative_mse_improvement": attempt["primary"]["relative_mse_improvement"],
                "paired_p_value": attempt["paired_sign_flip"]["p_value"],
                "gates": attempt["gate_counts"],
                "fold_selections": selections,
                "strata": attempt["strata"],
                "compute": attempt["compute"],
            }
        },
        "retained_lead": {
            "label": "SPECTRAL_CLOCK_CONSENSUS_TIMES_STELLAR_SURFACE_DENSITY",
            "origin_status": "COMBINATION",
            "status": "PROMOTED_TO_INDEPENDENT_CONFIRMATION_QUEUE",
            "observed_pattern": (
                "All five outer folds select a signed consensus of fixed-normalized Dn4000, "
                "D4000, Balmer, and H-beta clocks multiplied by fixed-normalized stellar "
                "surface density. Fitted additive log-dispersion coefficients are positive in "
                "all folds."
            ),
            "parameter_boundary": (
                "The mechanism and modulation are stable, but exact threshold, scale, and power "
                "are not identical across folds. No single post-response parameter cell is promoted."
            ),
            "reuse_rule": (
                "Do not retune on the 585 opened responses or open the 250 sealed confirmations. "
                "Any confirmation must freeze a deterministic family-level parameter rule before a "
                "fresh response and must compare against stellar-population mass-systematic and "
                "ordinary-relaxation controls."
            ),
        },
        "counterexamples_and_boundaries": [
            "165 of 750 exploration galaxies fail frozen response-quality checks",
            "the selected exact parameter ordinal changes across outer folds",
            "integrated spectral indices do not directly measure formation or settling time",
            "stellar-population-dependent mass-to-light errors can mimic added dynamical information",
            "ordinary merger and relaxation history can generate the same association without modified gravity",
            "the target is integrated stellar velocity dispersion, not a galaxy rotation curve, cluster, or lensing map",
        ],
        "not_established": [
            "a causal dynamical-age effect",
            "a historically new formula",
            "a modification of gravity or alternative to general relativity",
            "prediction of galaxy rotation curves, galaxy-cluster dynamics, or gravitational lensing",
            "independent confirmation of the retained family",
        ],
        "why_item_complete": (
            "The frozen real-data family passed all 13 exploration gates on 585 MaNGA galaxies, "
            "including every preregistered stratum, after 262,144 nested-selected formulas. The "
            "positive family is retained without opening confirmation or synthesizing a new exact "
            "formula from responses. Item 13 is the required causal-disambiguation step."
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
            "dynamical_age_cause_established": False,
            "retained_family_independently_confirmed": False,
            "confirmation_opened": False,
            "roadmap_item_12_complete": True,
            "roadmap_item_13_authorized_next": True,
            "alternative_to_gr_established": False,
            "historical_novelty_established": False,
        },
        "next_action": (
            "Advance to Item 13 relaxation and mergers on a fresh response. Freeze disturbance, "
            "asymmetry, close-pair, merger-stage, and kinematic-relaxation predictors before response "
            "access; test whether they explain or preserve the retained spectral-clock association."
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
        raise GravityItem12SynthesisError("Item 12 synthesis receipt drifted")


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
