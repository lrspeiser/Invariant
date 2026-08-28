"""Deterministic synthesis for gravity roadmap Item 14."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .sigma_core import canonical_json_bytes, canonical_sha256

ROADMAP_PATH = "docs/GRAVITY_HIDDEN_VARIABLE_AND_THEORY_SEARCH_GOALS.md"
ROADMAP_SHA256 = "b9db7e1aa76deaf28ce2e840cd3d7db4cf7a086b1c7ca473800780c264a0e772"
ATTEMPT_PATH = "runs/gravity/roadmap/item-14-gz3d-resonance-coherence-v1.json"
ATTEMPT_FILE_SHA256 = "484f8c2e3d6ab72f570acac9d346525a75109bb287282dfda2ad766769709935"
ATTEMPT_CONTENT_SHA256 = "e5061b71c8fa18d7788b09ea0d47dffbdd062bb7092dbe95458f8754f032cf98"
ATTEMPT_DECISION = "REJECT_ITEM14_GZ3D_RESONANCE_COHERENCE_EXPLORATION"
OUTPUT_PATH = "runs/gravity/roadmap/item-14-synthesis-v1.json"


class GravityItem14SynthesisError(RuntimeError):
    """Raised when an Item 14 input or synthesis invariant drifts."""


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_attempt(root: Path) -> dict[str, Any]:
    path = root / ATTEMPT_PATH
    if _sha256_file(path) != ATTEMPT_FILE_SHA256:
        raise GravityItem14SynthesisError("Item 14 attempt file changed")
    attempt = json.loads(path.read_text(encoding="utf-8"))
    content = dict(attempt)
    content_sha256 = content.pop("content_sha256", None)
    if content_sha256 != ATTEMPT_CONTENT_SHA256:
        raise GravityItem14SynthesisError("Item 14 content binding changed")
    if canonical_sha256(content) != ATTEMPT_CONTENT_SHA256:
        raise GravityItem14SynthesisError("Item 14 content hash changed")
    if attempt.get("decision") != ATTEMPT_DECISION:
        raise GravityItem14SynthesisError("Item 14 decision changed")
    if any(bool(value) for value in attempt["claims"].values()):
        raise GravityItem14SynthesisError("Item 14 attempt contains an overclaim")
    return attempt


def build_receipt(root: Path) -> dict[str, Any]:
    root = root.resolve()
    if _sha256_file(root / ROADMAP_PATH) != ROADMAP_SHA256:
        raise GravityItem14SynthesisError("stable roadmap changed")
    attempt = _load_attempt(root)
    if attempt["gate_counts"] != {"passed": 7, "required": 14}:
        raise GravityItem14SynthesisError("Item 14 gate count changed")
    if int(attempt["counts"]["candidate_cells"]) != 262144:
        raise GravityItem14SynthesisError("Item 14 candidate count changed")
    if int(attempt["counts"]["quality_passing_galaxies"]) != 204:
        raise GravityItem14SynthesisError("Item 14 quality count changed")
    for key in (
        "confirmation_response_rows",
        "post_response_formula_cells",
        "paid_model_calls",
    ):
        if int(attempt["counts"][key]) != 0:
            raise GravityItem14SynthesisError(f"Item 14 forbidden count changed: {key}")
    primary = attempt["primary_stellar_outer_to_inner_log_span_ratio"]
    secondary = attempt["secondary_halpha_outer_to_inner_log_span_ratio"]
    paired = attempt["paired_sign_flip"]
    if float(primary["relative_mse_improvement"]) >= 0:
        raise GravityItem14SynthesisError("Item 14 stellar rejection changed")
    if float(secondary["relative_mse_improvement"]) >= 0:
        raise GravityItem14SynthesisError("Item 14 H-alpha transfer rejection changed")
    if float(paired["p_value"]) <= 0.05:
        raise GravityItem14SynthesisError("Item 14 paired null changed")
    expected_failed_gates = {
        "coherence_beats_control_baseline",
        "coherence_paired_p_at_most",
        "coherence_relative_mse_improvement_at_least",
        "gain_positive_in_both_bar_vote_states",
        "gain_positive_in_both_prior_age_halves",
        "gain_positive_in_both_stellar_mass_halves",
        "secondary_halpha_transfer_beats_control",
    }
    failed_gates = {key for key, value in attempt["gate_checks"].items() if not value}
    if failed_gates != expected_failed_gates:
        raise GravityItem14SynthesisError("Item 14 failed-gate set changed")

    receipt: dict[str, Any] = {
        "schema_version": "invariant-gravity-item14-synthesis-1.0",
        "goal": "GRAVITY_ROADMAP_ITEM_14_RESONANCE_AND_COHERENCE_SYNTHESIS",
        "item_number": 14,
        "decision": "REJECT_ITEM14_MASK_COHERENCE_ADVANCE_ITEM15",
        "evidence": {
            "attempt": {
                "path": ATTEMPT_PATH,
                "file_sha256": ATTEMPT_FILE_SHA256,
                "content_sha256": ATTEMPT_CONTENT_SHA256,
                "decision": ATTEMPT_DECISION,
                "candidate_cells": attempt["counts"]["candidate_cells"],
                "quality_passing_galaxies": attempt["counts"]["quality_passing_galaxies"],
                "quality_failed_galaxies": attempt["counts"]["quality_failed_galaxies"],
                "primary_control_baseline": primary["control_baseline"],
                "primary_selected_full_model": primary[
                    "selected_mask_geometry_full_model"
                ],
                "primary_relative_mse_improvement": primary["relative_mse_improvement"],
                "paired_sign_flip": paired,
                "secondary_halpha": secondary,
                "resolved_ratio_distribution": attempt["resolved_ratio_distribution"],
                "gates": attempt["gate_counts"],
                "failed_gates": sorted(failed_gates),
                "fold_selections": primary["outer_fold_selections"],
                "strata": attempt["strata"],
                "compute": attempt["compute"],
            }
        },
        "tested_family": {
            "status": "REJECTED_AS_INCREMENTAL_PREDICTOR_IN_TESTED_MASK_GEOMETRY_SCOPE",
            "scope": (
                "The rejection covers the frozen rotation/reflection-invariant GZ3D spiral and "
                "bar mask amplitudes, phase relationships, pitch/twist summaries, radial mode "
                "persistence, harmonic coupling, entropy suppression, and declared nonlinear "
                "combinations for resolved annular line-of-sight velocity-span ratios."
            ),
            "observed_pattern": (
                "Nested selection worsens stellar held-out MSE by 2.58 percent with paired "
                "p=0.827. The same fold-selected cells worsen H-alpha MSE by 0.10 percent without "
                "candidate reselection. Five of six broad strata regress, and the only positive "
                "stratum gain is negligible."
            ),
            "boundary": (
                "Static image masks do not measure temporal pattern speed, corotation, Lindblad "
                "radii, long-lived resonance, deprojected circular speed, or a gravitational "
                "field equation. Those materially different representations remain open."
            ),
        },
        "retained_observation": {
            "label": "RESOLVED_OUTER_TO_INNER_LINE_OF_SIGHT_VELOCITY_SPAN_PATTERN",
            "status": "DESCRIPTIVE_TARGET_FOR_TIMESCALE_TEST_NOT_FORMULA_LEAD",
            "observed_pattern": (
                "Across 204 quality galaxies, the median outer-to-inner stellar velocity-span "
                "ratio is 1.356 and the H-alpha median is 1.231. Only 14.7 percent of stellar "
                "ratios and 40.2 percent of H-alpha ratios lie within 20 percent of unity."
            ),
            "boundary": (
                "This establishes a descriptive annular line-of-sight pattern in the selected "
                "sample, not flat circular rotation, identical star speeds, resonance, or an "
                "alternative-gravity mechanism."
            ),
            "reuse_rule": (
                "Do not retune on the 204 opened Item 14 responses or open the 80 sealed Item 14 "
                "confirmations. Test causal timescale variables on fresh identities or a "
                "materially independent response after freezing the Item 15 grammar."
            ),
        },
        "counterexamples_and_boundaries": [
            "36 of 240 exploration galaxies fail frozen response-quality checks",
            "four outer folds select a bar-arm-radius family but one selects four-arm amplitude",
            "one of five fitted stellar coefficients reverses sign",
            "the mask component regresses in both mass halves and both age halves",
            "the H-alpha transfer regresses without candidate reselection",
            "morphology and kinematics come from the same SDSS MaNGA ecosystem",
            "annular line-of-sight spans are not deprojected circular speeds or individual-star velocities",
        ],
        "not_established": [
            "that orbital resonance or collective modes never affect galaxy dynamics",
            "temporal pattern speed, corotation, or a cosmic-time synchronization mechanism",
            "a causal age-dependent baryonic mass correction",
            "a historically new formula",
            "a modification of gravity or alternative to general relativity",
            "prediction of galaxy clusters or gravitational lensing",
        ],
        "why_item_complete": (
            "The exact fresh-identity test passed every source, quality, boundary, and accounting "
            "gate; scored all 262,144 frozen cells with nested held-out selection; transferred "
            "the selected cells to H-alpha without reselection; and returned a scoped rejection. "
            "No confirmation response or post-response formula was opened, so Item 15 is the "
            "next numbered roadmap test."
        ),
        "counts": {
            "attempts": 1,
            "candidate_formula_cells": attempt["counts"]["candidate_cells"],
            "candidate_galaxy_score_evaluations": attempt["compute"][
                "candidate_galaxy_score_evaluations"
            ],
            "quality_galaxies": attempt["counts"]["quality_passing_galaxies"],
            "confirmation_rows_opened": attempt["counts"]["confirmation_response_rows"],
            "post_response_formula_generation": attempt["counts"][
                "post_response_formula_cells"
            ],
            "paid_model_calls": attempt["counts"]["paid_model_calls"],
        },
        "claim_boundaries": {
            "static_mask_geometry_family_rejected_in_scope": True,
            "resolved_annular_span_pattern_observed": True,
            "temporal_resonance_cause_established": False,
            "age_dependent_baryonic_mass_error_established": False,
            "confirmation_opened": False,
            "roadmap_item_14_complete": True,
            "roadmap_item_15_authorized_next": True,
            "alternative_to_gr_established": False,
            "historical_novelty_established": False,
        },
        "next_action": (
            "Advance to Item 15 timescale ratios on fresh real identities. Before response access, "
            "freeze dimensionless orbital, crossing, free-fall, star-formation, cooling, settling, "
            "and cosmic-time ratios plus null controls; test whether one universal timescale law "
            "predicts resolved outer/inner motion and transfers to an independent tracer."
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
        raise GravityItem14SynthesisError("Item 14 synthesis receipt drifted")


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
