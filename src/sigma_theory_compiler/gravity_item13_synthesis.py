"""Deterministic synthesis for gravity roadmap Item 13."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .sigma_core import canonical_json_bytes, canonical_sha256

ROADMAP_PATH = "docs/GRAVITY_HIDDEN_VARIABLE_AND_THEORY_SEARCH_GOALS.md"
ROADMAP_SHA256 = "b9db7e1aa76deaf28ce2e840cd3d7db4cf7a086b1c7ca473800780c264a0e772"
ATTEMPT_PATH = "runs/gravity/roadmap/item-13-manga-relaxation-mergers-v1.json"
ATTEMPT_FILE_SHA256 = "de52b4d84fd93925ef8f42c6625e33d9ae530b46584a9561d35a4ef77438310a"
ATTEMPT_CONTENT_SHA256 = "d92883fec791fd8a2ef4514584676a3d36c1ee53d1691e6f92c7a7334f6d9f0b"
ATTEMPT_DECISION = "REJECT_ITEM13_MANGA_RELAXATION_AND_MERGERS_EXPLORATION"
OUTPUT_PATH = "runs/gravity/roadmap/item-13-synthesis-v1.json"


class GravityItem13SynthesisError(RuntimeError):
    """Raised when an Item 13 input or synthesis invariant drifts."""


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_attempt(root: Path) -> dict[str, Any]:
    path = root / ATTEMPT_PATH
    if _sha256_file(path) != ATTEMPT_FILE_SHA256:
        raise GravityItem13SynthesisError("Item 13 attempt file changed")
    attempt = json.loads(path.read_text(encoding="utf-8"))
    content = dict(attempt)
    content_sha256 = content.pop("content_sha256", None)
    if content_sha256 != ATTEMPT_CONTENT_SHA256:
        raise GravityItem13SynthesisError("Item 13 content binding changed")
    if canonical_sha256(content) != ATTEMPT_CONTENT_SHA256:
        raise GravityItem13SynthesisError("Item 13 content hash changed")
    if attempt.get("decision") != ATTEMPT_DECISION:
        raise GravityItem13SynthesisError("Item 13 decision changed")
    if any(bool(value) for value in attempt["claims"].values()):
        raise GravityItem13SynthesisError("Item 13 attempt contains an overclaim")
    return attempt


def build_receipt(root: Path) -> dict[str, Any]:
    root = root.resolve()
    if _sha256_file(root / ROADMAP_PATH) != ROADMAP_SHA256:
        raise GravityItem13SynthesisError("stable roadmap changed")
    attempt = _load_attempt(root)
    if attempt["gate_counts"] != {"passed": 10, "required": 16}:
        raise GravityItem13SynthesisError("Item 13 gate count changed")
    if int(attempt["counts"]["candidate_cells"]) != 262144:
        raise GravityItem13SynthesisError("Item 13 candidate count changed")
    if int(attempt["counts"]["quality_passing_galaxies"]) != 243:
        raise GravityItem13SynthesisError("Item 13 quality count changed")
    if int(attempt["counts"]["confirmation_response_rows"]) != 0:
        raise GravityItem13SynthesisError("Item 13 confirmation boundary opened")
    if int(attempt["counts"]["post_response_formula_cells"]) != 0:
        raise GravityItem13SynthesisError("post-response formula entered Item 13")
    primary = attempt["primary"]
    paired = attempt["paired_sign_flip"]
    if float(primary["disturbance_relative_mse_improvement"]) >= 0:
        raise GravityItem13SynthesisError("Item 13 disturbance rejection changed")
    if float(paired["disturbance_after_age"]["p_value"]) <= 0.05:
        raise GravityItem13SynthesisError("Item 13 disturbance null changed")
    if float(primary["item12_age_replication_relative_mse_improvement"]) <= 0.23:
        raise GravityItem13SynthesisError("Item 12 age replication changed")
    if float(paired["item12_age_replication"]["p_value"]) != 0.001:
        raise GravityItem13SynthesisError("Item 12 age replication p-value changed")
    if float(primary["item12_age_persistence_relative_mse_improvement"]) <= 0.22:
        raise GravityItem13SynthesisError("Item 12 age persistence changed")
    if float(paired["item12_age_persistence_after_disturbance"]["p_value"]) != 0.001:
        raise GravityItem13SynthesisError("Item 12 age persistence p-value changed")
    failed_gates = {key for key, value in attempt["gate_checks"].items() if not value}
    expected_failed_gates = {
        "disturbance_beats_age_baseline",
        "disturbance_gain_positive_in_both_prior_age_halves",
        "disturbance_gain_positive_in_both_stellar_mass_halves",
        "disturbance_gain_positive_in_both_tidal_states",
        "disturbance_paired_p_at_most",
        "disturbance_relative_mse_improvement_at_least",
    }
    if failed_gates != expected_failed_gates:
        raise GravityItem13SynthesisError("Item 13 failed-gate set changed")

    receipt: dict[str, Any] = {
        "schema_version": "invariant-gravity-item13-synthesis-1.0",
        "goal": "GRAVITY_ROADMAP_ITEM_13_RELAXATION_AND_MERGERS_SYNTHESIS",
        "item_number": 13,
        "decision": "REJECT_ITEM13_DISTURBANCE_RETAIN_AGE_LEAD_ADVANCE_ITEM14",
        "evidence": {
            "attempt": {
                "path": ATTEMPT_PATH,
                "file_sha256": ATTEMPT_FILE_SHA256,
                "content_sha256": ATTEMPT_CONTENT_SHA256,
                "decision": ATTEMPT_DECISION,
                "candidate_cells": attempt["counts"]["candidate_cells"],
                "quality_passing_galaxies": attempt["counts"]["quality_passing_galaxies"],
                "quality_failed_galaxies": attempt["counts"]["quality_failed_galaxies"],
                "structural_baseline": primary["structural_baseline"],
                "item12_age_baseline": primary["item12_age_baseline"],
                "selected_disturbance_full_model": primary["selected_disturbance_full_model"],
                "disturbance_relative_mse_improvement": primary[
                    "disturbance_relative_mse_improvement"
                ],
                "item12_age_replication_relative_mse_improvement": primary[
                    "item12_age_replication_relative_mse_improvement"
                ],
                "item12_age_persistence_relative_mse_improvement": primary[
                    "item12_age_persistence_relative_mse_improvement"
                ],
                "paired_sign_flip": paired,
                "gates": attempt["gate_counts"],
                "failed_gates": sorted(failed_gates),
                "fold_selections": primary["outer_fold_selections"],
                "strata": attempt["strata"],
                "secondary_stellar_velocity_span": attempt["secondary_stellar_velocity_span"],
                "compute": attempt["compute"],
            }
        },
        "disturbance_family": {
            "status": "REJECTED_AS_INCREMENTAL_PREDICTOR_IN_TESTED_VISUAL_CAS_SCOPE",
            "scope": (
                "The rejection covers the frozen SDSS visual tidal indicator, TType=11 state, "
                "CAS asymmetry/clumpiness, and their declared nonlinear combinations on the "
                "fresh MaNGA integrated-dispersion response. It does not reject faded ancient "
                "mergers, close-pair dynamics, resolved non-equilibrium fields, or other history "
                "observables."
            ),
            "observed_pattern": (
                "Nested disturbance selection worsens held-out dispersion MSE by 1.67 percent "
                "relative to the frozen age baseline, with paired p=0.683. Only two of six broad "
                "strata have positive gains, and three different formula families are selected "
                "across folds."
            ),
            "secondary_boundary": (
                "The inherited disturbance cells improve stellar velocity-span MSE by only 0.49 "
                "percent without reselection; this un-gated secondary diagnostic is not promoted."
            ),
        },
        "retained_lead": {
            "label": "SPECTRAL_CLOCK_CONSENSUS_TIMES_STELLAR_SURFACE_DENSITY",
            "origin_status": "COMBINATION",
            "status": "REPLICATED_ON_DISJOINT_IDENTITIES_PENDING_CROSS_SOURCE_CONFIRMATION",
            "observed_pattern": (
                "The frozen unweighted consolidation of the five Item 12 selected cells reduces "
                "fresh held-out dispersion MSE by 23.06 percent versus structure alone, with "
                "paired p=0.001. It retains a 22.69 percent advantage after the selected visual "
                "disturbance term is controlled, also with p=0.001."
            ),
            "boundary": (
                "This is a preregistered disjoint-identity replication within the same SDSS MaNGA "
                "survey and response pipeline. It is not cross-survey confirmation, a direct age "
                "measurement, a causal result, or evidence for modified gravity."
            ),
            "reuse_rule": (
                "Do not retune on the 243 opened Item 13 responses or open the 100 sealed Item 13 "
                "confirmations. Preserve the exact frozen consolidation in later causal controls."
            ),
        },
        "counterexamples_and_boundaries": [
            "57 of 300 exploration galaxies fail frozen response-quality checks",
            "the selected disturbance family changes across outer folds",
            "visible tidal debris and CAS structure can miss old dynamically relaxed merger remnants",
            "morphology and stellar dynamics come from the same survey even though identities are disjoint",
            "integrated dispersion is not a resolved rotation curve, cluster observable, or lensing map",
        ],
        "not_established": [
            "that mergers or relaxation never affect galaxy dynamics",
            "a causal dynamical-age variable",
            "cross-survey confirmation or a historically new formula",
            "a modification of gravity or alternative to general relativity",
            "prediction of galaxy rotation curves, clusters, or gravitational lensing",
        ],
        "why_item_complete": (
            "The exact fresh-identity test met its quality and boundary gates, scored all 262,144 "
            "frozen disturbance cells with nested held-out selection, and returned a scoped "
            "rejection. The ordinary visible-disturbance explanation was tested without opening "
            "confirmation or generating post-response formulas, so Item 14 is the next numbered "
            "roadmap test."
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
            "visual_cas_disturbance_family_rejected_in_scope": True,
            "item12_age_family_disjoint_identity_replication": True,
            "item12_age_family_cross_source_confirmed": False,
            "merger_or_relaxation_cause_established": False,
            "confirmation_opened": False,
            "roadmap_item_13_complete": True,
            "roadmap_item_14_authorized_next": True,
            "alternative_to_gr_established": False,
            "historical_novelty_established": False,
        },
        "next_action": (
            "Advance to Item 14 resonance and coherence on a fresh real response. Freeze orbital, "
            "pattern-speed, mode-coupling, phase-locking, and long-lived coherence observables and "
            "their dimensional controls before response access; carry the exact Item 12 age "
            "consolidation as a fixed comparator without retuning it."
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
        raise GravityItem13SynthesisError("Item 13 synthesis receipt drifted")


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
