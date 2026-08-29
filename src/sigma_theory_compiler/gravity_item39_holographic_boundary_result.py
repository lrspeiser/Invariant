"""Build the bounded aggregate result for gravity-roadmap Item 39."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sigma_theory_compiler.gravity_item22_polarization_superposition import (
    _content_hashed,
    _read_json,
    _sha256_file,
    _verify_content_hash,
    _write_json,
)
from sigma_theory_compiler.gravity_item39_holographic_boundary import (
    GravityItem39Error,
    _source_path,
    load_config,
)


def build(root: Path, *, write: bool = True) -> dict[str, Any]:
    root = root.resolve()
    config = load_config(root)
    compute_path = _source_path(root, config, "compute_manifest")
    lensing_path = _source_path(root, config, "lensing_transfer_result")
    compute = _read_json(compute_path)
    lensing = _read_json(lensing_path)
    _verify_content_hash(compute, "Item 39 dynamics compute")
    _verify_content_hash(lensing, "Item 39 SWELLS result")
    if compute["decision"] != "INCONCLUSIVE_ITEM39_WALLABY_QUALITY_RETAINED":
        raise GravityItem39Error("unexpected Item 39 dynamics decision")
    if lensing["decision"] != "ITEM39_UNCHANGED_SWELLS_TRANSFER_IMPROVES_DIAGNOSTIC":
        raise GravityItem39Error("unexpected Item 39 lensing decision")

    candidate = compute["candidate_search"]["full_exploration_candidate"]
    dynamics = compute["primary_dynamics"]
    lens_primary = lensing["primary"]
    result = _content_hashed(
        {
            "schema_version": "invariant-gravity-item39-holographic-boundary-result-1.0",
            "item": 39,
            "decision": "NONPROMOTED_ITEM39_HOLOGRAPHIC_BOUNDARY_MIXED_DIAGNOSTIC",
            "hypothesis": config["hypothesis"],
            "selected_formula": candidate,
            "formula_interpretation": {
                "u": "baryonic acceleration divided by a0",
                "f_M": "fraction of total baryonic mass enclosed by the test radius",
                "x": "test radius divided by the frozen baryonic screen radius",
                "H": "sqrt(4*f_M*(1-f_M)*4*x/(1+x)^2)",
                "multiplier": "1+1.25*u^(-0.5)*(1+(u/1e8)^0.2)^(-5)*(0.05+0.95*sin(pi*H/2)^0.2)",
                "metric_closure": "Phi=Psi weak-field test closure",
            },
            "compute": {
                "raw_candidate_cells": compute["candidate_search"]["raw_candidates"],
                "admitted_candidates": compute["candidate_search"]["admitted_candidates"],
                "behavioral_equivalence_classes": compute["candidate_search"][
                    "behavioral_equivalence_classes"
                ],
                "candidate_point_evaluations": compute["candidate_search"][
                    "candidate_point_evaluations"
                ],
                "backend": compute["candidate_search"]["backend"],
                "device": compute["candidate_search"]["device"],
                "cpu_gpu_max_abs_log10_velocity_difference": compute["candidate_search"][
                    "cpu_gpu_max_abs_log10_velocity_difference"
                ],
                "paid_api_calls": 0,
                "paid_api_usd": 0.0,
            },
            "wallaby_dynamics": {
                "selected_exploration_galaxies": 60,
                "reserved_confirmation_galaxies": 15,
                "quality_passing_galaxies": dynamics["galaxies"],
                "quality_passing_points": dynamics["points"],
                "confirmation_response_rows": compute["protocol"]["confirmation_response_rows"],
                "quality": compute["quality"],
                "losses": dynamics["losses"],
                "strongest_ordinary_baseline": dynamics["strongest_ordinary_baseline"],
                "improvement_vs_strongest_percent": dynamics["improvement_vs_strongest_percent"],
                "paired_sign_flip": dynamics["paired_sign_flip"],
                "robustness": dynamics["robustness"],
                "counterexample_assessment": dynamics["counterexample_assessment"],
            },
            "unchanged_swells_lensing_diagnostic": {
                "evaluable_lenses": lensing["sample"]["evaluable_lenses"],
                "losses": lens_primary["losses"],
                "strongest_fixed_control": lens_primary["strongest_fixed_control"],
                "improvement_vs_strongest_percent": lens_primary[
                    "improvement_vs_strongest_percent"
                ],
                "candidate_improves_over_every_fixed_control": lens_primary[
                    "candidate_improves_over_every_fixed_control"
                ],
                "counterexample_assessment": lens_primary["counterexample_assessment"],
                "systematic_audits": lensing["systematic_audits"],
                "unblinded_diagnostic": True,
                "direct_image_likelihood": False,
            },
            "gates": {
                "dynamics_quality_passes": compute["gates"]["quality_passes"],
                "dynamics_beats_matched_ordinary_geometry": compute["gates"][
                    "beats_matched_ordinary_geometry"
                ],
                "dynamics_beats_flexible_radial_surface": compute["gates"][
                    "beats_flexible_radial_surface"
                ],
                "dynamics_paired_p_at_most_0p05": compute["gates"]["paired_p_at_most_0p05"],
                "dynamics_all_broad_strata_improve": compute["gates"]["all_broad_strata_improve"],
                "unchanged_lensing_transfer_improves": lens_primary[
                    "candidate_improves_over_every_fixed_control"
                ],
                "lensing_all_frozen_systematics_improve": all(
                    audit["improvement_vs_strongest_percent"] > 0.0
                    for audit in lensing["systematic_audits"].values()
                ),
                "same_formula_and_zero_slip_metric_used_for_motion_and_light": True,
                "confirmation_remains_sealed": compute["gates"]["confirmation_remains_sealed"],
                "hard_theoretical_veto_absent": True,
                "promotion_passes": False,
            },
            "claim_boundaries": [
                "The WALLABY quality floor failed: 23 of 60 exploration galaxies supplied usable curves, below both frozen thresholds.",
                "The selected formula lost to matched ordinary geometry and flexible radial/surface controls on WALLABY.",
                "The SWELLS diagnostic was unblinded after WALLABY selection and is based on published aperture summaries, not raw image likelihoods.",
                "A +0.25 dex stellar-mass audit reverses the SWELLS advantage, so baryonic mass uncertainty remains a live explanation.",
                "The Phi=Psi rule is a weak-field test closure, not a covariant derivation of a relativistic gravity theory.",
                "Holographic and entanglement concepts are known prior art; the selected lane is at most a potentially new observational synthesis.",
                "No empirical counterexample prunes the selected formula or its wider family.",
                "Dark matter is not excluded, modified gravity is not established, and historical novelty is not established.",
            ],
            "protocol": {
                "scientific_freeze_commit": config["scientific_freeze_commit"],
                "predictor_freeze_commit": config["predictor_freeze_commit"],
                "sample_freeze_commit": config["sample_freeze_commit"],
                "lensing_transfer_freeze_commit": config["lensing_transfer"]["freeze_commit"],
                "dynamics_compute_file_sha256": _sha256_file(compute_path),
                "lensing_result_file_sha256": _sha256_file(lensing_path),
                "post_response_candidate_cells": 0,
                "post_selection_lensing_candidate_cells": 0,
                "paid_model_calls": 0,
            },
            "next_action": "Preserve the selected Item 39 formula as a nonpromoted lead for a fresh quality-passing rotation sample and a prospectively blinded direct-lensing test. Advance the ordered mechanism search to Item 40, discrete or network gravity, without opening the 15 reserved WALLABY confirmation galaxies.",
        }
    )
    if write:
        _write_json(root / str(config["paths"]["result"]), result)
    return result


def check(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config = load_config(root)
    path = root / str(config["paths"]["result"])
    existing = _read_json(path)
    _verify_content_hash(existing, "Item 39 aggregate result")
    replay = build(root, write=False)
    if existing != replay:
        raise GravityItem39Error("Item 39 aggregate result replay drifted")
    return {
        "status": "ITEM39_AGGREGATE_RESULT_VALID",
        "decision": existing["decision"],
        "content_sha256": existing["content_sha256"],
        "result_file_sha256": _sha256_file(path),
        "promotion_passes": existing["gates"]["promotion_passes"],
        "confirmation_response_rows": existing["wallaby_dynamics"]["confirmation_response_rows"],
        "paid_model_calls": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "check"))
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    if args.command == "build":
        value = build(args.root)
        print(json.dumps({"decision": value["decision"]}, sort_keys=True))
    else:
        print(json.dumps(check(args.root), sort_keys=True))


if __name__ == "__main__":
    main()
