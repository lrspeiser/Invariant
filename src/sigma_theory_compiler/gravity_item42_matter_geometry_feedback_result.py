"""Build the aggregate Item 42 matter-geometry feedback result."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from sigma_theory_compiler.gravity_item22_polarization_superposition import (
    _content_hashed,
    _read_json,
    _sha256_file,
    _verify_content_hash,
    _write_json,
)
from sigma_theory_compiler.gravity_item42_matter_geometry_feedback import (
    GravityItem42Error,
    _source_path,
    load_config,
)


def build(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config = load_config(root)
    compute_path = _source_path(root, config, "compute_manifest")
    cluster_path = _source_path(root, config, "clash_transfer_result")
    compute = _read_json(compute_path)
    cluster = _read_json(cluster_path)
    _verify_content_hash(compute, "Item 42 dynamics compute manifest")
    _verify_content_hash(cluster, "Item 42 CLASH transfer")
    selected = compute["candidate_search"]["full_exploration_candidate"]
    if selected != cluster["selected_formula"]:
        raise GravityItem42Error("Item 42 transfer formula differs from dynamics selection")
    galaxy_no_feedback = compute["primary_dynamics"]["losses"]["matched_no_feedback"]
    galaxy_candidate = compute["primary_dynamics"]["losses"]["candidate"]
    galaxy_feedback_increment = 100.0 * (
        galaxy_no_feedback - galaxy_candidate
    ) / galaxy_no_feedback
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item42-aggregate-result-1.0",
            "item": 42,
            "decision": "NONPROMOTED_ITEM42_CROSS_SCALE_FEEDBACK_INCREMENT_RETAINED",
            "hypothesis": config["hypothesis"],
            "protocol": {
                "scientific_freeze_commit": config["scientific_freeze_commit"],
                "sample_freeze_commit": config["sample_freeze_commit"],
                "dynamics_compute_sha256": _sha256_file(compute_path),
                "dynamics_content_sha256": compute["content_sha256"],
                "cluster_transfer_sha256": _sha256_file(cluster_path),
                "cluster_content_sha256": cluster["content_sha256"],
                "post_response_candidate_cells": 0,
                "confirmation_response_rows": 0,
                "paid_model_calls": 0,
                "post_response_implementation_repair": compute["protocol"][
                    "post_response_implementation_repair"
                ],
            },
            "selected_formula": {
                **selected,
                "fixed_point": "K_ij proportional to exp(-|x_i-x_j|/0.2); w_i proportional to b_i*exp[5*tanh(0.2*|dH/dx|_i)]; H=normalize(Kw)",
                "motion": "u=gbar/a0; nu=1+u^(-0.6)/(1+u/3)*(0.05+0.95*H); v_pred=sqrt(nu)*v_bar",
                "light_proxy": "g_lens,pred=nu*g_bar under the frozen Phi=Psi weak-field closure",
            },
            "fresh_wallaby_dynamics": {
                "decision": compute["decision"],
                "quality": compute["quality"],
                "candidate_point_evaluations": compute["candidate_search"][
                    "candidate_point_evaluations"
                ],
                "backend": compute["candidate_search"]["backend"],
                "device": compute["candidate_search"]["device"],
                "losses": compute["primary_dynamics"]["losses"],
                "strongest_baseline": compute["primary_dynamics"]["strongest_baseline"],
                "improvement_vs_strongest_percent": compute["primary_dynamics"][
                    "improvement_vs_strongest_percent"
                ],
                "improvement_vs_matched_no_feedback_percent": galaxy_feedback_increment,
                "paired_sign_flip": compute["primary_dynamics"]["paired_sign_flip"],
                "robustness": compute["primary_dynamics"]["robustness"],
                "counterexample_assessment": compute["primary_dynamics"][
                    "counterexample_assessment"
                ],
            },
            "clash_cluster_diagnostic": {
                "decision": cluster["decision"],
                "clusters": cluster["data"]["clusters"],
                "radial_points": cluster["data"]["radial_points"],
                "convergence": cluster["convergence"],
                "losses": cluster["losses"],
                "loss_scope": cluster["loss_scope"],
                "strongest_fixed_control": cluster["strongest_fixed_control"],
                "improvement_vs_strongest_percent": cluster[
                    "improvement_vs_strongest_percent"
                ],
                "improvement_vs_matched_no_feedback_percent": cluster[
                    "improvement_vs_matched_no_feedback_percent"
                ],
                "counterexample_assessment": cluster["counterexample_assessment"],
                "role": cluster["data"]["role"],
            },
            "interpretation": {
                "material_improvement": "The active nonlinear feedback term improved over a matched beta=0 law in both fresh galaxy dynamics and the unchanged cluster transfer, by about 1.6% and 10.5% respectively.",
                "material_non_improvement": "The feedback law lost to gas-only MOND on galaxies and to MOND on the 19 converged clusters; its galaxy advantage over no feedback was not statistically persuasive.",
                "domain_failure": "The selected high-gain fixed point failed to converge for cluster A2261, so the cluster gate fails and the exact formula lacks complete domain coverage.",
                "data_limit": "The fresh galaxy sample has HI profiles but no accepted Legacy optical counterparts, so unmeasured stellar mass could explain part of the residual and prevents a complete-baryon gravity claim.",
                "research_value": "The same-direction incremental benefit across motion and lensing is worth a dedicated complete-baryon, convergence-stable replication study, but is not yet a paper-level discovery claim.",
                "counterexample_rule": "Every mismatch and the A2261 numerical failure remain recorded. Neither one counterexample nor a count alone kills a formula; genuinely independent unchanged replication is required for terminal tested-scope rejection.",
                "family_status": "The exact high-gain law is not promoted. Reinforcement, screening, gradient, and competing matter-geometry feedback mechanisms remain open for newly frozen formulations and data.",
            },
            "claim_boundaries": {
                "dark_matter_excluded": False,
                "modified_gravity_established": False,
                "historical_novelty_established": False,
                "covariant_theory_established": False,
                "complete_baryonic_inventory": False,
                "cluster_transfer_is_fresh_confirmation": False,
                "formula_pruned": False,
                "formula_family_pruned": False,
                "one_empirical_counterexample_is_veto": False,
                "confirmation_remains_sealed": True,
            },
            "next_action": "Proceed to Item 43 cosmological boundary coupling. Preserve Item 42's cross-scale feedback increment as a replication candidate, but do not tune it to the exposed failures.",
        }
    )


def write(root: Path) -> Path:
    config = load_config(root)
    path = root / str(config["paths"]["aggregate_result"])
    _write_json(path, build(root))
    return path


def check(root: Path) -> dict[str, Any]:
    config = load_config(root)
    path = root / str(config["paths"]["aggregate_result"])
    existing = _read_json(path)
    _verify_content_hash(existing, "Item 42 aggregate result")
    if existing != build(root):
        raise GravityItem42Error("Item 42 aggregate replay changed")
    return {
        "status": "ITEM42_AGGREGATE_REPLAY_VALID",
        "decision": existing["decision"],
        "content_sha256": existing["content_sha256"],
        "confirmation_response_rows": existing["protocol"]["confirmation_response_rows"],
        "paid_model_calls": existing["protocol"]["paid_model_calls"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("write", "check"))
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    result: Path | Mapping[str, Any]
    result = write(args.root) if args.command == "write" else check(args.root)
    print(result if isinstance(result, Path) else json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
