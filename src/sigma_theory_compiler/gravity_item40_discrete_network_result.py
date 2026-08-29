"""Build the aggregate Item 40 discrete/network-gravity result."""

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
from sigma_theory_compiler.gravity_item40_discrete_network import (
    GravityItem40Error,
    _source_path,
    load_config,
)


def build(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config = load_config(root)
    compute_path = _source_path(root, config, "compute_manifest")
    cluster_path = _source_path(root, config, "cluster_transfer_result")
    compute = _read_json(compute_path)
    cluster = _read_json(cluster_path)
    _verify_content_hash(compute, "Item 40 compute manifest")
    _verify_content_hash(cluster, "Item 40 CLASH transfer")
    selected = compute["candidate_search"]["full_exploration_candidate"]
    if selected != cluster["selected_formula"]:
        raise GravityItem40Error("Item 40 transfer formula differs from dynamics selection")
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item40-aggregate-result-1.0",
            "item": 40,
            "decision": "NONPROMOTED_ITEM40_DISCRETE_NETWORK_NEGATIVE_BUT_FAMILY_RETAINED",
            "hypothesis": config["hypothesis"],
            "protocol": {
                "scientific_freeze_commit": config["scientific_freeze_commit"],
                "predictor_freeze_commit": config["predictor_freeze_commit"],
                "sample_freeze_commit": config["sample_freeze_commit"],
                "dynamics_compute_sha256": _sha256_file(compute_path),
                "dynamics_content_sha256": compute["content_sha256"],
                "cluster_transfer_sha256": _sha256_file(cluster_path),
                "cluster_content_sha256": cluster["content_sha256"],
                "post_response_candidate_cells": 0,
                "confirmation_response_rows": 0,
                "paid_model_calls": 0,
            },
            "selected_formula": {
                **selected,
                "equation": "u=gbar/a0; nu=1+4*u^(-0.4)*[1+(u/1000)^0.2]^(-5)*[0.05+0.95*H_communicability^0.2]",
                "motion": "v_pred=sqrt(nu)*v_bar",
                "light_proxy": "g_lens,pred=nu*g_bar under the frozen Phi=Psi weak-field closure",
            },
            "wallaby_dynamics": {
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
                "counterexample_assessment": compute["primary_dynamics"][
                    "counterexample_assessment"
                ],
            },
            "clash_cluster_diagnostic": {
                "decision": cluster["decision"],
                "clusters": cluster["data"]["clusters"],
                "radial_points": cluster["data"]["radial_points"],
                "losses": cluster["losses"],
                "strongest_fixed_control": cluster["strongest_fixed_control"],
                "improvement_vs_strongest_percent": cluster[
                    "improvement_vs_strongest_percent"
                ],
                "counterexample_assessment": cluster["counterexample_assessment"],
                "role": cluster["data"]["role"],
            },
            "interpretation": {
                "material_improvement": "The graph candidate beat bare baryonic Newton and narrowly beat a matched ordinary-geometry ridge on fresh WALLABY exploration data.",
                "material_non_improvement": "It lost to MOND and the frozen Item 39 formula on galaxies, then lost to MOND on the unchanged CLASH transfer.",
                "counterexample_rule": "Raw mismatches are preserved. One mismatch never kills a formula; uncertainty, influence, data quality, and independent unchanged tests determine evidence weight.",
                "family_status": "The exact formula is not promoted, but graph/network gravity remains an open mechanism family because this finite, small, partly model-dependent audit cannot prune the family.",
            },
            "claim_boundaries": {
                "dark_matter_excluded": False,
                "modified_gravity_established": False,
                "historical_novelty_established": False,
                "covariant_theory_established": False,
                "cluster_transfer_is_fresh_confirmation": False,
                "formula_pruned": False,
                "formula_family_pruned": False,
                "one_empirical_counterexample_is_veto": False,
                "confirmation_remains_sealed": True,
            },
            "next_action": "Proceed to Item 41 with a new response-blind mechanism family; preserve Item 40 graph coordinates and failures for later hybrids, but do not tune them to these responses.",
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
    _verify_content_hash(existing, "Item 40 aggregate result")
    if existing != build(root):
        raise GravityItem40Error("Item 40 aggregate replay changed")
    return {
        "status": "ITEM40_AGGREGATE_REPLAY_VALID",
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
    result: Path | Mapping[str, Any] = write(args.root) if args.command == "write" else check(args.root)
    print(result if isinstance(result, Path) else json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
