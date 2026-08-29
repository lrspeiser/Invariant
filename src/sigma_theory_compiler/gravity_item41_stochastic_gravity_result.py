"""Build the aggregate Item 41 stochastic-gravity result."""

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
from sigma_theory_compiler.gravity_item41_stochastic_gravity import (
    GravityItem41Error,
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
    _verify_content_hash(compute, "Item 41 GHASP compute manifest")
    _verify_content_hash(cluster, "Item 41 CLASH transfer")
    selected = compute["candidate_search"]["full_retrospective_candidate"]
    if selected != cluster["selected_formula"]:
        raise GravityItem41Error("Item 41 transfer formula differs from GHASP selection")
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item41-aggregate-result-1.0",
            "item": 41,
            "decision": "NONPROMOTED_ITEM41_STOCHASTIC_LAW_NEGATIVE_BUT_FAMILY_RETAINED",
            "hypothesis": config["hypothesis"],
            "protocol": {
                "scientific_freeze_commit": config["scientific_freeze_commit"],
                "ghasp_compute_sha256": _sha256_file(compute_path),
                "ghasp_content_sha256": compute["content_sha256"],
                "cluster_transfer_sha256": _sha256_file(cluster_path),
                "cluster_content_sha256": cluster["content_sha256"],
                "post_response_candidate_cells": 0,
                "confirmation_values_read": 0,
                "paid_model_calls": 0,
            },
            "selected_formula": {
                **selected,
                "equation": "u=gbar/a0; m=0; S=0.56^2/[1+(u/10000)^3]",
                "one_cell_increment": "Delta ln(g)=sqrt(S)*xi, E[xi]=0, Var[xi]=1",
                "reading": "a nearly constant natural-log acceleration variance over the tested galaxy and cluster acceleration range",
            },
            "ghasp_paired_side_test": {
                "decision": compute["decision"],
                "quality": compute["quality"],
                "candidate_point_evaluations": compute["candidate_search"][
                    "candidate_point_evaluations"
                ],
                "backend": compute["candidate_search"]["backend"],
                "device": compute["candidate_search"]["device"],
                "losses": compute["joint_mean_variance_result"]["losses"],
                "strongest_control": compute["joint_mean_variance_result"][
                    "strongest_control"
                ],
                "improvement_vs_strongest_percent": compute[
                    "joint_mean_variance_result"
                ]["improvement_vs_strongest_percent"],
                "variance_spearman": compute["joint_mean_variance_result"][
                    "variance_vs_squared_side_difference_spearman"
                ],
                "robustness": compute["joint_mean_variance_result"]["robustness"],
                "counterexample_assessment": compute["joint_mean_variance_result"][
                    "counterexample_assessment"
                ],
            },
            "clash_cluster_diagnostic": {
                "decision": cluster["decision"],
                "clusters": cluster["data"]["clusters"],
                "radial_points": cluster["data"]["radial_points"],
                "losses": cluster["losses"],
                "strongest_primary_control": cluster["strongest_primary_control"],
                "improvement_vs_strongest_percent": cluster[
                    "improvement_vs_strongest_percent"
                ],
                "process_variance_range": cluster["process_variance_range"],
                "counterexample_assessment": cluster["counterexample_assessment"],
                "role": cluster["data"]["role"],
            },
            "interpretation": {
                "material_improvement": "The machinery now tests a stochastic law against paired galaxy-side means and variances, not only deterministic mean curves, and transfers the selected moments unchanged to lensing-scale residuals.",
                "material_non_improvement": "The selected nearly constant white-field variance lost to a homoskedastic control on GHASP and to an ordinary out-of-cluster heteroskedastic control on CLASH; it did not establish a scale-dependent stochastic signal.",
                "data_sensitivity": "The GHASP aggregate sign reverses when UGC5786 is removed, so the galaxy result is explicitly single-object-sensitive. The CLASH proxy is model-dependent and already exposed, so it is not counted as unchanged independent replication.",
                "counterexample_rule": "All mismatches remain recorded. Neither one counterexample nor a count alone kills a formula; uncertainty, influence, data quality, and genuinely independent unchanged replications determine evidence weight.",
                "family_status": "The exact formula is not promoted. White, colored, multiplicative-drift, and telegraph stochastic-gravity mechanisms remain available for new parameterizations, better covariance data, and fresh tests.",
            },
            "claim_boundaries": {
                "dark_matter_excluded": False,
                "modified_gravity_established": False,
                "stochastic_gravity_established": False,
                "historical_novelty_established": False,
                "covariant_theory_established": False,
                "cluster_transfer_is_fresh_confirmation": False,
                "formula_pruned": False,
                "formula_family_pruned": False,
                "one_empirical_counterexample_is_veto": False,
                "item28_confirmation_remains_sealed": True,
            },
            "next_action": "Proceed to Item 42 matter-geometry feedback with a newly frozen mechanism family. Preserve Item 41's failure map for later hybrids, but do not tune this formula to the exposed responses.",
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
    _verify_content_hash(existing, "Item 41 aggregate result")
    if existing != build(root):
        raise GravityItem41Error("Item 41 aggregate replay changed")
    return {
        "status": "ITEM41_AGGREGATE_REPLAY_VALID",
        "decision": existing["decision"],
        "content_sha256": existing["content_sha256"],
        "confirmation_values_read": existing["protocol"]["confirmation_values_read"],
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
