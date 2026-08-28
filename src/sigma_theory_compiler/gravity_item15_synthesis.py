"""Deterministic synthesis for gravity roadmap Item 15."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .sigma_core import canonical_json_bytes, canonical_sha256

ROADMAP_PATH = "docs/GRAVITY_HIDDEN_VARIABLE_AND_THEORY_SEARCH_GOALS.md"
ROADMAP_SHA256 = "b9db7e1aa76deaf28ce2e840cd3d7db4cf7a086b1c7ca473800780c264a0e772"
GALAXY_PATH = "runs/gravity/roadmap/item-15-manga-timescale-ratios-v1.json"
GALAXY_FILE_SHA256 = "2308098ec13e76459cd925e4d27e3c4ffc4edbe94d700cd931bf94864ec20702"
GALAXY_CONTENT_SHA256 = "eff43e00423c51e2f06e2a6c6db275cc77ad0d1c328ae5a30887b3ff16efb351"
GALAXY_DECISION = "INCONCLUSIVE_ITEM15_MANGA_TIMESCALE_QUALITY"
CLUSTER_PATH = "runs/gravity/roadmap/item-15-accept-lc2-timescale-ratios-v2.json"
CLUSTER_FILE_SHA256 = "9de4289267afd3090734250b98770baedfcecac8cce4fd223293f276426d445f"
CLUSTER_CONTENT_SHA256 = "bb0b12451de429b2697d2efe98655e1eb319c8bde0a2f6c228a7db9c83e1145f"
CLUSTER_DECISION = "REJECT_ITEM15_ACCEPT_LC2_TIMESCALE_EXPLORATION"
OUTPUT_PATH = "runs/gravity/roadmap/item-15-synthesis-v1.json"


class GravityItem15SynthesisError(RuntimeError):
    """Raised when an Item 15 input or synthesis invariant drifts."""


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_attempt(
    root: Path,
    *,
    path: str,
    file_sha256: str,
    content_sha256: str,
    decision: str,
) -> dict[str, Any]:
    source = root / path
    if _sha256_file(source) != file_sha256:
        raise GravityItem15SynthesisError(f"Item 15 attempt file changed: {path}")
    attempt = json.loads(source.read_text(encoding="utf-8"))
    content = dict(attempt)
    observed = content.pop("content_sha256", None)
    if observed != content_sha256 or canonical_sha256(content) != content_sha256:
        raise GravityItem15SynthesisError(f"Item 15 attempt content changed: {path}")
    if attempt.get("decision") != decision:
        raise GravityItem15SynthesisError(f"Item 15 attempt decision changed: {path}")
    return attempt


def build_receipt(root: Path) -> dict[str, Any]:
    root = root.resolve()
    if _sha256_file(root / ROADMAP_PATH) != ROADMAP_SHA256:
        raise GravityItem15SynthesisError("stable roadmap changed")
    galaxy = _load_attempt(
        root,
        path=GALAXY_PATH,
        file_sha256=GALAXY_FILE_SHA256,
        content_sha256=GALAXY_CONTENT_SHA256,
        decision=GALAXY_DECISION,
    )
    cluster = _load_attempt(
        root,
        path=CLUSTER_PATH,
        file_sha256=CLUSTER_FILE_SHA256,
        content_sha256=CLUSTER_CONTENT_SHA256,
        decision=CLUSTER_DECISION,
    )
    if galaxy["gate_counts"] != {"passed": 10, "required": 15}:
        raise GravityItem15SynthesisError("Item 15 galaxy gate count changed")
    if cluster["gate_counts"] != {"passed": 14, "required": 15}:
        raise GravityItem15SynthesisError("Item 15 cluster gate count changed")
    if int(galaxy["counts"]["quality_passing_galaxies"]) != 123:
        raise GravityItem15SynthesisError("Item 15 galaxy quality count changed")
    if int(cluster["counts"]["quality_passing_clusters"]) != 18:
        raise GravityItem15SynthesisError("Item 15 cluster quality count changed")
    galaxy_primary = galaxy["primary_stellar_outer_to_inner_log_span_ratio"]
    cluster_primary = cluster["primary_lensing_to_gas_log_mass_ratio"]
    if not 0.05 < float(galaxy_primary["relative_mse_improvement"]) < 0.06:
        raise GravityItem15SynthesisError("Item 15 galaxy hint changed")
    if not 0.09 < float(cluster_primary["relative_mse_improvement"]) < 0.10:
        raise GravityItem15SynthesisError("Item 15 cluster hint changed")
    if float(galaxy["paired_sign_flip"]["p_value"]) != 0.225:
        raise GravityItem15SynthesisError("Item 15 galaxy null changed")
    if float(cluster["full_selection_permutation"]["p_value"]) != 0.22:
        raise GravityItem15SynthesisError("Item 15 cluster null changed")
    failed_cluster_gates = {key for key, value in cluster["gate_checks"].items() if not bool(value)}
    if failed_cluster_gates != {"full_selection_permutation_p_at_most"}:
        raise GravityItem15SynthesisError("Item 15 cluster failed-gate set changed")
    cluster_families = _counts(
        str(row["selected_family"]) for row in cluster_primary["outer_fold_selections"]
    )
    if cluster_families != {
        "cooling_threshold_shell": 4,
        "direct_cooling_freefall_rewrite": 1,
    }:
        raise GravityItem15SynthesisError("Item 15 cluster selection pattern changed")
    for attempt in (galaxy, cluster):
        for key in (
            "confirmation_response_rows",
            "post_response_formula_cells",
            "paid_model_calls",
        ):
            if int(attempt["counts"][key]) != 0:
                raise GravityItem15SynthesisError(f"Item 15 forbidden count changed: {key}")
    if bool(galaxy["limitations"]["direct_hot_gas_cooling_time_tested"]):
        raise GravityItem15SynthesisError("galaxy attempt fabricated direct cooling")
    if not bool(cluster["claims"]["direct_hot_gas_cooling_time_tested"]):
        raise GravityItem15SynthesisError("cluster attempt removed direct cooling")

    receipt: dict[str, Any] = {
        "schema_version": "invariant-gravity-item15-synthesis-1.0",
        "goal": "GRAVITY_ROADMAP_ITEM_15_TIMESCALE_RATIOS_SYNTHESIS",
        "item_number": 15,
        "decision": "REJECT_ITEM15_PROMOTION_RETAIN_TIMESCALE_HINT_ADVANCE_ITEM16",
        "evidence": {
            "galaxy_attempt": {
                "path": GALAXY_PATH,
                "file_sha256": GALAXY_FILE_SHA256,
                "content_sha256": GALAXY_CONTENT_SHA256,
                "decision": GALAXY_DECISION,
                "quality_passing": galaxy["counts"]["quality_passing_galaxies"],
                "quality_failed": galaxy["counts"]["quality_failed_galaxies"],
                "primary": galaxy_primary,
                "secondary_halpha": galaxy["secondary_halpha_outer_to_inner_log_span_ratio"],
                "paired_sign_flip": galaxy["paired_sign_flip"],
                "gates": galaxy["gate_counts"],
                "failed_gates": sorted(
                    key for key, value in galaxy["gate_checks"].items() if not value
                ),
                "strata": galaxy["strata"],
                "compute": galaxy["compute"],
            },
            "direct_cooling_cluster_attempt": {
                "path": CLUSTER_PATH,
                "file_sha256": CLUSTER_FILE_SHA256,
                "content_sha256": CLUSTER_CONTENT_SHA256,
                "decision": CLUSTER_DECISION,
                "quality_passing": cluster["counts"]["quality_passing_clusters"],
                "quality_failed": cluster["counts"]["quality_failed_clusters"],
                "primary": cluster_primary,
                "measurement_error_weighted_robustness": cluster[
                    "measurement_error_weighted_robustness"
                ],
                "full_selection_permutation": cluster["full_selection_permutation"],
                "gates": cluster["gate_counts"],
                "failed_gates": sorted(failed_cluster_gates),
                "strata": cluster["strata"],
                "compute": cluster["compute"],
            },
        },
        "tested_scope": {
            "status": "PROMOTION_REJECTED_FOR_FROZEN_TIMESCALE_GRAMMARS",
            "galaxy_lane": (
                "The fresh MaNGA lane tests baryonic dynamical, mass-doubling, relaxation-null, "
                "and cosmic clock ratios against resolved stellar outer/inner velocity spans, "
                "with H-alpha transfer but no direct hot-gas cooling time."
            ),
            "cluster_lane": (
                "The contamination-safe ACCEPT/LC2 lane tests published direct cooling, "
                "gas-only free-fall, sound crossing, and cosmic clocks at 20, 50, and 100 kpc "
                "against a heterogeneous weak-lensing-to-inner-gas mass ratio."
            ),
            "boundary": (
                "This closes the two exact randomized scalar-ratio grammars as promoted laws. "
                "It does not reject action-derived time dynamics, causal history kernels, "
                "complete baryonic inventories, direct shear/image prediction, or every "
                "possible physical role for cooling and dynamical times."
            ),
        },
        "retained_lead": {
            "label": "NONPROMOTED_TIMESCALE_ORGANIZATION_HINT",
            "status": "RETAIN_WITHOUT_CONFIRMATION_ACCESS",
            "observed_pattern": (
                "Both independent lanes point in the favorable direction: 5.39 percent lower "
                "held-out stellar MSE in the quality-limited galaxy lane and 9.09 percent lower "
                "held-out lensing-to-gas MSE in the clean 18-cluster lane. The cluster gain is "
                "positive in every frozen cooling, temperature, redshift, and source-family "
                "slice and survives the fixed measurement-error weighting."
            ),
            "why_not_promoted": (
                "Neither search-aware null is significant (p=0.225 galaxies; p=0.22 clusters). "
                "The galaxy sample misses its frozen quality floor, the cluster sample is only "
                "18 systems, one cluster fold selects a known direct rewrite, and the four "
                "threshold-shell selections disagree in radius, modulation, threshold, and one "
                "coefficient sign. The common direction is therefore a lead, not one stable law."
            ),
            "reuse_rule": (
                "Do not retune either opened response or open the 80 MaNGA or five LC2 "
                "confirmations. Reuse the pattern only after an independent first-principles "
                "derivation or on a materially independent, frozen direct-observable dataset."
            ),
        },
        "counterexamples_and_boundaries": [
            "only 123 of 240 MaNGA exploration galaxies pass joint stellar/H-alpha quality",
            "the galaxy gain regresses in the slow-growth and higher-mass halves",
            "the galaxy search-aware paired null gives p=0.225",
            "the cluster full-selection permutation gives p=0.22",
            "the cluster error-weighted MSE gain is much smaller than its unweighted gain",
            "the cluster response is a heterogeneous GR/model-standardized M500 rather than direct shear",
            "gas mass inside 100 kpc omits stars and most outer intracluster gas",
            "the cluster free-fall clock intentionally omits the BCG and stellar baryons",
            "no exact formula, radius, modulation, or coefficient is stable across all five cluster folds",
        ],
        "not_established": [
            "that a dynamical or cooling age causes excess rotation or lensing",
            "that baryonic mass estimates are wrong because of age",
            "one universal timescale formula spanning galaxies and clusters",
            "prediction of direct weak- or strong-lensing observables",
            "a historically new formula",
            "a modification of gravity or alternative to general relativity",
        ],
        "why_item_complete": (
            "Item 15 now has two frozen real-data attempts covering the requested galaxy clocks "
            "and a materially independent direct hot-gas cooling lane. The second attempt passes "
            "all 18 source/quality checks, scores 262,144 pre-response cells on the RTX 5090, "
            "repeats the full nested selection in 99 null trials, and fails only the frozen "
            "significance gate. Both confirmation sets remain sealed. Additional tuning on these "
            "responses would reduce rather than increase evidence, so the numbered roadmap moves "
            "to Item 16 while retaining the bounded hint."
        ),
        "counts": {
            "attempts": 2,
            "candidate_formula_cells": int(galaxy["counts"]["candidate_cells"])
            + int(cluster["counts"]["candidate_cells"]),
            "candidate_observation_score_evaluations": int(
                galaxy["compute"]["candidate_galaxy_score_evaluations"]
            )
            + int(cluster["compute"]["candidate_scalar_score_evaluations_with_null"]),
            "quality_objects": int(galaxy["counts"]["quality_passing_galaxies"])
            + int(cluster["counts"]["quality_passing_clusters"]),
            "confirmation_rows_opened": 0,
            "post_response_formula_generation": 0,
            "paid_model_calls": 0,
        },
        "claim_boundaries": {
            "frozen_timescale_grammars_rejected_for_promotion": True,
            "nonpromoted_positive_hint_retained": True,
            "direct_hot_gas_cooling_lane_completed": True,
            "causal_timescale_mechanism_established": False,
            "universal_cross_scale_timescale_law_established": False,
            "confirmation_opened": False,
            "roadmap_item_15_complete": True,
            "roadmap_item_16_authorized_next": True,
            "alternative_to_gr_established": False,
            "historical_novelty_established": False,
        },
        "next_action": (
            "Advance to Item 16 QED-like gravity. Freeze a carrier/vertex/propagator grammar, "
            "classical-limit identities, conservation and stability rejection rules, and a real "
            "prediction target before opening a fresh response. The Item 15 hint may appear only "
            "as a fixed comparator unless a field derivation independently produces it."
        ),
        "content_sha256": None,
    }
    content = dict(receipt)
    content.pop("content_sha256")
    receipt["content_sha256"] = canonical_sha256(content)
    return receipt


def _counts(values: Any) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        result[value] = result.get(value, 0) + 1
    return result


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
        raise GravityItem15SynthesisError("Item 15 synthesis receipt drifted")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    if arguments.check:
        check_receipt(arguments.root)
    else:
        print(write_receipt(arguments.root))


if __name__ == "__main__":
    main()
