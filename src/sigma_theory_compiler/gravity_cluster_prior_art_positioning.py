"""Machine-audited prior-art positioning for the Item 59 cluster candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

CONFIG_PATH = Path("configs/gravity_cluster_prior_art_positioning_v1.json")
OUTPUT_PATH = Path("runs/gravity/publication-readiness/prior-art-positioning-v1.json")
CONFIG_SCHEMA = "invariant-gravity-cluster-prior-art-positioning-1.0"
RECEIPT_SCHEMA = "invariant-gravity-cluster-prior-art-positioning-receipt-1.0"
DATABASES = ("NASA_ADS", "arXiv", "Crossref", "OpenAlex", "INSPIRE_HEP", "journal_full_text")
SOURCE_IDS = (
    "PENNER_MODIFIED_GRAS_AQUAL_2026",
    "BEKENSTEIN_MILGROM_AQUAL_1984",
    "MATSAKOS_DIAFERIO_REFRACTED_GRAVITY_2016",
    "CHICONE_MASHHOON_NONLOCAL_POISSON_2012",
    "ZHAO_FAMAEY_EMOND_2012",
    "MOFFAT_MOG_ACCELERATION_2016",
    "VERLINDE_EMERGENT_GRAVITY_2016",
    "NFW_HALO_1996",
    "NAVARRO_EINASTO_LIKE_2004",
    "ARNAUD_GNFW_PRESSURE_2010",
)
COMPONENT_IDS = (
    "RATIONAL_OCCUPANCY_TRANSITION",
    "SOURCE_CONDITIONED_RESPONSE",
    "INWARD_NONLOCAL_KERNEL",
    "SYMMETRIC_NONLOCAL_KERNEL",
    "TWO_CHANNEL_ADDITIVE_LAW",
    "OUTER_BOUNDARY_FORWARD_MODEL",
    "CLUSTER_THERMODYNAMIC_ENDPOINT",
)


class GravityClusterPriorArtError(RuntimeError):
    """Raised when the prior-art audit or novelty boundary changes."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    ) + b"\n"


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _strict(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise GravityClusterPriorArtError(f"{label} keys changed")


def load_config(root: Path) -> dict[str, Any]:
    value = json.loads((root.resolve() / CONFIG_PATH).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise GravityClusterPriorArtError("prior-art config must be an object")
    validate_config(value)
    return value


def validate_config(config: Mapping[str, Any]) -> None:
    _strict(
        config,
        {
            "schema_version",
            "status",
            "audit_id",
            "audit_cutoff",
            "candidate",
            "database_search_audit",
            "retained_primary_sources",
            "equation_comparison",
            "candidate_adjudication",
            "human_review",
            "output_path",
        },
        "prior-art config",
    )
    if (
        config["schema_version"] != CONFIG_SCHEMA
        or config["status"] != "frozen_search_audit_human_novelty_review_open"
        or config["audit_id"] != "gravity-cluster-prior-art-positioning-v1"
        or config["output_path"] != OUTPUT_PATH.as_posix()
    ):
        raise GravityClusterPriorArtError("prior-art config identity changed")
    candidate = config["candidate"]
    if (
        candidate["candidate_id"] != "ITEM59-XCOP-CROSS-SCALE-BOUNDARY-BETA-1P5"
        or len(candidate["equations"]) != 3
        or candidate["historical_novelty_claim"] is not False
        or "K_in" not in candidate["equations"][2]
        or "K_sym" not in candidate["equations"][2]
    ):
        raise GravityClusterPriorArtError("candidate identity or novelty seal changed")
    searches = config["database_search_audit"]
    if tuple(row["database"] for row in searches) != DATABASES:
        raise GravityClusterPriorArtError("database search inventory changed")
    for row in searches:
        if (
            len(row["queries"]) < 3
            or not row["method"]
            or not row["result"]
            or row["authoritative_absence_claim"] is not False
        ):
            raise GravityClusterPriorArtError("database search overclaims corpus absence")
    sources = config["retained_primary_sources"]
    if tuple(row["source_id"] for row in sources) != SOURCE_IDS:
        raise GravityClusterPriorArtError("retained primary-source inventory changed")
    for row in sources:
        if (
            not str(row["url"]).startswith("https://")
            or not row["doi"]
            or not row["equation_anchor"]
            or not row["relevance"]
            or not row["relationship_label"]
        ):
            raise GravityClusterPriorArtError("primary-source equation record is incomplete")
    comparisons = config["equation_comparison"]
    if tuple(row["component_id"] for row in comparisons) != COMPONENT_IDS:
        raise GravityClusterPriorArtError("equation comparison inventory changed")
    classifications = {row["classification"] for row in comparisons}
    if not {
        "rewrite_of_known_transition_motif",
        "new_combination_of_known_motifs",
        "structurally_unmatched_exact_combination_novelty_unresolved",
        "known_method_not_candidate_novelty",
    }.issubset(classifications):
        raise GravityClusterPriorArtError("required prior-art labels are not separate")
    adjudication = config["candidate_adjudication"]
    if adjudication != {
        "exact_rewrite_identified": False,
        "known_family_components_identified": True,
        "known_combination_components_identified": True,
        "structurally_unmatched_exact_combination": True,
        "close_behavioral_equivalent_identified": True,
        "closest_behavioral_source_id": "PENNER_MODIFIED_GRAS_AQUAL_2026",
        "historical_novelty_resolved": False,
        "allowed_label": "potentially_new_exact_combination_of_known_motifs_with_a_close_2026_behavioral_neighbor",
        "prohibited_label": "historically_novel_gravity_law",
        "required_manuscript_action": adjudication["required_manuscript_action"],
    } or "named expert review" not in adjudication["required_manuscript_action"]:
        raise GravityClusterPriorArtError("candidate adjudication overclaims novelty")
    if config["human_review"] != {
        "cluster_astrophysicist_named_review": False,
        "modified_gravity_specialist_named_review": False,
        "historical_novelty_sentence_authorized": False,
    }:
        raise GravityClusterPriorArtError("human novelty-review gate changed")


def build_receipt(root: Path) -> dict[str, Any]:
    config = load_config(root.resolve())
    adjudication = config["candidate_adjudication"]
    body = {
        "schema_version": RECEIPT_SCHEMA,
        "audit_id": config["audit_id"],
        "decision": "PRIOR_ART_POSITIONED_CLOSE_2026_NEIGHBOR_HISTORICAL_NOVELTY_UNRESOLVED",
        "config_binding": {"path": CONFIG_PATH.as_posix(), "content_sha256": _sha(config)},
        "completed_goal_evidence": {
            "CP2.2": "six_required_literature_services_and_equation_level_full_text_audited",
            "CP2.3": "equation_by_equation_comparison_covers_required_modified_gravity_halo_and_pressure_families",
            "CP2.4": "rewrite_known_family_known_combination_structurally_unmatched_and_unresolved_labels_separated",
        },
        "blocked_goal_evidence": {
            "CP2.5": "named_cluster_astrophysicist_and_modified_gravity_reviews_not_obtained",
            "CP2.6": "historical_novelty_sentence_not_authorized_without_named_human_review",
        },
        "candidate_adjudication": adjudication,
        "closest_behavioral_neighbor": {
            "source_id": "PENNER_MODIFIED_GRAS_AQUAL_2026",
            "shared_structure": [
                "baryon_conditioned_cluster_acceleration",
                "nonlocal_radial_dependence",
                "complete_baryon_distribution_affects_interior",
                "outer_boundary_anchors_inward_integration",
            ],
            "decisive_difference": "Penner uses the response slope beta=-g/(r*g_prime) inside a modified GRAS/AQUAL ODE; Item 59 uses two fixed log-radius averages of q[g_bar/a0] in an additive empirical acceleration law.",
            "exact_rewrite": False,
        },
        "counts": {
            "databases_audited": len(config["database_search_audit"]),
            "primary_sources_retained": len(config["retained_primary_sources"]),
            "equation_components_compared": len(config["equation_comparison"]),
            "exact_rewrites_identified": 0,
            "close_behavioral_equivalents_identified": 1,
            "named_human_reviews": 0,
            "historical_novelty_claims_authorized": 0,
            "target_rows_opened": 0,
        },
        "claims": {
            "multi_database_search_complete_to_cutoff": True,
            "equation_comparison_complete": True,
            "classification_labels_separated": True,
            "exact_rewrite_identified": False,
            "close_behavioral_equivalent_identified": True,
            "structurally_unmatched_in_audited_corpus": True,
            "corpus_absence_is_authoritative": False,
            "historical_novelty_established": False,
            "human_review_complete": False,
            "independent_replication": False,
        },
        "next_action": "Obtain named review from a cluster astrophysicist and a modified-gravity specialist, with explicit attention to Penner 2026 equations 40-45, before writing any historical-novelty sentence.",
    }
    return {**body, "content_sha256": _sha(body)}


def validate_receipt(receipt: Mapping[str, Any], root: Path) -> None:
    body = dict(receipt)
    expected_hash = body.pop("content_sha256", None)
    if expected_hash != _sha(body) or dict(receipt) != build_receipt(root):
        raise GravityClusterPriorArtError("prior-art receipt changed")


def write_receipt(root: Path) -> Path:
    path = root.resolve() / OUTPUT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(build_receipt(root)))
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("write", "check", "status"))
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.command == "write":
        output: Any = str(write_receipt(root))
    elif args.command == "check":
        receipt = json.loads((root / OUTPUT_PATH).read_text(encoding="utf-8"))
        validate_receipt(receipt, root)
        output = {"status": "PASS", "content_sha256": receipt["content_sha256"]}
    else:
        receipt = build_receipt(root)
        output = {
            "decision": receipt["decision"],
            "candidate_adjudication": receipt["candidate_adjudication"],
            "next_action": receipt["next_action"],
        }
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
