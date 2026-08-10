from __future__ import annotations

import json
from typing import Any

from .promotion_orchestrator import ELIGIBILITY

EVALUATOR_ID = "galaxy-direct-observable-sealed-v1"
EVALUATOR_VERSION = "1.0.0"
PREDICTION_BUNDLES: dict[str, dict[str, Any]] = {}


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def galaxy_direct_observable_evaluator(
    candidate: dict[str, Any], context: dict[str, Any]
) -> dict[str, Any]:
    """Keep galaxy data sealed until a candidate supplies a registered prediction bundle."""

    if context.get("data_eligibility") != ELIGIBILITY:
        raise ValueError("galaxy evaluator context eligibility is not fail-closed")
    if candidate.get("data_eligibility") != ELIGIBILITY:
        raise ValueError("candidate eligibility is not fail-closed")
    provenance = candidate.get("direct_observable_prediction_provenance")
    if provenance is None:
        decision = "blocked"
        blocker = "missing_candidate_direct_observable_prediction_bundle"
        bundle_id = None
    else:
        required = {
            "bundle_id",
            "bundle_binding_sha256",
            "candidate_action_sha256",
            "prediction_content_sha256",
            "observable_contract_sha256",
        }
        if not isinstance(provenance, dict) or set(provenance) != required:
            raise ValueError("galaxy prediction provenance fields are not exact")
        if any(
            key in provenance
            for key in ("dark_matter_halo", "redshift_distance", "observations")
        ):
            raise ValueError("prediction provenance contains forbidden opened inputs")
        bundle_id = str(provenance["bundle_id"])
        registered = PREDICTION_BUNDLES.get(bundle_id)
        if registered is None:
            decision = "blocked"
            blocker = "unregistered_candidate_direct_observable_prediction_bundle"
        elif _canonical(provenance) != _canonical(registered):
            raise ValueError("galaxy prediction bundle binding mismatch")
        else:  # pragma: no cover - registry intentionally sealed in this version
            decision = "blocked"
            blocker = "direct_observable_data_access_not_authorized"
    return {
        "decision": decision,
        "blocker": blocker,
        "candidate_id": str(candidate["candidate_id"]),
        "input_lineage_sha256": str(context["input_lineage_sha256"]),
        "evaluator_id": EVALUATOR_ID,
        "evaluator_version": EVALUATOR_VERSION,
        "prediction_bundle_id": bundle_id,
        "registered_prediction_bundle_count": len(PREDICTION_BUNDLES),
        "observational_data_opened": False,
        "source_registrations_loaded": 0,
        "dark_matter_or_halo_inputs": False,
        "redshift_distance_inputs": False,
        "scope": (
            "sealed direct-observable galaxy evaluator scaffold; no score or scientific "
            "decision exists until an exact candidate prediction bundle and separately "
            "authorized observable contract are registered"
        ),
        "data_eligibility": dict(ELIGIBILITY),
    }
