from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .promotion_orchestrator import ELIGIBILITY

EVALUATOR_ID = "solar-known-answer-controls-v1"
EVALUATOR_VERSION = "1.0.0"


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


GR_SOLAR_BUNDLE = {
    "bundle_id": "einstein-hilbert-gr-solar-known-answer-v1",
    "role": "known_answer_control",
    "candidate_id": "KNOWN-ANSWER-EINSTEIN-HILBERT",
    "input_action_sha256": "8965f95177ca7e7d798d6163d184d62c5fa3aba0a7d11f32b407f71976d08d73",
    "artifacts": {
        "action": {
            "path": "runs/formal-controls-v1/action-health/einstein_hilbert_control/action-ir.json",
            "file_sha256": "85b6e0d05d31d1dadf9c725cab84d900a998d83378c3354124b78086b8d29709",
        },
        "health": {
            "path": "runs/formal-controls-v1/action-health/einstein_hilbert_control/action-health.json",
            "file_sha256": "d1a2fcc02d9453ad3c6c0ca744f7fc13a5477e471038d8454822054319a8ec93",
        },
        "reference": {
            "path": "runs/gr-reference/relativity_reference.json",
            "file_sha256": "b4ec6cdcfd53237896cdd9daef428c2dced8bbaf30d32a68834f21719080b457",
        },
    },
}

BUNDLES = {GR_SOLAR_BUNDLE["bundle_id"]: GR_SOLAR_BUNDLE}

REQUIRED_FORMAL_GATES = (
    "adm_decomposition",
    "adm_dirac",
    "covariant_identity",
    "covariant_variation",
    "field_contract",
    "generated_dirac_closure",
    "hamiltonian_stability",
    "higher_jet_regularity",
    "legendre_map",
    "parameter_domain",
    "principal_symbol",
    "static_dictionary_derivation",
)

REQUIRED_GOLDEN_CHECKS = {
    "schwarzschild_vacuum",
    "gr_ppn_recovery",
    "mercury_perihelion",
    "solar_limb_light_deflection",
    "shapiro_delay_geometry_control",
}


def bundle_binding(bundle: dict[str, Any]) -> str:
    return _sha(bundle)


def _load_artifact(root: Path, spec: dict[str, Any]) -> dict[str, Any]:
    path = root / spec["path"]
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != spec["file_sha256"]:
        raise ValueError(f"Solar artifact file hash mismatch: {spec['path']}")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise TypeError("Solar artifact must be a JSON object")
    return value


def _load_bundle(bundle: dict[str, Any]) -> dict[str, dict[str, Any]]:
    root = Path(__file__).resolve().parents[2]
    loaded = {
        name: _load_artifact(root, spec)
        for name, spec in bundle["artifacts"].items()
    }
    action_hash = bundle["input_action_sha256"]
    action = loaded["action"]
    if (
        action.get("content_sha256") != action_hash
        or _sha(action.get("canonical")) != action_hash
        or action.get("canonical", {}).get("source_role") != "known_answer_control"
    ):
        raise ValueError("Solar bundle action identity mismatch")
    health = loaded["health"]
    if (
        health.get("schema_version") != "sigma-action-health-1.0"
        or health.get("input_action_sha256") != action_hash
        or health.get("status") != "pass"
        or health.get("promotion_allowed") is not True
        or health.get("observational_gates_unsealed") is not False
    ):
        raise ValueError("Solar bundle formal-health prerequisite mismatch")
    reference = loaded["reference"]
    prerequisite = reference.get("formal_prerequisite", {})
    if (
        reference.get("schema_version") != "sigma-relativity-reference-1.0"
        or prerequisite.get("input_action_sha256") != action_hash
        or prerequisite.get("status") != "eligible"
        or prerequisite.get("mode") != "known_answer_reference"
        or prerequisite.get("observational_dataset_opened") is not False
        or prerequisite.get("candidate_dataset_manifest_may_be_audited") is not False
        or prerequisite.get("redshift_distance_allowed_by_default") is not False
        or prerequisite.get("supernova_default_status") != "excluded"
    ):
        raise ValueError("Solar reference prerequisite is not sealed and action-bound")
    gate_statuses = prerequisite.get("formal_gate_statuses", {})
    if any(gate_statuses.get(name) != "pass" for name in REQUIRED_FORMAL_GATES):
        raise ValueError("Solar reference formal gate is not passed")
    return loaded


def _golden_status(reference: dict[str, Any]) -> dict[str, str]:
    checks = reference.get("golden_checks")
    if not isinstance(checks, list):
        raise TypeError("Solar golden checks must be a list")
    statuses = {str(item.get("name")): str(item.get("status")) for item in checks}
    if set(statuses) != REQUIRED_GOLDEN_CHECKS:
        raise ValueError("Solar golden-check set mismatch")
    counts = reference.get("counts", {})
    if (
        counts.get("golden_total") != len(REQUIRED_GOLDEN_CHECKS)
        or counts.get("passed") != len(REQUIRED_GOLDEN_CHECKS)
        or counts.get("failed") != 0
        or counts.get("blocked") != 0
    ):
        raise ValueError("Solar golden-check accounting mismatch")
    return statuses


def _result(
    candidate: dict[str, Any],
    context: dict[str, Any],
    *,
    decision: str,
    blocker: str | None,
    bundle: dict[str, Any] | None,
    golden_statuses: dict[str, str] | None,
) -> dict[str, Any]:
    result = {
        "decision": decision,
        "candidate_id": str(candidate["candidate_id"]),
        "input_lineage_sha256": str(context["input_lineage_sha256"]),
        "evaluator_id": EVALUATOR_ID,
        "evaluator_version": EVALUATOR_VERSION,
        "bundle_id": bundle["bundle_id"] if bundle else None,
        "bundle_binding_sha256": bundle_binding(bundle) if bundle else None,
        "input_action_sha256": bundle["input_action_sha256"] if bundle else None,
        "golden_statuses": golden_statuses,
        "scope": (
            "action-hash-bound GR/Schwarzschild/PPN/Mercury/light-deflection/Shapiro "
            "known-answer controls only; no candidate observation is opened, no fitted PPN "
            "label is used as a search target, and pass is not observational support or novelty"
        ),
        "data_eligibility": dict(ELIGIBILITY),
    }
    if blocker is not None:
        result["blocker"] = blocker
    return result


def solar_known_answer_evaluator(
    candidate: dict[str, Any], context: dict[str, Any]
) -> dict[str, Any]:
    """Run Solar controls only for an exact immutable action-to-weak-field bundle."""

    if context.get("data_eligibility") != ELIGIBILITY:
        raise ValueError("promotion context eligibility is not fail-closed")
    provenance = candidate.get("solar_control_provenance")
    if provenance is None:
        return _result(
            candidate,
            context,
            decision="blocked",
            blocker="missing_exact_action_bound_solar_control_bundle",
            bundle=None,
            golden_statuses=None,
        )
    if not isinstance(provenance, dict) or set(provenance) != {
        "bundle_id",
        "bundle_binding_sha256",
        "input_action_sha256",
    }:
        raise ValueError("Solar control provenance fields are not exact")
    bundle = BUNDLES.get(str(provenance["bundle_id"]))
    if bundle is None:
        return _result(
            candidate,
            context,
            decision="blocked",
            blocker="unregistered_action_bound_solar_control_bundle",
            bundle=None,
            golden_statuses=None,
        )
    expected = {
        "bundle_id": bundle["bundle_id"],
        "bundle_binding_sha256": bundle_binding(bundle),
        "input_action_sha256": bundle["input_action_sha256"],
    }
    if provenance != expected:
        raise ValueError("Solar control provenance hash mismatch")
    if candidate.get("candidate_id") != bundle["candidate_id"]:
        raise ValueError("Solar known-answer bundle cannot be attached to a discovery candidate")
    loaded = _load_bundle(bundle)
    statuses = _golden_status(loaded["reference"])
    if any(status != "pass" for status in statuses.values()):
        return _result(
            candidate,
            context,
            decision="reject",
            blocker=None,
            bundle=bundle,
            golden_statuses=statuses,
        )
    return _result(
        candidate,
        context,
        decision="pass",
        blocker=None,
        bundle=bundle,
        golden_statuses=statuses,
    )
