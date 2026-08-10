from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .promotion_orchestrator import ELIGIBILITY

EVALUATOR_ID = "adm-dirac-principal-health-v1"
EVALUATOR_VERSION = "1.0.0"


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


KNOWN_ANSWER_BUNDLE = {
    "bundle_id": "einstein-aether-known-answer-v1",
    "role": "known_answer_control",
    "candidate_id": "KNOWN-ANSWER-EINSTEIN-AETHER",
    "input_action_sha256": "fceb6bc4b4329e6d12cae9f0ae5c0794b06745530a66c90d54f321cd278a0c35",
    "artifacts": {
        "action": {
            "path": "runs/formal-controls-v1/action-health/einstein_aether_control/action-ir.json",
            "file_sha256": "e434aa26ddd0f7e9e5771fa097677e4c5e077790a5c6db292714712e4a6bc01a",
            "content_sha256": "fceb6bc4b4329e6d12cae9f0ae5c0794b06745530a66c90d54f321cd278a0c35",
        },
        "adm": {
            "path": "runs/formal-controls-v1/action-health/einstein_aether_control/adm-ir.json",
            "file_sha256": "7dcb7d5339f1f5c395fb093bc02203a16e68b624c4a9f89dd71bf2b52a224194",
            "content_sha256": "9c133f446be644d9368b55b266593806cbc6764736507424c8227ad90bf44129",
        },
        "dirac": {
            "path": "runs/formal-controls-v1/action-health/einstein_aether_control/dirac-ir.json",
            "file_sha256": "4080190e418b619dd7f4ef0d1aa6a3e2de09468109e5fd8eca3a83445d01841f",
            "content_sha256": "1e7df0c40de875db4489ec3ac55e1ce8745f139435d2297855bf3c995aeb3ca1",
        },
        "principal": {
            "path": "runs/formal-controls-v1/action-health/einstein_aether_control/principal-ir.json",
            "file_sha256": "5fe2df9b148a604ade2992bd279b22b771b256182980bca8ab7ec95864d1e677",
            "content_sha256": "e9a2d269368adb01bf645ec39e9b09637704bc44c9f9070235cd8a2042aadf7c",
        },
        "health": {
            "path": "runs/formal-controls-v1/action-health/einstein_aether_control/action-health.json",
            "file_sha256": "aa6dfcc2f8d1a1b5bb1efa77f3ea010b74901fcde768e403647ed27d3092b9b0",
            "content_sha256": None,
        },
    },
}

GENERATED_REJECT_BUNDLE = {
    "bundle_id": "generated-q-q2-candidate-v1",
    "role": "generated_candidate",
    "candidate_id": "GF-cb4ebf3da5a74582",
    "ordinal": 723,
    "correction_expression": "+(q)+(q**2)",
    "input_action_sha256": "e4e4c16633d59e05905058b781d126b22bf53eeacc52e664ed7e28f5469216f2",
    "artifacts": {
        "action": {
            "path": "runs/generated-candidates/GF-cb4ebf3da5a74582/formal-health/action-ir.json",
            "file_sha256": "995a86614feced2161d34e81d97f7d207295610866458445c5a88c64b72b97e4",
            "content_sha256": "e4e4c16633d59e05905058b781d126b22bf53eeacc52e664ed7e28f5469216f2",
        },
        "adm": {
            "path": "runs/generated-candidates/GF-cb4ebf3da5a74582/formal-health/adm-ir.json",
            "file_sha256": "78e19871bf5c4cedc1a43fbe588b81de42296bb60c5295e8e67456abcfbe0274",
            "content_sha256": "947bca72c2f2d27f81f6aac909d5d406d3984435b65a4bbe3c80566e77c95088",
        },
        "dirac": {
            "path": "runs/generated-candidates/GF-cb4ebf3da5a74582/formal-health/dirac-ir.json",
            "file_sha256": "a2aed3c3b6d47b3f81dc10e7fb65ad2c91fe6cc0c9113dbd15443dfcb96f99f1",
            "content_sha256": "3f8839f3f3ea1a0bd24f2c4a5e75dab23f7dead7a8ac404ad8c8bed98a297d7f",
        },
        "principal": {
            "path": "runs/generated-candidates/GF-cb4ebf3da5a74582/formal-health/principal-ir.json",
            "file_sha256": "f86256a9fa6b307856c1ce905fa6290b117a6c625a489c66101c5cf0aadb78b1",
            "content_sha256": None,
        },
        "health": {
            "path": "runs/generated-candidates/GF-cb4ebf3da5a74582/formal-health/action-health.json",
            "file_sha256": "09a77e40aa769dd9dee48cee3037b3389726f17784f429a13e0beb5480a5ddb8",
            "content_sha256": None,
        },
    },
}

BUNDLES = {
    bundle["bundle_id"]: bundle for bundle in (KNOWN_ANSWER_BUNDLE, GENERATED_REJECT_BUNDLE)
}


def bundle_binding(bundle: dict[str, Any]) -> str:
    return _sha(bundle)


def _load_artifact(root: Path, name: str, spec: dict[str, Any]) -> dict[str, Any]:
    path = root / spec["path"]
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != spec["file_sha256"]:
        raise ValueError(f"formal artifact file hash mismatch: {spec['path']}")
    value = json.loads(raw)
    expected_content = spec["content_sha256"]
    if expected_content is not None:
        declared = value.get("content_sha256")
        body = {key: item for key, item in value.items() if key != "content_sha256"}
        content = value.get("canonical") if name == "action" else body
        if declared != expected_content or _sha(content) != declared:
            raise ValueError(f"formal artifact content hash mismatch: {spec['path']}")
    return value


def _load_bundle(bundle: dict[str, Any]) -> dict[str, dict[str, Any]]:
    root = Path(__file__).resolve().parents[2]
    loaded = {
        name: _load_artifact(root, name, spec)
        for name, spec in bundle["artifacts"].items()
    }
    action_hash = bundle["input_action_sha256"]
    action = loaded["action"]
    if action.get("content_sha256") != action_hash:
        raise ValueError("formal bundle action hash mismatch")
    expected_schemas = {
        "adm": "sigma-adm-ir-1.0",
        "dirac": "sigma-dirac-ir-1.0",
        "principal": "sigma-physical-principal-ir-1.0",
        "health": "sigma-action-health-1.0",
    }
    for name, schema in expected_schemas.items():
        artifact = loaded[name]
        if artifact.get("schema_version") != schema:
            raise ValueError(f"unsupported {name} artifact schema")
        if artifact.get("input_action_sha256") != action_hash:
            raise ValueError(f"{name} artifact is not action-hash-bound")
    if loaded["dirac"].get("input_adm_ir_sha256") != loaded["adm"].get(
        "content_sha256"
    ):
        raise ValueError("Dirac artifact is not ADM-hash-bound")
    if loaded["principal"].get("input_dirac_ir_sha256") != loaded["dirac"].get(
        "content_sha256"
    ):
        raise ValueError("principal artifact is not Dirac-hash-bound")
    health = loaded["health"]
    for name in ("adm", "dirac", "principal"):
        generated = health.get(f"generated_{name}_ir", {})
        expected = loaded[name].get("content_sha256")
        if generated.get("content_sha256") != expected and not (
            name == "principal" and expected is None
        ):
            raise ValueError(f"health artifact is not {name}-hash-bound")
    return loaded


def _result(
    candidate: dict[str, Any],
    context: dict[str, Any],
    *,
    decision: str,
    blocker: str | None,
    bundle: dict[str, Any] | None,
    gate_statuses: dict[str, str] | None,
) -> dict[str, Any]:
    result = {
        "decision": decision,
        "candidate_id": str(candidate["candidate_id"]),
        "input_lineage_sha256": str(context["input_lineage_sha256"]),
        "evaluator_id": EVALUATOR_ID,
        "evaluator_version": EVALUATOR_VERSION,
        "bundle_id": bundle["bundle_id"] if bundle is not None else None,
        "bundle_binding_sha256": bundle_binding(bundle) if bundle is not None else None,
        "input_action_sha256": (
            bundle["input_action_sha256"] if bundle is not None else None
        ),
        "gate_statuses": gate_statuses,
        "scope": (
            "action-hash-bound ADM decomposition, Dirac closure, Legendre/higher-jet "
            "regularity, parameter-domain, and physical principal-symbol classification "
            "only; it is not nonlinear total-energy proof, observational support, novelty, "
            "or a gravity-theory promotion"
        ),
        "data_eligibility": dict(ELIGIBILITY),
    }
    if blocker is not None:
        result["blocker"] = blocker
    return result


def adm_dirac_principal_health_evaluator(
    candidate: dict[str, Any], context: dict[str, Any]
) -> dict[str, Any]:
    """Evaluate only an explicit, immutable candidate-to-covariant-action mapping."""

    if context.get("data_eligibility") != ELIGIBILITY:
        raise ValueError("promotion context eligibility is not fail-closed")
    provenance = candidate.get("covariant_action_provenance")
    if provenance is None:
        return _result(
            candidate,
            context,
            decision="blocked",
            blocker="missing_exact_candidate_to_covariant_action_map",
            bundle=None,
            gate_statuses=None,
        )
    if not isinstance(provenance, dict) or set(provenance) != {
        "bundle_id",
        "bundle_binding_sha256",
        "input_action_sha256",
    }:
        raise ValueError("covariant action provenance fields are not exact")
    bundle = BUNDLES.get(str(provenance["bundle_id"]))
    if bundle is None:
        return _result(
            candidate,
            context,
            decision="blocked",
            blocker="unregistered_covariant_action_bundle",
            bundle=None,
            gate_statuses=None,
        )
    expected_provenance = {
        "bundle_id": bundle["bundle_id"],
        "bundle_binding_sha256": bundle_binding(bundle),
        "input_action_sha256": bundle["input_action_sha256"],
    }
    if provenance != expected_provenance:
        raise ValueError("candidate covariant action provenance hash mismatch")
    loaded = _load_bundle(bundle)
    action = loaded["action"]
    if bundle["role"] == "known_answer_control":
        if candidate.get("candidate_id") != bundle["candidate_id"]:
            raise ValueError("known-answer bundle cannot be attached to a discovery candidate")
    else:
        origin = action.get("canonical", {}).get("generator_origin", {})
        if (
            candidate.get("candidate_id") != bundle["candidate_id"]
            or int(candidate.get("ordinal", -1)) != bundle["ordinal"]
            or candidate.get("correction_expression") != bundle["correction_expression"]
            or origin.get("family_id") != bundle["candidate_id"]
            or int(origin.get("ordinal", -1)) != bundle["ordinal"]
            or origin.get("correction_expression") != bundle["correction_expression"]
        ):
            raise ValueError("generated candidate identity does not match its action IR")
    health_gates = loaded["health"].get("gates", {})
    gate_names = (
        "field_contract",
        "adm_decomposition",
        "adm_dirac",
        "generated_dirac_closure",
        "legendre_map",
        "higher_jet_regularity",
        "parameter_domain",
        "principal_symbol",
    )
    gate_statuses = {
        name: str(health_gates.get(name, {}).get("status", "unresolved"))
        for name in gate_names
    }
    gate_statuses.update(
        {
            "generated_adm_ir": str(loaded["adm"].get("status", "unresolved")),
            "generated_dirac_ir": str(loaded["dirac"].get("status", "unresolved")),
            "generated_principal_ir": str(
                loaded["principal"].get("status", "unresolved")
            ),
        }
    )
    rejected = sorted(name for name, status in gate_statuses.items() if status == "reject")
    unresolved = sorted(name for name, status in gate_statuses.items() if status != "pass")
    if rejected:
        return _result(
            candidate,
            context,
            decision="reject",
            blocker=None,
            bundle=bundle,
            gate_statuses=gate_statuses,
        ) | {"rejected_gates": rejected}
    if unresolved:
        return _result(
            candidate,
            context,
            decision="blocked",
            blocker="unresolved_adm_dirac_or_principal_gate",
            bundle=bundle,
            gate_statuses=gate_statuses,
        ) | {"unresolved_gates": unresolved}
    return _result(
        candidate,
        context,
        decision="pass",
        blocker=None,
        bundle=bundle,
        gate_statuses=gate_statuses,
    )
