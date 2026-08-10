from __future__ import annotations

import copy
import hashlib
import importlib
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

from .cubic_bssn_domain import certify_cubic_bssn_domain
from .promotion_orchestrator import ELIGIBILITY

SCHEMA_VERSION = "sigma-g3-componentwise-interval-domain-audit-1.0"


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_bound(root: Path, descriptor: dict[str, Any], *, content: bool = False) -> dict[str, Any]:
    path = root / descriptor["path"]
    if _file_sha(path) != descriptor["file_sha256"]:
        raise ValueError(f"bound file hash mismatch: {descriptor['path']}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {descriptor['path']}")
    if content:
        body = {key: item for key, item in value.items() if key != "content_sha256"}
        if value.get("content_sha256") != descriptor["content_sha256"] or _sha(body) != descriptor[
            "content_sha256"
        ]:
            raise ValueError(f"bound content hash mismatch: {descriptor['path']}")
    return value


def _resolve(entrypoint: str) -> Any:
    module_name, separator, attribute = entrypoint.partition(":")
    if not separator:
        raise ValueError("adapter must use module:function syntax")
    callback = getattr(importlib.import_module(module_name), attribute, None)
    if not callable(callback):
        raise TypeError(f"adapter is not callable: {entrypoint}")
    return callback


def _validate_target(record: dict[str, Any], target: dict[str, Any]) -> None:
    if record.get("seed_id") != target["seed_id"] or record.get("decision") != "blocked":
        raise ValueError("G3 predecessor identity or decision mismatch")
    if record.get("action_sha256") != target["action_sha256"]:
        raise ValueError("G3 action hash mismatch")
    if record["provenance"].get("binding_sha256") != target["predecessor_provenance_sha256"]:
        raise ValueError("G3 predecessor provenance mismatch")
    if (
        record["candidate_certificate"].get("content_sha256")
        != target["prior_candidate_certificate_sha256"]
    ):
        raise ValueError("G3 predecessor certificate mismatch")
    if record.get("first_missing_uniform_principal_premise") != (
        "componentwise_normalized_local_jet_box"
    ):
        raise ValueError("G3 predecessor principal blocker changed")


def _fraction(value: str) -> float:
    return float(Fraction(value))


def _candidate_adapter_inputs(
    target: dict[str, Any], domain: dict[str, Any], run: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    required_keys = {
        "contract_kind",
        "normalization",
        "frame",
        "phi_interval",
        "anchor_X_interval",
        "normal_gradient_padding",
        "spatial_gradient_component_abs",
        "symmetric_hessian_component_abs",
        "riemann_tetrad_component_abs",
        "lapse_interval",
        "lapse_log_spatial_gradient_component_abs",
        "extrinsic_curvature_component_abs",
        "slicing",
        "direction_sphere",
    }
    if set(domain) != required_keys:
        raise ValueError("G3 componentwise domain fields changed")
    if domain["contract_kind"] != "pointwise_local_jet_box_not_evolution_invariant":
        raise ValueError("G3 componentwise domain scope changed")
    phi_lower, phi_upper = map(Fraction, domain["phi_interval"])
    x_lower, x_upper = map(Fraction, domain["anchor_X_interval"])
    lapse_lower, lapse_upper = map(Fraction, domain["lapse_interval"])
    if not (
        phi_lower < phi_upper
        and 0 < x_lower < x_upper
        and 0 < lapse_lower < lapse_upper
        and domain["direction_sphere"].endswith("no sampling")
    ):
        raise ValueError("G3 componentwise domain is empty or malformed")
    ir_body = {
        "schema_version": "sigma-g3-candidate-bssn-adapter-ir-1.0",
        "source_action_sha256": target["action_sha256"],
        "normalization": domain["normalization"],
        "formulation_classification": {
            "canonical_G2": "x",
            "canonical_G3": "x/100",
            "G4_X": "0",
        },
        "derivation": {
            "compiler_G2": target["G2"],
            "compiler_G3": target["G3"],
            "compiler_G4": target["G4"],
            "source_normalized_G2": "0",
            "source_normalized_G3": "-x/100",
        },
    }
    ir = {**ir_body, "content_sha256": _sha(ir_body)}
    anchor_body = {
        "schema_version": "sigma-g3-local-box-anchor-1.0",
        "status": "pass_interval_certified",
        "role": "candidate_local_box_anchor_not_on_shell_trajectory",
        "coefficients": {},
        "trajectory_hull": {
            "u": {"lower": float(phi_lower), "upper": float(phi_upper)},
            "x": {"lower": float(x_lower), "upper": float(x_upper)},
        },
        "source_action_sha256": target["action_sha256"],
    }
    anchor = {**anchor_body, "content_sha256": _sha(anchor_body)}
    interval_config = {
        **run,
        "momentum_parameter_m": _fraction(domain["slicing"]["BSSN_m"]),
        "slicing_parameter_sigma": _fraction(domain["slicing"]["BSSN_sigma"]),
        "domain_extension": {
            "phi_padding": 0.0,
            "normal_gradient_padding": _fraction(domain["normal_gradient_padding"]),
            "spatial_gradient_abs": _fraction(domain["spatial_gradient_component_abs"]),
            "hessian_component_abs": _fraction(domain["symmetric_hessian_component_abs"]),
            "riemann_component_abs": _fraction(domain["riemann_tetrad_component_abs"]),
        },
        "frame_binding": domain["frame"],
        "lapse_interval": domain["lapse_interval"],
        "lapse_log_spatial_gradient_component_abs": domain[
            "lapse_log_spatial_gradient_component_abs"
        ],
        "extrinsic_curvature_component_abs": domain["extrinsic_curvature_component_abs"],
        "direction_sphere": domain["direction_sphere"],
    }
    return ir, anchor, interval_config


def _lapse_prerequisite(
    domain: dict[str, Any], lapse_evidence: dict[str, Any]
) -> dict[str, Any]:
    lower, upper = map(Fraction, domain["lapse_interval"])
    if lapse_evidence.get("generic_G2_lapse_hessian_contribution") != (
        "(G2_X*N**2 + G2_XX)/N**5"
    ):
        raise ValueError("generic G2 lapse contribution changed")
    if lapse_evidence["capability_boundary"].get("global_Delta_N_invertibility") != "unresolved":
        raise ValueError("generic lapse capability boundary changed")
    body = {
        "gauge": "timelike-gradient unitary gauge for the Dirac prerequisite only",
        "bound_lapse_interval": [str(lower), str(upper)],
        "positive_G2_multiplication_contribution": {
            "exact": "Delta_N^(G2)=N^-3",
            "lower": str(upper**-3),
            "upper": str(lower**-3),
            "uniformly_positive": True,
        },
        "full_candidate_operator": "Delta_N=M_(N^-3)+Delta_N^(G3)",
        "G3_remainder": {
            "status": "blocked",
            "reason": (
                "the action-specific functional lapse Hessian of -(X/100) box(phi), including "
                "derivative and boundary terms, is not derived by the existing adapter"
            ),
        },
        "local_full_operator_invertibility": "blocked",
        "global_boundary_domain": {
            "status": "blocked",
            "missing": [
                "explicit Delta_N^(G3) differential operator",
                "operator domain and asymptotic or inner boundary conditions",
                "coercive estimate or zero-mode exclusion for the complete operator",
            ],
        },
        "inference_rule": "positivity of one additive multiplication term cannot certify the full operator",
    }
    return {**body, "content_sha256": _sha(body)}


def build_g3_componentwise_interval_domain_audit(
    config: dict[str, Any], root: str | Path
) -> dict[str, Any]:
    root = Path(root).resolve()
    if config.get("data_eligibility") != ELIGIBILITY:
        raise ValueError("campaign eligibility is not fail-closed")
    predecessor = _load_bound(root, config["predecessor"], content=True)
    formal_report = _load_bound(root, config["formal_report"])
    for descriptor in config["adapter_sources"]:
        if _file_sha(root / descriptor["path"]) != descriptor["file_sha256"]:
            raise ValueError(f"adapter source hash mismatch: {descriptor['path']}")
    target = config["target_seed"]
    prior = next(item for item in predecessor["candidate_records"] if item["seed_id"] == target["seed_id"])
    _validate_target(prior, target)
    lapse_descriptor = config["lapse_adapter"]
    formal_check = next(
        (
            item
            for item in formal_report["checks"]
            if item["name"] == lapse_descriptor["id"]
        ),
        None,
    )
    passed, lapse_evidence = _resolve(lapse_descriptor["entrypoint"])()
    if (
        formal_check is None
        or formal_check["status"] != "pass"
        or not passed
        or lapse_evidence != formal_check["evidence"]
    ):
        raise ValueError("G3 lapse adapter replay mismatch")
    ir, anchor, interval_config = _candidate_adapter_inputs(
        target, config["componentwise_domain"], config["interval_run"]
    )
    principal = certify_cubic_bssn_domain(ir, anchor, interval_config)
    if principal.get("status") != "pass_uniform_local_jet_box":
        raise ValueError("candidate componentwise BSSN interval did not certify")
    bad_gradient = copy.deepcopy(interval_config)
    bad_gradient["domain_extension"]["spatial_gradient_abs"] = 1.1
    bad_gradient_result = certify_cubic_bssn_domain(ir, anchor, bad_gradient)
    bad_slicing = copy.deepcopy(interval_config)
    bad_slicing["slicing_parameter_sigma"] = 0.5
    bad_slicing_result = certify_cubic_bssn_domain(ir, anchor, bad_slicing)
    if bad_gradient_result.get("status") != "reject" or bad_slicing_result.get("status") != "reject":
        raise ValueError("G3 interval negative controls failed")
    lapse = _lapse_prerequisite(config["componentwise_domain"], lapse_evidence)
    gates = {
        "typed_action_and_predecessor": {"status": "pass"},
        "componentwise_domain_binding": {"status": "pass"},
        "uniform_principal_symbol": {"status": "pass"},
        "uniform_common_time_and_BSSN_slicing_cone": {"status": "pass"},
        "direction_sphere_coverage": {"status": "pass", "method": "no sampling"},
        "positive_G2_lapse_suboperator": {"status": "pass"},
        "complete_candidate_Delta_N": {"status": "blocked"},
        "distributed_Dirac_and_global_lapse": {"status": "blocked"},
        "global_hamiltonian_energy": {"status": "blocked"},
        "formal_prerequisite_completion": {"status": "blocked"},
    }
    provenance_body = {
        "predecessor_content_sha256": config["predecessor"]["content_sha256"],
        "action_sha256": target["action_sha256"],
        "predecessor_provenance_sha256": target["predecessor_provenance_sha256"],
        "prior_candidate_certificate_sha256": target["prior_candidate_certificate_sha256"],
        "componentwise_domain_sha256": _sha(config["componentwise_domain"]),
        "candidate_adapter_ir_sha256": ir["content_sha256"],
        "local_box_anchor_sha256": anchor["content_sha256"],
        "principal_certificate_sha256": principal["content_sha256"],
        "lapse_certificate_sha256": lapse["content_sha256"],
        "lapse_adapter_evidence_sha256": _sha(lapse_evidence),
        "data_eligibility": ELIGIBILITY,
    }
    record = {
        "seed_id": target["seed_id"],
        "action_sha256": target["action_sha256"],
        "decision": "blocked",
        "componentwise_domain": config["componentwise_domain"],
        "candidate_adapter_ir": ir,
        "local_box_anchor": anchor,
        "principal_common_cone_certificate": principal,
        "lapse_prerequisite": lapse,
        "negative_controls": {
            "non_timelike_gradient_box": bad_gradient_result,
            "invalid_BSSN_sigma": bad_slicing_result,
        },
        "gate_ledger": gates,
        "resolved_predecessor_blocker": "componentwise_normalized_local_jet_box",
        "first_missing_premise": "candidate_specific_full_Delta_N_operator",
        "necessary_condition_rejection_found": False,
        "provenance": {**provenance_body, "binding_sha256": _sha(provenance_body)},
        "solar_bundle": {"generated": False, "status": "blocked"},
    }
    body = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": config["campaign_id"],
        "source_bindings": {
            "predecessor": config["predecessor"],
            "formal_report": config["formal_report"],
            "adapter_sources": config["adapter_sources"],
        },
        "invoked_adapter_entrypoints": [
            "sigma_theory_compiler.cubic_bssn_domain:certify_cubic_bssn_domain",
            lapse_descriptor["entrypoint"],
        ],
        "target_seed_count": 1,
        "decision_counts": {"blocked": 1},
        "candidate_records": [record],
        "uniform_principal_common_cone_pass_count": 1,
        "full_formal_pass_count": 0,
        "solar_bundle_count": 0,
        "observational_data_opened": False,
        "paid_llm_spend_usd": 0.0,
        "data_eligibility": dict(ELIGIBILITY),
        "interpretation": (
            "The qualitative weak-cell label is replaced by a nonzero componentwise box. The "
            "existing interval machinery proves a uniform scalar principal symbol, common time "
            "covector, and separated BSSN slicing cone over every spatial direction. The full "
            "candidate lapse Hessian remains blocked because only its positive G2 multiplication "
            "part is available; the G3 differential/boundary remainder is not derived."
        ),
    }
    return {**body, "content_sha256": _sha(body)}
