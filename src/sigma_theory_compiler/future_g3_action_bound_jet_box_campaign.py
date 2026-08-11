"""Action-bound local-jet and periodic Dirac certificates for future cubic G3 actions."""

from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

from .cubic_bssn_domain import certify_cubic_bssn_domain
from .g3_full_lapse_dirac_operator_audit import (
    _coercivity_certificate,
    _derive_full_delta,
)
from .promotion_orchestrator import ELIGIBILITY

CONFIG_SCHEMA = "sigma-future-g3-action-bound-jet-box-config-1.0"
ARTIFACT_SCHEMA = "sigma-future-g3-action-bound-jet-box-campaign-1.0"
FIRST_BLOCKER = "asymptotically_flat_or_global_energy_domain_missing"
DOMAIN_FIELDS = [
    "normalization_binding",
    "frame_binding",
    "phi_interval",
    "anchor_X_interval",
    "normal_gradient_padding",
    "spatial_gradient_component_abs",
    "symmetric_hessian_component_abs",
    "riemann_tetrad_component_abs",
    "lapse_interval",
    "lapse_log_spatial_gradient_component_abs",
    "extrinsic_curvature_component_abs",
    "slicing_BSSN_sigma_m",
]


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_bound(root: Path, binding: dict[str, Any], label: str) -> dict[str, Any]:
    path = (root / binding["path"]).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} path escapes repository") from error
    if not path.is_file() or _file_sha(path) != binding["file_sha256"]:
        raise ValueError(f"{label} file hash mismatch")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{label} must contain an object")
    if "content_sha256" in binding:
        body = {key: item for key, item in value.items() if key != "content_sha256"}
        if (
            value.get("content_sha256") != binding["content_sha256"]
            or _sha(body) != binding["content_sha256"]
        ):
            raise ValueError(f"{label} content hash mismatch")
    return value


def _validate_domain(domain: dict[str, Any]) -> None:
    expected_keys = {
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
    if set(domain) != expected_keys or domain["contract_kind"] != (
        "candidate_action_bound_pointwise_local_jet_box_not_evolution_invariant"
    ):
        raise ValueError("future G3 action-bound componentwise domain changed")
    phi_lower, phi_upper = map(Fraction, domain["phi_interval"])
    x_lower, x_upper = map(Fraction, domain["anchor_X_interval"])
    lapse_lower, lapse_upper = map(Fraction, domain["lapse_interval"])
    positive = [
        Fraction(domain[key])
        for key in (
            "normal_gradient_padding",
            "spatial_gradient_component_abs",
            "symmetric_hessian_component_abs",
            "riemann_tetrad_component_abs",
            "lapse_log_spatial_gradient_component_abs",
            "extrinsic_curvature_component_abs",
        )
    ]
    if not (
        phi_lower < phi_upper
        and 0 < x_lower < x_upper
        and 0 < lapse_lower < lapse_upper
        and all(value > 0 for value in positive)
        and domain["slicing"] == {"BSSN_sigma": "1", "BSSN_m": "1"}
        and domain["direction_sphere"].endswith("no sampling")
    ):
        raise ValueError("future G3 action-bound componentwise box is empty or malformed")


def _registered_values(domain: dict[str, Any]) -> dict[str, Any]:
    return {
        "normalization_binding": domain["normalization"],
        "frame_binding": domain["frame"],
        "phi_interval": domain["phi_interval"],
        "anchor_X_interval": domain["anchor_X_interval"],
        "normal_gradient_padding": domain["normal_gradient_padding"],
        "spatial_gradient_component_abs": domain["spatial_gradient_component_abs"],
        "symmetric_hessian_component_abs": domain["symmetric_hessian_component_abs"],
        "riemann_tetrad_component_abs": domain["riemann_tetrad_component_abs"],
        "lapse_interval": domain["lapse_interval"],
        "lapse_log_spatial_gradient_component_abs": domain[
            "lapse_log_spatial_gradient_component_abs"
        ],
        "extrinsic_curvature_component_abs": domain["extrinsic_curvature_component_abs"],
        "slicing_BSSN_sigma_m": domain["slicing"],
    }


def _adapter_inputs(
    target: dict[str, Any], domain: dict[str, Any], run: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    beta = Fraction(target["beta"])
    expression = f"{beta.numerator}*x/{beta.denominator}"
    ir_body = {
        "schema_version": "sigma-g3-candidate-bssn-adapter-ir-1.0",
        "source_action_sha256": target["action_sha256"],
        "normalization": domain["normalization"],
        "formulation_classification": {
            "canonical_G2": "x",
            "canonical_G3": expression,
            "G4_X": "0",
        },
        "derivation": {
            "compiler_G2": "X_phi",
            "compiler_G3": f"({target['beta']})*X_phi",
            "compiler_G4": "1/2",
            "source_normalized_G2": "0",
            "source_normalized_G3": f"-{expression}",
        },
    }
    ir = {**ir_body, "content_sha256": _sha(ir_body)}
    phi_lower, phi_upper = map(Fraction, domain["phi_interval"])
    x_lower, x_upper = map(Fraction, domain["anchor_X_interval"])
    anchor_body = {
        "schema_version": "sigma-g3-local-box-anchor-1.0",
        "status": "pass_interval_certified",
        "role": "candidate_action_bound_local_box_anchor_not_on_shell_trajectory",
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
        "momentum_parameter_m": float(Fraction(domain["slicing"]["BSSN_m"])),
        "slicing_parameter_sigma": float(Fraction(domain["slicing"]["BSSN_sigma"])),
        "domain_extension": {
            "phi_padding": 0.0,
            "normal_gradient_padding": float(Fraction(domain["normal_gradient_padding"])),
            "spatial_gradient_abs": float(Fraction(domain["spatial_gradient_component_abs"])),
            "hessian_component_abs": float(Fraction(domain["symmetric_hessian_component_abs"])),
            "riemann_component_abs": float(Fraction(domain["riemann_tetrad_component_abs"])),
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


def _binary64(value: float) -> dict[str, Any]:
    return {"decimal": value, "exact_binary64_hex": float(value).hex()}


def _interval_certificate(
    target: dict[str, Any], domain: dict[str, Any], run: dict[str, Any]
) -> dict[str, Any]:
    ir, anchor, interval_config = _adapter_inputs(target, domain, run)
    certificate = certify_cubic_bssn_domain(ir, anchor, interval_config)
    if certificate.get("status") != "pass_uniform_local_jet_box":
        raise ValueError(f"candidate interval certificate rejected: {certificate.get('errors')}")
    proof = certificate["uniform_proof"]
    margins = {
        "common_time_covector_margin": _binary64(-proof["common_time_covector_upper_P00"]),
        "spatial_block_eigenvalue_lower": _binary64(proof["spatial_block_eigenvalue_lower"]),
        "characteristic_discriminant_lower": _binary64(proof["characteristic_discriminant_lower"]),
        "slicing_cone_separation": _binary64(-proof["slicing_cone_polynomial_upper"]),
    }
    body = {
        "candidate_id": target["candidate_id"],
        "action_sha256": target["action_sha256"],
        "beta": target["beta"],
        "candidate_adapter_ir": ir,
        "candidate_anchor": anchor,
        "candidate_interval_config": interval_config,
        "certify_cubic_bssn_domain_invoked": True,
        "direct_candidate_interval_run": True,
        "prior_interval_certificate_reused": False,
        "all_direction_method": proof["direction_sphere_method"],
        "adapter_certificate": certificate,
        "certified_margins": margins,
        "status": "pass_uniform_principal_and_common_cone",
        "scope": (
            "uniform pointwise principal/common-cone result on this exact action-bound box; "
            "not an evolution-invariant, asymptotically-flat, or global-energy result"
        ),
    }
    return {**body, "content_sha256": _sha(body)}


def _validate_operator_domain(domain: dict[str, Any]) -> None:
    required = {
        "contract_kind": (
            "candidate_action_bound_global_periodic_lapse_operator_domain_not_"
            "asymptotically_flat_energy_domain"
        ),
        "spatial_manifold": "connected_three_torus_with_periodic_coordinate_boundaries",
        "regularity": {
            "sobolev_index": "s>5/2",
            "canonical_fields": "H^s_periodic",
            "canonical_momenta": "H^{s-1}_periodic",
            "lapse_multiplier_domain": "L2_periodic",
            "lapse_multiplier_codomain": "L2_periodic",
        },
        "pointwise_cell": "the bound action-specific componentwise domain holds almost everywhere",
        "temporal_endpoint_variations": "zero",
        "spatial_boundary_terms": "zero_by_periodicity",
        "unitary_gauge": "phi=t on the future-timelike-gradient branch",
        "nonempty_witness": "constant center fields with N=1, K_ij=0, and vanishing Hessian",
    }
    if domain != required:
        raise ValueError("future G3 action-bound periodic operator domain changed")


def _periodic_dirac(
    target: dict[str, Any], domain_record: dict[str, Any], operator_domain: dict[str, Any]
) -> dict[str, Any]:
    beta = Fraction(target["beta"])
    derivation = _derive_full_delta(beta)
    coercivity = _coercivity_certificate(domain_record, beta, operator_domain)
    if (
        derivation["full_Delta_N"]["beta_specialization"] != str(beta)
        or set(derivation["exact_residuals"].values()) != {"0"}
        or coercivity["function_space_result"]["status"] != "pass"
        or coercivity["function_space_result"]["kernel"] != "{0}"
    ):
        raise ValueError("future G3 action-bound lapse/Dirac proof failed")
    body = {
        "candidate_id": target["candidate_id"],
        "action_sha256": target["action_sha256"],
        "beta": str(beta),
        "operator_domain": operator_domain,
        "full_Delta_N_derivation": derivation,
        "direct_candidate_coercivity_recompute": True,
        "prior_coercivity_certificate_reused": False,
        "coercivity_certificate": coercivity,
        "exact_Delta_N_lower_bound": coercivity["Delta_N_lower_bound"]["exact"],
        "exact_Delta_N_upper_bound": coercivity["Delta_N_lower_bound"][
            "upper_at_N_50_over_51_K_3_over_50"
        ],
        "exact_inverse_norm_upper": coercivity["function_space_result"]["inverse_norm_upper"],
        "periodic_distributed_Dirac": "pass",
        "asymptotically_flat_Dirac": "blocked",
        "global_energy": "blocked",
        "status": "pass_periodic_dirac_only",
        "scope": (
            "strict coercivity and zero kernel on the registered periodic L2 multiplication-"
            "operator domain only; no asymptotically-flat or global-energy inference"
        ),
    }
    return {**body, "content_sha256": _sha(body)}


def _validate_config(config: dict[str, Any]) -> None:
    if (
        config.get("schema_version") != CONFIG_SCHEMA
        or config.get("data_eligibility") != ELIGIBILITY
        or len(config.get("targets", [])) != 3
        or not config.get("output_path", "").startswith("runs/engine/")
    ):
        raise ValueError("future G3 action-bound config is invalid")
    expected_bindings = {
        "predecessor",
        "preflight",
        "predecessor_config",
        "predecessor_source",
        "componentwise_method_control",
        "lapse_dirac_method_control",
        "effective_metric_source",
        "lapse_source",
    }
    if set(config.get("bindings", {})) != expected_bindings:
        raise ValueError("future G3 action-bound predecessor bindings changed")


def build_future_g3_action_bound_jet_box_campaign(
    config: dict[str, Any], project_root: str | Path
) -> dict[str, Any]:
    """Directly certify three action-bound boxes while keeping AF/energy blocked."""

    _validate_config(config)
    root = Path(project_root).resolve()
    source_path = (root / config["adapter_source"]["path"]).resolve()
    if _file_sha(source_path) != config["adapter_source"]["file_sha256"]:
        raise ValueError("future G3 action-bound campaign source hash mismatch")
    json_bindings = {
        "predecessor",
        "preflight",
        "predecessor_config",
        "componentwise_method_control",
        "lapse_dirac_method_control",
    }
    loaded = {name: _load_bound(root, config["bindings"][name], name) for name in json_bindings}
    for name in set(config["bindings"]) - json_bindings:
        binding = config["bindings"][name]
        path = (root / binding["path"]).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError(f"{name} path escapes repository") from error
        if not path.is_file() or _file_sha(path) != binding["file_sha256"]:
            raise ValueError(f"{name} file hash mismatch")
    predecessor = loaded["predecessor"]
    if (
        predecessor.get("candidate_count") != 3
        or predecessor.get("decision_counts") != {"blocked": 3}
        or predecessor.get("nonzero_componentwise_box_pass_count") != 0
        or predecessor.get("uniform_principal_common_cone_pass_count") != 0
        or predecessor.get("uniform_Delta_N_coercivity_pass_count") != 0
        or predecessor.get("periodic_distributed_Dirac_pass_count") != 0
    ):
        raise ValueError("future G3 action-bound predecessor state changed")
    method = loaded["componentwise_method_control"]
    lapse_method = loaded["lapse_dirac_method_control"]
    if (
        method.get("uniform_principal_common_cone_pass_count") != 1
        or lapse_method.get("full_lapse_dirac_pass_count") != 1
    ):
        raise ValueError("future G3 method-control boundary changed")
    _validate_operator_domain(config["operator_domain"])
    records = []
    for target in config["targets"]:
        matches = [
            item
            for item in predecessor["candidate_records"]
            if item["candidate_id"] == target["candidate_id"]
        ]
        if len(matches) != 1:
            raise ValueError("future G3 action-bound target is not unique")
        prior = matches[0]
        expected = {
            "action_sha256": target["action_sha256"],
            "beta": target["beta"],
            "decision": "blocked",
            "first_blocker": "candidate_bound_nonzero_componentwise_normalized_local_jet_box_values",
            "content_sha256": target["predecessor_record_content_sha256"],
        }
        if any(prior.get(key) != value for key, value in expected.items()):
            raise ValueError("future G3 action-bound target binding changed")
        domain = target["componentwise_domain"]
        _validate_domain(domain)
        registered = _registered_values(domain)
        if set(registered) != set(DOMAIN_FIELDS) or any(
            value is None for value in registered.values()
        ):
            raise ValueError("future G3 action-bound registration did not fill all 12 fields")
        domain_body = {
            "domain_id": target["domain_id"],
            "candidate_id": target["candidate_id"],
            "action_sha256": target["action_sha256"],
            "beta": target["beta"],
            "contract_review": "explicit_new_mathematical_domain_registration_not_family_transfer",
            "registered_values": registered,
            "filled_field_count": 12,
            "unfilled_field_count": 0,
            "componentwise_domain": domain,
            "nonempty_rational_box_registered": True,
            "status": "pass_candidate_action_bound_domain_registered",
            "scope": (
                "reviewed sufficient local box for this action and adapter invocation; it is "
                "not asserted to follow from the qualitative beta-sized family label"
            ),
        }
        domain_certificate = {**domain_body, "content_sha256": _sha(domain_body)}
        interval = _interval_certificate(target, domain, config["interval_run"])
        periodic = _periodic_dirac(
            target,
            {"componentwise_domain": domain},
            config["operator_domain"],
        )
        gates = {
            "predecessor_action_binding": "pass",
            "twelve_componentwise_domain_fields": "pass_12_filled_0_missing",
            "nonzero_rational_local_jet_box": "pass",
            "uniform_local_jet_box_principal_common_cone": "pass",
            "exact_full_Delta_N_derivation": "pass",
            "uniform_Delta_N_coercivity": "pass",
            "periodic_distributed_Dirac": "pass",
            "asymptotically_flat_Dirac": "blocked",
            "global_energy": "blocked",
            "smaller_beta_or_family_label_transfer": "rejected_as_inference",
            "observational_data_seal": "pass",
        }
        provenance_body = {
            "predecessor_content_sha256": predecessor["content_sha256"],
            "predecessor_record_content_sha256": prior["content_sha256"],
            "candidate_id": target["candidate_id"],
            "action_sha256": target["action_sha256"],
            "domain_certificate_sha256": domain_certificate["content_sha256"],
            "interval_certificate_sha256": interval["content_sha256"],
            "periodic_dirac_sha256": periodic["content_sha256"],
            "data_eligibility": dict(ELIGIBILITY),
        }
        record_body = {
            "candidate_id": target["candidate_id"],
            "action_sha256": target["action_sha256"],
            "beta": target["beta"],
            "decision": "blocked",
            "first_blocker": FIRST_BLOCKER,
            "domain_registration": domain_certificate,
            "uniform_interval_certificate": interval,
            "lapse_and_periodic_dirac": periodic,
            "gate_ledger": gates,
            "necessary_condition_rejection_found": False,
            "full_formal_pass": False,
            "observational_data_opened": False,
            "data_eligibility": dict(ELIGIBILITY),
            "provenance": {**provenance_body, "binding_sha256": _sha(provenance_body)},
        }
        records.append({**record_body, "content_sha256": _sha(record_body)})
    body = {
        "schema_version": ARTIFACT_SCHEMA,
        "campaign_id": config["campaign_id"],
        "campaign_adapter_source": config["adapter_source"],
        "config_content_sha256": _sha(config),
        "source_bindings": config["bindings"],
        "candidate_count": 3,
        "decision_counts": {"blocked": 3},
        "candidate_records": records,
        "domain_registration_filled_field_count": 36,
        "domain_registration_missing_field_count": 0,
        "nonzero_componentwise_box_pass_count": 3,
        "uniform_principal_common_cone_pass_count": 3,
        "full_Delta_N_derivation_pass_count": 3,
        "uniform_Delta_N_coercivity_pass_count": 3,
        "periodic_distributed_Dirac_pass_count": 3,
        "asymptotically_flat_Dirac_pass_count": 0,
        "global_energy_pass_count": 0,
        "full_formal_pass_count": 0,
        "first_blocker_counts": {FIRST_BLOCKER: 3},
        "observational_data_opened": False,
        "dark_matter_or_halo_inputs": False,
        "redshift_distance_inputs": False,
        "paid_llm_spend_usd": 0.0,
        "synthetic_fixture_role": "none_used",
        "data_eligibility": dict(ELIGIBILITY),
        "interpretation": (
            "All 36 previously missing action-bound registration fields are filled. Direct "
            "candidate interval runs and exact beta-specific lapse/coercivity calculations pass "
            "all three boxes and their periodic Dirac domains. AF Dirac, global energy, and full "
            "formal promotion remain blocked."
        ),
    }
    return {**body, "content_sha256": _sha(body)}


def write_future_g3_action_bound_jet_box_campaign(
    config_path: str | Path, project_root: str | Path
) -> Path:
    root = Path(project_root).resolve()
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    artifact = build_future_g3_action_bound_jet_box_campaign(config, root)
    output = root / config["output_path"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_canonical(artifact) + "\n", encoding="utf-8")
    return output
