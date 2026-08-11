"""Constraint/boundary embedding audit for future Aether twist witnesses."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import os
from collections import Counter
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Any

import sympy as sp

from .aether_parameter_cell_formal_gate_campaign import _specialize
from .future_aether_candidate_formal_followup import (
    build_future_aether_candidate_formal_followup,
)
from .promotion_orchestrator import ELIGIBILITY

CONFIG_SCHEMA = "sigma-future-aether-constraint-boundary-embedding-audit-config-1.0"
RESULT_SCHEMA = "sigma-future-aether-constraint-boundary-embedding-audit-1.0"
TARGET_FAMILY = "AETHER_K1234_PARAMETER_CELL"
SOURCE_BLOCKER = "full_constraint_embedding_of_negative_static_twist_jet"
BLOCKER = "constraint_satisfying_asymptotically_Euclidean_completion_of_negative_twist_witness"
EXPECTED_CONTROLS = {
    "einstein_aether_generic_3plus1_legendre": (
        "sigma_theory_compiler.adm_aether:einstein_aether_3plus1_decomposition_control",
        "generic_exact_action_local_Legendre_and_lapse_divergence_structure",
    ),
    "einstein_aether_generic_lapse_shift_constraint_seeds": (
        "sigma_theory_compiler.adm_aether:einstein_aether_lapse_shift_constraint_seed_control",
        "generic_exact_action_Hamiltonian_and_momentum_constraint_seed_structure",
    ),
    "einstein_aether_spatial_diffeomorphism_algebra": (
        "sigma_theory_compiler.adm_aether:einstein_aether_spatial_diffeomorphism_control",
        "canonical_cotangent_lift_momentum_constraint_structure",
    ),
    "einstein_aether_restricted_nonlinear_total_energy": (
        "sigma_theory_compiler.adm_aether:einstein_aether_nonlinear_positive_energy_theorem_control",
        "boundary_identity_and_twisting_scope_exclusion_only",
    ),
}


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path.name} must contain an object")
    return value


def _bound_path(root: Path, binding: dict[str, Any], label: str) -> Path:
    if set(binding) - {"path", "file_sha256", "content_sha256"}:
        raise ValueError(f"{label} binding fields are invalid")
    path = (root / binding["path"]).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} path escapes repository") from error
    if not path.is_file() or _file_sha(path) != binding["file_sha256"]:
        raise ValueError(f"{label} file hash mismatch")
    return path


def _bound_json(
    root: Path, binding: dict[str, Any], label: str, *, content: bool = False
) -> dict[str, Any]:
    value = _load(_bound_path(root, binding, label))
    if content:
        body = {key: item for key, item in value.items() if key != "content_sha256"}
        if (
            value.get("content_sha256") != binding.get("content_sha256")
            or _sha(body) != binding["content_sha256"]
        ):
            raise ValueError(f"{label} content hash mismatch")
    return value


def _validate_config(config: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "campaign_id",
        "campaign_implementation",
        "source_followup_artifact",
        "source_followup_config",
        "source_followup_implementation",
        "adm_aether_source",
        "formal_report",
        "reviewed_controls",
        "budget",
        "observational_authorization",
        "data_eligibility",
        "external_paid_llm_calls",
    }
    if set(config) != required or config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("future Aether embedding-audit config is invalid")
    if config.get("data_eligibility") != ELIGIBILITY:
        raise ValueError("future Aether embedding-audit eligibility is open")
    if config.get("observational_authorization") is not False:
        raise ValueError("future Aether embedding-audit opened observations")
    if config.get("external_paid_llm_calls") is not False:
        raise ValueError("future Aether embedding-audit enabled paid LLM calls")
    budget = config["budget"]
    if set(budget) != {
        "maximum_candidates",
        "maximum_reviewed_control_replays",
        "maximum_paid_llm_spend_usd",
    } or budget != {
        "maximum_candidates": 14,
        "maximum_reviewed_control_replays": 4,
        "maximum_paid_llm_spend_usd": 0.0,
    }:
        raise ValueError("future Aether embedding-audit budget is not exact")
    descriptors = config["reviewed_controls"]
    registry = {
        item.get("formal_check_id"): (item.get("entrypoint"), item.get("applicability"))
        for item in descriptors
    }
    if len(descriptors) != 4 or registry != EXPECTED_CONTROLS:
        raise ValueError("future Aether embedding-audit control registry is incomplete")


def _replay_controls(
    config: dict[str, Any], root: Path, report: dict[str, Any]
) -> tuple[dict[str, Any], str]:
    source = _bound_path(root, config["adm_aether_source"], "ADM Aether source")
    checks = {item.get("name"): item for item in report.get("checks", [])}
    replayed = {}
    for descriptor in config["reviewed_controls"]:
        check_id = descriptor["formal_check_id"]
        module_name, separator, attribute = descriptor["entrypoint"].partition(":")
        callback = getattr(importlib.import_module(module_name), attribute, None)
        callback_source = (
            inspect.getsourcefile(inspect.unwrap(callback)) if callable(callback) else None
        )
        if not separator or not callable(callback) or callback_source is None:
            raise ValueError("reviewed Aether embedding control is not callable")
        if Path(callback_source).resolve() != source:
            raise ValueError("reviewed Aether embedding control source changed")
        evidence = callback()
        check = checks.get(check_id)
        if (
            not isinstance(check, dict)
            or check.get("status") != "pass"
            or evidence.get("passed") is not True
            or evidence != check.get("evidence")
            or _sha(evidence) != descriptor["evidence_sha256"]
        ):
            raise ValueError(f"reviewed Aether embedding control changed: {check_id}")
        replayed[check_id] = {
            "entrypoint": descriptor["entrypoint"],
            "applicability": descriptor["applicability"],
            "evidence_sha256": descriptor["evidence_sha256"],
            "scope": evidence["scope"],
            "status": "pass",
        }
    root_sha = _sha(
        [
            [key, value["evidence_sha256"], value["applicability"]]
            for key, value in sorted(replayed.items())
        ]
    )
    return replayed, root_sha


@lru_cache(maxsize=1)
def _affine_symbolic_control() -> dict[str, str]:
    """Derive the flat static affine-witness constraint residuals from K1..K4."""

    x0, x1, x2 = sp.symbols("x0 x1 x2", real=True)
    y = sp.symbols("y", positive=True)
    c1, c2, c3, c4, m2 = sp.symbols("c1 c2 c3 c4 M2", real=True)
    coordinates = (x0, x1, x2)
    twist = sp.Matrix([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])
    aether = sp.Matrix([sp.sqrt(y) - x1, x0, 0])
    chi = sp.sqrt(1 + (aether.T * aether)[0])
    d_chi = sp.Matrix([(aether.T * twist.row(i).T)[0] / chi for i in range(3)])
    k00, k11, k22, k01, k02, k12 = sp.symbols("K00 K11 K22 K01 K02 K12")
    extrinsic = sp.Matrix([[k00, k01, k02], [k01, k11, k12], [k02, k12, k22]])
    velocity = sp.Matrix(sp.symbols("V0:3"))
    spatial_block = twist + chi * extrinsic
    electric = velocity - extrinsic * aether
    spatial_normal = d_chi + extrinsic * aether
    normal_normal = ((aether.T * velocity)[0] - (aether.T * extrinsic * aether)[0]) / chi
    invariant_1 = sp.expand(
        normal_normal**2
        - (electric.T * electric)[0]
        - (spatial_normal.T * spatial_normal)[0]
        + sp.trace(spatial_block.T * spatial_block)
    )
    invariant_2 = sp.expand((normal_normal + sp.trace(spatial_block)) ** 2)
    invariant_3 = sp.expand(
        normal_normal**2 + 2 * (electric.T * spatial_normal)[0] + sp.trace(spatial_block**2)
    )
    acceleration_normal = -chi * normal_normal - (aether.T * spatial_normal)[0]
    acceleration_spatial = chi * electric + spatial_block.T * aether
    invariant_4 = sp.expand(
        -(acceleration_normal**2) + (acceleration_spatial.T * acceleration_spatial)[0]
    )
    lagrangian = sp.expand(
        m2 * (sp.trace(extrinsic.T * extrinsic) - sp.trace(extrinsic) ** 2) / 2
        - (c1 * invariant_1 + c2 * invariant_2 + c3 * invariant_3 - c4 * invariant_4) / 2
    )
    zero = {k00: 0, k11: 0, k22: 0, k01: 0, k02: 0, k12: 0, **{item: 0 for item in velocity}}
    p_w = sp.Matrix([sp.simplify(sp.diff(lagrangian, velocity[i]).subs(zero)) for i in range(3)])
    k_variables = (k00, k11, k22, k01, k02, k12)
    derivatives = [sp.simplify(sp.diff(lagrangian, item).subs(zero)) for item in k_variables]
    # K_ij=(dot(q)_ij-2D_(i N_j))/(2N).  For the six independent q_ij
    # coordinates, pi^ij=p_ij/2 off diagonal, as fixed by the reviewed metric
    # cotangent-lift convention.  Hence diagonals carry 1/2 and off diagonals 1/4.
    metric_momentum = sp.Matrix(
        [
            [derivatives[0] / 2, derivatives[3] / 4, derivatives[4] / 4],
            [derivatives[3] / 4, derivatives[1] / 2, derivatives[5] / 4],
            [derivatives[4] / 4, derivatives[5] / 4, derivatives[2] / 2],
        ]
    )
    h_core = sp.simplify(-lagrangian.subs(zero))
    h_divergence = sp.simplify(sum(sp.diff(chi * p_w[i], coordinates[i]) for i in range(3)))
    aether_momentum = sp.Matrix(
        [
            sp.simplify(
                sum(p_w[j] * twist[i, j] for j in range(3))
                - sum(sp.diff(p_w[j] * aether[i], coordinates[j]) for j in range(3))
            )
            for i in range(3)
        ]
    )
    gravitational_momentum = sp.Matrix(
        [
            sp.simplify(-2 * sum(sp.diff(metric_momentum[j, i], coordinates[j]) for j in range(3)))
            for i in range(3)
        ]
    )
    origin = {x0: 0, x1: 0, x2: 0}
    h_core_at_origin = sp.factor(h_core.subs(origin))
    h_divergence_at_origin = sp.factor(h_divergence.subs(origin))
    momentum_at_origin = (aether_momentum + gravitational_momentum).applyfunc(
        lambda item: sp.factor(item.subs(origin))
    )
    expected_h_core = sp.factor(c1 - c3 - c1 * y / (2 * (1 + y)) - c4 * y / 2)
    expected_h_divergence = -2 * (c3 + c4 * (1 + 2 * y))
    momentum_numerator = (
        -3 * c1 * y - 4 * c1 + 3 * c3 * y + 4 * c3 + 10 * c4 * y**2 + 18 * c4 * y + 8 * c4
    )
    expected_momentum = sp.Matrix(
        [sp.sqrt(y) * momentum_numerator / (2 * (1 + y) ** sp.Rational(3, 2)), 0, 0]
    )
    if (
        sp.factor(h_core_at_origin - expected_h_core) != 0
        or sp.factor(h_divergence_at_origin - expected_h_divergence) != 0
        or any(sp.factor(item) != 0 for item in momentum_at_origin - expected_momentum)
    ):
        raise ValueError("affine Aether constraint derivation changed")
    body = {
        "ansatz": (
            "q_ij=delta_ij, K_ij=0, N=1, partial_t A_i=0; "
            "A=(sqrt(y)-x^2,x^1,0), partial_i A_j=W_ij, W_12=1=-W_21"
        ),
        "normalization": "W2=2, WA2=y, chi=sqrt(1+A_i A_i)",
        "canonical_aether_momentum_at_origin": ("p_W=(0,sqrt(y)*(c3+c4*(1+y))/sqrt(1+y),0)"),
        "canonical_metric_momentum_at_origin": (
            "pi_12=pi_21=-y*(c1+c3)/(4*sqrt(1+y)); all other pi_ij=0"
        ),
        "hamiltonian_constraint_seed": "H=H_core+D_i(chi*p_W^i)-M2*R/2+metric_kinetic_terms",
        "flat_static_core_at_origin": str(expected_h_core),
        "lapse_divergence_at_origin": str(expected_h_divergence),
        "full_flat_static_H_residual_at_origin": str(
            sp.factor(expected_h_core + expected_h_divergence)
        ),
        "momentum_constraint_seed": (
            "D_i=-2*partial_j(pi^j_i)+p_W^j*partial_i(A_j)-partial_j(p_W^j*A_i)"
        ),
        "full_flat_static_D_residual_at_origin": str(expected_momentum),
        "momentum_residual_numerator": str(momentum_numerator),
        "derivation_scope": (
            "exact normalized affine two-jet subfamily on a flat static slice; this ansatz is "
            "not asymptotically Euclidean and is not a generic constraint solution"
        ),
    }
    return {**body, "content_sha256": _sha(body)}


def _candidate_affine_certificate(
    parameters: dict[str, str], witness: dict[str, Any]
) -> dict[str, Any]:
    c1, c3, c4 = (Fraction(parameters[name]) for name in ("c1", "c3", "c4"))
    y = Fraction(witness["tilt_squared_y"])
    if y <= 0 or witness.get("normalized_W2") != "2" or Fraction(witness["normalized_WA2"]) != y:
        raise ValueError("future Aether normalized twist witness changed")
    h_core = c1 - c3 - c1 * y / (2 * (1 + y)) - c4 * y / 2
    h_divergence = -2 * (c3 + c4 * (1 + 2 * y))
    h_residual = h_core + h_divergence
    momentum_numerator = (
        -3 * c1 * y - 4 * c1 + 3 * c3 * y + 4 * c3 + 10 * c4 * y**2 + 18 * c4 * y + 8 * c4
    )
    momentum_norm_squared = y * momentum_numerator**2 / (4 * (1 + y) ** 3)
    if h_core != Fraction(witness["C_y"]):
        raise ValueError("future Aether local twist Hamiltonian does not rederive")
    if h_residual == 0 or momentum_norm_squared == 0:
        raise ValueError("affine witness unexpectedly reached the constraint surface")
    body = {
        "tilt_squared_y": str(y),
        "normalized_local_twist_hamiltonian_H_core": str(h_core),
        "lapse_divergence_D_i_chi_pW_i": str(h_divergence),
        "flat_static_Hamiltonian_constraint_residual": str(h_residual),
        "flat_static_Hamiltonian_constraint_satisfied": False,
        "flat_static_momentum_constraint_residual": (
            f"(sqrt({y})*({momentum_numerator})/(2*(1+{y})^(3/2)),0,0)"
        ),
        "flat_static_momentum_constraint_residual_norm_squared": str(momentum_norm_squared),
        "flat_static_momentum_constraint_satisfied": False,
        "explicit_affine_ansatz_constraint_datum_status": "reject",
        "asymptotically_Euclidean": False,
        "asymptotic_failure": (
            "A_i grows linearly in the transverse radius and u^a does not approach the "
            "asymptotic unit normal"
        ),
        "completed_Aether_boundary_energy": "undefined_outside_AE_phase_space",
        "boundary_formula_if_AE_premises_were_met": (
            "M_ae=M_ADM-c14/(8*pi*G)*integral_infinity(r^a a_a)"
        ),
        "constraint_satisfying_negative_total_energy_datum_proven": False,
        "candidate_rejection_authorized": False,
    }
    return {**body, "content_sha256": _sha(body)}


def build_future_aether_constraint_boundary_embedding_audit(
    config: dict[str, Any], root: str | Path
) -> dict[str, Any]:
    root = Path(root).resolve()
    _validate_config(config)
    _bound_path(root, config["campaign_implementation"], "embedding-audit implementation")
    source_implementation = _bound_path(
        root, config["source_followup_implementation"], "source follow-up implementation"
    )
    source_config = _bound_json(root, config["source_followup_config"], "source follow-up config")
    source_artifact = _bound_json(
        root, config["source_followup_artifact"], "source follow-up artifact", content=True
    )
    if (
        Path(inspect.getsourcefile(build_future_aether_candidate_formal_followup) or "").resolve()
        != source_implementation
    ):
        raise ValueError("source follow-up implementation entrypoint changed")
    if build_future_aether_candidate_formal_followup(source_config, root) != source_artifact:
        raise ValueError("source follow-up artifact no longer replays")
    if (
        source_artifact.get("candidate_count") != 14
        or source_artifact.get("decision_counts") != {"blocked": 14}
        or source_artifact.get("first_blocker_counts") != {SOURCE_BLOCKER: 14}
        or source_artifact.get("candidate_rejection_authorized_count") != 0
        or source_artifact.get("full_candidate_specific_formal_completion_claimed") is not False
    ):
        raise ValueError("source follow-up decision scope changed")
    report = _bound_json(root, config["formal_report"], "formal report")
    reviewed, reviewed_root = _replay_controls(config, root, report)
    symbolic = _affine_symbolic_control()
    records = []
    h_residuals: Counter[str] = Counter()
    momentum_residuals: Counter[str] = Counter()
    for source in source_artifact["candidate_records"]:
        source_body = {key: value for key, value in source.items() if key != "content_sha256"}
        specialization = _specialize(source["parameters"])
        witness = specialization.get("finite_negative_twist_witness")
        if (
            source.get("content_sha256") != _sha(source_body)
            or source.get("family_id") != TARGET_FAMILY
            or source.get("decision") != "blocked"
            or source.get("first_blocker") != SOURCE_BLOCKER
            or source.get("candidate_rejection_authorized") is not False
            or source.get("exact_specialization") != specialization
            or not isinstance(witness, dict)
            or witness.get("local_hamiltonian_density_negative") is not True
        ):
            raise ValueError("source Aether follow-up record changed")
        certificate = _candidate_affine_certificate(source["parameters"], witness)
        h_residuals[certificate["flat_static_Hamiltonian_constraint_residual"]] += 1
        momentum_residuals[
            certificate["flat_static_momentum_constraint_residual_norm_squared"]
        ] += 1
        gates = {
            "source_candidate_action_and_negative_twist_witness_binding": {"status": "pass"},
            "generic_constraint_and_boundary_control_replay": {
                "status": "pass",
                "scope": "formula provenance only; no candidate pass inference",
                "reviewed_evidence_root_sha256": reviewed_root,
            },
            "normalized_affine_two_jet_completion": {
                "status": "pass",
                "scope": symbolic["derivation_scope"],
            },
            "flat_static_Hamiltonian_constraint": {
                "status": "reject_explicit_ansatz",
                "residual": certificate["flat_static_Hamiltonian_constraint_residual"],
            },
            "flat_static_momentum_constraint": {
                "status": "reject_explicit_ansatz",
                "residual_norm_squared": certificate[
                    "flat_static_momentum_constraint_residual_norm_squared"
                ],
            },
            "asymptotically_Euclidean_boundary_completion": {
                "status": "blocked",
                "boundary_contribution": "undefined_outside_AE_phase_space",
            },
            "constraint_satisfying_negative_completed_boundary_energy": {
                "status": "blocked",
                "reason": (
                    "the explicit affine completion violates both secondary constraints and "
                    "the local first jet does not supply an AE coupled elliptic completion"
                ),
            },
            "observational_data_seal": {"status": "pass"},
        }
        provenance_body = {
            "candidate_id": source["candidate_id"],
            "typed_action_ir_sha256": source["typed_action_ir_sha256"],
            "source_followup_record_sha256": source["content_sha256"],
            "source_exact_specialization_sha256": specialization["content_sha256"],
            "affine_constraint_certificate_sha256": certificate["content_sha256"],
            "reviewed_control_evidence_root_sha256": reviewed_root,
            "data_eligibility": ELIGIBILITY,
        }
        body = {
            "ordinal": source["ordinal"],
            "candidate_id": source["candidate_id"],
            "family_id": TARGET_FAMILY,
            "parameter_cell_id": source["parameter_cell_id"],
            "parameter_cell_lineage_sha256": source["parameter_cell_lineage_sha256"],
            "typed_action_ir_sha256": source["typed_action_ir_sha256"],
            "action_density_equivalence_sha256": source["action_density_equivalence_sha256"],
            "compilation_receipt_sha256": source["compilation_receipt_sha256"],
            "source_followup_record_sha256": source["content_sha256"],
            "parameters": source["parameters"],
            "exact_specialization": specialization,
            "affine_constraint_boundary_certificate": certificate,
            "gate_ledger": gates,
            "decision": "blocked",
            "first_blocker": BLOCKER,
            "formal_pass": False,
            "candidate_rejection_authorized": False,
            "constraint_satisfying_negative_total_energy_datum_proven": False,
            "full_formal_completion_claimed": False,
            "automatic_downstream_enqueue_performed": False,
            "solar_bundle_generated": False,
            "observational_data_opened": False,
            "data_eligibility": dict(ELIGIBILITY),
            "provenance": {**provenance_body, "binding_sha256": _sha(provenance_body)},
        }
        records.append({**body, "content_sha256": _sha(body)})
    if len(records) != 14:
        raise ValueError("future Aether embedding-audit target count changed")
    record_root = _sha(
        [
            [item["candidate_id"], item["typed_action_ir_sha256"], item["content_sha256"]]
            for item in records
        ]
    )
    provenance_body = {
        "source_followup_content_sha256": source_artifact["content_sha256"],
        "source_followup_record_registry_root_sha256": source_artifact[
            "candidate_record_registry_root_sha256"
        ],
        "source_followup_implementation_file_sha256": config["source_followup_implementation"][
            "file_sha256"
        ],
        "adm_aether_source_file_sha256": config["adm_aether_source"]["file_sha256"],
        "formal_report_file_sha256": config["formal_report"]["file_sha256"],
        "reviewed_control_evidence_root_sha256": reviewed_root,
        "affine_symbolic_control_sha256": symbolic["content_sha256"],
        "candidate_record_registry_root_sha256": record_root,
        "data_eligibility": ELIGIBILITY,
    }
    body = {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": config["campaign_id"],
        "immutable_config_sha256": _sha(config),
        "source_followup_binding": config["source_followup_artifact"],
        "candidate_count": len(records),
        "decision_counts": {"blocked": 14},
        "first_blocker_counts": {BLOCKER: 14},
        "formal_pass_count": 0,
        "candidate_rejection_authorized_count": 0,
        "constraint_satisfying_negative_total_energy_datum_count": 0,
        "exact_negative_local_twist_witness_count": 14,
        "explicit_affine_ansatz_constraint_reject_count": 14,
        "nonzero_Hamiltonian_constraint_residual_count": 14,
        "nonzero_momentum_constraint_residual_count": 14,
        "undefined_AE_boundary_contribution_count": 14,
        "Hamiltonian_constraint_residual_counts": dict(sorted(h_residuals.items())),
        "momentum_constraint_residual_norm_squared_counts": dict(
            sorted(momentum_residuals.items())
        ),
        "reviewed_control_replay_count": len(reviewed),
        "reviewed_control_evidence": reviewed,
        "reviewed_control_evidence_root_sha256": reviewed_root,
        "affine_symbolic_control": symbolic,
        "candidate_record_registry_root_sha256": record_root,
        "candidate_records": records,
        "bounded_embedding_audit_completed": True,
        "full_candidate_specific_formal_completion_claimed": False,
        "automatic_downstream_enqueue_performed": False,
        "solar_bundle_count": 0,
        "observational_data_opened": False,
        "dark_matter_or_halo_inputs": False,
        "redshift_distance_inputs": False,
        "paid_llm_spend_usd": 0.0,
        "data_eligibility": dict(ELIGIBILITY),
        "provenance": {**provenance_body, "binding_sha256": _sha(provenance_body)},
        "interpretation": (
            "Each of the fourteen negative local twist witnesses has an exact normalized flat, "
            "static affine two-jet completion. Every such completion has nonzero Hamiltonian and "
            "momentum constraint residuals and is not asymptotically Euclidean, so the explicit "
            "ansatz is rejected as initial data but no candidate theory is rejected. The first "
            "missing premise remains a candidate-bound AE solution of the coupled constraints "
            "whose completed Aether boundary energy is genuinely negative."
        ),
    }
    return {**body, "content_sha256": _sha(body)}


def publish_future_aether_constraint_boundary_embedding_audit(
    config: dict[str, Any], root: str | Path, target: str | Path
) -> dict[str, Any]:
    artifact = build_future_aether_constraint_boundary_embedding_audit(config, root)
    target = Path(target).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if _load(target) != artifact:
            raise ValueError("refusing to replace divergent future Aether embedding artifact")
        return artifact
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("xb") as handle:
        handle.write((_canonical(artifact) + "\n").encode())
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    config = _load(Path(arguments.config))
    artifact = publish_future_aether_constraint_boundary_embedding_audit(
        config, arguments.root, arguments.output
    )
    print(_canonical({key: value for key, value in artifact.items() if key != "candidate_records"}))


if __name__ == "__main__":
    main()
