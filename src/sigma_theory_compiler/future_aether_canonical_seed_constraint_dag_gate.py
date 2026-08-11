"""Canonical-seed and distributed-constraint DAG gate for future Aether."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

import sympy as sp

from .adm_aether import (
    einstein_aether_3plus1_decomposition_control,
    einstein_aether_spatial_diffeomorphism_control,
)
from .future_aether_finite_tilt_york_symbol_gate import YORK_SHELL_BLOCKER
from .future_aether_lower_order_coefficient_contract_gate import (
    BLOCKER as SOURCE_ELLIPTIC_BLOCKER,
)
from .future_aether_lower_order_coefficient_contract_gate import (
    build_future_aether_lower_order_coefficient_contract_gate,
)
from .future_aether_nonlinear_lift_characteristic_gate import CHARACTERISTIC_BLOCKER
from .promotion_orchestrator import ELIGIBILITY

CONFIG_SCHEMA = "sigma-future-aether-canonical-seed-constraint-dag-gate-config-1.0"
RESULT_SCHEMA = "sigma-future-aether-canonical-seed-constraint-dag-gate-1.0"
TARGET_FAMILY = "AETHER_K1234_PARAMETER_CELL"
BLOCKER = (
    "candidate_bound_spatially_distributed_canonical_H_core_and_"
    "metric_covariantized_H_D_Frechet_DAG_off_flat_seed_chart"
)
EXPECTED_CONTROLS = {
    "einstein_aether_generic_3plus1_legendre": (
        "sigma_theory_compiler.adm_aether:einstein_aether_3plus1_decomposition_control",
        "232ea12f7815b99e5a162f74ef4932d8bf2bada041d71fe71cc385be0152a353",
    ),
    "einstein_aether_spatial_diffeomorphism_algebra": (
        "sigma_theory_compiler.adm_aether:einstein_aether_spatial_diffeomorphism_control",
        "8156d1f721c3596e0530f9601612c2cdf9fac9d2a17e6fd8fd44881d4e216640",
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
        "source_lower_order_artifact",
        "source_lower_order_config",
        "source_lower_order_implementation",
        "source_compact_seed_artifact",
        "reviewed_adm_source",
        "formal_report",
        "reviewed_controls",
        "canonical_seed_ansatz",
        "budget",
        "observational_authorization",
        "data_eligibility",
        "external_paid_llm_calls",
    }
    if set(config) != required or config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("future Aether canonical-seed config is invalid")
    if config.get("data_eligibility") != ELIGIBILITY:
        raise ValueError("future Aether canonical-seed eligibility is open")
    if config.get("observational_authorization") is not False:
        raise ValueError("future Aether canonical-seed gate opened observations")
    if config.get("external_paid_llm_calls") is not False:
        raise ValueError("future Aether canonical-seed gate enabled paid LLM calls")
    if config.get("canonical_seed_ansatz") != {
        "q_ij": "delta_ij",
        "lapse": "1",
        "shift": "0",
        "K_ij": "0",
        "A_i": "F*delta_i1",
        "F": "10*(1-r^2)^4_+",
        "L_n_A_i": "0",
        "chi": "sqrt(1+F^2)",
    }:
        raise ValueError("future Aether canonical seed ansatz is not exact")
    if config.get("budget") != {
        "maximum_candidates": 14,
        "maximum_canonical_seed_candidates": 1,
        "maximum_reviewed_control_replays": 2,
        "maximum_paid_llm_spend_usd": 0.0,
    }:
        raise ValueError("future Aether canonical-seed budget is not exact")
    descriptors = config.get("reviewed_controls")
    if not isinstance(descriptors, list) or len(descriptors) != 2:
        raise ValueError("future Aether canonical-seed control registry is incomplete")
    registry = {
        item.get("formal_check_id"): (item.get("entrypoint"), item.get("evidence_sha256"))
        for item in descriptors
    }
    if registry != EXPECTED_CONTROLS:
        raise ValueError("future Aether canonical-seed reviewed controls changed")


def _reviewed_controls(
    config: dict[str, Any], root: Path, report: dict[str, Any]
) -> tuple[dict[str, Any], str]:
    source = _bound_path(root, config["reviewed_adm_source"], "reviewed ADM source")
    callbacks = {
        "einstein_aether_generic_3plus1_legendre": (
            einstein_aether_3plus1_decomposition_control
        ),
        "einstein_aether_spatial_diffeomorphism_algebra": (
            einstein_aether_spatial_diffeomorphism_control
        ),
    }
    report_checks = {item.get("name"): item for item in report.get("checks", [])}
    replayed: dict[str, Any] = {}
    for descriptor in config["reviewed_controls"]:
        check_id = descriptor["formal_check_id"]
        callback = callbacks[check_id]
        callback_source = Path(inspect.getsourcefile(inspect.unwrap(callback)) or "").resolve()
        evidence = callback()
        report_check = report_checks.get(check_id)
        if (
            callback_source != source
            or not isinstance(report_check, dict)
            or report_check.get("status") != "pass"
            or report_check.get("evidence") != evidence
            or evidence.get("passed") is not True
            or _sha(evidence) != descriptor["evidence_sha256"]
        ):
            raise ValueError(f"reviewed Aether canonical control changed: {check_id}")
        replayed[check_id] = {
            "entrypoint": descriptor["entrypoint"],
            "evidence_sha256": descriptor["evidence_sha256"],
            "scope": evidence["scope"],
            "status": "pass",
        }
    root_sha = _sha(
        [[key, item["evidence_sha256"]] for key, item in sorted(replayed.items())]
    )
    return replayed, root_sha


@lru_cache(maxsize=1)
def exact_static_canonical_seed_control() -> dict[str, Any]:
    """Derive pi and p_A for the exact static compact profile on the flat chart."""

    f, f1, f2, f3 = sp.symbols("F F_1 F_2 F_3", real=True)
    chi = sp.sqrt(1 + f**2)
    aether = sp.Matrix([f, 0, 0])
    derivative = sp.Matrix([[f1, 0, 0], [f2, 0, 0], [f3, 0, 0]])
    k11, k22, k33, k12, k13, k23 = sp.symbols("K11 K22 K33 K12 K13 K23")
    extrinsic = sp.Matrix(
        [[k11, k12, k13], [k12, k22, k23], [k13, k23, k33]]
    )
    electric = sp.Matrix(sp.symbols("W1:4", real=True))
    spatial_block = derivative + chi * extrinsic
    spatial_normal = sp.Matrix([f * f1 / chi, f * f2 / chi, f * f3 / chi]) + (
        extrinsic * aether
    )
    normal_normal = (aether.T * electric)[0] / chi
    invariant_1 = sp.expand(
        normal_normal**2
        - (electric.T * electric)[0]
        - (spatial_normal.T * spatial_normal)[0]
        + sp.trace(spatial_block.T * spatial_block)
    )
    acceleration_normal = sp.expand(
        -chi * normal_normal - (aether.T * spatial_normal)[0]
    )
    acceleration_spatial = sp.expand(chi * electric + spatial_block.T * aether)
    invariant_4 = sp.expand(
        -acceleration_normal**2 + (acceleration_spatial.T * acceleration_spatial)[0]
    )
    trace_k = sp.trace(extrinsic)
    k_squared = sp.trace(extrinsic.T * extrinsic)
    lagrangian = sp.expand(
        (k_squared - trace_k**2) / 2
        - (sp.Rational(1, 32) * invariant_1 - sp.Rational(1, 32) * invariant_4) / 2
    )
    k_variables = (k11, k22, k33, k12, k13, k23)
    zero = {item: 0 for item in (*k_variables, *electric)}
    aether_momentum = sp.Matrix(
        [sp.factor(sp.diff(lagrangian, item).subs(zero)) for item in electric]
    )
    raw_metric = [sp.factor(sp.diff(lagrangian, item).subs(zero)) for item in k_variables]
    metric_momentum = sp.Matrix(
        [
            [raw_metric[0] / 2, raw_metric[3] / 4, raw_metric[4] / 4],
            [raw_metric[3] / 4, raw_metric[1] / 2, raw_metric[5] / 4],
            [raw_metric[4] / 4, raw_metric[5] / 4, raw_metric[2] / 2],
        ]
    ).applyfunc(sp.factor)
    expected_aether = sp.Matrix([f * f1 / (32 * chi), 0, 0])
    expected_metric = sp.Matrix(
        [
            [(f**2 - 1) * f1 / (64 * chi), -f2 / (128 * chi), -f3 / (128 * chi)],
            [-f2 / (128 * chi), 0, 0],
            [-f3 / (128 * chi), 0, 0],
        ]
    )
    aether_residual = (aether_momentum - expected_aether).applyfunc(sp.factor)
    metric_residual = (metric_momentum - expected_metric).applyfunc(sp.factor)
    if aether_residual != sp.zeros(3, 1) or metric_residual != sp.zeros(3, 3):
        raise ValueError("static compact canonical momenta changed")

    # Omitting L_n chi=A^i W_i/chi removes a genuine p_A contribution.
    corrupted_aether = sp.Matrix([chi * f * f1 / 32, 0, 0])
    corrupted_residual = (corrupted_aether - expected_aether).applyfunc(sp.factor)
    if corrupted_residual == sp.zeros(3):
        raise ValueError("unit-branch negative control failed")

    coordinates = sp.symbols("x1:4", real=True)
    profile = sp.Function("F")(*coordinates)
    profile_chi = sp.sqrt(1 + profile**2)
    p_profile = sp.Matrix(
        [profile * sp.diff(profile, coordinates[0]) / (32 * profile_chi), 0, 0]
    )
    pi_profile = sp.zeros(3)
    pi_profile[0, 0] = (
        (profile**2 - 1) * sp.diff(profile, coordinates[0]) / (64 * profile_chi)
    )
    pi_profile[0, 1] = pi_profile[1, 0] = -sp.diff(
        profile, coordinates[1]
    ) / (128 * profile_chi)
    pi_profile[0, 2] = pi_profile[2, 0] = -sp.diff(
        profile, coordinates[2]
    ) / (128 * profile_chi)
    momentum_residual = []
    for i in range(3):
        cotangent_lift = -2 * sum(
            sp.diff(pi_profile[j, i], coordinates[j]) for j in range(3)
        )
        cotangent_lift += p_profile[0] * sp.diff(profile, coordinates[i])
        if i == 0:
            cotangent_lift -= sp.diff(p_profile[0] * profile, coordinates[0])
        momentum_residual.append(sp.factor(cotangent_lift))

    body = {
        "candidate_parameters": {"c1": "1/32", "c2": "0", "c3": "0", "c4": "1/32"},
        "chart": "q_ij=delta_ij,N=1,N^i=0,K_ij=0,L_n A_i=0",
        "profile": "A_i=F*delta_i1; F=10*(1-r^2)^4_+",
        "unit_branch": "chi=sqrt(1+F^2); L_n chi=A^i L_n A_i/chi=0",
        "canonical_aether_momentum": [str(item) for item in expected_aether],
        "canonical_metric_momentum": [
            [str(expected_metric[i, j]) for j in range(3)] for i in range(3)
        ],
        "outside_compact_support": "F=partial_i F=0 implies pi^ij=p_A^i=0",
        "momentum_constraint_density": (
            "D_i=-2*partial_j(pi^j_i)+p_A^j*partial_i(A_j)-"
            "partial_j(p_A^j*A_i)"
        ),
        "candidate_bound_flat_chart_D_residual": [str(item) for item in momentum_residual],
        "regularity": {
            "F": "C3_compact_support",
            "pi_and_p_A": "C2_compact_support",
            "D_residual": "C1_compact_support",
        },
        "unit_branch_negative_control": {
            "mutation": "omit_L_n_chi_equals_A_dot_L_n_A_over_chi",
            "corrupted_p_A": [str(item) for item in corrupted_aether],
            "exact_residual": [str(item) for item in corrupted_residual],
            "rejected_mutated_derivation": True,
        },
        "canonical_seed_point_registered": True,
        "constraint_satisfying_background": False,
        "scope": (
            "exact candidate-bound flat-chart canonical point and its distributed cotangent-lift "
            "momentum residual; no off-flat canonical Hamiltonian functional or Frechet DAG"
        ),
    }
    return {**body, "content_sha256": _sha(body)}


def _candidate_certificate(
    source: dict[str, Any], compact_source: dict[str, Any], controls_root: str
) -> dict[str, Any]:
    prior = source.get("lower_order_coefficient_contract_certificate")
    compact_records = {
        item["candidate_id"]: item for item in compact_source.get("candidate_records", [])
    }
    compact = compact_records.get(source["candidate_id"])
    if (
        not isinstance(prior, dict)
        or prior.get("lower_order_coefficient_contract_declared") is not True
        or prior.get("compact_profile_C3_weighted_jet_bounds_proven") is not True
        or not isinstance(compact, dict)
        or compact.get("parameters") != source.get("parameters")
        or compact.get("typed_action_ir_sha256") != source.get("typed_action_ir_sha256")
        or compact.get("finite_amplitude_negative_seed_certificate", {}).get(
            "seed_amplitude_squared"
        )
        != "100"
    ):
        raise ValueError("canonical seed source specialization changed")
    static = exact_static_canonical_seed_control()
    if static["candidate_parameters"] != source["parameters"]:
        raise ValueError("canonical seed candidate parameters changed")
    body = {
        "candidate_id": source["candidate_id"],
        "typed_action_ir_sha256": source["typed_action_ir_sha256"],
        "source_lower_order_record_sha256": source["content_sha256"],
        "source_compact_seed_record_sha256": compact["content_sha256"],
        "reviewed_ADM_controls_root_sha256": controls_root,
        "canonical_seed_point": static,
        "distributed_constraint_dependency_DAG": {
            "registered_nodes": {
                "flat_static_profile": "F_and_spatial_derivatives_through_order_3",
                "positive_unit_branch": "chi=sqrt(1+F^2)",
                "flat_chart_Legendre_image": "pi^ij_and_p_A^i_exact",
                "flat_chart_momentum_density": "D_i_exact_through_second_profile_derivatives",
                "principal_York_block": "bound_by_source_predecessor",
            },
            "missing_nodes": {
                "spatially_distributed_canonical_H_core": (
                    "H_core[q,A,pi,p_A] after the positive-unit-branch Legendre inversion"
                ),
                "off_flat_metric_covariantization": (
                    "pi[q,A,K,W] and p_A[q,A,K,W] in a neighborhood of q=delta"
                ),
                "H_D_Frechet_edges": (
                    "first and second derivatives with respect to phi,X and their jets"
                ),
                "B_and_C_tensor_registry": "3x4x4 order-one and 4x4 order-zero arrays",
            },
            "complete": False,
        },
        "reviewed_scope_obstruction": {
            "generic_3plus1_scope": (
                "construction of the spatially distributed canonical Hamiltonian and boundary "
                "terms remains separate in the bound reviewed control"
            ),
            "point_values_do_not_determine_Frechet_derivatives": True,
            "strongest_exact_conclusion": (
                "the canonical point is registered, but the requested lower-order coefficient "
                "DAG is underdetermined until an off-flat distributed canonical density "
                "functional is supplied"
            ),
        },
        "full_canonical_background_point_registered": True,
        "candidate_bound_flat_chart_D_residual_DAG_registered": True,
        "spatially_distributed_canonical_H_core_registered": False,
        "metric_covariantized_H_D_Frechet_DAG_registered": False,
        "distributed_lower_order_coefficient_registry_complete": False,
        "weighted_relative_lower_order_bound_proven": False,
        "weighted_Fredholm_isomorphism_proven": False,
        "full_operator_inverse_norm_proven": False,
        "nonlinear_remainder_majorant_proven": False,
        "completed_boundary_sign_persistence_proven": False,
        "constraint_satisfying_negative_total_energy_datum_proven": False,
        "candidate_rejection_authorized": False,
    }
    return {**body, "content_sha256": _sha(body)}


def build_future_aether_canonical_seed_constraint_dag_gate(
    config: dict[str, Any], root: str | Path
) -> dict[str, Any]:
    root = Path(root).resolve()
    _validate_config(config)
    _bound_path(root, config["campaign_implementation"], "canonical-seed implementation")
    predecessor_implementation = _bound_path(
        root, config["source_lower_order_implementation"], "lower-order implementation"
    )
    predecessor_config = _bound_json(
        root, config["source_lower_order_config"], "lower-order config"
    )
    predecessor = _bound_json(
        root, config["source_lower_order_artifact"], "lower-order artifact", content=True
    )
    compact = _bound_json(
        root, config["source_compact_seed_artifact"], "compact seed artifact", content=True
    )
    report = _bound_json(root, config["formal_report"], "formal report")
    callback_source = Path(
        inspect.getsourcefile(build_future_aether_lower_order_coefficient_contract_gate) or ""
    ).resolve()
    if callback_source != predecessor_implementation:
        raise ValueError("lower-order implementation entrypoint changed")
    if build_future_aether_lower_order_coefficient_contract_gate(predecessor_config, root) != predecessor:
        raise ValueError("lower-order artifact no longer replays")
    if (
        predecessor.get("candidate_count") != 14
        or predecessor.get("decision_counts") != {"blocked": 14}
        or predecessor.get("first_blocker_counts")
        != {
            CHARACTERISTIC_BLOCKER: 11,
            YORK_SHELL_BLOCKER: 2,
            SOURCE_ELLIPTIC_BLOCKER: 1,
        }
        or predecessor.get("full_canonical_background_point_registered_count") != 0
        or predecessor.get("candidate_rejection_authorized_count") != 0
        or compact.get("candidate_count") != 14
    ):
        raise ValueError("canonical-seed predecessor scope changed")
    reviewed, reviewed_root = _reviewed_controls(config, root, report)

    records = []
    blockers: Counter[str] = Counter()
    canonical_count = 0
    d_dag_count = 0
    for source in predecessor["candidate_records"]:
        source_body = {key: item for key, item in source.items() if key != "content_sha256"}
        if source.get("content_sha256") != _sha(source_body):
            raise ValueError("canonical-seed predecessor record changed")
        if source["first_blocker"] == SOURCE_ELLIPTIC_BLOCKER:
            certificate = _candidate_certificate(source, compact, reviewed_root)
            blocker = BLOCKER
            canonical_status = "pass"
            canonical_count += 1
            d_dag_count += 1
        else:
            certificate = None
            blocker = source["first_blocker"]
            canonical_status = "not_reached"
        blockers[blocker] += 1
        provenance_body = {
            "candidate_id": source["candidate_id"],
            "typed_action_ir_sha256": source["typed_action_ir_sha256"],
            "source_lower_order_record_sha256": source["content_sha256"],
            "canonical_seed_certificate_sha256": (
                certificate["content_sha256"] if certificate is not None else None
            ),
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
            "source_lower_order_record_sha256": source["content_sha256"],
            "parameters": source["parameters"],
            "exact_specialization": source["exact_specialization"],
            "canonical_seed_constraint_DAG_certificate": certificate,
            "gate_ledger": {
                "source_action_and_predecessor_binding": {"status": "pass"},
                "reviewed_ADM_control_replay": {"status": "pass"},
                "full_flat_chart_canonical_seed_point": {"status": canonical_status},
                "candidate_bound_flat_chart_D_residual_DAG": {"status": canonical_status},
                "spatially_distributed_canonical_H_core": {"status": "blocked"},
                "metric_covariantized_H_D_Frechet_DAG": {"status": "blocked"},
                "distributed_lower_order_B_C_registry": {"status": "blocked"},
                "weighted_Fredholm_and_full_inverse": {"status": "blocked"},
                "nonlinear_remainder": {"status": "blocked"},
                "completed_boundary_sign_persistence": {"status": "blocked"},
                "observational_data_seal": {"status": "pass"},
            },
            "decision": "blocked",
            "first_blocker": blocker,
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
    expected_blockers = {
        CHARACTERISTIC_BLOCKER: 11,
        YORK_SHELL_BLOCKER: 2,
        BLOCKER: 1,
    }
    if canonical_count != 1 or d_dag_count != 1 or dict(blockers) != expected_blockers:
        raise ValueError("canonical-seed candidate partition changed")
    record_root = _sha(
        [
            [item["candidate_id"], item["typed_action_ir_sha256"], item["content_sha256"]]
            for item in records
        ]
    )
    provenance_body = {
        "source_lower_order_content_sha256": predecessor["content_sha256"],
        "source_lower_order_record_registry_root_sha256": predecessor[
            "candidate_record_registry_root_sha256"
        ],
        "source_compact_seed_content_sha256": compact["content_sha256"],
        "reviewed_ADM_controls_root_sha256": reviewed_root,
        "candidate_record_registry_root_sha256": record_root,
        "data_eligibility": ELIGIBILITY,
    }
    body = {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": config["campaign_id"],
        "immutable_config_sha256": _sha(config),
        "source_lower_order_binding": config["source_lower_order_artifact"],
        "source_compact_seed_binding": config["source_compact_seed_artifact"],
        "reviewed_ADM_controls": reviewed,
        "candidate_count": 14,
        "decision_counts": {"blocked": 14},
        "first_blocker_counts": dict(sorted(blockers.items())),
        "formal_pass_count": 0,
        "candidate_rejection_authorized_count": 0,
        "constraint_satisfying_negative_total_energy_datum_count": 0,
        "forced_characteristic_candidate_count": 11,
        "York_symbol_shell_candidate_count": 2,
        "uniformly_elliptic_candidate_count": 1,
        "full_canonical_background_point_registered_count": 1,
        "candidate_bound_flat_chart_D_residual_DAG_registered_count": 1,
        "spatially_distributed_canonical_H_core_registered_count": 0,
        "metric_covariantized_H_D_Frechet_DAG_registered_count": 0,
        "distributed_lower_order_coefficient_registry_complete_count": 0,
        "weighted_relative_lower_order_bound_pass_count": 0,
        "weighted_Fredholm_isomorphism_pass_count": 0,
        "full_operator_inverse_norm_pass_count": 0,
        "nonlinear_remainder_bound_pass_count": 0,
        "completed_boundary_sign_persistence_count": 0,
        "candidate_record_registry_root_sha256": record_root,
        "candidate_records": records,
        "bounded_canonical_seed_constraint_DAG_gate_completed": True,
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
            "The sole uniformly elliptic candidate now has an exact finite-tilt flat-chart "
            "canonical seed point: q, A, pi and p_A are explicit compact profiles, and the "
            "canonical cotangent-lift momentum residual is derived as a distributed second-jet "
            "DAG. The reviewed generic control explicitly leaves the spatially distributed "
            "canonical Hamiltonian and boundary terms separate. A point value cannot determine "
            "the off-flat Frechet derivatives needed for the H/D order-one and order-zero "
            "coefficient arrays. No weighted Fredholm inverse, nonlinear remainder, boundary-sign "
            "persistence, constraint solution, formal pass, or theory rejection is claimed."
        ),
    }
    return {**body, "content_sha256": _sha(body)}


def validate_future_aether_canonical_seed_constraint_dag_gate(
    artifact: dict[str, Any],
) -> None:
    body = {key: item for key, item in artifact.items() if key != "content_sha256"}
    records = artifact.get("candidate_records")
    if (
        artifact.get("schema_version") != RESULT_SCHEMA
        or artifact.get("content_sha256") != _sha(body)
        or artifact.get("candidate_count") != 14
        or artifact.get("decision_counts") != {"blocked": 14}
        or artifact.get("formal_pass_count") != 0
        or artifact.get("candidate_rejection_authorized_count") != 0
        or artifact.get("constraint_satisfying_negative_total_energy_datum_count") != 0
        or artifact.get("full_canonical_background_point_registered_count") != 1
        or artifact.get("candidate_bound_flat_chart_D_residual_DAG_registered_count") != 1
        or artifact.get("spatially_distributed_canonical_H_core_registered_count") != 0
        or artifact.get("metric_covariantized_H_D_Frechet_DAG_registered_count") != 0
        or artifact.get("distributed_lower_order_coefficient_registry_complete_count") != 0
        or artifact.get("weighted_Fredholm_isomorphism_pass_count") != 0
        or artifact.get("full_operator_inverse_norm_pass_count") != 0
        or artifact.get("nonlinear_remainder_bound_pass_count") != 0
        or artifact.get("completed_boundary_sign_persistence_count") != 0
        or artifact.get("full_candidate_specific_formal_completion_claimed") is not False
        or artifact.get("automatic_downstream_enqueue_performed") is not False
        or artifact.get("solar_bundle_count") != 0
        or artifact.get("observational_data_opened") is not False
        or artifact.get("dark_matter_or_halo_inputs") is not False
        or artifact.get("redshift_distance_inputs") is not False
        or artifact.get("paid_llm_spend_usd") != 0.0
        or artifact.get("data_eligibility") != ELIGIBILITY
        or not isinstance(records, list)
        or len(records) != 14
    ):
        raise ValueError("future Aether canonical-seed artifact is inconsistent")
    blockers: Counter[str] = Counter()
    certificate_count = 0
    for record in records:
        record_body = {key: item for key, item in record.items() if key != "content_sha256"}
        provenance = record.get("provenance", {})
        provenance_body = {
            key: item for key, item in provenance.items() if key != "binding_sha256"
        }
        certificate = record.get("canonical_seed_constraint_DAG_certificate")
        if (
            record.get("content_sha256") != _sha(record_body)
            or record.get("decision") != "blocked"
            or record.get("formal_pass") is not False
            or record.get("candidate_rejection_authorized") is not False
            or record.get("constraint_satisfying_negative_total_energy_datum_proven") is not False
            or record.get("full_formal_completion_claimed") is not False
            or record.get("automatic_downstream_enqueue_performed") is not False
            or record.get("solar_bundle_generated") is not False
            or record.get("observational_data_opened") is not False
            or record.get("data_eligibility") != ELIGIBILITY
            or provenance.get("binding_sha256") != _sha(provenance_body)
        ):
            raise ValueError("future Aether canonical-seed record is inconsistent")
        blockers[record["first_blocker"]] += 1
        if certificate is not None:
            certificate_body = {
                key: item for key, item in certificate.items() if key != "content_sha256"
            }
            if (
                certificate.get("content_sha256") != _sha(certificate_body)
                or certificate.get("full_canonical_background_point_registered") is not True
                or certificate.get("candidate_bound_flat_chart_D_residual_DAG_registered")
                is not True
                or certificate.get("metric_covariantized_H_D_Frechet_DAG_registered")
                is not False
                or certificate.get("candidate_rejection_authorized") is not False
            ):
                raise ValueError("future Aether canonical-seed certificate is inconsistent")
            certificate_count += 1
    expected_blockers = {
        CHARACTERISTIC_BLOCKER: 11,
        YORK_SHELL_BLOCKER: 2,
        BLOCKER: 1,
    }
    expected_root = _sha(
        [
            [item["candidate_id"], item["typed_action_ir_sha256"], item["content_sha256"]]
            for item in records
        ]
    )
    if (
        dict(blockers) != expected_blockers
        or artifact.get("first_blocker_counts") != dict(sorted(expected_blockers.items()))
        or certificate_count != 1
        or artifact.get("candidate_record_registry_root_sha256") != expected_root
    ):
        raise ValueError("future Aether canonical-seed registry is inconsistent")


def publish_future_aether_canonical_seed_constraint_dag_gate(
    config: dict[str, Any], root: str | Path, target: str | Path
) -> dict[str, Any]:
    artifact = build_future_aether_canonical_seed_constraint_dag_gate(config, root)
    validate_future_aether_canonical_seed_constraint_dag_gate(artifact)
    target = Path(target).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if _load(target) != artifact:
            raise ValueError("refusing to replace divergent canonical-seed artifact")
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
    artifact = publish_future_aether_canonical_seed_constraint_dag_gate(
        _load(Path(arguments.config)), arguments.root, arguments.output
    )
    print(_canonical({key: item for key, item in artifact.items() if key != "candidate_records"}))


if __name__ == "__main__":
    main()
