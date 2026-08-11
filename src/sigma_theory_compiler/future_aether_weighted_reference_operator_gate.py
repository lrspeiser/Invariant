"""Weighted metric-reference operator gate for regular future Aether seeds."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any

from .future_aether_nonlinear_lift_characteristic_gate import CHARACTERISTIC_BLOCKER
from .future_aether_weighted_ift_contract_gate import BLOCKER as SOURCE_REGULAR_BLOCKER
from .future_aether_weighted_ift_contract_gate import (
    build_future_aether_weighted_ift_contract_gate,
)
from .promotion_orchestrator import ELIGIBILITY

CONFIG_SCHEMA = "sigma-future-aether-weighted-reference-operator-gate-config-1.0"
RESULT_SCHEMA = "sigma-future-aether-weighted-reference-operator-gate-1.0"
TARGET_FAMILY = "AETHER_K1234_PARAMETER_CELL"
BLOCKER = (
    "candidate_bound_Aether_constraint_variable_block_and_off_diagonal_principal_symbol_"
    "on_declared_weighted_spaces"
)
WEIGHT_DELTA = Fraction(-1, 2)


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
        "source_weighted_ift_artifact",
        "source_weighted_ift_config",
        "source_weighted_ift_implementation",
        "source_weak_field_artifact",
        "weighted_contract",
        "budget",
        "observational_authorization",
        "data_eligibility",
        "external_paid_llm_calls",
    }
    if set(config) != required or config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("future Aether weighted reference config is invalid")
    if config.get("data_eligibility") != ELIGIBILITY:
        raise ValueError("future Aether weighted reference eligibility is open")
    if config.get("observational_authorization") is not False:
        raise ValueError("future Aether weighted reference opened observations")
    if config.get("external_paid_llm_calls") is not False:
        raise ValueError("future Aether weighted reference enabled paid LLM calls")
    if config.get("weighted_contract") != {
        "spatial_dimension": 3,
        "weight_delta": "-1/2",
        "differentiability_order": 2,
        "metric_domain": "H^2_-1/2(R3;scalar) direct_sum H^2_-1/2(R3;R3)",
        "metric_codomain": "L^2_-5/2(R3;scalar) direct_sum L^2_-5/2(R3;R3)",
        "gauge": "h_ij=4*phi*delta_ij; K_ij=(L_X)_ij",
        "norm_convention": (
            "||u||^2_Hk_delta=sum_j=0^k integral_R3 (1+|x|^2)^(-(delta-j)-3/2)*|nabla^j u|^2 dx"
        ),
    }:
        raise ValueError("future Aether weighted reference contract is not exact")
    if config.get("budget") != {
        "maximum_candidates": 14,
        "maximum_regular_adm_candidates": 3,
        "maximum_symbol_dimension": 4,
        "maximum_paid_llm_spend_usd": 0.0,
    }:
        raise ValueError("future Aether weighted reference budget is not exact")


def exact_reference_symbol(unit_covector: tuple[Fraction, Fraction, Fraction]) -> dict[str, Any]:
    """Return the Planck-normalized scalar/York principal symbol at a unit covector."""

    norm = sum(item * item for item in unit_covector)
    if norm != 1:
        raise ValueError("reference symbol covector must have exact unit norm")
    york = [
        [2 * (Fraction(int(i == j)) + unit_covector[i] * unit_covector[j] / 3) for j in range(3)]
        for i in range(3)
    ]
    trace = sum(york[i][i] for i in range(3))
    trace_square = sum(york[i][j] * york[j][i] for i in range(3) for j in range(3))
    determinant = (
        york[0][0] * (york[1][1] * york[2][2] - york[1][2] * york[2][1])
        - york[0][1] * (york[1][0] * york[2][2] - york[1][2] * york[2][0])
        + york[0][2] * (york[1][0] * york[2][1] - york[1][1] * york[2][0])
    )
    if (
        trace != Fraction(20, 3)
        or trace_square != Fraction(136, 9)
        or determinant != Fraction(32, 3)
    ):
        raise ValueError("York principal symbol invariants changed")
    body = {
        "unit_covector": [str(item) for item in unit_covector],
        "Planck_normalized_scalar_symbol_eigenvalue": "4",
        "York_momentum_symbol_matrix": [[str(item) for item in row] for row in york],
        "York_symbol_eigenvalues": ["2", "2", "8/3"],
        "combined_symbol_eigenvalues": ["2", "2", "8/3", "4"],
        "combined_symbol_determinant": "128/3",
        "combined_principal_ellipticity_margin": "2",
        "principal_symbol_invertible": True,
    }
    return {**body, "content_sha256": _sha(body)}


def exact_ungauged_hamiltonian_negative_control() -> dict[str, Any]:
    """Exhibit an exact pure-diffeomorphism kernel of the ungauged scalar symbol."""

    covector = (Fraction(1), Fraction(0), Fraction(0))
    gauge_vector = (Fraction(0), Fraction(1), Fraction(0))
    perturbation = [
        [covector[i] * gauge_vector[j] + covector[j] * gauge_vector[i] for j in range(3)]
        for i in range(3)
    ]
    trace = sum(perturbation[i][i] for i in range(3))
    double_contraction = sum(
        covector[i] * perturbation[i][j] * covector[j] for i in range(3) for j in range(3)
    )
    curvature_symbol = trace - double_contraction
    if not any(perturbation[i][j] for i in range(3) for j in range(3)) or curvature_symbol != 0:
        raise ValueError("ungauged Hamiltonian negative control changed")
    body = {
        "unit_covector": [str(item) for item in covector],
        "pure_diffeomorphism_vector": [str(item) for item in gauge_vector],
        "metric_perturbation_h_ij": [[str(item) for item in row] for row in perturbation],
        "linearized_scalar_curvature_symbol": str(curvature_symbol),
        "nonzero_pure_gauge_symbol_kernel_witness": True,
        "ungauged_scalar_symbol_injective": False,
    }
    return {**body, "content_sha256": _sha(body)}


def _reference_certificate(source: dict[str, Any]) -> dict[str, Any]:
    prior = source["weighted_ift_contract_certificate"]
    if prior is None:
        raise ValueError("regular weighted reference candidate lacks prior certificate")
    controls = prior["available_exact_controls"]
    if (
        controls.get("scalar_conformal_reference_solution_exists") is not True
        or controls.get("vector_York_reference_solution_exists") is not True
    ):
        raise ValueError("reference conformal/York controls changed")
    axial = exact_reference_symbol((Fraction(1), Fraction(0), Fraction(0)))
    diagonal = exact_reference_symbol((Fraction(2, 3), Fraction(2, 3), Fraction(1, 3)))
    ungauged = exact_ungauged_hamiltonian_negative_control()
    body = {
        "candidate_id": source["candidate_id"],
        "typed_action_ir_sha256": source["typed_action_ir_sha256"],
        "source_weighted_ift_record_sha256": source["content_sha256"],
        "declared_metric_weighted_contract": {
            "spatial_dimension": 3,
            "weight_delta": str(WEIGHT_DELTA),
            "domain": "H^2_-1/2(R3;scalar) direct_sum H^2_-1/2(R3;R3)",
            "codomain": "L^2_-5/2(R3;scalar) direct_sum L^2_-5/2(R3;R3)",
            "gauge": "h_ij=4*phi*delta_ij; K_ij=(L_X)_ij",
            "norm_convention": (
                "||u||^2_Hk_delta=sum_j=0^k integral_R3 (1+|x|^2)^(-(delta-j)-3/2)*|nabla^j u|^2 dx"
            ),
            "reference_map": ("(phi,X)->(4*M2*Delta(phi),-2*(Delta(X)+(1/3)*grad(div(X))))"),
            "M2_premise": "M2>0; scalar symbol is Planck-normalized by M2",
        },
        "principal_symbol_certificates": {
            "axial_unit_covector": axial,
            "rational_diagonal_unit_covector": diagonal,
            "direction_independent_spectrum_proven_by_rank_one_projector_identity": True,
            "spectrum": ["2", "2", "8/3", "4"],
            "ellipticity_margin": "2",
        },
        "ungauged_negative_control": ungauged,
        "decaying_kernel_certificate": {
            "scalar": (
                "Delta(phi)=0 and phi in H^2_-1/2; integration by parts gives "
                "integral|grad(phi)|^2=0; decay removes constants"
            ),
            "York_vector": (
                "Delta_L(X)=0 and X in H^2_-1/2; integration by parts gives "
                "integral|L_X|^2=0; decay removes Euclidean conformal Killing fields"
            ),
            "reference_metric_kernel_trivial": True,
        },
        "compact_source_right_inverse": {
            "scalar_conformal_Newton_solution": True,
            "vector_decaying_York_solution": True,
            "scope": "registered compact frozen sources only",
        },
        "carried_candidate_controls": {
            "Aether_Legendre_inverse_bound": controls[
                "uniform_Aether_Legendre_block_inverse_bound"
            ],
            "negative_source_energy_margin_over_pi": controls[
                "strict_negative_source_energy_margin_over_pi"
            ],
        },
        "candidate_Aether_constraint_variables_declared": False,
        "candidate_Aether_constraint_principal_block_derived": False,
        "metric_Aether_off_diagonal_principal_symbol_derived": False,
        "full_coupled_Fredholm_operator_defined": False,
        "full_weighted_operator_isomorphism_proven": False,
        "computable_full_inverse_norm_proven": False,
        "nonlinear_remainder_majorant_proven": False,
        "completed_boundary_sign_persistence_proven": False,
        "candidate_rejection_authorized": False,
    }
    return {**body, "content_sha256": _sha(body)}


def build_future_aether_weighted_reference_operator_gate(
    config: dict[str, Any], root: str | Path
) -> dict[str, Any]:
    root = Path(root).resolve()
    _validate_config(config)
    _bound_path(root, config["campaign_implementation"], "weighted reference implementation")
    source_implementation = _bound_path(
        root, config["source_weighted_ift_implementation"], "weighted IFT implementation"
    )
    source_config = _bound_json(root, config["source_weighted_ift_config"], "weighted IFT config")
    source_artifact = _bound_json(
        root,
        config["source_weighted_ift_artifact"],
        "weighted IFT artifact",
        content=True,
    )
    weak_artifact = _bound_json(
        root, config["source_weak_field_artifact"], "weak-field artifact", content=True
    )
    callback_source = Path(
        inspect.getsourcefile(build_future_aether_weighted_ift_contract_gate) or ""
    ).resolve()
    if callback_source != source_implementation:
        raise ValueError("weighted IFT implementation entrypoint changed")
    if build_future_aether_weighted_ift_contract_gate(source_config, root) != source_artifact:
        raise ValueError("weighted IFT artifact no longer replays")
    if (
        source_artifact.get("candidate_count") != 14
        or source_artifact.get("decision_counts") != {"blocked": 14}
        or source_artifact.get("first_blocker_counts")
        != {CHARACTERISTIC_BLOCKER: 11, SOURCE_REGULAR_BLOCKER: 3}
        or source_artifact.get("reference_conformal_York_Aether_block_control_count") != 3
        or source_artifact.get("candidate_rejection_authorized_count") != 0
        or source_artifact["provenance"].get("source_weak_field_content_sha256")
        != weak_artifact.get("content_sha256")
    ):
        raise ValueError("weighted reference source scope changed")

    records = []
    blockers: Counter[str] = Counter()
    regular_count = 0
    for source in source_artifact["candidate_records"]:
        source_body = {key: value for key, value in source.items() if key != "content_sha256"}
        if source.get("content_sha256") != _sha(source_body):
            raise ValueError("weighted reference source record changed")
        if source["first_blocker"] == SOURCE_REGULAR_BLOCKER:
            certificate = _reference_certificate(source)
            blocker = BLOCKER
            status = "pass"
            regular_count += 1
        else:
            certificate = None
            blocker = CHARACTERISTIC_BLOCKER
            status = "not_reached"
        blockers[blocker] += 1
        provenance_body = {
            "candidate_id": source["candidate_id"],
            "typed_action_ir_sha256": source["typed_action_ir_sha256"],
            "source_weighted_ift_record_sha256": source["content_sha256"],
            "weighted_reference_certificate_sha256": (
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
            "source_weighted_ift_record_sha256": source["content_sha256"],
            "parameters": source["parameters"],
            "exact_specialization": source["exact_specialization"],
            "weighted_reference_operator_certificate": certificate,
            "gate_ledger": {
                "source_action_and_predecessor_binding": {"status": "pass"},
                "declared_metric_weighted_domain_codomain_and_gauge": {"status": status},
                "metric_reference_principal_ellipticity": {"status": status},
                "metric_reference_decaying_kernel": {"status": status},
                "registered_compact_source_right_inverse": {"status": status},
                "candidate_Aether_constraint_principal_block": {"status": "blocked"},
                "full_coupled_Fredholm_isomorphism_and_norm": {"status": "blocked"},
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
    if regular_count != 3 or dict(blockers) != {CHARACTERISTIC_BLOCKER: 11, BLOCKER: 3}:
        raise ValueError("future Aether weighted reference partition changed")
    record_root = _sha(
        [
            [item["candidate_id"], item["typed_action_ir_sha256"], item["content_sha256"]]
            for item in records
        ]
    )
    provenance_body = {
        "source_weighted_ift_content_sha256": source_artifact["content_sha256"],
        "source_weighted_ift_record_registry_root_sha256": source_artifact[
            "candidate_record_registry_root_sha256"
        ],
        "source_weak_field_content_sha256": weak_artifact["content_sha256"],
        "candidate_record_registry_root_sha256": record_root,
        "data_eligibility": ELIGIBILITY,
    }
    body = {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": config["campaign_id"],
        "immutable_config_sha256": _sha(config),
        "source_weighted_ift_binding": config["source_weighted_ift_artifact"],
        "source_weak_field_binding": config["source_weak_field_artifact"],
        "candidate_count": 14,
        "decision_counts": {"blocked": 14},
        "first_blocker_counts": dict(sorted(blockers.items())),
        "formal_pass_count": 0,
        "candidate_rejection_authorized_count": 0,
        "constraint_satisfying_negative_total_energy_datum_count": 0,
        "forced_characteristic_candidate_count": 11,
        "regular_ADM_candidate_count": 3,
        "declared_metric_weighted_contract_count": 3,
        "metric_reference_principal_ellipticity_pass_count": 3,
        "metric_reference_trivial_kernel_pass_count": 3,
        "registered_compact_source_right_inverse_count": 3,
        "ungauged_pure_diffeomorphism_negative_control_count": 3,
        "candidate_Aether_constraint_principal_block_pass_count": 0,
        "full_coupled_Fredholm_operator_defined_count": 0,
        "full_weighted_operator_isomorphism_pass_count": 0,
        "computable_full_inverse_norm_count": 0,
        "nonlinear_remainder_bound_pass_count": 0,
        "completed_boundary_sign_persistence_count": 0,
        "candidate_record_registry_root_sha256": record_root,
        "candidate_records": records,
        "bounded_weighted_reference_operator_gate_completed": True,
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
            "For the three regular candidates this gate declares the concrete delta=-1/2 metric "
            "Sobolev domain/codomain and conformal-York gauge, proves the exact direction-independent "
            "reference symbol spectrum (2,2,8/3,4), and proves the decaying reference kernel is "
            "trivial. Existing evidence supplies right inverses only for the registered compact "
            "frozen sources. The first missing premise is the candidate-bound Aether constraint "
            "variable block and its off-diagonal principal coupling on these spaces; therefore no "
            "full Fredholm/isomorphism, norm, nonlinear lift, boundary-sign, or rejection claim is made."
        ),
    }
    return {**body, "content_sha256": _sha(body)}


def publish_future_aether_weighted_reference_operator_gate(
    config: dict[str, Any], root: str | Path, target: str | Path
) -> dict[str, Any]:
    artifact = build_future_aether_weighted_reference_operator_gate(config, root)
    target = Path(target).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if _load(target) != artifact:
            raise ValueError("refusing to replace divergent weighted reference artifact")
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
    artifact = publish_future_aether_weighted_reference_operator_gate(
        _load(Path(arguments.config)), arguments.root, arguments.output
    )
    print(_canonical({key: value for key, value in artifact.items() if key != "candidate_records"}))


if __name__ == "__main__":
    main()
