"""AE no-go audit for flat static globally pure-twist Aether completions."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Any

import sympy as sp

from .aether_parameter_cell_formal_gate_campaign import _specialize
from .future_aether_constraint_boundary_embedding_audit import (
    build_future_aether_constraint_boundary_embedding_audit,
)
from .promotion_orchestrator import ELIGIBILITY

CONFIG_SCHEMA = "sigma-future-aether-pure-twist-ae-no-go-audit-config-1.0"
RESULT_SCHEMA = "sigma-future-aether-pure-twist-ae-no-go-audit-1.0"
TARGET_FAMILY = "AETHER_K1234_PARAMETER_CELL"
SOURCE_BLOCKER = (
    "constraint_satisfying_asymptotically_Euclidean_completion_of_negative_twist_witness"
)
BLOCKER = (
    "candidate_bound_AE_coupled_constraint_solution_beyond_flat_static_"
    "global_pure_twist_class_with_negative_completed_boundary_energy"
)


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
        "source_embedding_artifact",
        "source_embedding_config",
        "source_embedding_implementation",
        "budget",
        "observational_authorization",
        "data_eligibility",
        "external_paid_llm_calls",
    }
    if set(config) != required or config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("future Aether pure-twist AE no-go config is invalid")
    if config.get("data_eligibility") != ELIGIBILITY:
        raise ValueError("future Aether pure-twist AE no-go eligibility is open")
    if config.get("observational_authorization") is not False:
        raise ValueError("future Aether pure-twist AE no-go opened observations")
    if config.get("external_paid_llm_calls") is not False:
        raise ValueError("future Aether pure-twist AE no-go enabled paid LLM calls")
    if config.get("budget") != {
        "maximum_candidates": 14,
        "spatial_dimension": 3,
        "maximum_symbolic_linear_system_unknowns": 18,
        "maximum_paid_llm_spend_usd": 0.0,
    }:
        raise ValueError("future Aether pure-twist AE no-go budget is not exact")


@lru_cache(maxsize=1)
def _pure_twist_symbolic_control() -> dict[str, Any]:
    """Prove that a decaying Euclidean Killing covector is identically zero."""

    dimension = 3
    symmetric_pairs = tuple((i, j) for i in range(dimension) for j in range(i, dimension))
    variables = {
        (i, j, k): sp.Symbol(f"T{i}{j}{k}") for i, j in symmetric_pairs for k in range(dimension)
    }

    def second(i: int, j: int, k: int) -> sp.Symbol:
        return variables[(*sorted((i, j)), k)]

    # T_ijk=partial_i partial_j A_k is symmetric in i,j.  Differentiating
    # partial_i A_j+partial_j A_i=0 gives T_kij+T_kji=0.
    equations = [
        second(k, i, j) + second(k, j, i)
        for k in range(dimension)
        for i in range(dimension)
        for j in range(i, dimension)
    ]
    coefficient_matrix, _ = sp.linear_eq_to_matrix(equations, tuple(variables.values()))
    rank = int(coefficient_matrix.rank())
    unknown_count = len(variables)
    if rank != unknown_count or unknown_count != 18:
        raise ValueError("Euclidean pure-twist second-derivative no-go changed")

    x1, x2, x3, y, radial, eta_prime = sp.symbols(
        "x1 x2 x3 y R eta_prime", positive=True, finite=True
    )
    coordinates = sp.Matrix([x1, x2, x3])
    rotation = sp.Matrix([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])
    affine = sp.Matrix([sp.sqrt(y) - x2, x1, 0])
    # A_cut=eta(r^2) A_affine.  The antisymmetric eta*dA term drops from
    # E_ij=partial_(i A_j), leaving eta'*(x_i A_j+x_j A_i).
    symmetric_cutoff = eta_prime * (coordinates * affine.T + affine * coordinates.T)
    axis = {x1: radial, x2: 0, x3: 0}
    axis_residual = symmetric_cutoff.subs(axis)
    axis_norm_squared = sp.factor(sp.trace(axis_residual.T * axis_residual))
    expected_axis_norm_squared = sp.factor(eta_prime**2 * (4 * radial**2 * y + 2 * radial**4))
    if sp.factor(axis_norm_squared - expected_axis_norm_squared) != 0:
        raise ValueError("radial cutoff transition residual changed")

    affine_gradient = sp.Matrix(
        dimension,
        dimension,
        lambda i, j: sp.diff(affine[j], coordinates[i]),
    )
    affine_killing_residual = sp.simplify(affine_gradient + affine_gradient.T)
    affine_twist_norm_squared = sp.factor(sp.trace(affine_gradient.T * affine_gradient))
    if affine_killing_residual != sp.zeros(3) or affine_gradient != rotation:
        raise ValueError("affine pure-twist positive control changed")

    body = {
        "completion_class": {
            "spatial_manifold": "connected R^3 with q_ij=delta_ij",
            "gravitational_data": "K_ij=0, N=1, shift=0",
            "aether_data": "partial_t A_i=0 and partial_(i A_j)=0 globally",
            "AE_condition": "A_i=o(1) as r tends to infinity",
        },
        "differentiated_Killing_system": {
            "second_jet": "T_ijk=partial_i partial_j A_k=T_jik",
            "equations": "T_kij+T_kji=0",
            "unknown_count": unknown_count,
            "coefficient_rank": rank,
            "kernel_dimension": unknown_count - rank,
            "conclusion": "partial_i partial_j A_k=0",
        },
        "global_solution": "A_i=a_i+B_ij*x^j with B_ij=-B_ji",
        "AE_consequence": "A_i=o(1) forces a_i=0 and B_ij=0",
        "negative_twist_consequence": (
            "no nonzero local pure-twist witness has a completion in this class"
        ),
        "positive_control_non_AE_affine_rotation": {
            "field": "A=(sqrt(y)-x2,x1,0)",
            "Killing_residual": str(affine_killing_residual),
            "twist_gradient_norm_squared": str(affine_twist_norm_squared),
            "pure_twist": True,
            "AE": False,
            "lesson": "the AE premise is essential",
        },
        "positive_control_trivial_AE_solution": {
            "field": "A_i=0",
            "pure_twist": True,
            "AE": True,
            "contains_negative_twist_witness": False,
        },
        "negative_control_radial_cutoff": {
            "field": "A_cut=eta(r^2)*(sqrt(y)-x2,x1,0)",
            "center_conditions": "eta=1 and eta'=0 near r=0 preserve the exact local witness",
            "compact_support_condition": "eta=0 outside a finite radius",
            "transition_symmetric_gradient": "E_ij=eta'*(x_i*A_affine_j+x_j*A_affine_i)",
            "axis_transition_norm_squared": str(expected_axis_norm_squared),
            "strictly_positive_when": "R>0, y>0, eta'!=0",
            "rejected_false_claim": "a nonconstant compact radial cutoff remains globally pure twist",
        },
        "proved_obstruction": (
            "flat_static_global_pure_twist_AE_completion_of_nonzero_witness_is_impossible"
        ),
        "scope": (
            "exact kinematic no-go for the declared Euclidean completion class; no claim about "
            "non-pure-twist transition data, curved q_ij, nonzero K_ij/momenta, coupled "
            "constraint solutions, or completed total energy"
        ),
    }
    return {**body, "content_sha256": _sha(body)}


def _candidate_certificate(
    parameters: dict[str, str], witness: dict[str, Any], symbolic: dict[str, Any]
) -> dict[str, Any]:
    y = Fraction(witness["tilt_squared_y"])
    if (
        y <= 0
        or Fraction(witness["C_y"]) >= 0
        or witness.get("normalized_W2") != "2"
        or Fraction(witness["normalized_WA2"]) != y
    ):
        raise ValueError("future Aether negative twist witness changed")
    transition_norm_squared = 4 * y + 2
    if transition_norm_squared <= 0:
        raise ValueError("radial cutoff negative control lost strictness")
    body = {
        "parameters": parameters,
        "tilt_squared_y": str(y),
        "source_local_twist_coefficient": witness["C_y"],
        "center_witness_preserved_by_cutoff": True,
        "global_flat_static_pure_twist_AE_completion_exists": False,
        "obstruction_control_sha256": symbolic["content_sha256"],
        "normalized_axis_transition_choice": "R=1, eta'=1",
        "normalized_transition_symmetric_gradient_norm_squared": str(transition_norm_squared),
        "compact_cutoff_requires_non_pure_twist_transition": True,
        "Hamiltonian_constraint_solved": False,
        "momentum_constraint_solved": False,
        "completed_Aether_boundary_energy_defined": False,
        "constraint_satisfying_negative_total_energy_datum_proven": False,
        "candidate_rejection_authorized": False,
    }
    return {**body, "content_sha256": _sha(body)}


def build_future_aether_pure_twist_ae_no_go_audit(
    config: dict[str, Any], root: str | Path
) -> dict[str, Any]:
    root = Path(root).resolve()
    _validate_config(config)
    _bound_path(root, config["campaign_implementation"], "pure-twist AE no-go implementation")
    source_implementation = _bound_path(
        root, config["source_embedding_implementation"], "source embedding implementation"
    )
    source_config = _bound_json(root, config["source_embedding_config"], "source embedding config")
    source_artifact = _bound_json(
        root, config["source_embedding_artifact"], "source embedding artifact", content=True
    )
    callback_source = Path(
        inspect.getsourcefile(build_future_aether_constraint_boundary_embedding_audit) or ""
    ).resolve()
    if callback_source != source_implementation:
        raise ValueError("source embedding implementation entrypoint changed")
    if (
        build_future_aether_constraint_boundary_embedding_audit(source_config, root)
        != source_artifact
    ):
        raise ValueError("source embedding artifact no longer replays")
    if (
        source_artifact.get("candidate_count") != 14
        or source_artifact.get("decision_counts") != {"blocked": 14}
        or source_artifact.get("first_blocker_counts") != {SOURCE_BLOCKER: 14}
        or source_artifact.get("candidate_rejection_authorized_count") != 0
        or source_artifact.get("constraint_satisfying_negative_total_energy_datum_count") != 0
    ):
        raise ValueError("source Aether embedding decision scope changed")

    symbolic = _pure_twist_symbolic_control()
    records = []
    transition_norms: dict[str, int] = {}
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
        ):
            raise ValueError("source Aether embedding record changed")
        certificate = _candidate_certificate(source["parameters"], witness, symbolic)
        norm = certificate["normalized_transition_symmetric_gradient_norm_squared"]
        transition_norms[norm] = transition_norms.get(norm, 0) + 1
        gates = {
            "source_action_witness_and_embedding_audit_binding": {"status": "pass"},
            "Euclidean_Killing_second_jet_classification": {
                "status": "pass",
                "rank": symbolic["differentiated_Killing_system"]["coefficient_rank"],
                "unknown_count": symbolic["differentiated_Killing_system"]["unknown_count"],
            },
            "flat_static_global_pure_twist_AE_completion": {
                "status": "reject_completion_class",
                "reason": symbolic["proved_obstruction"],
            },
            "compact_radial_localization": {
                "status": "blocked_at_non_pure_twist_transition",
                "transition_norm_squared": norm,
            },
            "coupled_constraint_solution_beyond_obstructed_class": {
                "status": "blocked",
                "reason": (
                    "transition shear/expansion, curved metric or nonzero gravitational/Aether "
                    "momenta must be included and both secondary constraints solved"
                ),
            },
            "completed_AE_boundary_energy": {"status": "blocked"},
            "observational_data_seal": {"status": "pass"},
        }
        provenance_body = {
            "candidate_id": source["candidate_id"],
            "typed_action_ir_sha256": source["typed_action_ir_sha256"],
            "source_embedding_record_sha256": source["content_sha256"],
            "source_exact_specialization_sha256": specialization["content_sha256"],
            "pure_twist_no_go_certificate_sha256": certificate["content_sha256"],
            "symbolic_obstruction_control_sha256": symbolic["content_sha256"],
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
            "source_embedding_record_sha256": source["content_sha256"],
            "parameters": source["parameters"],
            "exact_specialization": specialization,
            "pure_twist_AE_no_go_certificate": certificate,
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
        raise ValueError("future Aether pure-twist AE no-go target count changed")
    record_root = _sha(
        [
            [item["candidate_id"], item["typed_action_ir_sha256"], item["content_sha256"]]
            for item in records
        ]
    )
    provenance_body = {
        "source_embedding_content_sha256": source_artifact["content_sha256"],
        "source_embedding_record_registry_root_sha256": source_artifact[
            "candidate_record_registry_root_sha256"
        ],
        "source_embedding_implementation_file_sha256": config["source_embedding_implementation"][
            "file_sha256"
        ],
        "symbolic_obstruction_control_sha256": symbolic["content_sha256"],
        "candidate_record_registry_root_sha256": record_root,
        "data_eligibility": ELIGIBILITY,
    }
    body = {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": config["campaign_id"],
        "immutable_config_sha256": _sha(config),
        "source_embedding_binding": config["source_embedding_artifact"],
        "candidate_count": len(records),
        "decision_counts": {"blocked": 14},
        "first_blocker_counts": {BLOCKER: 14},
        "formal_pass_count": 0,
        "candidate_rejection_authorized_count": 0,
        "constraint_satisfying_negative_total_energy_datum_count": 0,
        "flat_static_global_pure_twist_AE_completion_obstructed_count": 14,
        "compact_cutoff_non_pure_twist_transition_required_count": 14,
        "normalized_transition_symmetric_gradient_norm_squared_counts": dict(
            sorted(transition_norms.items())
        ),
        "symbolic_obstruction_control": symbolic,
        "candidate_record_registry_root_sha256": record_root,
        "candidate_records": records,
        "bounded_pure_twist_AE_no_go_audit_completed": True,
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
            "The flat, static, globally pure-twist completion class is now exhausted exactly: "
            "the Euclidean Killing equation forces A_i=a_i+B_ij x^j, and asymptotic decay "
            "forces a_i=B_ij=0. A compact radial cutoff can preserve each negative witness "
            "near its center, but it necessarily introduces a nonzero symmetric-gradient "
            "transition. All candidates remain blocked until such transition terms (or more "
            "general gravitational data) solve the coupled constraints and yield a defined, "
            "genuinely negative completed AE boundary energy."
        ),
    }
    return {**body, "content_sha256": _sha(body)}


def publish_future_aether_pure_twist_ae_no_go_audit(
    config: dict[str, Any], root: str | Path, target: str | Path
) -> dict[str, Any]:
    artifact = build_future_aether_pure_twist_ae_no_go_audit(config, root)
    target = Path(target).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if _load(target) != artifact:
            raise ValueError("refusing to replace divergent pure-twist AE no-go artifact")
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
    artifact = publish_future_aether_pure_twist_ae_no_go_audit(
        _load(Path(arguments.config)), arguments.root, arguments.output
    )
    print(_canonical({key: value for key, value in artifact.items() if key != "candidate_records"}))


if __name__ == "__main__":
    main()
