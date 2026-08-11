"""Weak-field AE constraint gate for the future Aether twist candidates."""

from __future__ import annotations

import argparse
import hashlib
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
from .future_aether_pure_twist_ae_no_go_audit import (
    build_future_aether_pure_twist_ae_no_go_audit,
)
from .promotion_orchestrator import ELIGIBILITY

CONFIG_SCHEMA = "sigma-future-aether-weak-field-ae-constraint-gate-config-1.0"
RESULT_SCHEMA = "sigma-future-aether-weak-field-ae-constraint-gate-1.0"
TARGET_FAMILY = "AETHER_K1234_PARAMETER_CELL"
SOURCE_BLOCKER = (
    "candidate_bound_AE_coupled_constraint_solution_beyond_flat_static_"
    "global_pure_twist_class_with_negative_completed_boundary_energy"
)
BLOCKER = (
    "finite_amplitude_candidate_bound_nonlinear_AE_coupled_constraint_solution_"
    "with_negative_completed_boundary_energy_beyond_positive_weak_field_quadratic_regime"
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
        "source_no_go_artifact",
        "source_no_go_config",
        "source_no_go_implementation",
        "budget",
        "observational_authorization",
        "data_eligibility",
        "external_paid_llm_calls",
    }
    if set(config) != required or config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("future Aether weak-field AE gate config is invalid")
    if config.get("data_eligibility") != ELIGIBILITY:
        raise ValueError("future Aether weak-field AE gate eligibility is open")
    if config.get("observational_authorization") is not False:
        raise ValueError("future Aether weak-field AE gate opened observations")
    if config.get("external_paid_llm_calls") is not False:
        raise ValueError("future Aether weak-field AE gate enabled paid LLM calls")
    if config.get("budget") != {
        "maximum_candidates": 14,
        "spatial_dimension": 3,
        "maximum_symbolic_polynomial_terms": 500,
        "maximum_paid_llm_spend_usd": 0.0,
    }:
        raise ValueError("future Aether weak-field AE gate budget is not exact")


def _integrate_unit_ball_polynomial(
    polynomial: sp.Expr, variables: tuple[sp.Symbol, sp.Symbol, sp.Symbol]
) -> sp.Expr:
    """Integrate a polynomial over the unit ball by exact monomial moments."""

    result = sp.S.Zero
    expanded = sp.Poly(sp.expand(polynomial), *variables)
    if len(expanded.terms()) > 500:
        raise ValueError("weak-field polynomial budget exceeded")
    for powers, coefficient in expanded.terms():
        if any(power % 2 for power in powers):
            continue
        a, b, c = (power // 2 for power in powers)
        total = a + b + c
        moment = (
            2
            * sp.gamma(a + sp.Rational(1, 2))
            * sp.gamma(b + sp.Rational(1, 2))
            * sp.gamma(c + sp.Rational(1, 2))
            / (sp.gamma(total + sp.Rational(3, 2)) * (2 * total + 3))
        )
        result += coefficient * moment
    return sp.factor(result)


@lru_cache(maxsize=1)
def _weak_field_symbolic_control() -> dict[str, Any]:
    """Derive the compact weak-field energy and linearized constraint solve."""

    x, y, z = sp.symbols("x y z", real=True)
    variables = (x, y, z)
    radius_squared = x**2 + y**2 + z**2
    cutoff = (1 - radius_squared) ** 4
    toroidal = sp.Matrix([-cutoff * y, cutoff * x, 0])
    gradient = sp.Matrix(
        3,
        3,
        lambda i, j: sp.diff(toroidal[j], variables[i]),
    )
    divergence = sp.factor(sp.trace(gradient))
    gradient_norm = sp.expand(sp.trace(gradient.T * gradient))
    crossed_gradient = sp.expand(sp.trace(gradient * gradient))
    gradient_integral = _integrate_unit_ball_polynomial(gradient_norm, variables)
    crossed_integral = _integrate_unit_ball_polynomial(crossed_gradient, variables)
    expected_gradient_integral = sp.Rational(524288, 2909907) * sp.pi
    if (
        divergence != 0
        or sp.factor(gradient_integral - expected_gradient_integral) != 0
        or crossed_integral != 0
    ):
        raise ValueError("compact toroidal weak-field control changed")

    k1, k2, k3 = sp.symbols("k1 k2 k3", real=True)
    wave = sp.Matrix([k1, k2, k3])
    wave_squared = sp.expand((wave.T * wave)[0])
    york_symbol = wave_squared * sp.eye(3) + wave * wave.T / 3
    transverse = sp.factor(wave_squared)
    longitudinal = sp.factor(sp.Rational(4, 3) * wave_squared)
    if sp.factor(york_symbol.det() - transverse**2 * longitudinal) != 0:
        raise ValueError("York-symbol ellipticity control changed")

    body = {
        "perturbative_completion_class": {
            "aether_seed": "A_i=epsilon*a_i with a_i in C_c^3(R^3)",
            "unit_normal_component": "u_perp=sqrt(1+epsilon^2*a_i*a_i)",
            "gravitational_correction": (
                "q_ij=delta_ij+4*epsilon^2*phi*delta_ij+O(epsilon^4), "
                "pi^ij=epsilon^2*(L_X)^ij+O(epsilon^4)"
            ),
            "asymptotic_extension": (
                "a_i=0 and the Aether is the unit normal outside compact support"
            ),
        },
        "quadratic_static_Aether_density": {
            "core_formula": (
                "rho_2=(1/2)*(c1*partial_i(a_j)*partial_i(a_j)"
                "+c2*(partial_i(a_i))^2+c3*partial_i(a_j)*partial_j(a_i))"
            ),
            "canonical_divergence_vector": (
                "Q_i^(2)=-c2*div(a)*a_i-c3*a_j*partial_i(a_j)+c4*a_j*partial_j(a_i)"
            ),
            "full_Hamiltonian_source": "S_H^(2)=rho_2+partial_i(Q_i^(2))",
            "compact_monopole": "integral S_H^(2)=integral rho_2=E_2",
            "c4_order": "quartic_in_epsilon",
            "integration_by_parts_identity": (
                "integral partial_i(a_j)*partial_j(a_i)=integral (partial_i(a_i))^2 for compact a"
            ),
            "integrated_formula": ("E_2=(1/2)*integral[c1*|grad a|^2+(c2+c3)*(div a)^2]"),
        },
        "linearized_Hamiltonian_completion": {
            "constraint": "4*M2*Delta(phi)+S_H^(2)=0",
            "green_solution": ("phi(x)=(1/(16*pi*M2))*integral S_H^(2)(y)/|x-y| d^3y"),
            "asymptotic_coefficient": "phi=E_2/(16*pi*M2*r)+O(r^-2)",
            "ADM_boundary_energy_coefficient": "M_ADM^(2)=E_2",
        },
        "linearized_momentum_completion": {
            "constraint": "-2*partial_j(L_X)^j_i+J_i^(2)=0",
            "York_operator": "Delta_L X_i=Delta X_i+(1/3)*partial_i div(X)",
            "Fourier_symbol": "|k|^2*delta_ij+(1/3)*k_i*k_j",
            "transverse_eigenvalue": str(transverse),
            "longitudinal_eigenvalue": str(longitudinal),
            "symbol_determinant": str(sp.factor(york_symbol.det())),
            "weighted_AE_conclusion": (
                "every compact J_i^(2) has the decaying York-potential solution; L_X=O(r^-2)"
            ),
        },
        "completed_Aether_boundary_energy": {
            "formula": "M_ae=M_ADM-c14/(8*pi*G)*integral_infinity(r^a*a_a)",
            "compact_seed_boundary_term": "zero_through_order_epsilon_squared",
            "quadratic_completed_energy": "M_ae^(2)=E_2",
        },
        "positive_control_compact_toroidal_seed": {
            "field_inside_unit_ball": "a=(-y,x,0)*(1-r^2)^4",
            "field_outside_unit_ball": "a=0",
            "regularity": "C^3_compact_support",
            "divergence": str(divergence),
            "integral_gradient_norm_squared": str(gradient_integral),
            "integral_crossed_gradient": str(crossed_integral),
            "candidate_energy": "E_2=(c1/2)*(524288*pi/2909907)>0",
        },
        "negative_control_excluded_coupling": {
            "couplings": "c1=-1,c2=0,c3=0",
            "same_toroidal_seed_energy": "-262144*pi/2909907",
            "caught_by": "c1_positive_coercivity_gate",
        },
        "boundary_control_noncompact_affine_rotation": {
            "field": "a=(-y,x,0)",
            "compact_support": False,
            "integration_by_parts_identity_authorized": False,
            "lesson": "the predecessor affine witness cannot enter the compact weak-field theorem",
        },
        "proved_scope": (
            "exact second-order weak-field constraint completion and completed-energy sign; "
            "not a full nonlinear Einstein-Aether constraint solution and not a finite-amplitude "
            "energy theorem"
        ),
    }
    return {**body, "content_sha256": _sha(body)}


def _candidate_certificate(parameters: dict[str, str], symbolic: dict[str, Any]) -> dict[str, Any]:
    c1 = Fraction(parameters["c1"])
    c23 = Fraction(parameters["c2"]) + Fraction(parameters["c3"])
    if c1 <= 0 or c23 < 0:
        raise ValueError("future Aether weak-field coercivity domain changed")
    toroidal_coefficient = c1 * Fraction(262144, 2909907)
    body = {
        "parameters": parameters,
        "c1": str(c1),
        "c2_plus_c3": str(c23),
        "quadratic_energy_identity": ("E_2=(1/2)*integral[c1*|grad a|^2+(c2+c3)*(div a)^2]"),
        "coercive_gradient_coefficient": str(c1 / 2),
        "nonnegative_divergence_coefficient": str(c23 / 2),
        "strictly_positive_for_every_nonzero_compact_seed": True,
        "compact_toroidal_seed_energy_over_pi": str(toroidal_coefficient),
        "linearized_Hamiltonian_constraint_completed": True,
        "linearized_momentum_constraint_completed": True,
        "linearized_completed_boundary_energy_negative": False,
        "finite_amplitude_nonlinear_constraint_solution_proven": False,
        "finite_amplitude_negative_completed_energy_proven": False,
        "candidate_rejection_authorized": False,
        "symbolic_control_sha256": symbolic["content_sha256"],
    }
    return {**body, "content_sha256": _sha(body)}


def build_future_aether_weak_field_ae_constraint_gate(
    config: dict[str, Any], root: str | Path
) -> dict[str, Any]:
    root = Path(root).resolve()
    _validate_config(config)
    _bound_path(root, config["campaign_implementation"], "weak-field gate implementation")
    source_implementation = _bound_path(
        root, config["source_no_go_implementation"], "source no-go implementation"
    )
    source_config = _bound_json(root, config["source_no_go_config"], "source no-go config")
    source_artifact = _bound_json(
        root, config["source_no_go_artifact"], "source no-go artifact", content=True
    )
    callback_source = Path(
        inspect.getsourcefile(build_future_aether_pure_twist_ae_no_go_audit) or ""
    ).resolve()
    if callback_source != source_implementation:
        raise ValueError("source no-go implementation entrypoint changed")
    if build_future_aether_pure_twist_ae_no_go_audit(source_config, root) != source_artifact:
        raise ValueError("source no-go artifact no longer replays")
    if (
        source_artifact.get("candidate_count") != 14
        or source_artifact.get("decision_counts") != {"blocked": 14}
        or source_artifact.get("first_blocker_counts") != {SOURCE_BLOCKER: 14}
        or source_artifact.get("candidate_rejection_authorized_count") != 0
        or source_artifact.get("constraint_satisfying_negative_total_energy_datum_count") != 0
    ):
        raise ValueError("source no-go decision scope changed")

    symbolic = _weak_field_symbolic_control()
    records = []
    c23_counts: Counter[str] = Counter()
    for source in source_artifact["candidate_records"]:
        source_body = {key: value for key, value in source.items() if key != "content_sha256"}
        specialization = _specialize(source["parameters"])
        if (
            source.get("content_sha256") != _sha(source_body)
            or source.get("family_id") != TARGET_FAMILY
            or source.get("decision") != "blocked"
            or source.get("first_blocker") != SOURCE_BLOCKER
            or source.get("candidate_rejection_authorized") is not False
            or source.get("exact_specialization") != specialization
        ):
            raise ValueError("source Aether no-go record changed")
        certificate = _candidate_certificate(source["parameters"], symbolic)
        c23_counts[certificate["c2_plus_c3"]] += 1
        gates = {
            "source_action_witness_and_no_go_binding": {"status": "pass"},
            "compact_support_integration_by_parts_identity": {"status": "pass"},
            "candidate_quadratic_energy_coercivity": {
                "status": "pass_positive",
                "gradient_coefficient": certificate["coercive_gradient_coefficient"],
                "divergence_coefficient": certificate["nonnegative_divergence_coefficient"],
            },
            "candidate_Aether_sourced_linearized_Hamiltonian_constraint": {
                "status": "pass",
                "scope": "second-order conformal Green completion",
            },
            "candidate_Aether_sourced_linearized_momentum_constraint": {
                "status": "pass",
                "scope": "second-order decaying York-potential completion",
            },
            "quadratic_completed_Aether_boundary_energy": {
                "status": "pass_nonnegative",
                "negative_direction_found": False,
            },
            "finite_amplitude_nonlinear_coupled_constraint_lift": {"status": "blocked"},
            "finite_amplitude_negative_completed_boundary_energy": {"status": "blocked"},
            "observational_data_seal": {"status": "pass"},
        }
        provenance_body = {
            "candidate_id": source["candidate_id"],
            "typed_action_ir_sha256": source["typed_action_ir_sha256"],
            "source_no_go_record_sha256": source["content_sha256"],
            "source_exact_specialization_sha256": specialization["content_sha256"],
            "weak_field_certificate_sha256": certificate["content_sha256"],
            "symbolic_control_sha256": symbolic["content_sha256"],
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
            "source_no_go_record_sha256": source["content_sha256"],
            "parameters": source["parameters"],
            "exact_specialization": specialization,
            "weak_field_AE_constraint_certificate": certificate,
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
        raise ValueError("future Aether weak-field target count changed")
    record_root = _sha(
        [
            [item["candidate_id"], item["typed_action_ir_sha256"], item["content_sha256"]]
            for item in records
        ]
    )
    provenance_body = {
        "source_no_go_content_sha256": source_artifact["content_sha256"],
        "source_no_go_record_registry_root_sha256": source_artifact[
            "candidate_record_registry_root_sha256"
        ],
        "source_no_go_implementation_file_sha256": config["source_no_go_implementation"][
            "file_sha256"
        ],
        "symbolic_control_sha256": symbolic["content_sha256"],
        "candidate_record_registry_root_sha256": record_root,
        "data_eligibility": ELIGIBILITY,
    }
    body = {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": config["campaign_id"],
        "immutable_config_sha256": _sha(config),
        "source_no_go_binding": config["source_no_go_artifact"],
        "candidate_count": len(records),
        "decision_counts": {"blocked": 14},
        "first_blocker_counts": {BLOCKER: 14},
        "formal_pass_count": 0,
        "candidate_rejection_authorized_count": 0,
        "constraint_satisfying_negative_total_energy_datum_count": 0,
        "weak_field_linearized_constraint_completion_count": 14,
        "strictly_positive_compact_quadratic_energy_count": 14,
        "weak_field_negative_completed_energy_direction_count": 0,
        "finite_amplitude_nonlinear_constraint_completion_count": 0,
        "c2_plus_c3_counts": dict(sorted(c23_counts.items())),
        "symbolic_weak_field_control": symbolic,
        "candidate_record_registry_root_sha256": record_root,
        "candidate_records": records,
        "bounded_weak_field_AE_constraint_gate_completed": True,
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
            "For every one of the fourteen candidates, compact Aether seeds source explicit "
            "decaying conformal/York solutions of the constraints at the first nontrivial "
            "weak-field backreaction order. Exact integration by parts reduces the completed "
            "quadratic energy to one half the integral of c1 times the gradient norm plus "
            "c2+c3 times the divergence norm. Here c1 is 1/32 and c2+c3 is nonnegative, so "
            "every nonzero compact seed has positive—not negative—quadratic completed energy. "
            "This rules out a negative-energy branch emerging perturbatively from the vacuum, "
            "but it neither solves the full nonlinear coupled constraints nor rejects a theory. "
            "The first missing premise is now a finite-amplitude nonlinear AE constraint lift "
            "whose completed Aether boundary energy is genuinely negative."
        ),
    }
    return {**body, "content_sha256": _sha(body)}


def publish_future_aether_weak_field_ae_constraint_gate(
    config: dict[str, Any], root: str | Path, target: str | Path
) -> dict[str, Any]:
    artifact = build_future_aether_weak_field_ae_constraint_gate(config, root)
    target = Path(target).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if _load(target) != artifact:
            raise ValueError("refusing to replace divergent weak-field Aether artifact")
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
    artifact = publish_future_aether_weak_field_ae_constraint_gate(
        _load(Path(arguments.config)), arguments.root, arguments.output
    )
    print(_canonical({key: value for key, value in artifact.items() if key != "candidate_records"}))


if __name__ == "__main__":
    main()
