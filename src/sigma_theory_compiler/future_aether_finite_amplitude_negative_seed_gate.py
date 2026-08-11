"""Finite-amplitude negative-source gate for future Aether candidates."""

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
from .future_aether_weak_field_ae_constraint_gate import (
    build_future_aether_weak_field_ae_constraint_gate,
)
from .promotion_orchestrator import ELIGIBILITY

CONFIG_SCHEMA = "sigma-future-aether-finite-amplitude-negative-seed-gate-config-1.0"
RESULT_SCHEMA = "sigma-future-aether-finite-amplitude-negative-seed-gate-1.0"
TARGET_FAMILY = "AETHER_K1234_PARAMETER_CELL"
SOURCE_BLOCKER = (
    "finite_amplitude_candidate_bound_nonlinear_AE_coupled_constraint_solution_"
    "with_negative_completed_boundary_energy_beyond_positive_weak_field_quadratic_regime"
)
BLOCKER = (
    "nonlinear_Einstein_Aether_constraint_lift_of_explicit_compact_negative_source_seed_"
    "with_sign_preserving_completed_boundary_energy"
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
        "source_weak_field_artifact",
        "source_weak_field_config",
        "source_weak_field_implementation",
        "budget",
        "observational_authorization",
        "data_eligibility",
        "external_paid_llm_calls",
    }
    if set(config) != required or config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("future Aether finite-amplitude gate config is invalid")
    if config.get("data_eligibility") != ELIGIBILITY:
        raise ValueError("future Aether finite-amplitude gate eligibility is open")
    if config.get("observational_authorization") is not False:
        raise ValueError("future Aether finite-amplitude gate opened observations")
    if config.get("external_paid_llm_calls") is not False:
        raise ValueError("future Aether finite-amplitude gate enabled paid LLM calls")
    if config.get("budget") != {
        "maximum_candidates": 14,
        "spatial_dimension": 3,
        "seed_amplitude_squared": 100,
        "maximum_symbolic_polynomial_terms": 700,
        "maximum_paid_llm_spend_usd": 0.0,
    }:
        raise ValueError("future Aether finite-amplitude gate budget is not exact")


def _integrate_unit_ball_polynomial(
    polynomial: sp.Expr, variables: tuple[sp.Symbol, sp.Symbol, sp.Symbol]
) -> sp.Expr:
    result = sp.S.Zero
    expanded = sp.Poly(sp.expand(polynomial), *variables)
    if len(expanded.terms()) > 700:
        raise ValueError("finite-amplitude polynomial budget exceeded")
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
def _finite_amplitude_symbolic_control() -> dict[str, Any]:
    x, y, z = sp.symbols("x y z", real=True)
    variables = (x, y, z)
    radius_squared = x**2 + y**2 + z**2
    profile = (1 - radius_squared) ** 4
    gradient = sp.Matrix([sp.diff(profile, coordinate) for coordinate in variables])
    gradient_norm = sp.expand((gradient.T * gradient)[0])
    axial_gradient = sp.expand(gradient[0] ** 2)
    acceleration_weight = sp.expand(profile**2 * gradient[0] ** 2)
    gradient_integral = _integrate_unit_ball_polynomial(gradient_norm, variables)
    axial_integral = _integrate_unit_ball_polynomial(axial_gradient, variables)
    acceleration_integral = _integrate_unit_ball_polynomial(acceleration_weight, variables)
    expected = (
        sp.Rational(262144, 255255) * sp.pi,
        sp.Rational(262144, 765765) * sp.pi,
        sp.Rational(8589934592, 148767396525) * sp.pi,
    )
    if any(
        sp.factor(actual - target) != 0
        for actual, target in zip(
            (gradient_integral, axial_integral, acceleration_integral),
            expected,
            strict=True,
        )
    ):
        raise ValueError("finite-amplitude compact seed control changed")
    worst_ratio = sp.factor((gradient_integral + 8 * axial_integral) / acceleration_integral)
    worst_bracket = sp.factor(gradient_integral + 8 * axial_integral - 100 * acceleration_integral)
    worst_energy_upper = sp.factor(sp.Rational(25, 16) * worst_bracket)
    if worst_ratio >= 100 or worst_energy_upper >= 0:
        raise ValueError("finite-amplitude worst-candidate negativity control changed")
    body = {
        "compact_seed": {
            "inside_unit_ball": "A_i=10*(1-r^2)^4*delta_i1",
            "outside_unit_ball": "A_i=0",
            "regularity": "C^3_compact_support",
            "maximum_tilt_squared": "100",
            "asymptotic_Aether": "unit_normal_outside_compact_support",
        },
        "exact_static_source_monopole": {
            "formula": (
                "E_static=(1/2)*integral[c1*t^2*|grad f|^2/(1+t^2*f^2)"
                "+(c2+c3)*t^2*(partial_1 f)^2-c4*t^4*f^2*(partial_1 f)^2]"
            ),
            "canonical_divergence_integral": "zero_for_compact_seed",
            "rigorous_upper_bound": (
                "E_static<=(t^2/2)*[c1*I_grad+(c2+c3)*I_axis-c4*t^2*I_acceleration]"
            ),
            "I_grad": str(gradient_integral),
            "I_axis": str(axial_integral),
            "I_acceleration": str(acceleration_integral),
            "worst_coupling_ratio": "(c2+c3)/c1=8",
            "worst_negativity_threshold_t_squared": str(worst_ratio),
            "chosen_t_squared": "100",
            "worst_energy_upper_bound": str(worst_energy_upper),
            "strictly_negative_for_all_candidate_cells": True,
        },
        "frozen_source_constraint_completion": {
            "Hamiltonian": ("4*M2*Delta(phi)+S_H[A;delta,0]=0 solved by the Newton potential"),
            "momentum": (
                "-2*partial_j(L_X)^j_i+J_i[A;delta,0]=0 solved by the decaying York potential"
            ),
            "source_support": "compact",
            "Aether_boundary_term": "zero_outside_compact_support",
            "linearized_completed_boundary_energy": (
                "equal_to_exact_frozen_static_source_monopole_and strictly negative"
            ),
        },
        "positive_control_weak_amplitude": {
            "t_squared": "1/2",
            "role": "inside predecessor positive quadratic/coercive regime",
        },
        "negative_control_removed_acceleration_term": {
            "mutation": "set c4=0",
            "conclusion": "the displayed quartic negativity proof is unavailable",
        },
        "scope": (
            "exact compact finite-amplitude Aether source and rigorous negative monopole; "
            "the conformal/York completion holds only for the frozen-source linearized "
            "Einstein constraints, not the full nonlinear Einstein-Aether constraint map"
        ),
    }
    return {**body, "content_sha256": _sha(body)}


def _candidate_certificate(parameters: dict[str, str], symbolic: dict[str, Any]) -> dict[str, Any]:
    c1 = Fraction(parameters["c1"])
    c23 = Fraction(parameters["c2"]) + Fraction(parameters["c3"])
    c4 = Fraction(parameters["c4"])
    if c1 != Fraction(1, 32) or c4 != c1 or not 0 <= c23 <= Fraction(1, 4):
        raise ValueError("future Aether finite-amplitude coupling domain changed")
    i_grad = Fraction(262144, 255255)
    i_axis = Fraction(262144, 765765)
    i_acceleration = Fraction(8589934592, 148767396525)
    upper_over_pi = 50 * (c1 * i_grad + c23 * i_axis - 100 * c4 * i_acceleration)
    if upper_over_pi >= 0:
        raise ValueError("candidate compact seed lost strict negative upper bound")
    body = {
        "parameters": parameters,
        "c1": str(c1),
        "c2_plus_c3": str(c23),
        "c4": str(c4),
        "seed_amplitude_squared": "100",
        "static_source_energy_upper_bound_over_pi": str(upper_over_pi),
        "exact_static_source_monopole_negative": True,
        "compact_asymptotically_Euclidean_Aether_seed": True,
        "frozen_source_linearized_Hamiltonian_constraint_completed": True,
        "frozen_source_linearized_momentum_constraint_completed": True,
        "negative_linearized_completed_boundary_energy_coefficient": True,
        "full_nonlinear_Einstein_Aether_constraint_solution_proven": False,
        "sign_preserving_nonlinear_boundary_completion_proven": False,
        "constraint_satisfying_negative_total_energy_datum_proven": False,
        "candidate_rejection_authorized": False,
        "symbolic_control_sha256": symbolic["content_sha256"],
    }
    return {**body, "content_sha256": _sha(body)}


def build_future_aether_finite_amplitude_negative_seed_gate(
    config: dict[str, Any], root: str | Path
) -> dict[str, Any]:
    root = Path(root).resolve()
    _validate_config(config)
    _bound_path(root, config["campaign_implementation"], "finite-amplitude implementation")
    source_implementation = _bound_path(
        root,
        config["source_weak_field_implementation"],
        "source weak-field implementation",
    )
    source_config = _bound_json(
        root, config["source_weak_field_config"], "source weak-field config"
    )
    source_artifact = _bound_json(
        root,
        config["source_weak_field_artifact"],
        "source weak-field artifact",
        content=True,
    )
    callback_source = Path(
        inspect.getsourcefile(build_future_aether_weak_field_ae_constraint_gate) or ""
    ).resolve()
    if callback_source != source_implementation:
        raise ValueError("source weak-field implementation entrypoint changed")
    if build_future_aether_weak_field_ae_constraint_gate(source_config, root) != source_artifact:
        raise ValueError("source weak-field artifact no longer replays")
    if (
        source_artifact.get("candidate_count") != 14
        or source_artifact.get("decision_counts") != {"blocked": 14}
        or source_artifact.get("first_blocker_counts") != {SOURCE_BLOCKER: 14}
        or source_artifact.get("strictly_positive_compact_quadratic_energy_count") != 14
        or source_artifact.get("weak_field_negative_completed_energy_direction_count") != 0
        or source_artifact.get("candidate_rejection_authorized_count") != 0
    ):
        raise ValueError("source weak-field decision scope changed")

    symbolic = _finite_amplitude_symbolic_control()
    records = []
    upper_counts: Counter[str] = Counter()
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
            raise ValueError("source weak-field candidate record changed")
        certificate = _candidate_certificate(source["parameters"], symbolic)
        upper_counts[certificate["static_source_energy_upper_bound_over_pi"]] += 1
        gates = {
            "source_action_and_weak_field_binding": {"status": "pass"},
            "explicit_compact_finite_amplitude_Aether_seed": {"status": "pass"},
            "exact_static_source_monopole": {
                "status": "pass_negative",
                "upper_bound_over_pi": certificate["static_source_energy_upper_bound_over_pi"],
            },
            "frozen_source_linearized_Hamiltonian_constraint": {"status": "pass"},
            "frozen_source_linearized_momentum_constraint": {"status": "pass"},
            "negative_linearized_completed_boundary_energy": {"status": "pass"},
            "full_nonlinear_Einstein_Aether_constraint_lift": {"status": "blocked"},
            "sign_preserving_completed_boundary_energy": {"status": "blocked"},
            "observational_data_seal": {"status": "pass"},
        }
        provenance_body = {
            "candidate_id": source["candidate_id"],
            "typed_action_ir_sha256": source["typed_action_ir_sha256"],
            "source_weak_field_record_sha256": source["content_sha256"],
            "source_exact_specialization_sha256": specialization["content_sha256"],
            "finite_amplitude_certificate_sha256": certificate["content_sha256"],
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
            "source_weak_field_record_sha256": source["content_sha256"],
            "parameters": source["parameters"],
            "exact_specialization": specialization,
            "finite_amplitude_negative_seed_certificate": certificate,
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
        raise ValueError("future Aether finite-amplitude target count changed")
    record_root = _sha(
        [
            [item["candidate_id"], item["typed_action_ir_sha256"], item["content_sha256"]]
            for item in records
        ]
    )
    provenance_body = {
        "source_weak_field_content_sha256": source_artifact["content_sha256"],
        "source_weak_field_record_registry_root_sha256": source_artifact[
            "candidate_record_registry_root_sha256"
        ],
        "source_weak_field_implementation_file_sha256": config["source_weak_field_implementation"][
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
        "source_weak_field_binding": config["source_weak_field_artifact"],
        "candidate_count": len(records),
        "decision_counts": {"blocked": 14},
        "first_blocker_counts": {BLOCKER: 14},
        "formal_pass_count": 0,
        "candidate_rejection_authorized_count": 0,
        "constraint_satisfying_negative_total_energy_datum_count": 0,
        "compact_finite_amplitude_Aether_seed_count": 14,
        "exact_negative_static_source_monopole_count": 14,
        "frozen_source_linearized_constraint_completion_count": 14,
        "negative_linearized_completed_boundary_energy_coefficient_count": 14,
        "full_nonlinear_constraint_completion_count": 0,
        "sign_preserving_nonlinear_boundary_completion_count": 0,
        "static_source_energy_upper_bound_over_pi_counts": dict(sorted(upper_counts.items())),
        "symbolic_finite_amplitude_control": symbolic,
        "candidate_record_registry_root_sha256": record_root,
        "candidate_records": records,
        "bounded_finite_amplitude_negative_seed_gate_completed": True,
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
            "Every candidate now has the same explicit compact C3 finite-amplitude Aether "
            "seed. Exact candidate-specialized bounds prove its full static flat-source "
            "Hamiltonian monopole is strictly negative, including the canonical divergence, "
            "and the compact source admits decaying conformal/York solutions of the frozen-source "
            "linearized Einstein constraints with a negative completed boundary coefficient. "
            "This is not yet constraint-satisfying nonlinear Einstein-Aether data: metric and "
            "Aether source terms must be solved together and the negative boundary sign must "
            "survive that lift. All candidates therefore remain blocked, not rejected."
        ),
    }
    return {**body, "content_sha256": _sha(body)}


def publish_future_aether_finite_amplitude_negative_seed_gate(
    config: dict[str, Any], root: str | Path, target: str | Path
) -> dict[str, Any]:
    artifact = build_future_aether_finite_amplitude_negative_seed_gate(config, root)
    target = Path(target).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if _load(target) != artifact:
            raise ValueError("refusing to replace divergent finite-amplitude Aether artifact")
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
    artifact = publish_future_aether_finite_amplitude_negative_seed_gate(
        _load(Path(arguments.config)), arguments.root, arguments.output
    )
    print(_canonical({key: value for key, value in artifact.items() if key != "candidate_records"}))


if __name__ == "__main__":
    main()
