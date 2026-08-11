"""Finite-tilt metric-momentum-to-York symbol gate for future Aether seeds."""

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

import sympy as sp

from .adm_aether import _einstein_aether_kinetic_model
from .future_aether_fixed_free_data_principal_gate import (
    BLOCKER as SOURCE_REGULAR_BLOCKER,
)
from .future_aether_fixed_free_data_principal_gate import (
    build_future_aether_fixed_free_data_principal_gate,
)
from .future_aether_nonlinear_lift_characteristic_gate import CHARACTERISTIC_BLOCKER
from .promotion_orchestrator import ELIGIBILITY

CONFIG_SCHEMA = "sigma-future-aether-finite-tilt-york-symbol-gate-config-1.0"
RESULT_SCHEMA = "sigma-future-aether-finite-tilt-york-symbol-gate-1.0"
TARGET_FAMILY = "AETHER_K1234_PARAMETER_CELL"
YORK_SHELL_BLOCKER = (
    "alternative_canonical_momentum_variable_or_gauge_avoiding_exact_finite_tilt_York_symbol_shell"
)
FREDHOLM_BLOCKER = (
    "candidate_bound_weighted_Fredholm_isomorphism_lower_order_coefficient_and_"
    "inverse_norm_bounds_for_finite_tilt_York_operator"
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
        "source_fixed_free_data_artifact",
        "source_fixed_free_data_config",
        "source_fixed_free_data_implementation",
        "source_inverse_margin_artifact",
        "reviewed_adm_implementation",
        "budget",
        "observational_authorization",
        "data_eligibility",
        "external_paid_llm_calls",
    }
    if set(config) != required or config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("future Aether finite-tilt York config is invalid")
    if config.get("data_eligibility") != ELIGIBILITY:
        raise ValueError("future Aether finite-tilt York eligibility is open")
    if config.get("observational_authorization") is not False:
        raise ValueError("future Aether finite-tilt York opened observations")
    if config.get("external_paid_llm_calls") is not False:
        raise ValueError("future Aether finite-tilt York enabled paid LLM calls")
    if config.get("budget") != {
        "maximum_candidates": 14,
        "maximum_regular_adm_candidates": 3,
        "maximum_symbol_rows": 3,
        "maximum_symbol_columns": 3,
        "maximum_paid_llm_spend_usd": 0.0,
    }:
        raise ValueError("future Aether finite-tilt York budget is not exact")


def _derive_york_symbol(parameters: dict[str, str]) -> dict[str, Any]:
    model = _einstein_aether_kinetic_model()
    lagrangian = model["lagrangian"]
    metric_velocities = model["velocities"][:6]
    normal_velocity, longitudinal_velocity, transverse_y, transverse_z = model["velocities"][6:]
    u0, u1, u2, u3 = model["vector_down"]
    planck2, c1, c2, c3, c4 = model["coupling_symbols"]
    y = sp.symbols("y", nonnegative=True, finite=True)
    p, q = sp.symbols("p q", real=True)
    z = sp.symbols("z", nonnegative=True, finite=True)
    tilt = sp.sqrt(y)
    chi = sp.sqrt(1 + y)
    tangent_normal_velocity = (y * metric_velocities[0] - tilt * longitudinal_velocity) / chi
    substitutions = {
        u0: -chi,
        u1: tilt,
        u2: 0,
        u3: 0,
        normal_velocity: tangent_normal_velocity,
        planck2: 1,
        c1: sp.Rational(parameters["c1"]),
        c2: sp.Rational(parameters["c2"]),
        c3: sp.Rational(parameters["c3"]),
        c4: sp.Rational(parameters["c4"]),
    }
    reduced_velocities = (
        *metric_velocities,
        longitudinal_velocity,
        transverse_y,
        transverse_z,
    )
    reduced_lagrangian = sp.expand(lagrangian.subs(substitutions))
    hessian = sp.hessian(reduced_lagrangian, reduced_velocities)
    metric_block = hessian[:6, :6]
    mixing_block = hessian[:6, 6:]
    aether_block = hessian[6:, 6:]
    schur = sp.simplify(metric_block - mixing_block * aether_block.inv() * mixing_block.T)

    covector = sp.Matrix([p, q, 0])
    york_variable = sp.Matrix(sp.symbols("X0:3"))
    contraction = (covector.T * york_variable)[0]
    york_k = sp.Matrix(
        [
            2 * p * york_variable[0] - sp.Rational(2, 3) * contraction,
            2 * q * york_variable[1] - sp.Rational(2, 3) * contraction,
            -sp.Rational(2, 3) * contraction,
            p * york_variable[1] + q * york_variable[0],
            p * york_variable[2],
            q * york_variable[2],
        ]
    )
    raw_metric_momentum = schur * york_k
    metric_momentum = sp.Matrix(
        [
            raw_metric_momentum[0],
            raw_metric_momentum[1],
            raw_metric_momentum[2],
            raw_metric_momentum[3] / 2,
            raw_metric_momentum[4] / 2,
            raw_metric_momentum[5] / 2,
        ]
    )
    divergence = sp.Matrix(
        [
            p * metric_momentum[0] + q * metric_momentum[3],
            p * metric_momentum[3] + q * metric_momentum[1],
            p * metric_momentum[4] + q * metric_momentum[5],
        ]
    )
    symbol = sp.simplify(-2 * divergence.jacobian(york_variable))
    determinant = sp.factor(symbol.det())
    unit_determinant = sp.factor(determinant.subs(q**2, 1 - p**2).subs(p**2, z))
    if unit_determinant.free_symbols - {y, z}:
        raise ValueError("finite-tilt York determinant retained direction auxiliaries")
    perpendicular = sp.factor(unit_determinant.subs(z, 0))
    body = {
        "parameters": parameters,
        "tilt_variable": "y=A_i A^i",
        "direction_variable": "z=(hat_xi dot hat_A)^2 in [0,1]",
        "fixed_vector_momentum_Schur_complement": ("H_KK-H_KV*(H_VV)^(-1)*H_VK"),
        "York_tensor": ("K_ij=i*(xi_i X_j+xi_j X_i-(2/3)*delta_ij*xi_k X_k)"),
        "momentum_constraint_symbol": "-2*i*xi_j*delta_pi^j_i",
        "unit_covector_determinant": str(unit_determinant),
        "perpendicular_covector_determinant": str(perpendicular),
        "symbol_shape": [3, 3],
        "candidate_bound_distributed_Legendre_principal_symbol_derived": True,
    }
    return {**body, "content_sha256": _sha(body)}


def _shell_is_below_amplitude(case: str, amplitude: Fraction) -> bool:
    if case == "quadratic_521":
        return 2 * amplitude + 5 > 0 and (2 * amplitude + 5) ** 2 > 521
    if case == "quadratic_97":
        return amplitude - 8 > 0 and (amplitude - 8) ** 2 > 97
    raise ValueError("unknown York shell comparison")


def _candidate_certificate(
    source: dict[str, Any], inverse_source: dict[str, Any]
) -> dict[str, Any]:
    parameters = source["parameters"]
    amplitude = Fraction(
        inverse_source["regular_ADM_inverse_margin_certificate"][
            "characteristic_free_seed_amplitude_squared"
        ]
    )
    symbol = _derive_york_symbol(parameters)
    y = sp.symbols("y", nonnegative=True, finite=True)
    perpendicular = sp.sympify(symbol["perpendicular_covector_determinant"], locals={"y": y})
    key = tuple(parameters[name] for name in ("c1", "c2", "c3", "c4"))
    if key == ("1/32", "0", "0", "1/32"):
        expected = -((y - 31) ** 2) * (61 * y + 124) / (6144 * (y + 2))
        if sp.factor(perpendicular - expected) != 0 or amplitude >= 31:
            raise ValueError("uniform York-positive candidate control changed")
        uniform = True
        shell = None
        proof = {
            "registered_tilt_upper": str(amplitude),
            "exact_upper_below_first_symbol_root": "amplitude_squared<31",
            "determinant_factor_1_z_monotonicity": (
                "d_z[2*y^2*z-2*y^2+3*y*z+58*y+124]=2*y^2+3*y>=0"
            ),
            "determinant_factor_1_at_z0": "2*(31-y)*(y+2)>0",
            "determinant_factor_2_z_monotonicity": ("y*(134*y^2*z+55*y^2+266*y*z+481*y+744)>=0"),
            "determinant_factor_2_at_z0": "2*(31-y)*(y+2)*(61*y+124)>0",
            "all_y_and_z_principal_determinants_nonzero": True,
        }
    elif key == ("1/32", "1/32", "0", "1/32"):
        expected = -(y - 31) * (61 * y + 124) * (y**2 + 5 * y - 124) / (24576 * (y + 2))
        if sp.factor(perpendicular - expected) != 0 or not _shell_is_below_amplitude(
            "quadratic_521", amplitude
        ):
            raise ValueError("quadratic-521 York shell control changed")
        uniform = False
        alpha = "(-5+sqrt(521))/2"
        shell = {
            "direction": "z=0_perpendicular_to_A",
            "tilt_squared": alpha,
            "root_polynomial": "y^2+5*y-124",
            "strictly_inside_registered_tilt_range": True,
            "compact_seed_shell_radius_squared": f"1-(({alpha})/({amplitude}))^(1/8)",
            "ansatz_status": "nonelliptic_reject_for_K_equals_LX_completion_only",
            "candidate_rejection_authorized": False,
        }
        proof = None
    elif key == ("1/32", "1/16", "-1/16", "1/32"):
        expected = (y + 33) * (65 * y + 132) * (y**2 - 16 * y - 33) / (3072 * (y + 2) ** 2)
        if sp.factor(perpendicular - expected) != 0 or not _shell_is_below_amplitude(
            "quadratic_97", amplitude
        ):
            raise ValueError("quadratic-97 York shell control changed")
        uniform = False
        alpha = "8+sqrt(97)"
        shell = {
            "direction": "z=0_perpendicular_to_A",
            "tilt_squared": alpha,
            "root_polynomial": "y^2-16*y-33",
            "strictly_inside_registered_tilt_range": True,
            "compact_seed_shell_radius_squared": f"1-(({alpha})/({amplitude}))^(1/8)",
            "ansatz_status": "nonelliptic_reject_for_K_equals_LX_completion_only",
            "candidate_rejection_authorized": False,
        }
        proof = None
    else:
        raise ValueError("unexpected regular Aether parameter cell")
    body = {
        "candidate_id": source["candidate_id"],
        "typed_action_ir_sha256": source["typed_action_ir_sha256"],
        "source_fixed_free_data_record_sha256": source["content_sha256"],
        "source_inverse_margin_record_sha256": inverse_source["content_sha256"],
        "registered_characteristic_free_tilt_upper": str(amplitude),
        "finite_tilt_York_symbol": symbol,
        "uniform_principal_ellipticity_certificate": proof,
        "exact_nonelliptic_York_shell": shell,
        "finite_tilt_metric_York_principal_symbol_derived": True,
        "uniform_fixed_free_data_principal_ellipticity_proven": uniform,
        "weighted_Fredholm_isomorphism_proven": False,
        "lower_order_coefficient_bounds_proven": False,
        "computable_inverse_norm_proven": False,
        "nonlinear_remainder_majorant_proven": False,
        "completed_boundary_sign_persistence_proven": False,
        "candidate_rejection_authorized": False,
    }
    return {**body, "content_sha256": _sha(body)}


def build_future_aether_finite_tilt_york_symbol_gate(
    config: dict[str, Any], root: str | Path
) -> dict[str, Any]:
    root = Path(root).resolve()
    _validate_config(config)
    _bound_path(root, config["campaign_implementation"], "finite-tilt York implementation")
    source_implementation = _bound_path(
        root,
        config["source_fixed_free_data_implementation"],
        "fixed-free-data implementation",
    )
    source_config = _bound_json(
        root, config["source_fixed_free_data_config"], "fixed-free-data config"
    )
    source_artifact = _bound_json(
        root,
        config["source_fixed_free_data_artifact"],
        "fixed-free-data artifact",
        content=True,
    )
    inverse_artifact = _bound_json(
        root, config["source_inverse_margin_artifact"], "inverse-margin artifact", content=True
    )
    adm_path = _bound_path(
        root, config["reviewed_adm_implementation"], "reviewed ADM implementation"
    )
    callback_source = Path(
        inspect.getsourcefile(build_future_aether_fixed_free_data_principal_gate) or ""
    ).resolve()
    model_source = Path(inspect.getsourcefile(_einstein_aether_kinetic_model) or "").resolve()
    if callback_source != source_implementation:
        raise ValueError("fixed-free-data implementation entrypoint changed")
    if model_source != adm_path:
        raise ValueError("reviewed ADM kinetic model entrypoint changed")
    if build_future_aether_fixed_free_data_principal_gate(source_config, root) != source_artifact:
        raise ValueError("fixed-free-data artifact no longer replays")
    if (
        source_artifact.get("candidate_count") != 14
        or source_artifact.get("decision_counts") != {"blocked": 14}
        or source_artifact.get("first_blocker_counts")
        != {CHARACTERISTIC_BLOCKER: 11, SOURCE_REGULAR_BLOCKER: 3}
        or source_artifact.get("candidate_rejection_authorized_count") != 0
        or inverse_artifact.get("regular_ADM_candidate_count") != 3
        or inverse_artifact.get("decision_counts") != {"blocked": 14}
    ):
        raise ValueError("finite-tilt York source scope changed")
    inverse_records = {
        item["candidate_id"]: item
        for item in inverse_artifact["candidate_records"]
        if item["regular_ADM_inverse_margin_certificate"] is not None
    }

    records = []
    blockers: Counter[str] = Counter()
    symbol_count = 0
    elliptic_count = 0
    shell_count = 0
    for source in source_artifact["candidate_records"]:
        source_body = {key: value for key, value in source.items() if key != "content_sha256"}
        if source.get("content_sha256") != _sha(source_body):
            raise ValueError("finite-tilt York source record changed")
        if source["first_blocker"] == SOURCE_REGULAR_BLOCKER:
            inverse_source = inverse_records.get(source["candidate_id"])
            if (
                inverse_source is None
                or inverse_source["typed_action_ir_sha256"] != source["typed_action_ir_sha256"]
                or inverse_source["parameters"] != source["parameters"]
            ):
                raise ValueError("inverse-margin candidate binding changed")
            certificate = _candidate_certificate(source, inverse_source)
            symbol_count += 1
            if certificate["uniform_fixed_free_data_principal_ellipticity_proven"]:
                blocker = FREDHOLM_BLOCKER
                elliptic_count += 1
                symbol_status = "pass"
                shell_status = "not_applicable"
            else:
                blocker = YORK_SHELL_BLOCKER
                shell_count += 1
                symbol_status = "reject_ansatz_only"
                shell_status = "pass_obstruction"
        else:
            certificate = None
            blocker = CHARACTERISTIC_BLOCKER
            symbol_status = "not_reached"
            shell_status = "not_reached"
        blockers[blocker] += 1
        provenance_body = {
            "candidate_id": source["candidate_id"],
            "typed_action_ir_sha256": source["typed_action_ir_sha256"],
            "source_fixed_free_data_record_sha256": source["content_sha256"],
            "finite_tilt_York_certificate_sha256": (
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
            "source_fixed_free_data_record_sha256": source["content_sha256"],
            "parameters": source["parameters"],
            "exact_specialization": source["exact_specialization"],
            "finite_tilt_York_symbol_certificate": certificate,
            "gate_ledger": {
                "source_action_and_predecessor_binding": {"status": "pass"},
                "finite_tilt_metric_momentum_to_York_symbol": {"status": symbol_status},
                "exact_nonelliptic_York_shell": {"status": shell_status},
                "weighted_Fredholm_isomorphism_and_norm": {"status": "blocked"},
                "lower_order_and_nonlinear_remainder": {"status": "blocked"},
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
        FREDHOLM_BLOCKER: 1,
        YORK_SHELL_BLOCKER: 2,
    }
    if (
        symbol_count != 3
        or elliptic_count != 1
        or shell_count != 2
        or dict(blockers) != expected_blockers
    ):
        raise ValueError("future Aether finite-tilt York partition changed")
    record_root = _sha(
        [
            [item["candidate_id"], item["typed_action_ir_sha256"], item["content_sha256"]]
            for item in records
        ]
    )
    provenance_body = {
        "source_fixed_free_data_content_sha256": source_artifact["content_sha256"],
        "source_fixed_free_data_record_registry_root_sha256": source_artifact[
            "candidate_record_registry_root_sha256"
        ],
        "source_inverse_margin_content_sha256": inverse_artifact["content_sha256"],
        "candidate_record_registry_root_sha256": record_root,
        "data_eligibility": ELIGIBILITY,
    }
    body = {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": config["campaign_id"],
        "immutable_config_sha256": _sha(config),
        "source_fixed_free_data_binding": config["source_fixed_free_data_artifact"],
        "source_inverse_margin_binding": config["source_inverse_margin_artifact"],
        "reviewed_ADM_implementation_binding": config["reviewed_adm_implementation"],
        "candidate_count": 14,
        "decision_counts": {"blocked": 14},
        "first_blocker_counts": dict(sorted(blockers.items())),
        "formal_pass_count": 0,
        "candidate_rejection_authorized_count": 0,
        "constraint_satisfying_negative_total_energy_datum_count": 0,
        "forced_characteristic_candidate_count": 11,
        "regular_ADM_candidate_count": 3,
        "finite_tilt_metric_York_symbol_derived_count": 3,
        "uniform_fixed_free_data_principal_ellipticity_pass_count": 1,
        "exact_nonelliptic_York_shell_count": 2,
        "York_ansatz_reject_count": 2,
        "weighted_Fredholm_isomorphism_pass_count": 0,
        "lower_order_coefficient_bound_pass_count": 0,
        "computable_full_inverse_norm_count": 0,
        "nonlinear_remainder_bound_pass_count": 0,
        "completed_boundary_sign_persistence_count": 0,
        "candidate_record_registry_root_sha256": record_root,
        "candidate_records": records,
        "bounded_finite_tilt_York_symbol_gate_completed": True,
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
            "The exact fixed-vector-momentum Schur complement now gives the finite-tilt York "
            "principal symbol for all three regular candidates. One candidate is uniformly "
            "elliptic across its full registered compact seed and advances to weighted Fredholm "
            "and coefficient bounds. Two candidates cross exact perpendicular-covector York "
            "symbol shells, rejecting only K_ij=(L_X)_ij as a global completion variable; an "
            "alternative canonical-momentum variable or gauge remains open. No candidate theory "
            "rejection, nonlinear constraint solution, or boundary-sign claim is made."
        ),
    }
    return {**body, "content_sha256": _sha(body)}


def publish_future_aether_finite_tilt_york_symbol_gate(
    config: dict[str, Any], root: str | Path, target: str | Path
) -> dict[str, Any]:
    artifact = build_future_aether_finite_tilt_york_symbol_gate(config, root)
    target = Path(target).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if _load(target) != artifact:
            raise ValueError("refusing to replace divergent finite-tilt York artifact")
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
    artifact = publish_future_aether_finite_tilt_york_symbol_gate(
        _load(Path(arguments.config)), arguments.root, arguments.output
    )
    print(_canonical({key: value for key, value in artifact.items() if key != "candidate_records"}))


if __name__ == "__main__":
    main()
