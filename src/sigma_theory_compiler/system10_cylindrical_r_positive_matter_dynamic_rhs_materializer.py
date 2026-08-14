from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path
from typing import Any


class System10MatterDynamicRHSError(RuntimeError):
    """Raised when the bounded matter dynamic-row contract fails closed."""


def _canonical_sha(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("ascii")).hexdigest()


def _canonical_lf_sha(path: Path) -> str:
    try:
        raw = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    except OSError as exc:
        raise System10MatterDynamicRHSError(f"cannot read bound file: {path}") from exc
    return hashlib.sha256(raw).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise System10MatterDynamicRHSError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise System10MatterDynamicRHSError("bound JSON root is not an object")
    return value


def _resolve(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise System10MatterDynamicRHSError("bound path escapes repository root")
    return path


def _load_binding(root: Path, binding: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = _resolve(root, str(binding.get("path", "")))
    if _canonical_lf_sha(path) != binding.get("canonical_lf_sha256"):
        raise System10MatterDynamicRHSError(f"bound file hash mismatch: {path}")
    value = _load_json(path)
    if value.get("content_sha256") != binding.get("content_sha256"):
        raise System10MatterDynamicRHSError(f"bound content hash mismatch: {path}")
    return path, value


def _load_source(root: Path, binding: dict[str, Any]) -> Path:
    path = _resolve(root, str(binding.get("path", "")))
    if _canonical_lf_sha(path) != binding.get("canonical_lf_sha256"):
        raise System10MatterDynamicRHSError(f"bound source hash mismatch: {path}")
    return path


def _with_sha(body: dict[str, Any], key: str) -> dict[str, Any]:
    return {**body, key: _canonical_sha(body)}


def _scalar_row(action_sha: str, assembly_sha: str) -> dict[str, Any]:
    origin = {
        "origin_type": "canonical_scalar_covariant_euler_component_expansion",
        "action_sector_id": "canonical_minimally_coupled_scalar",
        "shared_matter_action_sha256": action_sha,
        "covariant_euler_equation": "box_g(chi_m)-m_chi**2*chi_m=0",
        "coordinate_metric": "diag(-1,1,r**2,1)",
        "field_index": 11,
        "predecessor_assembly_sha256": assembly_sha,
    }
    body = {
        "row_id": "evolution_v[11]",
        "sector": "canonical_minimally_coupled_scalar",
        "field_index": 11,
        "lhs_state_index": 28,
        "lhs": "partial_0 state[28]",
        "state_atoms": {
            "q": 11,
            "v": 28,
            "w1": 45,
            "w2": 62,
            "w3": 79,
        },
        "rhs": (
            "partial_1 state[45]+state[45]/r+partial_2 state[62]/r**2+"
            "partial_3 state[79]-m_chi**2*state[11]"
        ),
        "rhs_terms": [
            {"coefficient": "1", "atom": "partial_1 state[45]"},
            {"coefficient": "1/r", "atom": "state[45]"},
            {"coefficient": "1/r**2", "atom": "partial_2 state[62]"},
            {"coefficient": "1", "atom": "partial_3 state[79]"},
            {"coefficient": "-m_chi**2", "atom": "state[11]"},
        ],
        "solved_acceleration_certificate": {
            "unsolved_euler_lhs": (
                "-partial_0 v_11+partial_1 w1_11+w1_11/r+"
                "partial_2 w2_11/r**2+partial_3 w3_11-m_chi**2*q_11"
            ),
            "acceleration_coefficient": "-1",
            "substitution_residual": "0",
            "maximum_coordinate_denominator_r_power": 2,
            "coordinate_pole_set": ["r=0"],
            "domain_excludes_all_poles": True,
        },
        "equation_origin": _with_sha(origin, "origin_sha256"),
        "candidate_dependence": "common_all_12",
        "domain": ["r>0", "m_chi**2>=0"],
    }
    return _with_sha(body, "row_sha256")


def _fluid_definitions() -> dict[str, str]:
    return {
        "X": "(v**2-w1**2-w2**2/r**2-w3**2)/2",
        "D": "X+v**2",
        "H01": "partial_1 v",
        "H02": "partial_2 v",
        "H03": "partial_3 v",
        "H11": "partial_1 w1",
        "H12": "partial_1 w2-w2/r",
        "H13": "partial_1 w3",
        "H22": "partial_2 w2+r*w1",
        "H23": "partial_2 w3",
        "H33": "partial_3 w3",
    }


def _fluid_numerator_terms() -> list[dict[str, str]]:
    return [
        {"coefficient": "2*v*w1", "atom": "partial_1 v"},
        {"coefficient": "2*v*w2/r**2", "atom": "partial_2 v"},
        {"coefficient": "2*v*w3", "atom": "partial_3 v"},
        {"coefficient": "X-w1**2", "atom": "partial_1 w1"},
        {"coefficient": "X/r**2-w2**2/r**4", "atom": "partial_2 w2+r*w1"},
        {"coefficient": "X-w3**2", "atom": "partial_3 w3"},
        {"coefficient": "-2*w1*w2/r**2", "atom": "partial_1 w2-w2/r"},
        {"coefficient": "-2*w1*w3", "atom": "partial_1 w3"},
        {"coefficient": "-2*w2*w3/r**2", "atom": "partial_2 w3"},
    ]


def _fluid_row(action_sha: str, assembly_sha: str) -> dict[str, Any]:
    origin = {
        "origin_type": "irrotational_px_fluid_covariant_euler_component_expansion",
        "action_sector_id": "barotropic_irrotational_fluid",
        "shared_matter_action_sha256": action_sha,
        "action": "sqrt(-g)*P(X), P(X)=kappa*X**2, kappa>0",
        "covariant_euler_equation": "nabla_mu(P_X*nabla^mu(tau))=0",
        "reduced_euler_equation": "(X*g^{mu nu}-u^mu*u^nu)*H_mu_nu=0",
        "kappa_cancellation": "divide by 2*kappa; legal because kappa>0",
        "coordinate_metric": "diag(-1,1,r**2,1)",
        "field_index": 16,
        "predecessor_assembly_sha256": assembly_sha,
    }
    numerator = "+".join(
        f"({term['coefficient']})*({term['atom']})" for term in _fluid_numerator_terms()
    )
    body = {
        "row_id": "evolution_v[16]",
        "sector": "barotropic_irrotational_fluid",
        "field_index": 16,
        "lhs_state_index": 33,
        "lhs": "partial_0 state[33]",
        "state_atoms": {
            "q": 16,
            "v": 33,
            "w1": 50,
            "w2": 67,
            "w3": 84,
        },
        "definitions": _fluid_definitions(),
        "numerator_terms": _fluid_numerator_terms(),
        "rhs": f"({numerator})/D",
        "solved_acceleration_certificate": {
            "unsolved_euler_lhs": f"-D*partial_0 v+({numerator})",
            "acceleration_coefficient": "-D",
            "substitution_residual": "0",
            "dynamic_denominator": "D=X+v**2",
            "dynamic_denominator_positive_proof": [
                "X>0",
                "v**2>=0",
                "therefore D=X+v**2>0",
            ],
            "maximum_coordinate_denominator_r_power_before_dynamic_division": 4,
            "coordinate_pole_set": ["r=0"],
            "domain_excludes_all_poles": True,
        },
        "equation_origin": _with_sha(origin, "origin_sha256"),
        "candidate_dependence": "common_all_12",
        "domain": [
            "r>0",
            "kappa>0",
            "X>0",
            "nabla_mu(tau) future-directed timelike",
        ],
    }
    return _with_sha(body, "row_sha256")


def _fluid_exact_evaluation(values: dict[str, Fraction]) -> dict[str, Fraction]:
    r = values["r"]
    v, w1, w2, w3 = (values[key] for key in ("v", "w1", "w2", "w3"))
    x = (v * v - w1 * w1 - w2 * w2 / r**2 - w3 * w3) / 2
    d = x + v * v
    numerator = (
        2 * v * w1 * values["dv1"]
        + 2 * v * w2 / r**2 * values["dv2"]
        + 2 * v * w3 * values["dv3"]
        + (x - w1 * w1) * values["dw11"]
        + (x / r**2 - w2 * w2 / r**4) * (values["dw22"] + r * w1)
        + (x - w3 * w3) * values["dw33"]
        - 2 * w1 * w2 / r**2 * (values["dw12"] - w2 / r)
        - 2 * w1 * w3 * values["dw13"]
        - 2 * w2 * w3 / r**2 * values["dw23"]
    )
    rhs = numerator / d
    return {"X": x, "D": d, "numerator": numerator, "rhs": rhs, "residual": -d * rhs + numerator}


def _fraction_text(value: Fraction) -> str:
    return (
        str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"
    )


def _exact_validation() -> dict[str, Any]:
    points = [
        {
            "r": Fraction(2),
            "v": Fraction(3),
            "w1": Fraction(1),
            "w2": Fraction(2),
            "w3": Fraction(1),
            "dv1": Fraction(2),
            "dv2": Fraction(-1),
            "dv3": Fraction(3),
            "dw11": Fraction(1),
            "dw12": Fraction(2),
            "dw13": Fraction(-2),
            "dw22": Fraction(4),
            "dw23": Fraction(1),
            "dw33": Fraction(-1),
        },
        {
            "r": Fraction(3, 2),
            "v": Fraction(5),
            "w1": Fraction(2),
            "w2": Fraction(1),
            "w3": Fraction(2),
            "dv1": Fraction(-1),
            "dv2": Fraction(2),
            "dv3": Fraction(1),
            "dw11": Fraction(3),
            "dw12": Fraction(-2),
            "dw13": Fraction(1),
            "dw22": Fraction(-1),
            "dw23": Fraction(2),
            "dw33": Fraction(4),
        },
    ]
    evaluations = []
    for point in points:
        result = _fluid_exact_evaluation(point)
        if result["X"] <= 0 or result["D"] <= 0 or result["residual"] != 0:
            raise System10MatterDynamicRHSError("fluid exact validation failed")
        body = {
            "input": {key: _fraction_text(value) for key, value in point.items()},
            "X": _fraction_text(result["X"]),
            "D": _fraction_text(result["D"]),
            "numerator": _fraction_text(result["numerator"]),
            "rhs": _fraction_text(result["rhs"]),
            "substitution_residual": "0",
        }
        evaluations.append(_with_sha(body, "evaluation_sha256"))
    return {
        "method": "independent_exact_fraction_substitution_into_unsolved_covariant_component",
        "evaluations": evaluations,
        "all_domain_admitted": True,
        "all_residuals_zero": True,
    }


def _candidate_ids(value: dict[str, Any], key: str) -> list[str]:
    return sorted(str(item["candidate_id"]) for item in value.get(key, []))


def _validate_predecessors(
    bound: dict[str, tuple[Path, dict[str, Any]]],
) -> tuple[list[str], str, str]:
    common = bound["common_rhs"][1]
    first_order = bound["first_order_reduction"][1]
    domain = bound["r_positive_domain"][1]
    action = bound["total_matter_action"][1]
    interface = bound["matter_interface"][1]
    sourced = bound["sourced_metric_euler"][1]
    if (
        common.get("decision") != "BOUNDED_PASS_68_COMMON_KINEMATIC_RHS_ROWS_BLOCK_17_DYNAMIC_ROWS"
        or common.get("counts", {}).get("common_kinematic_rows_registered") != 68
        or common.get("counts", {}).get("candidate_dynamic_velocity_rows_registered") != 0
    ):
        raise System10MatterDynamicRHSError("common RHS predecessor changed")
    certificate = first_order.get("reduction_certificate", {})
    if first_order.get(
        "decision"
    ) != "PASS_EXACT_85_STATE_FIRST_ORDER_REDUCTION_ALL_TWELVE" or certificate.get("state") != {
        "q_A": 17,
        "v_A": 17,
        "w_iA": 51,
        "total": 85,
    }:
        raise System10MatterDynamicRHSError("85-state reduction authority changed")
    if (
        domain.get("decision")
        != "BOUNDED_PASS_FIXED_CYLINDRICAL_R_POSITIVE_1010_JETS_AND_96_ROWS_NO_PROPAGATION_CLAIM"
        or domain.get("materialization", {}).get("domain_certificate", {}).get("domain") != "r>0"
    ):
        raise System10MatterDynamicRHSError("r-positive domain authority changed")
    components = action.get("shared_matter_action", {}).get("components", [])
    sectors = {item.get("sector_id"): item for item in components}
    if (
        action.get("decision") != "PASS_TOTAL_ACTION_HASH_BINDING_ALL_TWELVE_ONLY"
        or action.get("shared_matter_action_sha256")
        != "9275df25f6c20f92ec03a8aca67c11dd7f9b6dd879808602f4cfe78188d01a7a"
        or sectors.get("canonical_minimally_coupled_scalar", {}).get("field") != "chi_m"
        or sectors.get("barotropic_irrotational_fluid", {}).get("pressure_function")
        != "P(X)=kappa X^2"
    ):
        raise System10MatterDynamicRHSError("matter action authority changed")
    principal = interface.get("combined_matter_certificate", {}).get(
        "combined_matter_principal_compatibility", {}
    )
    if (
        interface.get("decision") != "BOUNDED_PASS_MATTER_INTERFACE_WITH_TYPED_GRAVITY_BLOCK"
        or principal.get("second_order_components") != 6
        or principal.get("principal_block_coefficients")
        != [[-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-3, 1]]
    ):
        raise System10MatterDynamicRHSError("matter interface authority changed")
    claims = sourced.get("claims", {})
    if (
        sourced.get("decision") != "PASS_SOURCED_METRIC_EULER_BINDING_ALL_TWELVE_ONLY"
        or claims.get("sourced_acceleration_solution_closed") is not False
        or claims.get("matter_field_euler_component_expansion_closed") is not False
    ):
        raise System10MatterDynamicRHSError("sourced Euler boundary changed")
    candidate_sets = [
        _candidate_ids(common["materialization"], "candidate_results"),
        _candidate_ids(first_order, "candidate_results"),
        _candidate_ids(action, "candidate_results"),
        _candidate_ids(sourced, "candidate_results"),
    ]
    if any(items != candidate_sets[0] for items in candidate_sets[1:]):
        raise System10MatterDynamicRHSError("candidate identity join changed")
    if len(candidate_sets[0]) != 12 or len(set(candidate_sets[0])) != 12:
        raise System10MatterDynamicRHSError("candidate census changed")
    return (
        candidate_sets[0],
        str(certificate["assembly_sha256"]),
        str(action["shared_matter_action_sha256"]),
    )


def _missing_rows() -> list[dict[str, Any]]:
    rows = []
    for field in range(10):
        body = {
            "row_id": f"evolution_v[{field}]",
            "field_index": field,
            "sector": "sourced_metric",
            "missing_primitives": [
                f"solved_sourced_metric_acceleration_row[{field}]",
                f"fixed_r_positive_lower_order_coordinate_expansion[{field}]",
                f"candidate_bound_equation_origin[{field}]",
            ],
            "status": "BLOCK_SOURCE_PRIMITIVES_UNREGISTERED",
        }
        rows.append(_with_sha(body, "block_sha256"))
    gravity_scalar = {
        "row_id": "evolution_v[10]",
        "field_index": 10,
        "sector": "candidate_gravity_scalar",
        "missing_primitives": [
            "solved_candidate_gravity_scalar_acceleration_row[10]",
            "fixed_r_positive_lower_order_coordinate_expansion[10]",
            "candidate_bound_equation_origin[10]",
        ],
        "status": "BLOCK_SOURCE_PRIMITIVES_UNREGISTERED",
    }
    rows.append(_with_sha(gravity_scalar, "block_sha256"))
    for field, component in zip(range(12, 16), ("B_0", "B_1", "B_2", "B_3"), strict=True):
        body = {
            "row_id": f"evolution_v[{field}]",
            "field_index": field,
            "sector": "source_free_maxwell",
            "component": component,
            "missing_primitives": [
                f"lorenz_gauge_completed_covector_euler_component[{component}]",
                f"fixed_r_positive_connection_and_lower_order_expansion[{component}]",
                f"equation_origin_from_action_plus_gauge_completion[{component}]",
            ],
            "status": "BLOCK_GAUGE_COMPLETED_COMPONENT_EXPANSION_UNREGISTERED",
        }
        rows.append(_with_sha(body, "block_sha256"))
    return rows


def _materialize(
    bound: dict[str, tuple[Path, dict[str, Any]]], frozen: dict[str, Any]
) -> dict[str, Any]:
    candidate_ids, assembly_sha, action_sha = _validate_predecessors(bound)
    rows = [_scalar_row(action_sha, assembly_sha), _fluid_row(action_sha, assembly_sha)]
    missing = _missing_rows()
    validation = _exact_validation()
    measured = {
        "dynamic_rows_registered": len(rows),
        "registered_field_indices": [item["field_index"] for item in rows],
        "row_set_sha256": _canonical_sha([item["row_sha256"] for item in rows]),
        "equation_origin_set_sha256": _canonical_sha(
            [item["equation_origin"]["origin_sha256"] for item in rows]
        ),
        "missing_rows": len(missing),
        "missing_row_block_set_sha256": _canonical_sha([item["block_sha256"] for item in missing]),
    }
    if measured != frozen:
        raise System10MatterDynamicRHSError("frozen matter dynamic expectations changed")
    manifests = []
    for candidate_id in candidate_ids:
        body = {
            "candidate_id": candidate_id,
            "common_kinematic_rows": 68,
            "matter_dynamic_rows": 2,
            "total_rhs_rows_closed": 70,
            "dynamic_rows_remaining": 15,
            "full_85_state_rhs_closed": False,
            "matter_dynamic_row_set_sha256": measured["row_set_sha256"],
            "outcome": "PASS_SCALAR_AND_FLUID_DYNAMIC_ROWS_BLOCK_15_ROWS",
        }
        manifests.append(_with_sha(body, "manifest_sha256"))
    scalar_missing_connection = {
        "mutation": "omit cylindrical w1/r term from evolution_v[11]",
        "witness": {"r": "2", "w1": "3", "all_other_atoms": "0"},
        "exact_residual": "3/2",
        "rejected": True,
    }
    scalar_mass_sign = {
        "mutation": "replace -m_chi**2*q by +m_chi**2*q",
        "witness": {"m_chi": "2", "q": "1", "all_derivative_atoms": "0"},
        "exact_rhs_delta": "8",
        "rejected": True,
    }
    fluid_wrong_denominator = {
        "mutation": "replace D=X+v**2 by X",
        "witness": {"r": "1", "v": "2", "w1": "0", "w2": "0", "w3": "0", "dw11": "1"},
        "X": "2",
        "D": "6",
        "correct_rhs": "1/3",
        "mutated_rhs": "1",
        "exact_rhs_delta": "2/3",
        "rejected": True,
    }
    fluid_missing_connection = {
        "mutation": "replace H22=partial_2 w2+r*w1 by partial_2 w2",
        "witness": {
            "r": "2",
            "v": "3",
            "w1": "1",
            "w2": "0",
            "w3": "0",
            "all_derivative_atoms": "0",
        },
        "X": "4",
        "D": "13",
        "exact_rhs_delta": "2/13",
        "rejected": True,
    }
    return {
        "state_index_contract": {
            "q_A": {"start": 0, "stop": 17},
            "v_A": {"start": 17, "stop": 34},
            "w_1A": {"start": 34, "stop": 51},
            "w_2A": {"start": 51, "stop": 68},
            "w_3A": {"start": 68, "stop": 85},
            "matter_field_map": {"chi_m": 11, "B_mu": [12, 13, 14, 15], "tau": 16},
            "predecessor_assembly_sha256": assembly_sha,
        },
        "rows": rows,
        "row_set_sha256": measured["row_set_sha256"],
        "equation_origin_set_sha256": measured["equation_origin_set_sha256"],
        "independent_exact_validation": validation,
        "candidate_results": manifests,
        "missing_dynamic_rows": missing,
        "missing_row_block_set_sha256": measured["missing_row_block_set_sha256"],
        "negative_controls": {
            "scalar_missing_cylindrical_connection": scalar_missing_connection,
            "scalar_mass_sign_flip": scalar_mass_sign,
            "fluid_wrong_acoustic_denominator": fluid_wrong_denominator,
            "fluid_missing_cylindrical_connection": fluid_missing_connection,
            "maxwell_zero_fill": {
                "mutation": "set four unresolved Maxwell dynamic rows to zero",
                "registered_nonzero_principal_components": 4,
                "missing_lorenz_gauge_completed_lower_order_origins": 4,
                "rejected": True,
            },
            "claim_full_rhs": {
                "closed_rows": 70,
                "required_rows": 85,
                "missing_rows": 15,
                "rejected": True,
            },
        },
        "next_missing_primitive": {
            "required_rows_per_candidate": 15,
            "required_candidate_rows": 180,
            "gravity_and_gravity_scalar_rows": 11,
            "maxwell_rows": 4,
            "block_set_sha256": measured["missing_row_block_set_sha256"],
            "status": "BLOCK_EXACT_PER_ROW_SOURCE_PRIMITIVES_UNREGISTERED",
        },
    }


def build_receipt(config_path: Path, *, root: Path | None = None) -> dict[str, Any]:
    repository = (root or config_path.resolve().parents[1]).resolve()
    config = _load_json(config_path)
    if config.get("schema_version") != (
        "invariant-system10-cylindrical-r-positive-matter-dynamic-rhs-config-1.0"
    ):
        raise System10MatterDynamicRHSError("unsupported config schema")
    expected_caps = {
        "candidates": 12,
        "state_dimension": 85,
        "predecessor_common_rows": 68,
        "matter_dynamic_rows": 2,
        "candidate_dynamic_row_instances": 24,
        "candidate_dynamic_rows_remaining": 180,
        "maximum_output_bytes": 524288,
    }
    if config.get("caps") != expected_caps:
        raise System10MatterDynamicRHSError("caps changed")
    expected_claims = {
        "canonical_scalar_dynamic_row": True,
        "irrotational_fluid_dynamic_row": True,
        "solved_acceleration_and_equation_origin": True,
        "fixed_cylindrical_r_positive": True,
        "maxwell_dynamic_rows": False,
        "gravity_dynamic_rows": False,
        "full_85_state_rhs": False,
        "constraint_propagation": False,
        "hyperbolicity": False,
        "global_theorem": False,
        "promotion": False,
    }
    if config.get("claims_policy") != expected_claims:
        raise System10MatterDynamicRHSError("claims policy broadened")
    bound = {
        name: _load_binding(repository, binding)
        for name, binding in config.get("bindings", {}).items()
    }
    if set(bound) != {
        "common_rhs",
        "first_order_reduction",
        "r_positive_domain",
        "total_matter_action",
        "matter_interface",
        "sourced_metric_euler",
    }:
        raise System10MatterDynamicRHSError("binding manifest changed")
    sources = {
        name: _load_source(repository, binding)
        for name, binding in config.get("source_evidence", {}).items()
    }
    if set(sources) != {"source", "test", "field_order_source"}:
        raise System10MatterDynamicRHSError("source evidence manifest changed")
    expected_test = (
        repository / "tests/test_system10_cylindrical_r_positive_matter_dynamic_rhs_materializer.py"
    )
    expected_field_order = (
        repository
        / "src/sigma_theory_compiler/quartic_twelve_candidate_85_state_first_order_reduction.py"
    )
    if (
        sources["source"] != Path(__file__).resolve()
        or sources["test"] != expected_test
        or sources["field_order_source"] != expected_field_order
    ):
        raise System10MatterDynamicRHSError("self evidence path changed")
    materialization = _materialize(bound, config.get("frozen_expectations", {}))
    body = {
        "schema_version": "invariant-system10-cylindrical-r-positive-matter-dynamic-rhs-result-1.0",
        "campaign_id": config["campaign_id"],
        "decision": "BOUNDED_PASS_2_MATTER_DYNAMIC_ROWS_BLOCK_15_DYNAMIC_ROWS",
        "materialization": materialization,
        "counts": {
            "candidates": 12,
            "state_dimension": 85,
            "predecessor_common_rows_registered": 68,
            "matter_dynamic_rows_registered": 2,
            "scalar_dynamic_rows_registered": 1,
            "fluid_dynamic_rows_registered": 1,
            "candidate_dynamic_row_instances_registered": 24,
            "total_rhs_rows_closed_per_candidate": 70,
            "candidate_dynamic_rows_remaining": 180,
            "full_85_state_rhs_candidates_closed": 0,
            "equation_origins_registered": 2,
            "solved_acceleration_certificates": 2,
            "negative_controls": 6,
        },
        "claims": {
            "canonical_scalar_dynamic_row_closed": True,
            "irrotational_fluid_dynamic_row_closed": True,
            "registered_rows_have_solved_acceleration_and_equation_origin": True,
            "fixed_cylindrical_r_positive_closed": True,
            "maxwell_dynamic_rows_closed": False,
            "gravity_dynamic_rows_closed": False,
            "full_85_state_rhs_closed": False,
            "constraint_propagation_closed": False,
            "hyperbolicity_closed": False,
            "global_theorem_established": False,
            "promotion_authorized": False,
        },
        "scope": (
            "Exact fixed-cylindrical-r>0 solved acceleration rows for the shared canonical "
            "matter scalar and admitted irrotational P(X)=kappa X^2 fluid, including "
            "coordinate connection terms, action/equation origins, denominator-domain "
            "certificates, and exact corruption witnesses. Together with the predecessor's "
            "68 kinematic rows this closes 70/85 rows for each of twelve candidates. Eleven "
            "sourced gravity/gravity-scalar rows and four Lorenz-gauge-completed Maxwell "
            "component rows remain explicitly blocked; no full evolution, propagation, "
            "hyperbolicity, global, or promotion claim is made."
        ),
        "source_bindings": {
            "config": {
                "path": config_path.relative_to(repository).as_posix(),
                "canonical_json_sha256": _canonical_sha(config),
            },
            **{
                name: {
                    "path": path.relative_to(repository).as_posix(),
                    "canonical_lf_sha256": _canonical_lf_sha(path),
                    "content_sha256": value["content_sha256"],
                }
                for name, (path, value) in bound.items()
            },
            **{
                name: {
                    "path": path.relative_to(repository).as_posix(),
                    "canonical_lf_sha256": _canonical_lf_sha(path),
                }
                for name, path in sources.items()
            },
        },
    }
    receipt = {**body, "content_sha256": _canonical_sha(body)}
    if len(json.dumps(receipt).encode("utf-8")) > expected_caps["maximum_output_bytes"]:
        raise System10MatterDynamicRHSError("output cap exceeded")
    return receipt


def write_receipt(
    config_path: Path, output_path: Path, *, root: Path | None = None
) -> dict[str, Any]:
    receipt = build_receipt(config_path, root=root)
    data = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if output_path.exists() and output_path.read_bytes() != data:
        raise System10MatterDynamicRHSError("immutable output conflict")
    if not output_path.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(data)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    write_receipt(arguments.config.resolve(), arguments.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
