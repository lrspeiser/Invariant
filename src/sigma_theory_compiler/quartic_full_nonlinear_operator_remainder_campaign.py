from __future__ import annotations

import hashlib
import json
from functools import cache
from math import ceil, inf, isfinite, nextafter
from pathlib import Path
from typing import Any

import sympy as sp

SCHEMA_VERSION = "sigma-quartic-full-nonlinear-operator-remainder-campaign-1.0"


class QuarticFullNonlinearOperatorRemainderError(ValueError):
    """Raised when the operator-level nonlinear remainder audit is inconsistent."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _content_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _content_hash_matches(campaign: dict[str, Any]) -> bool:
    body = {key: value for key, value in campaign.items() if key != "content_sha256"}
    return campaign.get("content_sha256") == _content_hash(body)


def _candidate_records(campaign: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item["candidate_id"]): item
        for item in campaign.get("certificates", [])
        if isinstance(item, dict) and "candidate_id" in item
    }


def _outward_integer(value: float) -> int:
    if not (isfinite(value) and value >= 0):
        raise QuarticFullNonlinearOperatorRemainderError(
            "a published Frechet envelope is invalid"
        )
    return ceil(nextafter(value, inf))


@cache
def generic_full_operator_remainder_control() -> tuple[bool, dict[str, Any]]:
    """Verify Taylor, Bony-partition, H7-product, and C4-obstruction controls."""

    y, h, t = sp.symbols("y h t", real=True, finite=True)
    coefficients = sp.symbols("a0:5", real=True, finite=True)
    polynomial = sum(coefficients[index] * y**index for index in range(5))
    shifted = polynomial.subs(y, y + t * h)
    taylor_integral = sp.integrate(
        (1 - t) * sp.diff(polynomial, y, 2).subs(y, y + t * h) * h**2,
        (t, 0, 1),
    )
    taylor_residual = sp.expand(
        polynomial.subs(y, y + h)
        - polynomial
        - sp.diff(polynomial, y) * h
        - taylor_integral
    )
    difference_residuals = {
        str(order): str(
            sp.expand(
                sp.diff(polynomial, y, order).subs(y, y + h)
                - sp.diff(polynomial, y, order)
                - sp.integrate(
                    sp.diff(shifted, y, order + 1) * h,
                    (t, 0, 1),
                )
            )
        )
        for order in range(1, 4)
    }

    fourier_l1_h7 = sp.sqrt(21) / (64 * sp.sqrt(sp.pi))
    h7_algebra = sp.factor(2**7 * fourier_l1_h7)
    dyadic_source_upper = sp.Integer(2) ** 7
    state_h7_square_to_q7 = sp.Integer(2) ** 15
    q7_bilinear_factor = sp.factor(
        dyadic_source_upper * state_h7_square_to_q7 * h7_algebra
    )

    shell_pairs = [(left, right) for left in range(-8, 9) for right in range(-8, 9)]
    low_high = {(left, right) for left, right in shell_pairs if left <= right - 3}
    high_low = {(left, right) for left, right in shell_pairs if right <= left - 3}
    resonant = {(left, right) for left, right in shell_pairs if abs(left - right) <= 2}
    partition_exact = bool(
        not (low_high & high_low)
        and not (low_high & resonant)
        and not (high_low & resonant)
        and low_high | high_low | resonant == set(shell_pairs)
    )

    frequency = sp.Symbol("N", integer=True, positive=True)
    x = sp.Symbol("x", real=True, finite=True)
    c4_family = sp.sin(frequency * x) / frequency**4
    derivative_supremum_powers = {
        str(order): str(frequency ** (order - 4)) for order in range(5)
    }
    fifth_derivative_at_zero = sp.simplify(sp.diff(c4_family, x, 5).subs(x, 0))
    omitted_resonant_witness = (0, 0) not in low_high | high_low
    corrupted_taylor = sp.expand(
        polynomial.subs(y, y + h)
        - polynomial
        - sp.diff(polynomial, y) * h
        - 2 * taylor_integral
    )
    corrupted_witness = sp.expand(corrupted_taylor.subs(dict(zip(coefficients, range(1, 6), strict=True))))

    passed = bool(
        taylor_residual == 0
        and set(difference_residuals.values()) == {"0"}
        and h7_algebra == 2 * sp.sqrt(21) / sp.sqrt(sp.pi)
        and q7_bilinear_factor == 2**23 * sp.sqrt(21) / sp.sqrt(sp.pi)
        and partition_exact
        and fifth_derivative_at_zero == frequency
        and omitted_resonant_witness
        and corrupted_witness != 0
    )
    return passed, {
        "control": "full operator Taylor remainder and exact H7/Bony boundary",
        "operator_Taylor_identities": {
            "quadratic_integral_remainder": (
                "F(Y+H)-F(Y)-DF(Y)H="
                "integral_0^1(1-t)D2F(Y+tH)[H,H]dt"
            ),
            "quadratic_scalar_residual": str(taylor_residual),
            "derivative_difference_identity": (
                "DkF(Y+H)-DkF(Y)="
                "integral_0^1 D(k+1)F(Y+tH)[H]dt"
            ),
            "orders_1_to_3_scalar_residuals": difference_residuals,
        },
        "H7_product_algebra": {
            "space": "R3",
            "fourier_convention": "unitary Fourier transform",
            "fourier_L1_embedding_constant": str(fourier_l1_h7),
            "weight_inequality": "<eta+zeta>^7<=2^6(<eta>^7+<zeta>^7)",
            "algebra_constant": str(h7_algebra),
            "bilinear_vector_bound": (
                "||B(H,K)||_H7(l2)<=C_alg||B||_(l2,l2->l2)"
                "||H||_H7(l2)||K||_H7(l2)"
            ),
            "dyadic_source_upper": str(dyadic_source_upper),
            "state_H7_square_to_Q7": str(state_h7_square_to_q7),
            "total_bilinear_Q7_factor": str(q7_bilinear_factor),
        },
        "Bony_partition": {
            "identity": "fg=T_f g+T_g f+R(f,g)",
            "low_high_rule": "j<=k-3",
            "high_low_rule": "k<=j-3",
            "resonant_rule": "|j-k|<=2",
            "finite_exact_audit_shell_range": [-8, 8],
            "audited_ordered_pairs": len(shell_pairs),
            "low_high_pairs": len(low_high),
            "high_low_pairs": len(high_low),
            "resonant_pairs": len(resonant),
            "partition_exact": partition_exact,
        },
        "C4_to_H7_obstruction": {
            "family": "f_N(x)=N^-4 sin(Nx), N>=1",
            "derivative_supremum_uppers_orders_0_to_4": derivative_supremum_powers,
            "uniform_C4_upper": "1",
            "fifth_derivative_at_zero": str(fifth_derivative_at_zero),
            "conclusion": (
                "uniform D0-D4 operator envelopes do not imply a D5 bound; "
                "therefore they cannot alone produce a general variable-coefficient "
                "H7 Nemytskii/paralinearization constant"
            ),
        },
        "negative_controls": {
            "double_Taylor_kernel": {
                "exact_witness_residual": str(corrupted_witness),
                "rejected": corrupted_witness != 0,
            },
            "omit_Bony_resonance": {
                "unassigned_shell_pair": [0, 0],
                "rejected": omitted_resonant_witness,
            },
            "infer_H7_from_C4_only": {
                "unbounded_witness": "D5 f_N(0)=N",
                "rejected": fifth_derivative_at_zero.has(frequency),
            },
        },
        "passed": passed,
    }


def _certify_candidate(
    jacobian: dict[str, Any],
    solved: dict[str, Any],
    global_energy: dict[str, Any],
    generic: dict[str, Any],
) -> dict[str, Any]:
    candidate_id = str(jacobian.get("candidate_id"))
    records = (solved, global_energy)
    if any(record.get("candidate_id") != candidate_id for record in records):
        raise QuarticFullNonlinearOperatorRemainderError("candidate identity mismatch")
    if any(
        record.get("coefficients") != jacobian.get("coefficients") for record in records
    ):
        raise QuarticFullNonlinearOperatorRemainderError(
            "candidate coefficient mismatch"
        )
    if (
        jacobian.get("status")
        != "pass_full_11x153_entrywise_arithmetic_mixed_tensors_fail_closed"
        or solved.get("status")
        != "pass_coordinate_atom_C4_solved_source_moser_envelopes"
        or global_energy.get("status")
        != "audit_global_H7_energy_single_source_remainder_lifespan_fail_closed"
    ):
        raise QuarticFullNonlinearOperatorRemainderError(
            "candidate prerequisite status mismatch"
        )
    if not (
        jacobian.get("full_11x153_source_Jacobian_entrywise_materialized") is True
        and jacobian.get("total_entries_entrywise_arithmetic") == 1683
        and jacobian.get("full_component_Frechet_tensors_orders_2_to_4_complete")
        is False
    ):
        raise QuarticFullNonlinearOperatorRemainderError(
            "full first-Jacobian arithmetic contract mismatch"
        )
    source = solved["solved_source_Frechet_derivatives"]
    if not (
        source.get("orders") == [0, 1, 2, 3, 4]
        and source.get("input_norm") == "153-coordinate-atom component l_infinity"
        and source.get("output_norm")
        == "11-acceleration-vector Euclidean 2-norm"
    ):
        raise QuarticFullNonlinearOperatorRemainderError(
            "solved-source C4 norm contract mismatch"
        )
    frechet_ceilings = {
        str(order): _outward_integer(
            float(source["2_norm_envelopes_numeric"][str(order)])
        )
        for order in range(5)
    }
    m2 = sp.Integer(frechet_ceilings["2"])
    m3 = sp.Integer(frechet_ceilings["3"])
    m4 = sp.Integer(frechet_ceilings["4"])
    algebra = sp.sympify(generic["H7_product_algebra"]["algebra_constant"])
    q7_factor = sp.sympify(
        generic["H7_product_algebra"]["total_bilinear_Q7_factor"]
    )
    frozen_cb = sp.factor(q7_factor * m2)
    taylor_cb = sp.factor(frozen_cb / 2)
    h7_lower = sp.sympify(global_energy["global_energy"]["H7_lower"])
    gamma = sp.sympify(
        global_energy["strongest_global_differential_inequality"]["Gamma_B"]
    )
    frozen_energy_increment = gamma * frozen_cb / h7_lower
    numeric = {
        "M2": int(m2),
        "M3": int(m3),
        "M4": int(m4),
        "frozen_full_D2_C_B": float(sp.N(frozen_cb, 18)),
        "quadratic_Taylor_C_B": float(sp.N(taylor_cb, 18)),
        "frozen_energy_D_increment": float(sp.N(frozen_energy_increment, 18)),
    }
    if any(not (isfinite(value) and value > 0) for value in numeric.values()):
        raise QuarticFullNonlinearOperatorRemainderError(
            "a quantitative operator constant is invalid"
        )
    provenance = jacobian["provenance"]
    return {
        "schema_version": (
            "sigma-quartic-full-nonlinear-operator-remainder-certificate-1.0"
        ),
        "status": (
            "pass_full_pointwise_C4_and_frozen_H7_operator_remainder_"
            "variable_H7_fail_closed"
        ),
        "candidate_id": candidate_id,
        "coefficients": jacobian["coefficients"],
        "source_identity_and_basis_provenance": {
            "full_entry_manifest_sha256": provenance["full_entry_manifest_sha256"],
            "principal_arithmetic_dag_sha256": provenance[
                "principal_arithmetic_dag_sha256"
            ],
            "coordinate_atom_basis_sha256": provenance[
                "coordinate_atom_basis_sha256"
            ],
            "state_basis_sha256": provenance["state_basis_sha256"],
            "source_Jacobian_shape": [11, 153],
            "entrywise_DF_arithmetic_roots": 1683,
        },
        "global_operator_envelopes": {
            "coordinate_component_radius": solved["coordinate_component_radius"],
            "input_norm": source["input_norm"],
            "output_norm": source["output_norm"],
            "outward_integer_ceilings": frechet_ceilings,
            "D2_bilinear_operator_upper": str(m2),
            "D3_trilinear_operator_upper": str(m3),
            "D4_quadrilinear_operator_upper": str(m4),
            "component_D2_D4_leaf_enumeration_needed_for_these_norm_bounds": False,
            "component_D2_D4_leaf_enumeration_completed": False,
        },
        "full_variable_pointwise_remainder": {
            "quadratic_Taylor_bound": (
                f"||F(Y+H)-F(Y)-DF(Y)H||2<={m2}/2*||H||inf^2"
            ),
            "DF_Lipschitz_bound": f"||DF(Y)-DF(Z)||<={m2}*||Y-Z||inf",
            "D2F_Lipschitz_bound": f"||D2F(Y)-D2F(Z)||<={m3}*||Y-Z||inf",
            "D3F_Lipschitz_bound": f"||D3F(Y)-D3F(Z)||<={m4}*||Y-Z||inf",
            "validity": "the joining line segment remains in the certified coordinate tube",
            "all_11_output_rows": True,
            "all_153_coordinate_directions": True,
            "closed": True,
        },
        "full_frozen_H7_quadratic_remainder": {
            "definition": "B_*(H,K)=D2F(Y_*)[H,K] for a fixed tube point Y_*",
            "bound": (
                f"||B_*(H,K)||H7(l2)<={algebra}*{m2}*"
                "||H||H7(l2)*||K||H7(l2)"
            ),
            "H7_algebra_constant": str(algebra),
            "dyadic_Q7_conversion_factor": str(q7_factor),
            "full_D2_B7_C_B": str(frozen_cb),
            "quadratic_Taylor_half_C_B": str(taylor_cb),
            "energy_D_increment_if_isolated": str(frozen_energy_increment),
            "all_direction_operator_bound_bypasses_component_tensor_enumeration": True,
            "closed": True,
        },
        "exact_Bony_ledger": {
            **generic["Bony_partition"],
            "frozen_bilinear_sum_controlled_by_H7_algebra": True,
            "variable_coefficient_high_frequency_controlled_from_C4_only": False,
        },
        "global_variable_coefficient_H7_remainder": {
            "closed": False,
            "reason": (
                "spatially differentiating D2F(Y(x)) through the H7 endpoint requires "
                "higher composition control not implied by D0-D4 envelopes"
            ),
            "rigorous_obstruction": generic["C4_to_H7_obstruction"],
            "minimum_unavailable_operator_orders": [5, 6, 7],
            "safe_sufficient_next_input": (
                "tube-uniform solved-source Frechet operator envelopes through order "
                "nine, followed by a quantitative vector-valued paracomposition theorem"
            ),
            "good_unknown_variable_remainder_closed": False,
            "global_H7_differential_inequality_closed": False,
            "global_dyadic_summation_applied": False,
            "nonlinear_lifespan_proved": False,
        },
        "numeric_constants": numeric,
        "precise_closure": (
            "The full 11-output, 153-direction pointwise nonlinear Taylor/Lipschitz "
            "remainder and the full fixed-coefficient D2 H7 bilinear term are bounded "
            "without enumerating D2-D4 components. The spatially variable H7 Bony "
            "remainder remains open."
        ),
    }


def run_quartic_full_nonlinear_operator_remainder_campaign(
    full_jacobian_campaign: dict[str, Any],
    solved_source_campaign: dict[str, Any],
    global_h7_campaign: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        campaigns = (full_jacobian_campaign, solved_source_campaign, global_h7_campaign)
        expected_statuses = (
            "pass_all_12_full_11x153_entrywise_arithmetic_mixed_tensors_fail_closed",
            "pass_all_12_coordinate_atom_C4_solved_source_moser_envelopes",
            (
                "audit_all_12_global_H7_energies_single_source_remainder_"
                "lifespans_fail_closed"
            ),
        )
        if config.get("schema_version") != SCHEMA_VERSION:
            raise QuarticFullNonlinearOperatorRemainderError(
                "unsupported campaign schema_version"
            )
        if tuple(campaign.get("status") for campaign in campaigns) != expected_statuses:
            raise QuarticFullNonlinearOperatorRemainderError(
                "campaign prerequisite status mismatch"
            )
        if not all(_content_hash_matches(campaign) for campaign in campaigns):
            raise QuarticFullNonlinearOperatorRemainderError(
                "campaign content hash mismatch"
            )
        full_links = full_jacobian_campaign["upstream_sha256"]
        global_links = global_h7_campaign["upstream_sha256"]
        solved_links = solved_source_campaign["upstream_sha256"]
        if (
            full_links["principal_source"] != global_links["source_jacobian"]
            or solved_links["coordinate_tube"] != global_links["tube"]
        ):
            raise QuarticFullNonlinearOperatorRemainderError(
                "Jacobian/solved-source/global-H7 provenance mismatch"
            )
        if (
            int(config["expected_candidate_count"]) != 12
            or int(config["source_rows"]) != 11
            or int(config["coordinate_atom_dimension"]) != 153
            or int(config["available_Frechet_order"]) != 4
            or int(config["target_sobolev_order"]) != 7
            or config.get("component_tensor_policy") != "operator_norm_bypass"
            or config.get("variable_coefficient_H7_policy") != "fail_closed"
            or config.get("global_H7_policy") != "fail_closed"
            or config.get("lifespan_policy") != "fail_closed"
        ):
            raise QuarticFullNonlinearOperatorRemainderError(
                "unsupported full operator-remainder contract"
            )
        generic_passed, generic = generic_full_operator_remainder_control()
        if not generic_passed:
            raise QuarticFullNonlinearOperatorRemainderError(
                "generic operator-remainder control failed"
            )
        maps = tuple(_candidate_records(campaign) for campaign in campaigns)
        candidate_ids = set(maps[0])
        if len(candidate_ids) != 12 or any(
            set(records) != candidate_ids for records in maps[1:]
        ):
            raise QuarticFullNonlinearOperatorRemainderError(
                "candidate-set mismatch"
            )
        certificates = [
            _certify_candidate(
                maps[0][candidate_id],
                maps[1][candidate_id],
                maps[2][candidate_id],
                generic,
            )
            for candidate_id in sorted(candidate_ids)
        ]
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": (
                "pass_all_12_full_pointwise_C4_and_frozen_H7_operator_"
                "remainders_variable_H7_fail_closed"
            ),
            "errors": [],
            "upstream_sha256": {
                "full_source_jacobian": full_jacobian_campaign["content_sha256"],
                "solved_source_C4": solved_source_campaign["content_sha256"],
                "global_H7": global_h7_campaign["content_sha256"],
            },
            "config_sha256": _content_hash(config),
            "generic_full_operator_remainder_control": generic,
            "counts": {
                "selected": len(certificates),
                "full_DF_entrywise_arithmetic_certificates": len(certificates),
                "full_pointwise_C4_operator_remainders_closed": len(certificates),
                "full_frozen_H7_all_direction_remainders_closed": len(certificates),
                "component_D2_D4_enumerations_bypassed_for_norm_bounds": len(
                    certificates
                ),
                "variable_coefficient_H7_remainders_closed": 0,
                "global_H7_closures": 0,
                "lifespans_proved": 0,
                "rejected": 0,
            },
            "certificates": certificates,
            "claim": (
                "All 12 candidates have full pointwise C4 nonlinear operator bounds and "
                "a full all-direction frozen D2 H7 bound without component-tensor "
                "enumeration. The variable-coefficient H7 remainder is rigorously "
                "fail-closed because C4 does not control D5, much less the H7 "
                "paracomposition ledger."
            ),
            "scope": certificates[0]["precise_closure"],
        }
    except (
        KeyError,
        TypeError,
        ValueError,
        QuarticFullNonlinearOperatorRemainderError,
    ) as error:
        errors.append(str(error))
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "reject",
            "errors": errors,
            "certificates": [],
            "counts": {
                "selected": 0,
                "full_DF_entrywise_arithmetic_certificates": 0,
                "full_pointwise_C4_operator_remainders_closed": 0,
                "full_frozen_H7_all_direction_remainders_closed": 0,
                "component_D2_D4_enumerations_bypassed_for_norm_bounds": 0,
                "variable_coefficient_H7_remainders_closed": 0,
                "global_H7_closures": 0,
                "lifespans_proved": 0,
                "rejected": 0,
            },
        }
    return {**body, "content_sha256": _content_hash(body)}


def write_quartic_full_nonlinear_operator_remainder_campaign(
    result: dict[str, Any], output: Path
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "campaign.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
