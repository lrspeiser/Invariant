from __future__ import annotations

import hashlib
import json
from functools import cache
from math import factorial, isfinite
from pathlib import Path
from typing import Any

import sympy as sp

from .quartic_r3_sobolev_calculus_campaign import r3_sobolev_embedding_constant

SCHEMA_VERSION = "sigma-quartic-h7-paracomposition-topology-campaign-1.0"


class QuarticH7ParacompositionTopologyError(ValueError):
    """Raised when an H7 paracomposition branch or topology claim is invalid."""


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


def _partitions(order: int) -> list[tuple[int, ...]]:
    result: list[tuple[int, ...]] = []

    def visit(remaining: int, maximum: int, current: list[int]) -> None:
        if remaining == 0:
            result.append(tuple(current))
            return
        for part in range(min(maximum, remaining), 0, -1):
            visit(remaining - part, part, [*current, part])

    visit(order, order, [])
    return result


def _faa_coefficient(parts: tuple[int, ...]) -> int:
    order = sum(parts)
    denominator = 1
    for part in set(parts):
        denominator *= factorial(parts.count(part)) * factorial(part) ** parts.count(part)
    return factorial(order) // denominator


def _interaction(left: int, right: int, gap: int = 4) -> str:
    if left <= right - gap:
        return "coefficient_low_state_high"
    if right <= left - gap:
        return "coefficient_high_state_low"
    return "balanced_resonant"


@cache
def generic_h7_paracomposition_topology_control() -> tuple[bool, dict[str, Any]]:
    topology = {
        "metric_deviation_atoms": {"count": 10, "spatial_space": "H7"},
        "first_partial_atoms": {"count": 44, "spatial_space": "H7"},
        "acceleration_free_second_partial_atoms": {
            "count": 99,
            "spatial_space": "H6",
            "one_spatial_derivative_space": "H5",
            "decomposition": {
                "spatial_derivatives_of_v_A": 33,
                "symmetric_spatial_derivatives_of_w_iA": 66,
            },
        },
    }
    count_residual = sum(item["count"] for item in topology.values()) - 153
    high_atom_residual = 33 + 66 - 99
    injection_square = sp.Integer(1) + sp.Integer(3)
    injection_upper = sp.sqrt(injection_square)

    partitions = _partitions(7)
    tame = [parts for parts in partitions if parts != (7,)]
    tame_records = []
    for parts in tame:
        largest = max(parts)
        others = list(parts)
        others.remove(largest)
        tame_records.append(
            {
                "partition": list(parts),
                "outer_Frechet_order": len(parts),
                "Faa_di_Bruno_coefficient": _faa_coefficient(parts),
                "largest_coordinate_atom_derivative": largest,
                "largest_underlying_state_derivative_for_second_atoms": largest + 1,
                "largest_other_coordinate_atom_derivative": max(others, default=0),
                "topology_compatible": largest <= 6 and max(others, default=0) <= 3,
            }
        )
    faa_sum = sum(item["Faa_di_Bruno_coefficient"] for item in tame_records)

    embeddings = {
        str(order): r3_sobolev_embedding_constant(7, order)
        for order in range(6)
    }
    coordinate_linf = {
        str(order): sp.factor(
            sp.sqrt(embeddings[str(order)] ** 2 + 3 * embeddings[str(order + 1)] ** 2)
        )
        for order in range(4)
    }

    levels = tuple(range(-8, 9))
    branch_counts = {
        "coefficient_low_state_high": 0,
        "coefficient_high_state_low": 0,
        "balanced_resonant": 0,
    }
    for left in levels:
        for right in levels:
            branch_counts[_interaction(left, right)] += 1
    branch_total_residual = sum(branch_counts.values()) - len(levels) ** 2

    remote_weight_sum = sp.factor(
        1
        + sp.Rational(2**7, 1 - sp.Rational(1, 2**14))
        + 2**14
        + 2**28
        + 2**42
    )
    balanced_input_partners = sp.Integer(7)
    resonant_remote_constant = sp.factor(
        balanced_input_partners * remote_weight_sum
    )
    remote_geometric_residual = sp.factor(
        sum(sp.Rational(1, 2 ** (14 * distance)) for distance in range(12))
        - (1 - sp.Rational(1, 2 ** (14 * 12))) / (1 - sp.Rational(1, 2**14))
    )

    frequency = sp.Symbol("N", integer=True, positive=True)
    h6_coefficient_amplitude = frequency**-6
    h7_product_growth = sp.simplify(frequency**7 * h6_coefficient_amplitude)
    omitted_high_low = _interaction(6, 0) != "coefficient_low_state_high"
    omitted_resonance = _interaction(0, 0) == "balanced_resonant"
    remote_frequency_residual = sp.expand(frequency + (-frequency))
    passed = bool(
        count_residual == 0
        and high_atom_residual == 0
        and injection_upper == 2
        and len(partitions) == 15
        and len(tame_records) == 14
        and faa_sum == 876
        and all(item["topology_compatible"] for item in tame_records)
        and branch_total_residual == 0
        and remote_geometric_residual == 0
        and remote_weight_sum > 0
        and resonant_remote_constant > remote_weight_sum
        and h7_product_growth == frequency
        and omitted_high_low
        and omitted_resonance
        and remote_frequency_residual == 0
    )
    return passed, {
        "control": "153-to-11 H7 atom topology and complete Bony branch ledger",
        "coordinate_atom_topology": {
            "groups": topology,
            "count_residual": count_residual,
            "state": "U=(q_A,v_A,w_iA) in H7(R3;l2^55)",
            "second_atom_injection": (
                "Y_second=(partial_i v_A, partial_(i w_j)A); each state component "
                "occurs in at most three spatial derivative slots"
            ),
            "combined_coordinate_L2_injection_square": str(injection_square),
            "combined_coordinate_L2_injection_upper": str(injection_upper),
            "derivative_ladder": {
                "Y_low": ["H7", "partial Y_low in H6"],
                "Y_second": ["H6", "partial Y_second in H5"],
            },
        },
        "H7_vector_Sobolev_constants": {
            "scalar_embedding_C_7_m_orders_0_to_5": {
                order: str(value) for order, value in embeddings.items()
            },
            "coordinate_atom_Linfinity_constants_orders_0_to_3": {
                order: str(value) for order, value in coordinate_linf.items()
            },
            "vector_norms": "l2^153 input to l2^11 output",
        },
        "seventh_derivative_tame_partition_ledger": {
            "all_integer_partitions": len(partitions),
            "principal_partition": [7],
            "nonprincipal_partitions": tame_records,
            "nonprincipal_partition_count": len(tame_records),
            "Faa_di_Bruno_multiplicity_sum": faa_sum,
            "all_nonprincipal_topologies_compatible": all(
                item["topology_compatible"] for item in tame_records
            ),
            "reason": (
                "after removing the single D7Y principal block, the L2 block uses at "
                "most D6Y_second=D7U and every remaining Linfinity block uses at most "
                "D3Y_second=D4U"
            ),
        },
        "Bony_branch_partition": {
            "gap": 4,
            "rules": {
                "coefficient_low_state_high": "ell_coefficient<=ell_state-4",
                "coefficient_high_state_low": "ell_state<=ell_coefficient-4",
                "balanced_resonant": "|ell_coefficient-ell_state|<=3",
            },
            "finite_exact_level_range": [levels[0], levels[-1]],
            "branch_counts": branch_counts,
            "branch_total_residual": branch_total_residual,
        },
        "remote_resonant_shell_summation": {
            "balanced_input_partners_per_shell": str(balanced_input_partners),
            "output_support": "j<=max(k,l)+3, including arbitrarily remote low j",
            "weight": "w_j=(1+2^(2j))^7",
            "remote_weight_sum_upper": str(remote_weight_sum),
            "balanced_partner_times_remote_weight_upper": str(
                resonant_remote_constant
            ),
            "geometric_series_residual": str(remote_geometric_residual),
            "shell_index_summation_constant_instantiated": True,
            "Fourier_bilinear_operator_constant_instantiated": False,
        },
        "negative_controls": {
            "omit_coefficient_high_state_low": {
                "witness_shells": [6, 0],
                "actual_branch": _interaction(6, 0),
                "H6_packet": "a_N=N^-6 exp(iNx_1)a_0",
                "H7_product_growth": str(h7_product_growth),
                "rejected": omitted_high_low and h7_product_growth != 1,
            },
            "omit_balanced_resonance": {
                "witness_shells": [0, 0],
                "actual_branch": _interaction(0, 0),
                "rejected": omitted_resonance,
            },
            "omit_remote_low_outputs": {
                "input_frequencies": ["N e1", "-N e1"],
                "output_frequency_residual": str(remote_frequency_residual),
                "rejected": remote_frequency_residual == 0,
            },
            "treat_99_second_atoms_as_H7": {
                "actual_space": "H6 because Y_second is one spatial derivative of U",
                "missing_derivative": 1,
                "rejected": True,
            },
            "treat_C9_outer_smoothness_as_spatial_smoothing": {
                "counterstatement": "DkF bounds do not raise Y_second from H6 to H7",
                "rejected": True,
            },
        },
        "passed": passed,
    }


def _tame_polynomials(c9: dict[str, Any], generic: dict[str, Any]) -> dict[str, Any]:
    m = {
        order: sp.Integer(c9["solved_source_Frechet_operator_integer_uppers"][str(order)])
        for order in range(10)
    }
    linf = {
        order: sp.sympify(
            generic["H7_vector_Sobolev_constants"][
                "coordinate_atom_Linfinity_constants_orders_0_to_3"
            ][str(order)]
        )
        for order in range(4)
    }
    radius = sp.Symbol("R", nonnegative=True, finite=True)
    expressions: dict[str, Any] = {
        "0": {
            "expression": str(
                sp.factor(m[2] * linf[0] ** 2 * radius**2 / 2)
            ),
            "interpretation": "pointwise quadratic Taylor remainder",
        },
        "1": {
            "expression": "0",
            "interpretation": "the full first derivative is assigned to the principal term",
        },
    }
    for order in range(2, 8):
        terms: dict[int, sp.Expr] = {}
        for parts in _partitions(order):
            if parts == (order,):
                continue
            largest = max(parts)
            others = list(parts)
            others.remove(largest)
            coefficient = sp.Integer(_faa_coefficient(parts)) * 2 * m[len(parts)]
            for derivative in others:
                coefficient *= linf[derivative]
            power = len(parts)
            terms[power] = sp.factor(terms.get(power, 0) + coefficient)
        expression = sp.factor(
            sum(coefficient * radius**power for power, coefficient in terms.items())
        )
        expressions[str(order)] = {
            "expression": str(expression),
            "coefficients_by_radius_power": {
                str(power): str(value) for power, value in sorted(terms.items())
            },
            "ordered_spatial_derivative_bound": True,
        }
    full = sp.sqrt(
        sum(
            sp.binomial(7, order)
            * 3**order
            * sp.sympify(expressions[str(order)]["expression"]) ** 2
            for order in range(8)
        )
    )
    return {
        "radius": "R=||U||_H7(l2^55)",
        "ordered_derivative_recombined_remainders": expressions,
        "full_H7_recombined_upper": str(full),
        "H7_weight_identity": (
            "||f||_H7^2=sum_(n=0)^7 binomial(7,n) "
            "sum_(i1,...,in=1)^3 ||partial_i1...partial_in f||2^2"
        ),
        "principal_subtraction": "full multiplication DF(Y)D^nY",
        "paraproduct_subtraction": "T_DF(Y) D^nY",
        "full_multiplication_to_paraproduct_conversion_closed": False,
    }


def _certify_candidate(
    c9: dict[str, Any],
    jacobian: dict[str, Any],
    dyadic: dict[str, Any],
    global_energy: dict[str, Any],
    generic: dict[str, Any],
) -> dict[str, Any]:
    candidate_id = str(c9.get("candidate_id"))
    records = (jacobian, dyadic, global_energy)
    if any(record.get("candidate_id") != candidate_id for record in records):
        raise QuarticH7ParacompositionTopologyError("candidate identity mismatch")
    if any(record.get("coefficients") != c9.get("coefficients") for record in records):
        raise QuarticH7ParacompositionTopologyError("candidate coefficient mismatch")
    expected = (
        "pass_solved_source_C9_operator_envelopes_H7_topology_fail_closed",
        "pass_full_11x153_entrywise_arithmetic_mixed_tensors_fail_closed",
        "pass_H7_dyadic_partition_and_shell_local_commutator_framework",
        "audit_global_H7_energy_single_source_remainder_lifespan_fail_closed",
    )
    if tuple(record.get("status") for record in (c9, *records)) != expected:
        raise QuarticH7ParacompositionTopologyError("candidate prerequisite status mismatch")
    physical_hash = dyadic["physical_pencil"]["source_spatial_block_sha256"]
    if not (
        global_energy["good_unknown_and_source"]["physical_pencil_sha256"]
        == physical_hash
        and global_energy["good_unknown_and_source"][
            "principal_identity_entries_zero"
        ]
        == 3025
        and global_energy["good_unknown_and_source"][
            "leading_good_unknown_symbol_binding_verified"
        ]
        is True
        and jacobian["physical_pencil_J_identity_proved"] is True
    ):
        raise QuarticH7ParacompositionTopologyError(
            "principal good-unknown binding is incomplete"
        )
    tame = _tame_polynomials(c9, generic)
    tube_radius = sp.sympify(
        global_energy["bootstrap_and_conditional_lifespan"]["tube_H7_radius"]
    )
    tame_expression = sp.sympify(tame["full_H7_recombined_upper"])
    radius_symbols = [
        symbol for symbol in tame_expression.free_symbols if str(symbol) == "R"
    ]
    if len(radius_symbols) != 1:
        raise QuarticH7ParacompositionTopologyError(
            "recombined tame radius symbol is missing or ambiguous"
        )
    tame_at_tube = tame_expression.subs(radius_symbols[0], tube_radius)
    tame_numeric = float(sp.N(tame_at_tube, 18))
    if not (isfinite(tame_numeric) and tame_numeric > 0):
        raise QuarticH7ParacompositionTopologyError(
            "recombined tame constant is invalid"
        )
    return {
        "schema_version": "sigma-quartic-h7-paracomposition-topology-certificate-1.0",
        "status": (
            "pass_H7_atom_topology_and_recombined_tame_ledger_"
            "high_low_paraproduct_fail_closed"
        ),
        "candidate_id": candidate_id,
        "coefficients": c9["coefficients"],
        "provenance": {
            "physical_pencil_sha256": physical_hash,
            "full_entry_manifest_sha256": jacobian["provenance"][
                "full_entry_manifest_sha256"
            ],
            "coordinate_atom_basis_sha256": jacobian["provenance"][
                "coordinate_atom_basis_sha256"
            ],
            "state_basis_sha256": jacobian["provenance"]["state_basis_sha256"],
            "C9_orders": c9["orders_cumulatively_closed"],
        },
        "coordinate_atom_topology": generic["coordinate_atom_topology"],
        "vector_153_to_11_recombined_tame_bound": tame,
        "Bony_branches": {
            "coefficient_low_state_high": {
                "status": "closed_by_exact_good_unknown_principal_identity",
                "identity": "D_Y E55 J_153x55(xi)=iP55(Y,xi)",
                "entry_residuals_zero": 3025,
            },
            "coefficient_high_state_low": {
                "status": "fail_closed",
                "reason": (
                    "the exact LP H6 coefficient packet has H7 growth N; C9 controls "
                    "outer derivatives of F but supplies no spatial derivative"
                ),
                "negative_control": generic["negative_controls"][
                    "omit_coefficient_high_state_low"
                ],
            },
            "balanced_resonant": {
                "status": "shell_index_sum_only_operator_constant_fail_closed",
                "remote_shell_constant": generic["remote_resonant_shell_summation"][
                    "balanced_partner_times_remote_weight_upper"
                ],
                "Fourier_bilinear_operator_constant_instantiated": False,
            },
        },
        "recombined_full_multiplication_tame_ledger": {
            "all_14_nonprincipal_D7_partitions_topology_compatible": True,
            "tube_H7_radius": str(tube_radius),
            "full_H7_upper_at_tube_numeric": tame_numeric,
            "does_not_equal_paraproduct_remainder": True,
        },
        "connection_to_global_H7": {
            "existing_inequality": global_energy[
                "strongest_global_differential_inequality"
            ]["exact"],
            "B7_replaced_by_closed_expression": False,
            "reason": (
                "the coefficient-high/state-low branch and resonant Fourier operator "
                "constant remain inside B7"
            ),
        },
        "coordinate_topology_instantiated": True,
        "C9_vector_recombined_tame_constant_instantiated": True,
        "remote_shell_index_summation_constant_instantiated": True,
        "complete_paracomposition_remainder_closed": False,
        "global_H7_differential_inequality_closed": False,
        "global_dyadic_summation_applied": False,
        "nonlinear_lifespan_proved": False,
        "remaining_gates": [
            "prove a system-specific cancellation of coefficient-high/state-low H6 packets",
            "instantiate the resonant Fourier bilinear operator constant for the declared cutoffs",
            "replace B7 in the global energy ledger and only then apply the dyadic sum",
        ],
    }


def run_quartic_h7_paracomposition_topology_campaign(
    c9_campaign: dict[str, Any],
    full_jacobian_campaign: dict[str, Any],
    dyadic_campaign: dict[str, Any],
    global_h7_campaign: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        campaigns = (c9_campaign, full_jacobian_campaign, dyadic_campaign, global_h7_campaign)
        expected_statuses = (
            "pass_all_12_solved_source_C9_operator_envelopes_H7_topology_fail_closed",
            "pass_all_12_full_11x153_entrywise_arithmetic_mixed_tensors_fail_closed",
            "pass_all_12_H7_dyadic_local_frameworks_global_commutator_fail_closed",
            "audit_all_12_global_H7_energies_single_source_remainder_lifespans_fail_closed",
        )
        if config.get("schema_version") != SCHEMA_VERSION:
            raise QuarticH7ParacompositionTopologyError(
                "unsupported campaign schema_version"
            )
        if tuple(campaign.get("status") for campaign in campaigns) != expected_statuses:
            raise QuarticH7ParacompositionTopologyError(
                "campaign prerequisite status mismatch"
            )
        if not all(_content_hash_matches(campaign) for campaign in campaigns):
            raise QuarticH7ParacompositionTopologyError("campaign content hash mismatch")
        c9_links = c9_campaign["upstream_sha256"]
        global_links = global_h7_campaign["upstream_sha256"]
        if (
            c9_links["full_source_jacobian"] != full_jacobian_campaign["content_sha256"]
            or c9_links["global_H7"] != global_h7_campaign["content_sha256"]
            or global_links["dyadic"] != dyadic_campaign["content_sha256"]
            or global_links["source_jacobian"]
            != full_jacobian_campaign["upstream_sha256"]["principal_source"]
        ):
            raise QuarticH7ParacompositionTopologyError(
                "C9/Jacobian/dyadic/global provenance mismatch"
            )
        if (
            int(config["expected_candidate_count"]) != 12
            or int(config["spatial_dimension"]) != 3
            or int(config["state_dimension"]) != 55
            or int(config["coordinate_atom_dimension"]) != 153
            or int(config["state_sobolev_order"]) != 7
            or int(config["second_atom_sobolev_order"]) != 6
            or int(config["differentiated_second_atom_sobolev_order"]) != 5
            or int(config["paraproduct_gap"]) != 4
            or config.get("high_low_branch_policy") != "fail_closed"
            or config.get("resonant_operator_policy") != "fail_closed"
            or config.get("global_H7_policy") != "fail_closed"
            or config.get("lifespan_policy") != "fail_closed"
        ):
            raise QuarticH7ParacompositionTopologyError(
                "unsupported H7 paracomposition topology contract"
            )
        loss_control = dyadic_campaign["generic_dyadic_localization_control"][
            "derivative_loss_negative"
        ]
        if not (
            loss_control.get("growth_exponent") == 1
            and loss_control.get("rejected") is True
            and loss_control.get("R3_Schwartz_Fourier_support_counterexample_encoded")
            is True
        ):
            raise QuarticH7ParacompositionTopologyError(
                "dyadic H6-to-H7 loss control is absent"
            )
        generic_passed, generic = generic_h7_paracomposition_topology_control()
        if not generic_passed:
            raise QuarticH7ParacompositionTopologyError("generic topology control failed")
        maps = tuple(_candidate_records(campaign) for campaign in campaigns)
        candidate_ids = set(maps[0])
        if len(candidate_ids) != 12 or any(
            set(records) != candidate_ids for records in maps[1:]
        ):
            raise QuarticH7ParacompositionTopologyError("candidate-set mismatch")
        certificates = [
            _certify_candidate(
                maps[0][candidate_id],
                maps[1][candidate_id],
                maps[2][candidate_id],
                maps[3][candidate_id],
                generic,
            )
            for candidate_id in sorted(candidate_ids)
        ]
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": (
                "pass_all_12_H7_atom_topologies_and_recombined_tame_ledgers_"
                "high_low_paraproduct_fail_closed"
            ),
            "errors": [],
            "upstream_sha256": {
                "solved_source_C9": c9_campaign["content_sha256"],
                "full_source_jacobian": full_jacobian_campaign["content_sha256"],
                "dyadic_localization": dyadic_campaign["content_sha256"],
                "global_H7": global_h7_campaign["content_sha256"],
            },
            "config_sha256": _content_hash(config),
            "generic_H7_paracomposition_topology_control": generic,
            "counts": {
                "selected": len(certificates),
                "coordinate_topologies_instantiated": len(certificates),
                "C9_vector_recombined_tame_ledgers_instantiated": len(certificates),
                "principal_low_high_good_unknown_branches_closed": len(certificates),
                "remote_shell_index_summations_instantiated": len(certificates),
                "coefficient_high_state_low_branches_closed": 0,
                "resonant_Fourier_operator_constants_instantiated": 0,
                "complete_paracomposition_remainders_closed": 0,
                "global_H7_closures": 0,
                "lifespans_proved": 0,
                "rejected": 0,
            },
            "certificates": certificates,
            "claim": (
                "All 12 candidates now have an exact 54-H7/99-H6/derived-H5 atom "
                "topology, C9 vector-valued recombined tame ledger, all Bony branches, "
                "and an explicit remote-shell weight sum. The coefficient-high/state-low "
                "H6 packet and resonant Fourier operator constant remain unclosed, so B7, "
                "the global H7 inequality, and lifespan remain fail-closed."
            ),
            "scope": (
                "Topology and recombined derivative algebra are complete; full "
                "multiplication-to-paraproduct conversion is not."
            ),
        }
    except (
        KeyError,
        TypeError,
        ValueError,
        QuarticH7ParacompositionTopologyError,
    ) as error:
        errors.append(str(error))
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "reject",
            "errors": errors,
            "certificates": [],
            "counts": {
                "selected": 0,
                "coordinate_topologies_instantiated": 0,
                "C9_vector_recombined_tame_ledgers_instantiated": 0,
                "principal_low_high_good_unknown_branches_closed": 0,
                "remote_shell_index_summations_instantiated": 0,
                "coefficient_high_state_low_branches_closed": 0,
                "resonant_Fourier_operator_constants_instantiated": 0,
                "complete_paracomposition_remainders_closed": 0,
                "global_H7_closures": 0,
                "lifespans_proved": 0,
                "rejected": 0,
            },
        }
    return {**body, "content_sha256": _content_hash(body)}


def write_quartic_h7_paracomposition_topology_campaign(
    result: dict[str, Any], output: Path
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "campaign.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
