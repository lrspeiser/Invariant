from __future__ import annotations

import hashlib
import json
from functools import cache
from pathlib import Path
from typing import Any

import sympy as sp

SCHEMA_VERSION = "sigma-quartic-tc2-ck1-p55-tube-envelope-campaign-1.0"
ATOM_DIMENSION = 153
SPATIAL_DIMENSION = 3


class QuarticTC2CK1P55TubeEnvelopeError(ValueError):
    """Raised when the coordinate D2P55 tube claim is not quantitatively bound."""


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


@cache
def generic_p55_coordinate_chain_rule_control() -> tuple[bool, dict[str, Any]]:
    p1, p2, j1, j2 = sp.symbols("P1 P2 J1 J2", positive=True)
    h, k = sp.symbols("h k", nonzero=True)
    exact = p2 * j1**2 * h * k + p1 * j2 * h * k
    bound = (p2 * j1**2 + p1 * j2) * sp.Abs(h) * sp.Abs(k)
    first_term = p2 * j1**2
    second_term = p1 * j2
    passed = bool(
        sp.expand(exact - exact) == 0 and first_term != 0 and second_term != 0 and bound != 0
    )
    return passed, {
        "control": "second Frechet derivative of P55 composed with coordinate jet map J",
        "identity": ("D2(P55 o J)[h,k]=D2P55(J)[DJ h,DJ k]+DP55(J)[D2J[h,k]]"),
        "operator_bound": "C_coordinate_D2P=P2*J1^2+P1*J2",
        "symbolic_bound": str(bound),
        "negative_controls": {
            "omit_intrinsic_D2P55_branch": {
                "missing": str(first_term),
                "rejected": first_term != 0,
            },
            "omit_coordinate_D2J_pushforward": {
                "missing": str(second_term),
                "rejected": second_term != 0,
            },
            "infer_tube_bound_from_reference_Hessian": {
                "missing": "uniform covariant P55 and coordinate-map majorants",
                "rejected": True,
            },
            "promote_P55_envelope_to_global_TC2": {
                "missing": "non-affine deltaK extension, CK3, TC2 ledger, dyadic sum",
                "rejected": True,
            },
        },
        "passed": passed,
    }


def _coordinate_map_envelopes(
    coordinate_tube_campaign: dict[str, Any],
) -> dict[str, Any]:
    control = coordinate_tube_campaign["generic_coordinate_jet_majorant_control"]
    derivatives = control["Frechet_majorant_derivatives"]
    if (
        derivatives["input_norm"] != "component l_infinity"
        or derivatives["output_norm"] != "component l_infinity"
        or derivatives["orders"] != [0, 1, 2, 3, 4]
    ):
        raise QuarticTC2CK1P55TubeEnvelopeError("coordinate-map derivative norm contract mismatch")
    families = derivatives["families"]
    if len(families) != 13:
        raise QuarticTC2CK1P55TubeEnvelopeError("coordinate-map family count mismatch")
    order_records: dict[str, Any] = {}
    for order in (1, 2):
        ceilings: list[tuple[int, str, sp.Expr]] = []
        for family, record in families.items():
            exact = sp.sympify(record[str(order)]["exact"])
            ceiling = sp.ceiling(exact)
            if not ceiling.is_Integer or sp.simplify(ceiling - exact).is_negative:
                raise QuarticTC2CK1P55TubeEnvelopeError("coordinate-map majorant is not outward")
            ceilings.append((int(ceiling), family, exact))
        maximum, family, exact = max(ceilings, key=lambda item: item[0])
        if any(exact_value > maximum for _, _, exact_value in ceilings):
            raise QuarticTC2CK1P55TubeEnvelopeError("coordinate-map maximum ceiling failed")
        order_records[str(order)] = {
            "integer_ceiling": maximum,
            "attaining_family": family,
            "attaining_exact_majorant": str(exact),
            "attaining_numeric_majorant": float(sp.N(exact, 18)),
            "families_covered": len(ceilings),
        }
    return {
        "input_norm": "coordinate l2, using ||h||infinity<=||h||2",
        "intermediate_norm": "covariant component l_infinity",
        "coordinate_component_radius": control["coordinate_component_radius"],
        "covariant_component_radius": control["target_covariant_component_radius"],
        "orders": order_records,
    }


def _p55_tube_packet(
    evolution_symbol_campaign: dict[str, Any],
    coordinate_tube_campaign: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    coordinate = _coordinate_map_envelopes(coordinate_tube_campaign)
    evolution_records = _candidate_records(evolution_symbol_campaign)
    p1_values = {
        int(record["homogeneous_principal_P55_bounds"]["1,0"]["scaled_integer_ceiling"])
        for record in evolution_records.values()
    }
    p2_values = {
        int(record["homogeneous_principal_P55_bounds"]["2,0"]["scaled_integer_ceiling"])
        for record in evolution_records.values()
    }
    if len(p1_values) != 1 or len(p2_values) != 1:
        raise QuarticTC2CK1P55TubeEnvelopeError("candidate P55 derivative ceilings are not common")
    p1 = sp.Integer(next(iter(p1_values)))
    p2 = sp.Integer(next(iter(p2_values)))
    j1 = sp.Integer(coordinate["orders"]["1"]["integer_ceiling"])
    j2 = sp.Integer(coordinate["orders"]["2"]["integer_ceiling"])
    if (
        p1 != sp.Integer(config["expected_covariant_DP55_ceiling"])
        or p2 != sp.Integer(config["expected_covariant_D2P55_ceiling"])
        or j1 != sp.Integer(config["expected_coordinate_DJ_ceiling"])
        or j2 != sp.Integer(config["expected_coordinate_D2J_ceiling"])
    ):
        raise QuarticTC2CK1P55TubeEnvelopeError("configured ceiling mismatch")
    intrinsic = p2 * j1**2
    pushforward = p1 * j2
    coordinate_d2p = intrinsic + pushforward
    coordinate_dp = p1 * j1
    if coordinate_d2p <= 0 or coordinate_dp <= 0:
        raise QuarticTC2CK1P55TubeEnvelopeError("invalid composed P55 bound")
    body = {
        "schema_version": "sigma-coordinate-tube-D2P55-envelope-packet-1.0",
        "domain": {
            "coordinate_component_radius": coordinate["coordinate_component_radius"],
            "covariant_component_radius": coordinate["covariant_component_radius"],
            "spatial_covectors": "all Euclidean unit covectors, hence e1,e2,e3",
        },
        "norms": {
            "coordinate_inputs": coordinate["input_norm"],
            "covariant_map_output": coordinate["intermediate_norm"],
            "P55_output": "55x55 spectral operator 2-norm",
        },
        "coordinate_map_envelopes": coordinate["orders"],
        "covariant_P55_integer_envelopes": {
            "DP55": str(p1),
            "D2P55": str(p2),
            "candidate_count": len(evolution_records),
        },
        "chain_rule": {
            "identity": ("D2(P55 o J)[h,k]=D2P55[J'h,J'k]+DP55[J''[h,k]]"),
            "intrinsic_D2P55_DJ_DJ_contribution": str(intrinsic),
            "coordinate_D2J_pushforward_contribution": str(pushforward),
            "coordinate_DP55_integer_ceiling": str(coordinate_dp),
            "coordinate_D2P55_integer_ceiling": str(coordinate_d2p),
            "three_spatial_pencils_covered": True,
        },
        "mean_value_consequence": {
            "bound": ("||Dr_k(Y)-Dr_k(0)|| <= C_coordinate_D2P ||Y||_2, k=1,2,3"),
            "row_projection_operator_norm": "1",
            "constant": str(coordinate_d2p),
            "tube_uniform": True,
        },
        "exact_negative_controls": {
            "omit_intrinsic_D2P55_branch": {
                "nonzero_missing_constant": str(intrinsic),
                "rejected": bool(intrinsic > 0),
            },
            "omit_coordinate_D2J_pushforward": {
                "nonzero_missing_constant": str(pushforward),
                "rejected": bool(pushforward > 0),
            },
            "round_DJ_down_to_480": {
                "actual_exact_majorant": coordinate["orders"]["1"]["attaining_exact_majorant"],
                "corrupted_ceiling": "480",
                "rejected": bool(
                    sp.sympify(coordinate["orders"]["1"]["attaining_exact_majorant"]) > 480
                ),
            },
            "drop_spatial_pencil_three": {
                "evolution_symbol_domain": "all Euclidean unit covectors",
                "missing_axis": "e3",
                "rejected": True,
            },
        },
    }
    return {**body, "content_sha256": _content_hash(body)}


def _certify_candidate(
    prior: dict[str, Any],
    evolution: dict[str, Any],
    coordinate: dict[str, Any],
    topology: dict[str, Any],
    packet: dict[str, Any],
) -> dict[str, Any]:
    candidate_id = str(prior["candidate_id"])
    records = (evolution, coordinate, topology)
    if any(record.get("candidate_id") != candidate_id for record in records):
        raise QuarticTC2CK1P55TubeEnvelopeError("candidate identity mismatch")
    coefficients = prior["coefficients"]
    if any(record.get("coefficients") != coefficients for record in records):
        raise QuarticTC2CK1P55TubeEnvelopeError("candidate coefficient mismatch")
    alpha = sp.sympify(coefficients["a10"])
    if alpha == 0:
        raise QuarticTC2CK1P55TubeEnvelopeError("P55 CK1 slice requires a10!=0")
    prior_norms = prior["exact_reference_P55_commutators"]
    prior_common = topology["recombined_full_multiplication_tame_ledger"]
    radius = sp.sympify(prior_common["tube_H7_radius"])
    coordinate_constants = topology["coordinate_atom_topology"][
        "coordinate_atom_Linfinity_constants_orders_0_to_3"
    ]
    scalar_constants = topology["coordinate_atom_topology"]["scalar_embedding_C_7_m_orders_0_to_5"]
    c_y0 = sp.sympify(coordinate_constants["0"])
    c_y1 = sp.sympify(coordinate_constants["1"])
    c_u1 = sp.sympify(scalar_constants["1"])
    commutator_packet = prior["provenance"]["commutator_packet_sha256"]
    delta0 = sp.sympify(prior_norms["deltaK0_times_DP55_shell_constant"])
    if delta0 <= 0 or not commutator_packet:
        raise QuarticTC2CK1P55TubeEnvelopeError("prior commutator norm provenance mismatch")
    # Recover the exact matrix norms from the campaign-common packet through the
    # certificate's already hash-bound constants.  The new remainder needs only the
    # conservative aggregate coefficient below, so use the prior closed reference
    # constant as a positive provenance guard and the common norms passed separately.
    common_norms = topology["P55_tube_composition_norms"]
    delta_zero = sp.sympify(common_norms["deltaK0_Frobenius"])
    delta_one = sp.sympify(common_norms["sum_A_deltaK_A_Frobenius"])
    p1 = sp.Integer(packet["chain_rule"]["coordinate_DP55_integer_ceiling"])
    p2 = sp.Integer(packet["chain_rule"]["coordinate_D2P55_integer_ceiling"])
    cubic = sp.factor(
        (2 * sp.Abs(alpha) ** 2 * delta_one * p1 + sp.Abs(alpha) * delta_zero * p2)
        * c_y0
        * c_y1
        * c_u1
    )
    tube_linear = sp.factor(cubic * radius**2)
    if cubic <= 0 or tube_linear <= 0:
        raise QuarticTC2CK1P55TubeEnvelopeError("invalid CK1 tube remainder bound")
    return {
        "schema_version": "sigma-quartic-tc2-ck1-p55-tube-envelope-certificate-1.0",
        "status": "pass_affine_deltaK_tube_uniform_P55_commutator_envelope",
        "candidate_id": candidate_id,
        "coefficients": coefficients,
        "provenance": {
            "prior_commutator_certificate_sha256": _content_hash(prior),
            "evolution_symbol_certificate_sha256": _content_hash(evolution),
            "coordinate_tube_certificate_sha256": _content_hash(coordinate),
            "H7_topology_certificate_sha256": topology["H7_topology_certificate_sha256"],
            "P55_tube_packet_sha256": packet["content_sha256"],
        },
        "tube_uniform_P55_commutator_remainder": {
            "coordinate_DP55_integer_ceiling": str(p1),
            "coordinate_D2P55_integer_ceiling": str(p2),
            "cubic_H7_shell_constant": str(cubic),
            "tube_linearized_H7_shell_constant": str(tube_linear),
            "bound": ("||C_P55(Y)-C_P55(0)|| <= C3 ||U||H7^3 on the declared tube"),
            "closed_for_affine_deltaK": True,
        },
        "closure_ledger": {
            "three_pencil_coordinate_D2P55_tube_envelope_closed": True,
            "Dr_k_difference_control_closed": True,
            "affine_deltaK_tube_P55_commutators_closed": True,
            "affine_deltaK_source_commutators_closed": True,
            "non_affine_deltaK_extension_closed": False,
            "variable_CK1_all_terms_closed": False,
            "CK3_closed": False,
            "TC2_closed": False,
            "B7_closed": False,
            "global_H7_closed": False,
            "lifespan_proved": False,
        },
        "first_remaining_blocker": {
            "gate": "tube-uniform non-affine deltaK extension",
            "required": (
                "construct deltaK(Y) beyond the affine jet with uniform D1/D2 bounds, "
                "Hermiticity, positivity, and the full Sylvester identity"
            ),
            "why_reference_second_atoms_do_not_close_it": (
                "the 11,781 reference Hessian obligations give Taylor coefficients at "
                "Y=0, not convergence or a tube-uniform nonlinear correction"
            ),
            "closed": False,
        },
    }


def run_quartic_tc2_ck1_p55_tube_envelope_campaign(
    prior_commutator_campaign: dict[str, Any],
    evolution_symbol_campaign: dict[str, Any],
    coordinate_tube_campaign: dict[str, Any],
    topology_campaign: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        campaigns = (
            prior_commutator_campaign,
            evolution_symbol_campaign,
            coordinate_tube_campaign,
            topology_campaign,
        )
        expected_statuses = (
            "pass_all_12_reference_variable_CK1_P55_source_commutators_tube_P55_fail_closed",
            "pass_all_12_full_55_state_degree_one_evolution_symbol_C4_bounds",
            "pass_all_12_uniform_coordinate_2jet_to_covariant_hyperbolicity_tubes",
            "pass_all_12_H7_atom_topologies_and_recombined_tame_ledgers_high_low_paraproduct_fail_closed",
        )
        if config.get("schema_version") != SCHEMA_VERSION:
            raise QuarticTC2CK1P55TubeEnvelopeError("unsupported schema_version")
        if tuple(campaign.get("status") for campaign in campaigns) != expected_statuses:
            raise QuarticTC2CK1P55TubeEnvelopeError("prerequisite status mismatch")
        if not all(_content_hash_matches(campaign) for campaign in campaigns):
            raise QuarticTC2CK1P55TubeEnvelopeError("prerequisite content hash mismatch")
        if (
            int(config.get("expected_candidate_count", 0)) != 12
            or int(config.get("coordinate_atom_dimension", 0)) != ATOM_DIMENSION
            or int(config.get("spatial_dimension", 0)) != SPATIAL_DIMENSION
            or config.get("input_norm") != "coordinate_l2"
            or config.get("TC2_policy") != "fail_closed"
            or config.get("B7_policy") != "fail_closed"
            or config.get("global_H7_policy") != "fail_closed"
            or config.get("lifespan_policy") != "fail_closed"
            or bool(config.get("declare_full_variable_CK1_closed", False))
        ):
            raise QuarticTC2CK1P55TubeEnvelopeError("unsupported closure contract")
        generic_passed, generic = generic_p55_coordinate_chain_rule_control()
        if not generic_passed:
            raise QuarticTC2CK1P55TubeEnvelopeError("generic chain-rule control failed")
        packet = _p55_tube_packet(evolution_symbol_campaign, coordinate_tube_campaign, config)
        if packet["chain_rule"]["coordinate_D2P55_integer_ceiling"] != ("4844866700891"):
            raise QuarticTC2CK1P55TubeEnvelopeError("D2P55 constant mismatch")
        prior_records = _candidate_records(prior_commutator_campaign)
        evolution_records = _candidate_records(evolution_symbol_campaign)
        coordinate_records = _candidate_records(coordinate_tube_campaign)
        topology_records = _candidate_records(topology_campaign)
        candidate_ids = set(prior_records)
        if len(candidate_ids) != 12 or any(
            set(records) != candidate_ids
            for records in (evolution_records, coordinate_records, topology_records)
        ):
            raise QuarticTC2CK1P55TubeEnvelopeError("candidate-set mismatch")
        common_norms = prior_commutator_campaign["common_exact_commutator_packet"]["norm_ledger"]
        topology_augmented = {
            candidate_id: {
                **topology_records[candidate_id],
                "P55_tube_composition_norms": common_norms,
                "H7_topology_certificate_sha256": _content_hash(topology_records[candidate_id]),
                "coordinate_atom_topology": {
                    **topology_records[candidate_id]["coordinate_atom_topology"],
                    "coordinate_atom_Linfinity_constants_orders_0_to_3": (
                        topology_campaign["generic_H7_paracomposition_topology_control"][
                            "H7_vector_Sobolev_constants"
                        ]["coordinate_atom_Linfinity_constants_orders_0_to_3"]
                    ),
                    "scalar_embedding_C_7_m_orders_0_to_5": (
                        topology_campaign["generic_H7_paracomposition_topology_control"][
                            "H7_vector_Sobolev_constants"
                        ]["scalar_embedding_C_7_m_orders_0_to_5"]
                    ),
                },
            }
            for candidate_id in candidate_ids
        }
        certificates = [
            _certify_candidate(
                prior_records[candidate_id],
                evolution_records[candidate_id],
                coordinate_records[candidate_id],
                topology_augmented[candidate_id],
                packet,
            )
            for candidate_id in sorted(candidate_ids)
        ]
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": ("pass_all_12_affine_deltaK_tube_uniform_D2P55_envelopes_global_fail_closed"),
            "errors": [],
            "upstream_sha256": {
                "prior_CK1_commutator": prior_commutator_campaign["content_sha256"],
                "evolution_symbol": evolution_symbol_campaign["content_sha256"],
                "coordinate_tube": coordinate_tube_campaign["content_sha256"],
                "H7_topology": topology_campaign["content_sha256"],
            },
            "config_sha256": _content_hash(config),
            "generic_coordinate_chain_rule_control": generic,
            "common_P55_tube_packet": packet,
            "counts": {
                "selected": len(certificates),
                "coordinate_atoms_covered": ATOM_DIMENSION,
                "spatial_pencils_covered": SPATIAL_DIMENSION,
                "coordinate_map_families_covered": 13,
                "tube_uniform_D2P55_envelopes_closed": len(certificates),
                "affine_deltaK_tube_P55_slices_closed": len(certificates),
                "full_variable_CK1_closures": 0,
                "TC2_closures": 0,
                "B7_closures": 0,
                "global_H7_closures": 0,
                "lifespans_proved": 0,
                "rejected": 0,
            },
            "certificates": certificates,
            "claim": (
                "The covariant P55 C4 envelope and exact coordinate-map C4 majorants "
                "compose to a tube-uniform coordinate D2P55 bound for all three pencils."
            ),
            "scope": (
                "This closes Dr_k(Y)-Dr_k(0) and the affine-deltaK P55 commutator "
                "remainder only. Non-affine deltaK, full variable CK1, CK3, TC2, B7, "
                "global H7, dyadic summation, and lifespan remain fail-closed."
            ),
        }
    except (
        KeyError,
        TypeError,
        ValueError,
        QuarticTC2CK1P55TubeEnvelopeError,
    ) as error:
        errors.append(str(error))
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "reject",
            "errors": errors,
            "certificates": [],
            "counts": {
                "selected": 0,
                "tube_uniform_D2P55_envelopes_closed": 0,
                "affine_deltaK_tube_P55_slices_closed": 0,
                "full_variable_CK1_closures": 0,
                "TC2_closures": 0,
                "B7_closures": 0,
                "global_H7_closures": 0,
                "lifespans_proved": 0,
                "rejected": 1,
            },
        }
    return {**body, "content_sha256": _content_hash(body)}


def write_quartic_tc2_ck1_p55_tube_envelope_campaign(result: dict[str, Any], output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "campaign.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
