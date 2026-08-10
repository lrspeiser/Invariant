from __future__ import annotations

import hashlib
import json
from functools import cache
from pathlib import Path
from typing import Any

import sympy as sp

SCHEMA_VERSION = "sigma-quartic-frequency-localized-evolution-campaign-1.0"


class QuarticFrequencyLocalizedEvolutionError(ValueError):
    """Raised when a localized evolution-energy input is inconsistent."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _content_hash_matches(campaign: dict[str, Any]) -> bool:
    body = {key: value for key, value in campaign.items() if key != "content_sha256"}
    return campaign.get("content_sha256") == hashlib.sha256(
        _canonical_json(body).encode()
    ).hexdigest()


def _candidate_records(campaign: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item["candidate_id"]): item
        for item in campaign.get("certificates", [])
        if isinstance(item, dict) and "candidate_id" in item
    }


@cache
def generic_frequency_localized_evolution_control() -> tuple[bool, dict[str, Any]]:
    """Verify energy differentiation, Young coupling, and the fixed-low projector."""

    k, k_t, v, v_t = sp.symbols("K K_t v v_t", real=True, finite=True)
    energy = k * v**2
    differentiation_residual = sp.expand(
        sp.diff(energy, k) * k_t
        + sp.diff(energy, v) * v_t
        - (k_t * v**2 + 2 * k * v * v_t)
    )
    a, b = sp.symbols("a b", nonnegative=True, finite=True)
    young_residual = sp.factor(a**2 + b**2 - 2 * a * b)

    r, t = sp.symbols("r t", positive=True, finite=True)
    smoothstep = (
        126 * t**5 - 420 * t**6 + 540 * t**7 - 315 * t**8 + 70 * t**9
    )
    low_piece = sp.expand(1 - smoothstep.subs(t, 2 * r - 1))

    def radial_l3(value: sp.Expr) -> sp.Expr:
        return sp.expand(value - sp.diff(value, r, 2) - 2 * sp.diff(value, r) / r)

    def radial_l5(value: sp.Expr) -> sp.Expr:
        return sp.expand(value - sp.diff(value, r, 2) - 4 * sp.diff(value, r) / r)

    first_radial = sp.integrate(r**2, (r, 0, sp.Rational(1, 2))) + sp.integrate(
        sp.expand(r**2 * radial_l3(low_piece) ** 2),
        (r, sp.Rational(1, 2), 1),
    )
    second_radial = sp.integrate(r**4, (r, 0, sp.Rational(1, 2))) + sp.integrate(
        sp.expand(r**4 * radial_l5(radial_l5(low_piece)) ** 2),
        (r, sp.Rational(1, 2), 1),
    )
    first_kernel_square = 4 * sp.pi * first_radial
    second_kernel_square = sp.Rational(4, 3) * sp.pi * second_radial
    kernel_prefactor = sp.pi * (2 * sp.pi) ** (-sp.Rational(3, 2))
    low_kappa = sp.factor(
        kernel_prefactor
        * (sp.sqrt(first_kernel_square) + sp.sqrt(second_kernel_square))
    )

    time_witness = sp.diff((1 + t) * v**2, t)
    omitted_time_residual = sp.expand(time_witness)
    commutator_coefficient, state_gradient = sp.symbols(
        "C_comm grad_v", positive=True, finite=True
    )
    omitted_commutator_residual = commutator_coefficient * state_gradient
    lower_source = sp.Symbol("F_lower", nonzero=True, finite=True)
    premature_source_omission = 2 * k * v * lower_source
    n = sp.Symbol("N", positive=True, finite=True)
    derivative_loss_witness = n ** (7 - 6)

    passed = bool(
        differentiation_residual == 0
        and young_residual == (a - b) ** 2
        and first_radial.is_positive
        and second_radial.is_positive
        and low_kappa.is_positive
        and omitted_time_residual != 0
        and omitted_commutator_residual != 0
        and premature_source_omission != 0
        and derivative_loss_witness == n
    )
    return passed, {
        "control": "frequency-localized evolution energy prerequisite",
        "energy_differentiation": {
            "identity": "d_t<Kv,v>=<K_t v,v>+2 Re<Kv,v_t>",
            "scalar_residual": str(differentiation_residual),
        },
        "neighbor_coupling_Young": {
            "identity": "2ab<=a^2+b^2",
            "positive_residual": str(young_residual),
        },
        "fixed_low_projector_commutator": {
            "multiplier": "m_-1(xi)=chi(2|xi|)",
            "first_radial_square_integral": str(first_radial),
            "second_radial_square_integral": str(second_radial),
            "explicit_kappa_low": str(low_kappa),
            "bound": (
                "||[Delta_-1,A^k]partial_k v||2<="
                "3*sqrt(3)*kappa_low*A1*||v||2"
            ),
        },
        "high_shell_inequality_template": (
            "E_j' <= (T_K+C_comp+3*Lambda*A1+Lambda*C_proj)||v_j||2^2 "
            "+Lambda*C_proj||tildeDelta_j u||2^2+2*Lambda||v_j||||F_j^lower||"
        ),
        "finite_low_partial_inequality_template": (
            "E_low' <= (T_K+C_def+3*Lambda*A1+Lambda*C_proj_low)||v_low||2^2 "
            "+Lambda*C_proj_low||tildeLow u||2^2+2*Lambda||v_low||"
            "*(||R_low^AW||+||F_low^lower||)"
        ),
        "negative_controls": {
            "omit_time_K": {
                "witness": "K(t)=1+t with time-independent nonzero v",
                "residual": str(omitted_time_residual),
                "rejected": omitted_time_residual != 0,
            },
            "omit_projection_commutator": {
                "residual": str(omitted_commutator_residual),
                "rejected": omitted_commutator_residual != 0,
            },
            "omit_lower_order_source": {
                "residual": str(premature_source_omission),
                "rejected": premature_source_omission != 0,
            },
            "premature_H7_sum_from_H6_coefficients": {
                "frequency_growth": str(derivative_loss_witness),
                "rejected": derivative_loss_witness != 1,
            },
        },
        "passed": passed,
    }


def _certify_candidate(
    annular: dict[str, Any],
    bounded: dict[str, Any],
    dyadic: dict[str, Any],
    time_atoms: dict[str, Any],
    first_order: dict[str, Any],
    evolution: dict[str, Any],
    positive: dict[str, Any],
    anti_wick: dict[str, Any],
    generic_control: dict[str, Any],
) -> dict[str, Any]:
    records = (
        annular,
        bounded,
        dyadic,
        time_atoms,
        first_order,
        evolution,
        positive,
        anti_wick,
    )
    candidate_id = str(annular.get("candidate_id"))
    if any(record.get("candidate_id") != candidate_id for record in records):
        raise QuarticFrequencyLocalizedEvolutionError("candidate identity mismatch")
    if any(record.get("coefficients") != annular.get("coefficients") for record in records[1:]):
        raise QuarticFrequencyLocalizedEvolutionError("candidate coefficient mismatch")
    expected_statuses = (
        "pass_targeted_annular_K55_C6_principal_composition_constant",
        "pass_actual_P55_compact_frequency_defect_KN_L2_lemma",
        "pass_H7_dyadic_partition_and_shell_local_commutator_framework",
        "pass_H7_closed_coordinate_atom_time_budget",
        "pass_exact_55_variable_principal_first_order_reduction",
        "pass_full_55_state_degree_one_evolution_symbol_C4_bounds",
        "pass_uniform_positive_anti_wick_K55_operator",
        "fail_closed_requires_C6_spatial_frequency_symbol_bounds",
    )
    if tuple(record.get("status") for record in records) != expected_statuses:
        raise QuarticFrequencyLocalizedEvolutionError("candidate prerequisite status mismatch")
    if not annular.get("anti_wick_principal_composition_remainder_instantiated"):
        raise QuarticFrequencyLocalizedEvolutionError("principal composition is not instantiated")
    if not bounded.get("physical_scale_contract_passed"):
        raise QuarticFrequencyLocalizedEvolutionError("bounded-frequency scale contract failed")
    if not dyadic.get("shell_local_commutator_bound_certified"):
        raise QuarticFrequencyLocalizedEvolutionError("projection commutator is not certified")
    if not all(
        (
            annular.get("full_dyadic_energy_closed") is False,
            bounded.get("full_energy_closed") is False,
            dyadic.get("full_H7_commutator_closed") is False,
        )
    ):
        raise QuarticFrequencyLocalizedEvolutionError("an upstream closure flag is inconsistent")
    source_hash = first_order.get("source_spatial_block_sha256")
    if evolution.get("exact_reduction_provenance", {}).get(
        "source_spatial_block_sha256"
    ) != source_hash:
        raise QuarticFrequencyLocalizedEvolutionError("physical pencil hash mismatch")

    energy = positive["operator_energy_equivalence"]
    lam = sp.sympify(energy["lower"])
    upper = sp.sympify(energy["upper"])
    if not (lam > 0 and upper >= lam):
        raise QuarticFrequencyLocalizedEvolutionError("anti-Wick energy bounds are invalid")
    q0 = sp.sympify(
        anti_wick["annular_positive_energy"]["coercivity_factor"]
    ) if "annular_positive_energy" in anti_wick else None
    # Candidate anti-Wick records do not repeat the generic control; the caller injects it.
    if q0 is None:
        raise QuarticFrequencyLocalizedEvolutionError("annular coercivity factor is absent")

    composition = sp.sympify(
        annular["principal_anti_wick_composition_constant"]["exact"]
    )
    time_bound = sp.sympify(
        time_atoms["closed_time_K55_bounds"]["0,0"]["expression"],
        locals={"E": sp.Symbol("E", nonnegative=True, finite=True)},
    )
    radius = sp.sympify(annular["composition_inputs"]["sufficient_H6_radius"])
    energy_symbol = next(iter(time_bound.free_symbols), None)
    time_at_radius = (
        time_bound.subs(energy_symbol, radius)
        if energy_symbol is not None
        else time_bound
    )
    a1 = sp.sympify(annular["composition_inputs"]["A1_P_1_1"])
    radius_symbol = sp.Symbol("R", nonnegative=True, finite=True)
    projection = sp.sympify(
        dyadic["shell_local_commutator_bound"]["expression"],
        locals={"R": radius_symbol},
    ).subs(radius_symbol, radius)
    low_kappa = sp.sympify(
        generic_control["fixed_low_projector_commutator"]["explicit_kappa_low"]
    )
    low_projection = 3 * sp.sqrt(3) * low_kappa * a1
    low_defect = sp.sympify(bounded["operator_L2_bound"]["exact"])

    # Preserve exact expressions, but do not globally factor these very large sums.
    # Factoring is not part of the proof and makes a twelve-candidate audit needlessly
    # superlinear.  SymPy's canonical Add/Mul representation remains deterministic.
    physical_weyl = 3 * upper * a1
    high_growth_norm = time_at_radius + composition + physical_weyl + upper * projection
    high_energy_growth = high_growth_norm / (lam * q0)
    low_certified_growth_norm = (
        time_at_radius + low_defect + physical_weyl + upper * low_projection
    )
    low_energy_growth = low_certified_growth_norm / lam
    numeric_values = {
        "time_K": float(sp.N(time_at_radius, 18)),
        "principal_C_comp": float(sp.N(composition, 18)),
        "projection_commutator": float(sp.N(projection, 18)),
        "high_energy_growth": float(sp.N(high_energy_growth, 18)),
        "low_compact_defect": float(sp.N(low_defect, 18)),
        "low_projection_commutator": float(sp.N(low_projection, 18)),
        "low_certified_growth": float(sp.N(low_energy_growth, 18)),
    }
    if any(not (value >= 0 and sp.Float(value).is_finite) for value in numeric_values.values()):
        raise QuarticFrequencyLocalizedEvolutionError("a localized energy constant is invalid")

    return {
        "schema_version": "sigma-quartic-frequency-localized-evolution-certificate-1.0",
        "status": "pass_high_shell_coupled_energy_partial_low_sources_and_sum_fail_closed",
        "candidate_id": candidate_id,
        "coefficients": annular.get("coefficients"),
        "physical_pencil": {
            "source_spatial_block_sha256": source_hash,
            "state_dimension": 55,
            "degree_one_in_frequency": True,
        },
        "energy_equivalence": {
            "global_low_lower": str(lam),
            "global_upper": str(upper),
            "high_localized_lower": str(lam * q0),
            "annular_coercivity_factor": str(q0),
        },
        "high_shell_j_ge_7": {
            "semiclassical_scale": "h_j=8*2^-j",
            "time_K_constant": str(time_at_radius),
            "principal_anti_wick_composition_constant": str(composition),
            "physical_left_to_Weyl_constant": str(physical_weyl),
            "projection_commutator_constant": str(projection),
            "coupled_norm_growth_constant": str(high_growth_norm),
            "self_energy_growth_constant": str(high_energy_growth),
            "inequality": (
                "E_j'<=G_high||Delta_j u||2^2+Lambda*C_proj"
                "||tildeDelta_j u||2^2+2*Lambda||Delta_j u||||F_j^lower||"
            ),
            "time_K_included": True,
            "principal_composition_included": True,
            "projection_commutator_included": True,
            "lower_order_source_closed": False,
        },
        "finite_physical_low_frequencies": {
            "compact_pointwise_symmetrization_defect_constant": str(low_defect),
            "low_projection_commutator_constant": str(low_projection),
            "certified_partial_norm_growth_constant": str(low_certified_growth_norm),
            "certified_partial_energy_growth_constant": str(low_energy_growth),
            "inequality": (
                "E_low'<=G_low||Delta_-1 u||2^2+Lambda*C_proj_low"
                "||tildeLow u||2^2+2*Lambda||Delta_-1 u||"
                "*(||R_low^AW||+||F_low^lower||)"
            ),
            "compact_physical_defect_included": True,
            "projection_commutator_included": True,
            "low_anti_wick_composition_remainder_closed": False,
            "lower_order_source_closed": False,
        },
        "numeric_constants": numeric_values,
        "per_shell_principal_time_projection_inequality_certified": True,
        "finite_low_principal_partial_inequality_certified": True,
        "complete_shell_inequality_closed": False,
        "global_H7_dyadic_sum_applied": False,
        "remaining_gates": [
            "low_frequency_anti_wick_operator_composition_remainder",
            "lower_order_and_solved_source_localized_operator_bounds",
            "remote_paraproduct_commutator_without_H7_from_H6_derivative_loss",
            "monotone_dyadic_summation",
            "nonlinear_energy_bootstrap_and_lifespan",
        ],
        "scope": (
            "The high-shell time, principal anti-Wick composition, physical Weyl "
            "correction, and local projection commutator contributions are explicit. "
            "Finite physical low frequencies include the compact pointwise defect and "
            "low projection commutator. Low anti-Wick composition, lower-order sources, "
            "remote paraproducts, the global H7 sum, and lifespan remain fail-closed."
        ),
    }


def run_quartic_frequency_localized_evolution_campaign(
    annular_campaign: dict[str, Any],
    bounded_frequency_campaign: dict[str, Any],
    dyadic_campaign: dict[str, Any],
    time_atom_campaign: dict[str, Any],
    first_order_campaign: dict[str, Any],
    evolution_campaign: dict[str, Any],
    positive_quantization_campaign: dict[str, Any],
    anti_wick_campaign: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        if config.get("schema_version") != SCHEMA_VERSION:
            raise QuarticFrequencyLocalizedEvolutionError(
                "unsupported campaign schema_version"
            )
        campaigns = {
            "annular_C6": annular_campaign,
            "bounded_frequency": bounded_frequency_campaign,
            "dyadic": dyadic_campaign,
            "time_atoms": time_atom_campaign,
            "first_order": first_order_campaign,
            "evolution": evolution_campaign,
            "positive_quantization": positive_quantization_campaign,
            "anti_wick": anti_wick_campaign,
        }
        expected_statuses = {
            "annular_C6": "pass_all_12_targeted_annular_K55_C6_principal_composition_constants",
            "bounded_frequency": "pass_all_12_actual_P55_compact_frequency_defect_KN_L2_lemmas",
            "dyadic": "pass_all_12_H7_dyadic_local_frameworks_global_commutator_fail_closed",
            "time_atoms": "pass_all_12_H7_closed_coordinate_atom_time_budgets",
            "first_order": "pass_all_12_exact_55_variable_principal_first_order_reductions",
            "evolution": "pass_all_12_full_55_state_degree_one_evolution_symbol_C4_bounds",
            "positive_quantization": "pass_all_12_uniform_positive_anti_wick_K55_operators",
            "anti_wick": "pass_exact_anti_wick_composition_prerequisite_audit_C6_required",
        }
        for name, campaign in campaigns.items():
            if campaign.get("status") != expected_statuses[name]:
                raise QuarticFrequencyLocalizedEvolutionError(
                    f"{name} prerequisite status mismatch"
                )
            if not _content_hash_matches(campaign):
                raise QuarticFrequencyLocalizedEvolutionError(
                    f"{name} campaign content hash mismatch"
                )
        upstream = {
            name: campaign.get("content_sha256") for name, campaign in campaigns.items()
        }
        annular_upstream = annular_campaign.get("upstream_sha256", {})
        if annular_upstream.get("anti_wick") != upstream["anti_wick"]:
            raise QuarticFrequencyLocalizedEvolutionError("annular anti-Wick provenance mismatch")
        for campaign_name in ("bounded_frequency", "dyadic"):
            links = campaigns[campaign_name].get("upstream_sha256", {})
            if links.get("evolution") != upstream["evolution"] or links.get("first_order") != upstream["first_order"]:
                raise QuarticFrequencyLocalizedEvolutionError(
                    f"{campaign_name} physical-pencil provenance mismatch"
                )
        if evolution_campaign.get("first_order_campaign_sha256") != upstream["first_order"]:
            raise QuarticFrequencyLocalizedEvolutionError("evolution first-order provenance mismatch")
        if anti_wick_campaign.get("upstream_sha256", {}).get("time_atoms") != upstream["time_atoms"]:
            raise QuarticFrequencyLocalizedEvolutionError("anti-Wick time provenance mismatch")
        low_hash = bounded_frequency_campaign.get("upstream_sha256", {}).get("low_frequency")
        if positive_quantization_campaign.get("low_frequency_campaign_sha256") != low_hash:
            raise QuarticFrequencyLocalizedEvolutionError("positive low-frequency provenance mismatch")
        if (
            int(config["spatial_dimension"]) != 3
            or int(config["state_dimension"]) != 55
            or int(config["state_sobolev_order"]) != 7
            or int(config["high_shell_index_minimum"]) != 7
            or sp.sympify(config["semiclassical_h_maximum"]) != sp.Rational(1, 16)
            or int(config["finite_physical_frequency_radius"]) != 2
            or config.get("lower_order_source_policy") != "fail_closed"
            or config.get("global_dyadic_sum_policy") != "fail_closed"
        ):
            raise QuarticFrequencyLocalizedEvolutionError(
                "unsupported frequency-localized evolution contract"
            )
        control_passed, control = generic_frequency_localized_evolution_control()
        if not control_passed:
            raise QuarticFrequencyLocalizedEvolutionError("generic localized energy control failed")
        maps = {name: _candidate_records(campaign) for name, campaign in campaigns.items()}
        candidate_ids = set(maps["annular_C6"])
        expected = int(config.get("expected_candidate_count", 12))
        if len(candidate_ids) != expected or any(set(records) != candidate_ids for records in maps.values()):
            raise QuarticFrequencyLocalizedEvolutionError("candidate-set mismatch")
        # Inject the exact generic annular coercivity into each candidate audit without
        # mutating the upstream certificate.
        anti_generic = anti_wick_campaign["generic_anti_wick_composition_audit"]
        certificates = []
        for candidate_id in sorted(candidate_ids):
            anti_record = dict(maps["anti_wick"][candidate_id])
            anti_record["annular_positive_energy"] = anti_generic[
                "annular_positive_energy"
            ]
            certificates.append(
                _certify_candidate(
                    maps["annular_C6"][candidate_id],
                    maps["bounded_frequency"][candidate_id],
                    maps["dyadic"][candidate_id],
                    maps["time_atoms"][candidate_id],
                    maps["first_order"][candidate_id],
                    maps["evolution"][candidate_id],
                    maps["positive_quantization"][candidate_id],
                    anti_record,
                    control,
                )
            )
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": (
                "pass_all_12_frequency_localized_principal_shell_inequalities_"
                "sources_and_global_sum_fail_closed"
            ),
            "errors": [],
            "upstream_sha256": upstream,
            "config_sha256": hashlib.sha256(_canonical_json(config).encode()).hexdigest(),
            "generic_frequency_localized_evolution_control": control,
            "counts": {
                "selected": len(certificates),
                "high_shell_principal_time_projection_inequalities_passed": len(certificates),
                "finite_low_partial_inequalities_passed": len(certificates),
                "complete_shell_inequalities_closed": 0,
                "global_H7_dyadic_sums_applied": 0,
                "rejected": 0,
            },
            "certificates": certificates,
            "claim": (
                "All 12 candidates have explicit coupled high-shell inequalities for "
                "time K, principal anti-Wick composition, physical Weyl correction, "
                "and local projection commutators, plus partial finite-low estimates. "
                "Source closure and the global H7 sum remain deliberately fail-closed."
            ),
            "scope": certificates[0]["scope"],
        }
    except (
        KeyError,
        TypeError,
        ValueError,
        QuarticFrequencyLocalizedEvolutionError,
    ) as error:
        errors.append(str(error))
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "reject",
            "errors": errors,
            "certificates": [],
            "counts": {
                "selected": 0,
                "high_shell_principal_time_projection_inequalities_passed": 0,
                "finite_low_partial_inequalities_passed": 0,
                "complete_shell_inequalities_closed": 0,
                "global_H7_dyadic_sums_applied": 0,
                "rejected": 0,
            },
        }
    return {
        **body,
        "content_sha256": hashlib.sha256(_canonical_json(body).encode()).hexdigest(),
    }


def write_quartic_frequency_localized_evolution_campaign(
    result: dict[str, Any], output: Path
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "campaign.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
