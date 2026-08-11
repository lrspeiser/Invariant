"""Fail-closed formula intake for Schlatter--Kastner transactional gravity."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path
from typing import Any

import sympy as sp

CONFIG_SCHEMA = "sigma-kastner-schlatter-transactional-gravity-intake-config-1.0"
RESULT_SCHEMA = "sigma-kastner-schlatter-transactional-gravity-intake-1.0"
SOURCE_PDF_SHA256 = "c2f671293d07b21397e745da00a3ce1a2193c00da647a2ebf4147612b76c1780"
FIRST_BLOCKER = "no_candidate_bound_fundamental_action_or_complete_variational_field_system"


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fraction(value: str) -> Fraction:
    return Fraction(value)


def _validate_config(config: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "campaign_id",
        "campaign_implementation",
        "primary_source",
        "intake_scope",
        "synthetic_vectors",
        "budget",
        "observational_authorization",
        "external_paid_llm_calls",
        "data_seals",
    }
    if set(config) != required or config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("transactional-gravity intake config is invalid")
    source = config["primary_source"]
    if source != {
        "arxiv_id": "2209.04025",
        "version": "v1",
        "title": "Gravity from Transactions: Fulfilling the Entropic Gravity Program",
        "authors": ["A. Schlatter", "R. E. Kastner"],
        "abstract_url": "https://arxiv.org/abs/2209.04025",
        "pdf_url": "https://arxiv.org/pdf/2209.04025v1",
        "pdf_sha256": SOURCE_PDF_SHA256,
        "page_count": 16,
    }:
        raise ValueError("primary source binding changed")
    if config["intake_scope"] != {
        "paper_equation_numbers": [
            33,
            34,
            35,
            38,
            39,
            42,
            44,
            45,
            55,
            59,
            60,
            62,
            65,
            68,
            69,
        ],
        "standard_poisson_pmf_is_implementation_not_paper_equation": True,
        "fundamental_action_requested": False,
        "observational_execution_requested": False,
        "cuda_execution_requested": False,
    }:
        raise ValueError("formula intake scope changed")
    if config["budget"] != {
        "maximum_formula_contracts": 8,
        "maximum_synthetic_checks": 8,
        "maximum_poisson_count": 8,
        "maximum_paid_llm_spend_usd": 0.0,
    }:
        raise ValueError("formula intake budget changed")
    if config["observational_authorization"] is not False:
        raise ValueError("observational execution was opened")
    if config["external_paid_llm_calls"] is not False:
        raise ValueError("paid LLM execution was opened")
    if config["data_seals"] != {
        "observational_data_opened": False,
        "dark_matter_or_halo_data_opened": False,
        "redshift_or_cosmology_data_opened": False,
        "solar_system_data_opened": False,
    }:
        raise ValueError("data seals changed")


def _equation_contracts() -> list[dict[str, Any]]:
    return [
        {
            "contract_id": "transaction_poisson_rate_and_pressure",
            "paper_pages": [9],
            "paper_equations": [33, 34],
            "paper_statement": (
                "Transaction counts are asserted to follow a Poisson stochastic process; "
                "a constant average rate q_gamma is transactions per four-volume."
            ),
            "equations_latex": [
                r"\bar\lambda(x_0)=\lim_{R\to0}N_R(x_0)/V_R",
                r"\Delta\bar\lambda/\Delta x_0=q_\gamma",
                r"\bar P_\gamma=-c h q_\gamma",
            ],
            "classification": "paper_proposal_with_qed_and_lorentz_invariance_assertions",
            "assumptions": [
                "constant average transaction rate",
                "rate is measured per four-volume",
                "small-volume and finite-difference limits in equation 33 exist",
                "the paper's cited Poisson/Lorentz-invariance premise applies",
                "smallest wavelength compatible with the time-energy inequality",
            ],
            "not_claimed": [
                "a microscopic rate derived from a registered action",
                "independence or stationarity beyond the stated constant-rate model",
            ],
        },
        {
            "contract_id": "standard_poisson_cuda_reference",
            "paper_pages": [9],
            "paper_equations": [],
            "equations_latex": [r"p(n\mid\mu)=e^{-\mu}\mu^n/n!"],
            "classification": "standard_implementation_of_paper_poisson_assertion_not_printed_equation",
            "assumptions": ["mu >= 0", "n is a nonnegative integer"],
            "not_claimed": ["the paper explicitly prints this PMF"],
        },
        {
            "contract_id": "transaction_cosmological_term",
            "paper_pages": [9, 10],
            "paper_equations": [35, 36],
            "equations_latex": [
                r"\Lambda=-(4\pi G/c^4)\bar P_\gamma=(4\pi G h/c^3)q_\gamma=4\pi^2l_P^2q_\gamma",
                r"R_{00}+\Lambda\delta_{00}=(4\pi G/c^4)T_{00}",
            ],
            "classification": "paper_derived_identification_with_explicit_premises",
            "assumptions": [
                r"the same page states l_P^2=G\hbar/c^3",
                "repulsive pressure sign convention",
                "homogeneous energy distribution near the origin",
            ],
            "normalization_note": (
                "The printed final equality needs clarification: with h=2*pi*hbar and "
                "l_P^2=G*hbar/c^3, its middle expression is 8*pi^2*l_P^2*q_gamma, "
                "not the printed 4*pi^2*l_P^2*q_gamma."
            ),
            "not_claimed": [
                "a measured value of Lambda",
                "dark-energy elimination",
                "cosmological perturbation equivalence",
            ],
        },
        {
            "contract_id": "einstein_trace_reversed_recovery_scope",
            "paper_pages": [9, 10],
            "paper_equations": [38, 39],
            "equations_latex": [
                r"R_{00}+\Lambda\delta_{00}=(8\pi G/c^4)(T_{00}-T\delta_{00}/2)",
                r"R_{\mu\nu}+\Lambda g_{\mu\nu}=(8\pi G/c^4)(T_{\mu\nu}-Tg_{\mu\nu}/2)",
            ],
            "classification": "paper_successive_generalization_not_action_equivalence_proof",
            "assumptions": [
                "local inertial/geodesic-normal coordinates around each point",
                "small ball of test systems and local volume-acceleration relation",
                "initial-rest calculation followed by local-rest-frame transformation",
                "known tensor transformation rules",
                "equation 38 holds in every local inertial frame around every point",
            ],
            "not_claimed": [
                "a covariant fundamental action",
                "Euler-Lagrange recovery",
                "global existence or boundary terms",
                "formal equivalence to general relativity",
            ],
        },
        {
            "contract_id": "schwarzschild_de_sitter_background",
            "paper_pages": [11],
            "paper_equations": [42, 44, 45],
            "equations_latex": [
                r"ds^2=f(r)c^2dt^2-f(r)^{-1}dr^2-r^2d\Omega^2",
                r"f(r)=1-r^2/R_0^2-R_S/r",
                r"R_0=\sqrt{3/\Lambda},\quad R_S=2GM/c^2",
                r"a_\infty=cH_0=c^2/R_0=c^2\sqrt{\Lambda/3},\quad a_0=a_\infty/2",
                r"a(r)=(r/R_0)a_0",
            ],
            "classification": "standard_solution_plus_paper_interpretive_use",
            "assumptions": [
                "static Schwarzschild-de Sitter chart",
                "the paper labels a(r) hypothetical in empty de Sitter space",
                "transactional interpretation denies physical pure-vacuum spacetime",
            ],
            "not_claimed": ["global chart coverage", "transactional ontology validation"],
        },
        {
            "contract_id": "sds_effective_entropy_quadratic",
            "paper_pages": [13],
            "paper_equations": [55, 59, 60],
            "equations_latex": [
                r"\bar S_{dS}(r)=S_{dS}(r)+\Delta S_{dS}(r)",
                r"(\bar a^2/a_0)S-\bar a|\Delta S_{dS}|-gS=0",
                r"\bar a=(a_0/2)(|\Delta S_{dS}|/S+\sqrt{(|\Delta S_{dS}|/S)^2+4g/a_0})",
            ],
            "classification": "paper_proposed_observer_consistency_relation",
            "assumptions": [
                "first-order weak-potential entropy change",
                "r is at or above the limit radius",
                "local clock/entropy products for two observers are equated",
                "effective entropy is positive",
            ],
            "not_claimed": ["derivation from a fundamental action", "unique relativistic completion"],
        },
        {
            "contract_id": "sds_mond_galaxy_relation",
            "paper_pages": [14, 15],
            "paper_equations": [62, 65, 68, 69],
            "equations_latex": [
                r"\bar a(r)=a_0|\Phi(r)|(1+\sqrt{1+c^4/(MGa_0)})",
                r"r>r_0=\sqrt{MG/a_0}",
                r"\bar a(r)\approx\sqrt{MGa_0}/r",
                r"v^2=\sqrt{MGa_0}",
            ],
            "classification": "paper_approximation_and_interpretive_mond_identification",
            "assumptions": [
                r"|\Phi|\ll1 and the paper's first-order entropy construction",
                r"r>r_0 and |\Phi|a_0\ll1",
                r"|\Phi|=MG/(c^2r)",
                r"\bar a=v^2/r for circular motion",
            ],
            "not_claimed": [
                "galaxy-data likelihood pass",
                "lensing or cluster agreement",
                "dark-matter elimination",
            ],
        },
        {
            "contract_id": "action_and_validation_absence_contract",
            "paper_pages": [15],
            "paper_equations": [],
            "equations_latex": [],
            "classification": "fail_closed_intake_boundary",
            "assumptions": [],
            "not_claimed": [
                "fundamental action",
                "formal GR equivalence",
                "dark-matter elimination",
                "dark-energy elimination",
                "observational pass",
                "theory validity",
            ],
        },
    ]


def _synthetic_checks(config: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    max_count = config["synthetic_vectors"]["poisson_max_count"]
    poisson_vectors = []
    recurrence_pass = True
    for mean_text in config["synthetic_vectors"]["poisson_means"]:
        mean = _fraction(mean_text)
        mu = float(mean)
        probabilities = [math.exp(-mu) * mu**n / math.factorial(n) for n in range(max_count + 1)]
        for n in range(max_count):
            recurrence_pass &= math.isclose(
                probabilities[n + 1] * (n + 1), probabilities[n] * mu, rel_tol=1e-15
            )
        poisson_vectors.append(
            {
                "mu": mean_text,
                "p_n_float64": [format(value, ".17g") for value in probabilities],
                "truncated_mass": format(sum(probabilities), ".17g"),
            }
        )
    checks.append(
        {
            "check_id": "poisson_float64_recurrence",
            "status": "pass" if recurrence_pass else "reject",
            "scope": "synthetic_numeric_implementation_only",
            "vectors": poisson_vectors,
        }
    )

    z = sp.symbols("z", positive=True)
    generating = sp.exp(z * (sp.symbols("t") - 1))
    t = next(symbol for symbol in generating.free_symbols if symbol.name == "t")
    mean_identity = sp.simplify(sp.diff(generating, t).subs(t, 1) - z)
    variance_identity = sp.simplify(
        (sp.diff(generating, t, 2) + sp.diff(generating, t)).subs(t, 1) - z**2 - z
    )
    checks.append(
        {
            "check_id": "poisson_symbolic_mean_variance",
            "status": "pass" if mean_identity == variance_identity == 0 else "reject",
            "scope": "standard_poisson_contract_not_paper_printed_equation",
            "mean_residual": str(mean_identity),
            "variance_residual": str(variance_identity),
        }
    )

    q, c, h_planck, hbar, grav, lp = sp.symbols(
        "q c h hbar G l_P", positive=True
    )
    pressure = -c * h_planck * q
    cosmological_from_pressure = -4 * sp.pi * grav * pressure / c**4
    cosmological_from_lp = 4 * sp.pi**2 * lp**2 * q
    pressure_middle_residual = sp.simplify(
        cosmological_from_pressure - 4 * sp.pi * grav * h_planck * q / c**3
    )
    printed_chain_residual = sp.simplify(
        cosmological_from_pressure.subs(h_planck, 2 * sp.pi * hbar).subs(
            hbar, lp**2 * c**3 / grav
        )
        - cosmological_from_lp
    )
    rate_vectors = []
    for rate in config["synthetic_vectors"]["transaction_rates"]:
        value = _fraction(rate)
        rate_vectors.append(
            {
                "q_gamma": rate,
                "normalized_pressure_c_equals_h_equals_1": str(-value),
                "normalized_lambda_lP_equals_1": format(4 * math.pi**2 * float(value), ".17g"),
            }
        )
    checks.append(
        {
            "check_id": "transaction_pressure_lambda_identity",
            "status": "block" if pressure_middle_residual == 0 and printed_chain_residual != 0 else "reject",
            "scope": "equation_34_35_algebra_and_same_page_planck_length_normalization",
            "pressure_to_middle_exact_residual": str(pressure_middle_residual),
            "printed_chain_exact_residual": str(printed_chain_residual),
            "first_missing_premise": "equation_35_h_vs_hbar_factor_normalization_clarification",
            "vectors": rate_vectors,
        }
    )

    dimension = sp.Integer(4)
    trace_reversed_rhs_trace = sp.simplify(1 - dimension / 2)
    standard_einstein_residual = sp.simplify(trace_reversed_rhs_trace + 1)
    checks.append(
        {
            "check_id": "einstein_trace_reversal_four_dimensions",
            "status": "pass" if standard_einstein_residual == 0 else "reject",
            "scope": "tensor_algebra_only_not_derivation_or_equivalence_proof",
            "rhs_trace_coefficient": str(trace_reversed_rhs_trace),
            "exact_residual": str(standard_einstein_residual),
        }
    )

    vector = config["synthetic_vectors"]["mond_quadratic"]
    a0 = sp.Rational(vector["a0"])
    entropy = sp.Rational(vector["entropy"])
    deficit = sp.Rational(vector["entropy_deficit_abs"])
    g = sp.Rational(vector["g"])
    acceleration = sp.simplify(
        a0 * (deficit / entropy + sp.sqrt((deficit / entropy) ** 2 + 4 * g / a0)) / 2
    )
    quadratic_residual = sp.simplify(
        acceleration**2 * entropy / a0 - acceleration * deficit - g * entropy
    )
    checks.append(
        {
            "check_id": "sds_entropy_quadratic_positive_root",
            "status": "pass" if quadratic_residual == 0 and acceleration > 0 else "reject",
            "scope": "equation_59_60_algebra_only",
            "positive_root": str(acceleration),
            "exact_residual": str(quadratic_residual),
        }
    )

    normalized = config["synthetic_vectors"]["sdsmond_normalized"]
    values = {key: sp.Rational(value) for key, value in normalized.items()}
    exact_62 = sp.simplify(
        values["a0"]
        * values["potential_abs"]
        * (
            1
            + sp.sqrt(
                1
                + values["c"] ** 4
                / (values["M"] * values["G"] * values["a0"])
            )
        )
    )
    checks.append(
        {
            "check_id": "sds_mond_main_formula_normalized",
            "status": "pass" if exact_62 == sp.Rational(1, 2) + sp.sqrt(2) / 2 else "reject",
            "scope": "equation_62_synthetic_evaluation_only",
            "exact_value": str(exact_62),
            "float64_reference": format(float(exact_62), ".17g"),
        }
    )

    mass, newton, scale = sp.symbols("M G a_0", positive=True)
    radius = sp.symbols("r", positive=True)
    deep_acceleration = sp.sqrt(mass * newton * scale) / radius
    velocity_squared = sp.simplify(deep_acceleration * radius)
    checks.append(
        {
            "check_id": "mond_circular_velocity_relation",
            "status": "pass" if velocity_squared == sp.sqrt(mass * newton * scale) else "reject",
            "scope": "equation_68_69_algebra_only",
            "exact_v_squared": str(velocity_squared),
            "exact_v_fourth": str(sp.expand(velocity_squared**2)),
        }
    )

    corrupted_root = -acceleration
    negative_residual = sp.simplify(
        corrupted_root**2 * entropy / a0 - corrupted_root * deficit - g * entropy
    )
    checks.append(
        {
            "check_id": "negative_control_wrong_quadratic_branch",
            "status": "pass" if negative_residual != 0 else "reject",
            "scope": "deterministic_negative_control",
            "mutation_rejected": negative_residual != 0,
            "exact_residual": str(negative_residual),
        }
    )
    return checks


def _validate_result(result: dict[str, Any]) -> None:
    if result.get("schema_version") != RESULT_SCHEMA:
        raise ValueError("transactional-gravity artifact schema changed")
    body = {key: value for key, value in result.items() if key != "content_sha256"}
    if result.get("content_sha256") != _sha(body):
        raise ValueError("transactional-gravity artifact content hash mismatch")
    contracts = result.get("formula_contracts")
    checks = result.get("synthetic_checks")
    if not isinstance(contracts, list) or len(contracts) != 8:
        raise ValueError("formula contract registry is incomplete")
    if not isinstance(checks, list) or len(checks) != 8:
        raise ValueError("synthetic check registry is incomplete")
    statuses = [check.get("status") for check in checks]
    if statuses.count("pass") != 7 or statuses.count("block") != 1 or "reject" in statuses:
        raise ValueError("synthetic check partition changed")
    exact_seals = {
        "fundamental_action_registered": False,
        "formal_gr_equivalence_proven": False,
        "dark_matter_elimination_proven": False,
        "dark_energy_elimination_proven": False,
        "observational_pass": False,
        "theory_validity_claimed": False,
        "cuda_execution_performed": False,
        "automatic_downstream_enqueue_performed": False,
    }
    if result.get("claim_seals") != exact_seals:
        raise ValueError("scientific claim seals changed")
    if result.get("decision") != "blocked" or result.get("first_blocker") != FIRST_BLOCKER:
        raise ValueError("intake decision is not fail-closed")
    if result.get("synthetic_preflight_counts") != {"pass": 7, "reject": 0, "block": 1}:
        raise ValueError("synthetic preflight counts changed")


def build_intake(config: dict[str, Any], root: Path) -> dict[str, Any]:
    """Build the source-bound equation intake and deterministic synthetic references."""

    _validate_config(config)
    implementation_path = Path(__file__).resolve()
    config_path = root / "configs/kastner_schlatter_transactional_gravity_intake.json"
    if _file_sha(config_path) != _sha_file_expected(config):
        raise ValueError("loaded config does not match the repository config file")
    contracts = _equation_contracts()
    checks = _synthetic_checks(config)
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": config["campaign_id"],
        "primary_source": config["primary_source"],
        "source_binding": {
            "official_pdf_sha256": SOURCE_PDF_SHA256,
            "config_path": "configs/kastner_schlatter_transactional_gravity_intake.json",
            "config_file_sha256": _file_sha(config_path),
            "implementation_path": (
                "src/sigma_theory_compiler/kastner_schlatter_transactional_gravity_intake.py"
            ),
            "implementation_file_sha256": _file_sha(implementation_path),
        },
        "action_contract": {
            "contract_kind": "equation_only_proposal_intake",
            "fundamental_action": None,
            "field_content_closed": False,
            "variational_principle_registered": False,
            "euler_lagrange_map_registered": False,
            "boundary_terms_registered": False,
            "candidate_action_hash": None,
        },
        "formula_contracts": contracts,
        "synthetic_checks": checks,
        "synthetic_preflight_counts": {
            "pass": sum(check["status"] == "pass" for check in checks),
            "reject": sum(check["status"] == "reject" for check in checks),
            "block": sum(check["status"] == "block" for check in checks),
        },
        "cuda_handoff_contract": {
            "ready_for_later_execution": True,
            "execution_performed": False,
            "numeric_type": "IEEE-754 binary64",
            "kernels": [
                "poisson_pmf_and_recurrence",
                "transaction_pressure_and_lambda",
                "sds_entropy_quadratic_positive_root",
                "sds_mond_equation_62",
                "mond_circular_velocity",
            ],
            "domain_guards": [
                "mu>=0 and integer n>=0",
                "q_gamma>=0",
                "a0>0, S>0, g>=0, |DeltaS_dS|>=0",
                "M>0, G>0, c>0, r>r0",
            ],
            "blocked_constant_normalization": (
                "equation 35 h-versus-hbar factor requires clarification before using its "
                "final Lambda normalization"
            ),
            "reference_vectors_sha256": _sha(checks),
        },
        "decision": "blocked",
        "first_blocker": FIRST_BLOCKER,
        "remaining_missing_premises": [
            FIRST_BLOCKER,
            "no_formal_transaction_process_to_lorentzian_continuum_derivation",
            "einstein_recovery_uses_local_and_transformation_assumptions_not_a_variational_proof",
            "no_candidate_bound_perturbation_initial_boundary_value_or_global_completion",
            "no_registered_galaxy_lensing_cluster_or_cosmology_likelihood",
        ],
        "claim_seals": {
            "fundamental_action_registered": False,
            "formal_gr_equivalence_proven": False,
            "dark_matter_elimination_proven": False,
            "dark_energy_elimination_proven": False,
            "observational_pass": False,
            "theory_validity_claimed": False,
            "cuda_execution_performed": False,
            "automatic_downstream_enqueue_performed": False,
        },
        "data_seals": config["data_seals"],
        "external_paid_llm_calls": False,
    }
    result["content_sha256"] = _sha(result)
    _validate_result(result)
    return result


def _sha_file_expected(config: dict[str, Any]) -> str:
    encoded = (json.dumps(config, indent=2, ensure_ascii=False) + "\n").encode()
    return hashlib.sha256(encoded).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    root = Path(__file__).resolve().parents[2]
    result = build_intake(config, root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "content_sha256": result["content_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
