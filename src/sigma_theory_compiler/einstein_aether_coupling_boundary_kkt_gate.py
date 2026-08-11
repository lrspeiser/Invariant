"""Exact aligned Einstein-Aether coupling-boundary KKT rank gate."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import sympy as sp

from .adm_aether import (
    _einstein_aether_kinetic_model,
    einstein_aether_reduced_principal_domain_control,
)

CONFIG_SCHEMA = "sigma-einstein-aether-coupling-boundary-kkt-gate-config-1.0"
RESULT_SCHEMA = "sigma-einstein-aether-coupling-boundary-kkt-gate-1.0"
FIRST_BLOCKER = (
    "generic_nonlinear_constraint_reduced_Hamiltonian_boundedness_and_boundary_"
    "completion_not_proven"
)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _content_sha(value: Mapping[str, Any]) -> str:
    return _sha({key: item for key, item in value.items() if key != "content_sha256"})


def _inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("Einstein-Aether KKT path escapes repository") from error
    return path


def _bound_file(root: Path, binding: Mapping[str, Any], label: str) -> Path:
    if set(binding) != {"path", "file_sha256"}:
        raise ValueError(f"{label} binding shape changed")
    path = _inside(root, str(binding["path"]))
    if not path.is_file() or _file_sha(path) != binding["file_sha256"]:
        raise ValueError(f"{label} file hash mismatch")
    return path


def _validate_config(config: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "campaign_id",
        "output_path",
        "source_bindings",
        "aligned_contract",
        "admission_policy",
        "seals",
    }
    if set(config) != required or config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("Einstein-Aether KKT config shape changed")
    if set(config.get("source_bindings", {})) != {"adm_aether_source", "formal_controls"}:
        raise ValueError("Einstein-Aether KKT source binding set changed")
    if config.get("aligned_contract", {}).get("unit_constraint") != "-u0^2+u_i*u_i+1=0":
        raise ValueError("Einstein-Aether KKT unit constraint changed")
    if config.get("aligned_contract", {}).get("metric_velocity_relation") != (
        "K_ij=-dot(gamma_ij)/2"
    ):
        raise ValueError("Einstein-Aether KKT velocity convention changed")
    if config.get("admission_policy") != {
        "ambient_normal_singularity_is_not_constrained_rank_loss": True,
        "KKT_or_tangent_rank_loss_is_a_boundary_obstruction": True,
        "linear_five_mode_positivity_is_not_global_nonlinear_stability": True,
        "boundary_rank_loss_is_not_automatic_theory_rejection": True,
        "observational_or_candidate_inference_allowed": False,
    }:
        raise ValueError("Einstein-Aether KKT admission policy changed")
    if any(config.get("seals", {}).values()):
        raise ValueError("Einstein-Aether KKT seal opened")


def _validate_formal_controls(path: Path) -> Mapping[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    checks = {item["name"]: item for item in report.get("checks", [])}
    required = {
        "einstein_aether_adm_kinetic_hessian",
        "einstein_aether_coupled_unit_normal",
        "einstein_aether_reduced_five_mode_principal_domain",
    }
    if not required.issubset(checks) or any(
        checks[name].get("status") != "pass" for name in required
    ):
        raise ValueError("Einstein-Aether predecessor formal controls changed")
    reduced = checks["einstein_aether_reduced_five_mode_principal_domain"]["evidence"]
    if reduced.get("necessary_and_sufficient_regular_domain") != [
        "1-c13 > 0",
        "0 < c14 < 2",
        "2c1-c1^2+c3^2 > 0",
        "c123(2+c13+3c2) > 0",
    ]:
        raise ValueError("Einstein-Aether reduced positivity chart changed")
    return reduced


def _matrices() -> dict[str, Any]:
    model = _einstein_aether_kinetic_model()
    hessian_k = model["hessian"]
    u0, u1, u2, u3 = model["vector_down"]
    planck2, c1, c2, c3, c4 = model["coupling_symbols"]
    aligned = hessian_k.subs({u0: 1, u1: 0, u2: 0, u3: 0})
    # K_ij=-dot(gamma_ij)/2 for all six independent symmetric components.
    pullback = sp.diag(*([-sp.Rational(1, 2)] * 6 + [1] * 4))
    ambient = sp.simplify(pullback.T * aligned * pullback)
    tangent_indices = [0, 1, 2, 3, 4, 5, 7, 8, 9]
    tangent = ambient.extract(tangent_indices, tangent_indices)
    constraint_gradient = sp.zeros(10, 1)
    constraint_gradient[6] = -2
    kkt = ambient.row_join(constraint_gradient).col_join(
        constraint_gradient.T.row_join(sp.zeros(1, 1))
    )
    return {
        "ambient": ambient,
        "tangent": tangent,
        "kkt": kkt,
        "constraint_gradient": constraint_gradient,
        "symbols": (planck2, c1, c2, c3, c4),
        "model": model,
    }


def _symbolic_factorization(matrices: Mapping[str, Any]) -> dict[str, Any]:
    ambient = matrices["ambient"]
    tangent = matrices["tangent"]
    kkt = matrices["kkt"]
    gradient = matrices["constraint_gradient"]
    planck2, c1, c2, c3, c4 = matrices["symbols"]
    c13 = c1 + c3
    c14 = c1 + c4
    trace = 2 * planck2 + c13 + 3 * c2
    normal_factor = sp.factor(
        2 * planck2 * c1
        + 2 * planck2 * c2
        + 2 * planck2 * c3
        + 2 * planck2 * c4
        + c1**2
        + 4 * c1 * c2
        + 2 * c1 * c3
        + c1 * c4
        + 4 * c2 * c3
        + 3 * c2 * c4
        + c3**2
        + c3 * c4
    )
    expected_ambient = sp.factor(-(c14**3) * (-planck2 + c13) ** 5 * normal_factor / 512)
    expected_tangent = sp.factor(c14**3 * (-planck2 + c13) ** 5 * trace / 512)
    expected_kkt = sp.factor(-(c14**3) * (-planck2 + c13) ** 5 * trace / 128)
    ambient_det = sp.factor(ambient.det())
    tangent_det = sp.factor(tangent.det())
    kkt_det = sp.factor(kkt.det())
    normality = sp.factor((gradient.T * ambient.inv() * gradient)[0])
    expected_normality = sp.factor(-4 * trace / normal_factor)
    identities = {
        "ambient_10x10": sp.factor(ambient_det - expected_ambient) == 0,
        "tangent_9x9": sp.factor(tangent_det - expected_tangent) == 0,
        "KKT_11x11": sp.factor(kkt_det - expected_kkt) == 0,
        "unit_normality_rational": sp.factor(normality - expected_normality) == 0,
        "KKT_equals_minus_four_tangent": sp.factor(kkt_det + 4 * tangent_det) == 0,
    }
    if not all(identities.values()):
        raise ValueError("Einstein-Aether determinant identity failed")
    return {
        "combinations": {
            "c13": "c1+c3",
            "c14": "c1+c4",
            "trace_factor": "2*M2+c13+3*c2",
            "normal_factor_D": str(normal_factor),
            "normal_factor_alternative": ("D=(c1+c2+c3+c4)*(2*M2+c1+c3+3*c2)-3*c2^2"),
        },
        "ambient_10x10_determinant": str(ambient_det),
        "expected_ambient_10x10_determinant": str(expected_ambient),
        "unit_normality_rational_factor": str(normality),
        "expected_unit_normality_rational_factor": str(expected_normality),
        "tangent_9x9_determinant": str(tangent_det),
        "expected_tangent_9x9_determinant": str(expected_tangent),
        "constrained_KKT_11x11_determinant": str(kkt_det),
        "expected_constrained_KKT_11x11_determinant": str(expected_kkt),
        "identity_checks": identities,
        "interpretation": (
            "D is only the Schur-complement normal factor of the ambient Hessian. The unit "
            "constraint removes that direction; constrained rank depends on c14, M2-c13, and "
            "2*M2+c13+3*c2, not D."
        ),
    }


def _point_record(
    matrices: Mapping[str, Any], couplings: Mapping[sp.Symbol, sp.Expr]
) -> dict[str, Any]:
    ambient = matrices["ambient"].subs(couplings)
    tangent = matrices["tangent"].subs(couplings)
    kkt = matrices["kkt"].subs(couplings)
    planck2, c1, c2, c3, c4 = matrices["symbols"]
    normal_factor = sp.factor(
        2 * planck2 * c1
        + 2 * planck2 * c2
        + 2 * planck2 * c3
        + 2 * planck2 * c4
        + c1**2
        + 4 * c1 * c2
        + 2 * c1 * c3
        + c1 * c4
        + 4 * c2 * c3
        + 3 * c2 * c4
        + c3**2
        + c3 * c4
    )
    factors = {
        "D": normal_factor,
        "c14": c1 + c4,
        "M2_minus_c13": planck2 - c1 - c3,
        "trace": 2 * planck2 + c1 + c3 + 3 * c2,
        "c123": c1 + c2 + c3,
        "vector_gradient": 2 * c1 - c1**2 + c3**2,
    }
    return {
        "couplings": {str(symbol): str(value) for symbol, value in couplings.items()},
        "factor_values": {
            name: str(sp.factor(value.subs(couplings))) for name, value in factors.items()
        },
        "ambient_rank": int(ambient.rank()),
        "tangent_rank": int(tangent.rank()),
        "KKT_rank": int(kkt.rank()),
        "ambient_determinant": str(sp.factor(ambient.det())),
        "tangent_determinant": str(sp.factor(tangent.det())),
        "KKT_determinant": str(sp.factor(kkt.det())),
        "ambient_nullspace": [[str(item) for item in vector] for vector in ambient.nullspace()],
        "tangent_nullspace": [[str(item) for item in vector] for vector in tangent.nullspace()],
        "KKT_nullspace": [[str(item) for item in vector] for vector in kkt.nullspace()],
    }


def _witnesses(matrices: Mapping[str, Any]) -> dict[str, Any]:
    planck2, c1, c2, c3, c4 = matrices["symbols"]
    d_only_chart = _point_record(
        matrices,
        {
            planck2: 1,
            c1: sp.Rational(27, 46),
            c2: sp.Rational(1, 10),
            c3: -sp.Rational(27, 46),
            c4: -sp.Rational(2, 23),
        },
    )
    d_only_independent_replay = _point_record(
        matrices,
        {
            planck2: 1,
            c1: sp.Rational(1, 10),
            c2: sp.Rational(1, 20),
            c3: 0,
            c4: -sp.Rational(11, 75),
        },
    )
    c14_boundary = _point_record(
        matrices,
        {
            planck2: 1,
            c1: sp.Rational(1, 10),
            c2: sp.Rational(1, 20),
            c3: 0,
            c4: -sp.Rational(1, 10),
        },
    )
    tensor_boundary = _point_record(
        matrices,
        {
            planck2: 1,
            c1: sp.Rational(1, 4),
            c2: 0,
            c3: sp.Rational(3, 4),
            c4: sp.Rational(1, 4),
        },
    )
    trace_boundary = _point_record(
        matrices,
        {
            planck2: 1,
            c1: sp.Rational(1, 4),
            c2: -sp.Rational(3, 4),
            c3: 0,
            c4: sp.Rational(1, 4),
        },
    )
    if not (
        d_only_chart["factor_values"]
        == {
            "D": "0",
            "c14": "1/2",
            "M2_minus_c13": "1",
            "trace": "23/10",
            "c123": "1/10",
            "vector_gradient": "27/23",
        }
        and (d_only_chart["ambient_rank"], d_only_chart["tangent_rank"], d_only_chart["KKT_rank"])
        == (9, 9, 11)
        and (c14_boundary["ambient_rank"], c14_boundary["tangent_rank"], c14_boundary["KKT_rank"])
        == (7, 6, 8)
        and (
            tensor_boundary["ambient_rank"],
            tensor_boundary["tangent_rank"],
            tensor_boundary["KKT_rank"],
        )
        == (5, 4, 6)
        and (
            trace_boundary["ambient_rank"],
            trace_boundary["tangent_rank"],
            trace_boundary["KKT_rank"],
        )
        == (10, 8, 10)
    ):
        raise ValueError("Einstein-Aether exact rank witness changed")
    return {
        "D_only_inside_five_mode_positivity_chart": {
            **d_only_chart,
            "five_mode_chart_checks": {
                "1_minus_c13_positive": True,
                "zero_less_c14_less_two": True,
                "vector_gradient_positive": True,
                "c123_times_trace_positive": True,
            },
            "conclusion": (
                "the ambient normal factor vanishes, but tangent and KKT rank are full and the "
                "registered five-mode reduced linear Hamiltonian chart remains positive"
            ),
        },
        "D_only_independent_replay": d_only_independent_replay,
        "true_constrained_boundaries": {
            "c14_equals_zero": c14_boundary,
            "M2_minus_c13_equals_zero": tensor_boundary,
            "two_M2_plus_c13_plus_3c2_equals_zero": trace_boundary,
        },
    }


def _reduced_chart_binding(reduced_source: Mapping[str, Any]) -> dict[str, Any]:
    replay = einstein_aether_reduced_principal_domain_control()
    if (
        replay.get("passed") is not True
        or replay.get("necessary_and_sufficient_regular_domain")
        != reduced_source.get("necessary_and_sufficient_regular_domain")
        or replay.get("aligned_full_legendre_determinant")
        != reduced_source.get("aligned_full_legendre_determinant")
    ):
        raise ValueError("Einstein-Aether reduced chart replay changed")
    return {
        "formal_control_status": "pass",
        "mode_count": replay["mode_count"],
        "physical_basis": replay["physical_basis"],
        "necessary_and_sufficient_linear_domain_M2_equals_one": replay[
            "necessary_and_sufficient_regular_domain"
        ],
        "aligned_full_legendre_determinant_K_basis": replay["aligned_full_legendre_determinant"],
        "shared_constrained_rank_boundaries": [
            "c14=0",
            "M2-c13=0 (M2=1 gives 1-c13=0)",
            "2*M2+c13+3*c2=0 (M2=1 gives 2+c13+3*c2=0)",
        ],
        "D_is_not_a_reduced_five_mode_boundary": True,
        "scope": (
            "exact aligned-Minkowski five-mode linearized kinetic/gradient positivity and "
            "hyperbolicity only; not a generic nonlinear reduced-Hamiltonian boundedness theorem"
        ),
    }


def _validate_result(result: Mapping[str, Any]) -> None:
    if result.get("schema_version") != RESULT_SCHEMA:
        raise ValueError("Einstein-Aether KKT result schema changed")
    if result.get("decision_counts") != {"pass": 0, "reject": 0, "blocked": 1}:
        raise ValueError("Einstein-Aether KKT decision partition changed")
    if result.get("gate_counts") != {
        "generic_symbolic_determinant_identities_pass": 5,
        "D_only_ambient_singular_constrained_full_rank_witnesses": 2,
        "true_constrained_rank_boundary_witnesses": 3,
        "five_mode_linear_positivity_chart_bindings": 1,
        "global_nonlinear_stability_pass": 0,
        "candidate_or_theory_reject": 0,
        "observational_pass": 0,
    }:
        raise ValueError("Einstein-Aether KKT gate counts changed")
    if not all(result.get("symbolic_factorization", {}).get("identity_checks", {}).values()):
        raise ValueError("Einstein-Aether KKT determinant identity lost")
    witness = result.get("exact_witnesses", {}).get("D_only_inside_five_mode_positivity_chart", {})
    if (
        witness.get("ambient_rank") != 9
        or witness.get("tangent_rank") != 9
        or witness.get("KKT_rank") != 11
        or not all(witness.get("five_mode_chart_checks", {}).values())
    ):
        raise ValueError("Einstein-Aether D-only witness lost")
    boundaries = result.get("exact_witnesses", {}).get("true_constrained_boundaries", {})
    if [
        boundaries.get(name, {}).get("KKT_rank")
        for name in (
            "c14_equals_zero",
            "M2_minus_c13_equals_zero",
            "two_M2_plus_c13_plus_3c2_equals_zero",
        )
    ] != [8, 6, 10]:
        raise ValueError("Einstein-Aether true boundary witness lost")
    if result.get("first_blocker") != FIRST_BLOCKER:
        raise ValueError("Einstein-Aether KKT first blocker changed")
    if any(result.get("claim_seals", {}).values()) or any(result.get("data_seals", {}).values()):
        raise ValueError("Einstein-Aether KKT seal opened")
    if (
        result.get("content_sha256") is not None
        and _content_sha(result) != result["content_sha256"]
    ):
        raise ValueError("Einstein-Aether KKT content hash mismatch")


def build_gate(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    root = config_path.parents[1]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    source_bindings = config["source_bindings"]
    source_path = _bound_file(root, source_bindings["adm_aether_source"], "ADM-Aether source")
    formal_path = _bound_file(root, source_bindings["formal_controls"], "formal controls")
    model_path = Path(inspect.getsourcefile(_einstein_aether_kinetic_model) or "").resolve()
    if model_path != source_path:
        raise ValueError("Einstein-Aether kinetic model source path changed")
    reduced_source = _validate_formal_controls(formal_path)
    matrices = _matrices()
    factorization = _symbolic_factorization(matrices)
    witnesses = _witnesses(matrices)
    reduced_chart = _reduced_chart_binding(reduced_source)
    implementation_path = Path(__file__).resolve()
    test_path = root / "tests/test_einstein_aether_coupling_boundary_kkt_gate.py"
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": config["campaign_id"],
        "scope": (
            "generic independent-c1..c4 aligned-unit-Aether ambient, tangent, and constrained "
            "KKT rank classification with exact five-mode linear positivity-chart binding"
        ),
        "source_bindings": {
            **source_bindings,
            "config": {
                "path": config_path.relative_to(root).as_posix(),
                "file_sha256": _file_sha(config_path),
            },
            "implementation": {
                "path": implementation_path.relative_to(root).as_posix(),
                "file_sha256": _file_sha(implementation_path),
            },
            "test": {
                "path": test_path.relative_to(root).as_posix(),
                "file_sha256": _file_sha(test_path),
            },
        },
        "aligned_contract": config["aligned_contract"],
        "symbolic_factorization": factorization,
        "exact_witnesses": witnesses,
        "reduced_five_mode_chart_binding": reduced_chart,
        "gate_counts": {
            "generic_symbolic_determinant_identities_pass": 5,
            "D_only_ambient_singular_constrained_full_rank_witnesses": 2,
            "true_constrained_rank_boundary_witnesses": 3,
            "five_mode_linear_positivity_chart_bindings": 1,
            "global_nonlinear_stability_pass": 0,
            "candidate_or_theory_reject": 0,
            "observational_pass": 0,
        },
        "decision_counts": {"pass": 0, "reject": 0, "blocked": 1},
        "decision": "constrained_coupling_boundary_rank_closed_nonlinear_stability_blocked",
        "first_blocker": FIRST_BLOCKER,
        "secondary_blockers": [
            "arbitrary_background_constraint_reduced_Hamiltonian_not_derived",
            "boundary_charge_and_global_energy_completion_not_registered_for_generic_twisting_Aether",
            "rank_loss_boundaries_are_strong_coupling_strata_not_automatic_theory_rejections",
        ],
        "claim_seals": {
            "global_nonlinear_Hamiltonian_stability_proven": False,
            "arbitrary_background_boundary_classification_proven": False,
            "candidate_rejection_authorized": False,
            "theory_rejection_authorized": False,
            "observational_pass": False,
        },
        "data_seals": dict(config["seals"]),
    }
    result["content_sha256"] = _content_sha(result)
    _validate_result(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--output")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    result = build_gate(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = (
        Path(args.output).resolve()
        if args.output
        else config_path.parents[1] / str(config["output_path"])
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
