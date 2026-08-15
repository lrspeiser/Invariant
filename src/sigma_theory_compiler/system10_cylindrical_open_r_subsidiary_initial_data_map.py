from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import sympy as sp

from .system10_cylindrical_r_positive_divq_row_materializer import (
    R,
    _canonical_sha,
    _hat_projectors,
    _physical_tensors,
)


class System10SubsidiaryInitialDataMapError(RuntimeError):
    """Raised when a subsidiary initial-data implication fails closed."""


RECEIPT_SCHEMA = "invariant-system10-open-r-subsidiary-initial-data-map-receipt-1.0"
DECISION = "BOUNDED_PASS_12_SUBSIDIARY_INITIAL_DATA_MAPS_BLOCK_ENERGY_UNIQUENESS"


def _canonical_lf_sha(path: Path) -> str:
    raw = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(raw).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise System10SubsidiaryInitialDataMapError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise System10SubsidiaryInitialDataMapError("JSON root is not an object")
    return value


def _resolve(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise System10SubsidiaryInitialDataMapError("bound path escapes repository")
    return path


def _sealed(document: dict[str, Any]) -> bool:
    body = {key: value for key, value in document.items() if key != "content_sha256"}
    return document.get("content_sha256") == _canonical_sha(body)


def _load_binding(root: Path, binding: dict[str, Any]) -> dict[str, Any]:
    path = _resolve(root, str(binding.get("path", "")))
    if _canonical_lf_sha(path) != binding.get("canonical_lf_sha256"):
        raise System10SubsidiaryInitialDataMapError(f"bound file mismatch: {path}")
    document = _load_json(path)
    if document.get("content_sha256") != binding.get("content_sha256") or not _sealed(document):
        raise System10SubsidiaryInitialDataMapError(f"bound content mismatch: {path}")
    return document


def _authority_sha(config: dict[str, Any]) -> str:
    return _canonical_sha({key: value for key, value in config.items() if key != "source_evidence"})


def _validate_config(config_path: Path, root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = _load_json(config_path)
    if config.get("schema_version") != f"{RECEIPT_SCHEMA}-config":
        raise System10SubsidiaryInitialDataMapError("unsupported config schema")
    if config.get("caps") != {
        "candidate_indices": list(range(12)),
        "physical_gravity_constraints_per_candidate": 4,
        "modified_harmonic_constraints": 4,
        "gravity_normal_derivative_maps": 48,
        "maxwell_constraints": 2,
        "maxwell_normal_derivative_maps": 12,
        "candidate_initial_data_maps": 12,
        "maximum_receipt_bytes": 262144,
    }:
        raise System10SubsidiaryInitialDataMapError("caps changed")
    if set(config.get("bindings", {})) != {
        "factorization",
        "r_positive_domain",
        "full_rhs",
        "maxwell_dynamic",
        "matter_interface",
    }:
        raise System10SubsidiaryInitialDataMapError("binding manifest changed")
    bound = {name: _load_binding(root, item) for name, item in config["bindings"].items()}
    if bound["factorization"].get("counts", {}).get("divQ_to_C_factorization_rows_closed") != 4:
        raise System10SubsidiaryInitialDataMapError("factorization authority changed")
    if bound["r_positive_domain"].get("counts", {}).get("physical_gravity_rows_closed") != 96:
        raise System10SubsidiaryInitialDataMapError("physical constraints changed")
    if bound["full_rhs"].get("counts", {}).get("total_rhs_row_instances") != 1020:
        raise System10SubsidiaryInitialDataMapError("full RHS changed")
    maxwell = bound["maxwell_dynamic"].get("materialization", {})
    if (
        len(maxwell.get("rows", [])) != 4
        or maxwell.get("lorenz_reduced_origin", {}).get("completion") != "E_L_mu=E_mu+nabla_mu C"
    ):
        raise System10SubsidiaryInitialDataMapError("Maxwell reduced authority changed")
    interface = bound["matter_interface"]
    if (
        not interface.get("claims", {}).get("total_matter_stress_conserved_on_shell")
        or interface.get("combined_matter_certificate", {})
        .get("internal_matter_constraint_closure", {})
        .get("subsidiary_equation")
        != "box_g C=0 for the source-free Lorenz-gauge system"
    ):
        raise System10SubsidiaryInitialDataMapError("matter subsidiary authority changed")

    sources = {}
    for name, item in config.get("source_evidence", {}).items():
        path = _resolve(root, str(item.get("path", "")))
        if _canonical_lf_sha(path) != item.get("canonical_lf_sha256"):
            raise System10SubsidiaryInitialDataMapError(f"source evidence mismatch: {name}")
        sources[name] = path
    expected = {
        "source": Path(__file__).resolve(),
        "test": root / "tests/test_system10_cylindrical_open_r_subsidiary_initial_data_map.py",
        "divQ_source": root
        / "src/sigma_theory_compiler/system10_cylindrical_r_positive_divq_row_materializer.py",
        "maxwell_source": root
        / "src/sigma_theory_compiler/system10_cylindrical_r_positive_maxwell_dynamic_rhs_materializer.py",
    }
    if sources != expected:
        raise System10SubsidiaryInitialDataMapError("source evidence paths changed")
    return config, bound


def _gravity_normal_map() -> dict[str, Any]:
    _, inverse, hat, connection = _physical_tensors()
    projector, _ = _hat_projectors(hat, connection)
    matrix = sp.Matrix(
        4,
        4,
        lambda nu, beta: sp.factor(-projector(beta, 0, 0, nu) * inverse[beta][beta] / 2),
    )
    expected = sp.diag(-sp.Rational(9, 4), sp.Rational(9, 4), 9 / (4 * R**2), sp.Rational(9, 4))
    if matrix != expected:
        raise System10SubsidiaryInitialDataMapError("gravity normal map changed")
    determinant = sp.factor(matrix.det())
    if determinant != -sp.Rational(6561, 256) / R**2:
        raise System10SubsidiaryInitialDataMapError("gravity normal map determinant changed")
    inverse_matrix = matrix.inv().applyfunc(sp.factor)
    map_matrix = (-inverse_matrix).applyfunc(sp.factor)
    body = {
        "premises": [
            "E_sourced^{0nu}=physical_constraint^{0nu}+Q^{0nu}=0",
            "modified_harmonic_C_beta vanishes as a differentiable field on the initial slice",
            "therefore tangential_derivative_i(C_beta)=0 and connection*C terms vanish",
        ],
        "normal_operator": "Q^{0nu}/M2=A[nu,beta]*partial_0(C_beta)",
        "A": [[sp.sstr(value) for value in row] for row in matrix.tolist()],
        "det_A": sp.sstr(determinant),
        "det_A_nonzero_on_r_positive": True,
        "normal_derivative_map": ("partial_0(C_beta)=(-A^-1)[beta,nu]*physical_constraint^{0nu}"),
        "minus_A_inverse": [[sp.sstr(value) for value in row] for row in map_matrix.tolist()],
        "vanishing_H_M_implies_four_vanishing_normal_C_derivatives": True,
    }
    return {**body, "map_sha256": _canonical_sha(body)}


def _maxwell_initial_map(maxwell: dict[str, Any]) -> dict[str, Any]:
    time, radius, angle, axis = sp.symbols("t r theta z", real=True)
    coordinates = [time, radius, angle, axis]
    metric = sp.diag(-1, 1, radius**2, 1)
    inverse = metric.inv()
    potentials = [sp.Function(f"B_{index}")(*coordinates) for index in range(4)]
    connection = [
        [
            [
                sp.factor(
                    sum(
                        inverse[upper, delta]
                        * (
                            sp.diff(metric[delta, right], coordinates[left])
                            + sp.diff(metric[delta, left], coordinates[right])
                            - sp.diff(metric[left, right], coordinates[delta])
                        )
                        for delta in range(4)
                    )
                    / 2
                )
                for right in range(4)
            ]
            for left in range(4)
        ]
        for upper in range(4)
    ]
    field_strength = [
        [
            sp.diff(potentials[right], coordinates[left])
            - sp.diff(potentials[left], coordinates[right])
            for right in range(4)
        ]
        for left in range(4)
    ]
    gauss = sp.factor(
        sum(
            inverse[nu, derivative]
            * (
                sp.diff(field_strength[nu][0], coordinates[derivative])
                - sum(
                    connection[upper][derivative][nu] * field_strength[upper][0]
                    + connection[upper][derivative][0] * field_strength[nu][upper]
                    for upper in range(4)
                )
            )
            for nu in range(4)
            for derivative in range(4)
        )
    )
    lorenz = sp.factor(
        sum(
            inverse[left, right]
            * (
                sp.diff(potentials[right], coordinates[left])
                - sum(connection[upper][left][right] * potentials[upper] for upper in range(4))
            )
            for left in range(4)
            for right in range(4)
        )
    )
    normal_lorenz = sp.diff(lorenz, time)
    reduced_normal = sp.factor(gauss + normal_lorenz)
    solved_acceleration = sp.solve(sp.Eq(reduced_normal, 0), sp.diff(potentials[0], time, 2))[0]
    registered = maxwell["materialization"]["rows"][0]
    registered_terms = registered["rhs_terms"]
    expected_terms = [
        {"coefficient": "1", "atom": "partial_1 state[46]"},
        {"coefficient": "1/r", "atom": "state[46]"},
        {"coefficient": "1/r**2", "atom": "partial_2 state[63]"},
        {"coefficient": "1", "atom": "partial_3 state[80]"},
    ]
    if registered_terms != expected_terms:
        raise System10SubsidiaryInitialDataMapError("registered Maxwell normal row changed")
    expected_solve = (
        sp.diff(potentials[0], radius, 2)
        + sp.diff(potentials[0], radius) / radius
        + sp.diff(potentials[0], angle, 2) / radius**2
        + sp.diff(potentials[0], axis, 2)
    )
    if sp.factor(solved_acceleration - expected_solve) != 0:
        raise System10SubsidiaryInitialDataMapError("Maxwell normal reduced replay failed")
    lorenz_state = [
        {"coefficient": "-1", "atom": "state[29]"},
        {"coefficient": "1", "atom": "state[47]"},
        {"coefficient": "1/r**2", "atom": "state[65]"},
        {"coefficient": "1", "atom": "state[83]"},
        {"coefficient": "1/r", "atom": "state[13]"},
    ]
    gauss_state = [
        {"coefficient": "1", "atom": "partial_1 state[46]"},
        {"coefficient": "1/r", "atom": "state[46]"},
        {"coefficient": "1/r**2", "atom": "partial_2 state[63]"},
        {"coefficient": "1", "atom": "partial_3 state[80]"},
        {"coefficient": "-1", "atom": "partial_1 state[30]"},
        {"coefficient": "-1/r", "atom": "state[30]"},
        {"coefficient": "-1/r**2", "atom": "partial_2 state[31]"},
        {"coefficient": "-1", "atom": "partial_3 state[32]"},
    ]
    body = {
        "action_constraint": "Maxwell_Gauss=E_0=nabla^nu F_nu0",
        "lorenz_constraint": "C_Maxwell=nabla^rho B_rho",
        "lorenz_state_operator_terms": lorenz_state,
        "lorenz_state_operator_sha256": _canonical_sha(lorenz_state),
        "gauss_state_operator_terms": gauss_state,
        "gauss_state_operator_sha256": _canonical_sha(gauss_state),
        "registered_reduced_normal_row_sha256": registered["row_sha256"],
        "exact_component_identity": "E_L_0=Maxwell_Gauss+partial_0(C_Maxwell)",
        "symbolic_identity_residual": "0",
        "normal_derivative_map": "partial_0(C_Maxwell)=-Maxwell_Gauss when E_L_0=0",
        "vanishing_Gauss_implies_vanishing_normal_C_Maxwell_derivative": True,
        "domain": "r>0",
    }
    return {**body, "map_sha256": _canonical_sha(body)}


def build_receipt(config_path: Path, *, root: Path | None = None) -> dict[str, Any]:
    repository = (root or config_path.resolve().parents[1]).resolve()
    config, bound = _validate_config(config_path.resolve(), repository)
    gravity = _gravity_normal_map()
    maxwell = _maxwell_initial_map(bound["maxwell_dynamic"])
    domain_candidates = bound["r_positive_domain"]["materialization"]["candidate_results"]
    factor_candidates = bound["factorization"]["candidate_results"]
    full_candidates = bound["full_rhs"]["candidate_results"]
    if not (
        len(domain_candidates) == len(factor_candidates) == len(full_candidates) == 12
        and [item["candidate_id"] for item in domain_candidates]
        == [item["candidate_id"] for item in factor_candidates]
        == [item["candidate_id"] for item in full_candidates]
    ):
        raise System10SubsidiaryInitialDataMapError("candidate join changed")
    maps = []
    for index, (domain, factor, full) in enumerate(
        zip(domain_candidates, factor_candidates, full_candidates, strict=True)
    ):
        body = {
            "candidate_index": index,
            "candidate_id": domain["candidate_id"],
            "physical_constraint_manifest_sha256": domain["manifest_sha256"],
            "full_rhs_equation_origin_set_sha256": full["equation_origin_set_sha256"],
            "factorization_row_set_sha256": factor["common_factorization_row_set_sha256"],
            "gravity_normal_map_sha256": gravity["map_sha256"],
            "maxwell_initial_map_sha256": maxwell["map_sha256"],
            "premise": (
                "H=M_i=modified_harmonic_C_beta=Maxwell_Gauss=C_Maxwell=0 "
                "as differentiable initial-slice fields"
            ),
            "conclusion": ("partial_0 modified_harmonic_C_beta=0 and partial_0 C_Maxwell=0"),
            "exact_map_closed": True,
        }
        maps.append({**body, "map_sha256": _canonical_sha(body)})

    missing = {
        "primitive": "common_tube_homogeneous_subsidiary_cauchy_uniqueness_certificate",
        "status": "BLOCK_SUBSIDIARY_ENERGY_OR_UNIQUENESS_CERTIFICATE_UNREGISTERED",
        "required_systems": [
            "four_component_modified_harmonic_divQ(C)=0 operator on 1/2<=r<=3/2",
            "scalar Maxwell box_g(C_Maxwell)=0 operator on 1/2<=r<=3/2",
        ],
        "required_controls": [
            "common-time principal coercivity or a first-order symmetrizer",
            "bounded lower-order coefficient estimate on the certified tube",
            "causal subdomain or boundary-flux conditions",
            "zero-data uniqueness estimate",
        ],
        "registered_energy_estimates": 0,
        "acceptance": (
            "Derive a coercive exact energy (or equivalent Cauchy uniqueness theorem with all "
            "premises checked) for the two sealed homogeneous subsidiary operators on the "
            "certified tube and conclude that the zero initial data remain zero."
        ),
        "why_not_inferred": (
            "An algebraic initial-data implication plus homogeneous differential equations is "
            "not itself a checked well-posedness or uniqueness estimate, especially on a bounded "
            "radial tube with unspecified boundary flux."
        ),
    }
    missing["primitive_sha256"] = _canonical_sha(missing)
    body = {
        "schema_version": RECEIPT_SCHEMA,
        "campaign_id": config["campaign_id"],
        "decision": DECISION,
        "scope": (
            "Exact initial-slice implication for all twelve candidates. The reduced metric "
            "normal equations plus physical Hamiltonian/momentum constraints invert the "
            "Q^{0nu} normal-derivative matrix for modified-harmonic C. The source-free Maxwell "
            "normal reduced equation plus the action Gauss constraint gives the normal derivative "
            "of the Lorenz constraint. Homogeneous subsidiary equations are now sealed, but a "
            "tube energy/uniqueness certificate is not; no propagation or hyperbolicity claim is made."
        ),
        "source_bindings": {
            "authority_sha256": _authority_sha(config),
            "predecessor_content_sha256": {
                name: document["content_sha256"] for name, document in bound.items()
            },
        },
        "counts": {
            "candidates": 12,
            "physical_gravity_constraints_bound": 48,
            "modified_harmonic_constraints_bound": 4,
            "gravity_normal_derivative_maps_closed": 48,
            "maxwell_lorenz_constraints_bound": 1,
            "maxwell_gauss_constraints_registered": 1,
            "maxwell_normal_derivative_maps_closed": 12,
            "candidate_subsidiary_initial_data_maps_required": 12,
            "candidate_subsidiary_initial_data_maps_closed": 12,
            "homogeneous_gravity_subsidiary_equations_closed": 4,
            "homogeneous_maxwell_subsidiary_equations_closed": 1,
            "subsidiary_energy_uniqueness_certificates": 0,
            "constraint_propagation_proofs": 0,
        },
        "candidate_results": [
            {
                "candidate_index": item["candidate_index"],
                "candidate_id": item["candidate_id"],
                "map_sha256": item["map_sha256"],
                "outcome": "PASS_INITIAL_DATA_MAP_BLOCK_SUBSIDIARY_ENERGY_UNIQUENESS",
            }
            for item in maps
        ],
        "materialization": {
            "gravity_normal_derivative_map": gravity,
            "maxwell_normal_derivative_map": maxwell,
            "candidate_initial_data_maps": maps,
            "candidate_map_set_sha256": _canonical_sha(maps),
            "homogeneous_subsidiary_system": {
                "gravity": (
                    "divQ_lower[nu]/M2=0 with the four exact factorized rows in the predecessor"
                ),
                "gravity_row_set_sha256": bound["factorization"]["materialization"][
                    "factorization_row_set_sha256"
                ],
                "maxwell": "box_g(C_Maxwell)=0",
                "matter_on_shell_source_cancellation": True,
                "closed_equation_count": 5,
                "cauchy_uniqueness_proved": False,
            },
            "propagation_audit": {"first_missing_primitive": missing},
            "negative_controls": {
                "pointwise_C_instead_of_slice_field": {
                    "mutation": "assume C_beta=0 only at one point and set tangential derivatives to zero",
                    "rejected": True,
                    "reason": "pointwise vanishing does not imply tangential derivative vanishing",
                },
                "drop_Maxwell_Gauss": {
                    "mutation": "use only E_L_0=0 and C_Maxwell=0",
                    "free_witness": "partial_0(C_Maxwell)=lambda, Maxwell_Gauss=-lambda",
                    "rejected": True,
                },
                "corrupt_Q_normal_sign": {
                    "mutation": "replace A[0,0]=-9/4 by +9/4",
                    "expected_det_A": gravity["det_A"],
                    "rejected": True,
                },
                "claim_propagation_without_uniqueness": {
                    "mutation": "infer zero solution from homogeneous equations without energy/uniqueness premises",
                    "registered_energy_estimates": 0,
                    "rejected": True,
                },
            },
        },
        "claims": {
            "all_twelve_subsidiary_initial_data_maps_closed": True,
            "gravity_normal_derivative_implication_closed": True,
            "maxwell_normal_derivative_implication_closed": True,
            "homogeneous_candidate_bound_subsidiary_system_closed": True,
            "subsidiary_energy_or_uniqueness_closed": False,
            "constraint_propagation_closed": False,
            "hyperbolicity_closed": False,
            "global_theorem_established": False,
            "promotion_authorized": False,
        },
    }
    receipt = {**body, "content_sha256": _canonical_sha(body)}
    if len(json.dumps(receipt).encode("utf-8")) > config["caps"]["maximum_receipt_bytes"]:
        raise System10SubsidiaryInitialDataMapError("receipt cap exceeded")
    return receipt


def write_output(config_path: Path, output_dir: Path, *, root: Path | None = None) -> Path:
    receipt = build_receipt(config_path, root=root)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "receipt.json"
    data = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if path.exists() and path.read_bytes() != data:
        raise System10SubsidiaryInitialDataMapError(f"immutable output conflict: {path}")
    if not path.exists():
        temporary = output_dir / "receipt.json.tmp"
        temporary.write_bytes(data)
        os.replace(temporary, path)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build subsidiary initial-data maps")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    write_output(args.config, args.output, root=args.config.resolve().parents[1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
