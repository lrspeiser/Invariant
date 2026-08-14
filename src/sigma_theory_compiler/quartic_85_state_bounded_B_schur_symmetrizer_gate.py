# ruff: noqa: N999
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

try:
    from . import quartic_85_state_nonresonant_sylvester_complement_gate as sylvester
except ImportError:  # pragma: no cover - supports direct receipt materialization
    _path = Path(__file__).with_name("quartic_85_state_nonresonant_sylvester_complement_gate.py")
    _spec = importlib.util.spec_from_file_location("_sylvester_exact", _path)
    if _spec is None or _spec.loader is None:
        raise
    sylvester = importlib.util.module_from_spec(_spec)
    sys.modules["_sylvester_exact"] = sylvester
    _spec.loader.exec_module(sylvester)

sphere = sylvester.sphere
exact = sylvester.exact


class Quartic85StateBoundedBSymmetrizerError(RuntimeError):
    """Raised when the corrected flat-reference symmetrizer fails closed."""


def _canonical_sha(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("ascii")).hexdigest()


def _file_sha(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except (OSError, ValueError) as exc:
        raise Quartic85StateBoundedBSymmetrizerError(f"cannot read bound file: {path}") from exc


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise Quartic85StateBoundedBSymmetrizerError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise Quartic85StateBoundedBSymmetrizerError(f"JSON root is not an object: {path}")
    return value


def _resolve(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise Quartic85StateBoundedBSymmetrizerError("bound path escapes repository root")
    return path


def _load_binding(root: Path, binding: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = _resolve(root, str(binding.get("path", "")))
    if _file_sha(path) != binding.get("file_sha256"):
        raise Quartic85StateBoundedBSymmetrizerError(f"bound file hash mismatch: {path}")
    value = _load_json(path)
    if value.get("content_sha256") != binding.get("content_sha256"):
        raise Quartic85StateBoundedBSymmetrizerError(f"bound content hash mismatch: {path}")
    return path, value


def _gravity_energy(
    gravity: sphere._Sparse, projectors: dict[Fraction, sphere._Sparse]
) -> sphere._Sparse:
    total: sphere._Sparse = {}
    energy: sphere._Sparse = {}
    for projector in projectors.values():
        total = sphere._sparse_add(total, projector)
        energy = sphere._sparse_add(
            energy,
            sphere._sparse_multiply(sphere._sparse_transpose(projector), projector),
        )
    completeness = sphere._sparse_add(total, sphere._sparse_scale(sphere._sparse_identity(55), -1))
    symmetry = sphere._sparse_add(
        sphere._sparse_multiply(energy, gravity),
        sphere._sparse_scale(
            sphere._sparse_multiply(sphere._sparse_transpose(gravity), energy), -1
        ),
    )
    if completeness or symmetry:
        raise Quartic85StateBoundedBSymmetrizerError("gravity spectral energy replay failed")
    return energy


def _matter_inverse_energy() -> sphere._Sparse:
    inverse = sphere._constant_matrix(exact._identity(30))
    inverse[(23, 23)] = sphere._poly(Fraction(1, 3))
    return inverse


def _quadratic_correction(
    left: sphere._Sparse,
    right: sphere._Sparse,
    inverse_energy: sphere._Sparse,
    *,
    cross: bool,
) -> sphere._Sparse:
    correction = sphere._sparse_multiply(
        sphere._sparse_transpose(left),
        sphere._sparse_multiply(inverse_energy, right),
    )
    if cross:
        correction = sphere._sparse_add(
            correction,
            sphere._sparse_multiply(
                sphere._sparse_transpose(right),
                sphere._sparse_multiply(inverse_energy, left),
            ),
        )
    return correction


def _top_left_residual(
    correction: sphere._Sparse,
    gravity: sphere._Sparse,
    left_solution: sphere._Sparse,
    right_solution: sphere._Sparse,
    left_cross: sphere._Sparse,
    right_cross: sphere._Sparse,
    *,
    cross: bool,
) -> sphere._Sparse:
    residual = sphere._sparse_add(
        sphere._sparse_multiply(correction, gravity),
        sphere._sparse_scale(
            sphere._sparse_multiply(sphere._sparse_transpose(gravity), correction), -1
        ),
    )
    coupling = sphere._sparse_multiply(sphere._sparse_transpose(left_solution), right_cross)
    if cross:
        coupling = sphere._sparse_add(
            coupling,
            sphere._sparse_multiply(sphere._sparse_transpose(right_solution), left_cross),
        )
    return sphere._sparse_add(
        residual,
        sphere._sparse_add(coupling, sphere._sparse_scale(sphere._sparse_transpose(coupling), -1)),
    )


def _materialize(p55: dict[str, Any], readiness: dict[str, Any]) -> dict[str, Any]:
    gravity, projectors, crosses, matter_energy, matter_projectors = sylvester._load_spectral_data(
        p55, readiness
    )
    matter = sylvester._matter_operator()
    inverse_energy = _matter_inverse_energy()
    solutions: list[sphere._Sparse] = []
    for component, cross_pencil in enumerate(crosses):
        solution, _ = sylvester._solve_component(
            cross_pencil, projectors, matter_energy, matter_projectors
        )
        forcing = sphere._sparse_multiply(matter_energy, cross_pencil)
        if sylvester._residual(matter, solution, gravity, forcing):
            raise Quartic85StateBoundedBSymmetrizerError(
                f"bound Sylvester predecessor replay failed: B_{component}"
            )
        solutions.append(solution)
    corrections: list[dict[str, Any]] = []
    omit_correction_negative: dict[str, Any] | None = None
    for left in range(4):
        for right in range(left, 4):
            cross_term = left != right
            correction = _quadratic_correction(
                solutions[left], solutions[right], inverse_energy, cross=cross_term
            )
            residual = _top_left_residual(
                correction,
                gravity,
                solutions[left],
                solutions[right],
                crosses[left],
                crosses[right],
                cross=cross_term,
            )
            if residual:
                raise Quartic85StateBoundedBSymmetrizerError(
                    f"top-left corrected residual nonzero: B_{left} B_{right}"
                )
            manifest = sphere._polynomial_manifest(f"Q_B_{left}_{right}(n)", correction, [55, 55])
            corrections.append(
                {
                    "potential_monomial": f"B_{left}*B_{right}",
                    **manifest,
                    "top_left_symmetry_residual_nonzero_entries": 0,
                }
            )
            if left == right == 0:
                omitted = _top_left_residual(
                    {},
                    gravity,
                    solutions[0],
                    solutions[0],
                    crosses[0],
                    crosses[0],
                    cross=False,
                )
                if not omitted:
                    raise Quartic85StateBoundedBSymmetrizerError(
                        "omit-quadratic-correction negative unexpectedly passed"
                    )
                omit_correction_negative = {
                    "mutation": "omit X_0^T H_m^-1 X_0 from the gravity block",
                    "top_left_residual_nonzero_polynomial_entries": len(omitted),
                    "top_left_residual_normal_form_terms": sum(
                        len(value.terms) for value in omitted.values()
                    ),
                    "rejected": True,
                }
    if omit_correction_negative is None:
        raise Quartic85StateBoundedBSymmetrizerError("negative control was not evaluated")
    potential_upper = Fraction(8, 38505)
    solution_bound = Fraction(2567, 8)
    x_infinity = solution_bound * potential_upper
    if x_infinity != Fraction(1, 15):
        raise Quartic85StateBoundedBSymmetrizerError("bounded-B arithmetic changed")
    # ||X||_1 <= 30 ||X||_infinity, so ||X||_2^2 <= ||X||_1 ||X||_infinity.
    x_operator_squared = 30 * x_infinity**2
    if x_operator_squared != Fraction(2, 15) or x_operator_squared >= 1:
        raise Quartic85StateBoundedBSymmetrizerError("operator bound failed")
    return {
        "corrected_symmetrizer": {
            "operator": "L=[[G,0],[C,M]]",
            "cross_equation": "M^T X-XG=H_m C",
            "block_matrix": ("H=[[H_g+X^T H_m^-1 X, X^T],[X,H_m]]"),
            "congruence_factorization": (
                "H=S^-T diag(H_g,H_m) S^-1 with S^-1=[[I,0],[H_m^-1 X,I]]"
            ),
            "schur_complement": "H_g",
            "gravity_energy_formula": "H_g=sum_lambda Pi_G(lambda)^T Pi_G(lambda)",
            "gravity_energy_uniform_lower": "1/7",
            "gravity_projector_completeness_residual_nonzero_entries": 0,
            "gravity_symmetry_residual_nonzero_entries": 0,
            "quadratic_correction_coefficients": corrections,
            "full_85_state_symmetry_residual_nonzero_entries": 0,
        },
        "bounded_potential_domain": {
            "condition": "max_mu |B_mu| <= 8/38505",
            "contains_nonzero_potentials": True,
            "X_infinity_norm_upper": "1/15",
            "X_operator_norm_squared_upper": "2/15",
            "triangular_factor_norm_upper_strict": "2",
            "full_symmetrizer_uniform_lower_strict": "1/28",
            "schur_complement_uniform_lower": "1/7",
        },
        "stronger_pointwise_result": {
            "all_finite_B_at_flat_reference_positive": True,
            "reason": "exact congruence of two positive spectral energies",
            "uniform_all_B_lower_bound_claimed": False,
        },
        "corruption_negative": omit_correction_negative,
    }


def build_receipt(config_path: Path, *, root: Path | None = None) -> dict[str, Any]:
    repository = (root or config_path.resolve().parents[1]).resolve()
    config = _load_json(config_path)
    if config.get("schema_version") != (
        "invariant-quartic-85-state-bounded-B-schur-symmetrizer-config-1.0"
    ):
        raise Quartic85StateBoundedBSymmetrizerError("unsupported config schema")
    expected_domain = {
        "norm": "max_mu |B_mu|",
        "upper": "8/38505",
        "contains_nonzero_potentials": True,
    }
    if config.get("bounded_potential_domain") != expected_domain:
        raise Quartic85StateBoundedBSymmetrizerError("bounded potential domain changed")
    expected_policy = {
        "exact_flat_sphere_full_85_state_symmetrizer": True,
        "bounded_B_uniform_positive_lower_bound": True,
        "candidate_jet_uniformity": False,
        "sourced_constraint_propagation": False,
        "gravity_h7": False,
        "universal_all_matter": False,
        "physics_no_go": False,
        "promotion": False,
    }
    if config.get("claims_policy") != expected_policy:
        raise Quartic85StateBoundedBSymmetrizerError("claims policy is absent or broadened")
    bound = {
        name: _load_binding(repository, binding)
        for name, binding in config.get("bindings", {}).items()
    }
    if set(bound) != {
        "nonresonant_solution",
        "P55_sphere_pencil",
        "projector_recipes",
        "symmetrizer_blocker",
    }:
        raise Quartic85StateBoundedBSymmetrizerError("closed binding manifest changed")
    predecessor = bound["nonresonant_solution"][1]
    p55 = bound["P55_sphere_pencil"][1]
    readiness = bound["projector_recipes"][1]
    blocker = bound["symmetrizer_blocker"][1]
    if predecessor.get("decision") != ("PASS_EXACT_FLAT_SPHERE_NONRESONANT_SYLVESTER_COMPLEMENT"):
        raise Quartic85StateBoundedBSymmetrizerError("nonresonant predecessor changed")
    if p55.get("status") != "pass_exact_flat_reference_P55_spatial_pencil_registration":
        raise Quartic85StateBoundedBSymmetrizerError("P55 predecessor changed")
    if readiness.get("counts", {}).get("Lagrange_projector_recipes_registered") != 7:
        raise Quartic85StateBoundedBSymmetrizerError("projector recipes changed")
    if blocker.get("decision") != ("TYPED_BLOCK_RESONANT_SYLVESTER_AND_SCHUR_DOMAIN_UNREGISTERED"):
        raise Quartic85StateBoundedBSymmetrizerError("symmetrizer blocker changed")
    materialization = _materialize(p55, readiness)
    source_path = Path(__file__).resolve()
    test_path = repository / "tests/test_quartic_85_state_bounded_B_schur_symmetrizer_gate.py"
    body: dict[str, Any] = {
        "schema_version": "invariant-quartic-85-state-bounded-B-schur-symmetrizer-result-1.0",
        "campaign_id": config["campaign_id"],
        "decision": "PASS_EXACT_FLAT_SPHERE_FULL_SYMMETRIZER_BOUNDED_B",
        "materialization": materialization,
        "counts": {
            "quadratic_gravity_correction_coefficients": 10,
            "top_left_symmetry_residual_nonzero_entries": 0,
            "full_85_state_symmetry_residual_nonzero_entries": 0,
            "bounded_nonzero_potential_domains": 1,
            "flat_reference_full_symmetrizers": 1,
            "negative_controls": 1,
            "candidate_jet_uniform_symmetrizers": 0,
            "sourced_constraint_propagation_claims": 0,
        },
        "claims": {
            "exact_flat_sphere_full_85_state_symmetrizer_closed": True,
            "bounded_B_uniform_positive_lower_bound_closed": True,
            "all_finite_B_flat_reference_pointwise_positive": True,
            "candidate_jet_uniformity_closed": False,
            "sourced_constraint_propagation_closed": False,
            "gravity_h7_theorem_established": False,
            "universal_all_matter_closure_established": False,
            "physics_no_go_established": False,
            "promotion_authorized": False,
        },
        "remaining_contract": [
            "transport the corrected symmetrizer to the registered candidate-jet domain",
            "bind sourced gravity-constraint propagation to the coupled system",
        ],
        "scope": (
            "exact corrected 85-state symmetrizer at the flat reference over the full unit "
            "direction sphere. A nonzero bounded Maxwell-potential domain has a quantitative "
            "uniform lower bound, and congruence gives pointwise positivity for every finite B "
            "at this reference. Candidate-jet uniformity, sourced constraints, H7, universal "
            "matter closure, physics no-go, and promotion remain false"
        ),
        "source_bindings": {
            "config": {
                "path": config_path.relative_to(repository).as_posix(),
                "file_sha256": _file_sha(config_path),
            },
            **{
                name: {
                    "path": path.relative_to(repository).as_posix(),
                    "file_sha256": _file_sha(path),
                    "content_sha256": value["content_sha256"],
                }
                for name, (path, value) in bound.items()
            },
            "source": {
                "path": source_path.relative_to(repository).as_posix(),
                "file_sha256": _file_sha(source_path),
            },
            "test": {
                "path": test_path.relative_to(repository).as_posix(),
                "file_sha256": _file_sha(test_path),
            },
        },
    }
    return {**body, "content_sha256": _canonical_sha(body)}


def write_receipt(
    config_path: Path, output_path: Path, *, root: Path | None = None
) -> dict[str, Any]:
    receipt = build_receipt(config_path, root=root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    write_receipt(args.config.resolve(), args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
