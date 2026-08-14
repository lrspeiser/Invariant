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
    from . import quartic_85_state_full_sphere_resonant_compatibility_gate as sphere
except ImportError:  # pragma: no cover - supports direct receipt materialization
    _path = Path(__file__).with_name("quartic_85_state_full_sphere_resonant_compatibility_gate.py")
    _spec = importlib.util.spec_from_file_location("_sphere_exact", _path)
    if _spec is None or _spec.loader is None:
        raise
    sphere = importlib.util.module_from_spec(_spec)
    sys.modules["_sphere_exact"] = sphere
    _spec.loader.exec_module(sphere)

exact = sphere.exact


class Quartic85StateNonresonantSylvesterError(RuntimeError):
    """Raised when the exact nonresonant Sylvester replay fails closed."""


def _canonical_sha(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("ascii")).hexdigest()


def _file_sha(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except (OSError, ValueError) as exc:
        raise Quartic85StateNonresonantSylvesterError(f"cannot read bound file: {path}") from exc


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise Quartic85StateNonresonantSylvesterError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise Quartic85StateNonresonantSylvesterError(f"JSON root is not an object: {path}")
    return value


def _resolve(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise Quartic85StateNonresonantSylvesterError("bound path escapes repository root")
    return path


def _load_binding(root: Path, binding: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = _resolve(root, str(binding.get("path", "")))
    if _file_sha(path) != binding.get("file_sha256"):
        raise Quartic85StateNonresonantSylvesterError(f"bound file hash mismatch: {path}")
    value = _load_json(path)
    if value.get("content_sha256") != binding.get("content_sha256"):
        raise Quartic85StateNonresonantSylvesterError(f"bound content hash mismatch: {path}")
    return path, value


def _matter_operator() -> sphere._Sparse:
    operator: sphere._Sparse = {}
    spatial_offsets = [24, 6, 12]
    for field in range(6):
        inverse_time = Fraction(1, 3 if field == 5 else 1)
        for axis, offset in enumerate(spatial_offsets):
            operator[(18 + field, offset + field)] = sphere._N[axis] * inverse_time
            operator[(offset + field, 18 + field)] = sphere._N[axis]
    return operator


def _load_spectral_data(
    p55: dict[str, Any], readiness: dict[str, Any]
) -> tuple[
    sphere._Sparse,
    dict[Fraction, sphere._Sparse],
    list[sphere._Sparse],
    sphere._Sparse,
    dict[int, sphere._Sparse],
]:
    axes = [
        exact._matrix_from_packet(
            next(item for item in p55["matrix_packets"] if item["name"] == f"P_{axis}")
        )
        for axis in (1, 2, 3)
    ]
    recipe_records = readiness["exact_Lagrange_projector_recipes"]["recipes"]
    roots = [
        Fraction(-1),
        Fraction(-1, 2),
        Fraction(-1, 3),
        Fraction(0),
        Fraction(1, 3),
        Fraction(1, 2),
        Fraction(1),
    ]
    recipes = {
        root: [
            Fraction(value)
            for value in next(
                item for item in recipe_records if Fraction(item["eigenvalue"]) == root
            )["coefficients_low_to_high"]
        ]
        for root in roots
    }
    gravity, gravity_projectors = sphere._full_projectors(axes, recipes)
    cross_axes = sphere._cross_axes(axes)
    cross_pencils: list[sphere._Sparse] = []
    for matrices in cross_axes:
        pencil: sphere._Sparse = {}
        for axis, matrix in enumerate(matrices):
            pencil = sphere._sparse_add(
                pencil,
                sphere._sparse_scale(sphere._constant_matrix(matrix), sphere._N[axis]),
            )
        cross_pencils.append(pencil)
    energy, matter_projectors = sphere._matter_projectors()
    return gravity, gravity_projectors, cross_pencils, energy, matter_projectors


def _solve_component(
    cross: sphere._Sparse,
    gravity_projectors: dict[Fraction, sphere._Sparse],
    energy: sphere._Sparse,
    matter_projectors: dict[int, sphere._Sparse],
) -> tuple[sphere._Sparse, list[dict[str, Any]]]:
    forcing = sphere._sparse_multiply(energy, cross)
    solution: sphere._Sparse = {}
    blocks: list[dict[str, Any]] = []
    for matter_root in (-1, 1):
        left_forcing = sphere._sparse_multiply(
            sphere._sparse_transpose(matter_projectors[matter_root]), forcing
        )
        for gravity_root, projector in gravity_projectors.items():
            block = sphere._sparse_multiply(left_forcing, projector)
            if gravity_root == matter_root:
                if block:
                    raise Quartic85StateNonresonantSylvesterError(
                        f"resonant block became nonzero: {matter_root}"
                    )
                continue
            inverse_gap = Fraction(1, 1) / (Fraction(matter_root) - gravity_root)
            solution = sphere._sparse_add(solution, sphere._sparse_scale(block, inverse_gap))
            blocks.append(
                {
                    "matter_root": str(matter_root),
                    "gravity_root": str(gravity_root),
                    "gap": str(Fraction(matter_root) - gravity_root),
                    "inverse_gap": str(inverse_gap),
                    "forcing_polynomial_entries": len(block),
                    "forcing_normal_form_terms": sum(len(value.terms) for value in block.values()),
                }
            )
    return solution, blocks


def _residual(
    matter: sphere._Sparse,
    solution: sphere._Sparse,
    gravity: sphere._Sparse,
    forcing: sphere._Sparse,
) -> sphere._Sparse:
    result = sphere._sparse_add(
        sphere._sparse_multiply(sphere._sparse_transpose(matter), solution),
        sphere._sparse_scale(sphere._sparse_multiply(solution, gravity), -1),
    )
    return sphere._sparse_add(result, sphere._sparse_scale(forcing, -1))


def _coefficient_upper(value: exact._Surd) -> Fraction:
    # sqrt(2) < 2 gives an exact rational majorant.
    return abs(value.rational) + 2 * abs(value.radical)


def _uniform_infinity_bound(matrix: sphere._Sparse, rows: int) -> Fraction:
    row_bounds = [Fraction(0) for _ in range(rows)]
    for (row, _), polynomial in matrix.items():
        row_bounds[row] += sum(
            _coefficient_upper(coefficient) for coefficient in polynomial.terms.values()
        )
    return max(row_bounds)


def _solution_manifest(component: int, solution: sphere._Sparse) -> dict[str, Any]:
    manifest = sphere._polynomial_manifest(f"X_B_{component}(n)", solution, [30, 55])
    bound = _uniform_infinity_bound(solution, 30)
    return {
        **manifest,
        "uniform_unit_sphere_infinity_norm_upper": str(bound),
        "bound_proof": ("|n_i|<=1 and |a+b*sqrt(2)|<=|a|+2|b|, followed by exact row sums"),
    }


def _constructive_negative(
    cross: sphere._Sparse,
    gravity: sphere._Sparse,
    gravity_projectors: dict[Fraction, sphere._Sparse],
    energy: sphere._Sparse,
    matter: sphere._Sparse,
    matter_projectors: dict[int, sphere._Sparse],
    solution: sphere._Sparse,
) -> dict[str, Any]:
    forcing = sphere._sparse_multiply(energy, cross)
    left = sphere._sparse_multiply(sphere._sparse_transpose(matter_projectors[1]), forcing)
    block = sphere._sparse_multiply(left, gravity_projectors[Fraction(1, 2)])
    if not block:
        raise Quartic85StateNonresonantSylvesterError("negative-control block is empty")
    # Correct inverse gap is 2. Replacing it by 1 subtracts one copy of this block.
    corrupted = sphere._sparse_add(solution, sphere._sparse_scale(block, -1))
    residual = _residual(matter, corrupted, gravity, forcing)
    if not residual:
        raise Quartic85StateNonresonantSylvesterError("inverse-gap corruption unexpectedly passed")
    return {
        "mutation": "replace inverse gap (1-1/2)^-1=2 by 1 in the B_0 block",
        "residual_nonzero_polynomial_entries": len(residual),
        "residual_normal_form_terms": sum(len(value.terms) for value in residual.values()),
        "residual_normal_form_sha256": sphere._polynomial_manifest(
            "corrupted_residual", residual, [30, 55]
        )["normal_form_sha256"],
        "rejected": True,
    }


def _materialize(p55: dict[str, Any], readiness: dict[str, Any]) -> dict[str, Any]:
    gravity, projectors, crosses, energy, matter_projectors = _load_spectral_data(p55, readiness)
    forcing_row_support = sorted({row for cross in crosses for row, _ in cross})
    if forcing_row_support != [19, 20, 21, 22]:
        raise Quartic85StateNonresonantSylvesterError("Maxwell-only matter forcing support changed")
    matter = _matter_operator()
    solutions: list[sphere._Sparse] = []
    manifests: list[dict[str, Any]] = []
    block_records: list[dict[str, Any]] = []
    for component, cross in enumerate(crosses):
        solution, blocks = _solve_component(cross, projectors, energy, matter_projectors)
        forcing = sphere._sparse_multiply(energy, cross)
        residual = _residual(matter, solution, gravity, forcing)
        if residual:
            raise Quartic85StateNonresonantSylvesterError(
                f"Sylvester residual nonzero: B_{component}"
            )
        solutions.append(solution)
        manifests.append(_solution_manifest(component, solution))
        block_records.extend(
            {"potential_component": f"B_{component}", **record} for record in blocks
        )
    nonempty_gaps = {
        abs(Fraction(record["gap"]))
        for record in block_records
        if record["forcing_polynomial_entries"] > 0
    }
    if not nonempty_gaps or min(nonempty_gaps) != Fraction(1, 2):
        raise Quartic85StateNonresonantSylvesterError("active spectral gap census changed")
    negative = _constructive_negative(
        crosses[0],
        gravity,
        projectors,
        energy,
        matter,
        matter_projectors,
        solutions[0],
    )
    return {
        "spectral_contract": {
            "gravity_roots": ["-1", "-1/2", "-1/3", "0", "1/3", "1/2", "1"],
            "active_matter_roots": ["-1", "1"],
            "inactive_forcing_sectors": [
                "matter zero-speed projector",
                "fluid +/-1/sqrt(3) projectors",
            ],
            "forcing_row_support_in_30_state_basis": forcing_row_support,
            "inactive_sector_annihilation_basis": (
                "C has rows only in Maxwell velocities 19..22; zero-speed left modes have "
                "no velocity support and fluid left modes have support only at fluid indices"
            ),
            "smallest_active_nonresonant_gap": "1/2",
            "largest_active_inverse_gap": "2",
            "solution_formula": ("X=sum_(s=+/-1,mu!=s) Pi_M(s)^T H_m C Pi_G(mu)/(s-mu)"),
        },
        "spectral_blocks": block_records,
        "solutions": manifests,
        "sylvester_residual_sphere_normal_form_nonzero_entries": [0, 0, 0, 0],
        "combined_arbitrary_B_infinity_norm_upper": str(
            sum(Fraction(item["uniform_unit_sphere_infinity_norm_upper"]) for item in manifests)
        ),
        "combined_bound_interpretation": (
            "||X(B,n)||_infinity <= bound*max_mu|B_mu| on the exact flat unit sphere"
        ),
        "corruption_negative": negative,
    }


def build_receipt(config_path: Path, *, root: Path | None = None) -> dict[str, Any]:
    repository = (root or config_path.resolve().parents[1]).resolve()
    config = _load_json(config_path)
    if config.get("schema_version") != (
        "invariant-quartic-85-state-nonresonant-sylvester-complement-config-1.0"
    ):
        raise Quartic85StateNonresonantSylvesterError("unsupported config schema")
    expected_policy = {
        "exact_flat_sphere_nonresonant_sylvester_solution": True,
        "exact_uniform_coefficient_bound": True,
        "candidate_jet_uniformity": False,
        "bounded_B_schur_positivity": False,
        "full_coupled_symmetrizer": False,
        "constraint_propagation": False,
        "gravity_h7": False,
        "physics_no_go": False,
        "promotion": False,
    }
    if config.get("claims_policy") != expected_policy:
        raise Quartic85StateNonresonantSylvesterError("claims policy is absent or broadened")
    bound = {
        name: _load_binding(repository, binding)
        for name, binding in config.get("bindings", {}).items()
    }
    if set(bound) != {
        "full_sphere_resonant_predecessor",
        "P55_sphere_pencil",
        "projector_recipes",
        "symmetrizer_blocker",
    }:
        raise Quartic85StateNonresonantSylvesterError("closed binding manifest changed")
    predecessor = bound["full_sphere_resonant_predecessor"][1]
    p55 = bound["P55_sphere_pencil"][1]
    readiness = bound["projector_recipes"][1]
    blocker = bound["symmetrizer_blocker"][1]
    if predecessor.get("decision") != (
        "PASS_EXACT_FULL_SPHERE_RESONANT_COMPATIBILITY_FLAT_REFERENCE"
    ):
        raise Quartic85StateNonresonantSylvesterError("sphere predecessor changed")
    if p55.get("status") != "pass_exact_flat_reference_P55_spatial_pencil_registration":
        raise Quartic85StateNonresonantSylvesterError("P55 predecessor changed")
    if readiness.get("counts", {}).get("Lagrange_projector_recipes_registered") != 7:
        raise Quartic85StateNonresonantSylvesterError("projector recipes changed")
    if blocker.get("decision") != ("TYPED_BLOCK_RESONANT_SYLVESTER_AND_SCHUR_DOMAIN_UNREGISTERED"):
        raise Quartic85StateNonresonantSylvesterError("symmetrizer blocker changed")
    materialization = _materialize(p55, readiness)
    source_path = Path(__file__).resolve()
    test_path = repository / (
        "tests/test_quartic_85_state_nonresonant_sylvester_complement_gate.py"
    )
    body: dict[str, Any] = {
        "schema_version": (
            "invariant-quartic-85-state-nonresonant-sylvester-complement-result-1.0"
        ),
        "campaign_id": config["campaign_id"],
        "decision": "PASS_EXACT_FLAT_SPHERE_NONRESONANT_SYLVESTER_COMPLEMENT",
        "materialization": materialization,
        "counts": {
            "potential_component_solutions": 4,
            "spectral_block_records": len(materialization["spectral_blocks"]),
            "exact_sylvester_residual_nonzero_entries": 0,
            "uniform_solution_bounds": 4,
            "negative_controls": 1,
            "candidate_jet_uniform_solutions": 0,
            "bounded_B_schur_domains": 0,
            "full_coupled_symmetrizers": 0,
        },
        "claims": {
            "exact_flat_sphere_nonresonant_sylvester_solution_closed": True,
            "exact_uniform_solution_coefficient_bound_closed": True,
            "candidate_jet_uniformity_closed": False,
            "bounded_B_schur_positivity_closed": False,
            "full_coupled_symmetrizer_closed": False,
            "constraint_propagation_closed": False,
            "gravity_h7_theorem_established": False,
            "physics_no_go_established": False,
            "promotion_authorized": False,
        },
        "remaining_contract": [
            "transport the flat-sphere solution and bound to the candidate-jet domain",
            "register a compatible gravity-energy inverse/lower bound in the same basis",
            "derive a nonzero bounded-B domain for Schur-complement positivity",
        ],
        "scope": (
            "exact constructive solution of M^T X-XG=H_m C on every active nonresonant "
            "spectral block at the flat reference over the full unit direction sphere, with "
            "exact rational coefficient bounds. The zero and +/-1 resonances are annihilated "
            "by the registered forcing/projector structure. Candidate-jet uniformity, bounded-B "
            "Schur positivity, a full symmetrizer, constraints, H7, and physics no-go remain false"
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
