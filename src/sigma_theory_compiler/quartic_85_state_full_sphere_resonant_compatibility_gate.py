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
    from . import quartic_85_state_resonant_projector_compatibility_gate as exact
except ImportError:  # pragma: no cover - supports direct receipt materialization
    _path = Path(__file__).with_name("quartic_85_state_resonant_projector_compatibility_gate.py")
    _spec = importlib.util.spec_from_file_location("_resonant_exact", _path)
    if _spec is None or _spec.loader is None:
        raise
    exact = importlib.util.module_from_spec(_spec)
    sys.modules["_resonant_exact"] = exact
    _spec.loader.exec_module(exact)


class Quartic85StateFullSphereCompatibilityError(RuntimeError):
    """Raised when the full-sphere exact compatibility replay fails closed."""


def _canonical_sha(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("ascii")).hexdigest()


def _file_sha(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except (OSError, ValueError) as exc:
        raise Quartic85StateFullSphereCompatibilityError(f"cannot read bound file: {path}") from exc


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise Quartic85StateFullSphereCompatibilityError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise Quartic85StateFullSphereCompatibilityError(f"JSON root is not an object: {path}")
    return value


def _resolve(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise Quartic85StateFullSphereCompatibilityError("bound path escapes repository root")
    return path


def _load_binding(root: Path, binding: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = _resolve(root, str(binding.get("path", "")))
    if _file_sha(path) != binding.get("file_sha256"):
        raise Quartic85StateFullSphereCompatibilityError(f"bound file hash mismatch: {path}")
    value = _load_json(path)
    if value.get("content_sha256") != binding.get("content_sha256"):
        raise Quartic85StateFullSphereCompatibilityError(f"bound content hash mismatch: {path}")
    return path, value


class _Polynomial:
    """Q(sqrt(2))[n1,n2,n3]/(n1^2+n2^2+n3^2-1) normal form."""

    def __init__(self, terms: dict[tuple[int, int, int], exact._Surd] | None = None):
        self.terms = {key: value for key, value in (terms or {}).items() if value != exact._ZERO}
        self._reduce()

    def _reduce(self) -> None:
        pending = list(self.terms.items())
        self.terms = {}
        while pending:
            (power1, power2, power3), value = pending.pop()
            if value == exact._ZERO:
                continue
            if power1 >= 2:
                pending.extend(
                    [
                        ((power1 - 2, power2, power3), value),
                        ((power1 - 2, power2 + 2, power3), -value),
                        ((power1 - 2, power2, power3 + 2), -value),
                    ]
                )
            else:
                key = (power1, power2, power3)
                self.terms[key] = self.terms.get(key, exact._ZERO) + value
        self.terms = {key: value for key, value in self.terms.items() if value != exact._ZERO}

    def __add__(self, other: object) -> _Polynomial:
        value = _poly(other)
        terms = self.terms.copy()
        for key, coefficient in value.terms.items():
            terms[key] = terms.get(key, exact._ZERO) + coefficient
        return _Polynomial(terms)

    __radd__ = __add__

    def __neg__(self) -> _Polynomial:
        return _Polynomial({key: -value for key, value in self.terms.items()})

    def __sub__(self, other: object) -> _Polynomial:
        return self + (-_poly(other))

    def __mul__(self, other: object) -> _Polynomial:
        value = _poly(other)
        terms: dict[tuple[int, int, int], exact._Surd] = {}
        for left, left_value in self.terms.items():
            for right, right_value in value.terms.items():
                key = tuple(left[index] + right[index] for index in range(3))
                terms[key] = terms.get(key, exact._ZERO) + left_value * right_value
        return _Polynomial(terms)

    __rmul__ = __mul__


_PZERO = _Polynomial()
_PONE = _Polynomial({(0, 0, 0): exact._ONE})
_N = [
    _Polynomial({tuple(1 if index == axis else 0 for index in range(3)): exact._ONE})
    for axis in range(3)
]
_Sparse = dict[tuple[int, int], _Polynomial]


def _poly(value: object) -> _Polynomial:
    if isinstance(value, _Polynomial):
        return value
    return _Polynomial({(0, 0, 0): exact._surd(value)})


def _constant_matrix(matrix: exact._Matrix) -> _Sparse:
    return {
        (row, column): _Polynomial({(0, 0, 0): value})
        for row, values in enumerate(matrix)
        for column, value in enumerate(values)
        if value != exact._ZERO
    }


def _sparse_add(left: _Sparse, right: _Sparse) -> _Sparse:
    result = left.copy()
    for key, value in right.items():
        result[key] = result.get(key, _PZERO) + value
    return {key: value for key, value in result.items() if value.terms}


def _sparse_scale(matrix: _Sparse, coefficient: object) -> _Sparse:
    return {
        key: value * coefficient for key, value in matrix.items() if (value * coefficient).terms
    }


def _sparse_multiply(left: _Sparse, right: _Sparse) -> _Sparse:
    rows: dict[int, list[tuple[int, _Polynomial]]] = {}
    for (row, column), value in right.items():
        rows.setdefault(row, []).append((column, value))
    result: _Sparse = {}
    for (row, pivot), left_value in left.items():
        for column, right_value in rows.get(pivot, []):
            key = (row, column)
            result[key] = result.get(key, _PZERO) + left_value * right_value
    return {key: value for key, value in result.items() if value.terms}


def _sparse_transpose(matrix: _Sparse) -> _Sparse:
    return {(column, row): value for (row, column), value in matrix.items()}


def _sparse_identity(size: int) -> _Sparse:
    return {(index, index): _PONE for index in range(size)}


def _evaluate(
    matrix: _Sparse, direction: tuple[int, int, int], rows: int, columns: int
) -> exact._Matrix:
    result = exact._zeros(rows, columns)
    for (row, column), polynomial in matrix.items():
        value = exact._ZERO
        for powers, coefficient in polynomial.terms.items():
            value += coefficient * Fraction(
                direction[0] ** powers[0] * direction[1] ** powers[1] * direction[2] ** powers[2]
            )
        result[row][column] = value
    return result


def _polynomial_manifest(name: str, matrix: _Sparse, shape: list[int]) -> dict[str, Any]:
    payload = [
        {
            "row": row,
            "column": column,
            "terms": [
                {"powers": list(powers), "coefficient": coefficient.text()}
                for powers, coefficient in sorted(polynomial.terms.items())
            ],
        }
        for (row, column), polynomial in sorted(matrix.items())
    ]
    return {
        "name": name,
        "shape": shape,
        "nonzero_polynomial_entries": len(payload),
        "normal_form_terms": sum(len(item["terms"]) for item in payload),
        "maximum_terms_per_entry": max(len(item["terms"]) for item in payload),
        "normal_form_sha256": _canonical_sha(payload),
        "sphere_normal_form": "power(n1)<=1 modulo n1^2+n2^2+n3^2-1",
    }


def _full_projectors(
    axes: list[exact._Matrix], recipes: dict[int, list[Fraction]]
) -> tuple[_Sparse, dict[int, _Sparse]]:
    pencil: _Sparse = {}
    for axis, matrix in enumerate(axes):
        pencil = _sparse_add(pencil, _sparse_scale(_constant_matrix(matrix), _N[axis]))
    powers = [_sparse_identity(55)]
    for _ in range(6):
        powers.append(_sparse_multiply(powers[-1], pencil))
    projectors: dict[int, _Sparse] = {}
    for sign, coefficients in recipes.items():
        projector: _Sparse = {}
        for power, coefficient in zip(powers, coefficients, strict=True):
            projector = _sparse_add(projector, _sparse_scale(power, coefficient))
        projectors[sign] = projector
    return pencil, projectors


def _metric_coefficient(coordinate: int, left: int, right: int) -> exact._Surd:
    pairs = [(a, b) for a in range(4) for b in range(a, 4)]
    pair = pairs[coordinate]
    if tuple(sorted((left, right))) != pair:
        return exact._ZERO
    return exact._ONE if pair[0] == pair[1] else exact._Surd(radical=Fraction(1, 2))


def _second_order_coefficients(
    potential_component: int,
) -> tuple[exact._Matrix, list[exact._Matrix], list[list[exact._Matrix]]]:
    signature = [-1, 1, 1, 1]
    pairs = [(a, b) for a in range(4) for b in range(a, 4)]
    coefficient_a = exact._zeros(6, 11)
    coefficient_b = [exact._zeros(6, 11) for _ in range(3)]
    coefficient_c = [[exact._zeros(6, 11) for _ in range(3)] for _ in range(3)]
    for equation in range(4):
        for coordinate, pair in enumerate(pairs):
            monomials: dict[tuple[int, int], exact._Surd] = {}
            for contracted in range(4):
                gamma = exact._surd(
                    signature[contracted] * signature[potential_component]
                ) * _metric_coefficient(coordinate, contracted, potential_component)
                if contracted == potential_component and pair[0] == pair[1]:
                    gamma += exact._surd(-signature[potential_component] * signature[pair[0]]) / 2
                if gamma != exact._ZERO:
                    key = tuple(sorted((equation, contracted)))
                    monomials[key] = monomials.get(key, exact._ZERO) - gamma
            row = 1 + equation
            coefficient_a[row][coordinate] = monomials.get((0, 0), exact._ZERO)
            for spatial in range(3):
                coefficient_b[spatial][row][coordinate] = monomials.get(
                    (0, spatial + 1), exact._ZERO
                )
                coefficient_c[spatial][spatial][row][coordinate] = monomials.get(
                    (spatial + 1, spatial + 1), exact._ZERO
                )
                for other in range(spatial + 1, 3):
                    value = monomials.get((spatial + 1, other + 1), exact._ZERO) / 2
                    coefficient_c[spatial][other][row][coordinate] = value
                    coefficient_c[other][spatial][row][coordinate] = value
    return coefficient_a, coefficient_b, coefficient_c


def _cross_axes(axes: list[exact._Matrix]) -> list[list[exact._Matrix]]:
    all_components: list[list[exact._Matrix]] = []
    spatial_offsets = [44, 11, 22]
    for component in range(4):
        coefficient_a, coefficient_b, coefficient_c = _second_order_coefficients(component)
        component_axes: list[exact._Matrix] = []
        for axis, gravity in enumerate(axes):
            cross = exact._zeros(30, 55)
            for matter_field in range(6):
                matter_a = Fraction(-3 if matter_field == 5 else -1)
                for column in range(55):
                    value = exact._ZERO
                    if 33 <= column < 44:
                        value -= coefficient_b[axis][matter_field][column - 33]
                    for flux, offset in enumerate(spatial_offsets):
                        if offset <= column < offset + 11:
                            value -= coefficient_c[axis][flux][matter_field][column - offset]
                    for pivot in range(11):
                        value -= coefficient_a[matter_field][pivot] * gravity[33 + pivot][column]
                    cross[18 + matter_field][column] = value / matter_a
            component_axes.append(cross)
        all_components.append(component_axes)
    return all_components


def _matter_projectors() -> tuple[_Sparse, dict[int, _Sparse]]:
    energy, _ = exact._matter_data()
    projectors: dict[int, _Sparse] = {}
    half = Fraction(1, 2)
    offsets = [24, 6, 12]
    for sign in (-1, 1):
        projector: _Sparse = {}
        for field in range(5):
            projector[(18 + field, 18 + field)] = _poly(half)
            for axis, offset in enumerate(offsets):
                projector[(18 + field, offset + field)] = _N[axis] * Fraction(sign, 2)
                projector[(offset + field, 18 + field)] = _N[axis] * Fraction(sign, 2)
                for other, other_offset in enumerate(offsets):
                    projector[(offset + field, other_offset + field)] = _N[axis] * _N[other] * half
        projectors[sign] = projector
    return _constant_matrix(energy), projectors


def _materialize(
    predecessor: dict[str, Any], p55: dict[str, Any], readiness: dict[str, Any]
) -> dict[str, Any]:
    axes = [
        exact._matrix_from_packet(
            next(item for item in p55["matrix_packets"] if item["name"] == f"P_{axis}")
        )
        for axis in (1, 2, 3)
    ]
    recipe_records = readiness["exact_Lagrange_projector_recipes"]["recipes"]
    recipes = {
        sign: [
            Fraction(value)
            for value in next(item for item in recipe_records if item["eigenvalue"] == str(sign))[
                "coefficients_low_to_high"
            ]
        ]
        for sign in (-1, 1)
    }
    pencil, gravity_projectors = _full_projectors(axes, recipes)
    projector_identity_records: list[dict[str, Any]] = []
    for sign, projector in gravity_projectors.items():
        idempotence = _sparse_add(
            _sparse_multiply(projector, projector), _sparse_scale(projector, -1)
        )
        eigenidentity = _sparse_add(
            _sparse_multiply(pencil, projector), _sparse_scale(projector, -sign)
        )
        if idempotence or eigenidentity:
            raise Quartic85StateFullSphereCompatibilityError(
                f"full Riesz projector identity failed: root={sign}"
            )
        projector_identity_records.append(
            {
                "root": str(sign),
                "idempotence_sphere_normal_form_nonzero_entries": 0,
                "eigenidentity_sphere_normal_form_nonzero_entries": 0,
            }
        )
    cross_axes = _cross_axes(axes)
    cross_pencils: list[_Sparse] = []
    for matrices in cross_axes:
        pencil_component: _Sparse = {}
        for axis, matrix in enumerate(matrices):
            pencil_component = _sparse_add(
                pencil_component, _sparse_scale(_constant_matrix(matrix), _N[axis])
            )
        cross_pencils.append(pencil_component)
    energy, matter_projectors = _matter_projectors()
    projection_results: list[dict[str, Any]] = []
    for sign in (-1, 1):
        left = _sparse_multiply(_sparse_transpose(matter_projectors[sign]), energy)
        for component, cross in enumerate(cross_pencils):
            residual = _sparse_multiply(_sparse_multiply(left, cross), gravity_projectors[sign])
            if residual:
                raise Quartic85StateFullSphereCompatibilityError(
                    f"sphere resonant residual nonzero: root={sign}, B_{component}"
                )
            projection_results.append(
                {
                    "root": str(sign),
                    "potential_component": f"B_{component}",
                    "sphere_normal_form_nonzero_entries": 0,
                    "sphere_normal_form_terms": 0,
                }
            )
    old_projectors = predecessor["materialization"]["vacuum"]["projectors"]
    axis_regression: list[dict[str, Any]] = []
    for axis, direction in enumerate(((1, 0, 0), (0, 1, 0), (0, 0, 1)), start=1):
        for sign in (-1, 1):
            full = _evaluate(gravity_projectors[sign], direction, 55, 55)
            restricted = exact._matrix_from_packet(old_projectors[str(sign)])
            difference = exact._subtract(full, restricted)
            outside = sum(
                full[row][column] != exact._ZERO
                for row in range(55)
                for column in range(55)
                if not (33 <= row < 55 and 33 <= column < 55)
            )
            axis_regression.append(
                {
                    "axis": axis,
                    "root": str(sign),
                    "full_projector_nonzero_entries": exact._nonzero_count(full),
                    "outside_restricted_companion_support": outside,
                    "difference_from_restricted_packet_nonzero_entries": exact._nonzero_count(
                        difference
                    ),
                }
            )
    expected = [
        (1, -1, 0, 0),
        (1, 1, 0, 0),
        (2, -1, 18, 44),
        (2, 1, 18, 44),
        (3, -1, 18, 44),
        (3, 1, 18, 44),
    ]
    observed = [
        (
            item["axis"],
            int(item["root"]),
            item["outside_restricted_companion_support"],
            item["difference_from_restricted_packet_nonzero_entries"],
        )
        for item in axis_regression
    ]
    if observed != expected:
        raise Quartic85StateFullSphereCompatibilityError("full-support regression changed")
    return {
        "sphere_ring": "Q(sqrt(2))[n1,n2,n3]/(n1^2+n2^2+n3^2-1)",
        "vacuum_pencil": _polynomial_manifest("P55(n)", pencil, [55, 55]),
        "full_vacuum_projectors": {
            str(sign): _polynomial_manifest(f"Pi_{sign}(P55(n))", matrix, [55, 55])
            for sign, matrix in gravity_projectors.items()
        },
        "full_projector_identity_reductions": projector_identity_records,
        "full_maxwell_cross_pencils": [
            _polynomial_manifest(f"C_B_{component}(n)", matrix, [30, 55])
            for component, matrix in enumerate(cross_pencils)
        ],
        "cross_axis_nonzero_counts": [
            [exact._nonzero_count(matrix) for matrix in component] for component in cross_axes
        ],
        "resonant_sphere_reductions": projection_results,
        "full_support_regression": axis_regression,
        "supersession": {
            "predecessor_content_sha256": predecessor["content_sha256"],
            "predecessor_e1_zero_conclusion_remains_valid": True,
            "predecessor_serialized_projector_and_cross_packets": (
                "companion-restricted e1 controls; non-authoritative for full-block extension"
            ),
            "authoritative_representation_from_this_receipt": (
                "full 55-state degree-six Riesz projectors and full 30x55 cross pencils"
            ),
            "predecessor_modified": False,
        },
    }


def build_receipt(config_path: Path, *, root: Path | None = None) -> dict[str, Any]:
    repository = (root or config_path.resolve().parents[1]).resolve()
    config = _load_json(config_path)
    if config.get("schema_version") != (
        "invariant-quartic-85-state-full-sphere-resonant-compatibility-config-1.0"
    ):
        raise Quartic85StateFullSphereCompatibilityError("unsupported config schema")
    expected_policy = {
        "full_55_state_riesz_projectors_over_unit_sphere": True,
        "full_30x55_maxwell_cross_pencil": True,
        "exact_unit_sphere_resonant_compatibility": True,
        "candidate_jet_uniformity": False,
        "nonresonant_sylvester_solution": False,
        "bounded_B_schur_positivity": False,
        "full_coupled_symmetrizer": False,
        "constraint_propagation": False,
        "gravity_h7": False,
        "physics_no_go": False,
        "promotion": False,
    }
    if config.get("claims_policy") != expected_policy:
        raise Quartic85StateFullSphereCompatibilityError("claims policy is absent or broadened")
    bound = {
        name: _load_binding(repository, binding)
        for name, binding in config.get("bindings", {}).items()
    }
    if set(bound) != {
        "restricted_e1_predecessor",
        "P55_sphere_pencil",
        "projector_recipes",
        "maxwell_mixed_principal",
    }:
        raise Quartic85StateFullSphereCompatibilityError("closed binding manifest changed")
    predecessor = bound["restricted_e1_predecessor"][1]
    p55 = bound["P55_sphere_pencil"][1]
    readiness = bound["projector_recipes"][1]
    maxwell = bound["maxwell_mixed_principal"][1]
    if predecessor.get("decision") != "PASS_EXACT_FLAT_E1_RESONANT_COMPATIBILITY_ALL_B_COMPONENTS":
        raise Quartic85StateFullSphereCompatibilityError("e1 predecessor changed")
    if p55.get("status") != "pass_exact_flat_reference_P55_spatial_pencil_registration":
        raise Quartic85StateFullSphereCompatibilityError("P55 sphere predecessor changed")
    if readiness.get("counts", {}).get("Lagrange_projector_recipes_registered") != 7:
        raise Quartic85StateFullSphereCompatibilityError("projector recipes changed")
    if maxwell.get("decision") != ("PASS_EXACT_NONZERO_MAXWELL_MIXED_BLOCK_AND_17_FIELD_PRINCIPAL"):
        raise Quartic85StateFullSphereCompatibilityError("Maxwell predecessor changed")
    materialization = _materialize(predecessor, p55, readiness)
    source_path = Path(__file__).resolve()
    test_path = repository / (
        "tests/test_quartic_85_state_full_sphere_resonant_compatibility_gate.py"
    )
    body: dict[str, Any] = {
        "schema_version": (
            "invariant-quartic-85-state-full-sphere-resonant-compatibility-result-1.0"
        ),
        "campaign_id": config["campaign_id"],
        "decision": "PASS_EXACT_FULL_SPHERE_RESONANT_COMPATIBILITY_FLAT_REFERENCE",
        "materialization": materialization,
        "counts": {
            "spatial_pencil_coefficients": 3,
            "full_vacuum_projectors": 2,
            "full_projector_identity_reductions": 4,
            "full_maxwell_cross_coefficients": 12,
            "resonant_sphere_reductions": 8,
            "resonant_normal_form_nonzero_entries": 0,
            "axis_support_regressions": 6,
            "nonresonant_sylvester_solutions": 0,
            "bounded_B_schur_domains": 0,
            "full_coupled_symmetrizers": 0,
        },
        "claims": {
            "full_55_state_riesz_projectors_over_unit_sphere_materialized": True,
            "full_30x55_maxwell_cross_pencil_materialized": True,
            "exact_unit_sphere_resonant_compatibility_closed": True,
            "candidate_jet_uniformity_closed": False,
            "nonresonant_sylvester_solution_closed": False,
            "bounded_B_schur_positivity_closed": False,
            "full_coupled_symmetrizer_closed": False,
            "constraint_propagation_closed": False,
            "gravity_h7_theorem_established": False,
            "physics_no_go_established": False,
            "promotion_authorized": False,
        },
        "remaining_contract": [
            "solve and bound the nonresonant Sylvester complement over the unit sphere",
            "transport from the flat reference to the registered candidate-jet domain",
            "prove bounded-Maxwell-potential Schur-complement positivity",
        ],
        "scope": (
            "exact flat-reference unit-sphere extension in the registered 85-state basis: "
            "full 55-state degree-six Riesz projectors and full 30x55 Maxwell cross pencils "
            "reduce all eight +/-1 resonant forcings to zero modulo the sphere relation. "
            "This supersedes only the predecessor packet representation, not its valid e1 "
            "zero conclusion. Nonresonant Sylvester bounds, candidate-jet uniformity, bounded-B "
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
