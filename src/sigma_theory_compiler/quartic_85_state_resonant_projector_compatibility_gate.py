from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any


class Quartic85StateResonantCompatibilityError(RuntimeError):
    """Raised when the exact flat-reference resonant audit cannot be replayed."""


def _canonical_sha(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("ascii")).hexdigest()


def _file_sha(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except (OSError, ValueError) as exc:
        raise Quartic85StateResonantCompatibilityError(f"cannot read bound file: {path}") from exc


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise Quartic85StateResonantCompatibilityError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise Quartic85StateResonantCompatibilityError(f"JSON root is not an object: {path}")
    return value


def _resolve(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise Quartic85StateResonantCompatibilityError("bound path escapes repository root")
    return path


def _load_binding(root: Path, binding: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = _resolve(root, str(binding.get("path", "")))
    if _file_sha(path) != binding.get("file_sha256"):
        raise Quartic85StateResonantCompatibilityError(f"bound file hash mismatch: {path}")
    value = _load_json(path)
    if value.get("content_sha256") != binding.get("content_sha256"):
        raise Quartic85StateResonantCompatibilityError(f"bound content hash mismatch: {path}")
    return path, value


@dataclass(frozen=True)
class _Surd:
    rational: Fraction = Fraction()
    radical: Fraction = Fraction()

    def __add__(self, other: object) -> _Surd:
        value = _surd(other)
        return _Surd(self.rational + value.rational, self.radical + value.radical)

    __radd__ = __add__

    def __neg__(self) -> _Surd:
        return _Surd(-self.rational, -self.radical)

    def __sub__(self, other: object) -> _Surd:
        return self + (-_surd(other))

    def __rsub__(self, other: object) -> _Surd:
        return _surd(other) - self

    def __mul__(self, other: object) -> _Surd:
        value = _surd(other)
        return _Surd(
            self.rational * value.rational + 2 * self.radical * value.radical,
            self.rational * value.radical + self.radical * value.rational,
        )

    __rmul__ = __mul__

    def __truediv__(self, other: object) -> _Surd:
        value = _surd(other)
        denominator = value.rational**2 - 2 * value.radical**2
        if not denominator:
            raise ZeroDivisionError("zero Q(sqrt(2)) denominator")
        return self * _Surd(value.rational / denominator, -value.radical / denominator)

    def text(self) -> str:
        if not self.radical:
            return str(self.rational)
        if not self.rational:
            if self.radical == 1:
                return "sqrt(2)"
            if self.radical == -1:
                return "-sqrt(2)"
            return f"{self.radical}*sqrt(2)"
        sign = "+" if self.radical > 0 else "-"
        magnitude = abs(self.radical)
        radical = "sqrt(2)" if magnitude == 1 else f"{magnitude}*sqrt(2)"
        return f"{self.rational}{sign}{radical}"


_ZERO = _Surd()
_ONE = _Surd(Fraction(1))
_Matrix = list[list[_Surd]]


def _surd(value: object) -> _Surd:
    if isinstance(value, _Surd):
        return value
    return _Surd(Fraction(value))  # type: ignore[arg-type]


def _parse_surd(text: str) -> _Surd:
    value = text.replace(" ", "")
    if "sqrt(2)" not in value:
        return _Surd(Fraction(value))
    prefix, suffix = value.split("sqrt(2)", 1)
    prefix = prefix.rstrip("*")
    if prefix in ("", "+"):
        coefficient = Fraction(1)
    elif prefix == "-":
        coefficient = Fraction(-1)
    else:
        coefficient = Fraction(prefix)
    if suffix:
        if not suffix.startswith("/"):
            raise Quartic85StateResonantCompatibilityError(f"unsupported surd: {text}")
        coefficient /= Fraction(suffix[1:])
    return _Surd(radical=coefficient)


def _zeros(rows: int, columns: int) -> _Matrix:
    return [[_ZERO for _ in range(columns)] for _ in range(rows)]


def _identity(size: int) -> _Matrix:
    result = _zeros(size, size)
    for index in range(size):
        result[index][index] = _ONE
    return result


def _transpose(matrix: _Matrix) -> _Matrix:
    return [list(row) for row in zip(*matrix, strict=True)]


def _multiply(left: _Matrix, right: _Matrix) -> _Matrix:
    if not left or not right or len(left[0]) != len(right):
        raise Quartic85StateResonantCompatibilityError("matrix product shape mismatch")
    result = _zeros(len(left), len(right[0]))
    for row in range(len(left)):
        for pivot, coefficient in enumerate(left[row]):
            if coefficient == _ZERO:
                continue
            for column, value in enumerate(right[pivot]):
                if value != _ZERO:
                    result[row][column] += coefficient * value
    return result


def _subtract(left: _Matrix, right: _Matrix) -> _Matrix:
    return [
        [left[row][column] - right[row][column] for column in range(len(left[0]))]
        for row in range(len(left))
    ]


def _scale(matrix: _Matrix, coefficient: object) -> _Matrix:
    return [[value * coefficient for value in row] for row in matrix]


def _submatrix(matrix: _Matrix, rows: range, columns: range) -> _Matrix:
    return [[matrix[row][column] for column in columns] for row in rows]


def _packet_body_hash(packet: dict[str, Any]) -> str:
    body = {key: value for key, value in packet.items() if key != "content_sha256"}
    return _canonical_sha(body)


def _matrix_from_packet(packet: dict[str, Any]) -> _Matrix:
    if packet.get("content_sha256") != _packet_body_hash(packet):
        raise Quartic85StateResonantCompatibilityError("sparse matrix packet seal mismatch")
    rows, columns = packet.get("shape", [0, 0])
    matrix = _zeros(rows, columns)
    for entry in packet.get("entries", []):
        matrix[entry["row"]][entry["column"]] = _parse_surd(entry["value"])
    return matrix


def _matrix_packet(name: str, matrix: _Matrix) -> dict[str, Any]:
    entries = [
        {"row": row, "column": column, "value": value.text()}
        for row, values in enumerate(matrix)
        for column, value in enumerate(values)
        if value != _ZERO
    ]
    body = {
        "schema_version": "invariant-exact-sparse-Qsqrt2-matrix-1.0",
        "name": name,
        "shape": [len(matrix), len(matrix[0])],
        "entries": entries,
        "nonzero_count": len(entries),
    }
    return {**body, "content_sha256": _canonical_sha(body)}


def _nonzero_count(matrix: _Matrix) -> int:
    return sum(value != _ZERO for row in matrix for value in row)


def _embed(matrix: _Matrix, rows: int, columns: int, row_offset: int, col_offset: int) -> _Matrix:
    result = _zeros(rows, columns)
    for row, values in enumerate(matrix):
        for column, value in enumerate(values):
            result[row_offset + row][col_offset + column] = value
    return result


def _matter_data() -> tuple[_Matrix, dict[int, _Matrix]]:
    operator = _zeros(30, 30)
    for field in range(6):
        operator[18 + field][24 + field] = _surd(Fraction(1, 3) if field == 5 else 1)
        operator[24 + field][18 + field] = _ONE
    energy = _identity(30)
    energy[23][23] = _surd(3)
    if _multiply(energy, operator) != _multiply(_transpose(operator), energy):
        raise Quartic85StateResonantCompatibilityError("matter energy does not symmetrize M")
    projectors: dict[int, _Matrix] = {}
    for sign in (-1, 1):
        projector = _zeros(30, 30)
        for field in range(5):
            projector[18 + field][18 + field] = _surd(Fraction(1, 2))
            projector[24 + field][24 + field] = _surd(Fraction(1, 2))
            projector[18 + field][24 + field] = _surd(Fraction(sign, 2))
            projector[24 + field][18 + field] = _surd(Fraction(sign, 2))
        if _multiply(projector, projector) != projector:
            raise Quartic85StateResonantCompatibilityError("matter projector is not idempotent")
        if _multiply(operator, projector) != _scale(projector, sign):
            raise Quartic85StateResonantCompatibilityError("matter projector eigenidentity failed")
        projectors[sign] = projector
    return energy, projectors


def _second_order_maxwell_coefficients(component: int) -> tuple[_Matrix, _Matrix, _Matrix]:
    coefficient_a = _zeros(6, 11)
    coefficient_b = _zeros(6, 11)
    coefficient_c = _zeros(6, 11)
    sqrt_two_half = _Surd(radical=Fraction(1, 2))

    def add(matrix: _Matrix, row: int, column: int, value: object) -> None:
        matrix[row][column] += _surd(value)

    if component == 0:
        for column in (0, 4, 7, 9):
            add(coefficient_a, 1, column, Fraction(-1, 2))
            add(coefficient_b, 2, column, Fraction(-1, 2))
        add(coefficient_b, 1, 1, sqrt_two_half)
        add(coefficient_c, 2, 1, sqrt_two_half)
    elif component == 1:
        add(coefficient_a, 1, 1, sqrt_two_half)
        add(coefficient_b, 2, 1, sqrt_two_half)
        for column, value in ((0, -1), (4, -1), (7, 1), (9, 1)):
            add(coefficient_b, 1, column, Fraction(value, 2))
            add(coefficient_c, 2, column, Fraction(value, 2))
    elif component == 2:
        add(coefficient_a, 1, 2, sqrt_two_half)
        add(coefficient_b, 2, 2, sqrt_two_half)
        add(coefficient_b, 1, 5, -sqrt_two_half)
        add(coefficient_c, 2, 5, -sqrt_two_half)
    elif component == 3:
        add(coefficient_a, 1, 3, sqrt_two_half)
        add(coefficient_b, 2, 3, sqrt_two_half)
        add(coefficient_b, 1, 6, -sqrt_two_half)
        add(coefficient_c, 2, 6, -sqrt_two_half)
    else:
        raise Quartic85StateResonantCompatibilityError("invalid Maxwell component")
    return coefficient_a, coefficient_b, coefficient_c


def _cross_coefficients(gravity_companion: _Matrix) -> list[_Matrix]:
    gravity_vv = _submatrix(gravity_companion, range(11), range(11))
    gravity_vw = _submatrix(gravity_companion, range(11), range(11, 22))
    coefficients: list[_Matrix] = []
    for component in range(4):
        coefficient_a, coefficient_b, coefficient_c = _second_order_maxwell_coefficients(component)
        cross = _zeros(30, 55)
        for matter_field in range(6):
            matter_a = Fraction(-3 if matter_field == 5 else -1)
            for gravity_field in range(11):
                velocity = coefficient_b[matter_field][gravity_field]
                spatial = coefficient_c[matter_field][gravity_field]
                for pivot in range(11):
                    velocity += (
                        coefficient_a[matter_field][pivot] * gravity_vv[pivot][gravity_field]
                    )
                    spatial += coefficient_a[matter_field][pivot] * gravity_vw[pivot][gravity_field]
                cross[18 + matter_field][33 + gravity_field] = -velocity / matter_a
                cross[18 + matter_field][44 + gravity_field] = -spatial / matter_a
        coefficients.append(cross)
    return coefficients


def _materialize(k55_document: dict[str, Any], p55_document: dict[str, Any]) -> dict[str, Any]:
    construction = k55_document.get("exact_K0_construction", {})
    k55_packet = construction.get("K0")
    if not isinstance(k55_packet, dict):
        raise Quartic85StateResonantCompatibilityError("flat K55 packet is absent")
    k55 = _matrix_from_packet(k55_packet)
    p1_packet = next(
        (item for item in p55_document.get("matrix_packets", []) if item.get("name") == "P_1"),
        None,
    )
    if not isinstance(p1_packet, dict):
        raise Quartic85StateResonantCompatibilityError("flat P1 packet is absent")
    p55 = _matrix_from_packet(p1_packet)
    if _subtract(_multiply(k55, p55), _multiply(_transpose(p55), k55)) != _zeros(55, 55):
        raise Quartic85StateResonantCompatibilityError("K55/P55 symmetrizer identity failed")
    gravity_companion = _submatrix(p55, range(33, 55), range(33, 55))
    source_projectors = {
        item.get("name"): item for item in construction.get("projector_packets", [])
    }
    gravity_projectors: dict[int, _Matrix] = {}
    for sign in (-1, 1):
        packet = source_projectors.get(f"Pi_{sign}")
        if not isinstance(packet, dict):
            raise Quartic85StateResonantCompatibilityError("vacuum light projector is absent")
        companion_projector = _matrix_from_packet(packet)
        if _multiply(companion_projector, companion_projector) != companion_projector:
            raise Quartic85StateResonantCompatibilityError("vacuum projector is not idempotent")
        if _multiply(gravity_companion, companion_projector) != _scale(companion_projector, sign):
            raise Quartic85StateResonantCompatibilityError("vacuum projector eigenidentity failed")
        gravity_projectors[sign] = _embed(companion_projector, 55, 55, 33, 33)
    matter_energy, matter_projectors = _matter_data()
    cross_coefficients = _cross_coefficients(gravity_companion)
    projection_records: list[dict[str, Any]] = []
    for sign in (-1, 1):
        left = _multiply(_transpose(matter_projectors[sign]), matter_energy)
        for component, cross in enumerate(cross_coefficients):
            projection = _multiply(_multiply(left, cross), gravity_projectors[sign])
            nonzero = _nonzero_count(projection)
            if nonzero:
                raise Quartic85StateResonantCompatibilityError(
                    f"nonzero resonant projection: sign={sign}, B_{component}"
                )
            projection_records.append(
                {
                    "root": str(sign),
                    "potential_component": f"B_{component}",
                    "shape": [30, 55],
                    "nonzero_entries": 0,
                    "matrix_sha256": _canonical_sha([["0"] * 55 for _ in range(30)]),
                }
            )
    corrupted = [row[:] for row in cross_coefficients[0]]
    corrupted[19][40] += _ONE
    corruptions: list[dict[str, Any]] = []
    for sign in (-1, 1):
        residual = _multiply(
            _multiply(_multiply(_transpose(matter_projectors[sign]), matter_energy), corrupted),
            gravity_projectors[sign],
        )
        count = _nonzero_count(residual)
        if count == 0:
            raise Quartic85StateResonantCompatibilityError(
                "resonant corruption negative unexpectedly passed"
            )
        first = next(
            (row, column, value.text())
            for row, values in enumerate(residual)
            for column, value in enumerate(values)
            if value != _ZERO
        )
        corruptions.append(
            {
                "root": str(sign),
                "nonzero_entries": count,
                "first_nonzero": {"row": first[0], "column": first[1], "value": first[2]},
            }
        )
    gravity_to_coupled = [
        *range(11),
        *range(51, 62),
        *range(68, 79),
        *range(17, 28),
        *range(34, 45),
    ]
    matter_to_coupled = [
        *range(11, 17),
        *range(62, 68),
        *range(79, 85),
        *range(28, 34),
        *range(45, 51),
    ]
    if sorted([*gravity_to_coupled, *matter_to_coupled]) != list(range(85)):
        raise Quartic85StateResonantCompatibilityError("85-state selection maps are incomplete")
    return {
        "basis": {
            "coupled_85_state_order": ["q_A", "v_A", "w_1A", "w_2A", "w_3A"],
            "field_order": ["gravity_11", "matter_6"],
            "gravity_block_order": ["q_g", "w_2g", "w_3g", "v_g", "w_1g"],
            "matter_block_order": ["q_m", "w_2m", "w_3m", "v_m", "w_1m"],
            "gravity_block_to_coupled85": gravity_to_coupled,
            "matter_block_to_coupled85": matter_to_coupled,
            "permutation_is_bijective": True,
        },
        "vacuum": {
            "K55": k55_packet,
            "projectors": {
                str(sign): _matrix_packet(f"vacuum_Pi_{sign}_55", gravity_projectors[sign])
                for sign in (-1, 1)
            },
            "K55_P55_symmetrizer_residual_nonzero_entries": 0,
            "projector_idempotence_residual_nonzero_entries": 0,
            "projector_eigenidentity_residual_nonzero_entries": 0,
        },
        "matter": {
            "Hm": _matrix_packet("matter_Hm_30", matter_energy),
            "projectors": {
                str(sign): _matrix_packet(f"matter_Pi_{sign}_30", matter_projectors[sign])
                for sign in (-1, 1)
            },
            "Hm_M_symmetrizer_residual_nonzero_entries": 0,
            "positive_diagonal": True,
        },
        "maxwell_cross_block": {
            "identity": "C(B,e1)=sum_{mu=0}^3 B_mu C^(mu)",
            "coefficient_packets": [
                _matrix_packet(f"C_B_{component}_30x55", matrix)
                for component, matrix in enumerate(cross_coefficients)
            ],
            "linearity_covers_arbitrary_B_mu_at_reference": True,
        },
        "resonant_projections": projection_records,
        "corruption_negative": {
            "mutation": "add 1 to C^(0)[19,40] in the registered block basis",
            "results": corruptions,
            "rejected": True,
        },
    }


def build_receipt(config_path: Path, *, root: Path | None = None) -> dict[str, Any]:
    repository = (root or config_path.resolve().parents[1]).resolve()
    config = _load_json(config_path)
    if config.get("schema_version") != (
        "invariant-quartic-85-state-resonant-projector-compatibility-config-1.0"
    ):
        raise Quartic85StateResonantCompatibilityError("unsupported config schema")
    expected_reference = {
        "background": "exact flat order-zero quartic reference",
        "spatial_direction": [1, 0, 0],
        "gravity_block_basis": ["q_g", "w_2g", "w_3g", "v_g", "w_1g"],
        "matter_block_basis": ["q_m", "w_2m", "w_3m", "v_m", "w_1m"],
        "background_maxwell_potential": ("B_mu arbitrary through four exact coefficient matrices"),
    }
    if config.get("reference_contract") != expected_reference:
        raise Quartic85StateResonantCompatibilityError("reference contract changed")
    expected_policy = {
        "exact_flat_reference_e1_basis_materialization": True,
        "exact_resonant_compatibility_all_four_potential_components": True,
        "all_spatial_directions": False,
        "uniform_candidate_jet_domain": False,
        "nonresonant_sylvester_solution": False,
        "full_coupled_symmetrizer": False,
        "sourced_constraint_propagation": False,
        "gravity_h7": False,
        "physics_no_go": False,
        "promotion": False,
    }
    if config.get("claims_policy") != expected_policy:
        raise Quartic85StateResonantCompatibilityError("claims policy is absent or broadened")
    bound = {
        name: _load_binding(repository, binding)
        for name, binding in config.get("bindings", {}).items()
    }
    if set(bound) != {
        "coupled_85_state_reduction",
        "flat_vacuum_K55",
        "flat_vacuum_P55",
        "maxwell_mixed_principal",
        "symmetrizer_blocker",
    }:
        raise Quartic85StateResonantCompatibilityError("closed binding manifest changed")
    reduction = bound["coupled_85_state_reduction"][1]
    k55_document = bound["flat_vacuum_K55"][1]
    p55_document = bound["flat_vacuum_P55"][1]
    maxwell = bound["maxwell_mixed_principal"][1]
    blocker = bound["symmetrizer_blocker"][1]
    if reduction.get("decision") != "PASS_EXACT_85_STATE_FIRST_ORDER_REDUCTION_ALL_TWELVE":
        raise Quartic85StateResonantCompatibilityError("85-state predecessor changed")
    if (
        k55_document.get("claims", {}).get("exact_flat_K0_constructed") is not True
        or k55_document.get("counts", {}).get("projectors_constructed") != 6
    ):
        raise Quartic85StateResonantCompatibilityError("flat K55 predecessor changed")
    if p55_document.get("status") != "pass_exact_flat_reference_P55_spatial_pencil_registration":
        raise Quartic85StateResonantCompatibilityError("flat P55 predecessor changed")
    if maxwell.get("decision") != "PASS_EXACT_NONZERO_MAXWELL_MIXED_BLOCK_AND_17_FIELD_PRINCIPAL":
        raise Quartic85StateResonantCompatibilityError("Maxwell predecessor changed")
    if blocker.get("decision") != "TYPED_BLOCK_RESONANT_SYLVESTER_AND_SCHUR_DOMAIN_UNREGISTERED":
        raise Quartic85StateResonantCompatibilityError("symmetrizer blocker changed")
    materialization = _materialize(k55_document, p55_document)
    source_path = Path(__file__).resolve()
    test_path = repository / "tests/test_quartic_85_state_resonant_projector_compatibility_gate.py"
    body: dict[str, Any] = {
        "schema_version": "invariant-quartic-85-state-resonant-projector-compatibility-result-1.0",
        "campaign_id": config["campaign_id"],
        "decision": "PASS_EXACT_FLAT_E1_RESONANT_COMPATIBILITY_ALL_B_COMPONENTS",
        "materialization": materialization,
        "counts": {
            "reference_points": 1,
            "spatial_directions": 1,
            "vacuum_K55_shape": [55, 55],
            "matter_Hm_shape": [30, 30],
            "cross_coefficient_shapes": 4,
            "cross_coefficient_nonzero_entries": 28,
            "resonant_roots": 2,
            "resonant_component_checks": 8,
            "resonant_projection_entries_checked": 13200,
            "resonant_projection_nonzero_entries": 0,
            "negative_controls": 1,
            "nonresonant_sylvester_solutions": 0,
            "full_coupled_symmetrizers": 0,
            "constraint_propagation_claims": 0,
        },
        "claims": {
            "exact_flat_reference_e1_basis_materialized": True,
            "exact_resonant_compatibility_all_four_potential_components": True,
            "all_spatial_directions_closed": False,
            "uniform_candidate_jet_domain_closed": False,
            "nonresonant_sylvester_solution_closed": False,
            "full_coupled_symmetrizer_closed": False,
            "sourced_constraint_propagation_closed": False,
            "gravity_h7_theorem_established": False,
            "physics_no_go_established": False,
            "promotion_authorized": False,
        },
        "remaining_contract": [
            "transport the projector/materialization identity from e1 over the direction sphere",
            "register candidate-jet Riesz projectors and solve the nonresonant Sylvester blocks",
            "prove a bounded Maxwell-potential Schur-complement positivity domain",
        ],
        "scope": (
            "exact Q(sqrt(2)) materialization at the flat order-zero reference and e1: the "
            "vacuum K55 and +/-1 projectors, matter Hm and +/-1 projectors, and all four "
            "30x55 Maxwell cross coefficients share one explicit permutation into the stored "
            "85-state basis. All eight resonant projections vanish, including arbitrary B_mu "
            "by linearity at this reference. Direction-sphere transport, candidate-jet "
            "uniformity, the nonresonant Sylvester solution, Schur positivity, a full "
            "symmetrizer, constraints, H7, and any physics no-go remain unclaimed"
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
