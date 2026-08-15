from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

SCHEMA = "sigma-quartic-tc2-d4-k55-taylor-order-zero-registration-1.0"
CONFIG_SCHEMA = "sigma-quartic-tc2-d4-k55-taylor-order-zero-registration-config-1.0"
STATUS = "block_coordinate_free_D4_recurrence_emitter_missing_255_symbolic_packets"
CONFIG_PATH = "configs/backgrounds/quartic_tc2_d4_k55_taylor_order_zero_registration.json"
SOURCE_PATH = "src/sigma_theory_compiler/quartic_tc2_d4_k55_taylor_order_zero_registration.py"
TEST_PATH = "tests/test_quartic_tc2_d4_k55_taylor_order_zero_registration.py"
REQUIRED_PACKETS = 304
PREDECESSOR_REGISTERED = 34
NEW_PACKETS = 15
REGISTERED_PACKETS = 49
MISSING_PACKETS = 255
REQUIRED_ROWS = 117_180
EXPECTED_UPSTREAMS = {
    "manifest_predecessor": {
        "path": "runs/physics-language/quartic-tc2-d4-coordinate-free-p55-taylor-order-zero-registration/campaign.json",
        "file_sha256": "e0814fa3fd8a046fec4cbd85c230faa652260dc934d4096f6e3c38b73381d091",
        "content_sha256": "71bbd9966e3212440ef92952eab7f06d860de3f59dfaa861753f03dd53850273",
    },
    "flat_action_metric": {
        "path": "runs/physics-language/quartic-tc2-d4-flat-action-metric-registration/campaign.json",
        "file_sha256": "ef0e7bb964947e4b19b6b62aa1e7e571abe66c17c23e60070956098d8e101941",
        "content_sha256": "82b89bb823c0382e6a80918dd03bc7a1d46f059845a41ced86922dd79c30fd73",
    },
    "P55_checkpoint": {
        "path": "runs/physics-language/quartic-tc2-d4-p55-checkpointable-materializer/result.json",
        "file_sha256": "9e79245d45c248d5abf1b2fd11b12bb7ca0c857286d31f8bf46b7ec7c43490b2",
        "content_sha256": "25ced42ac83bd592332f3d3a8bc97eeaac9c55b32ba4bb2be3e3da1fa6b503b7",
    },
    "projector_recipes": {
        "path": "runs/physics-language/quartic-tc2-d4-coordinate-free-symbolic-recurrence-emitter-readiness/campaign.json",
        "file_sha256": "f8846ef19ba340c24622839beedcf25547974c64a68ab9d7cb3490e979198001",
        "content_sha256": "893b5b5daacc749a593d4eddd709e0e61f63f9f7954f46f02c0fd6970f48badb",
    },
}
EXPECTED_STATUSES = {
    "manifest_predecessor": "block_coordinate_free_D4_recurrence_emitter_missing_270_symbolic_packets",
    "flat_action_metric": "pass_exact_flat_action_metric_h_plus_0_registration",
    "P55_checkpoint": "pass_exact_flat_reference_P55_spatial_pencil_registration",
    "projector_recipes": "block_coordinate_free_D4_recurrence_emitter_missing_symbolic_P_and_Taylor_packets",
}


class K55TaylorOrderZeroRegistrationError(ValueError):
    """Raised when exact K55 Taylor-order-zero registration fails closed."""


@dataclass(frozen=True)
class Surd:
    rational: Fraction = Fraction()
    radical: Fraction = Fraction()

    def __add__(self, other: object) -> Surd:
        value = _surd(other)
        return Surd(self.rational + value.rational, self.radical + value.radical)

    __radd__ = __add__

    def __neg__(self) -> Surd:
        return Surd(-self.rational, -self.radical)

    def __sub__(self, other: object) -> Surd:
        return self + (-_surd(other))

    def __rsub__(self, other: object) -> Surd:
        return _surd(other) - self

    def __mul__(self, other: object) -> Surd:
        value = _surd(other)
        return Surd(
            self.rational * value.rational + 2 * self.radical * value.radical,
            self.rational * value.radical + self.radical * value.rational,
        )

    __rmul__ = __mul__

    def __truediv__(self, other: object) -> Surd:
        value = _surd(other)
        denominator = value.rational**2 - 2 * value.radical**2
        if not denominator:
            raise ZeroDivisionError("zero Q(sqrt(2)) denominator")
        return self * Surd(value.rational / denominator, -value.radical / denominator)

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


ZERO = Surd()
ONE = Surd(Fraction(1))
Matrix = list[list[Surd]]


def _surd(value: object) -> Surd:
    if isinstance(value, Surd):
        return value
    return Surd(Fraction(value))  # type: ignore[arg-type]


def parse_surd(text: str) -> Surd:
    value = text.replace(" ", "")
    if "sqrt(2)" not in value:
        return Surd(Fraction(value))
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
            raise K55TaylorOrderZeroRegistrationError(f"unsupported surd: {text}")
        coefficient /= Fraction(suffix[1:])
    return Surd(radical=coefficient)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "ascii"
    )


def _content_hash(value: dict[str, Any]) -> str:
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    return hashlib.sha256(_canonical_bytes(body)).hexdigest()


def _hash_matches(value: dict[str, Any]) -> bool:
    return value.get("content_sha256") == _content_hash(value)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise K55TaylorOrderZeroRegistrationError(f"expected JSON object: {path}")
    return value


def _resolve_under(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root != path and root not in path.parents:
        raise K55TaylorOrderZeroRegistrationError("bound path escaped project root")
    return path


def zeros(rows: int, columns: int) -> Matrix:
    return [[ZERO for _ in range(columns)] for _ in range(rows)]


def identity(size: int) -> Matrix:
    result = zeros(size, size)
    for index in range(size):
        result[index][index] = ONE
    return result


def add(left: Matrix, right: Matrix) -> Matrix:
    return [
        [left[row][column] + right[row][column] for column in range(len(left[0]))]
        for row in range(len(left))
    ]


def scale(matrix: Matrix, coefficient: object) -> Matrix:
    return [[value * coefficient for value in row] for row in matrix]


def transpose(matrix: Matrix) -> Matrix:
    return [list(row) for row in zip(*matrix, strict=True)]


def multiply(left: Matrix, right: Matrix) -> Matrix:
    rows, inner, columns = len(left), len(right), len(right[0])
    if len(left[0]) != inner:
        raise K55TaylorOrderZeroRegistrationError("matrix product shape mismatch")
    result = zeros(rows, columns)
    for row in range(rows):
        for pivot in range(inner):
            coefficient = left[row][pivot]
            if coefficient == ZERO:
                continue
            for column in range(columns):
                if right[pivot][column] != ZERO:
                    result[row][column] += coefficient * right[pivot][column]
    return result


def submatrix(matrix: Matrix, rows: range, columns: range) -> Matrix:
    return [[matrix[row][column] for column in columns] for row in rows]


def matrix_from_packet(packet: dict[str, Any]) -> Matrix:
    rows, columns = packet["shape"]
    matrix = zeros(rows, columns)
    for entry in packet["entries"]:
        matrix[entry["row"]][entry["column"]] = parse_surd(entry["value"])
    return matrix


def matrix_packet(name: str, matrix: Matrix) -> dict[str, Any]:
    entries = [
        {"row": row, "column": column, "value": value.text()}
        for row, values in enumerate(matrix)
        for column, value in enumerate(values)
        if value != ZERO
    ]
    body = {
        "schema_version": "sigma-exact-sparse-Qsqrt2-matrix-1.0",
        "name": name,
        "shape": [len(matrix), len(matrix[0])],
        "entries": entries,
        "nonzero_count": len(entries),
    }
    return {**body, "content_sha256": _content_hash(body)}


def _projectors(companion: Matrix, recipes: list[dict[str, Any]]) -> dict[Fraction, Matrix]:
    powers = [identity(22)]
    for _ in range(6):
        powers.append(multiply(powers[-1], companion))
    projectors: dict[Fraction, Matrix] = {}
    for recipe in recipes:
        eigenvalue = Fraction(recipe["eigenvalue"])
        if eigenvalue == 0:
            continue
        coefficients = [Fraction(value) for value in recipe["coefficients_low_to_high"]]
        projector = zeros(22, 22)
        for coefficient, power in zip(coefficients, powers, strict=True):
            projector = add(projector, scale(power, coefficient))
        projectors[eigenvalue] = projector
    if set(projectors) != {
        Fraction(1),
        Fraction(-1),
        Fraction(1, 2),
        Fraction(-1, 2),
        Fraction(1, 3),
        Fraction(-1, 3),
    }:
        raise K55TaylorOrderZeroRegistrationError("projector spectrum mismatch")
    total = zeros(22, 22)
    for eigenvalue, projector in projectors.items():
        total = add(total, projector)
        if multiply(projector, projector) != projector:
            raise K55TaylorOrderZeroRegistrationError(f"projector idempotence failed: {eigenvalue}")
        if multiply(companion, projector) != scale(projector, eigenvalue):
            raise K55TaylorOrderZeroRegistrationError(
                f"projector eigenidentity failed: {eigenvalue}"
            )
    if total != identity(22):
        raise K55TaylorOrderZeroRegistrationError("projector completeness failed")
    return projectors


def construct_K0(
    p1_packet: dict[str, Any], h_packet: dict[str, Any], recipes: list[dict[str, Any]]
) -> dict[str, Any]:
    if not _hash_matches(p1_packet) or not _hash_matches(h_packet):
        raise K55TaylorOrderZeroRegistrationError("input matrix packet seal mismatch")
    physical = matrix_from_packet(p1_packet)
    h_plus = matrix_from_packet(h_packet)
    companion = submatrix(physical, range(33, 55), range(33, 55))
    coupling = submatrix(physical, range(33, 55), range(33))
    projectors = _projectors(companion, recipes)
    companion_energy = zeros(22, 22)
    for eigenvalue, projector in projectors.items():
        metric = (
            h_plus if eigenvalue == 1 else scale(h_plus, -1) if eigenvalue == -1 else identity(22)
        )
        companion_energy = add(
            companion_energy,
            multiply(multiply(transpose(projector), metric), projector),
        )
    inverse = zeros(22, 22)
    for eigenvalue, projector in projectors.items():
        inverse = add(inverse, scale(projector, Fraction(1, 1) / eigenvalue))
    if multiply(companion, inverse) != identity(22):
        raise K55TaylorOrderZeroRegistrationError("spectral companion inverse failed")
    cross = multiply(multiply(transpose(coupling), companion_energy), inverse)
    energy = zeros(55, 55)
    for index in range(33):
        energy[index][index] = ONE
    for row in range(33):
        for column in range(22):
            energy[row][33 + column] = cross[row][column]
            energy[33 + column][row] = cross[row][column]
    for row in range(22):
        for column in range(22):
            energy[33 + row][33 + column] = companion_energy[row][column]
    if energy != transpose(energy):
        raise K55TaylorOrderZeroRegistrationError("K0 symmetry failed")
    residual = add(multiply(energy, physical), scale(multiply(transpose(physical), energy), -1))
    nonzero_residuals = sum(value != ZERO for row in residual for value in row)
    if nonzero_residuals:
        raise K55TaylorOrderZeroRegistrationError("K0 symmetrizer residual nonzero")
    wrong_energy = zeros(55, 55)
    for index in range(33):
        wrong_energy[index][index] = ONE
    for row in range(22):
        for column in range(22):
            wrong_energy[33 + row][33 + column] = companion_energy[row][column]
    wrong_residual = add(
        multiply(wrong_energy, physical),
        scale(multiply(transpose(physical), wrong_energy), -1),
    )
    wrong_nonzero = sum(value != ZERO for row in wrong_residual for value in row)
    if not wrong_nonzero:
        raise K55TaylorOrderZeroRegistrationError("omit-cross negative control unexpectedly passed")
    projector_packets = [
        matrix_packet(f"Pi_{eigenvalue}", projectors[eigenvalue])
        for eigenvalue in sorted(projectors)
    ]
    return {
        "projector_packets": projector_packets,
        "companion_energy_G0": matrix_packet("G_0", companion_energy),
        "cross_block_X0": matrix_packet("X_0", cross),
        "K0": matrix_packet("K_0", energy),
        "projector_completeness_residual_nonzero_entries": 0,
        "projector_idempotence_residual_nonzero_entries": 0,
        "companion_inverse_residual_nonzero_entries": 0,
        "K0_symmetry_residual_nonzero_entries": 0,
        "K0_P1_symmetrizer_residual_nonzero_entries": 0,
        "omit_cross_block_symmetrizer_nonzero_entries": wrong_nonzero,
    }


def _validate_config(config: dict[str, Any]) -> None:
    if (
        config.get("schema_version") != CONFIG_SCHEMA
        or config.get("policy")
        != "construct_exact_K0_then_register_all_order_zero_packets_fail_closed"
        or not _hash_matches(config)
        or config.get("upstreams") != EXPECTED_UPSTREAMS
        or config.get("target")
        != {
            "required_symbolic_input_packets": REQUIRED_PACKETS,
            "predecessor_registered_packets": PREDECESSOR_REGISTERED,
            "new_K55_order_zero_packets": NEW_PACKETS,
            "expected_registered_packets": REGISTERED_PACKETS,
            "expected_missing_packets": MISSING_PACKETS,
            "required_output_rows": REQUIRED_ROWS,
        }
    ):
        raise K55TaylorOrderZeroRegistrationError("invalid K55 order-zero config")


def _load_bound(root: Path, name: str, binding: dict[str, Any]) -> dict[str, Any]:
    path = _resolve_under(root, binding["path"])
    document = _load_json(path)
    if (
        _file_sha256(path) != binding["file_sha256"]
        or not _hash_matches(document)
        or document.get("content_sha256") != binding["content_sha256"]
        or document.get("status") != EXPECTED_STATUSES[name]
        or document.get("errors", []) != []
    ):
        raise K55TaylorOrderZeroRegistrationError(f"upstream mismatch: {name}")
    return document


def build_campaign(project_root: Path, config_path: Path) -> dict[str, Any]:
    root = project_root.resolve()
    config = _load_json(config_path)
    _validate_config(config)
    upstreams = {
        name: _load_bound(root, name, binding) for name, binding in config["upstreams"].items()
    }
    p1_packet = upstreams["P55_checkpoint"]["matrix_packets"][0]
    h_packet = upstreams["flat_action_metric"]["exact_construction"]["h_plus_0"]
    recipes = upstreams["projector_recipes"]["exact_Lagrange_projector_recipes"]["recipes"]
    exact = construct_K0(p1_packet, h_packet, recipes)
    predecessor = upstreams["manifest_predecessor"]
    evaluations = predecessor["polarization_evaluations"]
    if len(evaluations) != NEW_PACKETS:
        raise K55TaylorOrderZeroRegistrationError("polarization evaluation mismatch")
    packets = []
    for evaluation in evaluations:
        body = {
            "schema_version": "sigma-coordinate-free-K55-Taylor-polarization-packet-1.0",
            "evaluation_id": evaluation["evaluation_id"],
            "evaluation_content_sha256": evaluation["content_sha256"],
            "Taylor_order": 0,
            "factorial_normalization": "1/0!=1",
            "shape": [55, 55],
            "K0_content_sha256": exact["K0"]["content_sha256"],
            "identity": "K55_evaluation_order_0=K_0",
            "jet_direction_independent_at_flat_reference": True,
        }
        packets.append({**body, "content_sha256": _content_hash(body)})
    manifest = json.loads(json.dumps(predecessor["required_symbolic_input_manifest"]))
    records = {row["input_id"]: row for row in manifest}
    family = records.get("polarized_K55_Taylor_packets")
    if (
        family is None
        or family.get("required_packets") != 75
        or family.get("registered_packets") != 0
    ):
        raise K55TaylorOrderZeroRegistrationError("manifest K55 boundary mismatch")
    family.update(
        {
            "registered_packets": NEW_PACKETS,
            "status": "partially_registered_all_15_Taylor_order_zero_packets",
            "registered_Taylor_orders": [0],
            "missing_Taylor_orders": [1, 2, 3, 4],
            "packet_content_sha256": [packet["content_sha256"] for packet in packets],
            "K0_content_sha256": exact["K0"]["content_sha256"],
        }
    )
    if sum(row["registered_packets"] for row in manifest) != REGISTERED_PACKETS:
        raise K55TaylorOrderZeroRegistrationError("updated manifest total mismatch")
    missing = [
        {
            "input_id": row["input_id"],
            "required_packets": row["required_packets"],
            "registered_packets": row["registered_packets"],
            "missing_packets": row["required_packets"] - row["registered_packets"],
        }
        for row in manifest
        if row["registered_packets"] < row["required_packets"]
    ]
    if sum(row["missing_packets"] for row in missing) != MISSING_PACKETS:
        raise K55TaylorOrderZeroRegistrationError("updated missing total mismatch")
    claims = {key: False for key, value in predecessor["claims"].items() if value is False}
    claims.update(
        {
            "exact_flat_K0_constructed": True,
            "all_15_K55_Taylor_order_zero_packets_registered": True,
            "K55_Taylor_orders_one_through_four_registered": False,
            "cold_full_symbol_build_used": False,
            "manifest_recomputed_from_exact_packets": True,
        }
    )
    body = {
        "schema_version": SCHEMA,
        "status": STATUS,
        "errors": [],
        "config_sha256": config["content_sha256"],
        "upstream_bindings": {
            name: {**binding, "verified": True} for name, binding in config["upstreams"].items()
        },
        "exact_K0_construction": exact,
        "registered_K55_Taylor_order_zero_packets": packets,
        "required_symbolic_input_manifest": manifest,
        "remaining_missing_inputs": missing,
        "bounded_emitter_checkpoint": {
            "complete": False,
            "first_missing_input": "polarized_P55_Taylor_packets/orders_1_through_4",
            "required_output_rows": REQUIRED_ROWS,
            "emitted_output_rows": 0,
            "emitted_rhs_rows": 0,
            "emitted_sparse_entries": 0,
        },
        "phase_two": {
            "decision": "BLOCK",
            "admitted": False,
            "attempted": False,
            "blocker": "255 required symbolic input packets remain unregistered",
        },
        "counts": {
            "upstream_seals_verified": 4,
            "full_symbol_build_calls": 0,
            "projectors_constructed": 6,
            "K0_sparse_entries": exact["K0"]["nonzero_count"],
            "K0_P1_symmetrizer_nonzero_residuals": 0,
            "omit_cross_block_symmetrizer_nonzero_residuals": exact[
                "omit_cross_block_symmetrizer_nonzero_entries"
            ],
            "new_K55_Taylor_order_zero_packets_registered": NEW_PACKETS,
            "required_symbolic_input_packets": REQUIRED_PACKETS,
            "predecessor_registered_symbolic_input_packets": PREDECESSOR_REGISTERED,
            "registered_symbolic_input_packets": REGISTERED_PACKETS,
            "missing_symbolic_input_packets": MISSING_PACKETS,
            "required_output_rows": REQUIRED_ROWS,
            "emitted_output_rows": 0,
            "phase_two_solve_attempts": 0,
        },
        "claims": claims,
        "negative_controls": {
            "replace_h_plus_0_with_identity": {"rejected": True},
            "omit_required_cross_block_X0": {"rejected": True},
            "infer_K55_orders_one_through_four_as_zero": {"rejected": True},
            "count_K0_without_symmetrizer_residual": {"rejected": True},
            "emit_rows_with_255_missing_packets": {"rejected": True},
            "promote_order_zero_K55_to_full_D4_or_H7": {"rejected": True},
        },
        "source_bindings": {
            "config": {"path": CONFIG_PATH, "file_sha256": _file_sha256(config_path)},
            "source": {"path": SOURCE_PATH, "file_sha256": _file_sha256(root / SOURCE_PATH)},
            "test": {"path": TEST_PATH, "file_sha256": _file_sha256(root / TEST_PATH)},
        },
        "scope": (
            "Constructs the exact flat K0 symmetrizer from sealed P1, h_plus_0, and projector "
            "recipes, then registers all 15 Taylor-order-zero K55 polarization packets. "
            "Orders one through four and 255 total symbolic packets remain missing; no "
            "recurrence row, D4 theorem, H7 closure, PDE theorem, or lifespan follows."
        ),
    }
    return {**body, "content_sha256": _content_hash(body)}


def validate_campaign(document: dict[str, Any], project_root: Path) -> None:
    expected = build_campaign(project_root.resolve(), project_root.resolve() / CONFIG_PATH)
    if document != expected or not _hash_matches(document):
        raise K55TaylorOrderZeroRegistrationError("campaign replay mismatch")


def write_campaign(document: dict[str, Any], output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "campaign.json"
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    document = build_campaign(args.project_root.resolve(), args.config.resolve())
    print(write_campaign(document, args.output.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
