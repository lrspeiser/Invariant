from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

SCHEMA = "sigma-quartic-tc2-d4-tc2-taylor-order-zero-registration-1.0"
CONFIG_SCHEMA = "sigma-quartic-tc2-d4-tc2-taylor-order-zero-registration-config-1.0"
STATUS = "block_coordinate_free_D4_recurrence_emitter_missing_240_symbolic_packets"
CONFIG_PATH = "configs/backgrounds/quartic_tc2_d4_tc2_taylor_order_zero_registration.json"
SOURCE_PATH = "src/sigma_theory_compiler/quartic_tc2_d4_tc2_taylor_order_zero_registration.py"
TEST_PATH = "tests/test_quartic_tc2_d4_tc2_taylor_order_zero_registration.py"
OUTPUT_PATH = (
    "runs/physics-language/quartic-tc2-d4-tc2-taylor-order-zero-registration/campaign.json"
)
REQUIRED_PACKETS = 304
PREDECESSOR_REGISTERED = 49
NEW_PACKETS = 15
REGISTERED_PACKETS = 64
MISSING_PACKETS = 240
REQUIRED_ROWS = 117_180
PREFACTOR = "a10"
EMBEDDED_Q_COLUMN = {33: Fraction(2), 37: Fraction(-8)}
OUTPUT_COVECTOR_INDEX = 54
EXPECTED_E1_LEGACY_SHA256 = "f439b4f7952f43600bd7078ba6f767f2de57de09a5a4477362a2eabe0aed967b"


class TC2TaylorOrderZeroRegistrationError(ValueError):
    """Raised when the exact TC2 Taylor-order-zero lane fails closed."""


@dataclass(frozen=True)
class Qsqrt2:
    rational: Fraction = Fraction()
    radical: Fraction = Fraction()

    def __add__(self, other: object) -> Qsqrt2:
        value = _qsqrt2(other)
        return Qsqrt2(self.rational + value.rational, self.radical + value.radical)

    __radd__ = __add__

    def __mul__(self, other: object) -> Qsqrt2:
        value = _qsqrt2(other)
        return Qsqrt2(
            self.rational * value.rational + 2 * self.radical * value.radical,
            self.rational * value.radical + self.radical * value.rational,
        )

    __rmul__ = __mul__

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
        coefficient = abs(self.radical)
        radical = "sqrt(2)" if coefficient == 1 else f"{coefficient}*sqrt(2)"
        return f"{self.rational}{sign}{radical}"


ZERO = Qsqrt2()


def _qsqrt2(value: object) -> Qsqrt2:
    if isinstance(value, Qsqrt2):
        return value
    return Qsqrt2(Fraction(value))  # type: ignore[arg-type]


def _parse_qsqrt2(text: str) -> Qsqrt2:
    value = text.replace(" ", "")
    if "sqrt(2)" not in value:
        return Qsqrt2(Fraction(value))
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
            raise TC2TaylorOrderZeroRegistrationError(f"unsupported Q(sqrt(2)) value: {text}")
        coefficient /= Fraction(suffix[1:])
    return Qsqrt2(radical=coefficient)


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


def _resolve_under(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root != path and root not in path.parents:
        raise TC2TaylorOrderZeroRegistrationError("bound path escaped project root")
    return path


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TC2TaylorOrderZeroRegistrationError(f"expected JSON object: {path}")
    return value


def _load_bound(root: Path, binding: dict[str, str]) -> dict[str, Any]:
    path = _resolve_under(root, binding["path"])
    value = _load_json(path)
    if (
        _file_sha256(path) != binding["file_sha256"]
        or value.get("content_sha256") != binding["content_sha256"]
        or not _hash_matches(value)
    ):
        raise TC2TaylorOrderZeroRegistrationError(f"upstream mismatch: {binding['path']}")
    return value


def _validate_raw_binding(root: Path, binding: dict[str, str]) -> None:
    path = _resolve_under(root, binding["path"])
    if not path.is_file() or _file_sha256(path) != binding["file_sha256"]:
        raise TC2TaylorOrderZeroRegistrationError("TC2 formula-source binding changed")


def _validate_config(config: dict[str, Any]) -> None:
    target = config.get("target", {})
    if (
        set(config)
        != {
            "schema_version",
            "policy",
            "upstreams",
            "formula_source",
            "target",
            "seals",
            "content_sha256",
        }
        or config.get("schema_version") != CONFIG_SCHEMA
        or config.get("policy")
        != "construct_exact_unit_TC2_order_zero_from_serialized_P55_fail_closed"
        or not _hash_matches(config)
        or set(config.get("upstreams", {}))
        != {"manifest_predecessor", "P55_order_zero", "registered_operator_reference"}
        or target.get("state_dimension") != 55
        or target.get("embedded_Q_column_10") != {"33": "2", "37": "-8"}
        or target.get("output_covector_index") != OUTPUT_COVECTOR_INDEX
        or target.get("scalar_prefactor") != PREFACTOR
        or target.get("polarization_evaluations") != NEW_PACKETS
        or target.get("Taylor_order") != 0
        or target.get("expected_sparse_linear_coefficients") != 8
        or target.get("expected_distinct_output_rows") != 8
        or target.get("expected_reference_e1_dense_sha256") != EXPECTED_E1_LEGACY_SHA256
        or target.get("predecessor_registered_packets") != PREDECESSOR_REGISTERED
        or target.get("new_packets") != NEW_PACKETS
        or target.get("registered_packets") != REGISTERED_PACKETS
        or target.get("missing_packets") != MISSING_PACKETS
        or target.get("required_packets") != REQUIRED_PACKETS
        or target.get("required_output_rows") != REQUIRED_ROWS
        or any(config.get("seals", {}).values())
    ):
        raise TC2TaylorOrderZeroRegistrationError("invalid TC2 order-zero config")


def _linear_tc2_packet(pencil: dict[str, Any]) -> dict[str, Any]:
    if (
        pencil.get("shape") != [55, 55]
        or pencil.get("variables") != ["n1", "n2", "n3"]
        or not _hash_matches(pencil)
    ):
        raise TC2TaylorOrderZeroRegistrationError("coordinate-free P55 packet mismatch")
    coefficients: dict[tuple[int, str], Qsqrt2] = {}
    for entry in pencil.get("entries", []):
        column = entry["column"]
        if column not in EMBEDDED_Q_COLUMN:
            continue
        for variable, text in entry["linear_coefficients"].items():
            key = (entry["row"], variable)
            coefficients[key] = coefficients.get(key, ZERO) + (
                _parse_qsqrt2(text) * EMBEDDED_Q_COLUMN[column]
            )
    by_row: dict[int, dict[str, str]] = {}
    for (row, variable), coefficient in sorted(coefficients.items()):
        if coefficient != ZERO:
            by_row.setdefault(row, {})[variable] = coefficient.text()
    entries = [
        {"row": row, "column": OUTPUT_COVECTOR_INDEX, "linear_coefficients": values}
        for row, values in sorted(by_row.items())
    ]
    coefficient_count = sum(len(row["linear_coefficients"]) for row in entries)
    if coefficient_count != 8 or len(entries) != 8:
        raise TC2TaylorOrderZeroRegistrationError("unexpected sparse TC2 order-zero census")
    body = {
        "schema_version": "sigma-exact-sparse-linear-Qsqrt2-TC2-order-zero-matrix-1.0",
        "name": "unit_TC2_0",
        "shape": [55, 55],
        "variables": ["n1", "n2", "n3"],
        "scalar_prefactor": PREFACTOR,
        "embedded_Q_column_10": [
            {"row": row, "value": str(value)} for row, value in EMBEDDED_Q_COLUMN.items()
        ],
        "output_covector_index": OUTPUT_COVECTOR_INDEX,
        "identity": "TC2_0(n;a10)=a10*P_0(n)*(2*e_33-8*e_37)*e_54^T",
        "entries": entries,
        "nonzero_linear_coefficient_count": coefficient_count,
        "distinct_output_rows": len(entries),
        "right_support_columns": [OUTPUT_COVECTOR_INDEX],
        "P55_order_zero_content_sha256": pencil["content_sha256"],
    }
    return {**body, "content_sha256": _content_hash(body)}


def _axis_dense_legacy_sha256(packet: dict[str, Any], variable: str) -> str:
    matrix = [["0" for _ in range(55)] for _ in range(55)]
    for entry in packet["entries"]:
        value = entry["linear_coefficients"].get(variable)
        if value is not None:
            matrix[entry["row"]][entry["column"]] = value
    return hashlib.sha256(_canonical_bytes(matrix)).hexdigest()


def _axis_certificates(packet: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "axis": index,
            "variable": variable,
            "legacy_dense_matrix_sha256": _axis_dense_legacy_sha256(packet, variable),
            "nonzero_entries": sum(
                variable in entry["linear_coefficients"] for entry in packet["entries"]
            ),
        }
        for index, variable in enumerate(("n1", "n2", "n3"), start=1)
    ]


def _validate_reference(reference: dict[str, Any], axes: list[dict[str, Any]]) -> None:
    rows = reference.get("exact_no_go", {}).get("registered_block_checks", [])
    variable = next((row for row in rows if row.get("name") == "variable_column_10"), None)
    if (
        variable is None
        or variable.get("block_sha256") != EXPECTED_E1_LEGACY_SHA256
        or axes[0]["legacy_dense_matrix_sha256"] != variable["block_sha256"]
        or variable.get("block_rank") != 1
        or variable.get("right_support_columns") != [OUTPUT_COVECTOR_INDEX]
    ):
        raise TC2TaylorOrderZeroRegistrationError("legacy e1 TC2 reference mismatch")


def _updated_manifest(
    predecessor: dict[str, Any],
    packet: dict[str, Any],
    registered: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    manifest = deepcopy(predecessor["required_symbolic_input_manifest"])
    records = {row["input_id"]: row for row in manifest}
    family = records.get("polarized_TC2_Taylor_packets")
    if (
        len(records) != 8
        or family is None
        or family.get("required_packets") != 75
        or family.get("registered_packets") != 0
        or predecessor.get("counts", {}).get("registered_symbolic_input_packets")
        != PREDECESSOR_REGISTERED
    ):
        raise TC2TaylorOrderZeroRegistrationError("predecessor TC2 manifest boundary mismatch")
    family.update(
        {
            "registered_packets": NEW_PACKETS,
            "status": "partially_registered_all_15_Taylor_order_zero_packets",
            "registered_Taylor_orders": [0],
            "missing_Taylor_orders": [1, 2, 3, 4],
            "unit_TC2_order_zero_content_sha256": packet["content_sha256"],
            "scalar_prefactor": PREFACTOR,
            "packet_content_sha256": [row["content_sha256"] for row in registered],
        }
    )
    if (
        sum(row["registered_packets"] for row in manifest) != REGISTERED_PACKETS
        or sum(row["required_packets"] for row in manifest) != REQUIRED_PACKETS
    ):
        raise TC2TaylorOrderZeroRegistrationError("updated TC2 manifest total mismatch")
    return manifest


def build_campaign(project_root: Path, config_path: Path) -> dict[str, Any]:
    root = project_root.resolve()
    config = _load_json(config_path)
    _validate_config(config)
    upstreams = {name: _load_bound(root, binding) for name, binding in config["upstreams"].items()}
    _validate_raw_binding(root, config["formula_source"])
    predecessor = upstreams["manifest_predecessor"]
    p55 = upstreams["P55_order_zero"]
    if (
        predecessor.get("status")
        != "block_coordinate_free_D4_recurrence_emitter_missing_255_symbolic_packets"
        or len(p55.get("polarization_evaluations", [])) != NEW_PACKETS
    ):
        raise TC2TaylorOrderZeroRegistrationError("TC2 predecessor status mismatch")
    packet = _linear_tc2_packet(p55["coordinate_free_P55_order_zero_matrix"])
    axes = _axis_certificates(packet)
    _validate_reference(upstreams["registered_operator_reference"], axes)
    evaluations = p55["polarization_evaluations"]
    registered = []
    for evaluation in evaluations:
        body = {
            "schema_version": ("sigma-coordinate-free-TC2-Taylor-polarization-packet-1.0"),
            "evaluation_id": evaluation["evaluation_id"],
            "evaluation_content_sha256": evaluation["content_sha256"],
            "Taylor_order": 0,
            "factorial_normalization": "1/0!=1",
            "unit_TC2_order_zero_content_sha256": packet["content_sha256"],
            "scalar_prefactor": PREFACTOR,
            "identity": "TC2_evaluation_order_0(n)=a10*unit_TC2_0(n)",
            "jet_direction_independent_at_flat_reference": True,
            "shape": [55, 55],
        }
        registered.append({**body, "content_sha256": _content_hash(body)})
    manifest = _updated_manifest(predecessor, packet, registered)
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
        raise TC2TaylorOrderZeroRegistrationError("updated missing-packet total mismatch")
    claims = {key: value for key, value in predecessor["claims"].items() if value is False}
    claims.update(
        {
            "exact_coordinate_free_unit_TC2_order_zero_constructed": True,
            "all_15_TC2_Taylor_order_zero_packets_registered": True,
            "TC2_Taylor_orders_one_through_four_registered": False,
            "manifest_recomputed_from_exact_packets": True,
            "cold_full_symbol_build_used": False,
        }
    )
    omit_minus_eight_count = sum(
        entry["column"] == 37 for entry in p55["coordinate_free_P55_order_zero_matrix"]["entries"]
    )
    if omit_minus_eight_count != 6:
        raise TC2TaylorOrderZeroRegistrationError("omit-minus-eight negative control changed")
    body = {
        "schema_version": SCHEMA,
        "status": STATUS,
        "errors": [],
        "config_sha256": config["content_sha256"],
        "upstream_bindings": {
            name: {**binding, "verified": True} for name, binding in config["upstreams"].items()
        },
        "formula_source_binding": {**config["formula_source"], "verified": True},
        "exact_TC2_order_zero_construction": {
            "formula": "TC2_0(n;a10)=a10*P_0(n)*(2*e_33-8*e_37)*e_54^T",
            "unit_matrix": packet,
            "axis_certificates": axes,
            "reference_e1_match": True,
            "reference_e1_legacy_dense_sha256": EXPECTED_E1_LEGACY_SHA256,
            "scalar_prefactor_factored_not_sampled": PREFACTOR,
        },
        "registered_TC2_Taylor_order_zero_packets": registered,
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
            "blocker": "240 required symbolic input packets remain unregistered",
        },
        "counts": {
            "upstream_seals_verified": 3,
            "required_symbolic_input_packets": REQUIRED_PACKETS,
            "predecessor_registered_symbolic_input_packets": PREDECESSOR_REGISTERED,
            "new_TC2_Taylor_order_zero_packets_registered": NEW_PACKETS,
            "registered_symbolic_input_packets": REGISTERED_PACKETS,
            "missing_symbolic_input_packets": MISSING_PACKETS,
            "unit_TC2_sparse_linear_coefficients": packet["nonzero_linear_coefficient_count"],
            "unit_TC2_distinct_output_rows": packet["distinct_output_rows"],
            "unit_TC2_right_support_columns": 1,
            "legacy_reference_axes_checked": 1,
            "omit_minus_eight_changed_coefficients": omit_minus_eight_count,
            "full_symbol_build_calls": 0,
            "required_output_rows": REQUIRED_ROWS,
            "emitted_output_rows": 0,
            "phase_two_solve_attempts": 0,
        },
        "claims": claims,
        "negative_controls": {
            "omit_minus_eight_embedded_Q_term": {
                "rejected": True,
                "changed_coefficients": 6,
            },
            "infer_TC2_orders_one_through_four_as_zero": {"rejected": True},
            "absorb_candidate_a10_into_a_sampled_constant": {"rejected": True},
            "count_order_zero_packet_as_higher_order_packet": {"rejected": True},
            "emit_rows_with_240_missing_packets": {"rejected": True},
            "promote_partial_TC2_family_to_full_D4_or_H7": {"rejected": True},
        },
        "source_bindings": {
            "config": {"path": CONFIG_PATH, "file_sha256": _file_sha256(config_path)},
            "source": {
                "path": SOURCE_PATH,
                "file_sha256": _file_sha256(root / SOURCE_PATH),
            },
            "test": {"path": TEST_PATH, "file_sha256": _file_sha256(root / TEST_PATH)},
        },
        "data_seals": deepcopy(config["seals"]),
        "scope": (
            "Registers only the 15 Taylor-order-zero TC2 polarization packets. The "
            "exact coordinate-free unit block is constructed from the sealed P0(n) "
            "pencil and the registered embedded-Q column, with candidate scalar a10 "
            "left factored. TC2 orders one through four still require erased state-jet "
            "derivatives. No recurrence row, complete D4 theorem, TC2 closure, H7 "
            "closure, PDE theorem, lifespan, physical "
            "no-go, or candidate rejection follows."
        ),
    }
    return {**body, "content_sha256": _content_hash(body)}


def validate_campaign(document: dict[str, Any], project_root: Path) -> None:
    expected = build_campaign(project_root.resolve(), project_root.resolve() / CONFIG_PATH)
    if document != expected or not _hash_matches(document):
        raise TC2TaylorOrderZeroRegistrationError("campaign replay mismatch")


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
