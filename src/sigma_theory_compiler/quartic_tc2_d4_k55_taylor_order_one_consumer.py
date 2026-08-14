from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

try:
    from .quartic_tc2_d4_k55_taylor_order_zero_registration import (
        ZERO,
        Matrix,
        add,
        identity,
        matrix_from_packet,
        matrix_packet,
        multiply,
        parse_surd,
        scale,
        submatrix,
        transpose,
        zeros,
    )
except ImportError:  # pragma: no cover - direct artifact construction
    from quartic_tc2_d4_k55_taylor_order_zero_registration import (
        ZERO,
        Matrix,
        add,
        identity,
        matrix_from_packet,
        matrix_packet,
        multiply,
        parse_surd,
        scale,
        submatrix,
        transpose,
        zeros,
    )

SCHEMA = "sigma-quartic-tc2-d4-k55-taylor-order-one-consumer-1.0"
CONFIG_SCHEMA = f"{SCHEMA.removesuffix('-1.0')}-config-1.0"
STATUS = "block_coordinate_free_K55_order_one_missing_K0_of_n_directional_lift"
CONFIG_PATH = "configs/backgrounds/quartic_tc2_d4_k55_taylor_order_one_consumer.json"
SOURCE_PATH = "src/sigma_theory_compiler/quartic_tc2_d4_k55_taylor_order_one_consumer.py"
TEST_PATH = "tests/test_quartic_tc2_d4_k55_taylor_order_one_consumer.py"
OUTPUT_PATH = "runs/physics-language/quartic-tc2-d4-k55-taylor-order-one-consumer/campaign.json"
AXIS_VARIABLES = ("n1", "n2", "n3")
EIGENVALUES = {
    Fraction(1),
    Fraction(-1),
    Fraction(1, 2),
    Fraction(-1, 2),
    Fraction(1, 3),
    Fraction(-1, 3),
}


class K55TaylorOrderOneConsumerError(ValueError):
    """Raised when exact K55 order-one replay fails closed."""


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
        raise K55TaylorOrderOneConsumerError("bound path escaped project root")
    return path


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise K55TaylorOrderOneConsumerError(f"expected JSON object: {path}")
    return value


def _load_bound(root: Path, binding: dict[str, str]) -> dict[str, Any]:
    path = _resolve_under(root, binding["path"])
    value = _load_json(path)
    if (
        _file_sha256(path) != binding["file_sha256"]
        or value.get("content_sha256") != binding["content_sha256"]
        or not _hash_matches(value)
    ):
        raise K55TaylorOrderOneConsumerError(f"upstream mismatch: {binding['path']}")
    return value


def _validate_config(config: dict[str, Any]) -> None:
    if (
        config.get("schema_version") != CONFIG_SCHEMA
        or config.get("policy")
        != ("construct_exact_K55_order_one_and_advance_only_if_coordinate_free_replay_passes")
        or not _hash_matches(config)
        or set(config.get("upstreams", {}))
        != {
            "manifest_predecessor",
            "K55_order_zero",
            "flat_action_metric",
            "flat_P55",
            "H_star_order_one",
            "projector_recipes",
        }
        or config.get("target", {}).get("required_K55_Taylor_order_one_packets") != 15
        or config.get("target", {}).get("manifest_registered_before") != 79
        or config.get("caps", {}).get("maximum_full_symbol_build_calls") != 0
        or config.get("caps", {}).get("maximum_output_rows_emitted") != 0
    ):
        raise K55TaylorOrderOneConsumerError("invalid K55 order-one consumer config")


def _nonzero_count(matrix: Matrix) -> int:
    return sum(value != ZERO for row in matrix for value in row)


def _subtract(left: Matrix, right: Matrix) -> Matrix:
    return add(left, scale(right, -1))


def _evaluate_polynomial_packet(packet: dict[str, Any], variable: str) -> Matrix:
    if not _hash_matches(packet) or packet.get("shape") not in ([55, 55], [22, 22]):
        raise K55TaylorOrderOneConsumerError("polynomial matrix packet boundary changed")
    rows, columns = packet["shape"]
    matrix = zeros(rows, columns)
    seen = set()
    for entry in packet.get("entries", []):
        coordinate = (entry["row"], entry["column"])
        if coordinate in seen:
            raise K55TaylorOrderOneConsumerError("duplicate polynomial matrix cell")
        value = ZERO
        if "constant" in entry:
            value += parse_surd(entry["constant"])
        if variable in entry.get("linear_coefficients", {}):
            value += parse_surd(entry["linear_coefficients"][variable])
        matrix[coordinate[0]][coordinate[1]] = value
        seen.add(coordinate)
    return matrix


def _powers(matrix: Matrix, maximum: int) -> list[Matrix]:
    result = [identity(len(matrix))]
    for _ in range(maximum):
        result.append(multiply(result[-1], matrix))
    return result


def _projector_pairs(
    companion0: Matrix,
    companion1: Matrix,
    recipes: list[dict[str, Any]],
) -> dict[Fraction, tuple[Matrix, Matrix]]:
    powers0 = _powers(companion0, 6)
    projectors0 = {}
    for recipe in recipes:
        eigenvalue = Fraction(recipe["eigenvalue"])
        if eigenvalue == 0:
            continue
        coefficients = [Fraction(value) for value in recipe["coefficients_low_to_high"]]
        projector0 = zeros(22, 22)
        for coefficient, power0 in zip(coefficients, powers0, strict=True):
            projector0 = add(projector0, scale(power0, coefficient))
        projectors0[eigenvalue] = projector0
    if set(projectors0) != EIGENVALUES:
        raise K55TaylorOrderOneConsumerError("projector spectrum changed")
    result = {}
    derivative_total = zeros(22, 22)
    for eigenvalue, projector0 in projectors0.items():
        projector1 = zeros(22, 22)
        for other, other_projector in projectors0.items():
            if other == eigenvalue:
                continue
            gap = eigenvalue - other
            projector1 = add(
                projector1,
                scale(
                    add(
                        multiply(multiply(other_projector, companion1), projector0),
                        multiply(multiply(projector0, companion1), other_projector),
                    ),
                    Fraction(1, 1) / gap,
                ),
            )
        idempotence1 = _subtract(
            add(multiply(projector1, projector0), multiply(projector0, projector1)),
            projector1,
        )
        commutator1 = _subtract(
            add(multiply(companion1, projector0), multiply(companion0, projector1)),
            add(multiply(projector1, companion0), multiply(projector0, companion1)),
        )
        if _nonzero_count(idempotence1) or _nonzero_count(commutator1):
            raise K55TaylorOrderOneConsumerError("projector derivative replay failed")
        derivative_total = add(derivative_total, projector1)
        result[eigenvalue] = (projector0, projector1)
    if set(result) != EIGENVALUES or _nonzero_count(derivative_total):
        raise K55TaylorOrderOneConsumerError("projector derivative completeness failed")
    return result


def _construct_reference_K1(
    p0: Matrix,
    p1: Matrix,
    k0: Matrix,
    h0: Matrix,
    h1: Matrix,
    recipes: list[dict[str, Any]],
) -> dict[str, Any]:
    companion0 = submatrix(p0, range(33, 55), range(33, 55))
    companion1 = submatrix(p1, range(33, 55), range(33, 55))
    lift0 = submatrix(p0, range(33, 55), range(33))
    lift1 = submatrix(p1, range(33, 55), range(33))
    pairs = _projector_pairs(companion0, companion1, recipes)
    k220 = zeros(22, 22)
    k221 = zeros(22, 22)
    inverse0 = zeros(22, 22)
    for eigenvalue, (projector0, projector1) in pairs.items():
        metric0 = h0 if eigenvalue == 1 else scale(h0, -1) if eigenvalue == -1 else identity(22)
        metric1 = h1 if eigenvalue == 1 else scale(h1, -1) if eigenvalue == -1 else zeros(22, 22)
        k220 = add(
            k220,
            multiply(multiply(transpose(projector0), metric0), projector0),
        )
        k221 = add(
            k221,
            add(
                add(
                    multiply(multiply(transpose(projector1), metric0), projector0),
                    multiply(multiply(transpose(projector0), metric1), projector0),
                ),
                multiply(multiply(transpose(projector0), metric0), projector1),
            ),
        )
        inverse0 = add(inverse0, scale(projector0, Fraction(1, 1) / eigenvalue))
    if k220 != submatrix(k0, range(33, 55), range(33, 55)):
        raise K55TaylorOrderOneConsumerError("reference K22 zero replay mismatch")
    if _nonzero_count(_subtract(multiply(companion0, inverse0), identity(22))):
        raise K55TaylorOrderOneConsumerError("reference companion inverse mismatch")
    inverse1 = scale(multiply(multiply(inverse0, companion1), inverse0), -1)
    f1 = add(
        add(
            multiply(multiply(transpose(lift1), k220), inverse0),
            multiply(multiply(transpose(lift0), k221), inverse0),
        ),
        multiply(multiply(transpose(lift0), k220), inverse1),
    )
    k1 = zeros(55, 55)
    for row in range(33):
        for column in range(22):
            k1[row][33 + column] = f1[row][column]
            k1[33 + column][row] = f1[row][column]
    for row in range(22):
        for column in range(22):
            k1[33 + row][33 + column] = k221[row][column]
    residual = _subtract(
        add(multiply(k1, p0), multiply(k0, p1)),
        add(multiply(transpose(p0), k1), multiply(transpose(p1), k0)),
    )
    if k1 != transpose(k1) or _nonzero_count(residual):
        raise K55TaylorOrderOneConsumerError("reference K55 order-one residual nonzero")
    return {
        "K55_Taylor_order_one_reference_e1": matrix_packet("K55_1_e1", k1),
        "projector_derivative_packets": 6,
        "projector_derivative_residual_nonzero_entries": 0,
        "K55_order_one_symmetry_residual_nonzero_entries": 0,
        "K55_order_one_symmetrizer_residual_nonzero_entries": 0,
    }


def build_campaign(project_root: Path, config_path: Path) -> dict[str, Any]:
    root = project_root.resolve()
    config = _load_json(config_path)
    _validate_config(config)
    upstreams = {name: _load_bound(root, binding) for name, binding in config["upstreams"].items()}
    predecessor = upstreams["manifest_predecessor"]
    h_result = upstreams["H_star_order_one"]
    if (
        predecessor.get("counts", {}).get("registered_symbolic_input_packets") != 79
        or h_result.get("status")
        != "pass_exact_15_H_star_plus_Taylor_order_one_packets_materialized"
        or len(h_result.get("packets", [])) != 15
    ):
        raise K55TaylorOrderOneConsumerError("consumer predecessor boundary changed")
    p0_packets = upstreams["flat_P55"]["matrix_packets"]
    p0_axes = [matrix_from_packet(packet) for packet in p0_packets]
    k0_receipt = upstreams["K55_order_zero"]
    k0 = matrix_from_packet(k0_receipt["exact_K0_construction"]["K0"])
    h0 = matrix_from_packet(upstreams["flat_action_metric"]["exact_construction"]["h_plus_0"])
    axis_residuals = []
    for p0 in p0_axes:
        residual = _subtract(multiply(k0, p0), multiply(transpose(p0), k0))
        axis_residuals.append(_nonzero_count(residual))
    if axis_residuals != [0, 128, 128]:
        raise K55TaylorOrderOneConsumerError("K0 spatial-axis obstruction changed")
    recipes = upstreams["projector_recipes"]["exact_Lagrange_projector_recipes"]["recipes"]
    p1_by_id = {
        packet["evaluation_id"]: packet
        for packet in predecessor["registered_P55_Taylor_order_one_packets"]
    }
    h1_by_id = {packet["evaluation_id"]: packet for packet in h_result["packets"]}
    if set(p1_by_id) != set(h1_by_id) or len(p1_by_id) != 15:
        raise K55TaylorOrderOneConsumerError("P55/H-star evaluation set mismatch")
    references = []
    for evaluation_id, p1_packet in p1_by_id.items():
        h1_packet = h1_by_id[evaluation_id]
        p1 = _evaluate_polynomial_packet(p1_packet["P55_Taylor_order_one_matrix"], "n1")
        h1 = _evaluate_polynomial_packet(h1_packet["H_star_plus_order_one_matrix"], "n1")
        exact = _construct_reference_K1(p0_axes[0], p1, k0, h0, h1, recipes)
        body = {
            "schema_version": "sigma-K55-Taylor-order-one-reference-e1-packet-1.0",
            "evaluation_id": evaluation_id,
            "evaluation_content_sha256": p1_packet["evaluation_content_sha256"],
            "P55_order_one_content_sha256": p1_packet["content_sha256"],
            "H_star_order_one_content_sha256": h1_packet["content_sha256"],
            **exact,
            "coordinate_free_admissible": False,
        }
        references.append({**body, "content_sha256": _content_hash(body)})
    manifest = json.loads(json.dumps(predecessor["required_symbolic_input_manifest"]))
    claims = {key: value for key, value in predecessor["claims"].items() if value is False}
    claims.update(
        {
            "all_15_reference_e1_K55_Taylor_order_one_packets_constructed": True,
            "all_15_coordinate_free_K55_Taylor_order_one_packets_registered": False,
            "manifest_advanced_beyond_79": False,
            "cold_full_symbol_build_used": False,
        }
    )
    body = {
        "schema_version": SCHEMA,
        "status": STATUS,
        "decision": "BLOCK_COORDINATE_FREE_ADMISSION",
        "errors": [],
        "config_sha256": config["content_sha256"],
        "upstream_bindings": {
            name: {**binding, "verified": True} for name, binding in config["upstreams"].items()
        },
        "flat_K0_spatial_axis_replay": {
            "axis_residual_nonzero_entries": dict(zip(AXIS_VARIABLES, axis_residuals, strict=True)),
            "coordinate_free_K0_admissible": False,
            "decisive_obstruction": (
                "the sealed K0 is an e1 reference: it symmetrizes P_1 but not P_2 or P_3"
            ),
        },
        "registered_reference_e1_K55_Taylor_order_one_packets": references,
        "minimal_missing_coordinate_free_contract": {
            "required_packets": 1,
            "registered_packets": 0,
            "packet": ("exact coordinate-free K0(n) or equivalent transverse/directional lift"),
            "required_fields": [
                "exact_sparse_polynomial_55x55_K0_of_n",
                "K0_of_n_symmetry",
                "K0_of_n_P0_of_n_minus_P0_of_n_transpose_K0_of_n_zero_on_sphere",
                "directional_projector_or_rotation_provenance",
                "unit_sphere_reduction_certificate",
            ],
        },
        "required_symbolic_input_manifest": manifest,
        "counts": {
            "upstream_seals_verified": 6,
            "H_star_plus_order_one_packets_validated": 15,
            "reference_e1_K55_Taylor_order_one_packets_constructed": 15,
            "coordinate_free_K55_Taylor_order_one_packets_registered": 0,
            "flat_K0_axis_e1_residual_nonzero_entries": 0,
            "flat_K0_axis_e2_residual_nonzero_entries": 128,
            "flat_K0_axis_e3_residual_nonzero_entries": 128,
            "registered_symbolic_input_packets": 79,
            "missing_symbolic_input_packets": 225,
            "full_symbol_build_calls": 0,
            "required_output_rows": 117180,
            "emitted_output_rows": 0,
        },
        "phase_two": {
            "decision": "BLOCK",
            "admitted": False,
            "attempted": False,
            "blocker": "coordinate-free K0(n)/directional lift is not serialized",
        },
        "claims": claims,
        "negative_controls": {
            "promote_e1_reference_packets_to_coordinate_free": {"rejected": True},
            "ignore_e2_e3_K0_residuals": {"rejected": True},
            "count_H_star_packets_as_K55_packets": {"rejected": True},
            "advance_manifest_without_full_n_replay": {"rejected": True},
            "emit_rows_with_225_missing_packets": {"rejected": True},
            "promote_partial_replay_to_D4_H7_or_PDE": {"rejected": True},
        },
        "source_bindings": {
            "config": {"path": CONFIG_PATH, "file_sha256": _file_sha256(config_path)},
            "source": {
                "path": SOURCE_PATH,
                "file_sha256": _file_sha256(root / SOURCE_PATH),
            },
            "test": {"path": TEST_PATH, "file_sha256": _file_sha256(root / TEST_PATH)},
        },
        "scope": (
            "Constructs and validates all 15 exact K55 Taylor-order-one packets at the "
            "sealed reference direction e1. The sealed K0 fails the e2/e3 flat-axis "
            "symmetrizer replay, so no packet is admitted as coordinate-free and the "
            "manifest remains 79/304. No recurrence row, D4, H7, PDE, or lifespan "
            "follows."
        ),
    }
    return {**body, "content_sha256": _content_hash(body)}


def validate_campaign(document: dict[str, Any], project_root: Path) -> None:
    expected = build_campaign(project_root.resolve(), project_root.resolve() / CONFIG_PATH)
    if document != expected or not _hash_matches(document):
        raise K55TaylorOrderOneConsumerError("campaign replay mismatch")


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
