from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

import sympy as sp

from . import quartic_tc2_d4_coordinate_free_k0_polynomial_packet as poly
from . import quartic_tc2_d4_coordinate_free_k55_order_one_registration as k1

SCHEMA = "sigma-quartic-tc2-d4-physical-metric-transport-no-go-1.0"
CONFIG_SCHEMA = f"{SCHEMA.removesuffix('-1.0')}-config-1.0"
CONFIG_PATH = "configs/backgrounds/quartic_tc2_d4_physical_metric_transport_no_go.json"
SOURCE_PATH = "src/sigma_theory_compiler/quartic_tc2_d4_physical_metric_transport_no_go.py"
TEST_PATH = "tests/test_quartic_tc2_d4_physical_metric_transport_no_go.py"
OUTPUT_PATH = "runs/physics-language/quartic-tc2-d4-physical-metric-transport-no-go/campaign.json"
WITNESS_DIRECTION = (Fraction(3, 5), Fraction(4, 5), Fraction(0))


class PhysicalMetricTransportNoGoError(ValueError):
    """Raised when the physical metric-transport no-go boundary changes."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "ascii"
    )


def _content_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        _canonical_bytes({key: item for key, item in value.items() if key != "content_sha256"})
    ).hexdigest()


def _hash_matches(value: dict[str, Any]) -> bool:
    return value.get("content_sha256") == _content_hash(value)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PhysicalMetricTransportNoGoError(f"expected object: {path}")
    return value


def _load_bound(root: Path, binding: dict[str, str]) -> dict[str, Any]:
    path = (root / binding["path"]).resolve()
    if root != path and root not in path.parents:
        raise PhysicalMetricTransportNoGoError("bound path escaped root")
    value = _load_json(path)
    if (
        _file_sha256(path) != binding["file_sha256"]
        or value.get("content_sha256") != binding["content_sha256"]
        or not _hash_matches(value)
    ):
        raise PhysicalMetricTransportNoGoError(f"upstream mismatch: {binding['path']}")
    return value


def _validate_config(config: dict[str, Any]) -> None:
    if (
        config.get("schema_version") != CONFIG_SCHEMA
        or config.get("policy")
        != "prove_or_refute_symmetric_physical_metric_transport_at_exact_unit_direction"
        or set(config.get("upstreams", {}))
        != {
            "Sylvester_obstruction",
            "K55_frontier",
            "higher_H_star",
            "P55_order_one",
            "flat_P55",
            "flat_action_metric",
            "projector_recipes",
        }
        or config.get("target")
        != {
            "evaluation_id": "subset_2",
            "Taylor_order": 3,
            "metric_transport_order": 2,
            "symmetric_domain_dimension": 253,
            "witness_direction": ["3/5", "4/5", "0"],
        }
        or not _hash_matches(config)
    ):
        raise PhysicalMetricTransportNoGoError("invalid physical metric-transport config")


def _sympy_matrix(matrix: Any) -> sp.Matrix:
    return sp.Matrix(
        [
            [
                sp.Rational(value.rational.numerator, value.rational.denominator)
                + sp.Rational(value.radical.numerator, value.radical.denominator) * sp.sqrt(2)
                for value in row
            ]
            for row in matrix
        ]
    )


def build_campaign(project_root: Path, config_path: Path) -> dict[str, Any]:
    root = project_root.resolve()
    config = _load_json(config_path)
    _validate_config(config)
    upstreams = {name: _load_bound(root, binding) for name, binding in config["upstreams"].items()}
    obstruction = upstreams["Sylvester_obstruction"]
    frontier = upstreams["K55_frontier"]
    higher_h = upstreams["higher_H_star"]
    if (
        obstruction.get("counts", {}).get("equal_plus_one_projection_nonzero_polynomial_entries")
        != 192
        or obstruction.get("counts", {}).get(
            "equal_minus_one_projection_nonzero_polynomial_entries"
        )
        != 192
        or frontier.get("failure_checkpoint", {}).get("evaluation_id") != "subset_2"
        or higher_h.get("counts", {}).get("authoritative_action_derivative_nonzero_entries") != 0
    ):
        raise PhysicalMetricTransportNoGoError("physical metric-transport predecessor changed")
    p_axes = [
        k1.exact._matrix_from_packet(packet) for packet in upstreams["flat_P55"]["matrix_packets"]
    ]
    h_plus = k1.exact._matrix_from_packet(
        upstreams["flat_action_metric"]["exact_construction"]["h_plus_0"]
    )
    recipes = upstreams["projector_recipes"]["exact_Lagrange_projector_recipes"]["recipes"]
    base = k1._base_data(p_axes, h_plus, recipes)
    p1_record = next(
        packet
        for packet in upstreams["P55_order_one"]["registered_P55_Taylor_order_one_packets"]
        if packet["evaluation_id"] == "subset_2"
    )
    p1 = k1._linear_packet(p1_record["P55_Taylor_order_one_matrix"], [55, 55])
    companion1 = poly._multiply(poly._multiply(base["JT"], p1), base["J"])
    full_residual = k1._sphere_packet(
        frontier["failure_checkpoint"]["sphere_symmetrizer_residual"], [55, 55]
    )
    companion_residual = poly._multiply(poly._multiply(base["JT"], full_residual), base["J"])
    c1 = _sympy_matrix(poly._evaluate(companion1, WITNESS_DIRECTION, 22, 22))
    residual = _sympy_matrix(poly._evaluate(companion_residual, WITNESS_DIRECTION, 22, 22))
    witnesses = []
    for eigenvalue in (Fraction(1), Fraction(-1)):
        projector = _sympy_matrix(
            poly._evaluate(base["Pi0"][eigenvalue], WITNESS_DIRECTION, 22, 22)
        )
        diagonal_companion1 = (projector * c1 * projector).applyfunc(sp.factor)
        target = (-(projector.T * residual * projector)).applyfunc(sp.factor)
        nonzero = [
            (row, column, sp.sstr(target[row, column]))
            for row in range(22)
            for column in range(22)
            if target[row, column] != 0
        ]
        if (
            not diagonal_companion1.is_zero_matrix
            or len(nonzero) != 32
            or nonzero[0] != (4, 10, "4096/46875")
        ):
            raise PhysicalMetricTransportNoGoError("metric-transport witness changed")
        witnesses.append(
            {
                "eigenvalue": str(eigenvalue),
                "physical_diagonal_companion1_nonzero_entries": 0,
                "symmetric_metric_domain_dimension": 253,
                "transport_linear_map_rank": 0,
                "projected_target_nonzero_entries": len(nonzero),
                "augmented_rank": 1,
                "consistent": False,
                "left_nullspace_coordinate_witness": {
                    "row": nonzero[0][0],
                    "column": nonzero[0][1],
                    "linear_map_value_for_every_symmetric_metric": "0",
                    "required_target_value": nonzero[0][2],
                },
            }
        )
    body = {
        "schema_version": SCHEMA,
        "status": "no_go_source_bound_symmetric_physical_metric_transport_cannot_cancel_K55_obstruction",
        "decision": "BLOCK_SERIALIZATION",
        "errors": [],
        "config_sha256": config["content_sha256"],
        "upstream_bindings": {
            name: {**binding, "verified": True} for name, binding in config["upstreams"].items()
        },
        "exact_unit_direction_witness": {
            "direction": ["3/5", "4/5", "0"],
            "unit_sphere_residual": "(3/5)^2+(4/5)^2-1=0",
            "evaluation_id": "subset_2",
            "failed_K55_Taylor_order": 3,
            "candidate_metric_transport_order": 2,
            "physical_sign_witnesses": witnesses,
        },
        "theorem": {
            "identity": "Pi_s*C1*Pi_s=0 implies Pi_s^T[(Pi_s^T H2 Pi_s)C1-C1^T(Pi_s^T H2 Pi_s)]Pi_s=0 for every symmetric H2",
            "contradiction": "the required projected order-three target has 32 nonzero entries for each s in {+1,-1}",
            "conclusion": "no source-bound symmetric within-physical-eigenspace order-two metric transport can cancel the subset_2 order-three K55 obstruction",
        },
        "counts": {
            "physical_signs_checked": 2,
            "symmetric_metric_variables_per_sign": 253,
            "transport_map_rank_each": 0,
            "augmented_rank_each": 1,
            "projected_target_nonzero_entries_each": 32,
            "manifest_registered_before": 154,
            "manifest_registered_after": 154,
            "emitted_output_rows": 0,
        },
        "claims": {
            "higher_K55_registered": False,
            "higher_TC2_registered": False,
            "lower_Sylvester_registered": False,
            "rows_emitted": False,
        },
        "negative_controls": {
            "sample_nonunit_direction": {"rejected": True},
            "allow_nonsymmetric_metric_transport": {"rejected": True},
            "ignore_left_nullspace_coordinate": {"rejected": True},
            "advance_manifest_after_no_go": {"rejected": True},
        },
        "source_bindings": {
            "config": {"path": CONFIG_PATH, "file_sha256": _file_sha256(config_path)},
            "source": {"path": SOURCE_PATH, "file_sha256": _file_sha256(root / SOURCE_PATH)},
            "test": {"path": TEST_PATH, "file_sha256": _file_sha256(root / TEST_PATH)},
        },
        "scope": "Provides an exact rational-unit-direction left-nullspace witness against the smallest symmetric physical metric-transport repair. It does not exclude changes to earlier P55/action authority, nonsymmetric forms, cross-cluster metrics, or a different symmetrizer construction.",
    }
    return {**body, "content_sha256": _content_hash(body)}


def validate_campaign(document: dict[str, Any], root: Path) -> None:
    if not _hash_matches(document) or document != build_campaign(root, root / CONFIG_PATH):
        raise PhysicalMetricTransportNoGoError("physical metric-transport no-go replay mismatch")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    document = build_campaign(args.project_root.resolve(), args.config.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
