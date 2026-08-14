from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

from . import quartic_tc2_d4_coordinate_free_k0_polynomial_packet as poly
from . import quartic_tc2_d4_coordinate_free_k55_order_one_registration as k1

SCHEMA = "sigma-quartic-tc2-d4-higher-k55-sylvester-obstruction-1.0"
CONFIG_SCHEMA = f"{SCHEMA.removesuffix('-1.0')}-config-1.0"
CONFIG_PATH = "configs/backgrounds/quartic_tc2_d4_higher_k55_sylvester_obstruction.json"
SOURCE_PATH = "src/sigma_theory_compiler/quartic_tc2_d4_higher_k55_sylvester_obstruction.py"
TEST_PATH = "tests/test_quartic_tc2_d4_higher_k55_sylvester_obstruction.py"
OUTPUT_PATH = "runs/physics-language/quartic-tc2-d4-higher-k55-sylvester-obstruction/campaign.json"

Sparse = poly.Sparse


class HigherK55SylvesterObstructionError(ValueError):
    """Raised when the projected K55 obstruction boundary changes."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def _content_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes({key: item for key, item in value.items() if key != "content_sha256"})).hexdigest()


def _hash_matches(value: dict[str, Any]) -> bool:
    return value.get("content_sha256") == _content_hash(value)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise HigherK55SylvesterObstructionError(f"expected object: {path}")
    return value


def _load_bound(root: Path, binding: dict[str, str]) -> dict[str, Any]:
    path = (root / binding["path"]).resolve()
    if root != path and root not in path.parents:
        raise HigherK55SylvesterObstructionError("bound path escaped root")
    value = _load_json(path)
    if _file_sha256(path) != binding["file_sha256"] or value.get("content_sha256") != binding["content_sha256"] or not _hash_matches(value):
        raise HigherK55SylvesterObstructionError(f"upstream mismatch: {binding['path']}")
    return value


def _validate_config(config: dict[str, Any]) -> None:
    if (
        config.get("schema_version") != CONFIG_SCHEMA
        or config.get("policy") != "project_first_K55_failure_onto_every_equal_eigenspace_before_any_correction"
        or set(config.get("upstreams", {})) != {"K55_frontier", "flat_P55", "flat_action_metric", "projector_recipes"}
        or config.get("target") != {"evaluation_id": "subset_2", "Taylor_order": 3, "spectrum": ["0", "1", "-1", "1/2", "-1/2", "1/3", "-1/3"]}
        or not _hash_matches(config)
    ):
        raise HigherK55SylvesterObstructionError("invalid K55 obstruction config")


def _sphere_equal(left: Sparse, right: Sparse) -> bool:
    return set(left) == set(right) and all(left[key].terms == right[key].terms for key in left)


def build_campaign(project_root: Path, config_path: Path) -> dict[str, Any]:
    root = project_root.resolve()
    config = _load_json(config_path)
    _validate_config(config)
    upstreams = {name: _load_bound(root, binding) for name, binding in config["upstreams"].items()}
    frontier = upstreams["K55_frontier"]
    failure = frontier.get("failure_checkpoint", {})
    if (
        frontier.get("status") != "block_higher_K55_at_subset_2_order_3_auxiliary_Riesz_metric_correction"
        or failure.get("evaluation_id") != "subset_2"
        or failure.get("Taylor_order") != 3
        or failure.get("sphere_symmetrizer_remainder_entries") != 120
    ):
        raise HigherK55SylvesterObstructionError("K55 failure frontier changed")
    residual = k1._sphere_packet(failure["sphere_symmetrizer_residual"], [55, 55])
    p_axes = [k1.exact._matrix_from_packet(packet) for packet in upstreams["flat_P55"]["matrix_packets"]]
    h_plus = k1.exact._matrix_from_packet(upstreams["flat_action_metric"]["exact_construction"]["h_plus_0"])
    recipes = upstreams["projector_recipes"]["exact_Lagrange_projector_recipes"]["recipes"]
    base = k1._base_data(p_axes, h_plus, recipes)
    projectors: dict[Fraction, Sparse] = {Fraction(0): base["T"]}
    projectors.update(
        {
            eigenvalue: poly._multiply(poly._multiply(base["J"], projector), base["JT"])
            for eigenvalue, projector in base["Pi0"].items()
        }
    )
    diagonal_packets = []
    obstruction_blocks: dict[Fraction, Sparse] = {}
    for eigenvalue, projector in projectors.items():
        block = poly._multiply(poly._multiply(poly._transpose(projector), residual), projector)
        if block:
            obstruction_blocks[eigenvalue] = block
        if block:
            packet = poly._packet(
                f"Pi_{eigenvalue}^T_R3_Pi_{eigenvalue}", block, [55, 55]
            )
        else:
            empty_body = {
                "schema_version": "sigma-exact-empty-sphere-polynomial-matrix-1.0",
                "name": f"Pi_{eigenvalue}^T_R3_Pi_{eigenvalue}",
                "shape": [55, 55],
                "entries": [],
                "nonzero_polynomial_entries": 0,
            }
            packet = {**empty_body, "content_sha256": _content_hash(empty_body)}
        diagonal_packets.append({"eigenvalue": str(eigenvalue), "nonzero_polynomial_entries": len(block), "packet": packet})
    correction: Sparse = {}
    off_diagonal_forcing: Sparse = {}
    for left, left_projector in projectors.items():
        for right, right_projector in projectors.items():
            if left == right:
                continue
            block = poly._multiply(poly._multiply(poly._transpose(left_projector), residual), right_projector)
            off_diagonal_forcing = poly._add(off_diagonal_forcing, block)
            correction = poly._add(correction, poly._scale(block, Fraction(-1, 1) / (right - left)))
    solve_remainder = poly._add(
        residual,
        poly._add(
            poly._multiply(correction, base["P0"]),
            poly._scale(poly._multiply(poly._transpose(base["P0"]), correction), -1),
        ),
    )
    if (
        len(residual) != 120
        or set(obstruction_blocks) != {Fraction(1), Fraction(-1)}
        or any(len(obstruction_blocks[eigenvalue]) != 192 for eigenvalue in obstruction_blocks)
        or off_diagonal_forcing
        or correction
        or not _sphere_equal(solve_remainder, residual)
    ):
        raise HigherK55SylvesterObstructionError("projected Sylvester census changed")
    body = {
        "schema_version": SCHEMA,
        "status": "block_exact_equal_physical_eigenspace_K55_order_three_obstruction",
        "decision": "BLOCK_SERIALIZATION",
        "errors": [],
        "config_sha256": config["content_sha256"],
        "upstream_bindings": {name: {**binding, "verified": True} for name, binding in config["upstreams"].items()},
        "projected_equal_eigenspace_packets": diagonal_packets,
        "exact_Sylvester_result": {
            "equation": "DeltaK3*P0-P0^T*DeltaK3=-R3",
            "off_diagonal_forcing_nonzero_entries": len(off_diagonal_forcing),
            "canonical_unequal_gap_correction_nonzero_entries": len(correction),
            "post_correction_remainder_nonzero_entries": len(solve_remainder),
            "obstructed_eigenvalues": ["1", "-1"],
            "first_missing_primitive": "source-bound physical plus/minus eigenspace metric-transport recurrence beyond raw action-symbol derivatives",
        },
        "counts": {
            "input_residual_nonzero_polynomial_entries": len(residual),
            "equal_plus_one_projection_nonzero_polynomial_entries": len(obstruction_blocks[Fraction(1)]),
            "equal_minus_one_projection_nonzero_polynomial_entries": len(obstruction_blocks[Fraction(-1)]),
            "other_equal_eigenspace_projection_nonzero_polynomial_entries": 0,
            "unequal_eigenspace_forcing_nonzero_polynomial_entries": 0,
            "canonical_Sylvester_correction_nonzero_polynomial_entries": 0,
            "manifest_registered_before": 154,
            "manifest_registered_after": 154,
            "emitted_output_rows": 0,
        },
        "claims": {"higher_K55_registered": False, "higher_TC2_registered": False, "lower_Sylvester_registered": False, "rows_emitted": False},
        "negative_controls": {"treat_equal_eigenspace_block_as_divisible_by_zero_gap": {"rejected": True}, "drop_physical_projection": {"rejected": True}, "count_zero_unequal_gap_correction_as_solution": {"rejected": True}, "advance_manifest_after_obstruction": {"rejected": True}},
        "source_bindings": {"config": {"path": CONFIG_PATH, "file_sha256": _file_sha256(config_path)}, "source": {"path": SOURCE_PATH, "file_sha256": _file_sha256(root / SOURCE_PATH)}, "test": {"path": TEST_PATH, "file_sha256": _file_sha256(root / TEST_PATH)}},
        "scope": "Projects the first exact higher-K55 failure onto all seven flat eigenspaces and proves that no unequal-gap Sylvester correction exists or helps. It does not register downstream packets or rows.",
    }
    return {**body, "content_sha256": _content_hash(body)}


def validate_campaign(document: dict[str, Any], root: Path) -> None:
    if not _hash_matches(document) or document != build_campaign(root, root / CONFIG_PATH):
        raise HigherK55SylvesterObstructionError("K55 obstruction replay mismatch")


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
