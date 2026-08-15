from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

try:
    from .quartic_tc2_d4_k55_taylor_order_zero_registration import (
        ONE,
        ZERO,
        Matrix,
        Surd,
        add,
        identity,
        matrix_from_packet,
        matrix_packet,
        multiply,
        scale,
        transpose,
        zeros,
    )
except ImportError:  # pragma: no cover
    from quartic_tc2_d4_k55_taylor_order_zero_registration import (
        ONE,
        ZERO,
        Matrix,
        Surd,
        add,
        identity,
        matrix_from_packet,
        matrix_packet,
        multiply,
        scale,
        transpose,
        zeros,
    )

SCHEMA = "sigma-quartic-tc2-d4-coordinate-free-k0-directional-lift-1.0"
CONFIG_SCHEMA = f"{SCHEMA.removesuffix('-1.0')}-config-1.0"
STATUS = "pass_exact_coordinate_free_K0_directional_lift_formula"
CONFIG_PATH = "configs/backgrounds/quartic_tc2_d4_coordinate_free_k0_directional_lift.json"
SOURCE_PATH = "src/sigma_theory_compiler/quartic_tc2_d4_coordinate_free_k0_directional_lift.py"
TEST_PATH = "tests/test_quartic_tc2_d4_coordinate_free_k0_directional_lift.py"
OUTPUT_PATH = (
    "runs/physics-language/quartic-tc2-d4-coordinate-free-k0-directional-lift/campaign.json"
)
DIRECTIONS = [
    (Fraction(1), Fraction(0), Fraction(0)),
    (Fraction(0), Fraction(1), Fraction(0)),
    (Fraction(0), Fraction(0), Fraction(1)),
    (Fraction(3, 5), Fraction(4, 5), Fraction(0)),
    (Fraction(1, 3), Fraction(2, 3), Fraction(2, 3)),
    (Fraction(0), Fraction(3, 5), Fraction(4, 5)),
]


class CoordinateFreeK0DirectionalLiftError(ValueError):
    """Raised when the exact directional-lift construction fails closed."""


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
        raise CoordinateFreeK0DirectionalLiftError("bound path escaped project root")
    return path


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CoordinateFreeK0DirectionalLiftError(f"expected JSON object: {path}")
    return value


def _load_bound(root: Path, binding: dict[str, str]) -> dict[str, Any]:
    path = _resolve_under(root, binding["path"])
    value = _load_json(path)
    if (
        _file_sha256(path) != binding["file_sha256"]
        or value.get("content_sha256") != binding["content_sha256"]
        or not _hash_matches(value)
    ):
        raise CoordinateFreeK0DirectionalLiftError(f"upstream mismatch: {binding['path']}")
    return value


def _validate_config(config: dict[str, Any]) -> None:
    if (
        config.get("schema_version") != CONFIG_SCHEMA
        or config.get("policy")
        != "construct_basis_free_K0_of_n_from_P55_projectors_and_action_metric"
        or not _hash_matches(config)
        or set(config.get("upstreams", {}))
        != {"flat_P55", "reference_K0", "flat_action_metric", "projector_recipes"}
        or config.get("target")
        != {
            "state_dimension": 55,
            "companion_dimension": 22,
            "transverse_zero_dimension": 33,
            "projector_degree": 6,
            "exact_direction_controls": 6,
            "reference_direction": ["1", "0", "0"],
        }
        or config.get("caps", {}).get("maximum_full_symbol_build_calls") != 0
        or config.get("caps", {}).get("maximum_output_rows_emitted") != 0
    ):
        raise CoordinateFreeK0DirectionalLiftError("invalid directional-lift config")


def _subtract(left: Matrix, right: Matrix) -> Matrix:
    return add(left, scale(right, -1))


def _nonzero_count(matrix: Matrix) -> int:
    return sum(value != ZERO for row in matrix for value in row)


def _normal_embedding(direction: tuple[Fraction, Fraction, Fraction]) -> Matrix:
    embedding = zeros(55, 22)
    for field in range(11):
        embedding[33 + field][field] = ONE
        for coefficient, offset in zip(direction, (44, 11, 22), strict=True):
            embedding[offset + field][11 + field] = Surd(coefficient)
    return embedding


def _projectors(companion: Matrix, recipes: list[dict[str, Any]]) -> dict[Fraction, Matrix]:
    powers = [identity(22)]
    for _ in range(6):
        powers.append(multiply(powers[-1], companion))
    result = {}
    total = zeros(22, 22)
    for recipe in recipes:
        eigenvalue = Fraction(recipe["eigenvalue"])
        if eigenvalue == 0:
            continue
        projector = zeros(22, 22)
        for coefficient, power in zip(
            map(Fraction, recipe["coefficients_low_to_high"]), powers, strict=True
        ):
            projector = add(projector, scale(power, coefficient))
        if multiply(projector, projector) != projector:
            raise CoordinateFreeK0DirectionalLiftError("projector idempotence failed")
        result[eigenvalue] = projector
        total = add(total, projector)
    if set(result) != {
        Fraction(1),
        Fraction(-1),
        Fraction(1, 2),
        Fraction(-1, 2),
        Fraction(1, 3),
        Fraction(-1, 3),
    } or total != identity(22):
        raise CoordinateFreeK0DirectionalLiftError("projector completeness failed")
    return result


def construct_at_direction(
    direction: tuple[Fraction, Fraction, Fraction],
    p_axes: list[Matrix],
    h_plus: Matrix,
    recipes: list[dict[str, Any]],
) -> dict[str, Any]:
    if sum(value * value for value in direction) != 1:
        raise CoordinateFreeK0DirectionalLiftError("direction is not on the unit sphere")
    physical = zeros(55, 55)
    for coefficient, axis in zip(direction, p_axes, strict=True):
        physical = add(physical, scale(axis, coefficient))
    embedding = _normal_embedding(direction)
    normal_projector = multiply(embedding, transpose(embedding))
    transverse = _subtract(identity(55), normal_projector)
    if multiply(transpose(embedding), embedding) != identity(22):
        raise CoordinateFreeK0DirectionalLiftError("normal embedding isometry failed")
    if _nonzero_count(multiply(transverse, embedding)):
        raise CoordinateFreeK0DirectionalLiftError("transverse-normal orthogonality failed")
    companion = multiply(multiply(transpose(embedding), physical), embedding)
    lift = multiply(multiply(transpose(embedding), physical), transverse)
    reconstruction = add(
        multiply(embedding, lift),
        multiply(multiply(embedding, companion), transpose(embedding)),
    )
    if reconstruction != physical:
        raise CoordinateFreeK0DirectionalLiftError("directional block reconstruction failed")
    projectors = _projectors(companion, recipes)
    companion_energy = zeros(22, 22)
    inverse = zeros(22, 22)
    for eigenvalue, projector in projectors.items():
        metric = (
            h_plus if eigenvalue == 1 else scale(h_plus, -1) if eigenvalue == -1 else identity(22)
        )
        companion_energy = add(
            companion_energy,
            multiply(multiply(transpose(projector), metric), projector),
        )
        inverse = add(inverse, scale(projector, Fraction(1, 1) / eigenvalue))
    if multiply(companion, inverse) != identity(22):
        raise CoordinateFreeK0DirectionalLiftError("companion inverse replay failed")
    cross = multiply(multiply(transpose(lift), companion_energy), inverse)
    symmetrizer = add(
        add(
            add(transverse, multiply(cross, transpose(embedding))),
            multiply(embedding, transpose(cross)),
        ),
        multiply(multiply(embedding, companion_energy), transpose(embedding)),
    )
    residual = _subtract(
        multiply(symmetrizer, physical),
        multiply(transpose(physical), symmetrizer),
    )
    if symmetrizer != transpose(symmetrizer) or _nonzero_count(residual):
        raise CoordinateFreeK0DirectionalLiftError("K0 directional replay failed")
    return {
        "direction": [str(value) for value in direction],
        "K0": matrix_packet("K0_direction_control", symmetrizer),
        "P55_symmetrizer_residual_nonzero_entries": 0,
        "normal_embedding_isometry_residual_nonzero_entries": 0,
        "directional_block_reconstruction_residual_nonzero_entries": 0,
    }


def build_campaign(project_root: Path, config_path: Path) -> dict[str, Any]:
    root = project_root.resolve()
    config = _load_json(config_path)
    _validate_config(config)
    upstreams = {name: _load_bound(root, binding) for name, binding in config["upstreams"].items()}
    p_axes = [matrix_from_packet(packet) for packet in upstreams["flat_P55"]["matrix_packets"]]
    h_plus = matrix_from_packet(upstreams["flat_action_metric"]["exact_construction"]["h_plus_0"])
    recipes = upstreams["projector_recipes"]["exact_Lagrange_projector_recipes"]["recipes"]
    controls = [construct_at_direction(value, p_axes, h_plus, recipes) for value in DIRECTIONS]
    reference = matrix_from_packet(upstreams["reference_K0"]["exact_K0_construction"]["K0"])
    if matrix_from_packet(controls[0]["K0"]) != reference:
        raise CoordinateFreeK0DirectionalLiftError("e1 K0 agreement failed")
    formula_body = {
        "schema_version": "sigma-coordinate-free-K0-directional-lift-formula-1.0",
        "variables": ["n1", "n2", "n3"],
        "domain": "n1^2+n2^2+n3^2=1",
        "normal_embedding": (
            "J(n):(v,w_n)->U; v occupies rows 33:44 and w_n="
            "n1*w1+n2*w2+n3*w3 occupies gradient rows"
        ),
        "identities": {
            "T": "I55-J*J^T",
            "C": "J^T*P55(n)*J",
            "L": "J^T*P55(n)*T",
            "Pi_lambda": "degree-6 Lagrange polynomial in C",
            "G": "sum Pi_lambda^T*H_lambda*Pi_lambda",
            "C_inverse": "sum lambda^-1*Pi_lambda",
            "F": "L^T*G*C_inverse",
            "K0": "T+F*J^T+J*F^T+J*G*J^T",
        },
        "block_theorem": {
            "P_decomposition": "P=J*L+J*C*J^T",
            "F_C_equals_L_transpose_G": True,
            "G_C_minus_C_transpose_G_zero": True,
            "conclusion": "K0(n)*P55(n)-P55(n)^T*K0(n)=0 on the unit sphere",
        },
        "reference_K0_content_sha256": upstreams["reference_K0"]["exact_K0_construction"]["K0"][
            "content_sha256"
        ],
    }
    formula = {**formula_body, "content_sha256": _content_hash(formula_body)}
    claims = {
        "coordinate_free_K0_directional_lift_formula_constructed": True,
        "e1_reference_K0_reproduced_exactly": True,
        "expanded_55x55_polynomial_K0_packet_emitted": False,
        "K55_Taylor_order_one_registered": False,
        "complete_coordinate_free_coefficient_map_emitted": False,
        "full_direction_sphere_D4_compatibility_proved": False,
        "global_H7_closed": False,
        "nonlinear_PDE_closure_proved": False,
        "lifespan_proved": False,
    }
    body = {
        "schema_version": SCHEMA,
        "status": STATUS,
        "errors": [],
        "config_sha256": config["content_sha256"],
        "upstream_bindings": {
            name: {**binding, "verified": True} for name, binding in config["upstreams"].items()
        },
        "coordinate_free_K0_formula": formula,
        "exact_direction_controls": controls,
        "polynomial_serialization_boundary": {
            "expanded_packet_emitted": False,
            "reason": (
                "the basis-free formula is exact; expansion and sphere-normal-form "
                "serialization remain a separate bounded emission step"
            ),
            "missing_cross_or_normalization_data": [],
        },
        "counts": {
            "upstream_seals_verified": 4,
            "degree_six_projector_recipes_used": 6,
            "exact_direction_controls": 6,
            "exact_direction_controls_passed": 6,
            "e1_reference_matrix_entries_compared": 3025,
            "e1_reference_matrix_mismatches": 0,
            "direction_control_symmetrizer_residual_nonzero_entries": 0,
            "full_symbol_build_calls": 0,
            "emitted_output_rows": 0,
        },
        "claims": claims,
        "negative_controls": {
            "reuse_fixed_e1_K0_for_all_directions": {"rejected": True},
            "choose_a_transverse_frame": {"rejected": True},
            "omit_cross_F": {"rejected": True},
            "count_formula_as_expanded_polynomial_packet": {"rejected": True},
            "promote_K0_formula_to_D4_H7_or_PDE": {"rejected": True},
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
            "Constructs an exact basis-free coordinate-free K0(n) formula from the "
            "full P55 pencil, degree-six projectors, and h_plus_0, reproducing the "
            "K0 and passing six exact sphere-direction controls. Expanded polynomial "
            "serialization, K55 order one, D4, H7, PDE, and lifespan remain open."
        ),
    }
    return {**body, "content_sha256": _content_hash(body)}


def validate_campaign(document: dict[str, Any], project_root: Path) -> None:
    expected = build_campaign(project_root.resolve(), project_root.resolve() / CONFIG_PATH)
    if document != expected or not _hash_matches(document):
        raise CoordinateFreeK0DirectionalLiftError("campaign replay mismatch")


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
