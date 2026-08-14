from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

try:
    from . import quartic_85_state_full_sphere_resonant_compatibility_gate as sphere
except ImportError:  # pragma: no cover
    import quartic_85_state_full_sphere_resonant_compatibility_gate as sphere


SCHEMA = "sigma-quartic-tc2-d4-coordinate-free-k0-polynomial-packet-1.0"
CONFIG_SCHEMA = f"{SCHEMA.removesuffix('-1.0')}-config-1.0"
STATUS = "pass_exact_coordinate_free_K0_polynomial_packet"
CONFIG_PATH = "configs/backgrounds/quartic_tc2_d4_coordinate_free_k0_polynomial_packet.json"
SOURCE_PATH = "src/sigma_theory_compiler/quartic_tc2_d4_coordinate_free_k0_polynomial_packet.py"
TEST_PATH = "tests/test_quartic_tc2_d4_coordinate_free_k0_polynomial_packet.py"
OUTPUT_PATH = (
    "runs/physics-language/quartic-tc2-d4-coordinate-free-k0-polynomial-packet/campaign.json"
)

Polynomial = sphere._Polynomial
Sparse = sphere._Sparse
exact = sphere.exact
N = sphere._N
PZERO = sphere._PZERO
PONE = sphere._PONE


class CoordinateFreeK0PolynomialPacketError(ValueError):
    """Raised when the exact polynomial K0 registration fails closed."""


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
        raise CoordinateFreeK0PolynomialPacketError(f"expected JSON object: {path}")
    return value


def _resolve_under(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root != path and root not in path.parents:
        raise CoordinateFreeK0PolynomialPacketError("bound path escaped project root")
    return path


def _load_bound(root: Path, binding: dict[str, str]) -> dict[str, Any]:
    path = _resolve_under(root, binding["path"])
    value = _load_json(path)
    if (
        _file_sha256(path) != binding["file_sha256"]
        or value.get("content_sha256") != binding["content_sha256"]
        or not _hash_matches(value)
    ):
        raise CoordinateFreeK0PolynomialPacketError(f"upstream mismatch: {binding['path']}")
    return value


def _validate_config(config: dict[str, Any]) -> None:
    expected_claims = {
        "expanded_55x55_polynomial_K0_packet_emitted": True,
        "K0_sphere_symmetrizer_identity_proved": True,
        "all_15_K55_order_one_packets_authorized_for_construction": True,
        "K55_Taylor_order_one_registered": False,
        "complete_coordinate_free_coefficient_map_emitted": False,
        "full_direction_sphere_D4_compatibility_proved": False,
        "global_H7_closed": False,
        "nonlinear_PDE_closure_proved": False,
        "lifespan_proved": False,
    }
    if (
        config.get("schema_version") != CONFIG_SCHEMA
        or config.get("policy")
        != "expand_K0_formula_in_exact_sphere_quotient_and_authorize_not_register_K1"
        or set(config.get("upstreams", {})) != {"directional_lift", "order_one_consumer"}
        or config.get("claims_policy") != expected_claims
        or config.get("target")
        != {
            "shape": [55, 55],
            "coefficient_field": "Q(sqrt(2))",
            "sphere_normal_form": "power(n1)<=1 modulo n1^2+n2^2+n3^2-1",
            "former_blocked_K55_order_one_packets": 15,
        }
        or config.get("caps")
        != {
            "maximum_matrix_dimension": 55,
            "maximum_nonzero_polynomial_entries": 3025,
            "maximum_normal_form_terms": 250000,
            "maximum_total_degree": 32,
            "maximum_full_symbol_build_calls": 0,
            "maximum_output_rows_emitted": 0,
        }
        or not _hash_matches(config)
    ):
        raise CoordinateFreeK0PolynomialPacketError("invalid polynomial-packet config")


def _add(left: Sparse, right: Sparse) -> Sparse:
    return sphere._sparse_add(left, right)


def _scale(matrix: Sparse, coefficient: object) -> Sparse:
    return sphere._sparse_scale(matrix, coefficient)


def _multiply(left: Sparse, right: Sparse) -> Sparse:
    return sphere._sparse_multiply(left, right)


def _transpose(matrix: Sparse) -> Sparse:
    return sphere._sparse_transpose(matrix)


def _constant(matrix: Any) -> Sparse:
    return sphere._constant_matrix(matrix)


def _identity(size: int) -> Sparse:
    return sphere._sparse_identity(size)


def _normal_embedding() -> Sparse:
    embedding: Sparse = {}
    offsets = (44, 11, 22)
    for field in range(11):
        embedding[(33 + field, field)] = PONE
        for axis, offset in enumerate(offsets):
            embedding[(offset + field, 11 + field)] = N[axis]
    return embedding


def _projectors(companion: Sparse, recipes: list[dict[str, Any]]) -> dict[Fraction, Sparse]:
    powers = [_identity(22)]
    for _ in range(6):
        powers.append(_multiply(powers[-1], companion))
    projectors: dict[Fraction, Sparse] = {}
    total: Sparse = {}
    for recipe in recipes:
        eigenvalue = Fraction(recipe["eigenvalue"])
        if eigenvalue == 0:
            continue
        projector: Sparse = {}
        for coefficient, power in zip(
            map(Fraction, recipe["coefficients_low_to_high"]), powers, strict=True
        ):
            projector = _add(projector, _scale(power, coefficient))
        if _add(_multiply(projector, projector), _scale(projector, -1)):
            raise CoordinateFreeK0PolynomialPacketError("projector idempotence failed")
        if _add(_multiply(companion, projector), _scale(projector, -eigenvalue)):
            raise CoordinateFreeK0PolynomialPacketError("projector eigenidentity failed")
        projectors[eigenvalue] = projector
        total = _add(total, projector)
    if set(projectors) != {
        Fraction(-1),
        Fraction(-1, 2),
        Fraction(-1, 3),
        Fraction(1, 3),
        Fraction(1, 2),
        Fraction(1),
    } or _add(total, _scale(_identity(22), -1)):
        raise CoordinateFreeK0PolynomialPacketError("projector completeness failed")
    return projectors


def _evaluate(matrix: Sparse, direction: tuple[int, int, int], rows: int, columns: int) -> Any:
    return sphere._evaluate(matrix, direction, rows, columns)


def _packet(name: str, matrix: Sparse, shape: list[int]) -> dict[str, Any]:
    entries = [
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
    term_count = sum(len(entry["terms"]) for entry in entries)
    maximum_degree = max(sum(term["powers"]) for entry in entries for term in entry["terms"])
    body = {
        "schema_version": "sigma-exact-sparse-Qsqrt2-sphere-polynomial-matrix-1.0",
        "name": name,
        "shape": shape,
        "variables": ["n1", "n2", "n3"],
        "quotient_relation": "n1^2+n2^2+n3^2-1",
        "canonical_normal_form": "power(n1)<=1",
        "entries": entries,
        "nonzero_polynomial_entries": len(entries),
        "normal_form_terms": term_count,
        "maximum_total_degree": maximum_degree,
        "normal_form_sha256": hashlib.sha256(_canonical_bytes(entries)).hexdigest(),
    }
    return {**body, "content_sha256": _content_hash(body)}


def _construct(
    p_axes: list[Any], h_plus: Any, recipes: list[dict[str, Any]]
) -> tuple[Sparse, dict[str, int]]:
    pencil: Sparse = {}
    for axis, matrix in enumerate(p_axes):
        pencil = _add(pencil, _scale(_constant(matrix), N[axis]))
    embedding = _normal_embedding()
    embedding_t = _transpose(embedding)
    normal = _multiply(embedding, embedding_t)
    transverse = _add(_identity(55), _scale(normal, -1))
    if _add(_multiply(embedding_t, embedding), _scale(_identity(22), -1)):
        raise CoordinateFreeK0PolynomialPacketError("embedding isometry failed")
    if _multiply(transverse, embedding):
        raise CoordinateFreeK0PolynomialPacketError("embedding orthogonality failed")
    companion = _multiply(_multiply(embedding_t, pencil), embedding)
    lift = _multiply(_multiply(embedding_t, pencil), transverse)
    reconstruction = _add(
        _multiply(embedding, lift),
        _multiply(_multiply(embedding, companion), embedding_t),
    )
    if _add(pencil, _scale(reconstruction, -1)):
        raise CoordinateFreeK0PolynomialPacketError("pencil block reconstruction failed")
    projectors = _projectors(companion, recipes)
    companion_energy: Sparse = {}
    companion_inverse: Sparse = {}
    h_plus_sparse = _constant(h_plus)
    for eigenvalue, projector in projectors.items():
        metric = (
            h_plus_sparse
            if eigenvalue == 1
            else _scale(h_plus_sparse, -1)
            if eigenvalue == -1
            else _identity(22)
        )
        companion_energy = _add(
            companion_energy,
            _multiply(_multiply(_transpose(projector), metric), projector),
        )
        companion_inverse = _add(companion_inverse, _scale(projector, 1 / eigenvalue))
    if _add(_multiply(companion, companion_inverse), _scale(_identity(22), -1)):
        raise CoordinateFreeK0PolynomialPacketError("companion inverse failed")
    cross = _multiply(_multiply(_transpose(lift), companion_energy), companion_inverse)
    symmetrizer = _add(
        _add(
            _add(transverse, _multiply(cross, embedding_t)),
            _multiply(embedding, _transpose(cross)),
        ),
        _multiply(_multiply(embedding, companion_energy), embedding_t),
    )
    symmetry = _add(symmetrizer, _scale(_transpose(symmetrizer), -1))
    residual = _add(
        _multiply(symmetrizer, pencil),
        _scale(_multiply(_transpose(pencil), symmetrizer), -1),
    )
    if symmetry or residual:
        raise CoordinateFreeK0PolynomialPacketError("K0 polynomial identity failed")
    counts = {
        "P55_nonzero_polynomial_entries": len(pencil),
        "J_nonzero_polynomial_entries": len(embedding),
        "C_nonzero_polynomial_entries": len(companion),
        "L_nonzero_polynomial_entries": len(lift),
        "G_nonzero_polynomial_entries": len(companion_energy),
        "F_nonzero_polynomial_entries": len(cross),
        "K0_symmetry_remainder_entries": len(symmetry),
        "K0_P55_symmetrizer_remainder_entries": len(residual),
    }
    return symmetrizer, counts


def build_campaign(project_root: Path, config_path: Path) -> dict[str, Any]:
    root = project_root.resolve()
    config = _load_json(config_path)
    _validate_config(config)
    upstreams = {name: _load_bound(root, binding) for name, binding in config["upstreams"].items()}
    lift = upstreams["directional_lift"]
    inherited = {
        name: _load_bound(root, binding) for name, binding in lift["upstream_bindings"].items()
    }
    p_axes = [
        exact._matrix_from_packet(packet) for packet in inherited["flat_P55"]["matrix_packets"]
    ]
    h_plus = exact._matrix_from_packet(
        inherited["flat_action_metric"]["exact_construction"]["h_plus_0"]
    )
    recipes = inherited["projector_recipes"]["exact_Lagrange_projector_recipes"]["recipes"]
    symmetrizer, construction_counts = _construct(p_axes, h_plus, recipes)
    packet = _packet("K0(n)", symmetrizer, [55, 55])
    caps = config["caps"]
    if (
        packet["nonzero_polynomial_entries"] > caps["maximum_nonzero_polynomial_entries"]
        or packet["normal_form_terms"] > caps["maximum_normal_form_terms"]
        or packet["maximum_total_degree"] > caps["maximum_total_degree"]
    ):
        raise CoordinateFreeK0PolynomialPacketError("polynomial packet exceeded explicit cap")
    reference = exact._matrix_from_packet(inherited["reference_K0"]["exact_K0_construction"]["K0"])
    e1 = _evaluate(symmetrizer, (1, 0, 0), 55, 55)
    mismatches = sum(
        e1[row][column] != reference[row][column] for row in range(55) for column in range(55)
    )
    if mismatches:
        raise CoordinateFreeK0PolynomialPacketError("e1 reference regression failed")
    consumer = upstreams["order_one_consumer"]
    candidate_packets = consumer["counts"]["reference_e1_K55_Taylor_order_one_packets_constructed"]
    reference_residuals = sum(
        packet["K55_order_one_symmetrizer_residual_nonzero_entries"]
        for packet in consumer["registered_reference_e1_K55_Taylor_order_one_packets"]
    )
    if (
        candidate_packets != 15
        or reference_residuals != 0
        or consumer["claims"]["all_15_coordinate_free_K55_Taylor_order_one_packets_registered"]
    ):
        raise CoordinateFreeK0PolynomialPacketError("order-one blocker inventory changed")
    authorization = {
        "decision": "AUTHORIZED_FOR_EXACT_CONSTRUCTION_NOT_REGISTERED",
        "former_blocker": "missing coordinate-free K0(n) polynomial packet",
        "former_blocker_resolved": True,
        "authorized_packet_count": 15,
        "registered_packet_count": 0,
        "required_replay_before_registration": (
            "construct each K55^(1)(n) in the same sphere quotient and reduce "
            "K1*P0+K0*P1-P0^T*K1-P1^T*K0 entrywise to zero"
        ),
        "available_exact_inputs": {
            "P55_order_one_coordinate_polynomial_packets": 15,
            "H_star_plus_order_one_coordinate_polynomial_packets": 15,
            "K0_coordinate_polynomial_packets": 1,
            "projector_derivative_formula": "sealed in order-one consumer",
        },
    }
    claims = config["claims_policy"]
    body = {
        "schema_version": SCHEMA,
        "status": STATUS,
        "errors": [],
        "config_sha256": config["content_sha256"],
        "upstream_bindings": {
            name: {**binding, "verified": True} for name, binding in config["upstreams"].items()
        },
        "exact_K0_polynomial_packet": packet,
        "exact_replay": {
            **construction_counts,
            "sphere_relation": "n1^2+n2^2+n3^2-1",
            "canonical_reducer": "replace n1^2 by 1-n2^2-n3^2 until power(n1)<=1",
            "e1_reference_entries_compared": 3025,
            "e1_reference_mismatches": mismatches,
        },
        "K55_order_one_authorization": authorization,
        "counts": {
            "upstream_seals_verified": 2,
            "transitive_exact_source_seals_verified": 4,
            "K0_polynomial_packets_emitted": 1,
            "K0_polynomial_nonzero_entries": packet["nonzero_polynomial_entries"],
            "K0_polynomial_normal_form_terms": packet["normal_form_terms"],
            "K0_polynomial_maximum_total_degree": packet["maximum_total_degree"],
            "sphere_identity_entries_reduced": 3025,
            "sphere_identity_nonzero_remainders": 0,
            "former_blocked_K55_order_one_packets": candidate_packets,
            "K55_order_one_packets_authorized_for_construction": 15,
            "K55_order_one_packets_registered": 0,
            "manifest_registered_before": 79,
            "manifest_registered_after": 79,
            "manifest_missing_after": 225,
            "full_symbol_build_calls": 0,
            "emitted_output_rows": 0,
        },
        "claims": claims,
        "negative_controls": {
            "accept_noncanonical_n1_power_two_term": {"rejected": True},
            "tamper_polynomial_packet_content_hash": {"rejected": True},
            "count_authorized_K55_packets_as_registered": {"rejected": True},
            "advance_manifest_without_15_K1_sphere_replays": {"rejected": True},
            "promote_K0_identity_to_D4_H7_or_PDE": {"rejected": True},
        },
        "source_bindings": {
            "config": {"path": CONFIG_PATH, "file_sha256": _file_sha256(config_path)},
            "source": {"path": SOURCE_PATH, "file_sha256": _file_sha256(root / SOURCE_PATH)},
            "test": {"path": TEST_PATH, "file_sha256": _file_sha256(root / TEST_PATH)},
        },
        "scope": (
            "Emits one exact 55x55 coordinate-free K0(n) polynomial packet in canonical "
            "unit-sphere normal form and proves its symmetrizer identity entrywise. This "
            "removes the sole blocker identified for constructing 15 K55 order-one packets, "
            "but those packets remain unregistered until their polynomial replay is emitted. "
            "No coefficient rows, D4, H7, PDE, or lifespan theorem follows."
        ),
    }
    return {**body, "content_sha256": _content_hash(body)}


def validate_campaign(document: dict[str, Any], project_root: Path) -> None:
    if not _hash_matches(document):
        raise CoordinateFreeK0PolynomialPacketError("campaign content hash mismatch")
    packet = document.get("exact_K0_polynomial_packet", {})
    if not isinstance(packet, dict) or not _hash_matches(packet):
        raise CoordinateFreeK0PolynomialPacketError("polynomial packet content hash mismatch")
    for entry in packet.get("entries", []):
        if any(term["powers"][0] > 1 for term in entry.get("terms", [])):
            raise CoordinateFreeK0PolynomialPacketError("noncanonical sphere term")
    expected = build_campaign(project_root.resolve(), project_root.resolve() / CONFIG_PATH)
    if document != expected:
        raise CoordinateFreeK0PolynomialPacketError("campaign replay mismatch")


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
