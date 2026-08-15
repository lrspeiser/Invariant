from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

try:
    from . import quartic_tc2_d4_coordinate_free_k0_polynomial_packet as poly
except ImportError:  # pragma: no cover
    import quartic_tc2_d4_coordinate_free_k0_polynomial_packet as poly


SCHEMA = "sigma-quartic-tc2-d4-coordinate-free-k55-order-one-registration-1.0"
CONFIG_SCHEMA = f"{SCHEMA.removesuffix('-1.0')}-config-1.0"
STATUS = "pass_exact_15_coordinate_free_K55_Taylor_order_one_packets_registered"
CONFIG_PATH = "configs/backgrounds/quartic_tc2_d4_coordinate_free_k55_order_one_registration.json"
SOURCE_PATH = (
    "src/sigma_theory_compiler/quartic_tc2_d4_coordinate_free_k55_order_one_registration.py"
)
TEST_PATH = "tests/test_quartic_tc2_d4_coordinate_free_k55_order_one_registration.py"
OUTPUT_PATH = (
    "runs/physics-language/quartic-tc2-d4-coordinate-free-k55-order-one-registration/campaign.json"
)

Sparse = poly.Sparse
Polynomial = poly.Polynomial
exact = poly.exact
N = poly.N
PONE = poly.PONE


class CoordinateFreeK55OrderOneRegistrationError(ValueError):
    """Raised when the coordinate-free K55 order-one replay fails closed."""


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
        raise CoordinateFreeK55OrderOneRegistrationError(f"expected object: {path}")
    return value


def _resolve_under(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root != path and root not in path.parents:
        raise CoordinateFreeK55OrderOneRegistrationError("bound path escaped root")
    return path


def _load_bound(root: Path, binding: dict[str, str]) -> dict[str, Any]:
    path = _resolve_under(root, binding["path"])
    value = _load_json(path)
    if (
        _file_sha256(path) != binding["file_sha256"]
        or value.get("content_sha256") != binding["content_sha256"]
        or not _hash_matches(value)
    ):
        raise CoordinateFreeK55OrderOneRegistrationError(f"upstream mismatch: {binding['path']}")
    return value


def _validate_config(config: dict[str, Any]) -> None:
    expected_claims = {
        "all_15_coordinate_free_K55_Taylor_order_one_packets_registered": True,
        "manifest_advanced_from_79_to_94": True,
        "complete_coordinate_free_coefficient_map_emitted": False,
        "full_direction_sphere_D4_compatibility_proved": False,
        "global_H7_closed": False,
        "nonlinear_PDE_closure_proved": False,
        "lifespan_proved": False,
    }
    if (
        config.get("schema_version") != CONFIG_SCHEMA
        or config.get("policy")
        != "register_only_after_all_15_differentiated_sphere_symmetrizer_identities_vanish"
        or set(config.get("upstreams", {}))
        != {
            "K0_polynomial",
            "P55_order_one",
            "H_star_order_one",
            "flat_P55",
            "flat_action_metric",
            "projector_recipes",
        }
        or config.get("target")
        != {
            "required_packets": 15,
            "shape_each": [55, 55],
            "manifest_registered_before": 79,
            "manifest_registered_after": 94,
            "manifest_missing_after": 210,
        }
        or config.get("caps")
        != {
            "maximum_packets": 15,
            "maximum_matrix_dimension": 55,
            "maximum_terms_per_packet": 250000,
            "maximum_total_degree": 32,
            "maximum_full_symbol_build_calls": 0,
            "maximum_output_rows_emitted": 0,
        }
        or config.get("claims_policy") != expected_claims
        or not _hash_matches(config)
    ):
        raise CoordinateFreeK55OrderOneRegistrationError("invalid K55 order-one config")


def _add(left: Sparse, right: Sparse) -> Sparse:
    return poly._add(left, right)


def _scale(matrix: Sparse, coefficient: object) -> Sparse:
    return poly._scale(matrix, coefficient)


def _multiply(left: Sparse, right: Sparse) -> Sparse:
    return poly._multiply(left, right)


def _transpose(matrix: Sparse) -> Sparse:
    return poly._transpose(matrix)


def _constant(matrix: Any) -> Sparse:
    return poly._constant(matrix)


def _identity(size: int) -> Sparse:
    return poly._identity(size)


def _sparse_equal(left: Sparse, right: Sparse) -> bool:
    return set(left) == set(right) and all(left[key].terms == right[key].terms for key in left)


def _linear_packet(packet: dict[str, Any], shape: list[int]) -> Sparse:
    if not _hash_matches(packet) or packet.get("shape") != shape:
        raise CoordinateFreeK55OrderOneRegistrationError("linear packet boundary changed")
    result: Sparse = {}
    seen: set[tuple[int, int]] = set()
    for entry in packet.get("entries", []):
        key = (entry["row"], entry["column"])
        if key in seen:
            raise CoordinateFreeK55OrderOneRegistrationError("duplicate linear packet cell")
        terms: dict[tuple[int, int, int], Any] = {}
        if "constant" in entry:
            terms[(0, 0, 0)] = exact._parse_surd(entry["constant"])
        for axis, variable in enumerate(("n1", "n2", "n3")):
            if variable in entry.get("linear_coefficients", {}):
                terms[tuple(1 if index == axis else 0 for index in range(3))] = exact._parse_surd(
                    entry["linear_coefficients"][variable]
                )
        polynomial = Polynomial(terms)
        if polynomial.terms:
            result[key] = polynomial
        seen.add(key)
    return result


def _sphere_packet(packet: dict[str, Any], shape: list[int]) -> Sparse:
    if not _hash_matches(packet) or packet.get("shape") != shape:
        raise CoordinateFreeK55OrderOneRegistrationError("sphere packet boundary changed")
    result: Sparse = {}
    for entry in packet.get("entries", []):
        key = (entry["row"], entry["column"])
        if key in result:
            raise CoordinateFreeK55OrderOneRegistrationError("duplicate sphere packet cell")
        terms = {
            tuple(term["powers"]): exact._parse_surd(term["coefficient"]) for term in entry["terms"]
        }
        if any(powers[0] > 1 for powers in terms):
            raise CoordinateFreeK55OrderOneRegistrationError("noncanonical sphere packet")
        result[key] = Polynomial(terms)
    return result


def _base_data(p_axes: list[Any], h_plus: Any, recipes: list[dict[str, Any]]) -> dict[str, Any]:
    pencil: Sparse = {}
    for axis, matrix in enumerate(p_axes):
        pencil = _add(pencil, _scale(_constant(matrix), N[axis]))
    embedding = poly._normal_embedding()
    embedding_t = _transpose(embedding)
    transverse = _add(_identity(55), _scale(_multiply(embedding, embedding_t), -1))
    companion = _multiply(_multiply(embedding_t, pencil), embedding)
    lift = _multiply(_multiply(embedding_t, pencil), transverse)
    projectors = poly._projectors(companion, recipes)
    h_plus_sparse = _constant(h_plus)
    energy: Sparse = {}
    inverse: Sparse = {}
    for eigenvalue, projector in projectors.items():
        metric = (
            h_plus_sparse
            if eigenvalue == 1
            else _scale(h_plus_sparse, -1)
            if eigenvalue == -1
            else _identity(22)
        )
        energy = _add(
            energy,
            _multiply(_multiply(_transpose(projector), metric), projector),
        )
        inverse = _add(inverse, _scale(projector, 1 / eigenvalue))
    return {
        "P0": pencil,
        "J": embedding,
        "JT": embedding_t,
        "T": transverse,
        "C0": companion,
        "L0": lift,
        "Pi0": projectors,
        "G0": energy,
        "N0": inverse,
        "H0": h_plus_sparse,
    }


def _projector_derivatives(base: dict[str, Any], companion1: Sparse) -> dict[Fraction, Sparse]:
    result: dict[Fraction, Sparse] = {}
    total: Sparse = {}
    companion0 = base["C0"]
    for eigenvalue, projector0 in base["Pi0"].items():
        projector1: Sparse = {}
        for other, other_projector in base["Pi0"].items():
            if other == eigenvalue:
                continue
            numerator = _add(
                _multiply(_multiply(other_projector, companion1), projector0),
                _multiply(_multiply(projector0, companion1), other_projector),
            )
            projector1 = _add(projector1, _scale(numerator, 1 / (eigenvalue - other)))
        idempotence = _add(
            _add(_multiply(projector1, projector0), _multiply(projector0, projector1)),
            _scale(projector1, -1),
        )
        commutator = _add(
            _add(_multiply(companion1, projector0), _multiply(companion0, projector1)),
            _scale(
                _add(_multiply(projector1, companion0), _multiply(projector0, companion1)),
                -1,
            ),
        )
        if idempotence or commutator:
            raise CoordinateFreeK55OrderOneRegistrationError("projector derivative identity failed")
        result[eigenvalue] = projector1
        total = _add(total, projector1)
    if total:
        raise CoordinateFreeK55OrderOneRegistrationError("projector derivative completeness failed")
    return result


def _construct_one(base: dict[str, Any], p1: Sparse, h1: Sparse) -> tuple[Sparse, dict[str, int]]:
    companion1 = _multiply(_multiply(base["JT"], p1), base["J"])
    lift1 = _multiply(_multiply(base["JT"], p1), base["T"])
    projector1 = _projector_derivatives(base, companion1)
    energy1: Sparse = {}
    for eigenvalue, projector0 in base["Pi0"].items():
        derivative = projector1[eigenvalue]
        metric0 = (
            base["H0"]
            if eigenvalue == 1
            else _scale(base["H0"], -1)
            if eigenvalue == -1
            else _identity(22)
        )
        metric1 = h1 if eigenvalue == 1 else _scale(h1, -1) if eigenvalue == -1 else {}
        energy1 = _add(
            energy1,
            _add(
                _add(
                    _multiply(_multiply(_transpose(derivative), metric0), projector0),
                    _multiply(_multiply(_transpose(projector0), metric1), projector0),
                ),
                _multiply(_multiply(_transpose(projector0), metric0), derivative),
            ),
        )
    inverse1 = _scale(_multiply(_multiply(base["N0"], companion1), base["N0"]), -1)
    cross1 = _add(
        _add(
            _multiply(_multiply(_transpose(lift1), base["G0"]), base["N0"]),
            _multiply(_multiply(_transpose(base["L0"]), energy1), base["N0"]),
        ),
        _multiply(_multiply(_transpose(base["L0"]), base["G0"]), inverse1),
    )
    k1 = _add(
        _add(_multiply(cross1, base["JT"]), _multiply(base["J"], _transpose(cross1))),
        _multiply(_multiply(base["J"], energy1), base["JT"]),
    )
    symmetry = _add(k1, _scale(_transpose(k1), -1))
    residual = _add(
        _add(_multiply(k1, base["P0"]), _multiply(base["K0"], p1)),
        _scale(
            _add(
                _multiply(_transpose(base["P0"]), k1),
                _multiply(_transpose(p1), base["K0"]),
            ),
            -1,
        ),
    )
    if symmetry or residual:
        raise CoordinateFreeK55OrderOneRegistrationError(
            "differentiated K55 sphere identity failed"
        )
    return k1, {
        "P1_nonzero_polynomial_entries": len(p1),
        "H_star_plus_1_nonzero_polynomial_entries": len(h1),
        "C1_nonzero_polynomial_entries": len(companion1),
        "L1_nonzero_polynomial_entries": len(lift1),
        "G1_nonzero_polynomial_entries": len(energy1),
        "K1_symmetry_remainder_entries": len(symmetry),
        "K1_differentiated_symmetrizer_remainder_entries": len(residual),
    }


def _updated_manifest(
    predecessor: dict[str, Any], packet_hashes: list[str]
) -> list[dict[str, Any]]:
    manifest = json.loads(json.dumps(predecessor["required_symbolic_input_manifest"]))
    row = next(item for item in manifest if item["input_id"] == "polarized_K55_Taylor_packets")
    if row["registered_packets"] != 15 or row["registered_Taylor_orders"] != [0]:
        raise CoordinateFreeK55OrderOneRegistrationError("K55 manifest predecessor changed")
    row["registered_packets"] = 30
    row["registered_Taylor_orders"] = [0, 1]
    row["missing_Taylor_orders"] = [2, 3, 4]
    row["packet_content_sha256"].extend(packet_hashes)
    row["status"] = "partially_registered_all_15_packets_at_Taylor_orders_zero_and_one"
    if sum(item["registered_packets"] for item in manifest) != 94:
        raise CoordinateFreeK55OrderOneRegistrationError("manifest total did not advance to 94")
    return manifest


def build_campaign(project_root: Path, config_path: Path) -> dict[str, Any]:
    root = project_root.resolve()
    config = _load_json(config_path)
    _validate_config(config)
    upstreams = {name: _load_bound(root, binding) for name, binding in config["upstreams"].items()}
    predecessor = upstreams["P55_order_one"]
    h_result = upstreams["H_star_order_one"]
    if (
        predecessor["counts"]["registered_symbolic_input_packets"] != 79
        or len(predecessor["registered_P55_Taylor_order_one_packets"]) != 15
        or len(h_result["packets"]) != 15
    ):
        raise CoordinateFreeK55OrderOneRegistrationError("first-order input boundary changed")
    p_axes = [
        exact._matrix_from_packet(packet) for packet in upstreams["flat_P55"]["matrix_packets"]
    ]
    h_plus = exact._matrix_from_packet(
        upstreams["flat_action_metric"]["exact_construction"]["h_plus_0"]
    )
    recipes = upstreams["projector_recipes"]["exact_Lagrange_projector_recipes"]["recipes"]
    base = _base_data(p_axes, h_plus, recipes)
    k0_packet = upstreams["K0_polynomial"]["exact_K0_polynomial_packet"]
    base["K0"] = _sphere_packet(k0_packet, [55, 55])
    reconstructed_k0, _ = poly._construct(p_axes, h_plus, recipes)
    if not _sparse_equal(base["K0"], reconstructed_k0):
        raise CoordinateFreeK55OrderOneRegistrationError("K0 packet replay mismatch")
    p1_by_id = {
        packet["evaluation_id"]: packet
        for packet in predecessor["registered_P55_Taylor_order_one_packets"]
    }
    h1_by_id = {packet["evaluation_id"]: packet for packet in h_result["packets"]}
    if set(p1_by_id) != set(h1_by_id) or len(p1_by_id) != 15:
        raise CoordinateFreeK55OrderOneRegistrationError("P1/H1 evaluation sets differ")
    packets = []
    caps = config["caps"]
    for evaluation_id, p1_record in p1_by_id.items():
        h1_record = h1_by_id[evaluation_id]
        p1 = _linear_packet(p1_record["P55_Taylor_order_one_matrix"], [55, 55])
        h1 = _linear_packet(h1_record["H_star_plus_order_one_matrix"], [22, 22])
        k1, replay = _construct_one(base, p1, h1)
        packet = poly._packet(f"K55_1_{evaluation_id}(n)", k1, [55, 55])
        if (
            packet["normal_form_terms"] > caps["maximum_terms_per_packet"]
            or packet["maximum_total_degree"] > caps["maximum_total_degree"]
        ):
            raise CoordinateFreeK55OrderOneRegistrationError("K1 packet exceeded cap")
        body = {
            "schema_version": "sigma-coordinate-free-K55-Taylor-order-one-packet-1.0",
            "evaluation_id": evaluation_id,
            "P55_order_one_content_sha256": p1_record["content_sha256"],
            "H_star_order_one_content_sha256": h1_record["content_sha256"],
            "K55_Taylor_order_one_matrix": packet,
            "exact_replay": replay,
        }
        packets.append({**body, "content_sha256": _content_hash(body)})
    packet_hashes = [packet["content_sha256"] for packet in packets]
    manifest = _updated_manifest(predecessor, packet_hashes)
    total_entries = sum(
        packet["K55_Taylor_order_one_matrix"]["nonzero_polynomial_entries"] for packet in packets
    )
    total_terms = sum(
        packet["K55_Taylor_order_one_matrix"]["normal_form_terms"] for packet in packets
    )
    maximum_degree = max(
        packet["K55_Taylor_order_one_matrix"]["maximum_total_degree"] for packet in packets
    )
    body = {
        "schema_version": SCHEMA,
        "status": STATUS,
        "errors": [],
        "config_sha256": config["content_sha256"],
        "upstream_bindings": {
            name: {**binding, "verified": True} for name, binding in config["upstreams"].items()
        },
        "registered_coordinate_free_K55_Taylor_order_one_packets": packets,
        "required_symbolic_input_manifest": manifest,
        "counts": {
            "upstream_seals_verified": 6,
            "K55_order_one_packets_required": 15,
            "K55_order_one_packets_registered": len(packets),
            "K55_order_one_nonzero_polynomial_entries_total": total_entries,
            "K55_order_one_normal_form_terms_total": total_terms,
            "K55_order_one_maximum_total_degree": maximum_degree,
            "differentiated_identity_matrix_entries_reduced": 15 * 3025,
            "differentiated_identity_nonzero_remainders": 0,
            "manifest_registered_before": 79,
            "manifest_registered_after": 94,
            "manifest_missing_after": 210,
            "required_output_rows": 117180,
            "emitted_output_rows": 0,
            "full_symbol_build_calls": 0,
        },
        "claims": config["claims_policy"],
        "negative_controls": {
            "accept_one_failed_differentiated_identity": {"rejected": True},
            "accept_fewer_than_15_evaluation_ids": {"rejected": True},
            "accept_noncanonical_sphere_packet": {"rejected": True},
            "advance_manifest_by_other_than_15": {"rejected": True},
            "promote_order_one_registration_to_D4_H7_or_PDE": {"rejected": True},
        },
        "source_bindings": {
            "config": {"path": CONFIG_PATH, "file_sha256": _file_sha256(config_path)},
            "source": {"path": SOURCE_PATH, "file_sha256": _file_sha256(root / SOURCE_PATH)},
            "test": {"path": TEST_PATH, "file_sha256": _file_sha256(root / TEST_PATH)},
        },
        "scope": (
            "Registers exactly 15 coordinate-free K55 Taylor-order-one polynomial packets "
            "after entrywise differentiated symmetrizer replay in the unit-sphere quotient. "
            "K55 orders two through four, recurrence rows, D4, H7, PDE, and lifespan remain open."
        ),
    }
    return {**body, "content_sha256": _content_hash(body)}


def validate_campaign(document: dict[str, Any], project_root: Path) -> None:
    if not _hash_matches(document):
        raise CoordinateFreeK55OrderOneRegistrationError("campaign content hash mismatch")
    expected = build_campaign(project_root.resolve(), project_root.resolve() / CONFIG_PATH)
    if document != expected:
        raise CoordinateFreeK55OrderOneRegistrationError("campaign replay mismatch")


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
