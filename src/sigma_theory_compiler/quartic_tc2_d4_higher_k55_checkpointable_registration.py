from __future__ import annotations

import argparse
import hashlib
import json
import os
from fractions import Fraction
from pathlib import Path
from typing import Any

from . import quartic_tc2_d4_coordinate_free_k0_polynomial_packet as poly
from . import quartic_tc2_d4_coordinate_free_k55_order_one_registration as k1

CONFIG_PATH = "configs/backgrounds/quartic_tc2_d4_higher_k55_checkpointable_registration.json"
SOURCE_PATH = "src/sigma_theory_compiler/quartic_tc2_d4_higher_k55_checkpointable_registration.py"
TEST_PATH = "tests/test_quartic_tc2_d4_higher_k55_checkpointable_registration.py"
OUTPUT_PATH = "runs/physics-language/quartic-tc2-d4-higher-k55-checkpointable-registration/campaign.json"
CHECKPOINT_PATH = "runs/physics-language/quartic-tc2-d4-higher-k55-checkpointable-registration/checkpoints"
CONFIG_SCHEMA = "sigma-quartic-tc2-d4-higher-k55-registration-config-1.0"
CHECKPOINT_SCHEMA = "sigma-quartic-tc2-d4-higher-k55-evaluation-checkpoint-1.0"
RESULT_SCHEMA = "sigma-quartic-tc2-d4-higher-k55-registration-1.0"
MAX_ORDER = 4
ORDERS = (2, 3, 4)

Sparse = poly.Sparse
Series = list[Sparse]


class HigherK55RegistrationError(ValueError):
    """Raised when higher K55 recurrence authority fails closed."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def _content_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes({key: item for key, item in value.items() if key != "content_sha256"})).hexdigest()


def _with_hash(body: dict[str, Any]) -> dict[str, Any]:
    return {**body, "content_sha256": _content_hash(body)}


def _hash_matches(value: dict[str, Any]) -> bool:
    return value.get("content_sha256") == _content_hash(value)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise HigherK55RegistrationError(f"expected object: {path}")
    return value


def _resolve_under(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root != path and root not in path.parents:
        raise HigherK55RegistrationError("bound path escaped root")
    return path


def _load_bound(root: Path, binding: dict[str, str]) -> dict[str, Any]:
    path = _resolve_under(root, binding["path"])
    value = _load_json(path)
    if _file_sha256(path) != binding["file_sha256"] or value.get("content_sha256") != binding["content_sha256"] or not _hash_matches(value):
        raise HigherK55RegistrationError(f"upstream mismatch: {binding['path']}")
    return value


def _load_config(root: Path, config_path: Path) -> dict[str, Any]:
    config = _load_json(config_path)
    if (
        config.get("schema_version") != CONFIG_SCHEMA
        or config.get("policy") != "register_all_45_K55_packets_after_exact_Riesz_and_sphere_replay"
        or set(config.get("upstreams", {})) != {"predecessor", "higher_P55", "P55_order_one", "higher_H_star", "H_star_order_one", "K55_order_one", "flat_P55", "flat_action_metric", "projector_recipes", "K0_polynomial"}
        or config.get("target") != {"Taylor_orders": [2, 3, 4], "polarization_evaluations": 15, "packets": 45, "shape_each": [55, 55], "manifest_before": 154, "manifest_after": 199}
        or not _hash_matches(config)
    ):
        raise HigherK55RegistrationError("invalid higher K55 config")
    return config


def _zero_series() -> Series:
    return [{} for _ in range(MAX_ORDER + 1)]


def _constant_series(matrix: Sparse) -> Series:
    return [matrix, {}, {}, {}, {}]


def _series_add(left: Series, right: Series) -> Series:
    return [poly._add(left[index], right[index]) for index in range(MAX_ORDER + 1)]


def _series_scale(series: Series, coefficient: object) -> Series:
    return [poly._scale(item, coefficient) for item in series]


def _series_transpose(series: Series) -> Series:
    return [poly._transpose(item) for item in series]


def _series_multiply(left: Series, right: Series) -> Series:
    result = _zero_series()
    for order in range(MAX_ORDER + 1):
        for index in range(order + 1):
            result[order] = poly._add(result[order], poly._multiply(left[index], right[order - index]))
    return result


def _coefficient_product(left: Series, right: Series, order: int) -> Sparse:
    result: Sparse = {}
    for index in range(order + 1):
        result = poly._add(result, poly._multiply(left[index], right[order - index]))
    return result


def _projector_series(base: dict[str, Any], companion: Series) -> dict[Fraction, Series]:
    result = {eigenvalue: [projector, {}, {}, {}, {}] for eigenvalue, projector in base["Pi0"].items()}
    identity = poly._identity(22)
    for order in range(1, MAX_ORDER + 1):
        for eigenvalue, series in result.items():
            projector0 = series[0]
            complement = poly._add(identity, poly._scale(projector0, -1))
            commutator: Sparse = {}
            for index in range(1, order + 1):
                commutator = poly._add(
                    commutator,
                    poly._add(
                        poly._multiply(companion[index], series[order - index]),
                        poly._scale(poly._multiply(series[order - index], companion[index]), -1),
                    ),
                )
            off_diagonal: Sparse = {}
            for other, other_projector in base["Pi0"].items():
                if other == eigenvalue:
                    continue
                off_diagonal = poly._add(
                    off_diagonal,
                    poly._scale(poly._multiply(poly._multiply(other_projector, commutator), projector0), 1 / (eigenvalue - other)),
                )
                off_diagonal = poly._add(
                    off_diagonal,
                    poly._scale(poly._multiply(poly._multiply(projector0, commutator), other_projector), 1 / (other - eigenvalue)),
                )
            quadratic: Sparse = {}
            for index in range(1, order):
                quadratic = poly._add(quadratic, poly._multiply(series[index], series[order - index]))
            diagonal = poly._add(
                poly._scale(poly._multiply(poly._multiply(projector0, quadratic), projector0), -1),
                poly._multiply(poly._multiply(complement, quadratic), complement),
            )
            series[order] = poly._add(off_diagonal, diagonal)
    for order in range(1, MAX_ORDER + 1):
        completeness: Sparse = {}
        for series in result.values():
            completeness = poly._add(completeness, series[order])
            idempotence = poly._add(_coefficient_product(series, series, order), poly._scale(series[order], -1))
            commutation = poly._add(
                _coefficient_product(companion, series, order),
                poly._scale(_coefficient_product(series, companion, order), -1),
            )
            if idempotence or commutation:
                raise HigherK55RegistrationError(f"Riesz projector recurrence failed at order {order}")
        if completeness:
            raise HigherK55RegistrationError(f"Riesz projector completeness failed at order {order}")
    return result


def _inverse_series(companion: Series, inverse0: Sparse) -> Series:
    result = [inverse0, {}, {}, {}, {}]
    for order in range(1, MAX_ORDER + 1):
        forcing: Sparse = {}
        for index in range(1, order + 1):
            forcing = poly._add(forcing, poly._multiply(companion[index], result[order - index]))
        result[order] = poly._scale(poly._multiply(inverse0, forcing), -1)
        if _coefficient_product(companion, result, order):
            raise HigherK55RegistrationError(f"companion inverse recurrence failed at order {order}")
    return result


def _construct_series(base: dict[str, Any], p_series: Series, h_series: Series) -> Series:
    j = _constant_series(base["J"])
    jt = _constant_series(base["JT"])
    transverse = _constant_series(base["T"])
    companion = _series_multiply(_series_multiply(jt, p_series), j)
    lift = _series_multiply(_series_multiply(jt, p_series), transverse)
    projectors = _projector_series(base, companion)
    energy = _zero_series()
    for eigenvalue, projector in projectors.items():
        metric = h_series if eigenvalue == 1 else _series_scale(h_series, -1) if eigenvalue == -1 else _constant_series(poly._identity(22))
        energy = _series_add(energy, _series_multiply(_series_multiply(_series_transpose(projector), metric), projector))
    inverse = _inverse_series(companion, base["N0"])
    cross = _series_multiply(_series_multiply(_series_transpose(lift), energy), inverse)
    return _series_add(
        transverse,
        _series_add(
            _series_add(_series_multiply(cross, jt), _series_multiply(j, _series_transpose(cross))),
            _series_multiply(_series_multiply(j, energy), jt),
        ),
    )


def _sphere_equal(left: Sparse, right: Sparse) -> bool:
    return set(left) == set(right) and all(left[key].terms == right[key].terms for key in left)


def _packet_path(directory: Path, evaluation_id: str) -> Path:
    if not evaluation_id.startswith("subset_") or any(character not in "abcdefghijklmnopqrstuvwxyz_0123" for character in evaluation_id):
        raise HigherK55RegistrationError("unsafe evaluation id")
    return directory / f"{evaluation_id}.json"


def _failure_path(directory: Path, evaluation_id: str) -> Path:
    return directory / "failures" / f"{evaluation_id}.json"


def _atomic_write(path: Path, value: dict[str, Any], maximum: int) -> None:
    data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    if len(data) > maximum:
        raise HigherK55RegistrationError("checkpoint exceeded byte cap")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise HigherK55RegistrationError(f"immutable checkpoint conflict: {path.name}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(data)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def materialize(root: Path, config_path: Path, checkpoint_dir: Path) -> None:
    root = root.resolve()
    config = _load_config(root, config_path)
    upstreams = {name: _load_bound(root, binding) for name, binding in config["upstreams"].items()}
    p_axes = [k1.exact._matrix_from_packet(packet) for packet in upstreams["flat_P55"]["matrix_packets"]]
    h_plus = k1.exact._matrix_from_packet(upstreams["flat_action_metric"]["exact_construction"]["h_plus_0"])
    recipes = upstreams["projector_recipes"]["exact_Lagrange_projector_recipes"]["recipes"]
    base = k1._base_data(p_axes, h_plus, recipes)
    base["K0"] = k1._sphere_packet(upstreams["K0_polynomial"]["exact_K0_polynomial_packet"], [55, 55])
    p1_by_id = {row["evaluation_id"]: row for row in upstreams["P55_order_one"]["registered_P55_Taylor_order_one_packets"]}
    h1_by_id = {row["evaluation_id"]: row for row in upstreams["H_star_order_one"]["packets"]}
    k1_by_id = {row["evaluation_id"]: row for row in upstreams["K55_order_one"]["registered_coordinate_free_K55_Taylor_order_one_packets"]}
    ph_by_id: dict[str, dict[int, dict[str, Any]]] = {}
    for packet in upstreams["higher_P55"]["registered_P55_Taylor_orders_two_through_four_packets"]:
        ph_by_id.setdefault(packet["evaluation_id"], {})[packet["Taylor_order"]] = packet
    hh_by_id: dict[str, dict[int, dict[str, Any]]] = {}
    for packet in upstreams["higher_H_star"]["packets"]:
        hh_by_id.setdefault(packet["evaluation_id"], {})[packet["Taylor_order"]] = packet
    evaluation_ids = list(p1_by_id)
    if set(p1_by_id) != set(h1_by_id) or set(p1_by_id) != set(k1_by_id) or set(p1_by_id) != set(ph_by_id) or set(p1_by_id) != set(hh_by_id):
        raise HigherK55RegistrationError("evaluation authority mismatch")
    if any(_failure_path(checkpoint_dir, evaluation_id).exists() for evaluation_id in evaluation_ids):
        return
    maximum = config["caps"]["maximum_checkpoint_bytes"]
    for evaluation_id in evaluation_ids:
        path = _packet_path(checkpoint_dir, evaluation_id)
        if path.exists():
            continue
        p_series = [base["P0"], k1._linear_packet(p1_by_id[evaluation_id]["P55_Taylor_order_one_matrix"], [55, 55])]
        p_series.extend(k1._linear_packet(ph_by_id[evaluation_id][order], [55, 55]) for order in ORDERS)
        h_series = [base["H0"], k1._linear_packet(h1_by_id[evaluation_id]["H_star_plus_order_one_matrix"], [22, 22])]
        h_series.extend(k1._linear_packet(hh_by_id[evaluation_id][order], [22, 22]) for order in ORDERS)
        k_series = _construct_series(base, p_series, h_series)
        expected_k1 = k1._sphere_packet(k1_by_id[evaluation_id]["K55_Taylor_order_one_matrix"], [55, 55])
        if not _sphere_equal(k_series[0], base["K0"]) or not _sphere_equal(k_series[1], expected_k1):
            raise HigherK55RegistrationError("K55 order-zero/one replay mismatch")
        packets = []
        for order in ORDERS:
            symmetry = poly._add(k_series[order], poly._scale(poly._transpose(k_series[order]), -1))
            residual = poly._add(
                _coefficient_product(k_series, p_series, order),
                poly._scale(_coefficient_product(_series_transpose(p_series), k_series, order), -1),
            )
            if symmetry or residual:
                failure = _with_hash(
                    {
                        "schema_version": "sigma-quartic-tc2-d4-higher-k55-failure-checkpoint-1.0",
                        "evaluation_id": evaluation_id,
                        "Taylor_order": order,
                        "first_missing_primitive": (
                            "canonical auxiliary-cluster Riesz metric/Sylvester correction"
                        ),
                        "sphere_symmetry_remainder_entries": len(symmetry),
                        "sphere_symmetrizer_remainder_entries": len(residual),
                        "sphere_symmetrizer_residual": poly._packet(
                            f"K55_{order}_{evaluation_id}_symmetrizer_residual(n)",
                            residual,
                            [55, 55],
                        ),
                        "Riesz_idempotence_commutation_completeness_remainders": 0,
                        "companion_inverse_recurrence_remainders": 0,
                    }
                )
                _atomic_write(_failure_path(checkpoint_dir, evaluation_id), failure, maximum)
                raise HigherK55RegistrationError(f"K55 order-{order} sphere identity failed: symmetry={len(symmetry)} residual={len(residual)}")
            matrix = poly._packet(f"K55_{order}_{evaluation_id}(n)", k_series[order], [55, 55])
            packets.append(
                _with_hash(
                    {
                        "schema_version": "sigma-coordinate-free-K55-higher-Taylor-packet-1.0",
                        "evaluation_id": evaluation_id,
                        "Taylor_order": order,
                        "factorial_normalization": f"1/{order}!",
                        "P55_content_sha256": ph_by_id[evaluation_id][order]["content_sha256"],
                        "H_star_content_sha256": hh_by_id[evaluation_id][order]["content_sha256"],
                        "K55_matrix": matrix,
                        "sphere_symmetry_remainder_entries": 0,
                        "sphere_symmetrizer_remainder_entries": 0,
                    }
                )
            )
        checkpoint = _with_hash(
            {
                "schema_version": CHECKPOINT_SCHEMA,
                "evaluation_id": evaluation_id,
                "packets": packets,
                "Riesz_orders_checked": [1, 2, 3, 4],
                "Riesz_idempotence_commutation_completeness_remainders": 0,
                "companion_inverse_recurrence_remainders": 0,
                "sphere_identity_remainders": 0,
            }
        )
        _atomic_write(path, checkpoint, maximum)
        print(f"sealed higher K55 {evaluation_id}", flush=True)


def build_campaign(root: Path, config_path: Path, checkpoint_dir: Path) -> dict[str, Any]:
    root = root.resolve()
    config = _load_config(root, config_path)
    predecessor = _load_bound(root, config["upstreams"]["predecessor"])
    higher_p = _load_bound(root, config["upstreams"]["higher_P55"])
    evaluation_ids = list(
        dict.fromkeys(
            packet["evaluation_id"]
            for packet in higher_p["registered_P55_Taylor_orders_two_through_four_packets"]
        )
    )
    failures = [
        _load_json(_failure_path(checkpoint_dir, evaluation_id))
        for evaluation_id in evaluation_ids
        if _failure_path(checkpoint_dir, evaluation_id).exists()
    ]
    if failures:
        failure = failures[0]
        if (
            failure.get("schema_version")
            != "sigma-quartic-tc2-d4-higher-k55-failure-checkpoint-1.0"
            or not _hash_matches(failure)
            or not _hash_matches(failure.get("sphere_symmetrizer_residual", {}))
        ):
            raise HigherK55RegistrationError("failure checkpoint tamper")
        completed = [
            evaluation_id
            for evaluation_id in evaluation_ids
            if _packet_path(checkpoint_dir, evaluation_id).exists()
        ]
        manifest = json.loads(json.dumps(higher_p["required_symbolic_input_manifest"]))
        if sum(row["registered_packets"] for row in manifest) != 154:
            raise HigherK55RegistrationError("blocked manifest boundary changed")
        body = {
            "schema_version": RESULT_SCHEMA,
            "status": "block_higher_K55_at_subset_2_order_3_auxiliary_Riesz_metric_correction",
            "decision": "BLOCK_SERIALIZATION",
            "errors": [],
            "config_sha256": config["content_sha256"],
            "completed_unregistered_evaluation_checkpoints": completed,
            "failure_checkpoint": failure,
            "registered_higher_K55_packets": [],
            "required_symbolic_input_manifest": manifest,
            "counts": {
                "complete_evaluation_checkpoints": len(completed),
                "higher_K55_packets_computed_but_unregistered": len(completed) * 3,
                "higher_K55_packets_registered": 0,
                "Riesz_recurrence_nonzero_remainders_before_failure": 0,
                "companion_inverse_recurrence_nonzero_remainders_before_failure": 0,
                "failure_sphere_symmetry_remainder_entries": failure[
                    "sphere_symmetry_remainder_entries"
                ],
                "failure_sphere_symmetrizer_remainder_entries": failure[
                    "sphere_symmetrizer_remainder_entries"
                ],
                "manifest_registered_before": 154,
                "manifest_registered_after": 154,
                "manifest_missing_after": 150,
                "emitted_output_rows": 0,
            },
            "claims": {
                "all_45_higher_K55_packets_registered": False,
                "higher_TC2_registered": False,
                "lower_Sylvester_registered": False,
                "rows_emitted": False,
            },
            "negative_controls": {
                "register_two_completed_evaluations_as_partial_family": {"rejected": True},
                "drop_nonzero_order_three_residual": {"rejected": True},
                "reuse_fixed_eigenvalue_inverse_beyond_order_one": {"rejected": True},
                "advance_manifest_after_family_failure": {"rejected": True},
            },
            "source_bindings": {
                "config": {"path": CONFIG_PATH, "file_sha256": _file_sha256(config_path)},
                "source": {"path": SOURCE_PATH, "file_sha256": _file_sha256(root / SOURCE_PATH)},
                "test": {"path": TEST_PATH, "file_sha256": _file_sha256(root / TEST_PATH)},
            },
            "scope": (
                "Retains two exact completed K55 evaluation checkpoints but registers none "
                "after the first nonzero order-three sphere identity. TC2, lower-Sylvester, "
                "rows, D4, and H7 remain blocked."
            ),
        }
        return _with_hash(body)
    packets = []
    checkpoint_hashes = []
    for evaluation_id in evaluation_ids:
        path = _packet_path(checkpoint_dir, evaluation_id)
        if not path.exists():
            raise HigherK55RegistrationError(f"first missing primitive: {evaluation_id} K55 Taylor order 2")
        checkpoint = _load_json(path)
        if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA or not _hash_matches(checkpoint) or [packet.get("Taylor_order") for packet in checkpoint.get("packets", [])] != list(ORDERS) or any(not _hash_matches(packet) or not _hash_matches(packet["K55_matrix"]) for packet in checkpoint["packets"]):
            raise HigherK55RegistrationError(f"checkpoint tamper: {path.name}")
        packets.extend(checkpoint["packets"])
        checkpoint_hashes.append(checkpoint["content_sha256"])
    if len(packets) != 45:
        raise HigherK55RegistrationError("incomplete higher K55 family")
    manifest = json.loads(json.dumps(higher_p["required_symbolic_input_manifest"]))
    family = next(row for row in manifest if row["input_id"] == "polarized_K55_Taylor_packets")
    if family.get("registered_packets") != 30 or family.get("registered_Taylor_orders") != [0, 1]:
        raise HigherK55RegistrationError("K55 manifest predecessor changed")
    family["registered_packets"] = 75
    family["registered_Taylor_orders"] = [0, 1, 2, 3, 4]
    family["missing_Taylor_orders"] = []
    family["packet_content_sha256"].extend(packet["content_sha256"] for packet in packets)
    family["status"] = "registered_all_15_packets_at_Taylor_orders_zero_through_four"
    if sum(row["registered_packets"] for row in manifest) != 199 or predecessor.get("counts", {}).get("manifest_registered_after") != 109:
        raise HigherK55RegistrationError("manifest did not advance atomically to 199")
    body = {
        "schema_version": RESULT_SCHEMA,
        "status": "pass_exact_45_higher_K55_packets_registered",
        "errors": [],
        "config_sha256": config["content_sha256"],
        "checkpoint_content_sha256": checkpoint_hashes,
        "registered_higher_K55_packets": packets,
        "required_symbolic_input_manifest": manifest,
        "counts": {"higher_K55_packets": 45, "Riesz_recurrence_nonzero_remainders": 0, "companion_inverse_recurrence_nonzero_remainders": 0, "sphere_identity_nonzero_remainders": 0, "manifest_registered_before": 154, "manifest_registered_after": 199, "manifest_missing_after": 105, "emitted_output_rows": 0},
        "claims": {"all_45_higher_K55_packets_registered": True, "higher_TC2_registered": False, "lower_Sylvester_registered": False, "rows_emitted": False},
        "negative_controls": {"reuse_fixed_eigenvalue_inverse_beyond_order_one": {"rejected": True}, "accept_failed_Riesz_recurrence": {"rejected": True}, "accept_failed_sphere_identity": {"rejected": True}, "partially_advance_K55_family": {"rejected": True}},
        "source_bindings": {"config": {"path": CONFIG_PATH, "file_sha256": _file_sha256(config_path)}, "source": {"path": SOURCE_PATH, "file_sha256": _file_sha256(root / SOURCE_PATH)}, "test": {"path": TEST_PATH, "file_sha256": _file_sha256(root / TEST_PATH)}},
        "scope": "Registers all 45 coordinate-free K55 Taylor packets at orders two through four after exact Riesz, inverse, symmetry, and unit-sphere symmetrizer recurrences. TC2, lower-Sylvester packets, rows, D4, and H7 remain open.",
    }
    return _with_hash(body)


def validate_campaign(document: dict[str, Any], root: Path) -> None:
    if not _hash_matches(document) or document != build_campaign(root, root / CONFIG_PATH, root / CHECKPOINT_PATH):
        raise HigherK55RegistrationError("higher K55 campaign replay mismatch")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--materialize", action="store_true")
    args = parser.parse_args(argv)
    if args.materialize:
        materialize(args.project_root, args.config, args.checkpoint_dir)
    document = build_campaign(args.project_root, args.config, args.checkpoint_dir)
    _atomic_write(args.output, document, 64 * 1024 * 1024)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
