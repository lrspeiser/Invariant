from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from . import quartic_tc2_d4_coordinate_free_k55_order_one_registration as common
except ImportError:  # pragma: no cover
    import quartic_tc2_d4_coordinate_free_k55_order_one_registration as common


SCHEMA = "sigma-quartic-tc2-d4-coordinate-free-tc2-order-one-registration-1.0"
CONFIG_SCHEMA = f"{SCHEMA.removesuffix('-1.0')}-config-1.0"
STATUS = "pass_exact_15_coordinate_free_TC2_Taylor_order_one_packets_registered"
CONFIG_PATH = "configs/backgrounds/quartic_tc2_d4_coordinate_free_tc2_order_one_registration.json"
SOURCE_PATH = (
    "src/sigma_theory_compiler/quartic_tc2_d4_coordinate_free_tc2_order_one_registration.py"
)
TEST_PATH = "tests/test_quartic_tc2_d4_coordinate_free_tc2_order_one_registration.py"
OUTPUT_PATH = (
    "runs/physics-language/quartic-tc2-d4-coordinate-free-tc2-order-one-registration/campaign.json"
)
EMBEDDED_Q = {33: 2, 37: -8}
OUTPUT_COLUMN = 54

Sparse = common.Sparse
Polynomial = common.Polynomial
exact = common.exact


class CoordinateFreeTC2OrderOneRegistrationError(ValueError):
    """Raised when the exact TC2 order-one replay fails closed."""


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
        raise CoordinateFreeTC2OrderOneRegistrationError(f"expected object: {path}")
    return value


def _resolve_under(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root != path and root not in path.parents:
        raise CoordinateFreeTC2OrderOneRegistrationError("bound path escaped root")
    return path


def _load_bound(root: Path, binding: dict[str, str]) -> dict[str, Any]:
    path = _resolve_under(root, binding["path"])
    value = _load_json(path)
    if (
        _file_sha256(path) != binding["file_sha256"]
        or value.get("content_sha256") != binding["content_sha256"]
        or not _hash_matches(value)
    ):
        raise CoordinateFreeTC2OrderOneRegistrationError(f"upstream mismatch: {binding['path']}")
    return value


def _validate_config(config: dict[str, Any]) -> None:
    expected_claims = {
        "all_15_coordinate_free_TC2_Taylor_order_one_packets_registered": True,
        "manifest_advanced_from_94_to_109": True,
        "complete_coordinate_free_coefficient_map_emitted": False,
        "full_direction_sphere_D4_compatibility_proved": False,
        "global_H7_closed": False,
        "nonlinear_PDE_closure_proved": False,
        "lifespan_proved": False,
    }
    if (
        config.get("schema_version") != CONFIG_SCHEMA
        or config.get("policy")
        != "differentiate_TC2_equals_a10_times_P_times_q_at_fixed_candidate_coefficient"
        or set(config.get("upstreams", {}))
        != {"manifest_predecessor", "P55_order_one", "TC2_order_zero"}
        or config.get("target")
        != {
            "required_packets": 15,
            "shape_each": [55, 55],
            "embedded_Q_column_10": {"33": "2", "37": "-8"},
            "output_covector_index": 54,
            "scalar_prefactor": "a10",
            "candidate_coefficient_derivative": "0",
            "manifest_registered_before": 94,
            "manifest_registered_after": 109,
            "manifest_missing_after": 195,
        }
        or config.get("caps")
        != {
            "maximum_packets": 15,
            "maximum_matrix_dimension": 55,
            "maximum_terms_per_packet": 1000,
            "maximum_total_degree": 1,
            "maximum_full_symbol_build_calls": 0,
            "maximum_output_rows_emitted": 0,
        }
        or config.get("claims_policy") != expected_claims
        or not _hash_matches(config)
    ):
        raise CoordinateFreeTC2OrderOneRegistrationError("invalid TC2 order-one config")


def _add(left: Sparse, right: Sparse) -> Sparse:
    return common._add(left, right)


def _scale(matrix: Sparse, coefficient: object) -> Sparse:
    return common._scale(matrix, coefficient)


def _construct_unit_tc2_one(p1: Sparse) -> Sparse:
    result: Sparse = {}
    for (row, column), polynomial in p1.items():
        if column not in EMBEDDED_Q:
            continue
        key = (row, OUTPUT_COLUMN)
        contribution = {key: polynomial * EMBEDDED_Q[column]}
        result = _add(result, contribution)
    return result


def _packet(name: str, matrix: Sparse) -> dict[str, Any]:
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
    terms = [term for entry in entries for term in entry["terms"]]
    body = {
        "schema_version": "sigma-exact-sparse-Qsqrt2-sphere-polynomial-TC2-order-one-1.0",
        "name": name,
        "shape": [55, 55],
        "variables": ["n1", "n2", "n3"],
        "scalar_prefactor": "a10",
        "candidate_coefficient_derivative": "0",
        "entries": entries,
        "nonzero_polynomial_entries": len(entries),
        "normal_form_terms": len(terms),
        "maximum_total_degree": max((sum(term["powers"]) for term in terms), default=0),
        "right_support_columns": sorted({entry["column"] for entry in entries}),
        "normal_form_sha256": hashlib.sha256(_canonical_bytes(entries)).hexdigest(),
    }
    return {**body, "content_sha256": _content_hash(body)}


def _jet_coefficient_certificate(records: list[dict[str, Any]]) -> dict[str, Any]:
    basis_directions = sorted(
        {
            binding["basis_jet_direction"]
            for record in records
            for binding in record["basis_bindings"]
        }
    )
    if basis_directions != ["G_01", "G_12", "H_01", "H_11"]:
        raise CoordinateFreeTC2OrderOneRegistrationError("registered jet basis changed")
    if any(
        "a10" in binding["basis_jet_direction"]
        for record in records
        for binding in record["basis_bindings"]
    ):
        raise CoordinateFreeTC2OrderOneRegistrationError("candidate coefficient jet present")
    return {
        "registered_basis_jet_directions": basis_directions,
        "candidate_parameter_directions_present": [],
        "a10_Taylor_order_one": "0",
        "product_rule": "TC2_1=a10*P1*q*e54^T+(a10)_1*P0*q*e54^T",
        "surviving_term": "a10*P1*q*e54^T",
        "omitted_term_proved_zero": "(a10)_1*P0*q*e54^T",
    }


def _updated_manifest(
    predecessor: dict[str, Any], packet_hashes: list[str]
) -> list[dict[str, Any]]:
    manifest = json.loads(json.dumps(predecessor["required_symbolic_input_manifest"]))
    family = next(item for item in manifest if item["input_id"] == "polarized_TC2_Taylor_packets")
    if family["registered_packets"] != 15 or family["registered_Taylor_orders"] != [0]:
        raise CoordinateFreeTC2OrderOneRegistrationError("TC2 manifest predecessor changed")
    family["registered_packets"] = 30
    family["registered_Taylor_orders"] = [0, 1]
    family["missing_Taylor_orders"] = [2, 3, 4]
    family["packet_content_sha256"].extend(packet_hashes)
    family["status"] = "partially_registered_all_15_packets_at_Taylor_orders_zero_and_one"
    if sum(item["registered_packets"] for item in manifest) != 109:
        raise CoordinateFreeTC2OrderOneRegistrationError("manifest did not advance to 109")
    return manifest


def build_campaign(project_root: Path, config_path: Path) -> dict[str, Any]:
    root = project_root.resolve()
    config = _load_json(config_path)
    _validate_config(config)
    upstreams = {name: _load_bound(root, binding) for name, binding in config["upstreams"].items()}
    predecessor = upstreams["manifest_predecessor"]
    p55 = upstreams["P55_order_one"]
    tc20 = upstreams["TC2_order_zero"]
    records = p55["registered_P55_Taylor_order_one_packets"]
    if (
        predecessor["counts"]["manifest_registered_after"] != 94
        or len(records) != 15
        or tc20["exact_TC2_order_zero_construction"]["scalar_prefactor_factored_not_sampled"]
        != "a10"
        or tc20["exact_TC2_order_zero_construction"]["unit_matrix"]["embedded_Q_column_10"]
        != [{"row": 33, "value": "2"}, {"row": 37, "value": "-8"}]
    ):
        raise CoordinateFreeTC2OrderOneRegistrationError("TC2 source boundary changed")
    coefficient_certificate = _jet_coefficient_certificate(records)
    packets = []
    caps = config["caps"]
    for record in records:
        p1 = common._linear_packet(record["P55_Taylor_order_one_matrix"], [55, 55])
        unit_tc21 = _construct_unit_tc2_one(p1)
        packet = _packet(f"unit_TC2_1_{record['evaluation_id']}(n)", unit_tc21)
        if (
            packet["normal_form_terms"] > caps["maximum_terms_per_packet"]
            or packet["maximum_total_degree"] > caps["maximum_total_degree"]
            or any(column != OUTPUT_COLUMN for column in packet["right_support_columns"])
        ):
            raise CoordinateFreeTC2OrderOneRegistrationError("TC2 packet exceeded boundary")
        body = {
            "schema_version": "sigma-coordinate-free-TC2-Taylor-order-one-packet-1.0",
            "evaluation_id": record["evaluation_id"],
            "evaluation_content_sha256": record["evaluation_content_sha256"],
            "P55_order_one_content_sha256": record["content_sha256"],
            "Taylor_order": 1,
            "factorial_normalization": "1/1!=1",
            "identity": "TC2_1(n)=a10*P1(n)*(2*e33-8*e37)*e54^T",
            "unit_TC2_Taylor_order_one_matrix": packet,
            "product_rule_nonzero_remainder_entries": 0,
        }
        packets.append({**body, "content_sha256": _content_hash(body)})
    manifest = _updated_manifest(predecessor, [packet["content_sha256"] for packet in packets])
    total_entries = sum(
        packet["unit_TC2_Taylor_order_one_matrix"]["nonzero_polynomial_entries"]
        for packet in packets
    )
    total_terms = sum(
        packet["unit_TC2_Taylor_order_one_matrix"]["normal_form_terms"] for packet in packets
    )
    zero_packets = sum(
        packet["unit_TC2_Taylor_order_one_matrix"]["normal_form_terms"] == 0 for packet in packets
    )
    body = {
        "schema_version": SCHEMA,
        "status": STATUS,
        "errors": [],
        "config_sha256": config["content_sha256"],
        "upstream_bindings": {
            name: {**binding, "verified": True} for name, binding in config["upstreams"].items()
        },
        "candidate_coefficient_derivative_certificate": coefficient_certificate,
        "registered_coordinate_free_TC2_Taylor_order_one_packets": packets,
        "required_symbolic_input_manifest": manifest,
        "counts": {
            "upstream_seals_verified": 3,
            "TC2_order_one_packets_required": 15,
            "TC2_order_one_packets_registered": len(packets),
            "TC2_order_one_zero_packets": zero_packets,
            "TC2_order_one_nonzero_polynomial_entries_total": total_entries,
            "TC2_order_one_normal_form_terms_total": total_terms,
            "TC2_order_one_maximum_total_degree": max(
                packet["unit_TC2_Taylor_order_one_matrix"]["maximum_total_degree"]
                for packet in packets
            ),
            "candidate_coefficient_derivative_nonzero_terms": 0,
            "product_rule_nonzero_remainders": 0,
            "manifest_registered_before": 94,
            "manifest_registered_after": 109,
            "manifest_missing_after": 195,
            "required_output_rows": 117180,
            "emitted_output_rows": 0,
            "full_symbol_build_calls": 0,
        },
        "claims": config["claims_policy"],
        "negative_controls": {
            "invent_nonzero_a10_derivative": {"rejected": True},
            "omit_minus_eight_embedded_Q_term": {"rejected": True},
            "accept_fewer_than_15_evaluation_ids": {"rejected": True},
            "advance_manifest_by_other_than_15": {"rejected": True},
            "promote_TC2_order_one_to_D4_H7_or_PDE": {"rejected": True},
        },
        "source_bindings": {
            "config": {"path": CONFIG_PATH, "file_sha256": _file_sha256(config_path)},
            "source": {"path": SOURCE_PATH, "file_sha256": _file_sha256(root / SOURCE_PATH)},
            "test": {"path": TEST_PATH, "file_sha256": _file_sha256(root / TEST_PATH)},
        },
        "scope": (
            "Registers exactly 15 coordinate-free TC2 Taylor-order-one packets for the "
            "sealed metric/action jet basis, where a10 is a fixed candidate coefficient. "
            "TC2 orders two through four, recurrence rows, D4, H7, PDE, and lifespan remain open."
        ),
    }
    return {**body, "content_sha256": _content_hash(body)}


def validate_campaign(document: dict[str, Any], project_root: Path) -> None:
    if not _hash_matches(document):
        raise CoordinateFreeTC2OrderOneRegistrationError("campaign content hash mismatch")
    expected = build_campaign(project_root.resolve(), project_root.resolve() / CONFIG_PATH)
    if document != expected:
        raise CoordinateFreeTC2OrderOneRegistrationError("campaign replay mismatch")


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
