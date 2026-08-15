from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA = "sigma-quartic-tc2-d4-h-star-order-one-checkpoint-readiness-gate-1.0"
CONFIG_SCHEMA = f"{SCHEMA.removesuffix('-1.0')}-config-1.0"
STATUS = "ready_checkpointable_H_star_plus_order_one_materializer"
CONFIG_PATH = "configs/backgrounds/quartic_tc2_d4_h_star_order_one_checkpoint_readiness_gate.json"
SOURCE_PATH = (
    "src/sigma_theory_compiler/quartic_tc2_d4_h_star_order_one_checkpoint_readiness_gate.py"
)
TEST_PATH = "tests/test_quartic_tc2_d4_h_star_order_one_checkpoint_readiness_gate.py"
OUTPUT_PATH = (
    "runs/physics-language/quartic-tc2-d4-h-star-order-one-checkpoint-readiness-gate/campaign.json"
)
BASIS_ATOMS = ["G_12", "G_01", "H_01", "H_11"]
EXPECTED_EVALUATIONS = [
    "subset_0",
    "subset_1",
    "subset_2",
    "subset_3",
    "subset_01",
    "subset_02",
    "subset_03",
    "subset_12",
    "subset_13",
    "subset_23",
    "subset_012",
    "subset_013",
    "subset_023",
    "subset_123",
    "subset_0123",
]


class HStarOrderOneReadinessError(ValueError):
    """Raised when the exact H-star order-one readiness boundary changes."""


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
        raise HStarOrderOneReadinessError("bound path escaped project root")
    return path


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise HStarOrderOneReadinessError(f"expected JSON object: {path}")
    return value


def _load_bound(root: Path, binding: dict[str, str]) -> dict[str, Any]:
    path = _resolve_under(root, binding["path"])
    value = _load_json(path)
    if (
        _file_sha256(path) != binding["file_sha256"]
        or value.get("content_sha256") != binding["content_sha256"]
        or not _hash_matches(value)
    ):
        raise HStarOrderOneReadinessError(f"upstream mismatch: {binding['path']}")
    return value


def _validate_formula_source(root: Path, binding: dict[str, Any]) -> dict[str, Any]:
    path = _resolve_under(root, binding["path"])
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = {
        node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    required = set(binding["required_functions"])
    if _file_sha256(path) != binding["file_sha256"] or not required <= functions:
        raise HStarOrderOneReadinessError(f"formula source mismatch: {binding['path']}")
    return {"functions_verified": len(required), "verified": True}


def _validate_config(config: dict[str, Any]) -> None:
    if (
        config.get("schema_version") != CONFIG_SCHEMA
        or config.get("policy")
        != "reuse_only_sealed_exact_action_derivatives_or_emit_checkpointable_contract"
        or not _hash_matches(config)
        or set(config.get("upstreams", {}))
        != {
            "K55_order_one_blocker",
            "P55_order_one_materializer",
            "flat_action_metric",
            "H_star_envelopes",
        }
        or set(config.get("formula_sources", {})) != {"symbol_builder", "action_pencil"}
        or config.get("target")
        != {
            "basis_jet_directions": BASIS_ATOMS,
            "basis_A_star_order_one_matrices": 4,
            "basis_B_star_order_one_axis_matrices": 12,
            "polarized_H_star_plus_order_one_packets": 15,
            "matrix_dimension": 22,
            "manifest_registered_before": 79,
            "manifest_registered_after_readiness": 79,
        }
        or config.get("caps", {}).get("maximum_symbol_builder_calls_per_C1_worker") != 1
        or config.get("caps", {}).get("maximum_output_rows_emitted") != 0
    ):
        raise HStarOrderOneReadinessError("invalid H-star readiness config")


def _audit_current_serialization(receipts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    blocker = receipts["K55_order_one_blocker"]
    contract = blocker.get("exact_K55_derivative_boundary", {}).get("missing_input_contract", {})
    if (
        blocker.get("status")
        != "block_K55_Taylor_order_one_missing_physical_H_star_order_one_packets"
        or contract.get("required_packets") != 15
        or contract.get("registered_packets") != 0
        or contract.get("shape_each") != [22, 22]
    ):
        raise HStarOrderOneReadinessError("K55 blocker boundary changed")
    p55 = receipts["P55_order_one_materializer"]
    packets = p55.get("packets", [])
    if (
        p55.get("status") != "pass_exact_15_P55_Taylor_order_one_packets_materialized"
        or len(packets) != 15
        or [packet.get("evaluation_id") for packet in packets] != EXPECTED_EVALUATIONS
        or any(
            set(packet) & {"A_star_order_one", "B_star_order_one", "H_star"} for packet in packets
        )
    ):
        raise HStarOrderOneReadinessError("P55 materializer boundary changed")
    atoms = sorted(
        {
            binding["basis_jet_direction"]
            for packet in packets
            for binding in packet.get("basis_bindings", [])
        }
    )
    if atoms != sorted(BASIS_ATOMS):
        raise HStarOrderOneReadinessError("basis atom set changed")
    flat = receipts["flat_action_metric"].get("exact_construction", {})
    if (
        flat.get("h_plus_0", {}).get("shape") != [22, 22]
        or flat.get("A_0", {}).get("shape") != [11, 11]
        or flat.get("B_0", {}).get("shape") != [11, 11]
    ):
        raise HStarOrderOneReadinessError("flat action metric boundary changed")
    envelopes = (
        receipts["H_star_envelopes"]
        .get("uniform_raw_mixed_derivative_envelopes", {})
        .get("H_star", {})
    )
    if (
        len(envelopes) != 15
        or any(set(record) != {"exact", "numeric"} for record in envelopes.values())
        or any("entries" in record or "shape" in record for record in envelopes.values())
    ):
        raise HStarOrderOneReadinessError("H-star envelope boundary changed")
    return {
        "exact_H_star_plus_order_one_packets_found": 0,
        "exact_A_star_order_one_matrix_packets_found": 0,
        "exact_B_star_order_one_axis_matrix_packets_found": 0,
        "P55_order_one_packets_inspected": len(packets),
        "flat_action_matrix_packets_inspected": 3,
        "scalar_H_star_envelope_records_rejected_as_coefficient_data": len(envelopes),
        "basis_jet_directions_recovered": atoms,
        "cold_symbol_build_used_in_audit": False,
    }


def build_campaign(project_root: Path, config_path: Path) -> dict[str, Any]:
    root = project_root.resolve()
    config = _load_json(config_path)
    _validate_config(config)
    receipts = {name: _load_bound(root, binding) for name, binding in config["upstreams"].items()}
    source_audits = {
        name: {
            **binding,
            **_validate_formula_source(root, binding),
        }
        for name, binding in config["formula_sources"].items()
    }
    audit = _audit_current_serialization(receipts)
    checkpoint_contract = {
        "schema_version": "sigma-H-star-plus-order-one-checkpoint-contract-1.0",
        "status": "READY_NOT_EXECUTED",
        "checkpoint_directory": "caller_owned_scratch",
        "phases": [
            {
                "phase": "C0",
                "units": 1,
                "action": "validate upstream and formula-source seals",
                "output": "c0-seals.json",
            },
            {
                "phase": "C1",
                "units": 4,
                "unit_key": "basis_jet_direction",
                "unit_ids": BASIS_ATOMS,
                "cold_dependency": "one _symbol_data call per worker invocation",
                "outputs_per_unit": {
                    "A_star_order_one_11x11_matrices": 1,
                    "B_star_order_one_axis_11x11_matrices": 3,
                },
                "durability": "seal each completed basis atom immediately",
            },
            {
                "phase": "C2",
                "units": 15,
                "unit_key": "polarization_evaluation",
                "cold_dependency": None,
                "output_each": "one exact sparse polynomial H_star_plus_1 22x22 packet",
            },
            {
                "phase": "C3",
                "units": 1,
                "action": "validate all packets and atomically seal portable result",
            },
        ],
        "exact_source_recipe": {
            "symbol_data": "data=_symbol_data()",
            "action_pencil": (
                "action=_first_order_generalized_pencil(data['action_symbol'],data['xi_lower'][0])"
            ),
            "basis_A_star_order_one": ("D_atom action['A'] at flat jets, alpha=1, m2=1, c20=0"),
            "basis_B_star_order_one_axis": (
                "D_atom D_nj action['B'] at n=0, flat jets, alpha=1, m2=1, c20=0"
            ),
            "polarized_metric": (
                "H_star_plus_1(n)=[[sum_j n_j B_star_1,j,A_star_1],[A_star_1,0_11]]"
            ),
            "negative_physical_metric": "H_star_minus_1=-H_star_plus_1",
        },
        "checkpoint_schemas": {
            "C1_required_fields": [
                "basis_jet_direction",
                "A_star_order_one_matrix",
                "three_B_star_order_one_axis_matrices",
                "flat_specialization",
                "source_content_sha256",
                "content_sha256",
            ],
            "C2_required_fields": [
                "evaluation_id",
                "evaluation_content_sha256",
                "basis_bindings",
                "H_star_plus_order_one_matrix",
                "symmetry_residual_nonzero_entries",
                "H_star_minus_negation_residual_nonzero_entries",
                "content_sha256",
            ],
        },
        "caps": config["caps"],
        "expected_counts_after_complete_run": {
            "basis_jet_packets": 4,
            "basis_A_star_order_one_matrices": 4,
            "basis_B_star_order_one_axis_matrices": 12,
            "polarized_H_star_plus_order_one_packets": 15,
        },
    }
    claims = {
        key: value
        for key, value in receipts["K55_order_one_blocker"]["claims"].items()
        if value is False
    }
    claims.update(
        {
            "H_star_plus_order_one_materializer_ready": True,
            "H_star_plus_order_one_packets_registered": False,
            "K55_Taylor_order_one_registered": False,
            "manifest_advanced_beyond_79": False,
            "cold_full_symbol_build_used": False,
        }
    )
    body = {
        "schema_version": SCHEMA,
        "status": STATUS,
        "decision": "READY_CHECKPOINTABLE_MATERIALIZER",
        "errors": [],
        "config_sha256": config["content_sha256"],
        "upstream_bindings": {
            name: {**binding, "verified": True} for name, binding in config["upstreams"].items()
        },
        "formula_source_bindings": source_audits,
        "current_serialization_audit": audit,
        "checkpointable_minimal_source_contract": checkpoint_contract,
        "counts": {
            "upstream_seals_verified": 4,
            "formula_source_functions_verified": sum(
                row["functions_verified"] for row in source_audits.values()
            ),
            "exact_H_star_plus_order_one_packets_found": 0,
            "inadmissible_scalar_H_star_envelope_records": 15,
            "required_basis_jet_packets": 4,
            "required_basis_matrix_packets": 16,
            "required_polarized_H_star_plus_order_one_packets": 15,
            "registered_polarized_H_star_plus_order_one_packets": 0,
            "registered_symbolic_input_packets": 79,
            "missing_symbolic_input_packets": 225,
            "full_symbol_build_calls": 0,
            "emitted_output_rows": 0,
        },
        "claims": claims,
        "negative_controls": {
            "treat_scalar_norm_envelope_as_matrix_coefficients": {"rejected": True},
            "recover_action_A_B_derivatives_from_P55_product": {"rejected": True},
            "infer_H_star_plus_order_one_as_zero": {"rejected": True},
            "count_readiness_schema_as_registered_packets": {"rejected": True},
            "advance_manifest_before_C1_C2_completion": {"rejected": True},
            "promote_readiness_to_D4_H7_or_PDE": {"rejected": True},
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
            "Audits sealed action, P55, and H-star envelope evidence without a cold "
            "build. No exact H-star order-one coefficient packet is present. The "
            "receipt closes a checkpointable 4-basis-atom to 15-polarization "
            "construction contract. It does not register H-star or K55 packets, emit "
            "recurrence rows, or prove D4, H7, PDE, lifespan, or candidate rejection."
        ),
    }
    return {**body, "content_sha256": _content_hash(body)}


def validate_campaign(document: dict[str, Any], project_root: Path) -> None:
    expected = build_campaign(project_root.resolve(), project_root.resolve() / CONFIG_PATH)
    if document != expected or not _hash_matches(document):
        raise HStarOrderOneReadinessError("campaign replay mismatch")


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
