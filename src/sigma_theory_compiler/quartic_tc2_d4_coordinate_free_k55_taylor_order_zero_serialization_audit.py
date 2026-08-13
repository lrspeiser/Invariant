from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA = "sigma-quartic-tc2-d4-coordinate-free-k55-taylor-order-zero-serialization-audit-1.0"
CONFIG_SCHEMA = (
    "sigma-quartic-tc2-d4-coordinate-free-k55-taylor-order-zero-serialization-audit-config-1.0"
)
STATUS = "block_K55_Taylor_order_zero_missing_exact_reference_action_metric"
PREDECESSOR_STATUS = "block_coordinate_free_D4_recurrence_emitter_missing_270_symbolic_packets"
CONFIG_PATH = (
    "configs/backgrounds/"
    "quartic_tc2_d4_coordinate_free_k55_taylor_order_zero_serialization_audit.json"
)
SOURCE_PATH = (
    "src/sigma_theory_compiler/"
    "quartic_tc2_d4_coordinate_free_k55_taylor_order_zero_serialization_audit.py"
)
TEST_PATH = "tests/test_quartic_tc2_d4_coordinate_free_k55_taylor_order_zero_serialization_audit.py"
REQUIRED_PACKETS = 304
REGISTERED_PACKETS = 34
MISSING_PACKETS = 270
REQUIRED_ROWS = 117_180
EXPECTED_UPSTREAMS = {
    "predecessor": {
        "path": "runs/physics-language/quartic-tc2-d4-coordinate-free-p55-taylor-order-zero-registration/campaign.json",
        "file_sha256": "e0814fa3fd8a046fec4cbd85c230faa652260dc934d4096f6e3c38b73381d091",
        "content_sha256": "71bbd9966e3212440ef92952eab7f06d860de3f59dfaa861753f03dd53850273",
    },
    "coordinate_jet_tube": {
        "path": "runs/physics-language/quartic-coordinate-jet-tube-campaign/campaign.json",
        "file_sha256": "a453b04a20734f477c0bbc0b02b206c156160f227be61c6c81df56b2d0c06861",
        "content_sha256": "f95ddf3262d9a5eb58d8425e296901ba940d63721a78fb5ac9f339336171b1ec",
    },
    "annular_K55_C6": {
        "path": "runs/physics-language/quartic-annular-k55-c6-campaign/campaign.json",
        "file_sha256": "bcc2b4184e5bcfb64d9a8a24ca095aa4067c18502c0c2f4956dcd8ad6f7fc527",
        "content_sha256": "55fa580fb91e37f48a8e6bd39c4c172c9aa4b3960336d42b88ceb211331b4e2f",
    },
}
EXPECTED_STATUSES = {
    "predecessor": PREDECESSOR_STATUS,
    "coordinate_jet_tube": "pass_all_12_uniform_coordinate_2jet_to_covariant_hyperbolicity_tubes",
    "annular_K55_C6": "pass_all_12_targeted_annular_K55_C6_principal_composition_constants",
}
EXPECTED_SOURCE = {
    "path": "src/sigma_theory_compiler/quartic_tc2_variable_sylvester_campaign.py",
    "file_sha256": "5df63ca3084654198c7ca23e8e7ba6e171aadfeff0ab6c5f1d2709b16f20937f",
    "required_function": "_reference_and_first_jet_packet",
}


class K55TaylorOrderZeroSerializationAuditError(ValueError):
    """Raised when the exact K55 serialization audit fails closed."""


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
        raise K55TaylorOrderZeroSerializationAuditError(f"expected JSON object: {path}")
    return value


def _resolve_under(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root != path and root not in path.parents:
        raise K55TaylorOrderZeroSerializationAuditError("bound path escaped project root")
    return path


def _validate_config(config: dict[str, Any]) -> None:
    if (
        config.get("schema_version") != CONFIG_SCHEMA
        or config.get("policy") != "construct_only_from_serialized_exact_matrices_else_typed_block"
        or not _hash_matches(config)
        or config.get("upstreams") != EXPECTED_UPSTREAMS
        or config.get("construction_source") != EXPECTED_SOURCE
        or config.get("target")
        != {
            "required_symbolic_input_packets": REQUIRED_PACKETS,
            "predecessor_registered_packets": REGISTERED_PACKETS,
            "K55_Taylor_packets_required": 75,
            "K55_order_zero_packets_requested": 15,
            "expected_registered_packets": REGISTERED_PACKETS,
            "expected_missing_packets": MISSING_PACKETS,
            "required_output_rows": REQUIRED_ROWS,
        }
        or config.get("resource_caps")
        != {
            "maximum_JSON_nodes_per_evidence_artifact": 250_000,
            "maximum_constructed_matrix_packets": 0,
            "maximum_output_rows_emitted": 0,
        }
    ):
        raise K55TaylorOrderZeroSerializationAuditError("invalid K55 audit config")


def _load_bound(root: Path, name: str, binding: dict[str, Any]) -> dict[str, Any]:
    path = _resolve_under(root, binding["path"])
    document = _load_json(path)
    if (
        _file_sha256(path) != binding["file_sha256"]
        or not _hash_matches(document)
        or document.get("content_sha256") != binding["content_sha256"]
        or document.get("status") != EXPECTED_STATUSES[name]
        or document.get("errors", []) != []
    ):
        raise K55TaylorOrderZeroSerializationAuditError(f"upstream mismatch: {name}")
    return document


def _validate_source(root: Path, binding: dict[str, Any]) -> dict[str, Any]:
    path = _resolve_under(root, binding["path"])
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    function = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == binding["required_function"]
        ),
        None,
    )
    markers = (
        "companion_projectors =",
        "h_plus0 =",
        "companion_energy0 =",
        "cross0 =",
        "energy0 = sp.zeros(STATE_DIMENSION)",
    )
    if (
        _file_sha256(path) != binding["file_sha256"]
        or function is None
        or any(marker not in text for marker in markers)
    ):
        raise K55TaylorOrderZeroSerializationAuditError("K55 construction source mismatch")
    return {
        "function_present": True,
        "construction_markers_verified": len(markers),
        "function_start_line": function.lineno,
        "function_end_line": function.end_lineno,
    }


def _serialization_audit(document: dict[str, Any], cap: int) -> dict[str, Any]:
    nodes = 0
    exact_55_records: list[str] = []
    exact_22_records: list[str] = []
    K55_named_keys: list[str] = []
    stack: list[tuple[str, Any]] = [("", document)]
    while stack:
        path, value = stack.pop()
        nodes += 1
        if nodes > cap:
            raise K55TaylorOrderZeroSerializationAuditError("JSON audit node cap exceeded")
        if isinstance(value, dict):
            if "k55" in path.lower():
                K55_named_keys.append(path)
            entries = value.get("entries")
            is_exact_entries = isinstance(entries, list) and all(
                isinstance(entry, dict) and {"row", "column", "value"} <= set(entry)
                for entry in entries
            )
            if value.get("shape") == [55, 55] and is_exact_entries:
                exact_55_records.append(path)
            if value.get("shape") == [22, 22] and is_exact_entries:
                exact_22_records.append(path)
            stack.extend((f"{path}.{key}" if path else key, child) for key, child in value.items())
        elif isinstance(value, list):
            stack.extend((f"{path}[{index}]", child) for index, child in enumerate(value))
    return {
        "JSON_nodes_audited": nodes,
        "K55_named_paths": len(K55_named_keys),
        "K55_named_path_examples": sorted(K55_named_keys)[:12],
        "exact_sparse_55x55_records": exact_55_records,
        "exact_sparse_22x22_records": exact_22_records,
        "constructible_K55_order_zero_packets": 0,
    }


def _minimal_contract() -> dict[str, Any]:
    return {
        "schema_version": "sigma-flat-reference-K55-order-zero-minimal-serialization-contract-1.0",
        "status": "MISSING_REQUIRED_SERIALIZATION",
        "smallest_sufficient_missing_primitive": {
            "packet_id": "flat_reference_action_metric_h_plus_0",
            "shape": [22, 22],
            "required_fields": [
                "ordered_companion_state_indices",
                "exact_sparse_rational_or_radical_entries",
                "symmetry_residual_zero",
                "matrix_content_sha256",
            ],
            "equivalent_source_blocks": (
                "exact flat action A_0 and B_0 11x11 matrices with h_plus_0=[[B_0,A_0],[A_0,0]]"
            ),
        },
        "already_available_exact_inputs": [
            "P_1, hence reference coupling C_0 and companion Q_0",
            "exact Lagrange projector polynomial recipes for the companion spectrum",
            "ordered 55-state convention",
        ],
        "deterministic_construction_after_registration": [
            "Pi_lambda=product_{mu!=lambda}(Q_0-mu*I)/(lambda-mu)",
            "G_0=sum Pi_lambda^T*g_lambda*Pi_lambda, with g_1=h_plus_0, g_-1=-h_plus_0, and g_lambda=I for lambda=+-1/2,+-1/3",
            "X_0=C_0^T*G_0*Q_0^{-1}",
            "K_0=[[I_33,X_0],[X_0^T,G_0]]",
        ],
        "required_replay_certificates": [
            "K_0=K_0^T entrywise",
            "K_0*P_1-P_1^T*K_0=0 entrywise",
            "15 Taylor-order-zero packets reference the same exact K_0 content seal",
        ],
        "why_bounds_do_not_suffice": (
            "coercivity constants, derivative envelopes, and anti-Wick composition constants "
            "do not determine any entry of h_plus_0 or K_0"
        ),
        "direct_alternative": "serialize one exact symmetric 55x55 K_0 packet",
    }


def build_campaign(project_root: Path, config_path: Path) -> dict[str, Any]:
    root = project_root.resolve()
    config = _load_json(config_path)
    _validate_config(config)
    documents = {
        name: _load_bound(root, name, binding) for name, binding in config["upstreams"].items()
    }
    source_audit = _validate_source(root, config["construction_source"])
    cap = config["resource_caps"]["maximum_JSON_nodes_per_evidence_artifact"]
    evidence_audits = {
        name: _serialization_audit(documents[name], cap)
        for name in ("coordinate_jet_tube", "annular_K55_C6")
    }
    if any(
        audit["exact_sparse_55x55_records"] or audit["exact_sparse_22x22_records"]
        for audit in evidence_audits.values()
    ):
        raise K55TaylorOrderZeroSerializationAuditError(
            "audit expectation changed: exact K55 primitive requires constructive review"
        )
    predecessor = documents["predecessor"]
    manifest = json.loads(json.dumps(predecessor["required_symbolic_input_manifest"]))
    records = {row["input_id"]: row for row in manifest}
    family = records.get("polarized_K55_Taylor_packets")
    if (
        len(records) != 8
        or family is None
        or family.get("required_packets") != 75
        or family.get("registered_packets") != 0
        or sum(row["registered_packets"] for row in manifest) != REGISTERED_PACKETS
    ):
        raise K55TaylorOrderZeroSerializationAuditError("predecessor manifest mismatch")
    claims = {key: False for key, value in predecessor["claims"].items() if value is False}
    claims.update(
        {
            "K55_Taylor_order_zero_packets_registered": False,
            "exact_K55_serialization_evidence_audited": True,
            "minimal_K55_order_zero_serialization_contract_closed": True,
            "manifest_recomputed_without_unjustified_advance": True,
        }
    )
    body = {
        "schema_version": SCHEMA,
        "status": STATUS,
        "errors": [],
        "config_sha256": config["content_sha256"],
        "upstream_bindings": {
            name: {**binding, "verified": True} for name, binding in config["upstreams"].items()
        },
        "construction_source_binding": {
            **config["construction_source"],
            **source_audit,
            "verified": True,
        },
        "serialized_K55_evidence_audit": evidence_audits,
        "minimal_missing_serialization_contract": _minimal_contract(),
        "required_symbolic_input_manifest": manifest,
        "remaining_missing_inputs": predecessor["remaining_missing_inputs"],
        "bounded_emitter_checkpoint": predecessor["bounded_emitter_checkpoint"],
        "phase_two": {
            "decision": "BLOCK",
            "admitted": False,
            "attempted": False,
            "blocker": "270 required symbolic input packets remain unregistered",
        },
        "counts": {
            "upstream_seals_verified": 3,
            "K55_evidence_artifacts_audited": 2,
            "K55_named_paths_audited": sum(
                audit["K55_named_paths"] for audit in evidence_audits.values()
            ),
            "exact_sparse_55x55_K55_packets_found": 0,
            "exact_sparse_22x22_action_metric_packets_found": 0,
            "new_K55_Taylor_order_zero_packets_registered": 0,
            "required_symbolic_input_packets": REQUIRED_PACKETS,
            "predecessor_registered_symbolic_input_packets": REGISTERED_PACKETS,
            "registered_symbolic_input_packets": REGISTERED_PACKETS,
            "missing_symbolic_input_packets": MISSING_PACKETS,
            "required_output_rows": REQUIRED_ROWS,
            "emitted_output_rows": 0,
            "phase_two_solve_attempts": 0,
        },
        "claims": claims,
        "negative_controls": {
            "substitute_coercivity_bounds_for_matrix_entries": {"rejected": True},
            "substitute_derivative_envelopes_for_reference_matrix": {"rejected": True},
            "infer_action_metric_from_P55_axis_packets": {"rejected": True},
            "assume_K55_order_zero_is_identity": {"rejected": True},
            "advance_manifest_without_exact_K55_packet": {"rejected": True},
            "promote_K55_bounds_to_full_D4_or_H7": {"rejected": True},
        },
        "source_bindings": {
            "config": {"path": CONFIG_PATH, "file_sha256": _file_sha256(config_path)},
            "source": {"path": SOURCE_PATH, "file_sha256": _file_sha256(root / SOURCE_PATH)},
            "test": {"path": TEST_PATH, "file_sha256": _file_sha256(root / TEST_PATH)},
        },
        "scope": (
            "Audits the strongest committed K55 tube and annular-C6 artifacts and records "
            "that they contain bounds, not an exact K55 reference matrix or flat action "
            "metric. The manifest remains 34/304. No recurrence row, D4 theorem, H7 closure, "
            "PDE theorem, lifespan, or candidate rejection follows."
        ),
    }
    return {**body, "content_sha256": _content_hash(body)}


def validate_campaign(document: dict[str, Any], project_root: Path) -> None:
    expected = build_campaign(project_root.resolve(), project_root.resolve() / CONFIG_PATH)
    if document != expected or not _hash_matches(document):
        raise K55TaylorOrderZeroSerializationAuditError("campaign replay mismatch")


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
