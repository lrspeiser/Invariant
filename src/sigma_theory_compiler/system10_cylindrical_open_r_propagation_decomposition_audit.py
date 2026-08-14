from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .system10_cylindrical_open_r_tangential_rhs_jets import (
    _validate_config as _validate_tangential_config,
)
from .system10_cylindrical_open_r_tangential_rhs_jets import _verify_packet
from .system10_cylindrical_r_positive_gravity_scalar_aw_materializer import (
    _canonical_lf_sha,
    _canonical_sha,
    _load_json,
    _resolve,
)


class System10PropagationDecompositionAuditError(RuntimeError):
    """Raised when a sealed propagation-audit authority changes."""


SCHEMA = "invariant-system10-open-r-propagation-decomposition-audit-1.0"
DECISION = "BLOCK_COORDINATE_OFF_SHELL_DECOMPOSITION_UNREGISTERED_AFTER_ALL_SPATIAL_RHS_JETS"


def _sealed(document: dict[str, Any]) -> bool:
    body = {key: value for key, value in document.items() if key != "content_sha256"}
    return document.get("content_sha256") == _canonical_sha(body)


def _load_binding(root: Path, binding: dict[str, Any]) -> dict[str, Any]:
    path = _resolve(root, binding["path"])
    if _canonical_lf_sha(path) != binding.get("canonical_lf_sha256"):
        raise System10PropagationDecompositionAuditError(f"bound file mismatch: {path}")
    document = _load_json(path)
    if document.get("content_sha256") != binding.get("content_sha256") or not _sealed(document):
        raise System10PropagationDecompositionAuditError(f"bound content mismatch: {path}")
    return document


def _authority_sha(config: dict[str, Any]) -> str:
    return _canonical_sha({key: value for key, value in config.items() if key != "source_evidence"})


def _validate_config(config_path: Path, root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = _load_json(config_path)
    if config.get("schema_version") != f"{SCHEMA}-config":
        raise System10PropagationDecompositionAuditError("unsupported config schema")
    if config.get("caps") != {
        "candidates": 12,
        "state_dimension": 85,
        "physical_gravity_rows": 96,
        "divQ_rows": 4,
        "full_rhs_rows": 1020,
        "radial_rhs_jets": 132,
        "tangential_rhs_jets": 264,
        "all_first_spatial_rhs_jets": 396,
        "required_coordinate_decomposition_rows": 48,
        "maximum_output_bytes": 262144,
    }:
        raise System10PropagationDecompositionAuditError("caps changed")

    tangential = config["bindings"]["tangential_rhs_jets"]
    tangential_config_path = _resolve(root, tangential["config_path"])
    if (
        _canonical_lf_sha(tangential_config_path) != tangential["config_canonical_lf_sha256"]
        or _canonical_sha(_load_json(tangential_config_path)) != tangential["config_content_sha256"]
    ):
        raise System10PropagationDecompositionAuditError("tangential config mismatch")
    tangential_config, _ = _validate_tangential_config(tangential_config_path, root)
    tangential_receipt = _load_binding(root, tangential["receipt"])
    if (
        tangential_receipt.get("decision") != "BOUNDED_PASS_ALL_TWELVE_264_TANGENTIAL_RHS_JETS"
        or tangential_receipt.get("counts", {}).get("tangential_rhs_jets") != 264
        or tangential_receipt.get("counts", {}).get("candidate_passes") != 12
    ):
        raise System10PropagationDecompositionAuditError("tangential receipt mismatch")
    packet_dir = _resolve(root, tangential["packet_dir"])
    seals = []
    for index in range(12):
        packet = _load_json(packet_dir / f"candidate-{index:02d}.json")
        _verify_packet(packet, tangential_config, index)
        expected = tangential_receipt["candidate_results"][index]["packet_content_sha256"]
        if packet["content_sha256"] != expected:
            raise System10PropagationDecompositionAuditError("tangential packet mismatch")
        seals.append(packet["content_sha256"])
    ordered_set = hashlib.sha256("".join(seals).encode("ascii")).hexdigest()
    if ordered_set != tangential_receipt["source_bindings"]["ordered_candidate_packet_set_sha256"]:
        raise System10PropagationDecompositionAuditError("tangential packet set mismatch")

    full_rhs = _load_binding(root, config["bindings"]["full_rhs"])
    constraints = _load_binding(root, config["bindings"]["physical_constraint_rows"])
    divq = _load_binding(root, config["bindings"]["divQ_rows"])
    identity = _load_binding(root, config["bindings"]["off_shell_identity"])
    predecessor = _load_binding(root, config["bindings"]["tangential_jet_blocker"])
    if (
        full_rhs.get("counts", {}).get("total_rhs_row_instances") != 1020
        or full_rhs.get("counts", {}).get("equation_origin_seals") != 1020
    ):
        raise System10PropagationDecompositionAuditError("full RHS authority changed")
    if constraints.get("counts", {}).get("physical_gravity_rows_closed") != 96:
        raise System10PropagationDecompositionAuditError("constraint authority changed")
    if divq.get("counts", {}).get("divq_rows_registered") != 4:
        raise System10PropagationDecompositionAuditError("divQ authority changed")
    if not identity.get("claims", {}).get("common_off_shell_covariant_sourced_identity_closed"):
        raise System10PropagationDecompositionAuditError("off-shell identity changed")
    if (
        predecessor.get("decision")
        != "BLOCK_TANGENTIAL_RHS_JETS_UNREGISTERED_AFTER_RADIAL_JET_CLOSURE"
    ):
        raise System10PropagationDecompositionAuditError("predecessor blocker changed")

    sources = {}
    for name, binding in config.get("source_evidence", {}).items():
        path = _resolve(root, binding["path"])
        if _canonical_lf_sha(path) != binding.get("canonical_lf_sha256"):
            raise System10PropagationDecompositionAuditError(f"source evidence mismatch: {name}")
        sources[name] = path
    if (
        set(sources) != {"source", "test"}
        or sources["source"] != Path(__file__).resolve()
        or sources["test"]
        != root / "tests/test_system10_cylindrical_open_r_propagation_decomposition_audit.py"
    ):
        raise System10PropagationDecompositionAuditError("source evidence paths changed")
    return config, {
        "tangential": tangential_receipt,
        "full_rhs": full_rhs,
        "identity": identity,
    }


def build_receipt(config_path: Path, *, root: Path | None = None) -> dict[str, Any]:
    repository = (root or config_path.resolve().parents[1]).resolve()
    config, bound = _validate_config(config_path.resolve(), repository)
    identity_formula = bound["identity"]["materialization"]["common_formula"][
        "maximal_common_off_shell_identity"
    ]
    witness = {
        "covariant_identity": identity_formula,
        "available_coordinate_rows": {
            "constraint_rows": [
                "Hamiltonian_E_nn",
                "momentum_E_n1",
                "momentum_E_n2",
                "momentum_E_n3",
            ],
            "divQ_rows": ["divQ_lower[0]", "divQ_lower[1]", "divQ_lower[2]", "divQ_lower[3]"],
            "rhs_rows_per_candidate": 85,
            "first_spatial_rhs_jets_per_candidate": 33,
        },
        "missing_link": (
            "No sealed row expands each lower-nu covariant divergence into partial_0 of the "
            "Hamiltonian/momentum projections, spatial derivatives of spatial metric Euler "
            "components, connection lower-order terms, matter Euler-force terms, and divQ_lower[nu]."
        ),
        "why_hash_matching_is_not_a_derivation": (
            "Equation-origin and row seals identify inputs but do not supply projection signs, "
            "normalization coefficients, index lowering, or connection terms."
        ),
    }
    witness["witness_sha256"] = _canonical_sha(witness)
    required_rows = [
        {
            "candidate_index": candidate,
            "lower_nu": nu,
            "required_output": f"coordinate_off_shell_decomposition_candidate_{candidate:02d}_nu_{nu}",
        }
        for candidate in range(12)
        for nu in range(4)
    ]
    missing = {
        "primitive": "candidate_bound_coordinate_off_shell_identity_decomposition_rows",
        "status": "BLOCK_48_COORDINATE_DECOMPOSITION_ROWS_UNREGISTERED",
        "required_rows": required_rows,
        "required_row_count": 48,
        "required_terms": [
            "partial_0 Hamiltonian_or_momentum_projection",
            "partial_i spatial_metric_Euler_components",
            "connection_times_metric_Euler_components",
            "E_phi_g times partial_nu_phi_g",
            "F_total_lower_nu",
            "minus_2_divQ_lower_nu",
        ],
        "acceptance": (
            "For every candidate and lower_nu=0..3, expand the registered covariant identity in "
            "the fixed cylindrical coordinate basis; substitute all 85 equation-origin rows and "
            "all 33 first spatial F jets; replay an exact zero residual; only then solve for a "
            "closed subsidiary row."
        ),
        "witness_sha256": witness["witness_sha256"],
    }
    missing["primitive_sha256"] = _canonical_sha(missing)
    body = {
        "schema_version": SCHEMA,
        "campaign_id": config["campaign_id"],
        "decision": DECISION,
        "scope": (
            "Exact propagation re-audit after closing all 396 first spatial derivatives of the "
            "11 candidate-bound dynamic RHS rows. The time derivative of every registered "
            "constraint row is now chain-rule formable, but the covariant off-shell identity has "
            "not been decomposed into the registered coordinate projections and equation origins. "
            "No subsidiary, propagation, energy, or hyperbolicity claim is made."
        ),
        "source_bindings": {"audit_authority_sha256": _authority_sha(config)},
        "counts": {
            "candidates": 12,
            "physical_gravity_rows_bound": 96,
            "divQ_rows_bound": 4,
            "full_rhs_rows_bound": 1020,
            "equation_origin_seals_bound": 1020,
            "radial_rhs_jets_bound": 132,
            "tangential_rhs_jets_bound": 264,
            "all_first_spatial_rhs_jets_bound": 396,
            "coordinate_decomposition_rows_required": 48,
            "coordinate_decomposition_rows_bound": 0,
            "candidate_subsidiary_systems_closed": 0,
            "constraint_propagation_proofs": 0,
        },
        "materialization": {
            "closed_predecessor_block": {
                "status": "PASS_ALL_TWELVE_TANGENTIAL_RHS_JETS",
                "closed_candidate_instances": 264,
                "source_receipt_content_sha256": bound["tangential"]["content_sha256"],
            },
            "coordinate_decomposition_witness": witness,
            "first_missing_primitive": missing,
            "negative_controls": {
                "infer_decomposition_from_origin_hashes": {
                    "mutation": "treat equation-origin hashes as coordinate projection coefficients",
                    "rejected": True,
                    "reason": "hash equality contains no tensor normalization or connection data",
                },
                "flip_normal_projection_sign": {
                    "mutation": "reverse the normal projection sign without a coordinate replay",
                    "rejected": True,
                    "reason": "changes partial_0 constraint coefficient and the subsidiary row",
                },
                "omit_spatial_euler_divergence": {
                    "mutation": "drop partial_i E_sourced_i_nu before applying evolution equations",
                    "rejected": True,
                    "reason": "the covariant divergence contains independently registered spatial components",
                },
            },
        },
        "claims": {
            "all_first_spatial_rhs_jets_closed": True,
            "coordinate_off_shell_decomposition_closed": False,
            "candidate_bound_subsidiary_system_closed": False,
            "constraint_propagation_closed": False,
            "subsidiary_energy_estimate_closed": False,
            "hyperbolicity_closed": False,
            "global_theorem_established": False,
            "promotion_authorized": False,
        },
    }
    receipt = {**body, "content_sha256": _canonical_sha(body)}
    if len(json.dumps(receipt).encode("utf-8")) > config["caps"]["maximum_output_bytes"]:
        raise System10PropagationDecompositionAuditError("receipt cap exceeded")
    return receipt


def write_receipt(config_path: Path, output_path: Path, *, root: Path | None = None) -> Path:
    receipt = build_receipt(config_path, root=root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output_path)
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit propagation after all spatial RHS jets")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    write_receipt(args.config, args.output, root=args.config.resolve().parents[1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
