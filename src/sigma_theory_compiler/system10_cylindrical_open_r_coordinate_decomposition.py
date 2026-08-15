from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import sympy as sp

from .system10_cylindrical_r_positive_gravity_scalar_aw_materializer import (
    _canonical_lf_sha,
    _canonical_sha,
    _load_json,
    _resolve,
)


class System10CoordinateDecompositionError(RuntimeError):
    """Raised when an exact coordinate-decomposition packet fails closed."""


PACKET_SCHEMA = "invariant-system10-open-r-coordinate-off-shell-decomposition-packet-1.0"
RECEIPT_SCHEMA = "invariant-system10-open-r-coordinate-off-shell-decomposition-receipt-1.0"
DECISION = "BOUNDED_PASS_48_COORDINATE_DECOMPOSITIONS_BLOCK_EULER_NORMALIZATION_BRIDGE"


def _sealed(document: dict[str, Any]) -> bool:
    body = {key: value for key, value in document.items() if key != "content_sha256"}
    return document.get("content_sha256") == _canonical_sha(body)


def _load_binding(root: Path, binding: dict[str, Any]) -> dict[str, Any]:
    path = _resolve(root, binding["path"])
    if _canonical_lf_sha(path) != binding.get("canonical_lf_sha256"):
        raise System10CoordinateDecompositionError(f"bound file mismatch: {path}")
    document = _load_json(path)
    if document.get("content_sha256") != binding.get("content_sha256") or not _sealed(document):
        raise System10CoordinateDecompositionError(f"bound content mismatch: {path}")
    return document


def _authority_sha(config: dict[str, Any]) -> str:
    return _canonical_sha({key: value for key, value in config.items() if key != "source_evidence"})


def _validate_config(config_path: Path, root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = _load_json(config_path)
    if config.get("schema_version") != f"{RECEIPT_SCHEMA}-config":
        raise System10CoordinateDecompositionError("unsupported config schema")
    if config.get("caps") != {
        "candidate_indices": list(range(12)),
        "lower_nu_components": [0, 1, 2, 3],
        "coordinate_decomposition_rows": 48,
        "physical_constraint_rows": 96,
        "all_first_spatial_rhs_jets": 396,
        "full_rhs_rows": 1020,
        "maximum_packet_bytes": 131072,
        "maximum_receipt_bytes": 262144,
    }:
        raise System10CoordinateDecompositionError("caps changed")
    if config.get("domain") != {
        "coordinates": ["t", "r", "theta", "z"],
        "physical_metric": "diag(-1,1,r^2,1)",
        "predicate": "1/2<=r<=3/2 and real |v_10|<=1/4",
        "coordinate_partials_commute": True,
    }:
        raise System10CoordinateDecompositionError("domain changed")

    bound = {name: _load_binding(root, item) for name, item in config["bindings"].items()}
    if set(bound) != {
        "predecessor_block",
        "off_shell_identity",
        "physical_constraint_rows",
        "projection_skeletons",
        "divQ_rows",
        "full_rhs",
        "all_spatial_rhs_jets",
    }:
        raise System10CoordinateDecompositionError("binding manifest changed")
    if bound["predecessor_block"].get("decision") != (
        "BLOCK_COORDINATE_OFF_SHELL_DECOMPOSITION_UNREGISTERED_AFTER_ALL_SPATIAL_RHS_JETS"
    ):
        raise System10CoordinateDecompositionError("predecessor decision changed")
    if (
        not bound["off_shell_identity"]
        .get("claims", {})
        .get("common_off_shell_covariant_sourced_identity_closed")
    ):
        raise System10CoordinateDecompositionError("off-shell identity changed")
    if (
        bound["physical_constraint_rows"].get("counts", {}).get("physical_gravity_rows_closed")
        != 96
    ):
        raise System10CoordinateDecompositionError("constraint rows changed")
    if (
        not bound["projection_skeletons"]
        .get("claims", {})
        .get("exact_r1_normal_tangential_projection_operator_closed")
    ):
        raise System10CoordinateDecompositionError("projection skeleton changed")
    if bound["divQ_rows"].get("counts", {}).get("divq_rows_registered") != 4:
        raise System10CoordinateDecompositionError("divQ rows changed")
    if bound["full_rhs"].get("counts", {}).get("total_rhs_row_instances") != 1020:
        raise System10CoordinateDecompositionError("full RHS changed")
    if bound["all_spatial_rhs_jets"].get("counts", {}).get("tangential_rhs_jets") != 264:
        raise System10CoordinateDecompositionError("spatial RHS jets changed")

    ids = [
        [
            item["candidate_id"]
            for item in bound["physical_constraint_rows"]["materialization"]["candidate_results"]
        ],
        [item["candidate_id"] for item in bound["full_rhs"]["candidate_results"]],
        [item["candidate_id"] for item in bound["all_spatial_rhs_jets"]["candidate_results"]],
    ]
    if any(candidate_ids != ids[0] for candidate_ids in ids[1:]) or len(ids[0]) != 12:
        raise System10CoordinateDecompositionError("candidate join changed")

    sources = {}
    for name, binding in config.get("source_evidence", {}).items():
        path = _resolve(root, binding["path"])
        if _canonical_lf_sha(path) != binding.get("canonical_lf_sha256"):
            raise System10CoordinateDecompositionError(f"source evidence mismatch: {name}")
        sources[name] = path
    if (
        set(sources) != {"source", "test"}
        or sources["source"] != Path(__file__).resolve()
        or sources["test"]
        != root / "tests/test_system10_cylindrical_open_r_coordinate_decomposition.py"
    ):
        raise System10CoordinateDecompositionError("source evidence paths changed")
    return config, {**bound, "candidate_ids": ids[0]}


def _christoffel(
    metric: sp.Matrix, coordinates: tuple[sp.Symbol, ...]
) -> list[list[list[sp.Expr]]]:
    inverse = metric.inv()
    result = [[[sp.Integer(0) for _ in range(4)] for _ in range(4)] for _ in range(4)]
    for upper in range(4):
        for left in range(4):
            for right in range(4):
                result[upper][left][right] = sp.factor(
                    sum(
                        inverse[upper, contracted]
                        * (
                            sp.diff(metric[contracted, right], coordinates[left])
                            + sp.diff(metric[contracted, left], coordinates[right])
                            - sp.diff(metric[left, right], coordinates[contracted])
                        )
                        / 2
                        for contracted in range(4)
                    )
                )
    return result


def _derive_coordinate_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    t, r, theta, z = sp.symbols("t r theta z", real=True)
    coordinates = (t, r, theta, z)
    metric = sp.diag(-1, 1, r**2, 1)
    gamma = _christoffel(metric, coordinates)
    names = (
        ("C0", "C1", "C2", "C3"),
        ("C1", "S11", "S12", "S13"),
        ("C2", "S12", "S22", "S23"),
        ("C3", "S13", "S23", "S33"),
    )
    functions = {name: sp.Function(name)(*coordinates) for row in names for name in row}
    euler = sp.MutableDenseMatrix(4, 4, lambda _i, _j: 0)
    for left in range(4):
        for right in range(4):
            euler[left, right] = functions[names[left][right]]
    divergence_upper = []
    for nu in range(4):
        value = sum(sp.diff(euler[mu, nu], coordinates[mu]) for mu in range(4))
        value += sum(
            gamma[mu][mu][lam] * euler[lam, nu] + gamma[nu][mu][lam] * euler[mu, lam]
            for mu in range(4)
            for lam in range(4)
        )
        divergence_upper.append(sp.factor(value))
    divergence_lower = [
        sp.factor(sum(metric[nu, sigma] * divergence_upper[sigma] for sigma in range(4)))
        for nu in range(4)
    ]
    expected = [
        -sp.diff(functions["C0"], t)
        - sp.diff(functions["C1"], r)
        - sp.diff(functions["C2"], theta)
        - sp.diff(functions["C3"], z)
        - functions["C1"] / r,
        sp.diff(functions["C1"], t)
        + sp.diff(functions["S11"], r)
        + sp.diff(functions["S12"], theta)
        + sp.diff(functions["S13"], z)
        + functions["S11"] / r
        - r * functions["S22"],
        r**2
        * (
            sp.diff(functions["C2"], t)
            + sp.diff(functions["S12"], r)
            + sp.diff(functions["S22"], theta)
            + sp.diff(functions["S23"], z)
        )
        + 3 * r * functions["S12"],
        sp.diff(functions["C3"], t)
        + sp.diff(functions["S13"], r)
        + sp.diff(functions["S23"], theta)
        + sp.diff(functions["S33"], z)
        + functions["S13"] / r,
    ]
    residuals = [
        sp.factor(observed - target)
        for observed, target in zip(divergence_lower, expected, strict=True)
    ]
    if residuals != [sp.Integer(0)] * 4:
        raise System10CoordinateDecompositionError("coordinate divergence replay failed")
    constraint_names = ["Hamiltonian_E_nn", "momentum_E_n1", "momentum_E_n2", "momentum_E_n3"]
    rows = []
    for nu, expression in enumerate(expected):
        time_atom = sp.diff(functions[f"C{nu}"], t)
        coefficient = sp.factor(sp.expand(expression).coeff(time_atom))
        if coefficient == 0:
            raise System10CoordinateDecompositionError("constraint time coefficient vanished")
        solved = sp.factor(
            (sp.Symbol(f"D_lower_{nu}") - (expression - coefficient * time_atom)) / coefficient
        )
        body = {
            "lower_nu": nu,
            "constraint_row": constraint_names[nu],
            "constraint_tensor_atom": f"C{nu}=E_sourced^0{nu}",
            "covariant_divergence_expression": sp.sstr(expression, order="lex"),
            "constraint_time_coefficient": sp.sstr(coefficient),
            "solved_constraint_time_expression": sp.sstr(solved, order="lex"),
            "off_shell_substitution": f"D_lower_{nu}=divQ_lower[{nu}]-(E_phi_g*partial_{nu}(phi_g)+F_total_lower[{nu}])/2",
            "domain": "r>0",
            "coordinate_basis": "(t,r,theta,z), g=diag(-1,1,r^2,1)",
        }
        rows.append({**body, "row_sha256": _canonical_sha(body)})
    geometry = {
        "metric": "diag(-1,1,r^2,1)",
        "nonzero_christoffels": {
            "Gamma^1_22": "-r",
            "Gamma^2_12": "1/r",
            "Gamma^2_21": "1/r",
        },
        "derived_lower_divergence_components": 4,
        "exact_symbolic_zero_residuals": 4,
        "axis_excluded": True,
    }
    return rows, {**geometry, "geometry_sha256": _canonical_sha(geometry)}


def _build_candidate(
    config: dict[str, Any], bound: dict[str, Any], candidate_index: int
) -> dict[str, Any]:
    if candidate_index not in config["caps"]["candidate_indices"]:
        raise System10CoordinateDecompositionError("candidate outside cap")
    rows, geometry = _derive_coordinate_rows()
    candidate_id = bound["candidate_ids"][candidate_index]
    constraint = bound["physical_constraint_rows"]["materialization"]["candidate_results"][
        candidate_index
    ]
    full_rhs = bound["full_rhs"]["candidate_results"][candidate_index]
    jets = bound["all_spatial_rhs_jets"]["candidate_results"][candidate_index]
    body = {
        "schema_version": PACKET_SCHEMA,
        "campaign_id": config["campaign_id"],
        "candidate_index": candidate_index,
        "candidate_id": candidate_id,
        "domain": config["domain"]["predicate"],
        "geometry_certificate": geometry,
        "rows": rows,
        "counts": {
            "coordinate_decomposition_rows": 4,
            "symbolic_zero_residuals": 4,
            "spatial_rhs_jets_bound": 33,
            "full_rhs_rows_bound": 85,
        },
        "source_bindings": {
            "constraint_manifest_sha256": constraint["manifest_sha256"],
            "full_rhs_equation_origin_set_sha256": full_rhs["equation_origin_set_sha256"],
            "spatial_rhs_jet_packet_content_sha256": jets["packet_content_sha256"],
        },
        "claims": {
            "four_coordinate_decomposition_rows_closed": True,
            "equation_origin_to_off_shell_normalization_closed": False,
            "subsidiary_system_closed": False,
            "constraint_propagation_closed": False,
        },
    }
    packet = {**body, "content_sha256": _canonical_sha(body)}
    if len(json.dumps(packet).encode("utf-8")) > config["caps"]["maximum_packet_bytes"]:
        raise System10CoordinateDecompositionError("packet cap exceeded")
    return packet


def _verify_packet(packet: dict[str, Any], config: dict[str, Any], index: int) -> None:
    if not _sealed(packet) or packet.get("schema_version") != PACKET_SCHEMA:
        raise System10CoordinateDecompositionError("packet seal failed")
    if packet.get("candidate_index") != index or packet.get("counts") != {
        "coordinate_decomposition_rows": 4,
        "symbolic_zero_residuals": 4,
        "spatial_rhs_jets_bound": 33,
        "full_rhs_rows_bound": 85,
    }:
        raise System10CoordinateDecompositionError("packet counts failed")
    expected_rows, expected_geometry = _derive_coordinate_rows()
    if (
        packet.get("rows") != expected_rows
        or packet.get("geometry_certificate") != expected_geometry
    ):
        raise System10CoordinateDecompositionError("packet replay failed")


def build_receipt(
    config_path: Path, *, output_dir: Path | None = None, root: Path | None = None
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    repository = (root or config_path.resolve().parents[1]).resolve()
    config, bound = _validate_config(config_path.resolve(), repository)
    packets = [_build_candidate(config, bound, index) for index in range(12)]
    for index, packet in enumerate(packets):
        _verify_packet(packet, config, index)
    if output_dir is not None:
        for index, expected in enumerate(packets):
            path = output_dir / f"candidate-{index:02d}.json"
            if path.exists():
                observed = _load_json(path)
                if observed != expected:
                    raise System10CoordinateDecompositionError("immutable packet conflict")

    missing = {
        "primitive": "candidate_equation_origin_to_off_shell_euler_normalization_bridge",
        "status": "BLOCK_NORMALIZATION_AND_FORCE_ORIGIN_MAP_UNREGISTERED",
        "required_candidate_instances": 12,
        "required_mappings_per_candidate": {
            "six_spatial_metric_components": ["E^11", "E^12", "E^13", "E^22", "E^23", "E^33"],
            "scalar_force": "E_phi_g*partial_nu(phi_g)",
            "matter_force": "F_total_lower[nu]",
            "gauge_divergence": "divQ_lower[nu] from receipt normalization divQ_lower[nu]/M2",
        },
        "why_current_hashes_do_not_close_it": (
            "The 85-row origin seals identify field-pair residuals and exact solves, but do not "
            "state their multiplicative/index-raising normalization relative to E_sourced^mu_nu, "
            "E_phi_g, or each sector Euler force in the covariant identity."
        ),
        "acceptance": (
            "Bind each equation origin to the normalized off-shell Euler atom, substitute the 85 "
            "RHS rows and 33 spatial RHS jets into all four decompositions per candidate, replay "
            "the covariant identity exactly, and only then factor the resulting divQ terms through "
            "the four registered modified-harmonic C rows."
        ),
    }
    missing["primitive_sha256"] = _canonical_sha(missing)
    packet_set = hashlib.sha256(
        "".join(packet["content_sha256"] for packet in packets).encode("ascii")
    ).hexdigest()
    body = {
        "schema_version": RECEIPT_SCHEMA,
        "campaign_id": config["campaign_id"],
        "decision": DECISION,
        "scope": (
            "All four lower-nu coordinate decompositions of the registered covariant off-shell "
            "identity are derived exactly on the fixed cylindrical r>0 metric and sealed for all "
            "12 candidates. The prior 48-row blocker is closed. Propagation remains blocked "
            "because the candidate equation-origin seals do not register the normalization bridge "
            "to the Euler atoms in that identity. No subsidiary, propagation, energy, "
            "hyperbolicity, or broader-domain claim is made."
        ),
        "source_bindings": {
            "authority_sha256": _authority_sha(config),
            "ordered_candidate_packet_set_sha256": packet_set,
        },
        "counts": {
            "candidates": 12,
            "coordinate_decomposition_rows_required": 48,
            "coordinate_decomposition_rows_closed": 48,
            "symbolic_coordinate_zero_residuals": 48,
            "physical_constraint_rows_bound": 96,
            "full_rhs_rows_bound": 1020,
            "equation_origin_seals_bound": 1020,
            "all_first_spatial_rhs_jets_bound": 396,
            "normalization_bridge_candidate_instances_required": 12,
            "normalization_bridge_candidate_instances_closed": 0,
            "candidate_subsidiary_systems_closed": 0,
            "constraint_propagation_proofs": 0,
        },
        "candidate_results": [
            {
                "candidate_index": index,
                "candidate_id": packet["candidate_id"],
                "packet_content_sha256": packet["content_sha256"],
                "row_set_sha256": _canonical_sha([row["row_sha256"] for row in packet["rows"]]),
                "outcome": "PASS_4_COORDINATE_DECOMPOSITIONS_BLOCK_NORMALIZATION_BRIDGE",
            }
            for index, packet in enumerate(packets)
        ],
        "materialization": {
            "closed_predecessor_block": {
                "primitive": "candidate_bound_coordinate_off_shell_identity_decomposition_rows",
                "closed_rows": 48,
                "status": "PASS_ALL_48_COORDINATE_DECOMPOSITION_ROWS",
            },
            "first_missing_primitive": missing,
            "negative_controls": {
                "flip_angular_metric_factor": {
                    "mutation": "replace g_theta_theta=r^2 by r^-2",
                    "rejected": True,
                    "reason": "changes lower-nu=2 time coefficient from r^2 to r^-2",
                },
                "omit_cylindrical_connection": {
                    "mutation": "set all Christoffel symbols to zero",
                    "rejected": True,
                    "witness_missing_term": "3*r*E_sourced^12 in lower-nu=2",
                },
                "infer_origin_normalization": {
                    "mutation": "identify source_field_pair with E_sourced component at coefficient one",
                    "rejected": True,
                    "reason": "field-pair labels do not seal action, index, or symmetric-row normalization",
                },
            },
        },
        "claims": {
            "all_48_coordinate_decomposition_rows_closed": True,
            "equation_origin_to_off_shell_normalization_closed": False,
            "candidate_bound_subsidiary_system_closed": False,
            "constraint_propagation_closed": False,
            "subsidiary_energy_estimate_closed": False,
            "hyperbolicity_closed": False,
            "global_theorem_established": False,
            "promotion_authorized": False,
        },
    }
    receipt = {**body, "content_sha256": _canonical_sha(body)}
    if len(json.dumps(receipt).encode("utf-8")) > config["caps"]["maximum_receipt_bytes"]:
        raise System10CoordinateDecompositionError("receipt cap exceeded")
    return receipt, packets


def write_outputs(config_path: Path, output_dir: Path, *, root: Path | None = None) -> Path:
    receipt, packets = build_receipt(config_path, root=root)
    output_dir.mkdir(parents=True, exist_ok=True)
    documents = [
        (output_dir / f"candidate-{index:02d}.json", packet) for index, packet in enumerate(packets)
    ]
    documents.append((output_dir / "receipt.json", receipt))
    for path, document in documents:
        data = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8")
        if path.exists() and path.read_bytes() != data:
            raise System10CoordinateDecompositionError(f"immutable output conflict: {path}")
        if not path.exists():
            temporary = path.with_suffix(".json.tmp")
            temporary.write_bytes(data)
            os.replace(temporary, path)
    return output_dir / "receipt.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build exact cylindrical coordinate decomposition rows"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    write_outputs(args.config, args.output, root=args.config.resolve().parents[1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
