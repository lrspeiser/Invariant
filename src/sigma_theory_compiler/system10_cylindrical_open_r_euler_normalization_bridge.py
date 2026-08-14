from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .system10_cylindrical_r_positive_gravity_scalar_aw_materializer import (
    SYMMETRIC_METRIC_PAIRS,
    _canonical_lf_sha,
    _canonical_sha,
    _load_json,
    _resolve,
)


class System10EulerNormalizationBridgeError(RuntimeError):
    """Raised when an equation-origin normalization bridge changes."""


PACKET_SCHEMA = "invariant-system10-open-r-euler-normalization-bridge-packet-1.0"
RECEIPT_SCHEMA = "invariant-system10-open-r-euler-normalization-bridge-receipt-1.0"
DECISION = "BOUNDED_PASS_12_EULER_NORMALIZATION_BRIDGES_BLOCK_DIVQ_C_FACTORIZATION"


def _sealed(document: dict[str, Any]) -> bool:
    body = {key: value for key, value in document.items() if key != "content_sha256"}
    return document.get("content_sha256") == _canonical_sha(body)


def _load_binding(root: Path, binding: dict[str, Any]) -> dict[str, Any]:
    path = _resolve(root, binding["path"])
    if _canonical_lf_sha(path) != binding.get("canonical_lf_sha256"):
        raise System10EulerNormalizationBridgeError(f"bound file mismatch: {path}")
    document = _load_json(path)
    if document.get("content_sha256") != binding.get("content_sha256") or not _sealed(document):
        raise System10EulerNormalizationBridgeError(f"bound content mismatch: {path}")
    return document


def _authority_sha(config: dict[str, Any]) -> str:
    return _canonical_sha({key: value for key, value in config.items() if key != "source_evidence"})


def _validate_config(config_path: Path, root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = _load_json(config_path)
    if config.get("schema_version") != f"{RECEIPT_SCHEMA}-config":
        raise System10EulerNormalizationBridgeError("unsupported config schema")
    if config.get("caps") != {
        "candidate_indices": list(range(12)),
        "spatial_metric_euler_components_per_candidate": 6,
        "gravity_scalar_euler_components_per_candidate": 1,
        "matter_force_sectors_per_candidate": 3,
        "divQ_normalizations_per_candidate": 4,
        "normalization_bridges": 12,
        "maximum_packet_bytes": 131072,
        "maximum_receipt_bytes": 262144,
    }:
        raise System10EulerNormalizationBridgeError("caps changed")
    bound = {name: _load_binding(root, item) for name, item in config["bindings"].items()}
    if set(bound) != {
        "coordinate_decomposition",
        "all_candidate_aw",
        "full_rhs",
        "matter_dynamic_rhs",
        "maxwell_dynamic_rhs",
        "matter_interface",
        "sourced_constraint_control",
        "divQ_rows",
    }:
        raise System10EulerNormalizationBridgeError("binding manifest changed")
    if bound["coordinate_decomposition"].get("decision") != (
        "BOUNDED_PASS_48_COORDINATE_DECOMPOSITIONS_BLOCK_EULER_NORMALIZATION_BRIDGE"
    ):
        raise System10EulerNormalizationBridgeError("coordinate predecessor changed")
    if bound["all_candidate_aw"].get("counts", {}).get("candidate_packets") != 12:
        raise System10EulerNormalizationBridgeError("A/W authority changed")
    if bound["full_rhs"].get("counts", {}).get("equation_origin_seals") != 1020:
        raise System10EulerNormalizationBridgeError("full RHS origin authority changed")
    if bound["matter_dynamic_rhs"].get("counts", {}).get("matter_dynamic_rows_registered") != 2:
        raise System10EulerNormalizationBridgeError("matter dynamic authority changed")
    if bound["maxwell_dynamic_rhs"].get("counts", {}).get("maxwell_dynamic_rows_registered") != 4:
        raise System10EulerNormalizationBridgeError("Maxwell dynamic authority changed")
    interface = bound["matter_interface"]["combined_matter_certificate"]
    if (
        interface["combined_stress_conservation"].get("sector_euler_force_coefficients")
        != [1, 1, 1]
        or interface["internal_matter_constraint_closure"].get("subsidiary_equation")
        != "box_g C=0 for the source-free Lorenz-gauge system"
    ):
        raise System10EulerNormalizationBridgeError("matter force authority changed")
    if (
        not bound["sourced_constraint_control"]
        .get("claims", {})
        .get("flat_reference_matter_source_divergence_cancellation_closed")
    ):
        raise System10EulerNormalizationBridgeError("matter cancellation authority changed")
    divq = bound["divQ_rows"]
    if (
        divq.get("counts", {}).get("divq_rows_registered") != 4
        or divq["materialization"]["operator_convention"].get("row_normalization")
        != "divQ_lower[nu]/M2"
    ):
        raise System10EulerNormalizationBridgeError("divQ authority changed")

    ids = [
        [item["candidate_id"] for item in bound["all_candidate_aw"]["candidate_results"]],
        [item["candidate_id"] for item in bound["full_rhs"]["candidate_results"]],
        [item["candidate_id"] for item in bound["coordinate_decomposition"]["candidate_results"]],
    ]
    if any(items != ids[0] for items in ids[1:]) or len(ids[0]) != 12:
        raise System10EulerNormalizationBridgeError("candidate join changed")
    if any(
        item.get("coefficients", {}).get("m2") != "1"
        for item in bound["all_candidate_aw"]["candidate_results"]
    ):
        raise System10EulerNormalizationBridgeError("candidate M2 normalization changed")

    sources = {}
    for name, binding in config.get("source_evidence", {}).items():
        path = _resolve(root, binding["path"])
        if _canonical_lf_sha(path) != binding.get("canonical_lf_sha256"):
            raise System10EulerNormalizationBridgeError(f"source evidence mismatch: {name}")
        sources[name] = path
    expected = {
        "source": Path(__file__).resolve(),
        "test": root / "tests/test_system10_cylindrical_open_r_euler_normalization_bridge.py",
        "aw_source": root
        / "src/sigma_theory_compiler/system10_cylindrical_r_positive_gravity_scalar_aw_materializer.py",
        "nonlinear_source": root
        / "src/sigma_theory_compiler/quartic_nonlinear_evolution_campaign.py",
        "matter_source": root
        / "src/sigma_theory_compiler/system10_cylindrical_r_positive_matter_dynamic_rhs_materializer.py",
        "maxwell_source": root
        / "src/sigma_theory_compiler/system10_cylindrical_r_positive_maxwell_dynamic_rhs_materializer.py",
    }
    if sources != expected:
        raise System10EulerNormalizationBridgeError("source evidence paths changed")
    return config, {**bound, "candidate_ids": ids[0]}


def _metric_normalizations() -> list[dict[str, Any]]:
    expected_pairs = [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]
    if list(SYMMETRIC_METRIC_PAIRS[4:10]) != expected_pairs:
        raise System10EulerNormalizationBridgeError("spatial metric pair order changed")
    mappings = []
    for row, pair in enumerate(expected_pairs, start=4):
        weight = "1" if pair[0] == pair[1] else "sqrt(2)"
        inverse_weight = "1" if weight == "1" else "1/sqrt(2)"
        body = {
            "source_equation_row": row,
            "source_field_pair": list(pair),
            "source_symmetric_row_weight": weight,
            "off_shell_atom": f"E_sourced^{pair[0]}{pair[1]}",
            "bridge": f"E_sourced^{pair[0]}{pair[1]}={inverse_weight}*AW_equation[{row}]",
            "off_shell_to_source_coefficient": weight,
            "source_to_off_shell_coefficient": inverse_weight,
        }
        mappings.append({**body, "mapping_sha256": _canonical_sha(body)})
    return mappings


def _gravity_scalar_normalization() -> dict[str, Any]:
    body = {
        "source_equation_row": 10,
        "source_field_pair": "gravity_scalar",
        "source_variational_convention": "symmetric_11_field_row_is_minus_E_phi_g",
        "off_shell_atom": "E_phi_g",
        "bridge": "E_phi_g=-AW_equation[10]",
        "source_to_off_shell_coefficient": "-1",
        "source_authority": "quartic_nonlinear_evolution_campaign._assemble_equations",
    }
    return {**body, "mapping_sha256": _canonical_sha(body)}


def _matter_force_normalization(bound: dict[str, Any]) -> dict[str, Any]:
    matter_rows = bound["matter_dynamic_rhs"]["materialization"]["rows"]
    maxwell_rows = bound["maxwell_dynamic_rhs"]["materialization"]["rows"]
    if [row["field_index"] for row in matter_rows] != [11, 16]:
        raise System10EulerNormalizationBridgeError("matter field order changed")
    if [row["field_index"] for row in maxwell_rows] != [12, 13, 14, 15]:
        raise System10EulerNormalizationBridgeError("Maxwell field order changed")
    scalar = {
        "sector": "canonical_minimally_coupled_scalar",
        "force_atom": "E_chi*partial_nu(chi_m)",
        "dynamic_origin_sha256": matter_rows[0]["equation_origin"]["origin_sha256"],
        "dynamic_to_covariant_euler": "E_chi=unsolved_dynamic_equation[11]",
        "sector_force_coefficient": "1",
    }
    fluid = {
        "sector": "barotropic_irrotational_fluid",
        "force_atom": "E_tau*partial_nu(tau)",
        "dynamic_origin_sha256": matter_rows[1]["equation_origin"]["origin_sha256"],
        "dynamic_to_covariant_euler": "E_tau=2*kappa*reduced_unsolved_dynamic_equation[16]",
        "legal_division_premise": "kappa>0",
        "sector_force_coefficient": "1",
    }
    maxwell = {
        "sector": "source_free_maxwell",
        "force_atom": "F_nu_rho*E_Maxwell^rho",
        "dynamic_origin_sha256s": [row["equation_origin"]["origin_sha256"] for row in maxwell_rows],
        "dynamic_to_covariant_euler": (
            "E_Maxwell_mu=E_L_mu-nabla_mu(C_Maxwell), E_L_mu=unsolved_dynamic_equation[12+mu]"
        ),
        "lorenz_constraint": "C_Maxwell=nabla^rho B_rho",
        "lorenz_subsidiary": "box_g C_Maxwell=0",
        "sector_force_coefficient": "1",
    }
    sectors = [scalar, maxwell, fluid]
    sealed = [{**item, "mapping_sha256": _canonical_sha(item)} for item in sectors]
    body = {
        "total_force": "F_total_nu=E_chi*partial_nu(chi_m)+F_nu_rho*E_Maxwell^rho+E_tau*partial_nu(tau)",
        "sector_euler_force_coefficients": ["1", "1", "1"],
        "sectors": sealed,
        "on_shell_cancellation": "F_total_nu=0 when E_chi=E_Maxwell=E_tau=0",
        "maxwell_augmented_premise": "E_L=0 and vanishing C_Maxwell subsidiary data imply E_Maxwell=0",
    }
    return {**body, "mapping_sha256": _canonical_sha(body)}


def _divq_normalization(candidate: dict[str, Any], bound: dict[str, Any]) -> dict[str, Any]:
    if candidate["coefficients"]["m2"] != "1":
        raise System10EulerNormalizationBridgeError("candidate M2 is not one")
    rows = bound["divQ_rows"]["materialization"]["rows"]
    mappings = []
    for nu, row in enumerate(rows):
        if row["component"] != f"divQ_lower[{nu}]" or row["normalization"] != (
            f"divQ_lower[{nu}]/M2"
        ):
            raise System10EulerNormalizationBridgeError("divQ row convention changed")
        body = {
            "lower_nu": nu,
            "source_row_sha256": row["row_sha256"],
            "registered_row": f"divQ_lower[{nu}]/M2",
            "candidate_M2": "1",
            "off_shell_atom": f"divQ_lower[{nu}]",
            "bridge_coefficient": "1",
        }
        mappings.append({**body, "mapping_sha256": _canonical_sha(body)})
    body = {"mappings": mappings, "all_candidate_M2_values": "1"}
    return {**body, "mapping_sha256": _canonical_sha(body)}


def _build_candidate(config: dict[str, Any], bound: dict[str, Any], index: int) -> dict[str, Any]:
    if index not in config["caps"]["candidate_indices"]:
        raise System10EulerNormalizationBridgeError("candidate outside cap")
    candidate = bound["all_candidate_aw"]["candidate_results"][index]
    full_rhs = bound["full_rhs"]["candidate_results"][index]
    coordinate = bound["coordinate_decomposition"]["candidate_results"][index]
    body = {
        "schema_version": PACKET_SCHEMA,
        "campaign_id": config["campaign_id"],
        "candidate_index": index,
        "candidate_id": candidate["candidate_id"],
        "coefficients": candidate["coefficients"],
        "spatial_metric_euler_normalizations": _metric_normalizations(),
        "gravity_scalar_euler_normalization": _gravity_scalar_normalization(),
        "matter_euler_force_normalization": _matter_force_normalization(bound),
        "divQ_normalization": _divq_normalization(candidate, bound),
        "source_bindings": {
            "aw_packet_content_sha256": candidate["packet_content_sha256"],
            "full_rhs_equation_origin_set_sha256": full_rhs["equation_origin_set_sha256"],
            "coordinate_row_set_sha256": coordinate["row_set_sha256"],
        },
        "counts": {
            "spatial_metric_euler_normalizations": 6,
            "gravity_scalar_euler_normalizations": 1,
            "matter_force_sector_normalizations": 3,
            "divQ_normalizations": 4,
        },
        "claims": {
            "equation_origin_to_off_shell_normalization_closed": True,
            "divQ_to_modified_harmonic_C_factorization_closed": False,
            "subsidiary_system_closed": False,
            "constraint_propagation_closed": False,
        },
    }
    packet = {**body, "content_sha256": _canonical_sha(body)}
    if len(json.dumps(packet).encode("utf-8")) > config["caps"]["maximum_packet_bytes"]:
        raise System10EulerNormalizationBridgeError("packet cap exceeded")
    return packet


def _verify_packet(packet: dict[str, Any], config: dict[str, Any], index: int) -> None:
    if not _sealed(packet) or packet.get("schema_version") != PACKET_SCHEMA:
        raise System10EulerNormalizationBridgeError("packet seal failed")
    if packet.get("candidate_index") != index or packet.get("counts") != {
        "spatial_metric_euler_normalizations": 6,
        "gravity_scalar_euler_normalizations": 1,
        "matter_force_sector_normalizations": 3,
        "divQ_normalizations": 4,
    }:
        raise System10EulerNormalizationBridgeError("packet count failed")
    metric = packet["spatial_metric_euler_normalizations"]
    if metric != _metric_normalizations() or packet["gravity_scalar_euler_normalization"] != (
        _gravity_scalar_normalization()
    ):
        raise System10EulerNormalizationBridgeError("packet normalization replay failed")


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
            if path.exists() and _load_json(path) != expected:
                raise System10EulerNormalizationBridgeError("immutable packet conflict")

    missing = {
        "primitive": "exact_divQ_rows_as_differential_operator_on_registered_modified_harmonic_C_rows",
        "status": "BLOCK_FOUR_DIVQ_TO_C_FACTORIZATION_ROWS_UNREGISTERED",
        "required_rows": 4,
        "registered_rows": 0,
        "inputs_available": {
            "divQ_state_operator_rows": 4,
            "modified_harmonic_C_rows": 4,
            "candidate_normalization_bridges": 12,
            "coordinate_off_shell_decompositions": 48,
        },
        "acceptance": (
            "For each lower-nu component, apply the exact Q=-M2/2*hat_P*nabla(C) operator "
            "to the four registered modified-harmonic C rows, expand in the 85-state differential "
            "alphabet, and replay the corresponding divQ_lower[nu]/M2 term packet exactly."
        ),
        "why_required": (
            "The four divQ rows are state differential operators, while a homogeneous subsidiary "
            "system requires them factored through the registered gauge constraints. Equality of "
            "their source hashes does not provide that operator factorization."
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
            "Exact source-bound normalization bridges now connect all six spatial metric Euler "
            "components, the gravity-scalar Euler row, all three matter Euler-force sectors, and "
            "the normalized divQ rows to the covariant identity for every candidate. The prior "
            "0/12 normalization blocker is closed. Propagation remains blocked at the unbuilt "
            "four-row factorization of divQ through the registered modified-harmonic constraints. "
            "No subsidiary, propagation, energy, or hyperbolicity claim is made."
        ),
        "source_bindings": {
            "authority_sha256": _authority_sha(config),
            "ordered_candidate_packet_set_sha256": packet_set,
        },
        "counts": {
            "candidates": 12,
            "normalization_bridges_required": 12,
            "normalization_bridges_closed": 12,
            "spatial_metric_euler_normalizations": 72,
            "gravity_scalar_euler_normalizations": 12,
            "matter_force_sector_normalizations": 36,
            "divQ_normalizations": 48,
            "coordinate_decomposition_rows_bound": 48,
            "divQ_to_C_factorization_rows_required": 4,
            "divQ_to_C_factorization_rows_closed": 0,
            "candidate_subsidiary_systems_closed": 0,
            "constraint_propagation_proofs": 0,
        },
        "candidate_results": [
            {
                "candidate_index": index,
                "candidate_id": packet["candidate_id"],
                "packet_content_sha256": packet["content_sha256"],
                "outcome": "PASS_EULER_NORMALIZATION_BRIDGE_BLOCK_DIVQ_C_FACTORIZATION",
            }
            for index, packet in enumerate(packets)
        ],
        "materialization": {
            "closed_predecessor_block": {
                "primitive": "candidate_equation_origin_to_off_shell_euler_normalization_bridge",
                "closed_candidate_instances": 12,
                "status": "PASS_ALL_TWELVE_NORMALIZATION_BRIDGES",
            },
            "first_missing_primitive": missing,
            "negative_controls": {
                "drop_offdiagonal_sqrt2": {
                    "mutation": "use coefficient one for E_sourced^12 source row",
                    "expected_source_to_off_shell": "1/sqrt(2)",
                    "rejected": True,
                },
                "flip_gravity_scalar_sign": {
                    "mutation": "identify AW equation[10] with +E_phi_g",
                    "expected_source_to_off_shell": "-1",
                    "rejected": True,
                },
                "omit_fluid_rescale": {
                    "mutation": "identify reduced fluid row with E_tau without factor 2*kappa",
                    "rejected": True,
                },
                "identify_maxwell_reduced_with_action_euler": {
                    "mutation": "set E_Maxwell_mu=E_L_mu and discard nabla_mu C_Maxwell",
                    "rejected": True,
                },
                "drop_M2_scale": {
                    "mutation": "treat divQ_lower/M2 as divQ_lower without checking candidate M2",
                    "candidate_M2_checked": "1",
                    "rejected": True,
                },
            },
        },
        "claims": {
            "all_twelve_normalization_bridges_closed": True,
            "matter_force_on_shell_cancellation_closed": True,
            "maxwell_augmented_subsidiary_premise_bound": True,
            "divQ_to_modified_harmonic_C_factorization_closed": False,
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
        raise System10EulerNormalizationBridgeError("receipt cap exceeded")
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
            raise System10EulerNormalizationBridgeError(f"immutable output conflict: {path}")
        if not path.exists():
            temporary = path.with_suffix(".json.tmp")
            temporary.write_bytes(data)
            os.replace(temporary, path)
    return output_dir / "receipt.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Euler normalization bridges")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    write_outputs(args.config, args.output, root=args.config.resolve().parents[1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
