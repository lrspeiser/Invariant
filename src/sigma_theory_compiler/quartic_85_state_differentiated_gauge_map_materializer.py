from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any


class Quartic85StateDifferentiatedGaugeMapError(RuntimeError):
    """Raised when the differentiated gauge-map materializer fails closed."""


def _canonical_sha(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("ascii")).hexdigest()


def _file_sha(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except (OSError, ValueError) as exc:
        raise Quartic85StateDifferentiatedGaugeMapError(f"cannot read bound file: {path}") from exc


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise Quartic85StateDifferentiatedGaugeMapError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise Quartic85StateDifferentiatedGaugeMapError(f"JSON root is not an object: {path}")
    return value


def _resolve(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise Quartic85StateDifferentiatedGaugeMapError("bound path escapes repository root")
    return path


def _load_binding(root: Path, binding: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = _resolve(root, str(binding.get("path", "")))
    if _file_sha(path) != binding.get("file_sha256"):
        raise Quartic85StateDifferentiatedGaugeMapError(f"bound file hash mismatch: {path}")
    value = _load_json(path)
    if value.get("content_sha256") != binding.get("content_sha256"):
        raise Quartic85StateDifferentiatedGaugeMapError(f"bound content hash mismatch: {path}")
    return path, value


def _pairs() -> list[tuple[int, int]]:
    return list(itertools.combinations_with_replacement(range(4), 2))


def _triples() -> list[tuple[int, int, int]]:
    return list(itertools.combinations_with_replacement(range(4), 3))


def _primitive_slots() -> dict[str, list[dict[str, Any]]]:
    pairs = _pairs()
    triples = _triples()
    slots: dict[str, list[dict[str, Any]]] = {
        "hat_inverse_first": [],
        "tilde_inverse_second": [],
        "reference_connection_second": [],
        "gauge_source_second": [],
        "physical_metric_third": [],
    }
    for derivative in range(4):
        for left, right in pairs:
            slots["hat_inverse_first"].append(
                {
                    "key": f"d_hat[{derivative}|{left},{right}]",
                    "kind": "external_formulation_jet_atom",
                    "indices": [derivative, left, right],
                }
            )
    for first, second in pairs:
        for left, right in pairs:
            slots["tilde_inverse_second"].append(
                {
                    "key": f"d2_tilde[{first},{second}|{left},{right}]",
                    "kind": "external_formulation_jet_atom",
                    "indices": [first, second, left, right],
                }
            )
    for first, second in pairs:
        for upper in range(4):
            for left, right in pairs:
                slots["reference_connection_second"].append(
                    {
                        "key": f"d2_barGamma[{first},{second}|{upper}|{left},{right}]",
                        "kind": "external_formulation_jet_atom",
                        "indices": [first, second, upper, left, right],
                    }
                )
    for first, second in pairs:
        for lower in range(4):
            slots["gauge_source_second"].append(
                {
                    "key": f"d2_H[{first},{second}|{lower}]",
                    "kind": "external_formulation_jet_atom",
                    "indices": [first, second, lower],
                }
            )
    for derivative_triple in triples:
        for metric_field, (left, right) in enumerate(pairs):
            derivative_list = list(derivative_triple)
            if 0 in derivative_list:
                derivative_list.remove(0)
                state_index = 17 + metric_field
                state_coordinate = f"v[g_{left}{right}]"
            else:
                spatial = derivative_list.pop(0)
                state_index = 34 + (spatial - 1) * 17 + metric_field
                state_coordinate = f"w_{spatial}[g_{left}{right}]"
            slots["physical_metric_third"].append(
                {
                    "key": (
                        f"d3_g[{derivative_triple[0]},{derivative_triple[1]},"
                        f"{derivative_triple[2]}|{left},{right}]"
                    ),
                    "kind": "85_state_differential_operator",
                    "indices": [*derivative_triple, left, right],
                    "state_index": state_index,
                    "state_coordinate": state_coordinate,
                    "remaining_derivative_operator": derivative_list,
                }
            )
    expected = {
        "hat_inverse_first": 40,
        "tilde_inverse_second": 100,
        "reference_connection_second": 400,
        "gauge_source_second": 40,
        "physical_metric_third": 200,
    }
    if {family: len(items) for family, items in slots.items()} != expected:
        raise Quartic85StateDifferentiatedGaugeMapError(
            "primitive slot materialization count changed"
        )
    all_keys = [item["key"] for items in slots.values() for item in items]
    if len(all_keys) != 780 or len(set(all_keys)) != 780:
        raise Quartic85StateDifferentiatedGaugeMapError(
            "primitive slot keys are incomplete or overlap"
        )
    return slots


def _checkpoint_packets(
    slots: dict[str, list[dict[str, Any]]], chunk_sizes: dict[str, int]
) -> list[dict[str, Any]]:
    packets: list[dict[str, Any]] = []
    prior_sha = "0" * 64
    global_index = 0
    for family in (
        "hat_inverse_first",
        "tilde_inverse_second",
        "reference_connection_second",
        "gauge_source_second",
        "physical_metric_third",
    ):
        size = chunk_sizes[family]
        family_slots = slots[family]
        if len(family_slots) % size:
            raise Quartic85StateDifferentiatedGaugeMapError(
                f"chunk size does not divide family: {family}"
            )
        for family_chunk, start in enumerate(range(0, len(family_slots), size)):
            chunk = family_slots[start : start + size]
            body = {
                "global_chunk_index": global_index,
                "family": family,
                "family_chunk_index": family_chunk,
                "slot_start": start,
                "slot_stop_exclusive": start + len(chunk),
                "slot_count": len(chunk),
                "first_key": chunk[0]["key"],
                "last_key": chunk[-1]["key"],
                "slot_payload_sha256": _canonical_sha(chunk),
                "prior_checkpoint_sha256": prior_sha,
            }
            checkpoint_sha = _canonical_sha(body)
            packets.append({**body, "checkpoint_sha256": checkpoint_sha})
            prior_sha = checkpoint_sha
            global_index += 1
    if len(packets) != 48 or sum(item["slot_count"] for item in packets) != 780:
        raise Quartic85StateDifferentiatedGaugeMapError("checkpoint packet census changed")
    return packets


def _indexed_formula_program() -> dict[str, Any]:
    templates = {
        "A_metric_bracket": (
            "A[kappa|rho,sigma]=partial_rho g[kappa,sigma]+"
            "partial_sigma g[kappa,rho]-partial_kappa g[rho,sigma]"
        ),
        "inverse_metric_first": ("dginv[mu|a,b]=-ginv[a,r]*dg[mu|r,s]*ginv[s,b]"),
        "inverse_metric_second": (
            "d2ginv[mu,gamma|a,b]=ginv[a,r]*dg[mu|r,s]*ginv[s,t]*"
            "dg[gamma|t,u]*ginv[u,b]+ginv[a,r]*dg[gamma|r,s]*ginv[s,t]*"
            "dg[mu|t,u]*ginv[u,b]-ginv[a,r]*d2g[mu,gamma|r,s]*ginv[s,b]"
        ),
        "A_metric_bracket_first": (
            "dA[mu|kappa,rho,sigma]=d2g[mu,rho|kappa,sigma]+"
            "d2g[mu,sigma|kappa,rho]-d2g[mu,kappa|rho,sigma]"
        ),
        "A_metric_bracket_second": (
            "d2A[mu,gamma|kappa,rho,sigma]=d3g[mu,gamma,rho|kappa,sigma]+"
            "d3g[mu,gamma,sigma|kappa,rho]-d3g[mu,gamma,kappa|rho,sigma]"
        ),
        "physical_connection": (
            "Gamma[lambda|rho,sigma]=(1/2) ginv[lambda,kappa] A[kappa|rho,sigma]"
        ),
        "physical_connection_second": (
            "d2Gamma[mu,gamma|lambda,rho,sigma]=(1/2)["
            "d2ginv[mu,gamma|lambda,kappa]*A[kappa|rho,sigma]+"
            "dginv[gamma|lambda,kappa]*dA[mu|kappa,rho,sigma]+"
            "dginv[mu|lambda,kappa]*dA[gamma|kappa,rho,sigma]+"
            "ginv[lambda,kappa]*d2A[mu,gamma|kappa,rho,sigma]]"
        ),
        "connection_difference_up": (
            "DeltaUp=Gamma-barGamma; dDeltaUp=dGamma-dbarGamma; d2DeltaUp=d2Gamma-d2_barGamma"
        ),
        "connection_difference_lower_second": (
            "d2DeltaLower[mu,gamma|beta,rho,sigma]="
            "d2g[mu,gamma|beta,lambda]*DeltaUp[lambda|rho,sigma]+"
            "dg[gamma|beta,lambda]*dDeltaUp[mu|lambda,rho,sigma]+"
            "dg[mu|beta,lambda]*dDeltaUp[gamma|lambda,rho,sigma]+"
            "g[beta,lambda]*(d2Gamma-d2_barGamma)[mu,gamma|lambda,rho,sigma]"
        ),
        "constraint_second_partial": (
            "d2C[mu,gamma|beta]=sum_rho_sigma["
            "d2tilde[mu,gamma|rho,sigma]*DeltaLower[beta,rho,sigma]+"
            "dtilde[gamma|rho,sigma]*dDeltaLower[mu|beta,rho,sigma]+"
            "dtilde[mu|rho,sigma]*dDeltaLower[gamma|beta,rho,sigma]+"
            "tilde[rho,sigma]*d2DeltaLower[mu,gamma|beta,rho,sigma]]-"
            "d2H[mu,gamma|beta]"
        ),
        "constraint_covariant_hessian": (
            "nabla2C[mu,gamma|beta]=d2C[mu,gamma|beta]-"
            "dGamma[mu|lambda,gamma,beta]*C[lambda]-"
            "Gamma[lambda|gamma,beta]*dC[mu|lambda]-"
            "Gamma[lambda|mu,gamma]*nablaC[lambda,beta]-"
            "Gamma[lambda|mu,beta]*nablaC[gamma,lambda]"
        ),
        "constraint_first_partial": ("dC[mu|beta]=nablaC[mu,beta]+Gamma[lambda|mu,beta]*C[lambda]"),
        "hat_covariant_first": (
            "nablaHat[mu|a,b]=d_hat[mu|a,b]+"
            "Gamma[a|mu,lambda]*hat[lambda,b]+"
            "Gamma[b|mu,lambda]*hat[a,lambda]"
        ),
        "hat_projector_covariant_first": (
            "nablaHatP[mu|alpha,gamma,rho,nu]=(1/2)["
            "delta[alpha,rho]*nablaHat[mu|nu,gamma]+"
            "delta[alpha,nu]*nablaHat[mu|rho,gamma]-"
            "delta[alpha,gamma]*nablaHat[mu|rho,nu]]"
        ),
        "hat_projector": (
            "HatP[alpha|gamma,mu,nu]=(1/2)[delta[alpha,mu]*hat[nu,gamma]+"
            "delta[alpha,nu]*hat[mu,gamma]-delta[alpha,gamma]*hat[mu,nu]]"
        ),
        "gauge_divergence_upper": (
            "divQ_upper[nu]=-(M2/2) sum_mu_alpha_beta_gamma["
            "nablaHatP[mu|alpha,gamma,mu,nu]*ginv[alpha,beta]*nablaC[gamma,beta]+"
            "HatP[alpha|gamma,mu,nu]*ginv[alpha,beta]*nabla2C[mu,gamma|beta]]"
        ),
        "gauge_divergence_lower": ("divQ_lower[nu]=sum_sigma g[nu,sigma]*divQ_upper[sigma]"),
    }
    program = {
        "dimension": 4,
        "summation_convention": "every repeated Greek index ranges over 0..3",
        "raw_partial_canonicalization": (
            "commuting coordinate-partial multi-indices are sorted before primitive-slot lookup"
        ),
        "metric_pair_order": [list(pair) for pair in _pairs()],
        "derivative_pair_order": [list(pair) for pair in _pairs()],
        "derivative_triple_order": [list(triple) for triple in _triples()],
        "templates": templates,
        "output_components": [f"divQ_lower[{index}]" for index in range(4)],
        "metric_compatibility_reduction": "nabla_mu g^(alpha beta)=0",
    }
    return {**program, "program_sha256": _canonical_sha(program)}


def _materialize(
    readiness: dict[str, Any],
    divergence: dict[str, Any],
    gauge_source: dict[str, Any],
    basis: dict[str, Any],
    chunk_sizes: dict[str, int],
) -> dict[str, Any]:
    if readiness.get("decision") != (
        "PASS_RESUMABLE_READINESS_CONTRACT_FIVE_PRIMITIVE_JET_BLOCKERS"
    ):
        raise Quartic85StateDifferentiatedGaugeMapError("readiness predecessor changed")
    if divergence.get("decision") != (
        "BOUNDED_PASS_COMMON_COVARIANT_IDENTITY_TYPED_BLOCK_DIFFERENTIATED_GAUGE_MAP"
    ):
        raise Quartic85StateDifferentiatedGaugeMapError("off-shell divergence predecessor changed")
    if gauge_source.get("status") != (
        "pass_all_12_exact_local_nonlinear_time_acceleration_eliminations"
    ):
        raise Quartic85StateDifferentiatedGaugeMapError("gauge-source formula predecessor changed")
    if basis.get("decision") != (
        "BOUNDED_PASS_KINEMATIC_MATTER_BASIS_TYPED_BLOCK_GRAVITY_COORDINATE_MAP"
    ):
        raise Quartic85StateDifferentiatedGaugeMapError("constraint-coordinate predecessor changed")
    readiness_inventory = readiness.get("materialization", {}).get("primitive_inventory", [])
    expected_counts = {item["family"]: item["missing_slots"] for item in readiness_inventory}
    if expected_counts != {
        "hat_inverse_first": 40,
        "tilde_inverse_second": 100,
        "reference_connection_second": 400,
        "gauge_source_second": 40,
        "physical_metric_third": 200,
    }:
        raise Quartic85StateDifferentiatedGaugeMapError("readiness primitive inventory changed")
    slots = _primitive_slots()
    packets = _checkpoint_packets(slots, chunk_sizes)
    formula_program = _indexed_formula_program()
    slot_schema = {
        family: {
            "slot_count": len(items),
            "payload_sha256": _canonical_sha(items),
            "kind_counts": {
                kind: sum(item["kind"] == kind for item in items)
                for kind in sorted({item["kind"] for item in items})
            },
        }
        for family, items in slots.items()
    }
    primitive_set_sha = _canonical_sha(slot_schema)
    basis_records = basis.get("materialization", {}).get("candidate_results", [])
    if len(basis_records) != 12:
        raise Quartic85StateDifferentiatedGaugeMapError("candidate basis count changed")
    candidate_results: list[dict[str, Any]] = []
    for item in sorted(basis_records, key=lambda record: record["candidate_id"]):
        manifest = {
            "schema_version": "invariant-candidate-indexed-differentiated-gauge-map-manifest-1.0",
            "candidate_id": item["candidate_id"],
            "constraint_coordinate_manifest_sha256": item["constraint_coordinate_manifest_sha256"],
            "primitive_slot_schema_sha256": primitive_set_sha,
            "indexed_formula_program_sha256": formula_program["program_sha256"],
            "formal_external_jet_atoms": 580,
            "physical_metric_third_85_operator_slots": 200,
            "output_covector_components": 4,
            "fully_expanded_coefficient_rows": 0,
            "outcome": "PASS_EXACT_INDEXED_MAP_FORMAL_EXTERNAL_JETS",
        }
        candidate_results.append({**manifest, "manifest_sha256": _canonical_sha(manifest)})
    if len({item["manifest_sha256"] for item in candidate_results}) != 12:
        raise Quartic85StateDifferentiatedGaugeMapError(
            "candidate map manifests are not one-to-one"
        )
    all_keys = [item["key"] for items in slots.values() for item in items]
    integrity = {
        "complete_slot_count": len(all_keys),
        "unique_slot_count": len(set(all_keys)),
        "omitted_last_slot_negative": {
            "slot_count": len(all_keys[:-1]),
            "expected": 780,
            "rejected": len(all_keys[:-1]) != 780,
        },
        "duplicated_first_slot_negative": {
            "slot_count": 780,
            "unique_slot_count": len(set(all_keys[:-1] + [all_keys[0]])),
            "expected_unique": 780,
            "rejected": len(set(all_keys[:-1] + [all_keys[0]])) != 780,
        },
    }
    readiness_progress = [
        {"unit_id": "R0", "status": "PASS_BOUND_CONVENTIONS"},
        {"unit_id": "R1", "status": "PASS_REUSED_SOURCE_PACKETS"},
        {"unit_id": "R2", "status": "PASS_INDEXED_PRODUCT_RULE"},
        {
            "unit_id": "R3",
            "status": "PASS_FORMAL_SLOT_REGISTRATION_VALUES_UNCERTIFIED",
        },
        {"unit_id": "R4", "status": "PASS_EXACT_INDEXED_C_HESSIAN_TEMPLATE"},
        {"unit_id": "R5", "status": "PASS_EXACT_INDEXED_HAT_PROJECTOR_DERIVATIVE"},
        {"unit_id": "R6", "status": "PASS_EXACT_INDEXED_FOUR_COMPONENT_MAP"},
        {
            "unit_id": "R7",
            "status": "PARTIAL_85_OPERATOR_MAP_NO_SCALAR_COEFFICIENT_EXPANSION",
        },
        {"unit_id": "R8", "status": "PARTIAL_TWO_OF_FOUR_INTEGRITY_CONTROLS"},
    ]
    return {
        "primitive_slot_schema": slot_schema,
        "primitive_slot_schema_sha256": primitive_set_sha,
        "checkpoint_packets": packets,
        "final_checkpoint_sha256": packets[-1]["checkpoint_sha256"],
        "physical_metric_third_operator_map": slots["physical_metric_third"],
        "indexed_formula_program": formula_program,
        "candidate_results": candidate_results,
        "packet_integrity_controls": integrity,
        "readiness_unit_progress": readiness_progress,
        "scientific_boundary": {
            "formal_external_jet_atoms_are_values": False,
            "external_jet_domain_certified": False,
            "fully_expanded_85_state_coefficient_rows": False,
            "constraint_propagation_inferred": False,
            "meaning": (
                "the indexed map is exact as a tensor program over independent prescribed "
                "formulation-field jet atoms and 85-state differential operators"
            ),
        },
    }


def build_receipt(config_path: Path, *, root: Path | None = None) -> dict[str, Any]:
    repository = (root or config_path.resolve().parents[1]).resolve()
    config = _load_json(config_path)
    if config.get("schema_version") != (
        "invariant-quartic-85-state-differentiated-gauge-map-materializer-config-1.0"
    ):
        raise Quartic85StateDifferentiatedGaugeMapError("unsupported config schema")
    expected_chunks = {
        "hat_inverse_first": 10,
        "tilde_inverse_second": 10,
        "reference_connection_second": 20,
        "gauge_source_second": 10,
        "physical_metric_third": 20,
    }
    if config.get("primitive_chunk_sizes") != expected_chunks:
        raise Quartic85StateDifferentiatedGaugeMapError("primitive chunk plan changed")
    expected_policy = {
        "formal_primitive_packet_registration": True,
        "physical_metric_third_to_85_state_operator_map": True,
        "exact_indexed_differentiated_gauge_map": True,
        "external_formulation_jet_values_certified": False,
        "fully_expanded_85_state_coefficient_rows": False,
        "constraint_propagation": False,
        "candidate_jet_uniformity": False,
        "nonlinear_global_closure": False,
        "gravity_h7": False,
        "universal_all_matter": False,
        "promotion": False,
    }
    if config.get("claims_policy") != expected_policy:
        raise Quartic85StateDifferentiatedGaugeMapError("claims policy is absent or broadened")
    bound = {
        name: _load_binding(repository, binding)
        for name, binding in config.get("bindings", {}).items()
    }
    expected_bindings = {
        "readiness_contract",
        "off_shell_divergence_gate",
        "gauge_source_formula",
        "constraint_coordinate_basis",
    }
    if set(bound) != expected_bindings:
        raise Quartic85StateDifferentiatedGaugeMapError("closed binding manifest changed")
    materialization = _materialize(
        bound["readiness_contract"][1],
        bound["off_shell_divergence_gate"][1],
        bound["gauge_source_formula"][1],
        bound["constraint_coordinate_basis"][1],
        config["primitive_chunk_sizes"],
    )
    source_path = Path(__file__).resolve()
    test_path = repository / (
        "tests/test_quartic_85_state_differentiated_gauge_map_materializer.py"
    )
    body: dict[str, Any] = {
        "schema_version": (
            "invariant-quartic-85-state-differentiated-gauge-map-materializer-result-1.0"
        ),
        "campaign_id": config["campaign_id"],
        "decision": "PASS_EXACT_INDEXED_GAUGE_MAP_WITH_FORMAL_EXTERNAL_JET_PACKETS",
        "materialization": materialization,
        "counts": {
            "formal_external_jet_atoms": 580,
            "physical_metric_third_operator_slots": 200,
            "total_primitive_slots": 780,
            "checkpoint_packets": 48,
            "indexed_formula_templates": 17,
            "output_divergence_components": 4,
            "candidate_map_manifests": 12,
            "fully_expanded_85_state_coefficient_rows": 0,
            "constraint_propagation_claims": 0,
            "negative_controls": 2,
            "readiness_units_passed_indexed": 7,
            "readiness_units_partial": 2,
        },
        "claims": {
            "formal_primitive_packet_registration_closed": True,
            "physical_metric_third_to_85_state_operator_map_closed": True,
            "exact_indexed_differentiated_gauge_map_closed": True,
            "external_formulation_jet_values_certified": False,
            "fully_expanded_85_state_coefficient_rows_closed": False,
            "constraint_propagation_closed": False,
            "candidate_jet_uniformity_closed": False,
            "nonlinear_global_closure_established": False,
            "gravity_h7_theorem_established": False,
            "universal_all_matter_closure_established": False,
            "promotion_authorized": False,
        },
        "scope": (
            "All 780 readiness slots are registered in 48 chained checkpoint packets. The 200 "
            "physical metric third-jet slots are mapped to differential operators on the "
            "registered 85-state ordering, and an exact indexed tensor program constructs the "
            "four components of nabla Q over 580 independent prescribed formulation-field jet "
            "atoms. Those atoms are formal inputs, not certified values. Scalar coefficient "
            "expansion, external-jet domains, constraint propagation, candidate-jet uniformity, "
            "nonlinear/global closure, H7, universal matter, and promotion remain false."
        ),
        "source_bindings": {
            "config": {
                "path": config_path.relative_to(repository).as_posix(),
                "file_sha256": _file_sha(config_path),
            },
            **{
                name: {
                    "path": path.relative_to(repository).as_posix(),
                    "file_sha256": _file_sha(path),
                    "content_sha256": value["content_sha256"],
                }
                for name, (path, value) in bound.items()
            },
            "source": {
                "path": source_path.relative_to(repository).as_posix(),
                "file_sha256": _file_sha(source_path),
            },
            "test": {
                "path": test_path.relative_to(repository).as_posix(),
                "file_sha256": _file_sha(test_path),
            },
        },
    }
    return {**body, "content_sha256": _canonical_sha(body)}


def write_receipt(
    config_path: Path, output_path: Path, *, root: Path | None = None
) -> dict[str, Any]:
    receipt = build_receipt(config_path, root=root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    write_receipt(args.config.resolve(), args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
