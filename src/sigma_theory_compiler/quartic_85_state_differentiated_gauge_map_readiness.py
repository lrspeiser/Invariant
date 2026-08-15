from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


class Quartic85StateDifferentiatedGaugeReadinessError(RuntimeError):
    """Raised when the differentiated-gauge readiness contract fails closed."""


def _canonical_sha(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("ascii")).hexdigest()


def _file_sha(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except (OSError, ValueError) as exc:
        raise Quartic85StateDifferentiatedGaugeReadinessError(
            f"cannot read bound file: {path}"
        ) from exc


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise Quartic85StateDifferentiatedGaugeReadinessError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise Quartic85StateDifferentiatedGaugeReadinessError(f"JSON root is not an object: {path}")
    return value


def _resolve(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise Quartic85StateDifferentiatedGaugeReadinessError("bound path escapes repository root")
    return path


def _load_binding(root: Path, binding: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = _resolve(root, str(binding.get("path", "")))
    if _file_sha(path) != binding.get("file_sha256"):
        raise Quartic85StateDifferentiatedGaugeReadinessError(f"bound file hash mismatch: {path}")
    value = _load_json(path)
    if value.get("content_sha256") != binding.get("content_sha256"):
        raise Quartic85StateDifferentiatedGaugeReadinessError(
            f"bound content hash mismatch: {path}"
        )
    return path, value


def _primitive_inventory(chunk_sizes: dict[str, int]) -> list[dict[str, Any]]:
    inventory = [
        {
            "family": "hat_inverse_first",
            "tensor_components": 10,
            "derivative_multi_indices": 4,
            "missing_slots": 40,
            "needed_for": "nabla_mu hat_P_alpha^(gamma mu nu)",
        },
        {
            "family": "tilde_inverse_second",
            "tensor_components": 10,
            "derivative_multi_indices": 10,
            "missing_slots": 100,
            "needed_for": "partial_(mu gamma) C_beta",
        },
        {
            "family": "reference_connection_second",
            "tensor_components": 40,
            "derivative_multi_indices": 10,
            "missing_slots": 400,
            "needed_for": "partial_(mu gamma) Delta_Gamma",
        },
        {
            "family": "gauge_source_second",
            "tensor_components": 4,
            "derivative_multi_indices": 10,
            "missing_slots": 40,
            "needed_for": "partial_(mu gamma) H_beta",
        },
        {
            "family": "physical_metric_third",
            "tensor_components": 10,
            "derivative_multi_indices": 20,
            "missing_slots": 200,
            "needed_for": "partial_(mu gamma) Gamma and the second derivative of C_beta",
        },
    ]
    for item in inventory:
        chunk_size = chunk_sizes.get(item["family"])
        if not isinstance(chunk_size, int) or chunk_size <= 0:
            raise Quartic85StateDifferentiatedGaugeReadinessError(
                f"invalid primitive chunk size: {item['family']}"
            )
        if item["missing_slots"] % chunk_size:
            raise Quartic85StateDifferentiatedGaugeReadinessError(
                f"chunk size does not divide slot count: {item['family']}"
            )
        item["chunk_size"] = chunk_size
        item["chunk_count"] = item["missing_slots"] // chunk_size
        item["status"] = "BLOCK_MISSING_REGISTRATION"
    if sum(item["missing_slots"] for item in inventory) != 780:
        raise Quartic85StateDifferentiatedGaugeReadinessError(
            "primitive missing-slot census changed"
        )
    if sum(item["chunk_count"] for item in inventory) != 48:
        raise Quartic85StateDifferentiatedGaugeReadinessError("primitive chunk census changed")
    return inventory


def _resume_units() -> list[dict[str, Any]]:
    return [
        {
            "unit_id": "R0",
            "operation": "seal index, sign, field-order, and formula conventions",
            "depends_on": [],
            "bounded_output": "one canonical convention manifest",
            "status": "PASS_REGISTERED",
        },
        {
            "unit_id": "R1",
            "operation": "reuse exact C_beta, nabla_gamma C_beta, and Q_mu_nu source packets",
            "depends_on": ["R0"],
            "bounded_output": "4 C entries, 16 nabla-C entries, 10 symmetric Q entries",
            "status": "PASS_CONSTRUCTIBLE_NO_COLD_RUN",
        },
        {
            "unit_id": "R2",
            "operation": "emit covariant product-rule dependency shell for nabla^mu Q_mu_nu",
            "depends_on": ["R1"],
            "bounded_output": "two nonzero branches plus one metric-compatibility zero branch",
            "status": "PASS_CONSTRUCTIBLE_NO_COLD_RUN",
        },
        {
            "unit_id": "R3",
            "operation": "register five primitive differentiated formulation-field packets",
            "depends_on": ["R0"],
            "bounded_output": "780 slots in 48 resumable chunks",
            "status": "BLOCK_PRIMITIVE_PACKETS_ABSENT",
        },
        {
            "unit_id": "R4",
            "operation": (
                "assemble second coordinate derivative and full covariant Hessian of C_beta"
            ),
            "depends_on": ["R1", "R3"],
            "bounded_output": (
                "40 independent symmetric partial-pair entries and 64 ordered covariant "
                "Hessian entries"
            ),
            "status": "BLOCKED_BY_R3",
        },
        {
            "unit_id": "R5",
            "operation": "assemble covariant derivative of the hat projector",
            "depends_on": ["R1", "R3"],
            "bounded_output": "one hashed sparse coefficient packet",
            "status": "BLOCKED_BY_R3",
        },
        {
            "unit_id": "R6",
            "operation": "contract the two product-rule branches into nabla^mu Q_mu_nu",
            "depends_on": ["R2", "R4", "R5"],
            "bounded_output": "4 exact covector components",
            "status": "BLOCKED_BY_R4_R5",
        },
        {
            "unit_id": "R7",
            "operation": "lower the four divergence components into 85-state differential rows",
            "depends_on": ["R6"],
            "bounded_output": "4 common rows and 12 candidate binding manifests",
            "status": "BLOCKED_BY_R6",
        },
        {
            "unit_id": "R8",
            "operation": "run omission, sign, field-order, and zero-fill corruption controls",
            "depends_on": ["R7"],
            "bounded_output": "at least 4 exact nonzero residual witnesses",
            "status": "BLOCKED_BY_R7",
        },
    ]


def _materialize(
    divergence: dict[str, Any],
    gauge_source: dict[str, Any],
    basis: dict[str, Any],
    coordinate_tube: dict[str, Any],
    chunk_sizes: dict[str, int],
) -> dict[str, Any]:
    if divergence.get("decision") != (
        "BOUNDED_PASS_COMMON_COVARIANT_IDENTITY_TYPED_BLOCK_DIFFERENTIATED_GAUGE_MAP"
    ):
        raise Quartic85StateDifferentiatedGaugeReadinessError(
            "off-shell divergence predecessor changed"
        )
    if gauge_source.get("status") != (
        "pass_all_12_exact_local_nonlinear_time_acceleration_eliminations"
    ):
        raise Quartic85StateDifferentiatedGaugeReadinessError(
            "gauge-source formula predecessor changed"
        )
    if basis.get("decision") != (
        "BOUNDED_PASS_KINEMATIC_MATTER_BASIS_TYPED_BLOCK_GRAVITY_COORDINATE_MAP"
    ):
        raise Quartic85StateDifferentiatedGaugeReadinessError("85-state basis predecessor changed")
    if coordinate_tube.get("status") != (
        "pass_all_12_uniform_coordinate_2jet_to_covariant_hyperbolicity_tubes"
    ):
        raise Quartic85StateDifferentiatedGaugeReadinessError(
            "coordinate two-jet predecessor changed"
        )
    coordinate_control = coordinate_tube.get("generic_coordinate_jet_majorant_control", {})
    coordinate_atoms = coordinate_control.get("bounded_coordinate_atoms", {})
    if (
        coordinate_control.get("control")
        != "uniform coordinate-state 2-jet to covariant-jet majorant theorem"
        or coordinate_atoms.get("total") != 153
        or coordinate_atoms.get("acceleration_free_symmetric_second_partial_components") != 99
    ):
        raise Quartic85StateDifferentiatedGaugeReadinessError(
            "coordinate two-jet atom census changed"
        )
    nonlinear = gauge_source.get("nonlinear_evolution_control", {})
    fields = nonlinear.get("formulation_fields", {})
    if fields.get("gauge_source") != "prescribed covector H_beta and first derivative":
        raise Quartic85StateDifferentiatedGaugeReadinessError(
            "registered gauge-source jet order changed"
        )
    primitive = _primitive_inventory(chunk_sizes)
    product_rule = {
        "input": ("Q^mu_nu=-(M2/2) hat_P_alpha^(gamma mu nu) g^(alpha beta) nabla_gamma C_beta"),
        "output": "nabla_mu Q^(mu)_nu",
        "branches": [
            {
                "branch": "projector_derivative",
                "formula": "(nabla hat_P)*g^-1*(nabla C)",
                "constructible_subset": ("connection action on registered order-zero hat metric"),
                "missing": "partial hat_inverse_metric",
            },
            {
                "branch": "physical_inverse_derivative",
                "formula": "hat_P*(nabla g^-1)*(nabla C)",
                "constructible_subset": "exactly zero by physical metric compatibility",
                "missing": None,
            },
            {
                "branch": "constraint_hessian",
                "formula": "hat_P*g^-1*(nabla nabla C)",
                "constructible_subset": (
                    "connection and first-connection corrections using C and nabla C"
                ),
                "missing": ("raw second C derivative from tilde2, reference2, H2, and metric3"),
            },
        ],
        "status": "PASS_DEPENDENCY_SHELL_ONLY",
    }
    constructible = {
        "exact_source_packets": [
            "C_beta (4 components)",
            "nabla_gamma C_beta (16 components)",
            "Q_mu_nu (10 symmetric components)",
        ],
        "exact_algebraic_reductions": [
            "nabla_mu g^(alpha beta)=0",
            "constant M2 has zero derivative",
            "all connection corrections using registered metric two-jet, C, and nabla C",
            "all first-tilde times first-Delta-Gamma product terms in partial^2 C",
        ],
        "cold_symbolic_work_executed": False,
        "coordinate_two_jet_evidence": {
            "bounded_coordinate_atoms": 153,
            "acceleration_free_second_partial_atoms": 99,
            "third_partial_atoms": 0,
        },
    }
    units = _resume_units()
    contract = {
        "primitive_inventory": primitive,
        "constructible_subset": constructible,
        "product_rule_shell": product_rule,
        "resume_units": units,
        "resume_order": ["R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8"],
        "checkpoint_rule": (
            "each primitive chunk is canonical JSON with family, slot interval, convention hash, "
            "payload hash, and predecessor hashes; assembly refuses gaps, overlaps, or zero-fill"
        ),
    }
    return {**contract, "readiness_contract_sha256": _canonical_sha(contract)}


def build_receipt(config_path: Path, *, root: Path | None = None) -> dict[str, Any]:
    repository = (root or config_path.resolve().parents[1]).resolve()
    config = _load_json(config_path)
    if config.get("schema_version") != (
        "invariant-quartic-85-state-differentiated-gauge-map-readiness-config-1.0"
    ):
        raise Quartic85StateDifferentiatedGaugeReadinessError("unsupported config schema")
    expected_chunks = {
        "hat_inverse_first": 10,
        "tilde_inverse_second": 10,
        "reference_connection_second": 20,
        "gauge_source_second": 10,
        "physical_metric_third": 20,
    }
    if config.get("primitive_chunk_sizes") != expected_chunks:
        raise Quartic85StateDifferentiatedGaugeReadinessError("primitive chunk plan changed")
    expected_policy = {
        "readiness_contract_complete": True,
        "constructible_subset_inventoried": True,
        "primitive_missing_jets_registered": False,
        "differentiated_gauge_map_constructed": False,
        "constraint_propagation": False,
        "candidate_jet_uniformity": False,
        "nonlinear_global_closure": False,
        "gravity_h7": False,
        "universal_all_matter": False,
        "promotion": False,
    }
    if config.get("claims_policy") != expected_policy:
        raise Quartic85StateDifferentiatedGaugeReadinessError(
            "claims policy is absent or broadened"
        )
    bound = {
        name: _load_binding(repository, binding)
        for name, binding in config.get("bindings", {}).items()
    }
    expected_bindings = {
        "off_shell_divergence_gate",
        "gauge_source_formula",
        "constraint_coordinate_basis",
        "coordinate_two_jet_tube",
    }
    if set(bound) != expected_bindings:
        raise Quartic85StateDifferentiatedGaugeReadinessError("closed binding manifest changed")
    materialization = _materialize(
        bound["off_shell_divergence_gate"][1],
        bound["gauge_source_formula"][1],
        bound["constraint_coordinate_basis"][1],
        bound["coordinate_two_jet_tube"][1],
        config["primitive_chunk_sizes"],
    )
    source_path = Path(__file__).resolve()
    test_path = repository / ("tests/test_quartic_85_state_differentiated_gauge_map_readiness.py")
    body: dict[str, Any] = {
        "schema_version": (
            "invariant-quartic-85-state-differentiated-gauge-map-readiness-result-1.0"
        ),
        "campaign_id": config["campaign_id"],
        "decision": "PASS_RESUMABLE_READINESS_CONTRACT_FIVE_PRIMITIVE_JET_BLOCKERS",
        "materialization": materialization,
        "counts": {
            "constructible_source_packet_families": 3,
            "constructible_algebraic_reductions": 4,
            "product_rule_branches": 3,
            "metric_compatibility_zero_branches": 1,
            "missing_primitive_jet_families": 5,
            "missing_primitive_slots": 780,
            "primitive_resume_chunks": 48,
            "resume_units": 9,
            "differentiated_gauge_maps_constructed": 0,
            "constraint_propagation_claims": 0,
            "cold_symbolic_runs": 0,
        },
        "claims": {
            "differentiated_gauge_map_readiness_contract_closed": True,
            "constructible_subset_inventory_closed": True,
            "primitive_differentiated_formulation_jets_registered": False,
            "differentiated_gauge_map_in_85_state_coordinates_closed": False,
            "constraint_propagation_closed": False,
            "candidate_jet_uniformity_closed": False,
            "nonlinear_global_closure_established": False,
            "gravity_h7_theorem_established": False,
            "universal_all_matter_closure_established": False,
            "promotion_authorized": False,
        },
        "scope": (
            "A deterministic, resumable readiness contract for differentiating the registered "
            "modified-harmonic gauge completion. It inventories the exact constructible source "
            "packets and product-rule branches, seals five absent primitive jet families as 780 "
            "slots in 48 chunks, and orders nine bounded resume units. It performs no cold "
            "symbolic contraction and does not construct nabla Q, constraint propagation, "
            "candidate-jet uniformity, nonlinear/global closure, H7, universal matter, or "
            "promotion."
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
