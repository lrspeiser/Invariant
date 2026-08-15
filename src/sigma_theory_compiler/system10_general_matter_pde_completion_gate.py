from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from fractions import Fraction
from pathlib import Path
from typing import Any


class System10GeneralMatterPDECompletionError(RuntimeError):
    """Raised when the general System 10 completion audit fails closed."""


def _canonical_sha(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("ascii")).hexdigest()


def _file_sha(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except (OSError, ValueError) as exc:
        raise System10GeneralMatterPDECompletionError(f"cannot read bound file: {path}") from exc


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise System10GeneralMatterPDECompletionError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise System10GeneralMatterPDECompletionError(f"JSON root is not an object: {path}")
    return value


def _resolve(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise System10GeneralMatterPDECompletionError("bound path escapes repository root")
    return path


def _load_binding(root: Path, binding: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = _resolve(root, str(binding.get("path", "")))
    if _file_sha(path) != binding.get("file_sha256"):
        raise System10GeneralMatterPDECompletionError(f"bound file hash mismatch: {path}")
    value = _load_json(path)
    if value.get("content_sha256") != binding.get("content_sha256"):
        raise System10GeneralMatterPDECompletionError(f"bound content hash mismatch: {path}")
    return path, value


def _pairs() -> list[tuple[int, int]]:
    return list(itertools.combinations_with_replacement(range(4), 2))


def _slot_families() -> list[dict[str, Any]]:
    pairs = _pairs()
    families: list[tuple[str, list[str], str]] = []
    families.append(
        (
            "hat_inverse_first",
            [
                f"d_hat[{derivative}|{left},{right}]"
                for derivative in range(4)
                for left, right in pairs
            ],
            "formal_atom_only",
        )
    )
    families.append(
        (
            "tilde_inverse_second",
            [
                f"d2_tilde[{first},{second}|{left},{right}]"
                for first, second in pairs
                for left, right in pairs
            ],
            "formal_atom_only",
        )
    )
    families.append(
        (
            "reference_connection_second",
            [
                f"d2_barGamma[{first},{second}|{upper}|{left},{right}]"
                for first, second in pairs
                for upper in range(4)
                for left, right in pairs
            ],
            "formal_atom_only",
        )
    )
    families.append(
        (
            "gauge_source_second",
            [f"d2_H[{first},{second}|{lower}]" for first, second in pairs for lower in range(4)],
            "formal_atom_only",
        )
    )
    families.extend(
        [
            ("hat_inverse_zero", [f"hat[{left},{right}]" for left, right in pairs], "absent"),
            ("tilde_inverse_zero", [f"tilde[{left},{right}]" for left, right in pairs], "absent"),
            (
                "tilde_inverse_first",
                [
                    f"d_tilde[{derivative}|{left},{right}]"
                    for derivative in range(4)
                    for left, right in pairs
                ],
                "absent",
            ),
            (
                "reference_connection_zero",
                [
                    f"barGamma[{upper}|{left},{right}]"
                    for upper in range(4)
                    for left, right in pairs
                ],
                "absent",
            ),
            (
                "reference_connection_first",
                [
                    f"d_barGamma[{derivative}|{upper}|{left},{right}]"
                    for derivative in range(4)
                    for upper in range(4)
                    for left, right in pairs
                ],
                "absent",
            ),
            ("gauge_source_zero", [f"H[{lower}]" for lower in range(4)], "absent"),
            (
                "gauge_source_first",
                [f"d_H[{derivative}|{lower}]" for derivative in range(4) for lower in range(4)],
                "absent",
            ),
            ("physical_metric_zero", [f"g[{left},{right}]" for left, right in pairs], "absent"),
            (
                "physical_metric_first",
                [
                    f"d_g[{derivative}|{left},{right}]"
                    for derivative in range(4)
                    for left, right in pairs
                ],
                "absent",
            ),
            (
                "physical_metric_second",
                [
                    f"d2_g[{first},{second}|{left},{right}]"
                    for first, second in pairs
                    for left, right in pairs
                ],
                "absent_including_10_sourced_time_accelerations",
            ),
        ]
    )
    records: list[dict[str, Any]] = []
    for family, slot_ids, registration in families:
        record = {
            "family": family,
            "requested_scalar_values": len(slot_ids),
            "certified_general_values": 0,
            "registration_state": registration,
            "slot_ids": slot_ids,
            "slot_ids_sha256": _canonical_sha(slot_ids),
            "status": "BLOCK_MISSING_CERTIFIED_GENERAL_VALUES",
        }
        records.append(record)
    all_ids = [slot for record in records for slot in record["slot_ids"]]
    if len(all_ids) != 1010 or len(set(all_ids)) != 1010:
        raise System10GeneralMatterPDECompletionError("general scalar slot census changed")
    return records


def _validate_predecessors(bound: dict[str, tuple[Path, dict[str, Any]]]) -> None:
    expected = {
        "general_scalar_expansion_predecessor": (
            "decision",
            "BOUNDED_PASS_FLAT_SCALAR_ROWS_TYPED_BLOCK_GENERAL_EXTERNAL_JETS",
        ),
        "indexed_gauge_map": (
            "decision",
            "PASS_EXACT_INDEXED_GAUGE_MAP_WITH_FORMAL_EXTERNAL_JET_PACKETS",
        ),
        "constraint_basis": (
            "decision",
            "BOUNDED_PASS_KINEMATIC_MATTER_BASIS_TYPED_BLOCK_GRAVITY_COORDINATE_MAP",
        ),
        "off_shell_divergence": (
            "decision",
            "BOUNDED_PASS_COMMON_COVARIANT_IDENTITY_TYPED_BLOCK_DIFFERENTIATED_GAUGE_MAP",
        ),
        "sourced_metric_euler": (
            "decision",
            "PASS_SOURCED_METRIC_EULER_BINDING_ALL_TWELVE_ONLY",
        ),
        "coupled_principal": (
            "decision",
            "PASS_EXACT_NONZERO_MAXWELL_MIXED_BLOCK_AND_17_FIELD_PRINCIPAL",
        ),
        "first_order_reduction": (
            "decision",
            "PASS_EXACT_85_STATE_FIRST_ORDER_REDUCTION_ALL_TWELVE",
        ),
        "flat_bounded_symmetrizer": (
            "decision",
            "PASS_EXACT_FLAT_SPHERE_FULL_SYMMETRIZER_BOUNDED_B",
        ),
        "matter_interface": (
            "decision",
            "BOUNDED_PASS_MATTER_INTERFACE_WITH_TYPED_GRAVITY_BLOCK",
        ),
    }
    for name, (field, value) in expected.items():
        if bound[name][1].get(field) != value:
            raise System10GeneralMatterPDECompletionError(f"predecessor decision changed: {name}")

    scalar = bound["general_scalar_expansion_predecessor"][1]
    block = scalar.get("materialization", {}).get("general_expansion_block", {})
    if (
        scalar.get("counts", {}).get("required_general_scalar_values_before_domain") != 1010
        or block.get("total_exact_scalar_values_before_domain") != 1010
        or block.get("zero_fill_forbidden") is not True
    ):
        raise System10GeneralMatterPDECompletionError("1,010-value blocker changed")
    indexed = bound["indexed_gauge_map"][1]
    if indexed.get("counts", {}).get("formal_external_jet_atoms") != 580:
        raise System10GeneralMatterPDECompletionError("formal external atom count changed")
    basis = bound["constraint_basis"][1]
    if (
        basis.get("counts", {}).get("physical_gravity_constraint_rows_required") != 96
        or basis.get("counts", {}).get("physical_gravity_constraint_rows_registered") != 0
    ):
        raise System10GeneralMatterPDECompletionError("physical gravity row blocker changed")
    sourced = bound["sourced_metric_euler"][1]
    if (
        sourced.get("counts", {}).get("sourced_metric_euler_bindings_passed") != 12
        or sourced.get("counts", {}).get("sourced_acceleration_solutions") != 0
    ):
        raise System10GeneralMatterPDECompletionError("sourced Euler boundary changed")
    if (
        bound["coupled_principal"][1].get("counts", {}).get("completed_17_field_principal_matrices")
        != 12
    ):
        raise System10GeneralMatterPDECompletionError("coupled principal count changed")
    if bound["first_order_reduction"][1].get("counts", {}).get("reductions_passed") != 12:
        raise System10GeneralMatterPDECompletionError("85-state reduction count changed")
    if (
        bound["flat_bounded_symmetrizer"][1]
        .get("claims", {})
        .get("candidate_jet_uniformity_closed")
        is not False
    ):
        raise System10GeneralMatterPDECompletionError("flat symmetrizer scope changed")
    if (
        bound["matter_interface"][1]
        .get("claims", {})
        .get("common_time_matter_principal_compatibility_closed")
        is not True
    ):
        raise System10GeneralMatterPDECompletionError("matter common-time prerequisite changed")


def _value_packet_audit(slot_families: list[dict[str, Any]]) -> dict[str, Any]:
    groups = {
        "differentiated_external_formulation_jets": {
            "families": [
                "hat_inverse_first",
                "tilde_inverse_second",
                "reference_connection_second",
                "gauge_source_second",
            ],
            "requested": 580,
        },
        "lower_formulation_field_jets": {
            "families": [
                "hat_inverse_zero",
                "tilde_inverse_zero",
                "tilde_inverse_first",
                "reference_connection_zero",
                "reference_connection_first",
                "gauge_source_zero",
                "gauge_source_first",
            ],
            "requested": 280,
        },
        "physical_metric_two_jet": {
            "families": [
                "physical_metric_zero",
                "physical_metric_first",
                "physical_metric_second",
            ],
            "requested": 150,
        },
    }
    by_name = {item["family"]: item for item in slot_families}
    packets = []
    for packet, spec in groups.items():
        requested = sum(by_name[name]["requested_scalar_values"] for name in spec["families"])
        if requested != spec["requested"]:
            raise System10GeneralMatterPDECompletionError(f"packet slot count changed: {packet}")
        body = {
            "packet": packet,
            "families": spec["families"],
            "requested_scalar_values": requested,
            "certified_general_values": 0,
            "missing_general_values": requested,
            "status": "BLOCK_MISSING_CERTIFIED_VALUE_PACKET",
        }
        packets.append({**body, "packet_sha256": _canonical_sha(body)})
    return {
        "slot_families": slot_families,
        "slot_family_count": len(slot_families),
        "requested_scalar_values": 1010,
        "certified_general_values": 0,
        "missing_general_values": 1010,
        "packets": packets,
        "zero_fill_forbidden": True,
        "attempt_outcome": "BLOCK_EXTERNAL_FORMULATION_AND_SOURCED_METRIC_VALUES_UNREGISTERED",
    }


def _domain_audit() -> dict[str, Any]:
    requirements = [
        {
            "requirement": "matter_common_time_domain",
            "status": "PASS_PREDECESSOR",
            "evidence": "X>0 common-time covector for scalar, Lorenz-Maxwell, and P(X)=kappa X^2 fluid",
        },
        {
            "requirement": "flat_reference_bounded_Maxwell_potential",
            "status": "PASS_FLAT_REFERENCE_ONLY",
            "evidence": "max_mu |B_mu| <= 8/38505 with lower bound 1/28",
        },
        {
            "requirement": "all_1010_general_values_registered",
            "status": "BLOCK",
            "evidence": "0 of 1010 certified general scalar values",
        },
        {
            "requirement": "external_jet_uniform_bounds_and_compatibility",
            "status": "BLOCK",
            "evidence": "no prescribed auxiliary-field jet domain is registered",
        },
        {
            "requirement": "general_metric_Lorentzian_and_common_time_margin",
            "status": "BLOCK",
            "evidence": "flat diagonal metric is not a candidate-jet-uniform domain",
        },
        {
            "requirement": "general_symmetrizer_positive_lower_bound",
            "status": "BLOCK",
            "evidence": "the exact 1/28 bound is flat-reference only",
        },
    ]
    missing = [item["requirement"] for item in requirements if item["status"] == "BLOCK"]
    return {
        "requirements": requirements,
        "passed_subrequirements": 2,
        "blocked_subrequirements": len(missing),
        "missing_requirements": missing,
        "general_common_domain_closed": False,
        "status": "BLOCK_GENERAL_FORMULATION_JET_AND_POSITIVITY_DOMAIN_UNREGISTERED",
    }


def _physical_row_audit(constraint_basis: dict[str, Any]) -> dict[str, Any]:
    candidates = constraint_basis.get("materialization", {}).get("candidate_results", [])
    if len(candidates) != 12:
        raise System10GeneralMatterPDECompletionError("constraint candidate count changed")
    rows: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda item: item["candidate_id"]):
        for component in range(4):
            rows.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "row": f"modified_harmonic_C[{component}]",
                    "dependency": "general lower formulation and physical metric jet values",
                    "status": "BLOCK_COEFFICIENT_ROW_UNREGISTERED",
                }
            )
        for row in ("Hamiltonian_E_nn", "momentum_E_n1", "momentum_E_n2", "momentum_E_n3"):
            rows.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "row": row,
                    "dependency": "candidate sourced Euler coordinate projection and sourced acceleration solve",
                    "status": "BLOCK_COEFFICIENT_ROW_UNREGISTERED",
                }
            )
    if len(rows) != 96:
        raise System10GeneralMatterPDECompletionError("physical gravity row census changed")
    return {
        "required_rows": rows,
        "required_rows_sha256": _canonical_sha(rows),
        "rows_required": 96,
        "rows_closed": 0,
        "modified_harmonic_rows_required": 48,
        "hamiltonian_momentum_rows_required": 48,
        "status": "BLOCK_ALL_96_GENERAL_PHYSICAL_GRAVITY_ROWS_UNREGISTERED",
    }


def _negative_controls(physical_rows: list[dict[str, Any]]) -> dict[str, Any]:
    correct = Fraction(-1, 2)
    corrupted = Fraction(1, 2)
    sign_delta = corrupted - correct
    if sign_delta != 1:
        raise System10GeneralMatterPDECompletionError("source sign negative changed")
    domain_requirements = {
        "all_1010_general_values_registered",
        "external_jet_uniform_bounds_and_compatibility",
        "general_metric_Lorentzian_and_common_time_margin",
        "general_symmetrizer_positive_lower_bound",
    }
    corrupted_domain = domain_requirements - {"external_jet_uniform_bounds_and_compatibility"}
    missing_domain = sorted(domain_requirements - corrupted_domain)
    truncated_rows = physical_rows[:-1]
    missing_row = physical_rows[-1]
    if len(truncated_rows) != 95 or not missing_domain:
        raise System10GeneralMatterPDECompletionError("negative-control construction changed")
    return {
        "source_sign_corruption": {
            "mutation": "metric source normalization -1/2 -> +1/2",
            "sector_coefficient_deltas": [str(sign_delta)] * 3,
            "rejected": True,
            "scope": "source-normalization contract; not a completed subsidiary residual",
        },
        "domain_omission": {
            "mutation": "declare the common domain closed without external-jet bounds",
            "missing_requirements": missing_domain,
            "rejected": True,
        },
        "constraint_row_drop": {
            "mutation": "drop the final candidate momentum row from the 96-row census",
            "observed_rows": len(truncated_rows),
            "expected_rows": 96,
            "missing_row": missing_row,
            "rejected": True,
        },
    }


def _materialize(bound: dict[str, tuple[Path, dict[str, Any]]]) -> dict[str, Any]:
    _validate_predecessors(bound)
    slots = _slot_families()
    values = _value_packet_audit(slots)
    domain = _domain_audit()
    physical = _physical_row_audit(bound["constraint_basis"][1])
    negatives = _negative_controls(physical["required_rows"])
    hyperbolicity = {
        "matter_common_time_principal_compatibility": "PASS",
        "candidate_coupled_17_field_principals": 12,
        "candidate_85_state_first_order_reductions": 12,
        "flat_reference_bounded_B_symmetrizer": "PASS_LOWER_BOUND_1/28",
        "general_candidate_jet_symmetrizers": 0,
        "general_common_time_positive_domains": 0,
        "status": "BLOCK_GENERAL_CANDIDATE_JET_SYMMETRIZER_AND_COMMON_DOMAIN",
        "scientific_boundary": (
            "the arbitrary-point principal registration and flat-reference symmetrizer do not "
            "supply a candidate-jet-uniform positive symmetrizer"
        ),
    }
    propagation = {
        "common_off_shell_covariant_sourced_identity": "PASS_ALL_12",
        "combined_matter_stress_conservation": "PASS_ON_SHELL",
        "indexed_differentiated_gauge_map": "PASS_FORMAL_ATOMS_ONLY",
        "general_scalar_gauge_rows": 0,
        "general_physical_gravity_constraint_rows": 0,
        "candidate_subsidiary_factorizations": 0,
        "candidate_initial_constraint_surface_maps": 0,
        "status": "BLOCK_GENERAL_GAUGE_ROWS_SUBSIDIARY_FACTORIZATION_AND_INITIAL_DATA_MAP",
    }
    return {
        "general_scalar_value_attempt": values,
        "common_domain_attempt": domain,
        "physical_gravity_row_attempt": physical,
        "coupled_hyperbolicity_attempt": hyperbolicity,
        "sourced_constraint_propagation_attempt": propagation,
        "negative_controls": negatives,
        "terminal_blocker": {
            "reason_code": "external_formulation_values_sourced_accelerations_and_general_domain_unregistered",
            "first_missing_registration": (
                "a candidate-compatible value packet for all 1,010 scalar slots, including the "
                "10 sourced metric accelerations, with uniform jet bounds and compatibility"
            ),
            "resume_order": [
                "register_1010_exact_general_scalar_values",
                "certify_common_formulation_and_matter_domain",
                "expand_48_modified_harmonic_rows",
                "solve_sourced_accelerations_and_expand_48_ADM_rows",
                "construct_candidate_jet_uniform_symmetrizers",
                "factor_the_sourced_subsidiary_system_and_bind_initial_constraints",
            ],
        },
    }


def build_receipt(config_path: Path, *, root: Path | None = None) -> dict[str, Any]:
    repository = (root or config_path.resolve().parents[1]).resolve()
    config = _load_json(config_path)
    if (
        config.get("schema_version")
        != "invariant-system10-general-matter-pde-completion-config-1.0"
    ):
        raise System10GeneralMatterPDECompletionError("unsupported config schema")
    if config.get("expected_candidates") != 12 or config.get("expected_state_dimension") != 85:
        raise System10GeneralMatterPDECompletionError("candidate or state dimension changed")
    expected_policy = {
        "required_general_scalar_slot_manifest": True,
        "physical_gravity_row_dependency_census": True,
        "matter_common_time_subgate": True,
        "flat_reference_bounded_symmetrizer_subgate": True,
        "general_scalar_values_materialized": False,
        "general_common_domain_closed": False,
        "physical_gravity_rows_closed": False,
        "general_coupled_hyperbolicity_closed": False,
        "general_common_time_positivity_closed": False,
        "sourced_constraint_propagation_closed": False,
        "candidate_jet_uniformity": False,
        "nonlinear_global_closure": False,
        "gravity_h7": False,
        "universal_all_matter": False,
        "promotion": False,
    }
    if config.get("claims_policy") != expected_policy:
        raise System10GeneralMatterPDECompletionError("claims policy is absent or broadened")
    bound = {
        name: _load_binding(repository, binding)
        for name, binding in config.get("bindings", {}).items()
    }
    expected_bindings = {
        "general_scalar_expansion_predecessor",
        "indexed_gauge_map",
        "constraint_basis",
        "off_shell_divergence",
        "sourced_metric_euler",
        "coupled_principal",
        "first_order_reduction",
        "flat_bounded_symmetrizer",
        "matter_interface",
    }
    if set(bound) != expected_bindings:
        raise System10GeneralMatterPDECompletionError("closed binding manifest changed")
    materialization = _materialize(bound)
    source_path = Path(__file__).resolve()
    test_path = repository / "tests/test_system10_general_matter_pde_completion_gate.py"
    body: dict[str, Any] = {
        "schema_version": "invariant-system10-general-matter-pde-completion-result-1.0",
        "campaign_id": config["campaign_id"],
        "decision": "TYPED_BLOCK_GENERAL_FORMULATION_JETS_DOMAIN_AND_GRAVITY_ROWS_UNREGISTERED",
        "materialization": materialization,
        "counts": {
            "candidates": 12,
            "state_dimension": 85,
            "general_scalar_value_slots_manifested": 1010,
            "general_scalar_values_certified": 0,
            "general_scalar_values_missing": 1010,
            "physical_gravity_rows_required": 96,
            "physical_gravity_rows_closed": 0,
            "coupled_17_field_principals_closed": 12,
            "first_order_85_state_reductions_closed": 12,
            "general_candidate_jet_symmetrizers": 0,
            "general_common_time_positive_domains": 0,
            "sourced_constraint_propagation_passes": 0,
            "negative_controls": 3,
        },
        "claims": {
            "required_general_scalar_slot_manifest_closed": True,
            "physical_gravity_row_dependency_census_closed": True,
            "matter_common_time_subgate_closed": True,
            "flat_reference_bounded_symmetrizer_subgate_closed": True,
            "general_scalar_values_materialized": False,
            "general_common_domain_closed": False,
            "all_96_physical_gravity_rows_closed": False,
            "general_coupled_hyperbolicity_closed": False,
            "general_common_time_positivity_closed": False,
            "sourced_constraint_propagation_closed": False,
            "candidate_jet_uniformity_closed": False,
            "nonlinear_global_closure_established": False,
            "gravity_h7_theorem_established": False,
            "universal_all_matter_closure_established": False,
            "promotion_authorized": False,
        },
        "scope": (
            "End-to-end System 10 audit for the registered scalar, source-free Maxwell, and "
            "irrotational-fluid pack across all 12 quartic candidates. It deterministically "
            "manifests every one of the 1,010 required general scalar value slots and all 96 "
            "physical gravity constraint rows, but certifies no missing value or row: 580 jets "
            "remain formal atoms, 280 lower formulation jets and 150 physical two-jets are "
            "unregistered, and the latter include 10 unsolved sourced accelerations. Matter "
            "common-time compatibility, arbitrary-point coupled principals, exact 85-state "
            "reductions, and the bounded-B flat-reference symmetrizer remain valid subresults. "
            "They do not prove a general domain, candidate-jet-uniform positivity, sourced "
            "constraint propagation, nonlinear/global closure, H7, universal matter, or promotion."
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
