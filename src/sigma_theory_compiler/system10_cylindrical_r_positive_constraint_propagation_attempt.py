from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import sympy as sp


class System10CylindricalPropagationError(RuntimeError):
    """Raised when the fixed-profile propagation audit fails closed."""


def _canonical_sha(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("ascii")).hexdigest()


def _file_sha(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise System10CylindricalPropagationError(f"cannot read bound file: {path}") from exc


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise System10CylindricalPropagationError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise System10CylindricalPropagationError("bound JSON root is not an object")
    return value


def _resolve(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise System10CylindricalPropagationError("bound path escapes repository root")
    return path


def _load_binding(root: Path, binding: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = _resolve(root, str(binding.get("path", "")))
    if _file_sha(path) != binding.get("file_sha256"):
        raise System10CylindricalPropagationError(f"bound file hash mismatch: {path}")
    value = _load_json(path)
    if value.get("content_sha256") != binding.get("content_sha256"):
        raise System10CylindricalPropagationError(f"bound content hash mismatch: {path}")
    return path, value


def _load_source(root: Path, binding: dict[str, Any]) -> Path:
    path = _resolve(root, str(binding.get("path", "")))
    if _file_sha(path) != binding.get("file_sha256"):
        raise System10CylindricalPropagationError(f"bound source hash mismatch: {path}")
    return path


def _candidate_ids(value: dict[str, Any], path: tuple[str, ...]) -> set[str]:
    current: Any = value
    for key in path:
        current = current[key]
    return {str(item["candidate_id"]) for item in current}


def _validate_predecessors(bound: dict[str, tuple[Path, dict[str, Any]]]) -> list[str]:
    domain = bound["r_positive_rows"][1]
    off_shell = bound["off_shell_identity"][1]
    differentiated = bound["indexed_divergence_map"][1]
    first_order = bound["principal_85_state_reduction"][1]
    historical = bound["historical_propagation_block"][1]
    if (
        domain.get("decision")
        != "BOUNDED_PASS_FIXED_CYLINDRICAL_R_POSITIVE_1010_JETS_AND_96_ROWS_NO_PROPAGATION_CLAIM"
        or domain.get("counts", {}).get("physical_gravity_rows_closed") != 96
        or domain.get("counts", {}).get("formulation_jet_rational_functions") != 1010
    ):
        raise System10CylindricalPropagationError("r-positive row authority changed")
    if (
        off_shell.get("decision")
        != "BOUNDED_PASS_COMMON_COVARIANT_IDENTITY_TYPED_BLOCK_DIFFERENTIATED_GAUGE_MAP"
        or off_shell.get("counts", {}).get("candidate_common_formula_hashes") != 12
    ):
        raise System10CylindricalPropagationError("off-shell identity authority changed")
    if (
        differentiated.get("decision")
        != "PASS_EXACT_INDEXED_GAUGE_MAP_WITH_FORMAL_EXTERNAL_JET_PACKETS"
        or differentiated.get("counts", {}).get("formal_external_jet_atoms") != 580
        or differentiated.get("counts", {}).get("physical_metric_third_operator_slots") != 200
        or differentiated.get("counts", {}).get("fully_expanded_85_state_coefficient_rows") != 0
    ):
        raise System10CylindricalPropagationError("indexed divergence authority changed")
    if (
        first_order.get("decision") != "PASS_EXACT_85_STATE_FIRST_ORDER_REDUCTION_ALL_TWELVE"
        or first_order.get("counts", {}).get("first_order_states_per_candidate") != 85
        or first_order.get("counts", {}).get("constraint_propagation_claims") != 0
    ):
        raise System10CylindricalPropagationError("85-state principal authority changed")
    if historical.get("decision") != (
        "TYPED_BLOCK_CANDIDATE_GRAVITY_CONSTRAINT_JET_DIVERGENCE_UNREGISTERED"
    ):
        raise System10CylindricalPropagationError("historical propagation block changed")
    candidate_sets = [
        _candidate_ids(domain, ("materialization", "candidate_results")),
        _candidate_ids(off_shell, ("materialization", "candidate_results")),
        _candidate_ids(differentiated, ("materialization", "candidate_results")),
        _candidate_ids(first_order, ("candidate_results",)),
    ]
    if any(candidate_set != candidate_sets[0] for candidate_set in candidate_sets[1:]):
        raise System10CylindricalPropagationError("candidate identity join changed")
    if len(candidate_sets[0]) != 12:
        raise System10CylindricalPropagationError("candidate census changed")
    return sorted(candidate_sets[0])


def _external_jet_closure(domain: dict[str, Any], indexed: dict[str, Any]) -> dict[str, Any]:
    packet = domain["materialization"]["formulation_jet_rational_functions"]
    families = {item["family"]: item for item in packet["families"]}
    schema = indexed["materialization"]["primitive_slot_schema"]
    expected = {
        "hat_inverse_first": 40,
        "tilde_inverse_second": 100,
        "reference_connection_second": 400,
        "gauge_source_second": 40,
    }
    registered = {}
    for family, count in expected.items():
        if families.get(family, {}).get("scalar_values") != count:
            raise System10CylindricalPropagationError(f"external jet family changed: {family}")
        if schema.get(family, {}).get("slot_count") != count:
            raise System10CylindricalPropagationError(f"indexed slot family changed: {family}")
        registered[family] = {
            "required": count,
            "r_positive_rational_values": count,
            "indexed_program_slots": count,
            "closed": True,
        }
    body = {
        "domain": "r>0",
        "external_formulation_jet_slots_required": 580,
        "external_formulation_jet_slots_valued": sum(
            item["r_positive_rational_values"] for item in registered.values()
        ),
        "families": registered,
        "maximum_input_denominator_r_power": packet["maximum_denominator_r_power"],
        "axis_r_zero_excluded": True,
    }
    return {**body, "closure_sha256": _canonical_sha(body)}


def _coefficient_at_unit_point(text: str) -> sp.Expr:
    r = sp.Symbol("r", positive=True)
    kappa = sp.Symbol("kappa", positive=True)
    value = sp.sympify(text, locals={"r": r, "kappa": kappa})
    return sp.factor(value.subs({r: 1, kappa: 1}))


def _lower_order_nonidentifiability_witness(domain: dict[str, Any]) -> dict[str, Any]:
    packets = domain["materialization"]["sourced_rational_rows"]
    for packet in packets:
        parsed = [
            (
                _coefficient_at_unit_point(term["coefficient"]),
                {factor["atom"]: int(factor["power"]) for factor in term["factors"]},
            )
            for term in packet["terms"]
        ]
        for coefficient, factors in parsed:
            targets = sorted(atom for atom in factors if atom.startswith("v_"))
            for target in targets:
                active = set(factors)
                derivative = sp.Integer(0)
                for other_coefficient, other_factors in parsed:
                    target_power = other_factors.get(target, 0)
                    if target_power and set(other_factors).issubset(active):
                        derivative += other_coefficient * target_power
                derivative = sp.factor(derivative)
                if derivative != 0:
                    body = {
                        "candidate_id": packet["candidate_id"],
                        "constraint_row": packet["row"],
                        "state_coordinate": target,
                        "evaluation": {
                            "r": "1",
                            "kappa": "1",
                            "active_atoms_equal_one": sorted(active),
                            "all_other_atoms_equal_zero": True,
                        },
                        "completion_0_lower_order_increment": "0",
                        "completion_1_lower_order_increment": f"partial_0({target}) += 1",
                        "principal_A_B_C_change": "0",
                        "constraint_time_derivative_delta": sp.sstr(derivative),
                        "nonzero": True,
                        "meaning": (
                            "the registered principal 85-state authority admits both algebraic "
                            "lower-order completions; the receipts contain no equation-origin map "
                            "that selects the physical completion"
                        ),
                    }
                    return {**body, "witness_sha256": _canonical_sha(body)}
    raise System10CylindricalPropagationError("no lower-order nonidentifiability witness found")


def _materialize(
    bound: dict[str, tuple[Path, dict[str, Any]]], caps: dict[str, int]
) -> dict[str, Any]:
    candidate_ids = _validate_predecessors(bound)
    domain = bound["r_positive_rows"][1]
    off_shell = bound["off_shell_identity"][1]
    indexed = bound["indexed_divergence_map"][1]
    first_order = bound["principal_85_state_reduction"][1]
    external = _external_jet_closure(domain, indexed)
    if external["external_formulation_jet_slots_valued"] != caps["external_jet_slots"]:
        raise System10CylindricalPropagationError("external jet closure count changed")
    program = indexed["materialization"]["indexed_formula_program"]
    common_formula = off_shell["materialization"]["common_formula"]
    if "nabla Q remains an unevaluated covariant divergence" not in common_formula["scope"]:
        raise System10CylindricalPropagationError("off-shell scientific boundary changed")
    if program["output_components"] != [
        "divQ_lower[0]",
        "divQ_lower[1]",
        "divQ_lower[2]",
        "divQ_lower[3]",
    ]:
        raise System10CylindricalPropagationError("divQ output convention changed")
    witness = _lower_order_nonidentifiability_witness(domain)
    first_missing = {
        "primitive": (
            "sourced_cylindrical_r_positive_divQ_lower_four_component_85_state_"
            "rational_differential_operator_rows"
        ),
        "required_rows": 4,
        "registered_rows": indexed["counts"]["fully_expanded_85_state_coefficient_rows"],
        "inputs_already_closed": {
            "indexed_formula_program_sha256": program["program_sha256"],
            "r_positive_external_jet_slots": 580,
            "physical_metric_third_85_operator_slots": 200,
            "domain": "r>0",
        },
        "acceptance": {
            "output_components": program["output_components"],
            "coefficient_domain": "exact rational functions on r>0",
            "denominator_zero_set_must_be_subset_of": ["r=0"],
            "zero_fill_forbidden": True,
            "candidate_manifests": 12,
            "exact_r1_replay_required": True,
        },
        "status": "BLOCK_UNMATERIALIZED",
    }
    next_missing = {
        "primitive": (
            "sourced_cylindrical_r_positive_candidate_full_85_state_evolution_rhs_"
            "with_equation_origin_map"
        ),
        "principal_rows_registered": 85,
        "full_lower_order_rhs_rows_registered": 0,
        "required": (
            "all 85 nonlinear lower-order evolution components, including candidate metric, "
            "gravity-scalar, scalar, Maxwell, and fluid equations, bound back to the sourced "
            "Euler rows and sufficient for the exact chain-rule time derivative of all 96 "
            "constraint rows"
        ),
        "exact_nonidentifiability_witness_sha256": witness["witness_sha256"],
        "status": "BLOCK_PRINCIPAL_ONLY",
    }
    candidate_attempts = []
    for candidate_id in candidate_ids:
        body = {
            "candidate_id": candidate_id,
            "r_positive_constraint_rows": 8,
            "off_shell_common_identity_bound": True,
            "indexed_divQ_program_bound": True,
            "external_formulation_jets_valued": 580,
            "fully_expanded_divQ_rows": 0,
            "full_85_state_lower_order_rhs_rows": 0,
            "subsidiary_system_closed": False,
            "energy_estimate_closed": False,
            "outcome": "BLOCK_FIRST_MISSING_DIVQ_RATIONAL_OPERATOR_ROWS",
        }
        candidate_attempts.append({**body, "manifest_sha256": _canonical_sha(body)})
    energy = {
        "domain_denominators_certified": True,
        "domain": "r>0",
        "maximum_available_input_denominator_r_power": domain["materialization"][
            "domain_certificate"
        ]["maximum_denominator_r_power"],
        "subsidiary_operator_available": False,
        "subsidiary_mass_matrix_available": False,
        "boundary_flux_available": False,
        "energy_functional_constructed": False,
        "reason": "an 85-state evolution symmetrizer cannot be reused as an unbuilt subsidiary-system energy",
    }
    negatives = {
        "infer_divQ_zero": {
            "mutation": "replace the unevaluated covariant divergence nabla Q by zero",
            "indexed_output_components_discarded": 4,
            "rejected": True,
        },
        "zero_fill_physical_third_operator": {
            "mutation": "set all physical metric third differential operators to zero",
            "registered_operator_slots_discarded": 200,
            "rejected": True,
        },
        "treat_principal_matrix_as_full_rhs": {
            "mutation": "set every unregistered lower-order evolution term to zero",
            "witness_constraint_time_derivative_delta": witness["constraint_time_derivative_delta"],
            "rejected": True,
        },
        "include_axis": {
            "mutation": "replace r>0 by r>=0",
            "denominator_zero_set": ["r=0"],
            "rejected": True,
        },
        "borrow_parent_energy": {
            "mutation": "reuse an 85-state evolution energy as a subsidiary-system energy",
            "subsidiary_operator_constructed": False,
            "rejected": True,
        },
    }
    return {
        "closed_inputs": {
            "r_positive_formulation_and_row_authority": True,
            "physical_gravity_rows": 96,
            "off_shell_common_covariant_identity": True,
            "indexed_divQ_tensor_program": True,
            "principal_85_state_reductions": 12,
            "external_jet_closure": external,
        },
        "candidate_attempts": candidate_attempts,
        "first_missing_primitive": {
            **first_missing,
            "primitive_sha256": _canonical_sha(first_missing),
        },
        "next_missing_primitive": {
            **next_missing,
            "primitive_sha256": _canonical_sha(next_missing),
        },
        "lower_order_nonidentifiability_witness": witness,
        "energy_control": {**energy, "control_sha256": _canonical_sha(energy)},
        "negative_controls": negatives,
        "attempt_chain_sha256": _canonical_sha(
            [item["manifest_sha256"] for item in candidate_attempts]
        ),
        "principal_reduction_assembly_sha256": first_order["reduction_certificate"][
            "assembly_sha256"
        ],
    }


def build_receipt(config_path: Path, *, root: Path | None = None) -> dict[str, Any]:
    repository = (root or config_path.resolve().parents[1]).resolve()
    config = _load_json(config_path)
    if config.get("schema_version") != (
        "invariant-system10-cylindrical-r-positive-constraint-propagation-attempt-config-1.0"
    ):
        raise System10CylindricalPropagationError("unsupported config schema")
    expected_caps = {
        "candidates": 12,
        "state_dimension": 85,
        "physical_gravity_rows": 96,
        "external_jet_slots": 580,
        "physical_metric_third_operator_slots": 200,
        "maximum_output_bytes": 1048576,
    }
    if config.get("caps") != expected_caps:
        raise System10CylindricalPropagationError("caps changed")
    expected_domain = {
        "predicate": "r>0",
        "excluded_axis": "r=0",
        "fixed_physical_metric": "diag(-1,1,r^2,1)",
        "arbitrary_formulation_functions": False,
    }
    if config.get("domain_contract") != expected_domain:
        raise System10CylindricalPropagationError("domain contract changed")
    expected_claims = {
        "r_positive_input_rows": True,
        "external_formulation_jet_values": True,
        "candidate_bound_subsidiary_system": False,
        "sourced_constraint_propagation": False,
        "subsidiary_energy_estimate": False,
        "arbitrary_formulation_functions": False,
        "general_hyperbolicity": False,
        "global_theorem": False,
        "promotion": False,
    }
    if config.get("claims_policy") != expected_claims:
        raise System10CylindricalPropagationError("claims policy broadened")
    bound = {
        name: _load_binding(repository, binding)
        for name, binding in config.get("bindings", {}).items()
    }
    if set(bound) != {
        "r_positive_rows",
        "off_shell_identity",
        "indexed_divergence_map",
        "principal_85_state_reduction",
        "historical_propagation_block",
    }:
        raise System10CylindricalPropagationError("binding manifest changed")
    sources = {
        name: _load_source(repository, binding)
        for name, binding in config.get("source_evidence", {}).items()
    }
    if set(sources) != {
        "r_positive_source",
        "off_shell_source",
        "indexed_divergence_source",
        "principal_reduction_source",
        "source",
        "test",
    }:
        raise System10CylindricalPropagationError("source evidence manifest changed")
    expected_test = (
        repository / "tests/test_system10_cylindrical_r_positive_constraint_propagation_attempt.py"
    )
    if sources["source"] != Path(__file__).resolve() or sources["test"] != expected_test:
        raise System10CylindricalPropagationError("self evidence path changed")
    materialization = _materialize(bound, expected_caps)
    body = {
        "schema_version": (
            "invariant-system10-cylindrical-r-positive-constraint-propagation-attempt-result-1.0"
        ),
        "campaign_id": config["campaign_id"],
        "decision": "TYPED_BLOCK_R_POSITIVE_DIVQ_ROWS_AND_FULL_EVOLUTION_UNREGISTERED",
        "materialization": materialization,
        "counts": {
            "candidates": 12,
            "physical_gravity_rows_available": 96,
            "external_formulation_jet_slots_valued": 580,
            "physical_metric_third_operator_slots_registered": 200,
            "indexed_divQ_programs": 1,
            "fully_expanded_divQ_rows_required": 4,
            "fully_expanded_divQ_rows_registered": 0,
            "principal_85_state_reductions": 12,
            "full_lower_order_85_state_rhs_rows_registered": 0,
            "candidate_subsidiary_systems_closed": 0,
            "sourced_constraint_propagation_proofs": 0,
            "subsidiary_energy_estimates": 0,
            "negative_controls": 5,
        },
        "claims": {
            "r_positive_rows_and_domain_closed": True,
            "external_formulation_jet_values_closed": True,
            "candidate_bound_subsidiary_system_closed": False,
            "sourced_constraint_propagation_closed": False,
            "subsidiary_energy_estimate_closed": False,
            "arbitrary_formulation_functions_closed": False,
            "general_hyperbolicity_closed": False,
            "global_theorem_established": False,
            "promotion_authorized": False,
        },
        "scope": (
            "Fail-closed propagation attempt on the exact fixed cylindrical r>0 profile. The "
            "96 constraint rows, all 580 external formulation-jet values required by the indexed "
            "divQ program, its 200 physical-third 85-state operator slots, the common off-shell "
            "identity, and all 12 principal 85-state reductions are bound. Propagation is not "
            "proved: the indexed divQ program has zero of four fully expanded rational operator "
            "rows, and the 85-state authority has no full nonlinear lower-order RHS/equation-origin "
            "map. An exact two-completion witness shows that the principal authority alone does "
            "not determine constraint time derivatives. No arbitrary-formulation, general "
            "hyperbolicity, global, energy, or promotion claim follows."
        ),
        "source_bindings": {
            "config": {
                "path": config_path.relative_to(repository).as_posix(),
                "canonical_json_sha256": _canonical_sha(config),
            },
            **{
                name: {
                    "path": path.relative_to(repository).as_posix(),
                    "file_sha256": _file_sha(path),
                    "content_sha256": value["content_sha256"],
                }
                for name, (path, value) in bound.items()
            },
            **{
                name: {
                    "path": path.relative_to(repository).as_posix(),
                    "file_sha256": _file_sha(path),
                }
                for name, path in sources.items()
            },
        },
    }
    receipt = {**body, "content_sha256": _canonical_sha(body)}
    if len(json.dumps(receipt).encode("utf-8")) > expected_caps["maximum_output_bytes"]:
        raise System10CylindricalPropagationError("output cap exceeded")
    return receipt


def write_receipt(
    config_path: Path, output_path: Path, *, root: Path | None = None
) -> dict[str, Any]:
    receipt = build_receipt(config_path, root=root)
    data = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if output_path.exists() and output_path.read_bytes() != data:
        raise System10CylindricalPropagationError("immutable output conflict")
    if not output_path.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(data)
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
