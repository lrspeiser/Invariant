from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import sympy as sp


class System10MaxwellDynamicRHSError(RuntimeError):
    """Raised when the cylindrical Lorenz-Maxwell row contract fails closed."""


def _canonical_sha(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("ascii")).hexdigest()


def _canonical_lf_sha(path: Path) -> str:
    try:
        raw = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    except OSError as exc:
        raise System10MaxwellDynamicRHSError(f"cannot read bound file: {path}") from exc
    return hashlib.sha256(raw).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise System10MaxwellDynamicRHSError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise System10MaxwellDynamicRHSError("bound JSON root is not an object")
    return value


def _resolve(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise System10MaxwellDynamicRHSError("bound path escapes repository root")
    return path


def _load_binding(root: Path, binding: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = _resolve(root, str(binding.get("path", "")))
    if _canonical_lf_sha(path) != binding.get("canonical_lf_sha256"):
        raise System10MaxwellDynamicRHSError(f"bound file hash mismatch: {path}")
    value = _load_json(path)
    if value.get("content_sha256") != binding.get("content_sha256"):
        raise System10MaxwellDynamicRHSError(f"bound content hash mismatch: {path}")
    return path, value


def _load_source(root: Path, binding: dict[str, Any]) -> Path:
    path = _resolve(root, str(binding.get("path", "")))
    if _canonical_lf_sha(path) != binding.get("canonical_lf_sha256"):
        raise System10MaxwellDynamicRHSError(f"bound source hash mismatch: {path}")
    return path


def _with_sha(body: dict[str, Any], key: str) -> dict[str, Any]:
    return {**body, key: _canonical_sha(body)}


def _row_terms(component: int) -> list[dict[str, str]]:
    radial_state = 46 + component
    angular_state = 63 + component
    axial_state = 80 + component
    terms = [
        {"coefficient": "1", "atom": f"partial_1 state[{radial_state}]"},
        {
            "coefficient": "-1/r" if component == 2 else "1/r",
            "atom": f"state[{radial_state}]",
        },
        {"coefficient": "1/r**2", "atom": f"partial_2 state[{angular_state}]"},
        {"coefficient": "1", "atom": f"partial_3 state[{axial_state}]"},
    ]
    if component == 1:
        terms.extend(
            [
                {"coefficient": "-1/r**2", "atom": "state[13]"},
                {"coefficient": "-2/r**3", "atom": "state[65]"},
            ]
        )
    if component == 2:
        terms.append({"coefficient": "2/r", "atom": "state[64]"})
    return terms


def _rhs_text(terms: list[dict[str, str]]) -> str:
    return "+".join(f"({term['coefficient']})*({term['atom']})" for term in terms)


def _maxwell_rows(
    action_sha: str, reduced_origin_sha: str, assembly_sha: str
) -> list[dict[str, Any]]:
    rows = []
    for component, field in enumerate(range(12, 16)):
        terms = _row_terms(component)
        origin = {
            "origin_type": "source_free_lorenz_reduced_maxwell_covector_component",
            "action_sector_id": "source_free_maxwell",
            "shared_matter_action_sha256": action_sha,
            "action_euler": "E_mu=nabla^nu F_nu_mu",
            "lorenz_constraint": "C=nabla^rho B_rho",
            "gauge_completion": "E_L_mu=E_mu+nabla_mu C",
            "reduced_equation": "box_g B_mu-R_mu^nu B_nu=0",
            "reduced_origin_sha256": reduced_origin_sha,
            "coordinate_metric": "diag(-1,1,r**2,1)",
            "curvature_specialization": "R_mu^nu=0",
            "component": f"B_{component}",
            "field_index": field,
            "predecessor_assembly_sha256": assembly_sha,
        }
        body = {
            "row_id": f"evolution_v[{field}]",
            "sector": "source_free_lorenz_maxwell",
            "component": f"B_{component}",
            "field_index": field,
            "lhs_state_index": 17 + field,
            "lhs": f"partial_0 state[{17 + field}]",
            "state_atoms": {
                "q": field,
                "v": 17 + field,
                "w1": 34 + field,
                "w2": 51 + field,
                "w3": 68 + field,
            },
            "rhs_terms": terms,
            "rhs": _rhs_text(terms),
            "solved_acceleration_certificate": {
                "unsolved_euler_lhs": f"-partial_0 state[{17 + field}]+({_rhs_text(terms)})",
                "acceleration_coefficient": "-1",
                "substitution_residual": "0",
                "maximum_coordinate_denominator_r_power": 3 if component == 1 else 2,
                "coordinate_pole_set": ["r=0"],
                "domain_excludes_all_poles": True,
            },
            "equation_origin": _with_sha(origin, "origin_sha256"),
            "candidate_dependence": "common_all_12",
            "domain": "r>0",
        }
        rows.append(_with_sha(body, "row_sha256"))
    return rows


def _independent_covariant_replay() -> dict[str, Any]:
    time, radius, angle, axis = sp.symbols("t r theta z", positive=True)
    coordinates = [time, radius, angle, axis]
    metric = sp.diag(-1, 1, radius**2, 1)
    inverse = metric.inv()
    potentials = [sp.Function(f"B_{index}")(*coordinates) for index in range(4)]
    christoffel = [
        [
            [
                sp.simplify(
                    sum(
                        inverse[upper, delta]
                        * (
                            sp.diff(metric[delta, second], coordinates[first])
                            + sp.diff(metric[delta, first], coordinates[second])
                            - sp.diff(metric[first, second], coordinates[delta])
                        )
                        for delta in range(4)
                    )
                    / 2
                )
                for second in range(4)
            ]
            for first in range(4)
        ]
        for upper in range(4)
    ]
    nonzero_christoffel = {
        f"Gamma^{upper}_{first}{second}": sp.sstr(christoffel[upper][first][second])
        for upper in range(4)
        for first in range(4)
        for second in range(4)
        if christoffel[upper][first][second] != 0
    }
    expected_christoffel = {
        "Gamma^1_22": "-r",
        "Gamma^2_12": "1/r",
        "Gamma^2_21": "1/r",
    }
    if nonzero_christoffel != expected_christoffel:
        raise System10MaxwellDynamicRHSError("cylindrical connection replay changed")

    ricci = sp.zeros(4)
    for first in range(4):
        for second in range(4):
            ricci[first, second] = sp.simplify(
                sum(
                    sp.diff(christoffel[upper][first][second], coordinates[upper])
                    - sp.diff(christoffel[upper][first][upper], coordinates[second])
                    + sum(
                        christoffel[upper][upper][lower] * christoffel[lower][first][second]
                        - christoffel[upper][second][lower] * christoffel[lower][first][upper]
                        for lower in range(4)
                    )
                    for upper in range(4)
                )
            )
    ricci_nonzero = sum(entry != 0 for entry in ricci)
    if ricci_nonzero != 0:
        raise System10MaxwellDynamicRHSError("cylindrical Ricci replay changed")

    first_covariant = [
        [
            sp.diff(potentials[component], coordinates[derivative])
            - sum(
                christoffel[upper][derivative][component] * potentials[upper] for upper in range(4)
            )
            for component in range(4)
        ]
        for derivative in range(4)
    ]
    solved = []
    for component in range(4):
        wave = sum(
            inverse[first, second]
            * (
                sp.diff(first_covariant[second][component], coordinates[first])
                - sum(
                    christoffel[upper][first][second] * first_covariant[upper][component]
                    for upper in range(4)
                )
                - sum(
                    christoffel[upper][first][component] * first_covariant[second][upper]
                    for upper in range(4)
                )
            )
            for first in range(4)
            for second in range(4)
        )
        acceleration = sp.solve(
            sp.Eq(sp.expand(wave), 0),
            sp.diff(potentials[component], time, 2),
        )[0]
        solved.append(sp.factor(acceleration))
    expected = [
        sp.diff(potentials[0], radius, 2)
        + sp.diff(potentials[0], radius) / radius
        + sp.diff(potentials[0], angle, 2) / radius**2
        + sp.diff(potentials[0], axis, 2),
        sp.diff(potentials[1], radius, 2)
        + sp.diff(potentials[1], radius) / radius
        + sp.diff(potentials[1], angle, 2) / radius**2
        + sp.diff(potentials[1], axis, 2)
        - potentials[1] / radius**2
        - 2 * sp.diff(potentials[2], angle) / radius**3,
        sp.diff(potentials[2], radius, 2)
        - sp.diff(potentials[2], radius) / radius
        + sp.diff(potentials[2], angle, 2) / radius**2
        + sp.diff(potentials[2], axis, 2)
        + 2 * sp.diff(potentials[1], angle) / radius,
        sp.diff(potentials[3], radius, 2)
        + sp.diff(potentials[3], radius) / radius
        + sp.diff(potentials[3], angle, 2) / radius**2
        + sp.diff(potentials[3], axis, 2),
    ]
    residuals = [
        sp.simplify(actual - reference) for actual, reference in zip(solved, expected, strict=True)
    ]
    if residuals != [0, 0, 0, 0]:
        raise System10MaxwellDynamicRHSError("covector wave component replay failed")
    return {
        "method": "independent_Levi_Civita_covector_box_expansion",
        "metric": "diag(-1,1,r**2,1)",
        "nonzero_christoffel": expected_christoffel,
        "ricci_tensor_nonzero_entries": ricci_nonzero,
        "components_solved": 4,
        "component_residuals": ["0", "0", "0", "0"],
        "all_residuals_zero": True,
        "derived_component_expressions": [sp.sstr(value) for value in solved],
    }


def _candidate_ids(value: dict[str, Any], key: str) -> list[str]:
    return sorted(str(item["candidate_id"]) for item in value.get(key, []))


def _validate_predecessors(
    bound: dict[str, tuple[Path, dict[str, Any]]],
) -> tuple[list[str], str, str, str]:
    matter = bound["matter_dynamic_rhs"][1]
    action = bound["total_matter_action"][1]
    interface = bound["matter_interface"][1]
    mixed = bound["maxwell_mixed_principal"][1]
    domain = bound["r_positive_domain"][1]
    if (
        matter.get("decision") != "BOUNDED_PASS_2_MATTER_DYNAMIC_ROWS_BLOCK_15_DYNAMIC_ROWS"
        or matter.get("counts", {}).get("total_rhs_rows_closed_per_candidate") != 70
        or matter.get("counts", {}).get("candidate_dynamic_rows_remaining") != 180
    ):
        raise System10MaxwellDynamicRHSError("70-row predecessor changed")
    components = action.get("shared_matter_action", {}).get("components", [])
    maxwell_action = next(
        (item for item in components if item.get("sector_id") == "source_free_maxwell"),
        None,
    )
    if (
        action.get("decision") != "PASS_TOTAL_ACTION_HASH_BINDING_ALL_TWELVE_ONLY"
        or not isinstance(maxwell_action, dict)
        or maxwell_action.get("field") != "B_mu"
        or maxwell_action.get("mass_term_removed") is not True
    ):
        raise System10MaxwellDynamicRHSError("Maxwell action authority changed")
    internal = interface.get("combined_matter_certificate", {}).get(
        "internal_matter_constraint_closure", {}
    )
    if (
        interface.get("decision") != "BOUNDED_PASS_MATTER_INTERFACE_WITH_TYPED_GRAVITY_BLOCK"
        or internal.get("maxwell_constraint") != "C=nabla_mu A^mu"
        or internal.get("subsidiary_equation")
        != "box_g C=0 for the source-free Lorenz-gauge system"
    ):
        raise System10MaxwellDynamicRHSError("Maxwell subsidiary authority changed")
    derivation = mixed.get("maxwell_metric_mixed_principal", {}).get("rnc_derivation", {})
    if (
        mixed.get("decision") != "PASS_EXACT_NONZERO_MAXWELL_MIXED_BLOCK_AND_17_FIELD_PRINCIPAL"
        or derivation.get("action_euler") != "E_nu=nabla^mu F_mu_nu"
        or derivation.get("lorenz_constraint") != "C=nabla^rho B_rho"
        or derivation.get("reduced_euler") != "E_L_nu=E_nu+nabla_nu C"
    ):
        raise System10MaxwellDynamicRHSError("Lorenz-reduced Euler origin changed")
    if (
        domain.get("decision")
        != "BOUNDED_PASS_FIXED_CYLINDRICAL_R_POSITIVE_1010_JETS_AND_96_ROWS_NO_PROPAGATION_CLAIM"
        or domain.get("materialization", {}).get("domain_certificate", {}).get("domain") != "r>0"
    ):
        raise System10MaxwellDynamicRHSError("r-positive domain authority changed")
    candidates = _candidate_ids(matter["materialization"], "candidate_results")
    if candidates != _candidate_ids(action, "candidate_results") or len(candidates) != 12:
        raise System10MaxwellDynamicRHSError("candidate identity join changed")
    state_contract = matter.get("materialization", {}).get("state_index_contract", {})
    if state_contract.get("matter_field_map", {}).get("B_mu") != [12, 13, 14, 15]:
        raise System10MaxwellDynamicRHSError("Maxwell field ordering changed")
    return (
        candidates,
        str(state_contract["predecessor_assembly_sha256"]),
        str(action["shared_matter_action_sha256"]),
        str(mixed["maxwell_metric_mixed_principal_sha256"]),
    )


def _gravity_blocks(previous_blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    retained = [item for item in previous_blocks if item.get("field_index") in range(11)]
    if [item.get("field_index") for item in retained] != list(range(11)):
        raise System10MaxwellDynamicRHSError("remaining gravity block set changed")
    return retained


def _materialize(
    bound: dict[str, tuple[Path, dict[str, Any]]], frozen: dict[str, Any]
) -> dict[str, Any]:
    candidates, assembly_sha, action_sha, reduced_sha = _validate_predecessors(bound)
    rows = _maxwell_rows(action_sha, reduced_sha, assembly_sha)
    replay = _independent_covariant_replay()
    previous_blocks = bound["matter_dynamic_rhs"][1]["materialization"]["missing_dynamic_rows"]
    remaining = _gravity_blocks(previous_blocks)
    measured = {
        "maxwell_rows_registered": len(rows),
        "registered_field_indices": [item["field_index"] for item in rows],
        "row_set_sha256": _canonical_sha([item["row_sha256"] for item in rows]),
        "equation_origin_set_sha256": _canonical_sha(
            [item["equation_origin"]["origin_sha256"] for item in rows]
        ),
        "remaining_rows": len(remaining),
        "remaining_block_set_sha256": _canonical_sha([item["block_sha256"] for item in remaining]),
    }
    if measured != frozen:
        raise System10MaxwellDynamicRHSError("frozen Maxwell dynamic expectations changed")
    manifests = []
    for candidate_id in candidates:
        body = {
            "candidate_id": candidate_id,
            "predecessor_rhs_rows": 70,
            "maxwell_dynamic_rows": 4,
            "total_rhs_rows_closed": 74,
            "dynamic_rows_remaining": 11,
            "full_85_state_rhs_closed": False,
            "maxwell_row_set_sha256": measured["row_set_sha256"],
            "outcome": "PASS_FOUR_MAXWELL_ROWS_BLOCK_11_GRAVITY_ROWS",
        }
        manifests.append(_with_sha(body, "manifest_sha256"))
    negatives = {
        "omit_B0_radial_connection": {
            "mutation": "omit state[46]/r from evolution_v[12]",
            "witness": {"r": "2", "state[46]": "3", "all_other_atoms": "0"},
            "exact_rhs_delta": "3/2",
            "rejected": True,
        },
        "omit_B1_algebraic_connection": {
            "mutation": "omit -state[13]/r**2 from evolution_v[13]",
            "witness": {"r": "2", "state[13]": "4", "all_other_atoms": "0"},
            "exact_rhs_delta": "1",
            "rejected": True,
        },
        "flip_B1_B2_cross_sign": {
            "mutation": "replace -2*state[65]/r**3 by +2*state[65]/r**3",
            "witness": {"r": "2", "state[65]": "4", "all_other_atoms": "0"},
            "exact_rhs_delta": "2",
            "rejected": True,
        },
        "flip_B2_radial_connection_sign": {
            "mutation": "replace -state[48]/r by +state[48]/r",
            "witness": {"r": "2", "state[48]": "3", "all_other_atoms": "0"},
            "exact_rhs_delta": "3",
            "rejected": True,
        },
        "omit_B2_B1_cross_connection": {
            "mutation": "omit 2*state[64]/r from evolution_v[14]",
            "witness": {"r": "2", "state[64]": "5", "all_other_atoms": "0"},
            "exact_rhs_delta": "5",
            "rejected": True,
        },
        "zero_fill_maxwell_rows": {
            "mutation": "replace all four Maxwell accelerations by zero",
            "nonzero_exact_component_rows": 4,
            "rejected": True,
        },
        "claim_full_rhs_or_propagation": {
            "closed_rows": 74,
            "required_rows": 85,
            "remaining_gravity_rows": 11,
            "constraint_propagation_proved_here": False,
            "rejected": True,
        },
    }
    return {
        "state_index_contract": {
            "B_mu_fields": [12, 13, 14, 15],
            "B_mu_velocity_states": [29, 30, 31, 32],
            "predecessor_assembly_sha256": assembly_sha,
        },
        "lorenz_reduced_origin": {
            "action_euler": "E_mu=nabla^nu F_nu_mu",
            "constraint": "C=nabla^rho B_rho",
            "completion": "E_L_mu=E_mu+nabla_mu C",
            "covector_wave": "box_g B_mu-R_mu^nu B_nu=0",
            "registered_mixed_principal_sha256": reduced_sha,
        },
        "rows": rows,
        "row_set_sha256": measured["row_set_sha256"],
        "equation_origin_set_sha256": measured["equation_origin_set_sha256"],
        "independent_covariant_replay": replay,
        "candidate_results": manifests,
        "remaining_dynamic_rows": remaining,
        "remaining_block_set_sha256": measured["remaining_block_set_sha256"],
        "negative_controls": negatives,
        "next_missing_primitive": {
            "required_rows_per_candidate": 11,
            "required_candidate_rows": 132,
            "sourced_metric_rows": 10,
            "candidate_gravity_scalar_rows": 1,
            "block_set_sha256": measured["remaining_block_set_sha256"],
            "status": "BLOCK_SOURCED_GRAVITY_ACCELERATION_PRIMITIVES_UNREGISTERED",
        },
    }


def build_receipt(config_path: Path, *, root: Path | None = None) -> dict[str, Any]:
    repository = (root or config_path.resolve().parents[1]).resolve()
    config = _load_json(config_path)
    if config.get("schema_version") != (
        "invariant-system10-cylindrical-r-positive-maxwell-dynamic-rhs-config-1.0"
    ):
        raise System10MaxwellDynamicRHSError("unsupported config schema")
    expected_caps = {
        "candidates": 12,
        "state_dimension": 85,
        "predecessor_rows": 70,
        "maxwell_dynamic_rows": 4,
        "candidate_row_instances": 48,
        "candidate_dynamic_rows_remaining": 132,
        "maximum_output_bytes": 524288,
    }
    if config.get("caps") != expected_caps:
        raise System10MaxwellDynamicRHSError("caps changed")
    expected_claims = {
        "four_lorenz_maxwell_dynamic_rows": True,
        "solved_acceleration_and_equation_origin": True,
        "fixed_cylindrical_r_positive": True,
        "gravity_dynamic_rows": False,
        "full_85_state_rhs": False,
        "constraint_propagation": False,
        "hyperbolicity": False,
        "global_theorem": False,
        "promotion": False,
    }
    if config.get("claims_policy") != expected_claims:
        raise System10MaxwellDynamicRHSError("claims policy broadened")
    bound = {
        name: _load_binding(repository, binding)
        for name, binding in config.get("bindings", {}).items()
    }
    if set(bound) != {
        "matter_dynamic_rhs",
        "total_matter_action",
        "matter_interface",
        "maxwell_mixed_principal",
        "r_positive_domain",
    }:
        raise System10MaxwellDynamicRHSError("binding manifest changed")
    sources = {
        name: _load_source(repository, binding)
        for name, binding in config.get("source_evidence", {}).items()
    }
    if set(sources) != {"source", "test"}:
        raise System10MaxwellDynamicRHSError("source evidence manifest changed")
    expected_test = (
        repository
        / "tests/test_system10_cylindrical_r_positive_maxwell_dynamic_rhs_materializer.py"
    )
    if sources["source"] != Path(__file__).resolve() or sources["test"] != expected_test:
        raise System10MaxwellDynamicRHSError("self evidence path changed")
    materialization = _materialize(bound, config.get("frozen_expectations", {}))
    body = {
        "schema_version": "invariant-system10-cylindrical-r-positive-maxwell-dynamic-rhs-result-1.0",
        "campaign_id": config["campaign_id"],
        "decision": "BOUNDED_PASS_4_MAXWELL_DYNAMIC_ROWS_BLOCK_11_GRAVITY_ROWS",
        "materialization": materialization,
        "counts": {
            "candidates": 12,
            "state_dimension": 85,
            "predecessor_rhs_rows_per_candidate": 70,
            "maxwell_dynamic_rows_registered": 4,
            "candidate_row_instances_registered": 48,
            "total_rhs_rows_closed_per_candidate": 74,
            "candidate_dynamic_rows_remaining": 132,
            "full_85_state_rhs_candidates_closed": 0,
            "equation_origins_registered": 4,
            "solved_acceleration_certificates": 4,
            "negative_controls": 7,
        },
        "claims": {
            "four_lorenz_maxwell_dynamic_rows_closed": True,
            "registered_rows_have_solved_acceleration_and_equation_origin": True,
            "fixed_cylindrical_r_positive_closed": True,
            "gravity_dynamic_rows_closed": False,
            "full_85_state_rhs_closed": False,
            "constraint_propagation_closed": False,
            "hyperbolicity_closed": False,
            "global_theorem_established": False,
            "promotion_authorized": False,
        },
        "scope": (
            "Exact solved fixed-cylindrical-r>0 acceleration rows for all four components "
            "of the registered source-free Lorenz-reduced Maxwell covector. The rows retain "
            "the radial, algebraic, and B_r-B_theta connection couplings, bind to the Maxwell "
            "action plus E_L=E+nabla C origin, and independently replay the Levi-Civita "
            "covector wave operator. Together with the predecessor this closes 74/85 RHS "
            "rows for each of twelve candidates. Ten sourced metric rows and one candidate "
            "gravity-scalar row remain blocked. No full evolution, constraint propagation, "
            "hyperbolicity, global, or promotion claim is made."
        ),
        "source_bindings": {
            "config": {
                "path": config_path.relative_to(repository).as_posix(),
                "canonical_json_sha256": _canonical_sha(config),
            },
            **{
                name: {
                    "path": path.relative_to(repository).as_posix(),
                    "canonical_lf_sha256": _canonical_lf_sha(path),
                    "content_sha256": value["content_sha256"],
                }
                for name, (path, value) in bound.items()
            },
            **{
                name: {
                    "path": path.relative_to(repository).as_posix(),
                    "canonical_lf_sha256": _canonical_lf_sha(path),
                }
                for name, path in sources.items()
            },
        },
    }
    receipt = {**body, "content_sha256": _canonical_sha(body)}
    if len(json.dumps(receipt).encode("utf-8")) > expected_caps["maximum_output_bytes"]:
        raise System10MaxwellDynamicRHSError("output cap exceeded")
    return receipt


def write_receipt(
    config_path: Path, output_path: Path, *, root: Path | None = None
) -> dict[str, Any]:
    receipt = build_receipt(config_path, root=root)
    data = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if output_path.exists() and output_path.read_bytes() != data:
        raise System10MaxwellDynamicRHSError("immutable output conflict")
    if not output_path.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(data)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    write_receipt(arguments.config.resolve(), arguments.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
