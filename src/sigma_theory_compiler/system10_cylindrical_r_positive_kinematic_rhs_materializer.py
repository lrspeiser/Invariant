from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


class System10KinematicRHSError(RuntimeError):
    """Raised when the bounded 85-state kinematic RHS contract fails closed."""


def _canonical_sha(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("ascii")).hexdigest()


def _canonical_lf_sha(path: Path) -> str:
    try:
        raw = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    except OSError as exc:
        raise System10KinematicRHSError(f"cannot read bound file: {path}") from exc
    return hashlib.sha256(raw).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise System10KinematicRHSError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise System10KinematicRHSError("bound JSON root is not an object")
    return value


def _resolve(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise System10KinematicRHSError("bound path escapes repository root")
    return path


def _load_binding(root: Path, binding: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = _resolve(root, str(binding.get("path", "")))
    if _canonical_lf_sha(path) != binding.get("canonical_lf_sha256"):
        raise System10KinematicRHSError(f"bound file hash mismatch: {path}")
    value = _load_json(path)
    if value.get("content_sha256") != binding.get("content_sha256"):
        raise System10KinematicRHSError(f"bound content hash mismatch: {path}")
    return path, value


def _load_source(root: Path, binding: dict[str, Any]) -> Path:
    path = _resolve(root, str(binding.get("path", "")))
    if _canonical_lf_sha(path) != binding.get("canonical_lf_sha256"):
        raise System10KinematicRHSError(f"bound source hash mismatch: {path}")
    return path


def _rhs_term(state_index: int, spatial_derivatives: list[int]) -> dict[str, Any]:
    body = {
        "coefficient": "1",
        "state_index": state_index,
        "spatial_derivatives": spatial_derivatives,
    }
    return {**body, "term_sha256": _canonical_sha(body)}


def _kinematic_rows(assembly_sha256: str) -> list[dict[str, Any]]:
    rows = []
    for field in range(17):
        origin = {
            "origin_type": "state_definition",
            "registered_definition": "v_A=partial_0 q_A",
            "field_index": field,
            "predecessor_assembly_sha256": assembly_sha256,
        }
        body = {
            "row_id": f"evolution_q[{field}]",
            "lhs_state_index": field,
            "lhs": f"partial_0 state[{field}]",
            "rhs_terms": [_rhs_term(17 + field, [])],
            "rhs": f"state[{17 + field}]",
            "equation_origin": {**origin, "origin_sha256": _canonical_sha(origin)},
            "candidate_dependence": "common_all_12",
            "domain": "r>0",
            "maximum_denominator_r_power": 0,
        }
        rows.append({**body, "row_sha256": _canonical_sha(body)})
    for spatial_coordinate in range(1, 4):
        for field in range(17):
            lhs_state = 34 + (spatial_coordinate - 1) * 17 + field
            origin = {
                "origin_type": "commuting_coordinate_partial_integrability",
                "registered_definitions": [
                    "w_iA=partial_i q_A",
                    "v_A=partial_0 q_A",
                ],
                "identity": "partial_0 partial_i q_A=partial_i partial_0 q_A",
                "spatial_coordinate": spatial_coordinate,
                "field_index": field,
                "predecessor_assembly_sha256": assembly_sha256,
            }
            body = {
                "row_id": f"evolution_w[{spatial_coordinate},{field}]",
                "lhs_state_index": lhs_state,
                "lhs": f"partial_0 state[{lhs_state}]",
                "rhs_terms": [_rhs_term(17 + field, [spatial_coordinate])],
                "rhs": f"partial_{spatial_coordinate} state[{17 + field}]",
                "equation_origin": {**origin, "origin_sha256": _canonical_sha(origin)},
                "candidate_dependence": "common_all_12",
                "domain": "r>0",
                "maximum_denominator_r_power": 0,
            }
            rows.append({**body, "row_sha256": _canonical_sha(body)})
    return rows


def _candidate_ids(value: dict[str, Any], key: str) -> list[str]:
    records = value.get(key, [])
    return sorted(str(item["candidate_id"]) for item in records)


def _validate_predecessors(bound: dict[str, tuple[Path, dict[str, Any]]]) -> tuple[list[str], str]:
    divq = bound["divq_rows"][1]
    first_order = bound["first_order_reduction"][1]
    domain = bound["r_positive_domain"][1]
    sourced = bound["sourced_metric_euler"][1]
    if (
        divq.get("decision") != "BOUNDED_PASS_FOUR_R_POSITIVE_DIVQ_ROWS_BLOCK_FULL_EVOLUTION_RHS"
        or divq.get("counts", {}).get("divq_rows_registered") != 4
        or divq.get("counts", {}).get("full_nonlinear_85_state_rhs_rows_registered") != 0
    ):
        raise System10KinematicRHSError("divQ predecessor changed")
    certificate = first_order.get("reduction_certificate", {})
    expected_assembly = {
        "state_order": ["q_A", "v_A", "w_1A", "w_2A", "w_3A"],
        "mass_diagonal_blocks": ["I17", "A17", "I17", "I17", "I17"],
        "evolution_nonzero_blocks": {
            "K_vv": "-sum_i n_i B_i",
            "K_vw_j": "-sum_i n_i C_ij",
            "K_w_i_v": "n_i I17",
        },
    }
    if (
        first_order.get("decision") != "PASS_EXACT_85_STATE_FIRST_ORDER_REDUCTION_ALL_TWELVE"
        or certificate.get("assembly") != expected_assembly
        or certificate.get("state")
        != {
            "q_A": 17,
            "v_A": 17,
            "w_iA": 51,
            "total": 85,
        }
        or certificate.get("corruption_negative", {}).get("rejected") is not True
    ):
        raise System10KinematicRHSError("85-state reduction authority changed")
    if (
        domain.get("decision")
        != "BOUNDED_PASS_FIXED_CYLINDRICAL_R_POSITIVE_1010_JETS_AND_96_ROWS_NO_PROPAGATION_CLAIM"
        or domain.get("materialization", {}).get("domain_certificate", {}).get("domain") != "r>0"
    ):
        raise System10KinematicRHSError("r-positive domain authority changed")
    claims = sourced.get("claims", {})
    if (
        sourced.get("decision") != "PASS_SOURCED_METRIC_EULER_BINDING_ALL_TWELVE_ONLY"
        or sourced.get("counts", {}).get("sourced_acceleration_solutions") != 0
        or claims.get("sourced_acceleration_solution_closed") is not False
        or claims.get("matter_field_euler_component_expansion_closed") is not False
    ):
        raise System10KinematicRHSError("sourced Euler boundary changed")
    candidate_sets = [
        _candidate_ids(first_order, "candidate_results"),
        _candidate_ids(sourced, "candidate_results"),
        _candidate_ids(divq["materialization"], "candidate_results"),
    ]
    if any(items != candidate_sets[0] for items in candidate_sets[1:]):
        raise System10KinematicRHSError("candidate identity join changed")
    if len(candidate_sets[0]) != 12 or len(set(candidate_sets[0])) != 12:
        raise System10KinematicRHSError("candidate census changed")
    return candidate_sets[0], str(certificate["assembly_sha256"])


def _materialize(
    bound: dict[str, tuple[Path, dict[str, Any]]], frozen: dict[str, Any]
) -> dict[str, Any]:
    candidate_ids, assembly_sha = _validate_predecessors(bound)
    rows = _kinematic_rows(assembly_sha)
    row_hashes = [item["row_sha256"] for item in rows]
    measured = {
        "row_count": len(rows),
        "q_definition_rows": sum(item["row_id"].startswith("evolution_q") for item in rows),
        "w_integrability_rows": sum(item["row_id"].startswith("evolution_w") for item in rows),
        "row_set_sha256": _canonical_sha(row_hashes),
        "equation_origin_set_sha256": _canonical_sha(
            [item["equation_origin"]["origin_sha256"] for item in rows]
        ),
    }
    if measured != frozen:
        raise System10KinematicRHSError("frozen kinematic expectations changed")
    manifests = []
    for candidate_id in candidate_ids:
        body = {
            "candidate_id": candidate_id,
            "common_kinematic_row_set_sha256": measured["row_set_sha256"],
            "complete_kinematic_rhs_rows": 68,
            "missing_candidate_dynamic_v_rows": 17,
            "full_85_state_rhs_closed": False,
            "outcome": "PASS_68_COMMON_KINEMATIC_ROWS_BLOCK_17_DYNAMIC_ROWS",
        }
        manifests.append({**body, "manifest_sha256": _canonical_sha(body)})
    witness_body = {
        "candidate_id": candidate_ids[0],
        "dynamic_row": "evolution_v[0]",
        "completion_0_lower_order_increment": "0",
        "completion_1_lower_order_increment": "q_0**2",
        "shared_registered_principal_A_Bi_Cij": True,
        "shared_68_kinematic_rows": True,
        "sourced_acceleration_solution_registered": False,
        "evaluation": {"q_0": "1", "all_other_state_and_derivative_atoms": "0"},
        "exact_rhs_delta": "1",
        "nonzero": True,
        "meaning": (
            "an algebraic increment changes no registered principal coefficient and no "
            "kinematic definition; the current sourced Euler authority supplies no solved "
            "85-state acceleration row that selects between these completions"
        ),
    }
    witness = {**witness_body, "witness_sha256": _canonical_sha(witness_body)}
    dropped_hashes = row_hashes[:-1]
    corrupted_w = dict(rows[17])
    corrupted_w["rhs_terms"] = [_rhs_term(17, [2])]
    corrupted_w["rhs"] = "partial_2 state[17]"
    corrupted_w.pop("row_sha256")
    corrupted_w_sha = _canonical_sha(corrupted_w)
    corrupted_hashes = list(row_hashes)
    corrupted_hashes[17] = corrupted_w_sha
    negatives = {
        "omit_last_integrability_row": {
            "observed_rows": 67,
            "expected_rows": 68,
            "mutated_row_set_sha256": _canonical_sha(dropped_hashes),
            "expected_row_set_sha256": measured["row_set_sha256"],
            "rejected": _canonical_sha(dropped_hashes) != measured["row_set_sha256"],
        },
        "wrong_spatial_derivative": {
            "mutation": "replace partial_1 v_0 by partial_2 v_0 in evolution_w[1,0]",
            "plane_wave_witness": {"n_1": 2, "n_2": 3, "amplitude": 5},
            "exact_residual": "5",
            "mutated_row_set_sha256": _canonical_sha(corrupted_hashes),
            "expected_row_set_sha256": measured["row_set_sha256"],
            "rejected": _canonical_sha(corrupted_hashes) != measured["row_set_sha256"],
        },
        "zero_fill_dynamic_velocity_rows": {
            "mutation": "infer all 17 unregistered dynamic v rows are zero",
            "nonidentifiability_witness_sha256": witness["witness_sha256"],
            "exact_rhs_delta": "1",
            "rejected": True,
        },
        "claim_full_rhs_from_kinematic_slice": {
            "closed_rows": 68,
            "required_rows": 85,
            "missing_rows": 17,
            "rejected": True,
        },
    }
    return {
        "state_index_contract": {
            "q_A": {"start": 0, "stop": 17},
            "v_A": {"start": 17, "stop": 34},
            "w_1A": {"start": 34, "stop": 51},
            "w_2A": {"start": 51, "stop": 68},
            "w_3A": {"start": 68, "stop": 85},
            "predecessor_assembly_sha256": assembly_sha,
        },
        "rows": rows,
        "row_set_sha256": measured["row_set_sha256"],
        "equation_origin_set_sha256": measured["equation_origin_set_sha256"],
        "candidate_results": manifests,
        "dynamic_row_nonidentifiability_witness": witness,
        "negative_controls": negatives,
        "next_missing_primitive": {
            "primitive": (
                "candidate_bound_fixed_cylindrical_r_positive_17_dynamic_velocity_rhs_rows_"
                "solved_from_sourced_metric_gravity_scalar_scalar_maxwell_fluid_euler_equations"
            ),
            "required_rows_per_candidate": 17,
            "required_candidate_rows": 204,
            "registered_rows": 0,
            "first_missing_source_primitives": [
                "solved_sourced_metric_and_gravity_scalar_acceleration_rows",
                "matter_field_euler_component_expansions",
                "candidate_bound_lower_order_equation_origin_map",
            ],
            "nonidentifiability_witness_sha256": witness["witness_sha256"],
            "status": "BLOCK_SOURCE_PRIMITIVES_UNREGISTERED",
        },
    }


def build_receipt(config_path: Path, *, root: Path | None = None) -> dict[str, Any]:
    repository = (root or config_path.resolve().parents[1]).resolve()
    config = _load_json(config_path)
    if config.get("schema_version") != (
        "invariant-system10-cylindrical-r-positive-kinematic-rhs-config-1.0"
    ):
        raise System10KinematicRHSError("unsupported config schema")
    expected_caps = {
        "candidates": 12,
        "state_dimension": 85,
        "common_kinematic_rows": 68,
        "candidate_dynamic_rows_remaining": 204,
        "maximum_output_bytes": 524288,
    }
    if config.get("caps") != expected_caps:
        raise System10KinematicRHSError("caps changed")
    expected_claims = {
        "common_kinematic_rhs_rows": True,
        "equation_origin_map": True,
        "fixed_cylindrical_r_positive": True,
        "candidate_dynamic_velocity_rhs": False,
        "full_85_state_rhs": False,
        "constraint_propagation": False,
        "hyperbolicity": False,
        "global_theorem": False,
        "promotion": False,
    }
    if config.get("claims_policy") != expected_claims:
        raise System10KinematicRHSError("claims policy broadened")
    bound = {
        name: _load_binding(repository, binding)
        for name, binding in config.get("bindings", {}).items()
    }
    if set(bound) != {
        "divq_rows",
        "first_order_reduction",
        "r_positive_domain",
        "sourced_metric_euler",
    }:
        raise System10KinematicRHSError("binding manifest changed")
    sources = {
        name: _load_source(repository, binding)
        for name, binding in config.get("source_evidence", {}).items()
    }
    if set(sources) != {"source", "test"}:
        raise System10KinematicRHSError("source evidence manifest changed")
    expected_test = (
        repository / "tests/test_system10_cylindrical_r_positive_kinematic_rhs_materializer.py"
    )
    if sources["source"] != Path(__file__).resolve() or sources["test"] != expected_test:
        raise System10KinematicRHSError("self evidence path changed")
    materialization = _materialize(bound, config.get("frozen_expectations", {}))
    body = {
        "schema_version": ("invariant-system10-cylindrical-r-positive-kinematic-rhs-result-1.0"),
        "campaign_id": config["campaign_id"],
        "decision": "BOUNDED_PASS_68_COMMON_KINEMATIC_RHS_ROWS_BLOCK_17_DYNAMIC_ROWS",
        "materialization": materialization,
        "counts": {
            "candidates": 12,
            "state_dimension": 85,
            "common_kinematic_rows_registered": 68,
            "q_definition_rows": 17,
            "w_integrability_rows": 51,
            "equation_origins_registered": 68,
            "candidate_row_instances_closed": 816,
            "candidate_dynamic_velocity_rows_required": 204,
            "candidate_dynamic_velocity_rows_registered": 0,
            "full_85_state_rhs_candidates_closed": 0,
            "constraint_propagation_proofs": 0,
            "negative_controls": 4,
        },
        "claims": {
            "common_kinematic_rhs_rows_closed": True,
            "equation_origin_map_closed_for_registered_rows": True,
            "fixed_cylindrical_r_positive_closed": True,
            "candidate_dynamic_velocity_rhs_closed": False,
            "full_85_state_rhs_closed": False,
            "constraint_propagation_closed": False,
            "hyperbolicity_closed": False,
            "global_theorem_established": False,
            "promotion_authorized": False,
        },
        "scope": (
            "Exact common 85-state kinematic RHS slice on the fixed cylindrical r>0 domain: "
            "17 state-definition rows partial_0 q_A=v_A and 51 commuting-partial "
            "integrability rows partial_0 w_iA=partial_i v_A, each with a source-bound equation "
            "origin. This closes 68 of 85 RHS rows for every candidate. The 17 candidate-dynamic "
            "velocity rows still lack solved sourced acceleration equations and lower-order "
            "equation origins, so full evolution, constraint propagation, hyperbolicity, global, "
            "and promotion claims remain blocked."
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
        raise System10KinematicRHSError("output cap exceeded")
    return receipt


def write_receipt(
    config_path: Path, output_path: Path, *, root: Path | None = None
) -> dict[str, Any]:
    receipt = build_receipt(config_path, root=root)
    data = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if output_path.exists() and output_path.read_bytes() != data:
        raise System10KinematicRHSError("immutable output conflict")
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
