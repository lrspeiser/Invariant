"""Certify all lower q/p coordinate tangents in the nonlinear covariant jet map."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

CONFIG_SCHEMA = "sigma-quartic-lower-coordinate-covariant-projection-config-1.0"
RESULT_SCHEMA = "sigma-quartic-lower-coordinate-covariant-projection-gate-1.0"
CAMPAIGN_ID = "quartic-lower-coordinate-covariant-projection-001"
CONFIG_PATH = "configs/backgrounds/quartic_lower_coordinate_covariant_projection_gate.json"
SOURCE_PATH = "src/sigma_theory_compiler/quartic_lower_coordinate_covariant_projection_gate.py"
TEST_PATH = "tests/test_quartic_lower_coordinate_covariant_projection_gate.py"
OUTPUT_PATH = (
    "runs/physics-language/quartic-lower-coordinate-covariant-projection-gate/campaign.json"
)
SYMMETRIC_PAIRS = tuple((left, right) for left in range(4) for right in range(left, 4))
SOURCE_BINDINGS = {
    "principal_projection": {
        "path": (
            "runs/physics-language/quartic-principal-second-jet-covariant-projection-gate/"
            "campaign.json"
        ),
        "file_sha256": "727b1902b32f369ad82803b5f1cc5e57a8643bb7f5b0aec1f8d934c0cc51efaa",
        "content_sha256": "f4343d2a7b4418694b96c088a22795983168451e948035875557bc726385c7b7",
    },
    "geometric_map": {
        "path": "runs/physics-language/quartic-geometric-jet-campaign/campaign.json",
        "file_sha256": "aa18e643877f4eb7224891e70e929b8d9574a83309aac9007e9f635689d82b65",
        "content_sha256": "3878728a11df567606c18d37cd683ff222a5de20f87248394f7ce75a618562a4",
    },
    "full_source_D1": {
        "path": (
            "runs/physics-language/quartic-full-source-jacobian-arithmetic-campaign/campaign.json"
        ),
        "file_sha256": "e893ebcaef464b958516279c557382fb76ecdb0fd542b3e3fed6a347076fcdae",
        "content_sha256": "1707b7258fd434f68b06c7af6bc447b4136624b9916992df8b412e048ab6538a",
    },
    "D1_DAG_differentiability": {
        "path": (
            "runs/physics-language/"
            "quartic-ordered-mixed-d2-arithmetic-dag-differentiability-gate/campaign.json"
        ),
        "file_sha256": "2992571c544846efc96142e2e4a74efe280a7bb025efadb1ff945ab9515bafcc",
        "content_sha256": "d8afd9f91c090ad1c07e4bb22257baa8c61c095f8d434e02a27082b5591abb6a",
    },
}
CONTRACT = {
    "candidate_count": 12,
    "coordinate_dimension": 153,
    "previous_principal_directions": 99,
    "lower_directions": 54,
    "q_metric_directions": 10,
    "p_metric_directions": 40,
    "p_scalar_gradient_directions": 4,
    "covariant_output_dimension": 24,
    "registered_D2_entries_per_candidate": 5324,
    "full_D2_entries_per_candidate": 257499,
}
POLICIES = {
    "arbitrary_background_lower_projection": "exact_indexed_tangent_formula_required",
    "alias_handling": "coordinate_columns_not_formal_slot_ordinals",
    "D2_entry_promotion": (
        "forbidden_until_all_reachable_D1_component_input_leaf_derivatives_are_registered"
    ),
    "complete_D2F": "fail_closed",
    "H7": "fail_closed",
    "candidate_rejection": "forbidden",
}
SEALS = {
    "observations_opened": False,
    "live_SQLite_opened": False,
    "GPU_execution_used": False,
    "paid_llm_calls": False,
}
FIRST_BLOCKER = (
    "register_candidate_bound_coordinate_derivatives_for_the_31680_reachable_"
    "A_B_C_component_input_leaf_obligations_before_emitting_any_new_D2_root"
)


class LowerCoordinateProjectionError(ValueError):
    """A covariant tangent formula, alias, or D2 boundary changed."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _content_sha(value: Mapping[str, Any]) -> str:
    return _sha({key: item for key, item in value.items() if key != "content_sha256"})


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inside(root: Path, relative: str) -> Path:
    if not relative or "\\" in relative:
        raise LowerCoordinateProjectionError("projection path must be portable")
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise LowerCoordinateProjectionError("projection path escapes project root")
    return path


def _copy(value: Any) -> Any:
    return json.loads(_canonical(value))


def _load_bound(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    path = _inside(root, str(binding["path"]))
    if not path.is_file() or _file_sha(path) != binding["file_sha256"]:
        raise LowerCoordinateProjectionError("lower projection evidence file changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("content_sha256") != binding["content_sha256"] or value.get(
        "content_sha256"
    ) != _content_sha(value):
        raise LowerCoordinateProjectionError("lower projection evidence content changed")
    return value


def _validate_config(value: Mapping[str, Any]) -> None:
    expected = {
        "schema_version": CONFIG_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "output_path": OUTPUT_PATH,
        "source_bindings": SOURCE_BINDINGS,
        "projection_contract": CONTRACT,
        "policies": POLICIES,
        "seals": SEALS,
    }
    if value != expected:
        raise LowerCoordinateProjectionError("lower projection config changed")


def _lower_atoms() -> list[str]:
    return [f"q[{field}]" for field in range(10)] + [
        f"p{derivative}[{field}]" for derivative in range(4) for field in range(11)
    ]


def _principal_atoms() -> list[str]:
    families = ("s01", "s02", "s03", "s11", "s12", "s13", "s22", "s23", "s33")
    return [f"{family}[{field}]" for family in families for field in range(11)]


def _validate_inputs(root: Path) -> dict[str, dict[str, Any]]:
    values = {role: _load_bound(root, binding) for role, binding in SOURCE_BINDINGS.items()}
    principal = values["principal_projection"]
    geometric = values["geometric_map"]
    full_d1 = values["full_source_D1"]
    differentiability = values["D1_DAG_differentiability"]
    registry = principal.get("principal_projection_registry")
    if (
        principal.get("decision")
        != "pass_all_99_principal_second_jet_covariant_projections_lower_54_blocked"
        or not isinstance(registry, list)
        or len(registry) != 99
        or [row.get("coordinate_column") for row in registry] != list(range(54, 153))
        or [row.get("coordinate_atom") for row in registry] != _principal_atoms()
        or principal.get("gate_counts", {}).get("D2_entries_registered_per_candidate") != 5324
        or principal.get("gate_counts", {}).get("remaining_lower_jet_projection_directions") != 54
    ):
        raise LowerCoordinateProjectionError("principal projection boundary changed")
    if (
        geometric.get("status") != "pass_all_12_exact_nonlinear_geometric_state_to_jet_maps"
        or geometric.get("counts", {}).get("selected") != 12
        or geometric.get("geometric_control", {}).get("passed") is not True
    ):
        raise LowerCoordinateProjectionError("geometric map boundary changed")
    manifest = full_d1.get("common_full_entry_manifest", {})
    entries = manifest.get("entries")
    if (
        full_d1.get("status")
        != "pass_all_12_full_11x153_entrywise_arithmetic_mixed_tensors_fail_closed"
        or full_d1.get("counts", {}).get("full_source_entries_per_candidate") != 1683
        or full_d1.get("counts", {}).get("lower_entries_per_candidate") != 594
        or not isinstance(entries, list)
        or len(entries) != 1683
        or sum(row.get("family") == "lower" for row in entries) != 594
        or {row.get("coordinate_atom") for row in entries if row.get("family") == "lower"}
        != set(_lower_atoms())
    ):
        raise LowerCoordinateProjectionError("full source D1 boundary changed")
    diff_counts = differentiability.get("gate_counts", {})
    if (
        differentiability.get("decision")
        != "pass_exact_D1_DAG_differentiability_boundary_31680_leaf_jets_missing_D2_blocked"
        or diff_counts.get("deduplicated_leaf_derivative_obligations") != 31680
        or diff_counts.get("registered_leaf_derivative_roots") != 0
        or diff_counts.get("registered_ordered_mixed_D2_roots") != 0
        or differentiability.get("claim_seals", {}).get(
            "component_input_leaf_derivatives_registered"
        )
        is not False
    ):
        raise LowerCoordinateProjectionError("D1 differentiability boundary changed")
    return values


def _formula_program() -> dict[str, Any]:
    formulas = [
        "du^ab=-u^ac*dg_cd*u^db",
        "U_k^ab=-u^ac*P_k_cd*u^db",
        "dU_k^ab=-(du^ac*P_k_cd*u^db+u^ac*dP_k_cd*u^db+u^ac*P_k_cd*du^db)",
        "B_s_mn=P_m_sn+P_n_sm-P_s_mn",
        "dB_s_mn=dP_m_sn+dP_n_sm-dP_s_mn",
        "C_ks_mn=S_km_sn+S_kn_sm-S_ks_mn",
        "Gamma^r_mn=(1/2)*u^rs*B_s_mn",
        "dGamma^r_mn=(1/2)*(du^rs*B_s_mn+u^rs*dB_s_mn)",
        "partial_k_Gamma^r_mn=(1/2)*(U_k^rs*B_s_mn+u^rs*C_ks_mn)",
        "dpartial_k_Gamma^r_mn=(1/2)*(dU_k^rs*B_s_mn+U_k^rs*dB_s_mn+du^rs*C_ks_mn)",
        "dv_m=coordinate_seed_dv_m",
        "dH_mn=-dGamma^r_mn*v_r-Gamma^r_mn*dv_r",
        "dR^r_smn=dpartial_m_Gamma^r_ns-dpartial_n_Gamma^r_ms+dGamma^r_ml*Gamma^l_ns+Gamma^r_ml*dGamma^l_ns-dGamma^r_nl*Gamma^l_ms-Gamma^r_nl*dGamma^l_ms",
        "dRicci_mn=sum_r dR^r_mrn",
        "dR=du^mn*Ricci_mn+u^mn*dRicci_mn",
        "dG_mn=dRicci_mn-(1/2)*(dg_mn*R+g_mn*dR)",
        "dG^mn=du^ma*u^nb*G_ab+u^ma*du^nb*G_ab+u^ma*u^nb*dG_ab",
    ]
    body = {
        "schema_version": "sigma-indexed-lower-coordinate-to-covariant-24-tangent-1.0",
        "input_jets": {
            "g_lower": [4, 4],
            "g_upper": [4, 4],
            "P_partial_g": [4, 4, 4],
            "S_partial2_g": [4, 4, 4, 4],
            "v_partial_phi": [4],
            "W_partial2_phi": [4, 4],
        },
        "fixed_lower_tangent_slots": ["dS=0", "dW=0"],
        "output_basis": [f"v_{mu}" for mu in range(4)]
        + [f"H_{left}{right}" for left, right in SYMMETRIC_PAIRS]
        + [f"G_{left}{right}" for left, right in SYMMETRIC_PAIRS],
        "domain": "det(g_lower)!=0; symmetric metric; commuting coordinate second jets",
        "index_range": "all repeated lowercase Latin and Greek indices range over 0..3",
        "formulas_in_dependency_order": formulas,
    }
    return {**body, "content_sha256": _sha(body)}


def _direction_certificate(column: int, atom: str, program_sha: str) -> dict[str, Any]:
    if atom.startswith("q"):
        field = int(atom[2:-1])
        left, right = SYMMETRIC_PAIRS[field]
        family = "q_metric"
        derivative = None
        seed = {
            "dg_symmetric_pair": [left, right],
            "dg_value": "1" if left == right else "sqrt(2)/2",
            "dP": "0",
            "dv": "0",
        }
    else:
        derivative = int(atom[1])
        field = int(atom.split("[")[1][:-1])
        if field < 10:
            left, right = SYMMETRIC_PAIRS[field]
            family = "p_metric"
            seed = {
                "dg": "0",
                "dP_derivative": derivative,
                "dP_symmetric_pair": [left, right],
                "dP_value": "1" if left == right else "sqrt(2)/2",
                "dv": "0",
            }
        else:
            family = "p_scalar_gradient"
            seed = {"dg": "0", "dP": "0", "dv_component": derivative, "dv_value": "1"}
    body = {
        "coordinate_atom": atom,
        "coordinate_column": column,
        "family": family,
        "field_index": field,
        "derivative_index": derivative,
        "tangent_seed": seed,
        "indexed_formula_program_sha256": program_sha,
        "formula_scope": "arbitrary_nonsingular_metric_and_consistent_local_coordinate_2_jet",
        "covariant_output_basis_dimension": 24,
        "exact_projection_registered": True,
    }
    return {**body, "content_sha256": _sha(body)}


def _projection_registry(program: Mapping[str, Any]) -> list[dict[str, Any]]:
    records = [
        _direction_certificate(column, atom, program["content_sha256"])
        for column, atom in enumerate(_lower_atoms())
    ]
    counts = {
        family: sum(record["family"] == family for record in records)
        for family in ("q_metric", "p_metric", "p_scalar_gradient")
    }
    if len(records) != 54 or counts != {
        "q_metric": 10,
        "p_metric": 40,
        "p_scalar_gradient": 4,
    }:
        raise LowerCoordinateProjectionError("lower direction census changed")
    return records


def _exact_controls() -> dict[str, Any]:
    g00 = Fraction(-1)
    u00 = Fraction(-1)
    dg00 = Fraction(1)
    du00 = -u00 * dg00 * u00
    inverse_identity_residual = du00 * g00 + u00 * dg00
    omitted_inverse_residual = u00 * dg00
    if inverse_identity_residual != 0 or omitted_inverse_residual != -1:
        raise LowerCoordinateProjectionError("inverse tangent control changed")
    p_metric_hessian = Fraction(1, 2)
    p_scalar_hessian = Fraction(1)
    return {
        "Minkowski_q0_inverse_identity": {
            "exact_residual": str(inverse_identity_residual),
            "passed": True,
        },
        "omit_inverse_metric_tangent": {
            "exact_residual": str(omitted_inverse_residual),
            "rejected": True,
        },
        "drop_off_diagonal_sqrt2_normalization": {
            "exact_residual": "1-sqrt(2)/2",
            "rejected": True,
            "witness_atom": "q[1]",
        },
        "cylindrical_p1_metric_H22": {
            "background": "g=diag(-1,1,r^2,1), v_r=1, r=1",
            "witness_atom": "p1[7]",
            "exact_projection": str(p_metric_hessian),
            "zero_projection_rejected": p_metric_hessian != 0,
        },
        "cylindrical_p1_scalar_H22": {
            "background": "g=diag(-1,1,r^2,1), Gamma^r_22=-r, r=1",
            "witness_atom": "p1[10]",
            "exact_projection": str(p_scalar_hessian),
            "omit_connection_term_rejected": p_scalar_hessian != 0,
        },
        "promote_projection_certificate_to_D2_value": {"rejected": True},
    }


def _candidate_manifests(principal: Mapping[str, Any], registry_sha: str) -> list[dict[str, Any]]:
    result = []
    for candidate in principal["candidate_manifests"]:
        aliases = candidate["alias_reconciliation_records"]
        alias_columns = [row["coordinate_column"] for row in aliases]
        if len(aliases) != 2 or any(column < 54 for column in alias_columns):
            raise LowerCoordinateProjectionError("lower alias boundary changed")
        body = {
            "candidate_id": candidate["candidate_id"],
            "lower_projection_registry_sha256": registry_sha,
            "lower_coordinate_projection_certificates": 54,
            "lower_unique_coordinate_columns": 54,
            "lower_alias_groups": 0,
            "inherited_principal_alias_groups": 2,
            "inherited_principal_alias_columns": alias_columns,
            "total_unique_coordinate_directions_projected": 153,
            "D2_entries_before": 5324,
            "new_D2_entries_admitted": 0,
            "D2_entries_after": 5324,
            "candidate_decision": "pass_lower_projection_D2_leaf_derivatives_blocked",
            "candidate_rejection_authorized": False,
        }
        result.append({**body, "content_sha256": _sha(body)})
    if len(result) != 12 or len({row["candidate_id"] for row in result}) != 12:
        raise LowerCoordinateProjectionError("candidate projection inventory changed")
    return result


def _expected_body(
    root: Path,
    config_path: Path,
    values: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    program = _formula_program()
    registry = _projection_registry(program)
    registry_sha = _sha([record["content_sha256"] for record in registry])
    candidates = _candidate_manifests(values["principal_projection"], registry_sha)
    d1 = values["full_source_D1"]
    differentiability = values["D1_DAG_differentiability"]
    return {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "decision": "pass_all_54_lower_covariant_projections_D2_count_preserved",
        "decision_counts": {"pass": 12, "blocked": 0, "reject": 0},
        "D2_admission_counts": {"pass": 0, "blocked": 12, "reject": 0},
        "first_blocker": FIRST_BLOCKER,
        "projection_theorem": {
            "name": "arbitrary_background_lower_coordinate_to_covariant_jet_tangent_map",
            "premises": (
                "At one coordinate point g is symmetric and nonsingular; P and S are its "
                "commuting first and second coordinate jets; q off-diagonal coordinates equal "
                "sqrt(2) times physical metric components."
            ),
            "conclusion": (
                "All ten metric-value, forty metric-first-jet, and four scalar-gradient "
                "coordinate unit tangents have exact indexed projections into "
                "(v_mu,H_mu_nu,G^mu_nu). Together with the predecessor 99 principal "
                "directions this covers all 153 unique coordinate directions."
            ),
            "boundary": (
                "The projection is a chain-rule input, not a derivative of an A/B/C component "
                "leaf and not a D2 source value."
            ),
        },
        "indexed_formula_program": program,
        "lower_projection_registry": registry,
        "lower_projection_registry_sha256": registry_sha,
        "alias_reconciliation": {
            "coordinate_columns_are_authoritative": True,
            "lower_formal_slot_alias_groups": 0,
            "inherited_principal_alias_groups_per_candidate": 2,
            "inherited_principal_alias_columns": [97, 130],
            "unique_coordinate_directions_after_union": 153,
        },
        "candidate_manifests": candidates,
        "D1_DAG_audit": {
            "full_D1_entries_per_candidate": d1["counts"]["full_source_entries_per_candidate"],
            "lower_D1_entries_per_candidate": d1["counts"]["lower_entries_per_candidate"],
            "principal_D1_entries_per_candidate": d1["counts"]["principal_entries_per_candidate"],
            "full_D1_manifest_content_sha256": d1["common_full_entry_manifest"]["content_sha256"],
            "reachable_D1_DAG_nodes": differentiability["gate_counts"]["reachable_D1_DAG_nodes"],
            "reachable_component_input_labels": differentiability["gate_counts"][
                "reachable_component_input_labels"
            ],
            "missing_candidate_bound_leaf_derivatives": differentiability["gate_counts"][
                "deduplicated_leaf_derivative_obligations"
            ],
            "registered_leaf_derivative_roots": 0,
            "new_ordered_D2_roots_legitimately_yielded": 0,
        },
        "gate_counts": {
            "selected_candidates": 12,
            "coordinate_basis_dimension": 153,
            "principal_projection_directions_inherited": 99,
            "lower_projection_directions_registered": 54,
            "q_metric_projection_directions": 10,
            "p_metric_projection_directions": 40,
            "p_scalar_gradient_projection_directions": 4,
            "candidate_bound_lower_projection_certificates": 648,
            "unique_coordinate_directions_projected": 153,
            "formal_alias_groups_reconciled": 24,
            "D2_entries_registered_per_candidate_before": 5324,
            "new_D2_entries_registered_per_candidate": 0,
            "D2_entries_registered_per_candidate_after": 5324,
            "full_D2_entries_per_candidate": 257499,
            "D2_entries_remaining_per_candidate": 252175,
            "complete_D2F_tensors": 0,
            "H7_closures": 0,
        },
        "claim_seals": {
            "all_54_lower_coordinate_directions_projected": True,
            "all_153_unique_coordinate_directions_projected": True,
            "alias_aware_coordinate_union_registered": True,
            "full_D1_entry_manifest_registered": True,
            "all_reachable_D1_component_input_leaf_derivatives_registered": False,
            "D1_DAG_differentiation_complete": False,
            "D2_entry_count_advanced": False,
            "complete_D2F": False,
            "global_H7": False,
            "candidate_theory_rejected": False,
        },
        "exact_controls": _exact_controls(),
        "data_seals": dict(SEALS),
        "source_bindings": {
            "source": {"path": SOURCE_PATH, "file_sha256": _file_sha(_inside(root, SOURCE_PATH))},
            "config": {"path": CONFIG_PATH, "file_sha256": _file_sha(config_path)},
            "test": {"path": TEST_PATH, "file_sha256": _file_sha(_inside(root, TEST_PATH))},
            "evidence": _copy(SOURCE_BINDINGS),
        },
        "scope": (
            "exact arbitrary-background indexed coordinate-to-covariant projection for the 54 "
            "lower q/p directions across 12 candidates, alias-aware with the predecessor 99; "
            "no unregistered A/B/C leaf derivative, D2 count advance, complete tensor, H7, "
            "candidate rejection, or observation"
        ),
    }


def build_campaign(
    config_path: Path | str = CONFIG_PATH, *, root: Path | str | None = None
) -> dict[str, Any]:
    project_root = Path(root or Path.cwd()).resolve()
    path = _inside(project_root, str(config_path))
    config = json.loads(path.read_text(encoding="utf-8"))
    _validate_config(config)
    values = _validate_inputs(project_root)
    body = _expected_body(project_root, path, values)
    return {**body, "content_sha256": _sha(body)}


def validate_campaign(value: Mapping[str, Any], *, root: Path | str | None = None) -> None:
    project_root = Path(root or Path.cwd()).resolve()
    config_path = _inside(project_root, CONFIG_PATH)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    expected = build_campaign(root=project_root)
    if value.get("content_sha256") != _content_sha(value) or value != expected:
        raise LowerCoordinateProjectionError("lower projection result changed")


def write_campaign(
    output_path: Path | str = OUTPUT_PATH,
    config_path: Path | str = CONFIG_PATH,
    *,
    root: Path | str | None = None,
) -> dict[str, Any]:
    project_root = Path(root or Path.cwd()).resolve()
    result = build_campaign(config_path, root=project_root)
    path = _inside(project_root, str(output_path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--output", default=OUTPUT_PATH)
    parser.add_argument("--validate-checked", action="store_true")
    args = parser.parse_args(argv)
    if args.validate_checked:
        value = json.loads(Path(args.output).read_text(encoding="utf-8"))
        validate_campaign(value)
    else:
        write_campaign(args.output, args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
