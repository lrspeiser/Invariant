"""Factor the fitted connection through a finite registered action-feature grid."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import sympy as sp

from .quartic_fitted_output_connection_covariant_origin_audit import (
    _validate_result as validate_origin_audit,
)
from .quartic_row0_arithmetic_expansion_campaign import _content_hash_matches

CONFIG_SCHEMA = (
    "sigma-quartic-fitted-output-connection-action-feature-factorization-config-1.0"
)
RESULT_SCHEMA = (
    "sigma-quartic-fitted-output-connection-action-feature-factorization-gate-1.0"
)
CAMPAIGN_ID = "quartic-fitted-output-connection-action-feature-factorization-001"
CONFIG_PATH = (
    "configs/backgrounds/"
    "quartic_fitted_output_connection_action_feature_factorization_gate.json"
)
OUTPUT_PATH = (
    "runs/physics-language/"
    "quartic-fitted-output-connection-action-feature-factorization-gate/campaign.json"
)
SOURCE_PATH = (
    "src/sigma_theory_compiler/"
    "quartic_fitted_output_connection_action_feature_factorization_gate.py"
)
TEST_PATH = (
    "tests/test_quartic_fitted_output_connection_action_feature_factorization_gate.py"
)
FIRST_BLOCKER = (
    "registered_covariant_derivation_functor_or_corrected_second_source_jet_required_"
    "to_promote_extensional_G4_X_factorization_to_action_origin"
)

EXPECTED_PREDECESSORS = {
    "fitted_output_connection_covariant_origin_audit": {
        "path": (
            "runs/physics-language/quartic-fitted-output-connection-covariant-origin-audit/"
            "campaign.json"
        ),
        "file_sha256": "828b3a128031011d7628745c22b75a828f98288392f16d32ae2354b589cb9728",
        "content_sha256": "1e96b08e8d451a6a8baed04757c9c6ee85b886ffbfeffadf526678ed27a2007f",
    },
    "candidate_pother_one_form_connection": {
        "path": (
            "runs/physics-language/quartic-candidate-pother-one-form-connection-gate/"
            "campaign.json"
        ),
        "file_sha256": "f43b5623a46dc19541cd5a6323bd8b0c1ea7b63c7fbcfdb1e26bfe35a2f82bf6",
        "content_sha256": "c79256b901eb6e7b543938cae6e6cf4b41e9fe2778aa37077604cf39a869f93d",
    },
}
EXPECTED_DIRECT_EVIDENCE = {
    "covariant_action": {
        "source": {
            "path": "src/sigma_theory_compiler/quartic_dirac_hamiltonian_campaign.py",
            "file_sha256": "581a8daa447c9fb4096ca3b68052575ea9ad6842e37f938151fdf0ed931af510",
        },
        "config": {
            "path": "configs/backgrounds/quartic_dirac_hamiltonian_campaign.json",
            "file_sha256": "a1a9f4c228a469c4b31388d356f437350bf86b29efc3022c5373e278c567acc7",
        },
        "test": {
            "path": "tests/test_quartic_dirac_hamiltonian_campaign.py",
            "file_sha256": "0b4c56a13d8636d65ccacffd83bd418b8c9dda1bf5f45e1b071846abfccc2594",
        },
        "artifact": {
            "path": "runs/physics-language/quartic-dirac-hamiltonian-campaign/campaign.json",
            "file_sha256": "68541766993d0d46f23dd2707c4e5db8bbf00dbdd9c442fc3802c1c2f7d9bb3f",
            "content_sha256": "69f6f67237020adab07f741b8de154465fa5d24984d78dfb2541da4567db2a47",
        },
    }
}
EXPECTED_CONTRACT = {
    "action_feature_basis": ["constant", "G2_X2", "G4_X"],
    "candidate_grid_G2_X2": ["-1", "0", "1"],
    "candidate_grid_G4_X": ["-1", "-1/2", "1/2", "1"],
    "candidate_independent_map": "exact_affine_map_over_Q_sqrt2",
    "expected_connection_coordinates": 22,
    "expected_design_rank": 3,
    "expected_grid_points": 12,
    "origin_admission": "extensional_factorization_is_necessary_not_sufficient",
}
EXPECTED_POLICIES = {
    "covariant_origin": "fail_closed_without_registered_functor_and_live_root_replay",
    "corrected_second_source_jet": "fail_closed",
    "cross_slice_D2F": "fail_closed",
    "complete_ordered_D2F": "fail_closed",
    "full_high_atom_identity": "fail_closed",
    "global_H7": "fail_closed",
    "nonlinear_PDE": "fail_closed",
    "lifespan": "fail_closed",
    "candidate_rejection": "forbidden",
}
EXPECTED_SEALS = {
    "observations_opened": False,
    "solar_system_inputs_opened": False,
    "cosmology_inputs_opened": False,
    "paid_llm_calls": False,
    "live_SQLite_opened": False,
    "GPU_execution_used": False,
}
CLAIM_SEALS = {
    "registered_action_feature_grid_bound": True,
    "candidate_independent_affine_factorization_exists": True,
    "factorization_unique_in_declared_three_feature_class": True,
    "all_264_fitted_values_factor_through_G4_X": True,
    "G2_X2_and_constant_coefficients_vanish_in_declared_fit": True,
    "factorization_is_covariant_derivation": False,
    "fitted_coefficients_with_action_root_provenance": False,
    "covariant_output_connection_derivation_registered": False,
    "corrected_second_source_jet_registered": False,
    "physical_two_sided_connection_registered": False,
    "cross_slice_D2F_entries_admitted": False,
    "complete_ordered_D2F_tensor_registered": False,
    "full_high_atom_good_unknown_identity_proved": False,
    "global_H7_energy_closed": False,
    "nonlinear_PDE_closed": False,
    "nonlinear_lifespan_proved": False,
    "candidate_theory_rejected": False,
    "observational_claim_made": False,
}
ACTION_TOP_KEYS = {
    "binding_campaign_sha256", "certificates", "claim", "config_sha256",
    "content_sha256", "counts", "errors", "negative_controls", "primary_sources",
    "schema_version", "scope", "source_ir_sha256", "status",
    "symmetrizer_campaign_sha256",
}
ACTION_RECORD_KEYS = {
    "adm_hessian_and_primary_constraint", "candidate_id", "certified_local_jet_embedding",
    "claim", "coefficients", "covariant_action_specialization", "dirac_chain",
    "forward_homogeneous_invariant_domain", "on_shell_local_flrw_witness",
    "on_shell_quadratic_physical_hamiltonian", "schema_version", "scope", "status",
}


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _content_sha(value: Mapping[str, Any]) -> str:
    return _sha({key: item for key, item in value.items() if key != "content_sha256"})


def _inside(root: Path, relative: str) -> Path:
    target = (root / relative).resolve()
    if target != root and root not in target.parents:
        raise ValueError("action feature factorization path escapes project root")
    return target


def _validate_config(value: Mapping[str, Any]) -> None:
    expected = {
        "schema_version": CONFIG_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "output_path": OUTPUT_PATH,
        "predecessors": EXPECTED_PREDECESSORS,
        "direct_evidence": EXPECTED_DIRECT_EVIDENCE,
        "factorization_contract": EXPECTED_CONTRACT,
        "policies": EXPECTED_POLICIES,
        "seals": EXPECTED_SEALS,
    }
    if value != expected:
        raise ValueError("action feature factorization config boundary changed")


def _load_bound(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    if set(binding) != {"path", "file_sha256", "content_sha256"}:
        raise ValueError("action feature factorization artifact binding changed")
    path = _inside(root, str(binding["path"]))
    if not path.is_file() or _file_sha(path) != binding["file_sha256"]:
        raise ValueError("action feature factorization artifact file hash changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("content_sha256") != binding["content_sha256"]:
        raise ValueError("action feature factorization artifact content binding changed")
    if not _content_hash_matches(value):
        raise ValueError("action feature factorization artifact content changed")
    return value


def _load_action(root: Path) -> dict[str, Any]:
    bundle = EXPECTED_DIRECT_EVIDENCE["covariant_action"]
    if set(bundle) != {"source", "config", "test", "artifact"}:
        raise ValueError("action feature factorization action bundle changed")
    for label in ("source", "config", "test"):
        binding = bundle[label]
        if set(binding) != {"path", "file_sha256"} or _file_sha(
            _inside(root, binding["path"])
        ) != binding["file_sha256"]:
            raise ValueError(f"action feature factorization direct {label} binding changed")
    action = _load_bound(root, bundle["artifact"])
    if (
        set(action) != ACTION_TOP_KEYS
        or action.get("status") != "pass_all_12_local_on_shell_adm_dirac_and_quadratic_hamiltonian"
        or action.get("counts")
        != {"local_on_shell_adm_dirac_hamiltonian_passed": 12, "rejected": 0, "selected": 12}
    ):
        raise ValueError("action feature factorization action schema changed")
    return action


def _records(value: Mapping[str, Any], key: str) -> dict[str, Mapping[str, Any]]:
    rows = value.get(key)
    if not isinstance(rows, list):
        raise TypeError("action feature factorization candidate records missing")
    result = {str(row.get("candidate_id")): row for row in rows if isinstance(row, Mapping)}
    if len(rows) != 12 or len(result) != 12:
        raise ValueError("action feature factorization candidate set changed")
    return result


def _action_features(row: Mapping[str, Any]) -> tuple[sp.Expr, sp.Expr]:
    if set(row) != ACTION_RECORD_KEYS:
        raise ValueError("action feature factorization action record schema changed")
    specialization = row.get("covariant_action_specialization")
    coefficients = row.get("coefficients")
    if not isinstance(specialization, Mapping) or set(specialization) != {"G2", "G3", "G4", "G5"}:
        raise ValueError("action feature factorization specialization schema changed")
    if not isinstance(coefficients, Mapping) or set(coefficients) != {
        "a01", "a10", "a20", "c02", "c11", "c20", "d01", "d10", "m2"
    }:
        raise ValueError("action feature factorization coefficient schema changed")
    a10, c20 = sp.sympify(coefficients["a10"]), sp.sympify(coefficients["c20"])
    expected = {
        "G2": f"X+({coefficients['c20']})*X^2",
        "G3": "0",
        "G4": f"1/2+({coefficients['a10']})*X",
        "G5": "0",
    }
    if specialization != expected:
        raise ValueError("action feature factorization specialization arithmetic changed")
    return c20, a10


def _connection_map(row: Mapping[str, Any]) -> dict[tuple[str, int, int, str], sp.Expr]:
    witness = row.get("free_variable_zero_connection_witness")
    if not isinstance(witness, Mapping) or witness.get("total_nonzero_count") != 22:
        raise ValueError("action feature factorization fitted witness changed")
    result: dict[tuple[str, int, int, str], sp.Expr] = {}
    for direction, field, atom_key in (
        ("Pother", "Pother_direction_nonzero_entries", "Pother_atom"),
        ("P10", "P10_direction_nonzero_entries", "P10_atom"),
    ):
        entries = witness.get(field)
        if not isinstance(entries, list):
            raise TypeError("action feature factorization sparse entries missing")
        for entry in entries:
            if not isinstance(entry, Mapping) or set(entry) != {
                atom_key, "input_row", "output_row", "value"
            }:
                raise ValueError("action feature factorization sparse entry schema changed")
            key = (direction, int(entry["output_row"]), int(entry["input_row"]), str(entry[atom_key]))
            if key in result:
                raise ValueError("action feature factorization duplicate coordinate")
            result[key] = sp.sympify(entry["value"])
    if len(result) != 22 or any(value == 0 for value in result.values()):
        raise ValueError("action feature factorization coordinate support changed")
    return result


def _coordinate_record(key: tuple[str, int, int, str], beta: sp.Matrix) -> dict[str, Any]:
    direction, output_row, input_row, atom = key
    return {
        "direction": direction,
        "output_row": output_row,
        "input_row": input_row,
        "atom": atom,
        "affine_coefficients": {
            "constant": str(beta[0]), "G2_X2": str(beta[1]), "G4_X": str(beta[2])
        },
    }


def _factorization(
    pother: Mapping[str, Any], action: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    fitted = _records(pother, "candidate_records")
    actions = _records(action, "certificates")
    if set(fitted) != set(actions):
        raise ValueError("action feature factorization predecessor candidates disagree")
    ids = sorted(fitted)
    feature_rows: list[list[sp.Expr]] = []
    values: dict[str, dict[tuple[str, int, int, str], sp.Expr]] = {}
    candidate_records: list[dict[str, Any]] = []
    for candidate_id in ids:
        if fitted[candidate_id].get("coefficients") != actions[candidate_id].get("coefficients"):
            raise ValueError("action feature factorization candidate coefficients disagree")
        g2_x2, g4_x = _action_features(actions[candidate_id])
        feature_rows.append([sp.Integer(1), g2_x2, g4_x])
        values[candidate_id] = _connection_map(fitted[candidate_id])
        candidate_records.append({
            "candidate_id": candidate_id,
            "action_features": {"constant": "1", "G2_X2": str(g2_x2), "G4_X": str(g4_x)},
            "action_specialization_sha256": _sha(actions[candidate_id]["covariant_action_specialization"]),
            "fitted_connection_sha256": _sha({
                "Pother_direction_nonzero_entries": fitted[candidate_id]["free_variable_zero_connection_witness"]["Pother_direction_nonzero_entries"],
                "P10_direction_nonzero_entries": fitted[candidate_id]["free_variable_zero_connection_witness"]["P10_direction_nonzero_entries"],
            }),
            "fitted_values_checked": 22,
            "factorization_residual_nonzero_count": 0,
            "gate_decision": "pass",
            "candidate_decision": "blocked",
            "candidate_rejection_authorized": False,
            "first_blocker": FIRST_BLOCKER,
        })
    coordinate_keys = sorted(next(iter(values.values())))
    if any(sorted(row) != coordinate_keys for row in values.values()):
        raise ValueError("action feature factorization coordinate support not universal")
    design = sp.Matrix(feature_rows)
    if design.rank() != 3:
        raise ValueError("action feature factorization design rank changed")
    left_inverse = (design.T * design).inv() * design.T
    coordinate_records = []
    for key in coordinate_keys:
        target = sp.Matrix([values[candidate_id][key] for candidate_id in ids])
        beta = sp.simplify(left_inverse * target)
        if design * beta != target or beta[0] != 0 or beta[1] != 0 or beta[2] == 0:
            raise ValueError("action feature factorization exact G4_X law failed")
        coordinate_records.append(_coordinate_record(key, beta))
    grid = {(str(row[1]), str(row[2])) for row in feature_rows}
    expected_grid = {(x, y) for x in ("-1", "0", "1") for y in ("-1", "-1/2", "1/2", "1")}
    if grid != expected_grid:
        raise ValueError("action feature factorization action grid changed")
    theorem = {
        "name": "unique_declared_affine_action_feature_factorization",
        "premises": (
            "The twelve candidates form the complete registered 3-by-4 grid in (G2_X2,G4_X), "
            "and each candidate carries the same 22-coordinate sparse fitted-connection support."
        ),
        "exact_result": (
            "The 12-by-3 design matrix [1,G2_X2,G4_X] has rank three. For every one of the "
            "22 coordinates, the unique exact affine fit has zero constant and G2_X2 "
            "coefficients and a nonzero G4_X coefficient; all 264 residuals vanish."
        ),
        "boundary": (
            "This proves extensional finite-grid compatibility and uniqueness only in the declared "
            "three-feature affine class. It does not derive a covariant output-bundle functor, bind "
            "the multipliers to action arithmetic, or register a corrected second source jet."
        ),
    }
    return coordinate_records, candidate_records, theorem


def _expected_body(
    root: Path, config_path: Path, origin: Mapping[str, Any], pother: Mapping[str, Any],
    action: Mapping[str, Any]
) -> dict[str, Any]:
    coordinates, candidates, theorem = _factorization(pother, action)
    return {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "decision": "pass_exact_unique_declared_affine_G4_X_factorization_origin_still_blocked",
        "decision_counts": {"pass": 12, "reject": 0, "blocked": 0},
        "downstream_admission_counts": {"pass": 0, "reject": 0, "blocked": 12},
        "gate_counts": {
            "selected": 12,
            "action_feature_grid_points": 12,
            "action_feature_design_rank": 3,
            "declared_affine_features": 3,
            "fitted_connection_coordinates_per_candidate": 22,
            "fitted_connection_values_checked": 264,
            "factorization_residual_nonzero_count": 0,
            "coordinates_with_zero_constant_coefficient": 22,
            "coordinates_with_zero_G2_X2_coefficient": 22,
            "coordinates_with_nonzero_G4_X_coefficient": 22,
            "fitted_coefficients_with_action_root_provenance": 0,
            "registered_covariant_derivation_functors": 0,
            "registered_corrected_second_source_jet_entries": 0,
            "cross_slice_D2F_entries_admitted": 0,
            "principal_high_atom_entries_missing_per_candidate": 106920,
            "complete_ordered_D2F_tensors_registered": 0,
            "full_high_atom_good_unknown_identities_proved": 0,
            "global_H7_closures": 0,
            "nonlinear_PDE_closures": 0,
            "lifespans_proved": 0,
        },
        "factorization_theorem": theorem,
        "feature_basis": EXPECTED_CONTRACT["action_feature_basis"],
        "universal_coordinate_map": coordinates,
        "candidate_records": candidates,
        "exact_controls": {
            "promote_finite_grid_fit_to_covariant_derivation": {"rejected": True},
            "infer_action_root_provenance_from_value_factorization": {"rejected": True},
            "admit_cross_slice_D2F_without_registered_functor": {
                "rejected": True, "cross_slice_entries_admitted": 0
            },
            "reject_candidates_from_missing_covariant_origin": {"rejected": True},
        },
        "first_blocker": FIRST_BLOCKER,
        "secondary_blockers": [
            "corrected_second_source_jet_not_registered",
            "general_covariant_typed_map_not_registered",
            "remaining_other_principal_by_other_principal_values_not_registered",
            "complete_high_atom_identity_TC1_TC2_TC3_TC5_B7_H7_PDE_lifespan_not_closed",
        ],
        "claim_seals": CLAIM_SEALS,
        "data_seals": EXPECTED_SEALS,
        "source_bindings": {
            "source": {"path": SOURCE_PATH, "file_sha256": _file_sha(_inside(root, SOURCE_PATH))},
            "config": {"path": CONFIG_PATH, "file_sha256": _file_sha(config_path)},
            "test": {"path": TEST_PATH, "file_sha256": _file_sha(_inside(root, TEST_PATH))},
            "direct_evidence": EXPECTED_DIRECT_EVIDENCE,
            **EXPECTED_PREDECESSORS,
        },
        "predecessor_decisions": {
            "origin_audit": origin["decision"],
            "pother_connection": pother["decision"],
        },
        "scope": (
            "candidate-bound exact affine factorization of the 22 fitted connection coordinates "
            "over the registered 12-point covariant-action feature grid; no covariant derivation, "
            "corrected second source jet, D2F admission, complete D2F, high-atom identity, H7, "
            "PDE, lifespan, candidate rejection, or observation"
        ),
    }


def _validate_source_bindings(value: Mapping[str, Any], root: Path) -> None:
    bindings = value.get("source_bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != {
        "source", "config", "test", "direct_evidence", *EXPECTED_PREDECESSORS
    }:
        raise ValueError("action feature factorization source binding keys changed")
    for label, relative in {"source": SOURCE_PATH, "config": CONFIG_PATH, "test": TEST_PATH}.items():
        if bindings[label] != {"path": relative, "file_sha256": _file_sha(_inside(root, relative))}:
            raise ValueError("action feature factorization local binding changed")
    if bindings["direct_evidence"] != EXPECTED_DIRECT_EVIDENCE:
        raise ValueError("action feature factorization direct evidence binding changed")
    for label, expected in EXPECTED_PREDECESSORS.items():
        if bindings[label] != expected:
            raise ValueError("action feature factorization predecessor binding changed")


def _validated_inputs(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    origin = _load_bound(root, EXPECTED_PREDECESSORS["fitted_output_connection_covariant_origin_audit"])
    validate_origin_audit(origin, root=root)
    pother = _load_bound(root, EXPECTED_PREDECESSORS["candidate_pother_one_form_connection"])
    # The origin audit replays this exact Pother binding transitively. Loading the same
    # artifact directly here binds the values used by the finite factorization without
    # repeating the expensive two-sided solve.
    action = _load_action(root)
    return origin, pother, action


def _validate_result(value: Mapping[str, Any], *, root: Path | None = None) -> None:
    validation_root = (root or Path(__file__).resolve().parents[2]).resolve()
    if value.get("content_sha256") != _content_sha(value):
        raise ValueError("action feature factorization content hash changed")
    _validate_source_bindings(value, validation_root)
    config_path = _inside(validation_root, CONFIG_PATH)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    origin, pother, action = _validated_inputs(validation_root)
    expected = _expected_body(validation_root, config_path, origin, pother, action)
    if {key: item for key, item in value.items() if key != "content_sha256"} != expected:
        raise ValueError("action feature factorization result boundary changed")


def build_gate(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    root = config_path.parents[2]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    origin, pother, action = _validated_inputs(root)
    body = _expected_body(root, config_path, origin, pother, action)
    result = {**body, "content_sha256": _sha(body)}
    _validate_result(result, root=root)
    return result


def write_gate(config_path: Path) -> Path:
    result = build_gate(config_path)
    root = config_path.resolve().parents[2]
    output = _inside(root, OUTPUT_PATH)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(CONFIG_PATH))
    args = parser.parse_args()
    print(write_gate(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
