"""Prove finite action-feature values do not identify the fitted connection jet."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import sympy as sp

CONFIG_SCHEMA = "sigma-quartic-fitted-output-connection-action-jet-nonidentifiability-config-1.0"
RESULT_SCHEMA = "sigma-quartic-fitted-output-connection-action-jet-nonidentifiability-gate-1.0"
CAMPAIGN_ID = "quartic-fitted-output-connection-action-jet-nonidentifiability-001"
CONFIG_PATH = (
    "configs/backgrounds/quartic_fitted_output_connection_action_jet_nonidentifiability_gate.json"
)
SOURCE_PATH = (
    "src/sigma_theory_compiler/"
    "quartic_fitted_output_connection_action_jet_nonidentifiability_gate.py"
)
TEST_PATH = "tests/test_quartic_fitted_output_connection_action_jet_nonidentifiability_gate.py"
OUTPUT_PATH = (
    "runs/physics-language/"
    "quartic-fitted-output-connection-action-jet-nonidentifiability-gate/campaign.json"
)
FIRST_BLOCKER = (
    "registered_local_covariant_variation_rule_or_corrected_second_source_jet_values_"
    "required_to_select_one_extension_from_the_exact_22_parameter_jet_ambiguity_family"
)
GRID = (sp.Integer(-1), sp.Rational(-1, 2), sp.Rational(1, 2), sp.Integer(1))
GRID_TEXT = ["-1", "-1/2", "1/2", "1"]

EXPECTED_PREDECESSOR = {
    "source": {
        "path": (
            "src/sigma_theory_compiler/"
            "quartic_fitted_output_connection_action_feature_factorization_gate.py"
        ),
        "file_sha256": "fdc647494ef3ce9be12adfca18bf3a4dd5e6778340ea2ff0b794da19b2ea439c",
    },
    "config": {
        "path": (
            "configs/backgrounds/"
            "quartic_fitted_output_connection_action_feature_factorization_gate.json"
        ),
        "file_sha256": "3a8265c24b1d3d150c6d13f7ca5d1261b79a54ad52d01346e26e5bdbc9e994c6",
    },
    "test": {
        "path": "tests/test_quartic_fitted_output_connection_action_feature_factorization_gate.py",
        "file_sha256": "000344ce2e4b42d55923b6b046aaf62baa8af41d00a0a9346e428e227c516260",
    },
    "artifact": {
        "path": (
            "runs/physics-language/"
            "quartic-fitted-output-connection-action-feature-factorization-gate/campaign.json"
        ),
        "file_sha256": "b13437eda5da6f976286380ea0d479aeed6d1ea9800b5ce61b867ad3493046ab",
        "content_sha256": "3dd8200fea3a83cff2dffbae98425335d1220e32b6ebf806736cd4d1937b7118",
    },
}
EXPECTED_CONTRACT = {
    "action_coordinate": "G4_X",
    "ambiguity_class": "coordinatewise_polynomial_degree_at_most_four_over_Q_sqrt2",
    "base_extension": "beta*g",
    "grid": GRID_TEXT,
    "null_polynomial": "(g+1)*(g+1/2)*(g-1/2)*(g-1)",
    "target_connection_coordinates": 22,
}
EXPECTED_POLICIES = {
    "covariant_output_connection_derivation": "fail_closed",
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
    "finite_grid_value_factorization_bound": True,
    "degree_four_null_polynomial_constructed": True,
    "all_22_coordinate_value_extensions_nonunique": True,
    "all_88_first_jet_samples_nonidentified": True,
    "all_88_second_jet_samples_nonidentified": True,
    "independent_22_parameter_ambiguity_family_constructed": True,
    "covariant_output_connection_derivation_registered": False,
    "corrected_second_source_jet_registered": False,
    "cross_slice_D2F_entries_admitted": False,
    "complete_ordered_D2F_tensor_registered": False,
    "full_high_atom_good_unknown_identity_proved": False,
    "global_H7_energy_closed": False,
    "nonlinear_PDE_closed": False,
    "nonlinear_lifespan_proved": False,
    "candidate_theory_rejected": False,
    "observational_claim_made": False,
}


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _copy_json(value: Any) -> Any:
    return json.loads(_canonical(value))


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _content_sha(value: Mapping[str, Any]) -> str:
    return _sha({key: item for key, item in value.items() if key != "content_sha256"})


def _inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise ValueError("action jet path escapes project root")
    return path


def _validate_config(value: Mapping[str, Any]) -> None:
    expected = {
        "schema_version": CONFIG_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "output_path": OUTPUT_PATH,
        "predecessor": EXPECTED_PREDECESSOR,
        "jet_ambiguity_contract": EXPECTED_CONTRACT,
        "policies": EXPECTED_POLICIES,
        "seals": EXPECTED_SEALS,
    }
    if value != expected:
        raise ValueError("action jet nonidentifiability config boundary changed")


def _load_predecessor(root: Path) -> dict[str, Any]:
    for label in ("source", "config", "test", "artifact"):
        binding = EXPECTED_PREDECESSOR[label]
        expected_keys = (
            {"path", "file_sha256", "content_sha256"}
            if label == "artifact"
            else {
                "path",
                "file_sha256",
            }
        )
        if set(binding) != expected_keys:
            raise ValueError("action jet predecessor binding schema changed")
        path = _inside(root, binding["path"])
        if not path.is_file() or _file_sha(path) != binding["file_sha256"]:
            raise ValueError(f"action jet predecessor {label} file hash changed")
    artifact_path = _inside(root, EXPECTED_PREDECESSOR["artifact"]["path"])
    value = json.loads(artifact_path.read_text(encoding="utf-8"))
    if value.get("content_sha256") != EXPECTED_PREDECESSOR["artifact"][
        "content_sha256"
    ] or value.get("content_sha256") != _content_sha(value):
        raise ValueError("action jet predecessor content binding changed")
    expected_top_keys = {
        "campaign_id",
        "candidate_records",
        "claim_seals",
        "content_sha256",
        "data_seals",
        "decision",
        "decision_counts",
        "downstream_admission_counts",
        "exact_controls",
        "factorization_theorem",
        "feature_basis",
        "first_blocker",
        "gate_counts",
        "predecessor_decisions",
        "schema_version",
        "scope",
        "secondary_blockers",
        "source_bindings",
        "universal_coordinate_map",
    }
    if (
        set(value) != expected_top_keys
        or value.get("schema_version")
        != "sigma-quartic-fitted-output-connection-action-feature-factorization-gate-1.0"
        or value.get("decision")
        != "pass_exact_unique_declared_affine_G4_X_factorization_origin_still_blocked"
        or value.get("decision_counts") != {"blocked": 0, "pass": 12, "reject": 0}
        or value.get("downstream_admission_counts")
        != {
            "blocked": 12,
            "pass": 0,
            "reject": 0,
        }
    ):
        raise ValueError("action jet predecessor status boundary changed")
    return value


def _coordinate_key(row: Mapping[str, Any]) -> tuple[str, int, int, str]:
    return (
        str(row["direction"]),
        int(row["output_row"]),
        int(row["input_row"]),
        str(row["atom"]),
    )


def _ambiguity_certificate(
    predecessor: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    coordinates = predecessor.get("universal_coordinate_map")
    if not isinstance(coordinates, list) or len(coordinates) != 22:
        raise ValueError("action jet predecessor coordinate count changed")
    g = sp.Symbol("g")
    null_polynomial = sp.expand(
        (g + 1) * (g + sp.Rational(1, 2)) * (g - sp.Rational(1, 2)) * (g - 1)
    )
    expected_null = g**4 - sp.Rational(5, 4) * g**2 + sp.Rational(1, 4)
    if null_polynomial != expected_null:
        raise AssertionError("action jet null polynomial expansion changed")
    first_derivative = sp.diff(null_polynomial, g)
    second_derivative = sp.diff(first_derivative, g)
    samples = []
    for point in GRID:
        value = sp.simplify(null_polynomial.subs(g, point))
        first = sp.simplify(first_derivative.subs(g, point))
        second = sp.simplify(second_derivative.subs(g, point))
        if value != 0 or first == 0 or second == 0:
            raise ValueError("action jet null-polynomial sample certificate failed")
        samples.append(
            {
                "G4_X": str(point),
                "null_value": str(value),
                "first_jet_lambda_coefficient": str(first),
                "second_jet_lambda_coefficient": str(second),
            }
        )
    records = []
    seen = set()
    for ordinal, row in enumerate(coordinates):
        if not isinstance(row, Mapping) or set(row) != {
            "direction",
            "output_row",
            "input_row",
            "atom",
            "affine_coefficients",
        }:
            raise ValueError("action jet predecessor coordinate schema changed")
        key = _coordinate_key(row)
        if key in seen:
            raise ValueError("action jet predecessor coordinates contain duplicates")
        seen.add(key)
        coefficients = row["affine_coefficients"]
        if (
            not isinstance(coefficients, Mapping)
            or coefficients.get("constant") != "0"
            or coefficients.get("G2_X2") != "0"
        ):
            raise ValueError("action jet predecessor factorization class changed")
        beta = sp.sympify(coefficients.get("G4_X"))
        if beta == 0:
            raise ValueError("action jet predecessor G4_X coefficient vanished")
        records.append(
            {
                "coordinate_ordinal": ordinal,
                "direction": key[0],
                "output_row": key[1],
                "input_row": key[2],
                "atom": key[3],
                "beta": str(beta),
                "extension_family": f"({beta})*g+lambda_{ordinal}*(g^4-5/4*g^2+1/4)",
                "registered_value_equalities": 4,
                "first_jet_ambiguities": 4,
                "second_jet_ambiguities": 4,
                "jet_identified": False,
            }
        )
    return records, {
        "expanded_null_polynomial": "g^4-5/4*g^2+1/4",
        "first_derivative": "4*g^3-5/2*g",
        "second_derivative": "12*g^2-5/2",
        "grid_samples": samples,
    }


def _expected_body(root: Path, config_path: Path, predecessor: Mapping[str, Any]) -> dict[str, Any]:
    records, certificate = _ambiguity_certificate(predecessor)
    return {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "decision": "pass_exact_finite_grid_first_and_second_jet_nonidentifiability",
        "decision_counts": {"pass": 12, "blocked": 0, "reject": 0},
        "downstream_admission_counts": {"pass": 0, "blocked": 12, "reject": 0},
        "first_blocker": FIRST_BLOCKER,
        "jet_nonidentifiability_theorem": {
            "name": "four_point_values_do_not_identify_first_or_second_action_feature_jets",
            "premises": (
                "Each of the 22 fitted coordinates is registered only through its values on "
                "G4_X in {-1,-1/2,1/2,1}, with value beta*g. No derivative functor or "
                "polynomial degree bound below four is registered."
            ),
            "exact_result": (
                "For every coordinate, beta*g + lambda*(g+1)*(g+1/2)*(g-1/2)*(g-1) "
                "agrees at all four registered values for every lambda, while both its first "
                "and second derivatives vary nontrivially with lambda at every grid point. "
                "The product construction gives 22 independent ambiguity parameters."
            ),
            "boundary": (
                "This is an identifiability obstruction from the registered finite value data, "
                "not a no-go theorem for a covariant action derivation. A registered local "
                "variation rule, derivative samples, or corrected second-source jet can select "
                "an extension."
            ),
        },
        "null_polynomial_certificate": certificate,
        "coordinate_ambiguity_records": records,
        "gate_counts": {
            "selected": 12,
            "fitted_connection_coordinates": 22,
            "registered_G4_X_grid_points": 4,
            "registered_value_equalities_replayed": 88,
            "independent_ambiguity_parameters": 22,
            "nonidentified_first_jet_samples": 88,
            "nonidentified_second_jet_samples": 88,
            "registered_covariant_derivation_functors": 0,
            "registered_corrected_second_source_jet_entries": 0,
            "cross_slice_D2F_entries_admitted": 0,
            "complete_ordered_D2F_tensors_registered": 0,
            "full_high_atom_good_unknown_identities_proved": 0,
            "global_H7_closures": 0,
            "nonlinear_PDE_closures": 0,
            "lifespans_proved": 0,
        },
        "claim_seals": dict(CLAIM_SEALS),
        "exact_controls": {
            "choose_linear_extension_without_derivative_evidence": {"rejected": True},
            "infer_covariant_functor_from_four_value_samples": {"rejected": True},
            "infer_corrected_second_source_jet_from_four_value_samples": {"rejected": True},
            "admit_cross_slice_D2F_from_unselected_extension": {
                "rejected": True,
                "cross_slice_entries_admitted": 0,
            },
            "reject_candidates_from_jet_nonidentifiability": {"rejected": True},
        },
        "data_seals": dict(EXPECTED_SEALS),
        "source_bindings": {
            "source": {"path": SOURCE_PATH, "file_sha256": _file_sha(_inside(root, SOURCE_PATH))},
            "config": {"path": CONFIG_PATH, "file_sha256": _file_sha(config_path)},
            "test": {"path": TEST_PATH, "file_sha256": _file_sha(_inside(root, TEST_PATH))},
            "predecessor": _copy_json(EXPECTED_PREDECESSOR),
        },
        "predecessor_decision": predecessor["decision"],
        "scope": (
            "candidate-bound exact nonidentifiability of first and second G4_X jets from the "
            "registered four-point fitted-connection values; no covariant functor, corrected "
            "second-source jet, D2F admission, complete D2F, high-atom identity, H7, PDE, "
            "lifespan, candidate rejection, or observation"
        ),
    }


def _validate_source_bindings(value: Mapping[str, Any], root: Path) -> None:
    bindings = value.get("source_bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != {
        "source",
        "config",
        "test",
        "predecessor",
    }:
        raise ValueError("action jet source binding keys changed")
    for label, relative in {
        "source": SOURCE_PATH,
        "config": CONFIG_PATH,
        "test": TEST_PATH,
    }.items():
        if bindings[label] != {
            "path": relative,
            "file_sha256": _file_sha(_inside(root, relative)),
        }:
            raise ValueError("action jet local source binding changed")
    if bindings["predecessor"] != EXPECTED_PREDECESSOR:
        raise ValueError("action jet predecessor binding changed")


def _validate_result(value: Mapping[str, Any], *, root: Path | None = None) -> None:
    validation_root = (root or Path(__file__).resolve().parents[2]).resolve()
    if value.get("content_sha256") != _content_sha(value):
        raise ValueError("action jet content hash changed")
    _validate_source_bindings(value, validation_root)
    config_path = _inside(validation_root, CONFIG_PATH)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    predecessor = _load_predecessor(validation_root)
    expected = _expected_body(validation_root, config_path, predecessor)
    if {key: item for key, item in value.items() if key != "content_sha256"} != expected:
        raise ValueError("action jet result boundary changed")


def build_gate(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    root = config_path.parents[2]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    predecessor = _load_predecessor(root)
    body = _expected_body(root, config_path, predecessor)
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
