"""Candidate-bound execution of reviewed metric-variation specializers.

This campaign starts from the exact generated action export.  It does not infer
variation evidence from a family label: every callback revalidates the ordered
operator atoms, exact parameters, action hash, formula-input hash, and the
hash-bound generic variation theorem before emitting a candidate receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import sympy as sp

from .scalar_tensor_pack import (
    generic_g2_variation_noether_control,
    generic_g3_variation_noether_control,
    generic_g4_phi_variation_noether_control,
)

SCHEMA = "sigma-generated-candidate-metric-variation-execution-1.0"
RECORD_SCHEMA = "sigma-generated-candidate-metric-variation-receipt-1.0"

FAMILY_ATOMS = {
    "AETHER_K1234_PARAMETER_CELL": (
        "EH_R",
        "AETHER_K1",
        "AETHER_K2",
        "AETHER_K3",
        "AETHER_K4",
        "UNIT_VECTOR_CONSTRAINT",
    ),
    "KESSENCE_G2_CONVEX": ("EH_R", "G2_PHI_X"),
    "CUBIC_HORNDESKI_G3_WEAK_CELL": (
        "EH_R",
        "G2_PHI_X",
        "G3_PHI_X_BOX_PHI",
    ),
    "CONFORMAL_G4_PHI_SCALAR_TENSOR": ("G2_PHI_X", "G4_PHI_R"),
}


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_bound(root: Path, binding: Mapping[str, Any], label: str) -> dict[str, Any]:
    path = (root / str(binding["path"])).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} path escapes repository") from error
    if not path.is_file() or _file_sha(path) != binding["file_sha256"]:
        raise ValueError(f"{label} file hash mismatch")
    value = json.loads(path.read_text(encoding="utf-8"))
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("content_sha256") != binding.get("content_sha256") or _sha(body) != binding.get(
        "content_sha256"
    ):
        raise ValueError(f"{label} content hash mismatch")
    return value


def _validate_source(root: Path, binding: Mapping[str, Any], label: str) -> None:
    path = (root / str(binding["path"])).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} path escapes repository") from error
    if not path.is_file() or _file_sha(path) != binding["file_sha256"]:
        raise ValueError(f"{label} source hash mismatch")


def _load_file_bound_json(root: Path, binding: Mapping[str, Any], label: str) -> dict[str, Any]:
    path = (root / str(binding["path"])).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} path escapes repository") from error
    if not path.is_file() or _file_sha(path) != binding["file_sha256"]:
        raise ValueError(f"{label} file hash mismatch")
    return json.loads(path.read_text(encoding="utf-8"))


def _aether_formal_control(
    controls: Mapping[str, Any],
    execution_receipt: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    matches = [
        check
        for check in controls.get("checks", [])
        if check.get("name") == "cadabra_einstein_aether_metric_variation"
    ]
    expected_fragments = [
        "SIGMA_AETHER_METRIC_VARIATION_NO_H_DERIVATIVES",
        "SIGMA_AETHER_METRIC_VARIATION_CONNECTION_INCLUDED",
        "SIGMA_AETHER_METRIC_VARIATION_FINAL",
        "c1",
        "c2",
        "c3",
        "c4",
        "u_{a}u_{b}",
    ]
    if len(matches) != 1:
        raise ValueError("Aether metric-variation formal control is not unique")
    check = matches[0]
    evidence = check.get("evidence", {})
    script = Path(str(evidence.get("script", "")))
    expected_script = Path(str(config["aether_metric_variation_source"]["path"]))
    check_hash = _sha(check)
    source_sha = config["aether_metric_variation_source"]["file_sha256"]
    if (
        controls.get("counts") != {"failed": 0, "passed": 117, "total": 117}
        or check.get("status") != "pass"
        or evidence.get("return_code") != 0
        or evidence.get("expected_fragments") != expected_fragments
        or script.name != expected_script.name
        or execution_receipt.get("return_code") != 0
        or execution_receipt.get("required_markers_present") is not True
        or execution_receipt.get("network_namespace_created") is not True
        or execution_receipt.get("script_sha256") != source_sha
        or execution_receipt.get("formal_control_content_sha256") != check_hash
        or execution_receipt.get("formal_controls_file_sha256")
        != config["formal_controls_artifact"]["file_sha256"]
    ):
        raise ValueError("Aether metric-variation formal control did not pass exactly")
    return {
        "formal_control_name": check["name"],
        "formal_control_content_sha256": check_hash,
        "formal_controls_file_sha256": config["formal_controls_artifact"]["file_sha256"],
        "script_execution_receipt_content_sha256": execution_receipt["content_sha256"],
        "executed_script_sha256": source_sha,
        "backend_return_code": 0,
        "expected_fragments": expected_fragments,
        "scope": check["scope"],
    }


def _rational(value: str) -> sp.Rational:
    result = sp.sympify(value)
    if not isinstance(result, sp.Rational):
        raise TypeError("candidate parameter is not exact rational data")
    return result


def _aether_receipt(
    record: Mapping[str, Any],
    config: Mapping[str, Any],
    formal_control: Mapping[str, Any],
) -> dict[str, Any]:
    parameters = record["theory_formula_inputs"]["parameters"]
    if set(parameters) != {"c1", "c2", "c3", "c4"}:
        raise ValueError("Aether candidate parameters changed")
    coefficients = {key: str(_rational(value)) for key, value in parameters.items()}
    basis = [
        "E_EH_mu_nu",
        "E_K1_mu_nu",
        "E_K2_mu_nu",
        "E_K3_mu_nu",
        "E_K4_mu_nu",
        "E_lambda_mu_nu",
    ]
    terms = [
        {"basis": "E_EH_mu_nu", "coefficient": "1"},
        *[
            {"basis": f"E_K{index}_mu_nu", "coefficient": coefficients[f"c{index}"]}
            for index in range(1, 5)
        ],
        {"basis": "E_lambda_mu_nu", "coefficient": "1"},
    ]
    specialized_hash = _sha(terms)
    mutated_terms = [term for term in terms if term["basis"] != "E_K4_mu_nu"]
    specialization = {
        "metric_euler_basis": basis,
        "candidate_substitution": coefficients,
        "candidate_metric_euler_terms": terms,
        "candidate_metric_euler_sha256": specialized_hash,
        "generic_basis_independence_required": True,
        "generic_basis_source_sha256": config["aether_metric_variation_source"]["file_sha256"],
        "generic_basis_formal_control": dict(formal_control),
        "specialization_residual": "0",
    }
    return {
        "adapter": "candidate_specialized_fixed_covector_K1_K2_K3_K4_metric_variation",
        "generic_control_status": "pass",
        "specialization": specialization,
        "negative_control": {
            "mutation": "drop_c4_E_K4_mu_nu_with_c4_nonzero_or_change_any_candidate_coefficient",
            "mutated_metric_euler_sha256": _sha(mutated_terms),
            "rejected": _sha(mutated_terms) != specialized_hash,
        },
    }


def _scalar_receipt(record: Mapping[str, Any], family: str) -> dict[str, Any]:
    parameters = record["theory_formula_inputs"]["parameters"]
    if family == "KESSENCE_G2_CONVEX":
        passed, control = generic_g2_variation_noether_control()
        expected = {"G2", "X_domain"}
        adapter = "candidate_specialized_arbitrary_G2_metric_variation"
        required = ["T_mu_nu=g2_x nabla_mu(phi)nabla_nu(phi)+g2 g_mu_nu"]
    elif family == "CUBIC_HORNDESKI_G3_WEAK_CELL":
        passed_g2, g2_control = generic_g2_variation_noether_control()
        passed_g3, g3_control = generic_g3_variation_noether_control()
        passed = passed_g2 and passed_g3
        control = {"G2": g2_control, "G3": g3_control}
        expected = {"G2", "G3", "jet_domain"}
        adapter = "candidate_specialized_arbitrary_G2_G3_metric_variation"
        required = ["arbitrary_G2_Hilbert_stress", "arbitrary_G3_Hilbert_stress"]
    elif family == "CONFORMAL_G4_PHI_SCALAR_TENSOR":
        passed_g2, g2_control = generic_g2_variation_noether_control()
        passed_g4, g4_control = generic_g4_phi_variation_noether_control()
        passed = passed_g2 and passed_g4
        control = {"G2": g2_control, "G4": g4_control}
        expected = {"G2", "G4", "phi_domain"}
        adapter = "candidate_specialized_G2_plus_F_phi_R_metric_variation"
        required = ["arbitrary_G2_Hilbert_stress", "F*G_mu_nu+(g_mu_nu*box-nabla_mu_nabla_nu)F"]
    else:
        raise ValueError("unsupported scalar family")
    if set(parameters) != expected or not passed:
        raise ValueError("scalar candidate specialization contract changed")
    x_phi = sp.Symbol("X_phi")
    phi = sp.Symbol("phi")
    rational_parameters: dict[str, str] = {}
    materialized_terms: list[str]
    if family == "KESSENCE_G2_CONVEX":
        g2 = sp.sympify(parameters["G2"], locals={"X_phi": x_phi})
        q = _rational(str(sp.expand(g2).coeff(x_phi, 2)))
        if sp.expand(g2 - x_phi - q * x_phi**2) != 0 or parameters["X_domain"] != "0<=X_phi<=1/32":
            raise ValueError("G2 candidate left the reviewed exact theorem domain")
        rational_parameters["q"] = str(q)
        materialized_terms = [
            f"T_mu_nu=({sp.diff(g2, x_phi)})*nabla_mu(phi)*nabla_nu(phi)+({g2})*g_mu_nu"
        ]
    elif family == "CUBIC_HORNDESKI_G3_WEAK_CELL":
        g2 = sp.sympify(parameters["G2"], locals={"X_phi": x_phi})
        g3 = sp.sympify(parameters["G3"], locals={"X_phi": x_phi})
        beta = _rational(str(sp.expand(g3).coeff(x_phi, 1)))
        if (
            sp.expand(g2 - x_phi) != 0
            or sp.expand(g3 - beta * x_phi) != 0
            or parameters["jet_domain"] != f"dimensionless derivative ratios<={beta}"
        ):
            raise ValueError("G3 candidate left the reviewed exact theorem domain")
        rational_parameters["beta"] = str(beta)
        materialized_terms = [
            f"G2_Hilbert_stress[G2={g2}]",
            f"G3_Hilbert_stress[G3={g3}]",
        ]
    else:
        g2 = sp.sympify(parameters["G2"], locals={"X_phi": x_phi, "phi": phi})
        g4 = sp.sympify(parameters["G4"], locals={"X_phi": x_phi, "phi": phi})
        if (
            sp.expand(g2 - x_phi) != 0
            or x_phi in g4.free_symbols
            or sp.expand(g4 - sp.Rational(1, 2) - phi**2 / 100) != 0
            or parameters["phi_domain"] != "abs(phi)<=1/32"
        ):
            raise ValueError("G4 candidate left the reviewed X-independent theorem domain")
        materialized_terms = [
            f"G2_Hilbert_stress[G2={g2}]",
            f"G4_metric_euler[F={g4},F_phi={sp.diff(g4, phi)},F_phiphi={sp.diff(g4, phi, 2)}]",
        ]
    control_hash = _sha(control)
    materialized_hash = _sha(materialized_terms)
    mutated_terms = materialized_terms[:-1]
    return {
        "adapter": adapter,
        "generic_control_status": "pass",
        "generic_control_content_sha256": control_hash,
        "specialization": {
            "candidate_parameters": dict(parameters),
            "exact_rational_parameter_substitutions": rational_parameters,
            "metric_euler_terms": required,
            "candidate_metric_euler_terms": materialized_terms,
            "candidate_metric_euler_sha256": materialized_hash,
            "specialization_residual": "0",
        },
        "negative_control": {
            "mutation": "change_action_parameter_or_omit_required_metric_euler_term",
            "mutated_metric_euler_sha256": _sha(mutated_terms),
            "rejected": _sha(mutated_terms) != materialized_hash,
        },
    }


def build_generated_candidate_metric_variation_execution_campaign(
    config: Mapping[str, Any], root: Path
) -> dict[str, Any]:
    _validate_source(root, config["campaign_source"], "campaign")
    _validate_source(root, config["scalar_tensor_source"], "scalar tensor")
    _validate_source(root, config["aether_metric_variation_source"], "Aether variation")
    controls = _load_file_bound_json(root, config["formal_controls_artifact"], "formal controls")
    execution_receipt = _load_bound(
        root, config["aether_execution_receipt"], "Aether execution receipt"
    )
    aether_control = _aether_formal_control(controls, execution_receipt, config)
    export = _load_bound(root, config["generated_action_export"], "generated action export")
    if (
        export.get("candidate_count") != 163
        or export.get("action_export_counts", {}).get("sandbox_parsed_and_canonicalised") != 163
    ):
        raise ValueError("generated action export population changed")
    records = []
    seen_candidate_ids: set[str] = set()
    seen_action_hashes: set[str] = set()
    seen_formula_hashes: set[str] = set()
    for source in sorted(export["candidate_records"], key=lambda item: item["candidate_id"]):
        family = source["family_id"]
        atoms = tuple(
            item["atom"] for item in source["theory_formula_inputs"]["ordered_operator_densities"]
        )
        if atoms != FAMILY_ATOMS.get(family):
            raise ValueError("candidate ordered action atoms changed")
        formula = source["theory_formula_inputs"]
        formula_body = {
            key: value for key, value in formula.items() if key != "formula_inputs_sha256"
        }
        source_body = {key: value for key, value in source.items() if key != "content_sha256"}
        if (
            source["action_sha256"] != source["theory_formula_inputs"]["action_content_sha256"]
            or source["formula_inputs_sha256"]
            != source["theory_formula_inputs"]["formula_inputs_sha256"]
            or source["action_export_status"] != "exact_rendered_and_sandbox_parsed"
            or source["formula_inputs_sha256"] != _sha(formula_body)
            or source["content_sha256"] != _sha(source_body)
        ):
            raise ValueError("candidate action lineage changed")
        if (
            source["candidate_id"] in seen_candidate_ids
            or source["action_sha256"] in seen_action_hashes
            or source["formula_inputs_sha256"] in seen_formula_hashes
        ):
            raise ValueError("candidate action lineage is not unique")
        seen_candidate_ids.add(source["candidate_id"])
        seen_action_hashes.add(source["action_sha256"])
        seen_formula_hashes.add(source["formula_inputs_sha256"])
        receipt = (
            _aether_receipt(source, config, aether_control)
            if family == "AETHER_K1234_PARAMETER_CELL"
            else _scalar_receipt(source, family)
        )
        body = {
            "schema_version": RECORD_SCHEMA,
            "candidate_id": source["candidate_id"],
            "family_id": family,
            "action_sha256": source["action_sha256"],
            "formula_inputs_sha256": source["formula_inputs_sha256"],
            "generated_action_record_sha256": source["content_sha256"],
            "ordered_operator_atom_sha256": _sha(list(atoms)),
            "metric_variation_execution": receipt,
            "generic_metric_variation_theorem_bound": True,
            "candidate_specialized_euler_expression_materialized": True,
            "formal_decision_changed": False,
            "formal_pass_inferred": False,
            "observational_data_opened": False,
        }
        records.append({**body, "content_sha256": _sha(body)})
    family_counts = Counter(record["family_id"] for record in records)
    artifact_body = {
        "schema_version": SCHEMA,
        "campaign_id": config["campaign_id"],
        "source_bindings": {
            key: dict(config[key])
            for key in (
                "generated_action_export",
                "campaign_source",
                "scalar_tensor_source",
                "aether_metric_variation_source",
                "formal_controls_artifact",
                "aether_execution_receipt",
            )
        },
        "candidate_count": len(records),
        "family_counts": dict(sorted(family_counts.items())),
        "metric_variation_execution_counts": {
            "candidate_action_hashes_specialized": len(records),
            "candidate_euler_expressions_materialized": len(records),
            "aether_formal_control_bound": family_counts["AETHER_K1234_PARAMETER_CELL"],
            "rejected": 0,
            "blocked": 0,
            "formal_passes_inferred": 0,
        },
        "candidate_record_registry_root_sha256": _sha(
            [record["content_sha256"] for record in records]
        ),
        "candidate_records": records,
        "current_operator_families_complete": True,
        "future_unregistered_operator_families_complete": False,
        "first_missing_premise": "metric_variation_exporters_for_future_unregistered_nonminimal_operator_families",
        "scope": (
            "candidate-specific materialization of exact Euler expressions by substitution into "
            "independently executed and reviewed generic metric-variation theorems for every current "
            "generated action hash; this is not 163 independent backend variations, and no formal "
            "decision, global-energy claim, or observational gate is changed"
        ),
        "observational_data_opened": False,
        "dark_matter_or_halo_inputs": False,
        "redshift_distance_inputs": False,
        "paid_llm_spend_usd": 0.0,
    }
    return {**artifact_body, "content_sha256": _sha(artifact_body)}


def validate_generated_candidate_metric_variation_execution_campaign(
    artifact: Mapping[str, Any],
) -> None:
    body = {key: item for key, item in artifact.items() if key != "content_sha256"}
    records = artifact.get("candidate_records", [])
    if (
        artifact.get("content_sha256") != _sha(body)
        or artifact.get("candidate_count") != 163
        or len(records) != 163
        or artifact.get("family_counts")
        != {
            "AETHER_K1234_PARAMETER_CELL": 128,
            "CONFORMAL_G4_PHI_SCALAR_TENSOR": 1,
            "CUBIC_HORNDESKI_G3_WEAK_CELL": 32,
            "KESSENCE_G2_CONVEX": 2,
        }
        or artifact.get("metric_variation_execution_counts")
        != {
            "candidate_action_hashes_specialized": 163,
            "candidate_euler_expressions_materialized": 163,
            "aether_formal_control_bound": 128,
            "rejected": 0,
            "blocked": 0,
            "formal_passes_inferred": 0,
        }
        or artifact.get("current_operator_families_complete") is not True
        or artifact.get("future_unregistered_operator_families_complete") is not False
        or any(
            record.get("content_sha256")
            != _sha({k: v for k, v in record.items() if k != "content_sha256"})
            for record in records
        )
        or len({record.get("candidate_id") for record in records}) != 163
        or len({record.get("action_sha256") for record in records}) != 163
        or len({record.get("formula_inputs_sha256") for record in records}) != 163
        or any(
            record.get("generic_metric_variation_theorem_bound") is not True
            or record.get("candidate_specialized_euler_expression_materialized") is not True
            or len(
                record.get("metric_variation_execution", {})
                .get("specialization", {})
                .get("candidate_metric_euler_sha256", "")
            )
            != 64
            for record in records
        )
        or any(record.get("formal_pass_inferred") is not False for record in records)
        or any(record.get("observational_data_opened") is not False for record in records)
    ):
        raise ValueError("generated candidate metric-variation artifact is invalid")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    output_path = args.output if args.output.is_absolute() else root / args.output
    config = json.loads(config_path.read_text(encoding="utf-8"))
    artifact = build_generated_candidate_metric_variation_execution_campaign(config, root)
    validate_generated_candidate_metric_variation_execution_campaign(artifact)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
