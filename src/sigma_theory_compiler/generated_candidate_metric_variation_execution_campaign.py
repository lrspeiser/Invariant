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

from .covariant_grammar_v3_seed_compilation_campaign import _compile_action_ir
from .grammar_v3_parameter_cell_compilation_campaign import _action_density_key
from .grammar_v3_parameter_cell_manifest_campaign import iter_parameter_cells
from .promotion_orchestrator import ELIGIBILITY
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

FAMILY_FIELDS = {
    "AETHER_K1234_PARAMETER_CELL": ("g_mu_nu", "u_mu", "lambda_u"),
    "KESSENCE_G2_CONVEX": ("g_mu_nu", "phi"),
    "CUBIC_HORNDESKI_G3_WEAK_CELL": ("g_mu_nu", "phi"),
    "CONFORMAL_G4_PHI_SCALAR_TENSOR": ("g_mu_nu", "phi"),
}

CONFIG_KEYS = {
    "schema_version",
    "campaign_id",
    "generated_action_export",
    "campaign_source",
    "scalar_tensor_source",
    "aether_metric_variation_source",
    "formal_controls_artifact",
    "aether_execution_receipt",
    "compilation_config",
    "maximum_candidates",
    "maximum_paid_llm_spend_usd",
    "observational_data_opened",
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


def _validate_config(config: Mapping[str, Any]) -> None:
    if (
        set(config) != CONFIG_KEYS
        or config.get("schema_version")
        != "sigma-generated-candidate-metric-variation-execution-config-1.0"
        or config.get("maximum_candidates") != 163
        or config.get("maximum_paid_llm_spend_usd") != 0.0
        or config.get("observational_data_opened") is not False
    ):
        raise ValueError("metric-variation execution config is not fail-closed")


def _bound_compiler_input(
    root: Path, compiler_config: Mapping[str, Any], key: str, label: str
) -> dict[str, Any]:
    binding = compiler_config.get(key)
    if not isinstance(binding, Mapping):
        raise TypeError(f"{label} binding is missing")
    return _load_bound(root, binding, label)


def _replay_action_registry(
    root: Path, config: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    compiler_config = _load_file_bound_json(
        root, config["compilation_config"], "compilation config"
    )
    manifest = _bound_compiler_input(
        root, compiler_config, "parameter_cell_manifest", "parameter-cell manifest"
    )
    seed_manifest = _bound_compiler_input(
        root, compiler_config, "source_seed_manifest", "source seed manifest"
    )
    _validate_source(root, compiler_config["compiler_semantics"], "compiler semantics")
    families = {
        item["family_id"]: item for item in seed_manifest["typed_family_seeds"]
    }
    manifest_binding = {
        "parameter_cell_manifest_content_sha256": manifest["content_sha256"],
        "parameter_cell_registry_root_sha256": manifest[
            "parameter_cell_registry_root_sha256"
        ],
    }
    registry: dict[str, dict[str, Any]] = {}
    alias_count = 0
    for cell in iter_parameter_cells(manifest, seed_manifest):
        family = families.get(cell["family_id"])
        if family is None:
            raise ValueError("typed compiler family is missing")
        pseudo_seed = {
            "seed_id": cell["parameter_cell_id"],
            "seed_lineage_sha256": cell["parameter_cell_lineage_sha256"],
            "family_id": cell["family_id"],
            "family_lineage_sha256": cell["family_lineage_sha256"],
            "theory_contract": cell["theory_contract"],
            "operator_atoms": cell["operator_atoms"],
            "parameters": cell["parameters"],
        }
        action = _compile_action_ir(pseudo_seed, family, manifest_binding)
        density_key = _action_density_key(cell)
        equivalence_sha256 = _sha(density_key)
        candidate_id = "G3A-" + equivalence_sha256[:24]
        formula_body = {
            "fields": action["fields"],
            "parameters": action["parameters"],
            "ordered_operator_densities": [
                {"atom": item["atom"], "density": item["density"]}
                for item in action["operators"]
            ],
            "action_content_sha256": action["content_sha256"],
        }
        replay = {
            "candidate_id": candidate_id,
            "family_id": action["family_id"],
            "action_sha256": action["content_sha256"],
            "action_density_equivalence_sha256": equivalence_sha256,
            "action_density_key": density_key,
            "formula_body": formula_body,
            "formula_inputs_sha256": _sha(formula_body),
            "matter_coupling": action["matter_coupling"],
            "data_eligibility": action["data_eligibility"],
        }
        existing = registry.get(candidate_id)
        if existing is not None:
            alias_count += 1
            if (
                existing["family_id"] != replay["family_id"]
                or existing["action_density_equivalence_sha256"]
                != replay["action_density_equivalence_sha256"]
                or existing["action_density_key"] != replay["action_density_key"]
                or existing["matter_coupling"] != replay["matter_coupling"]
                or existing["data_eligibility"] != replay["data_eligibility"]
            ):
                raise ValueError("typed compiler aliases do not replay identically")
        else:
            registry[candidate_id] = replay
    if (
        len(registry) != 163
        or alias_count != 93
        or any(item["data_eligibility"] != ELIGIBILITY for item in registry.values())
        or any(
            item["matter_coupling"] != {"metric": "g_mu_nu", "universal": True}
            for item in registry.values()
        )
    ):
        raise ValueError("typed compiler replay registry changed")
    return registry


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
    receipt_body = {
        key: value for key, value in execution_receipt.items() if key != "content_sha256"
    }
    expected_markers = expected_fragments[:3]
    if (
        controls.get("counts") != {"failed": 0, "passed": 117, "total": 117}
        or check.get("status") != "pass"
        or evidence.get("return_code") != 0
        or evidence.get("expected_fragments") != expected_fragments
        or script.parts[-len(expected_script.parts) :] != expected_script.parts
        or execution_receipt.get("schema_version")
        != "sigma-cadabra-script-execution-receipt-1.0"
        or execution_receipt.get("content_sha256") != _sha(receipt_body)
        or execution_receipt.get("return_code") != 0
        or execution_receipt.get("script_path") != expected_script.as_posix()
        or execution_receipt.get("required_markers") != expected_markers
        or execution_receipt.get("required_markers_present") is not True
        or execution_receipt.get("coefficient_symbols_present")
        != ["c1", "c2", "c3", "c4"]
        or execution_receipt.get("network_namespace_created") is not True
        or execution_receipt.get("script_sha256") != source_sha
        or execution_receipt.get("formal_control_content_sha256") != check_hash
        or execution_receipt.get("formal_controls_file_sha256")
        != config["formal_controls_artifact"]["file_sha256"]
    ):
        raise ValueError("Aether metric-variation formal control did not pass exactly")
    body = {
        "formal_control_name": check["name"],
        "formal_control_content_sha256": check_hash,
        "formal_controls_file_sha256": config["formal_controls_artifact"]["file_sha256"],
        "script_execution_receipt_content_sha256": execution_receipt["content_sha256"],
        "executed_script_sha256": source_sha,
        "executed_script_path": expected_script.as_posix(),
        "backend_return_code": 0,
        "required_markers": expected_markers,
        "expected_fragments": expected_fragments,
        "scope": check["scope"],
    }
    return {**body, "script_formal_control_binding_sha256": _sha(body)}


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
    mutated_terms = [dict(term) for term in terms]
    mutated_terms[1]["coefficient"] = str(_rational(coefficients["c1"]) + 1)
    domain_certificate = {
        "fields": list(FAMILY_FIELDS["AETHER_K1234_PARAMETER_CELL"]),
        "ordered_operator_atoms": list(FAMILY_ATOMS["AETHER_K1234_PARAMETER_CELL"]),
        "coefficient_domain": "exact_rational_c1_c2_c3_c4",
        "candidate_coefficients": coefficients,
        "status": "pass_exact_candidate_substitution",
    }
    specialization = {
        "metric_euler_basis": basis,
        "candidate_substitution": coefficients,
        "candidate_metric_euler_terms": terms,
        "candidate_metric_euler_sha256": specialized_hash,
        "generic_basis_independence_required": True,
        "generic_basis_source_sha256": config["aether_metric_variation_source"]["file_sha256"],
        "generic_basis_formal_control": dict(formal_control),
        "exact_formula_domain_certificate": domain_certificate,
        "exact_formula_domain_certificate_sha256": _sha(domain_certificate),
        "specialization_residual": "0",
    }
    return {
        "adapter": "candidate_specialized_fixed_covector_K1_K2_K3_K4_metric_variation",
        "generic_control_status": "pass",
        "specialization": specialization,
        "negative_control": {
            "mutation": "replace_candidate_c1_by_c1_plus_1",
            "mutated_candidate_metric_euler_terms": mutated_terms,
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
    domain_certificate: dict[str, Any]
    if family == "KESSENCE_G2_CONVEX":
        g2 = sp.sympify(parameters["G2"], locals={"X_phi": x_phi})
        q = _rational(str(sp.expand(g2).coeff(x_phi, 2)))
        if sp.expand(g2 - x_phi - q * x_phi**2) != 0 or parameters["X_domain"] != "0<=X_phi<=1/32":
            raise ValueError("G2 candidate left the reviewed exact theorem domain")
        rational_parameters["q"] = str(q)
        materialized_terms = [
            f"T_mu_nu=({sp.diff(g2, x_phi)})*nabla_mu(phi)*nabla_nu(phi)+({g2})*g_mu_nu"
        ]
        domain_certificate = {
            "G2": str(g2),
            "G2_X": str(sp.diff(g2, x_phi)),
            "G2_XX": str(sp.diff(g2, x_phi, 2)),
            "q_positive": bool(q > 0),
            "X_domain": parameters["X_domain"],
            "status": "pass_exact_G2_quadratic_domain",
        }
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
            "T2_mu_nu=p_mu*p_nu+X_phi*g_mu_nu",
            (
                f"T3_mu_nu=-({beta})*theta*p_mu*p_nu"
                f"-2*({beta})*q_(mu*p_nu)+({beta})*g_mu_nu*q_rho*p^rho"
            ),
        ]
        domain_certificate = {
            "G2": str(g2),
            "G3": str(g3),
            "G3_X": str(sp.diff(g3, x_phi)),
            "G3_XX": str(sp.diff(g3, x_phi, 2)),
            "beta_positive": bool(beta > 0),
            "jet_domain": parameters["jet_domain"],
            "status": "pass_exact_linear_G3_jet_domain",
        }
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
            "T2_mu_nu=p_mu*p_nu+X_phi*g_mu_nu",
            (
                f"H4_mu_nu=({g4})*G_mu_nu+g_mu_nu*((phi/50)*theta-X_phi/25)"
                "-(phi/50)*H_mu_nu-(1/50)*p_mu*p_nu"
            ),
        ]
        domain_certificate = {
            "G2": str(g2),
            "G4": str(g4),
            "G4_X": str(sp.diff(g4, x_phi)),
            "G4_XX": str(sp.diff(g4, x_phi, 2)),
            "G4_phi": str(sp.diff(g4, phi)),
            "G4_phiphi": str(sp.diff(g4, phi, 2)),
            "X_independence_exact": sp.diff(g4, x_phi) == 0,
            "phi_domain": parameters["phi_domain"],
            "status": "pass_exact_X_independent_conformal_G4_domain",
        }
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
            "exact_formula_domain_certificate": domain_certificate,
            "exact_formula_domain_certificate_sha256": _sha(domain_certificate),
            "specialization_residual": "0",
        },
        "negative_control": {
            "mutation": "change_action_parameter_or_omit_required_metric_euler_term",
            "mutated_candidate_metric_euler_terms": mutated_terms,
            "mutated_metric_euler_sha256": _sha(mutated_terms),
            "rejected": _sha(mutated_terms) != materialized_hash,
        },
    }


def _bind_candidate_specialization(
    source: Mapping[str, Any],
    replay: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    specialization = receipt["specialization"]
    euler_terms = specialization["candidate_metric_euler_terms"]
    domain = specialization["exact_formula_domain_certificate"]
    if (
        specialization["candidate_metric_euler_sha256"] != _sha(euler_terms)
        or specialization["exact_formula_domain_certificate_sha256"] != _sha(domain)
        or specialization.get("specialization_residual") != "0"
        or receipt.get("negative_control", {}).get("rejected") is not True
    ):
        raise ValueError("candidate Euler specialization did not verify exactly")
    binding_body = {
        "candidate_id": source["candidate_id"],
        "action_sha256": replay["action_sha256"],
        "action_density_equivalence_sha256": replay[
            "action_density_equivalence_sha256"
        ],
        "formula_inputs_sha256": replay["formula_inputs_sha256"],
        "candidate_metric_euler_sha256": specialization[
            "candidate_metric_euler_sha256"
        ],
        "formula_domain_certificate_sha256": specialization[
            "exact_formula_domain_certificate_sha256"
        ],
        "generic_control_sha256": _sha(
            {
                key: value
                for key, value in receipt.items()
                if key not in {"specialization", "negative_control"}
            }
        ),
    }
    return {
        **receipt,
        "candidate_specialization_binding": {
            **binding_body,
            "content_sha256": _sha(binding_body),
        },
        "candidate_backend_metric_variation_executed": False,
        "claim_scope": (
            "exact candidate substitution into a reviewed generic metric-variation theorem; "
            "not an independent candidate backend variation"
        ),
    }


def build_generated_candidate_metric_variation_execution_campaign(
    config: Mapping[str, Any], root: Path
) -> dict[str, Any]:
    _validate_config(config)
    _validate_source(root, config["campaign_source"], "campaign")
    _validate_source(root, config["scalar_tensor_source"], "scalar tensor")
    _validate_source(root, config["aether_metric_variation_source"], "Aether variation")
    controls = _load_file_bound_json(root, config["formal_controls_artifact"], "formal controls")
    execution_receipt = _load_bound(
        root, config["aether_execution_receipt"], "Aether execution receipt"
    )
    aether_control = _aether_formal_control(controls, execution_receipt, config)
    replay_registry = _replay_action_registry(root, config)
    export = _load_bound(root, config["generated_action_export"], "generated action export")
    if (
        export.get("candidate_count") != 163
        or export.get("action_export_counts", {}).get("sandbox_parsed_and_canonicalised") != 163
        or len(export.get("candidate_records", [])) != 163
        or export.get("candidate_record_registry_root_sha256")
        != _sha([item["content_sha256"] for item in export["candidate_records"]])
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
        replay = replay_registry.get(source["candidate_id"])
        if (
            replay is None
            or replay["family_id"] != family
            or tuple(formula.get("fields", [])) != FAMILY_FIELDS.get(family)
            or formula_body != replay["formula_body"]
            or source["action_sha256"] != replay["action_sha256"]
            or source["action_sha256"]
            != source["theory_formula_inputs"]["action_content_sha256"]
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
        receipt = _bind_candidate_specialization(source, replay, receipt)
        body = {
            "schema_version": RECORD_SCHEMA,
            "candidate_id": source["candidate_id"],
            "family_id": family,
            "action_sha256": source["action_sha256"],
            "formula_inputs_sha256": source["formula_inputs_sha256"],
            "generated_action_record_sha256": source["content_sha256"],
            "replayed_action_density_equivalence_sha256": replay[
                "action_density_equivalence_sha256"
            ],
            "ordered_operator_atom_sha256": _sha(list(atoms)),
            "metric_variation_execution": receipt,
            "generic_metric_variation_theorem_bound": True,
            "candidate_specialized_euler_expression_materialized": True,
            "candidate_formula_domain_validated": True,
            "candidate_action_hash_replayed": True,
            "candidate_backend_metric_variation_executed": False,
            "formal_decision_changed": False,
            "formal_pass_inferred": False,
            "observational_data_opened": False,
        }
        records.append({**body, "content_sha256": _sha(body)})
    if seen_candidate_ids != set(replay_registry):
        raise ValueError("generated export and typed compiler replay populations differ")
    family_counts = Counter(record["family_id"] for record in records)
    specialization_registry = [
        [
            record["candidate_id"],
            record["action_sha256"],
            record["formula_inputs_sha256"],
            record["metric_variation_execution"]["specialization"][
                "candidate_metric_euler_sha256"
            ],
            record["metric_variation_execution"]["candidate_specialization_binding"][
                "content_sha256"
            ],
        ]
        for record in records
    ]
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
                "compilation_config",
            )
        },
        "candidate_count": len(records),
        "family_counts": dict(sorted(family_counts.items())),
        "metric_variation_execution_counts": {
            "candidate_action_hashes_specialized": len(records),
            "candidate_euler_expressions_materialized": len(records),
            "typed_action_hashes_replayed": len(records),
            "exact_formula_domains_validated": len(records),
            "candidate_specializations_symbolically_verified": len(records),
            "candidate_backend_variations_executed": 0,
            "aether_formal_control_bound": family_counts["AETHER_K1234_PARAMETER_CELL"],
            "rejected": 0,
            "blocked": 0,
            "formal_passes_inferred": 0,
        },
        "candidate_record_registry_root_sha256": _sha(
            [record["content_sha256"] for record in records]
        ),
        "candidate_specialization_registry_root_sha256": _sha(specialization_registry),
        "candidate_records": records,
        "current_operator_families_complete": True,
        "future_unregistered_operator_families_complete": False,
        "first_missing_premise": "metric_variation_exporters_for_future_unregistered_nonminimal_operator_families",
        "scope": (
            "candidate-specific materialization and symbolic verification of exact Euler "
            "specializations by substitution into independently executed or reviewed generic "
            "metric-variation theorems for every current replayed action hash; this is not 163 "
            "independent backend variations, and no formal "
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
            "typed_action_hashes_replayed": 163,
            "exact_formula_domains_validated": 163,
            "candidate_specializations_symbolically_verified": 163,
            "candidate_backend_variations_executed": 0,
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
        or len(
            {
                record.get("metric_variation_execution", {})
                .get("specialization", {})
                .get("candidate_metric_euler_sha256")
                for record in records
            }
        )
        != 163
        or any(
            record.get("generic_metric_variation_theorem_bound") is not True
            or record.get("candidate_specialized_euler_expression_materialized") is not True
            or record.get("candidate_formula_domain_validated") is not True
            or record.get("candidate_action_hash_replayed") is not True
            or record.get("candidate_backend_metric_variation_executed") is not False
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
    specialization_registry = []
    for record in records:
        execution = record["metric_variation_execution"]
        specialization = execution["specialization"]
        domain = specialization["exact_formula_domain_certificate"]
        binding = execution["candidate_specialization_binding"]
        binding_body = {
            key: value for key, value in binding.items() if key != "content_sha256"
        }
        negative = execution["negative_control"]
        if (
            specialization["candidate_metric_euler_sha256"]
            != _sha(specialization["candidate_metric_euler_terms"])
            or specialization["exact_formula_domain_certificate_sha256"]
            != _sha(domain)
            or binding["content_sha256"] != _sha(binding_body)
            or binding["candidate_id"] != record["candidate_id"]
            or binding["action_sha256"] != record["action_sha256"]
            or binding["formula_inputs_sha256"] != record["formula_inputs_sha256"]
            or binding["candidate_metric_euler_sha256"]
            != specialization["candidate_metric_euler_sha256"]
            or negative["mutated_metric_euler_sha256"]
            != _sha(negative["mutated_candidate_metric_euler_terms"])
            or negative["mutated_metric_euler_sha256"]
            == specialization["candidate_metric_euler_sha256"]
            or execution.get("candidate_backend_metric_variation_executed") is not False
        ):
            raise ValueError("candidate metric-variation specialization binding is invalid")
        specialization_registry.append(
            [
                record["candidate_id"],
                record["action_sha256"],
                record["formula_inputs_sha256"],
                specialization["candidate_metric_euler_sha256"],
                binding["content_sha256"],
            ]
        )
    if artifact.get("candidate_record_registry_root_sha256") != _sha(
        [record["content_sha256"] for record in records]
    ) or artifact.get("candidate_specialization_registry_root_sha256") != _sha(
        specialization_registry
    ):
        raise ValueError("candidate metric-variation registry root is invalid")


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
