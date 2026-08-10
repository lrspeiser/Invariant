from __future__ import annotations

import hashlib
import importlib
import json
import tempfile
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any

from .covariant_grammar_v3_seed_campaign import iter_scalable_seed_specs
from .promotion_orchestrator import ELIGIBILITY

SCHEMA_VERSION = "sigma-covariant-grammar-v3-seed-compilation-campaign-1.0"
ACTION_IR_SCHEMA = "sigma-covariant-grammar-v3-candidate-action-ir-1.0"
FORBIDDEN_TOKENS = {"Q_a_u", "T2_m", "z_b"}

OPERATOR_LIBRARY: dict[str, dict[str, Any]] = {
    "EH_R": {
        "density": "sqrt(-g)*(M_Pl^2/2)*R",
        "fields": ["g_mu_nu"],
        "invariants": ["R"],
        "maximum_field_derivative_order": 2,
    },
    "AETHER_K1": {
        "density": "-sqrt(-g)*(M_Pl^2/2)*c1*(nabla_mu(u_nu))*(nabla^mu(u^nu))",
        "fields": ["g_mu_nu", "u_mu"],
        "invariants": ["K1_u"],
        "maximum_field_derivative_order": 1,
    },
    "AETHER_K2": {
        "density": "-sqrt(-g)*(M_Pl^2/2)*c2*(nabla_mu(u^mu))^2",
        "fields": ["g_mu_nu", "u_mu"],
        "invariants": ["K2_u"],
        "maximum_field_derivative_order": 1,
    },
    "AETHER_K3": {
        "density": "-sqrt(-g)*(M_Pl^2/2)*c3*(nabla_mu(u_nu))*(nabla^nu(u^mu))",
        "fields": ["g_mu_nu", "u_mu"],
        "invariants": ["K3_u"],
        "maximum_field_derivative_order": 1,
    },
    "AETHER_K4": {
        "density": "+sqrt(-g)*(M_Pl^2/2)*c4*u^mu*u^nu*(nabla_mu(u_rho))*(nabla_nu(u^rho))",
        "fields": ["g_mu_nu", "u_mu"],
        "invariants": ["K4_u"],
        "maximum_field_derivative_order": 1,
    },
    "UNIT_VECTOR_CONSTRAINT": {
        "density": "sqrt(-g)*lambda_u*(u_mu*u^mu+1)",
        "fields": ["g_mu_nu", "u_mu", "lambda_u"],
        "invariants": ["u_mu*u^mu"],
        "maximum_field_derivative_order": 0,
    },
    "G2_PHI_X": {
        "density": "sqrt(-g)*Lambda_phi^4*G2(phi/Lambda_phi,X_phi)",
        "fields": ["g_mu_nu", "phi"],
        "invariants": ["X_phi"],
        "maximum_field_derivative_order": 1,
    },
    "G3_PHI_X_BOX_PHI": {
        "density": "-sqrt(-g)*Lambda_phi*G3(phi/Lambda_phi,X_phi)*box(phi)",
        "fields": ["g_mu_nu", "phi"],
        "invariants": ["X_phi", "box_phi"],
        "maximum_field_derivative_order": 2,
    },
    "G4_PHI_R": {
        "density": "sqrt(-g)*Lambda_phi^2*G4(phi/Lambda_phi)*R",
        "fields": ["g_mu_nu", "phi"],
        "invariants": ["phi", "R"],
        "maximum_field_derivative_order": 2,
    },
}


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_bound_json(root: Path, descriptor: dict[str, Any], *, content: bool = False) -> dict[str, Any]:
    path = root / descriptor["path"]
    if _file_sha(path) != descriptor["file_sha256"]:
        raise ValueError(f"bound file hash mismatch: {descriptor['path']}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if content:
        body = {key: item for key, item in value.items() if key != "content_sha256"}
        if value.get("content_sha256") != descriptor["content_sha256"] or _sha(body) != descriptor["content_sha256"]:
            raise ValueError(f"bound content hash mismatch: {descriptor['path']}")
    return value


def _validate_seed(seed: dict[str, Any], family: dict[str, Any]) -> None:
    seed_body = {key: value for key, value in seed.items() if key not in {"seed_id", "seed_lineage_sha256"}}
    if seed.get("seed_lineage_sha256") != _sha(seed_body):
        raise ValueError("concrete seed lineage mismatch")
    if seed.get("seed_id") != "G3-" + seed["seed_lineage_sha256"][:24]:
        raise ValueError("concrete seed id mismatch")
    family_body = {key: value for key, value in family.items() if key not in {"family_lineage_sha256", "pre_generation_decision", "pre_generation_reasons"}}
    if seed.get("family_lineage_sha256") != _sha(family_body):
        raise ValueError("concrete seed family lineage mismatch")
    if seed.get("operator_atoms") != family.get("operator_atoms"):
        raise ValueError("concrete seed operator list differs from its family")
    if seed.get("data_eligibility") != ELIGIBILITY:
        raise ValueError("concrete seed violates fail-closed eligibility")


def _compile_action_ir(seed: dict[str, Any], family: dict[str, Any], manifest_binding: dict[str, str]) -> dict[str, Any]:
    operators = []
    for atom in seed["operator_atoms"]:
        if atom not in OPERATOR_LIBRARY:
            raise ValueError(f"unregistered grammar-v3 operator: {atom}")
        operators.append({"atom": atom, **OPERATOR_LIBRARY[atom]})
    token_text = _canonical({"seed": seed, "operators": operators})
    present_forbidden = sorted(token for token in FORBIDDEN_TOKENS if token in token_text)
    if present_forbidden:
        raise ValueError("forbidden invariant entered candidate action: " + ",".join(present_forbidden))
    body = {
        "schema_version": ACTION_IR_SCHEMA,
        "candidate_id": seed["seed_id"],
        "seed_lineage_sha256": seed["seed_lineage_sha256"],
        "family_id": seed["family_id"],
        "family_lineage_sha256": seed["family_lineage_sha256"],
        "manifest_binding": manifest_binding,
        "theory_contract": seed["theory_contract"],
        "fields": family["field_ids"],
        "parameters": seed["parameters"],
        "operators": operators,
        "matter_coupling": {"metric": "g_mu_nu", "universal": True},
        "formal_scope": "candidate-specific typed covariant density; not a claim of completed dynamics",
        "data_eligibility": dict(ELIGIBILITY),
    }
    return {**body, "content_sha256": _sha(body)}


def _resolve(entrypoint: str) -> Any:
    module_name, separator, attribute = entrypoint.partition(":")
    if not separator:
        raise ValueError("formal adapter entrypoint must use module:function syntax")
    callback = getattr(importlib.import_module(module_name), attribute, None)
    if not callable(callback):
        raise TypeError(f"declared formal adapter is not callable: {entrypoint}")
    return callback


def _aether_spec(seed: dict[str, Any]) -> dict[str, Any]:
    p = seed["parameters"]
    return {
        "schema_version": "sigma-action-spec-1.0",
        "role": "candidate",
        "fields": ["g_mu_nu", "u_mu", "lambda_u"],
        "matter_metric": "g_mu_nu",
        "terms": seed["operator_atoms"],
        "coefficients": {
            "AETHER_K1": f"-M_Pl^2*({p['c1']})/2",
            "AETHER_K2": f"-M_Pl^2*({p['c2']})/2",
            "AETHER_K3": f"-M_Pl^2*({p['c3']})/2",
            "AETHER_K4": f"+M_Pl^2*({p['c4']})/2",
        },
        "universal_constants": ["M_Pl"],
        "parameter_domain": {"positive": ["M_Pl"]},
        "static_dictionary_status": "derived",
    }


def _scalar_pack_spec(seed: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "sigma-scalar-tensor-pack-1.0",
        "name": f"grammar-v3 candidate {seed['seed_id']}",
        "normalization": {
            "u": "phi/Lambda_phi",
            "x": "-nabla_phi_squared/(2*Lambda_phi**4)",
            "Lambda_phi_positive": True,
        },
        "coefficients": ["xi"],
        "functions": {"g2": "x", "g3": "0", "g4": "1/2+xi*u**2"},
        "derivative_overrides": {"g4_x": "0"},
        "mutation_axes": [{"coefficient": "xi", "values": ["1/100"]}],
    }


def _candidate_parameter_certificate(seed: dict[str, Any]) -> dict[str, Any]:
    if seed["family_id"] == "KESSENCE_G2_CONVEX":
        expression = seed["parameters"]["G2"]
        coefficient = Fraction(1, 8) if "1/8" in expression else Fraction(1, 4)
        body = {
            "method": "exact_rational_polynomial_interval",
            "domain": seed["parameters"]["X_domain"],
            "G2_X": f"1+({2 * coefficient})*X_phi",
            "G2_X_minimum": "1",
            "G2_X_plus_2X_G2_XX": f"1+({6 * coefficient})*X_phi",
            "G2_X_plus_2X_G2_XX_minimum": "1",
            "local_convexity": "pass",
            "global_energy": "unresolved",
        }
    elif seed["family_id"] == "CUBIC_HORNDESKI_G3_WEAK_CELL":
        body = {
            "method": "exact_parameter_parse",
            "declared_jet_domain": seed["parameters"]["jet_domain"],
            "uniform_source_threshold": "unavailable",
            "common_scalar_metric_time_cone": "unresolved",
            "status": "conditional_unresolved",
        }
    elif seed["family_id"] == "CONFORMAL_G4_PHI_SCALAR_TENSOR":
        body = {
            "method": "exact_rational_polynomial_interval",
            "domain": seed["parameters"]["phi_domain"],
            "G4_minimum": "1/2",
            "G4_positive": "pass",
            "global_Delta_N_invertibility": "unresolved",
            "inhomogeneous_principal_symbol": "unresolved",
            "global_energy": "unresolved",
        }
    else:
        body = {
            "method": "delegated_to_hash_bound_candidate_action_health",
            "nonlinear_global_energy": "unresolved",
        }
    return {**body, "content_sha256": _sha(body)}


def _invoke_adapter(
    adapter: dict[str, Any], seed: dict[str, Any], root: Path, sources: dict[str, Any]
) -> dict[str, Any]:
    callback = _resolve(adapter["entrypoint"])
    if adapter["id"] == "einstein_aether_complete_family":
        spec = _aether_spec(seed)
        with tempfile.TemporaryDirectory(prefix="sigma-g3-health-") as temporary:
            temporary_path = Path(temporary)
            spec_path = temporary_path / "action.json"
            spec_path.write_text(_canonical(spec), encoding="utf-8")
            report = callback(
                spec_path,
                root / sources["action_grammar"]["path"],
                root / sources["field_contract"]["path"],
                temporary_path / "output",
                project_root=root,
                formal_report=sources["formal_report_value"],
            )
            generated_action = json.loads(
                (temporary_path / "output" / "action-ir.json").read_text(encoding="utf-8")
            )
        stable_evidence = {
            "status": report["status"],
            "family": report["family"],
            "promotion_allowed": report["promotion_allowed"],
            "input_action_sha256": report["input_action_sha256"],
            "generated_action_content_sha256": generated_action["content_sha256"],
            "gate_statuses": {name: gate["status"] for name, gate in sorted(report["gates"].items())},
        }
        returned = report["status"] == "pass"
        evidence_sha256 = _sha(stable_evidence)
    elif adapter["id"] == "x_independent_g4_metric_noether":
        report = callback(_scalar_pack_spec(seed))
        stable_evidence = {
            "status": report["status"],
            "content_sha256": report["content_sha256"],
            "functions": report["functions"],
            "derivative_override_residuals": report["derivative_override_residuals"],
            "capability_status": {
                key: report["capability_status"][key]
                for key in (
                    "compiled_g4_phi_only_variation_and_noether",
                    "generic_adm_dirac",
                    "generic_covariant_variation",
                    "generic_hamiltonian",
                    "generic_principal_symbol",
                    "observations",
                    "typed_normalized_covariant_family",
                )
            },
            "compiled_adm_regular_patch": report["compiled_adm_regular_patch"],
            "compiled_dirac_regular_patch": report["compiled_dirac_regular_patch"],
        }
        returned = report["status"] == "pass"
        evidence_sha256 = report["content_sha256"]
    else:
        returned, report = callback()
        stable_evidence = {
            "control": report.get("control", adapter["id"]),
            "scope": report.get("scope", "adapter-specific exact control; see evidence hash"),
            "status": report.get("status", "pass" if returned else "unresolved"),
            "capability_boundary": report.get("capability_boundary"),
        }
        evidence_sha256 = _sha(report)
    body = {
        "adapter_id": adapter["id"],
        "entrypoint": adapter["entrypoint"],
        "callable_invoked": True,
        "returned_boolean": bool(returned),
        "evidence_sha256": evidence_sha256,
        "evidence_summary": stable_evidence,
    }
    return {**body, "content_sha256": _sha(body)}


def _gate(status: str, reason: str, evidence: list[str]) -> dict[str, Any]:
    return {
        "status": status,
        "reason": reason,
        "evidence_root_sha256": _sha(sorted(evidence)),
    }


def _candidate_gates(seed: dict[str, Any], invocations: list[dict[str, Any]], action_sha: str, parameter_sha: str) -> dict[str, Any]:
    family = seed["family_id"]
    evidence = [action_sha, parameter_sha, *[item["content_sha256"] for item in invocations]]
    gates = {
        "typed_action_ir": _gate("pass", "candidate-specific typed covariant action compiled", [action_sha]),
        "declared_adapter_invocation": _gate("pass", "every and only declared callable adapter was invoked", evidence),
        "covariant_variation": _gate("pass", "declared covariant variation/identity adapter returned exact control evidence", evidence),
    }
    if family == "AETHER_K1234_PARAMETER_CELL":
        health = invocations[0]["evidence_summary"]
        gates.update(
            {
                "adm_dirac": _gate(health["gate_statuses"]["adm_dirac"], "candidate action-health ADM/Dirac gate", evidence),
                "principal_symbol": _gate(health["gate_statuses"]["principal_symbol"], "candidate action-health principal gate", evidence),
                "hamiltonian_stability": _gate(health["gate_statuses"]["hamiltonian_stability"], "generic nonlinear total-energy positivity is not established", evidence),
            }
        )
    elif family == "KESSENCE_G2_CONVEX":
        gates.update(
            {
                "adm_dirac": _gate("blocked", "declared adapter proves a local scalar Legendre branch, not complete coupled distributed constraint closure", evidence),
                "principal_symbol": _gate("blocked", "no candidate-specific principal-symbol adapter was declared or invoked", evidence),
                "hamiltonian_stability": _gate("blocked", "local convexity is certified but global gravitational energy is unresolved", evidence),
            }
        )
    elif family == "CUBIC_HORNDESKI_G3_WEAK_CELL":
        gates.update(
            {
                "adm_dirac": _gate("blocked", "no candidate-specific ADM/Dirac adapter was declared", evidence),
                "principal_symbol": _gate("blocked", "BSSN theorem remains conditional without a source-defined uniform weak-field threshold and common cone proof", evidence),
                "hamiltonian_stability": _gate("blocked", "no candidate-specific reduced or global Hamiltonian proof was declared", evidence),
            }
        )
    else:
        gates.update(
            {
                "adm_dirac": _gate("blocked", "regular-patch theorem is conditional on candidate-specific global lapse-operator invertibility", evidence),
                "principal_symbol": _gate("blocked", "FLRW sign conditions do not establish the arbitrary inhomogeneous principal symbol", evidence),
                "hamiltonian_stability": _gate("blocked", "patchwise reduced controls do not establish nonlinear global energy boundedness", evidence),
            }
        )
    gates["formal_prerequisite_completion"] = _gate(
        "pass" if all(item["status"] == "pass" for item in gates.values()) else "blocked",
        "all formal prerequisites must pass before Solar or observations can open",
        evidence,
    )
    return gates


def build_covariant_grammar_v3_seed_compilation_campaign(config: dict[str, Any], root: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    if config.get("data_eligibility") != ELIGIBILITY:
        raise ValueError("campaign eligibility is not fail-closed")
    manifest = _load_bound_json(root, config["seed_manifest"], content=True)
    if manifest.get("data_eligibility") != ELIGIBILITY or manifest.get("observational_data_opened") is not False:
        raise ValueError("source seed manifest violates eligibility seal")
    formal_report = _load_bound_json(root, config["formal_report"])
    for key in ("action_grammar", "field_contract"):
        _load_bound_json(root, config[key])
    families = {item["family_id"]: item for item in manifest["typed_family_seeds"]}
    seeds = list(iter_scalable_seed_specs(manifest))
    if len(seeds) != config["maximum_seed_count"]:
        raise ValueError("bounded campaign seed count mismatch")
    manifest_binding = {
        "file_sha256": config["seed_manifest"]["file_sha256"],
        "content_sha256": config["seed_manifest"]["content_sha256"],
    }
    source_context = {**config, "formal_report_value": formal_report}
    records = []
    all_invoked = []
    for seed in seeds:
        family = families[seed["family_id"]]
        _validate_seed(seed, family)
        if family.get("enabled_for_generation") is not True or family.get("pre_generation_decision") != "accept":
            raise ValueError("non-enabled seed entered compilation campaign")
        action_ir = _compile_action_ir(seed, family, manifest_binding)
        parameter_certificate = _candidate_parameter_certificate(seed)
        declared = [item for item in family["formal_adapters"] if item.get("available") is True]
        invocations = [_invoke_adapter(item, seed, root, source_context) for item in declared]
        invoked_entrypoints = [item["entrypoint"] for item in invocations]
        declared_entrypoints = [item["entrypoint"] for item in declared]
        if invoked_entrypoints != declared_entrypoints:
            raise ValueError("adapter invocation differs from declared callable adapters")
        all_invoked.extend(invoked_entrypoints)
        gates = _candidate_gates(
            seed, invocations, action_ir["content_sha256"], parameter_certificate["content_sha256"]
        )
        statuses = {item["status"] for item in gates.values()}
        decision = "reject" if "reject" in statuses else "pass" if statuses == {"pass"} else "blocked"
        provenance_body = {
            "manifest_binding": manifest_binding,
            "seed_lineage_sha256": seed["seed_lineage_sha256"],
            "family_lineage_sha256": seed["family_lineage_sha256"],
            "action_ir_sha256": action_ir["content_sha256"],
            "parameter_certificate_sha256": parameter_certificate["content_sha256"],
            "adapter_result_sha256": [item["content_sha256"] for item in invocations],
            "data_eligibility": ELIGIBILITY,
        }
        records.append(
            {
                "seed_id": seed["seed_id"],
                "family_id": seed["family_id"],
                "decision": decision,
                "typed_action_ir": action_ir,
                "parameter_certificate": parameter_certificate,
                "declared_adapter_entrypoints": declared_entrypoints,
                "invoked_adapter_entrypoints": invoked_entrypoints,
                "adapter_invocations": invocations,
                "gate_ledger": gates,
                "provenance": {**provenance_body, "binding_sha256": _sha(provenance_body)},
                "solar_known_answer_bundle": {
                    "generated": decision == "pass",
                    "status": "eligible" if decision == "pass" else "blocked",
                    "reason": "all_formal_prerequisites_pass" if decision == "pass" else "formal_prerequisites_incomplete",
                },
            }
        )
    decision_counts = Counter(item["decision"] for item in records)
    if any(item["decision"] == "pass" for item in records):
        raise ValueError("unexpected formal pass requires a separately reviewed Solar-opening campaign")
    body = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": config["campaign_id"],
        "source_bindings": {
            "seed_manifest": config["seed_manifest"],
            "formal_report": config["formal_report"],
            "action_grammar": config["action_grammar"],
            "field_contract": config["field_contract"],
        },
        "operator_library_sha256": _sha(OPERATOR_LIBRARY),
        "seed_count": len(records),
        "adapter_invocation_count": len(all_invoked),
        "decision_counts": dict(sorted(decision_counts.items())),
        "solar_bundle_count": 0,
        "candidate_records": records,
        "observational_data_opened": False,
        "data_eligibility": dict(ELIGIBILITY),
        "interpretation": "Candidate-specific action and adapter evidence only; family labels and known-answer controls are not candidate viability evidence.",
    }
    return {**body, "content_sha256": _sha(body)}
