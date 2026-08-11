"""Candidate-specific necessary formal gates for 128 Aether parameter cells."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any

from .covariant_grammar_v3_seed_compilation_campaign import _compile_action_ir
from .covariant_identities import einstein_aether_flrw_variation_control
from .grammar_v3_parameter_cell_compilation_campaign import _action_density_key
from .grammar_v3_parameter_cell_manifest_campaign import iter_parameter_cells
from .promotion_orchestrator import ELIGIBILITY

SCHEMA_VERSION = "sigma-aether-parameter-cell-formal-gate-campaign-1.0"
STATUS_SCHEMA_VERSION = "sigma-aether-parameter-cell-formal-gate-status-1.0"
TARGET_FAMILY = "AETHER_K1234_PARAMETER_CELL"
PREFLIGHT_PAYLOAD_SCHEMA = "sigma-grammar-v3-formal-preflight-work-1.0"
PREFLIGHT_RESULT_SCHEMA = "sigma-grammar-v3-formal-preflight-result-1.0"
REQUIRED_FORMAL_ADAPTERS = {
    "einstein_aether_global_tilt_legendre_strata",
    "einstein_aether_generic_dh_covariance",
    "einstein_aether_generic_hh_deformation_kinematics",
    "einstein_aether_linearized_physical_energy",
    "einstein_aether_restricted_nonlinear_total_energy",
    "maxwell_unit_aether_nonlinear_hamiltonian",
}
EXPECTED_OPERATORS = [
    "EH_R",
    "AETHER_K1",
    "AETHER_K2",
    "AETHER_K3",
    "AETHER_K4",
    "UNIT_VECTOR_CONSTRAINT",
]


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_bound(root: Path, binding: dict[str, Any]) -> dict[str, Any]:
    path = root / binding["path"]
    if not path.is_file() or _file_sha(path) != binding["file_sha256"]:
        raise ValueError(f"bound Aether campaign artifact changed: {binding['path']}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{binding['path']} must contain an object")
    expected = binding.get("content_sha256")
    if expected is not None:
        body = {key: item for key, item in value.items() if key != "content_sha256"}
        if value.get("content_sha256") != expected or _sha(body) != expected:
            raise ValueError(f"bound Aether campaign content changed: {binding['path']}")
    return value


def _fraction_map(parameters: dict[str, str]) -> dict[str, Fraction]:
    if set(parameters) != {"c1", "c2", "c3", "c4"}:
        raise ValueError("Aether parameter-cell coordinates changed")
    return {name: Fraction(value) for name, value in parameters.items()}


def _formal_adapter_evidence(
    config: dict[str, Any], report: dict[str, Any], root: Path
) -> dict[str, dict[str, Any]]:
    descriptors = config.get("formal_adapters", [])
    if {item.get("id") for item in descriptors} != REQUIRED_FORMAL_ADAPTERS:
        raise ValueError("required reviewed Aether formal adapter is missing")
    report_by_name = {item["name"]: item for item in report.get("checks", [])}
    evidence = {}
    for descriptor in descriptors:
        if set(descriptor) != {
            "id",
            "source_path",
            "source_file_sha256",
            "evidence_sha256",
        }:
            raise ValueError("reviewed Aether adapter descriptor fields changed")
        path = root / descriptor["source_path"]
        if not path.is_file() or _file_sha(path) != descriptor["source_file_sha256"]:
            raise ValueError(f"reviewed Aether adapter source changed: {descriptor['id']}")
        check = report_by_name.get(descriptor["id"])
        if (
            not isinstance(check, dict)
            or check.get("status") != "pass"
            or _sha(check.get("evidence")) != descriptor["evidence_sha256"]
            or check.get("evidence", {}).get("passed") is not True
        ):
            raise ValueError(f"reviewed Aether adapter evidence changed: {descriptor['id']}")
        evidence[descriptor["id"]] = check["evidence"]
    nonlinear = evidence["einstein_aether_restricted_nonlinear_total_energy"]
    tilt = evidence["einstein_aether_global_tilt_legendre_strata"]
    if (
        nonlinear.get("generic_status") != "unresolved"
        or nonlinear.get("energy_not_local_density") is not True
        or nonlinear.get("out_of_domain_controls", {})
        .get("aether_with_twist", {})
        .get("theorem_premise_rejected")
        is not True
        or tilt.get("passed") is not True
    ):
        raise ValueError("reviewed Aether nonlinear/tilt scope changed")
    return evidence


def _preflight_records(
    config: dict[str, Any],
    compilation: dict[str, Any],
    cell_manifest: dict[str, Any],
    seed_manifest: dict[str, Any],
    preflight_status: dict[str, Any],
) -> list[dict[str, Any]]:
    preflight_config = config["formal_preflight_config_document"]
    callback_descriptors = preflight_config["reviewed_adapters"]
    callback_root = _sha(
        [
            {
                "family_id": item["family_id"],
                "adapter_id": item["adapter_id"],
                "callback": item["callback"],
                "source_file_sha256": item["source_file_sha256"],
                "state": "reviewed_bound",
            }
            for item in callback_descriptors
        ]
    )
    if callback_root != preflight_status.get("callback_registry_root_sha256"):
        raise ValueError("formal-preflight callback registry changed")
    aether_descriptor = next(
        (item for item in callback_descriptors if item["family_id"] == TARGET_FAMILY),
        None,
    )
    if aether_descriptor is None:
        raise ValueError("Aether formal-preflight adapter is missing")
    evidence = einstein_aether_flrw_variation_control()
    stable = {
        "passed": evidence.get("passed"),
        "noether_residual": evidence.get("noether_residual"),
        "declared_point_rank": evidence.get("declared_point_rank"),
        "scope": evidence.get("scope"),
    }
    adapter_body = {
        "family_id": TARGET_FAMILY,
        "decision": "pass",
        "evidence_summary": stable,
        "scope": "cheap family prerequisite only; not full ADM or global energy",
    }
    if stable["passed"] is not True or stable["noether_residual"] != "0":
        raise ValueError("Aether formal-preflight replay failed")
    adapter = {**adapter_body, "content_sha256": _sha(adapter_body)}
    families = {
        item["family_id"]: item
        for item in seed_manifest["typed_family_seeds"]
        if item["enabled_for_generation"]
    }
    manifest_binding = {
        "parameter_cell_manifest_content_sha256": cell_manifest["content_sha256"],
        "parameter_cell_registry_root_sha256": cell_manifest["parameter_cell_registry_root_sha256"],
    }
    seen: set[str] = set()
    ordinal = 0
    records = []
    simplified = []
    for cell in iter_parameter_cells(cell_manifest, seed_manifest):
        equivalence_sha = _sha(_action_density_key(cell))
        if equivalence_sha in seen:
            continue
        seen.add(equivalence_sha)
        pseudo_seed = {
            "seed_id": cell["parameter_cell_id"],
            "seed_lineage_sha256": cell["parameter_cell_lineage_sha256"],
            "family_id": cell["family_id"],
            "family_lineage_sha256": cell["family_lineage_sha256"],
            "theory_contract": cell["theory_contract"],
            "operator_atoms": cell["operator_atoms"],
            "parameters": cell["parameters"],
        }
        action_ir = _compile_action_ir(pseudo_seed, families[cell["family_id"]], manifest_binding)
        payload_body = {
            "schema_version": PREFLIGHT_PAYLOAD_SCHEMA,
            "ordinal": ordinal,
            "candidate_id": "G3A-" + equivalence_sha[:24],
            "typed_action_ir_sha256": action_ir["content_sha256"],
            "action_density_equivalence_sha256": equivalence_sha,
            "representative_cell_id": cell["parameter_cell_id"],
            "representative_cell_lineage_sha256": cell["parameter_cell_lineage_sha256"],
            "family_id": cell["family_id"],
            "compilation_campaign_content_sha256": compilation["content_sha256"],
            "callback_registry_root_sha256": callback_root,
            "data_eligibility": dict(ELIGIBILITY),
        }
        payload = {**payload_body, "input_lineage_sha256": _sha(payload_body)}
        ordinal += 1
        if cell["family_id"] != TARGET_FAMILY:
            continue
        result_body = {
            "schema_version": PREFLIGHT_RESULT_SCHEMA,
            "candidate_id": payload["candidate_id"],
            "typed_action_ir_sha256": payload["typed_action_ir_sha256"],
            "input_lineage_sha256": payload["input_lineage_sha256"],
            "callback_registry_root_sha256": callback_root,
            "receipt_binding_gate": "pass",
            "family_prerequisite_gate": "pass",
            "adapter_evidence": adapter,
            "decision": "pass",
            "blocker": None,
            "expensive_adm_or_global_energy_run": False,
            "data_eligibility": {**ELIGIBILITY, "passed": True},
            "paid_llm_spend_usd": 0.0,
        }
        result_sha = _sha(result_body)
        simplified_record = {
            "candidate_id": payload["candidate_id"],
            "typed_action_ir_sha256": payload["typed_action_ir_sha256"],
            "family_id": TARGET_FAMILY,
            "state": "succeeded",
            "attempt": 1,
            "result_sha256": result_sha,
            "blocker": None,
            "error_text": None,
        }
        simplified.append(simplified_record)
        records.append(
            {
                "cell": cell,
                "action_ir": action_ir,
                "payload": payload,
                "preflight_result_sha256": result_sha,
                "preflight_record_sha256": _sha(simplified_record),
            }
        )
    if len(records) != 128:
        raise ValueError("Aether formal-preflight survivor count changed")
    for chunk_index in range(4):
        selected = simplified[chunk_index * 32 : (chunk_index + 1) * 32]
        if _sha(selected) != preflight_status["chunks"][chunk_index]["record_root_sha256"]:
            raise ValueError("Aether formal-preflight record chunk changed")
    return records


def _finite_negative_twist_witness(
    c1: Fraction, c3: Fraction, c4: Fraction
) -> dict[str, Any] | None:
    def coefficient(y: int) -> Fraction:
        return c1 - c3 - Fraction(y, 2) * (c1 / (1 + y) + c4)

    negative_is_guaranteed = c4 > 0 or (c4 == 0 and c1 / 2 - c3 < 0)
    if not negative_is_guaranteed:
        return None
    y = 1
    while coefficient(y) >= 0 and y <= 4096:
        y *= 2
    if y > 4096 or coefficient(y) >= 0:
        raise ValueError("failed to construct exact finite negative twist witness")
    body = {
        "tilt_squared_y": str(y),
        "orientation": "WA2=(y/2)*W2",
        "normalized_W2": "2",
        "normalized_WA2": str(y),
        "C_y": str(coefficient(y)),
        "unit_constraint_branch": "u_mu=(-sqrt(1+y),A_i)",
        "local_hamiltonian_density_negative": True,
        "full_gravitational_constraint_embedding_proven": False,
        "candidate_rejection_authorized_by_this_witness_alone": False,
    }
    return {**body, "content_sha256": _sha(body)}


def _specialize(parameters: dict[str, str]) -> dict[str, Any]:
    p = _fraction_map(parameters)
    c1, c2, c3, c4 = (p[name] for name in ("c1", "c2", "c3", "c4"))
    c13 = c1 + c3
    c14 = c1 + c4
    c123 = c1 + c2 + c3
    trace = 2 + c13 + 3 * c2
    vector_numerator = 2 * c1 - c1**2 + c3**2
    adm = {
        "one_minus_c13": 1 - c13,
        "c14": c14,
        "trace_factor": trace,
    }
    speeds = {
        "spin_2": 1 / (1 - c13),
        "spin_1": vector_numerator / (2 * c14 * (1 - c13)),
        "spin_0": c123 * (2 - c14) / (c14 * (1 - c13) * trace),
    }
    energies = {
        "spin_2": Fraction(1),
        "spin_1": vector_numerator / (1 - c13),
        "spin_0": c14 * (2 - c14),
    }
    principal_pass = all(value > 0 for value in speeds.values()) and c123 > 0
    characteristic_tilt_squared = {
        sector: str(1 / (speed - 1)) for sector, speed in speeds.items() if speed > 1
    }
    witness = _finite_negative_twist_witness(c1, c3, c4) if principal_pass else None
    if c4 > 0:
        twist_limit = "negative_infinity"
        uniform_status = "finite_negative_local_density_witness"
    else:
        limit = c1 / 2 - c3
        twist_limit = str(limit)
        if limit < 0:
            uniform_status = "finite_negative_local_density_witness"
        elif limit == 0:
            uniform_status = "positive_at_every_finite_tilt_but_no_uniform_gap"
        else:
            uniform_status = "uniform_positive_static_local_twist_gap"
    body = {
        "method": "exact_rational_candidate_specialization",
        "parameters": parameters,
        "combinations": {
            "c13": str(c13),
            "c14": str(c14),
            "c123": str(c123),
            "trace_factor": str(trace),
            "vector_numerator": str(vector_numerator),
        },
        "adm_aligned_regularity_factors": {key: str(value) for key, value in adm.items()},
        "adm_aligned_regular": all(value > 0 for value in adm.values()),
        "principal_speed_squared": {key: str(value) for key, value in speeds.items()},
        "linear_energy_coefficients": {key: str(value) for key, value in energies.items()},
        "principal_and_linear_mode_domain_pass": principal_pass,
        "global_unit_tilt_legendre_strata": {
            "finite_characteristic_tilt_squared": characteristic_tilt_squared,
            "globally_noncharacteristic_for_finite_unit_tilt": (
                principal_pass and not characteristic_tilt_squared
            ),
            "interpretation": (
                "a finite threshold is a characteristic slicing of that physical mode, "
                "not by itself a coupling-space strong-coupling rejection"
            ),
        },
        "static_twist_exact_coefficient": "C(y)=c1-c3-(y/2)*(c1/(1+y)+c4)",
        "static_twist_derivative": "C'(y)=-(1/2)*(c1/(1+y)^2+c4)<0",
        "static_twist_large_tilt_limit": twist_limit,
        "static_twist_domain_status": uniform_status,
        "finite_negative_twist_witness": witness,
        "restricted_positive_energy_coupling_domain": (0 <= c14 <= 2 and c13 <= 1),
    }
    return {**body, "content_sha256": _sha(body)}


def build_aether_parameter_cell_formal_gate_campaign(
    config: dict[str, Any], root: str | Path
) -> dict[str, Any]:
    root = Path(root).resolve()
    if config.get("data_eligibility") != ELIGIBILITY:
        raise ValueError("Aether parameter-cell campaign eligibility changed")
    if config.get("observational_authorization") is not False:
        raise ValueError("Aether parameter-cell campaign opened observations")
    bindings = config["source_bindings"]
    source_binding = bindings["campaign_source"]
    source_path = root / source_binding["path"]
    if not source_path.is_file() or _file_sha(source_path) != source_binding["file_sha256"]:
        raise ValueError("Aether parameter-cell campaign source changed")
    sources = {
        key: _load_bound(root, binding)
        for key, binding in bindings.items()
        if key not in {"campaign_source", "formal_preflight_config"}
    }
    preflight_config_path = root / bindings["formal_preflight_config"]["path"]
    if (
        not preflight_config_path.is_file()
        or _file_sha(preflight_config_path) != bindings["formal_preflight_config"]["file_sha256"]
    ):
        raise ValueError("formal-preflight config changed")
    preflight_config = json.loads(preflight_config_path.read_text(encoding="utf-8"))
    working_config = {**config, "formal_preflight_config_document": preflight_config}
    adapter_evidence = _formal_adapter_evidence(config, sources["formal_report"], root)
    preflight = _preflight_records(
        working_config,
        sources["compilation_campaign"],
        sources["parameter_cell_manifest"],
        sources["source_seed_manifest"],
        sources["formal_preflight_status"],
    )
    prior_twist = sources["prior_twist_audit"]
    prior_formulas = {
        record["twist_sector_certificate"]["nonlinear_static_pure_twist_sector"][
            "exact_hamiltonian"
        ]
        for record in prior_twist["candidate_records"]
    }
    if prior_formulas != {"H_twist=(M_Pl^2/2)*[(c1-c3)*W2-(c1/(1+y)+c4)*WA2]"}:
        raise ValueError("reviewed Aether static twist Hamiltonian changed")
    evidence_hashes = {item["id"]: item["evidence_sha256"] for item in config["formal_adapters"]}
    records = []
    decisions: Counter[str] = Counter()
    twist_statuses: Counter[str] = Counter()
    tilt_statuses: Counter[str] = Counter()
    for item in preflight:
        cell = item["cell"]
        action_ir = item["action_ir"]
        if (
            action_ir.get("content_sha256") != item["payload"]["typed_action_ir_sha256"]
            or [operator["atom"] for operator in action_ir.get("operators", [])]
            != EXPECTED_OPERATORS
            or action_ir.get("parameters") != cell["parameters"]
            or action_ir.get("data_eligibility") != ELIGIBILITY
        ):
            raise ValueError("Aether candidate action/preflight binding changed")
        specialization = _specialize(cell["parameters"])
        principal_pass = specialization["principal_and_linear_mode_domain_pass"]
        twist_status = specialization["static_twist_domain_status"]
        tilt_strata = specialization["global_unit_tilt_legendre_strata"]
        if not principal_pass:
            decision = "reject"
            blocker = "nonpositive_spin0_principal_numerator_c123"
            local_twist_gate = {"status": "not_evaluated_after_decisive_principal_rejection"}
        else:
            decision = "blocked"
            if twist_status == "finite_negative_local_density_witness":
                blocker = "full_constraint_embedding_of_negative_static_twist_jet"
                local_twist_gate = {
                    "status": "blocked",
                    "finding": twist_status,
                    "reason": (
                        "an exact negative unit-reduced local density is present, but the "
                        "reviewed formula excludes gravitational constraint solving and boundary energy"
                    ),
                }
            elif twist_status == "positive_at_every_finite_tilt_but_no_uniform_gap":
                blocker = "uniform_static_twist_coercivity"
                local_twist_gate = {"status": "blocked", "finding": twist_status}
            else:
                blocker = "generic_twisting_reduced_hamiltonian_and_global_energy"
                local_twist_gate = {"status": "pass", "finding": twist_status}
            twist_statuses[twist_status] += 1
            tilt_statuses[
                "globally_noncharacteristic_for_finite_unit_tilt"
                if tilt_strata["globally_noncharacteristic_for_finite_unit_tilt"]
                else "finite_characteristic_slicing_present"
            ] += 1
        gates = {
            "exact_action_and_preflight_record": {"status": "pass"},
            "candidate_specific_adm_legendre_regularity": {
                "status": "pass" if specialization["adm_aligned_regular"] else "reject"
            },
            "generic_regular_patch_dirac_closure": {"status": "pass"},
            "global_unit_tilt_legendre_strata": {
                "status": (
                    "pass"
                    if tilt_strata["globally_noncharacteristic_for_finite_unit_tilt"]
                    else "conditional"
                    if principal_pass
                    else "not_evaluated_after_principal_rejection"
                ),
                "finite_characteristic_tilt_squared": tilt_strata[
                    "finite_characteristic_tilt_squared"
                ],
            },
            "aligned_minkowski_principal_and_linear_modes": {
                "status": "pass" if principal_pass else "reject"
            },
            "candidate_specific_linear_energy": {
                "status": "pass" if principal_pass else "not_evaluated_after_principal_rejection"
            },
            "static_unit_reduced_pure_twist_local_energy": local_twist_gate,
            "restricted_positive_energy_coupling_domain": {"status": "pass"},
            "generic_twisting_constraint_reduced_hamiltonian": {"status": "blocked"},
            "global_positive_energy": {"status": "blocked"},
        }
        provenance_body = {
            "candidate_id": item["payload"]["candidate_id"],
            "action_sha256": action_ir["content_sha256"],
            "action_density_equivalence_sha256": item["payload"][
                "action_density_equivalence_sha256"
            ],
            "representative_cell_lineage_sha256": cell["parameter_cell_lineage_sha256"],
            "preflight_input_lineage_sha256": item["payload"]["input_lineage_sha256"],
            "preflight_result_sha256": item["preflight_result_sha256"],
            "preflight_record_sha256": item["preflight_record_sha256"],
            "specialization_sha256": specialization["content_sha256"],
            "formal_adapter_evidence_sha256": evidence_hashes,
            "data_eligibility": ELIGIBILITY,
        }
        record_body = {
            "candidate_id": item["payload"]["candidate_id"],
            "family_id": TARGET_FAMILY,
            "representative_cell_id": cell["parameter_cell_id"],
            "representative_cell_lineage_sha256": cell["parameter_cell_lineage_sha256"],
            "action_sha256": action_ir["content_sha256"],
            "action_density_equivalence_sha256": item["payload"][
                "action_density_equivalence_sha256"
            ],
            "preflight_input_lineage_sha256": item["payload"]["input_lineage_sha256"],
            "preflight_result_sha256": item["preflight_result_sha256"],
            "preflight_record_sha256": item["preflight_record_sha256"],
            "parameters": cell["parameters"],
            "exact_specialization": specialization,
            "gate_ledger": gates,
            "decision": decision,
            "blocker": blocker,
            "formal_pass": False,
            "solar_bundle_generated": False,
            "provenance": {
                **provenance_body,
                "binding_sha256": _sha(provenance_body),
            },
        }
        records.append({**record_body, "content_sha256": _sha(record_body)})
        decisions[decision] += 1
    record_root = _sha(
        [
            [record["candidate_id"], record["action_sha256"], record["content_sha256"]]
            for record in records
        ]
    )
    formal_evidence_body = {name: evidence_hashes[name] for name in sorted(adapter_evidence)}
    provenance_body = {
        "compilation_campaign_sha256": bindings["compilation_campaign"]["content_sha256"],
        "formal_preflight_status_sha256": bindings["formal_preflight_status"]["content_sha256"],
        "formal_preflight_aether_record_root_sha256": _sha(
            [
                [
                    item["payload"]["candidate_id"],
                    item["action_ir"]["content_sha256"],
                    item["preflight_result_sha256"],
                ]
                for item in preflight
            ]
        ),
        "candidate_gate_record_root_sha256": record_root,
        "formal_adapter_evidence_root_sha256": _sha(formal_evidence_body),
        "prior_twist_audit_sha256": bindings["prior_twist_audit"]["content_sha256"],
        "campaign_source_sha256": source_binding["file_sha256"],
        "data_eligibility": ELIGIBILITY,
    }
    body = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": config["campaign_id"],
        "source_bindings": bindings,
        "target_family": TARGET_FAMILY,
        "input_preflight_pass_count": 128,
        "candidate_count": len(records),
        "decision_counts": dict(sorted(decisions.items())),
        "gate_finding_counts": {
            "principal_spin0_degeneracy_reject": decisions["reject"],
            **dict(sorted(twist_statuses.items())),
            **dict(sorted(tilt_statuses.items())),
        },
        "formal_pass_count": 0,
        "solar_bundle_count": 0,
        "candidate_gate_record_root_sha256": record_root,
        "candidate_records": records,
        "reviewed_formal_adapter_evidence": formal_evidence_body,
        "static_twist_asymptotic_correction": {
            "exact_reused_expression": ("C(y)=c1-c3-(y/2)*(c1/(1+y)+c4)"),
            "exact_limit": ("negative_infinity_for_c4>0; c1/2-c3_for_c4=0"),
            "prior_reported_finite_infimum_not_reused": True,
            "reason": "the c4*y/2 term is unbounded for y>=0 when c4>0",
            "candidate_rejection_from_local_density_alone": False,
        },
        "observational_data_opened": False,
        "dark_matter_or_halo_inputs": False,
        "redshift_distance_inputs": False,
        "paid_llm_spend_usd": 0.0,
        "data_eligibility": dict(ELIGIBILITY),
        "provenance": {**provenance_body, "binding_sha256": _sha(provenance_body)},
        "interpretation": (
            "Two cells fail an exact spin-0 principal/linear-mode necessary condition. "
            "The other 126 remain blocked: 79 have an exact negative static unit-reduced "
            "local twist-density witness that is not yet embedded in the full gravitational "
            "constraint surface, eight lose a uniform large-tilt twist gap, and 39 retain that "
            "local gap but still lack a generic twisting reduced-Hamiltonian/global-energy theorem."
        ),
    }
    return {**body, "content_sha256": _sha(body)}


def build_aether_parameter_cell_formal_gate_status(
    campaign: dict[str, Any],
) -> dict[str, Any]:
    """Project the full replay into a portable, per-candidate hash-bound ledger."""
    body_without_hash = {key: value for key, value in campaign.items() if key != "content_sha256"}
    if campaign.get("content_sha256") != _sha(body_without_hash):
        raise ValueError("Aether parameter-cell campaign content binding changed")
    candidate_bindings = [
        {
            "candidate_id": record["candidate_id"],
            "action_sha256": record["action_sha256"],
            "action_density_equivalence_sha256": record["action_density_equivalence_sha256"],
            "preflight_input_lineage_sha256": record["preflight_input_lineage_sha256"],
            "preflight_result_sha256": record["preflight_result_sha256"],
            "preflight_record_sha256": record["preflight_record_sha256"],
            "specialization_sha256": record["exact_specialization"]["content_sha256"],
            "decision": record["decision"],
            "blocker": record["blocker"],
            "candidate_gate_record_sha256": record["content_sha256"],
        }
        for record in campaign["candidate_records"]
    ]
    binding_root = _sha(candidate_bindings)
    if binding_root != _sha(
        [
            {
                "candidate_id": record["candidate_id"],
                "action_sha256": record["action_sha256"],
                "action_density_equivalence_sha256": record["action_density_equivalence_sha256"],
                "preflight_input_lineage_sha256": record["preflight_input_lineage_sha256"],
                "preflight_result_sha256": record["preflight_result_sha256"],
                "preflight_record_sha256": record["preflight_record_sha256"],
                "specialization_sha256": record["exact_specialization"]["content_sha256"],
                "decision": record["decision"],
                "blocker": record["blocker"],
                "candidate_gate_record_sha256": record["content_sha256"],
            }
            for record in campaign["candidate_records"]
        ]
    ):
        raise AssertionError("candidate-binding projection is nondeterministic")
    body = {
        "schema_version": STATUS_SCHEMA_VERSION,
        "campaign_id": campaign["campaign_id"],
        "campaign_content_sha256": campaign["content_sha256"],
        "source_bindings": campaign["source_bindings"],
        "input_preflight_pass_count": campaign["input_preflight_pass_count"],
        "candidate_count": campaign["candidate_count"],
        "decision_counts": campaign["decision_counts"],
        "gate_finding_counts": campaign["gate_finding_counts"],
        "formal_pass_count": campaign["formal_pass_count"],
        "solar_bundle_count": campaign["solar_bundle_count"],
        "candidate_gate_record_root_sha256": campaign["candidate_gate_record_root_sha256"],
        "candidate_binding_root_sha256": binding_root,
        "candidate_bindings": candidate_bindings,
        "reviewed_formal_adapter_evidence": campaign["reviewed_formal_adapter_evidence"],
        "static_twist_asymptotic_correction": campaign["static_twist_asymptotic_correction"],
        "observational_data_opened": campaign["observational_data_opened"],
        "dark_matter_or_halo_inputs": campaign["dark_matter_or_halo_inputs"],
        "redshift_distance_inputs": campaign["redshift_distance_inputs"],
        "paid_llm_spend_usd": campaign["paid_llm_spend_usd"],
        "data_eligibility": campaign["data_eligibility"],
        "provenance": campaign["provenance"],
        "interpretation": campaign["interpretation"],
    }
    return {**body, "content_sha256": _sha(body)}
