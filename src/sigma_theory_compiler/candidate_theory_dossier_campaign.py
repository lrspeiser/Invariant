from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .promotion_orchestrator import ELIGIBILITY

SCHEMA_VERSION = "sigma-candidate-theory-dossier-campaign-1.0"
NODE_STATUSES = {"proven", "blocked", "calibration_only"}
SEED_IDS = {
    "G3-0b8cb2d5591bf50d2465978d",
    "G3-1ee308440d778dfbee8094d2",
    "G3-94086fa702500475b35ab002",
    "G3-a82c572555e5d79686bc4a4a",
    "G3-e8b35002cfc9c60691a2f67b",
    "G3-f9c598b70a77ea54009d8f18",
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


def _load_bound(root: Path, descriptor: dict[str, Any]) -> dict[str, Any]:
    path = root / descriptor["path"]
    if _file_sha(path) != descriptor["file_sha256"]:
        raise ValueError(f"bound file hash mismatch: {descriptor['path']}")
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = descriptor.get("content_sha256")
    if expected is not None:
        body = {key: item for key, item in value.items() if key != "content_sha256"}
        if value.get("content_sha256") != expected or _sha(body) != expected:
            raise ValueError(f"bound content hash mismatch: {descriptor['path']}")
    return value


def _record(source: dict[str, Any], seed_id: str) -> dict[str, Any]:
    records = [item for item in source.get("candidate_records", []) if item.get("seed_id") == seed_id]
    if len(records) != 1:
        raise ValueError(f"candidate record is not unique: {seed_id}")
    return records[0]


def _gate(record: dict[str, Any], gate_id: str, expected: str) -> dict[str, Any]:
    gate = record.get("gate_ledger", {}).get(gate_id)
    if not isinstance(gate, dict) or gate.get("status") != expected:
        raise ValueError(f"gate mismatch: {gate_id} expected {expected}")
    return gate


def _evidence(
    bindings: dict[str, Any], key: str, locator: str, evidence_sha256: str | None = None
) -> dict[str, Any]:
    descriptor = bindings[key]
    value = {
        "artifact_key": key,
        "artifact_path": descriptor["path"],
        "artifact_file_sha256": descriptor["file_sha256"],
        "artifact_content_sha256": descriptor.get("content_sha256"),
        "json_locator": locator,
    }
    if evidence_sha256 is not None:
        value["evidence_sha256"] = evidence_sha256
    return value


def _node(
    node_id: str,
    status: str,
    scope: str,
    evidence: list[dict[str, Any]],
    **extra: Any,
) -> dict[str, Any]:
    if status not in NODE_STATUSES or not evidence:
        raise ValueError(f"invalid or unbound dossier node: {node_id}")
    body = {
        "node_id": node_id,
        "status": status,
        "scope": scope,
        "evidence": evidence,
        **extra,
    }
    return {**body, "content_sha256": _sha(body)}


def _validate_inputs(
    sources: dict[str, dict[str, Any]], bindings: dict[str, Any]
) -> dict[str, Any]:
    manifest = sources["manifest"]
    compilation = sources["compilation"]
    records = compilation.get("candidate_records", [])
    if len(records) != 6 or {item.get("seed_id") for item in records} != SEED_IDS:
        raise ValueError("grammar-v3 seed set changed")
    if compilation.get("seed_count") != 6 or compilation.get("decision_counts") != {"blocked": 6}:
        raise ValueError("grammar-v3 compilation decision changed")
    for item in records:
        action = item["typed_action_ir"]
        if (
            item.get("decision") != "blocked"
            or action.get("candidate_id") != item["seed_id"]
            or action.get("content_sha256") != item["provenance"]["action_ir_sha256"]
            or action.get("data_eligibility") != ELIGIBILITY
        ):
            raise ValueError("compiled candidate action/provenance mismatch")
    controls = {
        item["control_id"]: item for item in manifest.get("known_answer_controls", [])
    }
    gr = controls.get("GR-EINSTEIN-HILBERT")
    if (
        gr is None
        or gr.get("role") != "hash_bound_known_answer_control"
        or gr.get("classification") != "certified_viable_known_answer"
        or gr.get("eligible_as_generated_candidate") is not False
        or gr.get("input_action_sha256")
        != "8965f95177ca7e7d798d6163d184d62c5fa3aba0a7d11f32b407f71976d08d73"
    ):
        raise ValueError("GR known-answer control changed")
    action_spec = sources["gr_action_spec"]
    health = sources["gr_health"]
    reference = sources["gr_reference"]
    if (
        action_spec.get("role") != "known_answer_control"
        or action_spec.get("terms") != ["EH_R"]
        or health.get("input_action_sha256") != gr["input_action_sha256"]
        or reference.get("counts") != {
            "blocked": 0,
            "failed": 0,
            "golden_total": 5,
            "passed": 5,
        }
        or reference.get("formal_prerequisite", {}).get(
            "observational_dataset_opened"
        )
        is not False
    ):
        raise ValueError("GR action, health, or reference mismatch")
    for gate_id in [
        "covariant_variation",
        "adm_dirac",
        "principal_symbol",
        "hamiltonian_stability",
    ]:
        if health.get("gates", {}).get(gate_id, {}).get("status") != "pass":
            raise ValueError(f"GR health gate changed: {gate_id}")
    for key, source in sources.items():
        if source.get("observational_data_opened") not in {None, False}:
            raise ValueError(f"source opened observations: {key}")
        eligibility = source.get("data_eligibility")
        if eligibility is not None and eligibility != ELIGIBILITY:
            raise ValueError(f"source eligibility changed: {key}")
    if set(bindings) != set(sources):
        raise ValueError("source binding set changed")
    return gr


def _action_nodes(
    record: dict[str, Any], bindings: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    action = record["typed_action_ir"]
    action_evidence = _evidence(
        bindings,
        "compilation",
        f"candidate_records[seed_id={record['seed_id']}].typed_action_ir",
        action["content_sha256"],
    )
    definition = _node(
        "defining_covariant_action",
        "proven",
        action["formal_scope"],
        [action_evidence],
        action_sha256=action["content_sha256"],
        fields=action["fields"],
        theory_contract=action["theory_contract"],
        matter_coupling=action["matter_coupling"],
    )
    terms = _node(
        "exact_typed_operator_terms",
        "proven",
        "exact operator densities copied from the hash-bound typed action IR",
        [action_evidence],
        operators=action["operators"],
        parameters=action["parameters"],
    )
    return definition, terms


def _base_candidate(
    record: dict[str, Any], bindings: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    definition, terms = _action_nodes(record, bindings)
    variation = _gate(record, "covariant_variation", "pass")
    euler = _node(
        "euler_lagrange_and_noether",
        "proven",
        variation["reason"],
        [
            _evidence(
                bindings,
                "compilation",
                f"candidate_records[seed_id={record['seed_id']}].gate_ledger.covariant_variation",
                variation["evidence_root_sha256"],
            )
        ],
    )
    lineage = {
        "manifest_content_sha256": record["provenance"]["manifest_binding"][
            "content_sha256"
        ],
        "seed_lineage_sha256": record["provenance"]["seed_lineage_sha256"],
        "family_lineage_sha256": record["provenance"]["family_lineage_sha256"],
        "compilation_binding_sha256": record["provenance"]["binding_sha256"],
        "action_ir_sha256": record["provenance"]["action_ir_sha256"],
    }
    return [definition, terms, euler], lineage


def _blocked_solar_node(
    record: dict[str, Any], bindings: dict[str, Any]
) -> dict[str, Any]:
    bundle = record["solar_known_answer_bundle"]
    if bundle != {
        "generated": False,
        "reason": "formal_prerequisites_incomplete",
        "status": "blocked",
    }:
        raise ValueError("candidate Solar bundle status changed")
    return _node(
        "solar_prediction_obligation",
        "blocked",
        "no candidate-specific Solar bundle was generated because formal prerequisites are incomplete",
        [
            _evidence(
                bindings,
                "compilation",
                f"candidate_records[seed_id={record['seed_id']}].solar_known_answer_bundle",
                record["provenance"]["binding_sha256"],
            )
        ],
    )


def _aether_dossier(
    record: dict[str, Any], sources: dict[str, dict[str, Any]], bindings: dict[str, Any]
) -> dict[str, Any]:
    nodes, lineage = _base_candidate(record, bindings)
    latest = _record(sources["aether_energy"], record["seed_id"])
    _gate(latest, "regular_patch_legendre_and_dirac_constraint_algebra", "pass")
    _gate(latest, "complete_generic_twisting_reduced_hamiltonian", "blocked")
    _gate(latest, "global_positive_energy", "blocked")
    principal = _gate(record, "principal_symbol", "pass")
    nodes.extend(
        [
            _node(
                "adm_dirac_obligation",
                "blocked",
                "regular-patch Dirac algebra is proven, but the complete generic twisting reduced Hamiltonian is absent",
                [
                    _evidence(
                        bindings,
                        "aether_energy",
                        f"candidate_records[seed_id={record['seed_id']}].gate_ledger",
                        latest["provenance"]["binding_sha256"],
                    )
                ],
                proven_subresults=["regular_patch_legendre_and_dirac_constraint_algebra"],
            ),
            _node(
                "principal_symbol_obligation",
                "proven",
                principal["reason"],
                [
                    _evidence(
                        bindings,
                        "compilation",
                        f"candidate_records[seed_id={record['seed_id']}].gate_ledger.principal_symbol",
                        principal["evidence_root_sha256"],
                    )
                ],
            ),
            _node(
                "global_energy_obligation",
                "blocked",
                "quadratic and static pure-twist subresults pass, but generic nonlinear twisting energy is not certified",
                [
                    _evidence(
                        bindings,
                        "aether_energy",
                        f"candidate_records[seed_id={record['seed_id']}].gate_ledger.global_positive_energy",
                        latest["provenance"]["binding_sha256"],
                    )
                ],
                proven_subresults=[
                    "constraint_reduced_quadratic_energy",
                    "static_pure_twist_nonlinear_coercivity",
                ],
            ),
            _blocked_solar_node(record, bindings),
        ]
    )
    return _candidate_dossier(record, nodes, lineage, "blocked")


def _g2_dossier(
    record: dict[str, Any], sources: dict[str, dict[str, Any]], bindings: dict[str, Any]
) -> dict[str, Any]:
    nodes, lineage = _base_candidate(record, bindings)
    formal = _record(sources["g2_formal"], record["seed_id"])
    energy = _record(sources["g2_energy"], record["seed_id"])
    _gate(formal, "complete_distributed_dirac_boundary_contract", "blocked")
    _gate(formal, "principal_symbol", "pass")
    _gate(formal, "common_time_cone", "pass")
    _gate(energy, "general_nonmaximal_positive_mass", "blocked")
    nodes.extend(
        [
            _node(
                "adm_dirac_obligation",
                "blocked",
                "local ADM and Dirac pair pass; complete distributed boundary closure remains blocked",
                [
                    _evidence(
                        bindings,
                        "g2_formal",
                        f"candidate_records[seed_id={record['seed_id']}].gate_ledger",
                        formal["provenance"]["binding_sha256"],
                    )
                ],
                proven_subresults=["coupled_adm_primary_and_legendre", "candidate_local_dirac_pair"],
            ),
            _node(
                "principal_symbol_obligation",
                "proven",
                "candidate-specific principal symbol and common time cone pass on the registered X cell",
                [
                    _evidence(
                        bindings,
                        "g2_formal",
                        f"candidate_records[seed_id={record['seed_id']}].gate_ledger.principal_symbol",
                        formal["provenance"]["binding_sha256"],
                    )
                ],
            ),
            _node(
                "global_energy_obligation",
                "blocked",
                "DEC and restricted maximal-slice positive mass pass; a hash-bound general nonmaximal theorem is absent",
                [
                    _evidence(
                        bindings,
                        "g2_energy",
                        f"candidate_records[seed_id={record['seed_id']}].gate_ledger",
                        energy["provenance"]["binding_sha256"],
                    )
                ],
                proven_subresults=["candidate_DEC_on_contract_domain", "restricted_maximal_slice_positive_mass"],
            ),
            _blocked_solar_node(record, bindings),
        ]
    )
    return _candidate_dossier(record, nodes, lineage, "blocked")


def _g3_dossier(
    record: dict[str, Any], sources: dict[str, dict[str, Any]], bindings: dict[str, Any]
) -> dict[str, Any]:
    nodes, lineage = _base_candidate(record, bindings)
    principal = _record(sources["g3_principal"], record["seed_id"])
    dirac = _record(sources["g3_dirac"], record["seed_id"])
    af = _record(sources["g3_af"], record["seed_id"])
    nonunitary = _record(sources["nonunitary_formal"], record["seed_id"])
    _gate(principal, "uniform_principal_symbol", "pass")
    _gate(dirac, "distributed_Dirac_on_periodic_cell", "pass")
    _gate(af, "uniform_lapse_Dirac_invertibility", "blocked")
    _gate(nonunitary, "global_energy", "blocked")
    nodes.extend(
        [
            _node(
                "adm_dirac_obligation",
                "blocked",
                "full lapse operator and periodic-cell Dirac closure pass, but AF uniform invertibility and constraint solution are absent",
                [
                    _evidence(
                        bindings,
                        "g3_dirac",
                        f"candidate_records[seed_id={record['seed_id']}].gate_ledger",
                        dirac["provenance"]["binding_sha256"],
                    ),
                    _evidence(
                        bindings,
                        "g3_af",
                        f"candidate_records[seed_id={record['seed_id']}].gate_ledger",
                        af["provenance"]["binding_sha256"],
                    ),
                ],
                proven_subresults=["full_candidate_Delta_N_derivation", "distributed_Dirac_on_periodic_cell"],
            ),
            _node(
                "principal_symbol_obligation",
                "proven",
                "uniform principal/common-cone proof is restricted to the registered componentwise cell and AF reference profile",
                [
                    _evidence(
                        bindings,
                        "g3_principal",
                        f"candidate_records[seed_id={record['seed_id']}].principal_common_cone_certificate",
                        principal["principal_common_cone_certificate"]["content_sha256"],
                    ),
                    _evidence(
                        bindings,
                        "g3_af",
                        f"candidate_records[seed_id={record['seed_id']}].principal_common_cone_certificate",
                        af["principal_common_cone_certificate"]["content_sha256"],
                    ),
                ],
            ),
            _node(
                "global_energy_obligation",
                "blocked",
                "the nonunitary formulation removes a chart obstruction, but no candidate-specific AF constraint solution or global energy proof exists",
                [
                    _evidence(
                        bindings,
                        "nonunitary_formal",
                        f"candidate_records[seed_id={record['seed_id']}].gate_ledger.global_energy",
                        nonunitary["provenance"]["binding_sha256"],
                    )
                ],
            ),
            _blocked_solar_node(record, bindings),
        ]
    )
    return _candidate_dossier(record, nodes, lineage, "blocked")


def _g4_dossier(
    record: dict[str, Any], sources: dict[str, dict[str, Any]], bindings: dict[str, Any]
) -> dict[str, Any]:
    nodes, lineage = _base_candidate(record, bindings)
    formal = _record(sources["nonunitary_formal"], record["seed_id"])
    solar = _record(sources["g4_solar"], record["seed_id"])
    source_class = _record(sources["g4_source_class"], record["seed_id"])
    tail = sources["g4_tail_theorem"]["candidate_records"][0]
    if formal.get("decision") != "pass":
        raise ValueError("G4 formal-pass decision changed")
    for gate_id in [
        "AF_constraint_and_gauge_formulation",
        "candidate_specific_positive_mass",
        "formal_prerequisite_completion",
        "nonunitary_formulation_bypass",
        "physical_principal_nondegeneracy_at_scalar_gradient_zero",
    ]:
        _gate(formal, gate_id, "pass")
    for gate_id in ["Newtonian_limit", "PPN_gamma_beta", "exact_GR_scalar_free_branch"]:
        _gate(solar, gate_id, "pass")
    _gate(solar, "registered_direct_observable_Solar_bundle", "blocked")
    _gate(source_class, "source_class_static_nonlinear_uniqueness", "pass")
    _gate(source_class, "registered_real_Sun_instantiation", "blocked")
    if tail.get("theorem_decision") != "pass" or tail.get("real_Sun_instantiation_decision") != "blocked":
        raise ValueError("G4 noncompact source theorem status changed")
    nodes.extend(
        [
            _node(
                "adm_dirac_obligation",
                "proven",
                "completed in the exact alternative nonunitary AF constraint/gauge formulation",
                [
                    _evidence(
                        bindings,
                        "nonunitary_formal",
                        f"candidate_records[seed_id={record['seed_id']}].gate_ledger.AF_constraint_and_gauge_formulation",
                        formal["provenance"]["binding_sha256"],
                    )
                ],
            ),
            _node(
                "principal_symbol_obligation",
                "proven",
                "physical principal nondegeneracy at vanishing scalar gradient is proven in the nonunitary formulation",
                [
                    _evidence(
                        bindings,
                        "nonunitary_formal",
                        f"candidate_records[seed_id={record['seed_id']}].nonunitary_bypass_certificate",
                        formal["nonunitary_bypass_certificate"]["content_sha256"],
                    )
                ],
            ),
            _node(
                "global_energy_obligation",
                "proven",
                "candidate-specific positive mass passes on the exact registered AF domain",
                [
                    _evidence(
                        bindings,
                        "nonunitary_formal",
                        f"candidate_records[seed_id={record['seed_id']}].gate_ledger.candidate_specific_positive_mass",
                        formal["provenance"]["binding_sha256"],
                    )
                ],
            ),
            _node(
                "solar_analytic_prediction_on_scalar_free_branch",
                "proven",
                "Newtonian and PPN predictions plus the exact GR scalar-free branch are derived for phi_infinity=0",
                [
                    _evidence(
                        bindings,
                        "g4_solar",
                        f"candidate_records[seed_id={record['seed_id']}].coupling_and_PPN_certificate",
                        solar["coupling_and_PPN_certificate"]["content_sha256"],
                    ),
                    _evidence(
                        bindings,
                        "g4_solar",
                        f"candidate_records[seed_id={record['seed_id']}].exact_scalar_free_branch_certificate",
                        solar["exact_scalar_free_branch_certificate"]["content_sha256"],
                    ),
                ],
            ),
            _node(
                "solar_synthetic_GR_known_answer",
                "calibration_only",
                "GR numeric golden checks calibrate the solver and are not discovery-candidate evidence",
                [
                    _evidence(
                        bindings,
                        "g4_solar",
                        f"candidate_records[seed_id={record['seed_id']}].GR_calibration_control",
                        solar["GR_calibration_control"]["content_sha256"],
                    )
                ],
            ),
            _node(
                "solar_source_branch_theorem",
                "proven",
                "compact and noncompact source classes are conditional theorems, not claims that the real Sun is in either class",
                [
                    _evidence(
                        bindings,
                        "g4_source_class",
                        f"candidate_records[seed_id={record['seed_id']}].source_class_coercivity_certificate",
                        source_class["source_class_coercivity_certificate"]["content_sha256"],
                    ),
                    _evidence(
                        bindings,
                        "g4_tail_theorem",
                        "candidate_records[0]",
                        tail["provenance"]["binding_sha256"],
                    ),
                ],
            ),
            _node(
                "solar_prediction_obligation",
                "blocked",
                "real-Sun source instantiation and an exact action-bound direct-observable prediction bundle are absent",
                [
                    _evidence(
                        bindings,
                        "g4_solar",
                        f"candidate_records[seed_id={record['seed_id']}].gate_ledger.registered_direct_observable_Solar_bundle",
                        solar["provenance"]["binding_sha256"],
                    ),
                    _evidence(
                        bindings,
                        "g4_tail_theorem",
                        "candidate_records[0].first_missing_premise",
                        tail["provenance"]["binding_sha256"],
                    ),
                ],
                first_missing_premise=tail["first_missing_premise"],
            ),
        ]
    )
    return _candidate_dossier(record, nodes, lineage, "blocked_after_formal_pass")


def _candidate_dossier(
    record: dict[str, Any], nodes: list[dict[str, Any]], lineage: dict[str, Any], status: str
) -> dict[str, Any]:
    if len({node["node_id"] for node in nodes}) != len(nodes):
        raise ValueError("duplicate dossier node")
    body = {
        "dossier_id": record["seed_id"],
        "role": "generated_candidate",
        "family_id": record["family_id"],
        "overall_status": status,
        "lineage": lineage,
        "hierarchy_nodes": nodes,
    }
    return {**body, "content_sha256": _sha(body)}


def _gr_dossier(
    gr: dict[str, Any], sources: dict[str, dict[str, Any]], bindings: dict[str, Any]
) -> dict[str, Any]:
    action = sources["gr_action_spec"]
    health = sources["gr_health"]
    reference = sources["gr_reference"]
    action_evidence = _evidence(
        bindings, "gr_action_spec", "document", gr["input_action_sha256"]
    )
    health_evidence = lambda gate: _evidence(
        bindings, "gr_health", f"gates.{gate}", health["gates"][gate].get(f"{gate}_ir_sha256")
    )
    nodes = [
        _node(
            "defining_covariant_action",
            "proven",
            "hash-bound Einstein-Hilbert known-answer action specification",
            [action_evidence],
            action_sha256=gr["input_action_sha256"],
            fields=action["fields"],
            matter_metric=action["matter_metric"],
        ),
        _node(
            "exact_typed_operator_terms",
            "proven",
            "exact action-spec term identifiers; no density was reconstructed by this dossier",
            [action_evidence],
            operator_terms=action["terms"],
            coefficients=action["coefficients"],
        ),
        _node(
            "euler_lagrange_and_noether",
            "proven",
            health["gates"]["covariant_variation"]["scope"],
            [health_evidence("covariant_variation")],
        ),
        _node(
            "adm_dirac_obligation",
            "proven",
            health["gates"]["adm_dirac"]["scope"],
            [health_evidence("adm_dirac")],
        ),
        _node(
            "principal_symbol_obligation",
            "proven",
            health["gates"]["principal_symbol"]["scope"],
            [
                _evidence(
                    bindings,
                    "gr_health",
                    "gates.principal_symbol",
                    health["gates"]["principal_symbol"]["principal_ir_sha256"],
                )
            ],
        ),
        _node(
            "local_reduced_energy_obligation",
            "proven",
            health["gates"]["hamiltonian_stability"]["scope"],
            [
                _evidence(
                    bindings,
                    "gr_health",
                    "gates.hamiltonian_stability",
                    health["gates"]["hamiltonian_stability"]["hamiltonian_ir_sha256"],
                )
            ],
        ),
        _node(
            "generic_nonlinear_total_energy_obligation",
            "blocked",
            health["gates"]["hamiltonian_stability"]["generic_nonlinear_total_energy"]["scope"],
            [health_evidence("hamiltonian_stability")],
            upstream_status="not_claimed",
        ),
        _node(
            "solar_known_answer_predictions",
            "calibration_only",
            "five GR golden checks validate the reference solver and do not authorize candidate observations",
            [_evidence(bindings, "gr_reference", "golden_checks")],
            golden_check_names=[item["name"] for item in reference["golden_checks"]],
            pass_count=reference["counts"]["passed"],
        ),
    ]
    body = {
        "dossier_id": "GR-EINSTEIN-HILBERT",
        "role": "known_answer_calibration_control",
        "eligible_as_generated_candidate": False,
        "overall_status": "calibration_control",
        "lineage": {
            "control_lineage_sha256": gr["control_lineage_sha256"],
            "input_action_sha256": gr["input_action_sha256"],
            "manifest_content_sha256": bindings["manifest"]["content_sha256"],
        },
        "hierarchy_nodes": nodes,
    }
    return {**body, "content_sha256": _sha(body)}


def build_candidate_theory_dossier_campaign(
    config: dict[str, Any], root: str | Path
) -> dict[str, Any]:
    root = Path(root).resolve()
    if config.get("data_eligibility") != ELIGIBILITY:
        raise ValueError("campaign eligibility is not fail-closed")
    if config.get("observational_authorization") is not False:
        raise ValueError("observational authorization must remain false")
    bindings = config["source_bindings"]
    sources = {key: _load_bound(root, value) for key, value in bindings.items()}
    gr = _validate_inputs(sources, bindings)
    compilation_records = {
        item["seed_id"]: item for item in sources["compilation"]["candidate_records"]
    }
    dossiers = [_gr_dossier(gr, sources, bindings)]
    for seed_id in sorted(SEED_IDS):
        record = compilation_records[seed_id]
        family = record["family_id"]
        if family == "AETHER_K1234_PARAMETER_CELL":
            dossier = _aether_dossier(record, sources, bindings)
        elif family == "KESSENCE_G2_CONVEX":
            dossier = _g2_dossier(record, sources, bindings)
        elif family == "CUBIC_HORNDESKI_G3_WEAK_CELL":
            dossier = _g3_dossier(record, sources, bindings)
        elif family == "CONFORMAL_G4_PHI_SCALAR_TENSOR":
            dossier = _g4_dossier(record, sources, bindings)
        else:
            raise ValueError(f"unsupported dossier family: {family}")
        dossiers.append(dossier)
    all_nodes = [node for dossier in dossiers for node in dossier["hierarchy_nodes"]]
    status_counts = Counter(node["status"] for node in all_nodes)
    if set(status_counts) - NODE_STATUSES:
        raise ValueError("unexpected dossier status")
    provenance_body = {
        "source_file_sha256": {
            key: descriptor["file_sha256"] for key, descriptor in sorted(bindings.items())
        },
        "source_content_sha256": {
            key: descriptor["content_sha256"]
            for key, descriptor in sorted(bindings.items())
            if descriptor.get("content_sha256") is not None
        },
        "dossier_content_sha256": {
            item["dossier_id"]: item["content_sha256"] for item in dossiers
        },
        "data_eligibility": ELIGIBILITY,
    }
    body = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": config["campaign_id"],
        "source_bindings": bindings,
        "dossier_count": len(dossiers),
        "known_answer_control_count": 1,
        "generated_candidate_count": 6,
        "hierarchy_node_status_counts": dict(sorted(status_counts.items())),
        "dossiers": dossiers,
        "provenance": {**provenance_body, "binding_sha256": _sha(provenance_body)},
        "observational_authorization": False,
        "observational_data_opened": False,
        "tracking_target_values_opened": False,
        "paid_llm_spend_usd": 0.0,
        "data_eligibility": dict(ELIGIBILITY),
        "interpretation": (
            "This deterministic dossier exposes the exact action-to-obligation hierarchy for "
            "one GR calibration control and six grammar-v3 candidates. A proven node is limited "
            "to its cited artifact scope; blocked nodes are not inferred from family labels; GR "
            "golden checks remain calibration-only; no observation was opened."
        ),
    }
    serialized = _canonical(body)
    if "C:\\Users\\" in serialized or "C:/Users/" in serialized:
        raise ValueError("host path leaked into dossier")
    return {**body, "content_sha256": _sha(body)}
