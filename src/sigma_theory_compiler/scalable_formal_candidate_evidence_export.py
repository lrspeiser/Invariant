from __future__ import annotations

import hashlib
import json
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .aether_parameter_cell_formal_gate_campaign import (
    build_aether_parameter_cell_formal_gate_campaign,
    build_aether_parameter_cell_formal_gate_status,
)
from .covariant_grammar_v3_seed_compilation_campaign import _compile_action_ir
from .g4_scalable_action_formal_followup import build_g4_scalable_action_formal_followup
from .grammar_v3_formal_preflight_service import GrammarV3FormalPreflightService
from .grammar_v3_g2_candidate_formal_service import GrammarV3G2CandidateFormalService
from .grammar_v3_g3_candidate_formal_service import GrammarV3G3CandidateFormalService
from .grammar_v3_parameter_cell_compilation_campaign import _action_density_key
from .grammar_v3_parameter_cell_manifest_campaign import iter_parameter_cells
from .grammar_v3_promotion_admission_service import GrammarV3PromotionAdmissionService
from .persistent_parallel_search import WorkLease
from .promotion_orchestrator import ELIGIBILITY

CONFIG_SCHEMA = "sigma-scalable-formal-candidate-evidence-export-config-1.0"
EXPORT_SCHEMA = "sigma-scalable-formal-candidate-evidence-export-1.0"
RECORD_SCHEMA = "sigma-scalable-formal-candidate-evidence-record-1.0"
RECORD_COLUMNS = (
    "candidate_id",
    "family_id",
    "action_sha256",
    "theory_formula_inputs",
    "alias_count",
    "alias_lineage_root_sha256",
    "preflight_decision",
    "preflight_result_sha256",
    "final_decision",
    "blocker",
    "evidence_source",
    "result_sha256",
    "direct_metrics",
    "metric_source_sha256",
    "comparison_data_class",
)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path.name} must contain an object")
    return value


def _validate_config(config: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "campaign_id",
        "parameter_manifest",
        "parameter_manifest_config",
        "compilation_artifact",
        "compilation_config",
        "preflight_status",
        "preflight_config",
        "promotion_status",
        "promotion_config",
        "aether_status",
        "aether_config",
        "g2_status",
        "g2_config",
        "g3_status",
        "g3_config",
        "g4_followup",
        "g4_followup_config",
        "budget",
        "data_eligibility",
        "external_paid_llm_calls",
    }
    if set(config) != required or config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("scalable formal evidence export config is invalid")
    if config.get("data_eligibility") != ELIGIBILITY:
        raise ValueError("scalable formal evidence export eligibility is not fail-closed")
    if config.get("external_paid_llm_calls") is not False:
        raise ValueError("scalable formal evidence export enabled paid LLM calls")
    budget = config.get("budget", {})
    if set(budget) != {
        "maximum_candidates",
        "maximum_parameter_cells",
        "maximum_output_bytes",
        "maximum_paid_llm_spend_usd",
    } or (
        int(budget["maximum_candidates"]) != 163
        or int(budget["maximum_parameter_cells"]) != 256
        or not 1024 * 1024 <= int(budget["maximum_output_bytes"]) <= 16 * 1024 * 1024
        or float(budget["maximum_paid_llm_spend_usd"]) != 0.0
    ):
        raise ValueError("scalable formal evidence export budget is invalid")


def _binding_path(root: Path, binding: dict[str, Any], label: str) -> Path:
    if set(binding) - {"path", "file_sha256", "content_sha256"} or not {
        "path",
        "file_sha256",
    }.issubset(binding):
        raise ValueError(f"{label} binding fields are invalid")
    path = (root / binding["path"]).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} path escapes repository") from error
    if not path.is_file() or _file_sha(path) != binding["file_sha256"]:
        raise ValueError(f"{label} file hash mismatch")
    return path


def _bound_json(
    root: Path, config: dict[str, Any], key: str, *, content: bool = False
) -> dict[str, Any]:
    binding = config[key]
    value = _load(_binding_path(root, binding, key))
    if content:
        body = {name: item for name, item in value.items() if name != "content_sha256"}
        if value.get("content_sha256") != binding.get("content_sha256") or _sha(body) != binding[
            "content_sha256"
        ]:
            raise ValueError(f"{key} content hash mismatch")
    return value


def _lease(payload: dict[str, Any]) -> WorkLease:
    return WorkLease(
        work_id="immutable-export-attestation",
        ordinal=int(payload["ordinal"]),
        lane="cpu",
        seed=0,
        attempt=1,
        max_attempts=1,
        payload=payload,
    )


def _status_records(
    work_items: list[dict[str, Any]], results: dict[str, dict[str, Any]], *, family: str
) -> list[dict[str, Any]]:
    records = []
    for payload in work_items:
        result = results[payload["candidate_id"]]
        record = {
            "candidate_id": payload["candidate_id"],
            "typed_action_ir_sha256": payload["typed_action_ir_sha256"],
            "preflight_result_sha256": payload["preflight_result_sha256"],
            "admission_result_sha256": payload["admission_result_sha256"],
        }
        if family == "g2":
            record["quadratic_coefficient"] = payload["quadratic_coefficient"]
        elif family == "g3":
            record["beta"] = payload["beta"]
            record["candidate_evidence_sha256"] = payload["candidate_evidence_sha256"]
        else:
            raise ValueError("unknown formal status family")
        record.update(
            state="succeeded",
            attempt=1,
            result_sha256=result["content_sha256"],
            first_missing_premise=result["first_missing_premise"],
            error_text=None,
        )
        records.append(record)
    return records


def _alias_groups(
    cells: list[dict[str, Any]],
    preflight_items: list[dict[str, Any]],
    source_manifest: dict[str, Any],
    cell_manifest: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    preflight_by_id = {item["candidate_id"]: item for item in preflight_items}
    groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        equivalence = _sha(_action_density_key(cell))
        candidate_id = "G3A-" + equivalence[:24]
        groups[candidate_id].append(cell)
    if set(groups) != set(preflight_by_id) or sum(map(len, groups.values())) != 256:
        raise ValueError("candidate alias groups do not match formal-preflight registry")
    output = {}
    families = {
        family["family_id"]: family
        for family in source_manifest["typed_family_seeds"]
        if family["enabled_for_generation"]
    }
    manifest_binding = {
        "parameter_cell_manifest_content_sha256": cell_manifest["content_sha256"],
        "parameter_cell_registry_root_sha256": cell_manifest[
            "parameter_cell_registry_root_sha256"
        ],
    }
    for candidate_id in sorted(groups):
        source = preflight_by_id[candidate_id]
        ordered = sorted(groups[candidate_id], key=lambda cell: int(cell["ordinal"]))
        representative = ordered[0]
        pseudo_seed = {
            "seed_id": representative["parameter_cell_id"],
            "seed_lineage_sha256": representative["parameter_cell_lineage_sha256"],
            "family_id": representative["family_id"],
            "family_lineage_sha256": representative["family_lineage_sha256"],
            "theory_contract": representative["theory_contract"],
            "operator_atoms": representative["operator_atoms"],
            "parameters": representative["parameters"],
        }
        action_ir = _compile_action_ir(
            pseudo_seed, families[representative["family_id"]], manifest_binding
        )
        if (
            source["representative_cell_id"] != representative["parameter_cell_id"]
            or source["representative_cell_lineage_sha256"]
            != representative["parameter_cell_lineage_sha256"]
            or source["typed_action_ir_sha256"] != action_ir["content_sha256"]
        ):
            raise ValueError("representative cell changed from formal preflight")
        aliases = []
        for cell in ordered[1:]:
            body = {
                "parameter_cell_id": cell["parameter_cell_id"],
                "parameter_cell_lineage_sha256": cell["parameter_cell_lineage_sha256"],
                "ordinal": int(cell["ordinal"]),
            }
            aliases.append({**body, "content_sha256": _sha(body)})
        formula_body = {
            "fields": action_ir["fields"],
            "parameters": action_ir["parameters"],
            "ordered_operator_densities": [
                {"atom": operator["atom"], "density": operator["density"]}
                for operator in action_ir["operators"]
            ],
            "action_content_sha256": action_ir["content_sha256"],
        }
        output[candidate_id] = {
            "action_density_equivalence_sha256": source[
                "action_density_equivalence_sha256"
            ],
            "representative_action_sha256": source["typed_action_ir_sha256"],
            "representative_cell_id": representative["parameter_cell_id"],
            "representative_cell_lineage_sha256": representative[
                "parameter_cell_lineage_sha256"
            ],
            "alias_count": len(aliases),
            "alias_lineage": aliases,
            "theory_formula_inputs": {
                **formula_body,
                "formula_inputs_sha256": _sha(formula_body),
            },
        }
    return output


def _final_evidence(
    candidate_id: str,
    family_id: str,
    *,
    aether: dict[str, dict[str, Any]],
    g2_results: dict[str, dict[str, Any]],
    g3_results: dict[str, dict[str, Any]],
    g3_evidence: dict[str, dict[str, Any]],
    g4_followup: dict[str, Any],
    preflight_result: dict[str, Any],
) -> dict[str, Any]:
    direct_metrics: dict[str, Any] = {}
    metric_source_sha256 = None
    if family_id == "AETHER_K1234_PARAMETER_CELL":
        source = aether[candidate_id]
        decision = source["decision"]
        blocker = source["blocker"]
        result_sha256 = source["content_sha256"]
        gate_ledger = {
            gate: value["status"] for gate, value in sorted(source["gate_ledger"].items())
        }
        if decision == "reject":
            specialization = source["exact_specialization"]
            direct_metrics = {
                "c123": specialization["combinations"]["c123"],
                "spin_0_principal_speed_squared": specialization[
                    "principal_speed_squared"
                ]["spin_0"],
            }
            metric_source_sha256 = specialization["content_sha256"]
        evidence_source = "aether_parameter_cell_formal_gate"
    elif family_id == "KESSENCE_G2_CONVEX":
        source = g2_results[candidate_id]
        decision = source["decision"]
        blocker = source["first_missing_premise"]
        result_sha256 = source["content_sha256"]
        gate_ledger = source["gate_ledger"]
        evidence_source = "grammar_v3_g2_candidate_formal"
    elif family_id == "CUBIC_HORNDESKI_G3_WEAK_CELL":
        source = g3_results[candidate_id]
        if candidate_id not in g3_evidence:
            raise ValueError("G3 candidate evidence is missing")
        decision = source["decision"]
        blocker = source["first_missing_premise"]
        result_sha256 = source["content_sha256"]
        gate_ledger = source["gate_ledger"]
        evidence_source = "grammar_v3_g3_candidate_formal"
    elif family_id == "CONFORMAL_G4_PHI_SCALAR_TENSOR":
        if (
            candidate_id != g4_followup.get("candidate_id")
            or preflight_result["decision"] != "blocked"
            or g4_followup.get("preflight_decision") != "blocked"
            or g4_followup.get("formal_followup_decision") != "pass"
        ):
            raise ValueError("G4 formal follow-up does not bind the blocked preflight candidate")
        decision = "pass"
        blocker = None
        result_sha256 = g4_followup["content_sha256"]
        gate_ledger = g4_followup["gate_ledger"]
        certificate = g4_followup["equivalence_certificate"]
        direct_metrics = {
            "action_density_projection_equal": certificate[
                "action_density_projection_equal"
            ],
            "equivalent_parameter_cell_alias_count": certificate[
                "equivalent_parameter_cell_alias_count"
            ],
            "formal_pass_count": g4_followup["formal_pass_count"],
        }
        metric_source_sha256 = certificate["content_sha256"]
        evidence_source = "g4_scalable_action_formal_followup"
    else:
        raise ValueError("unreviewed final-evidence family")
    if decision not in {"blocked", "pass", "reject"}:
        raise ValueError("scalable formal export found an unsupported final decision")
    rank_eligible = decision in {"pass", "reject"}
    if decision == "reject":
        comparison_data_class = "aether_aligned_minkowski_principal_necessary_condition"
    elif decision == "pass":
        comparison_data_class = "full_formal_action_evidence"
    else:
        comparison_data_class = None
    leaderboard = {
        "rank": None,
        "rank_eligible": rank_eligible,
        "evidence_status": decision,
        "gate_completeness": "complete_for_category" if rank_eligible else "incomplete",
        "comparison_data_class": comparison_data_class,
        "promotion_eligible": False,
    }
    return {
        "decision": decision,
        "blocker": blocker,
        "evidence_source": evidence_source,
        "result_sha256": result_sha256,
        "gate_ledger": gate_ledger,
        "direct_metrics": direct_metrics,
        "metric_source_sha256": metric_source_sha256,
        "leaderboard_contract": leaderboard,
    }


def iter_scalable_formal_candidate_evidence_records(
    export: dict[str, Any],
) -> list[dict[str, Any]]:
    """Expand the compact immutable table into leaderboard-safe candidate records."""
    if export.get("candidate_record_columns") != list(RECORD_COLUMNS):
        raise ValueError("scalable formal evidence record columns changed")
    records = []
    for row in export.get("candidate_records", []):
        if not isinstance(row, list) or len(row) != len(RECORD_COLUMNS):
            raise ValueError("scalable formal evidence compact record is invalid")
        item = dict(zip(RECORD_COLUMNS, row, strict=True))
        decision = item["final_decision"]
        rank_eligible = decision in {"pass", "reject"}
        item["leaderboard_contract"] = {
            "rank": None,
            "rank_eligible": rank_eligible,
            "evidence_status": decision,
            "gate_completeness": "complete_for_category" if rank_eligible else "incomplete",
            "comparison_data_class": item["comparison_data_class"],
            "promotion_eligible": False,
        }
        records.append(item)
    return records


def validate_scalable_formal_candidate_evidence_export(export: dict[str, Any]) -> None:
    body = {key: value for key, value in export.items() if key != "content_sha256"}
    if export.get("schema_version") != EXPORT_SCHEMA or export.get("content_sha256") != _sha(body):
        raise ValueError("scalable formal evidence export content hash mismatch")
    compact_records = export.get("candidate_records")
    records = iter_scalable_formal_candidate_evidence_records(export)
    if not isinstance(compact_records, list) or len(records) != 163:
        raise ValueError("scalable formal evidence export candidate count changed")
    if len({record.get("candidate_id") for record in records}) != 163:
        raise ValueError("scalable formal evidence export candidate identity collision")
    decisions = Counter()
    aliases = 0
    if export.get("candidate_record_registry_root_sha256") != _sha(compact_records):
        raise ValueError("scalable formal evidence record hash mismatch")
    for record in records:
        decisions[record["final_decision"]] += 1
        aliases += int(record["alias_count"])
        formula = record["theory_formula_inputs"]
        formula_body = {
            key: value for key, value in formula.items() if key != "formula_inputs_sha256"
        }
        if (
            set(formula_body)
            != {
                "fields",
                "parameters",
                "ordered_operator_densities",
                "action_content_sha256",
            }
            or formula.get("formula_inputs_sha256") != _sha(formula_body)
            or formula.get("action_content_sha256") != record["action_sha256"]
            or not formula["fields"]
            or not formula["ordered_operator_densities"]
        ):
            raise ValueError("candidate theory formula inputs or action hash changed")
        board = record["leaderboard_contract"]
        if record["final_decision"] == "blocked":
            if board != {
                "rank": None,
                "rank_eligible": False,
                "evidence_status": "blocked",
                "gate_completeness": "incomplete",
                "comparison_data_class": None,
                "promotion_eligible": False,
            }:
                raise ValueError("blocked candidate entered a scored leaderboard state")
        elif record["final_decision"] == "reject":
            if (
                record["family_id"] != "AETHER_K1234_PARAMETER_CELL"
                or board["rank"] is not None
                or board["rank_eligible"] is not True
                or board["evidence_status"] != "reject"
                or board["gate_completeness"] != "complete_for_category"
                or board["comparison_data_class"]
                != "aether_aligned_minkowski_principal_necessary_condition"
                or set(record["direct_metrics"])
                != {"c123", "spin_0_principal_speed_squared"}
                or record["metric_source_sha256"] is None
            ):
                raise ValueError("noncomparable or incomplete reject became rank eligible")
        elif record["final_decision"] == "pass":
            if (
                record["candidate_id"] != "G3A-e0eff4150989e3522dc6ba03"
                or record["action_sha256"]
                != "7dd636e53f7cc161feabcb02b1f575bc1da3bd6b84033e870d2d9024c6cd5d21"
                or record["preflight_decision"] != "blocked"
                or board["comparison_data_class"] != "full_formal_action_evidence"
                or set(record["direct_metrics"])
                != {
                    "action_density_projection_equal",
                    "equivalent_parameter_cell_alias_count",
                    "formal_pass_count",
                }
                or record["metric_source_sha256"] is None
            ):
                raise ValueError("G4 formal pass lineage or metrics changed")
        else:
            raise ValueError("formal evidence export contains unknown decision")
        metric_text = _canonical(record["direct_metrics"]).lower()
        if any(
            token in metric_text
            for token in (
                "aggregate",
                "truth_score",
                "overall_score",
                "dark_matter",
                "halo",
                "redshift",
            )
        ):
            raise ValueError("aggregate or forbidden metric entered candidate evidence")
        if record["direct_metrics"] and record["metric_source_sha256"] is None:
            raise ValueError("candidate metrics lack exact evidence provenance")
    if dict(sorted(decisions.items())) != {"blocked": 160, "pass": 1, "reject": 2} or aliases != 93:
        raise ValueError("scalable formal evidence final accounting changed")
    if export.get("final_decision_counts") != {"blocked": 160, "pass": 1, "reject": 2}:
        raise ValueError("scalable formal evidence aggregate decision counts changed")
    if export.get("formal_pass_count") != 1 or export.get("rank_eligible_count") != 3:
        raise ValueError("scalable formal evidence pass/rank contract changed")
    g4 = export.get("g4_followup_provenance", {})
    if g4 != {
        "content_sha256": "7f470af2f26051da8429cd9663ea846277a84d624555e2c4bd48baecc08989db",
        "binding_sha256": "c25eb1187844323a3db7a73f69615da25957eb7ba9049c9079e671ce48b6e370",
    }:
        raise ValueError("G4 formal follow-up provenance changed")
    if export.get("data_eligibility") != {**ELIGIBILITY, "passed": True}:
        raise ValueError("scalable formal evidence export opened forbidden data")


def build_scalable_formal_candidate_evidence_export(
    config: dict[str, Any], root: str | Path
) -> dict[str, Any]:
    _validate_config(config)
    root = Path(root).resolve()
    manifest = _bound_json(root, config, "parameter_manifest", content=True)
    _bound_json(root, config, "parameter_manifest_config")
    compilation = _bound_json(root, config, "compilation_artifact", content=True)
    _bound_json(root, config, "compilation_config")
    preflight_status = _bound_json(root, config, "preflight_status", content=True)
    preflight_config = _bound_json(root, config, "preflight_config")
    promotion_status = _bound_json(root, config, "promotion_status", content=True)
    promotion_config = _bound_json(root, config, "promotion_config")
    aether_status = _bound_json(root, config, "aether_status", content=True)
    aether_config = _bound_json(root, config, "aether_config")
    g2_status = _bound_json(root, config, "g2_status", content=True)
    g2_config = _bound_json(root, config, "g2_config")
    g3_status = _bound_json(root, config, "g3_status", content=True)
    g3_config = _bound_json(root, config, "g3_config")
    g4_followup = _bound_json(root, config, "g4_followup", content=True)
    g4_followup_config = _bound_json(root, config, "g4_followup_config")
    if (
        manifest.get("parameter_cell_count") != 256
        or compilation.get("unique_candidate_count") != 163
        or compilation.get("equivalent_duplicate_count") != 93
        or preflight_status.get("decision_counts") != {"blocked": 1, "pass": 162}
        or promotion_status.get("decision_counts") != {"pass": 162}
    ):
        raise ValueError("scalable formal evidence upstream accounting changed")
    if (
        build_g4_scalable_action_formal_followup(g4_followup_config, root) != g4_followup
        or g4_followup.get("provenance", {}).get("binding_sha256")
        != "c25eb1187844323a3db7a73f69615da25957eb7ba9049c9079e671ce48b6e370"
    ):
        raise ValueError("G4 formal follow-up replay or provenance changed")

    with tempfile.TemporaryDirectory(prefix="sigma-formal-evidence-") as temporary:
        temporary_root = Path(temporary)
        preflight = GrammarV3FormalPreflightService(
            temporary_root / "preflight", preflight_config, root
        )
        preflight_results = {
            item["candidate_id"]: preflight.execute_lease(_lease(item))
            for item in preflight.work_items
        }
        if (
            preflight.candidate_registry_root_sha256
            != preflight_status["candidate_registry_root_sha256"]
            or Counter(result["decision"] for result in preflight_results.values())
            != Counter(preflight_status["decision_counts"])
        ):
            raise ValueError("formal-preflight candidate replay changed")
        promotion = GrammarV3PromotionAdmissionService(
            temporary_root / "promotion", promotion_config, root
        )
        admission_results = {
            item["candidate_id"]: promotion.execute_lease(_lease(item))
            for item in promotion.work_items
        }
        if (
            promotion.eligible_candidate_registry_root_sha256
            != promotion_status["eligible_candidate_registry_root_sha256"]
            or Counter(result["decision"] for result in admission_results.values())
            != Counter(promotion_status["decision_counts"])
        ):
            raise ValueError("promotion-admission candidate replay changed")
        g2 = GrammarV3G2CandidateFormalService(
            temporary_root / "g2", g2_config, root
        )
        g2_results = {
            item["candidate_id"]: g2.execute_lease(_lease(item)) for item in g2.work_items
        }
        if (
            g2.candidate_registry_root_sha256 != g2_status["candidate_registry_root_sha256"]
            or g2.reviewed_adapter_registry_root_sha256
            != g2_status["reviewed_adapter_registry_root_sha256"]
            or _sha(_status_records(g2.work_items, g2_results, family="g2"))
            != g2_status["record_registry_root_sha256"]
        ):
            raise ValueError("G2 formal candidate replay changed")
        g3 = GrammarV3G3CandidateFormalService(
            temporary_root / "g3", g3_config, root
        )
        g3_results = {
            item["candidate_id"]: g3.execute_lease(_lease(item)) for item in g3.work_items
        }
        if (
            g3.candidate_registry_root_sha256 != g3_status["candidate_registry_root_sha256"]
            or g3.candidate_evidence_registry_root_sha256
            != g3_status["candidate_evidence_registry_root_sha256"]
            or g3.reviewed_adapter_registry_root_sha256
            != g3_status["reviewed_adapter_registry_root_sha256"]
            or _sha(_status_records(g3.work_items, g3_results, family="g3"))
            != g3_status["record_registry_root_sha256"]
        ):
            raise ValueError("G3 formal candidate replay changed")

    aether_campaign = build_aether_parameter_cell_formal_gate_campaign(aether_config, root)
    if build_aether_parameter_cell_formal_gate_status(aether_campaign) != aether_status:
        raise ValueError("Aether formal candidate replay changed")
    aether_by_id = {record["candidate_id"]: record for record in aether_campaign["candidate_records"]}
    aliases = _alias_groups(
        list(iter_parameter_cells(preflight.cell_manifest, preflight.source_manifest)),
        preflight.work_items,
        preflight.source_manifest,
        preflight.cell_manifest,
    )
    preflight_by_id = {item["candidate_id"]: item for item in preflight.work_items}
    admission_by_id = {item["candidate_id"]: item for item in promotion.work_items}
    records = []
    for candidate_id in sorted(preflight_by_id):
        source = preflight_by_id[candidate_id]
        preflight_result = preflight_results[candidate_id]
        lineage_source = aliases[candidate_id]
        formula_inputs = lineage_source["theory_formula_inputs"]
        lineage = {
            key: value
            for key, value in lineage_source.items()
            if key != "theory_formula_inputs"
        }
        if source["typed_action_ir_sha256"] != lineage["representative_action_sha256"]:
            raise ValueError("candidate representative action binding changed")
        admission_payload = admission_by_id.get(candidate_id)
        if admission_payload is None:
            admission = {
                "state": "not_run_preflight_blocked",
                "decision": "not_run",
                "result_sha256": None,
                "target_task_type": None,
            }
        else:
            admission_result = admission_results[candidate_id]
            admission = {
                "state": "completed",
                "decision": admission_result["decision"],
                "result_sha256": admission_result["content_sha256"],
                "target_task_type": admission_result["target_task_type"],
            }
        final = _final_evidence(
            candidate_id,
            source["family_id"],
            aether=aether_by_id,
            g2_results=g2_results,
            g3_results=g3_results,
            g3_evidence=g3.candidate_evidence,
            g4_followup=g4_followup,
            preflight_result=preflight_result,
        )
        body = {
            "schema_version": RECORD_SCHEMA,
            "candidate_id": candidate_id,
            "family_id": source["family_id"],
            "action_sha256": source["typed_action_ir_sha256"],
            "theory_formula_inputs": formula_inputs,
            "compilation_lineage": lineage,
            "preflight": {
                "input_lineage_sha256": source["input_lineage_sha256"],
                "result_sha256": preflight_result["content_sha256"],
                "decision": preflight_result["decision"],
                "blocker": preflight_result["blocker"],
            },
            "admission": admission,
            "final_evidence": final,
        }
        records.append({**body, "content_sha256": _sha(body)})
    decision_counts = dict(
        sorted(Counter(record["final_evidence"]["decision"] for record in records).items())
    )
    family_counts = {
        family: {
            "candidate_count": len(selected),
            "decision_counts": dict(
                sorted(Counter(item["final_evidence"]["decision"] for item in selected).items())
            ),
            "alias_count": sum(item["compilation_lineage"]["alias_count"] for item in selected),
        }
        for family in sorted({record["family_id"] for record in records})
        if (selected := [record for record in records if record["family_id"] == family])
    }
    bindings = {
        key: config[key]
        for key in (
            "parameter_manifest",
            "parameter_manifest_config",
            "compilation_artifact",
            "compilation_config",
            "preflight_status",
            "preflight_config",
            "promotion_status",
            "promotion_config",
            "aether_status",
            "aether_config",
            "g2_status",
            "g2_config",
            "g3_status",
            "g3_config",
            "g4_followup",
            "g4_followup_config",
        )
    }
    compact_records = [
        [
            record["candidate_id"],
            record["family_id"],
            record["action_sha256"],
            record["theory_formula_inputs"],
            record["compilation_lineage"]["alias_count"],
            _sha(record["compilation_lineage"]),
            record["preflight"]["decision"],
            record["preflight"]["result_sha256"],
            record["final_evidence"]["decision"],
            record["final_evidence"]["blocker"],
            record["final_evidence"]["evidence_source"],
            record["final_evidence"]["result_sha256"],
            record["final_evidence"]["direct_metrics"],
            record["final_evidence"]["metric_source_sha256"],
            record["final_evidence"]["leaderboard_contract"][
                "comparison_data_class"
            ],
        ]
        for record in records
    ]
    body = {
        "schema_version": EXPORT_SCHEMA,
        "campaign_id": config["campaign_id"],
        "source_bindings": bindings,
        "parameter_cell_count": 256,
        "candidate_count": len(records),
        "alias_count": sum(record["compilation_lineage"]["alias_count"] for record in records),
        "family_counts": family_counts,
        "final_decision_counts": decision_counts,
        "formal_pass_count": 1,
        "rank_eligible_count": sum(
            record["final_evidence"]["leaderboard_contract"]["rank_eligible"]
            for record in records
        ),
        "candidate_record_columns": list(RECORD_COLUMNS),
        "candidate_record_registry_root_sha256": _sha(compact_records),
        "alias_lineage_registry_root_sha256": _sha(
            [record["compilation_lineage"] for record in records]
        ),
        "leaderboard_contract": (
            "blocked candidates are rank=None/incomplete; exact Aether rejects and the "
            "reviewed G4 formal pass are eligible only within separate evidence classes; "
            "no global truth score"
        ),
        "g4_followup_provenance": {
            "content_sha256": g4_followup["content_sha256"],
            "binding_sha256": g4_followup["provenance"]["binding_sha256"],
        },
        "candidate_records": compact_records,
        "observational_data_opened": False,
        "paid_llm_spend_usd": 0.0,
        "data_eligibility": {**ELIGIBILITY, "passed": True},
    }
    export = {**body, "content_sha256": _sha(body)}
    if len(_canonical(export).encode()) > int(config["budget"]["maximum_output_bytes"]):
        raise ValueError("scalable formal evidence export exceeds bounded output budget")
    validate_scalable_formal_candidate_evidence_export(export)
    return export
