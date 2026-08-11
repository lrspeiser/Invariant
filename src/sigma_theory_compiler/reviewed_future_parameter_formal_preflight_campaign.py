"""Reviewed, candidate-bound preflight for the first future parameter chunk."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .covariant_grammar_v3_seed_compilation_campaign import _compile_action_ir
from .promotion_orchestrator import ELIGIBILITY
from .scalable_future_parameter_chunk_campaign import (
    build_future_parameter_manifest_chunk,
    compile_future_parameter_chunk,
)

CONFIG_SCHEMA = "sigma-reviewed-future-parameter-formal-preflight-config-1.0"
RESULT_SCHEMA = "sigma-reviewed-future-parameter-formal-preflight-1.0"
AETHER = "AETHER_K1234_PARAMETER_CELL"
G3 = "CUBIC_HORNDESKI_G3_WEAK_CELL"
EXPECTED_OPERATORS = {
    AETHER: [
        "EH_R",
        "AETHER_K1",
        "AETHER_K2",
        "AETHER_K3",
        "AETHER_K4",
        "UNIT_VECTOR_CONSTRAINT",
    ],
    G3: ["EH_R", "G2_PHI_X", "G3_PHI_X_BOX_PHI"],
}
EXPECTED_ADAPTERS = {
    (AETHER, "einstein_aether_flrw_variation_noether"),
    (G3, "generic_g2_variation_noether_identity"),
    (G3, "generic_g3_variation_noether_identity"),
    (G3, "generic_cubic_horndeski_bssn_hyperbolicity"),
    (G3, "generic_cubic_horndeski_scalar_effective_metric"),
    (G3, "generic_horndeski_l2_l4_unitary_adm_primary_degeneracy"),
    (G3, "generic_horndeski_l2_l4_unitary_distributed_dirac_closure"),
}


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


def _bound_path(root: Path, binding: dict[str, Any], label: str) -> Path:
    if set(binding) - {"path", "file_sha256", "content_sha256"}:
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
    root: Path, binding: dict[str, Any], label: str, *, content: bool = False
) -> dict[str, Any]:
    value = _load(_bound_path(root, binding, label))
    if content:
        body = {key: item for key, item in value.items() if key != "content_sha256"}
        if (
            value.get("content_sha256") != binding.get("content_sha256")
            or _sha(body) != binding["content_sha256"]
        ):
            raise ValueError(f"{label} content hash mismatch")
    return value


def _validate_config(config: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "campaign_id",
        "campaign_implementation",
        "source_status",
        "source_campaign_config",
        "source_campaign_implementation",
        "source_seed_manifest",
        "formal_report",
        "reviewed_specializers",
        "reviewed_adapters",
        "g3_center_calibration",
        "budget",
        "data_eligibility",
        "external_paid_llm_calls",
    }
    if set(config) != required or config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("reviewed future preflight config is invalid")
    if config.get("data_eligibility") != ELIGIBILITY:
        raise ValueError("reviewed future preflight eligibility is open")
    if config.get("external_paid_llm_calls") is not False:
        raise ValueError("reviewed future preflight enabled paid LLM calls")
    budget = config["budget"]
    if set(budget) != {
        "maximum_candidates",
        "maximum_adapter_invocations",
        "maximum_paid_llm_spend_usd",
    } or (
        int(budget["maximum_candidates"]) != 19
        or int(budget["maximum_adapter_invocations"]) != 7
        or float(budget["maximum_paid_llm_spend_usd"]) != 0.0
    ):
        raise ValueError("reviewed future preflight budget is not exact")
    if {item.get("family_id") for item in config["reviewed_specializers"]} != {
        AETHER,
        G3,
    }:
        raise ValueError("reviewed future specializer registry is incomplete")
    adapters = {
        (item.get("family_id"), item.get("formal_check_id")) for item in config["reviewed_adapters"]
    }
    if adapters != EXPECTED_ADAPTERS or len(config["reviewed_adapters"]) != len(EXPECTED_ADAPTERS):
        raise ValueError("reviewed future adapter registry is incomplete")
    if config["g3_center_calibration"] != {
        "X_phi": "1/2",
        "hessian_covariant": "zero",
        "G4": "1/2",
        "G4_X": "0",
        "BSSN_m": "1",
        "BSSN_sigma": "1",
    }:
        raise ValueError("G3 center calibration changed")


def _resolve(root: Path, descriptor: dict[str, Any], label: str) -> tuple[Any, Path]:
    required = {"entrypoint", "source_path", "source_file_sha256"}
    if not required <= set(descriptor):
        raise ValueError(f"{label} descriptor fields are incomplete")
    source = _bound_path(
        root,
        {
            "path": descriptor["source_path"],
            "file_sha256": descriptor["source_file_sha256"],
        },
        f"{label} source",
    )
    module_name, separator, attribute = descriptor["entrypoint"].partition(":")
    if not separator:
        raise ValueError(f"{label} entrypoint must use module:function")
    callback = getattr(importlib.import_module(module_name), attribute, None)
    callback_source = (
        inspect.getsourcefile(inspect.unwrap(callback)) if callable(callback) else None
    )
    if (
        not callable(callback)
        or callback_source is None
        or Path(callback_source).resolve() != source
    ):
        raise ValueError(f"{label} callback is not defined by its bound source")
    return callback, source


def _replay_adapters(
    config: dict[str, Any], root: Path, report: dict[str, Any]
) -> tuple[dict[str, list[dict[str, Any]]], str]:
    report_checks = {item["name"]: item for item in report.get("checks", [])}
    by_family: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    invocation_count = 0
    for descriptor in config["reviewed_adapters"]:
        if set(descriptor) != {
            "family_id",
            "formal_check_id",
            "entrypoint",
            "return_kind",
            "source_path",
            "source_file_sha256",
            "evidence_sha256",
            "applicability_contract",
        }:
            raise ValueError("reviewed formal adapter descriptor fields are invalid")
        callback, _ = _resolve(root, descriptor, "reviewed formal adapter")
        returned = callback()
        invocation_count += 1
        if descriptor["return_kind"] == "tuple":
            passed, evidence = returned
        elif descriptor["return_kind"] == "dict":
            evidence = returned
            passed = evidence.get("passed") is True
        else:
            raise ValueError("reviewed formal adapter return kind is unsupported")
        check = report_checks.get(descriptor["formal_check_id"])
        if (
            passed is not True
            or not isinstance(evidence, dict)
            or not isinstance(check, dict)
            or check.get("status") != "pass"
            or evidence != check.get("evidence")
            or _sha(evidence) != descriptor["evidence_sha256"]
        ):
            raise ValueError(
                f"reviewed formal adapter replay mismatch: {descriptor['formal_check_id']}"
            )
        body = {
            "formal_check_id": descriptor["formal_check_id"],
            "entrypoint": descriptor["entrypoint"],
            "evidence_sha256": descriptor["evidence_sha256"],
            "applicability_contract": descriptor["applicability_contract"],
            "status": "pass",
        }
        by_family[descriptor["family_id"]].append({**body, "content_sha256": _sha(body)})
    if invocation_count != int(config["budget"]["maximum_adapter_invocations"]):
        raise ValueError("reviewed formal adapter invocation count changed")
    stable = {
        family: sorted(items, key=lambda item: item["formal_check_id"])
        for family, items in sorted(by_family.items())
    }
    return stable, _sha(stable)


def _resolve_specializers(config: dict[str, Any], root: Path) -> tuple[dict[str, Any], str]:
    callbacks = {}
    registry = []
    for descriptor in config["reviewed_specializers"]:
        if set(descriptor) != {
            "family_id",
            "entrypoint",
            "source_path",
            "source_file_sha256",
            "applicability_contract",
        }:
            raise ValueError("reviewed specializer descriptor fields are invalid")
        callback, _ = _resolve(root, descriptor, "reviewed specializer")
        callbacks[descriptor["family_id"]] = callback
        registry.append(
            {
                "family_id": descriptor["family_id"],
                "entrypoint": descriptor["entrypoint"],
                "source_file_sha256": descriptor["source_file_sha256"],
                "applicability_contract": descriptor["applicability_contract"],
            }
        )
    return callbacks, _sha(sorted(registry, key=lambda item: item["family_id"]))


def _validate_source_status(
    status: dict[str, Any],
    source_config: dict[str, Any],
    chunk: dict[str, Any],
    compilation: dict[str, Any],
) -> None:
    if (
        status.get("schema_version") != "sigma-scalable-future-parameter-compilation-status-1.0"
        or status.get("immutable_config_sha256") != _sha(source_config)
        or status.get("input_cell_count") != 32
        or status.get("disposition_counts")
        != {"admitted_new_candidate": 19, "deduplicated_existing_candidate": 13}
        or status.get("future_chunk_content_sha256") != chunk["content_sha256"]
        or status.get("compilation_result_content_sha256") != compilation["content_sha256"]
        or status.get("receipt_registry_root_sha256") != compilation["receipt_registry_root_sha256"]
        or status.get("next_blocker") != "reviewed_formal_preflight_not_run_for_new_candidates"
        or status.get("data_eligibility") != ELIGIBILITY
        or status.get("paid_llm_spend_usd") != 0.0
    ):
        raise ValueError("future source status does not bind the exact 32/19/13 campaign")


def _candidate_action(
    cell: dict[str, Any], family: dict[str, Any], chunk: dict[str, Any]
) -> dict[str, Any]:
    pseudo_seed = {
        "seed_id": cell["parameter_cell_id"],
        "seed_lineage_sha256": cell["parameter_cell_lineage_sha256"],
        "family_id": cell["family_id"],
        "family_lineage_sha256": cell["family_lineage_sha256"],
        "theory_contract": cell["theory_contract"],
        "operator_atoms": cell["operator_atoms"],
        "parameters": cell["parameters"],
    }
    return _compile_action_ir(
        pseudo_seed,
        family,
        {
            "future_manifest_chunk_content_sha256": chunk["content_sha256"],
            "parameter_cell_registry_root_sha256": chunk["parameter_cell_registry_root_sha256"],
        },
    )


def _aether_record(
    receipt: dict[str, Any],
    cell: dict[str, Any],
    action: dict[str, Any],
    specializer: Any,
    adapters: list[dict[str, Any]],
) -> dict[str, Any]:
    if (
        [item["atom"] for item in action["operators"]] != EXPECTED_OPERATORS[AETHER]
        or action.get("parameters") != cell["parameters"]
        or cell.get("domain_contract")
        != "bounded_coefficients_only; formal stability remains unresolved"
    ):
        raise ValueError("future Aether action/domain specialization is not applicable")
    specialization = specializer(cell["parameters"])
    if (
        specialization.get("method") != "exact_rational_candidate_specialization"
        or specialization.get("parameters") != cell["parameters"]
        or specialization.get("adm_aligned_regular") is not True
    ):
        raise ValueError("reviewed Aether specialization replay failed")
    principal = specialization["principal_and_linear_mode_domain_pass"] is True
    decision = "pass" if principal else "reject"
    blocker = None if principal else "nonpositive_spin0_principal_numerator_c123"
    gates = {
        "receipt_action_and_lineage_binding": "pass",
        "reviewed_covariant_variation_noether": "pass",
        "exact_rational_action_domain_specialization": "pass",
        "adm_aligned_regularity": "pass",
        "principal_and_linear_mode_necessary_condition": decision,
        "observational_data_seal": "pass",
    }
    return {
        "reviewed_adapter_evidence": adapters,
        "exact_specialization": specialization,
        "gate_ledger": gates,
        "decision": decision,
        "first_blocker": blocker,
        "preflight_pass_scope": (
            "candidate_specific_Aether_formal_queue_only; no global energy or theory pass"
            if principal
            else None
        ),
        "next_required_formal_stage": (
            "candidate_specific_Aether_ADM_twist_constraint_and_global_energy_campaign"
            if principal
            else None
        ),
    }


def _g3_record(
    receipt: dict[str, Any],
    cell: dict[str, Any],
    action: dict[str, Any],
    specializer: Any,
    adapters: list[dict[str, Any]],
    center: dict[str, str],
) -> dict[str, Any]:
    if (
        [item["atom"] for item in action["operators"]] != EXPECTED_OPERATORS[G3]
        or action.get("parameters") != cell["parameters"]
        or cell.get("domain_contract")
        != "weak derivative cell only; common-cone proof remains unresolved"
    ):
        raise ValueError("future G3 action/domain specialization is not applicable")
    match = re.fullmatch(r"\((\d+/\d+)\)\*X_phi", cell["parameters"]["G3"])
    if match is None or cell["parameters"].get("G2") != "X_phi":
        raise ValueError("future G3 function specialization is unsupported")
    certificate = specializer(
        {
            "parameters": cell["parameters"],
            "g3_linear_x_coefficient": match.group(1),
        },
        center,
    )
    weak = certificate["declared_weak_cell_audit"]
    if (
        certificate["adm_primary"]["status"] != "pass"
        or certificate["center_principal_calibration"]["status"] != "pass_at_center_only"
        or weak.get("status") != "blocked"
        or any(
            weak.get(name) is not None
            for name in (
                "componentwise_gradient_bounds",
                "componentwise_hessian_bounds",
                "curvature_bounds",
                "frame_and_normalization_binding",
                "uniform_effective_metric_interval",
                "uniform_direction_sphere_cone_gap",
            )
        )
    ):
        raise ValueError("reviewed G3 domain-applicability replay changed")
    blocker = "componentwise_normalized_local_jet_box_and_uniform_cone_certificate_missing"
    gates = {
        "receipt_action_and_lineage_binding": "pass",
        "generic_covariant_G2_G3_variation_noether": "pass",
        "generic_adm_primary_prerequisite": "pass",
        "exact_center_principal_calibration": "pass_at_center_only",
        "declared_weak_cell_uniform_common_cone": "blocked",
        "observational_data_seal": "pass",
    }
    return {
        "reviewed_adapter_evidence": adapters,
        "exact_specialization": certificate,
        "gate_ledger": gates,
        "decision": "blocked",
        "first_blocker": blocker,
        "preflight_pass_scope": None,
        "next_required_formal_stage": (
            "register a candidate-bound componentwise normalized jet box, then run the "
            "reviewed interval common-cone adapter"
        ),
    }


def build_reviewed_future_parameter_formal_preflight(
    config: dict[str, Any], root: str | Path
) -> dict[str, Any]:
    """Replay the source chunk and evaluate exactly its 19 new action classes."""
    _validate_config(config)
    root = Path(root).resolve()
    _bound_path(root, config["campaign_implementation"], "reviewed campaign implementation")
    status = _bound_json(root, config["source_status"], "future source status", content=True)
    source_config = _bound_json(
        root, config["source_campaign_config"], "future source campaign config"
    )
    _bound_path(root, config["source_campaign_implementation"], "future source implementation")
    if source_config.get("compiler_implementation") != config["source_campaign_implementation"]:
        raise ValueError("future source implementation binding differs from source config")
    source_manifest = _bound_json(
        root, config["source_seed_manifest"], "source seed manifest", content=True
    )
    if source_config.get("source_seed_manifest") != config["source_seed_manifest"]:
        raise ValueError("future source seed binding differs from source config")
    report = _bound_json(root, config["formal_report"], "formal report")
    chunk = build_future_parameter_manifest_chunk(source_config, root)
    compilation = compile_future_parameter_chunk(source_config, root, chunk)
    _validate_source_status(status, source_config, chunk, compilation)
    adapters, adapter_root = _replay_adapters(config, root, report)
    specializers, specializer_root = _resolve_specializers(config, root)
    cells = {item["parameter_cell_id"]: item for item in chunk["parameter_cells"]}
    families = {
        item["family_id"]: item
        for item in source_manifest["typed_family_seeds"]
        if item["enabled_for_generation"]
    }
    new_receipts = [
        item for item in compilation["receipts"] if item["disposition"] == "admitted_new_candidate"
    ]
    if len(new_receipts) != int(config["budget"]["maximum_candidates"]):
        raise ValueError("future reviewed preflight candidate count changed")
    records = []
    decisions: Counter[str] = Counter()
    family_decisions: defaultdict[str, Counter[str]] = defaultdict(Counter)
    blockers: Counter[str] = Counter()
    for ordinal, receipt in enumerate(new_receipts):
        cell = cells[receipt["parameter_cell_id"]]
        family_id = receipt["family_id"]
        if family_id not in {AETHER, G3} or cell["family_id"] != family_id:
            raise ValueError("new candidate family lacks a reviewed preflight route")
        action = _candidate_action(cell, families[family_id], chunk)
        if (
            action["content_sha256"] != receipt["typed_action_ir_sha256"]
            or receipt["candidate_id"] != "G3A-" + receipt["action_density_equivalence_sha256"][:24]
            or receipt["decision"] != "pass"
            or receipt["data_eligibility"] != ELIGIBILITY
        ):
            raise ValueError("new candidate compilation receipt binding changed")
        if family_id == AETHER:
            evaluated = _aether_record(
                receipt,
                cell,
                action,
                specializers[AETHER],
                adapters[AETHER],
            )
        else:
            evaluated = _g3_record(
                receipt,
                cell,
                action,
                specializers[G3],
                adapters[G3],
                config["g3_center_calibration"],
            )
        record_body = {
            "ordinal": ordinal,
            "candidate_id": receipt["candidate_id"],
            "family_id": family_id,
            "parameter_cell_id": cell["parameter_cell_id"],
            "parameter_cell_lineage_sha256": cell["parameter_cell_lineage_sha256"],
            "compilation_receipt_sha256": receipt["content_sha256"],
            "typed_action_ir_sha256": action["content_sha256"],
            "action_density_equivalence_sha256": receipt["action_density_equivalence_sha256"],
            "parameters": cell["parameters"],
            "domain_contract": cell["domain_contract"],
            **evaluated,
            "expensive_candidate_specific_formal_run": False,
            "observational_data_opened": False,
            "data_eligibility": dict(ELIGIBILITY),
        }
        records.append({**record_body, "content_sha256": _sha(record_body)})
        decisions[evaluated["decision"]] += 1
        family_decisions[family_id][evaluated["decision"]] += 1
        if evaluated["first_blocker"] is not None:
            blockers[evaluated["first_blocker"]] += 1
    if decisions != Counter({"pass": 14, "reject": 2, "blocked": 3}):
        raise ValueError("future reviewed preflight decision partition changed")
    record_root = _sha(
        [
            [
                item["candidate_id"],
                item["typed_action_ir_sha256"],
                item["compilation_receipt_sha256"],
                item["content_sha256"],
            ]
            for item in records
        ]
    )
    provenance_body = {
        "source_status_content_sha256": status["content_sha256"],
        "source_campaign_config_sha256": _sha(source_config),
        "future_chunk_content_sha256": chunk["content_sha256"],
        "compilation_result_content_sha256": compilation["content_sha256"],
        "compilation_receipt_registry_root_sha256": compilation["receipt_registry_root_sha256"],
        "reviewed_adapter_evidence_root_sha256": adapter_root,
        "reviewed_specializer_registry_root_sha256": specializer_root,
        "candidate_record_registry_root_sha256": record_root,
        "data_eligibility": ELIGIBILITY,
    }
    body = {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": config["campaign_id"],
        "immutable_config_sha256": _sha(config),
        "source_status_binding": config["source_status"],
        "source_input_cell_count": 32,
        "source_new_candidate_count": 19,
        "source_deduplicated_candidate_count": 13,
        "candidate_count": len(records),
        "family_counts": dict(sorted(Counter(item["family_id"] for item in records).items())),
        "decision_counts": dict(sorted(decisions.items())),
        "family_decision_counts": {
            family: dict(sorted(counts.items()))
            for family, counts in sorted(family_decisions.items())
        },
        "first_blocker_counts": dict(sorted(blockers.items())),
        "candidate_record_registry_root_sha256": record_root,
        "candidate_records": records,
        "reviewed_adapter_invocation_count": len(config["reviewed_adapters"]),
        "reviewed_adapter_evidence_root_sha256": adapter_root,
        "reviewed_specializer_registry_root_sha256": specializer_root,
        "formal_preflight_completed": True,
        "full_candidate_specific_formal_completion_claimed": False,
        "observational_data_opened": False,
        "dark_matter_or_halo_inputs": False,
        "redshift_distance_inputs": False,
        "paid_llm_spend_usd": 0.0,
        "data_eligibility": dict(ELIGIBILITY),
        "promotion": {
            "eligible_for_candidate_specific_formal_queue": 14,
            "rejected_before_candidate_specific_formal_queue": 2,
            "blocked_pending_exact_domain_registration": 3,
            "automatic_downstream_enqueue_performed": False,
        },
        "provenance": {**provenance_body, "binding_sha256": _sha(provenance_body)},
        "interpretation": (
            "Fourteen exact Aether specializations pass only this reviewed cheap/formal "
            "preflight and may enter a separately bound candidate-specific formal queue. "
            "Two Aether specializations fail an exact principal-mode necessary condition. "
            "Three cubic-G3 actions remain blocked because qualitative derivative-ratio "
            "labels do not supply the componentwise normalized jet boxes required for a "
            "uniform common-cone theorem. No observation or family-label inference was used."
        ),
    }
    return {**body, "content_sha256": _sha(body)}


def publish_reviewed_future_parameter_formal_preflight(
    config: dict[str, Any], root: str | Path, target: str | Path
) -> dict[str, Any]:
    """Atomically publish an immutable portable campaign artifact."""
    artifact = build_reviewed_future_parameter_formal_preflight(config, root)
    target = Path(target).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if _load(target) != artifact:
            raise ValueError("refusing to replace divergent future formal-preflight artifact")
        return artifact
    temporary = target.with_suffix(target.suffix + ".tmp")
    encoded = (_canonical(artifact) + "\n").encode()
    with temporary.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    return artifact
