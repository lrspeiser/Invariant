"""Exact action-level formal follow-up for the scalable conformal-G4 action."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

from .covariant_grammar_v3_seed_compilation_campaign import _compile_action_ir
from .grammar_v3_formal_preflight_service import (
    PAYLOAD_SCHEMA,
    RESULT_SCHEMA,
    GrammarV3FormalPreflightService,
)
from .grammar_v3_parameter_cell_compilation_campaign import _action_density_key
from .grammar_v3_parameter_cell_manifest_campaign import iter_parameter_cells
from .promotion_orchestrator import ELIGIBILITY

SCHEMA_VERSION = "sigma-g4-scalable-action-formal-followup-1.0"
TARGET_FAMILY = "CONFORMAL_G4_PHI_SCALAR_TENSOR"
TARGET_SEED_ID = "G3-f9c598b70a77ea54009d8f18"


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_bound(root: Path, binding: dict[str, Any]) -> dict[str, Any]:
    path = (root / binding["path"]).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError("G4 follow-up binding escapes repository") from error
    if not path.is_file() or _file_sha(path) != binding["file_sha256"]:
        raise ValueError(f"G4 follow-up file binding changed: {binding['path']}")
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = binding.get("content_sha256")
    if expected is not None:
        body = {key: item for key, item in value.items() if key != "content_sha256"}
        if value.get("content_sha256") != expected or _sha(body) != expected:
            raise ValueError(f"G4 follow-up content binding changed: {binding['path']}")
    return value


def _resolve_preflight_callbacks(
    root: Path, preflight_config: dict[str, Any]
) -> tuple[dict[str, Any], str]:
    descriptors = preflight_config["reviewed_adapters"]
    callbacks: dict[str, Any] = {}
    registry = []
    for descriptor in descriptors:
        source = (root / descriptor["source_path"]).resolve()
        if not source.is_file() or _file_sha(source) != descriptor["source_file_sha256"]:
            raise ValueError("reviewed preflight callback source changed")
        module_name, separator, attribute = descriptor["callback"].partition(":")
        if not separator:
            raise ValueError("reviewed preflight callback entrypoint changed")
        callback = getattr(importlib.import_module(module_name), attribute, None)
        callback_source = (
            inspect.getsourcefile(inspect.unwrap(callback)) if callable(callback) else None
        )
        if (
            not callable(callback)
            or callback_source is None
            or Path(callback_source).resolve() != source
        ):
            raise ValueError("reviewed preflight callback provenance changed")
        callbacks[descriptor["family_id"]] = callback
        registry.append(
            {
                "family_id": descriptor["family_id"],
                "adapter_id": descriptor["adapter_id"],
                "callback": descriptor["callback"],
                "source_file_sha256": descriptor["source_file_sha256"],
                "state": "reviewed_bound",
            }
        )
    return callbacks, _sha(registry)


def _replay_preflight(
    root: Path,
    compilation: dict[str, Any],
    cell_manifest: dict[str, Any],
    seed_manifest: dict[str, Any],
    preflight_config: dict[str, Any],
    preflight_status: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    callbacks, callback_root = _resolve_preflight_callbacks(root, preflight_config)
    if callback_root != preflight_status["callback_registry_root_sha256"]:
        raise ValueError("formal-preflight callback registry changed")
    adapters = {
        family: GrammarV3FormalPreflightService._invoke_adapter(family, callback)
        for family, callback in callbacks.items()
    }
    families = {
        family["family_id"]: family
        for family in seed_manifest["typed_family_seeds"]
        if family["enabled_for_generation"]
    }
    manifest_binding = {
        "parameter_cell_manifest_content_sha256": cell_manifest["content_sha256"],
        "parameter_cell_registry_root_sha256": cell_manifest["parameter_cell_registry_root_sha256"],
    }
    seen: set[str] = set()
    detailed = []
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
            "schema_version": PAYLOAD_SCHEMA,
            "ordinal": len(detailed),
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
        adapter = adapters[cell["family_id"]]
        result_body = {
            "schema_version": RESULT_SCHEMA,
            "candidate_id": payload["candidate_id"],
            "typed_action_ir_sha256": payload["typed_action_ir_sha256"],
            "input_lineage_sha256": payload["input_lineage_sha256"],
            "callback_registry_root_sha256": callback_root,
            "receipt_binding_gate": "pass",
            "family_prerequisite_gate": adapter["decision"],
            "adapter_evidence": adapter,
            "decision": adapter["decision"],
            "blocker": (
                None if adapter["decision"] == "pass" else "family_prerequisite_not_passed"
            ),
            "expensive_adm_or_global_energy_run": False,
            "data_eligibility": {**ELIGIBILITY, "passed": True},
            "paid_llm_spend_usd": 0.0,
        }
        result_sha = _sha(result_body)
        simple = {
            "candidate_id": payload["candidate_id"],
            "typed_action_ir_sha256": payload["typed_action_ir_sha256"],
            "family_id": payload["family_id"],
            "state": "succeeded",
            "attempt": 1,
            "result_sha256": result_sha,
            "blocker": result_body["blocker"],
            "error_text": None,
        }
        detailed.append(
            {
                "cell": cell,
                "action_ir": action_ir,
                "payload": payload,
                "preflight_result": {**result_body, "content_sha256": result_sha},
                "preflight_record_sha256": _sha(simple),
            }
        )
        simplified.append(simple)
    if len(detailed) != 163:
        raise ValueError("formal-preflight unique candidate count changed")
    for chunk in preflight_status["chunks"]:
        start, stop = chunk["range"]["start"], chunk["range"]["stop"]
        if _sha(simplified[start:stop]) != chunk["record_root_sha256"]:
            raise ValueError("formal-preflight replay record root changed")
    targets = [item for item in detailed if item["cell"]["family_id"] == TARGET_FAMILY]
    if len(targets) != 1:
        raise ValueError("scalable conformal-G4 unique action count changed")
    aliases = [
        cell
        for cell in iter_parameter_cells(cell_manifest, seed_manifest)
        if cell["family_id"] == TARGET_FAMILY
    ]
    return targets[0], adapters[TARGET_FAMILY], {"cells": aliases, "count": len(aliases)}


def _density_projection(action_ir: dict[str, Any]) -> dict[str, Any]:
    parameters = action_ir["parameters"]
    if not {"G2", "G4"}.issubset(parameters):
        raise ValueError("conformal-G4 action parameters changed")
    return {
        "family_id": action_ir["family_id"],
        "family_lineage_sha256": action_ir["family_lineage_sha256"],
        "theory_contract": action_ir["theory_contract"],
        "fields": action_ir["fields"],
        "operators": action_ir["operators"],
        "action_parameters": {"G2": parameters["G2"], "G4": parameters["G4"]},
        "matter_coupling": action_ir["matter_coupling"],
    }


def build_g4_scalable_action_formal_followup(
    config: dict[str, Any], root: str | Path
) -> dict[str, Any]:
    root = Path(root).resolve()
    if config.get("data_eligibility") != ELIGIBILITY:
        raise ValueError("G4 scalable follow-up eligibility changed")
    if config.get("observational_authorization") is not False:
        raise ValueError("G4 scalable follow-up opened observations")
    source = config["source_bindings"]["campaign_source"]
    source_path = root / source["path"]
    if not source_path.is_file() or _file_sha(source_path) != source["file_sha256"]:
        raise ValueError("G4 scalable follow-up source changed")
    bindings = config["source_bindings"]
    inputs = {
        key: _load_bound(root, value) for key, value in bindings.items() if key != "campaign_source"
    }
    target, preflight_adapter, aliases = _replay_preflight(
        root,
        inputs["compilation_campaign"],
        inputs["parameter_cell_manifest"],
        inputs["source_seed_manifest"],
        inputs["formal_preflight_config"],
        inputs["formal_preflight_status"],
    )
    if target["preflight_result"]["decision"] != "blocked":
        raise ValueError("target conformal-G4 preflight decision changed")
    seed_record = next(
        (
            record
            for record in inputs["seed_compilation"]["candidate_records"]
            if record["seed_id"] == TARGET_SEED_ID
        ),
        None,
    )
    if seed_record is None:
        raise ValueError("reviewed conformal-G4 seed action is missing")
    formal_record = next(
        (
            record
            for record in inputs["reviewed_formal_pass"]["candidate_records"]
            if record["seed_id"] == TARGET_SEED_ID and record["family"] == "G4"
        ),
        None,
    )
    if formal_record is None or formal_record["decision"] != "pass":
        raise ValueError("reviewed conformal-G4 formal pass is missing")
    if formal_record["action_sha256"] != seed_record["typed_action_ir"]["content_sha256"]:
        raise ValueError("reviewed formal pass action binding changed")
    if {item["status"] for item in formal_record["gate_ledger"].values()} != {"pass"}:
        raise ValueError("reviewed conformal-G4 formal gate changed")
    scalable_action = target["action_ir"]
    seed_action = seed_record["typed_action_ir"]
    scalable_projection = _density_projection(scalable_action)
    seed_projection = _density_projection(seed_action)
    if scalable_projection != seed_projection:
        raise ValueError("scalable and reviewed G4 action densities are not exact-equivalent")
    scalable_limit = Fraction(scalable_action["parameters"]["phi_domain"].split("<=", 1)[1])
    seed_limit = Fraction(seed_action["parameters"]["phi_domain"].split("<=", 1)[1])
    if not 0 < scalable_limit <= seed_limit or seed_limit != 1:
        raise ValueError("scalable G4 domain is not a reviewed-domain restriction")
    alias_limits = {
        Fraction(cell["parameters"]["phi_domain"].split("<=", 1)[1]) for cell in aliases["cells"]
    }
    if aliases["count"] != 32 or max(alias_limits) != seed_limit:
        raise ValueError("G4 parameter-cell alias domain changed")
    projection_sha = _sha(scalable_projection)
    certificate_body = {
        "method": "exact_typed_density_projection_and_rational_domain_inclusion",
        "full_typed_action_hashes_equal": (
            scalable_action["content_sha256"] == seed_action["content_sha256"]
        ),
        "full_hash_difference_reason": (
            "candidate/lineage/manifest metadata and the representative phi-domain differ; "
            "the covariant density is compared separately"
        ),
        "scalable_action_sha256": scalable_action["content_sha256"],
        "reviewed_seed_action_sha256": seed_action["content_sha256"],
        "action_density_projection_sha256": projection_sha,
        "action_density_projection_equal": True,
        "operator_densities_equal": scalable_action["operators"] == seed_action["operators"],
        "action_parameters_equal": {
            key: scalable_action["parameters"][key] == seed_action["parameters"][key]
            for key in ("G2", "G4")
        },
        "universal_matter_coupling_equal": (
            scalable_action["matter_coupling"] == seed_action["matter_coupling"]
        ),
        "scalable_representative_domain": scalable_action["parameters"]["phi_domain"],
        "reviewed_seed_domain": seed_action["parameters"]["phi_domain"],
        "representative_domain_is_subset": True,
        "equivalent_parameter_cell_alias_count": aliases["count"],
        "all_alias_domains_inside_reviewed_domain": all(
            0 < value <= seed_limit for value in alias_limits
        ),
        "family_label_used_as_equivalence_evidence": False,
    }
    certificate = {**certificate_body, "content_sha256": _sha(certificate_body)}
    provenance_body = {
        "candidate_id": target["payload"]["candidate_id"],
        "action_density_equivalence_sha256": target["payload"]["action_density_equivalence_sha256"],
        "scalable_action_sha256": scalable_action["content_sha256"],
        "preflight_input_lineage_sha256": target["payload"]["input_lineage_sha256"],
        "preflight_result_sha256": target["preflight_result"]["content_sha256"],
        "preflight_record_sha256": target["preflight_record_sha256"],
        "preflight_adapter_sha256": preflight_adapter["content_sha256"],
        "reviewed_seed_action_sha256": seed_action["content_sha256"],
        "reviewed_formal_pass_record_sha256": _sha(formal_record),
        "equivalence_certificate_sha256": certificate["content_sha256"],
        "data_eligibility": ELIGIBILITY,
    }
    body = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": config["campaign_id"],
        "source_bindings": bindings,
        "candidate_count": 1,
        "candidate_id": target["payload"]["candidate_id"],
        "representative_cell_id": target["cell"]["parameter_cell_id"],
        "equivalent_parameter_cell_alias_count": aliases["count"],
        "preflight_decision": "blocked",
        "preflight_blocker": target["preflight_result"]["blocker"],
        "formal_followup_decision": "pass",
        "decision_counts": {"pass": 1},
        "formal_pass_count": 1,
        "necessary_condition_rejection_count": 0,
        "solar_bundle_count": 0,
        "equivalence_certificate": certificate,
        "gate_ledger": {
            "candidate_action_and_preflight_receipt_binding": {"status": "pass"},
            "full_typed_action_hash_identity": {
                "status": "not_equal_expected",
                "used_as_equivalence_proof": False,
            },
            "exact_covariant_density_equivalence": {"status": "pass"},
            "reviewed_domain_restriction": {"status": "pass"},
            "reviewed_formal_pass_action_binding": {"status": "pass"},
            "formal_prerequisite_completion": {"status": "pass"},
            "solar_and_observations": {"status": "sealed"},
        },
        "provenance": {**provenance_body, "binding_sha256": _sha(provenance_body)},
        "observational_data_opened": False,
        "dark_matter_or_halo_inputs": False,
        "redshift_distance_inputs": False,
        "paid_llm_spend_usd": 0.0,
        "data_eligibility": dict(ELIGIBILITY),
        "interpretation": (
            "The scalable candidate and reviewed seed have different full typed-IR hashes, "
            "but their exact covariant density projections are identical and the scalable "
            "representative domain is a rational subset of the reviewed domain. The reviewed "
            "candidate-specific formal pass therefore transfers; Solar and observations remain sealed."
        ),
    }
    return {**body, "content_sha256": _sha(body)}
