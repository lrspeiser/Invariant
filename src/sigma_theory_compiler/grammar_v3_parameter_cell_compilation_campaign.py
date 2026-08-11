from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .covariant_grammar_v3_seed_compilation_campaign import (
    FORBIDDEN_TOKENS,
    OPERATOR_LIBRARY,
    _compile_action_ir,
)
from .grammar_v3_parameter_cell_manifest_campaign import iter_parameter_cells
from .promotion_orchestrator import ELIGIBILITY

CONFIG_SCHEMA = "sigma-grammar-v3-parameter-cell-compilation-config-1.0"
CAMPAIGN_SCHEMA = "sigma-grammar-v3-parameter-cell-compilation-campaign-1.0"
RECEIPT_SCHEMA = "sigma-grammar-v3-parameter-cell-compilation-receipt-1.0"


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


def _bound(root: Path, binding: dict[str, Any], label: str) -> tuple[Path, dict[str, Any]]:
    if not {"path", "file_sha256"}.issubset(binding) or set(binding) - {
        "path",
        "file_sha256",
        "content_sha256",
        "required_gates",
    }:
        raise ValueError(f"{label} binding fields are invalid")
    path = (root / binding["path"]).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} path escapes repository") from error
    if not path.is_file() or _file_sha(path) != binding["file_sha256"]:
        raise ValueError(f"{label} file hash mismatch")
    value = _load(path)
    if "content_sha256" in binding:
        body = {key: item for key, item in value.items() if key != "content_sha256"}
        if value.get("content_sha256") != binding["content_sha256"] or _sha(body) != binding[
            "content_sha256"
        ]:
            raise ValueError(f"{label} content hash mismatch")
    return path, value


def _validate_config(config: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "campaign_id",
        "parameter_cell_manifest",
        "source_seed_manifest",
        "compiler_semantics",
        "field_contract",
        "action_policy",
        "finite_budget",
        "negative_controls",
        "data_eligibility",
        "external_paid_llm_calls",
    }
    if set(config) != required or config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("grammar-v3 cell compilation config is invalid")
    if config.get("data_eligibility") != ELIGIBILITY:
        raise ValueError("grammar-v3 cell compilation eligibility is not fail-closed")
    if config.get("external_paid_llm_calls") is not False:
        raise ValueError("grammar-v3 cell compilation enabled paid LLM calls")
    budget = config.get("finite_budget", {})
    if set(budget) != {"maximum_cells", "chunk_size", "maximum_action_terms"} or (
        int(budget["maximum_cells"]) != 256
        or not 1 <= int(budget["chunk_size"]) <= 64
        or not 1 <= int(budget["maximum_action_terms"]) <= 8
    ):
        raise ValueError("grammar-v3 cell compilation budget is invalid")


def _action_density_key(cell: dict[str, Any]) -> dict[str, Any]:
    parameters = cell["parameters"]
    family = cell["family_id"]
    if family == "AETHER_K1234_PARAMETER_CELL":
        action_parameters = parameters
    elif family == "KESSENCE_G2_CONVEX":
        action_parameters = {"G2": parameters["G2"]}
    elif family == "CUBIC_HORNDESKI_G3_WEAK_CELL":
        action_parameters = {"G2": parameters["G2"], "G3": parameters["G3"]}
    elif family == "CONFORMAL_G4_PHI_SCALAR_TENSOR":
        action_parameters = {"G2": parameters["G2"], "G4": parameters["G4"]}
    else:
        raise ValueError("unreviewed family reached action equivalence")
    return {
        "family_id": family,
        "family_lineage_sha256": cell["family_lineage_sha256"],
        "theory_contract": cell["theory_contract"],
        "operator_atoms": cell["operator_atoms"],
        "action_parameters": action_parameters,
        "universal_matter_metric": "g_mu_nu",
    }


def structural_policy_gates(
    action_ir: dict[str, Any],
    cell: dict[str, Any],
    family: dict[str, Any],
    field_contract: dict[str, Any],
    action_policy_descriptor: dict[str, Any],
    action_policy_document: dict[str, Any],
    maximum_action_terms: int,
) -> dict[str, Any]:
    allowed_fields = {item["id"] for item in field_contract["fields"]}
    generator_fields = set(family["field_ids"])
    operator_atoms = [item["atom"] for item in action_ir["operators"]]
    text = _canonical(action_ir)
    dynamical = {
        item["id"]
        for item in field_contract["fields"]
        if item.get("dynamical") is True
    }
    extra_dynamical = (generator_fields & dynamical) - {"g_mu_nu"}
    gates = {
        "cell_lineage": (
            action_ir.get("seed_lineage_sha256")
            == cell["parameter_cell_lineage_sha256"]
        ),
        "typed_family": (
            action_ir.get("family_id") == cell["family_id"]
            and action_ir.get("family_lineage_sha256") == cell["family_lineage_sha256"]
        ),
        "field_contract": (
            generator_fields.issubset(allowed_fields)
            and "J_b_mu" not in generator_fields
            and "psi_m" not in generator_fields
            and action_ir.get("fields") == family["field_ids"]
        ),
        "registered_operators": (
            operator_atoms == cell["operator_atoms"]
            and all(atom in OPERATOR_LIBRARY for atom in operator_atoms)
            and len(operator_atoms) <= maximum_action_terms
        ),
        "action_policy_bounds": (
            len(operator_atoms)
            <= min(
                maximum_action_terms,
                int(action_policy_document["bounds"]["maximum_terms"]),
            )
            and len(extra_dynamical)
            <= int(action_policy_document["bounds"]["maximum_extra_dynamical_fields"])
        ),
        "universal_matter_coupling": action_ir.get("matter_coupling")
        == {"metric": "g_mu_nu", "universal": True},
        "forbidden_tokens_absent": not any(token in text for token in FORBIDDEN_TOKENS),
        "data_eligibility": (
            action_ir.get("data_eligibility") == ELIGIBILITY
            and cell.get("data_eligibility") == ELIGIBILITY
        ),
    }
    required = set(action_policy_descriptor["required_gates"])
    if set(gates) != required:
        raise ValueError("action-policy required gate set changed")
    return gates


def _negative_audit(
    controls: list[dict[str, Any]],
    representative_ir: dict[str, Any],
    representative_cell: dict[str, Any],
    representative_family: dict[str, Any],
    field_contract: dict[str, Any],
    policy_descriptor: dict[str, Any],
    policy_document: dict[str, Any],
    maximum_terms: int,
) -> list[dict[str, Any]]:
    results = []
    for control in controls:
        mutated_ir = json.loads(_canonical(representative_ir))
        mutated_cell = json.loads(_canonical(representative_cell))
        mutated_family = json.loads(_canonical(representative_family))
        kind = control["kind"]
        if kind == "nonuniversal_matter":
            mutated_ir["matter_coupling"] = {"metric": "g_mu_nu", "universal": False}
        elif kind == "forbidden_field":
            mutated_family["field_ids"].append("J_b_mu")
            mutated_ir["fields"].append("J_b_mu")
        elif kind == "unsupported_operator":
            mutated_ir["operators"].append({"atom": "UNREVIEWED_OP"})
        elif kind == "lineage_tamper":
            mutated_ir["seed_lineage_sha256"] = "0" * 64
        elif kind == "forbidden_data":
            mutated_cell["data_eligibility"]["redshift_distance_inputs"] = True
        else:
            raise ValueError("unknown compilation negative-control kind")
        gates = structural_policy_gates(
            mutated_ir,
            mutated_cell,
            mutated_family,
            field_contract,
            policy_descriptor,
            policy_document,
            maximum_terms,
        )
        failed = sorted(name for name, passed in gates.items() if not passed)
        if failed != control["expected_failed_gates"]:
            raise ValueError("compilation negative control outcome changed")
        body = {
            "control_id": control["control_id"],
            "kind": kind,
            "decision": "reject",
            "failed_gates": failed,
        }
        results.append({**body, "content_sha256": _sha(body)})
    return results


def build_parameter_cell_compilation_campaign(
    config: dict[str, Any], root: str | Path
) -> dict[str, Any]:
    _validate_config(config)
    root = Path(root).resolve()
    _, manifest = _bound(root, config["parameter_cell_manifest"], "parameter-cell manifest")
    _, source = _bound(root, config["source_seed_manifest"], "source seed manifest")
    compiler_path = (root / config["compiler_semantics"]["path"]).resolve()
    if not compiler_path.is_file() or _file_sha(compiler_path) != config["compiler_semantics"][
        "file_sha256"
    ]:
        raise ValueError("compiler semantics file hash mismatch")
    _, field_contract = _bound(root, config["field_contract"], "field contract")
    _, action_policy_document = _bound(root, config["action_policy"], "action policy")
    cells = list(iter_parameter_cells(manifest, source))
    if len(cells) != int(config["finite_budget"]["maximum_cells"]):
        raise ValueError("parameter-cell compilation input exceeds its exact finite cap")
    families = {
        family["family_id"]: family
        for family in source["typed_family_seeds"]
        if family["enabled_for_generation"]
    }
    manifest_binding = {
        "parameter_cell_manifest_content_sha256": manifest["content_sha256"],
        "parameter_cell_registry_root_sha256": manifest[
            "parameter_cell_registry_root_sha256"
        ],
    }
    representative_by_equivalence: dict[str, dict[str, Any]] = {}
    receipts = []
    family_inputs: Counter[str] = Counter()
    family_unique: Counter[str] = Counter()
    family_duplicates: Counter[str] = Counter()
    gate_pass_counts: Counter[str] = Counter()
    for cell in cells:
        family = families[cell["family_id"]]
        pseudo_seed = {
            "seed_id": cell["parameter_cell_id"],
            "seed_lineage_sha256": cell["parameter_cell_lineage_sha256"],
            "family_id": cell["family_id"],
            "family_lineage_sha256": cell["family_lineage_sha256"],
            "theory_contract": cell["theory_contract"],
            "operator_atoms": cell["operator_atoms"],
            "parameters": cell["parameters"],
        }
        action_ir = _compile_action_ir(pseudo_seed, family, manifest_binding)
        gates = structural_policy_gates(
            action_ir,
            cell,
            family,
            field_contract,
            config["action_policy"],
            action_policy_document,
            int(config["finite_budget"]["maximum_action_terms"]),
        )
        failed = sorted(name for name, passed in gates.items() if not passed)
        if failed:
            raise ValueError("generated reviewed cell failed structural policy: " + ",".join(failed))
        gate_pass_counts.update(gates)
        equivalence_sha = _sha(_action_density_key(cell))
        representative = representative_by_equivalence.get(equivalence_sha)
        family_inputs[cell["family_id"]] += 1
        if representative is None:
            candidate_id = "G3A-" + equivalence_sha[:24]
            representative = {
                "candidate_id": candidate_id,
                "representative_cell_id": cell["parameter_cell_id"],
                "representative_action_ir_sha256": action_ir["content_sha256"],
            }
            representative_by_equivalence[equivalence_sha] = representative
            disposition = "compiled_representative"
            family_unique[cell["family_id"]] += 1
        else:
            candidate_id = representative["candidate_id"]
            disposition = "deduplicated_equivalent"
            family_duplicates[cell["family_id"]] += 1
        receipt_body = {
            "schema_version": RECEIPT_SCHEMA,
            "ordinal": cell["ordinal"],
            "parameter_cell_id": cell["parameter_cell_id"],
            "parameter_cell_lineage_sha256": cell["parameter_cell_lineage_sha256"],
            "family_id": cell["family_id"],
            "typed_action_ir_sha256": action_ir["content_sha256"],
            "action_density_equivalence_sha256": equivalence_sha,
            "candidate_id": candidate_id,
            "representative_cell_id": representative["representative_cell_id"],
            "disposition": disposition,
            "structural_gate_root_sha256": _sha(gates),
            "decision": "pass" if disposition == "compiled_representative" else "deduplicated",
            "data_eligibility": dict(ELIGIBILITY),
        }
        receipts.append({**receipt_body, "content_sha256": _sha(receipt_body)})
    unique = [receipt for receipt in receipts if receipt["disposition"] == "compiled_representative"]
    duplicates = [receipt for receipt in receipts if receipt["disposition"] == "deduplicated_equivalent"]
    chunk_size = int(config["finite_budget"]["chunk_size"])
    chunks = []
    for start in range(0, len(receipts), chunk_size):
        selected = receipts[start : start + chunk_size]
        body = {
            "chunk_index": len(chunks),
            "range": {"start": start, "stop": start + len(selected)},
            "receipt_root_sha256": _sha(
                [[receipt["parameter_cell_id"], receipt["content_sha256"]] for receipt in selected]
            ),
            "compiled_representative_count": sum(
                receipt["disposition"] == "compiled_representative" for receipt in selected
            ),
            "deduplicated_equivalent_count": sum(
                receipt["disposition"] == "deduplicated_equivalent" for receipt in selected
            ),
        }
        chunks.append({**body, "content_sha256": _sha(body)})
    negatives = _negative_audit(
        config["negative_controls"],
        _compile_action_ir(
            {
                "seed_id": cells[0]["parameter_cell_id"],
                "seed_lineage_sha256": cells[0]["parameter_cell_lineage_sha256"],
                "family_id": cells[0]["family_id"],
                "family_lineage_sha256": cells[0]["family_lineage_sha256"],
                "theory_contract": cells[0]["theory_contract"],
                "operator_atoms": cells[0]["operator_atoms"],
                "parameters": cells[0]["parameters"],
            },
            families[cells[0]["family_id"]],
            manifest_binding,
        ),
        cells[0],
        families[cells[0]["family_id"]],
        field_contract,
        config["action_policy"],
        action_policy_document,
        int(config["finite_budget"]["maximum_action_terms"]),
    )
    family_audit = {
        family_id: {
            "input_cells": family_inputs[family_id],
            "unique_actions": family_unique[family_id],
            "equivalent_duplicates": family_duplicates[family_id],
        }
        for family_id in sorted(family_inputs)
    }
    sample_receipts = []
    for family_id in sorted(family_inputs):
        selected = [receipt for receipt in receipts if receipt["family_id"] == family_id]
        for receipt in (selected[0], selected[-1]):
            sample_receipts.append(
                {
                    key: receipt[key]
                    for key in (
                        "parameter_cell_id",
                        "family_id",
                        "candidate_id",
                        "typed_action_ir_sha256",
                        "action_density_equivalence_sha256",
                        "representative_cell_id",
                        "disposition",
                        "content_sha256",
                    )
                }
            )
    body = {
        "schema_version": CAMPAIGN_SCHEMA,
        "campaign_id": config["campaign_id"],
        "parameter_cell_manifest_binding": config["parameter_cell_manifest"],
        "parameter_cell_registry_root_sha256": manifest[
            "parameter_cell_registry_root_sha256"
        ],
        "compiler_semantics_binding": config["compiler_semantics"],
        "field_contract_binding": config["field_contract"],
        "action_policy_binding": config["action_policy"],
        "input_parameter_cell_count": len(receipts),
        "compiled_action_ir_count": len(receipts),
        "unique_candidate_count": len(unique),
        "equivalent_duplicate_count": len(duplicates),
        "candidate_decision_counts": {"pass": len(unique), "reject": 0, "blocked": 0},
        "cell_disposition_counts": {
            "compiled_representative": len(unique),
            "deduplicated_equivalent": len(duplicates),
        },
        "family_audit": family_audit,
        "structural_gate_pass_counts": dict(sorted(gate_pass_counts.items())),
        "receipt_registry_root_sha256": _sha(
            [[receipt["parameter_cell_id"], receipt["content_sha256"]] for receipt in receipts]
        ),
        "unique_candidate_registry_root_sha256": _sha(
            [
                [
                    receipt["candidate_id"],
                    receipt["action_density_equivalence_sha256"],
                    receipt["typed_action_ir_sha256"],
                ]
                for receipt in unique
            ]
        ),
        "equivalent_duplicate_registry_root_sha256": _sha(
            [
                [
                    receipt["parameter_cell_id"],
                    receipt["candidate_id"],
                    receipt["representative_cell_id"],
                ]
                for receipt in duplicates
            ]
        ),
        "chunks": chunks,
        "chunk_registry_root_sha256": _sha(chunks),
        "sample_receipts": sample_receipts,
        "negative_control_results": negatives,
        "negative_control_counts": {"reject": len(negatives)},
        "expensive_formal_campaign_run": False,
        "formal_decision_counts": {},
        "next_execution_hook": {
            "chunk_count": len(chunks),
            "chunk_size": chunk_size,
            "resume_key": "chunk.content_sha256 + receipt_registry_root_sha256",
            "required_adapter": (
                "reviewed family-specific ADM/formal callback bound to candidate_id and "
                "typed_action_ir_sha256; missing adapters block"
            ),
        },
        "observational_data_opened": False,
        "data_eligibility": dict(ELIGIBILITY),
        "paid_llm_spend_usd": 0.0,
    }
    return {**body, "content_sha256": _sha(body)}
