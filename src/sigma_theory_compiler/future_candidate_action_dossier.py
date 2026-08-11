from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .covariant_grammar_v3_seed_compilation_campaign import _compile_action_ir
from .promotion_orchestrator import ELIGIBILITY
from .scalable_future_parameter_chunk_campaign import (
    build_future_parameter_manifest_chunk,
    compile_future_parameter_chunk,
)

CONFIG_SCHEMA = "sigma-future-candidate-action-dossier-config-1.0"
ARTIFACT_SCHEMA = "sigma-future-candidate-action-dossier-1.0"
DOSSIER_SCHEMA = "sigma-future-candidate-action-dossier-record-1.0"


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_bound(root: Path, binding: dict[str, Any], label: str) -> dict[str, Any]:
    path = (root / binding["path"]).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} path escapes repository") from error
    if not path.is_file() or _file_sha(path) != binding["file_sha256"]:
        raise ValueError(f"{label} file hash mismatch")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{label} must contain an object")
    expected = binding.get("content_sha256")
    if expected is not None:
        body = {key: item for key, item in value.items() if key != "content_sha256"}
        if value.get("content_sha256") != expected or _sha(body) != expected:
            raise ValueError(f"{label} content hash mismatch")
    return value


def _validate_config(config: dict[str, Any]) -> None:
    if (
        set(config)
        != {
            "schema_version",
            "campaign_id",
            "source_bindings",
            "budget",
            "data_eligibility",
            "observational_authorization",
            "external_paid_llm_calls",
        }
        or config.get("schema_version") != CONFIG_SCHEMA
        or set(config.get("source_bindings", {}))
        != {
            "future_campaign_config",
            "preflight",
            "aether_followup",
            "g3_followup",
        }
        or config.get("data_eligibility") != ELIGIBILITY
        or config.get("observational_authorization") is not False
        or config.get("external_paid_llm_calls") is not False
    ):
        raise ValueError("future candidate action dossier config is invalid")
    budget = config.get("budget", {})
    if (
        set(budget) != {"maximum_candidates", "maximum_output_bytes", "maximum_paid_llm_spend_usd"}
        or int(budget["maximum_candidates"]) != 19
        or not 128 * 1024 <= int(budget["maximum_output_bytes"]) <= 8 * 1024 * 1024
        or float(budget["maximum_paid_llm_spend_usd"]) != 0.0
    ):
        raise ValueError("future candidate action dossier budget is invalid")


def _action_map(
    root: Path, campaign_config: dict[str, Any]
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], dict[str, Any]]:
    chunk = build_future_parameter_manifest_chunk(campaign_config, root)
    compilation = compile_future_parameter_chunk(campaign_config, root, chunk)
    seed_manifest = _load_bound(
        root, campaign_config["source_seed_manifest"], "future seed manifest"
    )
    families = {
        family["family_id"]: family
        for family in seed_manifest["typed_family_seeds"]
        if family["enabled_for_generation"]
    }
    receipts = {receipt["parameter_cell_id"]: receipt for receipt in compilation["receipts"]}
    manifest_binding = {
        "future_manifest_chunk_content_sha256": chunk["content_sha256"],
        "parameter_cell_registry_root_sha256": chunk["parameter_cell_registry_root_sha256"],
    }
    output: dict[str, dict[str, Any]] = {}
    for cell in chunk["parameter_cells"]:
        receipt = receipts[cell["parameter_cell_id"]]
        if receipt["disposition"] != "admitted_new_candidate":
            continue
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
        action = _compile_action_ir(pseudo_seed, family, manifest_binding)
        if (
            action["content_sha256"] != receipt["typed_action_ir_sha256"]
            or action["family_id"] != receipt["family_id"]
            or receipt["candidate_id"] in output
        ):
            raise ValueError("future action recompilation or uniqueness changed")
        output[receipt["candidate_id"]] = {
            "action": action,
            "cell": cell,
            "receipt": receipt,
        }
    if len(output) != 19:
        raise ValueError("future action recompilation did not yield exactly 19 candidates")
    return output, chunk, compilation


def _display_action(action: dict[str, Any]) -> dict[str, Any]:
    densities = [operator["density"] for operator in action["operators"]]
    body = {
        "display_kind": "verbatim_ordered_covariant_density_concatenation",
        "display_text": "S = integral d^4x ["
        + " + ".join(f"({density})" for density in densities)
        + "]",
        "scope": (
            "Display-only concatenation of exact compiler-emitted densities. Family labels "
            "supply no action terms and no field equation is inferred."
        ),
    }
    return {**body, "content_sha256": _sha(body)}


def _node(
    node_id: str,
    status: str,
    scope: str,
    evidence: dict[str, Any],
    **extra: Any,
) -> dict[str, Any]:
    body = {
        "node_id": node_id,
        "status": status,
        "scope": scope,
        "evidence": evidence,
        **extra,
    }
    return {**body, "content_sha256": _sha(body)}


def build_future_candidate_action_dossier(
    config: dict[str, Any], root: str | Path
) -> dict[str, Any]:
    _validate_config(config)
    root = Path(root).resolve()
    bindings = config["source_bindings"]
    campaign_config = _load_bound(
        root, bindings["future_campaign_config"], "future campaign config"
    )
    preflight = _load_bound(root, bindings["preflight"], "future preflight")
    aether = _load_bound(root, bindings["aether_followup"], "future Aether follow-up")
    g3 = _load_bound(root, bindings["g3_followup"], "future G3 follow-up")
    action_map, chunk, compilation = _action_map(root, campaign_config)
    preflight_records = {
        record["candidate_id"]: record for record in preflight["candidate_records"]
    }
    aether_records = {record["candidate_id"]: record for record in aether["candidate_records"]}
    g3_records = {record["candidate_id"]: record for record in g3["candidate_records"]}
    dossiers = []
    for candidate_id in sorted(action_map):
        action_packet = action_map[candidate_id]
        action = action_packet["action"]
        preflight_record = preflight_records.get(candidate_id)
        if (
            preflight_record is None
            or preflight_record["typed_action_ir_sha256"] != action["content_sha256"]
        ):
            raise ValueError("future preflight action binding changed")
        if preflight_record["decision"] == "reject":
            decision = "reject"
            blocker = preflight_record["first_blocker"]
            formal_hash = preflight_record["content_sha256"]
            formal_source = "preflight"
            formal_scope = (
                "An exact principal-mode necessary condition rejects this action before "
                "candidate-specific formal execution."
            )
        elif preflight_record["family_id"] == "AETHER_K1234_PARAMETER_CELL":
            followup = aether_records.get(candidate_id)
            if (
                followup is None
                or followup["typed_action_ir_sha256"] != action["content_sha256"]
            ):
                raise ValueError("future Aether dossier binding changed")
            decision = followup["decision"]
            blocker = followup["first_blocker"]
            formal_hash = followup["content_sha256"]
            formal_source = "aether_followup"
            formal_scope = (
                "The local twist witness and the flat, static, globally pure-twist AE no-go "
                "and the positive weak-field quadratic theorem remain exact. All 14 candidates "
                "now also have an explicit compact C3 finite-amplitude Aether seed with a "
                "strictly negative static source monopole, a decaying frozen-source linearized "
                "conformal/York constraint completion, and a negative linearized completed-"
                "boundary-energy coefficient. Exact ADM characteristic-shell analysis shows that "
                "11 candidates are forced across a characteristic shell before the rigorous "
                "negative-source threshold, while two have certified characteristic-free negative "
                "amplitude windows and one is globally noncharacteristic. Those three now have "
                "exact uniform Aether Legendre-sector margins, inverse bounds, and strict negative "
                "source-energy margins. The weighted reference principal spectrum remains exact. "
                "The positive-unit-branch ADM audit now proves that A_i and p_A^i are prescribed "
                "free data rather than extra elliptic constraint unknowns: the second-order Aether "
                "diagonal is zero-dimensional and the 4x3 Aether off-diagonal principal columns "
                "vanish. Augmenting the four York unknowns with delta A instead produces a 4x7 "
                "symbol with a three-dimensional kernel. The actual finite-tilt metric-momentum-to-"
                "York principal symbol from the distributed Legendre map is now exact. One of the "
                "three regular candidates is uniformly principal elliptic throughout its registered "
                "seed. That candidate now also has an exact positive determinant gap, a rational "
                "pointwise principal-symbol inverse bound, and an elliptic homotopy to the Euclidean "
                "reference. Its distributed order-zero/one constraint coefficients and weighted "
                "kernel/coercivity estimate remain unregistered, so full Fredholm and operator "
                "inverse bounds do not follow. The other two cross "
                "exact perpendicular-covector York-variable shells; this rejects only K_ij=(L_X)_ij "
                "as their global completion variable, leaving an alternative canonical momentum or "
                "gauge open. No full Fredholm/isomorphism, full inverse-norm, nonlinear-remainder, or "
                "completed-boundary sign theorem is inferred. The other 11 retain the forced-"
                "characteristic blocker. Blocked is not rejection."
            )
        else:
            followup = g3_records.get(candidate_id)
            if followup is None or followup["action_sha256"] != action["content_sha256"]:
                raise ValueError("future G3 dossier binding changed")
            decision = followup["decision"]
            blocker = followup["first_blocker"]
            formal_hash = followup["content_sha256"]
            formal_source = "g3_followup"
            formal_scope = (
                "The action-bound local box, uniform principal/common cone, lapse coercivity, "
                "periodic Dirac, smooth AF-profile cone, and scalar-retained nonunitary "
                "BSSN/Bona-Masso principal gates pass. The prior radial conformal/pure-trace "
                "Lichnerowicz no-go is now extended to positive nonradial conformal factors and "
                "arbitrary smooth trace-free York tensors, including TT, longitudinal, and mixed "
                "pieces. The exact compensation theorem removes the pointwise |K|/v cap for the "
                "tracefree-compensated subclass. The latest theorem then removes conformal flatness "
                "itself: for any smooth complete AF three-geometry, including unrestricted nonzero "
                "Cotton tensor, the exact Hamiltonian residual is C-Y, where C is curvature surplus "
                "above c_star*v^2 and Y is the candidate York-source surplus. Thus any finite point "
                "with Y>C is excluded, strictly extending the earlier curvature-shortfall theorem "
                "into endpoint and above-threshold geometry. Exact controls reject an above-threshold "
                "surplus mismatch, retain C=Y only as a necessary Hamiltonian condition, and leave "
                "C>Y unexcluded. No AF metric/York datum or momentum solution is constructed, so "
                "candidate-specific pointwise surplus matching plus momentum closure, global energy, "
                "and full formal completion remain unproved. No action or theory is rejected."
            )
        status = {"reject": "rejected", "blocked": "blocked"}[decision]
        nodes = [
            _node(
                "exact_compiler_action",
                "proven",
                "Exact fields, parameters, and ordered covariant densities recompiled from the hash-bound future cell.",
                {
                    "source": "future_campaign_config",
                    "typed_action_ir_sha256": action["content_sha256"],
                    "compilation_receipt_sha256": action_packet["receipt"]["content_sha256"],
                },
            ),
            _node(
                "reviewed_formal_evidence",
                status,
                formal_scope,
                {"source": formal_source, "record_sha256": formal_hash},
                decision=decision,
                blocker=blocker,
            ),
            _node(
                "downstream_observational_evidence",
                "blocked",
                "No observation is opened and no observational ranking is authorized for this staged action.",
                {"source": "sealed_policy", "observation_opening_allowed": False},
            ),
        ]
        body = {
            "schema_version": DOSSIER_SCHEMA,
            "candidate_id": candidate_id,
            "family_id": action["family_id"],
            "parameter_cell_id": action_packet["cell"]["parameter_cell_id"],
            "action": {
                "action_sha256": action["content_sha256"],
                "fields": action["fields"],
                "parameters": action["parameters"],
                "ordered_operator_densities": action["operators"],
                "human_readable_action": _display_action(action),
                "matter_coupling": action["matter_coupling"],
            },
            "preflight_decision": preflight_record["decision"],
            "formal_decision": decision,
            "first_blocker": blocker,
            "comparison_contract": {
                "rank": None,
                "rank_eligible": False,
                "scientific_validity_inferred_from_formula": False,
                "promotion_eligible": False,
            },
            "hierarchy_nodes": nodes,
        }
        dossiers.append({**body, "content_sha256": _sha(body)})
    decisions = dict(sorted(Counter(item["formal_decision"] for item in dossiers).items()))
    if decisions != {"blocked": 17, "reject": 2}:
        raise ValueError("future dossier decision ledger changed")
    body = {
        "schema_version": ARTIFACT_SCHEMA,
        "campaign_id": config["campaign_id"],
        "candidate_count": len(dossiers),
        "family_counts": dict(sorted(Counter(item["family_id"] for item in dossiers).items())),
        "decision_counts": decisions,
        "ranked_candidate_count": 0,
        "dossiers": dossiers,
        "dossier_registry_root_sha256": _sha(
            [[item["candidate_id"], item["content_sha256"]] for item in dossiers]
        ),
        "source_roots": {
            "future_chunk_content_sha256": chunk["content_sha256"],
            "compilation_result_content_sha256": compilation["content_sha256"],
            "preflight_content_sha256": preflight["content_sha256"],
            "aether_followup_content_sha256": aether["content_sha256"],
            "g3_followup_content_sha256": g3["content_sha256"],
        },
        "data_eligibility": dict(ELIGIBILITY),
        "observational_authorization": False,
        "observational_data_opened": False,
        "paid_llm_spend_usd": 0.0,
    }
    artifact = {**body, "content_sha256": _sha(body)}
    if len((_canonical(artifact) + "\n").encode()) > int(config["budget"]["maximum_output_bytes"]):
        raise RuntimeError("future candidate action dossier exceeds output budget")
    return artifact


def validate_future_candidate_action_dossier(artifact: dict[str, Any]) -> None:
    body = {key: value for key, value in artifact.items() if key != "content_sha256"}
    if (
        artifact.get("schema_version") != ARTIFACT_SCHEMA
        or artifact.get("content_sha256") != _sha(body)
        or artifact.get("candidate_count") != 19
        or artifact.get("decision_counts") != {"blocked": 17, "reject": 2}
        or artifact.get("ranked_candidate_count") != 0
        or artifact.get("observational_authorization") is not False
        or artifact.get("observational_data_opened") is not False
        or artifact.get("paid_llm_spend_usd") != 0.0
        or len(artifact.get("dossiers", [])) != 19
    ):
        raise ValueError("future candidate action dossier artifact is invalid")
    for record in artifact["dossiers"]:
        record_body = {key: value for key, value in record.items() if key != "content_sha256"}
        action = record.get("action", {})
        display = action.get("human_readable_action", {})
        if (
            record.get("content_sha256") != _sha(record_body)
            or record.get("comparison_contract", {}).get("rank") is not None
            or record.get("comparison_contract", {}).get("rank_eligible") is not False
            or not action.get("ordered_operator_densities")
            or display.get("display_kind") != "verbatim_ordered_covariant_density_concatenation"
            or display.get("content_sha256")
            != _sha({key: value for key, value in display.items() if key != "content_sha256"})
            or any(
                node.get("content_sha256")
                != _sha({key: value for key, value in node.items() if key != "content_sha256"})
                for node in record.get("hierarchy_nodes", [])
            )
        ):
            raise ValueError("future candidate action dossier record is invalid")


def iter_future_candidate_action_dossiers(artifact: dict[str, Any]):
    validate_future_candidate_action_dossier(artifact)
    yield from artifact["dossiers"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the future candidate action dossier.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config", type=Path, default=Path("configs/future_candidate_action_dossier.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("runs/engine/future-candidate-action-dossier.json")
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    output_path = args.output if args.output.is_absolute() else root / args.output
    config = json.loads(config_path.read_text(encoding="utf-8"))
    artifact = build_future_candidate_action_dossier(config, root)
    validate_future_candidate_action_dossier(artifact)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
